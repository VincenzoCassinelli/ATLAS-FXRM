"""
ATLAS-FXRM  |  Corporate Liquidity & FX Stress-Testing Engine
=============================================================
Monte Carlo engine that simulates a multinational's cash runway, interest
coverage and liquidity VaR under FX / interest-rate shocks, and compares
hedging strategies (forwards vs. FX options).

All money values are USD millions unless stated otherwise. Base currency: USD.
Foreign currencies are quoted as USD per 1 unit of foreign currency.

Model summary
  * FX      : correlated lognormal monthly returns (Cholesky of historical cov)
  * Rates   : Vasicek short-rate process for USD; EUR/PEN base rates move with
              the same simulated change (parallel-shift assumption)
  * Ops     : EBITDA with multiplicative noise, cash conversion %, per currency
  * Debt    : fixed/floating tranches, amortisation, bullet maturities
  * Hedging : monthly strip of forwards or Garman-Kohlhagen options on the
              planned net foreign-currency cash flow, executed BEFORE the shock
"""
from __future__ import annotations

import argparse
import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

CCYS = ["USD", "EUR", "PEN"]
FOREIGN = ["EUR", "PEN"]
IDX = {c: i for i, c in enumerate(CCYS)}


# --------------------------------------------------------------------------- #
# 1. DATA LAYER  (yfinance / FRED with deterministic synthetic fallback)
# --------------------------------------------------------------------------- #
def synthetic_fx(years: int = 5, seed: int = 42) -> pd.DataFrame:
    """Offline fallback: correlated GBM calibrated to rough EUR/PEN behaviour."""
    rng = np.random.default_rng(seed)
    n = 252 * years
    vols = np.array([0.075, 0.055]) / np.sqrt(252)
    corr = np.array([[1.0, 0.30], [0.30, 1.0]])
    L = np.linalg.cholesky(corr)
    r = (rng.standard_normal((n, 2)) @ L.T) * vols
    px = np.array([1.08, 0.27]) * np.exp(np.cumsum(r, axis=0))
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return pd.DataFrame(px, index=idx, columns=FOREIGN)


def load_fx(years: int = 5, synthetic: bool = False) -> tuple[pd.DataFrame, str]:
    """Daily USD-per-foreign-unit rates for EUR and PEN."""
    if not synthetic:
        try:
            import yfinance as yf

            raw = yf.download(["EURUSD=X", "USDPEN=X"], period=f"{years}y",
                              auto_adjust=True, progress=False)["Close"]
            df = pd.DataFrame({"EUR": raw["EURUSD=X"], "PEN": 1 / raw["USDPEN=X"]}).dropna()
            if len(df) > 250:
                return df, "yfinance"
        except Exception:
            pass
    return synthetic_fx(years), "synthetic"


def load_yield_curve(synthetic: bool = False) -> tuple[pd.Series, str]:
    """US Treasury curve from FRED (public CSV endpoint, no API key needed)."""
    series = {"3M": "DGS3MO", "2Y": "DGS2", "5Y": "DGS5", "10Y": "DGS10"}
    if not synthetic:
        try:
            import requests

            out = {}
            for k, sid in series.items():
                resp = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", timeout=6)
                s = pd.to_numeric(pd.read_csv(io.StringIO(resp.text)).iloc[:, 1], errors="coerce").dropna()
                out[k] = float(s.iloc[-1]) / 100
            return pd.Series(out), "FRED"
        except Exception:
            pass
    return pd.Series({"3M": 0.040, "2Y": 0.039, "5Y": 0.041, "10Y": 0.043}), "default"


@dataclass
class Market:
    spot: np.ndarray            # USD per EUR, PEN
    cov_m: np.ndarray           # monthly log-return covariance (2x2)
    vol_a: np.ndarray           # annualised vol per currency
    r0: dict                    # short rates {"USD","EUR","PEN"}
    theta: float                # Vasicek long-run mean (USD)
    kappa: float = 0.35
    sigma_r: float = 0.012      # annual rate vol (1.2% abs)
    curve: pd.Series = None
    sources: dict = field(default_factory=dict)


def build_market(synthetic: bool = False) -> Market:
    fx, fx_src = load_fx(synthetic=synthetic)
    curve, cv_src = load_yield_curve(synthetic=synthetic)
    lr = np.log(fx).diff().dropna()
    return Market(
        spot=fx.iloc[-1].values.astype(float),
        cov_m=lr.cov().values * 21,
        vol_a=lr.std().values * np.sqrt(252),
        r0={"USD": float(curve["3M"]), "EUR": 0.020, "PEN": 0.045},
        theta=float(curve["2Y"]),
        curve=curve,
        sources={"fx": fx_src, "curve": cv_src, "fx_history": fx},
    )


# --------------------------------------------------------------------------- #
# 2. CORPORATE MODEL
# --------------------------------------------------------------------------- #
@dataclass
class Tranche:
    name: str
    ccy: str
    principal: float            # local currency, millions
    maturity_m: int             # bullet repayment month
    floating: bool = False
    spread: float = 0.0         # over local base rate if floating
    fixed_rate: float = 0.0
    amort_pa: float = 0.0       # % of original principal repaid per year


@dataclass
class Company:
    name: str
    opening_cash: dict          # local millions
    ebitda_m: dict              # monthly EBITDA, local millions
    debt: list
    cash_conversion: float = 0.60   # EBITDA -> operating cash (after capex/tax/WC)
    op_vol: float = 0.07            # monthly EBITDA noise
    rcf_undrawn: float = 100.0      # USD millions
    min_cash: float = 75.0          # USD millions operating floor
    icr_covenant: float = 5.0       # min EBITDA / interest (rolling 3m)


def sample_company() -> Company:
    return Company(
        name="Andes Global Holdings (sample)",
        opening_cash={"USD": 60.0, "EUR": 25.0, "PEN": 150.0},
        ebitda_m={"USD": 10.0, "EUR": 7.0, "PEN": 45.0},
        debt=[
            Tranche("USD Term Loan B", "USD", 500, 36, floating=True, spread=0.035, amort_pa=0.05),
            Tranche("EUR Senior Notes", "EUR", 200, 15, fixed_rate=0.045),
            Tranche("PEN Local Bond", "PEN", 400, 20, fixed_rate=0.070),
        ],
    )


@dataclass
class Scenario:
    name: str
    rate_shift_bps: float = 0.0
    usd_appreciation: dict = field(default_factory=dict)   # e.g. {"EUR":0.15,"PEN":0.15}
    vol_mult: float = 1.0
    ebitda_shock: float = 0.0

    @property
    def rate_shift(self):
        return self.rate_shift_bps / 1e4


DEFAULT_SCENARIOS = [
    Scenario("Base"),
    Scenario("Rates +200bps", rate_shift_bps=200),
    Scenario("USD +15% vs EUR/PEN", usd_appreciation={"EUR": 0.15, "PEN": 0.15}),
    Scenario("Combined stress", rate_shift_bps=200, usd_appreciation={"EUR": 0.15, "PEN": 0.15},
             vol_mult=1.5, ebitda_shock=-0.05),
]


@dataclass
class Hedge:
    name: str
    kind: str = "none"          # none | forward | option
    ratio: float = 0.0
    otm: float = 0.05           # option strike distance from forward


STRATEGIES = [
    Hedge("Unhedged"),
    Hedge("Forward 50%", "forward", 0.5),
    Hedge("Forward 100%", "forward", 1.0),
    Hedge("Option 100% (5% OTM)", "option", 1.0, 0.05),
]


# --------------------------------------------------------------------------- #
# 3. SIMULATION CORE
# --------------------------------------------------------------------------- #
def _schedule(tr: Tranche, T: int):
    """Deterministic opening balance & principal paid per month (local ccy)."""
    ob, pay, bal = np.zeros(T), np.zeros(T), tr.principal
    for t in range(1, T + 1):
        ob[t - 1] = bal
        p = min(bal, tr.principal * tr.amort_pa / 12)
        if t == tr.maturity_m:
            p = bal
        pay[t - 1] = p
        bal -= p
    return ob, pay


def _flows(co: Company, mkt: Market, delta, noise, ebitda_shock, T):
    """Local-currency EBITDA, interest and principal arrays shaped (n, T, 3)."""
    n = delta.shape[0]
    ebitda, interest, princ = (np.zeros((n, T, 3)) for _ in range(3))
    for c in CCYS:
        ebitda[:, :, IDX[c]] = co.ebitda_m.get(c, 0.0) * (1 + ebitda_shock) * noise[:, :, IDX[c]]
    for tr in co.debt:
        ob, pay = _schedule(tr, T)
        rate = np.maximum(mkt.r0[tr.ccy] + delta + tr.spread, 0) if tr.floating else tr.fixed_rate
        interest[:, :, IDX[tr.ccy]] += ob[None, :] * rate / 12
        princ[:, :, IDX[tr.ccy]] += pay[None, :]
    return ebitda, interest, princ


def garman_kohlhagen(S, K, T, rd, rf, sigma, call: bool):
    d1 = (np.log(S / K) + (rd - rf + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if call:
        return S * np.exp(-rf * T) * norm.cdf(d1) - K * np.exp(-rd * T) * norm.cdf(d2)
    return K * np.exp(-rd * T) * norm.cdf(-d2) - S * np.exp(-rf * T) * norm.cdf(-d1)


def simulate(co: Company, mkt: Market, sc: Scenario, hedge: Hedge | None = None,
             n: int = 10_000, T: int = 24, seed: int = 7, stochastic: bool = True) -> dict:
    """Run one scenario/hedge combo. Same seed => common random numbers across runs."""
    rng = np.random.default_rng(seed)
    hedge = hedge or Hedge("Unhedged")
    nz = n if stochastic else 1
    vm = sc.vol_mult if stochastic else 0.0
    dt = 1 / 12

    # --- FX paths (n, T+1, 3); USD column = 1 -------------------------------
    L = np.linalg.cholesky(mkt.cov_m + 1e-14 * np.eye(2))
    z = rng.standard_normal((nz, T, 2)) @ L.T * vm
    z -= 0.5 * np.diag(mkt.cov_m) * vm**2
    S0 = mkt.spot * np.array([1 / (1 + sc.usd_appreciation.get(c, 0.0)) for c in FOREIGN])
    S = np.ones((nz, T + 1, 3))
    S[:, 0, 1:] = S0
    S[:, 1:, 1:] = S0 * np.exp(np.cumsum(z, axis=1))

    # --- Vasicek USD short rate; shock is a persistent level shift -----------
    zr = rng.standard_normal((nz, T)) * vm
    r = np.empty((nz, T + 1))
    r[:, 0] = mkt.r0["USD"] + sc.rate_shift
    theta = mkt.theta + sc.rate_shift
    for t in range(T):
        r[:, t + 1] = r[:, t] + mkt.kappa * (theta - r[:, t]) * dt + mkt.sigma_r * np.sqrt(dt) * zr[:, t]
    delta = r[:, :T] - mkt.r0["USD"]

    noise = 1 + co.op_vol * (vm if stochastic else 0.0) * rng.standard_normal((nz, T, 3))
    ebitda, interest, princ = _flows(co, mkt, delta, noise, sc.ebitda_shock, T)
    net_local = ebitda * co.cash_conversion - interest - princ
    Sm = S[:, 1:, :]
    cf = (net_local * Sm).sum(axis=2)                       # (n, T) USD

    # --- Hedging overlay (struck at pre-shock spot with base rates) ----------
    premium_total = 0.0
    if hedge.kind != "none" and hedge.ratio > 0:
        E_pl = _flows(co, mkt, np.zeros((1, T)), np.ones((1, T, 3)), 0.0, T)
        expo = (E_pl[0] * co.cash_conversion - E_pl[1] - E_pl[2])[0]     # (T, 3)
        for c in FOREIGN:   # balance-sheet exposure: opening foreign cash hedged to horizon end
            expo[T - 1, IDX[c]] += co.opening_cash.get(c, 0.0)
        hedge_cf = np.zeros((nz, T))
        for ci, c in enumerate(FOREIGN, start=1):
            for t in range(1, T + 1):
                E = hedge.ratio * expo[t - 1, ci]
                if abs(E) < 1e-9:
                    continue
                yrs = t / 12
                F = mkt.spot[ci - 1] * np.exp((mkt.r0["USD"] - mkt.r0[c]) * yrs)
                St = Sm[:, t - 1, ci]
                if hedge.kind == "forward":
                    hedge_cf[:, t - 1] += E * (F - St)
                else:   # long exposure -> put, short exposure -> call
                    is_call = E < 0
                    K = F * (1 + hedge.otm) if is_call else F * (1 - hedge.otm)
                    prem = abs(E) * garman_kohlhagen(mkt.spot[ci - 1], K, yrs, mkt.r0["USD"],
                                                     mkt.r0[c], mkt.vol_a[ci - 1], is_call)
                    pay = np.maximum(St - K, 0) if is_call else np.maximum(K - St, 0)
                    hedge_cf[:, t - 1] += abs(E) * pay
                    hedge_cf[:, 0] -= prem
                    premium_total += prem
        cf = cf + hedge_cf

    # --- Cash, liquidity, ICR ------------------------------------------------
    cash0_val = sum(co.opening_cash.get(c, 0.0) * S[:, :, IDX[c]] for c in CCYS)   # (n, T+1) revalued
    cash = cash0_val.copy()
    cash[:, 1:] += np.cumsum(cf, axis=1)
    liq = cash + co.rcf_undrawn

    e_usd, i_usd = (ebitda * Sm).sum(2), (interest * Sm).sum(2)
    ce, ci_ = np.cumsum(e_usd, 1), np.cumsum(i_usd, 1)
    ce, ci_ = np.pad(ce, ((0, 0), (1, 0))), np.pad(ci_, ((0, 0), (1, 0)))
    icr_q = (ce[:, 3:] - ce[:, :-3]) / np.maximum(ci_[:, 3:] - ci_[:, :-3], 1e-9)

    below = cash[:, 1:] < co.min_cash
    runway = np.where(below.any(1), below.argmax(1) + 1, T)
    return dict(cash=cash, liq=liq, runway=runway, icr_min=icr_q.min(1),
                premium=premium_total, T=T)


def tail_stats(loss, level=0.95):
    var = np.percentile(loss, level * 100)
    return var, loss[loss >= var].mean()


def summarize(co, res, plan) -> dict:
    end_loss = plan["liq"][0, -1] - res["liq"][:, -1]               # shortfall vs plan
    min_loss = plan["liq"][0].min() - res["liq"].min(axis=1)
    v95, c95 = tail_stats(end_loss, .95)
    v99, c99 = tail_stats(end_loss, .99)
    return {
        "Liquidity VaR 95% ($M)": v95, "Liquidity CVaR 95% ($M)": c95,
        "Liquidity VaR 99% ($M)": v99, "Liquidity CVaR 99% ($M)": c99,
        "Peak-trough VaR 95% ($M)": np.percentile(min_loss, 95),
        "Mean end liquidity ($M)": res["liq"][:, -1].mean(),
        "P5 min liquidity ($M)": np.percentile(res["liq"].min(1), 5),
        "Runway P5 (months)": np.percentile(res["runway"], 5),
        "Runway median (months)": np.median(res["runway"]),
        "P(cash < min floor)": (res["runway"] < res["T"]).mean(),
        "P(liquidity < floor)": (res["liq"].min(1) < co.min_cash).mean(),
        "Min ICR P5 (x)": np.percentile(res["icr_min"], 5),
        "P(ICR covenant breach)": (res["icr_min"] < co.icr_covenant).mean(),
        "Hedge premium ($M)": res["premium"],
    }


def run_grid(co, mkt, scenarios=None, strategies=None, n=10_000, T=24, seed=7):
    """Scenario x strategy results table + the raw paths (for charting)."""
    scenarios, strategies = scenarios or DEFAULT_SCENARIOS, strategies or STRATEGIES
    plan = simulate(co, mkt, Scenario("Plan"), n=1, T=T, stochastic=False)
    rows, raw = [], {}
    for sc in scenarios:
        for h in strategies:
            res = simulate(co, mkt, sc, h, n=n, T=T, seed=seed)
            raw[(sc.name, h.name)] = res
            rows.append({"Scenario": sc.name, "Strategy": h.name, **summarize(co, res, plan)})
    return pd.DataFrame(rows), raw, plan


# --------------------------------------------------------------------------- #
# 4. HEDGING RECOMMENDATION ENGINE
# --------------------------------------------------------------------------- #
def recommend(df: pd.DataFrame, scenario: str) -> str:
    d = df[df.Scenario == scenario].set_index("Strategy")
    base = d.loc["Unhedged"]
    best = d["Liquidity CVaR 95% ($M)"].idxmin()
    fwd, opt = d.loc["Forward 100%"], d[d.index.str.startswith("Option")].iloc[0]
    b = base["Liquidity CVaR 95% ($M)"]
    red = lambda r: (1 - r["Liquidity CVaR 95% ($M)"] / b) if b > 0 else 0.0
    lines = [f"[{scenario}] Lowest tail risk: {best}.",
             f"  CVaR95 change vs unhedged: Forward 100% {-red(fwd):+.0%}, Option {-red(opt):+.0%} "
             f"(option premium ${opt['Hedge premium ($M)']:.1f}M)."]
    if red(opt) >= 0.8 * red(fwd) and red(opt) > 0.05:
        lines.append("  => Prefer OPTIONS: near-forward protection while keeping FX upside; "
                     "use forwards on near-dated committed outflows such as the EUR maturity wall.")
    elif red(fwd) > 0.05:
        lines.append("  => Prefer FORWARDS on committed exposures (debt service/maturities); "
                     "options only on uncertain operating inflows - they cost more than the protection added.")
    else:
        lines.append("  => FX hedging adds little incremental protection in this scenario; "
                     "tail risk is driven by rates/operations. Consider swapping or capping the floating USD loan.")
    if base["P(ICR covenant breach)"] > 0.2:
        lines.append(f"  ! Covenant risk: P(ICR breach)={base['P(ICR covenant breach)']:.0%} unhedged - "
                     "FX hedges will not fix this; address floating-rate exposure.")
    return "\n".join(lines)


def to_excel(df: pd.DataFrame, plan: dict, path_or_buf):
    with pd.ExcelWriter(path_or_buf, engine="openpyxl") as xw:
        df.round(3).to_excel(xw, sheet_name="Stress Results", index=False)
        pd.DataFrame({"Month": range(plan["liq"].shape[1]), "Plan liquidity ($M)": plan["liq"][0]}
                     ).to_excel(xw, sheet_name="Plan", index=False)


# --------------------------------------------------------------------------- #
# 5. CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="ATLAS-FXRM stress-testing engine")
    ap.add_argument("--paths", type=int, default=10_000)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--synthetic", action="store_true", help="skip yfinance/FRED, use synthetic data")
    ap.add_argument("--excel", default="atlas_fxrm_results.xlsx")
    a = ap.parse_args()

    mkt, co = build_market(a.synthetic), sample_company()
    print(f"Data: FX={mkt.sources['fx']} | Curve={mkt.sources['curve']} | USD 3M={mkt.r0['USD']:.2%}")
    df, _, plan = run_grid(co, mkt, n=a.paths, T=a.horizon)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    cols = ["Scenario", "Strategy", "Liquidity VaR 95% ($M)", "Liquidity CVaR 95% ($M)",
            "Runway P5 (months)", "P(liquidity < floor)", "P(ICR covenant breach)", "Hedge premium ($M)"]
    print(df[cols].round(2).to_string(index=False))
    print()
    for sc in df.Scenario.unique():
        print(recommend(df, sc))
    to_excel(df, plan, a.excel)
    print(f"\nExcel written: {a.excel}")


if __name__ == "__main__":
    main()
