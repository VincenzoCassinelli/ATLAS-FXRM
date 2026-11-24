"""
ATLAS-FXRM  |  Interactive dashboard
Run:  streamlit run dashboard.py
(set ATLAS_SYNTHETIC=1 to default to offline synthetic data)
"""
import html
import io
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import atlas_fxrm as A

st.set_page_config(page_title="ATLAS-FXRM", page_icon="🌐", layout="wide",
                   initial_sidebar_state="expanded")

P = dict(card="#111b2e", line="#1f2d4a", txt="#e6edf7", mut="#8a9bb8", teal="#22d3b6",
         amber="#f5b83d", red="#ff5d73", blue="#4c8dff", violet="#9b7bff")
SERIES = [P["blue"], P["teal"], P["amber"], P["violet"], P["red"]]

# ------------------------------------------------------------------ styling --
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}
.block-container {{ padding-top: 1.2rem; max-width: 1500px; }}
.stApp {{ background: radial-gradient(1200px 600px at 85% -10%, #14305a55, transparent),
                      radial-gradient(900px 500px at -10% 10%, #0f4c4a40, transparent), #0b1220; }}
[data-testid="stSidebar"] {{ background: #0e1727; border-right: 1px solid {P['line']}; }}
.hero {{ display:flex; justify-content:space-between; align-items:center; gap:20px; padding:22px 28px;
  border-radius:18px; border:1px solid {P['line']};
  background: linear-gradient(120deg, #12223d 0%, #0f2f3c 55%, #123a3a 100%); margin-bottom:14px; }}
.logo {{ font-size:2.0rem; font-weight:800; letter-spacing:.04em; color:{P['txt']}; }}
.logo b {{ color:{P['teal']}; }}
.sub {{ color:{P['mut']}; font-size:.92rem; margin-top:2px; }}
.chips {{ display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; }}
.chip {{ font-family:'JetBrains Mono',monospace; font-size:.72rem; padding:5px 10px; border-radius:999px;
  border:1px solid {P['line']}; color:{P['mut']}; background:#0b122066; }}
.chip.ok {{ color:{P['teal']}; border-color:#22d3b655; }}
.kpi {{ background:{P['card']}; border:1px solid {P['line']}; border-radius:14px; padding:14px 16px;
  border-top:3px solid var(--c); height:100%; transition:transform .15s; }}
.kpi:hover {{ transform:translateY(-2px); }}
.kpi.teal {{ --c:{P['teal']}; }} .kpi.amber {{ --c:{P['amber']}; }} .kpi.red {{ --c:{P['red']}; }} .kpi.blue {{ --c:{P['blue']}; }}
.kl {{ color:{P['mut']}; font-size:.72rem; text-transform:uppercase; letter-spacing:.09em; font-weight:600; }}
.kv {{ font-family:'JetBrains Mono',monospace; font-size:1.7rem; font-weight:700; color:{P['txt']}; margin-top:4px; }}
.ks {{ color:{P['mut']}; font-size:.74rem; margin-top:2px; }}
.card {{ background:{P['card']}; border:1px solid {P['line']}; border-radius:14px; padding:16px 20px; }}
.reco {{ border-left:4px solid {P['teal']}; font-size:.92rem; line-height:1.6; color:{P['txt']}; }}
.sec {{ font-size:.78rem; text-transform:uppercase; letter-spacing:.12em; color:{P['mut']}; font-weight:700; margin:18px 0 8px; }}
.stTabs [data-baseweb="tab-list"] {{ gap:6px; border-bottom:1px solid {P['line']}; }}
.stTabs [data-baseweb="tab"] {{ background:transparent; border-radius:10px 10px 0 0; padding:10px 18px; color:{P['mut']}; font-weight:600; }}
.stTabs [aria-selected="true"] {{ color:{P['teal']} !important; background:#22d3b60f; }}
.stButton>button, .stFormSubmitButton>button, .stDownloadButton>button {{ width:100%; border:0; border-radius:10px; font-weight:700;
  background:linear-gradient(90deg,{P['teal']},{P['blue']}); color:#06101f; padding:.6rem 1rem; }}
.stButton>button:hover, .stFormSubmitButton>button:hover, .stDownloadButton>button:hover {{ filter:brightness(1.1); color:#06101f; }}
[data-testid="stDataFrame"] {{ border:1px solid {P['line']}; border-radius:12px; }}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers --
def rgba(hex_, a):
    h = hex_.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"


def kpi(label, value, sub="", tone="teal"):
    return f'<div class="kpi {tone}"><div class="kl">{label}</div><div class="kv">{value}</div><div class="ks">{sub}</div></div>'


def tone_hi_bad(v, warn, bad):
    return "red" if v >= bad else "amber" if v >= warn else "teal"


def style(fig, h=420, title=None):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=h, margin=dict(l=10, r=10, t=50 if title else 20, b=10),
                      font=dict(family="Inter", color=P["txt"]), hovermode="x unified",
                      title=dict(text=title, font=dict(size=15)) if title else None,
                      legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor=P["line"], zeroline=False)
    fig.update_yaxes(gridcolor=P["line"], zeroline=False)
    return fig


@st.cache_resource(show_spinner="Loading market data (yfinance / FRED)...")
def get_market(synthetic):
    return A.build_market(synthetic)


# ------------------------------------------------------------------ sidebar --
default_syn = os.environ.get("ATLAS_SYNTHETIC") == "1"
with st.sidebar:
    st.markdown('<div class="logo" style="font-size:1.4rem">ATLAS<b>·FXRM</b></div>'
                '<div class="sub">Liquidity & FX stress engine</div>', unsafe_allow_html=True)
    with st.form("params"):
        st.markdown('<div class="sec">Simulation</div>', unsafe_allow_html=True)
        synthetic = st.toggle("Offline synthetic data", value=default_syn)
        n = st.select_slider("Monte Carlo paths", [2000, 5000, 10000, 25000, 50000], value=10000)
        T = st.slider("Horizon (months)", 12, 36, 24)
        seed = st.number_input("Random seed", value=7, step=1)
        st.markdown('<div class="sec">Custom shock</div>', unsafe_allow_html=True)
        bps = st.slider("Rate shift (bps)", -200, 400, 200, 25)
        usd = st.slider("USD appreciation vs EUR / PEN (%)", 0, 30, 15) / 100
        vol = st.slider("Volatility multiplier", 0.5, 3.0, 1.5, 0.1)
        ebd = st.slider("EBITDA shock (%)", -30, 10, -5) / 100
        st.markdown('<div class="sec">Treasury policy</div>', unsafe_allow_html=True)
        min_cash = st.number_input("Minimum cash floor ($M)", value=75.0)
        rcf = st.number_input("Undrawn RCF ($M)", value=100.0)
        icr = st.number_input("ICR covenant (x)", value=5.0, step=0.5)
        conv = st.slider("Cash conversion (% of EBITDA)", 30, 100, 60) / 100
        st.markdown('<div class="sec">Hedge design</div>', unsafe_allow_html=True)
        part = st.slider("Partial forward ratio (%)", 10, 90, 50, 10) / 100
        otm = st.slider("Option strike OTM (%)", 0, 15, 5) / 100
        submitted = st.form_submit_button("▶  RUN SIMULATION")

# --------------------------------------------------------------- main frame --
mkt = get_market(synthetic)
hero_slot = st.container()
ctl1, ctl2 = st.columns(2)
tabs = st.tabs(["Overview", "Liquidity Paths", "Tail Risk", "Hedging Lab", "Markets", "Company Model"])

# --- Company tab (editors are read before the run) --------------------------
base_co = A.sample_company()
with tabs[5]:
    st.markdown('<div class="sec">Debt tranches (editable - applied on next run)</div>', unsafe_allow_html=True)
    debt_df = st.data_editor(
        pd.DataFrame([t.__dict__ for t in base_co.debt]), num_rows="dynamic", hide_index=True,
        width="stretch", key="debt",
        column_config={"ccy": st.column_config.SelectboxColumn("ccy", options=A.CCYS),
                       "floating": st.column_config.CheckboxColumn("floating"),
                       "maturity_m": st.column_config.NumberColumn("maturity_m", min_value=1, step=1),
                       "spread": st.column_config.NumberColumn(format="%.4f"),
                       "fixed_rate": st.column_config.NumberColumn(format="%.4f"),
                       "amort_pa": st.column_config.NumberColumn(format="%.3f")})
    st.markdown('<div class="sec">Opening cash & monthly EBITDA (local currency, millions)</div>', unsafe_allow_html=True)
    ops_df = st.data_editor(
        pd.DataFrame({"opening_cash": base_co.opening_cash, "ebitda_m": base_co.ebitda_m}).T,
        width="stretch", key="ops")

try:
    tranches = [A.Tranche(r.name, r.ccy, float(r.principal), int(r.maturity_m), bool(r.floating),
                          float(r.spread or 0), float(r.fixed_rate or 0), float(r.amort_pa or 0))
                for r in debt_df.dropna(subset=["name", "ccy", "principal", "maturity_m"]).itertuples()]
    co = A.Company(name=base_co.name, opening_cash=ops_df.loc["opening_cash"].to_dict(),
                   ebitda_m=ops_df.loc["ebitda_m"].to_dict(), debt=tranches, cash_conversion=conv,
                   rcf_undrawn=rcf, min_cash=min_cash, icr_covenant=icr)
except Exception as e:
    st.error(f"Invalid company inputs: {e}")
    st.stop()

scenarios = A.DEFAULT_SCENARIOS + [A.Scenario("Custom stress", bps, {"EUR": usd, "PEN": usd}, vol, ebd)]
strategies = [A.Hedge("Unhedged"), A.Hedge(f"Forward {part:.0%}", "forward", part),
              A.Hedge("Forward 100%", "forward", 1.0),
              A.Hedge(f"Option 100% ({otm:.0%} OTM)", "option", 1.0, otm)]

if "res" not in st.session_state or submitted:
    with st.spinner(f"Simulating {len(scenarios) * len(strategies)} runs x {n:,} paths..."):
        df, raw, plan = A.run_grid(co, mkt, scenarios, strategies, n=n, T=T, seed=int(seed))
    st.session_state.res = dict(df=df, raw=raw, plan=plan, co=co, T=T, n=n,
                                strat=[s.name for s in strategies], scen=[s.name for s in scenarios])
R = st.session_state.res
df, raw, plan, co_r, T_r = R["df"], R["raw"], R["plan"], R["co"], R["T"]

with hero_slot:
    st.markdown(f"""<div class="hero"><div>
      <div class="logo">ATLAS<b>·FXRM</b></div>
      <div class="sub">Corporate Liquidity & FX Stress-Testing Engine &nbsp;|&nbsp; {html.escape(co_r.name)}</div></div>
      <div class="chips"><span class="chip ok">FX: {mkt.sources['fx']}</span>
      <span class="chip ok">CURVE: {mkt.sources['curve']}</span>
      <span class="chip">{R['n']:,} PATHS</span><span class="chip">{T_r}M HORIZON</span>
      <span class="chip">USD 3M {mkt.r0['USD']:.2%}</span>
      <span class="chip">EURUSD {mkt.spot[0]:.4f}</span></div></div>""", unsafe_allow_html=True)

focus = ctl1.selectbox("Focus scenario", R["scen"], index=len(R["scen"]) - 1)
fstrat = ctl2.selectbox("Focus hedge strategy", R["strat"], index=min(2, len(R["strat"]) - 1))
with st.sidebar:
    buf = io.BytesIO(); A.to_excel(df, plan, buf)
    st.download_button("⬇  Download Excel report", buf.getvalue(), "atlas_fxrm_results.xlsx")

row = lambda sc, stg: df[(df.Scenario == sc) & (df.Strategy == stg)].iloc[0]
u = row(focus, "Unhedged")
hedged = row(focus, fstrat)

# -------------------------------------------------------------- OVERVIEW ----
with tabs[0]:
    c = st.columns(6)
    cards = [
        ("Liquidity VaR 95%", f"${u['Liquidity VaR 95% ($M)']:.0f}M", "end-horizon shortfall vs plan",
         tone_hi_bad(u['Liquidity VaR 95% ($M)'], 25, 60)),
        ("Liquidity CVaR 95%", f"${u['Liquidity CVaR 95% ($M)']:.0f}M", "mean of worst 5%",
         tone_hi_bad(u['Liquidity CVaR 95% ($M)'], 30, 70)),
        ("Runway (P5)", f"{u['Runway P5 (months)']:.0f} mo", f"median {u['Runway median (months)']:.0f} mo",
         "red" if u['Runway P5 (months)'] < 12 else "amber" if u['Runway P5 (months)'] < T_r else "teal"),
        ("P(liquidity < floor)", f"{u['P(liquidity < floor)']:.1%}", "cash + RCF below floor",
         tone_hi_bad(u['P(liquidity < floor)'], .02, .10)),
        ("P(ICR breach)", f"{u['P(ICR covenant breach)']:.0%}", f"covenant {co_r.icr_covenant:.1f}x",
         tone_hi_bad(u['P(ICR covenant breach)'], .10, .30)),
        ("Min ICR (P5)", f"{u['Min ICR P5 (x)']:.1f}x", "rolling 3-month", "blue"),
    ]
    for col, (l, v, s, t) in zip(c, cards):
        col.markdown(kpi(l, v, s, t), unsafe_allow_html=True)

    st.markdown('<div class="sec">Tail risk map - CVaR 95% by scenario x hedge ($M)</div>', unsafe_allow_html=True)
    hm = df.pivot(index="Scenario", columns="Strategy", values="Liquidity CVaR 95% ($M)").loc[R["scen"], R["strat"]]
    fig = go.Figure(go.Heatmap(z=hm.values, x=hm.columns, y=hm.index, text=hm.round(0).astype(int).astype(str).values,
                               texttemplate="%{text}", textfont=dict(size=15, family="JetBrains Mono"),
                               colorscale=[[0, "#0f3b3a"], [.5, "#7a5a12"], [1, "#9b1c35"]], showscale=False,
                               hovertemplate="%{y}<br>%{x}<br>CVaR95: $%{z:.1f}M<extra></extra>"))
    st.plotly_chart(style(fig, 330).update_layout(hovermode="closest", yaxis=dict(autorange="reversed")),
                    width="stretch")

    a, b = st.columns([3, 2])
    with a:
        ic = df[df.Strategy == "Unhedged"].set_index("Scenario").loc[R["scen"]]
        fig = go.Figure(go.Bar(x=ic.index, y=ic["P(ICR covenant breach)"] * 100,
                               marker_color=[P["red"] if v > .3 else P["amber"] if v > .1 else P["teal"]
                                             for v in ic["P(ICR covenant breach)"]],
                               text=(ic["P(ICR covenant breach)"] * 100).round(0).astype(int).astype(str) + "%",
                               textposition="outside"))
        st.plotly_chart(style(fig, 320, "Covenant breach probability (%)").update_layout(hovermode="closest"),
                        width="stretch")
    with b:
        st.markdown('<div class="sec" style="margin-top:6px">Hedging recommendation</div>', unsafe_allow_html=True)
        txt = html.escape(A.recommend(df, focus)).replace("=&gt;", "→").replace("\n", "<br>")
        st.markdown(f'<div class="card reco">{txt}</div>', unsafe_allow_html=True)

# -------------------------------------------------------- LIQUIDITY PATHS ---
with tabs[1]:
    liq = raw[(focus, fstrat)]["liq"]
    x = list(range(liq.shape[1]))
    q = {k: np.percentile(liq, k, axis=0) for k in (5, 25, 50, 75, 95)}
    fig = go.Figure()
    for lo, hi, a_ in ((5, 95, .12), (25, 75, .25)):
        fig.add_trace(go.Scatter(x=x + x[::-1], y=list(q[hi]) + list(q[lo])[::-1], fill="toself",
                                 fillcolor=rgba(P["teal"], a_), line=dict(width=0), hoverinfo="skip",
                                 name=f"P{lo}-P{hi}"))
    fig.add_trace(go.Scatter(x=x, y=q[50], name="Median", line=dict(color=P["teal"], width=3)))
    fig.add_trace(go.Scatter(x=x, y=plan["liq"][0], name="Plan", line=dict(color=P["blue"], dash="dash")))
    fig.add_trace(go.Scatter(x=x, y=q[5], name="P5", line=dict(color=P["amber"], width=1.5)))
    fig.add_hline(y=co_r.min_cash, line=dict(color=P["red"], dash="dot"), annotation_text="Minimum floor",
                  annotation_font_color=P["red"])
    fig.update_xaxes(title="Month"); fig.update_yaxes(title="Liquidity: cash + undrawn RCF ($M)")
    st.plotly_chart(style(fig, 480, f"Liquidity fan - {focus} | {fstrat}"), width="stretch")

    cash = raw[(focus, fstrat)]["cash"]
    sample = cash[np.random.default_rng(1).choice(len(cash), min(60, len(cash)), replace=False)]
    fig = go.Figure()
    for p_ in sample:
        fig.add_trace(go.Scatter(x=x, y=p_, mode="lines", line=dict(width=1, color=rgba(P["blue"], .25)),
                                 showlegend=False, hoverinfo="skip"))
    fig.add_hline(y=co_r.min_cash, line=dict(color=P["red"], dash="dot"))
    fig.update_xaxes(title="Month"); fig.update_yaxes(title="Cash ($M)")
    st.plotly_chart(style(fig, 340, "Sample cash paths (60 of the simulation)"), width="stretch")

# -------------------------------------------------------------- TAIL RISK ---
with tabs[2]:
    def shortfall(stg):
        return plan["liq"][0, -1] - raw[(focus, stg)]["liq"][:, -1]
    l0, l1 = shortfall("Unhedged"), shortfall(fstrat)
    fig = go.Figure()
    for loss, name, col in ((l0, "Unhedged", P["red"]), (l1, fstrat, P["teal"])):
        fig.add_trace(go.Histogram(x=loss, name=name, nbinsx=70, opacity=.6, marker_color=col, histnorm="probability"))
        v, cv = A.tail_stats(loss, .95)
        fig.add_vline(x=v, line=dict(color=col, dash="dash"), annotation_text=f"VaR95 {v:.0f}",
                      annotation_font_color=col, annotation_position="top")
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="End-horizon liquidity shortfall vs plan ($M)  -  right = worse")
    st.plotly_chart(style(fig, 440, f"Shortfall distribution - {focus}").update_layout(hovermode="closest"),
                    width="stretch")
    k = st.columns(4)
    for col, (l, a_, b_) in zip(k, [("VaR 95%", *(A.tail_stats(l0, .95)[0], A.tail_stats(l1, .95)[0])),
                                    ("CVaR 95%", *(A.tail_stats(l0, .95)[1], A.tail_stats(l1, .95)[1])),
                                    ("VaR 99%", *(A.tail_stats(l0, .99)[0], A.tail_stats(l1, .99)[0])),
                                    ("CVaR 99%", *(A.tail_stats(l0, .99)[1], A.tail_stats(l1, .99)[1]))]):
        col.markdown(kpi(l, f"${a_:.0f}M → ${b_:.0f}M", "unhedged → focus hedge",
                         "teal" if b_ <= a_ else "red"), unsafe_allow_html=True)

# ----------------------------------------------------------- HEDGING LAB ----
with tabs[3]:
    d = df[df.Scenario == focus].set_index("Strategy").loc[R["strat"]]
    fig = go.Figure()
    for i, m in enumerate(["Liquidity VaR 95% ($M)", "Liquidity CVaR 95% ($M)", "Liquidity CVaR 99% ($M)"]):
        fig.add_trace(go.Bar(x=d.index, y=d[m], name=m.replace(" ($M)", ""), marker_color=SERIES[i]))
    fig.update_layout(barmode="group")
    st.plotly_chart(style(fig, 380, f"Risk by strategy - {focus}").update_layout(hovermode="closest"),
                    width="stretch")
    a, b = st.columns([3, 2])
    with a:
        tbl = d[["Mean end liquidity ($M)", "Hedge premium ($M)", "Runway P5 (months)",
                 "P(liquidity < floor)", "P(ICR covenant breach)"]].copy()
        tbl["CVaR95 reduction"] = 1 - d["Liquidity CVaR 95% ($M)"] / d.loc["Unhedged", "Liquidity CVaR 95% ($M)"]
        st.dataframe(tbl.style.format({"Mean end liquidity ($M)": "{:.1f}", "Hedge premium ($M)": "{:.2f}",
                                       "Runway P5 (months)": "{:.0f}", "P(liquidity < floor)": "{:.1%}",
                                       "P(ICR covenant breach)": "{:.1%}", "CVaR95 reduction": "{:+.1%}"}),
                     width="stretch")
    with b:
        txt = html.escape(A.recommend(df, focus)).replace("=&gt;", "→").replace("\n", "<br>")
        st.markdown(f'<div class="card reco">{txt}</div>', unsafe_allow_html=True)
        st.caption("Forwards lock the rate (zero cost, no upside). Options cost a premium but keep the favourable tail.")

# --------------------------------------------------------------- MARKETS ----
with tabs[4]:
    hist = mkt.sources["fx_history"]
    norm = hist / hist.iloc[0] * 100
    a, b = st.columns([3, 2])
    with a:
        fig = go.Figure()
        for i, cn in enumerate(norm.columns):
            fig.add_trace(go.Scatter(x=norm.index, y=norm[cn], name=f"{cn}/USD", line=dict(color=SERIES[i + 1], width=2)))
        st.plotly_chart(style(fig, 360, "FX history (rebased to 100)"), width="stretch")
    with b:
        cv = mkt.curve
        fig = go.Figure(go.Scatter(x=list(cv.index), y=cv.values * 100, mode="lines+markers",
                                   line=dict(color=P["amber"], width=3), marker=dict(size=9)))
        fig.update_yaxes(title="Yield (%)")
        st.plotly_chart(style(fig, 360, "US Treasury curve"), width="stretch")
    corr = np.corrcoef(np.log(hist).diff().dropna().values.T)[0, 1]
    k = st.columns(4)
    k[0].markdown(kpi("EUR vol (ann.)", f"{mkt.vol_a[0]:.1%}", "realised", "blue"), unsafe_allow_html=True)
    k[1].markdown(kpi("PEN vol (ann.)", f"{mkt.vol_a[1]:.1%}", "realised", "blue"), unsafe_allow_html=True)
    k[2].markdown(kpi("EUR/PEN corr", f"{corr:.2f}", "daily log returns", "amber"), unsafe_allow_html=True)
    k[3].markdown(kpi("Vasicek θ / κ", f"{mkt.theta:.2%} / {mkt.kappa:.2f}", "long-run mean / speed", "teal"),
                  unsafe_allow_html=True)

# ------------------------------------------------- COMPANY (ladder chart) ---
with tabs[5]:
    spot = {"USD": 1.0, "EUR": mkt.spot[0], "PEN": mkt.spot[1]}
    fig = go.Figure()
    for i, tr in enumerate(co_r.debt):
        _, pay = A._schedule(tr, T_r)
        fig.add_trace(go.Bar(x=list(range(1, T_r + 1)), y=pay * spot[tr.ccy], name=f"{tr.name} ({tr.ccy})",
                             marker_color=SERIES[i % len(SERIES)]))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Month"); fig.update_yaxes(title="Principal due (USD $M at spot)")
    st.plotly_chart(style(fig, 360, "Debt maturity ladder"), width="stretch")
