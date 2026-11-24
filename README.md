# ATLAS-FXRM
Corporate Liquidity & FX Stress-Testing Engine: Monte Carlo simulation of cash runway,
interest coverage and liquidity VaR/CVaR for a multi-currency corporate debt portfolio,
with forward-vs-option hedging recommendations.

## Run
    pip install -r requirements.txt
    python atlas_fxrm.py                 # live data (yfinance + FRED), prints table, writes Excel
    python atlas_fxrm.py --synthetic     # offline
    streamlit run dashboard.py           # interactive dashboard

## Edit the company
`sample_company()` in atlas_fxrm.py: opening cash, monthly EBITDA, and debt tranches
(currency, fixed/floating, spread, amortisation, maturity month). Add currencies by extending CCYS/FOREIGN and `build_market`.

## Method (key assumptions)
- FX: correlated lognormal (Cholesky of historical covariance), zero drift.
- Rates: Vasicek USD short rate; shocks are persistent level shifts; EUR/PEN floating bases move in parallel.
- Hedges struck at pre-shock spot with a monthly strip on planned net foreign cash flow + opening foreign cash (to horizon end).
  Forwards priced by covered interest parity; options by Garman-Kohlhagen at realised vol, 5% OTM.
- Liquidity VaR = 95th/99th percentile of (planned end-of-horizon liquidity - simulated liquidity); CVaR = tail mean.
- Simplifications: no refinancing, no tax/working-capital detail beyond `cash_conversion`, no counterparty/margin costs.

## Dashboard
`streamlit run dashboard.py` - dark terminal-style UI: sidebar controls + RUN button, KPI cards, tail-risk heatmap,
liquidity fan chart, VaR/CVaR distributions, hedging lab (forward vs option), FX/yield-curve view, editable debt &
cash model with maturity ladder, one-click Excel export. Theme lives in `.streamlit/config.toml`.
