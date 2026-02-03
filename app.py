# =============================================================
# 🏛️ Institutional Apollo / ENIGMA – Quant Terminal (Streamlit)
# v2 – Single-file, Streamlit Cloud–ready
#
# Added (requested):
# 1) Relative VaR / CVaR / ES (active returns based)
# 2) Ledoit–Wolf shrinkage correlation (auto-fallback)
# 3) Black–Litterman (views + confidence sliders) – per-asset absolute views
# 4) Tracking Error tab (weekly / monthly, banded chart + tables)
# 5) Institutional KPI scorecard (traffic-light)
# =============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# SciPy optional but recommended
try:
    from scipy import linalg  # noqa: F401
except Exception:
    linalg = None  # noqa: F841

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
APP_TITLE = "🏛️ Institutional Apollo / ENIGMA – Quant Terminal (v2)"
DEFAULT_RF_ANNUAL = 0.03
TRADING_DAYS = 252

os.environ["NUMEXPR_MAX_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "4"

st.set_page_config(
    page_title="Apollo / ENIGMA – Quant Terminal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# THEME (dark / institutional)
# -------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --bg: #0b1220;
        --card: #0f172a;
        --border: #1e293b;
        --text: #e5e7eb;
        --muted: #94a3b8;
        --good: #16a34a;
        --warn: #f59e0b;
        --bad:  #dc2626;
    }
    .kpi {
        background-color: var(--card);
        padding: 16px 14px;
        border-radius: 14px;
        border: 1px solid var(--border);
        text-align: center;
        box-shadow: 0 6px 18px rgba(0,0,0,0.20);
    }
    .kpi h2 { color: var(--text); margin: 0; font-size: 26px; line-height: 1.0; }
    .kpi p  { color: var(--muted); margin: 6px 0 0 0; font-size: 13px; letter-spacing: .3px; }

    .scorecard {
        background-color: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 14px 14px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.20);
        margin-bottom: 12px;
    }
    .rowflex {
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:14px;
        padding: 8px 2px;
        border-bottom: 1px solid rgba(148,163,184,0.15);
    }
    .rowflex:last-child { border-bottom: none; }
    .label { color: var(--muted); font-size: 13px; }
    .value { color: var(--text); font-size: 14px; font-weight: 600; }

    .light {
        width: 12px; height: 12px;
        border-radius: 999px;
        display:inline-block;
        margin-right: 10px;
        box-shadow: 0 0 0 3px rgba(255,255,255,0.04);
    }
    .lg { background: var(--good); }
    .lo { background: var(--warn); }
    .lr { background: var(--bad);  }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_prices(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            close = data["Close"]
        else:
            close = data.xs(data.columns.levels[0][0], axis=1, level=0)
    else:
        close = data
    close = close.dropna(how="all")
    return close

def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")

def aggregate_horizon_simple(returns: pd.Series, horizon_days: int) -> pd.Series:
    lr = np.log1p(returns).dropna()
    agg = lr.rolling(horizon_days).sum().dropna()
    return np.expm1(agg)

# -------------------------------------------------------------
# RISK METRICS
# -------------------------------------------------------------
def max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns.fillna(0)).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min()) if len(dd) else np.nan

def hist_var_cvar_es(returns: pd.Series, alpha: float = 0.95):
    r = returns.dropna()
    if len(r) < 5:
        return np.nan, np.nan, np.nan
    q = r.quantile(1 - alpha)
    tail = r[r <= q]
    cvar = tail.mean() if len(tail) else np.nan
    es = cvar
    return float(q), float(cvar), float(es)

def annualize_return(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    return float(r.mean() * TRADING_DAYS)

def annualize_vol(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    return float(r.std() * np.sqrt(TRADING_DAYS))

def sharpe_ratio(r: pd.Series, rf_daily: float) -> float:
    ann_ret = annualize_return(r)
    ann_vol = annualize_vol(r)
    if ann_vol == 0 or np.isnan(ann_vol):
        return np.nan
    return float((ann_ret - rf_daily * TRADING_DAYS) / ann_vol)

# -------------------------------------------------------------
# CORRELATION (PSD + Ledoit–Wolf optional)
# -------------------------------------------------------------
def _nearest_psd_corr(corr: pd.DataFrame) -> pd.DataFrame:
    C = corr.values.astype(float)
    try:
        w, v = np.linalg.eigh(C)
        w[w < 0] = 0.0
        psd = (v @ np.diag(w) @ v.T)
        d = np.sqrt(np.clip(np.diag(psd), 1e-12, None))
        psd = psd / d[:, None] / d[None, :]
        psd = np.clip(psd, -1.0, 1.0)
        return pd.DataFrame(psd, index=corr.index, columns=corr.columns)
    except Exception:
        return corr

def ledoit_wolf_corr(returns_df: pd.DataFrame):
    r = returns_df.dropna(how="any")
    if r.shape[0] < 20 or r.shape[1] < 2:
        return r.corr(), "sample (insufficient data for LW)"
    try:
        from sklearn.covariance import LedoitWolf  # optional
        lw = LedoitWolf().fit(r.values)
        cov = lw.covariance_
        std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        corr = cov / std[:, None] / std[None, :]
        corr = pd.DataFrame(corr, index=r.columns, columns=r.columns)
        return corr, "Ledoit–Wolf shrinkage"
    except Exception:
        return r.corr(), "sample (sklearn not available)"

# -------------------------------------------------------------
# EWMA SIGNAL
# -------------------------------------------------------------
def ewma_vol(returns: pd.Series, span: int) -> pd.Series:
    return returns.ewm(span=span, adjust=False).std()

def bollinger(series: pd.Series, window: int = 20, n_std: float = 2.0):
    m = series.rolling(window).mean()
    s = series.rolling(window).std()
    upper = m + n_std * s
    lower = m - n_std * s
    return m, upper, lower

# -------------------------------------------------------------
# TRACKING ERROR
# -------------------------------------------------------------
def resample_returns(returns: pd.Series, freq: str) -> pd.Series:
    r = returns.dropna()
    if len(r) == 0:
        return r
    return (1 + r).resample(freq).prod() - 1

def tracking_error(returns_active: pd.Series, freq: str = "W-FRI") -> float:
    ra = resample_returns(returns_active, freq)
    if len(ra) < 4:
        return np.nan
    if freq.startswith("W"):
        ann = np.sqrt(52)
    elif freq.startswith("M"):
        ann = np.sqrt(12)
    else:
        ann = np.sqrt(TRADING_DAYS)
    return float(ra.std() * ann)

def rolling_tracking_error(active: pd.Series, freq: str, window: int) -> pd.Series:
    ra = resample_returns(active, freq)
    if len(ra) < (window + 2):
        return pd.Series(dtype=float)
    if freq.startswith("W"):
        ann = np.sqrt(52)
    elif freq.startswith("M"):
        ann = np.sqrt(12)
    else:
        ann = np.sqrt(TRADING_DAYS)
    return ra.rolling(window).std() * ann

# -------------------------------------------------------------
# ROLLING BETA
# -------------------------------------------------------------
def rolling_beta(asset: pd.Series, benchmark: pd.Series, window: int = 60) -> pd.Series:
    a = asset.dropna()
    b = benchmark.dropna()
    idx = a.index.intersection(b.index)
    a = a.loc[idx]
    b = b.loc[idx]
    cov = a.rolling(window).cov(b)
    var = b.rolling(window).var()
    return cov / var

# -------------------------------------------------------------
# BLACK–LITTERMAN (per-asset absolute views)
# -------------------------------------------------------------
def black_litterman_posterior(
    returns_df: pd.DataFrame,
    prior_weights: np.ndarray,
    delta: float,
    tau: float,
    views_df: pd.DataFrame
) -> dict:
    assets = list(returns_df.columns)
    n = len(assets)
    r = returns_df.dropna(how="any")
    if r.shape[0] < 30 or n < 2:
        return {"ok": False, "error": "Insufficient data for BL."}

    Sigma = np.cov(r.values, rowvar=False)

    w_mkt = prior_weights.reshape(-1, 1)
    pi = (delta * Sigma @ w_mkt)  # daily

    views_df = views_df.copy()
    views_df = views_df[views_df["Asset"].isin(assets)]
    views_df = views_df.dropna(subset=["View", "Confidence"])

    if len(views_df) == 0:
        mu = pi
        w = np.linalg.pinv(delta * Sigma) @ mu
        w = np.clip(w, 0, None)
        w = w / (w.sum() + 1e-12)
        return {"ok": True, "assets": assets, "Sigma": Sigma, "pi_daily": pi.flatten(), "mu_daily": mu.flatten(), "weights": w.flatten(),
                "note": "No views provided: posterior equals equilibrium (prior)."}

    k = len(views_df)
    P = np.zeros((k, n))
    Q = np.zeros((k, 1))
    Omega = np.zeros((k, k))

    for i, row in enumerate(views_df.itertuples(index=False)):
        a = row.Asset
        view_ann = float(row.View)
        conf = float(row.Confidence)

        j = assets.index(a)
        P[i, j] = 1.0
        Q[i, 0] = view_ann / TRADING_DAYS

        conf = float(np.clip(conf, 1e-3, 0.999))
        base = tau * Sigma[j, j]
        scale = (1.0 / conf) - 1.0
        Omega[i, i] = max(base * scale, 1e-12)

    try:
        tauSigma = tau * Sigma
        inv_tauSigma = np.linalg.pinv(tauSigma)
        inv_Omega = np.linalg.pinv(Omega)

        A = inv_tauSigma + (P.T @ inv_Omega @ P)
        b = (inv_tauSigma @ pi) + (P.T @ inv_Omega @ Q)
        mu = np.linalg.pinv(A) @ b

        w = np.linalg.pinv(delta * Sigma) @ mu
        w = np.clip(w, 0, None)
        w = w / (w.sum() + 1e-12)

        return {"ok": True, "assets": assets, "Sigma": Sigma, "pi_daily": pi.flatten(), "mu_daily": mu.flatten(), "weights": w.flatten(),
                "note": "Posterior computed with per-asset absolute views."}
    except Exception as e:
        return {"ok": False, "error": f"BL failed: {e}"}

# -------------------------------------------------------------
# UI – SIDEBAR
# -------------------------------------------------------------
st.sidebar.title("⚙️ Control Panel")

universe = [
    "SPY", "QQQ", "IWM", "TLT", "IEF", "GLD", "SLV", "USO", "UNG",
    "DXY", "EURUSD=X", "USDTRY=X",
    "BTC-USD", "ETH-USD"
]

tickers = st.sidebar.multiselect("Assets", universe, default=["SPY", "TLT", "GLD"])

benchmark = st.sidebar.selectbox(
    "Benchmark",
    [t for t in universe if t not in tickers] if len(tickers) else universe,
    index=0
)

start = st.sidebar.date_input("Start", pd.to_datetime("2018-01-01"))
end = st.sidebar.date_input("End", pd.to_datetime("today"))

conf_level = st.sidebar.selectbox("VaR Confidence", [0.90, 0.95, 0.99], index=1)
horizon_days = st.sidebar.slider("Risk horizon (days)", 1, 20, 1)

rf_annual = st.sidebar.number_input("Risk-free (annual, decimal)", value=float(DEFAULT_RF_ANNUAL), step=0.005, format="%.3f")
rf_daily = float(rf_annual) / TRADING_DAYS

st.sidebar.markdown("---")
st.sidebar.subheader("EWMA Risk Signal Bands")
sig_g = st.sidebar.number_input("Green max", value=0.80, step=0.05, format="%.2f")
sig_o = st.sidebar.number_input("Orange max", value=1.20, step=0.05, format="%.2f")

st.sidebar.markdown("---")
st.sidebar.subheader("Tracking Error Bands (annualized)")
te_g = st.sidebar.number_input("Green max (TE)", value=0.05, step=0.01, format="%.2f")
te_o = st.sidebar.number_input("Orange max (TE)", value=0.10, step=0.01, format="%.2f")
te_roll_weeks = st.sidebar.slider("Rolling TE window (weeks)", 8, 52, 26)
te_roll_months = st.sidebar.slider("Rolling TE window (months)", 6, 36, 12)

st.sidebar.markdown("---")
st.sidebar.subheader("Black–Litterman")
delta = st.sidebar.slider("Risk aversion (delta)", 1.0, 10.0, 2.5, 0.1)
tau = st.sidebar.slider("Tau", 0.01, 0.20, 0.05, 0.01)

run = st.sidebar.button("🚀 Run Analysis", use_container_width=True)

# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
st.title(APP_TITLE)

if (not run) or (len(tickers) < 1):
    st.info("Select assets and click **Run Analysis**.")
    st.stop()

# -------------------------------------------------------------
# LOAD + PREP
# -------------------------------------------------------------
all_tickers = list(dict.fromkeys(tickers + [benchmark]))
prices = load_prices(all_tickers, start, end)

missing = [t for t in all_tickers if t not in prices.columns]
if missing:
    st.warning(f"Missing tickers (no data): {missing}")

prices = prices[[c for c in prices.columns if c in all_tickers]].dropna(how="all")
rets = to_returns(prices).dropna(how="all")

rets = rets.dropna(axis=1, thresh=max(20, int(0.6 * len(rets))))

available_assets = [t for t in tickers if t in rets.columns]
if benchmark not in rets.columns:
    st.error("Benchmark data not available. Choose a different benchmark.")
    st.stop()
if len(available_assets) == 0:
    st.error("No asset data available after cleaning. Try different tickers or date range.")
    st.stop()

asset_rets = rets[available_assets].dropna(how="any")
bench_ret = rets[benchmark].reindex(asset_rets.index).dropna()

idx = asset_rets.index.intersection(bench_ret.index)
asset_rets = asset_rets.loc[idx]
bench_ret = bench_ret.loc[idx]

n = len(available_assets)
w_eq = np.array([1.0 / n] * n)

def portfolio_returns(weights: np.ndarray) -> pd.Series:
    w = np.array(weights).reshape(-1)
    w = w / (w.sum() + 1e-12)
    return pd.Series(asset_rets.values @ w, index=asset_rets.index, name="Portfolio")

port_eq = portfolio_returns(w_eq)
active_eq = port_eq - bench_ret

# -------------------------------------------------------------
# BLACK–LITTERMAN INPUT (views via data editor)
# -------------------------------------------------------------
st.subheader("Portfolio Mode")
port_mode = st.radio(
    "Select portfolio weighting",
    ["Equal Weight", "Black–Litterman (posterior weights)"],
    horizontal=True
)

views_default = pd.DataFrame({
    "Asset": available_assets,
    "View": [np.nan] * len(available_assets),
    "Confidence": [0.50] * len(available_assets)
})
st.caption("Black–Litterman views: set annual return expectations (optional) and confidence (0..1). Leave View blank to ignore.")
views_df = st.data_editor(
    views_default,
    use_container_width=True,
    num_rows="fixed",
    hide_index=True,
    column_config={
        "View": st.column_config.NumberColumn("View (annual expected return)", format="%.3f"),
        "Confidence": st.column_config.NumberColumn("Confidence (0..1)", min_value=0.0, max_value=1.0, step=0.05, format="%.2f"),
    }
)

bl_result = black_litterman_posterior(asset_rets, w_eq, float(delta), float(tau), views_df)

if port_mode.startswith("Black") and (not bl_result.get("ok", False)):
    st.warning(bl_result.get("error", "Black–Litterman failed; using Equal Weight."))
    port_mode = "Equal Weight"

if port_mode.startswith("Black") and bl_result.get("ok", False):
    w_bl = np.array(bl_result["weights"])
    port = portfolio_returns(w_bl)
    active = port - bench_ret
else:
    w_bl = None
    port = port_eq.copy()
    active = active_eq.copy()

# -------------------------------------------------------------
# KPI PANEL
# -------------------------------------------------------------
ann_ret = annualize_return(port)
ann_vol = annualize_vol(port)
sharpe = sharpe_ratio(port, rf_daily)
mdd = max_drawdown(port)

port_h = aggregate_horizon_simple(port, int(horizon_days))
var_p, cvar_p, es_p = hist_var_cvar_es(port_h, conf_level)

active_h = aggregate_horizon_simple(active, int(horizon_days))
rvar, rcvar, res = hist_var_cvar_es(active_h, conf_level)

kpi_cols = st.columns(6)
kpis = [
    ("Annual Return", f"{ann_ret*100:.2f}%"),
    ("Annual Vol", f"{ann_vol*100:.2f}%"),
    ("Sharpe", f"{sharpe:.2f}"),
    ("Max Drawdown", f"{mdd*100:.2f}%"),
    (f"VaR ({int(conf_level*100)}%, {horizon_days}d)", f"{var_p*100:.2f}%"),
    (f"Rel VaR ({int(conf_level*100)}%, {horizon_days}d)", f"{rvar*100:.2f}%"),
]
for col, (name, value) in zip(kpi_cols, kpis):
    with col:
        st.markdown(f"<div class='kpi'><h2>{value}</h2><p>{name}</p></div>", unsafe_allow_html=True)

st.markdown("---")

# -------------------------------------------------------------
# TABS
# -------------------------------------------------------------
tab_overview, tab_risk, tab_active, tab_corr, tab_beta, tab_signal, tab_bl, tab_score = st.tabs(
    [
        "📊 Overview",
        "⚖️ Risk (VaR / CVaR / ES)",
        "🎯 Active Risk (Relative VaR + Tracking Error)",
        "🔗 Correlation (PSD + LW)",
        "🧷 Rolling Beta",
        "🧭 EWMA Risk Signal",
        "🧠 Black–Litterman",
        "🚦 Institutional Scorecard"
    ]
)

# -------------------------------------------------------------
# OVERVIEW
# -------------------------------------------------------------
with tab_overview:
    col1, col2 = st.columns([2, 1])

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=port.index, y=(1 + port).cumprod(), name=f"Portfolio ({port_mode})"))
        fig.add_trace(go.Scatter(x=bench_ret.index, y=(1 + bench_ret).cumprod(), name=f"Benchmark ({benchmark})", line=dict(dash="dot")))
        fig.update_layout(title="Cumulative Performance", height=460, legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Weights")
        if port_mode.startswith("Black") and w_bl is not None:
            dfw = pd.DataFrame({"Asset": available_assets, "Weight": w_bl})
            st.dataframe(dfw.style.format({"Weight": "{:.2%}"}), use_container_width=True, height=360)
            st.caption(bl_result.get("note", ""))
        else:
            dfw = pd.DataFrame({"Asset": available_assets, "Weight": w_eq})
            st.dataframe(dfw.style.format({"Weight": "{:.2%}"}), use_container_width=True, height=360)
            st.caption("Equal-weight portfolio.")

# -------------------------------------------------------------
# RISK TAB
# -------------------------------------------------------------
with tab_risk:
    st.subheader("Portfolio – Historical VaR / CVaR / ES (Horizon-adjusted)")
    df_port = pd.DataFrame({"Metric": ["VaR", "CVaR", "ES"], "Value": [var_p, cvar_p, es_p]})
    st.dataframe(df_port.style.format({"Value": "{:.2%}"}), use_container_width=True)

    st.markdown("### Asset-level Risk (same confidence + horizon)")
    rows = []
    for a in available_assets:
        ah = aggregate_horizon_simple(asset_rets[a], int(horizon_days))
        v, c, e = hist_var_cvar_es(ah, conf_level)
        rows.append([a, v, c, e])
    df_assets = pd.DataFrame(rows, columns=["Asset", "VaR", "CVaR", "ES"])
    st.dataframe(df_assets.style.format({"VaR":"{:.2%}", "CVaR":"{:.2%}", "ES":"{:.2%}"}), use_container_width=True)

    st.markdown("### Distribution Snapshot")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=port_h.dropna().values, name="Portfolio horizon returns", nbinsx=60))
    fig.add_vline(x=var_p, line_width=2, line_dash="dash", annotation_text="VaR", annotation_position="top left")
    fig.add_vline(x=cvar_p, line_width=2, line_dash="dot", annotation_text="CVaR/ES", annotation_position="top right")
    fig.update_layout(height=420, barmode="overlay")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# ACTIVE RISK TAB
# -------------------------------------------------------------
with tab_active:
    st.subheader("Relative VaR / CVaR / ES (Active Returns vs Benchmark)")
    df_rel = pd.DataFrame({"Metric": ["Rel VaR", "Rel CVaR", "Rel ES"], "Value": [rvar, rcvar, res]})
    st.dataframe(df_rel.style.format({"Value": "{:.2%}"}), use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=active_h.dropna().values, name="Active horizon returns", nbinsx=60))
    fig.add_vline(x=rvar, line_width=2, line_dash="dash", annotation_text="Rel VaR", annotation_position="top left")
    fig.add_vline(x=rcvar, line_width=2, line_dash="dot", annotation_text="Rel CVaR/ES", annotation_position="top right")
    fig.update_layout(title="Active Return Distribution (horizon-adjusted)", height=420, barmode="overlay")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Tracking Error (Weekly / Monthly) with Band Zones")

    te_w = tracking_error(active, "W-FRI")
    te_m = tracking_error(active, "M")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Weekly TE (ann.)", f"{te_w*100:.2f}%")
    with c2:
        st.metric("Monthly TE (ann.)", f"{te_m*100:.2f}%")
    with c3:
        st.metric("Active Mean (daily)", f"{active.mean()*100:.4f}%")

    roll_w = rolling_tracking_error(active, "W-FRI", int(te_roll_weeks))
    roll_m = rolling_tracking_error(active, "M", int(te_roll_months))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.10,
                        subplot_titles=(f"Rolling Weekly TE (window={te_roll_weeks}w)", f"Rolling Monthly TE (window={te_roll_months}m)"))

    if len(roll_w):
        fig.add_trace(go.Scatter(x=roll_w.index, y=roll_w, name="Rolling Weekly TE"), row=1, col=1)
        fig.add_hrect(y0=0.0, y1=te_g, fillcolor="green", opacity=0.10, line_width=0, row=1, col=1)
        fig.add_hrect(y0=te_g, y1=te_o, fillcolor="orange", opacity=0.10, line_width=0, row=1, col=1)
        fig.add_hrect(y0=te_o, y1=max(te_o*2, te_o+0.05), fillcolor="red", opacity=0.10, line_width=0, row=1, col=1)

    if len(roll_m):
        fig.add_trace(go.Scatter(x=roll_m.index, y=roll_m, name="Rolling Monthly TE"), row=2, col=1)
        fig.add_hrect(y0=0.0, y1=te_g, fillcolor="green", opacity=0.10, line_width=0, row=2, col=1)
        fig.add_hrect(y0=te_g, y1=te_o, fillcolor="orange", opacity=0.10, line_width=0, row=2, col=1)
        fig.add_hrect(y0=te_o, y1=max(te_o*2, te_o+0.05), fillcolor="red", opacity=0.10, line_width=0, row=2, col=1)

    fig.update_layout(height=650, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Weekly / Monthly Active Returns (latest 12)")
    wtbl = resample_returns(active, "W-FRI").dropna().tail(12).to_frame("Weekly Active Return")
    mtbl = resample_returns(active, "M").dropna().tail(12).to_frame("Monthly Active Return")
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(wtbl.style.format("{:.2%}"), use_container_width=True)
    with c2:
        st.dataframe(mtbl.style.format("{:.2%}"), use_container_width=True)

# -------------------------------------------------------------
# CORRELATION TAB
# -------------------------------------------------------------
with tab_corr:
    st.subheader("Correlation Matrix (PSD-safe)")
    corr_method = st.radio("Correlation estimator", ["Ledoit–Wolf (if available)", "Sample"], horizontal=True)

    if corr_method.startswith("Ledoit"):
        corr_raw, label = ledoit_wolf_corr(asset_rets)
    else:
        corr_raw, label = asset_rets.corr(), "sample"

    corr_psd = _nearest_psd_corr(corr_raw)
    st.caption(f"Estimator used: **{label}** → PSD projection applied.")

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_psd.values,
            x=corr_psd.columns,
            y=corr_psd.index,
            colorscale="RdBu",
            zmin=-1, zmax=1,
            colorbar=dict(title="corr")
        )
    )
    fig.update_layout(title="PSD-safe Correlation Heatmap", height=520)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# ROLLING BETA TAB
# -------------------------------------------------------------
with tab_beta:
    st.subheader("Rolling Beta vs Benchmark")
    beta_window = st.slider("Rolling window (days)", 20, 252, 60, 5)

    fig = go.Figure()
    for a in available_assets:
        b = rolling_beta(asset_rets[a], bench_ret, int(beta_window))
        fig.add_trace(go.Scatter(x=b.index, y=b, name=a))
    fig.update_layout(height=460, legend=dict(orientation="h"), title=f"Rolling Beta (window={beta_window}d) vs {benchmark}")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# EWMA RISK SIGNAL TAB
# -------------------------------------------------------------
with tab_signal:
    st.subheader("EWMA Volatility Risk Signal (Portfolio)")

    vol22 = ewma_vol(port, 22)
    vol33 = ewma_vol(port, 33)
    vol99 = ewma_vol(port, 99)

    ratio = (vol22 / (vol33 + vol99)).dropna()
    mid, up, low = bollinger(ratio, window=20, n_std=2.0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ratio.index, y=ratio, name="EWMA Ratio"))
    fig.add_trace(go.Scatter(x=mid.index, y=mid, name="BB Mid", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=up.index, y=up, name="BB Upper", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=low.index, y=low, name="BB Lower", line=dict(dash="dash")))

    fig.add_hrect(y0=0.0, y1=sig_g, fillcolor="green", opacity=0.10, line_width=0)
    fig.add_hrect(y0=sig_g, y1=sig_o, fillcolor="orange", opacity=0.10, line_width=0)
    fig.add_hrect(y0=sig_o, y1=max(sig_o*2.2, sig_o+0.5), fillcolor="red", opacity=0.10, line_width=0)

    fig.update_layout(height=520, title="EWMA Ratio + Bollinger Bands", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# BLACK–LITTERMAN TAB
# -------------------------------------------------------------
with tab_bl:
    st.subheader("Black–Litterman Posterior")
    if not bl_result.get("ok", False):
        st.warning(bl_result.get("error", "BL not available."))
    else:
        assets = bl_result["assets"]
        pi_ann = np.array(bl_result["pi_daily"]) * TRADING_DAYS
        mu_ann = np.array(bl_result["mu_daily"]) * TRADING_DAYS
        w_post = np.array(bl_result["weights"])

        df_bl = pd.DataFrame({
            "Asset": assets,
            "Prior π (annual)": pi_ann,
            "Posterior μ (annual)": mu_ann,
            "Posterior Weight": w_post
        })
        st.dataframe(
            df_bl.style.format({"Prior π (annual)": "{:.2%}", "Posterior μ (annual)": "{:.2%}", "Posterior Weight": "{:.2%}"}),
            use_container_width=True
        )

        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_bl["Asset"], y=df_bl["Posterior Weight"], name="Posterior Weights"))
        fig.update_layout(height=420, title="Black–Litterman Posterior Weights")
        st.plotly_chart(fig, use_container_width=True)

        st.caption(bl_result.get("note", ""))

# -------------------------------------------------------------
# INSTITUTIONAL SCORECARD TAB
# -------------------------------------------------------------
def _light_class(value: float, green_max: float, orange_max: float, reverse: bool = False) -> str:
    if value is None or np.isnan(value):
        return "lo"
    if reverse:
        if value >= green_max:
            return "lg"
        elif value >= orange_max:
            return "lo"
        else:
            return "lr"
    else:
        if value <= green_max:
            return "lg"
        elif value <= orange_max:
            return "lo"
        else:
            return "lr"

with tab_score:
    st.subheader("Institutional KPI Scorecard (Traffic-Light)")

    colA, colB, colC = st.columns(3)
    with colA:
        thr_ret_g = st.number_input("Return GREEN min (ann.)", value=0.08, step=0.01, format="%.2f")
        thr_ret_o = st.number_input("Return ORANGE min (ann.)", value=0.03, step=0.01, format="%.2f")
    with colB:
        thr_sh_g = st.number_input("Sharpe GREEN min", value=1.00, step=0.10, format="%.2f")
        thr_sh_o = st.number_input("Sharpe ORANGE min", value=0.50, step=0.10, format="%.2f")
    with colC:
        thr_mdd_g = st.number_input("MDD GREEN max (abs)", value=0.15, step=0.01, format="%.2f")
        thr_mdd_o = st.number_input("MDD ORANGE max (abs)", value=0.25, step=0.01, format="%.2f")

    te_week = tracking_error(active, "W-FRI")
    sig_last = float(ratio.dropna().iloc[-1]) if len(ratio.dropna()) else np.nan

    ret_light = _light_class(ann_ret, thr_ret_g, thr_ret_o, reverse=True)
    sh_light = _light_class(sharpe, thr_sh_g, thr_sh_o, reverse=True)
    mdd_light = _light_class(abs(mdd) if not np.isnan(mdd) else np.nan, thr_mdd_g, thr_mdd_o, reverse=False)

    vol_light = _light_class(ann_vol, 0.15, 0.25, reverse=False)
    var_light = _light_class(abs(var_p), 0.03, 0.06, reverse=False)
    te_light = _light_class(te_week, te_g, te_o, reverse=False)
    sig_light = _light_class(sig_last, sig_g, sig_o, reverse=False)
    rvar_light = _light_class(abs(rvar), 0.01, 0.03, reverse=False)

    score_items = [
        ("Annual Return", f"{ann_ret*100:.2f}%", ret_light),
        ("Sharpe", f"{sharpe:.2f}", sh_light),
        ("Annual Vol", f"{ann_vol*100:.2f}%", vol_light),
        (f"VaR ({int(conf_level*100)}%, {horizon_days}d)", f"{var_p*100:.2f}%", var_light),
        ("Max Drawdown", f"{mdd*100:.2f}%", mdd_light),
        ("Tracking Error (weekly, ann.)", f"{te_week*100:.2f}%", te_light),
        ("EWMA Signal (latest)", f"{sig_last:.2f}", sig_light),
        ("Rel VaR (active)", f"{rvar*100:.2f}%", rvar_light),
    ]

    st.markdown("<div class='scorecard'>", unsafe_allow_html=True)
    for label, val, cls in score_items:
        st.markdown(
            f"<div class='rowflex'><div><span class='light {cls}'></span><span class='label'>{label}</span></div>"
            f"<div class='value'>{val}</div></div>",
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption("Traffic-light thresholds are configurable. Green=healthy, Orange=watch, Red=alert. Not financial advice.")

st.caption("© Institutional Apollo / ENIGMA – Research & Risk Analytics | Single-file Streamlit Terminal")
