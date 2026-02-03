# =====================================================================
# INSTITUTIONAL QUANT RESEARCH TERMINAL — STREAMLIT CLOUD EDITION
# (Converted + Upgraded from your Dash prototype)
# Monte Carlo | Portfolio Optimizer | ML | Regimes | Black-Litterman | GARCH |
# Proxy Factors | Trading Signals | Risk Metrics
# =====================================================================

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize

from arch import arch_model

# Optional dependency (can fail on some Streamlit Cloud builds)
try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore
    _HMM_AVAILABLE = True
except Exception:
    GaussianHMM = None  # type: ignore
    _HMM_AVAILABLE = False


# ===========================
# UI CONFIG
# ===========================
st.set_page_config(
    page_title="Institutional Quant Research Terminal",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Institutional Quant Research Terminal — Hedge Fund Edition")
st.caption("Streamlit Cloud–ready. Modular analytics: portfolio, risk, regimes, ML, signals.")


# ===========================
# CONFIG / INPUTS
# ===========================
DEFAULT_TICKERS = ["^GSPC", "^IXIC", "^FTSE", "GC=F", "CL=F", "EURUSD=X", "USDTRY=X"]
BENCHMARK_DEFAULT = "^GSPC"

with st.sidebar:
    st.header("⚙️ Controls")

    tickers = st.multiselect("Tickers", DEFAULT_TICKERS, default=["^GSPC", "GC=F", "EURUSD=X"])
    benchmark = st.selectbox("Benchmark (for beta / relative risk)", options=list(dict.fromkeys([BENCHMARK_DEFAULT] + tickers)), index=0)

    c1, c2 = st.columns(2)
    with c1:
        start = st.date_input("Start", value=date(2015, 1, 1))
    with c2:
        end = st.date_input("End", value=date.today())

    st.divider()

    st.subheader("Portfolio / Risk")
    risk_free = st.number_input("Risk-free rate (annual, decimal)", min_value=-0.05, max_value=0.25, value=0.05, step=0.005)
    allow_short = st.checkbox("Allow shorting (optimizer)", value=False)

    st.divider()

    st.subheader("Monte Carlo")
    mc_sims = st.slider("Simulations", 100, 5000, 750, 50)
    mc_days = st.slider("Horizon (days)", 21, 756, 252, 21)

    st.divider()

    st.subheader("GARCH")
    garch_h = st.slider("Forecast horizon (days)", 5, 120, 30, 5)

    st.divider()

    st.subheader("Regimes")
    n_regimes = st.slider("Regime count", 2, 4, 2, 1)
    use_hmm = st.checkbox("Use HMM (if available)", value=True)

    st.divider()

    run = st.button("▶ Run analysis", use_container_width=True)


# ===========================
# HELPERS
# ===========================
def _as_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [str(x)]


@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_prices(tickers: List[str], start: date, end: date) -> pd.DataFrame:
    """
    Download Adj Close prices from yfinance.
    Handles single vs multi-ticker shapes and returns a clean DataFrame.
    """
    if not tickers:
        return pd.DataFrame()

    df = yf.download(
        tickers=_as_list(tickers),
        start=pd.Timestamp(start),
        end=pd.Timestamp(end) + pd.Timedelta(days=1),
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )

    if df is None or len(df) == 0:
        return pd.DataFrame()

    # yfinance returns:
    # - MultiIndex columns for multiple tickers (field, ticker)
    # - Single-level columns for single ticker (fields)
    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close" in df.columns.get_level_values(0)):
            out = df["Adj Close"].copy()
        elif ("Close" in df.columns.get_level_values(0)):
            out = df["Close"].copy()
        else:
            # last resort: pick the first field
            out = df.xs(df.columns.levels[0][0], axis=1, level=0).copy()
    else:
        # single ticker
        if "Adj Close" in df.columns:
            out = df[["Adj Close"]].copy()
            out.columns = tickers[:1]
        elif "Close" in df.columns:
            out = df[["Close"]].copy()
            out.columns = tickers[:1]
        else:
            out = df.copy()
            out.columns = tickers[: out.shape[1]]

    out = out.dropna(how="all")
    out = out.sort_index()
    return out


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    rets = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
    rets = rets.dropna(axis=1, how="all")
    return rets.dropna()


def annualize_returns_cov(returns: pd.DataFrame, periods: int = 252) -> Tuple[pd.Series, pd.DataFrame]:
    mu = returns.mean() * periods
    cov = returns.cov() * periods
    return mu, cov


def portfolio_stats(weights: np.ndarray, mu: pd.Series, cov: pd.DataFrame, rf: float) -> Dict[str, float]:
    w = np.asarray(weights).reshape(-1)
    port_ret = float(np.dot(w, mu.values))
    port_var = float(w @ cov.values @ w)
    port_vol = float(np.sqrt(max(port_var, 0.0)))
    sharpe = (port_ret - rf) / port_vol if port_vol > 0 else np.nan
    return {"return": port_ret, "vol": port_vol, "sharpe": sharpe}


def optimize_portfolio(mu: pd.Series, cov: pd.DataFrame, rf: float, allow_short: bool) -> np.ndarray:
    n = len(mu)
    if n == 0:
        return np.array([])

    x0 = np.repeat(1 / n, n)
    bounds = [(-1.0, 1.0)] * n if allow_short else [(0.0, 1.0)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def neg_sharpe(w):
        s = portfolio_stats(w, mu, cov, rf)["sharpe"]
        # penalize infeasible / degenerate
        if not np.isfinite(s):
            return 1e6
        return -s

    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=cons)
    w = res.x if res.success else x0
    return w


def monte_carlo_gbm(series: pd.Series, sims: int = 500, days: int = 252) -> np.ndarray:
    """
    Geometric Brownian Motion simulation on log returns.
    Returns normalized paths starting at 1.0.
    """
    x = series.dropna().values
    if len(x) < 30:
        return np.zeros((days, sims))

    lr = np.log1p(x)
    mu = lr.mean()
    sigma = lr.std(ddof=1)

    paths = np.zeros((days, sims), dtype=float)
    paths[0, :] = 1.0
    for t in range(1, days):
        shock = np.random.normal(mu, sigma, size=sims)
        paths[t, :] = paths[t - 1, :] * np.exp(shock)
    return paths


def garch_forecast_vol(series: pd.Series, horizon: int = 30) -> np.ndarray:
    """
    Fit GARCH(1,1) to daily returns (%), forecast variance for `horizon` steps.
    Returns daily volatility (decimal) array length=horizon.
    """
    x = series.dropna()
    if len(x) < 250:
        return np.full(horizon, np.nan)

    # ARCH package likes percent returns
    x_pct = (x * 100.0).astype(float)
    model = arch_model(x_pct, vol="Garch", p=1, q=1, dist="normal")
    res = model.fit(disp="off")
    fc = res.forecast(horizon=horizon, reindex=False)
    var = fc.variance.values[-1, :]  # shape (horizon,)
    vol_pct = np.sqrt(np.maximum(var, 0.0))
    return vol_pct / 100.0  # back to decimal


def black_litterman_basic(returns: pd.DataFrame, tau: float = 0.05, delta: float = 2.5) -> pd.Series:
    """
    Basic Black-Litterman with:
    - Market weights = equal weight
    - Views = equilibrium returns (identity P, Q=pi)
    Produces a stable "BL posterior mean" and normalized weights (long-only by default).
    """
    if returns.shape[1] < 2:
        return pd.Series([1.0], index=returns.columns)

    mu, cov = annualize_returns_cov(returns)
    n = len(mu)
    w_mkt = np.repeat(1 / n, n).reshape(-1, 1)

    # Implied equilibrium returns
    pi = delta * (cov.values @ w_mkt)  # (n,1)

    P = np.eye(n)
    Q = pi.copy()
    omega = np.diag(np.diag(P @ (tau * cov.values) @ P.T))  # diagonal uncertainty

    middle = np.linalg.inv(np.linalg.inv(tau * cov.values) + P.T @ np.linalg.inv(omega) @ P)
    mu_bl = middle @ (np.linalg.inv(tau * cov.values) @ pi + P.T @ np.linalg.inv(omega) @ Q)  # (n,1)

    # Translate expected returns to weights (simple, not full BL optimization):
    # allow negative exposures but normalize; then clip long-only for stability
    raw = mu_bl.flatten()
    w = raw / (np.sum(np.abs(raw)) + 1e-12)
    w = np.clip(w, 0.0, None)
    if w.sum() <= 0:
        w = np.repeat(1 / n, n)
    else:
        w = w / w.sum()

    return pd.Series(w, index=returns.columns)


def proxy_factor_betas(asset_ret: pd.Series, bench_ret: pd.Series, cross_ret: pd.DataFrame) -> pd.Series:
    """
    Proxy "Fama-French-like" factors (not the real Fama-French dataset):
    - MKT = benchmark
    - SMB ~ avg(cross-section) - benchmark
    - HML ~ benchmark - avg(cross-section)
    Returns OLS betas [MKT, SMB, HML].
    """
    df = pd.concat([asset_ret, bench_ret, cross_ret.mean(axis=1)], axis=1).dropna()
    df.columns = ["asset", "mkt", "xavg"]
    df["smb"] = df["xavg"] - df["mkt"]
    df["hml"] = df["mkt"] - df["xavg"]

    X = np.column_stack([df["mkt"].values, df["smb"].values, df["hml"].values])
    y = df["asset"].values

    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return pd.Series(beta, index=["beta_mkt", "beta_smb_proxy", "beta_hml_proxy"])


def ml_forecast_rf(series: pd.Series, horizon: int = 30, lags: int = 5, n_estimators: int = 300, seed: int = 42) -> pd.Series:
    """
    RandomForestRegressor forecast on lag features.
    Produces iterative next-horizon predictions (out-of-sample simulation).
    """
    from sklearn.ensemble import RandomForestRegressor  # local import for faster cold start

    x = series.dropna().astype(float)
    if len(x) < (lags + 50):
        return pd.Series([np.nan] * horizon)

    df = pd.DataFrame({"y": x})
    for i in range(1, lags + 1):
        df[f"lag{i}"] = df["y"].shift(i)
    df = df.dropna()

    X = df[[f"lag{i}" for i in range(1, lags + 1)]].values
    y = df["y"].values

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    model.fit(X, y)

    # iterative forecast
    history = x.values.tolist()
    preds = []
    for _ in range(horizon):
        feats = np.array([history[-i] for i in range(1, lags + 1)], dtype=float).reshape(1, -1)
        p = float(model.predict(feats)[0])
        preds.append(p)
        history.append(p)

    idx = pd.date_range(x.index[-1] + pd.tseries.offsets.BDay(), periods=horizon, freq="B")
    return pd.Series(preds, index=idx, name="rf_pred")


def detect_regimes(series: pd.Series, n_states: int = 2, use_hmm: bool = True) -> pd.Series:
    """
    Regime detection:
    - If HMM is available and use_hmm=True: GaussianHMM on standardized returns.
    - Else: simple volatility-quantile regimes (fallback).
    Returns a Series of regime labels aligned to series index.
    """
    x = series.dropna().astype(float)
    if len(x) < 200:
        return pd.Series([np.nan] * len(series), index=series.index, name="regime")

    if use_hmm and _HMM_AVAILABLE:
        z = (x - x.mean()) / (x.std(ddof=1) + 1e-12)
        model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=200, random_state=7)
        model.fit(z.values.reshape(-1, 1))
        states = model.predict(z.values.reshape(-1, 1))
        return pd.Series(states, index=x.index, name="regime_hmm")

    # fallback: rolling volatility quantiles
    vol = x.rolling(20).std(ddof=1).dropna()
    q = pd.qcut(vol, q=n_states, labels=False, duplicates="drop")
    reg = pd.Series(index=x.index, dtype=float)
    reg.loc[q.index] = q.astype(float)
    return reg.rename("regime_vol_quantile")


def signal_ma_crossover(price: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    fast_ma = price.rolling(fast).mean()
    slow_ma = price.rolling(slow).mean()
    sig = np.where(fast_ma > slow_ma, 1, -1)
    return pd.Series(sig, index=price.index, name="signal")


def plot_price(prices: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for c in prices.columns:
        fig.add_trace(go.Scatter(x=prices.index, y=prices[c], mode="lines", name=c))
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h"))
    return fig


def plot_cum_returns(returns: pd.DataFrame) -> go.Figure:
    cum = (1 + returns).cumprod()
    fig = go.Figure()
    for c in cum.columns:
        fig.add_trace(go.Scatter(x=cum.index, y=cum[c], mode="lines", name=c))
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h"))
    return fig


def plot_monte_carlo(paths: np.ndarray) -> go.Figure:
    fig = go.Figure()
    if paths.size == 0 or np.all(paths == 0):
        fig.add_annotation(text="Not enough data to simulate.", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=420)
        return fig

    # show 25/50/75 fan + a few sample paths
    p25 = np.nanpercentile(paths, 25, axis=1)
    p50 = np.nanpercentile(paths, 50, axis=1)
    p75 = np.nanpercentile(paths, 75, axis=1)

    x = np.arange(paths.shape[0])
    for i in range(min(12, paths.shape[1])):
        fig.add_trace(go.Scatter(x=x, y=paths[:, i], mode="lines", name=f"path {i+1}", opacity=0.25))

    fig.add_trace(go.Scatter(x=x, y=p25, mode="lines", name="p25"))
    fig.add_trace(go.Scatter(x=x, y=p50, mode="lines", name="p50"))
    fig.add_trace(go.Scatter(x=x, y=p75, mode="lines", name="p75"))

    fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h"))
    return fig


def plot_vol_forecast(vol: np.ndarray) -> go.Figure:
    fig = go.Figure()
    x = np.arange(1, len(vol) + 1)
    fig.add_trace(go.Bar(x=x, y=vol, name="Daily vol (forecast)"))
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20), xaxis_title="Day", yaxis_title="Volatility (decimal)")
    return fig


# ===========================
# MAIN
# ===========================
if not run:
    st.info("Set tickers and dates in the sidebar, then click **Run analysis**.")
    st.stop()

if not tickers:
    st.error("Please select at least one ticker.")
    st.stop()

with st.spinner("Downloading data & computing analytics..."):
    prices = load_prices(tickers, start, end)
    if prices.empty:
        st.error("No price data returned. Try changing tickers or date range.")
        st.stop()

    rets = to_returns(prices)

    # Ensure benchmark exists
    if benchmark not in rets.columns:
        # If benchmark not in selected tickers, try to download it
        bench_prices = load_prices([benchmark], start, end)
        if bench_prices.empty:
            st.warning("Benchmark not available; using first selected ticker as benchmark.")
            benchmark = rets.columns[0]
            bench_ret = rets[benchmark]
        else:
            bench_ret = to_returns(bench_prices).iloc[:, 0].rename(benchmark)
    else:
        bench_ret = rets[benchmark]

    mu_ann, cov_ann = annualize_returns_cov(rets)

# ===========================
# TABS
# ===========================
tab_overview, tab_portfolio, tab_mc, tab_garch, tab_bl, tab_factors, tab_ml, tab_regime, tab_signals = st.tabs(
    ["📊 Overview", "🧺 Portfolio", "🎲 Monte Carlo", "🌪️ GARCH Vol", "🧠 Black–Litterman", "🧩 Factor Betas", "🤖 ML Forecast", "🧭 Regimes", "📌 Signals"]
)

# ---------------------------
# OVERVIEW
# ---------------------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assets", f"{rets.shape[1]}")
    c2.metric("Obs (daily)", f"{rets.shape[0]}")
    c3.metric("Date range", f"{prices.index.min().date()} → {prices.index.max().date()}")
    c4.metric("Benchmark", benchmark)

    st.subheader("Prices")
    st.plotly_chart(plot_price(prices), use_container_width=True)

    st.subheader("Cumulative performance (from 1.0)")
    st.plotly_chart(plot_cum_returns(rets), use_container_width=True)

    st.subheader("Return stats (annualized)")
    stats_df = pd.DataFrame({
        "ann_return": mu_ann,
        "ann_vol": (rets.std() * np.sqrt(252)).reindex(mu_ann.index),
        "sharpe": (mu_ann - risk_free) / (rets.std() * np.sqrt(252)).replace(0, np.nan),
    })
    st.dataframe(stats_df.style.format("{:.3f}"), use_container_width=True)


# ---------------------------
# PORTFOLIO
# ---------------------------
with tab_portfolio:
    st.subheader("Portfolio optimizer (max Sharpe)")
    w_opt = optimize_portfolio(mu_ann, cov_ann, rf=risk_free, allow_short=allow_short)
    w_df = pd.DataFrame({"weight": w_opt}, index=rets.columns).sort_values("weight", ascending=False)

    st.dataframe(w_df.style.format({"weight": "{:.4f}"}), use_container_width=True)

    s = portfolio_stats(w_opt, mu_ann, cov_ann, risk_free)
    c1, c2, c3 = st.columns(3)
    c1.metric("Expected return (ann)", f"{s['return']:.2%}")
    c2.metric("Volatility (ann)", f"{s['vol']:.2%}")
    c3.metric("Sharpe", f"{s['sharpe']:.3f}")

    port_ret_series = (rets @ w_opt).rename("portfolio")
    bench_aligned = bench_ret.reindex(port_ret_series.index).dropna()
    port_aligned = port_ret_series.reindex(bench_aligned.index)

    # Beta vs benchmark (OLS)
    beta = np.nan
    if len(port_aligned) > 30 and port_aligned.std(ddof=1) > 0 and bench_aligned.std(ddof=1) > 0:
        beta = float(np.cov(port_aligned, bench_aligned, ddof=1)[0, 1] / (bench_aligned.var(ddof=1) + 1e-12))

    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.metric("Beta vs benchmark", f"{beta:.3f}" if np.isfinite(beta) else "n/a")

    # Drawdown
    eq = (1 + port_aligned).cumprod()
    dd = eq / eq.cummax() - 1.0
    c2.metric("Max drawdown", f"{dd.min():.2%}")
    c3.metric("Tracking error (ann)", f"{((port_aligned - bench_aligned).std(ddof=1) * np.sqrt(252)):.2%}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq, mode="lines", name="Portfolio"))
    fig.add_trace(go.Scatter(x=(1 + bench_aligned).cumprod().index, y=(1 + bench_aligned).cumprod(), mode="lines", name="Benchmark"))
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Active risk snapshot")
    active = (port_aligned - bench_aligned).rename("active")
    var_95 = np.nanpercentile(active, 5)
    cvar_95 = active[active <= var_95].mean()
    st.write(pd.DataFrame({
        "VaR 95% (daily)": [var_95],
        "CVaR/ES 95% (daily)": [cvar_95],
    }).style.format("{:.4%}"))


# ---------------------------
# MONTE CARLO
# ---------------------------
with tab_mc:
    st.subheader("Monte Carlo (GBM) — normalized price paths")
    sel = st.selectbox("Simulate which series?", options=list(rets.columns), index=0)
    paths = monte_carlo_gbm(rets[sel], sims=mc_sims, days=mc_days)
    st.plotly_chart(plot_monte_carlo(paths), use_container_width=True)

    # Terminal distribution
    terminal = paths[-1, :] if paths.size else np.array([])
    if terminal.size:
        st.caption("Terminal distribution summary")
        st.write(pd.DataFrame({
            "p05": [np.percentile(terminal, 5)],
            "p50": [np.percentile(terminal, 50)],
            "p95": [np.percentile(terminal, 95)],
        }).style.format("{:.3f}"))


# ---------------------------
# GARCH
# ---------------------------
with tab_garch:
    st.subheader("GARCH(1,1) volatility forecast")
    sel = st.selectbox("Vol forecast series", options=list(rets.columns), index=0, key="garch_sel")
    vol = garch_forecast_vol(rets[sel], horizon=garch_h)
    st.plotly_chart(plot_vol_forecast(vol), use_container_width=True)

    if np.isfinite(vol).any():
        ann_vol = vol * np.sqrt(252)
        st.write(pd.DataFrame({
            "mean_ann_vol": [np.nanmean(ann_vol)],
            "last_ann_vol": [ann_vol[-1] if np.isfinite(ann_vol[-1]) else np.nan],
        }).style.format("{:.2%}"))


# ---------------------------
# BLACK-LITTERMAN
# ---------------------------
with tab_bl:
    st.subheader("Black–Litterman (basic posterior weights)")
    st.caption("Uses equal-weight market prior + equilibrium views. Weights are stabilized to long-only.")
    w_bl = black_litterman_basic(rets)
    fig = go.Figure(go.Bar(x=w_bl.index, y=w_bl.values, name="BL weights"))
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(w_bl.to_frame("weight").style.format("{:.4f}"), use_container_width=True)


# ---------------------------
# FACTORS
# ---------------------------
with tab_factors:
    st.subheader("Proxy factor betas (NOT real Fama–French dataset)")
    st.caption("This is a *proxy* decomposition using benchmark + cross-section averages. Good for quick diagnostics.")

    betas = {}
    for c in rets.columns:
        betas[c] = proxy_factor_betas(rets[c], bench_ret, rets)

    betas_df = pd.DataFrame(betas).T
    st.dataframe(betas_df.style.format("{:.3f}"), use_container_width=True)

    # Plot market betas
    fig = go.Figure(go.Bar(x=betas_df.index, y=betas_df["beta_mkt"], name="beta_mkt"))
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------
# ML
# ---------------------------
with tab_ml:
    st.subheader("ML forecast (Random Forest on lag features)")
    sel = st.selectbox("Forecast series", options=list(rets.columns), index=0, key="ml_sel")
    horizon = st.slider("Forecast horizon (days)", 5, 90, 30, 5)
    lags = st.slider("Lags", 2, 20, 5, 1)

    pred = ml_forecast_rf(rets[sel], horizon=horizon, lags=lags)

    fig = go.Figure()
    hist = rets[sel].dropna().iloc[-252:]
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines", name="history"))
    fig.add_trace(go.Scatter(x=pred.index, y=pred.values, mode="lines", name="forecast"))
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h"), yaxis_title="daily return")
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Note: This is a simple baseline model. For production, prefer walk-forward validation + feature expansion.")


# ---------------------------
# REGIMES
# ---------------------------
with tab_regime:
    st.subheader("Regime detection")
    st.caption(f"HMM available: **{_HMM_AVAILABLE}**. If unavailable, app falls back to volatility-quantile regimes.")

    sel = st.selectbox("Regime series (returns)", options=list(rets.columns), index=0, key="reg_sel")
    reg = detect_regimes(rets[sel], n_states=n_regimes, use_hmm=use_hmm)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rets.index, y=rets[sel], mode="lines", name=f"{sel} returns"))
    # overlay regimes as markers (aligned indices)
    rr = reg.dropna()
    fig.add_trace(go.Scatter(x=rr.index, y=np.zeros(len(rr)), mode="markers", name="regime", marker=dict(size=6), text=rr.astype(str)))
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    st.write("Regime counts:")
    st.write(rr.value_counts(dropna=True).sort_index())


# ---------------------------
# SIGNALS
# ---------------------------
with tab_signals:
    st.subheader("Trading signal engine (MA crossover)")
    sel = st.selectbox("Signal series (price)", options=list(prices.columns), index=0, key="sig_sel")
    fast = st.slider("Fast MA", 5, 60, 20, 1)
    slow = st.slider("Slow MA", 10, 200, 50, 5)

    p = prices[sel].dropna()
    sig = signal_ma_crossover(p, fast=fast, slow=slow)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p.index, y=p.values, mode="lines", name="price"))
    fig.add_trace(go.Scatter(x=p.index, y=p.rolling(fast).mean(), mode="lines", name=f"MA{fast}"))
    fig.add_trace(go.Scatter(x=p.index, y=p.rolling(slow).mean(), mode="lines", name=f"MA{slow}"))

    # signal line at bottom (scaled)
    s_scaled = (sig * 0.02 * p.median()) + p.min()
    fig.add_trace(go.Scatter(x=p.index, y=s_scaled, mode="lines", name="signal (scaled)"))

    fig.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    st.write("Latest signal:", int(sig.dropna().iloc[-1]) if len(sig.dropna()) else "n/a")
