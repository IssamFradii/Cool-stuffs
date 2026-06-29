"""Streamlit dashboard for the NSS pillar-fit + residual-spline + bucket-risk
pipeline -- click-through version of run_demo.py's console output.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import bucket_risk
import calibration
import data_live
import data_synthetic
import nss_model
import residual_analysis
import spline_adjustment

st.set_page_config(page_title="NSS swap curve: pillar fit + residual spline + bucket risk", layout="wide")
st.title("Swap curve factor pipeline -- NSS pillar fit + residual spline + bucket risk")
st.caption(
    "Exact Nelson-Siegel-Svensson fit on the 2y/5y/10y/20y liquid pillars -> residuals on the full "
    "curve -> smoothing spline correction -> trader bucket delta projected onto level/slope/curvature/curvature2."
)

with st.sidebar:
    st.header("Market curve")
    source = st.radio("Source", ["Live Treasury (+ spread proxy)", "Synthetic"])
    if source.startswith("Live"):
        spread_bp = st.slider("Swap spread over Treasury (bp)", -50, 100, 20, step=5)
    else:
        seed = st.number_input("Seed", value=7, step=1)
        noise_std = st.slider("Noise std (%)", 0.0, 0.10, 0.02, step=0.01)
        wiggle_size = st.slider("Planted wiggle size (%)", 0.0, 0.20, 0.08, step=0.01)

    st.header("NSS shape")
    lambda1 = st.slider("lambda1 (curvature decay)", 0.3, 5.0, nss_model.DEFAULT_LAMBDA1, step=0.1)
    lambda2 = st.slider("lambda2 (curvature2 decay)", 0.5, 15.0, nss_model.DEFAULT_LAMBDA2, step=0.5)

    st.header("Residual correction")
    method = st.radio("Method", ["spline", "poly"])
    if method == "spline":
        s_auto = st.checkbox("Auto smoothing", value=True)
        s = None if s_auto else st.slider("Smoothing s", 0.0, 1.0, 0.05, step=0.01)
        spline_kwargs = {"s": s}
    else:
        degree = st.slider("Polynomial degree", 1, 6, 3)
        spline_kwargs = {"degree": degree}

    st.header("Trader bucket delta ($/bp)")
    bucket_df = pd.DataFrame({
        "tenor": bucket_risk.DEFAULT_BUCKET_T,
        "delta": bucket_risk.DEFAULT_BUCKET_DELTA,
    })
    bucket_df = st.data_editor(bucket_df, hide_index=True, num_rows="fixed")


@st.cache_data(show_spinner=False)
def load_curve(source, spread_bp=None, seed=None, noise_std=None, wiggle_size=None):
    if source.startswith("Live"):
        curve = data_live.fetch_treasury_curve()
        curve = data_live.to_pseudo_swap(curve, spread_bp=spread_bp)
        as_of = curve.attrs.get("as_of_date", "n/a")
    else:
        curve, _truth = data_synthetic.simulate_curve(seed=seed, noise_std=noise_std, wiggle_size=wiggle_size)
        as_of = "synthetic"
    return curve, as_of


try:
    if source.startswith("Live"):
        curve, as_of = load_curve(source, spread_bp=spread_bp)
    else:
        curve, as_of = load_curve(source, seed=seed, noise_std=noise_std, wiggle_size=wiggle_size)
except Exception as exc:
    st.error(f"Live fetch failed ({exc}), switch to Synthetic in the sidebar.")
    st.stop()

st.caption(f"Curve as of: {as_of} -- {len(curve)} tenor points")

pillar_rates = calibration.pillar_rates_from_curve(curve)
beta = calibration.fit_pillars(pillar_rates, lambda1=lambda1, lambda2=lambda2)

residuals = residual_analysis.compute_residuals(curve, beta, lambda1, lambda2)
resid_fn = spline_adjustment.fit_residual_adjustment(
    residuals["T"], residuals["residual"], method=method, **spline_kwargs
)

T_fine = np.linspace(curve["T"].min(), curve["T"].max(), 300)
nss_fine = nss_model.nss_rate(T_fine, beta, lambda1, lambda2)
combined_fine = spline_adjustment.combined_curve(T_fine, beta, lambda1, lambda2, resid_fn)

factor_delta = bucket_risk.project_bucket_delta(
    bucket_df["tenor"].to_numpy(), bucket_df["delta"].to_numpy(), lambda1, lambda2
)

col1, col2 = st.columns(2)

with col1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["T"], y=curve["rate_pct"], mode="markers", name="market", marker_color="black"))
    fig.add_trace(go.Scatter(x=T_fine, y=nss_fine, mode="lines", name="NSS (4 pillars)", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=T_fine, y=combined_fine, mode="lines", name="NSS + residual correction"))
    fig.update_layout(title="Market curve vs NSS vs combined fit", xaxis_title="maturity (y)", yaxis_title="rate (%)")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Fitted beta (level, slope, curvature, curvature2)")
    st.write({k: round(float(v), 4) for k, v in zip(["level", "slope", "curvature", "curvature2"], beta)})

with col2:
    fig2 = go.Figure()
    fig2.add_hline(y=0, line_color="grey")
    fig2.add_trace(go.Scatter(x=residuals["T"], y=residuals["residual"], mode="markers", name="residual", marker_color="red"))
    fig2.add_trace(go.Scatter(x=T_fine, y=resid_fn(T_fine), mode="lines", name=f"fitted {method}"))
    fig2.update_layout(title="Pillar-fit residuals + correction", xaxis_title="maturity (y)")
    st.plotly_chart(fig2, width="stretch")

    st.subheader("Bucket delta projected onto NSS factors")
    factor_view = pd.DataFrame({
        "factor": ["level", "slope", "curvature", "curvature2"],
        "delta": [factor_delta[f"delta_{f}"] for f in ["level", "slope", "curvature", "curvature2"]],
    })
    fig3 = go.Figure(go.Bar(x=factor_view["factor"], y=factor_view["delta"], marker_color="seagreen"))
    fig3.update_layout(title="Level / slope / curvature / curvature2 delta")
    st.plotly_chart(fig3, width="stretch")

st.subheader("Full curve table")
st.dataframe(residuals[["tenor_label", "T", "rate_pct", "nss_rate", "residual"]], hide_index=True, width="stretch")
