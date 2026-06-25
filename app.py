"""Streamlit dashboard for the SVI + hierarchical Bayes vol surface pipeline.

Simulate -> per-slice SVI calibration (CEM-seeded) -> hierarchical Bayesian
pooling (PyMC) and/or sequential Bayesian filter -> residuals/outliers ->
Ward clustering -> portfolio exposure readout, all in one place to click
through instead of reading run_demo.py's console output.
"""

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from scipy.cluster.hierarchy import linkage as scipy_linkage

import clustering as cl
import data_synthetic as ds
import portfolio_exposure as pe
import residual_analysis as ra
import sequential_bayes as sb
import svi_hierarchical as sh

st.set_page_config(page_title="Vol surface: SVI + hierarchical Bayes", layout="wide")
st.title("Vol surface factor pipeline -- SVI + hierarchical Bayes")
st.caption(
    "Synthetic SSVI surface -> CEM-seeded per-slice SVI fit -> hierarchical Bayesian pooling "
    "(and/or a sequential Bayesian filter) -> residual outliers -> Ward clustering -> "
    "portfolio level/slope/curvature + skew/wing exposure."
)

with st.sidebar:
    st.header("Simulation")
    n_days = st.slider("Days", 20, 150, 60, step=10)
    seed = st.number_input("Seed", value=7, step=1)
    n_outliers = st.slider("Planted outliers", 0, 15, 6)

    st.header("Hierarchical fit (PyMC)")
    draws = st.slider("Draws", 200, 1500, 400, step=100)
    tune = st.slider("Tune", 200, 1500, 400, step=100)
    chains = st.slider("Chains", 1, 4, 2)

    st.header("Sequential Bayesian filter")
    tenor_smooth = st.slider("Cross-tenor smoothing", 0.0, 0.8, 0.3, step=0.1)

    st.header("Outliers / clusters")
    z_thresh = st.slider("Outlier z-threshold", 1.5, 5.0, 3.0, step=0.5)
    n_clusters = st.slider("Clusters", 2, 8, 5)

    st.caption("Re-runs automatically on any change above (cached, so unchanged settings are instant).")


@st.cache_resource(show_spinner=False)
def run_pipeline(n_days, seed, n_outliers, draws, tune, chains, tenor_smooth):
    market_long, truth, tenor_labels = ds.simulate_surface(n_days=n_days, seed=seed, n_outliers=n_outliers)
    portfolio = ds.synthetic_portfolio(tenor_labels)

    raw_panel = sh.fit_raw_panel(market_long, tenor_labels)
    cleaned_panel, idata = sh.fit_hierarchical_svi(raw_panel, draws=draws, tune=tune, chains=chains)

    day_sigma, obs_sigma = sh.extract_noise_scales(idata)
    filtered_long = sb.sequential_filter(raw_panel, day_sigma, obs_sigma, tenor_smooth=tenor_smooth)
    filtered_panel = sb.to_param_panel(filtered_long, ds.TENORS)

    return {
        "market_long": market_long, "truth": truth, "tenor_labels": tenor_labels, "portfolio": portfolio,
        "raw_panel": raw_panel, "cleaned_panel": cleaned_panel, "filtered_panel": filtered_panel,
        "day_sigma": day_sigma, "obs_sigma": obs_sigma,
    }


with st.spinner("Simulating, calibrating, pooling, filtering... (the PyMC step is the slow part, cached after the first run)"):
    result = run_pipeline(n_days, seed, n_outliers, draws, tune, chains, tenor_smooth)

market_long = result["market_long"]
truth = result["truth"]
tenor_labels = result["tenor_labels"]
portfolio = result["portfolio"]
raw_panel = result["raw_panel"]
cleaned_panel = result["cleaned_panel"]
filtered_panel = result["filtered_panel"]

# Derived (cheap) analytics recomputed on every rerun so sidebar tweaks to
# z_thresh / n_clusters don't require re-running the expensive PyMC fit.
residuals = ra.compute_residuals(market_long, cleaned_panel)
flagged = ra.flag_outliers(residuals, z_thresh=z_thresh)
true_outliers = market_long.loc[market_long["is_true_outlier"], ["date", "tenor_label"]].drop_duplicates()
detected = flagged.loc[flagged["is_outlier"], ["date", "tenor_label"]].drop_duplicates()
hits = pd.merge(true_outliers, detected, on=["date", "tenor_label"], how="inner")

clustered, link = cl.cluster_svi_params(cleaned_panel, n_clusters=n_clusters)
cluster_summary = cl.characterize_clusters(clustered)

factors_batch, loadings = pe.term_structure_factors(cleaned_panel)
factors_seq, _ = pe.term_structure_factors(filtered_panel)

tab_terms, tab_skew, tab_resid, tab_position, tab_compare, tab_data = st.tabs(
    ["Term structure", "Skew & regimes", "Residuals & outliers", "Position", "Batch vs sequential", "Raw data"]
)

with tab_terms:
    st.subheader("Level / slope / curvature, true vs fitted")
    cols = st.columns(3)
    true_level = ds.BASE_ATM_VOL + truth["level"]
    for col, factor_name in zip(cols, ["level", "slope", "curvature"]):
        fig = go.Figure()
        if factor_name == "level":
            fig.add_trace(go.Scatter(x=truth["date"], y=true_level, name="true", line=dict(color="grey")))
        else:
            fig.add_trace(go.Scatter(x=truth["date"], y=truth[factor_name], name="true (latent)", line=dict(color="grey")))
        fig.add_trace(go.Scatter(x=factors_batch["date"], y=factors_batch[factor_name], name="hierarchical fit"))
        fig.update_layout(title=factor_name.capitalize(), height=320, margin=dict(t=40, b=20))
        col.plotly_chart(fig, use_container_width=True)

    st.subheader("Pillars: how each tenor loads on level / slope / curvature")
    st.caption(
        "Once the curve is fit structurally (NS-style basis on the ATM term structure), these "
        "loadings are read directly off the model -- no PCA needed to discover them."
    )
    loadings_long = loadings.reset_index().melt(id_vars="index", var_name="factor", value_name="loading")
    loadings_long = loadings_long.rename(columns={"index": "tenor_label"})
    fig = px.bar(loadings_long, x="tenor_label", y="loading", facet_col="factor", category_orders={"tenor_label": tenor_labels})
    fig.update_layout(height=350, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

with tab_skew:
    st.subheader("Skew (rho) and wing (eta) ground truth")
    c1, c2 = st.columns(2)
    fig_rho = px.line(truth, x="date", y="rho", title="True rho (skew tilt)")
    c1.plotly_chart(fig_rho, use_container_width=True)
    fig_eta = px.line(truth, x="date", y="eta", title="True eta (wing/curvature scale)")
    c2.plotly_chart(fig_eta, use_container_width=True)

    st.subheader("Ward clustering of the cleaned SVI parameters (replaces PCA)")
    st.dataframe(cluster_summary[["n_slices", "rho", "b", "sigma", "regime_tag"]], use_container_width=True)

    dendro = ff.create_dendrogram(clustered[cl.FEATURES].to_numpy(), linkagefun=lambda x: scipy_linkage(x, method="ward"))
    dendro.update_layout(title="Dendrogram (Ward linkage on a,b,rho,m,sigma)", height=400, margin=dict(t=40, b=20))
    st.plotly_chart(dendro, use_container_width=True)

with tab_resid:
    st.subheader("Residual ('vol add-on') heatmap -- market minus hierarchical SVI fit")
    pivot = residuals.pivot_table(index="date", columns="tenor_label", values="residual", aggfunc="mean")
    pivot = pivot[[t for t in tenor_labels if t in pivot.columns]]
    fig = go.Figure(go.Heatmap(z=pivot.to_numpy().T, x=pivot.index, y=pivot.columns, colorscale="RdBu", zmid=0))
    fig.update_layout(height=420, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Planted dislocated slices", len(true_outliers))
    c2.metric("Flagged by the model", len(detected))
    c3.metric("Recovered", f"{len(hits)}/{len(true_outliers)}")

    st.subheader("Flagged outlier slices")
    flagged_slices = (
        flagged.loc[flagged["is_outlier"]]
        .groupby(["date", "tenor_label"])["z"].agg(lambda s: s.abs().max())
        .reset_index()
        .sort_values("z", ascending=False)
    )
    flagged_slices["planted"] = flagged_slices.set_index(["date", "tenor_label"]).index.isin(
        true_outliers.set_index(["date", "tenor_label"]).index
    )
    st.dataframe(flagged_slices, use_container_width=True)

with tab_position:
    st.subheader("Portfolio ladder (edit live -- exposures below update immediately)")
    edited_portfolio = st.data_editor(portfolio, use_container_width=True, num_rows="fixed", key="portfolio_editor")

    last_date = market_long["date"].max()
    exposure_ts = pe.project_portfolio(edited_portfolio, loadings)
    exposure_skew = pe.skew_curvature_exposure(edited_portfolio, cleaned_panel, last_date)

    st.subheader(f"Exposure readout as of {last_date.date()}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Level", f"{exposure_ts['level_exposure']:,.0f}")
    c2.metric("Slope", f"{exposure_ts['slope_exposure']:,.0f}")
    c3.metric("Curvature", f"{exposure_ts['curvature_exposure']:,.0f}")
    c4.metric("Skew", f"{exposure_skew['skew_exposure']:,.0f}", help=f"avg rho={exposure_skew['rho_avg']:.2f}")
    c5.metric("Wing", f"{exposure_skew['wing_exposure']:,.0f}", help=f"avg scale={exposure_skew['wing_scale_avg']:.4f}")

    greeks_long = edited_portfolio.melt(id_vars="tenor_label", var_name="greek", value_name="value")
    fig = px.bar(greeks_long, x="tenor_label", y="value", facet_col="greek", category_orders={"tenor_label": tenor_labels})
    fig.update_layout(height=350, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

with tab_compare:
    st.subheader("Batch hierarchical (PyMC) vs sequential Bayesian filter")
    st.caption(
        "Sequential filter reuses the batch fit's own learned day_sigma/obs_sigma as its process "
        "and observation noise, then updates each tenor's parameters day by day in closed form -- "
        "no MCMC re-run needed for each new day."
    )
    cols = st.columns(3)
    for col, factor_name in zip(cols, ["level", "slope", "curvature"]):
        fig = go.Figure()
        true_series = true_level if factor_name == "level" else truth[factor_name]
        fig.add_trace(go.Scatter(x=truth["date"], y=true_series, name="true", line=dict(color="grey")))
        fig.add_trace(go.Scatter(x=factors_batch["date"], y=factors_batch[factor_name], name="batch"))
        fig.add_trace(go.Scatter(x=factors_seq["date"], y=factors_seq[factor_name], name="sequential"))
        fig.update_layout(title=factor_name.capitalize(), height=320, margin=dict(t=40, b=20))
        col.plotly_chart(fig, use_container_width=True)

    st.subheader("Learned noise scales (from the batch fit, reused by the sequential filter)")
    noise_df = pd.DataFrame({"day_sigma": result["day_sigma"], "obs_sigma": result["obs_sigma"]})
    st.dataframe(noise_df, use_container_width=True)

with tab_data:
    st.subheader("Raw simulated surface (long format)")
    st.dataframe(market_long.head(500), use_container_width=True)
    st.subheader("Stage A: independent per-slice SVI fits")
    st.dataframe(raw_panel, use_container_width=True)
    st.subheader("Stage B: hierarchically cleaned SVI panel")
    st.dataframe(cleaned_panel, use_container_width=True)
