"""End-to-end demo: simulate SSVI surface -> per-slice SVI calibration ->
hierarchical Bayesian pooling -> residual/outlier detection -> Ward
clustering -> portfolio exposure readout.
"""

import os
import time

import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

import clustering
import data_synthetic as ds
import portfolio_exposure as pe
import residual_analysis as ra
import svi_hierarchical as sh

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Simulating SSVI surface panel...")
    market_long, truth, tenor_labels = ds.simulate_surface(n_days=60, seed=7)
    portfolio = ds.synthetic_portfolio(tenor_labels)
    true_outliers = market_long.loc[market_long["is_true_outlier"], ["date", "tenor_label"]].drop_duplicates()
    print(f"  {market_long['date'].nunique()} days x {len(tenor_labels)} tenors, "
          f"{len(true_outliers)} planted dislocated slices")

    print("Stage A: per-slice raw SVI calibration (CEM-seeded)...")
    t0 = time.time()
    raw_panel = sh.fit_raw_panel(market_long, tenor_labels)
    print(f"  done in {time.time() - t0:.1f}s, {len(raw_panel)} slices")

    print("Stage B: hierarchical Bayesian pooling (PyMC)...")
    t0 = time.time()
    cleaned_panel, idata = sh.fit_hierarchical_svi(raw_panel, draws=600, tune=600, chains=2)
    print(f"  done in {time.time() - t0:.1f}s")

    print("Residuals and outlier flags...")
    residuals = ra.compute_residuals(market_long, cleaned_panel)
    flagged = ra.flag_outliers(residuals, z_thresh=3.0)
    detected = flagged.loc[flagged["is_outlier"], ["date", "tenor_label"]].drop_duplicates()
    hits = pd.merge(true_outliers, detected, on=["date", "tenor_label"], how="inner")
    print(f"  flagged {len(detected)} dislocated slices, recovered {len(hits)}/{len(true_outliers)} planted ones")

    print("Ward clustering of cleaned SVI params...")
    clustered, link = clustering.cluster_svi_params(cleaned_panel, n_clusters=5)
    cluster_summary = clustering.characterize_clusters(clustered)
    print(cluster_summary[["n_slices", "rho", "sigma", "regime_tag"]])

    print("Portfolio exposure readout...")
    factors, loadings = pe.term_structure_factors(cleaned_panel)
    last_date = market_long["date"].max()
    exposure_ts = pe.project_portfolio(portfolio, loadings)
    exposure_skew = pe.skew_curvature_exposure(portfolio, cleaned_panel, last_date)

    print(f"\n=== Trader summary (as of {last_date.date()}) ===")
    print(f"Level exposure     : {exposure_ts['level_exposure']:,.0f}")
    print(f"Slope exposure     : {exposure_ts['slope_exposure']:,.0f}")
    print(f"Curvature exposure : {exposure_ts['curvature_exposure']:,.0f}")
    print(f"Skew exposure (avg rho={exposure_skew['rho_avg']:.2f})    : {exposure_skew['skew_exposure']:,.0f}")
    print(f"Wing exposure (avg scale={exposure_skew['wing_scale_avg']:.4f}): {exposure_skew['wing_exposure']:,.0f}")

    _save_plots(truth, factors, residuals, link)
    print(f"\nPlots saved to {OUTPUT_DIR}")


def _save_plots(truth, factors, residuals, link):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(truth["date"], truth["level"] - truth["level"].mean(), label="true level (demeaned)")
    axes[0, 0].plot(factors["date"], factors["level"] - factors["level"].mean(), label="fitted level (demeaned)")
    axes[0, 0].legend()
    axes[0, 0].set_title("Level factor: true vs fitted")

    axes[0, 1].plot(truth["date"], truth["rho"], label="true rho")
    axes[0, 1].legend()
    axes[0, 1].set_title("Skew (rho) ground truth")

    pivot = residuals.pivot_table(index="date", columns="tenor_label", values="residual", aggfunc="mean")
    im = axes[1, 0].imshow(pivot.to_numpy().T, aspect="auto", cmap="coolwarm", vmin=-0.03, vmax=0.03)
    axes[1, 0].set_yticks(range(len(pivot.columns)))
    axes[1, 0].set_yticklabels(pivot.columns)
    axes[1, 0].set_title("Residual ('vol add-on') heatmap")
    fig.colorbar(im, ax=axes[1, 0])

    dendrogram(link, ax=axes[1, 1], no_labels=True)
    axes[1, 1].set_title("Ward clustering of SVI params")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "summary.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
