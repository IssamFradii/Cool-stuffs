"""End-to-end demo: fetch the market curve (live Treasury proxy-swap, or a
synthetic fallback) -> exact NSS fit on the 2/5/10/20 pillars -> residuals
on the full curve -> smoothing-spline correction -> combined interpolator
-> project an example trader bucket-delta ladder onto level/slope/
curvature/curvature2.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import bucket_risk
import calibration
import data_live
import data_synthetic
import nss_model
import residual_analysis
import spline_adjustment

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

LAMBDA1 = nss_model.DEFAULT_LAMBDA1
LAMBDA2 = nss_model.DEFAULT_LAMBDA2

EXAMPLE_BUCKET_T = bucket_risk.DEFAULT_BUCKET_T
EXAMPLE_BUCKET_DELTA = bucket_risk.DEFAULT_BUCKET_DELTA


def load_market_curve():
    try:
        curve = data_live.fetch_treasury_curve()
        curve = data_live.to_pseudo_swap(curve, spread_bp=20.0)
        source = f"live Treasury+spread proxy, as of {curve.attrs.get('as_of_date')}"
    except Exception as exc:
        print(f"  live fetch failed ({exc}), falling back to synthetic curve")
        curve, _truth = data_synthetic.simulate_curve()
        source = "synthetic"
    return curve, source


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading market curve...")
    curve, source = load_market_curve()
    print(f"  {len(curve)} tenor points ({source})")

    print("Stage A: exact NSS fit on the 2/5/10/20 pillars...")
    pillar_rates = calibration.pillar_rates_from_curve(curve)
    beta = calibration.fit_pillars(pillar_rates, lambda1=LAMBDA1, lambda2=LAMBDA2)
    print(f"  beta (level, slope, curvature, curvature2) = {np.round(beta, 4)}")

    print("Residuals across the full curve...")
    residuals = residual_analysis.compute_residuals(curve, beta, LAMBDA1, LAMBDA2)
    print(residuals[["tenor_label", "T", "rate_pct", "nss_rate", "residual"]].to_string(index=False))

    print("Stage B: smoothing spline on the residuals...")
    resid_fn = spline_adjustment.fit_residual_adjustment(
        residuals["T"], residuals["residual"], method="spline"
    )

    print("Bucket delta -> NSS factor risk...")
    factor_delta = bucket_risk.project_bucket_delta(EXAMPLE_BUCKET_T, EXAMPLE_BUCKET_DELTA, LAMBDA1, LAMBDA2)
    for name, value in factor_delta.items():
        print(f"  {name:18s} = {value:,.1f}")

    _save_plot(curve, residuals, beta, resid_fn, factor_delta)
    print(f"\nPlot saved to {OUTPUT_DIR}")


def _save_plot(curve, residuals, beta, resid_fn, factor_delta):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    T_fine = np.linspace(curve["T"].min(), curve["T"].max(), 300)
    nss_fine = nss_model.nss_rate(T_fine, beta, LAMBDA1, LAMBDA2)
    combined_fine = spline_adjustment.combined_curve(T_fine, beta, LAMBDA1, LAMBDA2, resid_fn)

    axes[0, 0].scatter(curve["T"], curve["rate_pct"], label="market", color="black", zorder=3)
    axes[0, 0].plot(T_fine, nss_fine, label="NSS (4 pillars)", linestyle="--")
    axes[0, 0].plot(T_fine, combined_fine, label="NSS + spline")
    axes[0, 0].legend()
    axes[0, 0].set_title("Market curve vs NSS vs combined fit")
    axes[0, 0].set_xlabel("maturity (y)")
    axes[0, 0].set_ylabel("rate (%)")

    axes[0, 1].axhline(0, color="grey", linewidth=0.8)
    axes[0, 1].scatter(residuals["T"], residuals["residual"], label="residual", color="tab:red")
    axes[0, 1].plot(T_fine, resid_fn(T_fine), label="fitted spline", color="tab:blue")
    axes[0, 1].legend()
    axes[0, 1].set_title("Pillar-fit residuals + spline")
    axes[0, 1].set_xlabel("maturity (y)")

    axes[1, 0].bar([str(t) for t in EXAMPLE_BUCKET_T], EXAMPLE_BUCKET_DELTA, color="tab:purple")
    axes[1, 0].set_title("Example trader bucket delta ($/bp)")
    axes[1, 0].set_xlabel("maturity (y)")

    labels = ["level", "slope", "curvature", "curvature2"]
    values = [factor_delta[f"delta_{label}"] for label in labels]
    axes[1, 1].bar(labels, values, color="tab:green")
    axes[1, 1].set_title("Bucket delta projected onto NSS factors")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "summary.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
