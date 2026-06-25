"""Express the book's exposure in level/slope/curvature and vanna-volga-native
terms once the surface is SVI-fit, instead of projecting onto statistical PCA
loadings -- the structural model's own parameters are already the factors.
"""

import numpy as np
import pandas as pd

import svi
from data_synthetic import NS_TAU, ns_loadings


def term_structure_factors(cleaned_panel, tau=NS_TAU):
    tenor_order = cleaned_panel[["tenor_label", "T"]].drop_duplicates().sort_values("T")
    tenor_labels = tenor_order["tenor_label"].to_numpy()
    T_grid = tenor_order["T"].to_numpy()

    slope_load, curv_load = ns_loadings(T_grid, tau)
    design = np.column_stack([np.ones_like(T_grid), slope_load, curv_load])
    loadings = pd.DataFrame(design, columns=["level", "slope", "curvature"], index=tenor_labels)

    rows = []
    for date, g in cleaned_panel.groupby("date"):
        g = g.set_index("tenor_label").loc[tenor_labels]
        atm_var = svi.raw_svi_total_variance(
            0.0, g["a"].to_numpy(), g["b"].to_numpy(), g["rho"].to_numpy(), g["m"].to_numpy(), g["sigma"].to_numpy()
        )
        sigma_atm = np.sqrt(np.maximum(atm_var, 1e-12) / T_grid)
        coef, *_ = np.linalg.lstsq(design, sigma_atm, rcond=None)
        rows.append({"date": date, "level": coef[0], "slope": coef[1], "curvature": coef[2]})

    return pd.DataFrame(rows), loadings


def project_portfolio(portfolio_ladder, loadings):
    vega = portfolio_ladder.set_index("tenor_label")["vega"].reindex(loadings.index)
    exposure = vega.to_numpy() @ loadings.to_numpy()
    return {
        "level_exposure": float(exposure[0]),
        "slope_exposure": float(exposure[1]),
        "curvature_exposure": float(exposure[2]),
    }


def skew_curvature_exposure(portfolio_ladder, cleaned_panel, as_of_date):
    g = cleaned_panel[cleaned_panel["date"] == as_of_date].set_index("tenor_label")
    rho_avg = g["rho"].mean()
    wing_avg = (g["b"] * g["sigma"]).mean()

    ladder = portfolio_ladder.set_index("tenor_label").reindex(g.index)
    return {
        "skew_exposure": float(ladder["vanna"].sum()) * rho_avg,
        "wing_exposure": float(ladder["volga"].sum()) * wing_avg,
        "rho_avg": float(rho_avg),
        "wing_scale_avg": float(wing_avg),
    }
