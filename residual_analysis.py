"""Residual ('vol add-on') surface from the hierarchically-cleaned SVI fit,
and outlier flagging on top of it.
"""

import numpy as np
import pandas as pd

import svi


def compute_residuals(market_long, cleaned_panel):
    panel_idx = cleaned_panel.set_index(["date", "tenor_label"])

    rows = []
    for (date, tenor_label), g in market_long.groupby(["date", "tenor_label"], sort=False):
        params = panel_idx.loc[(date, tenor_label)]
        T = g["T"].iloc[0]
        k = g["k"].to_numpy()
        iv_fit = svi.raw_svi_iv(k, T, params["a"], params["b"], params["rho"], params["m"], params["sigma"])
        resid = g["iv"].to_numpy() - iv_fit
        for kk, ivm, ivf, r in zip(k, g["iv"].to_numpy(), iv_fit, resid):
            rows.append({
                "date": date, "tenor_label": tenor_label, "T": T, "k": kk,
                "iv_market": ivm, "iv_fit": ivf, "residual": r,
            })
    return pd.DataFrame(rows)


def flag_outliers(residual_df, z_thresh=3.0):
    out = residual_df.copy()
    grp = out.groupby(["tenor_label", "k"])["residual"]
    mu = grp.transform("mean")
    sigma = grp.transform("std").replace(0, np.nan)
    out["z"] = (out["residual"] - mu) / sigma
    out["is_outlier"] = out["z"].abs() >= z_thresh
    return out
