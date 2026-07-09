"""Synthetic forward/vol/bpv panels for testing the pipeline end-to-end.

Not real market data -- swap this out for real historical panels (same shape:
dates x (tenor, expiry) MultiIndex columns) when you're back on your other
laptop. Built with a common level + slope factor (so PCA has real structure to
find) plus idiosyncratic noise per cell, and an optional injected anomaly that
richens the 1Y expiry at the 30Y tenor over the last 10 days -- a synthetic
stand-in for the "1Y30Y looks rich" scenario from the strategy discussion, so
the demo signals have something real to flag instead of pure noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EXPIRY_YEARS = {"6M": 0.5, "1Y": 1.0, "2Y": 2.0, "3Y": 3.0, "5Y": 5.0}


def generate_synthetic_surface(n_days: int = 252,
                                tenors: tuple = (2, 5, 10, 30),
                                expiries: tuple = ("6M", "1Y", "2Y", "3Y", "5Y"),
                                seed: int = 0,
                                inject_anomaly: bool = True):
    """Returns (forward_panel, vol_panel, bpv_panel, expiry_years)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    cols = pd.MultiIndex.from_product([tenors, expiries], names=["tenor", "expiry"])

    # common factors every cell partially loads on -- this is the structure
    # PCA is supposed to find and denoise against.
    level = np.zeros(n_days)
    slope = np.zeros(n_days)
    for t in range(1, n_days):
        level[t] = 0.98 * level[t - 1] + rng.normal(0, 0.0006)
        slope[t] = 0.98 * slope[t - 1] + rng.normal(0, 0.0003)

    forward_panel = pd.DataFrame(index=dates, columns=cols, dtype=float)
    vol_panel = pd.DataFrame(index=dates, columns=cols, dtype=float)
    bpv_panel = pd.DataFrame(index=dates, columns=cols, dtype=float)

    for tenor in tenors:
        for expiry in expiries:
            T = EXPIRY_YEARS[expiry]
            base_fwd = 0.03 + 0.0005 * tenor
            base_vol = 0.007 + 0.0004 * np.sqrt(T) - 0.00002 * tenor

            idio_fwd = np.cumsum(rng.normal(0, 0.00015, n_days))
            idio_vol = np.cumsum(rng.normal(0, 0.00003, n_days))

            fwd_loading = 1.0 + 0.05 * tenor / 30
            vol_loading = 1.0 + 0.1 * T / 5

            forward_panel[(tenor, expiry)] = base_fwd + level * fwd_loading * 0.5 + idio_fwd
            vol_panel[(tenor, expiry)] = base_vol + slope * vol_loading + idio_vol
            bpv_panel[(tenor, expiry)] = tenor * 0.95 + rng.normal(0, 0.02, n_days)

    vol_panel = vol_panel.clip(lower=0.001)
    bpv_panel = bpv_panel.clip(lower=0.1)

    if inject_anomaly and 30 in tenors and "1Y" in expiries:
        vol_panel.loc[vol_panel.index[-10]:, (30, "1Y")] += np.linspace(0, 0.0015, 10)

    return forward_panel, vol_panel, bpv_panel, EXPIRY_YEARS
