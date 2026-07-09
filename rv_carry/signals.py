"""The 4 RV carry signals, as specified:

1. raw calendar level        -> z-score against its own history
2. calendar / front-leg price ("DF"-normalized) -> z-score against its own history
3. PCA-denoise the raw calendar surface (all consecutive-expiry calendars for a
   tenor, across history) -> residual = actual - PCA reconstruction -> z-score
4. same as 3, but on the DF-normalized calendar surface

Each returns a single z-score for "today" (the last row of whatever panel/series
is passed in). Combine them however you like downstream (pipeline.py just
averages them).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .pca_tools import pca_fit, pca_reconstruct


def zscore(series: pd.Series, window: int | None = None) -> float:
    """Z-score of the series' last value against its own trailing history.

    window=None uses the full history; pass an int to use a rolling lookback
    instead (e.g. window=252 for the last year of daily data).
    """
    hist = series if window is None else series.iloc[-window:]
    mu, sd = hist.mean(), hist.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float((series.iloc[-1] - mu) / sd)


def signal_1_raw_carry_zscore(calendar: pd.Series, window: int | None = None) -> float:
    """Signal 1: z-score of the raw calendar level (price on spot, no adjustment)."""
    return zscore(calendar, window)


def signal_2_df_normalized_zscore(calendar: pd.Series, front_leg_price: pd.Series,
                                   window: int | None = None) -> float:
    """Signal 2: z-score of the calendar discounted by the front leg's own price
    (calendar / front-leg price) -- controls for the front leg's price level so
    the same vol-point calendar doesn't look bigger/smaller just because the
    front leg happens to be trading rich or cheap that day.
    """
    normalized = calendar / front_leg_price
    return zscore(normalized, window)


def signal_3_pca_residual_zscore(calendar_panel: pd.DataFrame, target_col: str,
                                  n_components: int = 2, window: int | None = None):
    """Signal 3: PCA-denoise the raw calendar surface (all consecutive-expiry
    calendars for one tenor, across history), take the residual of `target_col`
    (actual - what the top common factors alone would predict), z-score it.

    Returns (zscore, residual_series) -- the series is handy for plotting.
    """
    values = calendar_panel.values
    mean, components, _ = pca_fit(values, n_components)
    recon = pca_reconstruct(values, mean, components)
    j = calendar_panel.columns.get_loc(target_col)
    residual = pd.Series(values[:, j] - recon[:, j], index=calendar_panel.index)
    return zscore(residual, window), residual


def signal_4_pca_residual_df_zscore(normalized_panel: pd.DataFrame, target_col: str,
                                     n_components: int = 2, window: int | None = None):
    """Signal 4: identical mechanism to signal 3, run on the DF-normalized
    calendar surface instead of the raw one.
    """
    return signal_3_pca_residual_zscore(normalized_panel, target_col, n_components, window)
