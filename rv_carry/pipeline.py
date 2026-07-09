"""Orchestrates the 4 signals for one calendar (tenor, expiry_front, expiry_back)
into a single summary. This is the entry point the notebook calls.
"""

from __future__ import annotations

import numpy as np

from .calendars import calendar_series
from .signals import (
    signal_1_raw_carry_zscore,
    signal_2_df_normalized_zscore,
    signal_3_pca_residual_zscore,
    signal_4_pca_residual_df_zscore,
)


def run_rv_carry_pipeline(price_panel, tenor, expiry_front, expiry_back,
                           calendar_panel_for_tenor, normalized_panel_for_tenor,
                           n_components: int = 2, window: int | None = None) -> dict:
    """Run all 4 signals for the calendar (tenor, expiry_back - expiry_front).

    price_panel                : full ATM price surface (dates x (tenor, expiry))
    calendar_panel_for_tenor   : output of calendars.calendar_grid for this tenor
    normalized_panel_for_tenor : output of calendars.normalized_calendar_grid for this tenor
    """
    target_col = f"{expiry_back}-{expiry_front}"

    cal = calendar_series(price_panel, tenor, expiry_front, expiry_back)
    front_leg = price_panel[(tenor, expiry_front)]

    s1 = signal_1_raw_carry_zscore(cal, window)
    s2 = signal_2_df_normalized_zscore(cal, front_leg, window)
    s3, resid3 = signal_3_pca_residual_zscore(calendar_panel_for_tenor, target_col, n_components, window)
    s4, resid4 = signal_4_pca_residual_df_zscore(normalized_panel_for_tenor, target_col, n_components, window)

    scores = np.array([s1, s2, s3, s4])
    return {
        "tenor": tenor,
        "calendar": f"{expiry_back}{tenor}-{expiry_front}{tenor}",
        "signal_1_raw_zscore": s1,
        "signal_2_df_zscore": s2,
        "signal_3_pca_residual_zscore": s3,
        "signal_4_pca_residual_df_zscore": s4,
        "composite_score": float(scores.mean()),
        "signals_agree": bool(np.all(scores > 0) or np.all(scores < 0)),
        "_residual_3": resid3,
        "_residual_4": resid4,
    }
