"""Sequential (day-by-day) Bayesian updating of the per-tenor SVI parameters
-- a lighter, faster stepping stone toward a full Kalman/state-space model.

Each (tenor, parameter) is its own scalar random walk: today's prior is
yesterday's posterior diffused by a process variance, updated against
today's Stage-A point estimate via the standard Gaussian conjugate update.
Noise scales are inherited from the batch hierarchical fit's own learned
day_sigma/obs_sigma (svi_hierarchical.extract_noise_scales) rather than
estimated again from scratch. Unlike the batch model, this never re-touches
history: each new day is one closed-form update per (tenor, parameter), so
it is the version that could plausibly run live, once per session.
"""

import numpy as np
import pandas as pd

from svi_hierarchical import PARAM_NAMES, inverse_transform_params, transform_params


def sequential_filter(raw_panel, day_sigma_by_param, obs_sigma_by_param, tenor_smooth=0.3):
    dates = sorted(raw_panel["date"].unique())
    tenor_order = raw_panel[["tenor_label", "T"]].drop_duplicates().sort_values("T")
    tenor_labels = tenor_order["tenor_label"].tolist()
    n_tenors = len(tenor_labels)
    tenor_pos = {t: i for i, t in enumerate(tenor_labels)}

    panel = raw_panel[["date", "tenor_label"]].copy()
    transformed = transform_params(raw_panel)
    for name in PARAM_NAMES:
        panel[name] = transformed[name]
    panel = panel.set_index(["date", "tenor_label"])

    state_mean = {name: np.full(n_tenors, np.nan) for name in PARAM_NAMES}
    state_var = {name: np.full(n_tenors, np.nan) for name in PARAM_NAMES}

    rows = []
    for date in dates:
        for name in PARAM_NAMES:
            process_var = day_sigma_by_param[name] ** 2
            obs_var = max(obs_sigma_by_param[name] ** 2, 1e-10)

            for tenor_label in tenor_labels:
                if (date, tenor_label) not in panel.index:
                    continue
                ti = tenor_pos[tenor_label]
                obs_value = panel.loc[(date, tenor_label), name]

                if np.isnan(state_mean[name][ti]):
                    state_mean[name][ti] = obs_value
                    state_var[name][ti] = obs_var
                    continue

                prior_mean = state_mean[name][ti]
                prior_var = state_var[name][ti] + process_var
                post_var = 1.0 / (1.0 / prior_var + 1.0 / obs_var)
                post_mean = post_var * (prior_mean / prior_var + obs_value / obs_var)

                state_mean[name][ti] = post_mean
                state_var[name][ti] = post_var

            if tenor_smooth > 0 and n_tenors > 2:
                vals = state_mean[name]
                smoothed = vals.copy()
                smoothed[1:-1] = (1 - tenor_smooth) * vals[1:-1] + tenor_smooth * 0.5 * (vals[:-2] + vals[2:])
                state_mean[name] = smoothed

        for tenor_label in tenor_labels:
            ti = tenor_pos[tenor_label]
            row = {"date": date, "tenor_label": tenor_label}
            for name in PARAM_NAMES:
                row[name] = state_mean[name][ti]
                row[f"{name}_var"] = state_var[name][ti]
            rows.append(row)

    return pd.DataFrame(rows)


def to_param_panel(filtered_long, tenor_to_T):
    means = {name: filtered_long[name].to_numpy() for name in PARAM_NAMES}
    values = inverse_transform_params(means)

    panel = filtered_long[["date", "tenor_label"]].copy()
    panel["T"] = panel["tenor_label"].map(tenor_to_T)
    for key, arr in values.items():
        panel[key] = arr
    return panel
