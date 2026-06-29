"""Offline fallback market curve: a known NSS ground truth plus a couple of
planted local wiggles (richness/cheapness at specific tenors not on the
2/5/10/20 pillar grid) and noise -- lets the residual + spline stage be
validated against a known answer when there is no network access.
"""

import numpy as np
import pandas as pd

import nss_model

TENORS = {
    "3m": 0.25, "6m": 0.5, "1y": 1.0, "2y": 2.0, "3y": 3.0, "4y": 4.0, "5y": 5.0,
    "7y": 7.0, "10y": 10.0, "15y": 15.0, "20y": 20.0, "25y": 25.0, "30y": 30.0,
}

TRUE_BETA = np.array([4.2, -1.6, -2.2, 1.1])
TRUE_LAMBDA1 = 1.5
TRUE_LAMBDA2 = 5.0


def simulate_curve(seed=7, noise_std=0.02, wiggle_tenors=("7y", "15y"), wiggle_size=0.08):
    rng = np.random.default_rng(seed)
    tenor_labels = list(TENORS.keys())
    T = np.array([TENORS[t] for t in tenor_labels])

    base = nss_model.nss_rate(T, TRUE_BETA, TRUE_LAMBDA1, TRUE_LAMBDA2)

    wiggle = np.zeros_like(base)
    for wt in wiggle_tenors:
        wiggle[tenor_labels.index(wt)] += rng.choice([-1, 1]) * wiggle_size

    rate = base + wiggle + rng.normal(0, noise_std, size=base.shape)

    curve = pd.DataFrame({"tenor_label": tenor_labels, "T": T, "rate_pct": rate})
    truth = {
        "beta": TRUE_BETA, "lambda1": TRUE_LAMBDA1, "lambda2": TRUE_LAMBDA2,
        "wiggle_tenors": wiggle_tenors, "wiggle_size": wiggle_size,
    }
    return curve, truth
