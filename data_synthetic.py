"""Synthetic SSVI vol surface panel with known ground truth, plus a synthetic
portfolio Greeks ladder with a deliberately known factor tilt -- used to
validate the rest of the pipeline before any real data is plugged in.
"""

import numpy as np
import pandas as pd

import svi

TENORS = {
    "1m": 1 / 12, "2m": 2 / 12, "3m": 0.25, "6m": 0.5, "9m": 0.75,
    "1y": 1.0, "2y": 2.0, "3y": 3.0, "5y": 5.0, "7y": 7.0, "10y": 10.0,
}
K_GRID = np.linspace(-0.5, 0.5, 9)
BASE_ATM_VOL = 0.20
NS_TAU = 2.0


def ns_loadings(T, tau=NS_TAU):
    x = T / tau
    slope = (1 - np.exp(-x)) / x
    curv = slope - np.exp(-x)
    return slope, curv


def _ar1_path(n, mu, phi, sigma, rng):
    x = np.empty(n)
    x[0] = mu
    for t in range(1, n):
        x[t] = mu + phi * (x[t - 1] - mu) + rng.normal(0, sigma)
    return x


def simulate_surface(n_days=60, seed=7, n_outliers=6, noise_std=0.0025):
    rng = np.random.default_rng(seed)
    tenor_labels = list(TENORS.keys())
    T_grid = np.array([TENORS[t] for t in tenor_labels])
    slope_load, curv_load = ns_loadings(T_grid)

    level = _ar1_path(n_days, mu=0.0, phi=0.985, sigma=0.0030, rng=rng)
    slope = _ar1_path(n_days, mu=0.0, phi=0.970, sigma=0.0018, rng=rng)
    curvature = _ar1_path(n_days, mu=0.0, phi=0.950, sigma=0.0010, rng=rng)
    rho_t = np.clip(_ar1_path(n_days, mu=-0.35, phi=0.97, sigma=0.02, rng=rng), -0.85, 0.85)
    eta_t = np.clip(_ar1_path(n_days, mu=1.0, phi=0.97, sigma=0.04, rng=rng), 0.3, None)

    dates = pd.bdate_range("2025-01-02", periods=n_days)

    outlier_days = rng.choice(n_days, size=n_outliers, replace=False)
    outlier_tenors = rng.choice(len(tenor_labels), size=n_outliers, replace=True)
    outlier_set = set(zip(outlier_days.tolist(), outlier_tenors.tolist()))

    rows = []
    for di, date in enumerate(dates):
        sigma_atm = np.maximum(BASE_ATM_VOL + level[di] + slope[di] * slope_load + curvature[di] * curv_load, 0.02)
        theta = (sigma_atm ** 2) * T_grid
        for ti, (tenor_label, T) in enumerate(zip(tenor_labels, T_grid)):
            w = svi.ssvi_total_variance(K_GRID, theta[ti], rho_t[di], eta_t[di])
            iv = np.sqrt(np.maximum(w, 1e-8) / T) + rng.normal(0, noise_std, size=K_GRID.shape)

            is_outlier_slice = (di, ti) in outlier_set
            if is_outlier_slice:
                bump = np.zeros_like(iv)
                center = len(K_GRID) // 2
                bump[center - 1:center + 2] += rng.choice([-1, 1]) * rng.uniform(0.03, 0.06)
                iv = iv + bump

            for kk, ivv in zip(K_GRID, iv):
                rows.append({
                    "date": date, "tenor_label": tenor_label, "T": T, "k": float(kk),
                    "iv": float(ivv), "is_true_outlier": bool(is_outlier_slice),
                })

    market_long = pd.DataFrame(rows)
    truth = pd.DataFrame({
        "date": dates, "level": level, "slope": slope, "curvature": curvature,
        "rho": rho_t, "eta": eta_t,
    })
    return market_long, truth, tenor_labels


def synthetic_portfolio(tenor_labels):
    """Deliberate known tilt: long front-end vega / short belly / long back
    (a calendar-fly shape), short front skew (vanna), long mid-curve wings
    (volga) -- recovered later by portfolio_exposure.py as a sanity check.
    """
    vega = {
        "1m": 120_000, "2m": 90_000, "3m": 60_000, "6m": 20_000, "9m": -10_000,
        "1y": -40_000, "2y": -60_000, "3y": -30_000, "5y": 10_000, "7y": 30_000, "10y": 50_000,
    }
    vanna = {
        "1m": -8_000, "2m": -6_000, "3m": -4_000, "6m": -1_000, "9m": 1_000,
        "1y": 3_000, "2y": 5_000, "3y": 6_000, "5y": 4_000, "7y": 2_000, "10y": 1_000,
    }
    volga = {
        "1m": 2_000, "2m": 3_000, "3m": 4_000, "6m": 5_000, "9m": 4_500,
        "1y": 4_000, "2y": 3_000, "3y": 2_000, "5y": 1_000, "7y": 500, "10y": 200,
    }
    return pd.DataFrame([
        {"tenor_label": t, "vega": vega[t], "vanna": vanna[t], "volga": volga[t]}
        for t in tenor_labels
    ])
