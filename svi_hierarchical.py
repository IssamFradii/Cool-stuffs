"""Two-stage hierarchical pooling of per-slice raw-SVI parameters.

Stage A (fit_raw_panel) calibrates each (day, tenor) slice independently.
Stage B (fit_hierarchical_svi) treats those point estimates as noisy
observations of a smooth latent two-way (day x tenor) structure and shrinks
them with a Student-t likelihood, robust to dislocated slices without having
to flag them by hand first. Doing the nonlinear per-slice fit first and the
hierarchical pooling second (rather than one giant joint nonlinear model) is
what keeps this tractable to sample.
"""

import numpy as np
import pandas as pd
import pymc as pm

import svi

PARAM_NAMES = ["log_theta", "log_b", "rho_z", "m", "log_sigma"]


def _atm_theta(a, b, rho, m, sigma):
    """Total variance at k=0. Unlike 'a' alone, this is tenor-comparable --
    'a' is only the ATM level when m=0, and m drifts away from 0 as tenor
    grows (since SSVI's phi=eta/theta shrinks at long T), so pooling raw 'a'
    hierarchically pools a quantity that isn't the same thing at every tenor."""
    return a - b * rho * m + b * np.sqrt(m ** 2 + sigma ** 2)


def fit_raw_panel(market_long, tenor_labels):
    dates = list(market_long["date"].unique())
    date_index = {d: i for i, d in enumerate(dates)}
    tenor_index = {t: i for i, t in enumerate(tenor_labels)}

    rows = []
    for (date, tenor_label), g in market_long.groupby(["date", "tenor_label"], sort=False):
        T = g["T"].iloc[0]
        k = g["k"].to_numpy()
        iv = g["iv"].to_numpy()
        seed = date_index[date] * 1000 + tenor_index[tenor_label]
        seed_params = svi.cem_seed_slice(k, iv, T, seed=seed)
        fit = svi.calibrate_slice(k, iv, T, seed_params=seed_params)
        fit.update(date=date, tenor_label=tenor_label, T=T)
        rows.append(fit)
    return pd.DataFrame(rows)


def transform_params(raw_panel):
    a = raw_panel["a"].to_numpy()
    b = np.clip(raw_panel["b"].to_numpy(), 1e-6, None)
    rho = np.clip(raw_panel["rho"].to_numpy(), -0.999, 0.999)
    m = raw_panel["m"].to_numpy()
    sigma = np.clip(raw_panel["sigma"].to_numpy(), 1e-6, None)
    theta_atm = np.clip(_atm_theta(a, b, rho, m, sigma), 1e-8, None)
    return {
        "log_theta": np.log(theta_atm),
        "log_b": np.log(b),
        "rho_z": np.arctanh(rho),
        "m": m,
        "log_sigma": np.log(sigma),
    }


def inverse_transform_params(means):
    b = np.exp(means["log_b"])
    rho = np.tanh(means["rho_z"])
    m = means["m"]
    sigma = np.exp(means["log_sigma"])
    theta_atm = np.exp(means["log_theta"])
    a = theta_atm + b * rho * m - b * np.sqrt(m ** 2 + sigma ** 2)
    return {"a": a, "b": b, "rho": rho, "m": m, "sigma": sigma}


def fit_hierarchical_svi(raw_panel, draws=600, tune=600, chains=2, seed=7):
    dates = list(raw_panel["date"].unique())
    tenor_order = raw_panel[["tenor_label", "T"]].drop_duplicates().sort_values("T")
    tenor_labels = tenor_order["tenor_label"].tolist()
    day_idx = pd.Categorical(raw_panel["date"], categories=dates).codes
    tenor_idx = pd.Categorical(raw_panel["tenor_label"], categories=tenor_labels).codes
    n_days, n_tenors = len(dates), len(tenor_labels)

    transformed = transform_params(raw_panel)

    with pm.Model() as model:
        day_idx_data = pm.Data("day_idx", day_idx)
        tenor_idx_data = pm.Data("tenor_idx", tenor_idx)

        cleaned = {}
        for name in PARAM_NAMES:
            values = transformed[name]
            scale = float(np.std(values) + 1e-3)

            global_mean = pm.Normal(f"{name}_global", mu=float(np.mean(values)), sigma=2.0 * scale)
            day_sigma = pm.HalfNormal(f"{name}_day_sigma", sigma=scale)
            tenor_sigma = pm.HalfNormal(f"{name}_tenor_sigma", sigma=scale)

            day_raw = pm.Normal(f"{name}_day_raw", mu=0.0, sigma=1.0, shape=n_days)
            # Random walk *ordered by tenor* rather than iid categories: nearby
            # tenors shrink toward each other, so boundary tenors (1m, 10y) are
            # pulled by their one neighbor instead of an unrelated cross-sectional
            # mean -- iid tenor effects biased the longest tenor systematically.
            tenor_raw = pm.GaussianRandomWalk(
                f"{name}_tenor_raw", sigma=1.0, init_dist=pm.Normal.dist(0.0, 1.0), shape=n_tenors
            )
            day_effect = day_raw * day_sigma
            tenor_effect = tenor_raw * tenor_sigma

            mu = global_mean + day_effect[day_idx_data] + tenor_effect[tenor_idx_data]
            obs_sigma = pm.HalfNormal(f"{name}_obs_sigma", sigma=scale)
            pm.StudentT(f"{name}_obs", nu=4, mu=mu, sigma=obs_sigma, observed=values)

            cleaned[name] = pm.Deterministic(f"{name}_clean", mu)

        idata = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed, progressbar=False)

    means = {name: idata.posterior[f"{name}_clean"].mean(dim=["chain", "draw"]).values for name in PARAM_NAMES}
    cleaned_values = inverse_transform_params(means)

    cleaned_panel = raw_panel[["date", "tenor_label", "T"]].copy()
    for key, arr in cleaned_values.items():
        cleaned_panel[key] = arr
    return cleaned_panel, idata


def extract_noise_scales(idata):
    """Posterior-mean day-to-day and within-day observation noise scales,
    per transformed parameter -- reused by sequential_bayes.py so the fast
    recursive filter inherits noise levels the batch model already learned,
    instead of re-estimating them from scratch."""
    day_sigma = {name: float(idata.posterior[f"{name}_day_sigma"].mean()) for name in PARAM_NAMES}
    obs_sigma = {name: float(idata.posterior[f"{name}_obs_sigma"].mean()) for name in PARAM_NAMES}
    return day_sigma, obs_sigma
