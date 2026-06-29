"""Stage B of the pipeline: a small smoothing spline (or low-degree
polynomial) fit on the pillar-fit residuals, used to soak up the local
market wiggles the 4-parameter NSS can't represent. NSS fit + this residual
model together form the actual market-curve interpolator.
"""

import numpy as np
from scipy.interpolate import UnivariateSpline

import nss_model


def fit_residual_spline(T, residual, k=3, s=None):
    T = np.asarray(T, dtype=float)
    residual = np.asarray(residual, dtype=float)
    order = np.argsort(T)
    T_sorted, r_sorted = T[order], residual[order]

    k = min(k, len(T_sorted) - 1)
    if s is None:
        s = len(T_sorted) * np.var(r_sorted) * 0.5
    return UnivariateSpline(T_sorted, r_sorted, k=k, s=s)


def fit_residual_poly(T, residual, degree=3):
    coeffs = np.polyfit(np.asarray(T, dtype=float), np.asarray(residual, dtype=float), degree)
    return np.poly1d(coeffs)


def fit_residual_adjustment(T, residual, method="spline", **kwargs):
    if method == "spline":
        return fit_residual_spline(T, residual, **kwargs)
    if method == "poly":
        return fit_residual_poly(T, residual, **kwargs)
    raise ValueError(f"unknown method {method!r}, expected 'spline' or 'poly'")


def combined_curve(T, beta, lambda1, lambda2, residual_fn):
    T = np.asarray(T, dtype=float)
    return nss_model.nss_rate(T, beta, lambda1, lambda2) + residual_fn(T)
