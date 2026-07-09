"""Generic PCA fit/reconstruct via SVD -- no sklearn dependency.

Used to denoise a historical panel (dates x cells): fit the top-k common
factors across history, reconstruct what those factors alone would predict
for every date, and let the caller take (actual - reconstruction) as the
residual that the common factors don't explain.
"""

from __future__ import annotations

import numpy as np


def pca_fit(panel: np.ndarray, n_components: int):
    """panel: (n_dates, n_cells). Returns (mean, components, explained_variance).

    components has shape (n_components, n_cells): each row is one principal
    direction across the cells, fitted on how the cells co-moved historically.
    """
    panel = np.asarray(panel, dtype=float)
    mean = panel.mean(axis=0)
    centered = panel - mean
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    n_components = min(n_components, Vt.shape[0])
    components = Vt[:n_components]
    explained_variance = (S ** 2 / max(len(panel) - 1, 1))[:n_components]
    return mean, components, explained_variance


def pca_reconstruct(panel: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Project `panel` onto `components` and reconstruct -- the "denoised"
    surface implied by only the top-k common factors, same shape as `panel`.
    """
    panel = np.asarray(panel, dtype=float)
    centered = panel - mean
    scores = centered @ components.T
    return scores @ components + mean
