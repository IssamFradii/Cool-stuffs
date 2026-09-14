"""
The ex-ante historical correlation matrix and its spectral decomposition
(Section 3.2).

The identifying assumption of the paper is

    H = U Lambda U'          (historical correlation matrix, ex-ante period)
    Sigma = U Psi U'         (implied covariance matrix, Psi diagonal >= 0)

i.e. the *eigenvectors* are taken from history and held fixed, while the
*eigenvalues* are implied each week from swaption prices.  Economically: the
market prices swaptions using the same factors that drive the term structure
historically, but its view of each factor's variance differs from the
historical one.

This is equivalent to requiring that H and Sigma commute (footnote 16).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .conventions import N_FORWARDS
from . import paper_data


# ---------------------------------------------------------------------------
# Estimation from a forward-rate history
# ---------------------------------------------------------------------------
def correlation_from_forwards(forward_history: np.ndarray) -> np.ndarray:
    """
    Correlation matrix of weekly *log* changes in the 19 six-month forwards.

    Parameters
    ----------
    forward_history : (n_weeks, 19) array
        Forward rates in level terms (decimals), one row per weekly observation.

    Returns
    -------
    (19, 19) correlation matrix.
    """
    f = np.asarray(forward_history, dtype=float)
    if f.ndim != 2 or f.shape[1] != N_FORWARDS:
        raise ValueError(f"expected (n_weeks, {N_FORWARDS}) array, got {f.shape}")
    if np.any(f <= 0):
        raise ValueError("forward rates must be positive to take log changes")

    log_changes = np.diff(np.log(f), axis=0)
    return np.corrcoef(log_changes, rowvar=False)


def covariance_from_forwards(
    forward_history: np.ndarray, periods_per_year: float = 52.0
) -> np.ndarray:
    """
    Annualised covariance matrix of log changes in the forwards.

    The paper works with the historical *correlation* matrix to keep the
    scaling of the implied eigenvalues simple (footnote 15), but the covariance
    matrix is useful for comparison and for setting sensible starting values.
    """
    f = np.asarray(forward_history, dtype=float)
    log_changes = np.diff(np.log(f), axis=0)
    return np.cov(log_changes, rowvar=False) * periods_per_year


# ---------------------------------------------------------------------------
# Spectral decomposition
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FactorBasis:
    """
    The fixed factor basis of the model: eigenvectors of the ex-ante
    historical correlation matrix, ordered by descending eigenvalue.

    Attributes
    ----------
    eigenvectors : (19, 19) array
        Columns are the factors.  Column 0 is the level/parallel-shift factor,
        column 1 slope, column 2 curvature, column 3 the short-end factor
        (Figure 5).
    eigenvalues : (19,) array
        The corresponding *historical* eigenvalues.  Not used to price -- the
        implied eigenvalues replace them -- but useful as starting values and
        for the historical-vs-implied comparison of Tables 6 and 7.
    """

    eigenvectors: np.ndarray
    eigenvalues: np.ndarray

    def covariance(self, psi: np.ndarray) -> np.ndarray:
        """
        Build Sigma = U Psi U' from a vector of implied eigenvalues.

        `psi` may be shorter than 19, in which case the remaining eigenvalues
        are zero and the resulting Sigma has rank len(psi) -- this is exactly
        how the paper nests the N-factor models.
        """
        psi = np.asarray(psi, dtype=float)
        full = np.zeros(N_FORWARDS)
        full[: len(psi)] = psi
        return (self.eigenvectors * full) @ self.eigenvectors.T

    def rank(self, psi: np.ndarray) -> int:
        """Number of strictly positive eigenvalues in `psi`."""
        return int(np.sum(np.asarray(psi) > 0))


def spectral_decomposition(correlation: np.ndarray) -> FactorBasis:
    """
    H = U Lambda U', with eigenvalues sorted in descending order and
    eigenvector signs fixed so that the largest-magnitude loading of each
    factor is positive (otherwise the sign is arbitrary and plots flip).
    """
    h = np.asarray(correlation, dtype=float)
    h = 0.5 * (h + h.T)  # enforce exact symmetry

    eigenvalues, eigenvectors = np.linalg.eigh(h)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    for k in range(eigenvectors.shape[1]):
        dominant = np.argmax(np.abs(eigenvectors[:, k]))
        if eigenvectors[dominant, k] < 0:
            eigenvectors[:, k] *= -1.0

    return FactorBasis(eigenvectors=eigenvectors, eigenvalues=eigenvalues)


def paper_factor_basis() -> FactorBasis:
    """
    The factor basis of the paper: eigenvectors of the Table 2 ex-ante
    correlation matrix.  This is the default basis for the whole pipeline and
    needs no data.
    """
    return spectral_decomposition(paper_data.historical_correlation())


# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------
def to_correlation(covariance: np.ndarray) -> np.ndarray:
    """
    Standardise a covariance matrix into a correlation matrix.

    Rows/columns with zero variance (possible when Sigma is low rank) are
    returned as NaN rather than dividing by zero.
    """
    cov = np.asarray(covariance, dtype=float)
    sd = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(sd, sd)
    corr[~np.isfinite(corr)] = np.nan
    return corr


def nearest_psd(matrix: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """
    Project a symmetric matrix onto the positive semidefinite cone by clipping
    its eigenvalues at `floor`.  Used to guard the simulation against tiny
    negative eigenvalues from floating-point error.
    """
    m = 0.5 * (matrix + matrix.T)
    vals, vecs = np.linalg.eigh(m)
    vals = np.clip(vals, floor, None)
    return (vecs * vals) @ vecs.T
