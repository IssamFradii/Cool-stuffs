"""
Numerical primitives, implemented in pure numpy.

The package deliberately depends on numpy alone.  Everything the pipeline
needs from a wider scientific stack -- the normal distribution, a natural
cubic spline, and a bounded least-squares optimiser -- lives here.  That keeps
the model reproducible and immune to environment problems with compiled
extensions.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import math
import numpy as np

_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------------
# The standard normal distribution
# ---------------------------------------------------------------------------
_erf = np.vectorize(math.erf, otypes=[float])


def norm_cdf(x):
    """Standard normal CDF, via the error function."""
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=float) / _SQRT2))


def norm_pdf(x):
    """Standard normal density."""
    x = np.asarray(x, dtype=float)
    return _INV_SQRT_2PI * np.exp(-0.5 * x * x)


# Acklam's rational approximation to the inverse normal CDF, refined by one
# Halley step against norm_cdf.  Accurate to roughly machine precision.
_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00)
_P_LOW = 0.02425
_P_HIGH = 1.0 - _P_LOW


def norm_ppf(p):
    """Inverse standard normal CDF.  Accepts scalars or arrays."""
    scalar = np.isscalar(p) or (np.ndim(p) == 0)
    p = np.atleast_1d(np.asarray(p, dtype=float))
    x = np.empty_like(p)

    lower = p < _P_LOW
    upper = p > _P_HIGH
    middle = ~(lower | upper)

    if np.any(lower):
        q = np.sqrt(-2.0 * np.log(p[lower]))
        x[lower] = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
                   ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    if np.any(upper):
        q = np.sqrt(-2.0 * np.log(1.0 - p[upper]))
        x[upper] = -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
                    ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    if np.any(middle):
        q = p[middle] - 0.5
        r = q * q
        x[middle] = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / \
                    (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)

    # One Halley refinement.
    with np.errstate(over="ignore", invalid="ignore"):
        e = norm_cdf(x) - p
        u = e / np.maximum(norm_pdf(x), 1e-300)
        x = x - u / (1.0 + 0.5 * x * u)

    x[p <= 0.0] = -np.inf
    x[p >= 1.0] = np.inf
    return float(x[0]) if scalar else x


# ---------------------------------------------------------------------------
# Natural cubic spline
# ---------------------------------------------------------------------------
class CubicSpline:
    """
    Natural cubic spline interpolation (second derivative zero at both ends),
    which is the standard choice for par-curve interpolation.

    Matches scipy.interpolate.CubicSpline(..., bc_type="natural") to machine
    precision.  Extrapolates with the end cubics.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.ndim != 1 or y.shape != x.shape:
            raise ValueError("x and y must be 1-D arrays of equal length")
        if np.any(np.diff(x) <= 0):
            raise ValueError("x must be strictly increasing")
        n = len(x)
        if n < 3:
            raise ValueError("need at least three knots for a cubic spline")

        h = np.diff(x)

        # Solve the tridiagonal system for the second derivatives (moments).
        lower = np.zeros(n)
        diag = np.ones(n)
        upper = np.zeros(n)
        rhs = np.zeros(n)

        lower[1:-1] = h[:-1]
        diag[1:-1] = 2.0 * (h[:-1] + h[1:])
        upper[1:-1] = h[1:]
        rhs[1:-1] = 6.0 * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1])

        m = _solve_tridiagonal(lower, diag, upper, rhs)

        self.x = x
        self.y = y
        self.h = h
        self.m = m

    def __call__(self, xi) -> np.ndarray:
        xi = np.asarray(xi, dtype=float)
        idx = np.clip(np.searchsorted(self.x, xi) - 1, 0, len(self.x) - 2)

        x0 = self.x[idx]
        x1 = self.x[idx + 1]
        y0 = self.y[idx]
        y1 = self.y[idx + 1]
        m0 = self.m[idx]
        m1 = self.m[idx + 1]
        h = self.h[idx]

        a = x1 - xi
        b = xi - x0
        return (
            (m0 * a ** 3 + m1 * b ** 3) / (6.0 * h)
            + (y0 / h - m0 * h / 6.0) * a
            + (y1 / h - m1 * h / 6.0) * b
        )


def _solve_tridiagonal(
    lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Thomas algorithm for a tridiagonal system."""
    n = len(diag)
    c = np.zeros(n)
    d = np.zeros(n)

    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for i in range(1, n):
        denom = diag[i] - lower[i] * c[i - 1]
        c[i] = upper[i] / denom if i < n - 1 else 0.0
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / denom

    out = np.zeros(n)
    out[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        out[i] = d[i] - c[i] * out[i + 1]
    return out


# ---------------------------------------------------------------------------
# Bounded Levenberg-Marquardt least squares
# ---------------------------------------------------------------------------
@dataclass
class LeastSquaresResult:
    """Outcome of a Levenberg-Marquardt fit."""

    x: np.ndarray
    residuals: np.ndarray
    cost: float           # sum of squared residuals
    n_iterations: int
    n_evaluations: int
    converged: bool
    message: str

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean(self.residuals ** 2)))


def least_squares(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    lower: np.ndarray | float = -np.inf,
    upper: np.ndarray | float = np.inf,
    max_iter: int = 200,
    ftol: float = 1e-12,
    xtol: float = 1e-12,
    diff_step: float = 1e-6,
    verbose: bool = False,
) -> LeastSquaresResult:
    """
    Minimise 0.5 * sum(residual_fn(x)^2) subject to box constraints, using
    Levenberg-Marquardt with a forward-difference Jacobian and simple
    projection onto the box.

    The calibration problem here is small (at most seven parameters against
    thirty-four residuals) and the residual function is smooth once the
    Monte-Carlo seed is fixed, so LM is both fast and reliable.
    """
    x = np.clip(np.asarray(x0, dtype=float).copy(), lower, upper)
    n = len(x)

    r = np.asarray(residual_fn(x), dtype=float)
    cost = float(r @ r)
    n_eval = 1
    lam = 1e-3
    message = "maximum iterations reached"
    converged = False

    for iteration in range(max_iter):
        # Forward-difference Jacobian, with steps that respect the box.
        jac = np.empty((len(r), n))
        for j in range(n):
            step = diff_step * max(1.0, abs(x[j]))
            xj = x.copy()
            if np.isfinite(np.atleast_1d(upper)[0] if np.ndim(upper) else upper):
                hi = upper[j] if np.ndim(upper) else upper
            else:
                hi = np.inf
            if xj[j] + step > hi:
                step = -step
            xj[j] += step
            rj = np.asarray(residual_fn(xj), dtype=float)
            n_eval += 1
            jac[:, j] = (rj - r) / step

        jtj = jac.T @ jac
        jtr = jac.T @ r

        improved = False
        for _ in range(30):
            try:
                delta = np.linalg.solve(
                    jtj + lam * np.diag(np.maximum(np.diag(jtj), 1e-12)), -jtr
                )
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue

            x_new = np.clip(x + delta, lower, upper)
            r_new = np.asarray(residual_fn(x_new), dtype=float)
            n_eval += 1
            cost_new = float(r_new @ r_new)

            if cost_new < cost:
                step_norm = float(np.linalg.norm(x_new - x))
                rel_gain = (cost - cost_new) / max(cost, 1e-300)
                x, r, cost = x_new, r_new, cost_new
                lam = max(lam * 0.3, 1e-12)
                improved = True
                if rel_gain < ftol:
                    converged, message = True, "cost tolerance reached"
                elif step_norm < xtol:
                    converged, message = True, "step tolerance reached"
                break

            lam *= 10.0
            if lam > 1e12:
                break

        if verbose:
            print(f"  iter {iteration:3d}  cost {cost:.6e}  lambda {lam:.2e}")

        if converged:
            break
        if not improved:
            converged, message = True, "no further improvement"
            break

    return LeastSquaresResult(
        x=x,
        residuals=r,
        cost=cost,
        n_iterations=iteration + 1,
        n_evaluations=n_eval,
        converged=converged,
        message=message,
    )


# ---------------------------------------------------------------------------
# Time-series helpers used by the empirics
# ---------------------------------------------------------------------------
def serial_correlation(series: np.ndarray, lag: int = 1) -> float:
    """First-order (or lag-k) autocorrelation, ignoring NaNs."""
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) <= lag + 1:
        return float("nan")
    a, b = s[:-lag], s[lag:]
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / denom) if denom > 0 else float("nan")


def ar1_adjusted_tstat(series: np.ndarray) -> tuple[float, float]:
    """
    Mean and t-statistic corrected for first-order serial correlation, as used
    throughout Tables 8-10 ("adjusted for first-order serial correlation").

    With an AR(1) coefficient rho the variance of the sample mean is inflated
    by (1 + rho) / (1 - rho), so the standard error is scaled accordingly.
    """
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n < 3:
        return float("nan"), float("nan")

    mean = float(s.mean())
    rho = serial_correlation(s, 1)
    if not np.isfinite(rho):
        rho = 0.0
    rho = float(np.clip(rho, -0.99, 0.99))

    variance = float(s.var(ddof=1))
    inflation = (1.0 + rho) / (1.0 - rho)
    se = math.sqrt(max(variance * inflation / n, 1e-300))
    return mean, mean / se if se > 0 else float("nan")


def chi2_sf(x: float, df: int) -> float:
    """
    Upper tail probability of a chi-square distribution, via the regularised
    incomplete gamma function.  Used for the Table 5 likelihood ratio tests.
    """
    if x <= 0:
        return 1.0
    return float(_gammaincc(df / 2.0, x / 2.0))


def _gammaincc(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x)."""
    if x < a + 1.0:
        return 1.0 - _gamma_series(a, x)
    return _gamma_continued_fraction(a, x)


def _gamma_series(a: float, x: float, itmax: int = 1000, eps: float = 1e-14) -> float:
    """Series expansion for the regularised lower incomplete gamma P(a, x)."""
    if x <= 0:
        return 0.0
    ap = a
    total = 1.0 / a
    term = total
    for _ in range(itmax):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * eps:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_continued_fraction(
    a: float, x: float, itmax: int = 1000, eps: float = 1e-14
) -> float:
    """Lentz's continued fraction for the regularised upper gamma Q(a, x)."""
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, itmax + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
