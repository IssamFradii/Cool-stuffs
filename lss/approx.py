"""
Fast closed-form swaption approximation (paper footnote 18).

The paper notes that resimulating the whole term structure at every optimiser
iteration is unnecessary:

  "a least squares algorithm similar to Longstaff and Schwartz (2000) can be
   used to approximate forward swap rates as linear functions of the individual
   forward rates.  Given a covariance matrix, this linear approximation then
   implies closed-form expressions for the variance of individual forward swap
   rates at the expiration dates of the swaptions, which can then be used to
   provide a closed-form approximation to the value of the swaption.  This
   closed-form approximation can then be corrected for bias by an iterative
   process of comparing the simulated values given by the string market model
   to those implied by this approximation ...  With this type of algorithm,
   solving for the implied eigenvalues typically takes less than 10 seconds."

Two ingredients.

1. Integrated covariance under time homogeneity
   -------------------------------------------
   Sigma is indexed by *distance to becoming spot*: Sigma[i-1, j-1] is the
   annualised covariance between a forward i six-month periods from becoming
   spot and one j periods away.  A forward that is i periods away at time zero
   is i-k periods away at time t_k = k*delta.  So the covariance of the log
   changes of forwards i and j accumulated over [0, tau], tau = n*delta, is

       IntCov(i, j, n) = delta * sum_{k=0}^{n-1} Sigma[i-k-1, j-k-1]

   This single expression drives both swaptions and caps.  Its diagonal case
   reproduces the paper's cap prescription exactly: the variance of the caplet
   resetting at tau = i*delta is IntCov(i,i,i)/tau = mean of the first i
   diagonal elements of Sigma (Section 5.4).

2. Forward swap rate as a basket of forwards
   ------------------------------------------
   FSR(0,tau,T) is a smooth function of the forwards, so

       Var(ln FSR) ~= (1/FSR^2) sum_{i,j} w_i w_j F_i F_j IntCov(i,j,n)

   with w_i = dFSR/dF_i computed analytically.  Feeding that variance into the
   at-the-money-forward Black formula (equation 7) gives the approximate price.

The approximation is accurate to a fraction of a percent, and the residual bias
is removed by :class:`BiasCorrection`, which is refreshed against a full
simulation a handful of times during a calibration.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .conventions import DELTA, N_FORWARDS
from .curve import Curve
from .mathutil import norm_cdf
from . import smm


# ---------------------------------------------------------------------------
# Integrated covariance
# ---------------------------------------------------------------------------
def integrated_covariance(sigma: np.ndarray, n_steps: int) -> np.ndarray:
    """
    The matrix of covariances accumulated over [0, n_steps * delta].

    Returns a (19, 19) array `C` where `C[i-1, j-1]` is IntCov(i, j, n_steps)
    for forwards labelled by their time-zero distance-to-spot i, j = 1..19.
    Entries for which a forward would already have become the spot rate
    (i <= n_steps - 1) are left at zero; they are never used, because a
    swaption expiring at tau only involves forwards starting at or after tau.
    """
    sigma = np.asarray(sigma, dtype=float)
    n = N_FORWARDS
    out = np.zeros((n, n))
    for k in range(n_steps):
        # forward labelled i sits at index i-k; valid while i-k >= 1
        sub = sigma[: n - k, : n - k]
        out[k:, k:] += sub
    return out * DELTA


def caplet_variances(sigma: np.ndarray) -> np.ndarray:
    """
    Annualised Black variances for the caplets resetting at 0.5, 1.0, ..., 9.5.

    Element i-1 corresponds to the caplet resetting at tau = i * delta and is

        (1/i) * sum_{m=1}^{i} Sigma[m-1, m-1]

    i.e. the average of the first i elements of the main diagonal -- exactly
    the rule stated in Section 5.4: "the variance for the caplet maturing in
    six months is the first diagonal element, the variance for the caplet
    maturing in twelve months is the average of the first and second diagonal
    elements, etc."
    """
    diag = np.diag(np.asarray(sigma, dtype=float))
    return np.cumsum(diag) / np.arange(1, len(diag) + 1)


# ---------------------------------------------------------------------------
# Forward swap rate sensitivities
# ---------------------------------------------------------------------------
def forward_swap_weights(curve: Curve, expiry: float, tenor: float) -> np.ndarray:
    """
    w_i = d FSR(0, tau, T) / d F_i, for the 19 forwards.

    Using D(0, m*delta) = D(0, delta) * prod_{l=1}^{m-1} 1/(1 + delta F_l),

        d D(0, m*delta) / d F_i = -D(0, m*delta) * delta / (1 + delta F_i)
                                  for 1 <= i <= m-1, else 0

    and then differentiating FSR = (D_tau - D_T) / (delta * A).
    """
    end = expiry + tenor
    f = curve.forwards
    n = len(f)

    def d_bond(maturity: float) -> np.ndarray:
        """Gradient of D(0, maturity) with respect to the forwards."""
        m = int(round(maturity / DELTA))          # maturity = m * delta
        g = np.zeros(n)
        if m <= 1:
            return g                               # D(0, delta) is fixed
        idx = np.arange(1, m)                      # forwards 1 .. m-1
        g[idx - 1] = -curve.D(maturity) * DELTA / (1.0 + DELTA * f[idx - 1])
        return g

    n_pay = int(round(tenor / DELTA))
    pay_dates = [expiry + (p + 1) * DELTA for p in range(n_pay)]

    annuity = curve.annuity(expiry, end)
    d_annuity = np.sum([d_bond(t) for t in pay_dates], axis=0)

    d_tau = d_bond(expiry)
    d_end = d_bond(end)

    numerator = curve.D(expiry) - curve.D(end)
    fsr = numerator / (DELTA * annuity)

    return (d_tau - d_end) / (DELTA * annuity) - fsr * d_annuity / annuity


# ---------------------------------------------------------------------------
# The approximation
# ---------------------------------------------------------------------------
def approx_swaption_vol(
    curve: Curve, sigma: np.ndarray, expiry: float, tenor: float
) -> float:
    """
    Approximate Black volatility of the forward swap rate FSR(0, tau, T)
    implied by the covariance matrix Sigma.
    """
    n_steps = int(round(expiry / DELTA))
    cov = integrated_covariance(sigma, n_steps)

    w = forward_swap_weights(curve, expiry, tenor)
    f = curve.forwards
    fsr = curve.forward_swap_rate(expiry, expiry + tenor)

    elasticity = w * f                       # dFSR / dln F_i
    total_var = float(elasticity @ cov @ elasticity) / (fsr * fsr)
    return float(np.sqrt(max(total_var, 0.0) / expiry))


def approx_swaption_price(
    curve: Curve, sigma: np.ndarray, expiry: float, tenor: float
) -> float:
    """
    Approximate ATMF European swaption price, equation (7) evaluated at the
    approximate volatility.
    """
    vol = approx_swaption_vol(curve, sigma, expiry, tenor)
    end = expiry + tenor
    return (curve.D(expiry) - curve.D(end)) * (
        2.0 * norm_cdf(vol * np.sqrt(expiry) / 2.0) - 1.0
    )


class SwaptionApproximator:
    """
    Precomputed, curve-dependent pieces of the swaption approximation.

    The elasticities dFSR/dln F_i, the forward swap rates and the annuity
    levels all depend on the *curve only* -- not on Sigma.  Inside a
    calibration the curve is fixed and Sigma changes at every optimiser
    iteration, so hoisting these out of the loop is the difference between a
    calibration that takes a second and one that takes a minute.

    Prices are grouped by expiry so the integrated covariance is built once per
    distinct expiry rather than once per swaption.
    """

    def __init__(self, curve: Curve, grid: list[tuple[float, float]]):
        self.curve = curve
        self.grid = list(grid)

        f = curve.forwards
        self.elasticities = np.array(
            [forward_swap_weights(curve, e, t) * f for e, t in self.grid]
        )
        self.fsr = np.array(
            [curve.forward_swap_rate(e, e + t) for e, t in self.grid]
        )
        self.level = np.array(
            [curve.D(e) - curve.D(e + t) for e, t in self.grid]
        )
        self.expiry = np.array([e for e, _ in self.grid])
        self.sqrt_expiry = np.sqrt(self.expiry)

        # Group swaption indices by their expiry.
        self.by_expiry: dict[float, np.ndarray] = {}
        for e in sorted({e for e, _ in self.grid}):
            self.by_expiry[e] = np.array(
                [k for k, (ke, _) in enumerate(self.grid) if ke == e]
            )

    def volatilities(self, sigma: np.ndarray) -> np.ndarray:
        """Approximate Black volatilities for the whole grid."""
        sigma = np.asarray(sigma, dtype=float)
        vols = np.empty(len(self.grid))
        for expiry, idx in self.by_expiry.items():
            cov = integrated_covariance(sigma, int(round(expiry / DELTA)))
            e = self.elasticities[idx]
            total_var = np.einsum("ki,ij,kj->k", e, cov, e) / (self.fsr[idx] ** 2)
            vols[idx] = np.sqrt(np.maximum(total_var, 0.0) / expiry)
        return vols

    def prices(self, sigma: np.ndarray) -> np.ndarray:
        """Approximate ATMF swaption prices, equation (7)."""
        vols = self.volatilities(sigma)
        return self.level * (
            2.0 * norm_cdf(vols * self.sqrt_expiry / 2.0) - 1.0
        )


def approx_swaption_grid(
    curve: Curve,
    sigma: np.ndarray,
    grid: list[tuple[float, float]],
    approximator: SwaptionApproximator | None = None,
) -> np.ndarray:
    """
    Approximate prices for a whole swaption grid.

    Pass a prebuilt :class:`SwaptionApproximator` to avoid recomputing the
    curve-dependent terms; without one they are rebuilt on every call.
    """
    approximator = approximator or SwaptionApproximator(curve, grid)
    return approximator.prices(sigma)


# ---------------------------------------------------------------------------
# Bias correction against the full simulation
# ---------------------------------------------------------------------------
@dataclass
class BiasCorrection:
    """
    Multiplicative correction factors mapping approximate prices to simulated
    ones, one per swaption:

        factor_k = P_simulated_k / P_approx_k

    Refreshed at the current best-guess Sigma.  Because the ratio varies far
    more slowly in Sigma than the prices themselves, a handful of refreshes
    over a calibration is enough, and the optimiser then works entirely on the
    cheap closed form.
    """

    factors: np.ndarray
    n_refreshes: int = 0

    @classmethod
    def identity(cls, n: int) -> "BiasCorrection":
        return cls(factors=np.ones(n))

    def apply(self, approx_prices: np.ndarray) -> np.ndarray:
        return approx_prices * self.factors

    def refresh(
        self,
        curve: Curve,
        sigma: np.ndarray,
        grid: list[tuple[float, float]],
        config: "smm.SimulationConfig | None" = None,
        damping: float = 1.0,
        approximator: "SwaptionApproximator | None" = None,
    ) -> "BiasCorrection":
        """
        Re-estimate the factors by simulating at `sigma`.

        `damping` in (0, 1] blends the new factors with the old ones, which
        keeps the objective function from jumping between optimiser restarts.
        """
        result = smm.simulate(curve, sigma, config)
        simulated = smm.price_swaption_grid(result, curve, grid)
        approximate = approx_swaption_grid(curve, sigma, grid, approximator)

        with np.errstate(divide="ignore", invalid="ignore"):
            new = simulated / approximate
        new[~np.isfinite(new)] = 1.0
        new = np.clip(new, 0.5, 2.0)

        blended = (1.0 - damping) * self.factors + damping * new
        return BiasCorrection(factors=blended, n_refreshes=self.n_refreshes + 1)


def corrected_swaption_grid(
    curve: Curve,
    sigma: np.ndarray,
    grid: list[tuple[float, float]],
    correction: BiasCorrection | None = None,
    approximator: SwaptionApproximator | None = None,
) -> np.ndarray:
    """Approximate prices with the bias correction applied."""
    prices = approx_swaption_grid(curve, sigma, grid, approximator)
    return prices if correction is None else correction.apply(prices)
