"""
Solving for the implied covariance matrix (Section 3.2).

The identifying assumption is Sigma = U Psi U', with U the eigenvectors of the
ex-ante historical correlation matrix (fixed) and Psi a diagonal matrix of
non-negative implied eigenvalues (solved for, each week).

  "Since there are typically far more swaptions than eigenvalues, we solve for
   the implied eigenvalues by standard numerical optimization where the
   objective function is the root mean squared error (RMSE) of the percentage
   differences between the market price and the model price, taken over all
   swaptions."

An N-factor model is nested by solving for the first N eigenvalues and setting
the remaining 19 - N to zero, which is how the likelihood ratio tests of
Table 5 compare model ranks.

Two engines
-----------
``method="approx"``  (default) uses the closed-form approximation of
:mod:`lss.approx`, periodically bias-corrected against a full simulation.
This is the accelerated scheme of footnote 18 and takes well under a second
per week.

``method="simulation"`` re-simulates the term structure at every objective
evaluation, exactly as the body of Section 3.2 describes.  It is far slower
but is the reference implementation, and the test suite checks that the two
agree.

In both cases the same random seed is reused at every iteration, "to preserve
the differentiability of the objective function with respect to the
eigenvalue".
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .conventions import SWAPTION_GRID, N_FORWARDS
from .curve import Curve
from .historical import FactorBasis, paper_factor_basis, to_correlation
from .mathutil import least_squares, LeastSquaresResult
from . import approx, smm


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class CalibrationResult:
    """
    Outcome of fitting the string market model to one week of swaption prices.

    Attributes
    ----------
    psi : (n_factors,) array
        The implied eigenvalues -- the implied variances of the term structure
        factors.  These are what Figure 6 plots through time.
    sigma : (19, 19) array
        The implied covariance matrix U Psi U'.
    model_prices, market_prices : (34,) arrays
    percentage_errors : (34,) array
        100 * (model - market) / market.  These are the pricing errors
        summarised in Table 8.
    rmse : float
        Root mean squared percentage error, the objective value.  This is the
        series plotted in Figure 7.
    """

    psi: np.ndarray
    sigma: np.ndarray
    model_prices: np.ndarray
    market_prices: np.ndarray
    percentage_errors: np.ndarray
    rmse: float
    n_factors: int
    converged: bool
    n_evaluations: int
    message: str = ""
    grid: list[tuple[float, float]] = field(default_factory=lambda: list(SWAPTION_GRID))

    @property
    def sum_squared_errors(self) -> float:
        """Sum of squared *fractional* pricing errors, the Table 5 input."""
        return float(np.sum((self.percentage_errors / 100.0) ** 2))

    @property
    def implied_correlation(self) -> np.ndarray:
        """The implied correlation matrix, for Tables 6 and 7."""
        return to_correlation(self.sigma)

    @property
    def forward_vols(self) -> np.ndarray:
        """
        Implied annualised volatility of each forward in its i-th-from-last
        six-month period, i.e. sqrt of the main diagonal of Sigma.
        """
        return np.sqrt(np.clip(np.diag(self.sigma), 0.0, None))


# ---------------------------------------------------------------------------
# Starting values
# ---------------------------------------------------------------------------
def starting_eigenvalues(
    basis: FactorBasis, n_factors: int, target_vol: float = 0.15
) -> np.ndarray:
    """
    A sensible initial Psi.

    Because H is a *correlation* matrix, U Lambda U' has unit diagonal, so
    scaling the historical eigenvalues by v^2 produces a covariance matrix
    whose forwards all have volatility v.  Truncating to N factors lowers that
    somewhat, which is fine for a starting point.
    """
    return basis.eigenvalues[:n_factors] * target_vol * target_vol


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------
def _residuals_factory(
    curve: Curve,
    market_prices: np.ndarray,
    grid: list[tuple[float, float]],
    basis: FactorBasis,
    method: str,
    correction: approx.BiasCorrection | None,
    sim_config: smm.SimulationConfig | None,
    approximator: approx.SwaptionApproximator | None = None,
):
    """
    Build the residual function r(psi) = (model - market) / market.

    The eigenvalues are optimised *directly*, under a lower bound of zero,
    rather than through a squared reparameterisation.  Writing Psi = theta^2
    looks tidier but has a vanishing derivative at the origin: once a factor
    collapses to zero the Gauss-Newton step can never revive it, and with
    warm starts that stuck state propagates from week to week.  The paper
    notes the fourth implied eigenvalue "is often quite small and can actually
    be zero" (Section 5.1), so the optimiser has to be able to sit at zero and
    still leave it again.
    """

    def residuals(psi: np.ndarray) -> np.ndarray:
        psi = np.clip(np.asarray(psi, dtype=float), 0.0, None)
        sigma = basis.covariance(psi)
        if method == "approx":
            model = approx.corrected_swaption_grid(
                curve, sigma, grid, correction, approximator
            )
        else:
            result = smm.simulate(curve, sigma, sim_config)
            model = smm.price_swaption_grid(result, curve, grid)
        return (model - market_prices) / market_prices

    return residuals


# ---------------------------------------------------------------------------
# The calibration
# ---------------------------------------------------------------------------
def calibrate(
    curve: Curve,
    market_prices: np.ndarray,
    n_factors: int = 4,
    grid: list[tuple[float, float]] | None = None,
    basis: FactorBasis | None = None,
    method: str = "approx",
    sim_config: smm.SimulationConfig | None = None,
    n_refreshes: int = 3,
    psi0: np.ndarray | None = None,
    max_iter: int = 100,
    cascade: bool = True,
    verbose: bool = False,
) -> CalibrationResult:
    """
    Solve for the implied eigenvalues that best fit one week of swaption prices.

    Parameters
    ----------
    curve : Curve
        That week's term structure.
    market_prices : (34,) array
        Market swaption premia, obtained by putting the quoted Black vols
        through equation (7).
    n_factors : int
        Rank of the implied covariance matrix.  The paper settles on four.
    method : {"approx", "simulation"}
        See the module docstring.
    n_refreshes : int
        Number of bias-correction refreshes (``method="approx"`` only).  Each
        refresh costs one full simulation.
    cascade : bool
        Build the rank up one factor at a time, each rank starting from the
        solution of the one below.  The objective is not convex in Psi, and a
        cold start at rank four can settle into a worse optimum than rank
        three -- which would corrupt the incremental likelihood ratio tests of
        Table 5, since those compare sums of squared errors across ranks.
        Cascading guarantees the fit is monotonically non-worsening in the
        number of factors.  Ignored when `psi0` is supplied.

    Returns
    -------
    CalibrationResult
    """
    if method not in ("approx", "simulation"):
        raise ValueError("method must be 'approx' or 'simulation'")

    grid = list(grid) if grid is not None else list(SWAPTION_GRID)
    basis = basis or paper_factor_basis()
    sim_config = sim_config or smm.SimulationConfig()

    market_prices = np.asarray(market_prices, dtype=float)
    if market_prices.shape != (len(grid),):
        raise ValueError(
            f"expected {len(grid)} market prices, got {market_prices.shape}"
        )
    if np.any(market_prices <= 0):
        raise ValueError("market swaption prices must be positive")

    if n_factors < 1 or n_factors > N_FORWARDS:
        raise ValueError(f"n_factors must be between 1 and {N_FORWARDS}")

    if psi0 is not None:
        psi = np.asarray(psi0, dtype=float)
        if len(psi) < n_factors:
            # Warm start from a lower rank: seed the new factors small.
            psi = np.concatenate(
                [psi, np.full(n_factors - len(psi), 1e-4)]
            )
        psi = psi[:n_factors]
    elif cascade and n_factors > 1:
        lower = calibrate(
            curve,
            market_prices,
            n_factors=n_factors - 1,
            grid=grid,
            basis=basis,
            method=method,
            sim_config=sim_config,
            n_refreshes=n_refreshes,
            max_iter=max_iter,
            cascade=True,
        )
        psi = np.concatenate([lower.psi, [1e-4]])
    else:
        psi = starting_eigenvalues(basis, n_factors)

    psi = np.clip(psi, 0.0, None)

    correction = (
        approx.BiasCorrection.identity(len(grid)) if method == "approx" else None
    )
    approximator = (
        approx.SwaptionApproximator(curve, grid) if method == "approx" else None
    )

    total_evals = 0
    fit: LeastSquaresResult | None = None
    rounds = n_refreshes if method == "approx" else 1

    for round_index in range(max(rounds, 1)):
        if method == "approx":
            correction = correction.refresh(
                curve, basis.covariance(psi), grid, sim_config,
                approximator=approximator,
            )

        residuals = _residuals_factory(
            curve, market_prices, grid, basis, method, correction, sim_config,
            approximator,
        )
        fit = least_squares(
            residuals, psi, lower=0.0, max_iter=max_iter, verbose=verbose
        )
        psi = fit.x
        total_evals += fit.n_evaluations

        if verbose:
            print(
                f"round {round_index}: rmse {100 * fit.rmse:.3f}%  ({fit.message})"
            )

    psi = np.clip(psi, 0.0, None)
    sigma = basis.covariance(psi)

    # Always report the *simulated* prices, so the reported errors are the
    # model's, not the approximation's.
    result = smm.simulate(curve, sigma, sim_config)
    model_prices = smm.price_swaption_grid(result, curve, grid)
    pct = 100.0 * (model_prices - market_prices) / market_prices

    return CalibrationResult(
        psi=psi,
        sigma=sigma,
        model_prices=model_prices,
        market_prices=market_prices,
        percentage_errors=pct,
        rmse=float(np.sqrt(np.mean((pct / 100.0) ** 2))),
        n_factors=n_factors,
        converged=bool(fit.converged) if fit else False,
        n_evaluations=total_evals,
        message=fit.message if fit else "",
        grid=grid,
    )


def calibrate_series(
    curves: list[Curve],
    market_prices: np.ndarray,
    n_factors: int = 4,
    grid: list[tuple[float, float]] | None = None,
    basis: FactorBasis | None = None,
    method: str = "approx",
    sim_config: smm.SimulationConfig | None = None,
    warm_start: bool = True,
    progress: bool = False,
) -> list[CalibrationResult]:
    """
    Calibrate week by week over a whole sample.

    Parameters
    ----------
    curves : list of Curve, length n_weeks
    market_prices : (n_weeks, 34) array
    warm_start : bool
        Start each week from the previous week's solution.  Implied
        eigenvalues are highly persistent (Figure 6), so this roughly halves
        the iteration count without changing the optimum.

    Returns
    -------
    list of CalibrationResult, one per week.
    """
    grid = list(grid) if grid is not None else list(SWAPTION_GRID)
    basis = basis or paper_factor_basis()
    market_prices = np.asarray(market_prices, dtype=float)

    if market_prices.shape != (len(curves), len(grid)):
        raise ValueError(
            f"expected market prices of shape ({len(curves)}, {len(grid)}), "
            f"got {market_prices.shape}"
        )

    results: list[CalibrationResult] = []
    psi0 = None
    for week, curve in enumerate(curves):
        res = calibrate(
            curve,
            market_prices[week],
            n_factors=n_factors,
            grid=grid,
            basis=basis,
            method=method,
            sim_config=sim_config,
            psi0=psi0 if warm_start else None,
        )
        results.append(res)
        if warm_start:
            psi0 = res.psi
        if progress and (week + 1) % 10 == 0:
            print(f"  calibrated {week + 1}/{len(curves)} weeks", flush=True)

    return results


# ---------------------------------------------------------------------------
# Market prices from quoted volatilities
# ---------------------------------------------------------------------------
def swaption_prices_from_vols(
    curve: Curve, vols: np.ndarray, grid: list[tuple[float, float]] | None = None
) -> np.ndarray:
    """
    Convert quoted ATMF Black volatilities into premia using equation (7) --
    the market convention described in Section 4.
    """
    from .instruments import black_swaption_atmf

    grid = list(grid) if grid is not None else list(SWAPTION_GRID)
    vols = np.asarray(vols, dtype=float)
    return np.array(
        [black_swaption_atmf(curve, e, t, v) for (e, t), v in zip(grid, vols)]
    )
