"""
The String Market Model (Section 3.1) and its Monte-Carlo implementation
(Section 3.2).

Dynamics
--------
Each of the 19 six-month Libor forwards follows, under the risk-neutral
measure (equation 8):

    dF_i = alpha_i F_i dt + sigma_i F_i dZ_i

with the dZ_i *correlated across forwards* -- this is the "string" feature.
The covariance of dF_i/F_i and dF_j/F_j is collected in the 19x19
time-homogeneous matrix Sigma.

Rather than modelling the awkward drifts alpha_i, the model is implemented on
discount bond prices (equation 11):

    dD = r D dt + J^-1 sigma F dZ

Because discount bonds are traded assets, their risk-neutral drift is simply
the short rate -- the alpha_i never appear.  J is the Jacobian of the map from
bond prices to forwards.

Discretisation
--------------
Steps are six months, matching the semiannual fixed leg of the swaps and the
assumption that covariances are constant over six-month intervals.  The
discrete drift is applied exactly:

    E_k[ D(t+delta, T) ] = D(t, T) / D(t, t+delta)

which equals `r D dt` with r the simple yield on the shortest bond (the
paper's discretisation choice, footnote 13) and makes the numeraire-deflated
bonds exact martingales -- so the discrete model is arbitrage free, not merely
arbitrage free in the continuous-time limit.

Time homogeneity / roll-down
----------------------------
After six months the first forward becomes the spot rate.  The surviving
18 forwards have horizons 0.5 .. 9.0, so the relevant covariance matrix is the
leading 18x18 block of Sigma, and the Jacobian loses its first row and column.
This repeats each step until one forward remains (Section 3.2).

A note on the Jacobian of equation (11)
---------------------------------------
The matrix J printed in the paper omits the accrual factor: its entries are
-D(.50)/D^2(1.00) etc., whereas differentiating equation (10),
F_i = (1/delta)[D(T_i)/D(T_i+delta) - 1], gives those entries multiplied by
1/delta.  The printed matrix is therefore `delta * J_true`.  Using it as
written would scale every shock by delta = 1/2 and halve all model
volatilities.  This implementation uses the true Jacobian; the test suite
verifies the scaling directly by checking that the realised variance of
dF/F over one step equals Sigma_ii * delta, and that a rank-one Sigma
reproduces the Black caplet price.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .conventions import DELTA, N_FORWARDS, grid_index
from .curve import Curve
from .historical import nearest_psd
from .instruments import spot_swap_value_from_discounts


# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SimulationConfig:
    """
    Monte-Carlo settings.

    Parameters
    ----------
    n_paths : int
        Total number of paths.  With antithetic variates this must be even;
        the paper uses 2,000.
    seed : int
        The same seed is reused at every optimiser iteration so that the
        objective function stays differentiable in the eigenvalues
        (Section 3.2).
    antithetic : bool
        Antithetic variates to reduce simulation noise, as in the paper.
    max_steps : int
        Number of six-month steps to simulate.  Swaption expiries run to five
        years, so ten steps suffice for the calibration set.
    floor_discounts : bool
        Euler steps on bond prices can in principle produce a non-monotone or
        negative discount curve.  When True such curves are repaired (each
        bond floored below its predecessor) and counted in
        `SimulationResult.n_repairs`, rather than silently producing negative
        forwards.
    """

    n_paths: int = 2000
    seed: int = 12345
    antithetic: bool = True
    max_steps: int = 10
    floor_discounts: bool = True

    def __post_init__(self) -> None:
        if self.antithetic and self.n_paths % 2 != 0:
            raise ValueError("n_paths must be even when using antithetic variates")
        if self.max_steps < 1 or self.max_steps > 19:
            raise ValueError("max_steps must be between 1 and 19")


@dataclass
class SimulationResult:
    """
    Output of a string market model simulation.

    Attributes
    ----------
    states : list of arrays
        `states[k]` has shape (n_paths, 20 - k) and holds
        D(t_k, t_k + 0.5), ..., D(t_k, 10.0) for every path, where t_k = k/2.
        `states[0]` is the deterministic initial curve, broadcast.
    numeraire : (n_steps+1, n_paths) array
        `numeraire[k]` is the money-market discount factor back to time zero,
        prod_{j<k} D(t_j, t_j + 1/2), used to discount time-t_k cash flows.
    n_repairs : int
        Number of (path, step) discount curves that required repair.
    """

    states: list[np.ndarray]
    numeraire: np.ndarray
    n_repairs: int = 0

    @property
    def n_paths(self) -> int:
        return self.states[0].shape[0]

    def discounts_at(self, expiry: float) -> np.ndarray:
        """
        D(tau, tau + i/2) for i = 1, 2, ... at a grid expiry `tau`.
        Shape (n_paths, 20 - k) with k = tau / 0.5.
        """
        k = int(round(expiry / DELTA))
        if k >= len(self.states):
            raise ValueError(
                f"expiry {expiry} beyond simulated horizon "
                f"({(len(self.states) - 1) * DELTA} years)"
            )
        return self.states[k]

    def numeraire_at(self, expiry: float) -> np.ndarray:
        """Money-market discount factor from `tau` back to time zero."""
        k = int(round(expiry / DELTA))
        return self.numeraire[k]


# ---------------------------------------------------------------------------
# The Jacobian and its inverse
# ---------------------------------------------------------------------------
def jacobian_bands(discounts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    The two non-zero bands of the Jacobian J = dF/dD.

    `discounts` has shape (n_paths, m+1) and holds
    d[0] = D(t, t+delta)  (the non-stochastic short bond), then the m
    stochastic bonds d[1..m].

    Differentiating F_r+1 = (1/delta) [ d[r]/d[r+1] - 1 ]:

        J[r, r]     = -(1/delta) * d[r] / d[r+1]^2      (diagonal)
        J[r, r-1]   =  (1/delta) / d[r+1]               (sub-diagonal, r >= 1)

    Returns
    -------
    diag : (n_paths, m) array
    sub  : (n_paths, m) array, with sub[:, 0] unused (set to zero)
    """
    d = np.asarray(discounts, dtype=float)
    lead = d[:, :-1]    # d[r]   for r = 0..m-1
    trail = d[:, 1:]    # d[r+1] for r = 0..m-1

    diag = -(1.0 / DELTA) * lead / (trail * trail)
    sub = (1.0 / DELTA) / trail
    sub[:, 0] = 0.0
    return diag, sub


def solve_jacobian(diag: np.ndarray, sub: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """
    Solve J x = rhs for the lower-bidiagonal J, vectorised across paths.

    Forward substitution:
        x[0] = rhs[0] / diag[0]
        x[r] = ( rhs[r] - sub[r] * x[r-1] ) / diag[r]
    """
    m = rhs.shape[1]
    x = np.empty_like(rhs)
    x[:, 0] = rhs[:, 0] / diag[:, 0]
    for r in range(1, m):
        x[:, r] = (rhs[:, r] - sub[:, r] * x[:, r - 1]) / diag[:, r]
    return x


def forwards_from_state(discounts: np.ndarray) -> np.ndarray:
    """
    The m surviving six-month forwards implied by a state vector of m+1
    discount factors.  Shape (n_paths, m).
    """
    d = np.asarray(discounts, dtype=float)
    return (d[:, :-1] / d[:, 1:] - 1.0) / DELTA


# ---------------------------------------------------------------------------
# The simulation
# ---------------------------------------------------------------------------
def _shock_factor(sigma_block: np.ndarray, tol: float = 1e-14) -> np.ndarray:
    """
    A factor B with B B' = sigma_block, keeping only the strictly positive
    eigenvalues.  The implied Sigma is deliberately low rank (rank 4 of 19),
    so this returns an (m, rank) matrix and we draw only `rank` normals per
    path per step.
    """
    block = 0.5 * (sigma_block + sigma_block.T)
    vals, vecs = np.linalg.eigh(block)
    keep = vals > tol
    if not np.any(keep):
        return np.zeros((block.shape[0], 0))
    return vecs[:, keep] * np.sqrt(vals[keep])


def simulate(
    curve: Curve,
    sigma: np.ndarray,
    config: SimulationConfig | None = None,
) -> SimulationResult:
    """
    Simulate the term structure under the string market model.

    Parameters
    ----------
    curve : Curve
        The initial (time-zero) term structure.  The model fits it exactly.
    sigma : (19, 19) array
        The *annualised* covariance matrix of proportional changes in the
        forwards, time homogeneous: sigma[i-1, j-1] is the covariance between
        a forward i periods from becoming spot and one j periods from becoming
        spot.  Normally Sigma = U Psi U' from the factor basis.
    config : SimulationConfig, optional

    Returns
    -------
    SimulationResult
    """
    config = config or SimulationConfig()
    sigma = np.asarray(sigma, dtype=float)
    if sigma.shape != (N_FORWARDS, N_FORWARDS):
        raise ValueError(f"sigma must be {N_FORWARDS}x{N_FORWARDS}, got {sigma.shape}")
    sigma = nearest_psd(sigma)

    rng = np.random.default_rng(config.seed)
    n_paths = config.n_paths
    n_draw = n_paths // 2 if config.antithetic else n_paths

    # Step 0 state: the full initial curve D(0, 0.5) ... D(0, 10.0), 20 entries.
    state = np.broadcast_to(curve.discounts, (n_paths, len(curve.discounts))).copy()

    states = [state]
    numeraire = np.ones((config.max_steps + 1, n_paths))
    n_repairs = 0

    for k in range(config.max_steps):
        m = state.shape[1] - 1           # number of stochastic bonds / forwards
        if m < 1:
            break

        short = state[:, 0]              # D(t_k, t_k + delta), known at t_k
        stoch = state[:, 1:]             # the m stochastic bonds

        # --- money-market numeraire ---------------------------------------
        numeraire[k + 1] = numeraire[k] * short

        # --- correlated proportional shocks -------------------------------
        # z ~ N(0, Sigma_k * delta), Sigma_k the leading m x m block.
        factor = _shock_factor(sigma[:m, :m] * DELTA)
        rank = factor.shape[1]
        if rank > 0:
            w = rng.standard_normal((n_draw, rank))
            if config.antithetic:
                w = np.concatenate([w, -w], axis=0)
            z = w @ factor.T
        else:
            z = np.zeros((n_paths, m))

        # --- the diffusion term J^-1 (sigma F dZ) -------------------------
        fwd = forwards_from_state(state)         # (n_paths, m)
        rhs = fwd * z                            # sigma_i F_i dZ_i
        diag, sub = jacobian_bands(state)
        increment = solve_jacobian(diag, sub, rhs)

        # --- the exact discrete drift -------------------------------------
        # The shock is applied *before* dividing by the short bond.  Forwards
        # are homogeneous of degree zero in the bond prices, so the drift
        # leaves them unchanged; but the Jacobian was evaluated at the
        # undrifted state, so its linearisation F(d + x) is only exact if x is
        # added to d rather than to d/D(t,t+delta).  Adding after the division
        # would scale every shock by D(t,t+delta) and understate all model
        # volatilities by roughly that factor squared.  Taking the drift
        # outside preserves the exact martingale property either way, since
        # E[increment] = 0.
        new_state = (stoch + increment) / short[:, None]

        # --- guard against a degenerate Euler step ------------------------
        if config.floor_discounts:
            new_state, repairs = _repair_curve(new_state)
            n_repairs += repairs

        state = new_state
        states.append(state)

    return SimulationResult(states=states, numeraire=numeraire, n_repairs=n_repairs)


def _repair_curve(state: np.ndarray, eps: float = 1e-10) -> tuple[np.ndarray, int]:
    """
    Enforce a positive, strictly decreasing discount curve.

    An Euler step on bond prices can in principle push a discount factor above
    its predecessor (a negative forward rate) or below zero.  Such paths are
    repaired by capping each bond just below the one before it.  The count is
    reported so the caller can tell whether the discretisation is straining.
    """
    repaired = state.copy()
    bad = 0

    first_bad = repaired[:, 0] <= eps
    if np.any(first_bad):
        repaired[first_bad, 0] = eps
        bad += int(first_bad.sum())

    for j in range(1, repaired.shape[1]):
        violate = repaired[:, j] >= repaired[:, j - 1] * (1.0 - 1e-12)
        if np.any(violate):
            repaired[violate, j] = repaired[violate, j - 1] * (1.0 - 1e-12)
            bad += int(violate.sum())
        low = repaired[:, j] <= eps
        if np.any(low):
            repaired[low, j] = eps
            bad += int(low.sum())

    return repaired, bad


# ---------------------------------------------------------------------------
# Pricing off a simulation
# ---------------------------------------------------------------------------
def price_swaption(
    result: SimulationResult,
    curve: Curve,
    expiry: float,
    tenor: float,
    coupon: float | None = None,
) -> float:
    """
    Value an at-the-money-forward European swaption by simulation
    (Section 3.2, equation 12).

    At the expiry date tau the simulated discount factors give the value of
    the underlying spot-starting swap

        V(tau, tau, T, c) = (c/2) sum_i D(tau, tau+i/2) + D(tau, T) - 1

    The receiver pays max(0, V), the payer max(0, -V).  Each is discounted by
    the money-market factor and averaged over paths; since ATMF receivers and
    payers are worth the same, the reported price is the average of the two,
    which also cancels part of the simulation noise.
    """
    end = expiry + tenor
    if coupon is None:
        coupon = curve.forward_swap_rate(expiry, end)

    d = result.discounts_at(expiry)              # D(tau, tau + i/2), i = 1..
    n_pay = int(round(tenor / DELTA))
    if n_pay > d.shape[1]:
        raise ValueError(
            f"swap to {end}y needs {n_pay} bonds but only {d.shape[1]} were simulated"
        )
    legs = d[:, :n_pay]

    swap_value = spot_swap_value_from_discounts(legs, coupon)

    disc = result.numeraire_at(expiry)
    receiver = np.mean(np.maximum(swap_value, 0.0) * disc)
    payer = np.mean(np.maximum(-swap_value, 0.0) * disc)
    return float(0.5 * (receiver + payer))


def price_swaption_grid(
    result: SimulationResult,
    curve: Curve,
    grid: list[tuple[float, float]],
) -> np.ndarray:
    """Price a list of (expiry, tenor) swaptions off one simulation."""
    return np.array([price_swaption(result, curve, e, t) for e, t in grid])


def martingale_diagnostic(result: SimulationResult, curve: Curve) -> np.ndarray:
    """
    Check the discrete no-arbitrage condition: for each step k and each
    surviving maturity T,

        E[ D(t_k, T) * numeraire_k ]  ==  D(0, T)

    Returns the array of relative errors.  These should be at simulation-noise
    level (order 1/sqrt(n_paths)); a systematic drift indicates a bug in the
    drift or Jacobian handling.
    """
    errors = []
    for k, state in enumerate(result.states):
        if k == 0:
            continue
        disc = result.numeraire[k]
        deflated = np.mean(state * disc[:, None], axis=0)
        t_k = k * DELTA
        target = np.array(
            [curve.D(t_k + (j + 1) * DELTA) for j in range(state.shape[1])]
        )
        errors.append((deflated - target) / target)
    return np.concatenate(errors) if errors else np.zeros(0)
