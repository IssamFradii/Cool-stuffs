"""
Term structure construction (Section 4 of the paper).

Procedure exactly as described:

  "We first use the six-month and one-year Libor rates to solve for the
   six-month and one-year par rates.  We then use a standard cubic spline
   algorithm to interpolate the par curve at semiannual intervals.  Finally,
   we solve for six-month forward rates by bootstrapping the interpolated
   par curve."

Inputs are the market quotes available in the paper's data set:
    6M Libor, 1Y Libor  (money-market, simple act/360 style)
    2Y, 3Y, 4Y, 5Y, 7Y, 10Y midmarket par swap rates (semiannual fixed leg)

Following market convention (footnote 19) the swap curve is used as if it
were the riskless discount curve.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .mathutil import CubicSpline
from .conventions import (
    DELTA, N_FORWARDS, N_STEPS, BOND_MATURITIES, FORWARD_STARTS,
    SWAP_PILLARS, LIBOR_PILLARS, grid_index,
)


# ---------------------------------------------------------------------------
# Libor -> par rate
# ---------------------------------------------------------------------------
def libor_to_par(libor_6m: float, libor_1y: float) -> tuple[float, float]:
    """
    Convert the 6M and 1Y money-market Libor rates into equivalent par
    (semiannual bond-basis) rates, so they can sit on the same par curve as
    the quoted swap rates.

    A 6M Libor deposit has a single cash flow, so the 6M discount factor is
        D(0, .5) = 1 / (1 + delta * L_6m)
    and the 6M par rate is just the rate that prices a one-payment par bond,
    which is L_6m itself expressed on the same delta accrual.

    A 1Y Libor deposit pays once at 1Y, so
        D(0, 1) = 1 / (1 + 2*delta * L_1y)
    and the 1Y par rate c solves  c*delta*(D(0,.5) + D(0,1)) + D(0,1) = 1.
    """
    d_half = 1.0 / (1.0 + DELTA * libor_6m)
    d_one = 1.0 / (1.0 + 2.0 * DELTA * libor_1y)
    par_half = libor_6m
    par_one = (1.0 - d_one) / (DELTA * (d_half + d_one))
    return par_half, par_one


# ---------------------------------------------------------------------------
# Par curve interpolation + bootstrap
# ---------------------------------------------------------------------------
def interpolate_par_curve(
    pillar_maturities: np.ndarray,
    pillar_par_rates: np.ndarray,
) -> np.ndarray:
    """
    Cubic-spline interpolate the par curve onto the semiannual grid
    0.5, 1.0, ..., 10.0.  Returns an array of 20 par rates.
    """
    spline = CubicSpline(pillar_maturities, pillar_par_rates)
    return spline(BOND_MATURITIES)


def bootstrap_discounts(par_rates: np.ndarray) -> np.ndarray:
    """
    Bootstrap discount factors from a semiannual par curve.

    For a par swap/bond of maturity T = n*delta paying coupon c semiannually,
        c * delta * sum_{k=1}^{n} D(0, k*delta)  +  D(0, T) = 1
    so
        D(0, T) = (1 - c*delta*A_{n-1}) / (1 + c*delta)
    where A_{n-1} is the annuity over the first n-1 payment dates.
    """
    par_rates = np.asarray(par_rates, dtype=float)
    if par_rates.shape != (N_STEPS,):
        raise ValueError(f"expected {N_STEPS} par rates, got {par_rates.shape}")

    discounts = np.empty(N_STEPS)
    annuity = 0.0
    for n in range(N_STEPS):
        c = par_rates[n]
        discounts[n] = (1.0 - c * DELTA * annuity) / (1.0 + c * DELTA)
        annuity += discounts[n]

    if np.any(discounts <= 0) or np.any(np.diff(discounts) > 0):
        raise ValueError(
            "bootstrapped discount curve is not positive and decreasing; "
            "check the input par rates"
        )
    return discounts


def discounts_to_forwards(discounts: np.ndarray) -> np.ndarray:
    """
    The 19 six-month Libor forwards, equation (10):

        F_i = (1/delta) * [ D(0, T_i) / D(0, T_i + 1/2) - 1 ],   T_i = i/2

    Index i = 1..19 maps to horizons 0.5 .. 9.5 years, i.e. the forward
    covering the period [T_i, T_i + 1/2].
    """
    d = np.asarray(discounts, dtype=float)
    return (d[:-1] / d[1:] - 1.0) / DELTA


def forwards_to_discounts(forwards: np.ndarray, d_half: float) -> np.ndarray:
    """
    Inverse of :func:`discounts_to_forwards`.  Needs the (non-stochastic)
    six-month discount factor D(0, 0.5) to anchor the curve.
    """
    f = np.asarray(forwards, dtype=float)
    d = np.empty(len(f) + 1)
    d[0] = d_half
    for i in range(len(f)):
        d[i + 1] = d[i] / (1.0 + DELTA * f[i])
    return d


# ---------------------------------------------------------------------------
# Curve container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Curve:
    """
    A fully bootstrapped term structure on the semiannual grid.

    Attributes
    ----------
    discounts : (20,) array
        D(0, 0.5), D(0, 1.0), ..., D(0, 10.0)
    forwards : (19,) array
        Six-month Libor forwards F_i, horizons 0.5 .. 9.5
    par_rates : (20,) array
        The interpolated semiannual par curve used to build the discounts.
    """

    discounts: np.ndarray
    forwards: np.ndarray
    par_rates: np.ndarray

    # -- accessors ---------------------------------------------------------
    def D(self, T: float) -> float:
        """Discount factor D(0, T) for T on the semiannual grid (T=0 -> 1)."""
        if abs(T) < 1e-12:
            return 1.0
        return float(self.discounts[grid_index(T)])

    def annuity(self, start: float, end: float) -> float:
        """
        A(0, tau, T) = sum_{i=1}^{2(T-tau)} D(0, tau + i/2)

        The semiannual fixed-leg annuity of a forward swap, per equation (3).
        """
        n = int(round((end - start) / DELTA))
        if n <= 0:
            raise ValueError(f"empty annuity for start={start}, end={end}")
        return float(sum(self.D(start + (i + 1) * DELTA) for i in range(n)))

    def forward_swap_rate(self, start: float, end: float) -> float:
        """
        Forward swap rate, equation (3):

            FSR(0, tau, T) = 2 * [ D(0,tau) - D(0,T) ] / A(0, tau, T)

        With start = 0 this is the par swap rate of equation (2), which is the
        market-convention strike of a T-year cap.
        """
        return (self.D(start) - self.D(end)) / (DELTA * self.annuity(start, end))

    def par_swap_rate(self, maturity: float) -> float:
        """Spot par swap rate FSR(0, 0, T), equation (2)."""
        return self.forward_swap_rate(0.0, maturity)

    def zero_rate(self, T: float) -> float:
        """Continuously compounded zero rate."""
        return -np.log(self.D(T)) / T

    @property
    def short_discount(self) -> float:
        """D(0, 0.5): the shortest bond, non-stochastic over the first step."""
        return float(self.discounts[0])


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def build_curve(
    libor_6m: float,
    libor_1y: float,
    swap_rates: dict[float, float],
) -> Curve:
    """
    Build a :class:`Curve` from the market quotes the paper uses.

    Parameters
    ----------
    libor_6m, libor_1y : float
        Six-month and one-year Libor, as decimals (0.0575 for 5.75%).
    swap_rates : dict
        Par swap rates keyed by maturity, must cover 2, 3, 4, 5, 7 and 10 years.
    """
    missing = [t for t in SWAP_PILLARS if t not in swap_rates]
    if missing:
        raise ValueError(f"missing swap rates for maturities {missing}")

    par_half, par_one = libor_to_par(libor_6m, libor_1y)
    pillars = np.array(list(LIBOR_PILLARS) + list(SWAP_PILLARS), dtype=float)
    values = np.array([par_half, par_one] + [swap_rates[t] for t in SWAP_PILLARS])

    par_curve = interpolate_par_curve(pillars, values)
    discounts = bootstrap_discounts(par_curve)
    forwards = discounts_to_forwards(discounts)
    return Curve(discounts=discounts, forwards=forwards, par_rates=par_curve)


def curve_from_forwards(forwards: np.ndarray, libor_6m: float) -> Curve:
    """
    Build a curve directly from the 19 six-month forwards plus the spot 6M
    Libor rate.  Useful for synthetic data and for round-trip tests.
    """
    forwards = np.asarray(forwards, dtype=float)
    if forwards.shape != (N_FORWARDS,):
        raise ValueError(f"expected {N_FORWARDS} forwards, got {forwards.shape}")

    d_half = 1.0 / (1.0 + DELTA * libor_6m)
    discounts = forwards_to_discounts(forwards, d_half)

    # Recover the implied par curve for completeness.
    par = np.empty(N_STEPS)
    annuity = 0.0
    for n in range(N_STEPS):
        annuity += discounts[n]
        par[n] = (1.0 - discounts[n]) / (DELTA * annuity)

    return Curve(discounts=discounts, forwards=forwards, par_rates=par)
