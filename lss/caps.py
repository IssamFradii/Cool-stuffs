"""
Cap valuation from the implied covariance matrix (Section 5.4).

This is the heart of the paper's test.  Once Sigma has been implied from the
swaptions market, cap prices are *no longer free* -- they are determined:

  "The main diagonal of the implied covariance matrix represents the implied
   variance of the individual forward rates as they roll down in maturity and
   become the spot rate ...  Since this provides a complete specification of
   the volatility of all forwards, the main diagonal uniquely determines the
   values of individual caplets, which then determine the values of caps."

and

  "the variance for the caplet maturing in six months is the first diagonal
   element, the variance for the caplet maturing in twelve months is the
   average of the first and second diagonal elements, etc."

Caplets are simple European options on individual forward rates, and the
lognormal dynamics of equation (8) make the Black model *exact* for them, so
no simulation is needed here:

    caplet variance (annualised) for the caplet resetting at tau = i * delta
        = (1/i) * sum_{m=1}^{i} Sigma[m-1, m-1]

The wedge between the caps market and the swaptions market is then the
percentage difference between this model price and the quoted market price:

    error = 100 * (model - market) / market

A *positive* error means the swaptions market implies the cap is worth more
than it trades for -- the cap is cheap relative to swaptions.  That is the
sign convention of Tables 9 and 10, and the signal driving :mod:`lss.strategy`.

Monte-Carlo requirements
------------------------
Cap prices are an *extrapolative* use of the fitted Sigma.  The swaption
cross-section constrains Sigma mainly through elasticity-weighted quadratic
forms -- averages over the whole matrix -- whereas caps depend on its main
diagonal alone, which is pinned down only indirectly.  Simulation noise in the
calibration therefore propagates into the cap errors amplified, and because
noise in Psi tends to *raise* the implied caplet volatilities the resulting
bias is one-sided.

Measured on the synthetic sample, the Table 9 overall mean comes out at:

    2,000 paths (the paper's setting):  13.3 and 15.0 on two seeds
    4,000 paths:                         9.9
    8,000 paths:                        10.7 and 10.1 on two seeds

So 2,000 paths is not enough to estimate the wedge stably, and overstates it by
roughly a third.  Use 8,000 or more when the cap errors themselves are the
quantity of interest.  The swaption RMSE, by contrast, is largely insensitive
(3.5-4.2% across all of the above), which is why the problem is easy to miss.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .conventions import DELTA, CAP_MATURITIES
from .curve import Curve
from .approx import caplet_variances
from .instruments import (
    black_caplet_price, caplet_schedule, black_cap_price, cap_implied_vol,
)


# ---------------------------------------------------------------------------
# Model cap prices
# ---------------------------------------------------------------------------
def model_caplet_vols(sigma: np.ndarray) -> np.ndarray:
    """
    Implied Black volatility of each caplet, indexed by reset period:
    element i-1 is the caplet resetting at tau = i * delta.

    This is the square root of the running average of the main diagonal of
    Sigma -- see the module docstring.
    """
    return np.sqrt(np.clip(caplet_variances(sigma), 0.0, None))


def model_cap_price(
    curve: Curve,
    sigma: np.ndarray,
    maturity: float,
    strike: float | None = None,
) -> float:
    """
    Model price of a `maturity`-year cap on six-month Libor, implied by Sigma.

    The strike defaults to the `maturity`-year par swap rate, which is the
    market convention for a T-year cap (Section 4).  The first caplet is
    omitted, so a T-year cap has 2T - 1 caplets.
    """
    if strike is None:
        strike = curve.par_swap_rate(maturity)

    vols = model_caplet_vols(sigma)
    total = 0.0
    for reset, _ in caplet_schedule(maturity):
        i = int(round(reset / DELTA))          # reset = i * delta
        total += black_caplet_price(curve, reset, strike, vols[i - 1])
    return float(total)


def model_cap_prices(
    curve: Curve,
    sigma: np.ndarray,
    maturities: tuple[float, ...] = CAP_MATURITIES,
) -> np.ndarray:
    """Model prices for the quoted cap maturities."""
    return np.array([model_cap_price(curve, sigma, T) for T in maturities])


def model_cap_vols(
    curve: Curve,
    sigma: np.ndarray,
    maturities: tuple[float, ...] = CAP_MATURITIES,
) -> np.ndarray:
    """
    Model cap prices expressed back in the market's quoting convention: the
    single flat Black volatility that reproduces the model price.  Directly
    comparable with a Bloomberg cap vol.
    """
    out = np.empty(len(maturities))
    for k, T in enumerate(maturities):
        out[k] = cap_implied_vol(curve, T, model_cap_price(curve, sigma, T))
    return out


# ---------------------------------------------------------------------------
# Market cap prices
# ---------------------------------------------------------------------------
def market_cap_prices(
    curve: Curve,
    vols: np.ndarray,
    maturities: tuple[float, ...] = CAP_MATURITIES,
) -> np.ndarray:
    """
    Market cap prices from quoted flat Black volatilities, via equation (1)
    summed over the caplets (Section 4).
    """
    vols = np.asarray(vols, dtype=float)
    return np.array(
        [black_cap_price(curve, T, v) for T, v in zip(maturities, vols)]
    )


# ---------------------------------------------------------------------------
# The wedge
# ---------------------------------------------------------------------------
@dataclass
class CapValuation:
    """
    Model-versus-market comparison for one week's caps.

    Attributes
    ----------
    maturities : tuple of float
    model_prices, market_prices : arrays
    percentage_errors : array
        100 * (model - market) / market.  Positive => the cap is cheap
        relative to what the swaptions market implies.
    model_vols, market_vols : arrays
        The same comparison expressed in flat Black volatility points, which
        is how a trader would look at it.
    """

    maturities: tuple[float, ...]
    model_prices: np.ndarray
    market_prices: np.ndarray
    percentage_errors: np.ndarray
    model_vols: np.ndarray
    market_vols: np.ndarray

    @property
    def vol_spread(self) -> np.ndarray:
        """Model minus market cap vol, in volatility points."""
        return 100.0 * (self.model_vols - self.market_vols)


def value_caps(
    curve: Curve,
    sigma: np.ndarray,
    market_vols: np.ndarray,
    maturities: tuple[float, ...] = CAP_MATURITIES,
) -> CapValuation:
    """
    Compare model and market cap prices for one week -- one row of Table 9.

    Parameters
    ----------
    curve : Curve
        That week's term structure.
    sigma : (19, 19) array
        The implied covariance matrix from that week's swaption calibration.
    market_vols : array
        Quoted flat Black cap volatilities, same order as `maturities`.
    """
    market_vols = np.asarray(market_vols, dtype=float)
    model = model_cap_prices(curve, sigma, maturities)
    market = market_cap_prices(curve, market_vols, maturities)
    pct = 100.0 * (model - market) / market

    model_vols = np.array(
        [cap_implied_vol(curve, T, p) for T, p in zip(maturities, model)]
    )

    return CapValuation(
        maturities=tuple(maturities),
        model_prices=model,
        market_prices=market,
        percentage_errors=pct,
        model_vols=model_vols,
        market_vols=market_vols,
    )


# ---------------------------------------------------------------------------
# The Merton (1973) no-arbitrage bound
# ---------------------------------------------------------------------------
def merton_lower_bound(
    curve: Curve,
    sigma_rank_one: np.ndarray,
    maturities: tuple[float, ...] = CAP_MATURITIES,
) -> np.ndarray:
    """
    Cap prices implied by a *rank-one* covariance matrix, in which all forward
    rates are perfectly correlated (Section 5.4, Table 10).

    Merton (1973): a portfolio of options is worth at least as much as an
    option on the portfolio, with equality under perfect correlation.  A cap is
    a portfolio of options on individual forwards; a swaption is an option on a
    basket of them.  So the cap price implied by a one-factor fit to the
    swaptions market is a genuine *lower bound* for the market cap price, and
    the percentage errors of Table 10 should all be negative.

    Only 8 of 4,352 observations violate it in the paper -- caps and swaptions
    almost always satisfy the strict no-arbitrage restriction, even when they
    are inconsistent with each other.
    """
    return model_cap_prices(curve, sigma_rank_one, maturities)


def check_merton_bound(
    curve: Curve,
    sigma_rank_one: np.ndarray,
    market_vols: np.ndarray,
    maturities: tuple[float, ...] = CAP_MATURITIES,
) -> np.ndarray:
    """
    Percentage differences behind Table 10.  Values should be negative; a
    positive entry is an apparent violation of the Merton bound.
    """
    bound = merton_lower_bound(curve, sigma_rank_one, maturities)
    market = market_cap_prices(curve, market_vols, maturities)
    return 100.0 * (bound - market) / market
