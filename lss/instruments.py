"""
Cap, swap and swaption valuation -- the Black (1976) quoting conventions of
Section 2, plus the swap-value expressions used by the simulation.

A note on equation (1) of the paper
-----------------------------------
The caplet formula as typeset in the paper reads

    D(0,T) (a/360) [ F N(d) - K N(d - sqrt(sigma^2 tau / 2)) ]
    d = ( ln(F/K) + sqrt(sigma^2 tau / 2) ) / sqrt(sigma^2 tau)

The two `sqrt(sigma^2 tau / 2)` terms are typographical artifacts.  The
intended (and only internally consistent) expression is standard Black:

    d1 = ( ln(F/K) + sigma^2 tau / 2 ) / (sigma sqrt(tau))
    d2 = d1 - sigma sqrt(tau)

which is what is implemented here.  Read literally, the printed d2 would not
even reduce to the correct zero-volatility limit.  Equation (7), the
at-the-money-forward swaption formula, is printed correctly and is used as a
cross-check in the tests.
"""

from __future__ import annotations
import numpy as np
from .mathutil import norm_cdf, norm_pdf, norm_ppf

from .conventions import DELTA
from .curve import Curve


# ---------------------------------------------------------------------------
# Black (1976)
# ---------------------------------------------------------------------------
def black_call(forward: float, strike: float, vol: float, expiry: float) -> float:
    """
    Undiscounted Black (1976) call value on a lognormal forward.

    Returns the intrinsic value max(F-K, 0) when vol or expiry is zero.
    """
    total_vol = vol * np.sqrt(expiry)
    if total_vol <= 0 or forward <= 0 or strike <= 0:
        return max(forward - strike, 0.0)
    d1 = (np.log(forward / strike) + 0.5 * total_vol * total_vol) / total_vol
    d2 = d1 - total_vol
    return forward * norm_cdf(d1) - strike * norm_cdf(d2)


def black_put(forward: float, strike: float, vol: float, expiry: float) -> float:
    """Undiscounted Black (1976) put value (put/call parity)."""
    return black_call(forward, strike, vol, expiry) - (forward - strike)


def black_vega(forward: float, strike: float, vol: float, expiry: float) -> float:
    """
    Undiscounted Black vega, d(price)/d(vol).  Same for a call and a put.
    Units: price change per 1.00 of vol (i.e. per 100 vol points).
    """
    total_vol = vol * np.sqrt(expiry)
    if total_vol <= 0 or forward <= 0 or strike <= 0:
        return 0.0
    d1 = (np.log(forward / strike) + 0.5 * total_vol * total_vol) / total_vol
    return forward * norm_pdf(d1) * np.sqrt(expiry)


# ---------------------------------------------------------------------------
# Swap values
# ---------------------------------------------------------------------------
def forward_swap_value(curve: Curve, start: float, end: float, coupon: float) -> float:
    """
    Value to the receiver of fixed in a forward-start swap, equation (4):

        V(t, tau, T, c) = (c/2) sum_{i=1}^{2(T-tau)} D(t, tau+i/2)
                          + D(t, T) - D(t, tau)

    The last two terms are the redemption of the fixed leg and the value of
    the floating leg, which is worth par (i.e. D(t,tau)) at the start date.
    """
    return (
        coupon * DELTA * curve.annuity(start, end)
        + curve.D(end)
        - curve.D(start)
    )


def spot_swap_value_from_discounts(
    discounts_from_start: np.ndarray, coupon: float
) -> np.ndarray:
    """
    Value to the receiver of fixed in a swap that starts *now*, equation (12):

        V(tau, tau, T, c) = (c/2) sum_{i=1}^{2(T-tau)} D(tau, tau+i/2)
                            + D(tau, T) - 1

    Parameters
    ----------
    discounts_from_start : array, shape (..., n)
        D(tau, tau + i/2) for i = 1..n, where n = 2(T - tau).  The leading
        axes are broadcast, so this prices every simulated path at once.

    Returns
    -------
    array with the trailing axis reduced.
    """
    d = np.asarray(discounts_from_start, dtype=float)
    annuity = d.sum(axis=-1)
    return coupon * DELTA * annuity + d[..., -1] - 1.0


# ---------------------------------------------------------------------------
# Swaptions, Black quoting convention
# ---------------------------------------------------------------------------
def black_swaption_price(
    curve: Curve,
    expiry: float,
    tenor: float,
    vol: float,
    coupon: float | None = None,
    payer: bool = True,
) -> float:
    """
    Black (1976) European swaption price, equation (6):

        (1/2) A(0,tau,T) [ FSR N(d) - c N(d - sigma sqrt(tau)) ]

    The 1/2 is the semiannual accrual factor `delta` on the fixed leg.
    If `coupon` is None the swaption is struck at the money forward.
    """
    end = expiry + tenor
    fsr = curve.forward_swap_rate(expiry, end)
    strike = fsr if coupon is None else coupon
    level = DELTA * curve.annuity(expiry, end)
    fn = black_call if payer else black_put
    return level * fn(fsr, strike, vol, expiry)


def black_swaption_atmf(curve: Curve, expiry: float, tenor: float, vol: float) -> float:
    """
    At-the-money-forward European swaption price, equation (7):

        ( D(0,tau) - D(0,T) ) [ 2 N( sigma sqrt(tau) / 2 ) - 1 ]

    Payers and receivers have identical value at the money forward.  This is
    the formula the market uses to turn a quoted vol into a premium, and it is
    algebraically identical to :func:`black_swaption_price` at c = FSR.
    """
    end = expiry + tenor
    return (curve.D(expiry) - curve.D(end)) * (
        2.0 * norm_cdf(vol * np.sqrt(expiry) / 2.0) - 1.0
    )


def black_swaption_vega_atmf(
    curve: Curve, expiry: float, tenor: float, vol: float
) -> float:
    """
    Vega of an ATMF swaption under equation (7), d(price)/d(vol):

        ( D(0,tau) - D(0,T) ) * 2 * phi( sigma sqrt(tau)/2 ) * sqrt(tau)/2
    """
    end = expiry + tenor
    return (curve.D(expiry) - curve.D(end)) * norm_pdf(
        vol * np.sqrt(expiry) / 2.0
    ) * np.sqrt(expiry)


def swaption_implied_vol(
    curve: Curve, expiry: float, tenor: float, price: float
) -> float:
    """Invert equation (7) for the ATMF Black volatility."""
    end = expiry + tenor
    level = curve.D(expiry) - curve.D(end)
    if level <= 0:
        return float("nan")
    ratio = price / level
    if not (0.0 <= ratio < 1.0):
        return float("nan")
    # price = level * (2 N(x) - 1)  with x = vol sqrt(tau)/2
    x = norm_ppf(0.5 * (ratio + 1.0))
    return 2.0 * x / np.sqrt(expiry)


# ---------------------------------------------------------------------------
# Caps
# ---------------------------------------------------------------------------
def caplet_schedule(maturity: float) -> list[tuple[float, float]]:
    """
    (reset, payment) dates of the caplets in a `maturity`-year cap on six-month
    Libor.

    The first caplet is omitted by market convention because its cash flow is
    already fixed at time zero (footnote 2).  A T-year cap therefore has
    2T - 1 caplets, resetting at 0.5, 1.0, ..., T-0.5 and paying one period
    later.  For a five-year cap that is nine caplets paying at 1.0 ... 5.0.
    """
    n = int(round(maturity / DELTA))
    return [(i * DELTA, (i + 1) * DELTA) for i in range(1, n)]


def black_caplet_price(
    curve: Curve, reset: float, strike: float, vol: float
) -> float:
    """
    Black (1976) caplet value, equation (1):

        D(0, T) * delta * [ F N(d1) - K N(d2) ]

    where T = reset + delta is the payment date and the option expires at the
    reset date.  See the module docstring on the typography of equation (1).
    """
    payment = reset + DELTA
    fwd = (curve.D(reset) / curve.D(payment) - 1.0) / DELTA
    return curve.D(payment) * DELTA * black_call(fwd, strike, vol, reset)


def black_cap_price(
    curve: Curve,
    maturity: float,
    vol: float | np.ndarray,
    strike: float | None = None,
) -> float:
    """
    Price a cap as the sum of its caplets.

    Parameters
    ----------
    vol : float or array
        A single flat Black volatility (the market quoting convention -- one
        vol applied to every caplet in the cap), or a per-caplet array of
        volatilities, which is what the string market model produces.
    strike : float, optional
        Defaults to the `maturity`-year par swap rate, the market convention
        for the strike of a T-year cap (Section 4).
    """
    if strike is None:
        strike = curve.par_swap_rate(maturity)

    schedule = caplet_schedule(maturity)
    vols = np.broadcast_to(np.asarray(vol, dtype=float), (len(schedule),))

    return float(
        sum(
            black_caplet_price(curve, reset, strike, v)
            for (reset, _), v in zip(schedule, vols)
        )
    )


def black_cap_vega(
    curve: Curve, maturity: float, vol: float, strike: float | None = None
) -> float:
    """
    Vega of a cap quoted at a single flat volatility: the sum of the caplet
    vegas.  Units: price change per 1.00 of vol.
    """
    if strike is None:
        strike = curve.par_swap_rate(maturity)

    total = 0.0
    for reset, payment in caplet_schedule(maturity):
        fwd = (curve.D(reset) / curve.D(payment) - 1.0) / DELTA
        total += curve.D(payment) * DELTA * black_vega(fwd, strike, vol, reset)
    return total


def cap_implied_vol(
    curve: Curve, maturity: float, price: float, strike: float | None = None
) -> float:
    """
    Flat Black volatility that reproduces `price` for the cap -- the market
    quoting convention.  This is how a model cap price is converted into a
    comparable "model implied vol" for the wedge signal.
    """
    if strike is None:
        strike = curve.par_swap_rate(maturity)

    lo, hi = 1e-6, 5.0
    if black_cap_price(curve, maturity, hi, strike) < price:
        return float("nan")
    if black_cap_price(curve, maturity, lo, strike) > price:
        return float("nan")

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if black_cap_price(curve, maturity, mid, strike) < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    return 0.5 * (lo + hi)
