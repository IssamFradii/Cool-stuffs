"""Bachelier (normal-model) swaption pricing.

Rates in this whole package are in rate units (0.008 = 80bp), not percent.
This is a deliberately minimal pricer: it prices ATM (or off-ATM) swaptions
from a forward, a normal vol and an annuity (BPV), with no curve bootstrap.
Swap it out for your real pricer/curve stack later -- the rv_carry signals
downstream only need a price_surface DataFrame shaped the same way.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def bachelier_price(F, K, sigma, T, bpv, is_payer: bool = True):
    """Normal-model swaption price.

    F, K       : forward and strike, rate units (e.g. 0.032)
    sigma      : normal vol, rate units (e.g. 0.008 = 80bp)
    T          : expiry in years
    bpv        : annuity (basis point value) of the underlying swap
    is_payer   : True for payer, False for receiver

    d = (F-K) / (sigma * sqrt(T)); price = bpv * [(F-K) N(d) + sigma*sqrt(T)*phi(d)]
    (receiver uses the put-side mirror). All args broadcast via numpy.
    """
    F, K, sigma, T, bpv = (np.asarray(x, dtype=float) for x in (F, K, sigma, T, bpv))
    vol_sqrtT = sigma * np.sqrt(T)
    vol_sqrtT_safe = np.where(vol_sqrtT > 1e-12, vol_sqrtT, 1e-12)
    d = (F - K) / vol_sqrtT_safe

    if is_payer:
        price = bpv * ((F - K) * norm.cdf(d) + vol_sqrtT * norm.pdf(d))
        intrinsic = np.maximum(F - K, 0.0)
    else:
        price = bpv * ((K - F) * norm.cdf(-d) + vol_sqrtT * norm.pdf(d))
        intrinsic = np.maximum(K - F, 0.0)

    return np.where(vol_sqrtT > 1e-12, price, bpv * intrinsic)


def atm_price(F, sigma, T, bpv, is_payer: bool = True):
    """Convenience: Bachelier price at strike = forward (K = F)."""
    return bachelier_price(F, F, sigma, T, bpv, is_payer)
