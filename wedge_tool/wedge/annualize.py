"""Formulas from Brigo & Mercurio Section 6.20.

Notation (three instants S < T < U, six months apart):
    F1 = F(0; S, T)   semi-annual forward rate, expiry S  (year fraction tau1)
    F2 = F(0; T, U)   semi-annual forward rate, expiry T  (year fraction tau2)
    F  = F(0; S, U)   annual forward rate in which F1 and F2 are "nested"

    v1 = Black volatility of the S-expiry caplet (on F1)
    v2 = Black volatility of the T-expiry caplet (on F2)
    v_swaption = Black volatility of the S x 1 swaption (its one-period
                 underlying swap rate is exactly F)

All functions accept scalars or numpy arrays.
"""

import numpy as np


def annual_forward(f1, f2, tau1=0.5, tau2=0.5):
    """Annual forward rate nesting two semi-annual forwards, eq. (6.74).

    Derived from 1 + (tau1+tau2)*F = (1 + tau1*F1)(1 + tau2*F2).
    With tau1 = tau2 = 0.5 this is the book's F = (F1+F2)/2 + F1*F2/4.
    """
    return ((1.0 + tau1 * f1) * (1.0 + tau2 * f2) - 1.0) / (tau1 + tau2)


def nesting_weights(f1, f2, tau1=0.5, tau2=0.5):
    """Weights u1, u2 of the caplet vols inside the swaption vol, Section 6.20.

    u1 = (F1/2 + F1*F2/4) / F,  u2 = (F2/2 + F1*F2/4) / F  (semi-annual case).
    They are the loadings of dZ1 and dZ2 in the lognormal dynamics of F,
    with everything frozen at time 0.
    """
    f = annual_forward(f1, f2, tau1, tau2)
    denom = (tau1 + tau2) * f
    u1 = tau1 * f1 * (1.0 + tau2 * f2) / denom
    u2 = tau2 * f2 * (1.0 + tau1 * f1) / denom
    return u1, u2


def cap_implied_swaption_vol(f1, f2, v1, v2, rho=1.0, tau1=0.5, tau2=0.5):
    """S x 1 swaption vol implied by the two caplet vols, eq. (6.76) ("a2").

    v_swaption^2 ~= u1^2 v1^2 + u2^2 v2^2 + 2 rho u1 u2 v1 v2

    rho is the "infra-correlation" between the two adjacent semi-annual
    forward rates. The book uses rho = 1 (adjacent rates are strongly
    correlated) and notes the formula tends to slightly overestimate.
    """
    u1, u2 = nesting_weights(f1, f2, tau1, tau2)
    var = (u1 * v1) ** 2 + (u2 * v2) ** 2 + 2.0 * rho * u1 * u2 * v1 * v2
    return np.sqrt(var)


def implied_infra_correlation(v_swaption, f1, f2, v1, v2, tau1=0.5, tau2=0.5):
    """Infra-correlation rho backed out of market quotes, Section 7.9.

    Inverts eq. (6.76):
        rho = (v_swaption^2 - u1^2 v1^2 - u2^2 v2^2) / (2 u1 u2 v1 v2)

    A viable correlation lies in [-1, 1] and, between adjacent forward
    rates, is usually close to 1. Values far below that (or above 1) flag
    a misalignment between the cap and swaption markets in that bucket.
    """
    u1, u2 = nesting_weights(f1, f2, tau1, tau2)
    num = v_swaption**2 - (u1 * v1) ** 2 - (u2 * v2) ** 2
    return num / (2.0 * u1 * u2 * v1 * v2)
