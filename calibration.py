"""Stage A of the pipeline: fit the NSS level/slope/curvature/curvature2
betas exactly through the four liquid pillar points (2y, 5y, 10y, 20y),
given fixed lambda1/lambda2 -- a 4-equation, 4-unknown linear solve, so the
fit is exact at the pillars by construction. Anything away from the
pillars is left to the residual + spline stage.
"""

import numpy as np

import nss_model

PILLARS = (2.0, 5.0, 10.0, 20.0)


def fit_pillars(pillar_rates, pillar_T=PILLARS, lambda1=nss_model.DEFAULT_LAMBDA1,
                 lambda2=nss_model.DEFAULT_LAMBDA2):
    L = nss_model.loadings(pillar_T, lambda1, lambda2)
    beta = np.linalg.solve(L, np.asarray(pillar_rates, dtype=float))
    return beta


def pillar_rates_from_curve(curve, pillar_T=PILLARS, tol=1e-6):
    """Pull the four pillar rates out of a tidy curve (columns T, rate_pct)."""
    rates = []
    for t in pillar_T:
        match = curve.loc[np.isclose(curve["T"], t, atol=tol), "rate_pct"]
        if match.empty:
            raise ValueError(f"no curve point at T={t}y to use as a pillar")
        rates.append(float(match.iloc[0]))
    return np.array(rates)
