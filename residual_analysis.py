"""Residual between the pillar-fitted NSS curve and the full market curve --
zero at the 2/5/10/20 pillars by construction, non-zero everywhere else,
which is exactly the part the spline stage in spline_adjustment.py picks up.
"""

import nss_model


def compute_residuals(curve, beta, lambda1=nss_model.DEFAULT_LAMBDA1, lambda2=nss_model.DEFAULT_LAMBDA2):
    out = curve.copy()
    out["nss_rate"] = nss_model.nss_rate(out["T"].to_numpy(), beta, lambda1, lambda2)
    out["residual"] = out["rate_pct"] - out["nss_rate"]
    return out
