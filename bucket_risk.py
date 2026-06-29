"""Rotate a trader's tenor-bucket delta ladder into the NSS factor space
(level / slope / curvature / curvature2), the same chain-rule projection
used for PCA bucket-to-factor risk: dV/dbeta_k = sum_i (dV/drate_i) *
(drate_i/dbeta_k) = (J^T @ bucket_delta)[k], where J is the NSS loadings
matrix evaluated at the bucket tenors.
"""

import numpy as np

import nss_model

DEFAULT_BUCKET_T = [1, 2, 3, 5, 7, 10, 15, 20, 30]
DEFAULT_BUCKET_DELTA = [800, -1200, 400, 1500, -300, -2000, 600, 1100, -900]  # $ per bp, illustrative


def nss_jacobian(bucket_T, lambda1=nss_model.DEFAULT_LAMBDA1, lambda2=nss_model.DEFAULT_LAMBDA2):
    return nss_model.loadings(bucket_T, lambda1, lambda2)


def project_bucket_delta(bucket_T, bucket_delta, lambda1=nss_model.DEFAULT_LAMBDA1,
                          lambda2=nss_model.DEFAULT_LAMBDA2):
    J = nss_jacobian(bucket_T, lambda1, lambda2)
    param_delta = J.T @ np.asarray(bucket_delta, dtype=float)
    return {
        "delta_level": float(param_delta[0]),
        "delta_slope": float(param_delta[1]),
        "delta_curvature": float(param_delta[2]),
        "delta_curvature2": float(param_delta[3]),
    }
