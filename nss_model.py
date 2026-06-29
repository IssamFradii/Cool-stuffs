"""Nelson-Siegel-Svensson (NSS) curve: loadings, rate function, and the
Jacobian w.r.t. the four beta factors -- the linear core every other module
in this project builds on.

y(tau) = beta0*L0(tau) + beta1*L1(tau) + beta2*L2(tau) + beta3*L3(tau)

beta0..beta3 are level / slope / curvature / second-curvature ("4th
parameter", the Svensson hump term). lambda1, lambda2 are the two decay
constants -- fixed shape parameters, not part of the risk factor set.
"""

import numpy as np

DEFAULT_LAMBDA1 = 1.5
DEFAULT_LAMBDA2 = 5.0


def loadings(T, lambda1=DEFAULT_LAMBDA1, lambda2=DEFAULT_LAMBDA2):
    """Design matrix L(T), shape (n, 4): L[:, k] = d(rate)/d(beta_k).

    Because rate is linear in beta given fixed lambdas, this matrix is both
    the basis-function loadings *and* the Jacobian w.r.t. the betas.
    """
    T = np.asarray(T, dtype=float)
    x1 = T / lambda1
    x2 = T / lambda2

    L0 = np.ones_like(T)
    L1 = (1 - np.exp(-x1)) / x1
    L2 = L1 - np.exp(-x1)
    L3 = (1 - np.exp(-x2)) / x2 - np.exp(-x2)
    return np.column_stack([L0, L1, L2, L3])


def nss_rate(T, beta, lambda1=DEFAULT_LAMBDA1, lambda2=DEFAULT_LAMBDA2):
    beta = np.asarray(beta, dtype=float)
    return loadings(T, lambda1, lambda2) @ beta
