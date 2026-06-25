"""Raw SVI and SSVI parametrizations for a single implied vol surface, plus a
Cross-Entropy Method seed for the non-convex part of single-slice calibration.
"""

import numpy as np
from scipy.optimize import least_squares


def raw_svi_total_variance(k, a, b, rho, m, sigma):
    k = np.asarray(k, dtype=float)
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def raw_svi_iv(k, T, a, b, rho, m, sigma):
    w = raw_svi_total_variance(k, a, b, rho, m, sigma)
    return np.sqrt(np.maximum(w, 1e-12) / T)


def ssvi_phi_heston(theta, eta):
    return eta / theta


def ssvi_total_variance(k, theta, rho, eta):
    k = np.asarray(k, dtype=float)
    phi = ssvi_phi_heston(theta, eta)
    return 0.5 * theta * (1.0 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + (1.0 - rho ** 2)))


def slice_is_arbitrage_free(a, b, rho, m, sigma):
    """Necessary single-slice (butterfly) conditions -- not the full SSVI
    calendar-spread condition across slices."""
    if b < 0 or abs(rho) >= 1 or sigma <= 0:
        return False
    return a + b * sigma * np.sqrt(1.0 - rho ** 2) >= 0


def cem_seed_slice(k, market_iv, T, n_iter=20, population=300, elite_frac=0.15, seed=None):
    """Cross-Entropy Method search over the non-convex (rho, m, sigma); (a, b)
    are linear given those three, so each candidate is scored by closed-form
    simple-linear-regression instead of a per-candidate optimizer call. Fully
    vectorized across the population -- this is what keeps calibrating
    hundreds of slices fast.
    """
    rng = np.random.default_rng(seed)
    k = np.asarray(k, dtype=float)
    w_target = (np.asarray(market_iv, dtype=float) ** 2) * T
    w_mean = w_target.mean()

    k_lo, k_hi = float(k.min()), float(k.max())
    span = max(k_hi - k_lo, 1e-3)
    mean = np.array([0.0, 0.5 * (k_lo + k_hi), 0.3 * span + 1e-3])
    std = np.array([0.4, 0.4 * span + 1e-3, 0.3 * span + 1e-3])

    n_elite = max(4, int(population * elite_frac))
    best_loss, best_params = np.inf, None

    for _ in range(n_iter):
        rho_s = np.clip(rng.normal(mean[0], std[0], population), -0.995, 0.995)
        m_s = rng.normal(mean[1], std[1], population)
        sigma_s = np.abs(rng.normal(mean[2], std[2], population)) + 1e-4

        diff = k[None, :] - m_s[:, None]
        x = rho_s[:, None] * diff + np.sqrt(diff ** 2 + sigma_s[:, None] ** 2)  # (population, n_k)

        x_mean = x.mean(axis=1)
        var = ((x - x_mean[:, None]) ** 2).mean(axis=1)
        cov = ((x - x_mean[:, None]) * (w_target[None, :] - w_mean)).mean(axis=1)
        b_s = np.clip(cov / np.maximum(var, 1e-10), 1e-6, None)
        a_s = w_mean - b_s * x_mean

        w_fit = a_s[:, None] + b_s[:, None] * x
        losses = ((w_fit - w_target[None, :]) ** 2).mean(axis=1)

        elite_idx = np.argsort(losses)[:n_elite]
        if losses[elite_idx[0]] < best_loss:
            best_loss = losses[elite_idx[0]]
            i0 = elite_idx[0]
            best_params = (a_s[i0], b_s[i0], rho_s[i0], m_s[i0], sigma_s[i0])

        mean = np.array([rho_s[elite_idx].mean(), m_s[elite_idx].mean(), sigma_s[elite_idx].mean()])
        std = np.array([
            rho_s[elite_idx].std() + 1e-6,
            m_s[elite_idx].std() + 1e-6,
            sigma_s[elite_idx].std() + 1e-6,
        ])

    return best_params


def calibrate_slice(k, market_iv, T, seed_params=None):
    k = np.asarray(k, dtype=float)
    market_iv = np.asarray(market_iv, dtype=float)
    if seed_params is None:
        seed_params = cem_seed_slice(k, market_iv, T)
    a0, b0, rho0, m0, sigma0 = seed_params

    def residuals(p):
        a, b, rho, m, sigma = p
        return raw_svi_iv(k, T, a, b, rho, m, sigma) - market_iv

    lower = [-np.inf, 0.0, -0.999, -np.inf, 1e-6]
    upper = [np.inf, np.inf, 0.999, np.inf, np.inf]
    result = least_squares(residuals, x0=[a0, b0, rho0, m0, sigma0], bounds=(lower, upper))
    a, b, rho, m, sigma = result.x
    return {"a": a, "b": b, "rho": rho, "m": m, "sigma": sigma, "cost": result.cost}
