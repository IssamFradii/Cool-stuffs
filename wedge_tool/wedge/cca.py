"""Cascade Calibration Algorithm (CCA/RCCA), Brigo & Mercurio Section 7.4.

Calibrates the general piecewise-constant (TABLE 1) instantaneous
volatilities sigma[i, h] of a lognormal LIBOR market model EXACTLY to a
rectangular swaption volatility matrix, given an exogenous instantaneous
correlation matrix, by solving a cascade of scalar quadratic equations
(Algorithms 7.4.1 and 7.4.2 - no optimization involved).

Conventions (annual tenor):
    T_h = h+1 years, h = 0, 1, ...; T_{-1} = 0, so all year fractions = 1.
    Forward F_i = F(0; i, i+1) years, i = 1..M (1-based, stored 0-based).
    sigma[i, h] = instantaneous vol of F_i for t in (T_{h-2}, T_{h-1}],
                  i.e. entry (i, h) of TABLE 1, h = 1..i.
    Swaption (a, b): expiry T_a = a+1 years, payments T_{a+1}..T_b;
                     tenor = b - a years; Black vol V[a, tenor].

The cascade can produce negative or complex sigmas when the input matrix
is irregular (book, Section 7.4.1) - that is a diagnostic, not a bug, so
the computation is carried out in complex arithmetic and flagged.
"""

import numpy as np

# Rank-2 correlation angles used for the book's Table 7.5 calibration
# (Section 7.4.1): rho_ij = cos(theta_i - theta_j), forwards 1..19.
BOOK_ANGLES = np.array([
    0.0147, 0.0643, 0.1032, 0.1502, 0.1969, 0.2239, 0.2771, 0.2950, 0.3630,
    0.3810, 0.4217, 0.4836, 0.5204, 0.5418, 0.5791, 0.6496, 0.6679, 0.7126,
    0.7659,
])


def rank2_correlation(angles):
    """rho_ij = cos(theta_i - theta_j), eq. (6.47) rank-2 form."""
    th = np.asarray(angles, dtype=float)
    return np.cos(th[:, None] - th[None, :])


def exp_decay_correlation(n, beta):
    """rho_ij = exp(-beta |T_i - T_j|), the simplest full-rank form (6.44)."""
    idx = np.arange(n, dtype=float)
    return np.exp(-beta * np.abs(idx[:, None] - idx[None, :]))


def swap_weights(fwd, a, b):
    """Weights w_i(0), i = a+1..b, and swap rate S_{a,b}(0), Section 6.15.

    fwd is the 0-based array of annual forwards F_1..F_M; year fractions 1.
    Returns (w, s) with w aligned to forwards indices a+1..b (1-based).
    """
    f = fwd[a: b]                      # F_{a+1}..F_b, 0-based slice
    disc = np.cumprod(1.0 / (1.0 + f))  # prod_{j=a+1}^{i} 1/(1+F_j)
    w = disc / disc.sum()
    s = float(np.dot(w, f))
    return w, s


def cascade_calibration(vol_matrix, fwd, corr):
    """Rectangular Cascade Calibration (Algorithm 7.4.2).

    Parameters
    ----------
    vol_matrix : (s x s) array, entry [a, c-1] = Black vol of the swaption
        with expiry a+1 years and tenor c years (NaN allowed = skip cell,
        e.g. for a triangular calibration).
    fwd : array of annual forwards F_1..F_M with M >= 2s - 1.
    corr : (M x M) instantaneous correlation matrix (positive entries).

    Returns
    -------
    sigma : (M x s) complex array; sigma[i-1, h-1] is TABLE 1 entry
        (F_i, interval h). NaN where never determined.
    """
    V = np.asarray(vol_matrix, dtype=float)
    s = V.shape[0]
    if V.shape[1] != s:
        raise ValueError("vol_matrix must be square (s expiries x s tenors)")
    M = 2 * s - 1
    if len(fwd) < M or corr.shape[0] < M:
        raise ValueError(f"need at least {M} forwards/correlation rows")

    sig = np.full((len(fwd), s), np.nan, dtype=complex)

    for a in range(s):                 # expiry row: T_a = a+1 years
        t_alpha = float(a + 1)
        for c in range(1, s + 1):      # tenor in years
            b = a + c                  # last payment index (1-based forward)
            v = V[a, c - 1]
            if np.isnan(v):
                continue
            if a == 0 and c == 1:
                sig[0, 0] = v          # (V_{0,1})^2 = sigma_{1,1}^2
                continue

            w, srate = swap_weights(fwd, a, b)
            wf = w * fwd[a: b]         # w_i(0) F_i(0), i = a+1..b
            rho = corr[a: b, a: b]

            last_col = (b == s + a) and (a >= 1)
            # intervals h = 1..a+1 (columns 0..a of sig); unknown lives in
            # column a; on the last column ALL of forward b's columns 0..a
            # are unknown and assumed equal (Algorithm 7.4.2).
            n_known_int = 0 if last_col else a

            # A: quadratic coefficient of the unknown
            A = wf[-1] ** 2 * (1.0 if not last_col else float(a + 1))

            # B: cross terms unknown x known forwards (i = a+1..b-1)
            B = 0.0 + 0.0j
            for j in range(b - 1 - a):         # j-th forward = a+1+j (1-based)
                cross = 2.0 * wf[-1] * wf[j] * rho[-1, j]
                B += cross * sig[a + j, a]     # interval a+1 term
                if last_col:
                    B += cross * np.sum(sig[a + j, :a])
            # C: fully-known block minus the market target
            C = -t_alpha * srate**2 * v**2 + 0.0j
            for i in range(b - 1 - a):
                for j in range(b - 1 - a):
                    C += (wf[i] * wf[j] * rho[i, j]
                          * np.sum(sig[a + i, : a + 1] * sig[a + j, : a + 1]))
            if not last_col and a >= 1:
                # known part of forward b's own history + cross history
                C += wf[-1] ** 2 * np.sum(sig[b - 1, :a] ** 2)
                for j in range(b - 1 - a):
                    C += (2.0 * wf[-1] * wf[j] * rho[-1, j]
                          * np.sum(sig[b - 1, :a] * sig[a + j, :a]))

            disc = B * B - 4.0 * A * C
            root = (-B + np.lib.scimath.sqrt(disc)) / (2.0 * A)
            if last_col:
                sig[b - 1, : a + 1] = root
            else:
                sig[b - 1, a] = root

    return sig


def reprice_swaption_vol(sigma, fwd, corr, a, c):
    """LFM swaption vol from calibrated sigmas via eq. (7.1) - for checks."""
    b = a + c
    w, srate = swap_weights(fwd, a, b)
    wf = w * fwd[a: b]
    rho = corr[a: b, a: b]
    total = 0.0 + 0.0j
    for i in range(b - a):
        for j in range(b - a):
            total += (wf[i] * wf[j] * rho[i, j]
                      * np.sum(sigma[a + i, : a + 1] * sigma[a + j, : a + 1]))
    t_alpha = float(a + 1)
    return np.sqrt(total / (t_alpha * srate**2))


def diagnose(sigma):
    """Classify each calibrated entry: 'ok', 'negative', or 'imaginary'."""
    out = np.full(sigma.shape, "", dtype=object)
    determined = ~np.isnan(sigma.real)
    out[determined & (np.abs(sigma.imag) > 1e-12)] = "imaginary"
    out[determined & (np.abs(sigma.imag) <= 1e-12) & (sigma.real < 0)] = "negative"
    out[determined & (np.abs(sigma.imag) <= 1e-12) & (sigma.real >= 0)] = "ok"
    return out
