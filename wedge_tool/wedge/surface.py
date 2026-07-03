"""Cap-implied swaption vol surface and the wedge per expiry x tenor.

Builds, from annualized caplet volatilities alone, the swaption vol
surface a caps-calibrated LFM would imply, then compares it with the
market swaption matrix:

    wedge(expiry, tenor) = market vol - cap-implied vol

The cap-implied vol uses Rebonato's formula (6.67) under the simplest
assumption that keeps it market-data-only (as in Section 6.20 "a2"):
each annual forward F_i has constant instantaneous vol equal to its
annualized caplet vol v_i, so

    V_imp(a,b)^2 = sum_{i,j} w_i w_j F_i F_j rho_ij v_i v_j / S_{a,b}^2.

For tenor = 1 this collapses to the annualized caplet vol, consistent
with the per-bucket tool. Positive wedge = swaption RICH vs caps.
"""

import numpy as np
import pandas as pd

from .cca import swap_weights


def cap_implied_surface(fwd, caplet_vols, corr, n_expiry, n_tenor):
    """Cap-implied swaption vols, (n_expiry x n_tenor) array.

    fwd, caplet_vols: 0-based arrays for annual forwards F_1..F_M and
    their annualized caplet vols (need M >= n_expiry + n_tenor - 1).
    corr: (M x M) instantaneous correlation of the forwards.
    """
    fwd = np.asarray(fwd, dtype=float)
    v = np.asarray(caplet_vols, dtype=float)
    need = n_expiry + n_tenor - 1
    if len(fwd) < need or len(v) < need or corr.shape[0] < need:
        raise ValueError(f"need at least {need} forwards/vols/correlation rows")

    out = np.empty((n_expiry, n_tenor))
    for a in range(n_expiry):
        for c in range(1, n_tenor + 1):
            b = a + c
            w, srate = swap_weights(fwd, a, b)
            wf = w * fwd[a: b]
            vv = v[a: b]
            rho = corr[a: b, a: b]
            var = float((wf * vv) @ rho @ (wf * vv)) / srate**2
            out[a, c - 1] = np.sqrt(var)
    return out


def wedge_surface(market_matrix: pd.DataFrame, curve: pd.DataFrame, corr) -> dict:
    """Market minus cap-implied vol per (expiry, tenor).

    market_matrix : DataFrame indexed by expiry (years), columns = tenor
        (years), values = market Black vols (NaN allowed for missing quotes).
    curve : DataFrame with columns expiry, forward, caplet_vol (annual grid,
        expiry = 1..M years in order).
    corr : (M x M) correlation matrix of the annual forwards.

    Returns dict with DataFrames: 'market', 'implied', 'wedge' (all in the
    market matrix's shape).
    """
    n_expiry, n_tenor = market_matrix.shape
    implied = cap_implied_surface(
        curve.forward.to_numpy(), curve.caplet_vol.to_numpy(),
        corr, n_expiry, n_tenor,
    )
    implied_df = pd.DataFrame(implied, index=market_matrix.index,
                              columns=market_matrix.columns)
    return {
        "market": market_matrix,
        "implied": implied_df,
        "wedge": market_matrix - implied_df,
    }
