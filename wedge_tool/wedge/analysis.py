"""Rich/cheap wedge analysis per expiry bucket (Brigo & Mercurio 6.20 + 7.9)."""

import numpy as np
import pandas as pd

from .annualize import (
    nesting_weights,
    annual_forward,
    cap_implied_swaption_vol,
    implied_infra_correlation,
)

REQUIRED_COLUMNS = ["expiry", "swaption_vol", "f1", "f2", "caplet_vol_s", "caplet_vol_t"]


def analyze_wedges(
    quotes: pd.DataFrame,
    rho: float = 1.0,
    rho_low: float = 0.70,
    rho_high: float = 1.00,
    min_wedge: float = 0.005,
    tau1: float = 0.5,
    tau2: float = 0.5,
) -> pd.DataFrame:
    """Compute the cap/swaption wedge and a rich/cheap signal per bucket.

    Parameters
    ----------
    quotes : DataFrame with one row per expiry bucket and columns
        expiry        swaption expiry S in years (the bucket label)
        swaption_vol  market Black vol of the S x 1 swaption
        f1, f2        semi-annual forwards F(0;S,S+6m) and F(0;S+6m,S+1y)
        caplet_vol_s  market Black vol of the S-expiry caplet
        caplet_vol_t  market Black vol of the (S+6m)-expiry caplet
    rho : infra-correlation assumed when building the cap-implied
        swaption vol (the benchmark). The book uses 1.
    rho_low, rho_high : plausible band for the implied infra-correlation.
        Implied rho above the band = swaption RICH vs caps,
        below the band = swaption CHEAP vs caps.
    min_wedge : minimum |market - implied| vol gap (absolute, e.g. 0.005
        = 0.5 vol points) before a bucket may be flagged; smaller gaps
        are treated as noise and marked FAIR.
    tau1, tau2 : year fractions of the two nested forward rates.

    Returns a DataFrame with the inputs plus diagnostics, one row per bucket.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in quotes.columns]
    if missing:
        raise ValueError(f"quotes is missing columns: {missing}")

    df = quotes.copy().reset_index(drop=True)

    u1, u2 = nesting_weights(df.f1, df.f2, tau1, tau2)
    df["annual_forward"] = annual_forward(df.f1, df.f2, tau1, tau2)
    df["u1"] = u1
    df["u2"] = u2
    df["cap_implied_vol"] = cap_implied_swaption_vol(
        df.f1, df.f2, df.caplet_vol_s, df.caplet_vol_t, rho, tau1, tau2
    )
    df["wedge"] = df.swaption_vol - df.cap_implied_vol
    df["implied_rho"] = implied_infra_correlation(
        df.swaption_vol, df.f1, df.f2, df.caplet_vol_s, df.caplet_vol_t, tau1, tau2
    )

    # Cross-sectional z-score of the wedge: which buckets stick out most
    # against today's own curve (guards against a level shift fooling the
    # absolute threshold).
    wedge_std = df.wedge.std(ddof=0)
    df["wedge_zscore"] = 0.0 if wedge_std == 0 else (df.wedge - df.wedge.mean()) / wedge_std

    df["signal"] = [
        _classify(w, r, rho_low, rho_high, min_wedge)
        for w, r in zip(df.wedge, df.implied_rho)
    ]
    df["trade"] = [
        _suggest_trade(sig, exp, w1, w2)
        for sig, exp, w1, w2 in zip(df.signal, df.expiry, df.u1, df.u2)
    ]
    return df


def _classify(wedge, implied_rho, rho_low, rho_high, min_wedge):
    if abs(wedge) < min_wedge:
        return "FAIR"
    if implied_rho > rho_high and wedge > 0:
        return "RICH"
    if implied_rho < rho_low and wedge < 0:
        return "CHEAP"
    return "FAIR"


def _suggest_trade(signal, expiry, u1, u2):
    """Vega-neutral wedge trade for one bucket.

    From eq. (6.76) with rho ~ 1, v_swaption ~ u1*v1 + u2*v2, so per unit
    of swaption vega the offsetting caplet legs carry vegas u1 and u2.
    RICH swaption -> sell swaption vol, buy caplet vol; CHEAP -> reverse.
    """
    y = f"{expiry:g}y"
    caplets = f"{u1:.2f} vega {y} caplet + {u2:.2f} vega {y}+6m caplet"
    if signal == "RICH":
        return f"sell {y}x1y swaption straddle, buy {caplets}"
    if signal == "CHEAP":
        return f"buy {y}x1y swaption straddle, sell {caplets}"
    return "no trade"
