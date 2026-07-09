"""Build a price surface from forward/vol panels, and calendar spreads from it.

Panels are pandas DataFrames: rows = dates, columns = MultiIndex (tenor, expiry).
This shape is the one thing every signal in this package agrees on -- swap the
synthetic panels for real market panels later without touching signals.py.
"""

from __future__ import annotations

import pandas as pd

from .pricer import atm_price


def price_surface(forward_panel: pd.DataFrame, vol_panel: pd.DataFrame,
                   bpv_panel: pd.DataFrame, expiry_years: dict) -> pd.DataFrame:
    """ATM price panel, same shape (dates x (tenor, expiry)) as the inputs."""
    out = pd.DataFrame(index=forward_panel.index, columns=forward_panel.columns, dtype=float)
    for tenor, expiry in forward_panel.columns:
        F = forward_panel[(tenor, expiry)].values
        sigma = vol_panel[(tenor, expiry)].values
        bpv = bpv_panel[(tenor, expiry)].values
        T = expiry_years[expiry]
        out[(tenor, expiry)] = atm_price(F, sigma, T, bpv)
    return out


def calendar_series(price_panel: pd.DataFrame, tenor, expiry_front, expiry_back) -> pd.Series:
    """Calendar = back-leg price - front-leg price, same tenor, over time.

    e.g. calendar_series(prices, tenor=30, expiry_front="1Y", expiry_back="2Y")
    is the 2Y30Y - 1Y30Y calendar time series.
    """
    return price_panel[(tenor, expiry_back)] - price_panel[(tenor, expiry_front)]


def calendar_grid(price_panel: pd.DataFrame, tenor, expiries_ordered: list[str]) -> pd.DataFrame:
    """All consecutive-expiry calendars for one tenor, as a panel over time.

    Column labels are "{back}-{front}", e.g. "2Y-1Y", "3Y-2Y", ... This is the
    "calendar surface" that signals 3/4 run PCA denoising on: each column is one
    calendar, each row is a historical date.
    """
    cols = {}
    for front, back in zip(expiries_ordered[:-1], expiries_ordered[1:]):
        cols[f"{back}-{front}"] = calendar_series(price_panel, tenor, front, back)
    return pd.DataFrame(cols, index=price_panel.index)


def normalized_calendar_grid(price_panel: pd.DataFrame, tenor, expiries_ordered: list[str]) -> pd.DataFrame:
    """Same as calendar_grid, but each calendar is divided by its front-leg price
    (the "DF" / discounted normalization from signal 2, applied cell by cell so
    signal 4 can PCA-denoise the normalized surface the same way signal 3 does
    for the raw one).
    """
    cols = {}
    for front, back in zip(expiries_ordered[:-1], expiries_ordered[1:]):
        cal = calendar_series(price_panel, tenor, front, back)
        front_leg = price_panel[(tenor, front)]
        cols[f"{back}-{front}"] = cal / front_leg
    return pd.DataFrame(cols, index=price_panel.index)
