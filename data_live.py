"""Live market curve: US Treasury daily par yield curve (treasury.gov),
free and key-less, conveniently quoting exactly the 2y/5y/10y/20y pillars.
There is no free *live* swap-rate feed (ICE/Bloomberg/Refinitiv all require
a paid license), so this is used as the market curve proxy, optionally
bumped by a flat swap spread to stand in for a pseudo-swap curve.
"""

import datetime as dt
import io

import pandas as pd
import requests

TENOR_YEARS = {
    "1 Mo": 1 / 12, "1.5 Month": 1.5 / 12, "2 Mo": 2 / 12, "3 Mo": 0.25, "4 Mo": 4 / 12,
    "6 Mo": 0.5, "1 Yr": 1.0, "2 Yr": 2.0, "3 Yr": 3.0, "5 Yr": 5.0, "7 Yr": 7.0,
    "10 Yr": 10.0, "20 Yr": 20.0, "30 Yr": 30.0,
}

_URL_TEMPLATE = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
    "&field_tdr_date_value={year}&page&_format=csv"
)


def _fetch_year(year, timeout):
    resp = requests.get(_URL_TEMPLATE.format(year=year), timeout=timeout)
    resp.raise_for_status()
    return pd.read_csv(io.StringIO(resp.text))


def fetch_treasury_curve(year=None, timeout=10):
    """Most recent published daily par yield curve, tidied to (tenor_label, T, rate_pct)."""
    year = year or dt.date.today().year
    df = _fetch_year(year, timeout)
    if df.empty:
        df = _fetch_year(year - 1, timeout)

    latest = df.iloc[0]
    rows = [
        {"tenor_label": col, "T": years, "rate_pct": float(latest[col])}
        for col, years in TENOR_YEARS.items()
        if col in df.columns and pd.notna(latest[col])
    ]
    curve = pd.DataFrame(rows).sort_values("T").reset_index(drop=True)
    curve.attrs["as_of_date"] = str(latest["Date"])
    return curve


def to_pseudo_swap(curve, spread_bp=20.0):
    """Flat-spread stand-in for a swap curve: swap = Treasury + spread."""
    out = curve.copy()
    out["rate_pct"] = out["rate_pct"] + spread_bp / 100.0
    out.attrs.update(curve.attrs)
    out.attrs["spread_bp"] = spread_bp
    return out
