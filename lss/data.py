"""
Loading real market data.

The pipeline runs on synthetic data out of the box.  To run it on real
Bloomberg (or any other) quotes, drop three CSV files into a directory and
call :func:`load_market_data`.  Nothing else in the package changes.

Expected files
--------------
``rates.csv``
    date, libor_6m, libor_1y, swap_2y, swap_3y, swap_4y, swap_5y, swap_7y, swap_10y

``swaption_vols.csv``
    date, then one column per structure named ``<expiry>x<tenor>``, e.g.
    ``0.5x1``, ``1x1``, ``2x1``, ... , ``3x7``.  All 34 structures of Table 3.

``cap_vols.csv``
    date, cap_2y, cap_3y, cap_4y, cap_5y, cap_7y, cap_10y

Rates and volatilities may be given either in percent (5.75, 16.1) or as
decimals (0.0575, 0.161); the loader detects which and normalises to decimals.
Dates are parsed with pandas and rows are aligned across the three files on
their common dates.

The forward curve is then built from the rates by the paper's own procedure
(spline the par curve, bootstrap), so `forwards` is derived, not supplied.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from .conventions import SWAPTION_GRID, CAP_MATURITIES, SWAP_PILLARS
from .curve import build_curve
from .synthetic import MarketData


# ---------------------------------------------------------------------------
# Column naming
# ---------------------------------------------------------------------------
def swaption_column(expiry: float, tenor: float) -> str:
    """Column name for a swaption structure, e.g. (0.5, 1.0) -> '0.5x1'."""
    def fmt(x: float) -> str:
        return f"{x:g}"
    return f"{fmt(expiry)}x{fmt(tenor)}"


def cap_column(maturity: float) -> str:
    """Column name for a cap maturity, e.g. 2.0 -> 'cap_2y'."""
    return f"cap_{maturity:g}y"


RATE_COLUMNS = ["libor_6m", "libor_1y"] + [f"swap_{t:g}y" for t in SWAP_PILLARS]


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def _to_decimal(values: np.ndarray, what: str) -> np.ndarray:
    """
    Normalise rates or volatilities to decimals.

    Quotes above 1.0 are taken to be percentages.  Interest rates and implied
    volatilities in this market sit in the 1-30 range when quoted in percent
    and the 0.01-0.30 range as decimals, so the two are unambiguous -- but a
    mixed-unit file is rejected rather than silently mangled.
    """
    v = np.asarray(values, dtype=float)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        raise ValueError(f"{what}: no finite values")

    looks_percent = finite > 1.0
    if looks_percent.all():
        return v / 100.0
    if not looks_percent.any():
        return v
    raise ValueError(
        f"{what}: values span both percent and decimal ranges "
        f"(min {finite.min():g}, max {finite.max():g}); "
        "please use consistent units"
    )


def _read_csv(path: Path, what: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{what} file not found: {path}")
    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        raise ValueError(f"{path.name} must have a 'date' column")
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date").set_index("date")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_market_data(
    directory: str | Path,
    rates_file: str = "rates.csv",
    swaption_file: str = "swaption_vols.csv",
    cap_file: str = "cap_vols.csv",
) -> MarketData:
    """
    Load real market data from CSVs and build the weekly forward curves.

    Returns a :class:`~lss.synthetic.MarketData` with ``is_synthetic=False``,
    which every downstream module consumes identically to synthetic data.
    """
    directory = Path(directory)

    rates = _read_csv(directory / rates_file, "rates")
    swaptions = _read_csv(directory / swaption_file, "swaption volatility")
    caps = _read_csv(directory / cap_file, "cap volatility")

    missing = [c for c in RATE_COLUMNS if c not in rates.columns]
    if missing:
        raise ValueError(f"{rates_file} is missing columns: {missing}")

    swaption_cols = [swaption_column(e, t) for e, t in SWAPTION_GRID]
    missing = [c for c in swaption_cols if c not in swaptions.columns]
    if missing:
        raise ValueError(f"{swaption_file} is missing columns: {missing}")

    cap_cols = [cap_column(T) for T in CAP_MATURITIES]
    missing = [c for c in cap_cols if c not in caps.columns]
    if missing:
        raise ValueError(f"{cap_file} is missing columns: {missing}")

    # Align on dates present in all three files.
    dates = rates.index.intersection(swaptions.index).intersection(caps.index)
    if len(dates) == 0:
        raise ValueError("no dates common to all three files")

    rates = rates.loc[dates]
    swaptions = swaptions.loc[dates, swaption_cols]
    caps = caps.loc[dates, cap_cols]

    incomplete = (
        rates[RATE_COLUMNS].isna().any(axis=1)
        | swaptions.isna().any(axis=1)
        | caps.isna().any(axis=1)
    )
    if incomplete.any():
        dropped = int(incomplete.sum())
        dates = dates[~incomplete.to_numpy()]
        rates = rates.loc[dates]
        swaptions = swaptions.loc[dates]
        caps = caps.loc[dates]
        print(f"  dropped {dropped} week(s) with incomplete quotes")

    rate_values = _to_decimal(rates[RATE_COLUMNS].to_numpy(), "rates")
    swaption_vols = _to_decimal(swaptions.to_numpy(), "swaption volatilities")
    cap_vols = _to_decimal(caps.to_numpy(), "cap volatilities")

    # Build each week's curve, then extract the 19 forwards.
    forwards = np.empty((len(dates), 19))
    for week in range(len(dates)):
        row = rate_values[week]
        swap_rates = {t: row[2 + k] for k, t in enumerate(SWAP_PILLARS)}
        curve = build_curve(row[0], row[1], swap_rates)
        forwards[week] = curve.forwards

    return MarketData(
        dates=pd.DatetimeIndex(dates),
        forwards=forwards,
        swaption_vols=swaption_vols,
        cap_vols=cap_vols,
        libor_6m=rate_values[:, 0],
        swaption_grid=list(SWAPTION_GRID),
        cap_maturities=tuple(CAP_MATURITIES),
        is_synthetic=False,
    )


# ---------------------------------------------------------------------------
# Writing, for round-tripping synthetic data to disk
# ---------------------------------------------------------------------------
def save_market_data(data: MarketData, directory: str | Path) -> None:
    """
    Write a :class:`MarketData` out in the CSV layout the loader expects.

    Useful for inspecting the synthetic data, and as a template showing exactly
    what real files should look like.  Forwards are re-expressed as the
    underlying Libor and swap quotes so the files round-trip through the same
    curve build a real data set would go through.

    Note that this round trip is deliberately lossy for the forward curve.
    Only the eight market quotes are written, and reloading re-derives the
    nineteen forwards by splining them -- so a nineteen-point curve cannot come
    back unchanged (expect differences of tens of basis points on synthetic
    data).  Volatilities round-trip exactly.  This is not a defect: it is
    exactly what happens with real data, where eight quotes are observed and
    the rest of the curve is an interpolation choice.  It does mean real
    forward curves are spline-consistent by construction, while the synthetic
    generator's are not.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    rows = []
    for week in range(data.n_weeks):
        curve = data.curve(week)
        row = {
            "date": data.dates[week].date(),
            "libor_6m": 100.0 * data.libor_6m[week],
            "libor_1y": 100.0 * curve.par_rates[1],
        }
        for t in SWAP_PILLARS:
            row[f"swap_{t:g}y"] = 100.0 * curve.par_swap_rate(t)
        rows.append(row)
    pd.DataFrame(rows).to_csv(directory / "rates.csv", index=False)

    frame = pd.DataFrame(
        100.0 * data.swaption_vols,
        columns=[swaption_column(e, t) for e, t in data.swaption_grid],
    )
    frame.insert(0, "date", [d.date() for d in data.dates])
    frame.to_csv(directory / "swaption_vols.csv", index=False)

    frame = pd.DataFrame(
        100.0 * data.cap_vols,
        columns=[cap_column(T) for T in data.cap_maturities],
    )
    frame.insert(0, "date", [d.date() for d in data.dates])
    frame.to_csv(directory / "cap_vols.csv", index=False)
