"""
Where the BPV (= DV01, no notional) comes from.

`fva.quote(..., bpv=<source>)` accepts any callable (start, end) -> BPV in years.
The three calls it makes map 1:1 onto the legs of the replication - for a 6m1y10y FVA
(exercise 6M, forward term 18M, swap tenor 10Y):

    bpv(T2, T3) = bpv(1.5, 11.5) -> 18M10Y   forward swap the FVA delivers into
    bpv(T1, T3) = bpv(0.5, 11.5) -> 6M11Y    long leg of the mid-curve decomposition
    bpv(T1, T2) = bpv(0.5, 1.5)  -> 6M1Y     short leg of the mid-curve decomposition

Three sources are provided:
    CurveBpv    - the built-in curve (default, self-contained)
    TableBpv    - BPVs computed elsewhere and exported as CSV
    LegsCalcBpv - live adapter for rdws SwapLegsValueCalculator
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

import fva


def term_label(years: float) -> str:
    """0.5 -> '6M', 1.5 -> '18M', 10 -> '10Y' - the calculator's tenor style."""
    return fva.label_years(years).upper()


class CurveBpv:
    """BPV from the built-in discount curve: sum(tau_i * DF(t_i))."""

    def __init__(self, curve: fva.Curve) -> None:
        self.curve = curve

    def __call__(self, start: float, end: float) -> float:
        return self.curve.bpv(start, end)


class TableBpv:
    """BPVs computed outside this tool and handed over as a table.

    CSV columns: forward_term, swap_tenor, bpv   e.g.

        forward_term,swap_tenor,bpv
        18M,10Y,8.60
        6M,11Y,9.60
        6M,1Y,1.00

    `forward_term` is the swap start (= T2 for the forward swap, T1 for the mid-curve
    legs) and `swap_tenor` its length, exactly like SwapLegsValueCalculator's arguments.
    Anything missing from the table falls back to `fallback` (typically the curve) and
    is recorded in `.missing` so the app can report what still needs exporting.
    """

    def __init__(
        self, table: dict[tuple[float, float], float], fallback: Callable[[float, float], float]
    ) -> None:
        self.table = table
        self.fallback = fallback
        self.missing: set[tuple[str, str]] = set()

    @classmethod
    def from_frame(cls, df: pd.DataFrame, fallback: Callable[[float, float], float]) -> "TableBpv":
        cols = {c.lower().strip(): c for c in df.columns}
        need = ("forward_term", "swap_tenor", "bpv")
        if not all(c in cols for c in need):
            raise ValueError(f"BPV table needs columns {need}, got {list(df.columns)}")
        table: dict[tuple[float, float], float] = {}
        for _, row in df.iterrows():
            start = fva.parse_period(row[cols["forward_term"]])
            length = fva.parse_period(row[cols["swap_tenor"]])
            table[(round(start, 6), round(length, 6))] = float(row[cols["bpv"]])
        return cls(table, fallback)

    def __call__(self, start: float, end: float) -> float:
        key = (round(start, 6), round(end - start, 6))
        if key in self.table:
            return self.table[key]
        self.missing.add((term_label(start), term_label(end - start)))
        return self.fallback(start, end)


class LegsCalcBpv:
    """Adapter for rdws.utils.riskmetrics.SwapLegsValueCalculator.

    Mirrors the notebook's `get_legs`: every BPV is a forward swap value keyed by
    (forward_term, swap_tenor), so a single entry point covers the forward swap and
    both mid-curve decomposition legs.

        from rdws.utils.riskmetrics.SwapLegsValueCalculator import SwapLegsValueCalculator

        calculator_kwargs = dict(
            market=market,
            currency=CURRENCY,
            swap_curve_family=SWAP_CURVE_FAMILY,
            rate_stripping_configuration_name="PreTrade_Standard",
            rate_conventions_configuration_name="RateStatic_PreTrade",
            bootstrap_rates=False,
        )
        bpv = LegsCalcBpv(SwapLegsValueCalculator, calculator_kwargs, extract="bpv")
        res = fva.quote(curve, surface, 0.5, 1.0, 10.0, rho=1.0, bpv=bpv)

    `extract` names the BPV field on whatever `forward_swap_legs_value` returns; pass a
    callable if it needs unpacking.  Left as None it tries the usual names and raises
    with the list of available fields, so you only have to name it once.
    Results are cached: a full expiry x tenor grid asks for the same legs repeatedly.
    """

    _CANDIDATES = ("bpv", "BPV", "dv01", "DV01", "pv01", "PV01", "annuity", "Annuity")

    def __init__(
        self,
        calculator,
        calculator_kwargs: dict,
        extract: str | Callable[[object], float] | None = None,
        scale: float = 1.0,
    ) -> None:
        self.calculator = calculator
        self.kwargs = calculator_kwargs
        self.extract = extract
        self.scale = scale  # e.g. 1e-2 if the calculator returns $/bp per $1mm
        self._cache: dict[tuple[str, str], float] = {}

    def _value(self, obj: object) -> float:
        if callable(self.extract):
            return float(self.extract(obj))
        names = (self.extract,) if self.extract else self._CANDIDATES
        for name in names:
            if isinstance(obj, dict) and name in obj:
                return float(obj[name])
            if hasattr(obj, name):
                got = getattr(obj, name)
                return float(got() if callable(got) else got)
        available = list(obj) if isinstance(obj, dict) else [a for a in dir(obj) if a[0] != "_"]
        raise KeyError(
            f"Cannot find the BPV field on {type(obj).__name__}. Pass extract='<field>' "
            f"or a callable. Available: {available}"
        )

    def __call__(self, start: float, end: float) -> float:
        key = (term_label(start), term_label(end - start))
        if key not in self._cache:
            legs = self.calculator.forward_swap_legs_value(
                forward_term=key[0], swap_tenor=key[1], **self.kwargs
            )
            self._cache[key] = self._value(legs) * self.scale
        return self._cache[key]
