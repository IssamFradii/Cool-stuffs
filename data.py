"""Synthetic rates-desk data.

Everything the dashboard shows is generated here so the app runs with zero
setup. When you plug in the real thing, keep the FUNCTION SIGNATURES and the
column names identical -- then `app.py`, `charts.py` and `tables.py` do not
change at all. That is the whole point of keeping the data layer separate.

Replace, in this order of usefulness:
  * `MarketState.curve`          -> your bootstrapped par-swap curve
  * `MarketState.vol_surface`    -> your normal (bp) swaption vol cube slice
  * `MarketState.wedge`          -> the z-scores your wedge_tool already computes
  * `MarketState.positions`      -> the desk blotter
  * `MarketState.xva`            -> your CVA/DVA/FVA engine output
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

SEED = 20260729

# Grid definitions. Kept module-level so charts and tables agree on ordering.
CURVE_TENORS = ['1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '15Y', '20Y', '30Y']
CURVE_YEARS = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30], dtype=float)

EXPIRIES = ['1M', '3M', '6M', '1Y', '2Y', '5Y', '10Y']
TAILS = ['1Y', '2Y', '5Y', '10Y', '20Y', '30Y']

RISK_BUCKETS = ['3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '15Y', '20Y', '30Y']

COUNTERPARTIES = ['Meridian Bank', 'Northgate AM', 'Cordillera Cap',
                  'Halbrook Pension', 'Vantage Clearing', 'Astoria Funds']

INSTRUMENTS = [
    ('EUR 10Y Payer Swap', 'EUR', 'Swaps'),
    ('EUR 5Y Receiver Swap', 'EUR', 'Swaps'),
    ('EUR 2Y5Y Payer Swaption', 'EUR', 'Vol'),
    ('EUR 1Y10Y Straddle', 'EUR', 'Vol'),
    ('EUR 5Y30Y Steepener', 'EUR', 'Curve'),
    ('USD 10Y Payer Swap', 'USD', 'Swaps'),
    ('USD 3M/6M Basis', 'USD', 'Basis'),
    ('USD 1Y1Y Cap', 'USD', 'Vol'),
    ('USD 2Y10Y Flattener', 'USD', 'Curve'),
    ('GBP 10Y Receiver Swap', 'GBP', 'Swaps'),
    ('GBP 5Y5Y Payer Swaption', 'GBP', 'Vol'),
    ('EUR 30Y ASW', 'EUR', 'Basis'),
]


def _nelson_siegel(years: np.ndarray, beta0: float, beta1: float,
                   beta2: float, tau: float) -> np.ndarray:
    """Classic Nelson-Siegel zero curve -- gives a realistic curve shape."""
    x = years / tau
    loading = (1 - np.exp(-x)) / x
    return beta0 + beta1 * loading + beta2 * (loading - np.exp(-x))


@dataclass
class MarketState:
    """One snapshot of the desk. Call `tick()` to move it forward."""

    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(SEED))
    ticks: int = 0

    def __post_init__(self) -> None:
        # Curve parameters get nudged on every tick, which moves the whole
        # curve coherently rather than jiggling each tenor independently.
        self._beta = np.array([3.05, -1.35, 1.60])
        self._tau = 2.6
        self._curve_prev = self._build_curve(self._beta)
        self._vol_base = self._build_vol_surface()
        self._wedge = self._build_wedge()
        self._positions = self._build_positions()
        self._pnl_history = self._build_pnl_history()
        self._xva = self._build_xva()

    # -- builders ---------------------------------------------------------
    def _build_curve(self, beta: np.ndarray) -> np.ndarray:
        return _nelson_siegel(CURVE_YEARS, beta[0], beta[1], beta[2], self._tau)

    def _build_vol_surface(self) -> np.ndarray:
        """Normal (bp/day annualised) swaption vols, expiry x tail."""
        e = np.array([1 / 12, 0.25, 0.5, 1, 2, 5, 10])[:, None]
        t = np.array([1, 2, 5, 10, 20, 30])[None, :]
        # Vol falls with expiry (mean reversion) and with tail (averaging).
        surface = 92 * np.exp(-0.18 * np.log1p(e)) * (1 + 0.35 * np.exp(-0.25 * t))
        surface += self.rng.normal(0, 1.6, surface.shape)
        return surface

    def _build_wedge(self) -> np.ndarray:
        """Rich/cheap z-scores: model vol minus market vol, standardised.

        Positive = market trades ABOVE model -> rich -> a sell candidate.
        This is the same quantity your wedge_tool produces.
        """
        base = self.rng.normal(0, 1.0, (len(EXPIRIES), len(TAILS)))
        # Smooth it so neighbouring cells are related, like a real surface.
        smooth = base.copy()
        smooth[1:-1, :] = (base[:-2, :] + 2 * base[1:-1, :] + base[2:, :]) / 4
        smooth[:, 1:-1] = (smooth[:, :-2] + 2 * smooth[:, 1:-1] + smooth[:, 2:]) / 4
        smooth[2, 3] = 2.6      # a clear rich pocket, so the heatmap has a story
        smooth[5, 1] = -2.4     # and a clear cheap pocket
        return smooth * 1.15

    def _build_positions(self) -> pd.DataFrame:
        n = len(INSTRUMENTS)
        rng = self.rng
        notional = rng.choice([25, 50, 75, 100, 150, 200, 250], n) * 1e6
        side = rng.choice([1, -1], n)
        dv01 = side * notional / 1e6 * rng.uniform(0.4, 1.1, n) * 1e3
        return pd.DataFrame({
            'instrument': [i[0] for i in INSTRUMENTS],
            'ccy': [i[1] for i in INSTRUMENTS],
            'book': [i[2] for i in INSTRUMENTS],
            'notional': notional,
            'dv01': dv01,
            'mtm': rng.normal(0, 6e5, n),
            'day_pnl': rng.normal(0, 9e4, n),
            'vega': np.where([i[2] == 'Vol' for i in INSTRUMENTS],
                             rng.normal(0, 4e4, n), 0.0),
        })

    def _build_pnl_history(self) -> pd.DataFrame:
        days = 90
        rng = self.rng
        idx = [date.today() - timedelta(days=days - 1 - i) for i in range(days)]
        return pd.DataFrame({
            'date': idx,
            'Rates': np.cumsum(rng.normal(4_500, 62_000, days)),
            'Vol': np.cumsum(rng.normal(2_800, 38_000, days)),
            'Carry': np.cumsum(rng.normal(6_200, 9_000, days)),
        })

    def _build_xva(self) -> pd.DataFrame:
        rng = self.rng
        n = len(COUNTERPARTIES)
        epe = rng.uniform(4, 26, n) * 1e6
        return pd.DataFrame({
            'counterparty': COUNTERPARTIES,
            'rating': rng.choice(['AA-', 'A+', 'A', 'A-', 'BBB+'], n),
            'epe': epe,
            'cva': -epe * rng.uniform(0.004, 0.019, n),
            'dva': epe * rng.uniform(0.002, 0.008, n),
            'fva': -epe * rng.uniform(0.003, 0.014, n),
            'colva': -epe * rng.uniform(0.0005, 0.004, n),
            'csa': rng.choice(['CSA', 'CSA', 'CSA', 'No CSA', 'One-way'], n),
        })

    # -- live tick --------------------------------------------------------
    def tick(self) -> None:
        """Advance the market one step. Everything downstream re-reads state."""
        self.ticks += 1
        self._curve_prev = self.curve.copy() if self.ticks == 1 else self._curve_prev
        self._beta = self._beta + self.rng.normal(0, [0.010, 0.014, 0.020])
        self._vol_base += self.rng.normal(0, 0.22, self._vol_base.shape)
        self._wedge += self.rng.normal(0, 0.045, self._wedge.shape)
        self._wedge = np.clip(self._wedge, -3.2, 3.2)
        self._positions['mtm'] += self.rng.normal(0, 3.5e4, len(self._positions))
        self._positions['day_pnl'] += self.rng.normal(0, 1.4e4, len(self._positions))

    # -- accessors used by the UI ----------------------------------------
    @property
    def curve(self) -> np.ndarray:
        return self._build_curve(self._beta)

    @property
    def curve_prev(self) -> np.ndarray:
        return self._curve_prev

    @property
    def vol_surface(self) -> np.ndarray:
        return self._vol_base

    @property
    def wedge(self) -> np.ndarray:
        return self._wedge

    @property
    def positions(self) -> pd.DataFrame:
        return self._positions

    @property
    def pnl_history(self) -> pd.DataFrame:
        return self._pnl_history

    @property
    def xva(self) -> pd.DataFrame:
        return self._xva

    def risk_ladder(self) -> pd.DataFrame:
        """DV01 spread across the standard bucket set, plus a vega column."""
        rng = np.random.default_rng(SEED + 7)
        weights = rng.normal(0, 1, len(RISK_BUCKETS))
        weights[5] += 1.4     # a deliberate 5Y length
        weights[8] -= 1.8     # and a 15Y short
        total = self._positions['dv01'].sum()
        dv01 = weights / np.abs(weights).sum() * abs(total) * 3.2
        return pd.DataFrame({
            'bucket': RISK_BUCKETS,
            'dv01': dv01,
            'limit': np.abs(dv01).max() * 1.25,
            'vega': rng.normal(0, 2.2e4, len(RISK_BUCKETS)),
        })

    # -- headline numbers -------------------------------------------------
    def kpis(self) -> list[dict]:
        pos = self._positions
        xva = self._xva
        day = pos['day_pnl'].sum()
        mtd = day * 17.4 + 1.9e6
        dv01 = pos['dv01'].sum()
        vega = pos['vega'].sum()
        var95 = -abs(dv01) * 6.4 - 380_000
        fva = xva['fva'].sum()
        cva = xva['cva'].sum()
        # `tone` says whether the MOVE was good or bad; the arrow says which
        # way it moved. Keeping those separate matters: VaR going up is a rise
        # AND a bad thing, while DV01 going up is neither good nor bad -- it is
        # just a position. Colouring all three by sign would lie about two.
        def tone(delta: float, higher_is_better: bool | None) -> str:
            if higher_is_better is None:
                return 'neutral'
            return 'good' if (delta >= 0) == higher_is_better else 'bad'

        rows = [
            ('Day P&L', day, day / 1e6 * 3.1, 'vs yesterday', True),
            ('MTD P&L', mtd, 4.8, 'month to date', True),
            ('Net DV01', dv01, dv01 / 4200, 'per bp, all books', None),
            ('Net Vega', vega, -2.6, 'per vol point', None),
            ('VaR 95% 1d', var95, 1.9, 'historical sim', False),
            ('FVA', fva, fva / abs(cva) * 10, 'funding valuation adj', True),
        ]
        return [{'label': lbl, 'value': val, 'delta': d, 'hint': hint,
                 'tone': tone(d, better)}
                for lbl, val, d, hint, better in rows]


# --------------------------------------------------------------------------
# Formatting helpers -- used by both the tiles and the tables so a number
# never appears in two different shapes on the same screen.
# --------------------------------------------------------------------------
def fmt_ccy(x: float, decimals: int = 0) -> str:
    """Compact money: 1.2m / 340k / 812."""
    sign = '-' if x < 0 else ''
    a = abs(x)
    if a >= 1e9:
        return f'{sign}{a / 1e9:,.2f}bn'
    if a >= 1e6:
        return f'{sign}{a / 1e6:,.2f}m'
    if a >= 1e3:
        return f'{sign}{a / 1e3:,.0f}k'
    return f'{sign}{a:,.{decimals}f}'


def fmt_pct(x: float) -> str:
    return f'{x:+.2f}%'
