"""
The cap/swaption "wedge" relative-value strategy.

The paper stops short of trading its own result, but points straight at it:

  "Our results suggest the possibility that while buy-and-hold arbitrages may
   not be feasible, dynamic trading strategies exploiting inconsistencies in
   the relative valuation of caps and swaptions may be profitable."

and gives the two facts a strategy needs.  Table 9: cap pricing errors have
medians near zero but large, periodic, *positive* excursions -- so the wedge is
a mean-reverting spread, not a permanent level.  And those errors carry
first-order serial correlation around 0.67, so the dislocation persists long
enough to be traded at weekly frequency.

The trade
---------
Signal.  Each week, calibrate the model to swaptions, price the caps off the
main diagonal of the implied Sigma, and take the difference between the model
and market flat cap volatility:

    spread_t(T) = model_vol_t(T) - market_vol_t(T)          [vol points]

Volatility points rather than percentage price errors: they are comparable
across maturities and across time, and they are the units a trader and a
bid-offer are quoted in.  Positive spread => the swaptions market says the cap
should be dearer than it trades => the cap is *cheap*.

Entry.  Standardise the spread against its own history -- strictly causal,
using only weeks up to and including t -- and trade the extremes:

    z > +enter   buy the cap,  sell the swaption hedge
    z < -enter   sell the cap, buy the swaption hedge
    |z| < exit   flat

Hedging.  Two risks must go before the wedge is visible.
  * Volatility level: the position is made neutral to a proportional scaling
    of the whole covariance matrix, so a general rise or fall in implied vol
    nets out.  This is the sense in which caps are traded *against* swaptions.
  * Rates: the residual is delta-hedged with a par swap, matched on DV01.

What is left is the relative pricing of the two markets -- the wedge.

Costs.  Every leg crosses the bid-offer, which Bloomberg puts at about one
Black volatility point for these swaptions (footnote 23) -- roughly six percent
of an at-the-money-forward premium at a 16 percent vol.  The default charges
one vol point per leg per trade, applied on entry and exit, and nothing while a
position is merely held.

Honest accounting
-----------------
Positions are marked week to week at quoted market volatilities using the
paper's own identifying assumption (footnote 29): the position is re-marked at
the next week's quote for the same structure.  Maturities are held constant
across the one-week holding period, a standard constant-maturity
approximation for a weekly relative-value book.

Run on synthetic data this measures whether the machinery finds a dislocation
that was deliberately put there.  It is not evidence about the real market.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .conventions import DELTA, CAP_MATURITIES
from .curve import Curve
from .instruments import (
    black_cap_price, black_cap_vega, black_swaption_atmf,
    black_swaption_vega_atmf,
)
from .caps import model_cap_vols
from .empirics import SampleFit


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StrategyConfig:
    """
    Trading rules and cost assumptions.

    Parameters
    ----------
    entry_z, exit_z : float
        Enter when the standardised spread exceeds `entry_z`; close when it
        falls back inside `exit_z`.  A band, so the book is not churned by
        noise around the threshold.
    min_history : int
        Weeks of spread history required before the z-score is trusted and any
        trading is allowed.  The standardisation is causal, so early weeks are
        necessarily out of sample.
    lookback : int or None
        Rolling window for the z-score; None uses an expanding window.
    vol_point_cost : float
        Bid-offer charged per leg per trade, in Black volatility points.
    target_vega : float
        Position size, expressed as the cap-leg vega per 1.00 of volatility
        (i.e. per 100 vol points) per unit notional.  P&L scales linearly in
        this, so it only sets the units.
    delta_hedge : bool
        Neutralise residual rate risk with a par swap.
    max_hedge_instruments : int
        How many swaptions the hedge may use.  Every leg pays the bid-offer on
        entry and exit, so a wide hedge basket is self-defeating.
    maturities : tuple
        Which cap maturities to trade.
    """

    entry_z: float = 1.25
    exit_z: float = 0.0
    min_history: int = 26
    lookback: int | None = 52
    vol_point_cost: float = 1.0
    target_vega: float = 1.0
    delta_hedge: bool = True
    max_hedge_instruments: int = 3
    maturities: tuple[float, ...] = CAP_MATURITIES


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------
def wedge_spread(fit: SampleFit) -> pd.DataFrame:
    """
    The raw signal: model minus market flat cap volatility, in vol points.

    Positive => the cap is cheap relative to what the swaptions market implies.
    """
    data = fit.data
    spread = np.empty((fit.n_weeks, len(data.cap_maturities)))
    for week in range(fit.n_weeks):
        model = model_cap_vols(
            fit.curves[week], fit.results[week].sigma, data.cap_maturities
        )
        spread[week] = 100.0 * (model - data.cap_vols[week])

    return pd.DataFrame(
        spread,
        index=data.dates,
        columns=[f"{T:g}y" for T in data.cap_maturities],
    )


def causal_zscore(
    spread: pd.DataFrame, lookback: int | None = 52, min_history: int = 26
) -> pd.DataFrame:
    """
    Standardise each spread series using only information available at the
    time.

    The mean and standard deviation at week t are computed from weeks up to and
    including t.  Using the full-sample mean here would leak the future into
    the signal and is the single easiest way to manufacture a fake backtest.
    """
    if lookback is None:
        mean = spread.expanding(min_periods=min_history).mean()
        std = spread.expanding(min_periods=min_history).std(ddof=1)
    else:
        mean = spread.rolling(lookback, min_periods=min_history).mean()
        std = spread.rolling(lookback, min_periods=min_history).std(ddof=1)

    z = (spread - mean) / std.replace(0.0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan)


def target_positions(z: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """
    Convert z-scores into target positions in {-1, 0, +1}, with an entry/exit
    band and state carried forward while the signal sits inside the band.

    +1 means long the cap and short the swaption hedge.
    """
    positions = pd.DataFrame(
        0.0, index=z.index, columns=z.columns, dtype=float
    )
    for column in z.columns:
        state = 0.0
        values = z[column].to_numpy()
        out = np.zeros(len(values))
        for t, value in enumerate(values):
            if not np.isfinite(value):
                state = 0.0
            elif value >= config.entry_z:
                state = 1.0
            elif value <= -config.entry_z:
                state = -1.0
            elif abs(value) <= config.exit_z:
                state = 0.0
            out[t] = state
        positions[column] = out
    return positions


# ---------------------------------------------------------------------------
# Hedge construction
# ---------------------------------------------------------------------------
def swaption_hedge_basket(
    maturity: float,
    grid: list[tuple[float, float]],
    max_instruments: int = 3,
) -> list[int]:
    """
    Indices of the swaptions used to hedge a `maturity`-year cap.

    A cap spans forwards from six months out to its maturity, so the hedge is
    drawn from swaptions whose underlying swap ends at or before that point.
    If none qualify (a two-year cap against a grid starting at 1x1), the
    shortest available structures are used instead.

    Only a handful of structures are used, not every eligible one.  Each leg
    crosses the bid-offer on the way in and on the way out, so hedging a
    ten-year cap with all thirty-four swaptions would cost more than the wedge
    is worth -- and no desk would trade it that way.  Preference goes to
    structures whose underlying swap ends nearest the cap's maturity (so they
    cover the same forwards) and, among those, to the shortest expiries, which
    are the most liquid.
    """
    eligible = [k for k, (e, t) in enumerate(grid) if e + t <= maturity + 1e-9]
    if not eligible:
        shortest = min(e + t for e, t in grid)
        eligible = [k for k, (e, t) in enumerate(grid) if e + t <= shortest + 1e-9]

    eligible.sort(key=lambda k: (abs(grid[k][0] + grid[k][1] - maturity), grid[k][0]))
    return sorted(eligible[:max_instruments])


def hedge_weights(
    curve: Curve,
    maturity: float,
    cap_vol: float,
    grid: list[tuple[float, float]],
    swaption_vols: np.ndarray,
    basket: list[int],
) -> np.ndarray:
    """
    Weights on the swaption basket that make the package neutral to a parallel
    move in implied volatility.

    The cap's vega is matched by the basket's total vega, spread across the
    basket in proportion to each member's own vega -- the minimum-concentration
    solution to the single vega constraint, which avoids loading the whole
    hedge onto one illiquid structure.

    Returns an array over `basket`, in units of swaption notional per unit of
    cap notional.
    """
    cap_v = black_cap_vega(curve, maturity, cap_vol)

    vegas = np.array(
        [
            black_swaption_vega_atmf(curve, grid[k][0], grid[k][1], swaption_vols[k])
            for k in basket
        ]
    )
    total = float(vegas @ vegas)
    if total <= 0:
        return np.zeros(len(basket))
    # w proportional to vega, scaled so that w . vega == cap_vega
    return vegas * (cap_v / total)


def position_dv01(
    curve: Curve,
    maturity: float,
    cap_vol: float,
    grid: list[tuple[float, float]],
    swaption_vols: np.ndarray,
    basket: list[int],
    weights: np.ndarray,
    bump: float = 1e-4,
) -> float:
    """
    Rate sensitivity of the cap-minus-swaptions package, per basis point of
    parallel shift in the forward curve, holding volatilities fixed.
    """
    from .curve import curve_from_forwards

    def package(c: Curve) -> float:
        value = black_cap_price(c, maturity, cap_vol, strike=None)
        for w, k in zip(weights, basket):
            value -= w * black_swaption_atmf(
                c, grid[k][0], grid[k][1], swaption_vols[k]
            )
        return value

    up = curve_from_forwards(curve.forwards + bump, curve.forwards[0] + bump)
    down = curve_from_forwards(curve.forwards - bump, curve.forwards[0] - bump)
    return (package(up) - package(down)) / 2.0


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
@dataclass
class BacktestResult:
    """
    Outcome of a wedge backtest.

    Attributes
    ----------
    pnl : DataFrame
        Weekly P&L per cap maturity, net of costs, in units of cap notional.
    gross_pnl, costs : DataFrame
        The same before costs, and the costs themselves.
    positions : DataFrame
        Target position each week, in {-1, 0, +1}.
    spread, zscore : DataFrame
        The signal and its causal standardisation.
    """

    pnl: pd.DataFrame
    gross_pnl: pd.DataFrame
    costs: pd.DataFrame
    positions: pd.DataFrame
    spread: pd.DataFrame
    zscore: pd.DataFrame
    config: StrategyConfig

    @property
    def total(self) -> pd.Series:
        """Equally weighted P&L across the traded maturities."""
        return self.pnl.mean(axis=1)

    def equity_curve(self) -> pd.Series:
        """Cumulative P&L of the equally weighted book."""
        return self.total.cumsum()


@dataclass
class UnitEconomics:
    """
    The week-by-week economics of a *unit* position, independent of the
    trading rules.

    `unit_pnl[t, T]` is the P&L from holding a +1 package (long the cap, short
    the vega-matched swaption hedge, delta hedged) from week t to week t+1.
    `unit_cost[t, T]` is the cash cost of one unit of turnover at week t, per
    one Black volatility point of bid-offer.

    None of this depends on the entry threshold, the exit band, or the assumed
    bid-offer -- only on the calibrated fit and the hedge construction.  So it
    is computed once and reused across every configuration, which is what makes
    :func:`sensitivity` cheap: the repricing is the expensive part, and a
    fifteen-cell grid would otherwise repeat it fifteen times.
    """

    unit_pnl: pd.DataFrame
    unit_cost: pd.DataFrame
    spread: pd.DataFrame
    columns: list[str]


def unit_economics(
    fit: SampleFit, config: StrategyConfig | None = None
) -> UnitEconomics:
    """
    Reprice the cap-versus-swaptions package week by week.

    See :class:`UnitEconomics`.  Only `target_vega`, `delta_hedge` and
    `max_hedge_instruments` of the config are used.
    """
    config = config or StrategyConfig()
    data = fit.data
    grid = data.swaption_grid

    tradable = [T for T in config.maturities if T in data.cap_maturities]
    if not tradable:
        raise ValueError("none of the configured maturities are in the data")
    columns = [f"{T:g}y" for T in tradable]
    cap_index = {T: list(data.cap_maturities).index(T) for T in tradable}

    spread = wedge_spread(fit)[columns]

    n = fit.n_weeks
    unit_pnl = pd.DataFrame(0.0, index=data.dates, columns=columns)
    unit_cost = pd.DataFrame(0.0, index=data.dates, columns=columns)

    for week in range(n - 1):
        curve_now = fit.curves[week]
        curve_next = fit.curves[week + 1]
        vols_now = data.swaption_vols[week]
        vols_next = data.swaption_vols[week + 1]
        shift = float(np.mean(curve_next.forwards - curve_now.forwards) / 1e-4)

        for position, (column, maturity) in enumerate(zip(columns, tradable)):
            k_cap = cap_index[maturity]
            cap_vol_now = data.cap_vols[week][k_cap]
            cap_vol_next = data.cap_vols[week + 1][k_cap]

            basket = swaption_hedge_basket(
                maturity, grid, config.max_hedge_instruments
            )
            weights = hedge_weights(
                curve_now, maturity, cap_vol_now, grid, vols_now, basket
            )

            # Scale so the cap leg carries the configured vega.
            cap_v = black_cap_vega(curve_now, maturity, cap_vol_now)
            scale = config.target_vega / cap_v if cap_v > 0 else 0.0

            unit_cost.iloc[week, position] = scale * _round_trip_cost(
                curve_now, maturity, cap_vol_now, grid, vols_now,
                basket, weights, 1.0,
            )

            # --- mark to market one week forward ---------------------------
            strike = curve_now.par_swap_rate(maturity)
            change = (
                black_cap_price(curve_next, maturity, cap_vol_next, strike)
                - black_cap_price(curve_now, maturity, cap_vol_now, strike)
            )

            for w, k in zip(weights, basket):
                expiry, tenor = grid[k]
                change -= w * (
                    black_swaption_atmf(curve_next, expiry, tenor, vols_next[k])
                    - black_swaption_atmf(curve_now, expiry, tenor, vols_now[k])
                )

            if config.delta_hedge:
                change -= position_dv01(
                    curve_now, maturity, cap_vol_now, grid, vols_now,
                    basket, weights,
                ) * shift

            unit_pnl.iloc[week, position] = change * scale

    return UnitEconomics(
        unit_pnl=unit_pnl, unit_cost=unit_cost, spread=spread, columns=columns
    )


def backtest(
    fit: SampleFit,
    config: StrategyConfig | None = None,
    economics: UnitEconomics | None = None,
) -> BacktestResult:
    """
    Run the wedge strategy over a calibrated sample.

    The week-t position is formed from the week-t signal and earns the return
    from week t to week t+1, so no observation is used before it exists.

    Pass a precomputed :class:`UnitEconomics` to avoid repricing when sweeping
    configurations; it is rebuilt from `fit` when omitted.
    """
    config = config or StrategyConfig()
    economics = economics or unit_economics(fit, config)

    spread = economics.spread
    z = causal_zscore(spread, config.lookback, config.min_history)
    positions = target_positions(z, config)

    turnover = positions.diff().abs().fillna(positions.abs())

    gross = positions * economics.unit_pnl
    costs = turnover * economics.unit_cost * config.vol_point_cost

    return BacktestResult(
        pnl=gross - costs,
        gross_pnl=gross,
        costs=costs,
        positions=positions,
        spread=spread,
        zscore=z,
        config=config,
    )


def _round_trip_cost(
    curve: Curve,
    maturity: float,
    cap_vol: float,
    grid: list[tuple[float, float]],
    swaption_vols: np.ndarray,
    basket: list[int],
    weights: np.ndarray,
    vol_points: float,
) -> float:
    """
    Cash cost of crossing the bid-offer on every leg once, in price units per
    unit of cap notional.

    A `vol_points` move in Black volatility costs vega * vol_points / 100 on
    each leg, since vega is quoted per 1.00 of volatility.
    """
    half_spread = vol_points / 100.0
    cost = black_cap_vega(curve, maturity, cap_vol) * half_spread
    for w, k in zip(weights, basket):
        cost += abs(w) * black_swaption_vega_atmf(
            curve, grid[k][0], grid[k][1], swaption_vols[k]
        ) * half_spread
    return float(cost)


# ---------------------------------------------------------------------------
# Performance statistics
# ---------------------------------------------------------------------------
def summarize(result: BacktestResult, periods_per_year: float = 52.0) -> pd.DataFrame:
    """
    Per-maturity and aggregate performance statistics, gross and net.

    Both are reported because they answer different questions.  Gross P&L says
    whether the *signal* has any economic content -- whether the wedge is real
    and converges.  Net P&L says whether it survives the bid-offer, which for
    these instruments is wide enough to decide the question on its own.

    `breakeven_vol_points` is the bid-offer, in Black volatility points per leg
    per trade, at which net P&L reaches zero.  Compare it with the roughly one
    volatility point Bloomberg quotes (footnote 23): a break-even far below one
    means the strategy cannot be traded, however good the signal looks.

    The t-statistic is on the mean weekly P&L.  With around 128 weekly
    observations a Sharpe ratio needs a t-statistic near 2 before it means much.
    """
    rows = []
    columns = list(result.pnl.columns)
    series = {column: column for column in columns}

    for name in list(series) + ["Total"]:
        if name == "Total":
            net = result.total.to_numpy()
            gross = result.gross_pnl.mean(axis=1).to_numpy()
            costs = result.costs.mean(axis=1).to_numpy()
            active = (result.positions != 0.0).any(axis=1).to_numpy()
            turnover = result.positions.diff().abs().sum(axis=1).to_numpy()
        else:
            net = result.pnl[name].to_numpy()
            gross = result.gross_pnl[name].to_numpy()
            costs = result.costs[name].to_numpy()
            active = result.positions[name].to_numpy() != 0.0
            turnover = result.positions[name].diff().abs().to_numpy()

        mean = float(np.mean(net))
        std = float(np.std(net, ddof=1))
        gross_mean = float(np.mean(gross))
        gross_std = float(np.std(gross, ddof=1))

        traded = net[active]
        equity = np.cumsum(net)
        drawdown = equity - np.maximum.accumulate(equity)

        total_cost = float(np.sum(costs))
        total_gross = float(np.sum(gross))
        breakeven = (
            result.config.vol_point_cost * total_gross / total_cost
            if total_cost > 0
            else np.inf
        )

        rows.append(
            {
                "book": name,
                "gross_pnl": total_gross,
                "costs": total_cost,
                "net_pnl": float(np.sum(net)),
                "gross_sharpe": gross_mean / gross_std * np.sqrt(periods_per_year)
                if gross_std > 0 else np.nan,
                "net_sharpe": mean / std * np.sqrt(periods_per_year)
                if std > 0 else np.nan,
                "gross_t_stat": gross_mean / (gross_std / np.sqrt(len(gross)))
                if gross_std > 0 else np.nan,
                "net_t_stat": mean / (std / np.sqrt(len(net)))
                if std > 0 else np.nan,
                "hit_rate": float(np.mean(traded > 0)) if traded.size else np.nan,
                "weeks_in_position": int(np.sum(active)),
                "round_trips": float(np.nansum(turnover)) / 2.0,
                "max_drawdown": float(np.min(drawdown)) if len(drawdown) else np.nan,
                "breakeven_vol_points": breakeven,
            }
        )

    return pd.DataFrame(rows).set_index("book")


def sensitivity(
    fit: SampleFit,
    entry_grid: tuple[float, ...] = (0.75, 1.0, 1.25, 1.5, 2.0),
    cost_grid: tuple[float, ...] = (0.0, 0.5, 1.0),
    exit_z: float = 0.0,
    base: StrategyConfig | None = None,
) -> pd.DataFrame:
    """
    Net P&L across entry thresholds and bid-offer assumptions.

    A single backtest number is close to meaningless here, because the entry
    threshold trades off two things that move in opposite directions: a lower
    threshold gives more trades and more statistical power but pays the
    bid-offer far more often, while a higher one is cheap but may leave only a
    handful of round trips, on which no t-statistic means anything.

    Reporting the grid makes that trade-off visible instead of hiding it behind
    a tuned parameter, and shows whether any profit is robust to the choice or
    an artifact of one cell.  The `round_trips` column is the one to read
    first: a cell with two round trips is an anecdote, not a backtest.
    """
    base = base or StrategyConfig()
    # Reprice once; every cell below is then just arithmetic on positions.
    economics = unit_economics(fit, base)
    rows = []
    for entry in entry_grid:
        for cost in cost_grid:
            config = StrategyConfig(
                entry_z=entry,
                exit_z=exit_z,
                min_history=base.min_history,
                lookback=base.lookback,
                vol_point_cost=cost,
                target_vega=base.target_vega,
                delta_hedge=base.delta_hedge,
                max_hedge_instruments=base.max_hedge_instruments,
                maturities=base.maturities,
            )
            result = backtest(fit, config, economics)
            stats = summarize(result).loc["Total"]
            rows.append(
                {
                    "entry_z": entry,
                    "cost_vol_points": cost,
                    "gross_pnl": stats["gross_pnl"],
                    "net_pnl": stats["net_pnl"],
                    "net_sharpe": stats["net_sharpe"],
                    "net_t_stat": stats["net_t_stat"],
                    "round_trips": stats["round_trips"],
                    "weeks_in_position": stats["weeks_in_position"],
                }
            )
    return pd.DataFrame(rows)


def signal_diagnostics(
    result: BacktestResult, truth: np.ndarray | None = None
) -> pd.DataFrame:
    """
    How well the signal behaves, independent of the P&L.

    `truth` is the injected wedge from the synthetic generator, in vol points.
    When supplied, the correlation between the recovered spread and the true
    dislocation shows directly whether the calibration is finding what was put
    there -- a much sharper test of the pipeline than the P&L, which also
    depends on the trading rules.
    """
    rows = []
    for k, column in enumerate(result.spread.columns):
        spread = result.spread[column]
        row = {
            "maturity": column,
            "spread_mean": float(spread.mean()),
            "spread_std": float(spread.std(ddof=1)),
            "spread_serial_corr": float(spread.autocorr(1)),
            "pct_weeks_in_position": float(
                np.mean(result.positions[column].to_numpy() != 0.0)
            ),
        }
        if truth is not None:
            true_series = pd.Series(truth[:, k], index=spread.index)
            row["corr_with_true_wedge"] = float(spread.corr(true_series))
        rows.append(row)
    return pd.DataFrame(rows).set_index("maturity")
