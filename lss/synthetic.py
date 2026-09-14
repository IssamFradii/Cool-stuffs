"""
Paper-calibrated synthetic market data.

WHAT THIS IS AND IS NOT
-----------------------
This module fabricates a data set whose *summary statistics* match Tables 1,
3 and 4 of the paper: the level, dispersion and persistence of the forward
curve, the swaption volatility surface, and the cap volatility curve, together
with the volatility spike of the autumn 1998 hedge-fund crisis.

It is scaffolding, not evidence.  It exists so the whole pipeline -- curve
build, calibration, cap valuation, the empirical tables and the trading
strategy -- can be run, inspected and tested end to end before any real
Bloomberg data is available.  Backtest results computed on it measure whether
the machinery works, NOT whether the strategy makes money.  Nothing here is a
forecast, and no economic conclusion should be drawn from it.

Swap in real data through :mod:`lss.data` and every downstream number becomes
meaningful; nothing else in the package changes.

Design
------
* Forwards: a mean-reverting (AR(1) in logs) process on the 19 six-month
  forwards, with innovations correlated by the paper's Table 2 matrix, and
  per-forward means, standard deviations and serial correlations taken from
  Table 1.

* Swaption volatilities: a persistent common volatility level, plus the crisis
  spike, plus structure-specific noise, scaled so each of the 34 structures
  reproduces its Table 3 mean and standard deviation.

* Cap volatilities: driven by the same common volatility level -- so caps and
  swaptions are usually consistent -- plus a deliberately injected, persistent,
  mean-reverting *wedge* component.  That wedge is what Table 9 detects and
  what the strategy trades.  Its persistence is set to the ~0.67 first-order
  serial correlation the paper reports for cap pricing errors.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .conventions import SWAPTION_GRID, CAP_MATURITIES, N_FORWARDS
from . import paper_data
from .curve import Curve, curve_from_forwards


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------
@dataclass
class MarketData:
    """
    One sample of market data, real or synthetic.

    Attributes
    ----------
    dates : DatetimeIndex, length n_weeks
    forwards : (n_weeks, 19) array
        Six-month Libor forwards, decimals.
    swaption_vols : (n_weeks, 34) array
        ATMF Black volatilities, decimals, ordered as `swaption_grid`.
    cap_vols : (n_weeks, 6) array
        Flat Black cap volatilities, decimals, ordered as `cap_maturities`.
    libor_6m : (n_weeks,) array
        Spot six-month Libor, used to anchor each curve.
    """

    dates: pd.DatetimeIndex
    forwards: np.ndarray
    swaption_vols: np.ndarray
    cap_vols: np.ndarray
    libor_6m: np.ndarray
    swaption_grid: list[tuple[float, float]]
    cap_maturities: tuple[float, ...]
    is_synthetic: bool = True

    def __post_init__(self) -> None:
        n = len(self.dates)
        if self.forwards.shape != (n, N_FORWARDS):
            raise ValueError(f"forwards must be ({n}, {N_FORWARDS})")
        if self.swaption_vols.shape != (n, len(self.swaption_grid)):
            raise ValueError(f"swaption_vols must be ({n}, {len(self.swaption_grid)})")
        if self.cap_vols.shape != (n, len(self.cap_maturities)):
            raise ValueError(f"cap_vols must be ({n}, {len(self.cap_maturities)})")

    @property
    def n_weeks(self) -> int:
        return len(self.dates)

    def curve(self, week: int) -> Curve:
        """Bootstrapped term structure for one week."""
        return curve_from_forwards(self.forwards[week], self.libor_6m[week])

    def curves(self) -> list[Curve]:
        """All weekly curves."""
        return [self.curve(i) for i in range(self.n_weeks)]

    def crisis_mask(self) -> np.ndarray:
        """Boolean mask for the ten-week LTCM window (Section 5.3)."""
        return np.asarray(
            (self.dates >= pd.Timestamp(paper_data.CRISIS_START))
            & (self.dates <= pd.Timestamp(paper_data.CRISIS_END))
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ar1_path(
    n: int,
    persistence: float,
    innovations: np.ndarray,
    target_std: float,
) -> np.ndarray:
    """
    A zero-mean AR(1) path with the requested unconditional standard deviation.

    x_t = rho x_{t-1} + eps_t, with Var(eps) chosen so Var(x) = target_std^2.
    The path is started from its stationary distribution so there is no burn-in
    transient.
    """
    rho = float(np.clip(persistence, -0.995, 0.995))
    scale = target_std * np.sqrt(1.0 - rho * rho)

    x = np.empty(n)
    x[0] = innovations[0] * target_std
    for t in range(1, n):
        x[t] = rho * x[t - 1] + scale * innovations[t]
    return x


def _crisis_bump(dates: pd.DatetimeIndex, height: float, decay_weeks: float = 5.0) -> np.ndarray:
    """
    The autumn 1998 volatility spike.

    Rises sharply at the LTCM announcement week (11 September 1998) and decays
    exponentially, matching the shape in Figures 2, 4 and 7: implied vols and
    pricing errors jump for roughly ten weeks and then subside.
    """
    start = pd.Timestamp(paper_data.CRISIS_START)
    weeks_since = (dates - start).days / 7.0
    bump = np.where(
        weeks_since >= 0, height * np.exp(-weeks_since / decay_weeks), 0.0
    )
    # A short ramp in the fortnight before the announcement.
    ramp = np.where(
        (weeks_since < 0) & (weeks_since > -2),
        height * 0.3 * (1.0 + weeks_since / 2.0),
        0.0,
    )
    return np.asarray(bump + ramp, dtype=float)


def weekly_dates(start: str, end: str) -> pd.DatetimeIndex:
    """Friday-closing weekly observation dates, inclusive."""
    return pd.date_range(start=start, end=end, freq="W-FRI")


# ---------------------------------------------------------------------------
# Forward curve simulation
# ---------------------------------------------------------------------------
def generate_forwards(
    n_weeks: int,
    rng: np.random.Generator,
    correlation: np.ndarray | None = None,
    means: np.ndarray | None = None,
    stds: np.ndarray | None = None,
    persistences: np.ndarray | None = None,
) -> np.ndarray:
    """
    Simulate the 19 six-month Libor forwards.

    Log-forwards follow correlated AR(1) processes around the Table 1 means,
    with the Table 1 standard deviations and serial correlations, and
    cross-sectional innovation correlation given by the Table 2 matrix.
    """
    correlation = (
        paper_data.historical_correlation() if correlation is None else correlation
    )
    means = paper_data.FORWARD_MEAN if means is None else means
    stds = paper_data.FORWARD_STD if stds is None else stds
    persistences = (
        paper_data.TABLE1_FORWARD_STATS[:, 5] if persistences is None else persistences
    )

    # Correlated standard normal innovations.
    vals, vecs = np.linalg.eigh(0.5 * (correlation + correlation.T))
    factor = vecs * np.sqrt(np.clip(vals, 0.0, None))
    shocks = rng.standard_normal((n_weeks, N_FORWARDS)) @ factor.T

    # Work in logs so rates stay positive, matching the log-change convention
    # used for the historical correlation matrix.
    log_mean = np.log(means)
    log_std = stds / means            # first-order: sd(log x) ~ sd(x)/E[x]

    out = np.empty((n_weeks, N_FORWARDS))
    for i in range(N_FORWARDS):
        path = _ar1_path(n_weeks, persistences[i], shocks[:, i], log_std[i])
        out[:, i] = np.exp(log_mean[i] + path)
    return out


# ---------------------------------------------------------------------------
# Volatility surfaces
# ---------------------------------------------------------------------------
def generate_swaption_vols(
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
    common_level: np.ndarray,
    grid: list[tuple[float, float]] | None = None,
    idiosyncratic_share: float = 0.03,
    crisis_distortion_points: float = 2.3,
) -> np.ndarray:
    """
    Build the 34-structure ATMF swaption volatility surface.

    Each structure is its Table 3 mean, plus a loading on the common
    (standardised) volatility level, plus structure-specific noise.  The
    loading and noise are scaled so the structure reproduces its Table 3
    standard deviation, with `idiosyncratic_share` of the variance idiosyncratic
    -- which is what leaves the surface with a shape the four-factor model has
    to work to fit.

    Crisis distortion
    -----------------
    A rise in the *level* of volatility does not by itself strain the model:
    the four eigenvalues simply scale up and the fit stays good.  What the
    paper documents in autumn 1998 is different -- the surface takes on a
    *shape* no time-homogeneous four-factor model can reproduce, and the RMSE
    jumps from two or three percent to as much as sixteen (Figure 7):

      "either the time-homogeneous specification of the string market model is
       too restrictive, or quoted prices in the swaptions market were
       inconsistent with the absence of arbitrage"

    with traders reporting thin liquidity and questionable quotes.  That is
    modelled here as a fixed but arbitrary per-structure pattern, scaled by the
    crisis bump: a dislocation in the shape of the surface rather than its
    level.  Set `crisis_distortion_points` to zero to switch it off.
    """
    grid = list(grid) if grid is not None else list(SWAPTION_GRID)
    n = len(dates)
    stats = paper_data.TABLE3_SWAPTION_VOLS

    out = np.empty((n, len(grid)))
    common_loading = np.sqrt(1.0 - idiosyncratic_share)
    idio_loading = np.sqrt(idiosyncratic_share)

    # A fixed, mean-zero pattern across the surface; the same shape every week,
    # switched on only during the crisis.
    pattern = rng.standard_normal(len(grid))
    pattern -= pattern.mean()
    pattern /= pattern.std()
    bump = _crisis_bump(dates, crisis_distortion_points)

    for k, key in enumerate(grid):
        mean_pct, std_pct, _, _, serial = stats[key]
        idio = _ar1_path(n, serial, rng.standard_normal(n), 1.0)
        combined = common_loading * common_level + idio_loading * idio
        out[:, k] = (mean_pct + std_pct * combined + bump * pattern[k]) / 100.0

    return np.clip(out, 0.02, None)


def generate_cap_vols(
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
    common_level: np.ndarray,
    wedge_linear: float = 0.35,
    wedge_convex: float = 0.55,
    wedge_persistence: float = 0.67,
    crisis_wedge_points: float = 1.1,
    maturities: tuple[float, ...] = CAP_MATURITIES,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the cap volatility curve, and the wedge that was injected into it.

    Caps load on the same common volatility level as swaptions, so the two
    markets are usually consistent.  On top of that sits a persistent,
    mean-reverting *wedge*: a dislocation between the caps market and the
    swaptions market, of the kind the paper documents in Table 9.

    Sign convention
    ---------------
    The returned `wedge` is *cheapness in volatility points*, the same
    convention as :func:`lss.strategy.wedge_spread`: positive means the cap
    trades below what the swaptions market implies, so it is subtracted from
    the quoted cap volatility.  A backtest can therefore be scored against it
    directly, and the correlation between the recovered spread and the injected
    wedge should be strongly *positive*.

    Shape
    -----
    The dislocation is deliberately right-skewed, not symmetric, because that
    is what Table 9 shows: median errors within three percent of zero at every
    maturity but means of 6 to 23 percent, "primarily due to periodic large
    positive errors, resulting in a skewed, somewhat bimodal distribution of
    errors".  A symmetric wedge would also drive market cap prices below the
    one-factor Merton bound far more often than the 0.18% of observations the
    paper reports in Table 10.

    So the wedge is a linear term plus a one-sided convex term in a persistent
    latent factor: usually near zero, occasionally large, and almost always in
    the "cap is cheap" direction.  Persistence defaults to 0.67, the
    first-order serial correlation the paper reports for cap pricing errors.

    Returns
    -------
    cap_vols : (n_weeks, 6) array
    wedge : (n_weeks, 6) array
        The injected cheapness, in volatility points (not decimals).
    """
    n = len(dates)
    stats = paper_data.TABLE4_CAP_VOLS

    # A shared wedge factor plus a smaller maturity-specific one: dislocations
    # hit the whole cap curve but not uniformly.
    shared = _ar1_path(n, wedge_persistence, rng.standard_normal(n), 1.0)
    crisis = _crisis_bump(dates, crisis_wedge_points)

    cap_vols = np.empty((n, len(maturities)))
    wedge = np.empty((n, len(maturities)))

    for k, T in enumerate(maturities):
        mean_pct, std_pct, _, _, _, _ = stats[T]

        specific = _ar1_path(n, wedge_persistence, rng.standard_normal(n), 1.0)
        latent = 0.75 * shared + 0.66 * specific

        # Short caps dislocate more than long ones: in Table 9 the two-year
        # mean error is more than twice the overall mean.
        severity = float(np.sqrt(10.0 / T))

        cheapness = severity * (
            wedge_linear * latent
            + wedge_convex * np.maximum(latent, 0.0) ** 2
            + crisis
        )

        wedge[:, k] = cheapness
        cap_vols[:, k] = (mean_pct + std_pct * common_level - cheapness) / 100.0

    return np.clip(cap_vols, 0.02, None), wedge


# ---------------------------------------------------------------------------
# Top-level generators
# ---------------------------------------------------------------------------
def generate_market_data(
    start: str = paper_data.SAMPLE_START,
    end: str = paper_data.SAMPLE_END,
    seed: int = 20250913,
    vol_persistence: float = 0.93,
    crisis_vol_sd: float = 2.6,
    with_wedge: bool = True,
) -> tuple[MarketData, np.ndarray]:
    """
    Generate a full in-sample data set: curves, swaption vols and cap vols.

    Parameters
    ----------
    vol_persistence : float
        Serial correlation of the common volatility level.  Table 3 reports
        0.89-0.95 for the individual structures.
    crisis_vol_sd : float
        Height of the autumn 1998 spike, in standard deviations of the common
        volatility level.  Table 3 shows the 0.5-into-1 vol reaching 27% against
        a 14.6% mean and 3.4% standard deviation, roughly 3.6 sd.
    with_wedge : bool
        If False, caps are generated with no dislocation at all.  Useful as a
        null: the strategy should find nothing to trade.

    Returns
    -------
    data : MarketData
    wedge : (n_weeks, 6) array
        The injected cap/swaption dislocation in volatility points -- the
        "truth" a backtest can be scored against.  Zero when `with_wedge` is
        False.
    """
    rng = np.random.default_rng(seed)
    dates = weekly_dates(start, end)
    n = len(dates)

    # The common volatility level: persistent, mean zero, unit variance, with
    # the crisis spike superimposed.
    common = _ar1_path(n, vol_persistence, rng.standard_normal(n), 1.0)
    common = common + _crisis_bump(dates, crisis_vol_sd)

    forwards = generate_forwards(n, rng)
    libor_6m = forwards[:, 0] * (1.0 + 0.01 * rng.standard_normal(n))

    swaption_vols = generate_swaption_vols(dates, rng, common)

    if with_wedge:
        cap_vols, wedge = generate_cap_vols(dates, rng, common)
    else:
        cap_vols, wedge = generate_cap_vols(
            dates, rng, common, wedge_linear=0.0, wedge_convex=0.0,
            crisis_wedge_points=0.0,
        )

    data = MarketData(
        dates=dates,
        forwards=forwards,
        swaption_vols=swaption_vols,
        cap_vols=cap_vols,
        libor_6m=libor_6m,
        swaption_grid=list(SWAPTION_GRID),
        cap_maturities=tuple(CAP_MATURITIES),
        is_synthetic=True,
    )
    return data, wedge


def generate_exante_forwards(
    start: str = paper_data.EXANTE_START,
    end: str = paper_data.EXANTE_END,
    seed: int = 19920117,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """
    Generate the five-year ex-ante forward history from which the historical
    correlation matrix H would be estimated (262 weekly observations).

    In practice the pipeline defaults to the paper's own Table 2 matrix, which
    is a genuine model input; this is here so the estimation step itself can be
    exercised and tested.
    """
    rng = np.random.default_rng(seed)
    dates = weekly_dates(start, end)
    return dates, generate_forwards(len(dates), rng)
