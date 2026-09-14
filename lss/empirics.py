"""
The empirical tests of Section 5 -- Tables 5 through 10.

  Table 5   likelihood ratio tests for the number of implied factors
  Table 6   average implied correlations minus historical correlations
  Table 7   standard deviations of the implied correlations
  Table 8   percentage swaption valuation errors, by structure
  Table 9   percentage cap valuation errors, four-factor model
  Table 10  percentage cap valuation errors, one-factor Merton bound
  Figure 6  time series of the implied eigenvalues
  Figure 7  time series of the swaption RMSE

All t-statistics are adjusted for first-order serial correlation, as the paper
states for Tables 8, 9 and 10.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .conventions import SWAPTION_GRID, CAP_MATURITIES
from .curve import Curve
from .calibrate import CalibrationResult, calibrate_series, swaption_prices_from_vols
from .caps import value_caps, check_merton_bound
from .historical import FactorBasis, paper_factor_basis, to_correlation
from .mathutil import serial_correlation, ar1_adjusted_tstat, chi2_sf
from . import paper_data, smm
from .synthetic import MarketData


# ---------------------------------------------------------------------------
# Running the calibration over a sample
# ---------------------------------------------------------------------------
@dataclass
class SampleFit:
    """
    Everything produced by fitting the model to a whole sample.

    Attributes
    ----------
    results : list of CalibrationResult, one per week
    curves : list of Curve, one per week
    """

    data: MarketData
    curves: list[Curve]
    results: list[CalibrationResult]
    n_factors: int

    @property
    def n_weeks(self) -> int:
        return len(self.results)

    @property
    def rmse(self) -> np.ndarray:
        """Weekly RMSE of percentage swaption pricing errors -- Figure 7."""
        return np.array([100.0 * r.rmse for r in self.results])

    @property
    def eigenvalues(self) -> np.ndarray:
        """(n_weeks, n_factors) implied eigenvalues -- Figure 6."""
        return np.array([r.psi for r in self.results])

    @property
    def swaption_errors(self) -> np.ndarray:
        """(n_weeks, 34) percentage swaption pricing errors."""
        return np.array([r.percentage_errors for r in self.results])

    @property
    def sum_squared_errors(self) -> float:
        """Total SSE across all weeks and swaptions -- the Table 5 input."""
        return float(sum(r.sum_squared_errors for r in self.results))

    def sigma(self, week: int) -> np.ndarray:
        return self.results[week].sigma


def fit_sample(
    data: MarketData,
    n_factors: int = 4,
    basis: FactorBasis | None = None,
    method: str = "approx",
    sim_config: smm.SimulationConfig | None = None,
    progress: bool = False,
) -> SampleFit:
    """
    Calibrate the string market model week by week across the whole sample.
    """
    basis = basis or paper_factor_basis()
    curves = data.curves()

    market_prices = np.array(
        [
            swaption_prices_from_vols(curve, data.swaption_vols[w], data.swaption_grid)
            for w, curve in enumerate(curves)
        ]
    )

    results = calibrate_series(
        curves,
        market_prices,
        n_factors=n_factors,
        grid=data.swaption_grid,
        basis=basis,
        method=method,
        sim_config=sim_config,
        progress=progress,
    )
    return SampleFit(data=data, curves=curves, results=results, n_factors=n_factors)


# ---------------------------------------------------------------------------
# Table 5 -- how many implied factors?
# ---------------------------------------------------------------------------
def table5_factor_tests(
    data: MarketData,
    max_factors: int = 7,
    basis: FactorBasis | None = None,
    method: str = "approx",
    sim_config: smm.SimulationConfig | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """
    Incremental likelihood ratio tests for the number of implied factors.

      "Under the null hypothesis of equality, 128 x 34 = 4,352 times the
       difference between the logarithms of the sum of squared errors for N and
       N+1 factors is asymptotically distributed as a chi-square variate with
       128 degrees of freedom."

    The paper finds two-vs-one, three-vs-two and four-vs-three significant and
    everything beyond insignificant -- hence four factors.
    """
    basis = basis or paper_factor_basis()
    n_obs = data.n_weeks * len(data.swaption_grid)
    dof = data.n_weeks

    sse: dict[int, float] = {}
    for n in range(1, max_factors + 2):
        if progress:
            print(f"  fitting {n}-factor model...", flush=True)
        fit = fit_sample(
            data, n_factors=n, basis=basis, method=method, sim_config=sim_config
        )
        sse[n] = fit.sum_squared_errors

    rows = []
    for n in range(1, max_factors + 1):
        statistic = n_obs * (np.log(sse[n]) - np.log(sse[n + 1]))
        rows.append(
            {
                "N": n,
                "N+1": n + 1,
                "sse_N": sse[n],
                "sse_N+1": sse[n + 1],
                "statistic": statistic,
                "p_value": chi2_sf(statistic, dof),
                "paper_statistic": paper_data.TABLE5_LR_FULL.get((n, n + 1), np.nan),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tables 6 and 7 -- implied versus historical correlations
# ---------------------------------------------------------------------------
def implied_correlations(fit: SampleFit) -> np.ndarray:
    """(n_weeks, 19, 19) stack of weekly implied correlation matrices."""
    return np.array([to_correlation(r.sigma) for r in fit.results])


def table6_correlation_differences(
    fit: SampleFit, historical: np.ndarray | None = None
) -> pd.DataFrame:
    """
    Average implied correlation minus historical correlation.

    The paper finds these differences are almost all negative -- implied
    correlations are lower than historical ones, typically by .05 to .10 and
    occasionally by .20, with the largest gaps between the two-to-three year
    and seven-to-nine year forwards.  The one notable positive difference is
    between the first and second forwards.
    """
    historical = (
        paper_data.historical_correlation() if historical is None else historical
    )
    average = np.nanmean(implied_correlations(fit), axis=0)
    labels = [f"{(i + 1) * 0.5:.2f}" for i in range(19)]
    return pd.DataFrame(average - historical, index=labels, columns=labels)


def table7_correlation_dispersion(fit: SampleFit) -> pd.DataFrame:
    """Standard deviation through time of each implied correlation."""
    dispersion = np.nanstd(implied_correlations(fit), axis=0)
    labels = [f"{(i + 1) * 0.5:.2f}" for i in range(19)]
    return pd.DataFrame(dispersion, index=labels, columns=labels)


# ---------------------------------------------------------------------------
# Table 8 -- swaption valuation errors by structure
# ---------------------------------------------------------------------------
def _error_summary(series: np.ndarray, label: dict) -> dict:
    mean, tstat = ar1_adjusted_tstat(series)
    return {
        **label,
        "mean": mean,
        "t_stat": tstat,
        "std": float(np.nanstd(series, ddof=1)),
        "min": float(np.nanmin(series)),
        "median": float(np.nanmedian(series)),
        "max": float(np.nanmax(series)),
        "serial_corr": serial_correlation(series),
    }


def table8_swaption_errors(fit: SampleFit) -> pd.DataFrame:
    """
    Summary statistics for percentage swaption valuation errors, by structure.

    The paper finds the errors highly serially correlated, the largest and most
    positive for five-year-expiry swaptions (all means above four percent), and
    most other means consequently negative because the fitting procedure
    imposes an implicit adding-up-to-zero constraint.
    """
    errors = fit.swaption_errors
    rows = [
        _error_summary(errors[:, k], {"N": expiry, "M": tenor})
        for k, (expiry, tenor) in enumerate(fit.data.swaption_grid)
    ]
    frame = pd.DataFrame(rows)

    overall = _error_summary(errors.reshape(-1), {"N": np.nan, "M": np.nan})
    # The paper averages the columns, except min/median/max which pool.
    overall["mean"] = float(frame["mean"].mean())
    overall["t_stat"] = float(frame["t_stat"].mean())
    overall["std"] = float(frame["std"].mean())
    overall["serial_corr"] = float(frame["serial_corr"].mean())
    frame.loc[len(frame)] = overall
    frame.index = list(frame.index[:-1]) + ["Overall"]
    return frame


# ---------------------------------------------------------------------------
# Tables 9 and 10 -- cap valuation errors
# ---------------------------------------------------------------------------
def cap_errors(fit: SampleFit) -> np.ndarray:
    """
    (n_weeks, 6) percentage cap valuation errors from the fitted model.

    Positive means the swaptions market implies a higher cap value than the
    caps market quotes -- the cap is cheap relative to swaptions.
    """
    data = fit.data
    out = np.empty((fit.n_weeks, len(data.cap_maturities)))
    for week in range(fit.n_weeks):
        valuation = value_caps(
            fit.curves[week],
            fit.results[week].sigma,
            data.cap_vols[week],
            data.cap_maturities,
        )
        out[week] = valuation.percentage_errors
    return out


def table9_cap_errors(fit: SampleFit) -> pd.DataFrame:
    """
    Summary statistics for percentage cap valuation errors (four factors).

    The paper rejects equality at every maturity: means run from 23.3 percent
    at two years down to 5.7 percent at five years, all positive and
    significant.  But the medians are all within three percent of zero -- the
    two markets are usually consistent, and the positive means come from
    periodic large dislocations, giving a skewed, somewhat bimodal error
    distribution.
    """
    errors = cap_errors(fit)
    rows = []
    for k, maturity in enumerate(fit.data.cap_maturities):
        row = _error_summary(errors[:, k], {"maturity": maturity})
        paper = paper_data.TABLE9_CAP_ERRORS.get(maturity)
        row["paper_mean"] = paper[0] if paper else np.nan
        row["paper_median"] = paper[4] if paper else np.nan
        rows.append(row)

    frame = pd.DataFrame(rows)
    overall = _error_summary(errors.reshape(-1), {"maturity": np.nan})
    overall["mean"] = float(frame["mean"].mean())
    overall["t_stat"] = float(frame["t_stat"].mean())
    overall["std"] = float(frame["std"].mean())
    overall["serial_corr"] = float(frame["serial_corr"].mean())
    overall["paper_mean"] = paper_data.TABLE9_OVERALL[0]
    overall["paper_median"] = paper_data.TABLE9_OVERALL[4]
    frame.loc[len(frame)] = overall
    frame.index = list(frame.index[:-1]) + ["Overall"]
    return frame


def table10_merton_bound(
    data: MarketData,
    basis: FactorBasis | None = None,
    method: str = "approx",
    sim_config: smm.SimulationConfig | None = None,
    fit_1f: SampleFit | None = None,
) -> pd.DataFrame:
    """
    Percentage cap valuation errors from the ONE-factor model.

    With a rank-one covariance matrix all forwards are perfectly correlated, so
    the model cap price is a Merton (1973) lower bound for the market price: a
    portfolio of options must be worth at least an option on the portfolio.
    Every error should therefore be negative.  In the paper only 8 of 4,352
    observations are positive.
    """
    fit = fit_1f or fit_sample(
        data, n_factors=1, basis=basis, method=method, sim_config=sim_config
    )

    errors = np.empty((fit.n_weeks, len(data.cap_maturities)))
    for week in range(fit.n_weeks):
        errors[week] = check_merton_bound(
            fit.curves[week],
            fit.results[week].sigma,
            data.cap_vols[week],
            data.cap_maturities,
        )

    rows = []
    for k, maturity in enumerate(data.cap_maturities):
        row = _error_summary(errors[:, k], {"maturity": maturity})
        row["n_positive"] = int(np.sum(errors[:, k] > 0))
        paper = paper_data.TABLE10_CAP_ERRORS_1F.get(maturity)
        row["paper_mean"] = paper[0] if paper else np.nan
        row["paper_n_positive"] = paper[6] if paper else np.nan
        rows.append(row)

    frame = pd.DataFrame(rows)
    overall = _error_summary(errors.reshape(-1), {"maturity": np.nan})
    overall["mean"] = float(frame["mean"].mean())
    overall["t_stat"] = float(frame["t_stat"].mean())
    overall["std"] = float(frame["std"].mean())
    overall["serial_corr"] = float(frame["serial_corr"].mean())
    overall["n_positive"] = int(frame["n_positive"].sum())
    overall["paper_mean"] = -17.279
    overall["paper_n_positive"] = 8
    frame.loc[len(frame)] = overall
    frame.index = list(frame.index[:-1]) + ["Overall"]
    return frame


# ---------------------------------------------------------------------------
# Figures 6 and 7
# ---------------------------------------------------------------------------
def figure6_eigenvalues(fit: SampleFit) -> pd.DataFrame:
    """Time series of the implied eigenvalues -- the implied factor variances."""
    values = fit.eigenvalues
    columns = [f"eigenvalue_{k + 1}" for k in range(values.shape[1])]
    return pd.DataFrame(values, index=fit.data.dates, columns=columns)


def figure7_rmse(fit: SampleFit) -> pd.Series:
    """
    Time series of the weekly swaption RMSE, in percent.

    The paper reports a median of 3.10 percent and a standard deviation of
    2.98 percent, generally two to three percent outside the crisis, spiking
    as high as 16 percent in autumn 1998.
    """
    return pd.Series(fit.rmse, index=fit.data.dates, name="rmse_percent")


def rmse_summary(fit: SampleFit) -> dict:
    """Headline RMSE statistics, comparable with Section 5.3."""
    rmse = fit.rmse
    mask = fit.data.crisis_mask()
    return {
        "median": float(np.median(rmse)),
        "mean": float(np.mean(rmse)),
        "std": float(np.std(rmse, ddof=1)),
        "max": float(np.max(rmse)),
        "median_ex_crisis": float(np.median(rmse[~mask])) if (~mask).any() else np.nan,
        "median_in_crisis": float(np.median(rmse[mask])) if mask.any() else np.nan,
        "n_weeks_above_5pct": int(np.sum(rmse > 5.0)),
        "paper_median": 3.10,
        "paper_std": 2.98,
    }
