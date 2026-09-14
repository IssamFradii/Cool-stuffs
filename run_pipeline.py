"""
End-to-end driver for the Longstaff / Santa-Clara / Schwartz cap-swaption
pipeline.

    python run_pipeline.py                   # synthetic data, full run
    python run_pipeline.py --quick           # fewer paths/weeks, for a fast check
    python run_pipeline.py --data-dir ./mydata   # real CSVs (see lss/data.py)
    python run_pipeline.py --factor-tests    # also run the Table 5 LR tests (slow)
    python run_pipeline.py --no-wedge        # null test: no dislocation injected

Outputs are printed and written to ./outputs as CSVs.
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lss import (
    calibrate, caps, data as data_mod, empirics, historical, paper_data,
    smm, strategy, synthetic,
)
from lss.conventions import CAP_MATURITIES, SWAPTION_GRID


# ---------------------------------------------------------------------------
def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def fmt(frame: pd.DataFrame, floatfmt: str = "%.3f") -> str:
    with pd.option_context(
        "display.width", 200,
        "display.max_columns", 40,
        "display.float_format", lambda v: floatfmt % v,
    ):
        return frame.to_string()


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default=None,
                        help="directory of real CSVs; omit to use synthetic data")
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--factors", type=int, default=4,
                        help="rank of the implied covariance matrix")
    parser.add_argument("--paths", type=int, default=8000,
                        help="Monte-Carlo paths. The paper uses 2,000, but the "
                             "cap wedge of Table 9 is not converged there: it "
                             "is noisy across seeds and biased high. Use 8,000+ "
                             "for stable cap errors; 2,000 to reproduce the "
                             "paper's own setting.")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--quick", action="store_true",
                        help="short sample and fewer paths, for a fast check")
    parser.add_argument("--factor-tests", action="store_true",
                        help="run the Table 5 likelihood ratio tests (slow)")
    parser.add_argument("--no-wedge", action="store_true",
                        help="synthetic data with no injected dislocation")
    parser.add_argument("--method", choices=("approx", "simulation"),
                        default="approx")
    parser.add_argument("--save-data", action="store_true",
                        help="write the synthetic data out as CSVs")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sim_config = smm.SimulationConfig(
        n_paths=400 if args.quick else args.paths, seed=args.seed
    )

    # ---------------------------------------------------------------- data --
    banner("1. DATA")
    truth = None
    if args.data_dir:
        print(f"Loading real market data from {args.data_dir}")
        market = data_mod.load_market_data(args.data_dir)
    else:
        end = "1997-12-26" if args.quick else paper_data.SAMPLE_END
        print("Generating synthetic data calibrated to Tables 1, 3 and 4.")
        print("NOTE: synthetic data exercises the pipeline; it is not evidence")
        print("      about the real market.")
        market, truth = synthetic.generate_market_data(
            end=end, with_wedge=not args.no_wedge
        )
        if args.save_data:
            data_mod.save_market_data(market, out_dir / "synthetic_data")
            print(f"  wrote CSVs to {out_dir / 'synthetic_data'}")

    print(f"  {market.n_weeks} weekly observations, "
          f"{market.dates[0].date()} to {market.dates[-1].date()}")
    print(f"  {len(market.swaption_grid)} swaption structures, "
          f"{len(market.cap_maturities)} cap maturities")
    print(f"  6M Libor forward: mean {100 * market.forwards[:, 0].mean():.2f}%, "
          f"10y point: {100 * market.forwards[:, -1].mean():.2f}%")
    print(f"  swaption vols: {100 * market.swaption_vols.mean():.2f}% average, "
          f"peak {100 * market.swaption_vols.max():.2f}%")
    print(f"  cap vols:      {100 * market.cap_vols.mean():.2f}% average, "
          f"peak {100 * market.cap_vols.max():.2f}%")

    # ------------------------------------------------------- factor basis --
    banner("2. FACTOR BASIS (ex-ante historical correlation matrix, Table 2)")
    basis = historical.paper_factor_basis()
    total = basis.eigenvalues.sum()
    print("  eigenvalue   share of variance   cumulative")
    cumulative = 0.0
    for k in range(6):
        share = basis.eigenvalues[k] / total
        cumulative += share
        print(f"  {k + 1:>2}  {basis.eigenvalues[k]:8.4f}   "
              f"{100 * share:7.2f}%          {100 * cumulative:6.2f}%")
    print("  Factors 1-4 are level, slope, curvature and a short-end factor")
    print("  (Figure 5); the fourth is the one Knez-Litterman-Scheinkman find.")

    # --------------------------------------------------------- calibration --
    banner(f"3. CALIBRATION ({args.factors}-factor string market model)")
    start = time.time()
    fit = empirics.fit_sample(
        market, n_factors=args.factors, basis=basis,
        method=args.method, sim_config=sim_config, progress=True,
    )
    print(f"  calibrated {fit.n_weeks} weeks in {time.time() - start:.1f}s")

    summary = empirics.rmse_summary(fit)
    print()
    print("  Swaption pricing RMSE (percent of market price)  [Figure 7]")
    print(f"    median            {summary['median']:6.2f}%   "
          f"(paper: {summary['paper_median']:.2f}%)")
    print(f"    standard dev.     {summary['std']:6.2f}%   "
          f"(paper: {summary['paper_std']:.2f}%)")
    print(f"    maximum           {summary['max']:6.2f}%")
    print(f"    median ex-crisis  {summary['median_ex_crisis']:6.2f}%")
    print(f"    median in crisis  {summary['median_in_crisis']:6.2f}%")
    print(f"    weeks above 5%    {summary['n_weeks_above_5pct']:4d}")

    eigen = empirics.figure6_eigenvalues(fit)
    eigen.to_csv(out_dir / "figure6_eigenvalues.csv")
    empirics.figure7_rmse(fit).to_csv(out_dir / "figure7_rmse.csv")
    print()
    print("  Implied eigenvalues (implied factor variances)  [Figure 6]")
    print(fmt(eigen.describe().loc[["mean", "std", "min", "max"]], "%.4f"))

    # ----------------------------------------------- implied correlations --
    banner("4. IMPLIED vs HISTORICAL CORRELATIONS (Tables 6 and 7)")
    diff = empirics.table6_correlation_differences(fit)
    disp = empirics.table7_correlation_dispersion(fit)
    diff.to_csv(out_dir / "table6_correlation_differences.csv")
    disp.to_csv(out_dir / "table7_correlation_dispersion.csv")

    off_diagonal = diff.to_numpy()[np.triu_indices(19, k=1)]
    print(f"  mean off-diagonal difference   {np.nanmean(off_diagonal):+.4f}")
    print(f"  share negative                 "
          f"{100 * np.nanmean(off_diagonal < 0):.1f}%")
    print(f"  most negative                  {np.nanmin(off_diagonal):+.4f}")
    print(f"  1st-vs-2nd forward difference  {diff.iloc[1, 0]:+.4f}")
    print("  Paper: implied correlations are generally LOWER than historical,")
    print("  mostly by .05 to .10, the exception being the first two forwards.")

    # ------------------------------------------------- swaption residuals --
    banner("5. SWAPTION VALUATION ERRORS (Table 8)")
    table8 = empirics.table8_swaption_errors(fit)
    table8.to_csv(out_dir / "table8_swaption_errors.csv")
    print(fmt(table8.head(12)))
    print("  ...")
    print(fmt(table8.tail(3)))

    # ---------------------------------------------------------- the wedge --
    banner("6. CAP VALUATION ERRORS (Table 9) -- THE CAP/SWAPTION WEDGE")
    table9 = empirics.table9_cap_errors(fit)
    table9.to_csv(out_dir / "table9_cap_errors.csv")
    print(fmt(table9))
    print()
    print("  Positive => the swaptions market implies a higher cap value than")
    print("  the caps market quotes, i.e. the cap is cheap.")

    # --------------------------------------------------- the Merton bound --
    banner("7. MERTON (1973) NO-ARBITRAGE BOUND (Table 10)")
    print("  Fitting the rank-one model (all forward correlations = 1)...")
    fit1 = empirics.fit_sample(
        market, n_factors=1, basis=basis, method=args.method,
        sim_config=sim_config,
    )
    table10 = empirics.table10_merton_bound(market, fit_1f=fit1)
    table10.to_csv(out_dir / "table10_merton_bound.csv")
    print(fmt(table10))
    print()
    total_positive = int(table10.loc["Overall", "n_positive"])
    total_obs = market.n_weeks * len(market.cap_maturities)
    print(f"  Violations: {total_positive} of {total_obs} "
          f"({100 * total_positive / total_obs:.2f}%).")
    print("  A portfolio of options must be worth at least an option on the")
    print("  portfolio, so these should be negative. Paper: 8 of 4,352 (.18%).")

    # ------------------------------------------------------- factor tests --
    if args.factor_tests:
        banner("8. HOW MANY IMPLIED FACTORS? (Table 5)")
        print("  This refits the whole sample at each rank; it takes a while.")
        table5 = empirics.table5_factor_tests(
            market, max_factors=5, basis=basis, method=args.method,
            sim_config=sim_config, progress=True,
        )
        table5.to_csv(out_dir / "table5_factor_tests.csv")
        print(fmt(table5))
        print(f"  chi-square(128) 99% critical value: {paper_data.CHI2_128_99:.4f}")
        print("  Paper: 2v1, 3v2 and 4v3 significant; 5v4 and beyond are not.")

    # ---------------------------------------------------------- strategy --
    banner("9. THE WEDGE STRATEGY")
    config = strategy.StrategyConfig()
    print(f"  entry |z| >= {config.entry_z}, exit |z| <= {config.exit_z}, "
          f"causal z-score over {config.lookback} weeks")
    print(f"  bid-offer {config.vol_point_cost} vol point per leg per trade, "
          f"delta hedged: {config.delta_hedge}")

    result = strategy.backtest(fit, config)
    stats = strategy.summarize(result)
    config_zero = strategy.StrategyConfig(
        entry_z=config.entry_z, exit_z=config.exit_z,
        max_hedge_instruments=config.max_hedge_instruments, vol_point_cost=0.0,
    )
    stats.to_csv(out_dir / "strategy_performance.csv")
    result.pnl.to_csv(out_dir / "strategy_pnl.csv")
    result.spread.to_csv(out_dir / "strategy_spread.csv")

    print()
    print(fmt(stats, "%.4f"))
    print()
    print("  gross_pnl  = before transaction costs (is the signal real?)")
    print("  net_pnl    = after costs (is it tradable?)")
    print("  breakeven_vol_points = bid-offer per leg at which net P&L is zero;")
    print("               compare with the ~1.0 vol point Bloomberg quotes.")

    print()
    print("  Sensitivity to entry threshold and bid-offer")
    sens = strategy.sensitivity(fit)
    sens.to_csv(out_dir / "strategy_sensitivity.csv")
    print(fmt(sens.set_index(["entry_z", "cost_vol_points"]), "%.4f"))
    print("  Read round_trips first: a cell with a couple of round trips is an")
    print("  anecdote, not a backtest.")

    print()
    print("  Signal diagnostics")
    diagnostics = strategy.signal_diagnostics(result, truth)
    print(fmt(diagnostics))
    if truth is not None:
        column = "corr_with_true_wedge"
        if column in diagnostics:
            print(f"  Mean correlation with the injected wedge: "
                  f"{diagnostics[column].mean():.3f}")
            print("  (Synthetic only: this is the sharpest test that the")
            print("   calibration recovers a dislocation that was really there.)")

    if market.is_synthetic:
        print()
        if args.no_wedge:
            print("  THIS IS THE NULL RUN: no dislocation was injected at all.")
            print("  Any P&L above is what the strategy earns from nothing --")
            print("  a mean-reversion rule on a noisily estimated signal profits")
            print("  in sample because the estimation noise itself mean-reverts.")
            print("  Compare it with the main run before believing that result.")
        else:
            print("  REMINDER: this backtest ran on synthetic data. It shows the")
            print("  machinery works end to end. It says nothing about real P&L.")
            print()
            print("  RUN THE CONTROL:  python run_pipeline.py --no-wedge")
            print("  It generates data with NO dislocation and runs the same")
            print("  strategy. On the shipped settings the null earns as much as")
            print("  the real thing, which means this P&L is not evidence of")
            print("  wedge capture. The signal diagnostic above -- correlation")
            print("  with the injected wedge -- is the test that does hold.")

    print()
    print(f"All tables written to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
