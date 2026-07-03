"""Rich/cheap analysis of the cap/swaption wedge, per expiry bucket.

Usage:
    python run_analysis.py
    python run_analysis.py --data data/brigo_table_7_13.csv --rho 1.0
    python run_analysis.py --rho-low 0.7 --rho-high 1.0 --min-wedge 0.005

Reads a CSV of market quotes (see data/brigo_table_7_13.csv for the format),
prints the wedge table with rich/cheap signals and suggested trades, and
writes output/wedge_report.csv and output/wedge_report.png.
"""

import argparse
import os

import pandas as pd

from wedge.analysis import analyze_wedges
from wedge.report import print_report, plot_report

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", default=os.path.join(HERE, "data", "brigo_table_7_13.csv"),
                   help="CSV of market quotes, one row per expiry bucket")
    p.add_argument("--rho", type=float, default=1.0,
                   help="infra-correlation used to build the cap-implied vol (default 1.0)")
    p.add_argument("--rho-low", type=float, default=0.70,
                   help="lower edge of the plausible implied-rho band (default 0.70)")
    p.add_argument("--rho-high", type=float, default=1.00,
                   help="upper edge of the plausible implied-rho band (default 1.00)")
    p.add_argument("--min-wedge", type=float, default=0.005,
                   help="minimum |wedge| to flag, absolute vol (default 0.005 = 0.5 pts)")
    p.add_argument("--out", default=os.path.join(HERE, "output"),
                   help="output directory (default ./output)")
    args = p.parse_args()

    quotes = pd.read_csv(args.data)
    result = analyze_wedges(
        quotes,
        rho=args.rho,
        rho_low=args.rho_low,
        rho_high=args.rho_high,
        min_wedge=args.min_wedge,
    )

    print_report(result, rho=args.rho)

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "wedge_report.csv")
    png_path = os.path.join(args.out, "wedge_report.png")
    result.to_csv(csv_path, index=False)
    plot_report(result, rho=args.rho, rho_low=args.rho_low,
                rho_high=args.rho_high, path=png_path)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
