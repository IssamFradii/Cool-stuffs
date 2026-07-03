"""Validate the tool against numbers published in Brigo & Mercurio.

Section 7.9 applies formula (6.76) with rho = 1 to the Table 7.13 data
(February 1, 2002) and publishes both the annualized caplet volatilities
and the implied infra-correlations. If our implementation is right we
must reproduce those numbers to the book's printed precision.

Run:  python tests/test_book_numbers.py   (or pytest tests/)
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wedge.annualize import cap_implied_swaption_vol, implied_infra_correlation

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "brigo_table_7_13.csv")

# Published in Section 7.9, "Annualized caplet volatilities" (3 decimals)
BOOK_ANNUALIZED_VOLS = [0.178, 0.185, 0.163, 0.155, 0.152,
                        0.145, 0.138, 0.134, 0.129, 0.125]

# Published in Section 7.9, "Infra-correlations" (3 decimals)
BOOK_IMPLIED_RHO = [1.022, 0.388, 0.543, 0.536, 0.444,
                    0.493, 0.533, 0.560, 0.586, 0.598]


def test_annualized_caplet_vols_match_book():
    df = pd.read_csv(DATA)
    vols = cap_implied_swaption_vol(df.f1, df.f2, df.caplet_vol_s,
                                    df.caplet_vol_t, rho=1.0)
    np.testing.assert_allclose(np.round(vols, 3), BOOK_ANNUALIZED_VOLS, atol=5e-4)


def test_implied_infra_correlations_match_book():
    df = pd.read_csv(DATA)
    rho = implied_infra_correlation(df.swaption_vol, df.f1, df.f2,
                                    df.caplet_vol_s, df.caplet_vol_t)
    np.testing.assert_allclose(np.round(rho, 3), BOOK_IMPLIED_RHO, atol=2e-3)


if __name__ == "__main__":
    test_annualized_caplet_vols_match_book()
    print("OK: annualized caplet vols match Section 7.9")
    test_implied_infra_correlations_match_book()
    print("OK: implied infra-correlations match Section 7.9")
