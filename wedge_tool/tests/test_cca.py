"""Validate the CCA against Brigo & Mercurio Section 7.4.1.

The book calibrates the TABLE 1 volatilities to the 10x10 swaption
matrix of Table 7.4 (May 16, 2000; rows 6y, 8y, 9y linearly interpolated
from the quoted 7x10 matrix), Table 7.1 forwards, and the rank-2
correlation from the printed angles, publishing the result in Table 7.5
- including its famous negative entries. We must reproduce it.

Note: Table 7.4 prints the interpolated rows rounded to 3 decimals; the
book calibrated on the unrounded interpolation, so we rebuild rows 6y,
8y, 9y exactly before comparing (the cascade amplifies input rounding).

Run:  python tests/test_cca.py   (or pytest tests/)
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wedge.cca import (
    BOOK_ANGLES,
    cascade_calibration,
    diagnose,
    rank2_correlation,
    reprice_swaption_vol,
)

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Book Table 7.5 (fitted GPC instantaneous vols, 4 decimals), all 19 rows
BOOK_TABLE_7_5 = [
    [0.1800],
    [0.1548, 0.2039],
    [0.1285, 0.1559, 0.2329],
    [0.1178, 0.1042, 0.1656, 0.2437],
    [0.1091, 0.0988, 0.0973, 0.1606, 0.2483],
    [0.1131, 0.0734, 0.0781, 0.1009, 0.1618, 0.2627],
    [0.1040, 0.0984, 0.0502, 0.0737, 0.1128, 0.1633, 0.2633],
    [0.0940, 0.1052, 0.0938, 0.0319, 0.0864, 0.0969, 0.1684, 0.2731],
    [0.1065, 0.0790, 0.0857, 0.0822, 0.0684, 0.0536, 0.0921, 0.1763, 0.2848],
    [0.1013, 0.0916, 0.0579, 0.1030, 0.1514, -0.0316, 0.0389, 0.0845, 0.1634, 0.2777],
    [0.0916, 0.0916, 0.0787, 0.0431, 0.0299, 0.2088, -0.0383, 0.0746, 0.0948, 0.1854],
    [0.0827, 0.0827, 0.0827, 0.0709, 0.0488, 0.0624, 0.1561, -0.0103, 0.0731, 0.0911],
    [0.0744, 0.0744, 0.0744, 0.0744, 0.0801, 0.0576, 0.0941, 0.1231, -0.0159, 0.0610],
    [0.0704, 0.0704, 0.0704, 0.0704, 0.0704, 0.1009, 0.0507, 0.0817, 0.1203, -0.0210],
    [0.0725, 0.0725, 0.0725, 0.0725, 0.0725, 0.0725, 0.1002, 0.0432, 0.0619, 0.1179],
    [0.0753, 0.0753, 0.0753, 0.0753, 0.0753, 0.0753, 0.0753, 0.0736, 0.0551, 0.0329],
    [0.0719, 0.0719, 0.0719, 0.0719, 0.0719, 0.0719, 0.0719, 0.0719, 0.0708, 0.0702],
    [0.0690, 0.0690, 0.0690, 0.0690, 0.0690, 0.0690, 0.0690, 0.0690, 0.0690, 0.0680],
    [0.0663, 0.0663, 0.0663, 0.0663, 0.0663, 0.0663, 0.0663, 0.0663, 0.0663, 0.0663],
]


def _book_matrix_unrounded():
    vmat = pd.read_csv(os.path.join(DATA, "swaption_matrix.csv"), index_col=0).to_numpy()
    v = vmat.copy()
    v[5] = 0.5 * (vmat[4] + vmat[6])                  # 6y = (5y + 7y)/2
    v[7] = vmat[6] + (vmat[9] - vmat[6]) / 3.0        # 8y
    v[8] = vmat[6] + 2.0 * (vmat[9] - vmat[6]) / 3.0  # 9y
    return v


def _calibrate():
    curve = pd.read_csv(os.path.join(DATA, "annual_curve.csv"))
    corr = rank2_correlation(BOOK_ANGLES)
    v = _book_matrix_unrounded()
    sigma = cascade_calibration(v, curve.forward.to_numpy(), corr)
    return sigma, v, curve, corr


def test_sigma_matches_book_table_7_5():
    sigma, *_ = _calibrate()
    for i, row in enumerate(BOOK_TABLE_7_5):
        got = sigma[i, : len(row)].real
        tol = 1e-4 if i < 7 else 3e-3   # deeper rows accumulate input rounding
        np.testing.assert_allclose(got, row, atol=tol,
                                   err_msg=f"row {i + 1} of Table 7.5")
    # the book's headline diagnostic: negative sigmas on the sub-diagonal
    flags = diagnose(sigma)
    for i, h in [(9, 5), (10, 6), (11, 7), (12, 8), (13, 9)]:
        assert flags[i, h] == "negative", f"expected negative sigma at ({i+1},{h+1})"
    # everything determined must be real for this input
    determined = ~np.isnan(sigma.real)
    assert np.abs(sigma.imag[determined]).max() < 1e-10


def test_calibration_reprices_input_exactly():
    sigma, v, curve, corr = _calibrate()
    fwd = curve.forward.to_numpy()
    for a in range(10):
        for c in range(1, 11):
            model = reprice_swaption_vol(sigma, fwd, corr, a, c)
            assert abs(model.real - v[a, c - 1]) < 1e-10, (a + 1, c)


if __name__ == "__main__":
    test_sigma_matches_book_table_7_5()
    print("OK: calibrated sigmas match book Table 7.5 (incl. the negative entries)")
    test_calibration_reprices_input_exactly()
    print("OK: calibrated model reprices all 100 input swaption vols exactly")
