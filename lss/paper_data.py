"""
Hard-coded data tables transcribed from the published paper.

These serve two purposes:
  1. The ex-ante historical correlation matrix H (Table 2) is a genuine *model
     input*: every in-sample result in the paper uses the eigenvectors of this
     matrix.  Shipping it means the pipeline runs correctly with no data at all.
  2. Tables 1, 3, 4, 9 and 10 are used to calibrate the synthetic data generator
     and to benchmark our reproduced results against the published ones.
"""

from __future__ import annotations
import numpy as np

from .conventions import N_FORWARDS

# ---------------------------------------------------------------------------
# TABLE 2 -- Correlation matrix of weekly log changes in six-month Libor
# forward rates, ex-ante period 17-Jan-1992 to 17-Jan-1997 (262 observations).
# Lower triangle as printed in the paper; horizons .50 .. 9.50.
# ---------------------------------------------------------------------------
_TABLE2_LOWER: list[list[float]] = [
    [1.000],
    [.340, 1.000],
    [.579, .950, 1.000],
    [.547, .946, .995, 1.000],
    [.544, .904, .964, .983, 1.000],
    [.519, .812, .883, .918, .975, 1.000],
    [.476, .771, .841, .884, .954, .996, 1.000],
    [.440, .731, .800, .849, .929, .984, .996, 1.000],
    [.410, .692, .762, .815, .901, .967, .986, .997, 1.000],
    [.388, .656, .727, .781, .871, .944, .969, .986, .996, 1.000],
    [.373, .621, .693, .747, .838, .914, .943, .966, .983, .995, 1.000],
    [.365, .585, .659, .711, .798, .873, .905, .933, .957, .978, .994, 1.000],
    [.360, .546, .621, .668, .747, .816, .850, .882, .912, .942, .970, .991, 1.000],
    [.358, .501, .576, .615, .681, .739, .773, .807, .844, .882, .922, .960, .988, 1.000],
    [.355, .448, .522, .551, .600, .643, .674, .709, .749, .796, .847, .901, .950, .986, 1.000],
    [.349, .391, .462, .479, .507, .531, .557, .591, .634, .686, .748, .816, .884, .944, .986, 1.000],
    [.340, .333, .400, .404, .411, .414, .434, .465, .508, .564, .633, .712, .797, .879, .945, .987, 1.000],
    [.329, .279, .341, .333, .320, .302, .315, .343, .385, .442, .515, .603, .700, .799, .886, .952, .989, 1.000],
    [.317, .233, .290, .271, .239, .202, .209, .232, .271, .329, .404, .497, .603, .714, .819, .903, .961, .991, 1.000],
]


def _symmetrise(lower: list[list[float]]) -> np.ndarray:
    n = len(lower)
    m = np.zeros((n, n))
    for i, row in enumerate(lower):
        for j, v in enumerate(row):
            m[i, j] = v
            m[j, i] = v
    return m


HISTORICAL_CORRELATION = _symmetrise(_TABLE2_LOWER)
assert HISTORICAL_CORRELATION.shape == (N_FORWARDS, N_FORWARDS)


# ---------------------------------------------------------------------------
# TABLE 1 -- Summary statistics for the 19 six-month Libor forward rates,
# in-sample period 24-Jan-1997 to 2-Jul-1999 (128 weekly observations).
# Rates in percent.
# columns: mean, std, min, median, max, first-order serial correlation
# ---------------------------------------------------------------------------
TABLE1_FORWARD_STATS = np.array([
    [5.752, .501, 4.406, 5.840, 6.714, .974],   # 0.50
    [5.917, .522, 4.511, 5.941, 7.004, .950],   # 1.00
    [5.993, .614, 4.367, 5.982, 7.275, .962],   # 1.50
    [6.071, .615, 4.549, 6.042, 7.374, .962],   # 2.00
    [6.129, .557, 4.885, 6.061, 7.324, .964],   # 2.50
    [6.184, .513, 5.130, 6.086, 7.289, .965],   # 3.00
    [6.248, .519, 5.212, 6.122, 7.373, .964],   # 3.50
    [6.307, .524, 5.285, 6.152, 7.449, .963],   # 4.00
    [6.361, .529, 5.349, 6.192, 7.518, .962],   # 4.50
    [6.409, .534, 5.402, 6.232, 7.579, .962],   # 5.00
    [6.451, .538, 5.445, 6.260, 7.632, .961],   # 5.50
    [6.488, .540, 5.478, 6.296, 7.677, .961],   # 6.00
    [6.519, .542, 5.499, 6.322, 7.713, .960],   # 6.50
    [6.543, .541, 5.509, 6.351, 7.740, .958],   # 7.00
    [6.561, .540, 5.508, 6.357, 7.757, .954],   # 7.50
    [6.572, .538, 5.494, 6.373, 7.765, .949],   # 8.00
    [6.576, .535, 5.467, 6.396, 7.761, .941],   # 8.50
    [6.573, .533, 5.428, 6.413, 7.746, .929],   # 9.00
    [6.562, .531, 5.376, 6.413, 7.719, .912],   # 9.50
])
assert TABLE1_FORWARD_STATS.shape == (N_FORWARDS, 6)

FORWARD_MEAN = TABLE1_FORWARD_STATS[:, 0] / 100.0
FORWARD_STD = TABLE1_FORWARD_STATS[:, 1] / 100.0


# ---------------------------------------------------------------------------
# TABLE 3 -- Summary statistics for the 34 ATMF European swaption Black
# volatilities (percent), 128 weekly observations.
# key: (expiry N, tenor M) -> (mean, std, min, median, serial correlation)
#
# NOTE: the "Maximum" column of Table 3 in the source PDF is corrupted for
# every row but the first (it repeats the Minimum column), so it is not
# transcribed.  Only the first row's maximum (27.00) is trustworthy.
# ---------------------------------------------------------------------------
TABLE3_SWAPTION_VOLS: dict[tuple[float, float], tuple[float, float, float, float, float]] = {
    (0.5, 1.0): (14.60, 3.37, 9.70, 13.50, .926),
    (1.0, 1.0): (16.10, 2.98, 11.80, 15.35, .945),
    (2.0, 1.0): (16.66, 2.27, 13.10, 16.20, .940),
    (3.0, 1.0): (16.42, 1.99, 13.00, 16.10, .932),
    (4.0, 1.0): (16.08, 1.72, 13.00, 15.95, .941),
    (5.0, 1.0): (15.73, 1.48, 12.90, 15.70, .933),
    (0.5, 2.0): (15.35, 3.10, 10.40, 14.45, .919),
    (1.0, 2.0): (16.08, 2.59, 12.20, 15.50, .943),
    (2.0, 2.0): (16.31, 2.05, 13.00, 16.00, .936),
    (3.0, 2.0): (16.02, 1.74, 12.90, 15.85, .939),
    (4.0, 2.0): (15.70, 1.51, 12.80, 15.70, .931),
    (5.0, 2.0): (15.38, 1.34, 12.70, 15.50, .925),
    (0.5, 3.0): (15.34, 2.93, 10.60, 14.65, .913),
    (1.0, 3.0): (15.89, 2.35, 12.20, 15.30, .937),
    (2.0, 3.0): (16.00, 1.84, 12.90, 15.70, .933),
    (3.0, 3.0): (15.72, 1.57, 12.80, 15.65, .927),
    (4.0, 3.0): (15.42, 1.37, 12.70, 15.50, .929),
    (5.0, 3.0): (15.09, 1.23, 12.60, 15.20, .920),
    (0.5, 4.0): (15.32, 2.73, 10.80, 14.70, .906),
    (1.0, 4.0): (15.67, 2.14, 12.10, 15.30, .930),
    (2.0, 4.0): (15.72, 1.64, 12.80, 15.65, .933),
    (3.0, 4.0): (15.45, 1.42, 12.70, 15.50, .930),
    (4.0, 4.0): (15.14, 1.27, 12.60, 15.20, .918),
    (5.0, 4.0): (14.80, 1.14, 12.50, 15.00, .917),
    (0.5, 5.0): (15.30, 2.57, 11.00, 14.70, .897),
    (1.0, 5.0): (15.42, 1.93, 12.00, 15.20, .926),
    (2.0, 5.0): (15.45, 1.49, 12.70, 15.40, .924),
    (3.0, 5.0): (15.18, 1.32, 12.60, 15.25, .932),
    (4.0, 5.0): (14.85, 1.18, 12.50, 15.00, .919),
    (5.0, 5.0): (14.48, 1.04, 12.50, 14.70, .909),
    (0.5, 7.0): (15.09, 2.41, 11.00, 14.60, .887),
    (1.0, 7.0): (15.10, 1.71, 12.00, 15.00, .914),
    (2.0, 7.0): (15.03, 1.38, 12.40, 15.05, .921),
    (3.0, 7.0): (14.77, 1.24, 12.30, 14.90, .913),
}
assert len(TABLE3_SWAPTION_VOLS) == 34


# ---------------------------------------------------------------------------
# TABLE 4 -- Summary statistics for Libor cap Black volatilities (percent).
# maturity -> (mean, std, min, median, max, serial correlation)
# ---------------------------------------------------------------------------
TABLE4_CAP_VOLS: dict[float, tuple[float, float, float, float, float, float]] = {
    2.0:  (15.34, 3.19, 10.60, 14.25, 28.00, .945),
    3.0:  (16.43, 2.75, 12.10, 15.60, 25.20, .940),
    4.0:  (16.75, 2.41, 12.50, 16.10, 23.50, .932),
    5.0:  (16.84, 2.21, 12.90, 16.30, 22.75, .941),
    7.0:  (16.46, 1.86, 12.75, 16.15, 20.87, .933),
    10.0: (15.97, 1.55, 12.60, 15.90, 19.75, .919),
}


# ---------------------------------------------------------------------------
# TABLE 5 -- Likelihood ratio tests for the number of implied factors.
# (N, N+1) -> test statistic, full sample.  chi2_128 critical value 168.1332.
# ---------------------------------------------------------------------------
TABLE5_LR_FULL = {
    (1, 2): 3352.183, (2, 3): 2220.194, (3, 4): 284.847, (4, 5): 86.211,
    (5, 6): 13.204, (6, 7): 21.991, (7, 8): 3.959,
}
CHI2_128_99 = 168.1332
CHI2_64_99 = 93.2169

# ---------------------------------------------------------------------------
# TABLE 9 -- Percentage cap valuation errors, four-factor model.
# maturity -> (mean, t-stat, std, min, median, max, serial correlation)
# ---------------------------------------------------------------------------
TABLE9_CAP_ERRORS = {
    2.0:  (23.326, 3.50, 6.669, -19.484, 2.589, 113.307, .663),
    3.0:  (11.782, 2.73, 4.319, -11.359, -.705, 63.361, .664),
    4.0:  (7.666, 2.36, 3.245, -9.739, -2.120, 47.687, .646),
    5.0:  (5.665, 2.14, 2.644, -9.191, -2.385, 38.071, .641),
    7.0:  (6.021, 2.58, 2.332, -8.941, -.054, 32.379, .685),
    10.0: (7.864, 3.57, 2.200, -4.592, 2.883, 32.578, .734),
}
TABLE9_OVERALL = (10.387, 2.81, 3.568, -19.484, .862, 113.307, .672)

# ---------------------------------------------------------------------------
# TABLE 10 -- Percentage cap valuation errors, ONE-factor model (all forward
# correlations = 1).  Merton (1973) implies these should all be negative:
# a portfolio of options >= an option on a portfolio.
# maturity -> (mean, t-stat, std, min, median, max, n_positive, serial corr)
# ---------------------------------------------------------------------------
TABLE10_CAP_ERRORS_1F = {
    2.0:  (-25.286, -7.76, 3.259, -62.054, -23.639, -9.359, 0, .885),
    3.0:  (-23.400, -14.12, 1.657, -46.757, -22.431, -8.721, 0, .827),
    4.0:  (-20.288, -24.39, .832, -33.633, -19.423, -13.697, 0, .717),
    5.0:  (-17.500, -31.59, .554, -26.244, -16.969, -6.238, 0, .631),
    7.0:  (-11.320, -27.77, .408, -20.658, -11.132, 3.732, 1, .529),
    10.0: (-5.884, -8.63, .682, -13.457, -5.954, 12.071, 7, .716),
}

# Sample period facts used by the empirics module.
N_WEEKS = 128
N_SWAPTIONS = 34
SAMPLE_START = "1997-01-24"
SAMPLE_END = "1999-07-02"
EXANTE_START = "1992-01-17"
EXANTE_END = "1997-01-17"

# The hedge-fund (LTCM) crisis window: RMSEs exceed 5% for the ten weeks from
# 11-Sep-1998 to 13-Nov-1998 (Section 5.3).
CRISIS_START = "1998-09-11"
CRISIS_END = "1998-11-13"

# Bloomberg bid-offer for these swaptions is about one Black vol point, which
# for a 16% vol is roughly 6% of an ATMF swaption premium (footnote 23).
BID_OFFER_VOL_POINTS = 1.0


def historical_correlation() -> np.ndarray:
    """The paper's Table 2 ex-ante correlation matrix (19x19)."""
    return HISTORICAL_CORRELATION.copy()


def market_swaption_vol_means() -> dict[tuple[float, float], float]:
    """Mean Black vol (decimal) per swaption structure, from Table 3."""
    return {k: v[0] / 100.0 for k, v in TABLE3_SWAPTION_VOLS.items()}


def market_cap_vol_means() -> dict[float, float]:
    """Mean Black cap vol (decimal) per maturity, from Table 4."""
    return {k: v[0] / 100.0 for k, v in TABLE4_CAP_VOLS.items()}
