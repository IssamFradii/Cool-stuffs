"""
Market conventions and the fixed semiannual grid used throughout the paper.

The paper works on an idealised semiannual grid out to 10 years:

  * Discount bonds       D(0, 0.5), D(0, 1.0), ..., D(0, 10.0)          -> 20 maturities
  * Six-month forwards   F_i = F(t, T_i, T_i + 1/2), T_i = i/2, i=1..19 -> 19 forwards
                         (horizons 0.5y through 9.5y, covering [0.5,1.0] ... [9.5,10.0])

The first forward F(t, 0, 0.5) is the *spot* six-month rate and is not stochastic,
which is why there are 19 stochastic forwards and 19 stochastic bonds
D(.,1.0) ... D(.,10.0)  (see paper footnote 11).
"""

from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------- grid ----
N_FORWARDS = 19            # 19 six-month Libor forwards, horizons 0.5 .. 9.5
DELTA = 0.5                # accrual fraction of a six-month period
HORIZON = 10.0             # final maturity of the modelled curve, years
N_STEPS = 20               # number of six-month steps to reach the horizon

# Maturities of the discount bonds D(0, T): 0.5, 1.0, ..., 10.0
BOND_MATURITIES = np.arange(1, N_STEPS + 1) * DELTA

# Start dates T_i of the 19 six-month forwards: 0.5, 1.0, ..., 9.5
FORWARD_STARTS = np.arange(1, N_FORWARDS + 1) * DELTA

# Standard quoted swap maturities used to build the curve (Section 4)
SWAP_PILLARS = (2.0, 3.0, 4.0, 5.0, 7.0, 10.0)
LIBOR_PILLARS = (0.5, 1.0)

# ------------------------------------------------------- swaption grid ----
# Table 3: the 34 standard "N into M" ATMF European swaption structures with
# final swap maturity T = N + M <= 10.
#   for M in {1,2,3,4,5}: N in {0.5,1,2,3,4,5}   -> 30
#   for M = 7           : N in {0.5,1,2,3}       ->  4
# (Figure 3 notes the 4-into-7 and 5-into-7 are excluded as less liquid.)
def swaption_grid() -> list[tuple[float, float]]:
    """Return the 34 (expiry N, tenor M) swaption structures of Table 3."""
    grid: list[tuple[float, float]] = []
    for tenor in (1.0, 2.0, 3.0, 4.0, 5.0, 7.0):
        for expiry in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0):
            if expiry + tenor <= HORIZON:
                grid.append((expiry, tenor))
    return grid


SWAPTION_GRID = swaption_grid()
assert len(SWAPTION_GRID) == 34, f"expected 34 swaptions, got {len(SWAPTION_GRID)}"

# Table 4: quoted cap maturities
CAP_MATURITIES = (2.0, 3.0, 4.0, 5.0, 7.0, 10.0)


def on_grid(t: float, tol: float = 1e-9) -> bool:
    """True if t lies on the semiannual grid."""
    return abs(t / DELTA - round(t / DELTA)) < tol


def grid_index(t: float) -> int:
    """Index of maturity t on the bond grid (t=0.5 -> 0, t=1.0 -> 1, ...)."""
    if not on_grid(t):
        raise ValueError(f"maturity {t} is not on the semiannual grid")
    return int(round(t / DELTA)) - 1
