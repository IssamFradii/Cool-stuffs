"""Reproduce the worked example of the paper (Appendix, "An Example: 6m1y10y", p.8).

Paper assumptions:
    sigma 6m1y = 37nv, 6m11y = 99nv, 1y10y = 100nv, 18m10y = 99nv
    DV01/BPV:  6m1y = 1.0, 6m11y = 9.6, 18m10y = 8.6
    6m discount factor = 1, correlation of the mid-curve = 1

Paper results:
    sigma_MC = 107nv,  sigma_FVA = 95nv
    d sigma_FVA / d sigma_18m10y = 1.6,  d sigma_FVA / d sigma_MC = 0.6
    vega 18m10y = 8.4bp, vega mid-curve = 4.9bp
    100k/nv FVA  ->  146mm FVA notional, buy 186mm 18m10y, sell 116mm mid-curve
    mid-curve leg -> sell 66k/nv 6m11y, buy 6k/nv 6m1y
"""

import math

import fva

T0, T1, T2, T3 = 0.0, 0.5, 1.5, 11.5
S23, S13, S12 = 99.0, 99.0, 37.0  # 18m10y, 6m11y, 6m1y
D23, D13, D12 = 8.6, 9.6, 1.0  # BPV = DV01
RHO = 1.0
C = fva.SQRT_2_OVER_PI

ok = True


def check(name, got, expected, tol, unit=""):
    global ok
    good = abs(got - expected) <= tol
    ok &= good
    print(f"  [{'OK ' if good else 'FAIL'}] {name:<38} {got:>10.3f}{unit}   paper ~{expected}{unit}")


print("\n=== Paper example: 6m1y10y FVA (closed-form blocks) ===")

sig_mc = fva.midcurve_vol(D13, S13, D12, S12, D23, RHO)
check("sigma_MC", sig_mc, 107.0, 1.0, "nv")

sig_fva = fva.fva_vol(T1, T2, S23, sig_mc)
check("sigma_FVA", sig_fva, 95.0, 0.6, "nv")

d_fva_d23 = T2 * S23 / ((T2 - T1) * sig_fva)
d_fva_dmc = -T1 * sig_mc / ((T2 - T1) * sig_fva)
check("d sigma_FVA / d sigma_18m10y", d_fva_d23, 1.6, 0.05)
check("d sigma_FVA / d sigma_MC", abs(d_fva_dmc), 0.6, 0.05)

vega_van23 = C * math.sqrt(T2 - T0) * D23  # bp per 1nv
vega_mc = C * math.sqrt(T1 - T0) * D23
vega_fva = C * math.sqrt(T2 - T1) * D23  # DF(6m) = 1 in the paper
check("vega 18m10y straddle", vega_van23, 8.4, 0.05, "bp")
check("vega mid-curve", vega_mc, 4.9, 0.05, "bp")

target = 100_000.0  # 100k/nv
n_fva = target / (vega_fva * 1e-4) / 1e6
n_van23 = (target * d_fva_d23) / (vega_van23 * 1e-4) / 1e6
n_mc = (target * d_fva_dmc) / (vega_mc * 1e-4) / 1e6
check("FVA notional", n_fva, 146.0, 1.5, "mm")
check("18m10y notional (buy)", n_van23, 186.0, 2.0, "mm")
check("mid-curve notional (sell)", abs(n_mc), 116.0, 2.0, "mm")

d_mc_d13 = (D13**2 * S13 - RHO * D13 * D12 * S12) / (sig_mc * D23**2)
d_mc_d12 = (D12**2 * S12 - RHO * D13 * D12 * S13) / (sig_mc * D23**2)

# The paper quotes the mid-curve leg as a round -60k/nv (it rounds 0.558 -> 0.6);
# starting from that same -60k it gets 66k of 6m11y and 6k of 6m1y.
v_mc_paper = -60_000.0
check("6m11y vega from paper's -60k MC", abs(v_mc_paper * d_mc_d13) / 1e3, 66.0, 1.5, "k/nv")
check("6m1y vega from paper's -60k MC", abs(v_mc_paper * d_mc_d12) / 1e3, 6.0, 1.5, "k/nv")

# Unrounded chain rule from the 100k/nv FVA target (what the library returns).
v_mc = target * d_fva_dmc  # -55.8k/nv, i.e. the paper's -60k before rounding
print(
    f"  (unrounded chain: MC {v_mc / 1e3:.1f}k/nv -> 6m11y {v_mc * d_mc_d13 / 1e3:.1f}k/nv, "
    f"6m1y {v_mc * d_mc_d12 / 1e3:+.1f}k/nv)"
)

# --------------------------------------------------------------------------------------
print("\n=== Same trade, end-to-end from the loaded ATM matrix + curve ===")

import numpy as np
import pandas as pd

df = pd.read_csv("data/atm_vol.csv", index_col=0)
surface = fva.VolSurface(
    [fva.parse_period(x) for x in df.index],
    [fva.parse_period(c) for c in df.columns],
    df.to_numpy(float),
)
curve = fva.Curve.flat(0.0225)  # flat par curve that puts the 18m10y BPV near the paper's 8.6

res = fva.quote(curve, surface, strike=0.5, expiry=1.0, tenor=10.0, rho=1.0)
print(f"  trade                    {fva.trade_name(0.5, 1.0, 10.0)}")
check("sigma 18m10y (from surface)", res.sig_23, 99.0, 0.5, "nv")
check("sigma 6m11y  (from surface)", res.sig_13, 99.0, 1.0, "nv")
check("sigma 6m1y   (from surface)", res.sig_12, 37.0, 0.5, "nv")
check("BPV/DV01 18m10y (from curve)", res.dv01_23, 8.6, 0.3)
check("BPV/DV01 6m11y  (from curve)", res.dv01_13, 9.6, 0.5)
check("BPV/DV01 6m1y   (from curve)", res.dv01_12, 1.0, 0.1)
check("sigma_MC", res.sig_mc, 107.0, 2.0, "nv")
check("sigma_FVA", res.sig_fva, 95.0, 2.0, "nv")
check("discount vs spot 1y10y", res.discount_vs_spot * 100, -5.0, 2.0, "%")

rep = fva.replicate(res, 100_000.0)
print(
    f"  hedge: buy {rep.notional_van23 / 1e6:.0f}mm 18m10y ({rep.vega_van23 / 1e3:.0f}k/nv), "
    f"sell {abs(rep.notional_mc) / 1e6:.0f}mm mid-curve ({abs(rep.vega_mc) / 1e3:.0f}k/nv)"
)
print(
    f"         then sell {abs(rep.vega_13) / 1e3:.0f}k/nv 6m11y, "
    f"buy {abs(rep.vega_12) / 1e3:.0f}k/nv 6m1y"
)

# --------------------------------------------------------------------------------------
print("\n=== Sanity: variance additivity & grid ===")
# rebuild sigma_{T2,T3} from sigma_FVA and sigma_MC: (T2-T0)s23^2 = (T2-T1)sF^2 + (T1-T0)sMC^2
lhs = res.t2 * res.sig_23**2
rhs = (res.t2 - res.t1) * res.sig_fva**2 + res.t1 * res.sig_mc**2
check("variance additivity", lhs, rhs, 1e-6)

# rho = 1 must give the cheapest mid-curve -> the richest FVA vol
hi = fva.quote(curve, surface, 0.5, 1.0, 10.0, rho=1.0).sig_fva
lo = fva.quote(curve, surface, 0.5, 1.0, 10.0, rho=0.9).sig_fva
ok &= hi > lo
print(f"  [{'OK ' if hi > lo else 'FAIL'}] sigma_FVA(rho=1)={hi:.1f} > sigma_FVA(rho=0.9)={lo:.1f}")

g = fva.grid(curve, surface, strike=0.5, rho=1.0)
n_ok = int(np.isfinite(g["sig_fva"]).sum())
print(f"  grid 6m strike: {g['sig_fva'].shape[0]}x{g['sig_fva'].shape[1]}, {n_ok} finite cells")
ok &= n_ok > 0

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
raise SystemExit(0 if ok else 1)
