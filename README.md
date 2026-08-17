# FVA — Forward Volatility Agreement

Implementation of *The Convexity Maven — "The FVA: Forward Volatility Agreement"*
(H. Bassman / D. Mohney, Credit Suisse US Rates Trading, 27 Aug 2013), plus a Streamlit
front-end.

```
streamlit run app.py     # dashboard
python test_fva.py       # reproduces the paper's worked example
```

## Timeline & convention

```
T0 ------------ T1 ------------ T2 --------------------- T3
today       strike date     expiry date            swap maturity
```

The FVA is a straddle **struck** at `T1` at the then-ATM forward, **expiring** at `T2`
into a swap maturing at `T3`. A 6m1y10y FVA → `T1 = 0.5`, `T2 = 1.5`, `T3 = 11.5`.
Everything is normal (bp) vol.

## What is loaded, and once only

The **ATM normal vol matrix** (expiry × tenor) is the single market vol input. Every vol
used downstream is interpolated off it:

| symbol | what | 6m1y10y example |
|---|---|---|
| `sigma_{T2,T3}` | vanilla, expiry `T2`, tenor `T3-T2` | 18m10y |
| `sigma_{T1,T3}` | mid-curve long leg, expiry `T1`, tenor `T3-T1` | 6m11y |
| `sigma_{T1,T2}` | mid-curve short leg, expiry `T1`, tenor `T2-T1` | 6m1y |
| `sigma_spot` | same expiry/tenor seen spot, for the forward discount | 1y10y |

BPV (= DV01, **no notional**: `sum(tau_i · DF(t_i))`, the paper's `DV01_18m10y = 8.6`)
comes from the curve, and the mid-curve **correlation is an input, default 1**.

## The maths (paper → code)

| paper | code |
|---|---|
| `sigma_MC` from the DV01-weighted spread option formula (p.6) | `fva.midcurve_vol` |
| `(T2-T1) s_FVA² = (T2-T0) s_{2,3}² − (T1-T0) s_MC²` (p.5) | `fva.fva_vol` |
| `P_FVA = sqrt(2/pi)·sqrt(T2-T1)·s_FVA·DV01_{T2,T3}·(1+r/2)^{2(T1-T0)}` (p.6) | `fva.price_fva` |
| vega ratios `d s_FVA/d s_{2,3}`, `d s_FVA/d s_MC` (p.7) | `FVAResult.d_fva_d23`, `.d_fva_dmc` |
| `d s_MC/d s_{T1,T3}`, `d s_MC/d s_{T1,T2}` (p.7-8) | `FVAResult.d_mc_d13`, `.d_mc_d12` |
| the whole hedge book (pp.3-5, 8) | `fva.replicate` |

Two deliberate departures from the printed text:

- **`d sigma_MC / d sigma` denominator.** The paper prints `DV01²_{T2,T3}` in a place
  where the algebra gives `DV01_{T2,T3}` (a typo — the printed version is not
  dimensionally consistent). The code uses the correct form; with the paper's own inputs
  it returns 1.1163, which is exactly the ratio the paper then applies (60k × 1.1163 ≈ 66k
  of 6m11y).
- **Rounding.** The paper rounds `d s_FVA/d s_MC` 0.558 → 0.6 and carries a round -60k/nv
  mid-curve leg. The library keeps the unrounded chain: from 100k/nv of FVA it returns
  -55.8k mid-curve, -62.3k 6m11y, +6.5k 6m1y. From the paper's own -60k it returns 67.0k
  and 7.0k, matching their 66k / 6k.

`test_fva.py` checks both paths. Everything else lands on the paper: `sigma_MC` 106.2 (107),
`sigma_FVA` 95.2 (95), ratio 1.560 (1.6), vegas 8.40bp / 4.85bp (8.4 / 4.9), notionals
145.7 / 185.6 / 115.0mm (146 / 186 / 116).

A useful cross-check falls out of the algebra: the two straddles replicating the mid-curve
always carry the **same notional, opposite signs** (`|n_6m11y| = |n_6m1y|`), which is the
"clients trade notionally neutral" remark on p.9.

## Feeding your own BPVs

`fva.quote(..., bpv=<source>)` takes any callable `(start, end) -> BPV`. The three calls map
1:1 onto your `get_legs` decomposition:

```python
bpv(T2, T3)  ->  18M10Y   forward swap          # forward_term=18M, swap_tenor=10Y
bpv(T1, T3)  ->  6M11Y    mid-curve long leg    # forward_term=6M,  swap_tenor=11Y
bpv(T1, T2)  ->  6M1Y     mid-curve short leg   # forward_term=6M,  swap_tenor=1Y
```

`bpv_sources.py` ships three implementations:

- `CurveBpv` — the built-in curve (default).
- `TableBpv` — BPVs computed elsewhere, as CSV (`forward_term, swap_tenor, bpv`). The
  **BPV / DV01** tab exports the exact list of legs a given strike date needs, so you can
  fill the `bpv` column from your own calculator and upload it back (sidebar → BPV source →
  Table). Anything missing falls back to the curve and is reported.
- `LegsCalcBpv` — live adapter for `rdws...SwapLegsValueCalculator`:

```python
from rdws.utils.riskmetrics.SwapLegsValueCalculator import SwapLegsValueCalculator
from bpv_sources import LegsCalcBpv

bpv = LegsCalcBpv(SwapLegsValueCalculator, calculator_kwargs, extract="bpv")
res = fva.quote(curve, surface, strike=0.5, expiry=1.0, tenor=10.0, rho=1.0, bpv=bpv)
```

`extract` names the BPV field on whatever `forward_swap_legs_value` returns — left as `None`
it tries the usual names and raises with the list of available fields. Calls are cached, so
a full grid hits each leg once. The curve is still used for discounting the premium
(`(1+r/2)^{2T1} = 1/DF(T1)`) and for the forward rates shown in the trade blotter.

## App tabs

1. **FVA grid** — pick the strike date (sidebar, default 6m); the grid is expiry × tenor on
   the axes of the loaded matrix. Cell (1y, 10y) at strike 6m = the paper's 6m1y10y. Views:
   `sigma_FVA`, discount vs spot (the "100nv spot, 95nv six months forward" view), `sigma_MC`,
   the vanilla leg, price, vega, vega ratio.
2. **Trade detail** — one trade: legs, BPVs, forward rates, sensitivities, the full
   replication book for a target vega, prices, and a `rho` sensitivity ladder.
3. **BPV / DV01** — the BPV grids and weights that feed the mid-curve formula, plus the
   export of legs to compute externally.
4. **Inputs** — edit the vol matrix and the curve, see DF / zero / par / BPV.
5. **Paper check** — the worked example side by side with the paper's printed numbers.

Cells where `(T2-T0)·s_{2,3}² − (T1-T0)·s_MC² <= 0` have no forward vol (the surface and
`rho` imply negative forward variance); they come back as NaN and the app warns.

## Files

| file | |
|---|---|
| `fva.py` | curve/BPV, vol surface, mid-curve & FVA vols, greeks, replication, grids |
| `bpv_sources.py` | pluggable BPV sources (curve / table / SwapLegsValueCalculator) |
| `app.py` | Streamlit dashboard |
| `test_fva.py` | paper reproduction + sanity checks |
| `data/atm_vol.csv` | default ATM normal vol matrix (bp) |
| `data/curve.csv` | default par swap curve |
