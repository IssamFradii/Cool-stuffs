# The Relative Valuation of Caps and Swaptions — full pipeline

A complete, runnable implementation of

> Longstaff, F. A., P. Santa-Clara and E. S. Schwartz (2000/2001),
> *"The Relative Valuation of Caps and Swaptions: Theory and Empirical Evidence"*
> (`4-00.pdf` in this folder; published in the *Journal of Finance*, 2001).

Everything in the paper is here: the string market model, the implied-covariance
calibration, all of Tables 5–10 and Figures 5–7, plus the trading strategy the
paper gestures at in its conclusion but never runs.

It runs today with **no data**, on a synthetic sample calibrated to the paper's
own summary tables. Drop in real Bloomberg CSVs and nothing else changes.

```bash
python run_pipeline.py            # full run, ~2 minutes
python run_pipeline.py --quick    # fast smoke test
python tests/test_pipeline.py     # 50 tests, no pytest needed
```

Only dependencies are **numpy** and **pandas**. (Deliberately: the normal
distribution, the cubic spline and the Levenberg–Marquardt solver are all
implemented in `lss/mathutil.py` and unit-tested against scipy.)

---

## 1. What the paper actually says

Caps and swaptions are traded as separate products with separate models, but
no-arbitrage links them:

* A **cap** is a *portfolio of options* — one option on each individual forward
  Libor rate.
* A **swaption** is an *option on a portfolio* — one option on the forward swap
  rate, which is very nearly a linear basket of those same forwards.

Merton (1973): an option on a portfolio is worth no more than a portfolio of
options, with equality only under perfect correlation. So **the correlation
structure of forward rates is exactly what determines cap prices relative to
swaption prices**. Pin down that correlation matrix from one market and the
other market's prices are no longer free — they are implied.

The paper does this in four moves:

1. Model the 19 six-month Libor forwards as a *string* — each point on the curve
   its own Brownian motion, correlated across points, summarised by a single
   19×19 covariance matrix `Σ`.
2. Each week, **invert the swaptions market for `Σ`** — the matrix analogue of
   backing out an implied volatility.
3. Read the **main diagonal of `Σ`** off, which uniquely determines every caplet
   volatility and hence every cap price.
4. Compare those implied cap prices with quoted cap prices. The gap is **the
   wedge**.

Headline findings: four implied factors (not the usual two or three); implied
correlations lower than historical; swaptions well fitted (~3% RMSE) except for
a ten-week blow-up during the 1998 LTCM crisis; and cap prices that sit near
the swaption-implied values *most* of the time but dislocate periodically and
substantially.

---

## 2. The model

### Dynamics

Each forward is lognormal with correlated shocks (eq. 8):

```
dF_i = α_i F_i dt + σ_i F_i dZ_i,        Cov(dF_i/F_i, dF_j/F_j) = Σ_ij dt
```

The awkward drifts `α_i` are avoided entirely by working in **discount bond
prices**, which are traded assets and so drift at the short rate (eq. 11):

```
dD = r D dt + J⁻¹ σ F dZ
```

where `J = ∂F/∂D` is bidiagonal. Implemented in `lss/smm.py`.

### Identification — the key trick

A general N-factor model needs `N²` parameters but there are only `N(N+1)/2`
distinct covariances, so it is not identified. The paper fixes the
**eigenvectors** from the ex-ante historical correlation matrix and implies only
the **eigenvalues**:

```
H = U Λ U'     (historical, 1992–97, Table 2 — shipped in lss/paper_data.py)
Σ = U Ψ U'     (implied, Ψ diagonal ≥ 0 — solved for each week)
```

Economically: the market prices swaptions off the same factors that drive the
curve historically, but with its own forward-looking view of each factor's
variance. A rank-N model is nested by solving for the first N eigenvalues and
zeroing the rest.

### Time homogeneity and roll-down

`Σ_ij` depends only on *distance to becoming spot*. After six months the first
forward becomes the spot rate; the surviving 18 use the leading 18×18 block of
`Σ`, and the Jacobian drops its first row and column. This single indexing
convention drives both swaptions and caps.

### Caps come free

Because `Σ` is time-homogeneous, the variance of the caplet resetting at
`τ = i·δ` is the **running average of the first `i` diagonal elements of `Σ`**.
No simulation needed — lognormal forwards make Black exact for caplets. This is
the paper's central restriction, in `lss/caps.py`.

---

## 3. Three details where the paper as printed is wrong or silent

These matter, and getting them wrong silently breaks everything downstream.

**(a) The Jacobian in equation (11) is missing the accrual factor.** The printed
matrix has entries `−D(.50)/D²(1.00)`, but differentiating eq. (10),
`F_i = (1/δ)[D(T_i)/D(T_i+δ) − 1]`, gives those entries times `1/δ`. The printed
matrix is `δ·J_true`. Used literally it scales every shock by `δ = ½` and halves
all model volatilities. We use the true Jacobian —
`test_jacobian_scaling_reproduces_sigma` verifies the realised variance of
`dF/F` over one step equals `Σ_ii·δ`, and
`test_rank_one_simulation_matches_black_caplet` confirms the simulation
reproduces Black to within 1%.

**(b) Equation (1) is typeset with two artifacts.** It prints
`d − √(σ²τ/2)` where standard Black requires `d₁ − σ√τ`, and the same spurious
root in the numerator of `d`. Read literally it does not even have the right
zero-volatility limit. We implement standard Black and cross-check against
eq. (7), which *is* printed correctly.

**(c) The order of drift and diffusion in the discrete step.** Over a six-month
Euler step these interact, and the paper does not say which comes first. Adding
the shock *after* dividing by `D(t,t+δ)` scales every shock by that factor and
came in ~6.5% low on variance in testing. Adding it *before* is exact to first
order and still leaves the deflated bonds exact martingales. See the comment in
`lss/smm.py`.

Also worth flagging: Table 3's "Maximum" column in the source PDF is corrupted
for every row but the first (it repeats the Minimum column), so it is not
transcribed.

---

## 4. Layout

| File | What it does |
|---|---|
| `lss/mathutil.py` | numpy-only normal CDF/PDF/PPF, natural cubic spline, bounded Levenberg–Marquardt, χ² tail, AR(1)-adjusted t-stats |
| `lss/conventions.py` | the semiannual grid, the 34 swaption structures, cap maturities |
| `lss/paper_data.py` | Tables 1–10 transcribed, incl. the Table 2 correlation matrix (a real model input) |
| `lss/curve.py` | Libor+swaps → spline the par curve → bootstrap → 19 forwards |
| `lss/instruments.py` | Black caplets/caps/swaptions (eqs 1, 6, 7), swap values (eqs 4, 5, 12) |
| `lss/historical.py` | `H = UΛU'`, the fixed factor basis |
| `lss/smm.py` | the string market model simulation (eq. 11) |
| `lss/approx.py` | the fast closed-form approximation of footnote 18 |
| `lss/calibrate.py` | implied-eigenvalue solver |
| `lss/caps.py` | cap prices from the diagonal of `Σ`; the Merton bound |
| `lss/empirics.py` | Tables 5–10, Figures 6–7 |
| `lss/strategy.py` | the wedge trading strategy |
| `lss/synthetic.py` | paper-calibrated synthetic data |
| `lss/data.py` | real-CSV loaders |

### Speed

The paper's footnote 18 describes accelerating calibration by approximating the
forward swap rate as a linear basket of forwards, giving a closed-form swaption
price, then bias-correcting against the simulation. That is implemented
(`lss/approx.py`): the approximation is within **0.6% RMS** of a 20,000-path
simulation and ~400× faster. Calibration runs ~0.1 s/week warm-started; the full
128-week sample takes about 18 seconds, against the paper's "less than 10
seconds" on a 750 MHz Pentium III.

---

## 5. Results reproduced

Run on synthetic data calibrated to the paper's tables.

All figures below use 8,000 paths — see §9(a) on why the paper's 2,000 are not
enough for these particular numbers.

**Table 10 — the Merton no-arbitrage bound** (one-factor cap errors, %). This is
the strongest validation, because it is a pure no-arbitrage restriction with
nothing tuned to it:

| Maturity | 2y | 3y | 4y | 5y | 7y | 10y | Overall |
|---|---|---|---|---|---|---|---|
| **This code** | −26.55 | −23.91 | −20.77 | −16.95 | −10.10 | −5.03 | **−17.22** |
| **Paper** | −25.29 | −23.40 | −20.29 | −17.50 | −11.32 | −5.88 | **−17.28** |

**Table 9 — four-factor cap errors** (%, the wedge):

| Maturity | 2y | 3y | 4y | 5y | 7y | 10y | Overall |
|---|---|---|---|---|---|---|---|
| **This code** | 23.64 | 12.57 | 7.66 | 6.58 | 7.50 | 8.47 | **11.07** |
| **Paper** | 23.33 | 11.78 | 7.67 | 5.67 | 6.02 | 7.86 | **10.39** |

Also reproduced: four significant factors with the expected
level/slope/curvature/short-end shapes; implied correlations systematically
**below** historical (~95% of off-diagonals negative, mean −0.10) with the
first-vs-second forward the lone positive exception, exactly as in Table 6;
swaption RMSE around 3–4% outside the crisis, spiking during it.

Swaption RMSE: median 3.63% (paper 3.10%), rising to 8.19% during the crisis
window with a maximum of 13.9% — the Figure 7 shape.

**One honest gap.** The paper's Table 9 *medians* are near zero (overall +0.86)
while its *means* are strongly positive — a very skewed, bimodal error
distribution. The synthetic generator reproduces the means and the skew's
direction but not its full extremity (overall median +6.95 against +0.86). That
is a limitation of the fabricated data, not of the model code: the distributional
shape of real dislocations is not recoverable from six published summary
statistics. It does not affect the strategy, which trades a causally demeaned
z-score and so is invariant to the level.

---

## 6. The strategy

The paper's closing line:

> "dynamic trading strategies exploiting inconsistencies in the relative
> valuation of caps and swaptions may be profitable."

**Signal.** `spread_t(T) = model_cap_vol − market_cap_vol`, in vol points.
Positive ⇒ the cap is cheap versus what swaptions imply.

**Trade.** Standardise causally (rolling 52-week, no lookahead), enter beyond
`|z| ≥ 1.25`, exit at zero. Long the cap, short a vega-matched swaption basket
(capped at 3 instruments), DV01-hedged with a par swap. Costs at 1 Black vol
point per leg per trade, per the paper's footnote 23.

### What the backtest says

Two separate questions, with two different answers.

**Does the signal carry information about the wedge? Yes.** The recovered
spread correlates **+0.25 to +0.58** (mean +0.39) with the dislocation the
generator actually injected. The calibration does find what was put there.

**Is the backtest P&L evidence of that? No — and this is the important part.**

Running the identical strategy on synthetic data generated with **no wedge at
all** (`--no-wedge`) is the control, and it performs just as well:

| | gross | net @ 1.0 vp | net t-stat | round trips |
|---|---|---|---|---|
| **With wedge injected** | 0.113 | 0.046 | 0.25 | 10 |
| **Null: no wedge at all** | **0.124** | **0.064** | **0.36** | 9 |

The null is not weaker. It is marginally *stronger*. And the spread's standard
deviation is nearly the same in both (3.57 vs 3.69 vol points at 2y), which
explains why: the injected wedge contributes only a small share of the spread's
variance. The rest is estimation noise — idiosyncratic swaption quotes
perturbing `Ψ`, plus Monte-Carlo noise — and that noise is *persistent and
mean-reverting*, because it comes from a persistent calibration.

A mean-reversion rule applied to a noisily estimated signal will therefore show
positive in-sample gross P&L whether or not there is anything real to trade.
Without the null run, the +0.113 looks like a result. With it, it is a control
failure.

The cost frontier is consistent with that reading — nothing is significant
anywhere:

| entry z | gross | net @ 1.0 vp | net t-stat | round trips |
|---|---|---|---|---|
| 0.75 | 0.158 | −0.102 | −0.54 | 39 |
| 1.00 | 0.160 | −0.007 | −0.04 | 25 |
| 1.25 | 0.113 | 0.046 | 0.25 | 10 |
| 1.50 | 0.120 | 0.086 | 0.48 | 5 |
| 2.00 | 0.139 | 0.106 | 0.60 | 5 |

Every net t-statistic is below 1. The cells that look best rest on five round
trips. And the bid-offer binds hard: break-even is about 1.7 vol points against
the ~1 quoted, so lowering the threshold to buy statistical power destroys the
P&L on costs.

**Conclusion.** On this data the pipeline demonstrably recovers the wedge
(correlation with truth), but the trading rule does not convert that into
demonstrable P&L — the noise floor is of the same order as the signal. This
matches the paper's own caution that it works with *quotations*, not
transactions, and so stops short of claiming arbitrage.

**Always run the null.** `python run_pipeline.py --no-wedge` takes the same time
as the main run and is the difference between a result and an artifact.

## 7. Using real data

Put three CSVs in a directory and point at it:

```bash
python run_pipeline.py --data-dir ./mydata
```

* `rates.csv` — `date, libor_6m, libor_1y, swap_2y, swap_3y, swap_4y, swap_5y, swap_7y, swap_10y`
* `swaption_vols.csv` — `date`, then `0.5x1, 1x1, 2x1, ... 3x7` (all 34)
* `cap_vols.csv` — `date, cap_2y, cap_3y, cap_4y, cap_5y, cap_7y, cap_10y`

Percent or decimals both work (auto-detected; mixed units are rejected rather
than silently mangled). `python run_pipeline.py --save-data` writes the
synthetic set out in exactly this layout as a template.

Note the forward curve is an *interpolation choice*, not data: only eight
quotes are observed and the 19 forwards are splined from them. So saving and
reloading reproduces volatilities exactly but moves synthetic forwards by tens
of basis points, because an arbitrary 19-point curve is not recoverable from
eight quotes. Real curves are spline-consistent by construction; the synthetic
generator's are not.

Two things to decide with real data that the synthetic path chooses for you:

1. **The factor basis.** The default is the paper's Table 2 matrix (1992–97).
   For a different sample, estimate your own via
   `historical.spectral_decomposition(historical.correlation_from_forwards(...))`
   over a genuinely *ex-ante* window — using in-sample data here would leak.
2. **Cap tenor.** The paper assumes caps on six-month Libor to match the
   swaption grid, noting real 3M-Libor cap vols run an eighth to a quarter vol
   point higher. Adjust the quotes if your caps are on 3M.

---

## 8. Testing

```bash
python tests/test_pipeline.py      # or: python -m pytest tests -q
```

50 tests. The ones that actually pin the model down:

* `test_rank_one_simulation_matches_black_caplet` — with a rank-one `Σ` every
  forward is exactly lognormal, so the simulation **must** reproduce Black.
  Matches to <1%.
* `test_simulation_is_a_martingale` — numeraire-deflated bonds have expectation
  equal to today's price; catches any drift error.
* `test_jacobian_scaling_reproduces_sigma` — realised `Var(dF/F)` equals
  `Σ_ii·δ`; this is what caught the missing `1/δ` in eq. (11).
* `test_calibration_recovers_a_known_sigma` — generate prices from a known `Ψ`,
  recover it. The identification test.
* `test_rank_one_cap_price_is_a_merton_lower_bound` — the no-arbitrage bound.
* `test_causal_zscore_uses_no_future_information` — the z-score at week *t* is
  unchanged by anything after week *t*. The lookahead-leak guard.
* `test_signal_recovers_the_injected_wedge` — end-to-end: does the calibration
  find a dislocation that was really put there?
* `test_null_data_still_produces_a_spread` — the control. With no dislocation
  injected the recovered spread does *not* collapse: it keeps most of its
  variance and its persistence. That is why a positive backtest P&L is not by
  itself evidence, and why `--no-wedge` must always be run.
* `test_unit_economics_is_config_independent` and
  `test_costs_scale_linearly_with_bid_offer` — guard the repricing/rules split
  that makes the sensitivity sweep cheap.

---

## 9. Two things the paper doesn't mention

### (a) 2,000 Monte-Carlo paths are not enough for the cap wedge

The paper simulates 2,000 paths. That is ample for the swaption fit but **not**
for the cap errors of Table 9, which are the paper's headline result.

Cap prices are an *extrapolative* use of the fitted `Σ`. The swaption
cross-section constrains `Σ` through elasticity-weighted quadratic forms —
averages over the whole matrix — whereas caps depend on its **main diagonal
alone**, which is pinned down only indirectly. Calibration noise therefore
propagates into the cap errors amplified, and since noise in `Ψ` tends to
*raise* implied caplet vols, the bias is one-sided.

Table 9 overall mean on the same synthetic sample, varying only the simulation:

| paths | seed | overall mean | 2y mean | swaption RMSE |
|---|---|---|---|---|
| 1,000 | 99 | 10.63 | 21.85 | 3.89 |
| 2,000 | 99 | **13.31** | 27.69 | 4.18 |
| 2,000 | 12345 | **15.01** | 31.61 | 3.47 |
| 4,000 | 12345 | 9.85 | 22.21 | 3.68 |
| 8,000 | 7 | 10.72 | 22.49 | 3.60 |
| 8,000 | 99 | 10.13 | 21.18 | 3.46 |

At 2,000 paths two seeds give 13.3 and 15.0 — noisy *and* inflated by about a
third. By 8,000 the estimate has converged (10.7 vs 10.1 across seeds).

The trap is that the **swaption RMSE barely moves** (3.5–4.2% throughout), so
the calibration looks perfectly healthy while the quantity of interest is off.
`run_pipeline.py` therefore defaults to 8,000 paths; pass `--paths 2000` to
reproduce the paper's own setting.

### (b) Ψ is not always well identified

Worth knowing before reading Figure 6 too literally.

When the swaption data is genuinely generated by a rank-4 model, the
calibration recovers all four eigenvalues tightly — within 0.02
(`test_calibration_recovers_a_known_sigma`). But when the market surface is
*not* spanned by the model — which is the real-world case, and is why there are
pricing errors at all — the objective can have several near-equivalent minima.

Fitting rank 3 to the paper's own Table 3 mean volatilities, these two
eigenvalue vectors both give an RMSE of ~5.1%:

```
Ψ = [0.288, 0.153, 0.045]
Ψ = [0.316, 0.224, 0.000]
```

despite differing by up to 21% on individual swaption prices. The *fit* is
pinned down; the decomposition into factor variances is not, along one flat
direction.

This matters for interpretation. The paper reads the fitted eigenvalues as
implied factor variances and reads economic content off their time series
(Figure 6: "the volatility of this factor suddenly increased by a factor of
nearly ten"). Some of that week-to-week movement may be the optimiser sliding
along a flat direction rather than the market revising its view. Two mitigations
are implemented:

* **Rank cascade** (`cascade=True`): each rank starts from the solution of the
  one below, which guarantees the fit improves monotonically in the number of
  factors. Without it, a cold rank-4 start can land *worse* than rank 3 — which
  would corrupt the Table 5 likelihood ratio tests, since those compare sums of
  squared errors across ranks.
* **Warm starting** across weeks, which keeps the series on the same branch
  rather than hopping between equivalent optima.

A related trap, found and fixed here: parameterising `Ψ = θ²` to enforce
non-negativity has a vanishing derivative at the origin, so a factor that
collapses to zero can never recover, and with warm starts that stuck state
propagates for weeks. The eigenvalues are optimised directly under a lower
bound of zero instead. This mattered a lot — it cut the worst weekly RMSE from
16.2% to 7.3%.

---

## 10. Known limitations

* **Caplet/swaption expiries sit on the semiannual grid.** The paper's idealised
  setup. Real quote dates need date-rolling and day-count conventions.
* **ATM only.** No smile — the paper's data set is ATMF, and the Black
  quoting convention assumes it. Away-from-the-money would need a smile model.
* **Constant-maturity marking in the backtest.** Positions are re-marked at next
  week's quote for the same structure — the paper's own identifying assumption
  (footnote 29) — rather than aged by one week.
* **Synthetic median skew**, as described in §5.
* **The hedging comparison of Section 6** (Black vs SMM hedging R², 89.28% vs
  89.35% for receivers) is not implemented; it is a side analysis that does not
  feed the wedge.
