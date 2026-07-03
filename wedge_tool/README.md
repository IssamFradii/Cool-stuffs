# Cap/Swaption Wedge — Rich/Cheap Analysis per Bucket

A small Python tool that measures, for each expiry bucket, whether **swaptions
are trading rich or cheap relative to caps/floors** — the so-called
**cap/swaption wedge** — and suggests the corresponding relative-value trade.

Everything is based on **Brigo & Mercurio, *Interest Rate Models — Theory and
Practice* (2nd ed.)**:

- **Section 6.20** — *Connecting Caplet and S×1-Swaption Volatilities*:
  the formulas linking caplet vols to one-period swaption vols.
- **Section 7.9** — *Cascade Calibration and the cap market*: the same
  formulas applied to real market data (Table 7.13, February 1, 2002) to
  diagnose cap-vs-swaption misalignment.

---

## 1. The idea in plain words

Two markets trade volatility on (almost) the same underlying rates:

- The **cap market** trades caplets: options on **6-month forward rates**.
- The **swaption market** trades options on swap rates. The shortest one, the
  **S×1 swaption** (expiry `S` years, on a 1-year swap), is an option on an
  **annual forward rate**.

An annual forward rate `F(0; S, S+1y)` is just an algebraic combination of
the two 6-month forwards sitting inside it, `F1 = F(0; S, S+6m)` and
`F2 = F(0; S+6m, S+1y)` (book eq. **6.74**):

```
F = (F1 + F2)/2 + F1·F2/4
```

So the volatility of `F` (what the swaption market quotes) must be tied to
the volatilities of `F1` and `F2` (what the cap market quotes). Freezing the
weights at today's values, Brigo & Mercurio derive (eq. **6.76**):

```
v_swaption² ≈ u1²·v1² + u2²·v2² + 2·ρ·u1·u2·v1·v2

u1 = (F1/2 + F1·F2/4) / F        v1 = S-expiry caplet vol
u2 = (F2/2 + F1·F2/4) / F        v2 = (S+6m)-expiry caplet vol
ρ  = "infra-correlation" between F1 and F2 (close to 1 in reality)
```

That gives two mirror-image diagnostics **per expiry bucket S = 1y, 2y, …**:

1. **The wedge** — plug the caplet vols into (6.76) with ρ = 1 to get the
   *cap-implied* swaption vol, then compare with the market quote:

   `wedge = market swaption vol − cap-implied vol`

   - `wedge > 0` → the swaption is **RICH** versus caps (sell swaption vol,
     buy cap vol)
   - `wedge < 0` → the swaption is **CHEAP** versus caps (buy swaption vol,
     sell cap vol)

2. **The implied infra-correlation** — invert (6.76) and solve for ρ. A real
   correlation between adjacent 6-month rates should sit close to 1 (say in
   [0.7, 1.0]). If the market quotes imply ρ = 0.4, or ρ > 1, the two markets
   are inconsistent in that bucket — same signal, expressed as a correlation.

The tool computes both, classifies each bucket as **RICH / CHEAP / FAIR**, and
prints the vega-neutral trade (from (6.76) with ρ≈1, `v_swpt ≈ u1·v1 + u2·v2`,
so per unit of swaption vega the hedge is `u1` units of the first caplet's
vega and `u2` of the second's).

---

## 2. Project layout

```
wedge_tool/
├── app.py                     ← Streamlit dashboard (wedge monitor + heatmap + CCA)
├── run_analysis.py            ← command-line version of the bucket analysis
├── data/
│   ├── brigo_table_7_13.csv   ← S×1 bucket quotes (book Table 7.13, Feb-2002)
│   ├── annual_curve.csv       ← annual forwards + annualized caplet vols (Table 7.1)
│   └── swaption_matrix.csv    ← 10×10 swaption vol matrix (Table 7.4, May-2000)
├── wedge/
│   ├── annualize.py           ← formulas 6.74 / 6.76 and their inversion
│   ├── analysis.py            ← wedge per bucket, signals, suggested trades
│   ├── surface.py             ← cap-implied swaption surface → wedge heatmap
│   ├── cca.py                 ← Cascade Calibration Algorithm (7.4.1 / 7.4.2)
│   └── report.py              ← console table + chart
├── tests/
│   ├── test_book_numbers.py   ← reproduces the book's §7.9 numbers
│   └── test_cca.py            ← reproduces the book's Table 7.5 calibration
├── output/                    ← created on first run (CSV + PNG)
└── requirements.txt
```

## 3. How to run it

### The dashboard (recommended)

```powershell
cd "c:\Users\moham\Desktop\FVA AND interate modelling\wedge_tool"
pip install -r requirements.txt
streamlit run app.py
```

Your browser opens with three tabs:

1. **📊 S×1 wedge buckets** — the rich/cheap monitor: signal cards, the wedge
   bar chart (red = swaption rich vs caps, blue = cheap), implied ρ against
   its plausible band, and the trade table.
2. **🗺️ Wedge heatmap** — the wedge per **expiry × tenor**: cap-implied vols
   for every swaption (Rebonato's formula 6.67 fed with annualized caplet
   vols) minus market vols. Red cells = swaption rich vs the cap market,
   blue = cheap. Hover any cell for market/implied detail.
3. **⚙️ CCA calibration** — the book's Cascade Calibration (Algorithm
   7.4.1/7.4.2): exact, closed-form calibration of the LFM piecewise-constant
   vols to the whole swaption matrix. Negative σ entries (highlighted red)
   are the book's own diagnostic of misaligned/illiquid quotes.

Upload your own CSVs in the sidebar (all three fall back to the book's demo
data), and set ρ, the plausible band, and the correlation shape there too.

### The command line

```powershell
python run_analysis.py
```

You get a table like this (Feb-2002 book data):

```
  bucket  mkt swpt vol %  cap-implied %  wedge (vol pts)  implied rho signal
 1y x 1y           17.90          17.80            +0.10        1.022   FAIR
 2y x 1y           15.40          18.48            -3.08        0.388  CHEAP
 3y x 1y           14.30          16.28            -1.98        0.543  CHEAP
 ...
```

plus `output/wedge_report.png` with three panels (market vs cap-implied vol,
the wedge itself, and the implied ρ against its plausible band) and
`output/wedge_report.csv` with all columns.

Options:

```
python run_analysis.py --help
  --data       path to your own CSV (same columns as the sample)
  --rho        infra-correlation used for the benchmark vol   (default 1.0)
  --rho-low    lower edge of the plausible implied-ρ band     (default 0.70)
  --rho-high   upper edge of the plausible implied-ρ band     (default 1.00)
  --min-wedge  min |wedge| before flagging, absolute vol      (default 0.005)
  --out        output directory                               (default ./output)
```

### Using your own market data

Three CSVs feed the dashboard (each falls back to the book demo if not
uploaded). All rates/vols as decimals.

**1. S×1 bucket quotes** (tab 1) — one row per expiry bucket:

| column | meaning |
|---|---|
| `expiry` | swaption expiry S in years |
| `swaption_vol` | market Black vol of the S×1y swaption |
| `f1` | forward rate F(0; S, S+6m) |
| `f2` | forward rate F(0; S+6m, S+1y) |
| `caplet_vol_s` | Black vol of the S-expiry caplet |
| `caplet_vol_t` | Black vol of the (S+6m)-expiry caplet |

**2. Annual curve** (tabs 2–3) — one row per year 1..M:
`expiry, forward, caplet_vol` where `forward` = F(0; i, i+1) and
`caplet_vol` = the **annualized** caplet vol (semi-annual caplet vols
annualized via formula 6.76, or your own annual-tenor caplet strip). For a
10×10 matrix you need M ≥ 19 (max expiry + max tenor − 1).

**3. Swaption matrix** (tabs 2–3) — first column `expiry` (years), one
column per tenor (years), values = market Black vols. Missing quotes: leave
empty (heatmap skips them; CCA needs a full square matrix).

Note: caplet vols are *forward-forward* (spot) vols per caplet, not the flat
cap vols quoted for whole caps — strip caplet vols from cap quotes first
(book Section 6.4.3).

## 4. Checking it is right

The book publishes the outputs for both demo datasets, and the tests
reproduce them:

- **§7.9** — annualized caplet vols `0.178, 0.185, 0.163, …` and implied
  infra-correlations `1.022, 0.388, 0.543, …`: all twenty numbers match.
- **Table 7.5** — the full CCA calibration of the May-2000 10×10 matrix,
  including its five *negative* σ entries: reproduced to the book's printed
  precision, and the calibrated model reprices all 100 input swaption vols
  to ~1e-13 bp (the CCA is exact by construction).

```powershell
python tests\test_book_numbers.py
python tests\test_cca.py
```

## 5. Read before trading this (book, end of Section 7.9)

Brigo & Mercurio close with warnings that are part of the strategy:

- **Rebonato**: illiquidity, agency problems and VaR limits weaken the
  "quasi-arbitrageurs" who would keep the two markets consistent — cap and
  swaption vol surfaces **can stay non-congruent for a long time**. A wedge
  can persist or widen; this is relative value, not arbitrage.
- **Pelsser**: losses at several banks in the late 1990s are attributed to
  models that assumed a **fixed relation** between caplet and swaption vols.
- The ρ = 1 benchmark tends to slightly **overstate** the cap-implied vol
  (formula 6.76 overprices; see the a1/a2 tests in Section 6.20), which biases
  buckets toward looking CHEAP. Watch the *change* in the wedge over time and
  its cross-sectional z-score, not just its sign.
