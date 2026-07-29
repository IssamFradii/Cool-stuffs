# Rates Desk Dashboard

A trading dashboard in pure Python — KPI tiles, a swap curve, P&L attribution,
two heatmaps (vol surface + rich/cheap wedge), a DV01 ladder, a live positions
blotter and an XVA table. One long scrollable page, light and dark mode.

Built with [NiceGUI](https://nicegui.io).

```bash
pip install -r requirements.txt
python app.py
# open http://localhost:8080
```

Flip the **Live** switch in the header to start the market ticking.

---

## Why NiceGUI and not Streamlit / Dash

| | Streamlit | Dash | **NiceGUI** |
|---|---|---|---|
| Re-runs your whole script on every click | yes | no | **no** |
| Live push updates (websocket) | bolt-on | callbacks | **built in** |
| Pro data grid | 3rd party | 3rd party | **AG Grid built in** |
| Charts | Plotly | Plotly | **ECharts + Plotly + Matplotlib** |
| Underneath | Tornado | Flask | **FastAPI** |

The re-run model is the real problem. Streamlit re-executes the file top to
bottom on every widget change, so a dashboard with a live feed and a big
blotter gets slow and flickery. NiceGUI keeps state on the server and pushes
only the diff over a websocket, which is what makes 2-second ticking on a
12-column grid feel smooth.

The two built-ins matter too: **AG Grid** is the table component actual
trading desks use (sorting, column filters, pinned columns, virtual scrolling,
in-place value flashes), and **Apache ECharts** does heatmaps, candlesticks
and large time series better than Plotly.

Runners-up worth knowing: **Reflex** (compiles Python to a React app — very
capable, but needs Node and a build step) and **FastHTML** (Python + HTMX,
elegant and very new, but you assemble the chart and table layers yourself).

---

## File map

| File | What lives there |
|---|---|
| `app.py` | Page layout and all state. Read this first — it reads top-to-bottom like the page. |
| `theme.py` | **Every colour in the app.** Light + dark tokens, page CSS, AG Grid variables. |
| `data.py` | Synthetic market data. **The only file you replace with real feeds.** |
| `charts.py` | ECharts option builders. Pure functions: `(palette, data) -> dict`. |
| `tables.py` | AG Grid option builders. |

Nothing except `theme.py` contains a hex code, and nothing except `data.py`
invents a number. That separation is what makes the thing easy to extend.

---

## Plugging in real data

Open `data.py` and replace the bodies of these — keep the shapes and the
column names and no other file needs to change:

| Property / method | Expected shape |
|---|---|
| `MarketState.curve` | `np.ndarray`, one par rate per `CURVE_TENORS` entry |
| `MarketState.vol_surface` | `np.ndarray` of shape `(len(EXPIRIES), len(TAILS))`, normal vol in bp |
| `MarketState.wedge` | same shape, rich/cheap z-scores (**+ = rich = sell**) |
| `MarketState.positions` | DataFrame: `instrument, ccy, book, notional, dv01, vega, mtm, day_pnl` |
| `MarketState.xva` | DataFrame: `counterparty, rating, csa, epe, cva, dva, fva, colva` |
| `MarketState.risk_ladder()` | DataFrame: `bucket, dv01, limit, vega` |
| `MarketState.pnl_history` | DataFrame: `date, Rates, Vol, Carry` |
| `MarketState.tick()` | mutate state in place; the UI re-reads it every 2s |

`wedge` is the natural hook for the `wedge_tool` output — feed its z-score
matrix straight in and the diverging heatmap becomes a live rich/cheap monitor.

---

## The visual rules this follows

These are not taste, they are the things that make a dashboard readable, and
they're worth keeping as you extend it:

- **Sequential = one hue, light → dark.** The vol surface is a blue ramp, never
  a rainbow. On dark mode the ramp is reversed so low values recede into the
  background instead of glowing.
- **Diverging = two opposite hues + a neutral grey middle.** The wedge heatmap
  is blue↔red through grey, forced symmetric so `+1.5` and `−1.5` are equally
  loud and zero reads as nothing.
- **Categorical colours are assigned in a fixed order and never cycled.** Series
  1/2/3 are always blue/orange/aqua, so "Rates is blue" survives a filter change.
- **Never a dual-axis chart.** Two measures on two y-scales invents a
  correlation that isn't in the data. Use two charts.
- **Colour is never the only cue.** Every P&L number carries an explicit `+`/`−`,
  every KPI delta carries an arrow icon, and every heatmap has a Table view with
  the exact numbers — colourblind-safe and copy-pasteable.
- **Direct-label selectively.** The 10Y point on the curve, the biggest DV01
  bucket, the leading P&L series. Never a number on every point.
- **Filters live in one row above everything they scope**, so you can always
  tell that two cards show the same slice of the book.

---

## Ideas to build next

- Swap `data.py` for a real curve bootstrapper and your swaption cube
- Click a wedge cell → drill into that expiry/tail's vol history
- A P&L explain waterfall (delta / gamma / vega / theta / residual)
- Limit breaches: highlight a DV01 bucket over limit and push a toast
- `ui.download` on each grid for a CSV export
- Auth via `nicegui.app.add_middleware` before anyone else sees it
