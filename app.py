"""
FVA - Forward Volatility Agreement | Streamlit front-end.

    streamlit run app.py

Workflow: the ATM normal vol matrix is loaded ONCE (sidebar) and every vol used
downstream - sigma_{T2,T3}, sigma_{T1,T3}, sigma_{T1,T2} - is interpolated off that
single surface.  BPV (= DV01, no notional) of the forward swap and of the two
mid-curve legs comes from the curve.  Mid-curve correlation is an input, default 1.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import bpv_sources
import fva

HERE = Path(__file__).parent

st.set_page_config(page_title="FVA - Forward Volatility Agreement", layout="wide")

# --------------------------------------------------------------------------------------
# palette (light chart surface)
# --------------------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
# sequential blue, capped at step 500 so black cell labels stay >= 4:1 on every cell
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#2a78d6", "#256abf"]
DIVERGING = [[0.0, "#256abf"], [0.5, "#f0efec"], [1.0, "#e34948"]]


def heat(
    z: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    value_fmt: str = ".1f",
    unit: str = "",
    diverging: bool = False,
    strike_label: str = "",
    height: int = 460,
) -> go.Figure:
    """Expiry (rows) x tenor (cols) heatmap. Sequential for magnitude, diverging for signed."""
    zc = np.asarray(z, float)
    if diverging:
        lim = np.nanmax(np.abs(zc)) if np.isfinite(zc).any() else 1.0
        colorscale, zmid, zmin, zmax = DIVERGING, 0.0, -lim, lim
    else:
        colorscale, zmid, zmin, zmax = SEQ, None, None, None

    hover = (
        f"<b>{strike_label}%{{y}}%{{x}}</b><br>"
        if strike_label
        else "<b>%{y} x %{x}</b><br>"
    ) + f"expiry %{{y}} · tenor %{{x}}<br>value %{{z:{value_fmt}}}{unit}<extra></extra>"

    fig = go.Figure(
        go.Heatmap(
            z=zc,
            x=x_labels,
            y=y_labels,
            colorscale=colorscale,
            zmid=zmid,
            zmin=zmin,
            zmax=zmax,
            xgap=2,  # 2px surface gap between cells
            ygap=2,
            texttemplate=f"%{{z:{value_fmt}}}",
            textfont={"size": 11, "color": INK},
            hovertemplate=hover,
            colorbar={
                "outlinewidth": 0,
                "tickfont": {"color": INK_MUTED, "size": 11},
                "thickness": 12,
                "len": 0.85,
            },
        )
    )
    fig.update_layout(
        height=height,
        margin={"l": 76, "r": 8, "t": 56, "b": 8},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": 'system-ui, -apple-system, "Segoe UI", sans-serif', "color": INK},
        xaxis={
            "title": {"text": "Tenor (swap tail)", "font": {"color": INK_MUTED, "size": 12}, "standoff": 6},
            "side": "top",
            "type": "category",
            "tickfont": {"color": INK_MUTED, "size": 12},
            "showgrid": False,
            "ticks": "",
            "automargin": True,
        },
        yaxis={
            "title": {
                "text": "Expiry (after strike date)",
                "font": {"color": INK_MUTED, "size": 12},
                "standoff": 8,
            },
            "autorange": "reversed",
            "type": "category",
            "tickfont": {"color": INK_MUTED, "size": 12},
            "showgrid": False,
            "ticks": "",
            "automargin": True,
        },
    )
    return fig


def frame(z: np.ndarray, x_labels: list[str], y_labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame(z, index=pd.Index(y_labels, name="expiry \\ tenor"), columns=x_labels)


def show_grid(
    z: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    caption: str,
    value_fmt: str = ".1f",
    unit: str = "",
    diverging: bool = False,
    strike_label: str = "",
    key: str = "",
) -> None:
    st.caption(caption)
    st.plotly_chart(
        heat(z, x_labels, y_labels, value_fmt, unit, diverging, strike_label),
        width="stretch",
        theme=None,  # keep the palette below, not Streamlit's plotly template
        key=f"plot_{key}",
    )
    df = frame(z, x_labels, y_labels)
    with st.expander("Table view / export"):
        st.dataframe(df.style.format(f"{{:{value_fmt}}}"), width="stretch")
        st.download_button(
            "Download CSV",
            df.to_csv().encode(),
            file_name=f"fva_{key or 'grid'}.csv",
            mime="text/csv",
            key=f"dl_{key}",
        )


# --------------------------------------------------------------------------------------
# Sidebar - inputs loaded once
# --------------------------------------------------------------------------------------
st.sidebar.title("Inputs")

# ---- 1. the ATM vol matrix, loaded once -------------------------------------------
st.sidebar.subheader("1 · ATM normal vol matrix")
up = st.sidebar.file_uploader("CSV: rows = expiry, cols = tenor (bp)", type=["csv"])

if "vol_df" not in st.session_state:
    st.session_state.vol_df = pd.read_csv(HERE / "data" / "atm_vol.csv")
    st.session_state.vol_src = "data/atm_vol.csv"

if up is not None and st.session_state.get("vol_upload") != up.name:
    st.session_state.vol_df = pd.read_csv(up)
    st.session_state.vol_src = up.name
    st.session_state.vol_upload = up.name

st.sidebar.caption(f"loaded: `{st.session_state.vol_src}` (edit it in the Inputs tab)")

interp_mode = st.sidebar.selectbox(
    "Surface interpolation",
    ["vol", "variance"],
    help="Bilinear in vol, or linear in total variance (sigma^2 * T) along the expiry axis. "
    "Flat extrapolation outside the grid.",
)

# ---- 2. curve -> BPV / DV01 --------------------------------------------------------
st.sidebar.subheader("2 · Curve (for BPV = DV01)")
curve_mode = st.sidebar.radio("Curve input", ["Flat par rate", "Par swap curve"], horizontal=True)
freq = st.sidebar.selectbox("Fixed leg frequency", [1, 2, 4], index=1)

if curve_mode == "Flat par rate":
    flat = st.sidebar.number_input("Par rate (%)", value=2.25, step=0.05, format="%.3f")
    curve = fva.Curve.flat(flat / 100.0, freq=freq)
else:
    if "curve_df" not in st.session_state:
        st.session_state.curve_df = pd.read_csv(HERE / "data" / "curve.csv")
    cdf = st.session_state.curve_df
    mats = [fva.parse_period(x) for x in cdf.iloc[:, 0]]
    rates = [float(r) / 100.0 for r in cdf.iloc[:, 1]]
    curve = fva.Curve.from_par_swaps(mats, rates, freq=freq)

bpv_mode = st.sidebar.radio(
    "BPV source",
    ["Curve above", "Table (CSV)"],
    horizontal=True,
    help="BPV = DV01 without notional. Use 'Table' to feed BPVs computed elsewhere "
    "(e.g. SwapLegsValueCalculator): columns forward_term, swap_tenor, bpv.",
)
bpv_src = bpv_sources.CurveBpv(curve)
if bpv_mode == "Table (CSV)":
    bpv_file = st.sidebar.file_uploader("BPV table CSV", type=["csv"], key="bpv_csv")
    if bpv_file is not None:
        try:
            bpv_src = bpv_sources.TableBpv.from_frame(pd.read_csv(bpv_file), curve.bpv)
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"BPV table: {exc}")
    else:
        st.sidebar.caption("no table yet - using the curve meanwhile")

# ---- 3. correlation ---------------------------------------------------------------
st.sidebar.subheader("3 · Mid-curve correlation")
rho = st.sidebar.number_input(
    "rho (T1,T3) vs (T1,T2)",
    min_value=-1.0,
    max_value=1.0,
    value=1.0,
    step=0.01,
    format="%.2f",
    help="Correlation between the T1->T3 and T1->T2 rates in the DV01-weighted spread "
    "option formula. Default 1 (the paper's assumption); the market mid-curve vol "
    "usually trades above the rho=1 level.",
)

# ---- 4. strike date + target vega -------------------------------------------------
st.sidebar.subheader("4 · FVA strike date (T1)")

vol_df = st.session_state.vol_df
try:
    expiries = np.array([fva.parse_period(x) for x in vol_df.iloc[:, 0]], float)
    tenors = np.array([fva.parse_period(c) for c in vol_df.columns[1:]], float)
    vols = vol_df.iloc[:, 1:].to_numpy(float)
    surface = fva.VolSurface(expiries, tenors, vols, mode=interp_mode)
except Exception as exc:  # noqa: BLE001
    st.sidebar.error(f"Cannot read the vol matrix: {exc}")
    st.stop()

exp_labels = [fva.label_years(e) for e in expiries]
ten_labels = [fva.label_years(t) for t in tenors]

default_idx = exp_labels.index("6m") if "6m" in exp_labels else 0
strike_label = st.sidebar.selectbox("Strike date", exp_labels, index=default_idx)
custom = st.sidebar.text_input("or type one (e.g. 9m, 1.5y)", value="")
if custom.strip():
    try:
        strike = fva.parse_period(custom)
        strike_label = fva.label_years(strike)
    except ValueError:
        st.sidebar.error(f"Cannot parse '{custom}'")
        strike = fva.parse_period(strike_label)
else:
    strike = fva.parse_period(strike_label)

target_vega = st.sidebar.number_input(
    "Target FVA vega (\\$ per 1nv)", value=100_000, step=10_000, format="%d"
)

# --------------------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------------------
st.title("FVA · Forward Volatility Agreement")
st.caption(
    "The Convexity Maven, H. Bassman / D. Mohney, Credit Suisse US Rates Trading, 27-Aug-2013 · "
    "a straddle struck at T1 at the then-ATM forward, expiring T2 into a swap maturing T3."
)

grids = fva.grid(curve, surface, strike, rho, bpv=bpv_src)

missing = getattr(bpv_src, "missing", set())
if missing:
    st.warning(
        f"{len(missing)} forward swap(s) are not in the uploaded BPV table - the curve was "
        "used for those. Missing: "
        + ", ".join(f"{a}{b}" for a, b in sorted(missing)[:12])
        + (" ..." if len(missing) > 12 else "")
    )

tab_grid, tab_trade, tab_bpv, tab_inputs, tab_paper = st.tabs(
    ["FVA grid", "Trade detail", "BPV / DV01", "Inputs", "Paper check"]
)

# --------------------------------------------------------------------------------------
# FVA grid
# --------------------------------------------------------------------------------------
with tab_grid:
    st.subheader(f"Strike date {strike_label} · expiry × tenor")
    st.caption(
        f"Each cell = an FVA struck in {strike_label}, then expiring in <row> into a <col> swap. "
        f"Cell (1y, 10y) = the paper's {strike_label}1y10y. rho = {rho:.2f}."
    )

    view = st.radio(
        "View",
        [
            "sigma_FVA (bp)",
            "Discount vs spot (%)",
            "sigma_MC (bp)",
            "Vanilla sigma_{T2,T3} (bp)",
            "Price (bp, paid at strike date)",
            "Vega (bp per 1nv)",
            "d sigma_FVA / d sigma_{T2,T3}",
        ],
        horizontal=True,
    )

    spec = {
        "sigma_FVA (bp)": ("sig_fva", ".1f", "nv", False, "Forward volatility of the FVA."),
        "Discount vs spot (%)": (
            "discount",
            "+.1f",
            "%",
            True,
            "sigma_FVA vs today's spot vol of the SAME expiry x tenor - the paper's "
            "'spot 1y10y at 100nv, six months forward at 95nv, a 5% discount'. "
            "Blue = forward cheaper than spot.",
        ),
        "sigma_MC (bp)": (
            "sig_mc",
            ".1f",
            "nv",
            False,
            "Mid-curve vol from the BPV-weighted spread option formula (struck today, "
            "expiring T1, on the T2->T3 forward swap).",
        ),
        "Vanilla sigma_{T2,T3} (bp)": (
            "sig_23",
            ".1f",
            "nv",
            False,
            "The long vanilla leg read off the loaded surface (e.g. 18m10y).",
        ),
        "Price (bp, paid at strike date)": (
            "price_fva",
            ".1f",
            "bp",
            False,
            "sqrt(2/pi) sqrt(T2-T1) sigma_FVA BPV_{T2,T3} (1+r/2)^{2 T1}.",
        ),
        "Vega (bp per 1nv)": (
            "vega_fva",
            ".2f",
            "bp",
            False,
            "dP_FVA / d sigma_FVA, in bp of notional per 1bp of vol.",
        ),
        "d sigma_FVA / d sigma_{T2,T3}": (
            "d_fva_d23",
            ".2f",
            "",
            False,
            "Vega ratio to the vanilla hedge (1.6 in the paper's 6m1y10y example).",
        ),
    }
    key, vfmt, unit, div, caption = spec[view]
    show_grid(
        grids[key],
        ten_labels,
        exp_labels,
        caption,
        vfmt,
        unit,
        div,
        strike_label=strike_label,
        key=key,
    )

    bad = int(np.sum(~np.isfinite(grids["sig_fva"])))
    if bad:
        st.warning(
            f"{bad} cell(s) have no forward vol: (T2-T0)·sigma_23² - (T1-T0)·sigma_MC² <= 0, "
            "i.e. the loaded surface and rho imply a negative forward variance there."
        )

# --------------------------------------------------------------------------------------
# Trade detail
# --------------------------------------------------------------------------------------
with tab_trade:
    c1, c2 = st.columns(2)
    e_lab = c1.selectbox("Expiry after strike (T2-T1)", exp_labels, index=exp_labels.index("1y") if "1y" in exp_labels else 0)
    t_lab = c2.selectbox("Tenor (T3-T2)", ten_labels, index=ten_labels.index("10y") if "10y" in ten_labels else 0)
    e, t = fva.parse_period(e_lab), fva.parse_period(t_lab)

    r = fva.quote(curve, surface, strike, e, t, rho, bpv=bpv_src)
    rep = fva.replicate(r, float(target_vega))
    name = fva.trade_name(strike, e, t)

    st.subheader(f"{name} FVA")
    m = st.columns(5)
    m[0].metric("sigma_FVA", f"{r.sig_fva:.1f} nv")
    m[1].metric(
        f"vs spot {fva.label_years(e)}{fva.label_years(t)} {r.sig_spot:.1f}nv",
        f"{r.discount_vs_spot * 100:+.1f} %",
    )
    m[2].metric("sigma_MC", f"{r.sig_mc:.1f} nv")
    m[3].metric("Price @ strike date", f"{r.price_fva:.1f} bp")
    m[4].metric("BPV/DV01 T2→T3", f"{r.dv01_23:.2f}")

    st.markdown(
        f"**Timeline** — T0 = 0 · T1 = {r.t1:g}y (strike) · T2 = {r.t2:g}y (expiry) · "
        f"T3 = {r.t3:g}y (maturity)"
    )

    left, right = st.columns(2)

    with left:
        st.markdown("##### Legs read off the loaded ATM matrix")
        st.dataframe(
            pd.DataFrame(
                {
                    "leg": [
                        f"vanilla  {fva.label_years(r.t2)}{fva.label_years(r.t3 - r.t2)}",
                        f"mid-curve leg  {fva.label_years(r.t1)}{fva.label_years(r.t3 - r.t1)}",
                        f"mid-curve leg  {fva.label_years(r.t1)}{fva.label_years(r.t2 - r.t1)}",
                    ],
                    "role": ["sigma_{T2,T3}", "sigma_{T1,T3}", "sigma_{T1,T2}"],
                    "vol (nv)": [r.sig_23, r.sig_13, r.sig_12],
                    "BPV = DV01": [r.dv01_23, r.dv01_13, r.dv01_12],
                    "fwd rate (%)": [r.fwd_23 * 100, r.fwd_13 * 100, r.fwd_12 * 100],
                }
            ).style.format({"vol (nv)": "{:.2f}", "BPV = DV01": "{:.3f}", "fwd rate (%)": "{:.3f}"}),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            f"BPV-weighted replication of the forward swap: "
            f"pay {r.dv01_13 / r.dv01_23:.3f} × the {fva.label_years(r.t3 - r.t1)} rate, "
            f"receive {r.dv01_12 / r.dv01_23:.3f} × the {fva.label_years(r.t2 - r.t1)} rate "
            f"(both {fva.label_years(r.t1)} forward)."
        )

        st.markdown("##### Sensitivities")
        st.dataframe(
            pd.DataFrame(
                {
                    "sensitivity": [
                        "d sigma_FVA / d sigma_{T2,T3}",
                        "d sigma_FVA / d sigma_MC",
                        "d sigma_MC / d sigma_{T1,T3}",
                        "d sigma_MC / d sigma_{T1,T2}",
                    ],
                    "value": [r.d_fva_d23, r.d_fva_dmc, r.d_mc_d13, r.d_mc_d12],
                }
            ).style.format({"value": "{:+.3f}"}),
            hide_index=True,
            width="stretch",
        )

    with right:
        st.markdown(f"##### Replication for {target_vega / 1e3:,.0f}k/nv of FVA vega")
        book = pd.DataFrame(
            {
                "instrument": [
                    f"FVA {name}",
                    f"vanilla straddle {fva.label_years(r.t2)}{fva.label_years(r.t3 - r.t2)}",
                    f"mid-curve {name}",
                    f"  └ straddle {fva.label_years(r.t1)}{fva.label_years(r.t3 - r.t1)}",
                    f"  └ straddle {fva.label_years(r.t1)}{fva.label_years(r.t2 - r.t1)}",
                ],
                "side": [
                    "buy",
                    "buy" if rep.vega_van23 > 0 else "sell",
                    "sell" if rep.vega_mc < 0 else "buy",
                    "sell" if rep.vega_13 < 0 else "buy",
                    "sell" if rep.vega_12 < 0 else "buy",
                ],
                "vega (k/nv)": [
                    target_vega / 1e3,
                    abs(rep.vega_van23) / 1e3,
                    abs(rep.vega_mc) / 1e3,
                    abs(rep.vega_13) / 1e3,
                    abs(rep.vega_12) / 1e3,
                ],
                "notional (mm)": [
                    rep.notional_fva / 1e6,
                    abs(rep.notional_van23) / 1e6,
                    abs(rep.notional_mc) / 1e6,
                    abs(rep.notional_13) / 1e6,
                    abs(rep.notional_12) / 1e6,
                ],
            }
        )
        st.dataframe(
            book.style.format({"vega (k/nv)": "{:,.1f}", "notional (mm)": "{:,.0f}"}),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Rows 2-3 are the FVA replication (long the vanilla, short the mid-curve); "
            "rows 4-5 replicate that mid-curve with two straddles expiring at the strike date. "
            "Notionals use each straddle's own expiry and BPV."
        )

        st.markdown("##### Prices (bp of notional)")
        st.dataframe(
            pd.DataFrame(
                {
                    "instrument": ["FVA (paid at T1)", "FVA (PV today)", "vanilla T2×(T3-T2)", "mid-curve"],
                    "price": [r.price_fva, r.price_fva_pv, r.price_van23, r.price_mc],
                    "vega": [r.vega_fva, r.vega_fva / r.compounding, r.vega_van23, r.vega_mc],
                }
            ).style.format({"price": "{:.2f}", "vega": "{:.2f}"}),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            f"r(0,{fva.label_years(r.t1)}) = {r.r_t1 * 100:.3f}% s.a. → premium compounding "
            f"(1+r/2)^(2·T1) = {r.compounding:.4f} = 1/DF(T1)."
        )

    st.markdown("##### Sensitivity of sigma_FVA to the correlation input")
    rhos = np.round(np.arange(0.80, 1.0001, 0.02), 4)
    sens = pd.DataFrame(
        {
            "rho": rhos,
            "sigma_MC (nv)": [
                fva.quote(curve, surface, strike, e, t, x, bpv=bpv_src).sig_mc for x in rhos
            ],
            "sigma_FVA (nv)": [
                fva.quote(curve, surface, strike, e, t, x, bpv=bpv_src).sig_fva for x in rhos
            ],
        }
    )
    st.dataframe(
        sens.style.format({"rho": "{:.2f}", "sigma_MC (nv)": "{:.2f}", "sigma_FVA (nv)": "{:.2f}"}),
        hide_index=True,
        width="stretch",
    )

# --------------------------------------------------------------------------------------
# BPV / DV01
# --------------------------------------------------------------------------------------
with tab_bpv:
    st.subheader(f"BPV = DV01 (no notional) · strike date {strike_label}")
    st.caption(
        "BPV of the forward swap and of the two mid-curve legs, computed from the curve: "
        "sum(tau_i · DF(t_i)), i.e. bp of price per bp of rate (the paper's DV01 = 8.6 for "
        "18m10y). Multiply by 100 to read it as \\$ per bp per \\$1mm of notional."
    )
    which = st.radio(
        "Grid",
        [
            "Forward swap T2→T3",
            "Mid-curve leg T1→T3",
            "Mid-curve leg T1→T2",
            "Weight T1→T3 / T2→T3",
            "Weight T1→T2 / T2→T3",
            "Forward swap rate (%)",
        ],
        horizontal=True,
    )
    bpv_spec = {
        "Forward swap T2→T3": ("dv01_23", ".2f", "", "BPV of the swap the FVA delivers into."),
        "Mid-curve leg T1→T3": ("dv01_13", ".2f", "", "BPV of the long leg of the mid-curve spread."),
        "Mid-curve leg T1→T2": ("dv01_12", ".2f", "", "BPV of the short leg of the mid-curve spread."),
        "Weight T1→T3 / T2→T3": (
            "ratio_13",
            ".3f",
            "",
            "BPV weight applied to sigma_{T1,T3} in the mid-curve formula.",
        ),
        "Weight T1→T2 / T2→T3": (
            "ratio_12",
            ".3f",
            "",
            "BPV weight applied to sigma_{T1,T2} in the mid-curve formula.",
        ),
        "Forward swap rate (%)": ("fwd_23", ".3f", "%", "Forward par swap rate T2→T3."),
    }
    k, f_, u_, cap = bpv_spec[which]
    z = grids[k] * (100.0 if k == "fwd_23" else 1.0)
    show_grid(z, ten_labels, exp_labels, cap, f_, u_, key=f"bpv_{k}")

    st.markdown("##### Feed your own BPVs")
    st.caption(
        "Every leg this strike date needs, as (forward_term, swap_tenor) - the same keys "
        "SwapLegsValueCalculator.forward_swap_legs_value takes. Export it, fill the `bpv` "
        "column from your own calculator, and upload it back as the BPV source in the sidebar."
    )
    legs: dict[tuple[str, str], float] = {}
    for e_ in expiries:
        for t_ in tenors:
            t1_, t2_, t3_ = strike, strike + e_, strike + e_ + t_
            for s_, en_ in ((t2_, t3_), (t1_, t3_), (t1_, t2_)):
                legs[(bpv_sources.term_label(s_), bpv_sources.term_label(en_ - s_))] = curve.bpv(
                    s_, en_
                )
    legs_df = pd.DataFrame(
        [{"forward_term": a, "swap_tenor": b, "bpv": v} for (a, b), v in sorted(legs.items())]
    )
    st.dataframe(legs_df.style.format({"bpv": "{:.4f}"}), hide_index=True, height=240)
    st.download_button(
        f"Download the {len(legs_df)} legs needed for strike {strike_label} (CSV)",
        legs_df.to_csv(index=False).encode(),
        file_name=f"bpv_legs_{strike_label}.csv",
        mime="text/csv",
    )

# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------
with tab_inputs:
    st.subheader("ATM normal vol matrix (loaded once, used everywhere)")
    st.caption(
        "Rows = option expiry, columns = swap tenor, values in bp normal vol. Edit cells or "
        "add rows here; upload a CSV in the sidebar to change the tenor columns."
    )
    edited = st.data_editor(
        st.session_state.vol_df,
        num_rows="dynamic",
        width="stretch",
        key="vol_editor",
    )
    if not edited.equals(st.session_state.vol_df):
        st.session_state.vol_df = edited
        st.rerun()

    st.plotly_chart(
        heat(vols, ten_labels, exp_labels, ".0f", "nv", height=420),
        width="stretch",
        theme=None,
        key="plot_surface",
    )

    st.subheader("Curve")
    if curve_mode == "Par swap curve":
        cedit = st.data_editor(
            st.session_state.curve_df, num_rows="dynamic", width="stretch", key="curve_editor"
        )
        if not cedit.equals(st.session_state.curve_df):
            st.session_state.curve_df = cedit
            st.rerun()
    else:
        st.caption(f"Flat par curve at {flat:.3f}%, fixed leg {freq}×/year.")

    pts = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
    st.dataframe(
        pd.DataFrame(
            {
                "t": [fva.label_years(p) for p in pts],
                "DF": [curve.df(p) for p in pts],
                "zero (cont, %)": [curve.zero(p) * 100 for p in pts],
                "par swap rate (%)": [curve.fwd_swap_rate(0.0, p) * 100 for p in pts],
                "BPV 0→t": [curve.bpv(0.0, p) for p in pts],
            }
        ).style.format(
            {"DF": "{:.5f}", "zero (cont, %)": "{:.3f}", "par swap rate (%)": "{:.3f}", "BPV 0→t": "{:.3f}"}
        ),
        hide_index=True,
        width="stretch",
    )

# --------------------------------------------------------------------------------------
# Paper check
# --------------------------------------------------------------------------------------
with tab_paper:
    st.subheader("Worked example of the paper (Appendix p.8) — 6m1y10y, 100k/nv")
    st.caption(
        "Hard-coded paper assumptions: sigma 6m1y = 37, 6m11y = 99, 1y10y = 100, 18m10y = 99; "
        "BPV 6m1y = 1.0, 6m11y = 9.6, 18m10y = 8.6; DF(6m) = 1; rho = 1."
    )
    P = {"t1": 0.5, "t2": 1.5, "s23": 99.0, "s13": 99.0, "s12": 37.0, "d23": 8.6, "d13": 9.6, "d12": 1.0}
    smc = fva.midcurve_vol(P["d13"], P["s13"], P["d12"], P["s12"], P["d23"], 1.0)
    sfva = fva.fva_vol(P["t1"], P["t2"], P["s23"], smc)
    dfva23 = P["t2"] * P["s23"] / ((P["t2"] - P["t1"]) * sfva)
    dfvamc = -P["t1"] * smc / ((P["t2"] - P["t1"]) * sfva)
    vega23 = fva.SQRT_2_OVER_PI * math.sqrt(P["t2"]) * P["d23"]
    vegamc = fva.SQRT_2_OVER_PI * math.sqrt(P["t1"]) * P["d23"]
    vegafva = fva.SQRT_2_OVER_PI * math.sqrt(P["t2"] - P["t1"]) * P["d23"]
    st.dataframe(
        pd.DataFrame(
            {
                "quantity": [
                    "sigma_MC (nv)",
                    "sigma_FVA (nv)",
                    "d sigma_FVA / d sigma_18m10y",
                    "d sigma_FVA / d sigma_MC",
                    "vega 18m10y (bp)",
                    "vega mid-curve (bp)",
                    "FVA notional (mm)",
                    "buy 18m10y (mm)",
                    "sell mid-curve (mm)",
                ],
                "this model": [
                    smc,
                    sfva,
                    dfva23,
                    abs(dfvamc),
                    vega23,
                    vegamc,
                    1e5 / (vegafva * 1e-4) / 1e6,
                    1e5 * dfva23 / (vega23 * 1e-4) / 1e6,
                    abs(1e5 * dfvamc / (vegamc * 1e-4)) / 1e6,
                ],
                "paper": [107, 95, 1.6, 0.6, 8.4, 4.9, 146, 186, 116],
            }
        ).style.format({"this model": "{:.2f}", "paper": "{:.2f}"}),
        hide_index=True,
        width="stretch",
    )
    st.info(
        "The paper's last step (sell 66k/nv 6m11y, buy 6k/nv 6m1y) starts from a rounded "
        "-60k/nv mid-curve leg (0.558 rounded to 0.6). From that same -60k this model returns "
        "67.0k and 7.0k; the unrounded chain from 100k/nv of FVA gives -55.8k mid-curve, "
        "-62.3k 6m11y, +6.5k 6m1y.",
        icon="ℹ️",
    )
