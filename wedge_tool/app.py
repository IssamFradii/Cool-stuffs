"""Streamlit dashboard: rich/cheap cap/swaption wedge monitor.

Run with:  streamlit run app.py
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from wedge.analysis import analyze_wedges
from wedge.cca import (
    BOOK_ANGLES,
    cascade_calibration,
    diagnose,
    exp_decay_correlation,
    rank2_correlation,
    reprice_swaption_vol,
)
from wedge.surface import wedge_surface

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

BLUE, AQUA, RED, GRAY = "#2a78d6", "#1baf7a", "#e34948", "#898781"
DIVERGING = [[0.0, BLUE], [0.5, "#f0efec"], [1.0, RED]]

st.set_page_config(page_title="Wedge Monitor", page_icon="📐", layout="wide")


@st.cache_data
def load_csv(path, **kw):
    return pd.read_csv(path, **kw)


def read_upload(upload, demo_path, label, **kw):
    if upload is not None:
        try:
            return pd.read_csv(upload, **kw), f"{label}: uploaded file"
        except Exception as e:
            st.sidebar.error(f"{label}: could not read upload ({e}); using demo data")
    return load_csv(demo_path, **kw), f"{label}: book demo data"


# ---------------- sidebar ----------------
st.sidebar.title("Wedge Monitor")
st.sidebar.caption("Brigo & Mercurio §6.20 / §7.4 / §7.9")

st.sidebar.subheader("Market data (CSV)")
up_buckets = st.sidebar.file_uploader("S×1 bucket quotes", type="csv",
    help="columns: expiry, swaption_vol, f1, f2, caplet_vol_s, caplet_vol_t")
up_curve = st.sidebar.file_uploader("Annual curve (forwards + caplet vols)", type="csv",
    help="columns: expiry, forward, caplet_vol — annual grid 1..M years")
up_matrix = st.sidebar.file_uploader("Swaption vol matrix", type="csv",
    help="first column: expiry (years); other columns: tenor in years; values: Black vols")

buckets, src1 = read_upload(up_buckets, os.path.join(DATA, "brigo_table_7_13.csv"), "Buckets")
curve, src2 = read_upload(up_curve, os.path.join(DATA, "annual_curve.csv"), "Curve")
vmat, src3 = read_upload(up_matrix, os.path.join(DATA, "swaption_matrix.csv"),
                         "Matrix", index_col=0)
for s in (src1, src2, src3):
    st.sidebar.caption("· " + s)

st.sidebar.subheader("Parameters")
rho_bench = st.sidebar.slider("Benchmark infra-correlation ρ", 0.0, 1.0, 1.0, 0.05,
    help="ρ used to build the cap-implied vol (book uses 1.0)")
rho_low, rho_high = st.sidebar.slider("Plausible implied-ρ band", 0.0, 1.2, (0.70, 1.00), 0.05)
min_wedge_pts = st.sidebar.slider("Min |wedge| to flag (vol pts)", 0.0, 2.0, 0.5, 0.1)
corr_kind = st.sidebar.radio("Forward-rate correlation (surface & CCA)",
                             ["Book rank-2 angles", "Exponential decay"])
beta = st.sidebar.slider("Decay β  (ρᵢⱼ = e^{−β|i−j|})", 0.0, 0.5, 0.10, 0.01,
                         disabled=(corr_kind == "Book rank-2 angles"))


def correlation_matrix(m):
    if corr_kind == "Book rank-2 angles" and m <= len(BOOK_ANGLES):
        return rank2_correlation(BOOK_ANGLES)[:m, :m]
    if corr_kind == "Book rank-2 angles":
        st.warning(f"Book angles cover 19 forwards but {m} are needed — "
                   "using exponential decay instead.")
    return exp_decay_correlation(m, beta)


tab1, tab2, tab3 = st.tabs(["📊  S×1 wedge buckets", "🗺️  Wedge heatmap", "⚙️  CCA calibration"])

# ---------------- tab 1: per-bucket rich/cheap ----------------
with tab1:
    try:
        res = analyze_wedges(buckets, rho=rho_bench, rho_low=rho_low,
                             rho_high=rho_high, min_wedge=min_wedge_pts / 100.0)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    n_rich = int((res.signal == "RICH").sum())
    n_cheap = int((res.signal == "CHEAP").sum())
    widest = res.iloc[res.wedge.abs().idxmax()]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RICH buckets (swaption > caps)", n_rich)
    c2.metric("CHEAP buckets (swaption < caps)", n_cheap)
    c3.metric("Widest wedge", f"{widest.expiry:g}y×1y",
              f"{100 * widest.wedge:+.2f} vol pts", delta_color="off")
    c4.metric("Avg implied ρ", f"{res.implied_rho.mean():.2f}")

    labels = [f"{e:g}y" for e in res.expiry]

    fig = go.Figure()
    fig.add_bar(x=labels, y=100 * res.wedge,
                marker_color=[RED if w > 0 else BLUE for w in res.wedge],
                hovertemplate=("%{x}×1y<br>wedge %{y:+.2f} vol pts<br>"
                               "market %{customdata[0]:.2f}% · cap-implied "
                               "%{customdata[1]:.2f}%<extra></extra>"),
                customdata=np.c_[100 * res.swaption_vol, 100 * res.cap_implied_vol])
    fig.add_hline(y=0, line_color=GRAY, line_width=1)
    fig.update_layout(title="Wedge per bucket — market minus cap-implied "
                            "(red = swaption rich, blue = cheap)",
                      yaxis_title="vol pts", height=340,
                      margin=dict(l=40, r=20, t=60, b=30))
    st.plotly_chart(fig, width="stretch")

    col_a, col_b = st.columns([3, 2])
    with col_a:
        show = pd.DataFrame({
            "bucket": [f"{e:g}y × 1y" for e in res.expiry],
            "market %": (100 * res.swaption_vol).round(2),
            "cap-implied %": (100 * res.cap_implied_vol).round(2),
            "wedge (pts)": (100 * res.wedge).round(2),
            "implied ρ": res.implied_rho.round(3),
            "signal": res.signal,
            "suggested trade": res.trade,
        })

        def _color(sig):
            color = {"RICH": RED, "CHEAP": BLUE}.get(sig, GRAY)
            return f"color: {color}; font-weight: 600"

        st.dataframe(show.style.map(_color, subset=["signal"]),
                     hide_index=True, height=420, width="stretch")
    with col_b:
        fig2 = go.Figure()
        fig2.add_hrect(y0=rho_low, y1=rho_high, fillcolor="#f0efec",
                       line_width=0, layer="below")
        fig2.add_scatter(x=labels, y=res.implied_rho, mode="markers+lines",
                         marker=dict(color=BLUE, size=9), line=dict(color=BLUE, width=1),
                         hovertemplate="%{x}: implied ρ = %{y:.3f}<extra></extra>")
        fig2.add_hline(y=1.0, line_dash="dot", line_color=GRAY)
        fig2.update_layout(title="Implied infra-correlation vs plausible band",
                           yaxis_title="ρ", height=420,
                           margin=dict(l=40, r=20, t=60, b=30))
        st.plotly_chart(fig2, width="stretch")

    st.caption("Formula (6.76): v²ₛwₚₜ ≈ u₁²v₁² + u₂²v₂² + 2ρu₁u₂v₁v₂ — implied ρ "
               "far below ~0.7–1.0 (or above 1) flags cap/swaption misalignment. "
               "§7.9 warning: the two markets can stay non-congruent — relative "
               "value, not arbitrage.")

# ---------------- tab 2: wedge heatmap ----------------
with tab2:
    n_exp, n_ten = vmat.shape
    need = n_exp + n_ten - 1
    if len(curve) < need:
        st.error(f"The annual curve has {len(curve)} rows but the matrix needs "
                 f"{need} annual forwards (max expiry + max tenor − 1).")
    else:
        corr = correlation_matrix(len(curve))
        surf = wedge_surface(vmat, curve, corr)
        wedge_pts = 100 * surf["wedge"]

        fig = go.Figure(go.Heatmap(
            z=wedge_pts.to_numpy(),
            x=[f"{c}y" for c in wedge_pts.columns],
            y=[f"{r}y" for r in wedge_pts.index],
            colorscale=DIVERGING, zmid=0.0,
            colorbar=dict(title="vol pts"),
            text=np.round(wedge_pts.to_numpy(), 2), texttemplate="%{text}",
            customdata=np.dstack([100 * surf["market"].to_numpy(),
                                  100 * surf["implied"].to_numpy()]),
            hovertemplate=("expiry %{y} · tenor %{x}<br>wedge %{z:.2f} pts<br>"
                           "market %{customdata[0]:.2f}% · cap-implied "
                           "%{customdata[1]:.2f}%<extra></extra>"),
        ))
        fig.update_layout(title="Wedge surface: market swaption vol − cap-implied vol "
                                "(red = rich vs caps, blue = cheap)",
                          xaxis_title="Tenor", yaxis_title="Expiry",
                          yaxis_autorange="reversed", height=520,
                          margin=dict(l=60, r=20, t=60, b=40))
        st.plotly_chart(fig, width="stretch")

        st.caption("Cap-implied vols: Rebonato's formula (6.67) with each annual "
                   "forward at constant vol equal to its annualized caplet vol "
                   "(the §6.20 'a2' assumption); tenor 1y collapses to the "
                   "annualized caplet vol. Correlation: " + corr_kind.lower() + ".")

        with st.expander("Show the underlying surfaces (market / cap-implied)"):
            cc1, cc2 = st.columns(2)
            cc1.write("Market vols (%)")
            cc1.dataframe((100 * surf["market"]).round(2), width="stretch")
            cc2.write("Cap-implied vols (%)")
            cc2.dataframe((100 * surf["implied"]).round(2), width="stretch")

# ---------------- tab 3: CCA ----------------
with tab3:
    st.write("**Cascade Calibration (Algorithm 7.4.1 / 7.4.2)** — calibrates the "
             "LFM piecewise-constant volatilities *exactly* to the swaption matrix, "
             "with correlations given exogenously. No optimization: a cascade of "
             "closed-form quadratics, solved left→right, top→down.")
    n_exp, n_ten = vmat.shape
    if n_exp != n_ten:
        st.error("CCA needs a square swaption matrix (as many expiries as tenors).")
    elif len(curve) < 2 * n_exp - 1:
        st.error(f"The annual curve must have at least {2 * n_exp - 1} rows for a "
                 f"{n_exp}×{n_ten} matrix.")
    else:
        corr = correlation_matrix(len(curve))
        sigma = cascade_calibration(vmat.to_numpy(), curve.forward.to_numpy(), corr)
        flags = diagnose(sigma)
        n_neg = int((flags == "negative").sum())
        n_imag = int((flags == "imaginary").sum())

        fwd = curve.forward.to_numpy()
        errs = [abs(reprice_swaption_vol(sigma, fwd, corr, a, c).real
                    - vmat.to_numpy()[a, c - 1])
                for a in range(n_exp) for c in range(1, n_ten + 1)]

        m1, m2, m3 = st.columns(3)
        m1.metric("Repricing error (max)", f"{1e4 * max(errs):.2e} bp")
        m2.metric("Negative σ entries", n_neg,
                  "misaligned quotes" if n_neg else "clean", delta_color="off")
        m3.metric("Imaginary σ entries", n_imag,
                  "pathological quotes" if n_imag else "clean", delta_color="off")

        m = 2 * n_exp - 1
        shown = np.where(flags[:m] == "imaginary", np.abs(sigma[:m]), sigma[:m].real)
        sig_df = pd.DataFrame(
            np.where(np.isnan(sigma[:m].real), np.nan, shown.astype(float)),
            index=[f"F{i+1} ({i+1}→{i+2}y)" for i in range(m)],
            columns=[f"({h}y,{h+1}y]" if h else "(0,1y]" for h in range(n_ten)],
        )

        def _mark(v):
            if pd.isna(v):
                return "color: #c3c2b7"
            if v < 0:
                return f"color: {RED}; font-weight: 700"
            return ""

        st.write("Calibrated instantaneous volatilities σ (TABLE 1 layout: row = "
                 "forward rate, column = time interval; negative entries in red):")
        st.dataframe(sig_df.style.format("{:.4f}", na_rep="—").map(_mark),
                     height=520, width="stretch")
        if n_imag:
            st.error("Some entries came out imaginary — the input matrix is too "
                     "irregular in those areas (book §7.4.1). Their |σ| is shown.")
        st.caption("Negative/imaginary σ's are the book's diagnostic of misaligned "
                   "or illiquid swaption quotes (§7.4.1); smoothing the input "
                   "matrix typically removes them. A clean, all-positive table "
                   "means the matrix is internally consistent with the chosen "
                   "correlations.")
