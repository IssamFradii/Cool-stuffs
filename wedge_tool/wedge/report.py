"""Console table and chart for the wedge analysis."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Palette (light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"   # series 1 / diverging cool pole (swaption cheap)
AQUA = "#1baf7a"   # series 2
RED = "#e34948"    # diverging warm pole (swaption rich)
BAND = "#f0efec"   # neutral gray band


def print_report(df: pd.DataFrame, rho: float) -> None:
    out = pd.DataFrame(
        {
            "bucket": [f"{e:g}y x 1y" for e in df.expiry],
            "mkt swpt vol %": (100 * df.swaption_vol).round(2),
            "cap-implied %": (100 * df.cap_implied_vol).round(2),
            "wedge (vol pts)": (100 * df.wedge).round(2),
            "implied rho": df.implied_rho.round(3),
            "z": df.wedge_zscore.round(2),
            "signal": df.signal,
            "suggested trade": df.trade,
        }
    )
    print(f"\nCap/swaption wedge per bucket (cap-implied vol built with rho = {rho:g})")
    print("wedge = market swaption vol - cap-implied vol; positive = swaption RICH vs caps\n")
    print(out.to_string(index=False))

    n_rich = (df.signal == "RICH").sum()
    n_cheap = (df.signal == "CHEAP").sum()
    print(
        f"\nSummary: {n_rich} bucket(s) RICH, {n_cheap} bucket(s) CHEAP, "
        f"{(df.signal == 'FAIR').sum()} FAIR out of {len(df)}."
    )
    print(
        "Caveat (Brigo & Mercurio 7.9): cap and swaption markets can stay\n"
        "non-congruent for long periods; treat signals as relative value,\n"
        "not arbitrage."
    )


def plot_report(df: pd.DataFrame, rho: float, rho_low: float, rho_high: float, path: str) -> None:
    x = np.arange(len(df))
    labels = [f"{e:g}y" for e in df.expiry]

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(8.6, 9.2), sharex=True, facecolor=SURFACE,
        gridspec_kw={"height_ratios": [1.2, 1.0, 1.0], "hspace": 0.42},
    )

    for ax in (ax1, ax2, ax3):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, length=0)

    # Panel 1: market vs cap-implied vol
    ax1.plot(x, 100 * df.swaption_vol, color=BLUE, linewidth=2, marker="o",
             markersize=5, label="Market S×1y swaption vol")
    ax1.plot(x, 100 * df.cap_implied_vol, color=AQUA, linewidth=2, marker="o",
             markersize=5, label=f"Cap-implied vol (ρ = {rho:g})")
    ax1.annotate("market", (x[-1], 100 * df.swaption_vol.iloc[-1]),
                 xytext=(8, 0), textcoords="offset points", color=BLUE,
                 fontsize=9, va="center")
    ax1.annotate("cap-implied", (x[-1], 100 * df.cap_implied_vol.iloc[-1]),
                 xytext=(8, 0), textcoords="offset points", color=AQUA,
                 fontsize=9, va="center")
    ax1.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=INK_2)
    ax1.set_ylabel("Black vol (%)", color=INK_2, fontsize=9)
    ax1.set_title("Cap/swaption wedge per expiry bucket",
                  color=INK, fontsize=13, loc="left", pad=14, fontweight="bold")
    ax1.set_xlim(-0.5, len(df) - 0.5 + 1.2)

    # Panel 2: the wedge, diverging around zero
    colors = [RED if w > 0 else BLUE for w in df.wedge]
    ax2.bar(x, 100 * df.wedge, width=0.62, color=colors)
    ax2.axhline(0, color=BASELINE, linewidth=1)
    for i in (int(df.wedge.idxmax()), int(df.wedge.idxmin())):
        w = 100 * df.wedge.iloc[i]
        ax2.annotate(f"{w:+.2f}", (x[i], w), xytext=(0, 5 if w > 0 else -12),
                     textcoords="offset points", ha="center", color=INK_2, fontsize=9)
    lo, hi = 100 * df.wedge.min(), 100 * df.wedge.max()
    span = hi - lo
    ax2.set_ylim(lo - 0.18 * span, hi + 0.18 * span)
    ax2.set_ylabel("Wedge (vol pts)", color=INK_2, fontsize=9)
    ax2.set_title("Market minus cap-implied  (red = swaption rich, blue = cheap)",
                  color=INK_2, fontsize=10, loc="left", pad=12)

    # Panel 3: implied infra-correlation vs plausible band
    ax3.axhspan(rho_low, rho_high, color=BAND, zorder=0)
    ax3.axhline(1.0, color=BASELINE, linewidth=1, linestyle=(0, (4, 3)))
    ax3.vlines(x, 0, df.implied_rho, color=BLUE, linewidth=2)
    ax3.plot(x, df.implied_rho, "o", color=BLUE, markersize=6)
    ax3.annotate(f"plausible band [{rho_low:g}, {rho_high:g}]",
                 (0.99, rho_high), xycoords=("axes fraction", "data"),
                 xytext=(0, 3), textcoords="offset points",
                 ha="right", color=MUTED, fontsize=8.5)
    ax3.set_ylabel("Implied ρ", color=INK_2, fontsize=9)
    ax3.set_title("Implied infra-correlation per bucket (inverting eq. 6.76)",
                  color=INK_2, fontsize=10, loc="left")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_xlabel("Swaption expiry bucket", color=INK_2, fontsize=9)

    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
