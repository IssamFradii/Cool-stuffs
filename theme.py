"""Design tokens: the single place where every colour in the dashboard lives.

Charts and tables ask this module for colours -- they never hard-code a hex.
That means switching light/dark, or re-skinning the whole desk to your bank's
brand, is a change in ONE file.

Rules baked in here (they are what stop a dashboard looking amateur):

* Categorical colours (identity: "Rates" vs "Vol" vs "Carry") are used in a
  fixed order and never cycled. Past 8 series you fold the tail into "Other".
* Sequential ramps (magnitude: a vol surface) are ONE hue, light -> dark.
  Never a rainbow.
* Diverging ramps (polarity: rich vs cheap, +DV01 vs -DV01) are two opposite
  hues with a NEUTRAL GREY midpoint, so "zero" reads as nothing.
* Status colours (good / warning / critical) are reserved. They never get
  reused as "series number 4", and they always ship with a sign or a label
  next to them so the meaning never rests on colour alone.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Categorical slots -- fixed order, never cycled.
# --------------------------------------------------------------------------
_SERIES_LIGHT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                 '#e87ba4', '#008300', '#4a3aa7', '#e34948']
_SERIES_DARK = ['#3987e5', '#d95926', '#199e70', '#c98500',
                '#d55181', '#008300', '#9085e9', '#e66767']

# --------------------------------------------------------------------------
# Sequential ramp -- one hue (blue), lightest to darkest.
# --------------------------------------------------------------------------
_BLUE_RAMP = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5',
              '#256abf', '#184f95', '#0d366b']

# --------------------------------------------------------------------------
# Status palette -- fixed, never themed, never used for a series.
# --------------------------------------------------------------------------
_STATUS = {
    'good': '#0ca30c',
    'warning': '#fab219',
    'serious': '#ec835a',
    'critical': '#d03b3b',
}

LIGHT: dict = {
    'mode': 'light',
    'surface': '#fcfcfb',                  # card + chart background
    'plane': '#f9f9f7',                    # page background behind the cards
    'ink': '#0b0b0b',                      # primary text
    'ink2': '#52514e',                     # secondary text
    'muted': '#898781',                    # axis ticks, captions
    'grid': '#e1e0d9',                     # hairline gridlines (solid, never dashed)
    'axis': '#c3c2b7',                     # baseline / axis rule
    'border': 'rgba(11,11,11,0.10)',       # 1px ring around cards
    'series': _SERIES_LIGHT,
    # low value recedes toward the light surface -> lightest step first
    'sequential': _BLUE_RAMP,
    # two opposite hues + neutral grey middle
    'diverging': ['#2a78d6', '#f0efec', '#e34948'],
    'pos': '#2a78d6',                      # positive / long  (cool pole)
    'neg': '#e34948',                      # negative / short (warm pole)
    'up_text': '#006300',                  # P&L up, as TEXT (needs more contrast)
    'down_text': '#d03b3b',                # P&L down, as TEXT
    **_STATUS,
}

DARK: dict = {
    'mode': 'dark',
    'surface': '#1a1a19',
    'plane': '#0d0d0d',
    'ink': '#ffffff',
    'ink2': '#c3c2b7',
    'muted': '#898781',
    'grid': '#2c2c2a',
    'axis': '#383835',
    'border': 'rgba(255,255,255,0.10)',
    'series': _SERIES_DARK,
    # low value recedes toward the DARK surface -> darkest step first
    'sequential': list(reversed(_BLUE_RAMP)),
    'diverging': ['#3987e5', '#383835', '#e66767'],
    'pos': '#3987e5',
    'neg': '#e66767',
    'up_text': '#0ca30c',
    'down_text': '#e66767',
    **_STATUS,
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def palette(dark: bool) -> dict:
    """Return the whole token set for the active mode."""
    return DARK if dark else LIGHT


# --------------------------------------------------------------------------
# Colour maths.
#
# Charting libraries blend between colour stops in sRGB, which is wrong: sRGB
# is not perceptually uniform, so a blue->grey->red ramp passes through muddy
# browns and the middle of the scale turns to sludge. That is very visible on
# a dark surface, where the neutral midpoint is a dark grey.
#
# We blend in OKLab instead -- a perceptual space -- and hand the chart a set
# of pre-computed stops. The ANCHORS are still the palette hexes above; only
# the path between them changes.
# --------------------------------------------------------------------------


def hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    return '#%02x%02x%02x' % tuple(min(255, max(0, round(c))) for c in rgb)


def _srgb_to_linear(c: float) -> float:
    c /= 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c: float) -> float:
    c = 12.92 * c if c <= 0.0031308 else 1.055 * (max(c, 0.0) ** (1 / 2.4)) - 0.055
    return c * 255


def _to_oklab(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    r, g, b = (_srgb_to_linear(c) for c in rgb)
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = l ** (1 / 3), m ** (1 / 3), s ** (1 / 3)
    return (0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_)


def _from_oklab(lab: tuple[float, float, float]) -> tuple[float, float, float]:
    big_l, a, b = lab
    l_ = big_l + 0.3963377774 * a + 0.2158037573 * b
    m_ = big_l - 0.1055613458 * a - 0.0638541728 * b
    s_ = big_l - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    return (_linear_to_srgb(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
            _linear_to_srgb(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
            _linear_to_srgb(-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s))


def oklab_ramp(anchors: list[str], steps: int = 11) -> list[str]:
    """Expand a few anchor colours into `steps` stops, blended in OKLab."""
    labs = [_to_oklab(hex_to_rgb(h)) for h in anchors]
    out = []
    for i in range(steps):
        pos = i / (steps - 1) * (len(labs) - 1)
        lo = min(int(pos), len(labs) - 2)
        f = pos - lo
        a, b = labs[lo], labs[lo + 1]
        out.append(rgb_to_hex(_from_oklab(
            tuple(a[k] + (b[k] - a[k]) * f for k in range(3)))))  # type: ignore[arg-type]
    return out


def sample_ramp(ramp: list[str], t: float) -> tuple[float, float, float]:
    """The colour a chart will paint at normalised position `t` (0..1)."""
    t = min(max(t, 0.0), 1.0)
    pos = t * (len(ramp) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ramp) - 1)
    f = pos - lo
    a, b = hex_to_rgb(ramp[lo]), hex_to_rgb(ramp[hi])
    return tuple(a[i] + (b[i] - a[i]) * f for i in range(3))  # type: ignore[return-value]


def best_ink(rgb: tuple[float, float, float]) -> str:
    """Black or white for a label sitting ON `rgb` -- whichever contrasts more.

    In-cell numbers on a heatmap sit on a fill that changes cell to cell, so
    one fixed label colour is always unreadable somewhere. Choosing per cell
    keeps every number legible instead of most of them.
    """
    def channel(c: float) -> float:
        c /= 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (channel(c) for c in rgb)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return '#0b0b0b' if (lum + 0.05) / 0.05 >= 1.05 / (lum + 0.05) else '#ffffff'


# --------------------------------------------------------------------------
# Page-level CSS.
#
# Written against CSS custom properties so the light/dark values swap in one
# place, and so AG Grid (which styles itself from --ag-* variables) inherits
# exactly the same palette as the charts.
# --------------------------------------------------------------------------
def page_css() -> str:
    def block(p: dict) -> str:
        return f"""
      --surface: {p['surface']};
      --plane: {p['plane']};
      --ink: {p['ink']};
      --ink2: {p['ink2']};
      --muted: {p['muted']};
      --hairline: {p['grid']};
      --ring: {p['border']};
      --accent: {p['series'][0]};
      --pos: {p['up_text']};
      --neg: {p['down_text']};
    """

    return f"""
    :root {{ {block(LIGHT)} }}
    body.body--dark {{ {block(DARK)} }}

    body {{
      background: var(--plane);
      color: var(--ink);
      font-family: {FONT};
    }}

    /* Cards: flat surface + one hairline ring. No heavy shadows -- a trading
       screen should read as dense and calm, not as a stack of paper. */
    .desk-card {{
      background: var(--surface);
      border: 1px solid var(--ring);
      border-radius: 10px;
      padding: 14px 16px;
      box-shadow: none;
    }}
    .desk-title {{
      font-size: 12px;
      font-weight: 600;
      letter-spacing: .04em;
      text-transform: uppercase;
      color: var(--ink2);
    }}
    .desk-sub {{ font-size: 11px; color: var(--muted); }}

    /* Hero / stat-tile numbers: proportional figures on purpose --
       tabular-nums makes a big "121" look loose. */
    .desk-stat {{
      font-size: 30px;
      font-weight: 600;
      line-height: 1.1;
      color: var(--ink);
      font-variant-numeric: proportional-nums;
    }}
    /* Table + axis numbers DO align vertically, so they get tabular figures. */
    .nicegui-aggrid {{ font-variant-numeric: tabular-nums; }}

    .desk-sticky {{
      position: sticky;
      top: 0;
      z-index: 40;
      background: var(--plane);
      border-bottom: 1px solid var(--ring);
      backdrop-filter: blur(6px);
    }}

    /* AG Grid inherits the same tokens as the charts. */
    .nicegui-aggrid {{
      --ag-background-color: var(--surface);
      --ag-foreground-color: var(--ink);
      --ag-header-background-color: var(--surface);
      --ag-header-text-color: var(--ink2);
      --ag-border-color: var(--hairline);
      --ag-row-border-color: var(--hairline);
      --ag-row-hover-color: color-mix(in srgb, var(--accent) 10%, transparent);
      --ag-selected-row-background-color: color-mix(in srgb, var(--accent) 16%, transparent);
      --ag-accent-color: var(--accent);
      --ag-font-family: {FONT};
      --ag-font-size: 12px;
      --ag-header-font-size: 11px;
      --ag-row-height: 30px;
      --ag-header-height: 32px;
      --ag-wrapper-border-radius: 8px;
    }}
    .pnl-pos {{ color: var(--pos); }}
    .pnl-neg {{ color: var(--neg); }}
    """
