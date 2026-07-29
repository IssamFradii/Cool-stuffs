"""ECharts option builders.

Every function here takes the palette `p` plus data, and returns a plain dict
that NiceGUI hands to ECharts. No function touches the UI -- that keeps them
trivially testable and means a chart can be re-themed by calling it again with
the other palette.

House style applied everywhere:
  * 2px lines, hairline SOLID gridlines, no chart junk
  * a crosshair tooltip on time series, a per-cell tooltip on heatmaps
  * a legend whenever there are >= 2 series (identity is never colour-alone)
  * direct labels used SELECTIVELY -- the endpoint or the extreme, never a
    number on every point
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from theme import FONT, best_ink, oklab_ramp, sample_ramp

# --------------------------------------------------------------------------
# Shared chart chrome
# --------------------------------------------------------------------------


def _base(p: dict) -> dict:
    return {
        'backgroundColor': 'transparent',
        'textStyle': {'fontFamily': FONT, 'color': p['ink2']},
        'animationDuration': 300,
    }


def _tooltip(p: dict, **kw) -> dict:
    return {
        'backgroundColor': p['surface'],
        'borderColor': p['axis'],
        'borderWidth': 1,
        'padding': [8, 10],
        'textStyle': {'color': p['ink'], 'fontSize': 12, 'fontFamily': FONT},
        'extraCssText': 'border-radius:8px;box-shadow:0 4px 16px rgba(0,0,0,.14);',
        **kw,
    }


def _legend(p: dict, **kw) -> dict:
    """A legend is always present for >= 2 series."""
    return {
        'show': True,
        'top': 0,
        'right': 0,
        'itemWidth': 10,
        'itemHeight': 10,
        'itemGap': 14,
        'icon': 'roundRect',
        'textStyle': {'color': p['ink2'], 'fontSize': 11, 'fontFamily': FONT},
        **kw,
    }


def _cat_axis(p: dict, data: list[str], name: str = '') -> dict:
    return {
        'type': 'category',
        'data': data,
        'name': name,
        'nameTextStyle': {'color': p['muted'], 'fontSize': 10},
        'axisLine': {'lineStyle': {'color': p['axis'], 'width': 1}},
        'axisTick': {'show': False},
        'axisLabel': {'color': p['muted'], 'fontSize': 11},
        'splitLine': {'show': False},
    }


def _val_axis(p: dict, name: str = '', **kw) -> dict:
    return {
        'type': 'value',
        'name': name,
        'nameTextStyle': {'color': p['muted'], 'fontSize': 10, 'align': 'left'},
        'axisLine': {'show': False},
        'axisTick': {'show': False},
        'axisLabel': {'color': p['muted'], 'fontSize': 11},
        # solid hairlines only -- dashed gridlines read as "threshold"
        'splitLine': {'lineStyle': {'color': p['grid'], 'width': 1, 'type': 'solid'}},
        **kw,
    }


# --------------------------------------------------------------------------
# 1. Par swap curve -- 2 series, one direct label on the benchmark tenor
# --------------------------------------------------------------------------


def curve_chart(p: dict, tenors: list[str], today: np.ndarray,
                prev: np.ndarray) -> dict:
    bench = tenors.index('10Y')
    # Direct-label ONE point (the 10Y benchmark) rather than all nine. ECharts
    # attaches line labels to symbols, so the other points get symbolSize 0
    # instead of the series turning symbols off wholesale.
    today_data: list = [{'value': round(float(v), 3), 'symbolSize': 0}
                        for v in today]
    today_data[bench] = {
        'value': round(float(today[bench]), 3),
        'symbolSize': 9,
        'label': {
            'show': True, 'position': 'top', 'distance': 10,
            'color': p['ink'], 'fontSize': 11, 'fontWeight': 600,
            'formatter': f'10Y {today[bench]:.3f}%',
        },
    }
    return {
        **_base(p),
        'tooltip': _tooltip(p, trigger='axis',
                            axisPointer={'type': 'cross',
                                         'lineStyle': {'color': p['muted'], 'width': 1},
                                         'crossStyle': {'color': p['muted'], 'width': 1},
                                         'label': {'backgroundColor': p['ink2']}},
                            valueFormatter=':.3f'),
        'legend': _legend(p, data=['Today', '1 week ago']),
        'grid': {'left': 8, 'right': 48, 'top': 34, 'bottom': 4, 'containLabel': True},
        'xAxis': _cat_axis(p, tenors, 'tenor'),
        'yAxis': _val_axis(p, 'par rate %', scale=True,
                           axisLabel={'color': p['muted'], 'fontSize': 11,
                                      'formatter': '{value}%'}),
        'series': [
            {
                'name': '1 week ago',
                'type': 'line',
                'data': [round(float(v), 3) for v in prev],
                'smooth': 0.25,
                'showSymbol': False,
                'symbolSize': 8,
                'lineStyle': {'width': 2, 'color': p['series'][1], 'opacity': 0.75},
                'itemStyle': {'color': p['series'][1]},
                'z': 1,
            },
            {
                'name': 'Today',
                'type': 'line',
                'data': today_data,
                'smooth': 0.25,
                'showSymbol': True,
                'lineStyle': {'width': 2, 'color': p['series'][0]},
                'itemStyle': {'color': p['series'][0],
                              'borderColor': p['surface'], 'borderWidth': 2},
                'z': 2,
            },
        ],
    }


# --------------------------------------------------------------------------
# 2. Cumulative P&L attribution -- 3 series, endpoint label on the biggest
# --------------------------------------------------------------------------


def pnl_chart(p: dict, hist: pd.DataFrame) -> dict:
    cols = ['Rates', 'Vol', 'Carry']
    dates = [d.strftime('%d %b') for d in hist['date']]
    # Direct-label only the series that ends largest -- the one that matters.
    leader = max(cols, key=lambda c: abs(hist[c].iloc[-1]))

    series = []
    for i, col in enumerate(cols):
        colour = p['series'][i]
        series.append({
            'name': col,
            'type': 'line',
            'data': [round(float(v)) for v in hist[col]],
            'showSymbol': False,
            'symbolSize': 8,
            'smooth': 0.2,
            'lineStyle': {'width': 2, 'color': colour},
            'itemStyle': {'color': colour},
            'emphasis': {'focus': 'series'},
            'endLabel': {
                'show': col == leader,
                'distance': 6,
                'color': p['ink'],      # text wears ink, the line carries identity
                'fontSize': 11,
                'fontWeight': 600,
                'formatter': f'{col} {hist[col].iloc[-1] / 1e6:+.2f}m',
            },
        })

    return {
        **_base(p),
        'tooltip': _tooltip(p, trigger='axis',
                            axisPointer={'type': 'cross',
                                         'lineStyle': {'color': p['muted'], 'width': 1},
                                         'crossStyle': {'color': p['muted'], 'width': 1},
                                         'label': {'backgroundColor': p['ink2']}}),
        'legend': _legend(p, data=cols),
        'grid': {'left': 8, 'right': 96, 'top': 34, 'bottom': 4, 'containLabel': True},
        'xAxis': {**_cat_axis(p, dates), 'boundaryGap': False,
                  'axisLabel': {'color': p['muted'], 'fontSize': 11, 'interval': 14}},
        'yAxis': _val_axis(p, 'cum. P&L', scale=True,
                           axisLabel={'color': p['muted'], 'fontSize': 11,
                                      'formatter': '{value}'}),
        'series': series,
    }


# --------------------------------------------------------------------------
# 3. Heatmaps
# --------------------------------------------------------------------------


def _heatmap(p: dict, xs: list[str], ys: list[str], z: np.ndarray, *,
             anchors: list[str], vmin: float, vmax: float,
             decimals: int, unit: str,
             x_name: str, y_name: str) -> dict:
    """Shared heatmap body. ECharts wants [x_index, y_index, value] triples."""
    # Blend the palette anchors in OKLab so the middle of the scale stays
    # clean instead of turning to mud (see theme.oklab_ramp).
    ramp = oklab_ramp(anchors, 11)

    data = []
    span = (vmax - vmin) or 1.0
    for yi, _ in enumerate(ys):
        for xi, _ in enumerate(xs):
            v = float(z[yi, xi])
            t = (v - vmin) / span
            data.append({
                'value': [xi, yi, round(v, decimals)],
                # in-cell number stays legible whatever the fill underneath is
                'label': {'color': best_ink(sample_ramp(ramp, t))},
            })

    # A heatmap datum is [x_index, y_index, value], so a template formatter
    # would print "expiry 3 x tail 2". Look the indices back up against the
    # axis labels instead. The ':' prefix sends this to the browser as JS.
    tooltip_js = (
        'p => { const X = %s, Y = %s;'
        ' return `${Y[p.value[1]]} &times; ${X[p.value[0]]}'
        '<br/><b>${p.value[2]}</b> %s`; }'
    ) % (json.dumps(xs), json.dumps(ys), unit)

    return {
        **_base(p),
        'tooltip': _tooltip(p, trigger='item', **{':formatter': tooltip_js}),
        'grid': {'left': 8, 'right': 12, 'top': 10, 'bottom': 48, 'containLabel': True},
        'xAxis': {**_cat_axis(p, xs, x_name), 'splitArea': {'show': False}},
        'yAxis': {**_cat_axis(p, ys, y_name), 'splitArea': {'show': False},
                  'inverse': True},
        'visualMap': {
            'type': 'continuous',
            'min': vmin,
            'max': vmax,
            'calculable': True,
            'realtime': True,
            'orient': 'horizontal',
            'left': 'center',
            'bottom': 2,
            'itemWidth': 12,
            'itemHeight': 160,
            'precision': decimals,
            'textStyle': {'color': p['muted'], 'fontSize': 10, 'fontFamily': FONT},
            'handleStyle': {'borderColor': p['surface'], 'borderWidth': 2},
            'inRange': {'color': ramp},
        },
        'series': [{
            'type': 'heatmap',
            'data': data,
            'label': {'show': True, 'fontSize': 10, 'fontFamily': FONT,
                      'formatter': '{@[2]}'},
            # a 2px gap in the surface colour separates cells -- never a border
            'itemStyle': {'borderColor': p['surface'], 'borderWidth': 2},
            'emphasis': {'itemStyle': {'borderColor': p['ink'], 'borderWidth': 2}},
            'progressive': 0,
        }],
    }


def vol_surface_heatmap(p: dict, expiries: list[str], tails: list[str],
                        z: np.ndarray) -> dict:
    """Magnitude -> SEQUENTIAL: one hue, low end recedes into the surface."""
    return _heatmap(
        p, tails, expiries, z,
        anchors=p['sequential'],
        vmin=float(np.floor(z.min())), vmax=float(np.ceil(z.max())),
        decimals=1, unit='bp normal vol',
        x_name='tail', y_name='expiry',
    )


# A z-score scale is conventionally read against sigma, and 2 sigma is where a
# trade becomes actionable. Fixing the ends here (rather than scaling to the
# day's biggest outlier) means the colours mean the same thing tomorrow, and
# one freak cell can't flatten the rest of the surface to grey.
WEDGE_LIMIT = 2.0


def wedge_heatmap(p: dict, expiries: list[str], tails: list[str],
                  z: np.ndarray) -> dict:
    """Polarity -> DIVERGING: two opposite hues, neutral grey at zero.

    Positive = market above model = rich = sell candidate. The scale is
    symmetric, so +1.5 and -1.5 are equally loud and zero reads as nothing.
    """
    return _heatmap(
        p, tails, expiries, z,
        anchors=p['diverging'],
        vmin=-WEDGE_LIMIT, vmax=WEDGE_LIMIT,
        decimals=2, unit='z-score (+ rich / &minus; cheap)',
        x_name='tail', y_name='expiry',
    )


# --------------------------------------------------------------------------
# 4. Risk ladder -- sign is the story, so colour encodes polarity
# --------------------------------------------------------------------------


def risk_ladder_chart(p: dict, ladder: pd.DataFrame) -> dict:
    peak = ladder['dv01'].abs().idxmax()
    bars = []
    for i, v in enumerate(ladder['dv01']):
        positive = v >= 0
        bars.append({
            'value': round(float(v)),
            'itemStyle': {
                'color': p['pos'] if positive else p['neg'],
                # 4px rounded data-end, square where it meets the baseline
                'borderRadius': [4, 4, 0, 0] if positive else [0, 0, 4, 4],
            },
            # label only the single biggest bucket, not all eleven
            'label': {
                'show': i == peak,
                'position': 'top' if positive else 'bottom',
                'color': p['ink'], 'fontSize': 11, 'fontWeight': 600,
                'formatter': f'{v / 1e3:+,.0f}k',
            },
        })

    limit = float(ladder['limit'].iloc[0])
    return {
        **_base(p),
        'tooltip': _tooltip(p, trigger='axis',
                            axisPointer={'type': 'shadow',
                                         'shadowStyle': {'color': p['border']}}),
        'grid': {'left': 8, 'right': 12, 'top': 26, 'bottom': 4, 'containLabel': True},
        'xAxis': _cat_axis(p, list(ladder['bucket']), 'bucket'),
        'yAxis': _val_axis(p, 'DV01', scale=True),
        'series': [{
            'name': 'DV01',
            'type': 'bar',
            'data': bars,
            'barMaxWidth': 26,
            'barCategoryGap': '32%',   # the 2px+ gap between adjacent fills
            'markLine': {
                'silent': True,
                'symbol': 'none',
                'label': {'color': p['muted'], 'fontSize': 10,
                          'formatter': 'limit', 'position': 'insideEndTop'},
                'lineStyle': {'color': p['warning'], 'width': 1, 'type': 'dashed'},
                'data': [{'yAxis': limit}, {'yAxis': -limit}],
            },
        }],
    }


# --------------------------------------------------------------------------
# 5. XVA charge stack -- three same-sign components, stacked to a total
# --------------------------------------------------------------------------


def xva_chart(p: dict, xva: pd.DataFrame) -> dict:
    df = xva.copy()
    df['total'] = df['cva'] + df['fva'] + df['colva']
    df = df.sort_values('total')                      # worst charge at the top
    cols = [('cva', 'CVA'), ('fva', 'FVA'), ('colva', 'ColVA')]

    series = []
    for i, (key, name) in enumerate(cols):
        series.append({
            'name': name,
            'type': 'bar',
            'stack': 'charge',
            'data': [round(float(v) / 1e3) for v in df[key]],
            'barMaxWidth': 20,
            'itemStyle': {
                'color': p['series'][i],
                # 2px surface gap between stacked segments
                'borderColor': p['surface'],
                'borderWidth': 2,
                'borderRadius': 2,
            },
            'emphasis': {'focus': 'series'},
        })

    return {
        **_base(p),
        'tooltip': _tooltip(p, trigger='axis',
                            axisPointer={'type': 'shadow',
                                         'shadowStyle': {'color': p['border']}},
                            valueFormatter=':,.0f'),
        'legend': _legend(p, data=[n for _, n in cols]),
        'grid': {'left': 8, 'right': 16, 'top': 34, 'bottom': 4, 'containLabel': True},
        'xAxis': _val_axis(p, 'k', scale=True),
        'yAxis': _cat_axis(p, list(df['counterparty'])),
        'series': series,
    }
