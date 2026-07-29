"""AG Grid builders.

AG Grid is the table component real trading desks use: sorting, column
filters, pinning, resizing and virtual scrolling all come for free, so a
20-row blotter and a 20,000-row one are the same code.

Each function returns the AG Grid *options dict*. `app.py` wraps it in
`ui.aggrid(...)`. Values are formatted in the browser (JS `valueFormatter`)
rather than pre-formatted in Python, so sorting and filtering still operate on
the real numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Reusable JS value formatters. The ':' prefix on the key tells NiceGUI to
# send the string as JavaScript instead of as a literal.
# --------------------------------------------------------------------------
JS_NUM0 = "p => p.value == null ? '' : Number(p.value).toLocaleString('en-US', {maximumFractionDigits: 0})"
JS_NUM2 = "p => p.value == null ? '' : Number(p.value).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})"
JS_SIGNED0 = ("p => p.value == null ? '' : (p.value > 0 ? '+' : '') + "
              "Number(p.value).toLocaleString('en-US', {maximumFractionDigits: 0})")
JS_SIGNED2 = ("p => p.value == null ? '' : (p.value > 0 ? '+' : '') + "
              "Number(p.value).toFixed(2)")
JS_MONEY = """p => {
  if (p.value == null) return '';
  const s = p.value < 0 ? '-' : '';
  const a = Math.abs(p.value);
  if (a >= 1e9) return s + (a / 1e9).toFixed(2) + 'bn';
  if (a >= 1e6) return s + (a / 1e6).toFixed(2) + 'm';
  if (a >= 1e3) return s + (a / 1e3).toFixed(0) + 'k';
  return s + a.toFixed(0);
}"""
# Signed variant, for any money column that is also colour-coded: the sign is
# what carries the meaning, the colour only reinforces it.
JS_MONEY_SIGNED = JS_MONEY.replace(
    "const s = p.value < 0 ? '-' : '';",
    "const s = p.value < 0 ? '-' : '+';")

# Colour is a REINFORCEMENT here, never the only cue: every one of these
# columns also renders an explicit +/- sign.
PNL_CLASS_RULES = {'pnl-pos': 'x > 0', 'pnl-neg': 'x < 0'}

_DEFAULT_COL = {
    'sortable': True,
    'filter': True,
    'resizable': True,
    'suppressMovable': False,
}


def _num(header: str, field: str, fmt: str = JS_NUM0, *,
         flex: float = 1, signed: bool = False, width: int | None = None) -> dict:
    col = {
        'headerName': header,
        'field': field,
        'type': 'numericColumn',
        ':valueFormatter': fmt,
        'flex': flex,
    }
    if width:
        col['minWidth'] = width
    if signed:
        col['cellClassRules'] = PNL_CLASS_RULES
    return col


def _grid(column_defs: list[dict], row_data: list[dict], **kw) -> dict:
    return {
        'defaultColDef': _DEFAULT_COL,
        'columnDefs': column_defs,
        'rowData': row_data,
        'animateRows': True,
        'suppressCellFocus': True,
        'rowSelection': {'mode': 'singleRow', 'checkboxes': False},
        **kw,
    }


# --------------------------------------------------------------------------
# Positions blotter
# --------------------------------------------------------------------------


def positions_grid(pos: pd.DataFrame) -> dict:
    cols = [
        {'headerName': 'Instrument', 'field': 'instrument',
         'flex': 2.4, 'minWidth': 210, 'pinned': 'left'},
        {'headerName': 'CCY', 'field': 'ccy', 'flex': 0.6, 'minWidth': 70},
        {'headerName': 'Book', 'field': 'book', 'flex': 0.8, 'minWidth': 80},
        _num('Notional', 'notional', JS_MONEY, flex=1),
        _num('DV01', 'dv01', JS_SIGNED0, flex=1, signed=True),
        _num('Vega', 'vega', JS_SIGNED0, flex=1, signed=True),
        _num('MTM', 'mtm', JS_MONEY_SIGNED, flex=1, signed=True),
        _num('Day P&L', 'day_pnl', JS_SIGNED0, flex=1, signed=True),
    ]
    # A stable row id lets AG Grid animate a changed value in place instead of
    # tearing the table down and rebuilding it on every tick.
    return _grid(cols, pos.round(2).to_dict('records')) | {
        ':getRowId': "p => p.data.instrument"}


# --------------------------------------------------------------------------
# Risk ladder (the table-view twin of the DV01 bar chart)
# --------------------------------------------------------------------------


def risk_grid(ladder: pd.DataFrame) -> dict:
    df = ladder.copy()
    df['util'] = (df['dv01'].abs() / df['limit'] * 100).round(1)
    cols = [
        {'headerName': 'Bucket', 'field': 'bucket', 'flex': 1, 'minWidth': 80},
        _num('DV01', 'dv01', JS_SIGNED0, flex=1.4, signed=True),
        _num('Vega', 'vega', JS_SIGNED0, flex=1.2, signed=True),
        _num('Limit use %', 'util', JS_NUM2, flex=1.2),
    ]
    return _grid(cols, df.round(2).to_dict('records')) | {
        ':getRowId': "p => p.data.bucket"}


# --------------------------------------------------------------------------
# XVA by counterparty
# --------------------------------------------------------------------------


def xva_grid(xva: pd.DataFrame) -> dict:
    df = xva.copy()
    df['total'] = df['cva'] + df['dva'] + df['fva'] + df['colva']
    cols = [
        {'headerName': 'Counterparty', 'field': 'counterparty',
         'flex': 2, 'minWidth': 170, 'pinned': 'left'},
        {'headerName': 'Rating', 'field': 'rating', 'flex': 0.8, 'minWidth': 80},
        {'headerName': 'CSA', 'field': 'csa', 'flex': 0.9, 'minWidth': 90},
        _num('EPE', 'epe', JS_MONEY, flex=1),
        _num('CVA', 'cva', JS_MONEY_SIGNED, flex=1, signed=True),
        _num('DVA', 'dva', JS_MONEY_SIGNED, flex=1, signed=True),
        _num('FVA', 'fva', JS_MONEY_SIGNED, flex=1, signed=True),
        _num('ColVA', 'colva', JS_MONEY_SIGNED, flex=1, signed=True),
        _num('Total XVA', 'total', JS_MONEY_SIGNED, flex=1.1, signed=True),
    ]
    return _grid(cols, df.round(2).to_dict('records')) | {
        ':getRowId': "p => p.data.counterparty"}


# --------------------------------------------------------------------------
# Heatmap table view
#
# Every heatmap ships with this twin. A colour scale alone is not an
# accessible way to read a value, so the exact numbers are always one click
# away -- and they are copy-pasteable, which a heatmap never is.
# --------------------------------------------------------------------------


def matrix_grid(row_label: str, rows: list[str], cols_: list[str],
                z: np.ndarray, decimals: int = 2) -> dict:
    fmt = JS_NUM2 if decimals == 2 else JS_NUM0
    cols = [{'headerName': row_label, 'field': '_row',
             'flex': 1, 'minWidth': 90, 'pinned': 'left'}]
    cols += [_num(c, c, fmt, flex=1, width=76) for c in cols_]
    data = [{'_row': r, **{c: round(float(z[i, j]), decimals)
                           for j, c in enumerate(cols_)}}
            for i, r in enumerate(rows)]
    return _grid(cols, data) | {':getRowId': "p => p.data._row"}
