"""Rates desk dashboard -- NiceGUI.

Run:   python app.py        (then open http://localhost:8080)

Why NiceGUI rather than Streamlit/Dash:
  * it does NOT re-run your whole script on every interaction, so live ticking
    data and a 12-column blotter stay smooth;
  * AG Grid and ECharts are first-class built-ins, not third-party wrappers;
  * it is a normal FastAPI app underneath, so it deploys like any web service.

Layout of this file, top to bottom, mirrors the layout of the page:
  header -> filters -> KPI tiles -> curve & P&L -> heatmaps -> risk -> blotter
  -> XVA.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Lets you launch as `python app.py` or `python dash/app.py` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nicegui import ui  # noqa: E402

import charts  # noqa: E402
import tables  # noqa: E402
from data import (CURVE_TENORS, EXPIRIES, TAILS, MarketState,  # noqa: E402
                  fmt_ccy, fmt_pct)
from theme import page_css, palette  # noqa: E402

TICK_SECONDS = 2.0


class Desk:
    """One dashboard instance. NiceGUI creates one per connected browser tab,
    so two traders can hold different filters without fighting each other."""

    def __init__(self) -> None:
        self.state = MarketState()
        self.dark = True
        self.live = False
        self.ccy: list[str] = ['EUR', 'USD', 'GBP']
        self.book = 'All'
        self.lookback = 90
        self._charts: list[tuple] = []      # (element, builder) for re-theming
        self._grids: list[tuple] = []       # (element, builder) for live data

    # -- palette ---------------------------------------------------------
    @property
    def p(self) -> dict:
        return palette(self.dark)

    # -- filtered views of the data --------------------------------------
    def positions(self):
        df = self.state.positions
        df = df[df['ccy'].isin(self.ccy)]
        if self.book != 'All':
            df = df[df['book'] == self.book]
        return df

    def pnl_history(self):
        return self.state.pnl_history.tail(self.lookback)

    # -- element registration --------------------------------------------
    def chart(self, builder, classes: str):
        el = ui.echart(builder(self.p)).classes(classes)
        self._charts.append((el, builder))
        return el

    def grid(self, builder, classes: str):
        el = ui.aggrid(builder(), theme='balham').classes(classes)
        self._grids.append((el, builder))
        return el

    def redraw(self, *, data: bool = True, theme: bool = False) -> None:
        if theme or data:
            for el, builder in self._charts:
                el.options.clear()
                el.options.update(builder(self.p))
                el.update()
        if data:
            for el, builder in self._grids:
                el.options['rowData'] = builder()['rowData']
                el.update()
            self.kpi_row.refresh()

    # -- event handlers ---------------------------------------------------
    def on_tick(self) -> None:
        if not self.live:
            return
        self.state.tick()
        self.stamp.set_text(datetime.now().strftime('%H:%M:%S'))
        self.redraw()

    def on_theme(self, value: bool) -> None:
        self.dark = value
        self.darkmode.value = value
        self.redraw(data=False, theme=True)

    def on_filter(self) -> None:
        self.redraw()

    # ====================================================================
    # UI
    # ====================================================================
    def build(self) -> None:
        ui.add_css(page_css())
        self.darkmode = ui.dark_mode(self.dark)

        self._header()
        with ui.column().classes('w-full max-w-[1680px] mx-auto px-5 pb-14 gap-5'):
            self._filters()
            self.kpi_row()
            self._curve_and_pnl()
            self._surfaces()
            self._risk()
            self._blotter()
            self._xva()
            ui.label('Synthetic data — swap data.py for your own feed. '
                     'Rich/cheap z-score convention: positive = market above '
                     'model = sell candidate.').classes('desk-sub pt-2')

        ui.timer(TICK_SECONDS, self.on_tick)

    # -- header ------------------------------------------------------------
    def _header(self) -> None:
        with ui.element('div').classes('desk-sticky w-full mb-5'):
            with ui.row().classes('w-full max-w-[1680px] mx-auto px-5 py-3 '
                                  'items-center gap-4 flex-nowrap'):
                ui.label('Rates Desk').classes('text-[17px] font-semibold')
                ui.label('EUR / USD / GBP linear + vol').classes('desk-sub')
                ui.space()

                with ui.row().classes('items-center gap-2'):
                    ui.icon('schedule').classes('text-[15px]') \
                        .style(f"color:{self.p['muted']}")
                    self.stamp = ui.label(datetime.now().strftime('%H:%M:%S')) \
                        .classes('desk-sub')

                ui.switch('Live', value=self.live,
                          on_change=lambda e: setattr(self, 'live', e.value)) \
                    .props('dense color=primary').classes('text-[12px]')

                ui.switch('Dark', value=self.dark,
                          on_change=lambda e: self.on_theme(e.value)) \
                    .props('dense color=primary').classes('text-[12px]')

    # -- filter row --------------------------------------------------------
    def _filters(self) -> None:
        """ONE filter row above everything it scopes.

        Per-chart filters are the classic dashboard mistake: the reader can no
        longer tell whether two cards are showing the same slice of the book.
        """
        with ui.row().classes('desk-card w-full items-center gap-6 flex-wrap py-3'):
            ui.label('Filters').classes('desk-title')

            ui.select(['EUR', 'USD', 'GBP'], value=self.ccy, multiple=True,
                      label='Currency',
                      on_change=lambda e: (setattr(self, 'ccy', e.value or []),
                                           self.on_filter())) \
                .props('dense outlined options-dense use-chips').classes('min-w-[210px]')

            ui.select(['All', 'Swaps', 'Vol', 'Curve', 'Basis'], value=self.book,
                      label='Book',
                      on_change=lambda e: (setattr(self, 'book', e.value),
                                           self.on_filter())) \
                .props('dense outlined options-dense').classes('min-w-[140px]')

            ui.select({30: 'Last 30d', 60: 'Last 60d', 90: 'Last 90d'},
                      value=self.lookback, label='P&L window',
                      on_change=lambda e: (setattr(self, 'lookback', e.value),
                                           self.on_filter())) \
                .props('dense outlined options-dense').classes('min-w-[150px]')

            ui.space()
            ui.label('scopes every card below').classes('desk-sub')

    # -- KPI tiles ---------------------------------------------------------
    @ui.refreshable_method
    def kpi_row(self) -> None:
        p = self.p
        pos = self.positions()
        kpis = self.state.kpis()
        # Recompute the position-derived tiles against the filtered book.
        kpis[0]['value'] = pos['day_pnl'].sum()
        kpis[2]['value'] = pos['dv01'].sum()
        kpis[3]['value'] = pos['vega'].sum()

        tone_colour = {'good': p['up_text'], 'bad': p['down_text'],
                       'neutral': p['ink2']}

        with ui.row().classes('w-full gap-4 flex-wrap'):
            for k in kpis:
                up = k['delta'] >= 0
                colour = tone_colour[k['tone']]
                with ui.element('div').classes(
                        'desk-card flex-1 min-w-[195px]'):
                    ui.label(k['label']).classes('desk-title')
                    ui.label(fmt_ccy(k['value'])).classes('desk-stat mt-1')
                    with ui.row().classes('items-center gap-1.5 mt-1 flex-nowrap'):
                        # arrow = direction, colour = good/bad, number = size.
                        # Three cues, so the meaning never rests on colour alone.
                        ui.icon('arrow_upward' if up else 'arrow_downward') \
                            .classes('text-[13px]').style(f'color:{colour}')
                        ui.label(fmt_pct(k['delta'])) \
                            .classes('text-[12px] font-semibold') \
                            .style(f'color:{colour}')
                        ui.label(k['hint']).classes('desk-sub truncate')

    # -- curve + P&L -------------------------------------------------------
    def _curve_and_pnl(self) -> None:
        with ui.row().classes('w-full gap-4 flex-nowrap max-lg:flex-wrap'):
            with ui.element('div').classes('desk-card flex-1 min-w-[380px]'):
                self._card_head('Par swap curve',
                                'EUR 6M, mid — today vs one week ago')
                self.chart(
                    lambda p: charts.curve_chart(
                        p, CURVE_TENORS, self.state.curve, self.state.curve_prev),
                    'w-full h-[300px]')

            with ui.element('div').classes('desk-card flex-1 min-w-[380px]'):
                self._card_head('Cumulative P&L attribution',
                                'rates / vol / carry, filtered window')
                self.chart(lambda p: charts.pnl_chart(p, self.pnl_history()),
                           'w-full h-[300px]')

    # -- heatmaps ----------------------------------------------------------
    def _surfaces(self) -> None:
        with ui.row().classes('w-full gap-4 flex-nowrap max-xl:flex-wrap'):
            self._heatmap_card(
                'Swaption vol surface', 'normal vol, bp — expiry x tail',
                lambda p: charts.vol_surface_heatmap(
                    p, EXPIRIES, TAILS, self.state.vol_surface),
                lambda: tables.matrix_grid('Expiry', EXPIRIES, TAILS,
                                           self.state.vol_surface, decimals=2))

            self._heatmap_card(
                'Rich / cheap wedge', 'z-score — positive = rich = sell candidate',
                lambda p: charts.wedge_heatmap(
                    p, EXPIRIES, TAILS, self.state.wedge),
                lambda: tables.matrix_grid('Expiry', EXPIRIES, TAILS,
                                           self.state.wedge, decimals=2))

    def _heatmap_card(self, title: str, sub: str, chart_builder,
                      grid_builder) -> None:
        """A heatmap plus its table-view twin.

        A continuous colour scale is not an accessible way to read a value, so
        the exact numbers are always one click away. This is a VIEW switch, not
        a filter -- it never changes which data is on screen.
        """
        with ui.element('div').classes('desk-card flex-1 min-w-[440px]'):
            with ui.row().classes('w-full items-start gap-3 flex-nowrap'):
                with ui.column().classes('gap-0'):
                    ui.label(title).classes('desk-title')
                    ui.label(sub).classes('desk-sub')
                ui.space()
                view = ui.toggle({'chart': 'Chart', 'table': 'Table'},
                                 value='chart') \
                    .props('dense unelevated no-caps size=sm toggle-color=primary')

            with ui.element('div').classes('w-full mt-2') \
                    .bind_visibility_from(view, 'value', value='chart'):
                self.chart(chart_builder, 'w-full h-[400px]')
            with ui.element('div').classes('w-full mt-2') \
                    .bind_visibility_from(view, 'value', value='table'):
                self.grid(grid_builder, 'w-full h-[400px]')

    # -- risk ladder -------------------------------------------------------
    def _risk(self) -> None:
        ladder = self.state.risk_ladder()
        with ui.row().classes('w-full gap-4 flex-nowrap max-lg:flex-wrap'):
            with ui.element('div').classes('desk-card flex-[1.6] min-w-[420px]'):
                self._card_head('DV01 ladder',
                                'sign is the story — blue long, red short')
                self.chart(
                    lambda p: charts.risk_ladder_chart(p, self.state.risk_ladder()),
                    'w-full h-[300px]')

            with ui.element('div').classes('desk-card flex-1 min-w-[340px]'):
                self._card_head('Bucketed risk', 'the same numbers, exactly')
                self.grid(lambda: tables.risk_grid(self.state.risk_ladder()),
                          'w-full h-[300px]')

    # -- blotter -----------------------------------------------------------
    def _blotter(self) -> None:
        with ui.element('div').classes('desk-card w-full'):
            self._card_head('Positions blotter',
                            'click a header to sort, the filter icon to filter')
            self.grid(lambda: tables.positions_grid(self.positions()),
                      'w-full h-[380px] mt-2')

    # -- XVA ---------------------------------------------------------------
    def _xva(self) -> None:
        with ui.row().classes('w-full gap-4 flex-nowrap max-xl:flex-wrap'):
            with ui.element('div').classes('desk-card flex-1 min-w-[400px]'):
                self._card_head('Valuation adjustments',
                                'charge stack by counterparty, thousands')
                self.chart(lambda p: charts.xva_chart(p, self.state.xva),
                           'w-full h-[300px]')

            with ui.element('div').classes('desk-card flex-[1.4] min-w-[460px]'):
                self._card_head('XVA detail', 'CVA / DVA / FVA / ColVA')
                self.grid(lambda: tables.xva_grid(self.state.xva),
                          'w-full h-[300px] mt-2')

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _card_head(title: str, sub: str) -> None:
        with ui.column().classes('gap-0 mb-1'):
            ui.label(title).classes('desk-title')
            ui.label(sub).classes('desk-sub')


@ui.page('/')
def index() -> None:
    ui.page_title('Rates Desk')
    Desk().build()


if __name__ in {'__main__', '__mp_main__'}:
    ui.run(title='Rates Desk', port=8080, favicon='📈',
           dark=True, reload=False, show=False)
