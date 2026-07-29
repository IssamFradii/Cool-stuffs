"""Freeze ANY NiceGUI app into one standalone .html file. No edits to the app.

    python export.py app.py                 -> app.html
    python export.py app.py -o desk.html    -> pick the name
    python export.py --url http://localhost:8080   -> freeze an already-running app

The output is a single self-contained file. Double-click it on any machine --
no Python, no install, no server, works offline.

------------------------------------------------------------------------------
HOW IT WORKS  (worth reading once -- it explains what survives and what doesn't)

NiceGUI puts the ENTIRE element tree, with every widget's configuration, into
the HTML it serves, as one JSON blob. That includes the full ECharts options of
every chart and the full AG Grid options of every table. So we do not have to
guess anything or re-implement your app -- we just read it out.

The exporter then:
  1. starts your app and waits for it to print its URL,
  2. reads that JSON blob to collect every chart and grid config,
  3. opens the page in a headless browser and captures what the browser
     actually produced -- the rendered DOM plus every CSS rule (Quasar, the
     Tailwind classes generated at runtime, and anything you added with
     `ui.add_css`), so the layout is identical rather than re-guessed,
  4. inlines the fonts, icons and images it finds as data URIs,
  5. writes the frozen DOM together with the ECharts and AG Grid libraries and
     a short script that re-creates every chart and grid from step 2.

WHAT STILL WORKS in the exported file -- all of it runs in the browser anyway:
  * every chart: tooltips, crosshairs, legend toggles, heatmap colour-scale
    dragging, zoom, resize
  * every table: sorting, column filters, resizing, pinned columns, scrolling
  * all layout, fonts, icons, colours, light or dark exactly as captured

WHAT DOES NOT: anything wired to a Python callback -- buttons, switches,
dropdowns, `ui.timer`. Those need Python running; that is the whole point of a
server. The exporter greys them out and labels them, rather than leaving dead
controls that look clickable.

If you need one of those controls to keep working offline, that logic has to
live in JavaScript instead of Python. `export_handmade.py` in this folder is
the worked example -- it keeps the filters and toggles alive by shipping them
as ~150 lines of JS.
------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE / 'vendor'

# Widgets that need a Python round-trip. Frozen, they would look clickable and
# do nothing, which is worse than clearly showing they are inert.
INERT_SELECTORS = [
    '.q-btn', '.q-toggle', '.q-checkbox', '.q-radio', '.q-slider', '.q-input',
    '.q-select', '.q-btn-toggle', '.q-tab', '.q-uploader', '.q-editor',
    '.nicegui-select', '.nicegui-joystick', '.nicegui-codemirror',
]


# --------------------------------------------------------------------------
# 1. Start the app (or reuse a running one)
# --------------------------------------------------------------------------
def launch(app_path: Path, timeout: float = 90.0) -> tuple[subprocess.Popen, str]:
    """Run the app and wait for the URL NiceGUI prints on startup."""
    proc = subprocess.Popen(
        [sys.executable, str(app_path)], cwd=str(app_path.parent),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='replace', bufsize=1,
    )
    deadline = time.time() + timeout
    banner = []
    while time.time() < deadline:
        line = proc.stdout.readline() if proc.stdout else ''
        if not line:
            if proc.poll() is not None:
                raise RuntimeError('app exited before serving:\n' + ''.join(banner))
            continue
        banner.append(line)
        m = re.search(r'https?://(?:localhost|127\.0\.0\.1):(\d+)', line)
        if m:
            url = f'http://127.0.0.1:{m.group(1)}'
            _wait_for(url, deadline)
            return proc, url
    proc.kill()
    raise RuntimeError('timed out waiting for the app to start:\n' + ''.join(banner))


def _wait_for(url: str, deadline: float) -> None:
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2).read(1)
            return
        except Exception:
            time.sleep(0.3)
    raise RuntimeError(f'{url} never became reachable')


# --------------------------------------------------------------------------
# 2. Read the widget configs out of the served HTML
# --------------------------------------------------------------------------
def read_elements(url: str, route: str) -> dict:
    html = urllib.request.urlopen(urllib.parse.urljoin(url, route)).read().decode('utf-8')
    m = re.search(r'parseElements\(String\.raw`(.*?)`\)', html, re.S)
    if not m:
        raise RuntimeError('no NiceGUI element tree in the page -- is this a NiceGUI app?')
    return json.loads(_unescape(m.group(1)))


def _unescape(raw: str) -> str:
    """Undo the escaping NiceGUI applies to the element blob.

    The JSON is embedded inside a JS template literal, so NiceGUI escapes the
    characters that would break out of it. This mirrors `parseElements` in
    nicegui.js exactly, including the order -- '&amp;' has to come last or
    '&amp;gt;' would decode twice.
    """
    for token, char in (('&#36;', '$'), ('&#96;', '`'), ('&gt;', '>'),
                        ('&lt;', '<'), ('&amp;', '&')):
        raw = raw.replace(token, char)
    return raw


def collect_widgets(elements: dict) -> tuple[dict, dict]:
    """-> ({dom_id: echarts_options}, {dom_id: aggrid_options})"""
    charts, grids = {}, {}
    for eid, el in elements.items():
        tag = el.get('tag') or ''
        opts = (el.get('props') or {}).get('options')
        if opts is None:
            continue
        if tag == 'nicegui-echart':
            charts[f'c{eid}'] = opts
        elif tag == 'nicegui-aggrid':
            grids[f'c{eid}'] = opts
    return charts, grids


# --------------------------------------------------------------------------
# 3. Capture the rendered page
# --------------------------------------------------------------------------
CAPTURE_JS = r"""
(ids) => {
  // Every CSS rule the browser ended up with: Quasar, NiceGUI, the Tailwind
  // utilities generated at runtime, and anything from ui.add_css. Reading it
  // back from the browser means we never have to guess which files to fetch.
  // url(...) inside a stylesheet is relative to THAT stylesheet, not to the
  // page -- fonts live next to fonts.css, several folders down. Resolve them
  // here, where each sheet's own href is known, so Python only ever sees
  // absolute URLs it can fetch.
  const absolutise = (text, base) =>
    text.replace(/url\((["']?)([^)"']+)\1\)/g, (whole, q, ref) => {
      if (/^(data:|https?:|#)/.test(ref)) return whole;
      try { return 'url(' + q + new URL(ref, base).href + q + ')'; }
      catch (e) { return whole; }
    });

  const dump = (sheet) => {
    let out = '';
    let rules;
    try { rules = sheet.cssRules; } catch (e) { return ''; }
    const base = sheet.href || document.baseURI;
    for (const r of rules) {
      if (r.styleSheet) out += dump(r.styleSheet) + '\n';   // @import
      else out += absolutise(r.cssText, base) + '\n';
    }
    return out;
  };
  const css = Array.from(document.styleSheets).map(dump).join('\n');

  // Empty the widget containers -- their content is canvas pixels and
  // generated table DOM, which we rebuild from the captured configs instead.
  for (const id of ids) {
    const el = document.getElementById(id);
    if (el) { el.innerHTML = ''; el.removeAttribute('_echarts_instance_'); }
  }

  // Drop NiceGUI's own scripts. They boot Vue, Quasar and the websocket from
  // ES modules served by the app; re-running them in a file:// page throws and
  // there is nothing for them to do -- the DOM they would have built is
  // already captured below.
  document.querySelectorAll('script, link[rel="modulepreload"]')
    .forEach(function (n) { n.remove(); });

  return {
    css: css,
    bodyClass: document.body.className,
    bodyHTML: document.body.innerHTML,
    title: document.title,
    dark: document.body.classList.contains('body--dark'),
  };
}
"""


def capture(url: str, route: str, widget_ids: list[str], wait_ms: int) -> dict:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit('This needs Playwright:\n'
                         '    pip install playwright\n'
                         '    python -m playwright install chromium')

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={'width': 1680, 'height': 1000})
        page.goto(urllib.parse.urljoin(url, route), wait_until='networkidle')
        page.wait_for_timeout(wait_ms)          # let charts and grids settle
        result = page.evaluate(CAPTURE_JS, widget_ids)
        browser.close()
    return result


# --------------------------------------------------------------------------
# 4. Inline fonts / icons / images so the file works offline
# --------------------------------------------------------------------------
EXTRA_MIME = {'.woff2': 'font/woff2', '.woff': 'font/woff', '.ttf': 'font/ttf',
              '.otf': 'font/otf', '.eot': 'application/vnd.ms-fontobject',
              '.svg': 'image/svg+xml'}


def inline_assets(css: str, body: str, base_url: str) -> tuple[str, str]:
    """Pull every font, icon and image into the file as a data: URI."""
    cache: dict[str, str | None] = {}
    origin = '{0.scheme}://{0.netloc}'.format(urllib.parse.urlsplit(base_url))

    def data_uri(ref: str) -> str | None:
        if ref.startswith(('data:', '#')):
            return None
        # Only fetch from the app itself. A CDN link is left alone: it needs
        # the network either way, and mirroring third-party assets into the
        # file is not ours to decide.
        full = urllib.parse.urljoin(base_url, ref)
        if not full.startswith(origin):
            return None
        if full in cache:
            return cache[full]
        try:
            raw = urllib.request.urlopen(full, timeout=20).read()
        except Exception:
            cache[full] = None
            return None
        ext = Path(urllib.parse.urlsplit(full).path).suffix.lower()
        mime = (EXTRA_MIME.get(ext) or mimetypes.guess_type(full)[0]
                or 'application/octet-stream')
        uri = f'data:{mime};base64,{base64.b64encode(raw).decode()}'
        cache[full] = uri
        return uri

    def css_sub(m: re.Match) -> str:
        quote, ref = m.group(1), m.group(2)
        uri = data_uri(ref)
        return m.group(0) if uri is None else f'url({quote}{uri}{quote})'

    css = re.sub(r'url\((["\']?)([^)"\']+)\1\)', css_sub, css)

    def img_sub(m: re.Match) -> str:
        uri = data_uri(m.group(2))
        return m.group(0) if uri is None else f'{m.group(1)}="{uri}"'

    body = re.sub(r'\b(src|href)="([^"]+)"', img_sub, body)
    return css, body


# --------------------------------------------------------------------------
# 5. Emit the standalone file
# --------------------------------------------------------------------------
def to_js(obj) -> str:
    """JSON, except keys starting with ':' carry raw JavaScript.

    NiceGUI uses that prefix for options that are functions rather than data --
    an AG Grid valueFormatter, an ECharts tooltip builder. json.dumps would
    quote them into dead strings, so we emit a JS literal instead.
    """
    if isinstance(obj, dict):
        parts = []
        for k, v in obj.items():
            if isinstance(k, str) and k.startswith(':'):
                parts.append(f'{json.dumps(k[1:])}:{v}')
            else:
                parts.append(f'{json.dumps(str(k))}:{to_js(v)}')
        return '{' + ','.join(parts) + '}'
    if isinstance(obj, (list, tuple)):
        return '[' + ','.join(to_js(v) for v in obj) + ']'
    if isinstance(obj, float) and obj != obj:
        return 'null'
    return json.dumps(obj)


BOOT_JS = r"""
(function () {
  agGrid.ModuleRegistry.registerModules([agGrid.AllCommunityModule]);
  var themes = { quartz: agGrid.themeQuartz, balham: agGrid.themeBalham,
                 material: agGrid.themeMaterial, alpine: agGrid.themeAlpine };
  var dark = document.body.classList.contains('body--dark');
  var live = [];

  Object.keys(CHARTS).forEach(function (id) {
    var el = document.getElementById(id);
    if (!el) return;
    var c = echarts.init(el, null, { renderer: 'canvas' });
    c.setOption(CHARTS[id]);
    live.push(c);
  });

  Object.keys(GRIDS).forEach(function (id) {
    var el = document.getElementById(id);
    if (!el) return;
    var o = GRIDS[id];
    el.setAttribute('data-ag-theme-mode', dark ? 'dark' : 'light');
    o.theme = (themes[o.theme] || agGrid.themeQuartz)
      .withPart(agGrid.colorSchemeVariable);
    agGrid.createGrid(el, o);
  });

  window.addEventListener('resize', function () {
    live.forEach(function (c) { c.resize(); });
  });
})();
"""

SHELL = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="__SCHEME__">
<title>__TITLE__</title>
<style>
__CSS__
</style>
<style>
/* Controls that needed the Python server are inert here. Showing that is
   kinder than a button that silently does nothing. */
__INERT__ {
  opacity: .5 !important;
  pointer-events: none !important;
  cursor: not-allowed !important;
}
</style>
</head>
<body class="__BODYCLASS__">
__BODY__
<script>__ECHARTS_JS__</script>
<script>__AGGRID_JS__</script>
<script>
var CHARTS = __CHARTS__;
var GRIDS = __GRIDS__;
__BOOT__
</script>
</body>
</html>
"""


def build(cap: dict, charts: dict, grids: dict, mark_inert: bool) -> str:
    html = SHELL
    for token, value in (
        ('__TITLE__', cap['title'] or 'Dashboard'),
        ('__SCHEME__', 'dark' if cap['dark'] else 'light'),
        ('__BODYCLASS__', cap['bodyClass']),
        ('__CSS__', cap['css']),
        ('__INERT__', ', '.join(INERT_SELECTORS) if mark_inert else '.__none__'),
        ('__BODY__', cap['bodyHTML']),
        ('__ECHARTS_JS__', (VENDOR / 'echarts.min.js').read_text(encoding='utf-8')),
        ('__AGGRID_JS__', (VENDOR / 'ag-grid-community.min.js').read_text(encoding='utf-8')),
        ('__CHARTS__', to_js(charts)),
        ('__GRIDS__', to_js(grids)),
        ('__BOOT__', BOOT_JS),
    ):
        html = html.replace(token, value)
    return html


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description='Freeze a NiceGUI app into one standalone HTML file.',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('app', nargs='?', default='app.py',
                    help='the NiceGUI script to freeze (default: app.py)')
    ap.add_argument('-o', '--out', help='output file (default: <app>.html)')
    ap.add_argument('--url', help='freeze an app that is already running '
                                  'instead of starting one')
    ap.add_argument('--route', default='/', help='which page to freeze')
    ap.add_argument('--wait', type=int, default=3500,
                    help='ms to let the page settle before capturing')
    ap.add_argument('--keep-controls', action='store_true',
                    help='do NOT grey out the controls that need Python')
    args = ap.parse_args()

    proc = None
    try:
        if args.url:
            url = args.url.rstrip('/')
            print(f'using running app at {url}')
        else:
            app_path = Path(args.app).resolve()
            if not app_path.exists():
                raise SystemExit(f'no such file: {app_path}')
            print(f'starting {app_path.name} ...')
            proc, url = launch(app_path)
            print(f'  serving on {url}')

        elements = read_elements(url, args.route)
        charts, grids = collect_widgets(elements)
        print(f'found {len(charts)} chart(s) and {len(grids)} table(s) '
              f'in {len(elements)} elements')

        print('capturing the rendered page ...')
        cap = capture(url, args.route, list(charts) + list(grids), args.wait)

        print('inlining fonts and images ...')
        cap['css'], cap['bodyHTML'] = inline_assets(
            cap['css'], cap['bodyHTML'], urllib.parse.urljoin(url, args.route))

        out = Path(args.out) if args.out else Path(args.app).with_suffix('.html')
        out.write_text(build(cap, charts, grids, not args.keep_controls),
                       encoding='utf-8')
        print(f'\n{out}  ({out.stat().st_size / 1e6:.1f} MB)')
        print('Double-click it. No Python needed on that machine.')
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == '__main__':
    main()
