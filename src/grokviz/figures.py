"""Phase `render_report_figures`: screenshot the website's own figures for the reports, paper and README.

Local explanation: serves site/ on a free local port, opens the dashboard in headless Chromium (Playwright) with
the shared state set through the URL hash, waits until each requested figure has finished drawing, and saves a
PNG of that figure (light theme) to reports/figures/. A few whole-page screenshots go to docs/images/ for the README.
Global context: one plotting implementation for everything — the report figures are exactly what readers see on
the site, so they cannot drift apart.
Sources: Playwright for Python (https://playwright.dev/python/docs/screenshots);
http.server (https://docs.python.org/3/library/http.server.html).
"""

from __future__ import annotations  # postponed annotations

import functools  # bind directory to the handler
import logging  # module logger
import threading  # background HTTP server
from http.server import ThreadingHTTPServer  # static server

from playwright.sync_api import sync_playwright  # headless browser

from grokviz.config import REPO_ROOT  # output folders
from grokviz.context import PipelineContext  # phase argument
from grokviz.serve import NoCacheHandler  # no-cache static handler

log = logging.getLogger("grokviz.figures")

# (figure id, URL-hash state) — what the reports and the paper show.
REPORT_FIGURES = [
    ("loss-curves", "run=paper&step=40001"),
    ("progress-measures", "run=paper&step=40001"),
    ("gini-curves", "run=paper&step=40001"),
    ("ssw-curves", "run=paper&step=40001"),
    ("we-fourier-bars", "run=paper&step=40001"),
    ("wl-fourier-bars", "run=paper&step=40001"),
    ("we-fourier-heatmap", "run=paper&step=40001"),
    ("attn-grids-all", "run=paper&step=40001"),
    ("neuron-grid", "run=paper&step=40001&neuron=0"),
    ("neuron-freq-scatter", "run=paper&step=40001"),
    ("embed-circles", "run=paper&step=40001"),
    ("interference", "run=paper&step=40001&a=100&b=50"),
    ("freq-ablation-bars", "run=paper&step=40001"),
    ("patching-result", "run=paper&step=40001"),
    ("grad-alignment", "run=paper&step=40001"),
    ("wneur-fourier", "run=paper&step=40001&head=0"),
    ("neuron-freq-map", "run=paper&step=40001"),
    ("neuron-b2-map", "run=paper&step=40001"),
    ("logit-fourier-2d", "run=paper&step=40001"),
    ("cluster-ablations", "run=paper&step=40001"),
    ("adam-alignment", "run=wd1_s0&step=40000"),
    ("adam-vs-decay", "run=wd1_s0&step=40000"),
    ("update-norms", "run=wd1_s0&step=40000"),
    ("seed-test-loss", "run=wd1_s0&step=40000"),
    ("seed-key-freqs", "run=paper"),
]
PAGES = [("dashboard.html#run=paper&step=9301", "dashboard.png"), ("chapters/12-fourier-multiplication.html", "chapter.png")]


class QuietHandler(NoCacheHandler):
    """Same handler without per-request access logs (they would flood the pipeline log)."""

    def log_message(self, format, *args) -> None:  # noqa: A002 — signature defined by BaseHTTPRequestHandler
        pass


def serve(directory) -> tuple[ThreadingHTTPServer, int]:
    """Start a background static server on a free port; return (server, port)."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(directory)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def render_report_figures(ctx: PipelineContext) -> None:
    """Phase entry point: PNGs of report figures and README screenshots."""
    out = ctx.cfg.reports_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    images = REPO_ROOT / "docs" / "images"
    images.mkdir(parents=True, exist_ok=True)
    server, port = serve(ctx.cfg.site_dir)
    base = f"http://127.0.0.1:{port}"
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1400, "height": 1000}, color_scheme="light", device_scale_factor=2)
            for fid, state in REPORT_FIGURES:
                page.goto("about:blank")  # fresh document: the dashboard reads its state from the #hash only when it loads
                page.goto(f"{base}/dashboard.html?still=1#{state}")  # still: no motion, no ⤢ button (web/js/core/motion.js)
                # on paper: no interactive "How to read it" toggles, no selection outlines, no sticky control bar overlapping the figure
                page.add_style_tag(content=".fig details, .grid-mark { display: none; } .controlbar, .topbar { position: static; }")
                fig = page.locator(f"#fig-{fid}")
                fig.scroll_into_view_if_needed()
                page.wait_for_function(f"!document.querySelector('#fig-{fid} .loading')", timeout=60_000)  # figure drawn
                page.wait_for_timeout(800)  # let Plotly finish transitions
                fig.screenshot(path=str(out / f"{fid}.png"))
                log.info("figure %s → %s", fid, out / f"{fid}.png")
            page.set_viewport_size({"width": 1400, "height": 900})
            for path, name in PAGES:
                page.goto("about:blank")  # a fresh document: same-page #hash changes (and reloads) keep the old scroll position
                page.goto(f"{base}/{path}")
                page.wait_for_timeout(6000)
                page.screenshot(path=str(images / name))
                log.info("screenshot %s → %s", path, images / name)
            browser.close()
    finally:
        server.shutdown()
