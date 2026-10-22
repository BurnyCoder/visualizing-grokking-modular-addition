"""Every page of the built site loads in a real browser, draws all its figures and logs no errors (milestone M9/M16b);
offscreen figures do not redraw.

Global context: figures are ES modules that fetch binary data and draw lazily, so a broken array name, a JavaScript
exception or a failed fetch only shows up in a browser. This drives headless Chromium (Playwright) over every page,
scrolls so every lazy figure is created, waits until none says "Loading…", and fails on console errors, uncaught
exceptions or figures that report "Could not draw this". It also records the dashboard's first contentful paint
against the plan's 3 s budget, and checks the render pipeline's visibility gate through the window.grokviz.anim hooks
(web/js/core/hooks.js). Skipped until the site (with data) has been built.
Sources: Playwright for Python events (https://playwright.dev/python/docs/api/class-page#page-event-console);
Paint Timing API (https://developer.mozilla.org/docs/Web/API/PerformancePaintTiming).
"""

import pytest  # markers, skip, fixtures

from grokviz.config import load_config  # where site/ lives
from grokviz.figures import serve  # quiet background static server (also used for the report figures)

SITE = load_config().site_dir
pytestmark = [pytest.mark.browser, pytest.mark.skipif(not (SITE / "data" / "manifest.json").exists(), reason="site not built with data")]
SETTLED = "async () => { await window.grokviz.anim.settled(); return window.grokviz.anim.stats().renders; }"  # render counts once idle


def pages() -> list[str]:
    """Every HTML page of the built site, relative to its root."""
    return sorted(p.relative_to(SITE).as_posix() for p in SITE.rglob("*.html") if "data" not in p.relative_to(SITE).parts)


@pytest.fixture(scope="module")
def browser_page():
    """(page, base URL): one headless Chromium page on a quiet local server of the built site, shared by this module."""
    from playwright.sync_api import sync_playwright

    server, port = serve(SITE)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            yield browser.new_page(viewport={"width": 1400, "height": 900}), f"http://127.0.0.1:{port}"
            browser.close()
    finally:
        server.shutdown()


def test_every_page_draws_without_errors(browser_page):
    """No console error, page error or failed figure on any page; dashboard first paint under 3 s."""
    page, base = browser_page
    problems: list[str] = []
    page.on("console", lambda m: problems.append(f"{page.url}: console {m.text}") if m.type == "error" else None)
    page.on("pageerror", lambda e: problems.append(f"{page.url}: {e}"))
    for path in pages():
        page.goto(f"{base}/{path}")
        y = 0
        while y < page.evaluate("document.body.scrollHeight"):  # the page grows as figures draw, so re-measure
            page.evaluate(f"window.scrollTo(0, {y})")  # bring every lazy figure near the viewport
            page.wait_for_timeout(150)
            y += 700
        try:  # every lazy figure has replaced its "Loading…" placeholder
            page.wait_for_function("!document.querySelector('.fig .loading, .widget .loading')", timeout=30_000)
        except Exception:  # report which ones are stuck instead of a bare timeout
            stuck = page.evaluate("[...document.querySelectorAll('.fig, .widget')].filter(f => f.querySelector('.loading')).map(f => f.id || f.className)")
            problems.append(f"{path}: still loading after 30 s: {stuck}")
        failed = page.evaluate("[...document.querySelectorAll('.fig, .widget')].filter(f => f.textContent.includes('Could not draw')).map(f => f.id)")
        problems += [f"{path}: figure {f} could not draw" for f in failed]
    page.goto(f"{base}/dashboard.html")
    page.wait_for_load_state("load")
    fcp = page.evaluate("performance.getEntriesByType('paint').find(e => e.name === 'first-contentful-paint')?.startTime ?? 0")
    assert not problems, "\n".join(problems[:30])
    assert fcp < 3000, f"dashboard first contentful paint {fcp:.0f} ms"


def test_offscreen_figures_wait_until_visible(browser_page):
    """While the step changes, a figure scrolled out of view does not redraw (figures/common.js watch); it redraws
    exactly once when it is scrolled back, and visible figures follow every change."""
    page, base = browser_page
    page.goto(f"{base}/dashboard.html")
    page.wait_for_function("window.grokviz?.anim")
    page.evaluate("document.querySelector('#fig-we-svals').scrollIntoView()")
    page.wait_for_function("document.querySelector('#fig-we-svals .js-plotly-plot')")  # created and drawn once
    seen = page.evaluate(SETTLED)["we-svals"]
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(300)  # the visibility observer reports on the next frames
    top = page.evaluate(SETTLED)["loss-curves"]
    for i in (10, 20, 30):  # three different checkpoints via the control bar's slider
        page.evaluate("i => { const s = document.getElementById('ctl-step'); s.value = String(i); s.dispatchEvent(new Event('input')); }", i)
        renders = page.evaluate(SETTLED)
    assert renders["we-svals"] == seen, "an offscreen figure redrew"
    assert renders["loss-curves"] == top + 3, "a visible figure missed a step"
    page.evaluate("document.querySelector('#fig-we-svals').scrollIntoView()")
    page.wait_for_timeout(300)
    assert page.evaluate(SETTLED)["we-svals"] == seen + 1, "a figure scrolled back did not redraw exactly once"
