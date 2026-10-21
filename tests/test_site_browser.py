"""Every page of the built site loads in a real browser, draws all its figures and logs no errors (milestone M9/M16b).

Global context: figures are ES modules that fetch binary data and draw lazily, so a broken array name, a JavaScript
exception or a failed fetch only shows up in a browser. This drives headless Chromium (Playwright) over every page,
scrolls so every lazy figure is created, waits until none says "Loading…", and fails on console errors, uncaught
exceptions or figures that report "Could not draw this". It also records the dashboard's first contentful paint
against the plan's 3 s budget. Skipped until the site (with data) has been built.
Sources: Playwright for Python events (https://playwright.dev/python/docs/api/class-page#page-event-console);
Paint Timing API (https://developer.mozilla.org/docs/Web/API/PerformancePaintTiming).
"""

import pytest  # markers, skip

from grokviz.config import load_config  # where site/ lives
from grokviz.figures import serve  # quiet background static server (also used for the report figures)

SITE = load_config().site_dir
pytestmark = [pytest.mark.browser, pytest.mark.skipif(not (SITE / "data" / "manifest.json").exists(), reason="site not built with data")]


def pages() -> list[str]:
    """Every HTML page of the built site, relative to its root."""
    return sorted(p.relative_to(SITE).as_posix() for p in SITE.rglob("*.html") if "data" not in p.relative_to(SITE).parts)


def test_every_page_draws_without_errors():
    """No console error, page error or failed figure on any page; dashboard first paint under 3 s."""
    from playwright.sync_api import sync_playwright

    server, port = serve(SITE)
    problems: list[str] = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1400, "height": 900})
            page.on("console", lambda m: problems.append(f"{page.url}: console {m.text}") if m.type == "error" else None)
            page.on("pageerror", lambda e: problems.append(f"{page.url}: {e}"))
            for path in pages():
                page.goto(f"http://127.0.0.1:{port}/{path}")
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
            page.goto(f"http://127.0.0.1:{port}/dashboard.html")
            page.wait_for_load_state("load")
            fcp = page.evaluate("performance.getEntriesByType('paint').find(e => e.name === 'first-contentful-paint')?.startTime ?? 0")
            browser.close()
    finally:
        server.shutdown()
    assert not problems, "\n".join(problems[:30])
    assert fcp < 3000, f"dashboard first contentful paint {fcp:.0f} ms"
