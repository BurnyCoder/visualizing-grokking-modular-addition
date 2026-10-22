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
    page.evaluate("document.querySelector('#fig-loss-curves').scrollIntoView()")
    page.wait_for_timeout(300)  # the visibility observer reports on the next frames
    top = page.evaluate(SETTLED)["loss-curves"]
    for i in (10, 20, 30):  # three different checkpoints via the control bar's slider
        page.evaluate("i => { const s = document.querySelector('#ctl-player input[type=range]'); s.value = String(i); s.dispatchEvent(new Event('input')); }", i)
        renders = page.evaluate(SETTLED)
    assert renders["we-svals"] == seen, "an offscreen figure redrew"
    assert renders["loss-curves"] == top + 3, "a visible figure missed a step"
    page.evaluate("document.querySelector('#fig-we-svals').scrollIntoView()")
    page.wait_for_timeout(300)
    assert page.evaluate(SETTLED)["we-svals"] == seen + 1, "a figure scrolled back did not redraw exactly once"


MEDIAN_FRAME = """async () => { const d = []; let last = performance.now(); const t0 = last;
  await new Promise((done) => { const f = (t) => { d.push(t - last); last = t; if (t - t0 < 2000) requestAnimationFrame(f); else done(); }; requestAnimationFrame(f); });
  d.sort((a, b) => a - b); return d[d.length >> 1]; }"""  # median animation-frame interval over 2 s, in ms


def test_dashboard_playback(browser_page):
    """▶ plays the selected run from the start of its range (web/js/core/player.js): about 10 checkpoints per second
    at 1×, smooth frames, and the URL is written only once playback stops."""
    page, base = browser_page
    page.goto("about:blank")
    page.goto(f"{base}/dashboard.html")
    page.wait_for_function("document.querySelector('#ctl-player .pb-label')?.textContent.startsWith('step')")
    page.evaluate(SETTLED)
    page.click("#ctl-player .pb-play")
    median = page.evaluate(MEDIAN_FRAME)
    assert "step=" not in page.url, "the URL hash was rewritten while playing"
    page.click("#ctl-player .pb-play")  # pause
    step = int(page.inner_text("#ctl-player .pb-label").split()[1].replace(",", ""))
    assert 1000 <= step <= 3500, f"after ~2 s at 1× the paper run should be ~20 checkpoints (2,000 steps) in, got {step}"
    assert median <= 50, f"median frame {median:.0f} ms while playing"
    page.wait_for_timeout(400)  # hash writes are debounced by 150 ms
    assert f"step={step}" in page.url
    # with the model-driven panels on screen (grids recomputed per weight snapshot, tables built in the web worker)
    page.evaluate("document.querySelector('#fig-neuron-grid').scrollIntoView({block: 'start'})")
    page.wait_for_function("!document.querySelector('#fig-neuron-grid .loading')", timeout=30_000)
    page.evaluate(SETTLED)
    page.click("#ctl-player .pb-play")
    median = page.evaluate(MEDIAN_FRAME)
    page.click("#ctl-player .pb-play")
    assert median <= 50, f"median frame {median:.0f} ms while playing with model panels on screen"
    assert page.evaluate("performance.getEntriesByType('resource').some((e) => e.name.includes('model/worker.js'))"), "model tables not built in the worker"


def test_chapter_animation_scroll_play_and_reduced_motion(browser_page):
    """An {{anim:id}} figure plays while on screen, pauses when scrolled away, never moves the page's own step, and
    under prefers-reduced-motion shows its still poster frame without autoplay; ⤢ opens it in a dialog that Esc closes."""
    page, base = browser_page
    fig = "#fig-pcorrect-grid"
    page.goto("about:blank")
    page.goto(f"{base}/chapters/08-grokking.html")
    page.evaluate(f"document.querySelector('{fig}').scrollIntoView({{block: 'center'}})")
    page.wait_for_function(f"document.querySelector('{fig} .pb-play')?.textContent === '❚❚'", timeout=20_000)
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_function(f"document.querySelector('{fig} .pb-play').textContent === '▶'", timeout=5_000)
    assert "step=" not in page.url, "the animation moved the page's step"
    page.click(f"{fig} .fig-expand")
    assert page.locator(f".fig-modal {fig}").count() == 1 and page.evaluate("document.activeElement.classList.contains('fig-modal-close')")
    page.keyboard.press("Escape")
    assert page.locator(".fig-modal").count() == 0
    page.emulate_media(reduced_motion="reduce")
    try:
        page.goto("about:blank")
        page.goto(f"{base}/chapters/08-grokking.html")
        page.evaluate(f"document.querySelector('{fig}').scrollIntoView({{block: 'center'}})")
        page.wait_for_function(f"document.querySelector('{fig} .pb-label')?.textContent.startsWith('step')", timeout=20_000)
        poster = page.inner_text(f"{fig} .pb-label")
        page.wait_for_timeout(1500)
        assert page.inner_text(f"{fig} .pb-play") == "▶" and page.inner_text(f"{fig} .pb-label") == poster, "autoplay under reduced motion"
    finally:
        page.emulate_media(reduced_motion="no-preference")


def test_hero_and_tour(browser_page):
    """The hero's ▶ plays the run through grokking on the page player, with blended frames labelled while playing; the
    tour plays its first step's range and its text carries bound numbers."""
    page, base = browser_page
    hero = "#fig-hero-grokking"
    page.goto("about:blank")
    page.goto(f"{base}/dashboard.html")
    page.wait_for_function(f"document.querySelectorAll('{hero} .fig-note').length === 3 && !document.querySelector('{hero} .loading')", timeout=30_000)
    page.click(f"{hero} .hero-play")
    page.wait_for_timeout(1500)
    assert page.inner_text(f"{hero} .hero-play").startswith("❚❚")
    assert "blended" in page.inner_text(f"{hero} .hero-tile:nth-child(2) .fig-note"), "no blending label while playing"
    page.click(f"{hero} .hero-play")
    page.evaluate(SETTLED)
    assert "stored frame" in page.inner_text(f"{hero} .hero-tile:nth-child(2) .fig-note"), "paused view must be a stored frame"
    page.click("#ctl-tour")
    page.wait_for_timeout(800)
    assert page.locator(f"{hero}.tour-focus").count() == 1 and page.inner_text("#ctl-player .pb-play") == "❚❚"
    page.click("#tour .tour-next")
    page.evaluate(SETTLED)
    assert page.locator("#tour .tour-text [data-num='task.n_train']").count() == 1
    assert page.inner_text("#ctl-player .pb-label").startswith("step 1,601")
    page.click("#tour .tour-close")


def test_new_panels_show_the_trained_network(browser_page):
    """At the end of the paper's run the forward-flow diagram answers 100 + 50 = 37 (mod 113) correctly, the sigmoid
    panel reports its fits, and alpha-growth recognises k = 14 as a key frequency."""
    import re

    page, base = browser_page
    page.goto("about:blank")
    page.goto(f"{base}/dashboard.html#step=40001&a=100&b=50&freq=14")
    for fid in ("forward-flow", "sigmoid-linearisation", "alpha-growth"):
        page.evaluate(f"document.querySelector('#fig-{fid}').scrollIntoView()")
        page.wait_for_function(f"document.querySelector('#fig-{fid} .fig-note')?.textContent.length > 20", timeout=30_000)
    page.evaluate(SETTLED)
    flow = page.evaluate("document.querySelector('#fig-forward-flow svg').textContent")  # SVG: no innerText
    assert "answer 37" in flow and "✓ correct" in flow, flow[-200:]
    assert re.search(r"σ\(s₀ − s₁\) explains \d+\.\d %", page.inner_text("#fig-sigmoid-linearisation .fig-note"))
    assert "a key frequency of this run" in page.inner_text("#fig-alpha-growth .fig-body > .fig-note")
