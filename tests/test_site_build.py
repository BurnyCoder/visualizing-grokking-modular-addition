"""The website builds and satisfies its reading contract (milestone M9).

Global context: the build enforces that every term/symbol is defined before use (in chapter order), every figure,
widget, quote, citation and number referenced by the content exists, glossary entries are one sourced sentence and
quotes respect the length policy. This test runs the real build into a temporary folder with placeholder numbers
(the real ones come from the analysis), so it needs neither the GPU nor the downloaded data.
"""

import json  # placeholder summary
import re  # find referenced number keys

from grokviz.config import REPO_ROOT  # content location


def test_site_builds_and_content_rules_hold(tmp_path, monkeypatch):
    """Build into tmp; no content errors; key pages and a definition are present."""
    keys = set()
    for path in (REPO_ROOT / "content").rglob("*.*"):
        if path.suffix in (".md", ".yaml"):
            keys |= set(re.findall(r"\{\{num:([\w.-]+)\}\}", path.read_text(encoding="utf-8")))
    (tmp_path / "data").mkdir()
    summary = {k: {"value": 1.0, "fmt": ".1f"} for k in keys}
    summary["rchecks.rows"] = {"value": [], "fmt": "str"}
    (tmp_path / "data" / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setenv("GROKVIZ_SITE_DIR", str(tmp_path))
    from grokviz.site_build import build_site  # imported after the env override takes effect in load_config

    build_site()  # raises RuntimeError listing every content violation
    assert (tmp_path / "dashboard.html").exists() and (tmp_path / "glossary.html").exists()
    ch1 = (tmp_path / "chapters" / "01-clock-arithmetic.html").read_text(encoding="utf-8")
    assert 'data-term="modulo"' in ch1 and "<dfn" in ch1
    glossary = json.loads((tmp_path / "assets" / "glossary.json").read_text(encoding="utf-8"))
    assert "grokking" in glossary["terms"] and "WE" in glossary["symbols"]


def test_scientific_numbers_use_superscript_exponents():
    """Prose numbers like the paper's 2.41e-7 loss read as 2.41×10⁻⁷; exponent 0 shows the bare mantissa."""
    from grokviz.formatting import fmt_number

    assert fmt_number({"value": 2.41e-7, "fmt": "sci"}) == "2.41×10⁻⁷"
    assert fmt_number({"value": 3.2e5, "fmt": "sci"}) == "3.20×10⁵"
    assert fmt_number({"value": 5.27, "fmt": "sci"}) == "5.27"
    assert fmt_number({"value": 0.9624, "fmt": "pct"}) == "96.2 %"


def test_figures_use_gated_watch():
    """Every figure subscribes through watch(body, keys, fn): the old watch(keys, fn) skipped the visibility gate and
    the latest-wins coalescing (web/js/figures/common.js), so offscreen panels re-rendered and frames landed out of order."""
    old = [f"{p.name}:{i}" for p in (REPO_ROOT / "web" / "js").rglob("*.js")
           for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if re.search(r"\bwatch\(\s*\[", line)]
    assert not old, f"two-argument watch( calls left: {old}"


def test_anim_and_scene_macros():
    """{{anim:id}} renders the figure as a scoped, scroll-played animation configured by its `anim:` block in
    figures.yaml; {{scene:id}} renders a scrollytelling marker from content/scenes.yaml; unknown ids and figures
    without an `anim:` block are build errors."""
    from grokviz import site_build as sb

    sb._md = sb.markdown()
    figs = {"f": {"title": "F", "what": "w", "anim": {"run": "wd1_s0", "range": "grokking", "poster": 15400}}, "g": {"title": "G", "what": "w"}}
    b = sb.Build(glossary={}, symbols={}, quotes={}, refs={}, figures=figs, summary={}, js_figures={"f", "g"}, js_widgets=set(),
                 scenes={"s1": {"play": {"from": 1000, "to": 15000}}})
    html = sb.expand("{{anim:f}}", b, "p")
    assert "data-scope" in html and 'data-run="wd1_s0"' in html and 'data-step="15400"' in html and 'class="anim-bar"' in html
    anim = json.loads(re.search(r"data-anim='([^']*)'", html).group(1))
    assert anim["range"] == "grokking"
    scene = sb.expand("{{scene:s1}}", b, "p")
    assert scene.startswith('<div class="scene"') and json.loads(re.search(r"data-scene='([^']*)'", scene).group(1))["play"]["to"] == 15000
    assert "fig-expand" in sb.expand("{{fig:g}}", b, "p"), "every figure has an enlarge button"
    assert not b.errors
    sb.expand("{{anim:g}} {{anim:missing}} {{scene:missing}}", b, "p")
    assert len(b.errors) == 3, b.errors


def test_tour_steps_bind_numbers_and_are_checked():
    """Tour text goes through the macro expander (numbers come from summary.json, not typed by hand); a step must name
    a dashboard panel, and a play range needs from < to."""
    from grokviz import site_build as sb

    sb._md = sb.markdown()
    b = sb.Build(glossary={}, symbols={}, quotes={}, refs={}, figures={}, summary={"task.n_train": {"value": 3830, "fmt": "int"}},
                 js_figures=set(), js_widgets=set(), root="./")
    raw = [{"title": "T", "panel": "loss-curves", "play": {"from": 0, "to": 100}, "text": "{{num:task.n_train}} sums"},
           {"title": "U", "panel": "nope", "text": "x"}, {"title": "V", "panel": "loss-curves", "play": {"from": 9, "to": 1}, "text": "y"}]
    steps = sb.tour_steps(raw, b, {"loss-curves"})
    assert 'data-num="task.n_train"' in steps[0]["text"] and steps[0]["play"] == {"from": 0, "to": 100}
    assert len(b.errors) == 2, b.errors


def test_figure_data_keys_cannot_be_prose_numbers():
    """Summary entries with fmt "json" (hypothesis tables for the JavaScript panels) are rejected in {{num:}}."""
    from grokviz import site_build as sb

    sb._md = sb.markdown()
    b = sb.Build(glossary={}, symbols={}, quotes={}, refs={}, figures={}, js_figures=set(), js_widgets=set(),
                 summary={"hypotheses.per_run": {"value": {"wd1_s0": {}}, "fmt": "json"}, "task.n_train": {"value": 3830, "fmt": "int"}})
    assert "3,830" in sb.expand("{{num:task.n_train}}", b, "p") and not b.errors
    sb.expand("{{num:hypotheses.per_run}}", b, "p")
    assert len(b.errors) == 1
