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
