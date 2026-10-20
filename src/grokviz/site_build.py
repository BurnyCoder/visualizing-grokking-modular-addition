"""Phase `build_site`: render the static website from content/ (markdown + YAML), templates/ and web/ into site/.

Local explanation: chapters are markdown files with small macros that the build expands into HTML
  [[dfn:id]] / [[dfn:id|text]]      — the place a term is defined (highlighted, tooltip from glossary.yaml)
  [[term:id]] / [[term:id|text]]    — a later use of a defined term (tooltip, link to the glossary)
  [[symdef:id]] / [[sym:id]]        — definition / use of a mathematical symbol (KaTeX + one-sentence card)
  {{quote:id}}                      — an exact quotation from quotes.yaml (verified against the PDF page)
  {{num:key}}                       — a number from site/data/summary.json (so prose cannot drift from data)
  {{fig:id}} / {{widget:id}}        — a live figure from figures.yaml / an interactive teaching widget
  {{cite:key}}                      — a numbered link into references.yaml
and `::: example|nuance|note|object` containers become callout boxes; $…$ and $$…$$ become KaTeX math.
While expanding, the build enforces the reading contract and fails on any violation: every term and symbol is
defined (dfn/symdef) before it is used in reading order, every glossary entry is one sentence with a source,
every figure/number/quote/citation exists, and quotes respect the length policy (docs/content-style.md).
Global context: this is the only producer of site/ HTML; it imports no torch, so the GitHub Pages workflow can
run it on a CPU-only runner (`python -m grokviz.site_build`).
Sources: markdown-it-py (https://markdown-it-py.readthedocs.io/), mdit-py-plugins dollarmath/container
(https://mdit-py-plugins.readthedocs.io/), Jinja2 (https://jinja.palletsprojects.com/).
"""

from __future__ import annotations  # postponed annotations

import json  # summary + glossary json
import logging  # module logger
import re  # macro expansion
import shutil  # copy assets
from dataclasses import dataclass, field  # build state
from pathlib import Path  # paths

import yaml  # content files
from jinja2 import Environment, FileSystemLoader, select_autoescape  # templates
from markdown_it import MarkdownIt  # markdown → HTML
from mdit_py_plugins.container import container_plugin  # ::: callouts
from mdit_py_plugins.dollarmath import dollarmath_plugin  # $math$

from grokviz.config import REPO_ROOT, load_config  # paths

log = logging.getLogger("grokviz.site_build")
CONTENT, TEMPLATES, WEB = REPO_ROOT / "content", REPO_ROOT / "templates", REPO_ROOT / "web"
MACRO = re.compile(r"\[\[(dfn|term|sym|symdef):([\w.-]+)(?:\|([^\]]+))?\]\]|\{\{(quote|num|fig|widget|cite):([\w./-]+)\}\}")
CALLOUTS = {"example": "Worked example", "nuance": "Nuance", "note": "Note", "object": "", "definition": "Definition", "check": "Control"}
QUOTE_MAX_WORDS, QUOTE_CHAPTER_MAX = 25, 100  # copyright policy (NOTICE.md, docs/content-style.md)


def load_yaml(name: str):
    """Read content/<name>.yaml."""
    return yaml.safe_load((CONTENT / name).read_text(encoding="utf-8"))


@dataclass
class Build:
    """Everything the expansion needs plus the reading-order bookkeeping."""

    glossary: dict
    symbols: dict
    quotes: dict
    refs: dict
    figures: dict
    summary: dict
    js_figures: set
    js_widgets: set
    defined_terms: dict = field(default_factory=dict)  # term → page where defined
    defined_syms: dict = field(default_factory=dict)  # symbol → page where defined
    used_refs: list = field(default_factory=list)  # citation keys in first-use order
    errors: list = field(default_factory=list)
    root: str = "../"


def markdown() -> MarkdownIt:
    """Markdown renderer: CommonMark + tables + raw HTML + $math$ + ::: containers."""
    md = MarkdownIt("commonmark", {"html": True, "typographer": True}).enable("table")
    md.use(dollarmath_plugin, allow_space=True, double_inline=True)
    for name, title in CALLOUTS.items():
        def render(self, tokens, idx, options, env, name=name, title=title):
            if tokens[idx].nesting == 1:
                head = f'<span class="callout-title">{title}</span>' if title else ""
                return f'<div class="callout {name}{" objectbox" if name == "object" else ""}">{head}\n'
            return "</div>\n"
        md.use(container_plugin, name, render=render)
    md.add_render_rule("math_inline", lambda self, t, i, o, e: f'<span class="math inline">{_esc(t[i].content)}</span>')
    md.add_render_rule("math_block", lambda self, t, i, o, e: f'<div class="math block">{_esc(t[i].content)}</div>\n')
    md.add_render_rule("math_inline_double", lambda self, t, i, o, e: f'<div class="math block">{_esc(t[i].content)}</div>')
    return md


def _esc(s: str) -> str:
    """Minimal HTML escaping for math source (KaTeX reads textContent)."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fmt_number(entry) -> str:
    """Format a summary entry {value, fmt} for prose."""
    if not isinstance(entry, dict):
        entry = {"value": entry}
    v, f = entry["value"], entry.get("fmt", "g")
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    if v is None:
        return "—"
    if f == "int":
        return f"{int(round(v)):,}"
    if f == "pct":
        return f"{100 * v:.1f} %"
    if f == "sci":
        return f"{v:.2e}".replace("e-0", "×10⁻").replace("e-", "×10⁻").replace("e+0", "×10^").replace("e+", "×10^")
    return f"{v:{f}}" if not isinstance(v, str) else v


def expand(text: str, b: Build, page: str) -> str:
    """Expand macros in reading order, recording definitions and reporting violations."""
    quoted_words = 0

    def sub(m: re.Match) -> str:
        nonlocal quoted_words
        kind, key, label = m.group(1), m.group(2), m.group(3)
        if kind in ("dfn", "term"):
            g = b.glossary.get(key)
            if g is None:
                b.errors.append(f"{page}: term '{key}' missing from glossary.yaml")
                return label or key
            text_ = label or g["name"]
            if kind == "dfn":
                if key in b.defined_terms:
                    b.errors.append(f"{page}: term '{key}' defined twice (first in {b.defined_terms[key]})")
                b.defined_terms[key] = page
                return f'<dfn class="term" id="term-{key}" data-term="{key}" tabindex="0">{text_}</dfn>'
            if key not in b.defined_terms:
                b.errors.append(f"{page}: term '{key}' used before it is defined")
            return f'<a class="term" data-term="{key}" href="{b.root}glossary.html#{key}">{text_}</a>'
        if kind in ("sym", "symdef"):
            s = b.symbols.get(key)
            if s is None:
                b.errors.append(f"{page}: symbol '{key}' missing from symbols.yaml")
                return key
            if kind == "symdef":
                b.defined_syms.setdefault(key, page)
            elif key not in b.defined_syms:
                b.errors.append(f"{page}: symbol '{key}' used before it is defined")
            return f'<span class="sym" data-sym="{key}" tabindex="0"><span class="math inline">{_esc(label or s["tex"])}</span></span>'
        kind, key = m.group(4), m.group(5)
        if kind == "quote":
            q = b.quotes.get(key)
            if q is None:
                b.errors.append(f"{page}: quote '{key}' missing")
                return ""
            words = len(q["text"].split())
            quoted_words += words
            if words > QUOTE_MAX_WORDS:
                b.errors.append(f"{page}: quote '{key}' has {words} words (> {QUOTE_MAX_WORDS})")
            ref = b.refs[q["source"]]
            where = f", {q['where']}" if q.get("where") else ""
            cite = f'{ref["short"]}{where} · <a href="{q.get("url", ref["url"])}">source</a>'
            line_start = m.start() == 0 or m.string[m.start() - 1] == "\n"  # macro begins a line
            line_end = m.end() == len(m.string) or m.string[m.end()] == "\n"  # macro ends a line
            if line_start and line_end:  # alone on its line → a block quotation
                return f'<blockquote class="quote">“{q["text"]}”<cite>— {cite}</cite></blockquote>'
            return f'<q class="quote-inline">{q["text"]}</q> <span class="quote-cite">({cite})</span>'  # inside a sentence
        if kind == "num":
            if key not in b.summary:
                b.errors.append(f"{page}: number '{key}' missing from summary.json")
                return "?"
            return f'<span class="num" data-num="{key}">{fmt_number(b.summary[key])}</span>'
        if kind == "fig":
            if key not in b.figures:
                b.errors.append(f"{page}: figure '{key}' missing from figures.yaml")
                return ""
            if key not in b.js_figures:
                b.errors.append(f"{page}: figure '{key}' has no JS renderer in web/js/figures/index.js")
            return render_figure(key, b)
        if kind == "widget":
            if key not in b.js_widgets:
                b.errors.append(f"{page}: widget '{key}' has no JS implementation in web/js/widgets/index.js")
            return f'<div class="widget" data-widget="{key}"><p class="loading">Loading…</p></div>'
        if kind == "cite":
            if key not in b.refs:
                b.errors.append(f"{page}: reference '{key}' missing from references.yaml")
                return "[?]"
            if key not in b.used_refs:
                b.used_refs.append(key)
            n = list(b.refs).index(key) + 1
            return f'<a class="cite" href="{b.root}references.html#{key}" title="{b.refs[key]["short"]}">[{n}]</a>'
        return m.group(0)

    out = MACRO.sub(sub, text)
    if quoted_words > QUOTE_CHAPTER_MAX:
        b.errors.append(f"{page}: {quoted_words} quoted words (> {QUOTE_CHAPTER_MAX})")
    return out


_env: Environment | None = None
_md: MarkdownIt | None = None


def env() -> Environment:
    """Jinja environment (autoescape for .html templates)."""
    global _env
    _env = _env or Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
    return _env


def md_inline(text: str | None, b: Build, page: str) -> str:
    """Render a short markdown string (figure descriptions) with macros, without the <p> wrapper."""
    if not text:
        return ""
    html = _md.render(expand(text, b, page)).strip()
    return html[3:-4] if html.startswith("<p>") and html.endswith("</p>") and html.count("<p>") == 1 else html


def render_figure(fid: str, b: Build) -> str:
    """HTML of one figure from its metadata (shared by chapters and dashboard)."""
    f = dict(b.figures[fid], id=fid)
    page = f"figure:{fid}"
    f["what_html"] = md_inline(f.get("what"), b, page)
    f["read_html"] = _md.render(expand(f.get("read", ""), b, page)) if f.get("read") else ""
    f["eq_html"] = f'<div class="math block">{_esc(f["eq"])}</div>' if f.get("eq") else ""
    f["expect_html"] = md_inline(f.get("expect"), b, page)
    f["cite_html"] = md_inline(f.get("cite"), b, page)
    return env().get_template("figure.html").render(fig=f, root=b.root)


def js_ids(path: Path) -> set[str]:
    """Ids registered in a JS registry file (keys of the exported object literal)."""
    return set(re.findall(r'^\s*"([\w.-]+)":', path.read_text(encoding="utf-8"), flags=re.M)) if path.exists() else set()


def check_glossary(glossary: dict, errors: list) -> None:
    """Glossary policy: name + one sentence (≤ 35 words) + at least one source URL per entry."""
    for key, g in glossary.items():
        d = g.get("def", "")
        if not d.endswith(".") or len(d.split()) > 35:
            errors.append(f"glossary '{key}': definition must be one sentence of ≤ 35 words ending with '.'")
        if not g.get("src"):
            errors.append(f"glossary '{key}': missing source")


def build_site(ctx=None) -> None:
    """Phase entry point (also `python -m grokviz.site_build`): render every page into site/."""
    global _md
    cfg = ctx.cfg if ctx else load_config()
    out = cfg.site_dir
    _md = markdown()
    summary_path = out / "data" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    b = Build(glossary=load_yaml("glossary.yaml"), symbols=load_yaml("symbols.yaml"), quotes=load_yaml("quotes.yaml"),
              refs=load_yaml("references.yaml"), figures=load_yaml("figures.yaml"), summary=summary,
              js_figures=js_ids(WEB / "js" / "figures" / "index.js"), js_widgets=js_ids(WEB / "js" / "widgets" / "index.js"))
    check_glossary(b.glossary, b.errors)
    chapters = load_yaml("chapters.yaml")
    macros = {f"\\{k}": v for k, v in (load_yaml("macros.yaml") or {}).items()}
    common = {"first_chapter": chapters[0]["slug"]}
    (out / "chapters").mkdir(parents=True, exist_ok=True)
    pages_html = []
    for i, ch in enumerate(chapters):  # reading order = definition order
        ch["number"] = i
        src = (CONTENT / "chapters" / f"{ch['slug']}.md").read_text(encoding="utf-8")
        body = _md.render(expand(src, b, ch["slug"]))
        sources = [md_inline(s, b, ch["slug"]) for s in ch.get("sources", [])]
        pages_html.append((ch, body, sources))
    for i, (ch, body, sources) in enumerate(pages_html):
        html = env().get_template("chapter.html").render(
            page_title=f"{ch['title']} — Grokking, Visualized", description=ch["summary"], root="../", nav="learn",
            body_class="chapter-page", page_config={"root": "../", "page": "chapter", "macros": macros},
            chapters=chapters, chapter=dict(ch, sources=sources), body_html=body,
            prev=chapters[i - 1] if i > 0 else None, next=chapters[i + 1] if i + 1 < len(chapters) else None, **common)
        (out / "chapters" / f"{ch['slug']}.html").write_text(html, encoding="utf-8")
    b.root = "./"  # top-level pages
    dash = load_yaml("dashboard.yaml")
    groups = [{**g, "blurb": md_inline(g.get("blurb", ""), b, "dashboard"),
               "panels": [render_figure(fid, b) for fid in g["panels"]]} for g in dash["groups"]]
    for g in dash["groups"]:
        for fid in g["panels"]:
            if fid not in b.js_figures:
                b.errors.append(f"dashboard: figure '{fid}' has no JS renderer")
    tour = load_yaml("tour.yaml")
    html = env().get_template("dashboard.html").render(
        page_title="Dashboard — Grokking, Visualized", description="Every visualization of the grokking transformer, linked and scrubbable over training.",
        root="./", nav="dashboard", body_class="dashboard-page", groups=groups, intro=md_inline(dash["intro"], b, "dashboard"),
        page_config={"root": "./", "page": "dashboard", "macros": macros, "tour": tour}, **common)
    (out / "dashboard.html").write_text(html, encoding="utf-8")
    for name, title in (("index", "Grokking, Visualized"), ("glossary", "Glossary"), ("references", "Sources"), ("errata", "Errata and nuances"), ("reproduce", "Reproduce")):
        body = page_body(name, b)
        html = env().get_template("page.html").render(page_title=f"{title} — Grokking, Visualized" if name != "index" else title,
                                                      description="Explorable explanation of how a transformer groks modular addition.", root="./",
                                                      nav=name, body_class=f"{name}-page", body_html=body,
                                                      page_config={"root": "./", "page": name, "macros": macros}, **common)
        (out / f"{name}.html").write_text(html, encoding="utf-8")
    assets = out / "assets"
    shutil.copytree(WEB, assets, dirs_exist_ok=True)  # overwrite in place (a preview server may hold files open on Windows)
    glossary_json = {"terms": {k: {"name": v["name"], "def": _md.renderInline(v["def"]), "src": source_links(v)} for k, v in b.glossary.items()},
                     "symbols": {k: {"name": f'<span class="math inline">{_esc(v["tex"])}</span> — {v["name"]}', "def": _md.renderInline(v["def"])} for k, v in b.symbols.items()}}
    (assets / "glossary.json").write_text(json.dumps(glossary_json), encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")  # GitHub Pages: serve files/folders starting with _ as-is
    if b.errors:
        for e in b.errors:
            log.error("content check: %s", e)
        raise RuntimeError(f"site build: {len(b.errors)} content errors (see log)")
    log.info("built %d chapters + dashboard + 5 pages into %s", len(chapters), out)


def source_links(entry: dict) -> str:
    """'Source: <a>…</a>' links for a glossary entry."""
    srcs = entry.get("src", [])
    srcs = [srcs] if isinstance(srcs, str) else srcs
    return "Source: " + ", ".join(f'<a href="{u}">{re.sub(r"^https?://(www\.)?", "", u).split("/")[0]}</a>' for u in srcs)


def page_body(name: str, b: Build) -> str:
    """Body HTML of a top-level page: markdown file in content/pages/, or a generated glossary/references list."""
    if name == "glossary":
        items = sorted(b.glossary.items(), key=lambda kv: kv[1]["name"].lower())
        rows = "".join(f'<dt id="{k}">{v["name"]}</dt><dd>{_md.renderInline(v["def"])} <span class="fig-note">{source_links(v)}'
                       f'{" · defined in " + chapter_link(b.defined_terms[k]) if k in b.defined_terms else ""}</span></dd>' for k, v in items)
        syms = "".join(f'<tr><td><span class="math inline">{_esc(v["tex"])}</span></td><td>{v["name"]}</td><td>{_md.renderInline(v["def"])}</td></tr>'
                       for k, v in b.symbols.items())
        return (f"<h1>Glossary</h1><p>Every term used on this site, with a one-sentence definition and a source. "
                f"Terms are introduced in reading order in the chapters; hovering a dotted term anywhere shows its card.</p>"
                f"<dl class=\"glossary\">{rows}</dl><h2>Symbols</h2><table class=\"symtable\"><tr><th>Symbol</th><th>Name</th><th>Meaning</th></tr>{syms}</table>")
    if name == "references":
        items = "".join(f'<li id="{k}">{v["full"]} <a href="{v["url"]}">{v["url"]}</a>{(" — " + v["note"]) if v.get("note") else ""}</li>' for k, v in b.refs.items())
        notice = _md.render((REPO_ROOT / "NOTICE.md").read_text(encoding="utf-8").replace("# NOTICE —", "## Licenses —"))
        return f"<h1>Sources</h1><ol class=\"refs\">{items}</ol><div id=\"licenses\">{notice}</div>"
    path = CONTENT / "pages" / f"{name}.md"
    html = _md.render(expand(path.read_text(encoding="utf-8"), b, name)) if path.exists() else f"<h1>{name}</h1>"
    if "<!-- RCHECK_TABLE -->" in html:  # reproduce page: generated paper-vs-ours table (src/grokviz/rchecks.py)
        html = html.replace("<!-- RCHECK_TABLE -->", rcheck_table(b.summary.get("rchecks.rows", {}).get("value", [])))
    return html


def rcheck_table(rows: list[dict]) -> str:
    """HTML table of the reproduction checks (paper / ARENA value, ours, match, note)."""
    def f(v):
        if isinstance(v, float):
            return f"{v:.4g}"
        if isinstance(v, list):
            return "[" + ", ".join(f(x) for x in v) + "]"
        return str(v)
    mark = {True: "yes", False: "<strong>no</strong>", None: "—"}
    body = "".join(f"<tr><td>{r['id']}</td><td>{r['quantity']}</td><td>{f(r['paper'])}</td><td>{f(r['ours'])}</td><td>{mark[r['match']]}</td>"
                   f"<td>{r['source']}</td><td>{r['note']}</td></tr>" for r in rows)
    return ("<table><tr><th>ID</th><th>Quantity</th><th>Paper / ARENA</th><th>Ours</th><th>Match</th><th>Where</th><th>Note</th></tr>"
            f"{body}</table>")


def chapter_link(slug: str) -> str:
    """Link to a chapter by slug (used in the glossary)."""
    return f'<a href="chapters/{slug}.html">{slug}</a>' if not slug.startswith(("figure:", "dashboard")) else slug


if __name__ == "__main__":  # CPU-only entry point for CI: python -m grokviz.site_build
    import sys

    from grokviz.logs import setup_logging

    c = load_config()
    setup_logging(c.logs_dir, tag="site_build")
    try:
        build_site()
    except RuntimeError as exc:
        logging.getLogger("grokviz").error("%s", exc)
        sys.exit(1)
