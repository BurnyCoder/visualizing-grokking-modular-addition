"""Phase `build_paper_pdf`: compile the Typst paper `paper/main.typ` into `reports/paper.pdf`.

Local explanation: every number the paper prints comes from the same `site/data/summary.json` the website uses,
formatted by the website's own `fmt_number`, and is handed to Typst as `sys.inputs` (a string dictionary). Figures
are the PNGs that `render_report_figures` screenshots from the website. The paper therefore cannot disagree with
the site, and no number in it is typed by hand.
Global context: last phase of the pipeline; the PDF is the compressed write-up of reports/*.md.
Sources: typst Python bindings (https://pypi.org/project/typst/), `typst.compile(input, output, root, sys_inputs)`;
Typst `sys.inputs` (https://typst.app/docs/reference/foundations/sys/).
"""

from __future__ import annotations  # postponed annotations

import json  # summary.json
import logging  # module logger

import typst  # Typst compiler (Rust, via Python bindings)

from grokviz.config import REPO_ROOT  # paper/ location and Typst root
from grokviz.context import PipelineContext  # phase argument
from grokviz.formatting import fmt_number, fmt_value  # same number formatting as the website

log = logging.getLogger("grokviz.paper")


def paper_inputs(summary: dict, hypotheses: dict | None) -> dict[str, str]:
    """Typst sys.inputs (strings only): every prose number formatted as on the website, plus two JSON strings the
    paper decodes into tables — the R-check rows (as display strings) and the hypothesis verdicts."""
    # prose numbers (structured entries — the R-check rows, the verdict dict — are passed as JSON below)
    out = {k: fmt_number(v).replace("**", "") for k, v in summary.items() if not isinstance(v.get("value"), dict) and k != "rchecks.rows"}
    rows = summary.get("rchecks.rows", {}).get("value", [])
    mark = {True: "yes", False: "no", None: "—"}
    out["rchecks.json"] = json.dumps([[r["id"], r["quantity"], fmt_value(r["paper"]), fmt_value(r["ours"]), mark[r["match"]]] for r in rows])
    out["hypotheses.json"] = json.dumps(hypotheses["verdict"] if hypotheses else {}, default=float)
    return out


def build_paper_pdf(ctx: PipelineContext) -> None:
    """Phase entry point: reports/paper.pdf from paper/main.typ, site/data/summary.json and the hypothesis table."""
    summary = json.loads((ctx.cfg.site_dir / "data" / "summary.json").read_text(encoding="utf-8"))
    hyp_file = ctx.cfg.reports_dir / "tables" / "hypotheses.json"
    hypotheses = json.loads(hyp_file.read_text(encoding="utf-8")) if hyp_file.exists() else None
    out = ctx.cfg.reports_dir / "paper.pdf"
    typst.compile(str(REPO_ROOT / "paper" / "main.typ"), output=str(out), root=str(REPO_ROOT), sys_inputs=paper_inputs(summary, hypotheses))
    log.info("paper → %s (%d kB, %d numbers bound)", out, out.stat().st_size // 1024, len(summary))
