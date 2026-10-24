"""The Typst paper compiles from bound numbers (phase build_paper_pdf).

Global context: paper/main.typ reads every number measured on our runs from sys.inputs; this test feeds placeholder inputs shaped like the
real ones (formatted prose numbers, the R-check rows and the hypothesis verdicts as JSON) and checks that a multi-page
PDF comes out and that no ⟨missing⟩ marker is printed for a key we supplied.
Sources: typst Python bindings (https://pypi.org/project/typst/); PyMuPDF text extraction (https://pymupdf.readthedocs.io/).
"""

import re  # find the keys the paper reads

import pymupdf  # read the compiled PDF
import typst  # compiler

from grokviz.config import REPO_ROOT  # paper location
from grokviz.paper import paper_inputs  # the function under test


def test_paper_compiles_with_bound_numbers(tmp_path):
    """Every n("key") in main.typ gets a value; the PDF has several pages, the author line and ends with the AI statement."""
    source = (REPO_ROOT / "paper" / "main.typ").read_text(encoding="utf-8")
    # Typst ends an embedded expression at ";" and drops it: prose must write #n("key")\; to keep the semicolon
    assert not re.search(r'#n\("[\w.]+"\);', source), "unescaped ';' after #n(...) is swallowed by Typst"
    keys = set(re.findall(r'#?n\("([\w.]+)"\)', source))
    summary = {k: {"value": 0.5, "fmt": ".2f"} for k in keys}
    summary["rchecks.rows"] = {"value": [{"id": "R1", "quantity": "loss", "paper": 2.41e-7, "ours": 2.412e-7, "match": True}], "fmt": "str"}
    summary["hypotheses.verdict"] = {"value": {"H1": {"supported": True}}, "fmt": "json"}  # structured data entry from summary.py
    verdict = {"H1": {"successes": 11, "n": 11, "p_holm": 0.002, "supported": True},
               "H5": {"wd1_grokked": 11, "wd1_n": 11, "wd0_grokked": 0, "wd0_n": 5, "p_holm": 0.0002, "supported": True}}
    out = tmp_path / "paper.pdf"
    typst.compile(str(REPO_ROOT / "paper" / "main.typ"), output=str(out), root=str(REPO_ROOT),
                  sys_inputs=paper_inputs(summary, {"verdict": verdict}))
    doc = pymupdf.open(out)
    text = "".join(page.get_text() for page in doc)
    assert doc.page_count >= 5 and "Libor Burian" in text and "⟨" not in text
    assert "Claude Code" in doc[-1].get_text() and "AI statement" in doc[-1].get_text()  # disclosure on the last page
