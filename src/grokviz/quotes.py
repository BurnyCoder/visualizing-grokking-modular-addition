"""Verify that every quotation on the site appears verbatim on the cited page of the cited PDF.

Local explanation: extracts each page's text with PyMuPDF, normalises both the page and the quote the same way
(Unicode NFKC, ligatures, curly quotes → straight, line-break hyphenation removed, whitespace collapsed,
case-folded), and checks that the quote is a substring of its page; it also reports the page(s) where the
quote was actually found, so a wrong page number is caught too.
Global context: the website promises exact quotes; tests/test_quotes.py runs this against the PDFs that the
`acquire_paper_sources` phase downloaded (they are never redistributed).
Sources: PyMuPDF Page.get_text (https://pymupdf.readthedocs.io/en/latest/page.html#Page.get_text);
unicodedata.normalize NFKC (https://docs.python.org/3/library/unicodedata.html#unicodedata.normalize).
"""

from __future__ import annotations  # postponed annotations

import re  # whitespace/hyphenation clean-up
import unicodedata  # NFKC normalisation
from pathlib import Path  # PDF paths

import pymupdf  # PDF text extraction

QUOTE_MARKS = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", "−": "-"})  # typographic → ASCII


def normalise(text: str) -> str:
    """Comparable form of a text fragment (see module docstring)."""
    t = unicodedata.normalize("NFKC", text).translate(QUOTE_MARKS)  # NFKC also splits ligatures like "ﬁ" → "fi"
    t = re.sub(r"-\s*\n\s*", "", t)  # "generaliz-\ning" → "generalizing" (line-break hyphenation)
    t = re.sub(r"\s+", " ", t)  # newlines and runs of spaces → one space
    return t.strip().casefold()


def page_texts(pdf: Path) -> list[str]:
    """Normalised text of every page (index 0 = page 1)."""
    with pymupdf.open(pdf) as doc:
        return [normalise(page.get_text()) for page in doc]


def verify(quotes: dict, pdfs: dict[str, Path]) -> list[str]:
    """Return a list of problems (empty = every quote found verbatim on its cited page)."""
    problems = []
    cache: dict[str, list[str]] = {}
    for qid, q in quotes.items():
        pdf = pdfs.get(q["source"])
        if pdf is None or not pdf.exists():
            problems.append(f"{qid}: no PDF available for source {q['source']}")
            continue
        if q["source"] not in cache:  # parse each PDF once (setdefault would re-parse it for every quote)
            cache[q["source"]] = page_texts(pdf)
        pages = cache[q["source"]]
        needle = normalise(q["text"])
        found = [i + 1 for i, page in enumerate(pages) if needle in page]
        if not found:
            problems.append(f"{qid}: text not found in {pdf.name}")
        elif q["page"] not in found:
            problems.append(f"{qid}: cited page {q['page']} but found on page(s) {found}")
    return problems
