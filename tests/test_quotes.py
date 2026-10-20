"""Every quote in content/quotes.yaml must appear verbatim on its cited PDF page (needs the downloaded PDFs).

Global context: guards the site's "exact quotes" promise against typos and wrong page numbers; skipped when the
PDFs have not been downloaded yet (run `uv run grokviz --phases acquire_paper_sources` first).
"""

import pytest  # skip marker
import yaml  # quotes.yaml

from grokviz.config import REPO_ROOT, load_config  # paths
from grokviz.quotes import normalise, verify  # module under test

PDFS = {"nanda2023": "2301.05217v3.pdf", "power2022": "2201.02177v1.pdf"}


def test_normalise_handles_hyphenation_and_quotes():
    """Line-break hyphens, ligatures and curly quotes do not prevent a match."""
    assert normalise("gener-\nalizing “grok” ﬁne") == normalise('generalizing "grok" fine')


def test_quotes_are_verbatim_on_cited_pages():
    """All quotes found on exactly the page they cite."""
    folder = load_config().data_dir / "external" / "papers"
    pdfs = {k: folder / v for k, v in PDFS.items()}
    if not all(p.exists() for p in pdfs.values()):
        pytest.skip("reference PDFs not downloaded")
    quotes = yaml.safe_load((REPO_ROOT / "content" / "quotes.yaml").read_text(encoding="utf-8"))
    assert verify(quotes, pdfs) == []
