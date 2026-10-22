"""The exported site data matches its manifest and fits the hosting budget (docs/deployment.md, docs/data-formats.md).

Global context: the website fetches every array by the path, dtype and shape written in site/data/manifest.json; a file
that disagrees with its entry would silently draw garbage. GitHub Pages refuses sites above 1 GB
(https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits), and very large single files make
figures slow to appear, so both are bounded here. Skipped until `export_site_data` has run.
"""

import hashlib  # sha256 of each file
import json  # manifest
import math  # element counts

import pytest  # skip marker

from grokviz.config import load_config  # where site/ lives

DATA = load_config().site_dir / "data"
ITEMSIZE = {"f32": 4, "f16": 2, "i16": 2, "u8q": 1}  # bytes per element of each export dtype (export.Writer)
pytestmark = pytest.mark.skipif(not (DATA / "manifest.json").exists(), reason="site data not exported yet")


def test_every_array_file_matches_its_manifest_entry():
    """Size = elements × itemsize = bytes field, and the sha256 matches, for every array of every run."""
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    total, largest = 0, (0, "")
    for run in manifest["runs"]:
        for name, entry in run["arrays"].items():
            raw = (DATA / entry["file"]).read_bytes()
            assert len(raw) == entry["bytes"] == math.prod(entry["shape"]) * ITEMSIZE[entry["dtype"]], f"{run['id']}/{name}"
            assert hashlib.sha256(raw).hexdigest().startswith(entry["sha256"]), f"{run['id']}/{name}"  # manifest keeps a 16-hex prefix
            total, largest = total + len(raw), max(largest, (len(raw), f"{run['id']}/{name}"))
    assert total < 1e9, f"site data {total / 1e6:.0f} MB exceeds the 1 GB GitHub Pages limit"
    assert largest[0] < 16e6, f"largest file {largest[1]} is {largest[0] / 1e6:.1f} MB"


def test_every_prose_number_is_in_the_summary():
    """summary.json holds a value for every number key (the site build fails otherwise; this pins the data side)."""
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    assert all("value" in v for v in summary.values()) and "rchecks.n_match" in summary and "hyp.summary" in summary


def test_every_run_can_animate_and_the_budget_holds():
    """Every run ships weight snapshots across training (export.snapshot_steps), the attention lookup tables and no
    unused reference weights, and the whole site data stays under the project's 0.8 GB budget (docs/deployment.md)."""
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    total = sum(f.stat().st_size for f in DATA.rglob("*") if f.is_file())
    assert total < 0.8e9, f"site data {total / 1e6:.0f} MB exceeds the 0.8 GB budget"
    for run in manifest["runs"]:
        grokked = bool(run["phases"].get("t3"))
        assert len(run["weight_steps"]) >= (15 if grokked else 10), f"{run['id']}: {len(run['weight_steps'])} weight snapshots"
        assert "attn_C" in run["arrays"] and "w_ref_f32" not in run["arrays"], run["id"]
    summary = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))
    assert {"wd1_s0", "wd1_s10"} <= set(summary["hypotheses.per_run"]["value"]) and summary["hypotheses.per_run"]["fmt"] == "json"
