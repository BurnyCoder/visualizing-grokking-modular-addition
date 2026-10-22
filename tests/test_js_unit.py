"""Run the pure JavaScript unit tests (tests/js/unit/*.test.mjs) with Node's built-in test runner.

Global context: the site's update pipeline (batched store, scoped stores, latest-wins rendering) and its overlay maths
(step cursor positions) are plain ES modules without DOM access at import time, so Node can test them directly; the
browser-only parts are covered by tests/test_site_browser.py.
Source: Node.js test runner (https://nodejs.org/api/test.html).
"""

import shutil  # find node
import subprocess  # run node --test
from pathlib import Path  # test file discovery

import pytest  # skip if node missing

ROOT = Path(__file__).resolve().parents[1]  # repository root
UNIT = sorted((ROOT / "tests" / "js" / "unit").glob("*.test.mjs"))  # explicit file list: no shell globbing needed on Windows


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_js_unit_tests_pass():
    """Every Node unit test file must pass."""
    assert UNIT, "no JavaScript unit tests found"
    result = subprocess.run(["node", "--test", *map(str, UNIT)], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
