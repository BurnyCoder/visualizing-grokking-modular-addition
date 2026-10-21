"""How numbers are printed in prose, tables and the paper — one implementation for every output.

Local explanation: `fmt_number` turns a summary entry {value, fmt} into text (thousands separators, percentages,
scientific notation with Unicode superscripts); `fmt_value` prints the mixed scalars/lists of the R-check table.
Global context: the website (site_build.py), the R-check markdown table (rchecks.py) and the Typst paper (paper.py)
all import these, so the same number reads identically everywhere.
Sources: Python format specification mini-language (https://docs.python.org/3/library/string.html#formatspec);
str.translate (https://docs.python.org/3/library/stdtypes.html#str.translate).
"""

from __future__ import annotations  # postponed annotations

SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")  # str.translate table for exponents


def fmt_number(entry) -> str:
    """Format a summary entry {value, fmt} for prose."""
    if not isinstance(entry, dict):
        entry = {"value": entry}
    v, f = entry["value"], entry.get("fmt", "g")
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    if v is None:
        return "—"
    if isinstance(v, (str, dict)):  # pre-formatted text (or a structured value that has no prose form)
        return v if isinstance(v, str) else ""
    if f == "int":
        return f"{int(round(v)):,}"
    if f == "pct":
        return f"{100 * v:.1f} %"
    if f == "sci":  # 2.41e-07 → "2.41×10⁻⁷" (mantissa from Python's e-format, exponent in Unicode superscripts)
        mant, exp = f"{v:.2e}".split("e")
        return mant if int(exp) == 0 else f"{mant}×10{str(int(exp)).translate(SUPERSCRIPT)}"
    return f"{v:{f}}"


def fmt_value(v) -> str:
    """Compact text for an R-check value: 4 significant digits, lists in brackets, strings as given."""
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, list):
        return "[" + ", ".join(fmt_value(x) for x in v) + "]"
    return str(v)
