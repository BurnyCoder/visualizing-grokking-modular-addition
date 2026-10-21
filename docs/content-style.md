# Content style and checks

The site promises that a reader with minimal background can follow it. The build (`src/grokviz/site_build.py`)
enforces the mechanical parts of that promise; the rest is editorial.

## Enforced by the build (fails on violation)

- **Define before use.** A term is defined once with `[[dfn:id]]` and used later with `[[term:id]]`; symbols with
  `[[symdef:id]]` / `[[sym:id]]`. Chapters are checked in reading order (`content/chapters.yaml`); figure captions are
  checked where they are placed.
- **No plain forward uses (term-shadow lint).** A glossary word (multi-word, or 6+ letters) must not appear in plain text
  in a chapter before the chapter that defines it. Such a chapter must link the word once with
  `[[preview:id|text]]`, which shows the definition card ahead of the definition. Chapter 0's overview and the symbol
  table use this.
- **Links.** Every relative link and asset in the built HTML must exist. `data/` is excluded because it arrives at
  deploy time.
- **Glossary entries** (`content/glossary.yaml`): one sentence, ≤ 35 words, ending with a period, with ≥ 1 source URL.
- **Quotes** (`content/quotes.yaml`): ≤ 25 words each, ≤ 100 quoted words per chapter; rendered with source and page.
  `tests/test_quotes.py` verifies every quote verbatim against the cited PDF page.
- **References:** every `{{cite:key}}` exists in `content/references.yaml` (numbered in file order).
- **Numbers:** every `{{num:key}}` exists in `site/data/summary.json` — prose numbers are never typed by hand.
- **Figures/widgets:** every `{{fig:id}}` / `{{widget:id}}` has metadata and a registered JS implementation.

## Editorial rules

- One idea per paragraph; every new idea gets a worked example (`::: example`) with the simplest numbers that work,
  and the real network's numbers next to it.
- Explanations are our own words; quotes are short and used only for definitions and key claims (arXiv
  non-exclusive license, see NOTICE.md). Videos are linked, never transcribed.
- Claims about the network cite either the paper/ARENA or a bound number from our analysis; if our number differs
  from the paper's, both are shown and the difference is explained (errata page).
- Hedging follows the evidence: “the paper claims / we measured / we could not reproduce”.
- Charts: log axes are labelled, phases shaded, colours fixed per frequency/run, no dual axes, sequential maps for
  non-negative data and diverging maps (centred at 0) for signed data.
