# AGENTS.md — instructions for coding agents working in this repo

## What this repo is
A reproducible pipeline (`src/grokviz/`, Python, uv) plus a static website that visualizes the
modular-addition grokking transformer of Nanda et al. 2023 (arXiv:2301.05217). The original
notebooks live in the git submodule
`ai-mechanistic-interpretability-transformer-modular-addition-grokking/` — treat it as read-only
reference; port code from it with `Source:` links instead of editing it.

## Environment
- Always use the local venv managed by uv: `uv sync`, then `uv run <cmd>` (no global installs).
- Python 3.12 (`.python-version`). Configuration comes from `.env` (copy `.env.example`).
- On Windows set `PYTHONUTF8=1` (already in `.env.example`) so logs with math symbols don't crash.

## Conventions
- One wrapper (`src/grokviz/pipeline.py`) calls clearly named phases imported from modules.
- Every module, function and non-trivial line carries a comment with a local explanation, the
  global context, and a `Source:` link when the logic comes from a paper, notebook, ARENA or docs.
- Fourier basis convention follows the paper's own code: index 0 = constant, `2k-1` = cos k, `2k` = sin k.
- Quotes from papers: short and exact (≤25 words), with page number; see `NOTICE.md`.
- Tests first (pytest). Large artifacts (`runs/`, `data/`, `site/`, `logs/`) are gitignored.
