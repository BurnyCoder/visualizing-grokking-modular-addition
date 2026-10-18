"""Command-line interface: `uv run grokviz --phases <names|all> [--skip ...] [--runs ...] [--force]`.

Local explanation: parses arguments, exports process-level settings (UTF-8, cuBLAS determinism,
HF token) from the .env files, loads the typed config, starts timestamped logging, and hands control
to `grokviz.pipeline.run_pipeline`. Returns a process exit code instead of raising.
Global context: the only way users run the project; also importable as `main(argv)` for tests.
Source: argparse (https://docs.python.org/3/library/argparse.html).
"""

from __future__ import annotations  # postponed annotations

import argparse  # command-line parsing
import logging  # top-level logger for fatal errors
import sys  # process exit code

from grokviz.config import export_process_env, load_config  # .env-driven settings


def _csv(value: str) -> list[str]:
    """argparse type: "a,b , c" → ["a", "b", "c"] (empty items dropped)."""
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    """Define the CLI options shared by every invocation."""
    parser = argparse.ArgumentParser(prog="grokviz", description="Grokking, Visualized — pipeline runner")
    parser.add_argument("--phases", type=_csv, default=["all"], help="comma list of phases, or 'all'")
    parser.add_argument("--skip", type=_csv, default=[], help="comma list of phases to skip")
    parser.add_argument("--runs", type=_csv, default=[], help="run-id prefixes to process (default: all)")
    parser.add_argument("--force", action="store_true", help="recompute outputs that already exist")
    parser.add_argument("--verbose", action="store_true", help="log DEBUG messages too")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse args, set up config + logging, run the pipeline; return 0 on success, 1 on failure."""
    args = build_parser().parse_args(argv)
    export_process_env()  # must precede torch/huggingface_hub initialisation
    cfg = load_config()  # typed settings
    # Imports below pull in torch; they come after export_process_env so CUBLAS settings take effect.
    from grokviz.context import PipelineContext
    from grokviz.logs import setup_logging
    from grokviz.pipeline import run_pipeline

    log_path = setup_logging(cfg.logs_dir, level=logging.DEBUG if args.verbose else logging.INFO)
    ctx = PipelineContext(cfg=cfg, log_path=log_path, force=args.force, runs=args.runs)
    try:
        run_pipeline(ctx, args.phases, args.skip)
    except Exception as exc:  # phase failures were already logged with a full traceback by phase_timer
        logging.getLogger("grokviz").error("pipeline stopped: %s: %s", type(exc).__name__, exc)
        return 1
    finally:
        for handler in logging.getLogger("grokviz").handlers:  # release the log file (Windows locks open files)
            handler.close()
    return 0


if __name__ == "__main__":  # `python -m grokviz`
    sys.exit(main())
