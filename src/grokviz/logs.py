"""Timestamped logging to a file and the terminal at the same time.

Local explanation: `setup_logging` attaches two handlers to the "grokviz" logger — one writing UTF-8
lines to `logs/<YYYYmmdd-HHMMSS>_<tag>.log`, one echoing to the terminal in real time — and
`phase_timer` wraps a block with start/done/elapsed messages.
Global context: every pipeline phase logs through `logging.getLogger("grokviz.<module>")`, so one
run of the pipeline produces one complete, timestamped record that can be inspected afterwards.
Sources: logging cookbook (https://docs.python.org/3/howto/logging-cookbook.html#logging-to-multiple-destinations);
`TextIOWrapper.reconfigure` for UTF-8 consoles on Windows
(https://docs.python.org/3/library/io.html#io.TextIOWrapper.reconfigure).
"""

from __future__ import annotations  # postponed evaluation of annotations

import contextlib  # contextmanager decorator for phase_timer
import logging  # standard-library logging framework
import sys  # access to the stdout/stderr streams
import time  # wall-clock timing of phases
from datetime import datetime  # timestamp for the log file name
from pathlib import Path  # filesystem paths
from typing import Iterator  # return type of the generator-based context manager

# One format for both destinations: time with milliseconds | level | logger name | message.
LOG_FORMAT = "%(asctime)s.%(msecs)03d | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"  # ISO-like date, local time


def setup_logging(logs_dir: Path, tag: str = "pipeline", level: int = logging.INFO) -> Path:
    """Configure the "grokviz" logger to write to a new timestamped file and to stdout; return the file."""
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252, which cannot print ω or →
        if hasattr(stream, "reconfigure"):  # real text streams support it; pytest capture objects may not
            stream.reconfigure(encoding="utf-8", errors="replace")
    logs_dir.mkdir(parents=True, exist_ok=True)  # create logs/ on first use
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # sortable timestamp for the file name
    log_path = logs_dir / f"{stamp}_{tag}.log"  # e.g. logs/20261018-171500_pipeline.log
    logger = logging.getLogger("grokviz")  # parent of every module logger in the package
    for handler in list(logger.handlers):  # make repeated calls idempotent (tests, notebooks)
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)  # shared formatter
    file_handler = logging.FileHandler(log_path, encoding="utf-8")  # full record on disk
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)  # real-time echo in the terminal
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(level)  # INFO by default; DEBUG when --verbose is passed
    logger.propagate = False  # avoid duplicate lines through the root logger
    logger.info("logging to %s", log_path)  # first line of every log says where it lives
    return log_path


@contextlib.contextmanager
def phase_timer(logger: logging.Logger, name: str) -> Iterator[None]:
    """Log "phase <name>: start", run the block, then log "done" with the elapsed seconds (or the failure)."""
    start = time.perf_counter()  # monotonic high-resolution clock
    logger.info("phase %s: start", name)
    try:
        yield  # run the phase body
    except Exception:  # log the traceback into the file, then re-raise so the CLI exits non-zero
        logger.exception("phase %s: FAILED after %.1f s", name, time.perf_counter() - start)
        raise
    logger.info("phase %s: done in %.1f s", name, time.perf_counter() - start)
