"""Pipeline context: the single object passed to every phase function.

Local explanation: bundles the loaded `Config` with the command-line options that affect every phase
(which runs to process, whether to recompute cached outputs).
Global context: phases have the uniform signature `phase(ctx: PipelineContext) -> None`, which lets
`grokviz.pipeline` call them from an ordered registry without knowing their internals.
"""

from __future__ import annotations  # postponed annotations

from dataclasses import dataclass, field  # simple immutable record
from pathlib import Path  # type of the log file path

from grokviz.config import Config  # typed settings loaded from .env files


@dataclass(frozen=True)
class PipelineContext:
    """Settings + CLI options shared by all phases."""

    cfg: Config  # all tunable settings (see .env.example)
    log_path: Path  # file this invocation logs to (recorded in run metadata for provenance)
    force: bool = False  # recompute outputs even when a phase finds them already done
    runs: list[str] = field(default_factory=list)  # run-id filter; empty means "all runs"

    def wants(self, run_id: str) -> bool:
        """True when `run_id` is selected by the --runs filter (prefix match, so "wd1" selects all wd1_s*)."""
        return not self.runs or any(run_id.startswith(prefix) for prefix in self.runs)
