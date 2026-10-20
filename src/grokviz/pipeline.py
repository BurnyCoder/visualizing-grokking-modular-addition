"""The single wrapper: an ordered registry of clearly named pipeline phases and the loop that runs them.

Local explanation: `PHASES` maps each phase name to the function (imported from its own module) that
implements it; `run_pipeline` resolves the user's selection, then runs each phase inside a timer that
logs start/done/failure.
Global context: this is the top of the code hierarchy — reading `PHASES` top to bottom tells you the
whole data flow (environment → paper checkpoints → training → analysis → site/report export).
Phases are added here milestone by milestone as their modules are implemented.
"""

from __future__ import annotations  # postponed annotations

import logging  # pipeline-level logger
from typing import Callable  # type of phase functions

from grokviz.checkpoint_metrics import compute_checkpoint_metrics  # curves over training
from grokviz.context import PipelineContext  # uniform phase argument
from grokviz.envcheck import check_environment  # phase 1: environment and device check
from grokviz.export import export_site_data  # binary arrays + manifest for the website
from grokviz.figures import render_report_figures  # site figures → report PNGs
from grokviz.geometry import compute_geometry  # activation / output geometry frames
from grokviz.gradients import compute_gradient_dynamics  # exact full-batch gradients per checkpoint
from grokviz.hf_publish import publish_runs_to_hub, publish_site_data  # Hugging Face uploads
from grokviz.hypotheses import analyse_final_models, test_hypotheses  # preregistered tests
from grokviz.logs import phase_timer  # start/done/elapsed logging around each phase
from grokviz.phases import detect_phases  # objective phase boundaries
from grokviz.rchecks import validate_against_paper  # reproduce paper/ARENA numbers
from grokviz.site_build import build_site  # static website
from grokviz.sources import acquire_paper_sources, convert_paper_run, validate_paper_run  # paper run
from grokviz.train import train_all_runs  # our own runs (seeds + controls), trained together

log = logging.getLogger("grokviz.pipeline")  # logger for the wrapper itself

# Ordered registry: execution order is the order of this dict (dicts preserve insertion order).
PHASES: dict[str, Callable[[PipelineContext], None]] = {
    "check_environment": check_environment,  # versions, GPU, determinism settings
    "acquire_paper_sources": acquire_paper_sources,  # pinned HF checkpoint + arXiv PDFs (sha-checked)
    "convert_paper_run": convert_paper_run,  # → runs/paper/ in our run format
    "validate_paper_run": validate_paper_run,  # recomputed losses must match the logged ones
    "validate_against_paper": validate_against_paper,  # R-checks on the epoch-40k model
    "train_all_runs": train_all_runs,  # 22 preregistered runs (docs/preregistration.md)
    "compute_checkpoint_metrics": compute_checkpoint_metrics,  # every metric at every checkpoint
    "compute_gradient_dynamics": compute_gradient_dynamics,  # what the gradients point at, over training
    "compute_geometry": compute_geometry,  # circles, PCA, output-distribution geometry
    "detect_phases": detect_phases,  # memorisation / circuit formation / cleanup boundaries
    "analyse_final_models": analyse_final_models,  # ablations, Gini nulls, patching on final checkpoints
    "test_hypotheses": test_hypotheses,  # H1–H5 + secondary statistics
    "publish_runs_to_hub": publish_runs_to_hub,  # our runs + model card + collection on Hugging Face
    "export_site_data": export_site_data,  # site/data/manifest.json + arrays
    "publish_site_data": publish_site_data,  # site data → HF dataset, pinned in site-data.lock
    "build_site": build_site,  # content/ + templates/ + web/ → site/
    "render_report_figures": render_report_figures,  # PNGs for reports, paper and README
}


def select_phases(requested: list[str], skip: list[str]) -> list[str]:
    """Expand "all", validate names, drop skipped ones; returned names keep registry order."""
    names = list(PHASES) if requested == ["all"] else requested  # "all" means every registered phase
    unknown = [n for n in names + skip if n not in PHASES]  # typos must fail loudly, not be ignored
    if unknown:
        raise ValueError(f"unknown phase(s): {unknown}; known: {list(PHASES)}")
    return [n for n in PHASES if n in names and n not in skip]  # registry order, minus skips


def run_pipeline(ctx: PipelineContext, requested: list[str], skip: list[str] | None = None) -> None:
    """Run the selected phases in order; any exception stops the pipeline after being logged."""
    selected = select_phases(requested, skip or [])
    log.info("phases to run: %s", ", ".join(selected))
    for name in selected:  # sequential: later phases consume earlier phases' outputs
        with phase_timer(log, name):
            PHASES[name](ctx)
