"""Phase `detect_phases`: objective, pre-specified boundaries of memorisation / circuit formation / cleanup.

Local explanation: the paper drew its phase boundaries (1.4k, 9.4k, 14k epochs) by eye. To compare runs
without circularity we derive boundaries only from train/test loss, accuracy and the sum of squared
weights — never from the progress measures (restricted/excluded loss, Gini) whose behaviour we then test:
  t1 (end of memorisation)    = first step after which the train loss falls by < 1 % per 100 steps,
  t2 (end of circuit formation) = first checkpoint after t1 with test loss ≤ 0.2 × median test loss over [t1, t1 + 2000],
  t3 (end of cleanup)          = first checkpoint after t2 with SSW ≤ SSW_final + 0.05·(SSW(t2) − SSW_final),
                                 SSW_final = median SSW over the last 5000 steps,
  t_grok                        = first checkpoint with test accuracy ≥ 0.99.
The constants (1 %, 0.2, 0.05) were calibrated once on the paper run (giving 1.6k / 9.3k / 14.0k vs the
paper's 1.4k / 9.4k / 14k) and frozen in docs/operationalizations.md before any of our runs was analysed.
Global context: phase bands shade every training-time plot on the website and define the time points of the
preregistered hypotheses H1/H2.
Sources: paper Sec. 5.2 phase descriptions (https://arxiv.org/abs/2301.05217).
"""

from __future__ import annotations  # postponed annotations

import json  # phases.json
import logging  # module logger

import numpy as np  # arrays

from grokviz.checkpoint_metrics import run_dirs  # selected run folders
from grokviz.context import PipelineContext  # phase argument
from grokviz.runstore import RunStore  # run access

log = logging.getLogger("grokviz.phases")

TRAIN_PLATEAU_REL_DROP = 0.01  # t1: train loss decreases by less than 1 % over the next 100 steps
GROK_TEST_FRACTION = 0.2  # t2: test loss at 20 % of its post-memorisation plateau
CLEANUP_SSW_FRACTION = 0.05  # t3: SSW within 5 % of the way from SSW(t2) to its final level
PAPER_BOUNDARIES = {"t1": 1400, "t2": 9400, "t3": 14000}  # paper Sec. 5.2 (epochs; drawn by eye)


def _first(steps: np.ndarray, condition: np.ndarray, after: int) -> int | None:
    """First step strictly after `after` where `condition` holds (None if never)."""
    idx = np.nonzero(condition & (steps > after))[0]
    return int(steps[idx[0]]) if len(idx) else None


def detect(curve_steps: np.ndarray, train_loss: np.ndarray, ck_steps: np.ndarray, test_loss: np.ndarray,
           test_acc: np.ndarray, ssw: np.ndarray) -> dict:
    """Apply the frozen rules to one run's arrays; any boundary may be None if the run never reaches it."""
    rel_drop = (train_loss[:-100] - train_loss[100:]) / np.maximum(train_loss[:-100], 1e-300)  # 100-step relative decrease
    plateau = np.nonzero((rel_drop < TRAIN_PLATEAU_REL_DROP) & (curve_steps[:-100] > 100))[0]
    t1 = int(curve_steps[plateau[0]]) if len(plateau) else None
    t2 = t3 = None
    if t1 is not None:
        window = (ck_steps >= t1) & (ck_steps <= t1 + 2000)
        t2 = _first(ck_steps, test_loss <= GROK_TEST_FRACTION * np.median(test_loss[window]), t1)
    if t2 is not None:
        ssw_final = np.median(ssw[ck_steps >= ck_steps[-1] - 5000])
        ssw_t2 = ssw[np.searchsorted(ck_steps, t2)]
        t3 = _first(ck_steps, ssw <= ssw_final + CLEANUP_SSW_FRACTION * (ssw_t2 - ssw_final), t2)
    t_grok = _first(ck_steps, test_acc >= 0.99, -1)
    return {"t1": t1, "t2": t2, "t3": t3, "t_grok": t_grok}


def detect_phases(ctx: PipelineContext) -> None:
    """Phase entry point: write analysis/phases.json for every run with checkpoint metrics."""
    for run_dir in run_dirs(ctx):
        path = run_dir / "analysis" / "ckpt_metrics.npz"
        if not path.exists():
            log.warning("%s: no ckpt_metrics.npz yet, skipping", run_dir.name)
            continue
        store = RunStore(run_dir)
        m = np.load(path)
        c = store.curves()
        steps = c.get("train_loss_steps", np.arange(len(c["train_loss"])))
        result = detect(steps, c["train_loss"], m["steps"], m["loss_test"], m["acc_test"], m["ssw_total"])
        result["rules"] = {"train_plateau_rel_drop": TRAIN_PLATEAU_REL_DROP, "grok_test_fraction": GROK_TEST_FRACTION,
                           "cleanup_ssw_fraction": CLEANUP_SSW_FRACTION}
        if store.meta.get("kind") == "paper":
            result["paper_boundaries"] = PAPER_BOUNDARIES
        (run_dir / "analysis" / "phases.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        log.info("%s: t1=%s t2=%s t3=%s t_grok=%s", store.run_id, result["t1"], result["t2"], result["t3"], result["t_grok"])
