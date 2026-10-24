"""Tests for the run format and the paper-run phases (milestone M3).

Global context: every analysis iterates over checkpoints through `RunStore`; the paper run must load with
the exact layout and reproduce its own logged losses, otherwise all downstream figures would be wrong.
"""

import json  # read validation.json

import numpy as np  # curves
import pytest  # markers
import torch  # tensors

from grokviz.__main__ import main  # CLI entry point (runs phases like a user)
from grokviz.model import GrokTransformer, ModelShape  # architecture
from grokviz.runstore import RunStore, write_run  # module under test


def test_runstore_roundtrip(tmp_path):
    """Stacked checkpoints written by write_run come back identical, per index, as loadable models."""
    models = [GrokTransformer(ModelShape(), seed=s) for s in (1, 2, 3)]  # three fake checkpoints
    sds = [m.state_dict() for m in models]
    stacked = {k: torch.stack([sd[k] for sd in sds]) for k in sds[0]}
    meta = {"run_id": "fake", "ckpt_steps": [0, 100, 200], "model_shape": ModelShape().__dict__}
    write_run(tmp_path / "fake", meta, stacked, {"train_loss": np.arange(3.0)})
    store = RunStore(tmp_path / "fake")
    assert store.ckpt_steps == [0, 100, 200] and store.run_id == "fake"
    assert torch.equal(store.state_dict(1)["embed.W_E"], sds[1]["embed.W_E"])
    assert torch.equal(store.state_dict(-1)["unembed.W_U"], sds[2]["unembed.W_U"])
    loaded = [(t, step, m.embed.W_E.clone()) for t, step, m in store.iter_models()]
    assert [(t, s) for t, s, _ in loaded] == [(0, 0), (1, 100), (2, 200)]
    assert torch.equal(loaded[2][2], sds[2]["embed.W_E"])
    assert store.curves()["train_loss"].tolist() == [0.0, 1.0, 2.0]


@pytest.mark.network
@pytest.mark.slow
def test_paper_run_phases_reproduce_logged_losses(monkeypatch):
    """Acquire → convert → validate the paper run like a user; validation.json must report every check ok."""
    exit_code = main(["--phases", "acquire_paper_sources,convert_paper_run,validate_paper_run"])
    assert exit_code == 0
    from grokviz.config import load_config

    cfg = load_config()
    report = json.loads((cfg.runs_dir / "paper" / "analysis" / "validation.json").read_text(encoding="utf-8"))
    assert all(c["ok"] for c in report["checks"])
    assert report["epoch40k_matches_paper"]
    store = RunStore(cfg.runs_dir / "paper")
    assert len(store.ckpt_steps) == 500 and store.ckpt_steps[400] == 40001


@pytest.mark.skipif(__import__("os").name != "nt", reason="the read-only attribute blocks deletion only on Windows")
def test_remove_tree_deletes_read_only_folders(tmp_path):
    """runstore.remove_tree removes a tree whose folder and file carry the read-only bit (on Windows plain
    shutil.rmtree fails there with "Access is denied", which stopped export_site_data and render_media)."""
    import os
    import stat

    from grokviz.runstore import remove_tree

    d = tmp_path / "data" / "runs" / "paper"
    d.mkdir(parents=True)
    (d / "w.bin").write_bytes(b"\0")
    os.chmod(d / "w.bin", stat.S_IREAD)
    os.chmod(d, stat.S_IREAD | stat.S_IEXEC)
    remove_tree(tmp_path / "data")
    assert not (tmp_path / "data").exists()
