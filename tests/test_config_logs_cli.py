"""Tests for configuration loading, logging setup and the CLI wrapper (milestone M1).

Global context: every pipeline phase reads its settings from `grokviz.config.load_config`, writes
timestamped logs through `grokviz.logs.setup_logging`, and is dispatched by `grokviz.__main__.main`.
These tests pin that contract before the scientific code is written (test-driven development).
"""

from pathlib import Path  # filesystem paths for temporary log directories

import pytest  # test framework: fixtures (tmp_path, monkeypatch) and markers

from grokviz import config as config_module  # module under test: .env-driven configuration
from grokviz.__main__ import main  # CLI entry point under test
from grokviz.logs import setup_logging  # logging setup under test


def test_defaults_come_from_env_example(monkeypatch):
    """With no .env and no overrides, values equal the documented defaults in .env.example."""
    # Remove any GROKVIZ_* variable leaking from the developer's shell so only file defaults apply.
    for key in list(config_module.os.environ):
        if key.startswith("GROKVIZ_"):
            monkeypatch.delenv(key, raising=False)
    # Load with an env file that does not exist, so only .env.example supplies values.
    cfg = config_module.load_config(env_file=Path("does-not-exist.env"))
    # Paper mainline task/model settings (Nanda et al. 2023, Sec. 3).
    assert cfg.p == 113
    assert cfg.frac_train == pytest.approx(0.3)
    assert (cfg.d_model, cfg.d_mlp, cfg.n_heads, cfg.d_head, cfg.n_ctx) == (128, 512, 4, 32, 3)
    # Optimiser settings from the paper's released training code.
    assert cfg.lr == pytest.approx(1e-3)
    assert cfg.weight_decay == pytest.approx(1.0)
    assert (cfg.beta1, cfg.beta2) == (pytest.approx(0.9), pytest.approx(0.98))
    # Run matrix: 11 confirmatory seeds 0..10 (power analysis in docs/preregistration.md).
    assert cfg.wd1_seeds == list(range(11))


def test_environment_variables_override_files(monkeypatch):
    """A real environment variable wins over both .env and .env.example."""
    monkeypatch.setenv("GROKVIZ_P", "7")  # override the modulus
    cfg = config_module.load_config(env_file=Path("does-not-exist.env"))
    assert cfg.p == 7


@pytest.mark.parametrize(
    ("spec", "expected"),
    [("0-2", [0, 1, 2]), ("0-2,5", [0, 1, 2, 5]), ("3", [3]), ("", [])],
)
def test_parse_seed_list(spec, expected):
    """Seed specs accept inclusive ranges and comma lists."""
    assert config_module.parse_seed_list(spec) == expected


def test_setup_logging_writes_unicode_to_timestamped_file(tmp_path):
    """Log lines (including math symbols) reach a timestamped file under the logs directory."""
    log_path = setup_logging(tmp_path, tag="unit")  # configure handlers; returns the file path
    import logging  # standard library logging, configured by setup_logging

    logging.getLogger("grokviz.test").info("ω_k = 2πk/p → key frequencies")  # non-ASCII on purpose
    for handler in logging.getLogger("grokviz").handlers:  # flush so the file is complete
        handler.flush()
    text = log_path.read_text(encoding="utf-8")  # read back what was written
    assert log_path.parent == tmp_path and log_path.name.endswith("_unit.log")
    assert "ω_k = 2πk/p → key frequencies" in text


def test_cli_runs_check_environment_phase(tmp_path, monkeypatch):
    """`grokviz --phases check_environment` exits 0 and logs the phase start and end."""
    monkeypatch.setenv("GROKVIZ_LOGS_DIR", str(tmp_path))  # keep test logs out of the repo
    exit_code = main(["--phases", "check_environment"])  # run the CLI in-process
    assert exit_code == 0
    log_files = list(tmp_path.glob("*.log"))  # exactly one timestamped log per invocation
    assert len(log_files) == 1
    text = log_files[0].read_text(encoding="utf-8")
    assert "phase check_environment: start" in text and "phase check_environment: done" in text


def test_cli_rejects_unknown_phase(tmp_path, monkeypatch):
    """Unknown phase names fail fast with a non-zero exit code instead of silently doing nothing."""
    monkeypatch.setenv("GROKVIZ_LOGS_DIR", str(tmp_path))
    assert main(["--phases", "no_such_phase"]) != 0


@pytest.mark.gpu
def test_auto_device_is_blackwell_gpu():
    """On the development machine 'auto' resolves to the RTX 5070 (compute capability 12.0)."""
    import torch  # only imported here so CPU-only test runs never need CUDA

    from grokviz.envcheck import resolve_device

    device = resolve_device("auto")
    assert device.type == "cuda"
    assert torch.cuda.get_device_capability(device) == (12, 0)
