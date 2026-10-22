"""Which checkpoints' weights the website gets (src/grokviz/export.py snapshot_steps).

Global context: the browser recomputes every per-input view (attention, neurons, logits, the forward pass) from
float16 weight snapshots, so a run can be animated through training only where it has snapshots. Dense snapshots
cover memorisation, circuit formation and cleanup; sparse ones the rest, keeping the site under its size budget.
"""

from grokviz.export import snapshot_steps  # function under test

STEPS = list(range(0, 40001, 100))  # our runs: a checkpoint every 100 steps


def test_dense_through_cleanup_then_sparse():
    """Every 1,000 steps up to t3 + 2,000, then every 4,000, always including the first and last checkpoint."""
    got = [STEPS[i] for i in snapshot_steps(STEPS, t3=12900, fine=1000, coarse=4000)]
    assert got[:3] == [0, 1000, 2000] and 14000 in got and 15000 not in got and 16000 in got and got[-1] == 40000
    assert len(got) == 22 and got == sorted(set(got))


def test_runs_that_never_grok_get_sparse_snapshots():
    """No detected t3 (no weight decay, random labels): every 4,000 steps."""
    assert [STEPS[i] for i in snapshot_steps(STEPS, t3=None, fine=1000, coarse=4000)] == list(range(0, 40001, 4000))


def test_no_spacing_keeps_the_final_checkpoint_only():
    """An empty setting (GROKVIZ_SITE_WEIGHT_SNAPSHOTS=) reproduces the old export: final weights only."""
    assert snapshot_steps(STEPS, t3=12900, fine=0, coarse=0) == [len(STEPS) - 1]
