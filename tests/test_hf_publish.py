"""Tests for the Hugging Face cards written by src/grokviz/hf_publish.py.

Global context: the model card is the first thing a Hub visitor reads about our runs, so its counts must match the
run folders (random-label controls are not trained on (a + b) mod 113).
"""

import json  # meta.json fixtures

from grokviz.hf_publish import model_card  # function under test


def test_model_card_counts_task_and_random_label_runs(tmp_path):
    """The card names how many runs learn (a + b) mod 113 and how many are random-label controls, and lists each."""
    for name, fn in (("wd1_s0", "add"), ("wd0_s0", "add"), ("rand_s0", "rand")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "meta.json").write_text(json.dumps({"title": f"{name} title", "fn_name": fn}))
    (tmp_path / "paper").mkdir()  # the paper's run is never part of our upload
    (tmp_path / "paper" / "meta.json").write_text(json.dumps({"title": "paper", "fn_name": "add"}))
    text, runs = model_card(tmp_path)
    assert runs == ["rand_s0", "wd0_s0", "wd1_s0"]
    assert "3 one-layer transformers" in text and "2 on (a + b) mod 113 and 1 random-label control" in text
    assert "- `rand_s0` — rand_s0 title" in text
