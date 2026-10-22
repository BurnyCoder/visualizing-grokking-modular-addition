"""Bridge test: prepare PyTorch reference outputs, then run the JavaScript model tests with Node (milestone M8).

Global context: the website computes per-input activations in the browser with web/js/model/grokmodel.js;
this test guarantees those numbers equal the PyTorch model's (the one every analysis uses), for float32
weights and for the float16 weights actually shipped to the site.
Source: Node.js test runner (https://nodejs.org/api/test.html).
"""

import json  # fixture
import shutil  # find node
import subprocess  # run node --test
from types import SimpleNamespace  # minimal grid for the ablation reference

import numpy as np  # binary weights
import pytest  # skip if node missing
import torch  # reference model

from grokviz import fourier, interventions, losses, task  # basis, ablation reference, loss, inputs
from grokviz.model import GrokTransformer  # reference implementation

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_js_model_matches_pytorch(tmp_path):
    """Write weights + reference outputs, run tests/js/grokmodel.test.mjs, require all Node tests to pass."""
    torch.manual_seed(0)
    model = GrokTransformer(seed=11)
    with torch.no_grad():  # give biases non-zero values so their handling is tested too
        model.blocks[0].mlp.b_in.normal_(0, 0.3)
        model.blocks[0].mlp.b_out.normal_(0, 0.3)
    state = model.state_dict()
    names = list(state)
    flat = np.concatenate([state[n].numpy().reshape(-1) for n in names]).astype("<f4")
    (tmp_path / "weights_f32.bin").write_bytes(flat.tobytes())
    (tmp_path / "weights_f16.bin").write_bytes(flat.astype("<f2").tobytes())
    m16 = GrokTransformer(seed=11)  # same model with weights rounded to float16 (what the site ships)
    m16.load_state_dict({n: torch.from_numpy(state[n].numpy().astype(np.float16).astype(np.float32)) for n in names})
    tokens = task.all_pairs(113)
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)
        logits16 = m16(tokens)
    rng = np.random.default_rng(0)
    rows = rng.choice(113 * 113, 64, replace=False)
    samples = [{"a": int(r // 113), "b": int(r % 113), "logits": logits[r, -1].tolist(), "logits_f16": logits16[r, -1].tolist(),
                "pre": cache["blocks.0.mlp.pre"][r, -1].tolist()} for r in rows]
    L, labels = logits[:, -1, :113].double(), (tokens[:, 0] + tokens[:, 1]) % 113  # all-pairs logits and right answers
    grid = SimpleNamespace(p=113, F=fourier.fourier_basis(113)[0].double())  # the two fields ablate_logit_freqs reads
    loss = lambda x: losses.grid_loss(x, labels).item()  # noqa: E731 — cross-entropy over all pairs
    ablation = {"original": loss(L), "remove_3_7": loss(interventions.ablate_logit_freqs(L, grid, zero=[3, 7])),
                "keep_1_2": loss(interventions.ablate_logit_freqs(L, grid, keep_only=[1, 2]))}
    ref = {"p": 113, "layout": [{"name": n, "shape": list(state[n].shape)} for n in names], "samples": samples, "ablation": ablation,
           "attn_grid_h0": cache["blocks.0.attn.pattern"][:, 0, 2, 0].tolist(), "neuron_grid_7": cache["blocks.0.mlp.post"][:, -1, 7].tolist()}
    (tmp_path / "reference.json").write_text(json.dumps(ref), encoding="utf-8")
    result = subprocess.run(["node", "--test", str(ROOT / "tests" / "js" / "grokmodel.test.mjs")], capture_output=True, text=True,
                            env={**__import__("os").environ, "GROKVIZ_JS_FIXTURE": str(tmp_path)}, cwd=ROOT)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-2000:]
