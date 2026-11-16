"""Regression tests for the C0 unblocking fixes.

Pure-tensor; no IgLM weights, no GPU, no network.

  * compute_logp mask alignment invariant  (correct only by accident - pin it)
  * A2C advantage normalization is collective-safe and count-weighted
  * tune.py saves a config that still contains optimizer.name
  * StructureBasedMetric tolerates option dicts without do_refine
"""

import json
import sys
import types

import pytest
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutput

from protein_tune_rl.optimizer.a2c import A2C
from protein_tune_rl.util.util import compute_logp


class FixedLogitsModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("_logits", logits)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, **kw):
        return CausalLMOutput(logits=self._logits)


# --------------------------------------------------------------------------- #
# compute_logp mask alignment
# --------------------------------------------------------------------------- #
def test_compute_logp_mask_counts_actions_through_the_stop_token():
    """compute_logp's mask is shifted one position relative to the actions.

    `logp_mask = attention_mask[:, -A-1:-1]` pairs action i with the mask of
    action i-1 (and action 0 with the final prompt token, always attended).
    That shift exactly cancels the sampler marking the stop token's own
    attention_mask as False, so the token set scored here matches the
    sampler's logp_sum. It is correct BY ACCIDENT.

    If the deferred stop-token masking change is ever implemented, this test
    fails and PPO's importance ratio silently stops matching its own old
    log-probs. Do not "fix" this test; fix the caller.
    """
    B, V = 1, 6
    init_size, A = 3, 4           # prompt 3 tokens, 4 generated positions
    T = init_size + A

    logits = torch.zeros(B, T, V)
    model = FixedLogitsModel(logits)

    # Sampler emits 2 real tokens, then the stop token, then padding.
    # attention_mask over the action block is [True, True, False, False]:
    # the stop token's own entry is False (set from post-update `finished`).
    attention_mask = torch.tensor(
        [[True, True, True, True, True, False, False]], dtype=torch.bool
    )
    state = {
        "input_ids": torch.zeros(B, T, dtype=torch.long),
        "attention_mask": attention_mask,
        "position_ids": torch.arange(T).unsqueeze(0),
    }
    action = torch.ones(B, A, dtype=torch.long)

    logp = compute_logp(model, state, action)

    # Uniform logits over V, so every counted token contributes -log(V).
    # The sampler counts 3 tokens (2 real + the stop action), because its
    # step_mask uses the PRE-update `finished`. The shifted mask must agree.
    expected_tokens = 3
    assert torch.isclose(
        logp, torch.tensor([-expected_tokens * torch.log(torch.tensor(float(V)))])
    ), (
        "compute_logp mask alignment changed: it no longer counts exactly the "
        "tokens the sampler's logp_sum counts. PPO's importance ratio and the "
        "KL estimate are now inconsistent with the sampler."
    )


# --------------------------------------------------------------------------- #
# A2C advantage normalization
# --------------------------------------------------------------------------- #
def _bare_a2c():
    """An A2C instance without the DDP critic (which needs a process group)."""
    a = A2C.__new__(A2C)
    a.normalize_adv = True
    return a


def test_normalize_advantages_is_count_weighted():
    a = _bare_a2c()
    adv = torch.tensor([[1.0, 2.0, 3.0, 0.0], [4.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.bool)

    out = a._normalize_advantages(adv, mask)
    valid = out[mask]

    # Pooled over the 4 unmasked entries (1,2,3,4), not averaged per row.
    assert torch.isclose(valid.mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(valid.std(unbiased=False), torch.tensor(1.0), atol=1e-3)
    # Masked positions are untouched.
    assert out[0, 3] == 0.0 and out[1, 1] == 0.0


def test_normalize_advantages_handles_no_valid_actions():
    """Zero valid actions must return cleanly, not raise and not skip a collective.

    Previously the all_reduce sat inside `if valid_adv.numel() > 0`, so a rank
    in this state skipped it while its peers blocked, hanging the job.
    """
    a = _bare_a2c()
    adv = torch.zeros(2, 3)
    mask = torch.zeros(2, 3, dtype=torch.bool)
    out = a._normalize_advantages(adv, mask)
    assert torch.equal(out, adv)


def test_normalize_advantages_single_valid_action_is_not_divided_by_zero():
    a = _bare_a2c()
    adv = torch.tensor([[5.0, 0.0]])
    mask = torch.tensor([[1, 0]], dtype=torch.bool)
    out = a._normalize_advantages(adv, mask)
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------- #
# tune.py config snapshot
# --------------------------------------------------------------------------- #
def test_saved_config_retains_optimizer_name(tmp_path):
    """The saved config must survive a trainer popping keys from the live dict."""
    from protein_tune_rl.tune import ProteinTuneRL

    cfg = {"optimizer": {"name": "a2c", "learning_rate": 1e-5}, "trainer": {}}
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(cfg))

    obj = ProteinTuneRL.__new__(ProteinTuneRL)
    live = obj._load_config(str(cfg_file))

    # Simulate OnlineRLTrainer.__init__ mutating the live config.
    live["optimizer"].pop("name")

    obj.exp_output_dir = tmp_path
    obj._save_config()

    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["optimizer"]["name"] == "a2c"


# --------------------------------------------------------------------------- #
# StructureBasedMetric default options
# --------------------------------------------------------------------------- #
def test_structure_metric_accepts_options_without_do_refine(monkeypatch):
    """SASA and FoldingConfidence build option dicts lacking do_refine.

    Indexing those keys raised KeyError during construction, so no structure
    reward could ever be instantiated.
    """
    # Stub igfold so the import in structure.py succeeds without the package.
    fake = types.ModuleType("igfold")
    fake.IgFoldRunner = lambda **kw: types.SimpleNamespace(
        antiberty=types.SimpleNamespace(model=types.SimpleNamespace(to=lambda d: None)),
        models=[],
    )
    monkeypatch.setitem(sys.modules, "igfold", fake)

    import importlib

    mod = importlib.import_module("protein_tune_rl.metrics.structure")
    importlib.reload(mod)

    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        monkeypatch.setattr(mod.dist, "get_rank", lambda: 0)

    # The pre-fix code raised KeyError('do_refine') here.
    m = mod.StructureBasedMetric("igfold", options={"do_renum": False})
    assert m.options.get("do_refine") in (None, False)
