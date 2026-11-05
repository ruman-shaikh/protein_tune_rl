"""Unit tests for the online-RL correctness fixes.

These exercise the pure-tensor logic (no IgLM weights / no GPU required):
  * util.compute_logp constrained-support masking            (fix A6)
  * A2C._compute_gae truncation bootstrap                    (fix A2C truncation)
  * KLPenalty adaptive-beta controller math + scalar beta    (fix A1)

Run:  PYTHONPATH=. /home/ruman/venvs/general/bin/python -m pytest tests/test_fixes.py -v
"""

import math

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutput

from protein_tune_rl.util.util import compute_logp
from protein_tune_rl.optimizer.a2c import A2C
from protein_tune_rl.protein_trainer.online_rl_trainer import KLPenalty
from protein_tune_rl.metrics.ss_perc_sheet import PercBetaSheet


class FixedLogitsModel(nn.Module):
    """Minimal stand-in for a causal LM: returns a fixed logits tensor."""

    def __init__(self, logits):
        super().__init__()
        self.register_buffer("_logits", logits)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, **kw):
        return CausalLMOutput(logits=self._logits)


# --------------------------------------------------------------------------- #
# A6: compute_logp constrained-support masking
# --------------------------------------------------------------------------- #
def test_compute_logp_masking_renormalizes_over_constrained_support():
    B, T, V = 1, 3, 5  # state length 3, action length 2, vocab 5
    # Position 0 puts a lot of mass on the constrained token id 0.
    logits = torch.zeros(B, T, V)
    logits[0, 0, 0] = 5.0  # constrained token dominates at first action position
    model = FixedLogitsModel(logits)

    state = {
        "input_ids": torch.zeros(B, T, dtype=torch.long),
        "attention_mask": torch.ones(B, T, dtype=torch.long),
        "position_ids": torch.arange(T).unsqueeze(0),
    }
    action = torch.tensor([[1, 1]])  # never a constrained id
    constrained = torch.tensor([0], dtype=torch.long)

    logp_full = compute_logp(model, state, action)
    logp_masked = compute_logp(model, state, action, constrained)

    # Masking token 0 frees its probability mass, so the (unconstrained) action
    # token's log-prob must strictly increase.
    assert logp_masked.item() > logp_full.item()

    # Cross-check by hand. The constrained token is masked at EVERY position in
    # the slice (matching the sampler, which masks it at every generation step).
    row0 = logits[0, 0].clone()
    row0[0] = float("-inf")
    row1 = logits[0, 1].clone()
    row1[0] = float("-inf")
    lp0 = torch.log_softmax(row0, dim=-1)[1]
    lp1 = torch.log_softmax(row1, dim=-1)[1]
    assert math.isclose(logp_masked.item(), (lp0 + lp1).item(), rel_tol=1e-6)


def test_compute_logp_default_unchanged():
    B, T, V = 2, 4, 6
    torch.manual_seed(0)
    logits = torch.randn(B, T, V)
    model = FixedLogitsModel(logits)
    state = {
        "input_ids": torch.zeros(B, T, dtype=torch.long),
        "attention_mask": torch.ones(B, T, dtype=torch.long),
        "position_ids": torch.arange(T).unsqueeze(0).expand(B, T),
    }
    action = torch.tensor([[1, 2, 3], [4, 5, 0]])  # length 3
    # None => behaves exactly as before the fix.
    out = compute_logp(model, state, action, None)
    assert out.shape == (B,)


# --------------------------------------------------------------------------- #
# A2C GAE: truncation bootstrap
# --------------------------------------------------------------------------- #
def _make_a2c():
    a2c = A2C.__new__(A2C)  # bypass model construction
    a2c.gamma = 0.9
    a2c.gae_lambda = 0.95
    return a2c


def test_gae_terminated_has_no_bootstrap():
    a2c = _make_a2c()
    R = 2.0
    rewards = torch.tensor([[0.0, 0.0, R]])
    values = torch.tensor([[0.5, 0.7, 0.9]])
    mask = torch.ones(1, 3)
    last_values = torch.tensor([1.3])

    adv_term = a2c._compute_gae(
        rewards, values, mask, last_values=last_values, truncated=torch.tensor([0.0])
    )
    adv_none = a2c._compute_gae(rewards, values, mask)  # legacy path

    # A truly terminated episode must ignore last_values entirely.
    assert torch.allclose(adv_term, adv_none, atol=1e-6)
    # Terminal advantage = R - V(s_{T-1}), no bootstrap.
    assert math.isclose(adv_term[0, 2].item(), R - values[0, 2].item(), rel_tol=1e-6)


def test_gae_truncated_bootstraps_last_value():
    a2c = _make_a2c()
    R = 2.0
    rewards = torch.tensor([[0.0, 0.0, R]])
    values = torch.tensor([[0.5, 0.7, 0.9]])
    mask = torch.ones(1, 3)
    last_values = torch.tensor([1.3])

    adv_trunc = a2c._compute_gae(
        rewards, values, mask, last_values=last_values, truncated=torch.tensor([1.0])
    )
    adv_term = a2c._compute_gae(
        rewards, values, mask, last_values=last_values, truncated=torch.tensor([0.0])
    )

    # Terminal step should now bootstrap: delta gains gamma * V(s_T).
    expected_terminal = R + a2c.gamma * last_values[0].item() - values[0, 2].item()
    assert math.isclose(adv_trunc[0, 2].item(), expected_terminal, rel_tol=1e-6)
    # And it must differ from the terminated case by exactly gamma * V(s_T).
    diff = adv_trunc[0, 2].item() - adv_term[0, 2].item()
    assert math.isclose(diff, a2c.gamma * last_values[0].item(), rel_tol=1e-6)


def test_gae_padding_rows_stay_zero_and_dont_leak_bootstrap():
    a2c = _make_a2c()
    # Episode of true length 2 inside a width-3 batch (last column padded).
    rewards = torch.tensor([[0.0, 1.0, 0.0]])
    values = torch.tensor([[0.4, 0.6, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    last_values = torch.tensor([5.0])  # large; must NOT leak in (row terminated early)

    adv = a2c._compute_gae(
        rewards, values, mask, last_values=last_values, truncated=torch.tensor([0.0])
    )
    # Padded position contributes zero advantage.
    assert adv[0, 2].item() == 0.0
    # Real terminal (index 1) = R - V, no bootstrap from the padded column.
    assert math.isclose(adv[0, 1].item(), 1.0 - 0.6, rel_tol=1e-6)


# --------------------------------------------------------------------------- #
# A1: KLPenalty adaptive-beta controller
# --------------------------------------------------------------------------- #
def _kl_sequences(seq_len=4, init_size=2):
    return {
        "input_ids": torch.zeros(1, seq_len, dtype=torch.long),
        "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        "position_ids": torch.arange(seq_len).unsqueeze(0),
        "init_size": init_size,
    }


def test_klpenalty_beta_constant_when_no_target():
    ref = FixedLogitsModel(torch.zeros(1, 4, 5))  # uniform ref
    kl = KLPenalty(ref, weight=0.05, target=None, device=torch.device("cpu"))
    logp = torch.tensor([-1.0])
    out = kl(logp, _kl_sequences())

    assert kl.beta == 0.05  # untouched
    # returned penalty == -beta * (logp - ref_logp)
    ref_logp = 2 * math.log(1 / 5)  # two action tokens, uniform over 5
    assert math.isclose(out.item(), -0.05 * (logp.item() - ref_logp), rel_tol=1e-6)


def test_klpenalty_adaptive_beta_uses_correct_formula():
    ref = FixedLogitsModel(torch.zeros(1, 4, 5))
    target = 2.0
    kl = KLPenalty(ref, weight=0.05, target=target, device=torch.device("cpu"))
    logp = torch.tensor([-1.0])

    ref_logp = 2 * math.log(1 / 5)
    mean_KL = logp.item() - ref_logp
    e_t = max(-0.2, min(0.2, (mean_KL - target) / target))
    expected_beta = 0.05 * (1 + 0.1 * e_t)

    kl(logp, _kl_sequences())

    assert isinstance(kl.beta, float)  # scalar, not a tensor (the old bug)
    assert math.isclose(kl.beta, expected_beta, rel_tol=1e-6)  # float32 precision


def test_klpenalty_target_actually_matters():
    # Regression guard for `KL - target/target` (== KL - 1): with the old bug
    # the beta update was identical for every target. It must not be now.
    ref = FixedLogitsModel(torch.zeros(1, 4, 5))
    logp = torch.tensor([-1.0])

    kl_a = KLPenalty(ref, weight=0.05, target=1.5, device=torch.device("cpu"))
    kl_b = KLPenalty(ref, weight=0.05, target=3.0, device=torch.device("cpu"))
    kl_a(logp, _kl_sequences())
    kl_b(logp, _kl_sequences())

    assert not math.isclose(kl_a.beta, kl_b.beta, rel_tol=1e-9)


# --------------------------------------------------------------------------- #
# B1: region-aware beta-sheet reward
# --------------------------------------------------------------------------- #
def test_ss_perc_sheet_region_only_scores_infill_span():
    pre, infill, post = "EVQLVESGG", "WYWFWY", "LTVSS"
    heavy = pre + infill + post
    chains = {
        "H": heavy,
        "L": "DIQMTQ",
        "seq_pre_mask": pre,
        "seq_post_mask": post,
    }

    whole = PercBetaSheet(region_only=False)(chains)
    region = PercBetaSheet(region_only=True)(chains)

    # region_only must equal scoring the infilled span in isolation ...
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    expected = ProteinAnalysis(infill).secondary_structure_fraction()[2]
    assert math.isclose(region, expected, rel_tol=1e-9)
    # ... and differ from the diluted whole-chain score.
    assert not math.isclose(region, whole, rel_tol=1e-6)


def test_ss_perc_sheet_default_is_backward_compatible():
    chains = {"H": "EVQLWYWF", "L": "DIQMTQ"}
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    expected = ProteinAnalysis("EVQLWYWF" + "DIQMTQ").secondary_structure_fraction()[2]
    assert math.isclose(PercBetaSheet()(chains), expected, rel_tol=1e-9)
