"""End-to-end optimizer-step tests on a tiny GPT-2 (CPU, no IgLM weights).

Validates the model-dependent fixes:
  * A2C.step runs with constrained_ids + truncated and updates the policy.
  * PPO.step (baseline='state_value') computes the critic on the PROMPT ONLY
    (fix A3 -- no action leakage) and updates policy + value.

Run:  PYTHONPATH=. /home/ruman/venvs/general/bin/python tests/test_optimizer_integration.py
"""

import os

import pytest
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from transformers import GPT2Config, GPT2LMHeadModel

from protein_tune_rl.models.decoder import Decoder
from protein_tune_rl.optimizer.a2c import A2C
from protein_tune_rl.optimizer.ppo import PPO
from protein_tune_rl.util.util import compute_logp


def _init_dist():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29517")
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)


@pytest.fixture(scope="module", autouse=True)
def distributed_process_group():
    """Provide the single-process group required by DDP under pytest."""
    _init_dist()
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def _tiny_policy(vocab=20):
    cfg = GPT2Config(
        vocab_size=vocab,
        n_positions=32,
        n_embd=16,
        n_layer=2,
        n_head=2,
    )
    model = GPT2LMHeadModel(cfg)
    return DDP(Decoder(model, "gpt2"))


def _batch(vocab=20, B=2, init_size=3, action_len=4):
    seq_len = init_size + action_len
    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab, (B, seq_len))
    attention_mask = torch.ones(B, seq_len, dtype=torch.bool)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(B, seq_len).contiguous()
    action_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "init_size": init_size,
        "action_mask": action_mask,
        "truncated": torch.tensor([False, True]),
    }, B, init_size


def _snapshot(module):
    return [p.detach().clone() for p in module.parameters()]


def _changed(before, module):
    return any(not torch.equal(b, a) for b, a in zip(before, module.parameters()))


def test_a2c_step():
    vocab = 20
    policy = _tiny_policy(vocab)
    a2c = A2C(
        policy=policy,
        learning_rate=1e-2,
        constrained_ids=[0, 1, 2],  # exercise the masking path
        normalize_advantage=True,
        max_grad_norm=1.0,
    )
    batch, B, _ = _batch(vocab)
    reward = torch.tensor([1.0, -0.5])

    p_before = _snapshot(a2c.policy)
    v_before = _snapshot(a2c.state_value)
    a2c.step(reward, reward.mean(), torch.zeros(B), torch.zeros(B), batch)

    assert _changed(p_before, a2c.policy), "A2C did not update the policy"
    assert _changed(v_before, a2c.state_value), "A2C did not update the critic"
    print("PASS  A2C.step: policy + critic updated (constrained_ids + truncated OK)")


def test_ppo_state_value_uses_prompt_only():
    vocab = 20
    policy = _tiny_policy(vocab)
    ppo = PPO(
        policy=policy,
        learning_rate=1e-2,
        minibatch_size=2,
        baseline="state_value",
        constrained_ids=[0, 1, 2],
    )
    batch, B, init_size = _batch(vocab)
    reward = torch.tensor([1.0, -0.5])

    # Record the sequence length the critic is actually called with.
    seen_lengths = []
    real_value = ppo.state_value

    class Recorder:
        def __call__(self, **kw):
            seen_lengths.append(kw["input_ids"].shape[1])
            return real_value(**kw)

        def parameters(self):
            return real_value.parameters()

    ppo.state_value = Recorder()

    # Realistic old_logp (ratios start at 1), as during real training.
    state = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "position_ids": batch["position_ids"],
    }
    action = batch["input_ids"][:, init_size:]
    with torch.no_grad():
        old_logp = compute_logp(policy, state, action, ppo.constrained_ids)

    p_before = _snapshot(policy)
    v_before = _snapshot(real_value)
    ppo.step(reward, reward.mean(), old_logp, torch.zeros(B), batch)

    assert seen_lengths, "critic was never called"
    assert all(n == init_size for n in seen_lengths), (
        f"A3 regression: critic saw non-prompt lengths {seen_lengths} "
        f"(expected all == init_size={init_size})"
    )
    assert _changed(p_before, policy), "PPO did not update the policy"
    assert _changed(v_before, real_value), "PPO did not update the value net"
    print(
        f"PASS  PPO.step: critic saw only prompt-length inputs {seen_lengths} "
        f"(init_size={init_size}); policy + value updated"
    )


if __name__ == "__main__":
    _init_dist()
    try:
        test_a2c_step()
        test_ppo_state_value_uses_prompt_only()
        print("\nAll integration checks passed.")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
