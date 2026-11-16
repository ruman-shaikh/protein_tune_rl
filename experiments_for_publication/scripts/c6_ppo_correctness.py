"""C6 - PPO implementation correctness, measured properly.

Two measurements the earlier gate (U2) left incomplete.

C6a  Importance-ratio deviation on BASE IgLM, not a degenerate trained policy.
     On a frozen, unchanged policy a correct PPO gives ratio == 1 exactly.
     The pre-fix code evaluated PPO/reference log-probs over the full
     vocabulary while the sampler masked 12 control tokens to -inf, so the
     two measures differ. U2 measured 0.286 mean |ratio-1| on a2c_s82000,
     whose output distribution is degenerate; base IgLM is the distribution
     the published runs actually started from.

C6b  Baseline action-leakage, measured rather than argued.
     PPO's StateValue baseline must be a function of the prompt only. The
     pre-fix code conditioned it on prompt + sampled completion. Since the
     reward is a deterministic function of the completion, a
     completion-conditioned critic can in principle predict it exactly,
     driving the advantage to zero and removing the learning signal.

     An untrained critic head correlates with nothing, so instead we measure
     the INFORMATION AVAILABLE to each view: ridge-regress the pooled final
     hidden state onto the realised reward, held-out R^2, for prompt-only vs
     prompt+completion. That is an upper bound on what a critic could learn
     from each view.
"""

import json
import os
import sys
import warnings
from datetime import timedelta

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

sys.path.insert(0, "/home/ruman/repos/protein_tune_rl")
from protein_tune_rl.metrics import create_metric  # noqa: E402
from protein_tune_rl.protein_trainer.online_rl_trainer import OnlineRLSampler  # noqa
from protein_tune_rl.protein_trainer.trainer import Trainer  # noqa
from protein_tune_rl.util.util import compute_logp  # noqa: E402

REPO = "/home/ruman/repos/protein_tune_rl"
OUT = f"{REPO}/experiments_for_publication/results"
N_BATCHES = 8


class Sampler(Trainer, OnlineRLSampler):
    def __init__(self, config):
        Trainer.__init__(self, config)
        OnlineRLSampler.__init__(self, config)


def main():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "25177")   # distinct from the training run
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo",
                            rank=0, world_size=1, timeout=timedelta(seconds=180))
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    os.makedirs(OUT, exist_ok=True)

    # Base IgLM, no checkpoint: the distribution the published runs started from.
    cfg = json.load(open(f"{REPO}/configs/examples/a2c_iglm_hcdr3_beta_sheet.json"))
    cfg["dataset"]["data_directory"] = f"{REPO}/data/examples/eval_data_50_v3.csv"
    cfg.pop("checkpoint", None)
    cfg.setdefault("KL_penalty", {"weight": 0.0, "target": None})
    s = Sampler(cfg)
    m = cfg["metric"][0]
    metric = create_metric(name=m["name"])(**m["params"])

    cons = torch.as_tensor(s._constrained_ids, dtype=torch.long, device=s.device)
    base = s._unwrap_ddp_model(s.policy)

    devs, rewards, h_prompt, h_full = [], [], [], []
    torch.manual_seed(20261116)
    for bi, batch in enumerate(iter(s.dataloader)):
        if bi >= N_BATCHES:
            break
        tok = s.collator(batch)
        with torch.no_grad():
            seqs, *_ = s._sample_batch(s.policy, tok)

        init = tok["init_size"]
        state = {k: tok[k].to(s.device) for k in
                 ("input_ids", "attention_mask", "position_ids")}
        action = tok["input_ids"][:, init:].to(s.device)

        # --- C6a: ratio over full vocab vs constrained support ---
        with torch.no_grad():
            lp_con = compute_logp(s.policy, state, action, cons)
            lp_full = compute_logp(s.policy, state, action, None)
        devs.append((torch.exp(lp_full - lp_con) - 1.0).abs().cpu())

        # --- C6b: information available to each critic view ---
        prompt_state = {k: v[:, :init] for k, v in state.items()}
        with torch.no_grad():
            for store, st in ((h_prompt, prompt_state), (h_full, state)):
                o = base.model(input_ids=st["input_ids"],
                               attention_mask=st["attention_mask"],
                               position_ids=st["position_ids"],
                               output_hidden_states=True, return_dict=True)
                hs = o["hidden_states"][-1]
                mask = st["attention_mask"].unsqueeze(-1).to(hs.dtype)
                pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)
                store.append(pooled.float().cpu().numpy())

        for i, full in enumerate(seqs):
            rewards.append(float(metric({
                "L": batch["LC"][i], "H": full,
                "seq_pre_mask": tok["seq_pre_mask"][i],
                "seq_post_mask": tok["seq_post_mask"][i]})))

    d = torch.cat(devs).numpy()
    y = np.array(rewards)
    Xp = np.concatenate(h_prompt)
    Xf = np.concatenate(h_full)

    print("=== C6a  importance-ratio deviation, BASE IgLM (frozen, unchanged) ===")
    print("    a correct implementation gives |ratio - 1| == 0")
    print(f"    n={len(d)} sequences")
    print(f"    mean |ratio-1| = {d.mean():.4f}   median = {np.median(d):.4f}"
          f"   max = {d.max():.4f}")
    print(f"    >1% off: {100*(d>0.01).mean():.0f}%   "
          f">5% off: {100*(d>0.05).mean():.0f}%   "
          f">20% off (PPO clip band): {100*(d>0.20).mean():.0f}%")
    print("    for reference, U2 measured 0.286 on the degenerate a2c_s82000 policy")

    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score

    print("\n=== C6b  information available to each critic view ===")
    print("    held-out R^2 predicting the realised reward from the pooled")
    print("    final hidden state (5-fold CV, ridge). Upper bound on what a")
    print(f"    critic could learn from each view. n={len(y)}\n")
    rows = []
    for label, X in (("prompt only (correct)", Xp),
                     ("prompt + completion (leaking)", Xf)):
        sc = cross_val_score(Ridge(alpha=1.0), X, y, cv=5, scoring="r2")
        print(f"    {label:<32} R^2 = {sc.mean():+.3f}  (+/- {sc.std():.3f})")
        rows.append(dict(view=label, r2_mean=sc.mean(), r2_std=sc.std()))
    print(f"\n    reward spread across completions: std = {y.std():.3f}"
          f"  (range {y.min():.2f}..{y.max():.2f})")
    print("    that spread is the policy-gradient signal; a leaking baseline")
    print("    absorbs it into the advantage and the gradient vanishes")

    pd.DataFrame(rows).assign(
        ratio_dev_mean=d.mean(), ratio_dev_median=float(np.median(d)),
        ratio_dev_max=d.max(), n_sequences=len(d),
        reward_std=y.std(),
    ).to_csv(f"{OUT}/c6_ppo_correctness.csv", index=False)
    print(f"\nartifact: {OUT}/c6_ppo_correctness.csv")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
