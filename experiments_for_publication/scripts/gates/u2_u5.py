"""U2 (P2 correctness) + U5-partial (P5 endpoint agreement).

U2a  importance-ratio deviation: constrained vs full-vocab log-probs on a
     frozen, unchanged policy. A correct implementation gives ratio == 1.
U2b  adaptive-KL controller: buggy `KL - target/target` vs correct
     `(KL - target)/target`, simulated on the REAL KL trajectory from a
     retained train_log.csv.
U2c  baseline action-leakage: size the within-context reward signal that a
     completion-conditioned critic can absorb.
U5   rank correlation among the in-silico endpoints available without IgFold.
"""

import glob
import json
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from scipy.stats import spearmanr

sys.path.insert(0, "/home/ruman/repos/protein_tune_rl")
from protein_tune_rl.protein_trainer.online_rl_trainer import OnlineRLSampler  # noqa
from protein_tune_rl.protein_trainer.trainer import Trainer  # noqa
from protein_tune_rl.util.util import compute_logp  # noqa

REPO = "/home/ruman/repos/protein_tune_rl"
EVAL = f"{REPO}/data/examples/eval_data_50_v3.csv"


class S(Trainer, OnlineRLSampler):
    def __init__(self, config):
        Trainer.__init__(self, config)
        OnlineRLSampler.__init__(self, config)


def u2a_u2c():
    run_dir = sorted(glob.glob(f"{REPO}/test_run/online_rl_trainer/*/*steps_200000*"))[0]
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    cfg["dataset"]["data_directory"] = EVAL
    cfg["checkpoint"] = os.path.join(run_dir, "a2c_s82000.ckpt")
    cfg.setdefault("KL_penalty", {"weight": 0.0, "target": None})
    s = S(cfg)

    from protein_tune_rl.metrics import create_metric
    m = cfg["metric"][0]
    metric = create_metric(name=m["name"])(**m["params"])

    cons = torch.as_tensor(s._constrained_ids, dtype=torch.long, device=s.device)

    devs, ctx_rewards = [], []
    torch.manual_seed(7)
    for batch in iter(s.dataloader):
        tok = s.collator(batch)
        with torch.no_grad():
            seqs, logp_sampler, *_ = s._sample_batch(s.policy, tok)
        state = {k: tok[k].to(s.device) for k in
                 ("input_ids", "attention_mask", "position_ids")}
        action = tok["input_ids"][:, tok["init_size"]:].to(s.device)
        with torch.no_grad():
            lp_con = compute_logp(s.policy, state, action, cons)
            lp_full = compute_logp(s.policy, state, action, None)
        ratio = torch.exp(lp_full - lp_con)
        devs.append((ratio - 1.0).abs().cpu())

        # within-context reward spread (the signal leakage would absorb)
        for i, full in enumerate(seqs):
            ctx_rewards.append(float(metric({
                "L": batch["LC"][i], "H": full,
                "seq_pre_mask": tok["seq_pre_mask"][i],
                "seq_post_mask": tok["seq_post_mask"][i]})))
        if len(devs) >= 6:
            break

    d = torch.cat(devs).numpy()
    print("=== U2a  importance-ratio deviation (frozen, unchanged policy) ===")
    print("    correct implementation => |ratio - 1| == 0")
    print(f"    full-vocab vs constrained-support, n={len(d)} sequences")
    print(f"    mean |ratio-1| = {d.mean():.4f}   median = {np.median(d):.4f}"
          f"   max = {d.max():.4f}")
    print(f"    >1% off: {100*(d>0.01).mean():.0f}% of sequences;"
          f"  >5% off: {100*(d>0.05).mean():.0f}%")

    r = np.array(ctx_rewards)
    print("\n=== U2c  signal a completion-conditioned critic can absorb ===")
    print(f"    reward is a deterministic function of the completion, so a critic")
    print(f"    seeing the completion can in principle predict it exactly.")
    print(f"    reward spread across sampled completions: std = {r.std():.3f}"
          f"  (range {r.min():.2f}..{r.max():.2f})")
    print(f"    that std IS the policy-gradient signal; leakage drives it to ~0")

    del s
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def u2b():
    print("\n=== U2b  adaptive-KL controller, buggy vs correct ===")
    logs = [p for p in glob.glob(f"{REPO}/test_run/online_rl_trainer/**/train_log.csv",
                                 recursive=True)]
    best = max(logs, key=lambda p: sum(1 for _ in open(p)))
    df = pd.read_csv(best)
    kl = df["KL_mean"].to_numpy() if "KL_mean" in df else None
    if kl is None:
        print("    no KL_mean column; skipped")
        return
    kl = np.nan_to_num(kl, nan=0.0)
    print(f"    real KL trajectory from {os.path.basename(os.path.dirname(best))}")
    print(f"    {len(kl)} steps, KL range {kl.min():.3f}..{kl.max():.3f}")

    for target in (6.0, 12.0):
        for label, fn in (
            ("correct (KL-t)/t", lambda k, t: (k - t) / t),
            ("buggy   KL-t/t  ", lambda k, t: k - t / t),
        ):
            beta, K_b = 0.02, 0.1
            for k in kl:
                e = float(np.clip(fn(k, target), -0.2, 0.2))
                beta = beta * (1 + K_b * e)
            print(f"    target={target:>5.1f}  {label} -> final beta = {beta:.6g}"
                  f"  ({beta/0.02:.3g}x initial)")
    print("    NOTE: every shipped config has target=null, so this controller")
    print("          never executed in any retained run (guarded by `if self.target")
    print("          is not None`). The bug is real but was never reached.")


def u5():
    print("\n=== U5 (partial)  endpoint agreement, no IgFold required ===")
    df = pd.read_csv(f"{REPO}/data/examples/training_data_v3.csv")
    sub = df.sample(n=250, random_state=0).reset_index(drop=True)
    from protein_tune_rl.metrics.esm_scoring import ESMSequenceScoring
    sc_reg = ESMSequenceScoring(model_name="facebook/esm2_t6_8M_UR50D",
                                region_only=True, reduction="mean",
                                local_files_only=True)
    sc_ch = ESMSequenceScoring(model_name="facebook/esm2_t6_8M_UR50D",
                               region_only=False, reduction="mean",
                               local_files_only=True)
    rows = []
    for _, r in sub.iterrows():
        hc, cdr = str(r.HC), str(r.HCDR3)
        i = hc.find(cdr)
        if i < 0:
            continue
        pre, post = hc[:i], hc[i + len(cdr):]
        ch = {"H": hc, "L": "", "seq_pre_mask": pre, "seq_post_mask": post}
        rows.append(dict(sheet=r.ss_perc_sheet, sasa=r.sasa, cdr_len=len(cdr),
                         esm_region=sc_reg(ch), esm_chain=sc_ch(ch)))
    e = pd.DataFrame(rows)
    cols = ["sheet", "sasa", "cdr_len", "esm_region", "esm_chain"]
    print(f"    n={len(e)} native antibodies; Spearman rho\n")
    print("              " + "".join(f"{c:>12}" for c in cols))
    for a in cols:
        line = f"{a:<12}"
        for b in cols:
            line += f"{spearmanr(e[a], e[b]).correlation:>12.2f}"
        print(line)


def main():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "24877")
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo",
                            rank=0, world_size=1, timeout=timedelta(seconds=120))
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    u2a_u2c()
    u2b()
    u5()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
