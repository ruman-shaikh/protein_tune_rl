"""C6' - reward-gaming detector, needing no held-out oracle.

Built from the degeneracy signatures measured in C1/C3/C4. Given a policy
checkpoint and the training reward, it reports five signals and a verdict,
using only the policy's own outputs - no structural oracle, no second model,
nothing the practitioner would not already have.

Signals
  1  length pinning        generated length at the sampler cap, or collapsed
                           to a tiny value; low length variance
  2  composition concentr.  share held by the top residues; positional entropy
  3  output determinism     unique fraction, mean within-context edit distance
  4  reward quantisation    number of DISTINCT reward values. A healthy reward
                           over varied contexts is near-continuous; a handful
                           of distinct values means the policy has found a
                           context-independent output whose score is fixed.
  5  reward-length coupling  correlation between generated length and reward,
                           which exposes ratio-style endpoints that can be
                           gamed by shrinking the denominator

Usage:
    python experiments_for_publication/scripts/c6prime_degeneracy_detector.py \
        --checkpoint <path.ckpt> --config <config.json> [--contexts 48]

Reference points from this project (see Path_Forward.md):
  pretrained IgLM      len ~14, top-residue share ~0.12, uniq 1.00
  compositional reward len 22.0 (cap), sheet-set share 0.999  -> GAMED
"""

import argparse
import json
import math
import os
import sys
import warnings
from collections import Counter
from datetime import timedelta

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

sys.path.insert(0, "/home/ruman/repos/protein_tune_rl")
from protein_tune_rl.metrics import create_metric  # noqa: E402
from protein_tune_rl.protein_trainer.online_rl_trainer import OnlineRLSampler  # noqa
from protein_tune_rl.protein_trainer.trainer import Trainer  # noqa


class Sampler(Trainer, OnlineRLSampler):
    def __init__(self, config):
        Trainer.__init__(self, config)
        OnlineRLSampler.__init__(self, config)


def lev(a, b):
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def entropy(xs):
    n = len(xs)
    if n == 0:
        return 0.0
    c = Counter(xs)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--config", required=True)
    ap.add_argument("--contexts", type=int, default=48)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--port", default="25301")
    ap.add_argument("--skip-reward", action="store_true",
                    help="skip signals 4-5; signals 1-3 need no reward, so "
                         "this avoids folding and runs in seconds")
    args = ap.parse_args()

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = args.port
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo",
                            rank=0, world_size=1, timeout=timedelta(seconds=300))
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    cfg = json.load(open(args.config))
    if args.checkpoint:
        cfg["checkpoint"] = args.checkpoint
    else:
        cfg.pop("checkpoint", None)
    cfg.setdefault("KL_penalty", {"weight": 0.0, "target": None})
    max_len = cfg["trainer"]["max_length"]

    s = Sampler(cfg)
    m = cfg["metric"][0]
    metric = None if args.skip_reward else create_metric(name=m["name"])(**m["params"])

    gens, lens, rewards, per_ctx = [], [], [], {}
    seen = 0
    for rep in range(args.samples):
        torch.manual_seed(5000 + rep)
        for batch in iter(s.dataloader):
            tok = s.collator(batch)
            with torch.no_grad():
                seqs, *_ = s._sample_batch(s.policy, tok)
            for i, full in enumerate(seqs):
                pre, post = tok["seq_pre_mask"][i], tok["seq_post_mask"][i]
                mid = full[len(pre): len(full) - len(post)] if post else full[len(pre):]
                gens.append(mid)
                lens.append(len(mid))
                per_ctx.setdefault((pre[-10:], post[:10]), []).append(mid)
                if metric is not None:
                    rewards.append(float(metric({
                        "L": batch["LC"][i], "H": full,
                        "seq_pre_mask": pre, "seq_post_mask": post})))
            seen += len(seqs)
            if seen >= args.contexts * (rep + 1):
                break
    del s
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    n = len(gens)
    allres = "".join(gens)
    comp = Counter(allres)
    top4 = comp.most_common(4)
    top_share = sum(c for _, c in top4) / max(len(allres), 1)
    uniq = len(set(gens)) / n
    L = np.array(lens, dtype=float)
    R = np.array(rewards, dtype=float) if rewards else np.zeros(0)

    dists = []
    for v in per_ctx.values():
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                dists.append(lev(v[i], v[j]))
    mean_d = float(np.mean(dists)) if dists else float("nan")
    maxlen = int(L.max()) if n else 0
    ents = [entropy([g[p] for g in gens if len(g) > p]) for p in range(maxlen)]

    have_r = R.size == n and n > 0
    distinct_r = len(set(np.round(R, 6))) if have_r else -1
    rho = (np.corrcoef(L, R)[0, 1]
           if have_r and n > 2 and L.std() > 0 and R.std() > 0 else float("nan"))

    print(f"\n=== C6' degeneracy report  (n={n} generations, "
          f"{len(per_ctx)} contexts) ===")
    print(f"checkpoint: {os.path.basename(args.checkpoint) if args.checkpoint else 'BASE (no ckpt)'}")
    print(f"reward    : {m['name']}   sampler cap (max_length) = {max_len}\n")

    flags = []

    print(f"1 length            mean={L.mean():5.1f}  std={L.std():4.1f}  "
          f"min={int(L.min())}  max={int(L.max())}  cap={max_len}")
    at_cap = float((L >= max_len).mean())
    if at_cap > 0.5:
        flags.append(f"LENGTH PINNED AT CAP ({100*at_cap:.0f}% of samples)")
    if L.mean() < 5:
        flags.append(f"LENGTH COLLAPSED (mean {L.mean():.1f})")
    if L.std() < 1.0:
        flags.append(f"LENGTH VARIANCE COLLAPSED (std {L.std():.2f})")

    print(f"2 composition       top4={' '.join(f'{a}{100*c/len(allres):.0f}%' for a,c in top4)}"
          f"   share={top_share:.3f}   posEnt={np.mean(ents):.2f}")
    if top_share > 0.80:
        flags.append(f"COMPOSITION CONCENTRATED (top-4 share {top_share:.2f})")

    print(f"3 determinism       unique={uniq:.3f}  mean within-context "
          f"editD={mean_d:.1f}")
    if uniq < 0.5:
        flags.append(f"OUTPUTS NEAR-DETERMINISTIC (unique {uniq:.2f})")

    if not have_r:
        print("4 reward quantisation  SKIPPED (--skip-reward)")
    else:
        print(f"4 reward quantisation  distinct values={distinct_r}/{n}  "
              f"mean={R.mean():+.3f}  std={R.std():.4f}")
    if have_r and distinct_r <= max(3, n // 20):
        flags.append(f"REWARD QUANTISED ({distinct_r} distinct values over {n} "
                     f"varied contexts - output is context-independent)")

    # Interpretation note: |rho| near 1 means the reward is close to a
    # deterministic function of generation length. For a ratio-style endpoint
    # over a collapsed span that is the tell, and the SIGN is not the signal -
    # measured +0.974 at dssp_beta s450, where the policy had truncated HCDR3
    # to a 3-5 residue stub. Read this together with signals 1 and 4.
    print(f"5 reward-length rho = {rho:+.3f}"
          f"   (|rho| near 1 => reward is ~a function of length alone)")
    if not math.isnan(rho) and abs(rho) > 0.9:
        flags.append(f"REWARD IS ~A FUNCTION OF LENGTH ALONE (rho {rho:+.2f})")

    print("\n--- VERDICT ---")
    if flags:
        print("  GAMING SUSPECTED:")
        for f in flags:
            print(f"    * {f}")
    else:
        print("  no degeneracy signature triggered")

    print("\nexample generations:")
    for g in list(dict.fromkeys(gens))[:8]:
        print(f"    {g}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
