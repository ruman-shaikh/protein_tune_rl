"""U4 - what does the trained policy actually emit, and does diversity collapse?

Samples the retained A2C checkpoint ladder on fixed held-out contexts with
fixed seeds, then reports per checkpoint:
  * what the policy emits: generated length, sheet-residue fraction, top residues
  * the training reward it achieves
  * diversity: unique fraction, within-context mean edit distance, positional entropy

Completes U1 (did the policy take the composition exploit?) and gates P4.
"""

import json
import os
import sys
import glob
import math
import random
from collections import Counter
from datetime import timedelta

import torch
import torch.distributed as dist

sys.path.insert(0, "/home/ruman/repos/protein_tune_rl")
from protein_tune_rl.protein_trainer.online_rl_trainer import OnlineRLSampler  # noqa
from protein_tune_rl.protein_trainer.trainer import Trainer  # noqa

REPO = "/home/ruman/repos/protein_tune_rl"
SHEET = set("VIYFWLT")
REPEATS = 8          # samples per context, different seeds
EVAL = f"{REPO}/data/examples/eval_data_50_v3.csv"


class S(Trainer, OnlineRLSampler):
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


def entropy(col):
    n = len(col)
    if n == 0:
        return 0.0
    c = Counter(col)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def run_ckpt(cfg_base, ckpt, metric):
    config = json.loads(json.dumps(cfg_base))
    config["dataset"]["data_directory"] = EVAL
    config["checkpoint"] = ckpt
    config.setdefault("KL_penalty", {"weight": 0.0, "target": None})
    s = S(config)

    per_ctx = {}
    rewards = []
    for rep in range(REPEATS):
        torch.manual_seed(1000 + rep)
        random.seed(1000 + rep)
        for batch in iter(s.dataloader):
            tok = s.collator(batch)
            with torch.no_grad():
                seqs, *_ = s._sample_batch(s.policy, tok)
            for i, full in enumerate(seqs):
                pre = tok["seq_pre_mask"][i]
                post = tok["seq_post_mask"][i]
                infill = full[len(pre): len(full) - len(post)] if post else full[len(pre):]
                key = (pre[-12:], post[:12])
                per_ctx.setdefault(key, []).append(infill)
                rewards.append(
                    float(metric({"L": batch["LC"][i], "H": full,
                                  "seq_pre_mask": pre, "seq_post_mask": post}))
                )
    del s
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gens = [g for v in per_ctx.values() for g in v]
    if not gens:
        return None
    lens = [len(g) for g in gens]
    allres = "".join(gens)
    sheet_frac = sum(1 for c in allres if c in SHEET) / max(len(allres), 1)
    top = Counter(allres).most_common(4)

    uniq = len(set(gens)) / len(gens)
    dists = []
    for v in per_ctx.values():
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                dists.append(lev(v[i], v[j]))
    mean_d = sum(dists) / len(dists) if dists else 0.0
    maxlen = max(lens)
    ents = [entropy([g[p] for g in gens if len(g) > p]) for p in range(maxlen)]
    mean_ent = sum(ents) / len(ents) if ents else 0.0

    return dict(
        n=len(gens), reward=sum(rewards) / len(rewards),
        mlen=sum(lens) / len(lens), maxlen=maxlen,
        sheet=sheet_frac, top=top, uniq=uniq, meand=mean_d, ent=mean_ent,
    )


def main():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "24811")
    dist.init_process_group(
        "nccl" if torch.cuda.is_available() else "gloo",
        rank=0, world_size=1, timeout=timedelta(seconds=120))
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    run_dir = sorted(glob.glob(f"{REPO}/test_run/online_rl_trainer/*/*steps_200000*"))[0]
    print(f"run dir: {os.path.basename(run_dir)}")
    cfg_base = json.load(open(os.path.join(run_dir, "config.json")))
    print(f"config optimizer key = {cfg_base.get('optimizer')}   "
          f"<- 'name' missing is the tune.py save-order bug")
    print(f"KL weight={cfg_base['KL_penalty']['weight']} "
          f"target={cfg_base['KL_penalty']['target']}")

    from protein_tune_rl.metrics import create_metric
    m = cfg_base["metric"][0]
    metric = create_metric(name=m["name"])(**m["params"])

    ck = {}
    for p in glob.glob(os.path.join(run_dir, "a2c_s*.ckpt")):
        ck[int(os.path.basename(p).split("_s")[1].split(".")[0])] = p
    steps = sorted(ck)
    pick = [s for s in steps if s % 10000 == 0 or s in (steps[0], steps[-1])]
    print(f"{len(steps)} ckpts available; sampling {len(pick)}: {pick}\n")

    print(f"{'step':>7}{'reward':>9}{'gen_len':>9}{'sheet%':>8}"
          f"{'uniq%':>8}{'editD':>7}{'posEnt':>8}   top residues")
    print("-" * 86)
    base = None
    for st in pick:
        r = run_ckpt(cfg_base, ck[st], metric)
        if r is None:
            continue
        if base is None:
            base = r
        top = " ".join(f"{a}:{100*c/len(''.join([]))if False else c}" for a, c in r["top"])
        tot = sum(c for _, c in r["top"])
        top = " ".join(f"{a}{100*c/ (r['n']*r['mlen']):.0f}%" for a, c in r["top"])
        print(f"{st:>7}{r['reward']:>9.3f}{r['mlen']:>9.1f}{100*r['sheet']:>7.1f}%"
              f"{100*r['uniq']:>7.1f}%{r['meand']:>7.1f}{r['ent']:>8.2f}   {top}")
        del tot

    print(f"\n(reward is the training metric; sheet set = VIYFWLT; "
          f"{REPEATS} samples x 48 contexts per checkpoint)")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
