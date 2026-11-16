"""C3 - does the reward-maximising edit actually produce beta-sheet?

The closing leg of the argument. Four conditions on matched held-out contexts:

  native        the real HCDR3
  polyV_maxlen  hand-written attack: "V" * max_length
  policy_82k    sampled from the retained A2C checkpoint
  iglm_base     sampled from pretrained IgLM, no RL

Each sequence is folded with IgFold and assigned secondary structure with
pydssp, giving the beta-strand fraction actually predicted inside HCDR3,
alongside the training reward the same sequence receives.

Two phases, each resumable: phase 1 writes sequences, phase 2 appends one row
per completed fold and skips anything already present. Interrupting costs at
most one fold.

    python experiments_for_publication/scripts/c3_structural_measurement.py

Environment: stubs `wandb` before importing igfold - see Path_Forward.md
"C1/C2 completion notes" for why.
"""

import json
import os
import sys
import types
import warnings
from datetime import timedelta

warnings.filterwarnings("ignore")

_stub = types.ModuleType("wandb")
_stub.__spec__ = None
sys.modules.setdefault("wandb", _stub)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pydssp  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from Bio.PDB import PDBParser  # noqa: E402

sys.path.insert(0, "/home/ruman/repos/protein_tune_rl")
from protein_tune_rl.metrics import create_metric  # noqa: E402
from protein_tune_rl.protein_trainer.online_rl_trainer import OnlineRLSampler  # noqa
from protein_tune_rl.protein_trainer.trainer import Trainer  # noqa

REPO = "/home/ruman/repos/protein_tune_rl"
OUT = f"{REPO}/experiments_for_publication/results"
WORK = "/tmp/claude-1000/c3_folds"
SEQ_CSV = f"{OUT}/c3_sequences.csv"
STRUCT_CSV = f"{OUT}/c3_structure.csv"
MAX_LEN = 22
BACKBONE = ("N", "CA", "C", "O")
CKPT_GLOB = "test_run/online_rl_trainer/*/*steps_200000*/a2c_s82000.ckpt"


class Sampler(Trainer, OnlineRLSampler):
    def __init__(self, config):
        Trainer.__init__(self, config)
        OnlineRLSampler.__init__(self, config)


def _config(ckpt=None):
    import glob

    run_dir = os.path.dirname(glob.glob(f"{REPO}/{CKPT_GLOB}")[0])
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    cfg["dataset"]["data_directory"] = f"{REPO}/data/examples/eval_data_50_v3.csv"
    cfg.setdefault("KL_penalty", {"weight": 0.0, "target": None})
    if ckpt:
        cfg["checkpoint"] = ckpt
    else:
        cfg.pop("checkpoint", None)
    return cfg, run_dir


def phase1_sequences():
    """Generate the matched sequence set. Idempotent: skips if CSV exists."""
    if os.path.exists(SEQ_CSV):
        df = pd.read_csv(SEQ_CSV)
        print(f"phase 1: reusing {SEQ_CSV} ({len(df)} rows)")
        return df

    import glob

    ckpt = glob.glob(f"{REPO}/{CKPT_GLOB}")[0]
    cfg, _ = _config(ckpt)
    m = cfg["metric"][0]
    metric = create_metric(name=m["name"])(**m["params"])

    def sample(config, label):
        s = Sampler(config)
        got = {}
        torch.manual_seed(20261116)
        for batch in iter(s.dataloader):
            tok = s.collator(batch)
            with torch.no_grad():
                seqs, *_ = s._sample_batch(s.policy, tok)
            for i, full in enumerate(seqs):
                pre, post = tok["seq_pre_mask"][i], tok["seq_post_mask"][i]
                cdr = full[len(pre): len(full) - len(post)] if post else full[len(pre):]
                got[(pre, post)] = (cdr, batch["LC"][i])
        del s
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"  sampled {len(got)} contexts from {label}")
        return got

    print("phase 1: sampling")
    pol = sample(_config(ckpt)[0], "policy a2c_s82000")
    base = sample(_config(None)[0], "pretrained IgLM (no RL)")

    native = pd.read_csv(f"{REPO}/data/examples/eval_data_50_v3.csv")
    nat = {}
    for _, r in native.iterrows():
        hc, cdr = str(r.HC), str(r.HCDR3)
        i = hc.find(cdr)
        if i >= 0:
            nat[(hc[:i], hc[i + len(cdr):])] = (cdr, str(r.LC))

    rows = []
    for idx, key in enumerate(sorted(pol.keys() & nat.keys())):
        pre, post = key
        light = nat[key][1]
        for cond, cdr in (
            ("native", nat[key][0]),
            ("polyV_maxlen", "V" * MAX_LEN),
            ("policy_82k", pol[key][0]),
            ("iglm_base", base[key][0] if key in base else None),
        ):
            if cdr is None or len(cdr) == 0:
                continue
            heavy = pre + cdr + post
            rows.append(dict(
                context=idx, condition=cond, cdr=cdr, cdr_len=len(cdr),
                region_start=len(pre), heavy=heavy, light=light,
                reward=float(metric({"L": light, "H": heavy,
                                     "seq_pre_mask": pre, "seq_post_mask": post})),
            ))
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(SEQ_CSV, index=False)
    print(f"phase 1: wrote {len(df)} sequences over "
          f"{df.context.nunique()} contexts x {df.condition.nunique()} conditions")
    return df


def strand_fraction(pdb, chain, start, length):
    model = next(PDBParser(QUIET=True).get_structure("s", pdb).get_models())
    res = [r for r in model[chain] if all(a in r for a in BACKBONE)]
    coords = np.array([[r[a].get_coord() for a in BACKBONE] for r in res],
                      dtype=np.float32)
    ss = pydssp.assign(coords, out_type="c3")
    region = ss[start:start + length]
    return (float(np.mean(ss == "E")),
            float(np.mean(region == "E")) if len(region) else float("nan"),
            "".join(region))


def phase2_fold(seqs):
    os.makedirs(WORK, exist_ok=True)
    done = set()
    if os.path.exists(STRUCT_CSV):
        prev = pd.read_csv(STRUCT_CSV)
        done = set(zip(prev.context, prev.condition))
        print(f"phase 2: resuming, {len(done)} folds already done")

    todo = [r for _, r in seqs.iterrows() if (r.context, r.condition) not in done]
    if not todo:
        print("phase 2: nothing to do")
        return
    print(f"phase 2: folding {len(todo)} sequences (~{1.5*len(todo)/60:.1f} min)")

    from igfold import IgFoldRunner

    runner = IgFoldRunner(num_models=1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner.antiberty.model.to(dev)
    for mm in runner.models:
        mm.to(dev)

    header = not os.path.exists(STRUCT_CSV)
    for n, r in enumerate(todo, 1):
        pdb = f"{WORK}/c{r.context}_{r.condition}.pdb"
        try:
            runner.fold(pdb, sequences={"H": r.heavy, "L": r.light},
                        do_refine=False, do_renum=False)
            whole, region, dssp = strand_fraction(pdb, "H", r.region_start, r.cdr_len)
            ok = True
        except Exception as e:  # folding or parsing failed; record and continue
            print(f"  FAILED c{r.context} {r.condition}: {type(e).__name__}: {e}")
            whole, region, dssp, ok = float("nan"), float("nan"), "", False
        pd.DataFrame([dict(
            context=r.context, condition=r.condition, cdr=r.cdr,
            cdr_len=r.cdr_len, reward=r.reward,
            strand_frac_region=region, strand_frac_chain=whole,
            dssp_region=dssp, folded=ok,
        )]).to_csv(STRUCT_CSV, mode="a", header=header, index=False)
        header = False
        if os.path.exists(pdb):
            os.remove(pdb)
        if n % 25 == 0:
            print(f"  {n}/{len(todo)}")


def report():
    df = pd.read_csv(STRUCT_CSV)
    ok = df[df.folded.astype(str).str.lower() == "true"]
    print(f"\n=== C3 RESULT  (n={len(ok)} folded, {len(df)-len(ok)} failed) ===")
    print(f"{'condition':<16}{'n':>4}{'reward':>9}{'beta_region':>13}"
          f"{'beta_chain':>12}{'cdr_len':>9}")
    order = ["native", "iglm_base", "policy_82k", "polyV_maxlen"]
    for c in order:
        g = ok[ok.condition == c]
        if not len(g):
            continue
        print(f"{c:<16}{len(g):>4}{g.reward.mean():>9.3f}"
              f"{g.strand_frac_region.mean():>13.3f}"
              f"{g.strand_frac_chain.mean():>12.3f}{g.cdr_len.mean():>9.1f}")
    from scipy.stats import spearmanr
    v = ok.dropna(subset=["reward", "strand_frac_region"])
    rho = spearmanr(v.reward, v.strand_frac_region)
    print(f"\nreward vs predicted HCDR3 beta-strand fraction:"
          f"  Spearman rho = {rho.correlation:+.3f}  (p = {rho.pvalue:.2g}, n={len(v)})")
    print(f"\nartifact: {STRUCT_CSV}")


def main():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "24901")
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo",
                            rank=0, world_size=1, timeout=timedelta(seconds=180))
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    seqs = phase1_sequences()
    dist.destroy_process_group()
    phase2_fold(seqs)
    report()


if __name__ == "__main__":
    main()
