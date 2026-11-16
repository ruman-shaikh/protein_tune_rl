"""C4a - attackability screen for STRUCTURAL endpoints (no training).

Before spending 6-7 GPU-hours per training run on an endpoint, establish
whether that endpoint can be moved by trivially degenerate edits at all.
For each candidate reward we ask: how many native standard deviations can a
hand-written HCDR3 gain?

One IgFold call per (context, attack) yields three endpoints at once:
  sasa_chain          Shrake-Rupley SASA of the predicted complex
  igfold_confidence   1 / (1 + mean predicted RMSD)   [the repo's definition]
  dssp_beta_region    pydssp beta-strand fraction inside HCDR3

Composition endpoints need no folding and are computed here too, so the
structural and compositional columns share identical sequences.

ESM endpoints are screened by the sibling script c4b_esm_attackability.py,
which has no dependency on this one and can run concurrently.

Resumable: appends one row per completed fold, skips rows already present.
"""

import os
import sys
import types
import warnings

warnings.filterwarnings("ignore")

_stub = types.ModuleType("wandb")
_stub.__spec__ = None
sys.modules.setdefault("wandb", _stub)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pydssp  # noqa: E402
import torch  # noqa: E402
from Bio.PDB import PDBParser  # noqa: E402
from Bio.PDB.SASA import ShrakeRupley  # noqa: E402
from Bio.SeqUtils.ProtParam import ProteinAnalysis  # noqa: E402

REPO = "/home/ruman/repos/protein_tune_rl"
OUT = f"{REPO}/experiments_for_publication/results"
WORK = "/tmp/claude-1000/c4a_folds"
CSV = f"{OUT}/c4a_structural_attackability.csv"
MAX_LEN = 22
N_CTX = 20                     # screen, not a final measurement
BACKBONE = ("N", "CA", "C", "O")
SHEET_MEAN, SHEET_STD = 0.3784639359426365, 0.017445864484741506


def attacks(cdr):
    """Hand-written HCDR3 variants. No model, no search."""
    return {
        "native": cdr,
        "polyV_max": "V" * MAX_LEN,
        "polyG_max": "G" * MAX_LEN,
        "polyW_max": "W" * MAX_LEN,
        "polyP_max": "P" * MAX_LEN,
        "VY_repeat": "VY" * (MAX_LEN // 2),
        "minimal": cdr[:2] if len(cdr) >= 2 else "AA",
        "anchored_V": (cdr[:2] + "V" * (MAX_LEN - 4) + cdr[-2:]) if len(cdr) >= 4 else "V" * MAX_LEN,
    }


def composition_reward(heavy, light):
    frac = ProteinAnalysis(heavy + light).secondary_structure_fraction()[2]
    return (frac - SHEET_MEAN) / SHEET_STD


def composition_region(mid):
    if not mid:
        return float("nan")
    return ProteinAnalysis(mid).secondary_structure_fraction()[2]


def structural(pdb, out, start, length):
    struct = PDBParser(QUIET=True).get_structure("s", pdb)
    ShrakeRupley().compute(struct, level="S")
    sasa = float(struct.sasa)

    model = next(struct.get_models())
    res = [r for r in model["H"] if all(a in r for a in BACKBONE)]
    coords = np.array([[r[a].get_coord() for a in BACKBONE] for r in res],
                      dtype=np.float32)
    ss = pydssp.assign(coords, out_type="c3")
    region = ss[start:start + length]
    beta = float(np.mean(region == "E")) if len(region) else float("nan")

    # Repo definition: 1 / (1 + mean per-residue predicted RMSD)
    rmsd = out.prmsd.square().mean(dim=-1).sqrt().squeeze(0)
    conf = float(1.0 / (1.0 + rmsd.mean().cpu().numpy()))
    return sasa, conf, beta


def main():
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    df = pd.read_csv(f"{REPO}/data/examples/eval_data_50_v3.csv").head(N_CTX)

    done = set()
    if os.path.exists(CSV):
        prev = pd.read_csv(CSV)
        done = set(zip(prev.context, prev.attack))
        print(f"resuming: {len(done)} rows already present")

    jobs = []
    for idx, r in df.iterrows():
        hc, lc, cdr = str(r.HC), str(r.LC), str(r.HCDR3)
        i = hc.find(cdr)
        if i < 0:
            continue
        pre, post = hc[:i], hc[i + len(cdr):]
        for name, mid in attacks(cdr).items():
            if (idx, name) not in done:
                jobs.append((idx, name, mid, pre, post, lc))

    if not jobs:
        print("nothing to do")
    else:
        print(f"folding {len(jobs)} sequences (~{1.5*len(jobs)/60:.1f} min)")
        from igfold import IgFoldRunner
        runner = IgFoldRunner(num_models=1)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        runner.antiberty.model.to(dev)
        for m in runner.models:
            m.to(dev)

        header = not os.path.exists(CSV)
        for n, (idx, name, mid, pre, post, lc) in enumerate(jobs, 1):
            heavy = pre + mid + post
            pdb = f"{WORK}/c{idx}_{name}.pdb"
            try:
                out = runner.fold(pdb, sequences={"H": heavy, "L": lc},
                                  do_refine=False, do_renum=False)
                sasa, conf, beta = structural(pdb, out, len(pre), len(mid))
                ok = True
            except Exception as e:
                print(f"  FAILED c{idx} {name}: {type(e).__name__}: {e}")
                sasa = conf = beta = float("nan")
                ok = False
            pd.DataFrame([dict(
                context=idx, attack=name, cdr=mid, cdr_len=len(mid),
                composition_reward=composition_reward(heavy, lc),
                composition_region=composition_region(mid),
                sasa_chain=sasa, igfold_confidence=conf,
                dssp_beta_region=beta, folded=ok,
            )]).to_csv(CSV, mode="a", header=header, index=False)
            header = False
            if os.path.exists(pdb):
                os.remove(pdb)
            if n % 25 == 0:
                print(f"  {n}/{len(jobs)}")

    # ---- report: attackability in native standard deviations ---------------
    d = pd.read_csv(CSV)
    d = d[d.folded.astype(str).str.lower() == "true"]
    cols = ["composition_reward", "composition_region", "sasa_chain",
            "igfold_confidence", "dssp_beta_region"]
    nat = d[d.attack == "native"]

    print(f"\n=== C4a structural attackability (n={d.context.nunique()} contexts) ===")
    print("gain in native standard deviations; + means the attack raises the endpoint\n")
    print(f"{'attack':<13}" + "".join(f"{c.replace('_','.')[:13]:>15}" for c in cols))
    for a in d.attack.unique():
        if a == "native":
            continue
        row = f"{a:<13}"
        for c in cols:
            s = nat[c].std()
            g = (d[d.attack == a][c].mean() - nat[c].mean()) / (s if s else float("nan"))
            row += f"{g:>15.2f}"
        print(row)

    print(f"\nnative means: " + "  ".join(
        f"{c}={nat[c].mean():.3f}" for c in cols))
    print(f"native stds:  " + "  ".join(
        f"{c}={nat[c].std():.3f}" for c in cols))
    print(f"\nartifact: {CSV}")


if __name__ == "__main__":
    main()
