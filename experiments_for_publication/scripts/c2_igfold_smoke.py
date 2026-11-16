"""C2 - IgFold + pydssp setup smoke test.

Proves the structural pipeline works end to end on this hardware before C3
commits to a full measurement:

  sequence -> IgFold prediction -> backbone coords -> pydssp -> beta-strand
  fraction inside HCDR3

Conditions are deliberately minimal (native, poly-V at max_length, anchored).
Real policy generations are measured in C3.

Environment note: this venv's `wandb` is broken independently of IgFold
(wandb 0.25.1 vs protobuf 7.36.0 - its generated _pb2 modules are
incompatible, so `import wandb` fails on its own). pytorch_lightning 1.8.6,
which IgFold imports, eagerly imports wandb for WandbLogger. We stub wandb
rather than upgrade a shared package. Remove the stub once wandb is repaired.
"""

import json
import sys
import types
import warnings

warnings.filterwarnings("ignore")

# Must precede any igfold import.
_stub = types.ModuleType("wandb")
_stub.__spec__ = None
sys.modules.setdefault("wandb", _stub)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pydssp  # noqa: E402
import torch  # noqa: E402
from Bio.PDB import PDBParser  # noqa: E402

sys.path.insert(0, "/home/ruman/repos/protein_tune_rl")

REPO = "/home/ruman/repos/protein_tune_rl"
OUT = f"{REPO}/experiments_for_publication/results"
WORK = "/tmp/claude-1000/c2_igfold_work"
MAX_LEN = 22
BACKBONE = ("N", "CA", "C", "O")


def strand_fraction(pdb_path, chain_id, start, end):
    """Beta-strand fraction over residues [start, end) of `chain_id`.

    Indices are 0-based positions in the chain's residue order, matching how
    the sequence was constructed.
    """
    model = next(PDBParser(QUIET=True).get_structure("s", pdb_path).get_models())
    residues = [r for r in model[chain_id] if all(a in r for a in BACKBONE)]
    coords = np.array(
        [[r[a].get_coord() for a in BACKBONE] for r in residues], dtype=np.float32
    )
    ss = pydssp.assign(coords, out_type="c3")
    region = ss[start:end]
    whole = float(np.mean(ss == "E"))
    if len(region) == 0:
        return whole, float("nan"), "".join(ss)
    return whole, float(np.mean(region == "E")), "".join(ss)


def main():
    import os

    os.makedirs(WORK, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    df = pd.read_csv(f"{REPO}/data/examples/eval_data_50_v3.csv")
    r = df.iloc[0]
    hc, lc, cdr = str(r.HC), str(r.LC), str(r.HCDR3)
    i = hc.find(cdr)
    pre, post = hc[:i], hc[i + len(cdr):]

    conditions = {
        "native": cdr,
        "polyV_maxlen": "V" * MAX_LEN,
        "anchored": cdr[:2] + "V" * (MAX_LEN - 4) + cdr[-2:],
    }

    from igfold import IgFoldRunner

    print("constructing IgFoldRunner (downloads weights on first run)...")
    runner = IgFoldRunner(num_models=1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner.antiberty.model.to(dev)
    for m in runner.models:
        m.to(dev)
    print(f"  ready on {dev}")

    rows = []
    for name, mid in conditions.items():
        heavy = pre + mid + post
        pdb = f"{WORK}/{name}.pdb"
        runner.fold(pdb, sequences={"H": heavy, "L": lc}, do_refine=False,
                    do_renum=False)
        whole, region, ss = strand_fraction(pdb, "H", len(pre), len(pre) + len(mid))
        rows.append(dict(condition=name, cdr=mid, cdr_len=len(mid),
                         heavy_len=len(heavy),
                         strand_frac_region=region, strand_frac_chain=whole))
        print(f"\n{name}: HCDR3={mid}")
        print(f"  beta-strand fraction  region={region:.3f}  whole-H={whole:.3f}")
        print(f"  region DSSP: {ss[len(pre):len(pre)+len(mid)]}")

    peak = (torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0
    print(f"\npeak GPU alloc during folding: {peak:.0f} MB "
          f"(card has 6141 MB)")

    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/c2_smoke.csv", index=False)

    import igfold
    import pytorch_lightning
    import transformers
    import Bio
    versions = {
        "igfold": getattr(igfold, "__version__", "0.4.0"),
        "antiberty": "1.0.0",
        "pytorch_lightning": pytorch_lightning.__version__,
        "pydssp": getattr(pydssp, "__version__", "0.9.2"),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "biopython": Bio.__version__,
        "peak_gpu_mb_folding": round(peak),
        "dssp": "pydssp (pure python; no system mkdssp, which needs sudo)",
        "wandb_workaround": "stubbed; venv wandb 0.25.1 broken vs protobuf 7.36.0",
    }
    with open(f"{OUT}/c2_versions.json", "w") as f:
        json.dump(versions, f, indent=2)
    print(f"\nwrote {OUT}/c2_smoke.csv and c2_versions.json")


if __name__ == "__main__":
    main()
