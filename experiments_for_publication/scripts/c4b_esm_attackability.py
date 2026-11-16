"""C4b - attackability screen for ESM endpoints (no training, no folding).

Companion to c4a_structural_attackability.py, with the same attack set and
the same contexts, so the two result tables join on (context, attack). No
dependency between them: run concurrently.

Screens ESM-2 masked pseudo-log-likelihood as a *reward* rather than as a
check, at two scopes (generated region, whole heavy chain) and two model
sizes, to establish whether it is exploitable by degenerate edits and whether
exploitability depends on model capacity.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, "/home/ruman/repos/protein_tune_rl")
from protein_tune_rl.metrics.esm_scoring import ESMSequenceScoring  # noqa: E402

REPO = "/home/ruman/repos/protein_tune_rl"
OUT = f"{REPO}/experiments_for_publication/results"
CSV = f"{OUT}/c4b_esm_attackability.csv"
MAX_LEN = 22
N_CTX = 20
MODELS = [("facebook/esm2_t6_8M_UR50D", "8M"),
          ("facebook/esm2_t33_650M_UR50D", "650M")]


def attacks(cdr):
    """Identical to c4a so the tables join."""
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


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(f"{REPO}/data/examples/eval_data_50_v3.csv").head(N_CTX)

    done = set()
    if os.path.exists(CSV):
        prev = pd.read_csv(CSV)
        done = set(zip(prev.context, prev.attack, prev.model))
        print(f"resuming: {len(done)} rows already present")

    header = not os.path.exists(CSV)
    for model_name, tag in MODELS:
        pending = [
            (idx, r) for idx, r in df.iterrows()
            if any((idx, a, tag) not in done for a in attacks(str(r.HCDR3)))
        ]
        if not pending:
            print(f"{tag}: nothing to do")
            continue
        print(f"{tag}: scoring {len(pending)} contexts")
        reg = ESMSequenceScoring(model_name=model_name, region_only=True,
                                 reduction="mean", local_files_only=True)
        chain = ESMSequenceScoring(model_name=model_name, region_only=False,
                                   reduction="mean", local_files_only=True,
                                   model=reg.model, tokenizer=reg.tokenizer)
        for n, (idx, r) in enumerate(pending, 1):
            hc, cdr = str(r.HC), str(r.HCDR3)
            i = hc.find(cdr)
            if i < 0:
                continue
            pre, post = hc[:i], hc[i + len(cdr):]
            for name, mid in attacks(cdr).items():
                if (idx, name, tag) in done:
                    continue
                ch = {"H": pre + mid + post, "L": "",
                      "seq_pre_mask": pre, "seq_post_mask": post}
                try:
                    pll_r, pll_c = reg(ch), chain(ch)
                    ok = True
                except Exception as e:
                    print(f"  FAILED c{idx} {name} {tag}: {type(e).__name__}: {e}")
                    pll_r = pll_c = float("nan")
                    ok = False
                pd.DataFrame([dict(
                    context=idx, attack=name, model=tag, cdr=mid,
                    cdr_len=len(mid), esm_pll_region=pll_r,
                    esm_pll_chain=pll_c, scored=ok,
                )]).to_csv(CSV, mode="a", header=header, index=False)
                header = False
            if n % 5 == 0:
                print(f"  {n}/{len(pending)}")
        del reg, chain
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    d = pd.read_csv(CSV)
    d = d[d.scored.astype(str).str.lower() == "true"]
    print(f"\n=== C4b ESM attackability (n={d.context.nunique()} contexts) ===")
    print("gain in native standard deviations; + means the attack raises the endpoint\n")
    for tag in d.model.unique():
        m = d[d.model == tag]
        nat = m[m.attack == "native"]
        print(f"--- ESM-2 {tag} ---")
        print(f"{'attack':<13}{'pll.region':>14}{'pll.chain':>13}")
        for a in m.attack.unique():
            if a == "native":
                continue
            row = f"{a:<13}"
            for c in ("esm_pll_region", "esm_pll_chain"):
                s = nat[c].std()
                g = (m[m.attack == a][c].mean() - nat[c].mean()) / (s if s else float("nan"))
                row += f"{g:>14.2f}"
            print(row)
        print(f"  native: region={nat.esm_pll_region.mean():.3f} "
              f"chain={nat.esm_pll_chain.mean():.3f}\n")
    print(f"artifact: {CSV}")


if __name__ == "__main__":
    main()
