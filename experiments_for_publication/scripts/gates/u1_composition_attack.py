"""U1 - viability gate for P1: is `ss_perc_sheet` a composition exploit?

No training, no RL. Asks whether the training reward can be driven to the
level RL actually reached (~4.2 normalised) by editing HCDR3 composition
alone, and what that costs in an independent oracle (ESM-2 masked PLL).

Attacks, all applied to held-out contexts:
  A  substitute: replace HCDR3 with native-length sheet residues (upper
     bound on substitution-only gain at fixed length)
  B  extend: replace HCDR3 with k sheet residues, k up to max_length=22
     (the policy's real action space)
  C  anchored: keep native HCDR3 first 2 / last 2 residues, fill the middle
     with sheet residues up to max_length (a more plausible-looking exploit)

Reference points taken from the repo, not invented:
  reward = (sheet_fraction(H+L) - 0.3784639359426365) / 0.017445864484741506
  RL ceiling observed in test_run/ train_log.csv: reward_mean ~= 4.2
"""

import json
import sys

import pandas as pd
import torch
from Bio.SeqUtils.ProtParam import ProteinAnalysis

sys.path.insert(0, "/home/ruman/repos/protein_tune_rl")
from protein_tune_rl.metrics.esm_scoring import ESMSequenceScoring  # noqa: E402

REPO = "/home/ruman/repos/protein_tune_rl"
MEAN, STD = 0.3784639359426365, 0.017445864484741506
MAX_LEN = 22           # trainer max_length in the shipped configs
RL_CEILING = 4.2       # best reward_mean seen in retained train_log.csv
SHEET = "VIYFWLT"      # biopython >=1.82 sheet set, verified empirically
FILL = "V"             # single sheet residue used for the fill


def sheet_fraction(seq: str) -> float:
    return ProteinAnalysis(seq).secondary_structure_fraction()[2]


def reward(heavy: str, light: str) -> float:
    return (sheet_fraction(heavy + light) - MEAN) / STD


def build(pre: str, mid: str, post: str) -> str:
    return pre + mid + post


def main():
    df = pd.read_csv(f"{REPO}/data/examples/eval_data_50_v3.csv")
    print(f"held-out contexts: {len(df)}")

    rows = []
    for _, r in df.iterrows():
        hc, lc, cdr = str(r.HC), str(r.LC), str(r.HCDR3)
        i = hc.find(cdr)
        if i < 0:
            continue
        pre, post = hc[:i], hc[i + len(cdr):]
        L = len(cdr)

        native_r = reward(hc, lc)

        # A: substitute at native length
        a_seq = build(pre, FILL * L, post)
        a_r = reward(a_seq, lc)

        # B: extend to the best k <= MAX_LEN
        best_b = max(
            ((reward(build(pre, FILL * k, post), lc), k) for k in range(1, MAX_LEN + 1)),
            key=lambda t: t[0],
        )
        b_r, b_k = best_b
        b_seq = build(pre, FILL * b_k, post)

        # C: anchored - keep 2 native residues at each end, fill the middle
        c_r, c_k, c_seq = -1e9, None, None
        for k in range(4, MAX_LEN + 1):
            mid = cdr[:2] + FILL * (k - 4) + cdr[-2:]
            s = build(pre, mid, post)
            v = reward(s, lc)
            if v > c_r:
                c_r, c_k, c_seq = v, k, s

        rows.append(
            dict(
                cdr=cdr, cdr_len=L,
                native=native_r,
                subst=a_r, subst_seq=a_seq,
                extend=b_r, extend_k=b_k, extend_seq=b_seq,
                anchored=c_r, anchored_k=c_k, anchored_seq=c_seq,
                pre=pre, post=post,
            )
        )

    res = pd.DataFrame(rows)
    print(f"usable contexts: {len(res)}  (HCDR3 located in HC)\n")

    print("=== REWARD (normalised, same definition as training) ===")
    print(f"{'condition':<28}{'mean':>9}{'std':>9}{'max':>9}  {'% of RL ceiling':>16}")
    for name, col in (
        ("native (no edit)", "native"),
        ("A substitute @native len", "subst"),
        ("B extend to max_length", "extend"),
        ("C anchored + extend", "anchored"),
    ):
        m, s, mx = res[col].mean(), res[col].std(), res[col].max()
        print(f"{name:<28}{m:9.3f}{s:9.3f}{mx:9.3f}  {100*m/RL_CEILING:15.0f}%")
    print(f"\nRL ceiling (retained logs, reward_mean max) = {RL_CEILING}")
    print(f"native CDR length: mean={res.cdr_len.mean():.1f} "
          f"min={res.cdr_len.min()} max={res.cdr_len.max()}")
    print(f"chosen fill length: extend k mean={res.extend_k.mean():.1f} "
          f"(max_length={MAX_LEN}), anchored k mean={res.anchored_k.mean():.1f}")

    # ---- independent oracle: ESM-2 masked PLL -------------------------------
    print("\n=== ESM-2 masked pseudo-log-likelihood (independent oracle) ===")
    print("model: facebook/esm2_t6_8M_UR50D (only cached variant)")
    scorer_region = ESMSequenceScoring(
        model_name="facebook/esm2_t6_8M_UR50D",
        region_only=True, reduction="mean", local_files_only=True,
    )
    scorer_chain = ESMSequenceScoring(
        model_name="facebook/esm2_t6_8M_UR50D",
        region_only=False, reduction="mean", local_files_only=True,
    )

    sub = res.head(25)  # 25 contexts is ample for a viability gate
    out = {k: {"region": [], "chain": []} for k in
           ("native", "subst", "extend", "anchored")}
    for _, r in sub.iterrows():
        native_h = build(r.pre, r.cdr, r.post)
        for key, seq in (
            ("native", native_h), ("subst", r.subst_seq),
            ("extend", r.extend_seq), ("anchored", r.anchored_seq),
        ):
            mid_len = len(seq) - len(r.pre) - len(r.post)
            chains = {
                "H": seq, "L": "",
                "seq_pre_mask": r.pre, "seq_post_mask": r.post,
            }
            out[key]["region"].append(scorer_region(chains))
            out[key]["chain"].append(scorer_chain(chains))
            del mid_len

    print(f"\nscored {len(sub)} contexts x 4 conditions")
    print(f"{'condition':<28}{'PLL region':>12}{'PLL chain':>12}"
          f"{'d region':>11}{'d chain':>10}")
    base_r = sum(out['native']['region']) / len(sub)
    base_c = sum(out['native']['chain']) / len(sub)
    for name, key in (
        ("native (no edit)", "native"),
        ("A substitute @native len", "subst"),
        ("B extend to max_length", "extend"),
        ("C anchored + extend", "anchored"),
    ):
        mr = sum(out[key]["region"]) / len(sub)
        mc = sum(out[key]["chain"]) / len(sub)
        print(f"{name:<28}{mr:12.3f}{mc:12.3f}{mr-base_r:11.3f}{mc-base_c:10.3f}")

    print("\n=== example sequences (context 0) ===")
    r0 = res.iloc[0]
    print(f"  native   HCDR3: {r0.cdr}   (reward {r0.native:+.2f})")
    print(f"  A subst  HCDR3: {FILL*r0.cdr_len}   (reward {r0.subst:+.2f})")
    print(f"  B extend HCDR3: {FILL*int(r0.extend_k)}   (reward {r0.extend:+.2f})")
    anchored_mid = r0.cdr[:2] + FILL*(int(r0.anchored_k)-4) + r0.cdr[-2:]
    print(f"  C anchor HCDR3: {anchored_mid}   (reward {r0.anchored:+.2f})")

    res.drop(columns=["subst_seq", "extend_seq", "anchored_seq", "pre", "post"]).to_csv(
        "/tmp/claude-1000/-home-ruman-repos-protein-tune-rl/"
        "96ed25a6-8640-45c3-84b5-94a3428973fb/scratchpad/u1_results.csv", index=False
    )
    print("\nper-context results saved to scratchpad/u1_results.csv")


if __name__ == "__main__":
    main()
