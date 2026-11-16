# Gate scripts (16 November 2026)

The viability gates whose results are recorded in `../../Path_Forward.md` §1.
Preserved verbatim as run. They are **exploratory**, not the publication
pipeline: single seed, hard-coded paths, printed output rather than saved
artifacts.

| Script | Gate | Result |
| --- | --- | --- |
| `u1_composition_attack.py` | P1: is the reward a composition exploit? | Fired. `"V"*22` scores 3.338 vs 3.366 for the 82k-step policy. |
| `u4_ladder.py` | P1/P4: what does the policy emit, does diversity collapse? | Policy emits 99.9% sheet residues at length 22 from step 10k. Diversity does **not** collapse. |
| `u2_u5.py` | P2 correctness + P5 endpoint agreement | Importance ratio off by 0.286. KL controller never executed. Endpoints uncorrelated. |
| `time_sampling.py` | Throughput feasibility | 81 seq/s at batch 8, 295 MB peak VRAM. |

Run with the general venv from the repo root, e.g.

```bash
/home/ruman/venvs/general/bin/python \
  experiments_for_publication/scripts/gates/u1_composition_attack.py
```

C1 re-implements U1 properly (all contexts, ESM-2 650M, saved artifacts).
