# Parallel HPC runs on Imperial CX3

Companion to `../HPC_RUNBOOK.md`. That runbook covers connection, the one-time
container build, and single-job submission. This directory adds **parallel
multi-method training + evaluation + reporting**:

| File | Role |
| --- | --- |
| `protein_tune_rl.def` | Container definition (adds `pytest` for the fail-fast gate). |
| `train_array.pbs` | PBS **job array** — one method per index, all run in parallel. |
| `evaluate.py` | Held-out generation + reward scoring → `eval_metrics.csv` (test numbers). |
| `make_report.py` | Aggregates train + eval across methods → `summary.csv`, plot, `report.md`. |

Each array index is fully independent (no temporal dependency), so the scheduler
runs them concurrently across GPUs. Every job runs the unit-test gate first,
trains (checkpointing via `check_point_freq`, training numbers in `train_log.csv`),
then — for the online methods — runs held-out evaluation.

## Index → method

| Index | Config | Trainer |
| --- | --- | --- |
| 0 | `ppo_iglm_hcdr3_beta_sheet.json` | online |
| 1 | `a2c_iglm_hcdr3_beta_sheet.json` | online |
| 2 | `reinforce_iglm_hcdr3_beta_sheet.json` | online |
| 3 | `explicit_kl_reinforce_iglm_hcdr3_beta_sheet.json` | online |
| 4 | `dro_iglm_hcdr3_beta_sheet.json` | offline (train only) |
| 5 | `dpo_iglm_hcdr3_beta_sheet.json` | offline (train only) |

## Turnkey workflow

```bash
# --- LOCAL: connect VPN first, then sync code + weights (see HPC_RUNBOOK.md §3–4) ---
cd /home/ruman/repos
rsync -az --info=progress2 --exclude '.git/' --exclude '.venv/' \
  --exclude '__pycache__/' --exclude 'IgLM/' --exclude 'test_run/' \
  --exclude '.apptainer/' --exclude 'protein-tune-rl-gpu.sif' \
  -e 'ssh -F /dev/null' \
  protein_tune_rl/ ras24@login.cx3.hpc.imperial.ac.uk:protein_tune_rl/

# --- CX3: build the container from the repo file (once; adds pytest) ---
cd ~/protein_tune_rl
mkdir -p .apptainer/cache .apptainer/tmp
APPTAINER_CACHEDIR="$PWD/.apptainer/cache" APPTAINER_TMPDIR="$PWD/.apptainer/tmp" \
  apptainer build protein-tune-rl-gpu.sif hpc/protein_tune_rl.def

# --- CX3: patch example configs to container IgLM path (once) ---
apptainer exec --bind "$PWD:/workspace/protein_tune_rl" protein-tune-rl-gpu.sif \
  python /workspace/protein_tune_rl/configs/examples/patch_iglm_dir.py \
    --value /workspace/protein_tune_rl/IgLM/iglm/trained_models/IgLM-S --overwrite

# --- CX3: submit all six methods in parallel ---
qsub hpc/train_array.pbs
#   subset (online only):        qsub -J 0-3 hpc/train_array.pbs
#   longer runs (override steps): qsub -v STEPS=20000 hpc/train_array.pbs

# --- CX3: monitor ---
qstat -tu ras24                 # -t expands array subjobs
tail -f protein_rl_array.o<jobnum>.<idx>

# --- LOCAL: download results (see HPC_RUNBOOK.md §Download) ---
rsync -az -e 'ssh -F /dev/null' \
  ras24@login.cx3.hpc.imperial.ac.uk:protein_tune_rl/test_run/ \
  /home/ruman/repos/protein_tune_rl/test_run/

# --- LOCAL: build the comparison report ---
python hpc/make_report.py --root test_run --out report
#   -> report/summary.csv, report/training_curves.png, report/report.md
```

## Notes

- **Region-aware reward (`region_only`)** is available in `ss_perc_sheet` but is
  **off by default**. If you enable it, recompute the reward `mean`/`std` for the
  infilled-span distribution — the shipped constants are for the whole chain, and
  the large uncalibrated magnitude can destabilize PPO.
- **Structure rewards (`folding_confidence`, `sasa`)** need a separate container
  built with the repo's `folding` extra — the default image omits IgFold.
- **Reproducibility:** each array element writes the exact config it ran to
  `test_run/array_<job>_idx<i>/config_used.json`.
