# Path Forward — Checkpointed Plan

**Repository:** `ruman-shaikh/protein_tune_rl`, branch `new_algo`
**Created:** 16 November 2026
**Design constraint:** every checkpoint ends in a durable, committed artifact so
work can be paused, the machine shut down, and the next session resumed from
this file alone. Nothing below depends on a chat session staying alive.

---

## 1. The validated finding (do not re-derive this)

Gates run on 16 November 2026 against commit `3ef9fdc`. Hardware: RTX 4050
laptop, 6 GB. All numbers measured, not estimated.

**The `ss_perc_sheet` endpoint is satisfiable by length-capped composition
padding, and the trained policy converges on exactly that.**

| Evidence | Value |
|---|---|
| Hand-written `"V"*22`, no training, 50 held-out contexts | reward **3.338** |
| A2C after 82,000 steps (`a2c_s82000.ckpt`) | reward **3.366** |
| Best hand-written single context | **6.24** (RL ceiling 4.2) |
| Policy output at step ≥10,000 | length pinned at **22**, **99.9%** sheet-set residues |
| Top emitted residues at `s82000` | T 34%, Y 29%, V 12%, L 9% — all VIYFWLT |

Supporting results:

- **ESM-2 masked PLL is co-exploitable.** Region PLL: native −2.98 →
  `"V"*22` **−0.29**. Homopolymers of V, A, W and a VY-repeat all score
  ≈ −0.2 to −0.35, while native, shuffled-native and uniform-random all score
  ≈ −3.1 to −3.3. Masked PLL measures *predictability*, not plausibility: one
  masked residue in a run of identical residues is trivially inferable.
- **Diversity does not collapse.** 100% unique sequences, mean within-context
  edit distance 9–13, positional entropy ~1.9, at every checkpoint from 10,000
  to 82,000. The reward depends only on composition, so its optimum is a vast
  flat plateau. Standard diversity metrics report a healthy run throughout.
- **Endpoints do not agree.** Spearman ρ on 250 native antibodies: reward vs
  SASA −0.06, vs ESM-region −0.09, vs ESM-chain 0.02, vs CDR length −0.07.
  ESM-region vs ESM-chain only 0.34. SASA vs CDR length 0.38 (SASA is partly a
  length proxy).
- **Upstream PPO importance ratio is wrong by ~29%.** On a frozen, unchanged
  policy, full-vocab vs constrained-support log-probs give mean
  |ratio − 1| = 0.286 (median 0.281, max 0.502), 100% of sequences off by >5%.
  PPO's clip band is ±0.2, so ratios sat outside it before any policy change.
  *Caveat: measured on `s82000`, which is degenerate. Must be re-measured on
  base IgLM — see C6.*
- **Adaptive-KL controller bug never executed.** Every shipped config has
  `"target": null`, and `KLPenalty.__call__` guards the branch with
  `if self.target is not None`. Simulated on the real 4,337-step KL trajectory
  both buggy and correct forms diverge (β × 3.1e36 vs × 6.8e34), because
  observed KL (up to 62) saturates `e_t` at +0.2 every step. Footnote only.

**Reward definition, for the record:**
`reward = (sheet_fraction(H+L) − 0.3784639359426365) / 0.017445864484741506`,
where `sheet_fraction` is Biopython `secondary_structure_fraction()[2]`
= fraction of **V, I, Y, F, W, L, T**. Scope is whole chain H+L, so a ~13-residue
CDR is dilution-limited; the only way to move the metric is to lengthen the CDR
to `max_length` and fill it with sheet residues.

**Reproducibility landmine:** `setup.cfg` lists `biopython` unpinned. Before
v1.82 the tuple order was flipped, so index `[2]` returned *helix* (EMALK). Any
install with biopython <1.82 silently optimises a different property while every
label still says sheet. Installed here: 1.86. Stored CSV values match the
post-1.82 convention. **Pin biopython >=1.82.**

## 2. Paper direction

*The β-sheet endpoint used in published antibody-infilling RL is satisfiable by
length-capped composition padding; the trained policy converges on exactly that;
and none of the standard guardrails — proxy reward, masked-PLL plausibility,
sequence-diversity metrics — detect it. Only structural evaluation adjudicates.*

Second, independent contribution: the published PPO baseline was mis-implemented
(§C6).

Scope discipline: no binding, affinity, developability or therapeutic claims.
The bundled data have no antigens, complexes or affinity labels.

## 3. What is still unmeasured

1. **The closing claim.** That length-22 poly-sheet padding is not actually
   β-sheet. Asserted everywhere above, measured nowhere. Needs IgFold + DSSP.
2. **U2c baseline action-leakage.** Argued analytically; never measured with a
   real `StateValue`.
3. **Everything is single-seed.** The gates were viability checks. No number
   above is yet publishable evidence.

---

## 4. Checkpoints

Rules for every checkpoint:

- Ends with a committed artifact. `test_run/` is **gitignored and not backed up**
  — commit the small result CSVs under `experiments_for_publication/results/`
  so progress survives a lost or stolen laptop.
- Update §5 (Progress) and commit before shutting down.
- "Safe to shut down" marks a clean boundary.

**Schedule around network access.** C1 and C2 need downloads. Do both while
connected; everything else runs offline.

### C0 — Unblock the code (½ day, offline)

Four fixes, all verified still present at `3ef9fdc`:

| File | Fix |
|---|---|
| `metrics/structure.py:31` | `options["do_refine"]` → `.get("do_refine")`, same for `use_openmm`. Currently `SASA` and `FoldingConfidence` both raise `KeyError` on their own defaults. Blocks all of C2–C3. |
| `optimizer/a2c.py:180` | `normalize_across_processes` contains an `all_reduce` but sits inside `if valid_adv.numel() > 0`. One rank skipping it hangs the job. Move the collective outside the conditional; use a count-weighted global mean/var since per-rank counts differ. |
| `tune.py` | `_setup_tune()` pops `optimizer.name` before `_save_config()`, so saved `config.json` cannot reproduce a run. Deep-copy before constructing the trainer, or save first. |
| `protein_trainer/trainer.py:65,106` | `device_ids[0]` is `None` on CPU and `torch.cuda.set_rng_state` is unconditional. Guard both. Low priority on a GPU box; cheap. |

Also add a regression test pinning the `compute_logp` mask invariant: the mask
is shifted one position (`attention_mask[:, -A-1:-1]`), which exactly cancels the
stop token's `attention_mask=False`. It is **correct by accident**. If anyone
implements the deferred stop-token fix, PPO's importance ratio silently breaks.
Test that ratio == 1 on a frozen unchanged policy.

Pin `biopython>=1.82` in `setup.cfg`.

**Artifact:** commit. **Safe to shut down.**

### C1 — Lock the composition-attack result (½ day, NEEDS NETWORK)

Promote the gate script into `experiments_for_publication/scripts/` as a
reproducible experiment. All 50 held-out contexts, the three attack variants
(substitute / extend / anchored), plus per-context output.

**Confirm the ESM finding at 650M.** The gate used `esm2_t6_8M_UR50D`, the
weakest variant and the only one cached. Download `facebook/esm2_t33_650M_UR50D`
(~2.6 GB, fits 6 GB). The co-exploitability mechanism is architectural, so I
expect it to persist, but it must be shown at a credible model size or a reviewer
will dismiss it as a toy-model artifact.

**Artifact:** `results/c1_composition_attack.csv`, `results/c1_esm_scopes.csv`.
**Safe to shut down.**

### C2 — IgFold + DSSP setup and smoke test (½ day, NEEDS NETWORK)

IgFold has **never run on this machine** — verified: no `folding_workspace/`,
no stray `.pdb`/`.fasta`, `igfold`/`antiberty`/`pyrosetta`/`openmm` absent from
all three venvs, no cached weights, and every run ever produced used
`ss_perc_sheet`. This is first-time setup, not re-enabling something.

1. `pip install igfold` into `/home/ruman/venvs/general` (pulls AntiBERTy;
   downloads 4 IgFold checkpoints + AntiBERTy weights on first run).
2. Install DSSP (`mkdssp`) and confirm Biopython's `DSSP` wrapper finds it.
3. Run with `do_refine: False` — refinement is what needs PyRosetta/OpenMM, and
   neither is wanted.
4. Smoke test: fold three sequences (one native, one `"V"*22`, one `s82000`
   generation), run DSSP, confirm β-strand assignments come out and peak VRAM
   fits 6 GB.
5. Record exact installed versions.

Expect a possible torch-version fight. If IgFold cannot be made to work, fall
back to a different antibody-capable predictor; **do not** substitute ESMFold
(~2.8B params, needs ≥16 GB — will not run on this card).

**Artifact:** `results/c2_smoke.json`, `results/c2_versions.txt`.
**Safe to shut down.**

### C3 — The structural measurement (1 day, offline, GPU)

The decisive leg. Four conditions on matched held-out contexts:

1. native HCDR3
2. `"V"*22` hand-written attack
3. `s82000` policy generations
4. base IgLM samples (no RL)

Fold each with IgFold, run DSSP on the predictions, and report the **actual
assigned β-strand fraction within HCDR3**, plus loop geometry, SASA and model
confidence. Pilot ~50 sequences per condition; expand if variance demands.

Interpretation discipline: B-factor is pLDDT only if the producer wrote it there.
pLDDT is local prediction *confidence*, not folding evidence. RMSD is meaningful
only against a genuine matched reference — predicted-vs-predicted RMSD is not
native accuracy.

**Resumability:** fold in batches and append to the CSV after each batch, so a
shutdown mid-run loses at most one batch. Make the script skip sequences already
present in the output CSV.

**Artifact:** `results/c3_structure.csv`. **Safe to shut down between batches.**

### C4 — Multi-seed training matrix (1–2 days unattended, offline, GPU)

The first publishable evidence. 5 seeds × {REINFORCE, PPO (fixed), A2C} × 3 KL
weights, 2,000 steps, batch 8 — about 45 runs. Measured throughput is 81 seq/s,
so each run is minutes to tens of minutes; the matrix is roughly a long day
unattended. **Time 20 real training steps first** and extrapolate before
launching the full matrix.

Resume recipe, because this is the one checkpoint that spans shutdowns:

- set `"fixed_experiment_directory": true` so the output directory is stable
  across restarts (otherwise `tune.py` appends a fresh timestamp)
- set `"check_point_freq": 200`
- to resume, add `"checkpoint": "<path to latest .ckpt>"` to the config
- resume replays the dataloader to skip completed steps — fine at 2,000
- single GPU only: resuming with a different world size breaks
  (`rng_state["cuda"][rank]` indexes a list sized by the original world size)

Report against **measured KL**, not just step, so strong optimisation is not
confused with unconstrained drift. Freeze held-out contexts before any tuning.
Aggregate within context first, then bootstrap over contexts and seeds —
thousands of candidates from 50 prompts are not thousands of independent
examples.

**Artifact:** `results/c4_matrix_summary.csv` plus per-run `train_log.csv` left
in `test_run/`. **Safe to shut down at any checkpoint boundary.**

### C5 — Endpoint-agreement benchmark (½ day, offline)

Re-run the §1 correlation matrix with the structural endpoints from C2–C3 added:
DSSP β-fraction, IgFold confidence, IgFold-derived SASA. This is what converts
"the endpoints disagree" into "here is which endpoints can be trusted, and
structural assignment is the only one that separates the attack from native."

**Artifact:** `results/c5_endpoint_matrix.csv` + figure. **Safe to shut down.**

### C6 — PPO correctness, measured properly (½ day, offline)

1. Re-measure the 29% importance-ratio deviation on **base IgLM**, not `s82000`.
2. Close U2c: build a real `StateValue`, score prompt-only vs prompt+completion
   on identical rollouts, correlate each with realised reward. Requires a config
   with `"baseline": "state_value"` — the shipped PPO template uses `"mean"`, so
   the learned critic is never constructed and the correction is dormant.

**Artifact:** `results/c6_ppo_correctness.csv`. **Safe to shut down.**

### C7 — Figures and draft (offline)

Figure 1: hand-written attack vs 82,000-step policy, same axis.
Figure 2: emitted length and sheet-fraction vs training step.
Figure 3: diversity metrics flat while reward saturates.
Figure 4: DSSP β-fraction by condition — the adjudicating panel.
Figure 5: endpoint correlation matrix.
Table 1: importance-ratio deviation, before vs after.

Target an ML-for-biology workshop (GEM / MLSB / ICLR bio workshops). Verify
deadlines at the time — the previously drafted venue analysis was written
11 November 2026 and its cycle dates need rechecking.

---

## 5. Progress

Update and commit this table at every checkpoint boundary.

| CP | Description | Network | Status | Artifact |
|---|---|---|---|---|
| C0 | Unblock the code | no | **DONE** 16 Nov 2026 | `tests/test_c0_fixes.py`, 23 tests pass |
| C1 | Lock composition attack, ESM-650M | **yes** | **network part DONE** 16 Nov; script promotion pending (offline) | 650M weights cached |
| C2 | IgFold + DSSP setup | **yes** | **DONE** 16 Nov 2026 | `results/c2_smoke.csv`, `results/c2_versions.json` |
| C3 | Structural measurement | no | **DONE** 16 Nov 2026 | `results/c3_structure.csv`, `results/c3_sequences.csv` |
| C4 | Multi-seed matrix | no | not started | — |
| C5 | Endpoint benchmark | no | not started | — |
| C6 | PPO correctness | no | not started | — |
| C7 | Figures and draft | no | not started | — |

**Gate status from 16 Nov 2026:** P1 fired decisively. P2(a) fired (29%, promoted
to a contribution). P2(b) dead (never executed). P3 removed by instruction.
P4 inverted — diversity does not collapse; diversity metrics fail to detect the
pathology, which is a stronger result. P5 fired and was promoted from fallback to
a core component.

**Gate scripts** are preserved verbatim under `scripts/gates/` (see its README
for what each one established). They are exploratory — single seed, hard-coded
paths, printed output. C1 re-implements U1 properly with saved artifacts.

### C0 completion notes (16 Nov 2026)

All four fixes applied, plus the biopython pin. 23 tests pass (was 17).

- `metrics/structure.py` — `.get()` for `do_refine`/`use_openmm`; structure
  metrics can now be constructed. Unblocks C2/C3.
- `optimizer/a2c.py` — new `_normalize_advantages`: the `all_reduce` is now
  reached on every rank, and statistics are pooled by count rather than
  averaged per rank. `normalize_across_processes` is no longer used here
  (it remains correct for PPO, whose minibatches are equal-sized across ranks).
- `tune.py` — deep-copies the config at load and saves that snapshot, so
  `config.json` retains `optimizer.name`. **Runs from here on are reproducible
  from their own saved config; earlier run dirs are not.**
- `protein_trainer/trainer.py` — CUDA RNG gather and restore are now guarded,
  so CPU checkpointing works and resuming at a different world size logs a
  warning instead of raising `IndexError`.

The new `test_compute_logp_mask_counts_actions_through_the_stop_token` pins the
shifted-mask invariant. If it ever fails, fix the caller, not the test.

### C1/C2 completion notes (16 Nov 2026) — ALL NETWORK WORK IS DONE

Everything remaining in the plan runs offline.

**ESM co-exploitability confirmed at 650M.** Region-only masked PLL:

| Sequence | ESM-2 8M | ESM-2 650M |
|---|---|---|
| native CDR | −3.121 | −2.337 |
| shuffled native | −3.332 | −3.667 |
| uniform random (22) | −3.237 | −3.163 |
| poly-V (22) | −0.313 | −0.246 |
| poly-A (22) | −0.348 | −0.301 |
| VY-repeat | −0.351 | −0.105 |

Not a toy-model artifact. The 650M model is *more* discriminating — it
penalises shuffled-native harder than the 8M does, which the 8M barely
separates — and still rates `"V"*22` about 2.1 nats/residue more likely than
the native CDR. Peak GPU 2,984 MB of 6,141.

**Structural pipeline works, and the closing leg already points the right way.**
IgFold 0.4.0 + AntiBERTy 1.0.0 + pydssp 0.9.2, folding at ~1.2–1.9 s/sequence,
peak 441 MB. On context 0 (n=1, C3 does this properly):

| Condition | HCDR3 β-strand fraction | whole-H |
|---|---|---|
| native (`ASLNWDWYFDF`) | **0.636** | 0.547 |
| poly-V at max_length | **0.136** | 0.461 |
| anchored | 0.318 | 0.492 |

The edit that *maximises* the β-sheet reward reduces predicted β-strand
content. At ~1.5 s/fold, C3's 200 folds is roughly 5 minutes.

**Superseded by C3 — do not cite these numbers.** This single context showed
0.636 → 0.136, a five-fold drop. At n=46 the effect is real and highly
significant but far smaller (0.414 → 0.298). The n=1 result was not
representative. Use the C3 numbers below.

**Environment notes, carried forward:**

- `igfold` installs additively — torch 2.5.1 and transformers 4.46.3 were
  already satisfied and were **not** changed. Six new packages; reversible
  with `pip uninstall`.
- **This venv's `wandb` is broken independently of this work** (wandb 0.25.1
  vs protobuf 7.36.0: its generated `_pb2` modules are incompatible, so
  `import wandb` fails on its own). `pytorch_lightning` 1.8.6, which IgFold
  imports, eagerly imports wandb. Scripts stub `wandb` in `sys.modules` before
  importing igfold rather than upgrading a shared package. **If IgFold is ever
  used as a training reward**, `metrics/structure.py` imports igfold at module
  level, so either repair wandb or add a guarded stub there.
- No system `mkdssp` (needs sudo). `pydssp` provides the DSSP algorithm in
  pure Python and needs no system package.

### C3 completion notes (16 Nov 2026) — the closing leg, measured

46 matched held-out contexts × 4 conditions = 184 structures. **All 184 folded,
zero failures.** IgFold, `do_refine=False`, pydssp C3 assignment, β-strand
fraction taken over the HCDR3 span of the predicted heavy chain.

| Condition | n | reward | **β-strand (HCDR3)** | β-strand (whole H) | CDR len |
|---|---|---|---|---|---|
| native | 46 | 0.102 | **0.414** | 0.505 | 13.6 |
| `iglm_base` (no RL) | 46 | −0.093 | **0.374** | 0.499 | 14.6 |
| `policy_82k` | 46 | 3.325 | **0.298** | 0.482 | 22.0 |
| `polyV_maxlen` | 46 | 3.330 | **0.277** | 0.477 | 22.0 |

**Reward vs predicted HCDR3 β-strand fraction: Spearman ρ = −0.339,
p = 2.5 × 10⁻⁶, n = 184.**

What this establishes:

1. **The reward is anti-correlated with the property it is named after.** Not
   merely uninformative — *negatively* associated, significantly so.
2. **Maximising it reduces predicted β-strand content in the designed loop**
   by about 28% relative (0.414 → 0.298) while the reward rises roughly 32× in
   normalised units (0.102 → 3.325).
3. **The trained policy and the one-line string hack are indistinguishable on
   both axes** — reward 3.325 vs 3.330, β-strand 0.298 vs 0.277. 82,000
   optimisation steps bought nothing a hand-written constant does not.
4. Whole-chain β-strand barely moves (0.505 → 0.482): the dilution that forces
   the length exploit also hides it from any whole-chain readout.

Honest effect size: the drop is moderate, not catastrophic. The policy still
produces loops with ~0.30 β-strand fraction, because IgFold places strands at
the CDR anchors regardless of the fill. The defensible claim is the
anti-correlation plus the policy/one-liner equivalence, **not** "RL destroys
β-sheet structure."

Remaining caveats: single seed; one folding predictor (C5 or O4 should add a
second); β-strand assignment depends on IgFold's loop placement, which is the
least reliable part of any antibody structure prediction.
