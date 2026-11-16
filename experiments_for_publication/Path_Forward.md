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
- **Importance-ratio deviation — RETRACTED in its strong form, see C6 below.**
  U2 measured mean |ratio − 1| = 0.286 on `a2c_s82000`. C6 re-measured on
  **base IgLM** and got **0.0008**. The earlier claim that "the published PPO
  baseline was mis-implemented" does not survive; at initialisation the bug is
  numerically negligible. What survives is that its severity *scales with
  policy drift*.
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

## 2. Paper direction — main venue (revised 16 Nov 2026)

**Target: a research paper, not a workshop note.** Benchmark/evaluation track
(NeurIPS Datasets & Benchmarks) or a computational-biology journal
(Bioinformatics, PLOS Computational Biology).

*In-silico endpoints for antibody sequence RL are systematically exploitable.
We characterise a set of endpoints under optimisation pressure, show that the
standard guardrails — masked-PLL plausibility scoring and sequence-diversity
metrics — fail to detect the failure, and provide a diagnostic that does.*

Three contributions:

1. **An attackability characterisation** of the in-silico endpoints this
   literature actually uses: compositional β-sheet, SASA, IgFold confidence,
   DSSP β-strand, ESM-2 masked PLL. Which survive optimisation pressure, which
   do not, and by what mechanism.
2. **A demonstration that the standard guardrails fail together**, which is the
   part the field has no statement of. Plausibility scoring is co-exploitable;
   diversity metrics stay healthy throughout.
3. **A gaming detector** that needs no held-out oracle, built from the
   degeneracy signatures measured here: generation length pinning at the
   sampler cap, composition concentration into a small residue set, and reward
   saturation while an independent oracle diverges.

The β-sheet case stops being the thesis and becomes the worked example. That
also answers the strongest reviewer objection — "you attacked an obviously bad
reward" — because the claim is about the *class* of cheap endpoints.

Secondary contribution, methods appendix: the upstream PPO importance ratio is
wrong by ~29% (§C6). It also establishes that the baselines here are sound.

**Why this framing and not "better antibodies."** The bundled data have no
antigens, no complexes and no affinity or expression labels, and there is no
wet lab. No functional claim is available from this project at any effort
level. Benchmark and diagnostic contributions are the categories that do not
require functional validation, so they are the only routes to a research-level
venue here. Binding claims would need external data (SAbDab complexes, or a
public affinity dataset for offline DRO) and are a separate project.

**Hardware is not the constraint.** IgLM-S is 1.4M parameters; measured
throughput is 81 sequences/s and folding is ~1.5 s/structure, both on a 6 GB
laptop GPU at a fraction of capacity. Only ESMFold/AlphaFold-class models are
out of reach.

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

### C4 — Endpoint attackability screen (½ day, offline) **[RESTRUCTURED]**

The gate that must precede the expensive training. Before committing 6–7 GPU
hours per run to any endpoint, establish whether that endpoint can be moved by
trivially degenerate edits — the same discipline that made U1 worth running.

Eight hand-written HCDR3 attacks (`polyV_max`, `polyG_max`, `polyW_max`,
`polyP_max`, `VY_repeat`, `minimal`, `anchored_V`, vs `native`) scored against
every candidate endpoint, reported as gain in **native standard deviations**.

Split into two scripts with no dependency between them, so they run
concurrently (measured: 1.3 GB combined on a 6 GB card):

- `c4a_structural_attackability.py` — one IgFold call per (context, attack)
  yields SASA, IgFold confidence and DSSP β-strand together, plus the
  compositional endpoints on identical sequences.
- `c4b_esm_attackability.py` — ESM-2 masked PLL as a *reward*, at two scopes
  and two model sizes, to separate exploitability from model capacity.

Both resumable: one appended row per completed item, skipping what is present.
Tables join on `(context, attack)`.

**Decision rule for C4b-prime (training):** only endpoints that resist the
hand-written attacks are worth training against. An endpoint a constant string
already maximises needs no RL run to condemn it.

### C4-prime — Train against the surviving endpoints (the research core)

For each endpoint that survives C4, run RL against it and measure whether
optimisation finds an exploit the hand-written attacks missed. This is where
the compute goes.

Cost from measured rates: folding-based rewards cost ~1.5 s/sequence, so
~12 s/step at batch 8 — about **6–7 h per 2,000-step run**, against ~15 min for
the compositional reward. Budget 2–3 seeds per surviving endpoint;
~50–80 GPU-hours total, which suits intermittent laptop use.

Use the C4 §Resume recipe below. Report against **measured KL**, not step.

### C4-resume — recipe for any long training run

- set `"fixed_experiment_directory": true` so the output directory is stable
  across restarts (otherwise `tune.py` appends a fresh timestamp)
- set `"check_point_freq": 200`
- to resume, add `"checkpoint": "<path to latest .ckpt>"` to the config
- resume replays the dataloader to skip completed steps — fine at 2,000
- single GPU only: resuming with a different world size now logs a warning
  rather than raising, but RNG continuity is lost (C0)

The old plan's 45-run matrix (5 seeds × 3 algorithms × 3 KL weights on the
compositional reward) is **cut**. Under the main-venue framing the algorithm
comparison is not the contribution; 3 seeds on the compositional reward is
enough to establish the worked example, and the compute belongs in C4-prime.

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
| C4 | Endpoint attackability screen (restructured) | no | **DONE** 16 Nov 2026 | `results/c4a_*.csv`, `results/c4b_*.csv` |
| C4′ | Train against surviving endpoints (research core) | no | blocked on C4 | — |
| C5 | Endpoint benchmark table | no | blocked on C4′ | — |
| C6 | PPO correctness | no | not started (independent — parallelisable) | — |
| C6′ | Gaming detector | no | blocked on C4′ | — |
| C7 | Figures and draft | no | not started | — |

**Dependency graph** (for parallel scheduling): C4a ∥ C4b ∥ C6 are mutually
independent. C4′ needs C4. C5 and C6′ need C4′. C7 needs everything.
Measured concurrency headroom: C4a (IgFold, 0.44 GB) + C4b (ESM-650M, 3 GB)
ran together at 1.3 GB peak on a 6 GB card.

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

### C4 completion notes (16 Nov 2026) — the endpoint taxonomy

20 held-out contexts × 8 hand-written HCDR3 attacks, no training, no search.
Reported as gain in **native standard deviations**; positive means the attack
raises the endpoint. C4a (structural, 160 folds) and C4b (ESM, 320 scores) ran
concurrently at 1.3 GB combined.

| Endpoint | Best attack | Gain | Verdict |
|---|---|---|---|
| ESM-2 PLL, region, 650M | `VY_repeat` | **+6.58σ** | gameable (worst) |
| composition β-sheet, region | `polyV`/`polyW`/`VY` | **+5.15σ** | gameable |
| SASA (whole complex) | `polyP_max` | **+2.98σ** | gameable |
| composition β-sheet, H+L | `polyV`/`polyW`/`VY` | **+2.36σ** | gameable |
| ESM-2 PLL, whole chain, 650M | `VY_repeat` | +1.52σ | weakly gameable |
| **IgFold confidence** | — | all attacks **negative** | **resistant** |
| **DSSP β-strand, region** | — | all attacks **negative** | **resistant** |

The split falls exactly on the cheap/structural line: every sequence-level
endpoint is gameable by a hand-written constant; both coordinate-derived
endpoints resist all eight attacks.

Findings that matter for the paper:

1. **The plausibility oracle is more gameable than the reward it checks** —
   ESM-2 PLL at +6.58σ against the compositional reward's +2.36σ.
2. **Three published endpoints, three distinct mechanisms.** β-sheet falls to
   sheet-set residues; SASA falls to poly-proline (rigid extended residues
   maximise exposed surface); ESM PLL falls to *any* homopolymer — poly-G and
   poly-P raise it ~+5.8σ while *lowering* the β-sheet reward by −1.64σ. This
   is what makes it a class result rather than one bad reward choice, and it
   answers the strongest reviewer objection.
3. **Model capacity halves the ESM exploit but does not remove it**: region PLL
   gain falls from +13.3σ (8M) to +6.2σ (650M). Still six sigma.
4. **`region_only` scoring is more gameable than whole-chain** (+5.15σ vs
   +2.36σ). The repo offers region-only as the improvement, since it removes
   framework dilution — but dilution was what limited the exploit.
5. Internal consistency check passed: `polyV`, `polyW` and `VY_repeat` give
   identical composition gains, as they must, the sheet set being VIYFWLT.

**Decision for C4′:** only the two resistant endpoints are worth training
against. "Resistant to eight hand-written attacks at n=20" is *not*
"unexploitable by search", and closing that gap is the research question.

### C4′a — make structural rewards trainable, and measure the real cost

Prerequisite checkpoint, cheap, must precede C4′b:

1. **Write a DSSP β-strand metric.** It does not exist — C4a computed it in a
   standalone script. Needs a metric class (fold → pydssp → β-fraction over the
   generated span), factory registration, and normalisation constants from
   native data.
2. **Smoke-test both structural rewards in the training loop.** C0 unblocked
   construction, but no training run has ever called a structure metric. The
   per-sample folding, 3-retry logic and `fail_value` fallback are untested
   under training.
3. **Measure seconds/step on an idle GPU.** C4a ran at 4.2 s/fold under
   contention vs 1.5 s/fold clean — the difference between a 2,000-step run
   taking ~7 h or ~18 h.
4. **Verify terminate/resume explicitly**: run ~20 steps, kill the process,
   resume from the checkpoint, confirm it continues. The step-skip in
   `OnlineRLTrainer.run` sits *before* the collator and reward call, so a
   resume advances the dataloader without re-folding — important, since
   re-folding 14,400 structures to reach step 1,800 would cost ~6 h. Cheap,
   but never tested with a structural reward.
5. For folding rewards set `check_point_freq` to ~50, not 200: at ~12 s/step a
   200-step gap is 40 minutes of exposure.

### C4′a completion notes (16 Nov 2026) — all four items done

**1. DSSP metric written.** `metrics/dssp_beta.py`, `DSSPBetaStrand`,
registered in the factory as `dssp_beta`. Folds with IgFold, assigns secondary
structure with pydssp, returns the β-strand fraction over the generated span.
Normalisation from C4a natives (region mean 0.419, std 0.108). Behaves
correctly as a reward: **native +2.013, poly-V-22 −2.617** — it penalises
exactly what the compositional reward rewards.

**2. `structure.py` now self-heals the wandb import.** `_ensure_wandb_importable`
installs a stub *only* if the installed wandb raises on import, leaving a
working wandb untouched. Without this, every structure reward was unusable in
training for an unrelated dependency fault. Scripts no longer need their own
stub.

**3. Measured cost: ~6 s/step** (78 s for 12 steps; 5.7 s between consecutive
step logs), batch 8, one fold per sample, idle GPU. **Far cheaper than the
7–18 h estimate: a 2,000-step run is ≈3.3 h.**

| | per step | 2,000 steps |
|---|---|---|
| compositional reward | ~0.3 s | ~15 min |
| **structural reward (measured)** | **~6 s** | **~3.3 h** |

C4′b budget: 2 endpoints × 3 seeds × 3.3 h ≈ **20 GPU-hours**. Suits
intermittent laptop use.

**4. Terminate/resume verified bit-exact.** A fresh 12-step run and a run
resumed from the step-4 checkpoint produced *identical* reward values at every
shared step (step 4: −0.13346260786056519; step 5: 0.15311908721923828;
step 11: −0.09035690128803253). RNG state, dataloader replay and optimiser
state all survive a kill. With `check_point_freq: 50` at ~6 s/step, the
exposure window is ~5 minutes.

**Configs for C4′b** are in `configs/experiments/`:
`a2c_iglm_hcdr3_dssp_beta.json` and
`a2c_iglm_hcdr3_folding_confidence.json`, both with
`fixed_experiment_directory: true` and `check_point_freq: 50`.

### C6 completion notes (16 Nov 2026) — NEGATIVE, with a retraction

**C6a. The 29% importance-ratio finding is retracted in its strong form.**

| Policy | mean \|ratio − 1\| | >20% (PPO clip band) |
|---|---|---|
| `a2c_s82000` (degenerate; what U2 used) | 0.286 | 100% |
| **base IgLM** (what published runs start from) | **0.0008** | **0%** |

Three orders of magnitude smaller at initialisation. Base IgLM places almost
no probability mass on the 12 masked control tokens, so renormalising over the
constrained support barely changes the log-probs; the degenerate trained policy
had drifted onto them. The earlier claim — "the published PPO baseline was
mis-implemented" — **does not survive and is withdrawn.**

What survives, and is worth one paragraph in the methods: **the bug's severity
scales with policy drift**, 0.08% at initialisation rising to 29% once the
policy has degenerated. It is invisible early in training and becomes severe
exactly when the policy is going wrong, which reinforces the paper's main
theme rather than constituting a contribution of its own.

**C6b. Baseline action-leakage: inconclusive.** Held-out R² predicting the
realised reward from the pooled final hidden state (5-fold ridge, n=48):
prompt-only −0.035, prompt+completion +0.091, both ±0.36–0.40. No leakage
demonstrated. n is far too small, and a pooled-hidden-state probe may be the
wrong instrument for what a trained critic head could extract. Either scale n
by an order of magnitude or drop the claim.

**Effect on the paper: none.** The thesis rests on C1/C3/C4 endpoint
attackability, untouched. P2/C6 was always scoped as a methods appendix; it is
now a methods footnote. Found before submission rather than in review.
