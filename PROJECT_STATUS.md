# ProteinTuneRL Project Status

Last updated: 2026-11-05

## Current objective

Extend ProteinTuneRL with independent protein-language-model evaluation and
reference-based structural evaluation:

1. ESM-2 masked pseudo-log-likelihood scoring.
2. PDB structure comparison for generated antibody regions.

## Completed

- Added `ESMSequenceScoring` with:
  - Hugging Face ESM masked-language-model support.
  - Whole-heavy-chain and generated-region scoring.
  - Mean or sum pseudo-log-likelihood reduction.
  - Batched residue masking and optional score normalization.
- Added `PDBStructureEvaluator` with:
  - Framework-based rigid alignment.
  - Framework backbone RMSD.
  - CDR-region backbone RMSD for equal-length loops.
  - Region-length comparison.
  - Whole-chain and region SASA.
  - Mean region B-factor extraction.
- Added `scripts/compare_pdb_structures.py` for JSON-producing command-line
  structural comparisons.
- Added `scripts/evaluate_esm.py` for standalone whole-chain and HCDR3 scoring.
- Registered `esm_scoring` in the metric factory.
- Added focused usage and interpretation guidance in
  `docs/ESM_PDB_EVALUATION.md`.
- Made the DDP optimizer integration checks initialize and clean up their
  process group when collected by pytest.
- Fixed A2C and PPO device placement to follow the supplied policy instead of
  moving auxiliary tensors and critics to CUDA whenever any GPU is visible.

## Current handoff

- Local ESM-2 smoke demonstration completed on five held-out sequences (see below).
- Publishing RL fixes, evaluator modules, regression tests, and the research report
  on the fork's `new_algo` branch.
- Six upstream issues were submitted under `ruman-shaikh`:
  https://github.com/llnl/protein_tune_rl/issues/1 through /6. These are issues,
  not pull requests or merged contributions.
- Imperial HPC access has ended. Cluster scaffolding is removed; research logs,
  checkpoints, datasets, report sources, and figures remain intact. Ignored local
  run outputs and model weights are not backed up by pushing the repository.

## Validation status

- Revalidated on 2026-11-05 with Python 3.10.19: 17 tests passed in 4.81 seconds.
  CPU/Gloo integration checks ran outside the socket-restricted sandbox. One
  unrelated Google API dependency emitted a Python-version support warning.
- Existing RL unit tests: 10 passing.
- Optimizer integration checks: 2 passing under pytest and as standalone checks.
- New ESM and PDB unit tests: 5 passing.
- Complete test suite: 17 passing.
- Cached ESM-2 GPU smoke test: 5 held-out sequences scored in 9.5 seconds;
  mean whole-chain log probability -2.367 and mean HCDR3 log probability
  -3.051.

## Next steps

1. Expand cached ESM-2 evaluation beyond the existing five-sequence smoke test.
2. Generate a small ESM score summary for generated-region versus whole-chain
   evaluation.
3. Validate PDB evaluation on a matched experimental/predicted structure pair
   when one is supplied.

## Known limitations

- CPU optimizer tests are supported, but training checkpoint save/load still assumes
  CUDA RNG state. CPU training is not end-to-end checkpoint-safe.
- Stop-token credit, empty-action distributed advantage normalisation, and folding
  default-option/retry-cleanup edge cases remain unresolved; current tests do not
  establish exhaustive multi-rank correctness.
- ESM masked pseudo-log-likelihood requires one model forward pass per scored
  residue, although masks are batched to reduce overhead.
- PDB region RMSD is reported only for equal-length regions. Length difference
  remains available as a separate metric.
- Mean PDB B-factor is reported generically. It should be interpreted as pLDDT
  only when the structure producer explicitly stores pLDDT in that field.
- The existing online trainer supports one reward metric per run. ESM scoring
  can be used as that metric or as a separate evaluator; multi-objective reward
  composition remains future work.
- The bundled antibody datasets do not contain antigen structures, antibody-
  antigen complex PDB identifiers, experimental docking poses, epitopes,
  paratopes, or binding-affinity measurements. They are therefore unsuitable
  as docking ground truth.

## Docking data assessment

- Current data provide paired heavy/light sequences, annotated CDR sequences,
  computed SASA, beta-sheet composition, and proxy-derived preference labels.
- A defensible docking benchmark requires an external set of experimentally
  resolved antibody-antigen complexes, with antigen chains and native complex
  poses retained.
- SAbDab is the preferred source because it consistently annotates antibody
  structures harvested from the PDB and includes antibody-antigen complexes.
- Docking should remain a downstream, optional evaluation workflow. Native
  complexes can be separated into antibody and antigen inputs, redocked, and
  assessed by pose/interface recovery before generated antibodies are ranked.
