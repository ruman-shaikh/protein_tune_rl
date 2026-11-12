# Research Directions by ChatGPT — ProteinTuneRL Publication Plan

**Prepared:** 12 November 2026  
**Repository:** [ruman-shaikh/protein_tune_rl](https://github.com/ruman-shaikh/protein_tune_rl), branch **new_algo**  
**Research basis:** GitHub code and project report review; independent Deep Research literature survey (November 2026); comparison with the sibling document [Research Directions by Claude](Research_Directions_by_Claude.md).  
**Status:** Research hypotheses and proposed protocols, **not** completed experimental findings.

## Executive decision

The strongest opportunity is a **correctness-controlled investigation of proxy reward exploitation in antibody CDR-H3 reinforcement learning**. The central question is:

> When an IgLM-based policy improves a cheap antibody-design reward, how much of that improvement reflects meaningful changes in independently assessed sequence and predicted structure, and how much depends on implementation details, policy drift or an easily exploited reward definition?

Two paper theses remain open until experiments resolve them:

- **Thesis A — correctness and scientific conclusions:** Correcting KL control, constrained-token probability accounting and PPO value-baseline errors changes measured policy drift, optimisation stability or algorithm ranking.
- **Thesis B — proxy validity (preferred):** The fork's example beta-sheet composition reward can be increased by amino-acid enrichment without corresponding improvement in independent sequence or coordinate-derived structural measures. Correct KL-constrained optimisation may change this trade-off.

The most publishable outcome is **a reproducible, mechanistically explained effect**, including a robust negative result. Adding textbook A2C+GAE to an existing framework is an implementation contribution; a new scientific finding comes from the experiments.

**Scope:** computational antibody sequence infilling, RL implementation correctness, reward design, sequence-distribution drift and predicted structure. Binding affinity, antigen specificity and therapeutic utility require different ground truth.

## 1. Prior art and the actual novelty gap

| Work | Established result / relevance | What our paper must add |
| --- | --- | --- |
| [IgLM — Shuai, Ruffolo & Gray (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11018345/) | Antibody infilling by a pretrained sequence model and evaluation of generated designs. | A controlled analysis of post-training rather than novelty in infilling itself. |
| [LLNL ProteinTuneRL — Lee et al. (2025)](https://www.biorxiv.org/content/10.1101/2025.08.08.669419v1) and [code](https://github.com/LLNL/protein_tune_rl) | Online REINFORCE/PPO with KL regularisation and offline optimisation for antibody sequence infilling, including antibody-relevant proxy/experimental tasks. | Show that a specified implementation correction or reward choice changes conclusions on a matched problem. |
| [GermRL (2026)](https://www.biorxiv.org/content/10.64898/2026.06.08.730660v1) | Antibody-specific RL with attention to biological priors and plausibility. | A distinct claim in IgLM / CDR-H3 infilling, not a generic claim that antibody RL works. |
| [AbOptimizer (2026)](https://www.nature.com/articles/s41467-026-78533-6) | Multi-property antibody editing and experimental validation. | State computational evidence at its proper level; do not claim validated functional improvement from model scores. |
| [iGC (2026 preprint)](https://arxiv.org/abs/2611.05270) | RL over a learned somatic-hypermutation prior, addressing exploitation / biological constraints. | Contrast our focus on implementation correctness and the semantics of a particular reward. |
| [Goodhart-style failures in TCR generation (2026)](https://academic.oup.com/bioinformatics/article/42/Supplement_2/btag499/8767262) | Surrogate reward optimisation can separate from biological realism. | Demonstrate or refute that failure mode specifically for ProteinTuneRL's IgLM / CDR-H3 setting. |
| [RLXF (2026)](https://www.nature.com/articles/s41467-026-77557-2) | Protein-language-model RL alignment toward functional objectives. | Use independent evaluation and conservative biological claims. |

**Important distinction:** The **fork's example** beta-sheet reward calls Biopython's amino-acid-composition statistic; it is **not equivalent** to a structural beta-sheet measurement and must not automatically be described as the reward used in all experiments in the original LLNL paper. Compare matched reward definitions when making original-versus-fork claims.

## 2. Repository baseline: confirmed versus experimental

Checked on branch **new_algo**:

- **Implemented:** token-level A2C+GAE for variable-length infilling; KL-controller and constrained-probability corrections; prompt-only learned PPO baseline; truncation bootstrap; ESM-2 masked pseudo-log-likelihood evaluator; framework-aligned PDB evaluator; regression/integration tests.
- **Documented validation:** 17 tests passing (10 RL unit, 2 optimiser integration, 5 ESM/PDB) and a five-sequence ESM-2 smoke test. These validate components, not trained-policy superiority.
- **Existing figures:** historical single runs, not a fair multi-seed algorithm comparison. Current actual 2,000-step configurations use batch size 8 and maximum generated length 22; PPO template is a 10-step template, so publication runs require explicit complete configs.
- **Config-specific blockers for meaningful ablations:** example KL settings have **target: null**, so adaptive-target controller logic is dormant. The PPO template has **baseline: mean**, so the learned **state_value** baseline correction is dormant until enabled. A2C's learning rate differs from PPO/REINFORCE in existing examples. Existing figures must not be treated as controlled algorithm evidence.
- **Reward definition:** protein_tune_rl/metrics/ss_perc_sheet.py computes ProteinAnalysis(sequence).secondary_structure_fraction()[2]; whole-chain scoring uses H+L, while region_only scoring uses the generated HCDR3. Recompute normalisation constants for the selected reward scope.
- **Available data:** the included CSV files support sequence experiments, but do not include antigen–antibody complexes, matched experimental binding labels or docking ground truth. Structural comparison tooling needs external matched structures where native-reference measurements are desired.
- **Engineering checks before final training:** inspect stop-token credit, empty-action cases, distributed advantage normalisation, folding fallback paths and logging semantics; retain the actual executed configuration and model/hash for each run. Current REINFORCE/A2C/optimizer tests and README provide a starting point.

Repository anchors: [project status](../PROJECT_STATUS.md), [report](../paper/main.tex), [README](../README.md), [PPO template](../configs/examples/ppo_iglm_hcdr3_beta_sheet_template.json), [A2C config](../configs/examples/a2c_iglm_hcdr3_beta_sheet.json), [reward implementation](../protein_tune_rl/metrics/ss_perc_sheet.py), [RL tests](../tests/test_fixes.py), [ESM/PDB tests](../tests/test_esm_and_pdb.py).

## 3. Five publication-critical experiments (ranked)

### E1 — Original versus corrected RL: one-fix-at-a-time ablations

**Hypothesis.** At least one correctness repair measurably changes the reward–KL frontier, sample efficiency, training stability, or conclusions about which algorithm performs best.

**Design.** Start from a frozen upstream commit, then construct controlled versions with identical IgLM-S weights, prompts, generation constraints, data order, seed, optimiser settings and reward:

1. Original/upstream PPO, with the chosen feature active.
2. Corrected PPO with the same feature active.
3. Corrected PPO with **one** correction reverted at a time.
4. Optional: corrected all-fixes-together condition and interactions, if effects are coupled.

**Critical, separable ablations:**

- **Probability-support fix:** sampler constraints versus PPO/reference log-probabilities evaluated on the same support. On a frozen unchanged policy, importance ratios should equal one to numerical tolerance; quantify the incorrect ratio and KL estimate under mismatched support. Verify both synthetic and sampled-sequence cases.
- **Adaptive KL controller:** choose a **non-null target** and a prespecified desired KL. Compare original versus corrected controller and monitor target tracking, coefficient trajectory, and realised KL. **With target null this experiment does not test the controller fix.**
- **PPO learned value baseline:** enable **baseline = state_value**; compare completion-conditioned (action-leaking) and prompt-only predictions. The **baseline = mean** setting cannot test this correction.
- **A2C truncation bootstrap:** separate A2C experiment with forced maximum-length truncation versus genuine stop-token termination; test advantage bias and end-to-end learning. This is not a PPO ablation.
- **Gradient clipping:** treat as an optimisation stability ablation, not the same category as a mathematically invalid log-probability estimate.
- **CPU checkpoint issue:** a reproducibility/engineering correction, not a central biological performance claim.

**Outcomes.** Unpenalised task reward, independently recomputed sequence-level log-ratio KL estimate (with its estimator and limitations identified), target-KL tracking, importance-ratio distribution, entropy, gradient norms, divergence/failure rate and policy changes. Record coefficients **before and after** updates when analysing adaptive KL. Audit that stored reward and KL logs match their documented semantics.

**Figure/table.** Before/after importance-ratio diagnostic; fixed-versus-adaptive KL tracking; one-row-per-ablation effect sizes with paired seed intervals; reward versus empirical KL curves.

**Decision criterion.** A reproducible shift in control accuracy, measured behaviour or scientific ranking supports Thesis A. If effects are negligible in the tested regime, report that outcome and emphasise Thesis B.

**Priority:** First to execute; establishes a reliable foundation for the rest.

### E2 — Is the beta-sheet reward a composition exploit? (mechanistic negative controls)

**Hypothesis.** The example composition statistic can be optimised through residue-frequency changes without improved coordinate-derived HCDR3 beta-sheet structure.

**Conditions, all on identical held-out antibody contexts:**

- Unmodified IgLM sampling (no RL).
- Corrected PPO and corrected A2C+GAE trained on the example reward.
- **Best-of-N selection control:** draw N candidates from IgLM and retain candidates with the highest composition score; match and separately report candidate/oracle evaluation budget.
- **Composition-only mutation control:** bias editable HCDR3 positions toward the residues that the Biopython statistic counts (V/I/Y/F/W/L/T), while matching mutation/edit budget and checking valid sequences.
- Reward **whole H+L** versus **HCDR3-only**, with independently calibrated normalisation.

Biopython's sequence-level sheet propensity is based on amino-acid identity, not folding. Thus a strong rise in this reward alone does not establish new beta-sheet geometry. See the [Biopython source](https://github.com/biopython/biopython/blob/master/Bio/SeqUtils/ProtParam.py).

**Primary measurements.** Raw composition score; targeted residue-frequency shifts; positional entropy; sequence identity to starting HCDR3; uniqueness/diversity; ESM-2 and antibody-specific model scores (E3); coordinate-derived sheet assignment (E4).

**Proposed final sampling scale.** Pilot with 25–50 held-out contexts. For final sequence analysis, target approximately **200 cluster-held-out contexts × 32 generation samples per method** if compute permits; these numbers are design proposals and should be power/precision checked.

**Figure.** Composition reward versus independent plausibility/structure, annotated by method and reward scope; per-position amino-acid enrichment.

**Decision criterion.** If simple composition baselines approach RL reward gains without matching independent sequence/structural outcomes, the proxy is demonstrably easy to exploit. If the proxy correlates well with independent measures, that result narrows or redirects Thesis B.

### E3 — Held-out, independent sequence evaluation

**Hypothesis.** Changes in proxy reward need not track antibody sequence plausibility, and KL/reward design affects the gap.

**Design.** Evaluate the **same frozen candidates** from E2 (plus corresponding original/corrected PPO checkpoints) using multiple independent metrics:

1. **ESM-2 masked pseudo-log-likelihood**: whole-heavy-chain and HCDR3-only, with consistent per-residue normalisation. Your fork already implements the metric.
2. **Antibody-specific LM**: select a documented, compatible AntiBERTy / AntiBERTa / AbLang-family model; specify whether its score is a pseudo-likelihood, anomaly score or another quantity. Do not assume incomparable model scores have the same numerical meaning.
3. **IgLM reference likelihood**: report separately as **prior retention**, not an independent biological oracle.
4. **Sequence diversity**: exact unique fraction, pairwise edit distances/identity within prompt, position-wise entropy, length distribution; optional simple developability metrics reported separately.

**Controls.** Same prompt/framework, candidate count, decoding temperature and evaluation length handling; include pretrained IgLM, a composition optimiser, original PPO, corrected PPO and A2C when available. Prevent score leakage by ensuring an evaluator is never also used as a training reward in that comparison.

**Outcomes.** Independent-model score distributions, effect sizes relative to IgLM, reward–plausibility frontiers, diversity. Keep each metric separate; avoid an arbitrary synthetic “quality” composite.

**Statistical unit.** Aggregate candidate measurements **within each held-out antibody context first**. Bootstrap over contexts and, for training uncertainty, over independent seeds. Thousands of candidates from the same few prompts are not thousands of independent biological examples.

**Figure/table.** Held-out metric matrix by method; scatterplots of raw reward versus separate evaluators with context- or seed-aware uncertainty.

**Decision criterion.** Robust reward gains accompanied by neutral or worsening independent-model compatibility support a computational proxy-mismatch finding, **not** experimental loss of binding.

### E4 — Structural reality check for the reward name

**Hypothesis.** Increased *compositional* beta-sheet propensity does not necessarily yield increased **predicted** HCDR3 beta-sheet structure.

**Design.** Predefine a manageable, balanced subset from E2: random samples across reward quantiles from each condition, plus extremes, with identical selection rules and lengths matched or stratified. Pilot approximately 50 sequences per condition; expand according to variability and inference cost. Fold using a fixed antibody-capable structure predictor, frozen settings and one versioned model. Run a secondary-structure assignment method (e.g. DSSP) on the predicted coordinates.

**Outcomes.** Actual assigned beta-strand residue fraction in HCDR3, structural clashes/loop geometry, accessibility, model confidence, and optionally agreement between different predictors. RMSD is meaningful **only** with a legitimate native/matched reference; do not interpret predicted-to-predicted RMSD as native accuracy.

The existing PDB evaluator can contribute framework alignment, region RMSD for equal-length loops, SASA and B-factors. Interpret stored B-factors as pLDDT only if generated that way. pLDDT measures local **prediction confidence**, not affinity or demonstrated folding. See [AlphaFold confidence documentation](https://www.alphafold.ebi.ac.uk/faq).

**Figure.** Compositional reward versus coordinate-assigned HCDR3 beta-strand fraction, with predictor uncertainty and representative structures.

**Decision criterion.** Poor or inconsistent association supports the mechanistic mismatch; a positive relationship is scientifically informative and would shift the story toward corrected RL performance or other properties.

### E5 — Fair PPO / A2C+GAE / REINFORCE comparison

**Hypothesis.** Token-level credit assignment may change sample efficiency or the reward–KL–independent-plausibility frontier relative to sequence-level alternatives.

**Design.** Use a single primary task and common IgLM-S initialisation, training prompts, reward definition, generation limit, batch size, seed set and data split:

- PPO with all corrections.
- A2C+GAE with all relevant corrections.
- REINFORCE with all relevant corrections.
- Untrained IgLM as a no-RL reference.
- Optionally original PPO to connect to E1, and offline DRO as a distinct baseline if task/data are appropriate.

**Tuning policy.** Use equal **search budgets**, not necessarily identical numerical learning rates (different algorithms can require different scales). Starting grid proposal: learning rate {1e-6, 3e-6, 1e-5}; fixed KL coefficient {0.003, 0.01, 0.03}; two tuning seeds. Select via held-out reward **subject to a prespecified KL ceiling** or use a clearly reported Pareto analysis. Freeze selected settings before testing.

**Final protocol proposal.** Five independent training seeds; 2,000 optimisation steps × batch 8 as initial shared budget (16,000 sampled training examples/run), with frozen checkpoint evaluations at steps 0, 250, 500, 1,000, 1,500 and 2,000. Use **fixed held-out prompts and fixed generation seeds** for each checkpoint. Distinguish number of samples from number of distinct sequence contexts and report oracle calls, training FLOPs/GPU hours when measured, and wall-clock.

**Outcomes.** Unpenalised reward; reference KL; diversity; ESM-2 and antibody-LM scores; sample efficiency; seed variability; failure rates. Report results both against step and against **measured KL** to avoid conflating strong optimisation with unconstrained drift.

**Figure.** Multi-seed reward–KL Pareto fronts with 95% intervals; held-out plausibility and diversity versus training budget.

**Decision criterion.** If A2C consistently improves a controlled frontier it is a useful secondary result. If it does not, this supports the interpretation that the reward and the constraint dominate the choice of policy-gradient optimiser.

## 4. Experimental quality requirements

### Dataset and leakage controls

- Record dataset source/version and deduplicate heavy-chain/HCDR3 contexts; prefer similarity-clustered train/validation/test splitting and report clustering rule/identity threshold.
- Freeze test contexts before hyperparameter selection. Avoid selecting top model checkpoints using the final test set.
- Preserve native heavy/light pairing where available; keep HCDR3 generation constraints consistent across methods.
- Define precisely whether the experiment scores H alone, H+L or the infilled span. With whole-chain scores, most residues may remain unchanged; control for that dilution explicitly.

### Statistical analysis

- **5 independent training seeds** for headline claims; run short pilots and at least 2–3 exploratory seeds first, then freeze protocol.
- Use paired seed comparisons for original/corrected variants; bootstrap over independent held-out antibody contexts for sequence metrics; use hierarchical/bootstrap inference when combining seed and context variance.
- Predefine the primary endpoint (e.g. area under the held-out **raw-reward-versus-KL** curve under a KL ceiling; or difference in DSSP-assigned HCDR3 beta-strand fraction at matched reward), effect size, uncertainty interval and correction across multiple ablations.
- Report per-seed data and full distributions, not only smoothed mean curves; distinguish descriptive comparisons from statistically supported findings.
- Track invalid sequences, early stopping, failed folding, truncated generations and model errors as explicit denominators.

### Instrumentation and run reproducibility

Before final training, verify:

- Consistent token support for sampling, PPO importance ratios and reference log-probs.
- KL penalty coefficient and **pre-/post-update** logging; the logged KL estimate must be independently checked rather than only inferred by dividing penalised values by a possibly updated beta.
- Raw task reward logged separately from the KL-penalised training objective.
- A2C terminal reward placement, truncation bootstrap, stop-token handling and valid-action masks.
- Exact config, Git SHA, dependencies, model-weight identifier/hash, dataset split, seed, hardware, elapsed time, and checkpoint path recorded per run.
- Evaluate all methods with the **same generation constraints and evaluation code**; do not compare current historical figures as if they were a matched benchmark.

### Compute and feasibility

The repository does not establish benchmarkable GPU hours for new experiments. Run **one short timing pilot per method**, log throughput and memory, then estimate the full matrix. The main run budget will consist of dozens of 2,000-step training runs if every sweep and ablation is attempted; use staged narrowing (synthetic correctness tests → short pilots → selected conditions → five-seed finals) to conserve compute. Sequence-based evaluation reuses candidates and adds no RL retraining; folding will likely be a separate, more expensive inference budget.

## 5. Priority, minimum paper package and decision gates

| Sequence | Minimum action | Output / decision |
| --- | --- | --- |
| **Gate 0** | Lock upstream SHA; reproduce one original and one corrected run; audit config, logs, invariants and splits. | Validated protocol and tested ablation switches. |
| **Gate 1 (E1)** | Run original/corrected and active-feature correctness ablations. | Do fixes have measurable scientific consequences? |
| **Gate 2 (E2)** | Run no-RL, PPO/A2C and two simple composition controls on the same reward. | Is reward improvement explained by residue composition? |
| **Gate 3 (E3)** | Apply ESM-2 and antibody-LM evaluators to the very same held-out candidate set. | Does the proxy improvement transfer to independent sequence models? |
| **Gate 4 (E4)** | Test coordinate-derived structure on a balanced subset. | Is the term “beta-sheet improvement” structurally supported? |
| **Gate 5 (E5)** | Publish a matched multi-seed algorithm comparison. | Does token-level A2C materially change the frontier? |

**Minimum strong workshop story:** E1 + E2 + E3 + E5, with E4 as the most valuable addition because it directly tests the semantics of the beta-sheet reward. To support a strong statement about actual predicted beta structure, include E4. A pure correctness paper can centre E1 and E5 if the effects are substantial and reproducible.

**Conditional pivot:**

- If proxy reward and structure/PLM scores diverge → paper on **reward exploitation and control** (preferred).
- If the reward is a useful surrogate but implementation fixes change performance → paper on **correctness of KL-constrained antibody RL**.
- If both differences are small → technically valuable reproduction/benchmark release, with a narrower workshop or reproducibility focus.
- If A2C improves the held-out frontier beyond corrected PPO → highlight credit assignment as a secondary methods result; investigate whether it generalises to a second reward task.

## 6. Optional extensions after the core results

1. **A second real structural objective:** revisit LLNL's IgFold-based SASA / RMSD-style tasks, keeping training and held-out evaluators distinct. This tests whether results are unique to a composition shortcut.
2. **Offline DRO / experimental labels:** use a public experimentally measured antibody property dataset with cluster-held-out labels; report Spearman correlation and confidence intervals without claiming full prospective binder improvement.
3. **Diversity-aware objective:** add and ablate explicit sequence diversity regularisation, comparing reward–diversity Pareto fronts to current KL-only runs.
4. **Cross-predictor structural validation:** use two distinct structure predictors for a representative subset to assess predictor-specific conclusions.
5. **Multi-objective optimisation:** combine independent objectives only after single-reward reliability is demonstrated; compare against matched single-objective controls.

Antigen-conditioned docking or claimed binding improvements require external matched complex structures or experimentally measured binding labels; the included CSV files alone do not supply those endpoints.

## 7. Paper design and meeting-ready summary

**Candidate title (Thesis B):** *When Antibody Reinforcement Learning Optimises a Proxy: A Correctness-Controlled Study of Reward Exploitation in CDR-H3 Infilling*

**Alternative title (Thesis A):** *Implementation Correctness Changes the Reward–Divergence Trade-off in Antibody Sequence Reinforcement Learning* (conditional on evidence).

**Eight-to-ten-slide discussion outline for Dr Faraz Janan:**

1. Antibody CDR-H3 infilling: biological/ML task.
2. LLNL ProteinTuneRL and the pretrained IgLM policy.
3. Exact fork contributions versus inherited upstream code.
4. Why sampler/log-prob/KL/baseline correctness matters.
5. Why the example beta-sheet reward is compositional, not a folded structure.
6. E1: before/after correctness ablations and target-KL control.
7. E2: composition-based negative controls.
8. E3/E4: held-out sequence and structural verification.
9. E5: matched, multi-seed PPO/A2C/REINFORCE experiment.
10. Decision gates, compute pilot, proposed venue strategy and supervisor feedback.

The message to the supervisor is a **falsifiable study**, not a forecasted winner: “Does correcting online RL implementation and measuring independently of the reward reveal a different conclusion about antibody-sequence optimisation?”

## 8. Primary references and implementation sources

**Research papers:**

1. Shuai, Ruffolo & Gray (2023), *IgLM: Infilling language modeling for antibody sequence design*. [Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11018345/).
2. Lee, Hayes, Vashchenko & Landajuela (2025), *Reinforcement Learning for Antibody Sequence Infilling*. [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.08.08.669419v1).
3. Ludwig, Chungyoun & Gray (2026), *GermRL*. [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.06.08.730660v1).
4. Qiu et al. (2026), *Multiproperty optimization of therapeutic antibodies through sequence editing via large language model and reinforcement learning*. [Nature Communications](https://www.nature.com/articles/s41467-026-78533-6).
5. Nguyen et al. (2026), *iGC: Antibody Optimization via Reinforcement Learning over a Learned Somatic Hypermutation Prior*. [arXiv](https://arxiv.org/abs/2611.05270).
6. Zhang et al. (2026), *Mitigating Goodhart's law in epitope-conditioned TCR generation using plug-and-play reward designs*. [Bioinformatics](https://academic.oup.com/bioinformatics/article/42/Supplement_2/btag499/8767262).
7. Blalock et al. (2026), *Functional alignment of protein language models via reinforcement learning*. [Nature Communications](https://www.nature.com/articles/s41467-026-77557-2).
8. Hong et al. (2026), *Comprehensive evaluation of antibody language models across multiple antibody design tasks*. [Genome Biology](https://link.springer.com/article/10.1186/s13059-026-04323-4).

**Implementation and evaluator interpretation:**

- [LLNL reference repository](https://github.com/LLNL/protein_tune_rl).
- [Six issues submitted from this fork](https://github.com/LLNL/protein_tune_rl/issues?q=is%3Aissue+author%3Aruman-shaikh) — upstream issues/proposals, not merged pull requests.
- [Biopython beta-sheet propensity source](https://github.com/biopython/biopython/blob/master/Bio/SeqUtils/ProtParam.py).
- [AlphaFold confidence FAQ](https://www.alphafold.ebi.ac.uk/faq).
- [ESM repository](https://github.com/facebookresearch/esm).

**Document relationship:** This file provides a correctness- and proxy-validity-first experimental design based on the Deep Research report and direct repository inspection. [Research Directions by Claude](Research_Directions_by_Claude.md) supplies an additional, partly complementary perspective emphasising credit assignment, reward–KL–diversity frontiers, workshop feasibility, and optional multi-task extensions. Both should be retained for the publication discussion.
