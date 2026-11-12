# Turning the ProteinTuneRL Fork into a Publishable Paper: A Critical Peer Review and Experimental Roadmap

## TL;DR

- The verified contribution in `ruman-shaikh/protein_tune_rl` (branch `new_algo`) is an engineering and correctness extension of LLNL's ProteinTuneRL, not yet a scientific result: it adds a token-level A2C critic with Generalised Advantage Estimation (GAE), fixes several genuine online-RL bugs in the upstream PPO/KL implementation, and adds reward-independent ESM-2 and structure-based evaluators. There are currently no multi-seed, controlled, or trained-policy benchmark results, so as it stands the work is not acceptance-worthy at any peer-reviewed venue without new experiments.
- The single strongest publishable research question the code is positioned to answer is narrow and honest: "For short, variable-length antibody CDR-H3 infilling under sparse terminal rewards, does finer-grained credit assignment (token-level A2C+GAE) change the proxy-reward versus policy-drift (KL) trade-off relative to REINFORCE and sequence-level PPO, and does any apparent reward gain survive held-out-oracle and structural validation rather than being reward hacking?" This is a focused RL-methods-for-antibodies study, not a claim of better binders.
- The realistic target is a workshop paper (GEM/MLSB at NeurIPS 2026, or a 2027 ICLR bio workshop), contingent on running the controlled multi-seed, multi-task, held-out-oracle experiments described in the roadmap below. A main-conference ML paper (NeurIPS/ICML/ICLR) or a computational-biology journal/proceedings paper (ISMB, Bioinformatics, PLOS Comp Biol) is not realistic on the current evidence and would require in vitro data or a substantially novel algorithmic contribution the fork does not yet contain.

## What the Code Actually Contains (Verified Against the Repos)

**Upstream baseline (LLNL/protein_tune_rl).** The original framework, from Lee, Hayes, Vashchenko and Landajuela, "Reinforcement Learning for Antibody Sequence Infilling" (bioRxiv 2025.08.08.669419), uses IgLM, a transformer infilling model, to redesign CDR-H3 loops while preserving framework context. It implements online RL (REINFORCE and PPO with KL regularization) and offline RL via Direct Reward Optimization (DRO), a value-head MSE method adapted from Richemond et al. In the paper the online rewards are structure-derived proxies computed with IgFold: beta-sheet percentage, solvent accessible surface area (SASA), and predicted RMSD to a target backbone. The paper reports that all RL-tuned models beat vanilla IgLM on the proxy, that online REINFORCE reaches the highest reward (for example 13310 ± 223 on the beta-sheet task) but at a cost in IgLM log-likelihood or plausibility (dropping to about −5.82 ± 0.704) relative to the lower-reward PPO/DRO rows, with the authors demonstrating that "KL regularization is essential for maintaining a balance between score optimization and sequence plausibility," and that KL regularization strength beta controls the exploration/exploitation trade-off. Offline DRO is evaluated on nine experimental antibody datasets by Spearman correlation between model log-likelihood and the measured property. Compute was 8 V100 GPUs. This is the prior art the fork builds on, and it already occupies the obvious "RL for antibody infilling" claim.

**The fork's genuine additions (verified from README, PROJECT_STATUS, the six upstream issues, and the subagent check):**

1. **Token-level A2C with GAE (the headline "new algorithm").** A per-position value head, masked GAE with configurable discount and lambda, terminal reward placed at the final valid action, and an explicit distinction between stop-token termination (no bootstrap) and max-length truncation (bootstrap from the final state). Upstream has no A2C, so this is a real addition to the framework. It is, however, a textbook RL algorithm applied to a new setting, not a new algorithm.
2. **Online-RL correctness fixes, filed as upstream issues #1-#4.** These are the most intellectually substantive part of the diff. Issue #1: the adaptive KL controller computed `KL - target/target` (i.e. `KL - 1`) instead of `(KL - target)/target`, and promoted a scalar coefficient into a per-example tensor carried across batches and ranks. Issue #2: the sampler masks control tokens to minus infinity but the PPO and reference-KL log-probabilities were normalized over the full vocabulary, so even an unchanged policy produced an importance ratio different from one, biasing PPO and the KL estimate. Issue #3: the PPO "state_value" baseline conditioned on the prompt plus the sampled completion rather than the prompt only, making an action-independent baseline depend on the action. Issue #4: CPU training cannot checkpoint because save/restore assumes CUDA RNG state.
3. **Reward-independent evaluation (issue #5 plus structure tools).** Batched ESM-2 masked pseudo-log-likelihood scoring of the heavy chain or the generated span, and a structure comparison script (framework alignment, backbone RMSD, region-length difference, SASA, mean B-factor/pLDDT). This is exactly the right instinct for avoiding circular evaluation, because the training reward and the evaluator are different models.
4. **Tests and reproducibility scaffolding.** 17 CPU tests on synthetic tensors, fake ESM models, synthetic PDBs and a tiny randomly initialised GPT-2; figure-reproduction scripts from retained run logs.

**What the fork does NOT contain (critical caveats, verified).** The repository is unusually honest and self-disclaiming, which a reviewer will respect but which also defines the work's current ceiling:

- No multi-seed or controlled benchmark exists. The README states verbatim that "algorithm names do not imply a controlled or multi-seed benchmark" and that existing figures are "individual historical runs, not multi-seed evidence that one algorithm produces better antibodies," with run settings differing across methods.
- The only "validation" beyond unit tests is a five-sequence cached ESM-2 GPU smoke test (mean whole-chain log probability about -2.367, HCDR3 about -3.051), explicitly labelled "implementation smoke-test results, not trained-policy benchmarks."
- `ss_perc_sheet` (the beta-sheet reward in the fork's example config) is Biopython's amino-acid composition statistic, not a structure prediction. This is a serious evaluation-validity problem if used as a scientific endpoint.
- The online trainer supports only one reward metric per run; multi-objective composition is future work. Region RMSD is undefined for unequal loop lengths. B-factor equals pLDDT only when the structure producer writes pLDDT into that field.
- The bundled data have no antigen complexes and no experimental affinity labels, so the repo cannot by itself provide binding ground truth. Imperial HPC access has ended and cluster scaffolding was removed, though logs, checkpoints and figures were retained.
- The six issues are bug reports and proposals, not merged pull requests, so the contribution is not yet recognised upstream.

**Verification limits.** The subagent could confirm the README, PROJECT_STATUS content, the six issue titles and descriptions, and the A2C design as described, but could not open the raw `paper/main.tex`, the report PDF, or the actual `protein_tune_rl/` source files to quote the implementation line by line or extract in-paper quantitative tables. Any specific numeric claim inside the fork's own report should therefore be treated as unverified until the code and logs are run. A reviewer, and the professor, should insist on re-running from the committed configs before trusting any historical figure.

## Where This Sits in the 2025-2026 Literature

The space of RL and preference optimization for protein and antibody generative models is now crowded, which raises the novelty bar sharply. The honest framing is that the fork's methods are individually well precedented; only the specific combination and the antibody-infilling correctness analysis are unclaimed.

- **RL fine-tuning of protein LMs.** ProtRL / DPO_pLM (Stocco et al., Reinforcement Learning Guides Generative Protein Language Models, arXiv 2412.12979) fine-tunes autoregressive pLMs (including ZymCTRL) with KL-regularized RL and DPO to hit external rewards. RLXF (Functional alignment of protein language models via reinforcement learning, bioRxiv 2025.05.02.651993) aligns pLMs to experimental fitness across five families. ProteinRL (Sternke and Karpiak, NeurIPS 2023 GenBio) is the earlier property-directed version. ProteinZero (Wang et al., arXiv 2506.07459) does online RL (RAFT and GRPO) for inverse folding with an embedding-level diversity regularizer against mode collapse; its GRPO variant surpasses all baselines, enhancing FoldX ddG by 19-21% and leading on scRMSD < 2 Angstrom fraction and pLDDT, cross-checked across ESMFold, AlphaFold3 and FoldX.
- **Preference optimization (DPO family).** ProteinDPO (Widatalla et al., now in Nature Methods, s41592-026-03137-3) aligns a structure-conditioned model to stability. Multi-objective binder DPO (Mistani and Mysore, arXiv 2403.04187), g-DPO (Ferragu et al., arXiv 2510.19474, accepted at FM4LS NeurIPS 2025 and MLSB EurIPS 2025), AnnoDPO, DPLM-Evo, and Designability Preference Optimization all exist. This means a plain "DPO/PPO for antibodies" paper is already too incremental.
- **Antibody-specific RL, the nearest competitors.** GermRL (Ludwig, Chungyoun and Gray, JHU, bioRxiv 2026.06.08.730660) uses GRPO on ProGen2-OAS. Per the paper: "Before reinforcement learning (GRPO), ProGen2-OAS generated antibodies whose foldability deteriorated past ~10 mutations from germline. After GRPO, ProGen2-OAS generates antibody sequences up to 35 mutations from germline while maintaining foldability." Its reference list cites the exact LLNL paper this fork extends (entry [39]: "Chak Shing Lee, Conor F Hayes, Denis Vashchenko, and Mikel Landajuela. Reinforcement learning for antibody sequence infilling. bioRxiv, August 2025."). AB-Gen (PPO on a GPT policy for CDR-H3) and Structured Q-learning for antibody design predate it. BetterBodies (arXiv 2409.16298) is RL-guided diffusion for antibody sequences. This is the single most important competitive cluster for the fork: antibody-specific RL with diversity and structural plausibility is being actively published by established labs.
- **Reward hacking, diversity and mode collapse.** This is where the fork's framing ("proxy optimisation versus policy drift") is most current. "KL-regularized RL is designed to mode collapse" (arXiv 2510.20817) argues collapse is intrinsic to the objective. GFlowNet-vs-GRPO work on therapeutic peptides (bioRxiv 2026.01.05.697258) shows GRPO collapses without explicit diversity penalties while GFlowNets keep diversity for free. SGRPO (arXiv 2605.08659) pushes set-level utility-diversity frontiers. These give the fork a live, citable scientific question.
- **Oracle validity and circular evaluation.** Multiple 2025-2026 works show why reward-independent, held-out evaluation matters. Per Garcia et al. (Protein Science 2026, pro.70453; bioRxiv 2025.07.29.667290), evaluated on 614 experimentally characterized de novo monomers from 11 studies, "ESMFold and AlphaFold2 average pLDDT had the highest average AUCs across articles (0.71 ± 0.13 and 0.71 ± 0.17)," and in bootstrapping "ESMFold's average pLDDT had the highest AUC ... (0.73±0.04) whereas MPNN score had the lowest AUC (0.66±0.05)" at distinguishing design success. "RL for Sequence Design Leveraging Protein Language Models" (arXiv 2407.03154) and the Multiflow/distillation line explicitly separate a training oracle (ESMFold) from an evaluation oracle (AlphaFold2) to detect oracle overfitting. This literature both motivates the fork's independent evaluators and sets the standard the experiments must meet.

Net assessment: the fork is downstream of a fast-moving field. Its defensible niche is not "RL improves antibodies" (claimed by the LLNL paper and GermRL) but "a careful, correctness-first, held-out-oracle study of credit assignment and the reward-plausibility-diversity trade-off in short variable-length CDR infilling."

## The Strongest Publishable Research Questions

In priority order of scientific value given the code:

- **RQ1 (credit assignment).** Does token-level A2C+GAE change the reward-versus-KL and reward-versus-diversity Pareto frontier for short CDR-H3 infilling compared with sequence-level PPO and REINFORCE, under matched hyperparameters and multiple seeds? This is the fork's natural claim.
- **RQ2 (reward hacking / oracle validity).** When a model maximizes a cheap proxy (IgFold-derived SASA/beta-sheet or a composition statistic), how much of the gain survives a held-out oracle (ESMFold/AlphaFold2 pLDDT, independent structure metrics) that was not used in training? This directly tests the "proxy optimisation versus policy drift" thesis and is the most novel-feeling angle.
- **RQ3 (correctness matters).** Do the four upstream bugs (adaptive-KL normalization, vocabulary-support mismatch, baseline leakage, checkpoint) materially change conclusions, i.e. does fixing them alter the measured reward/KL trade-off or the ranking of algorithms? A clean "these implementation details change the scientific conclusion" result is publishable at a workshop and is genuinely useful to the community.
- **RQ4 (diversity preservation).** Does token-level credit assignment, or an added diversity term, mitigate the mode collapse that KL-regularized RL is known to induce, measured by sequence entropy, unique-sequence fraction and edit-distance spread?

## Experimental Roadmap (the core of this report)

Overarching principles the roadmap enforces, each tied to a reviewer expectation:

- Never evaluate on the training reward alone. Always report a held-out oracle not used in training, to avoid circular evaluation.
- Every headline comparison uses at least 5 random seeds, reports mean and standard deviation or a bootstrap confidence interval, and uses a stated statistical test (for example a paired test across matched prompts with multiple-comparison correction).
- Hyperparameters are matched across algorithms, or tuned with equal budget per algorithm, and the protocol is stated. The fork's own issue #6 concedes this is required.
- Report reward, plausibility (ESM-2 pseudo-log-likelihood and IgLM log-likelihood), diversity, and structure jointly, because any one in isolation can be reward-hacked.

Compute tiers assume modest academic resources. Low = single GPU, hours to a couple of days, or CPU for evaluation. Medium = single GPU over days, or a few GPUs. High = several GPUs for one to two weeks, or wet-lab, which is out of scope here.

### ESSENTIAL experiments (required for any credible workshop submission)

**E1. Controlled multi-seed algorithm comparison on a shared task.**

- Hypothesis: token-level A2C+GAE achieves a reward-versus-KL and reward-versus-plausibility trade-off at least comparable to, and in some regime better than, sequence-level PPO and REINFORCE for CDR-H3 infilling.
- Methodology: fix one well-defined structure-based reward (a real IgFold-derived metric, not the Biopython composition statistic), one prompt set, identical IgLM-S initialization, identical sampling budget. Sweep KL strength beta over at least three values. Run REINFORCE, PPO (post-fix), A2C+GAE, and DRO. At least 5 seeds each.
- Baselines: vanilla IgLM (no RL), REINFORCE, PPO, DRO. These are the LLNL methods, so the comparison is apples to apples.
- Metrics: mean reward; KL to reference; ESM-2 pseudo-log-likelihood; IgLM log-likelihood; wall-clock and sample efficiency (reward per oracle call).
- Publishable evidence: a Pareto plot showing where, if anywhere, A2C+GAE dominates, with error bars and a significance test. A credible negative result ("finer credit assignment does not help for short loops, and here is why") is also publishable at a workshop and is scientifically honest.
- Compute: Low to Medium.

**E2. Held-out oracle / reward-hacking check (the novelty anchor).**

- Hypothesis: a non-trivial fraction of proxy-reward improvement does not transfer to an independent structural oracle, and the gap grows as beta decreases (weaker KL).
- Methodology: train on the cheap proxy; evaluate every checkpoint with an independent oracle never used in training, for example ESMFold pLDDT and scRMSD, plus the fork's structure comparison script. Plot training-oracle reward and held-out-oracle score versus training step and versus beta. Quantify the divergence point.
- Baselines: pretrained IgLM; each RL algorithm from E1.
- Metrics: correlation between proxy reward and held-out oracle across generations; designability (pLDDT greater than 70 and scRMSD less than 2 Angstrom); fraction of "reward up, oracle down" samples.
- Publishable evidence: a clear demonstration that proxy gains partially fail to generalize, with the KL strength that best balances proxy gain and held-out retention. This is the result most likely to be seen as a contribution rather than an engineering report.
- Compute: Medium (ESMFold inference on a GPU; batchable).

**E3. Diversity and mode-collapse characterization.**

- Hypothesis: lowering KL raises proxy reward but collapses CDR-H3 diversity; credit-assignment choice shifts this curve.
- Methodology: for every run in E1, sample many completions per prompt and compute diversity metrics; relate diversity to beta and to algorithm.
- Baselines: IgLM sampling at matched temperature; the natural CDR-H3 diversity of the prompt set.
- Metrics: unique-sequence fraction, pairwise edit-distance distribution, positional entropy, and germline distance (connecting to GermRL's framing).
- Publishable evidence: a reward-diversity frontier per algorithm, showing whether token-level A2C mitigates or worsens collapse. Ties directly to the 2025-2026 diversity literature.
- Compute: Low (sequence metrics) to Medium (if structure-filtered).

**E4. Ablation and correctness study of the four fixes.**

- Hypothesis: at least one upstream bug (most likely the vocabulary-support mismatch in issue #2 or the adaptive-KL normalization in issue #1) materially changes the measured KL/reward trade-off or algorithm ranking.
- Methodology: toggle each fix on and off in an otherwise identical pipeline; measure effect on reward, measured-versus-target KL, and importance-ratio statistics (for issue #2, show the ratio departs from 1 for an unchanged policy before the fix).
- Baselines: the buggy upstream behavior versus the fixed behavior.
- Metrics: realized KL versus target KL; importance-ratio histograms; final reward and plausibility.
- Publishable evidence: a table showing which fixes change conclusions. This converts "bug fixes" into a methodological lesson, which is a legitimate workshop contribution and strengthens E1-E3 by proving the baselines are implemented correctly.
- Compute: Low to Medium.

**E5. Statistical and reproducibility rigor layer (not optional).**

- Report seeds, confidence intervals, the exact statistical test and correction, and the full hyperparameter grid. Fix the `ss_perc_sheet` endpoint: either replace it with a genuine secondary-structure assignment (for example DSSP on a predicted structure) or clearly restrict it to a sanity-check role, never a scientific endpoint. Verify every historical figure by re-running from the committed configs before reuse.
- Compute: Low.

### OPTIONAL extensions (raise the ceiling toward a stronger workshop or short paper)

**O1. Two additional reward tasks for generality.** Replicate E1-E3 on SASA and pRMSD (the other two LLNL tasks) so conclusions are not single-task artifacts. Medium.

**O2. A diversity-aware or set-level objective.** Add an explicit diversity penalty or a GFlowNet-style or SGRPO-style set-level reward and test whether it moves the reward-diversity frontier, as the peptide GFlowNet and SGRPO papers report for other modalities. This edges toward a genuine algorithmic contribution. Medium to High.

**O3. Offline DRO on a real experimental dataset with a held-out split.** Use one of the public antibody binding or developability datasets, train offline, and evaluate ranking on a held-out split, reporting Spearman correlation with proper uncertainty. This connects the in silico story to real measurements without needing a wet lab. Medium.

**O4. Multi-objective reward composition.** Since the trainer currently supports one reward per run, implementing and testing a two-objective reward (for example structure plus plausibility) would both remove a stated limitation and align with multi-objective binder-design literature. Medium.

**O5. Structure-based validation with two independent folders.** Fold a sample of designs with both ESMFold and an AlphaFold-family model and report agreement, to show the structural conclusions are not an artifact of one predictor. Medium to High.

**O6. Cross-oracle transfer (train on proxy, distill, evaluate on a stronger oracle).** Following the Multiflow and distillation precedent, test whether training on the cheap proxy overfits the proxy relative to a stronger held-out oracle. High.

Experiments that would be needed for a main-conference or journal claim but are out of scope under the stated constraints: any wet-lab expression or binding validation, and large-scale multi-model, multi-family benchmarks. These should be named in the paper as future work, not attempted.

## Venues, Deadlines and Feasibility (verified against official sources, as of 11 November 2026)

The honest bottom line: the realistic near-term target is an ML-for-biology workshop with the E1-E5 experiments done. Computational-biology proceedings venues with early-2026 deadlines have already passed for the 2026 cycle, and the 2027 cycle is the practical window for most of them. The student finishes in September 2026, so a late-2026 or 2027 workshop fits the timeline.

**Machine-learning workshops (best fit).**

- **GEM at NeurIPS 2026 (Atlanta, 13 December 2026).** The in-silico track fits this work. Per the official gembio.ai site the submission deadline was 30 August 2026, 11:59 PM AoE (notification 11 October 2026, camera-ready 25 November 2026), aligned with NeurIPS 2026 workshop guidance (suggested contribution date 29 August 2026). That deadline has passed, so the 2026 edition is not available. GEM recurs at ICLR and NeurIPS, so the next GEM (ICLR 2027) is the realistic target. Feasibility: high if E1-E5 are completed; GEM explicitly welcomes in-silico work and negative or analysis results.
- **MLSB (Machine Learning in Structural Biology), NeurIPS workshop.** The 2025 edition ran at NeurIPS and EurIPS on 7 December 2025. A 2026/2027 edition is the target; verify the call when posted. Feasibility: medium to high. MLSB is competitive but values rigorous methods and honest evaluation; E2 and E4 are the kind of content it rewards. The PRX Life MLSB collection offers an archival route for extended versions.
- **ICLR 2027 bio workshops (GEMBio, MLGenX, Gen2).** For reference, the ICLR 2026 cycle deadlines were early February 2026 (GEM/GEMBio about 2-5 February 2026, MLGenX 2 February 2026, Gen2 7-11 February 2026), all past. The equivalent ICLR 2027 workshop deadlines will fall around late January to early February 2027, which is the most realistic concrete target given the roadmap workload. Feasibility: high for a workshop paper with E1-E5.

**Machine-learning main conferences (not realistic now).** NeurIPS 2026's main-track deadline was 6 May 2026 (past). ICLR 2026's was 24 September 2025 (past); ICLR 2027's will be around September 2026-2027 cycle timing. A main-track submission would need a novel algorithm and strong multi-task, multi-seed results with structural validation, which the fork does not yet have. Recommendation: do not target a main ML conference with the current scope.

**Computational-biology venues (mostly 2027 cycle).**

- **ISMB 2026 (Washington DC, 12-16 July 2026).** Proceedings deadline was 20 January 2026 and abstract deadline 9 April 2026, both past; proceedings publish in Bioinformatics (Oxford University Press). The abstract/poster route for ISMB 2027 (deadlines likely around April 2027) is a feasible way to present in-progress work. Feasibility for a proceedings paper: low without substantially more (ideally experimental) evidence.
- **ECCB 2026 (Geneva, 31 August-4 September 2026).** Proceedings deadline was 16 March 2026 and abstracts 30 April 2026, both past.
- **MLCB 2026 (New York, 16-17 November 2026).** Paper deadline was around 1-3 July 2026 (extended), past. MLCB 2027 (likely mid-2027 deadline) is a strong fit for an 8-page version if E1-E5 plus one or two optional experiments are done. Feasibility: medium for 2027.
- **RECOMB 2027 (around May 2027; abstract deadlines in the preceding autumn).** Possible but competitive; better suited to a more complete methods contribution.
- **Journals (PLOS Computational Biology, Bioinformatics).** Open-ended deadlines. Realistic only after the full roadmap, and stronger with O3 (real experimental dataset) included.

**Recommended plan.** Target an ML-for-biology workshop in the ICLR 2027 or NeurIPS 2027 cycle. Do E1-E5 as the minimum viable paper, add O1 and O3 if time allows, and frame the paper around RQ2 (held-out-oracle / reward-hacking) and RQ3 (correctness changes conclusions) rather than a bare "A2C beats PPO" claim, which the evidence may not support and which the field would find incremental.

## Caveats and Reviewer-Style Warnings

- The work's current honesty is an asset, but a reviewer will read the absence of multi-seed results as the paper not yet existing. Everything hinges on running E1-E5.
- Do not overclaim biology. The fork cannot speak to binding or developability without antigen complexes or experimental labels. The paper must stay within "proxy optimisation, plausibility, diversity and predicted structure."
- Avoid circular evaluation at all costs. If the beta-sheet composition statistic is both the reward and part of the evaluation, the result is uninterpretable. Separate training reward from evaluation oracle everywhere (E2).
- Novelty is the main risk. Token-level A2C, KL control and ESM-based evaluation are all individually known. The contribution must be the controlled analysis and the correctness findings, positioned explicitly against GermRL, ProteinZero, DPO_pLM/ProtRL and the mode-collapse literature, not against a strawman.
- Attribution. The paper must credit the LLNL framework and the Lee et al. paper prominently and scope the claim to the delta. The six issues are unmerged, so describe them as proposed fixes validated in the fork, with the exact buggy-versus-fixed evidence from E4.
- Reproducibility. Because the subagent could not open the raw source and the historical figures are single runs with differing settings, re-run every figure from committed configs before trusting it, and release seeds and configs with the paper.

## Sources

- [LLNL/protein_tune_rl](https://github.com/llnl/protein_tune_rl)
- [Reinforcement Learning for Antibody Sequence Infilling (bioRxiv 2025.08.08.669419)](https://www.biorxiv.org/content/10.1101/2025.08.08.669419v1)
- [Reinforcement Learning for Antibody Sequence Infilling (full PDF)](https://www.biorxiv.org/content/10.1101/2025.08.08.669419v1.full.pdf)
- [LLNL upstream issue #3](https://github.com/llnl/protein_tune_rl/issues/3)
- [LLNL upstream issue #4](https://github.com/llnl/protein_tune_rl/issues/4)
- [ruman-shaikh/protein_tune_rl](https://github.com/ruman-shaikh/protein_tune_rl)
- [Reinforcement Learning Guides Generative Protein Language Models (arXiv 2412.12979)](https://arxiv.org/pdf/2412.12979)
- [ProteinZero (arXiv 2506.07459)](https://arxiv.org/html/2506.07459v5)
- [GermRL (bioRxiv 2026.06.08.730660)](https://www.biorxiv.org/content/10.64898/2026.06.08.730660v1.full)
- [Reinforcement Learning for Antibody Sequence Infilling (bioRxiv full PDF, alternate link)](https://www.biorxiv.org/content/10.1101/2025.08.08.669419.full.pdf)
- [GFlowNet vs GRPO for therapeutic peptides (bioRxiv 2026.01.05.697258)](https://www.biorxiv.org/content/10.64898/2026.01.05.697258v1.full.pdf)
- [PRX Life special collection on Machine Learning in Structural Biology](https://journals.aps.org/prxlife/edannounce/Submit-Your-Paper-to-%3Ci%3EPRX-Life%3C/i%3E%E2%80%99s-Special-Collection-on-Machine-Learning-in-Structural-Biology)
- [ISMB conference guide](https://casrai.org/guides/ismb)
- [ECCB 2026 call for papers](https://easychair.org/cfp/ECCB2026)
- [Conference listing post on X](https://x.com/pkoo562/status/2057144930322006446?lang=en)
- [DataBio conferences list](https://databio.org/conferences/)
- [Genomics conference list](https://conferenceagent.vercel.app/field/genomics/)
