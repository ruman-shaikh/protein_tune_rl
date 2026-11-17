# Concepts: everything this project assumes you know

A primer for picking the project back up after time away. Starts from the
biology, ends at the findings. Nothing here is new work — it explains the
vocabulary that `Path_Forward.md` uses without defining.

---

## 1. The biology, in the amount we actually need

**Antibody.** A Y-shaped protein the immune system uses to recognise things.
The two arms each have a **heavy chain** (HC) and a **light chain** (LC). Our
data give both as amino-acid strings, ~115 and ~110 residues.

**Framework vs CDR.** Most of a chain is **framework** — structurally
conserved scaffolding, nearly the same between antibodies. Stuck into it are
six short **CDRs** (Complementarity-Determining Regions), three on each chain,
and these are what actually touch the target. They vary enormously between
antibodies.

**HCDR3** (= CDR-H3, heavy chain CDR 3) is the one we work on. It is the most
variable, the longest, and the dominant contributor to what an antibody binds.
In our data it averages **13.5 residues** and ranges 6–27.

Why that matters here: **HCDR3 length is itself functionally important.** An
antibody with a 5-residue HCDR3 is not a working antibody, however good its
other numbers look. That single fact is what makes one of our findings
damning rather than merely curious.

**Amino acids** are written as single letters — `A` alanine, `V` valine, `G`
glycine, and so on, 20 of them. A sequence like `ARDLFLCLGGLNHLDY` is a loop
read left to right.

**Secondary structure** is the local shape the backbone folds into. Three
categories matter:
- **β-strand** (`E` in DSSP notation) — extended, pairs up sideways into sheets
- **α-helix** (`H`) — coiled
- **loop/coil** (`-`) — neither

Antibody frameworks are largely β-sheet. The CDR loops sitting between them
are mostly loop, with strand at the anchor points where they join the
framework. *That geometric fact is the mechanism behind our second exploit.*

---

## 2. The model: IgLM

**IgLM** (Shuai, Ruffolo & Gray) is a GPT-2-style language model trained on
antibody sequences. Ours is **IgLM-S**, the small one: **1.4 M parameters,
6.6 MB**. Tiny by modern standards, which is why everything runs on a laptop.

**Infilling** is how it is used. Rather than generating a sequence left to
right, you hand it a sequence with a hole and it fills the hole:

```
input :  [HEAVY] [HUMAN] E V Q L ... [MASK] ... V T V S S [SEP]
output:                              ARDLFLCLGGLNHLDY
```

So the framework is **frozen context** and only HCDR3 is generated. The model
emits residues one at a time until it emits a **stop token**, or until it hits
`max_length` (22 in our configs).

Two consequences that come up constantly:
- The policy controls ~13 of ~225 residues. Anything measured over the *whole
  chain* is therefore **diluted** — the part under control can barely move it.
- The policy chooses its own **length**, by deciding when to stop. Length is an
  action, not a constant. Both of our exploits turn out to be length exploits.

---

## 3. Reinforcement learning, as applied here

Standard RL vocabulary, mapped onto this task:

| RL term | Here |
|---|---|
| **policy** | IgLM, the thing being fine-tuned |
| **state** | the masked framework plus whatever has been generated so far |
| **action** | the next residue (or the stop token) |
| **episode** | generating one complete HCDR3 |
| **reward** | one number scoring the finished antibody |
| **horizon** | 8–22 steps. Very short, by RL standards |

The reward arrives **only at the end** — you cannot score half a loop. This is
a **sparse terminal reward**, and it creates the **credit assignment** problem:
if the finished loop scored well, which of the 14 individual residue choices
deserves the credit?

**KL penalty / regularisation.** Left alone, RL will push the policy anywhere
that raises the reward, including off the distribution of real antibodies. So a
frozen copy of the original IgLM is kept as a **reference model**, and the
policy is penalised for diverging from it. The divergence measure is
**KL (Kullback–Leibler) divergence**, in nats; the penalty weight is **β**.

- β too high → the policy never moves
- β too low → the policy drifts into nonsense

Our historical runs reached **KL 18–91 nats**, which is enormous drift, and
several collapsed outright.

### The algorithms

**REINFORCE.** The simplest policy gradient. Push up the probability of
whole sequences that scored above average, push down those below. One number
per sequence; every residue in it gets the same credit. `baseline` = the
batch-mean reward, subtracted to reduce variance.

**PPO** (Proximal Policy Optimization). Adds a safety rail. It computes the
**importance ratio** — how much more likely the new policy is to produce this
sequence than the old one — and **clips** it to ±0.2 so no single update can
move the policy too far. Optionally learns a **critic** (a **value function**,
`StateValue`) to predict the expected reward from the prompt, used as a smarter
baseline than the batch mean.

**A2C** (Advantage Actor-Critic). The `new_algo` contribution. Learns a
**per-token** critic (`TokenValue`) rather than one value per sequence, so each
residue position gets its own estimate. Combines them with **GAE**
(Generalized Advantage Estimation) — a weighted blend of short- and long-range
estimates controlled by **γ** (discount) and **λ** — to produce a per-token
**advantage**: how much better this action was than expected.

**DPO / DRO** (Direct Preference / Reward Optimization). Offline methods:
learn from a fixed dataset of (prompt, response, score) rather than by
generating. Present in the repo, not central to our findings.

**Terminated vs truncated.** An episode that emits the stop token *ended*. One
cut off at `max_length` was *interrupted* and would have continued. These need
different handling in GAE — a real ending gets no future value, an
interruption gets a bootstrap estimate. The fork implements this distinction;
most protein-RL code does not.

---

## 4. The endpoints — the heart of the project

An **endpoint** (or **proxy**, or **reward**, or **oracle** depending on role)
is any number you compute to say how good a generated antibody is. The whole
project is about whether these numbers mean what they claim.

### Sequence-level (cheap, no folding)

**`ss_perc_sheet` — "β-sheet percentage".** The reward in the published work
we extend. It calls Biopython's `secondary_structure_fraction()`, which is
**not a structure prediction at all** — it is a fixed lookup that counts what
fraction of residues belong to the set **{V, I, Y, F, W, L, T}**. Seven of the
twenty amino acids. No folding, no coordinates, no geometry.

> **Version trap.** Before Biopython 1.82 this function returned
> `(sheet, turn, helix)` while its docs claimed `(helix, turn, sheet)`. The
> code takes index `[2]`. So on an old Biopython the same code silently scores
> **helix** propensity (the set `{E, M, A, L, K}`) instead. We pinned
> `biopython>=1.82` for exactly this reason.

**ESM-2 masked pseudo-log-likelihood.** ESM-2 is a large protein language
model (we use the 8M and 650M variants). To score a sequence you mask each
residue in turn and ask how confidently the model recovers it, then average.
Intended as an independent **plausibility** check: does this look like a real
protein?

> **It measures predictability, not plausibility.** Mask one residue in a run
> of 22 identical ones and the answer is free from its neighbours. So any
> repetitive sequence scores superbly. This is the reason it fails as a
> safeguard.

### Structure-level (requires folding)

**IgFold.** A fast antibody-specific structure predictor (Graylab). Takes the
heavy and light sequences, returns 3-D coordinates as a **PDB file**, in about
1.5 seconds. It also returns **pRMSD**, its own per-residue confidence
estimate.

**`folding_confidence`.** Derived from pRMSD as `1 / (1 + mean pRMSD)`. Higher
means the predictor is more sure. Note carefully: this measures **prediction
confidence**, not correctness, and certainly not function.

**SASA** (Solvent Accessible Surface Area). How much of the folded protein's
surface is exposed to solvent, via the Shrake–Rupley algorithm on the
predicted coordinates. The other reward used in the published work.

**DSSP.** The standard algorithm for reading secondary structure *off
coordinates* — it finds backbone hydrogen-bond patterns and labels each residue
`H`, `E` or `-`. This is the real thing that `ss_perc_sheet` only pretends to
be. We use **pydssp**, a pure-Python implementation, because the system `mkdssp`
binary needs root to install.

**`dssp_beta`** (we wrote this). Fold with IgFold → run DSSP → report the
β-strand content of the generated loop. Two modes:
- `mode="fraction"` — the *proportion* of loop residues that are strand
- `mode="count"` — the *number* of strand residues

That distinction turned out to matter enormously; see §6.

### Not available to us

**ESMFold / AlphaFold** need ≥16 GB of GPU memory. The laptop has 6 GB. So the
general-purpose structure predictors are out, and IgFold is the only folding
option. **Binding affinity, expression, immunogenicity** need experimental
labels or a wet lab; our data have neither, plus no antigens and no
antibody–antigen complexes. **No functional claim is reachable from this
project**, at any effort level.

---

## 5. Normalisation, and why σ keeps appearing

Raw endpoint values have wildly different units — a fraction in [0,1], a SASA
in the thousands of Å². So each is **normalised**: `(value − mean) / std`,
where mean and std come from native antibodies. A reward of `+3.3` means *3.3
standard deviations above the natural distribution*.

This makes endpoints comparable, which is what lets us say "ESM-2 is gameable
by +6.6σ, SASA by +3.0σ" in one table.

**Caveat worth remembering:** normalisation constants are measured at native
lengths. Once the policy generates 5-residue or 22-residue loops, it is far
outside that distribution, so the normalised magnitudes stop being strictly
comparable. The *direction* and the *length collapse* remain solid; precise
reward values at the extremes do not.

---

## 6. What we actually found

**Finding 1 — the reward is a composition exploit.**
`"V" × 22`, typed by hand with no model, scores **3.338**. A2C after 82,000
training steps scores **3.366**. The policy pins length at the cap (22.0) and
fills it with the seven counted residues (99.9%). It is padding, not design.

**Finding 2 — it moves against the thing it is named after.**
Folding 184 structures and running DSSP: Spearman **ρ = −0.339, p = 2.5×10⁻⁶**.
Maximising "β-sheet" *reduces* real predicted β-strand in the loop by ~28%.

**Finding 3 — it is a class, not one bad choice.** Five endpoints attacked with
eight hand-written strings: ESM-2 PLL gameable by +6.6σ, composition-region
+5.2σ, SASA +3.0σ, composition-whole-chain +2.4σ. IgFold confidence and
`dssp_beta` resisted all eight. Crucially, **three different mechanisms** —
poly-valine beats β-sheet, poly-proline beats SASA, any homopolymer beats
ESM-2. Poly-glycine *raises* ESM-2 by ~5.8σ while *lowering* the β-sheet reward
by 1.6σ, so these are genuinely independent.

**Finding 4 — the safeguards fail.** ESM-2 plausibility prefers the exploit
(confirmed at both 8M and 650M, so not a toy-model artifact). Sequence
diversity stays at 94–100% unique in every failure, because a
composition-only reward has a vast flat optimum rather than a peak — so there
is no pressure to collapse onto one sequence, and a diversity dashboard reads
healthy throughout.

**Finding 5 — the structural fix also fails, by a new mechanism.**
Training on `dssp_beta` (fraction mode) collapsed HCDR3 from 13.6 to **5.0
residues**. The generated loops are **exact prefixes of base IgLM's own
outputs** — `ARGGG` from `ARGGGSSWYFDG`. The policy is not finding a special
motif; it emits the natural germline opening (`AR`/`AK`) and **fires the stop
token early**. Because β-strand *fraction* is a ratio over the generated span,
a short stub flush against the framework inherits the framework's own strand
and the ratio lands high. It deletes the loop rather than build a sheet.

This is why **hand-written attack screens are a lower bound**: we screened this
endpoint with eight attacks, it passed all of them, and RL found something we
had not imagined. Our own `minimal` attack even scored −3.6σ here, because we
built it from the native first two residues; RL found the short sequences that
*do* score.

**Finding 6 — the detector.** Five signals from the policy's own generations,
no oracle needed: length pinning or collapse, length variance collapse,
composition concentration, output determinism, and **reward quantisation**
(our own: only 4 distinct reward values across 144 generations from 81,891
different antibodies means the output has gone context-independent). Validated
three ways — silent on untrained IgLM, fires on both exploits, through
*complementary* signals each time. And it fires at **step 250 of 2,000**, with
87% of the compute unspent.

**One retraction, recorded deliberately.** We measured a 29% error in PPO's
importance ratio and called it a mis-implemented baseline. Re-measuring on
*base* IgLM rather than the degenerate trained policy gave **0.08%**. The
strong claim was withdrawn. What survives is narrower: the bug's severity
scales with policy drift, so it is invisible early and severe exactly when the
policy is already going wrong.

---

## 7. Mitigations, and the open question

**Count instead of fraction.** If a ratio invites truncation, score the
*number* of strand residues. A 5-residue stub then caps at 5, below the ~5.3
that a native-length loop already provides. Measured: the exploit `ARGGG` goes
from **+1.676** (fraction) to **−1.959** (count), while native stays positive.

**But the inverse failure is obvious**, and we are testing rather than
assuming it:

> **Prediction:** ratio endpoints invite truncation; count endpoints invite
> elongation. Neither is length-neutral.

A 22-residue span that genuinely forms strand would score ~+14σ under count.
If that is what the count run finds, the paper gains a clean symmetry — both
naive normalisations of the *same* structural measurement are gameable, in
opposite directions — and the remaining candidate is explicit
**length-matching**: penalise deviation from native loop length.

---

## 8. Where things live

| Path | What |
|---|---|
| `experiments_for_publication/Path_Forward.md` | the plan, all measured results, checkpoint status |
| `experiments_for_publication/CONCEPTS.md` | this file |
| `experiments_for_publication/scripts/` | the experiment scripts (C2, C3, C4a/b, C6′) |
| `experiments_for_publication/scripts/gates/` | the original viability gates, preserved as run |
| `experiments_for_publication/results/` | committed result CSVs |
| `protein_tune_rl/metrics/` | the endpoints: `ss_perc_sheet`, `sasa`, `folding_confidence`, `dssp_beta`, `esm_scoring` |
| `protein_tune_rl/optimizer/` | `reinforce.py`, `ppo.py`, `a2c.py`, `dpo.py`, `dro.py` |
| `protein_tune_rl/protein_trainer/online_rl_trainer.py` | the training loop, sampler, KL penalty |
| `configs/experiments/` | the C4′ run configs |
| `test_run/` | run outputs and checkpoints — **gitignored, laptop-only** |

**The one operational thing to remember:** `test_run/` is not in git. Result
CSVs are committed; the 35 MB checkpoints are not. Copy a `train_log.csv` into
`results/` before pausing a run if you want its history preserved.
