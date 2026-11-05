# ESM-2 and PDB evaluation

ProteinTuneRL provides two evaluation paths that are independent of the reward
used for RL training. ESM-2 estimates sequence plausibility, while the PDB
evaluator compares a predicted structure with a known reference structure.

## ESM-2 sequence scoring

`ESMSequenceScoring` computes masked pseudo-log-likelihood. Each selected amino
acid is masked and its log probability under ESM-2 is recorded. The mean score
is length-normalized; a larger, less negative value indicates that the sequence
is more compatible with the model's learned protein distribution. It is not an
experimental binding-affinity or developability measurement.

Run a held-out CSV evaluation with:

```bash
python scripts/evaluate_esm.py \
  data/examples/eval_data_50_v3.csv results/esm2_eval.csv \
  --device cuda --local-files-only
```

The input must contain `HC` and `HCDR3` columns by default. The command reports
both whole-heavy-chain and HCDR3 mean log probabilities. Use `--limit 5` for a
short smoke run. Omit `--local-files-only` when the requested Hugging Face model
is not already cached.

The metric factory also accepts `esm_scoring`:

```json
{
  "name": "esm_scoring",
  "params": {
    "model_name": "facebook/esm2_t6_8M_UR50D",
    "region_only": true,
    "reduction": "mean",
    "mask_batch_size": 32
  }
}
```

Pseudo-log-likelihood requires one masked prediction per scored residue. Masks
are evaluated in batches, but this metric is still substantially more expensive
than a single model forward pass. It is most practical as an offline evaluator.

## PDB structure comparison

Compare a predicted heavy-chain structure with a reference PDB using:

```bash
python scripts/compare_pdb_structures.py reference.pdb predicted.pdb \
  --reference-region 95:107 --predicted-region 95:107 \
  --reference-chain H --predicted-chain H --output comparison.json
```

Region bounds are one-based, inclusive positions in the selected chain after
non-amino-acid records are excluded. They are sequence positions, not PDB
residue identifiers. The evaluator aligns the structures on framework backbone
atoms and reports framework RMSD, region RMSD, region-length difference, SASA,
and predicted-region mean B-factor. Region RMSD is undefined when region lengths
differ. A B-factor can be read as pLDDT only when the structure producer stores
pLDDT in that field.

## Reference-data limitation

The bundled CSV files do not contain PDB identifiers, antigen chains, native
antibody-antigen complexes, or experimental affinities. They cannot support a
ground-truth structural or docking benchmark. PDB comparison requires a matched
external reference structure. Docking remains outside the current scope.
