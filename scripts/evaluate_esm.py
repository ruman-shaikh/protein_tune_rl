#!/usr/bin/env python3
"""Score antibody heavy chains and annotated regions with ESM-2."""

import argparse
from pathlib import Path

import pandas as pd
import torch

from protein_tune_rl.metrics.esm_scoring import ESMSequenceScoring


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--sequence-column", default="HC")
    parser.add_argument("--region-column", default="HCDR3")
    parser.add_argument("--model", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mask-batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def locate_unique_region(sequence, region):
    start = sequence.find(region)
    if start < 0:
        raise ValueError(f"region {region!r} is absent from the heavy chain")
    if sequence.find(region, start + 1) >= 0:
        raise ValueError(f"region {region!r} occurs more than once in the heavy chain")
    return start, start + len(region)


def main():
    args = parse_args()
    frame = pd.read_csv(args.input_csv)
    required = {args.sequence_column, args.region_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing CSV columns: {', '.join(sorted(missing))}")
    if args.limit is not None:
        frame = frame.head(args.limit).copy()

    scorer = ESMSequenceScoring(
        model_name=args.model,
        reduction="mean",
        mask_batch_size=args.mask_batch_size,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    full_scores, region_scores = [], []
    for sequence, region in zip(
        frame[args.sequence_column].astype(str),
        frame[args.region_column].astype(str),
    ):
        bounds = locate_unique_region(sequence, region)
        full_scores.append(scorer.score(sequence))
        region_scores.append(scorer.score(sequence, region=bounds))

    frame["esm2_full_mean_logp"] = full_scores
    frame["esm2_region_mean_logp"] = region_scores
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    print(f"scored {len(frame)} sequences on {args.device}; wrote {args.output_csv}")
    print(frame[["esm2_full_mean_logp", "esm2_region_mean_logp"]].describe())


if __name__ == "__main__":
    main()
