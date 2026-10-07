"""Held-out evaluation: generate infills with a trained policy and score them.

Produces the "test numbers" that complement the per-step training log. Reuses
the training-time sampler (OnlineRLSampler) so generation + reward scoring are
identical to training, then reports held-out reward statistics.

Usage (inside the container, GPU node):

    python hpc/evaluate.py \
        --config configs/examples/a2c_iglm_hcdr3_beta_sheet.json \
        --run-dir <timestamped run dir under test_run/...> \
        --eval-data data/examples/eval_data_50_v3.csv

Writes ``eval_metrics.csv`` (summary) and ``eval_predictions.csv`` (per-sequence)
into the run dir.
"""

import argparse
import copy
import glob
import json
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from protein_tune_rl.protein_trainer.online_rl_trainer import OnlineRLSampler
from protein_tune_rl.protein_trainer.trainer import Trainer


class HeldoutEvaluator(Trainer, OnlineRLSampler):
    """Load a checkpoint and score generations on a held-out dataset."""

    def __init__(self, config):
        Trainer.__init__(self, config)
        OnlineRLSampler.__init__(self, config)
        assert len(self.metrics) == 1, "single-metric evaluation only"
        self.metric = list(self.metrics.values())[0]

    @torch.no_grad()
    def evaluate(self):
        rows = []
        for batch in iter(self.dataloader):
            tokenized = self.collator(batch)
            sampled_seqs, *_ = self._sample_batch(self.policy, tokenized)
            for i, seq in enumerate(sampled_seqs):
                chains = {
                    "L": batch["LC"][i],
                    "H": seq,
                    "seq_pre_mask": tokenized["seq_pre_mask"][i],
                    "seq_post_mask": tokenized["seq_post_mask"][i],
                }
                rows.append(
                    {"reward": float(self.metric(chains)), "generated_H": seq}
                )
        return pd.DataFrame(rows)


def _latest_checkpoint(run_dir):
    ckpts = glob.glob(os.path.join(run_dir, "*.ckpt"))
    ckpts += glob.glob(os.path.join(run_dir, "*_model_step_*.bin"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found under {run_dir}")
    # Prefer the highest step number (…_s<step>.ckpt), else newest mtime.
    def step_of(p):
        base = os.path.basename(p)
        digits = "".join(c if c.isdigit() else " " for c in base).split()
        return int(digits[-1]) if digits else -1

    return max(ckpts, key=lambda p: (step_of(p), os.path.getmtime(p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True, help="timestamped training output dir")
    ap.add_argument("--eval-data", default="data/examples/eval_data_50_v3.csv")
    ap.add_argument("--checkpoint", default=None, help="override auto-detected ckpt")
    args = ap.parse_args()

    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "24601"
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, rank=0, world_size=1, timeout=timedelta(seconds=60))
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    with open(args.config) as f:
        config = json.load(f)

    ckpt = args.checkpoint or _latest_checkpoint(args.run_dir)
    eval_config = copy.deepcopy(config)
    eval_config["dataset"]["data_directory"] = args.eval_data
    eval_config["checkpoint"] = ckpt
    # No KL model / optimizer needed for pure generation-based evaluation.
    eval_config.setdefault("KL_penalty", {"weight": 0.0, "target": None})

    evaluator = HeldoutEvaluator(eval_config)
    df = evaluator.evaluate()

    summary = pd.DataFrame(
        {
            "checkpoint": [os.path.basename(ckpt)],
            "eval_data": [os.path.basename(args.eval_data)],
            "n": [len(df)],
            "reward_mean": [float(df["reward"].mean())],
            "reward_std": [float(df["reward"].std())],
            "reward_min": [float(df["reward"].min())],
            "reward_max": [float(df["reward"].max())],
        }
    )
    summary.to_csv(os.path.join(args.run_dir, "eval_metrics.csv"), index=False)
    df.to_csv(os.path.join(args.run_dir, "eval_predictions.csv"), index=False)
    print("Held-out eval:", summary.to_dict("records")[0])

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
