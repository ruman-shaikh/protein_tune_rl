"""Timing pilot: measure generation throughput for P3/P4 feasibility.

Reuses the training-time sampler so numbers reflect the real code path.
Reports sequences/sec at several batch sizes, plus checkpoint load cost.
"""

import json
import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist

sys.path.insert(0, "/home/ruman/repos/protein_tune_rl")

from protein_tune_rl.protein_trainer.online_rl_trainer import OnlineRLSampler
from protein_tune_rl.protein_trainer.trainer import Trainer


class Sampler(Trainer, OnlineRLSampler):
    def __init__(self, config):
        Trainer.__init__(self, config)
        OnlineRLSampler.__init__(self, config)


def main():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "24733")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, rank=0, world_size=1,
                            timeout=timedelta(seconds=120))
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    cfg_path = "/home/ruman/repos/protein_tune_rl/configs/examples/a2c_iglm_hcdr3_beta_sheet.json"
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None

    results = {}
    for bs in (8, 32, 64):
        with open(cfg_path) as f:
            config = json.load(f)
        config["dataset"]["data_directory"] = (
            "/home/ruman/repos/protein_tune_rl/data/examples/eval_data_50_v3.csv"
        )
        config["trainer"]["batch_size"] = bs
        config.setdefault("KL_penalty", {"weight": 0.0, "target": None})
        if ckpt:
            config["checkpoint"] = ckpt

        t0 = time.time()
        s = Sampler(config)
        build_s = time.time() - t0

        loader = iter(s.dataloader)
        # warm-up batch (CUDA init, kernel autotune)
        batch = next(loader)
        with torch.no_grad():
            s._sample_batch(s.policy, s.collator(batch))

        n_batches, n_seqs, t_start = 0, 0, time.time()
        with torch.no_grad():
            for batch in loader:
                tok = s.collator(batch)
                out = s._sample_batch(s.policy, tok)
                n_seqs += len(out[0])
                n_batches += 1
                if n_batches >= 3:
                    break
        elapsed = time.time() - t_start

        if n_batches:
            per_batch = elapsed / n_batches
            rate = n_seqs / elapsed
            results[bs] = (per_batch, rate)
            print(f"bs={bs:3d}  build={build_s:5.1f}s  "
                  f"{per_batch*1000:7.0f} ms/batch  {rate:7.1f} seq/s  "
                  f"({n_seqs} seqs in {elapsed:.2f}s)")
        else:
            print(f"bs={bs:3d}  no full batches available")

        del s
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if torch.cuda.is_available():
        print(f"\npeak GPU alloc: {torch.cuda.max_memory_allocated()/1e6:.0f} MB")

    print("\n--- extrapolations (best observed rate) ---")
    if results:
        best_bs, (pb, rate) = max(results.items(), key=lambda kv: kv[1][1])
        print(f"best: bs={best_bs} at {rate:.1f} seq/s")
        for label, n in (
            ("P3 probe: 50 ctx x 32 samples = 1600 seqs", 1600),
            ("P4 one checkpoint: 50 ctx x 32 = 1600 seqs", 1600),
            ("P4 ladder: 10 ckpts x 1600 seqs", 16000),
            ("P4 full ladder: 41 ckpts x 1600 seqs", 41 * 1600),
        ):
            secs = n / rate
            print(f"  {label:48s} -> {secs/60:7.1f} min  ({secs/3600:.2f} h)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
