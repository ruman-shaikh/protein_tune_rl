"""Regenerate paper/figures/reward_curves.pdf from the short local comparison.

Reads the three train_log.csv files produced by a matched REINFORCE/PPO/A2C run
and plots the KL-inclusive training reward (20-step moving average).
Usage: python make_figure.py <run_root>
  where <run_root>/{reinforce,ppo,a2c}/**/train_log.csv exist.
"""
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

root = sys.argv[1] if len(sys.argv) > 1 else "run"
runs = {"REINFORCE": "reinforce", "PPO": "ppo", "A2C+GAE": "a2c"}

plt.figure(figsize=(7, 4.3))
for label, key in runs.items():
    matches = glob.glob(os.path.join(root, key, "**", "train_log.csv"), recursive=True)
    if not matches:
        continue
    d = pd.read_csv(matches[0])
    y = d["reward_mean"].rolling(20, min_periods=1).mean()
    plt.plot(d["num_samples"], y, label=label, linewidth=1.8, alpha=0.9)

plt.xlabel("training samples")
plt.ylabel("reward mean (KL-inclusive)")
plt.title(r"Online RL on the CDR-H3 $\beta$-sheet task (20-step moving average)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/reward_curves.pdf")
print("wrote figures/reward_curves.pdf")
