"""Regenerate the four per-method training plots from test_run data.

Reproduces notebook.ipynb's inspect_online_run() as a script: a 2x2 grid of
reward and KL (moving average) versus timestep and versus training samples,
one figure per method. Fixes the notebook's method-label bug (it derived the
label from the date prefix of the run directory) by mapping each run explicitly.

Usage: python make_plots.py [test_run_root]
Writes figures/{ppo,a2c,reinforce,explicit_kl}.pdf
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else "test_run"
BASE = os.path.join(ROOT, "online_rl_trainer", "ss_perc_sheet")

# run directory -> (method label, output stem)
RUNS = {
    "2026_01_12_04_18_59_online_rl_trainer_steps_2000_bs_8_lr_1e-05": ("PPO", "ppo"),
    "2026_01_12_04_31_00_online_rl_trainer_steps_2000_bs_8_lr_5e-06": ("A2C+GAE", "a2c"),
    "2026_01_12_04_36_22_online_rl_trainer_steps_2000_bs_8_lr_1e-05": (
        "REINFORCE",
        "reinforce",
    ),
    "2026_01_12_04_52_08_online_rl_trainer_steps_2000_bs_8_lr_1e-05": (
        "Explicit-KL REINFORCE",
        "explicit_kl",
    ),
}
MA_WINDOW = 200


def moving_avg(x, w):
    return np.convolve(x, np.ones(w) / w, mode="valid")


def make_plot(run_name, method, stem):
    df = pd.read_csv(os.path.join(BASE, run_name, "train_log.csv"))
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    reward, kl = df["reward_mean"].values, df["KL_mean"].values
    t = df.index.values
    x = df["num_samples"].values

    reward_ma, kl_ma = moving_avg(reward, MA_WINDOW), moving_avg(kl, MA_WINDOW)
    t_ma, x_ma = t[MA_WINDOW - 1 :], x[MA_WINDOW - 1 :]

    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    panels = [
        (axs[0, 0], t_ma, reward_ma, "Timestep", "Reward mean", "Reward vs Timestep"),
        (axs[0, 1], t_ma, kl_ma, "Timestep", "KL mean", "KL vs Timestep"),
        (axs[1, 0], x_ma, reward_ma, "Training samples", "Reward mean", "Reward vs Samples"),
        (axs[1, 1], x_ma, kl_ma, "Training samples", "KL mean", "KL vs Samples"),
    ]
    for ax, xs, ys, xl, yl, title in panels:
        ax.plot(xs, ys, linewidth=1.6)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"{method}: {title}")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{method} (200-step moving average)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs("figures", exist_ok=True)
    out = f"figures/{stem}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(
        f"{method:24s} -> {out}  "
        f"(final reward {reward[-1]:.3f}, final KL {kl[-1]:.4f})"
    )


if __name__ == "__main__":
    for run_name, (method, stem) in RUNS.items():
        make_plot(run_name, method, stem)
