"""Aggregate training + held-out evaluation numbers across methods.

Walks a results root (default ``test_run/``), collects each run's training curve
(``train_log.csv``) and held-out metrics (``eval_metrics.csv``), and writes:
  * ``summary.csv``          -- one row per run (method, final train reward, eval)
  * ``training_curves.png``  -- reward vs samples, one line per method
  * ``report.md``            -- human-readable comparison table

Usage:  python hpc/make_report.py --root test_run --out report
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _md_table(df):
    """Render a DataFrame as a markdown table without external deps."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df.iterrows():
        cells = []
        for v in r:
            cells.append(f"{v:.4g}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _method_name(run_dir):
    # Checkpoint tag (<method>_s<step>.ckpt) is the most reliable signal, since
    # tune.py pops optimizer.name out of the saved config.json.
    ckpts = glob.glob(os.path.join(run_dir, "*_s*.ckpt"))
    tag = None
    if ckpts:
        base = os.path.basename(sorted(ckpts)[0])
        tag = base.split("_s")[0]
    kl = None
    for cfg_name in ("config_used.json", "config.json"):
        path = os.path.join(run_dir, cfg_name)
        if os.path.exists(path):
            c = json.load(open(path))
            kl = c.get("KL_penalty", {}).get("weight")
            if tag is None:
                tag = c.get("optimizer", {}).get("name") or c.get("trainer", {}).get(
                    "name"
                )
            break
    if tag is None:
        tag = os.path.basename(run_dir)
    if tag == "reinforce" and kl is not None:
        return f"reinforce (KL={kl})"
    return tag


def _find_runs(root):
    logs = glob.glob(os.path.join(root, "**", "train_log.csv"), recursive=True)
    logs += glob.glob(os.path.join(root, "**", "*_trainer_log.csv"), recursive=True)
    return sorted({os.path.dirname(p) for p in logs})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="test_run")
    ap.add_argument("--out", default="report")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    plt.figure(figsize=(8, 5))
    for run_dir in _find_runs(args.root):
        method = _method_name(run_dir)
        row = {"method": method, "run_dir": run_dir}

        log_path = next(
            (
                p
                for p in (
                    os.path.join(run_dir, "train_log.csv"),
                    *glob.glob(os.path.join(run_dir, "*_trainer_log.csv")),
                )
                if os.path.exists(p)
            ),
            None,
        )
        if log_path:
            log = pd.read_csv(log_path)
            ycol = "reward_mean" if "reward_mean" in log else log.columns[-1]
            xcol = "num_samples" if "num_samples" in log else log.index.name or None
            x = log[xcol] if xcol in log else range(len(log))
            row["train_steps"] = len(log)
            row["final_train_reward"] = float(log[ycol].iloc[-1])
            row["best_train_reward"] = float(log[ycol].max())
            plt.plot(list(x), log[ycol], label=method, alpha=0.85)

        eval_path = os.path.join(run_dir, "eval_metrics.csv")
        if os.path.exists(eval_path):
            ev = pd.read_csv(eval_path).iloc[0]
            row["eval_reward_mean"] = float(ev["reward_mean"])
            row["eval_reward_std"] = float(ev["reward_std"])
            row["eval_n"] = int(ev["n"])
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(args.out, "summary.csv")
    summary.to_csv(summary_path, index=False)

    plt.xlabel("training samples")
    plt.ylabel("reward mean")
    plt.title("Training reward by method")
    plt.legend()
    plt.grid(alpha=0.3)
    plot_path = os.path.join(args.out, "training_curves.png")
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")

    report_cols = [
        c
        for c in (
            "method",
            "train_steps",
            "final_train_reward",
            "best_train_reward",
            "eval_reward_mean",
            "eval_reward_std",
            "eval_n",
        )
        if c in summary.columns
    ]
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write("# ProteinTuneRL run report\n\n")
        f.write(f"Runs found under `{args.root}`: {len(summary)}\n\n")
        if report_cols:
            f.write(_md_table(summary[report_cols]))
            f.write("\n\n")
        f.write("![training curves](training_curves.png)\n")

    print(f"Wrote {summary_path}, {plot_path}, and report.md ({len(summary)} runs)")


if __name__ == "__main__":
    main()
