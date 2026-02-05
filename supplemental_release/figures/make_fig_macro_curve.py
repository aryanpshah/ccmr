#!/usr/bin/env python3
"""
Generate Figure: Macro Dice vs label budget (Figure 3).

Reproduces: Macro foreground Dice curve.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", type=str, default="artifacts/budget_macro_metrics.csv")
    ap.add_argument("--out_dir", type=str, default="artifacts")
    ap.add_argument("--data_root", type=str, default=".")
    ap.add_argument("--splits_dir", type=str, default="splits")
    args = ap.parse_args()

    print("Reproduces: Figure 3 (macro Dice vs budget)")

    path = Path(args.metrics_csv)
    if not path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {path}. Run aggregate_tables.py first.")
    df = pd.read_csv(path)
    df["Budget"] = df["Budget"].astype(str)
    budgets = [int(b.replace("L", "")) for b in df["Budget"]]
    budgets = sorted(budgets)
    means = [float(df[df["Budget"] == "L%d" % b]["Dice_mean"].iloc[0]) for b in budgets]
    sds = [float(df[df["Budget"] == "L%d" % b]["Dice_sd"].iloc[0]) for b in budgets]

    plt.figure()
    plt.errorbar(budgets, means, yerr=sds, marker="o", capsize=4)
    plt.xlabel("Label Budget (Number of Labeled Volumes)")
    plt.ylabel("Macro Foreground Dice")
    plt.xticks(budgets, [str(b) for b in budgets])
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dice_vs_budget.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("[OK]", out_path)
