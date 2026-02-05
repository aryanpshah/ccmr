#!/usr/bin/env python3
"""
Generate Figure: Per-case macro Dice distribution (Figure 4).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csvs", type=str, nargs="+",
                    default=["artifacts/L5_per_case.csv", "artifacts/L10_per_case.csv",
                             "artifacts/L20_per_case.csv", "artifacts/L40_per_case.csv"])
    ap.add_argument("--out_dir", type=str, default="artifacts")
    ap.add_argument("--data_root", type=str, default=".")
    ap.add_argument("--splits_dir", type=str, default="splits")
    args = ap.parse_args()

    print("Reproduces: Figure 4 (boxplot per-case macro Dice)")

    data = []
    labels = []
    for p in args.input_csvs:
        path = Path(p)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "fg_mean_dice" in df.columns:
            data.append(df["fg_mean_dice"].dropna().values)
        else:
            data.append([])
        labels.append(path.stem.split("_")[0])

    if not data:
        raise FileNotFoundError("No per-case CSVs found. Run compute_metrics.py first.")

    plt.figure()
    plt.boxplot(data, tick_labels=labels, showfliers=True)
    plt.xlabel("Label Budget")
    plt.ylabel("Per-Case Macro Dice")
    plt.grid(True, axis="y", alpha=0.3)
    plt.ylim(0, 1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / "boxplot_fg_dice.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("[OK]", out_dir / "boxplot_fg_dice.png")
