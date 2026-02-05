#!/usr/bin/env python3
"""
Generate Figure: Per-structure Dice and HD95 vs budget (Figure 6).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

STRUCT = {1: "LV", 2: "RV", 3: "LA", 4: "RA", 5: "Aorta", 6: "PA", 7: "SVC", 8: "IVC"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dice_csv", type=str, default="artifacts/per_structure_dice_by_budget.csv")
    ap.add_argument("--hd95_csv", type=str, default="artifacts/per_structure_hd95_by_budget.csv")
    ap.add_argument("--out_dir", type=str, default="artifacts")
    ap.add_argument("--data_root", type=str, default=".")
    ap.add_argument("--splits_dir", type=str, default="splits")
    args = ap.parse_args()

    print("Reproduces: Figure 6 (per-structure Dice and HD95)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dd = pd.read_csv(args.dice_csv) if Path(args.dice_csv).exists() else None
    hd = pd.read_csv(args.hd95_csv) if Path(args.hd95_csv).exists() else None

    if dd is not None:
        budget_cols = [c for c in dd.columns if c in ["L5", "L10", "L20", "L40"]]
        budgets = [5, 10, 20, 40]
        plt.figure()
        for _, row in dd.iterrows():
            ys = [row.get(c, float("nan")) for c in budget_cols]
            plt.plot(budgets[:len(ys)], ys, marker="o", label=row["Structure"])
        plt.xlabel("Label Budget")
        plt.ylabel("Mean Dice")
        plt.xticks(budgets, [str(b) for b in budgets])
        plt.legend(ncol=2, fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(out_dir / "per_class_dice_vs_budget.png", dpi=200, bbox_inches="tight")
        plt.close()
        print("[OK]", out_dir / "per_class_dice_vs_budget.png")

    if hd is not None:
        budget_cols = [c for c in hd.columns if "HD95" in c and "mm" in c]
        budgets = [5, 10, 20, 40]
        plt.figure()
        for _, row in hd.iterrows():
            ys = []
            for b in budgets:
                c = "HD95_L%d_mm" % b
                ys.append(row.get(c, float("nan")) if c in row else float("nan"))
            import math
            if any(not (isinstance(v, float) and math.isnan(v)) for v in ys):
                plt.plot(budgets, ys, marker="o", label=row["Structure"])
        plt.xlabel("Label Budget")
        plt.ylabel("Mean HD95 (mm)")
        plt.legend(ncol=2, fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "per_class_hd95_vs_budget.png", dpi=200, bbox_inches="tight")
        plt.close()
        print("[OK]", out_dir / "per_class_hd95_vs_budget.png")
