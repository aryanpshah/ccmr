#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

BUDGETS = [(5, 985), (10, 986), (20, 987), (40, 988)]
CLASSES = 8

def load_summary(did: int):
    p = Path("outputs/metrics") / f"D{did}_summary.json"
    return json.loads(p.read_text())

def load_per_case_fg(did: int):
    p = Path("outputs/metrics") / f"D{did}_per_case.csv"
    lines = p.read_text().strip().splitlines()
    hdr = lines[0].split(",")
    idx = hdr.index("fg_mean_dice")
    vals = []
    for line in lines[1:]:
        vals.append(float(line.split(",")[idx]))
    return np.array(vals, dtype=float)

def main():
    outdir = Path("outputs/plots")
    outdir.mkdir(parents=True, exist_ok=True)

    budgets = [b for b, _ in BUDGETS]

    # 1) FG mean Dice vs budget
    fg_mean = []
    fg_std = []
    for b, did in BUDGETS:
        s = load_summary(did)
        fg_mean.append(s["mean"]["fg_mean_dice"])
        fg_std.append(s["std"]["fg_mean_dice"])

    plt.figure()
    plt.plot(budgets, fg_mean, marker="o")
    plt.xlabel("Number of labeled volumes (L)")
    plt.ylabel("Foreground mean Dice (1..8)")
    plt.title("Dice vs label budget")
    plt.savefig(outdir / "dice_vs_budget.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 2) Per-class Dice vs budget
    plt.figure()
    for k in range(1, CLASSES + 1):
        ys = []
        for b, did in BUDGETS:
            s = load_summary(did)
            ys.append(s["mean"][f"dice_{k}"])
        plt.plot(budgets, ys, marker="o", label=f"class{k}")
    plt.xlabel("Number of labeled volumes (L)")
    plt.ylabel("Mean Dice")
    plt.title("Per-structure Dice vs label budget")
    plt.legend(ncol=2, fontsize=8)
    plt.savefig(outdir / "per_class_dice_vs_budget.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 3) Per-class HD95 vs budget
    plt.figure()
    for k in range(1, CLASSES + 1):
        ys = []
        for b, did in BUDGETS:
            s = load_summary(did)
            ys.append(s["mean"][f"hd95_{k}"])
        plt.plot(budgets, ys, marker="o", label=f"class{k}")
    plt.xlabel("Number of labeled volumes (L)")
    plt.ylabel("Mean HD95 (mm)")
    plt.title("Per-structure HD95 vs label budget")
    plt.legend(ncol=2, fontsize=8)
    plt.savefig(outdir / "per_class_hd95_vs_budget.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 4) Per-case FG Dice distribution (boxplot)
    data = [load_per_case_fg(did) for _, did in BUDGETS]
    plt.figure()
    plt.boxplot(data, labels=[str(b) for b, _ in BUDGETS], showfliers=True)
    plt.xlabel("Label budget (L)")
    plt.ylabel("Per-case foreground mean Dice")
    plt.title("Per-case Dice distribution vs label budget")
    plt.savefig(outdir / "boxplot_fg_dice.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("[OK] wrote plots to outputs/plots")

if __name__ == "__main__":
    main()
