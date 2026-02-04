#!/usr/bin/env python3
import json, csv
from pathlib import Path
import matplotlib.pyplot as plt

BUDGETS = [(5,985),(10,986),(20,987),(40,988)]
K = 8


STRUCT_NAMES = {1:"LV",2:"RV",3:"LA",4:"RA",5:"Aorta",6:"PA",7:"SVC",8:"IVC"}
def load_summary(did: int):
    return json.loads((Path("outputs/metrics")/f"D{did}_summary.json").read_text())

def load_per_case_fg_dice(did: int):
    p = Path("outputs/metrics")/f"D{did}_per_case.csv"
    vals = []
    with p.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            vals.append(float(row["fg_mean_dice"]))
    return vals

def save(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    out_plots = Path("outputs/plots")
    out_fr = Path("outputs/final_report")
    out_plots.mkdir(parents=True, exist_ok=True)
    out_fr.mkdir(parents=True, exist_ok=True)

    budgets = [b for b,_ in BUDGETS]
    sums = {did: load_summary(did) for _,did in BUDGETS}

    # --- HD95 vs Budget (macro FG) ---
    hd_mean = []
    hd_sd = []
    for b,did in BUDGETS:
        hd_mean.append(sums[did]["mean"]["fg_mean_hd95_mm"])
        hd_sd.append(sums[did]["std"]["fg_mean_hd95_mm"])

    plt.figure()
    plt.errorbar(budgets, hd_mean, yerr=hd_sd, marker="o", capsize=4)
    plt.title("Foreground Mean HD95 vs Label Budget")
    plt.xlabel("Label Budget (Number of Labeled Volumes)")
    plt.ylabel("Foreground Mean HD95 (mm, Macro-Averaged Across Structures)")
    plt.xticks(budgets, [str(b) for b in budgets])
    plt.grid(True, alpha=0.3)
    save(out_plots/"hd95_vs_budget.png")

    # --- Per-Structure HD95 vs Budget ---
    plt.figure()
    for k in range(1, K+1):
        ys = [sums[did]["mean"][f"hd95_{k}_mm"] for _,did in BUDGETS]
        plt.plot(budgets, ys, marker="o", label=f"{STRUCT_NAMES.get(k, f'Structure {k}')}")
    plt.title("Per-Structure HD95 vs Label Budget")
    plt.xlabel("Label Budget (Number of Labeled Volumes)")
    plt.ylabel("Mean HD95 (mm)")
    plt.xticks(budgets, [str(b) for b in budgets])
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=8)
    save(out_plots/"per_class_hd95_vs_budget.png")

    # --- Boxplot: per-case FG mean Dice ---
    data = []
    labels = []
    for b,did in BUDGETS:
        data.append(load_per_case_fg_dice(did))
        labels.append(str(b))

    plt.figure()
    plt.boxplot(data, tick_labels=labels, showfliers=True)
    plt.title("Per-Case Foreground Mean Dice Distribution")
    plt.xlabel("Label Budget (Number of Labeled Volumes)")
    plt.ylabel("Foreground Mean Dice")
    plt.grid(True, axis="y", alpha=0.3)
    save(out_plots/"boxplot_fg_dice.png")

    # Copy refreshed plots into final_report (overwrite)
    for name in ["per_class_hd95_vs_budget.png", "boxplot_fg_dice.png"]:
        src = out_plots/name
        (out_fr/name).write_bytes(src.read_bytes())

    print("[OK] Wrote prettified HD95 + boxplot plots to outputs/plots and outputs/final_report")

if __name__ == "__main__":
    main()
