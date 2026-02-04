import json
from pathlib import Path
import matplotlib.pyplot as plt

# (label budget, dataset id)
BUDGETS = [(5, 985), (10, 986), (20, 987), (40, 988)]
K = 8

STRUCT_NAMES = {
    1: "LV", 2: "RV", 3: "LA", 4: "RA",
    5: "Aorta", 6: "PA", 7: "SVC", 8: "IVC"
}

def load_summary(did: int) -> dict:
    p = Path("outputs/metrics") / f"D{did}_summary.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {p}. If this pod does not have outputs/metrics/, run:\n"
            f"  python scripts/recompute_metrics_from_preds.py\n"
        )
    return json.loads(p.read_text())

def pick_key(d: dict, candidates: list[str]) -> str:
    for c in candidates:
        if c in d:
            return c
    raise KeyError(f"None of these keys exist: {candidates}. Sample keys: {list(d.keys())[:40]}")

def main():
    out_plots = Path("outputs/plots")
    out_final = Path("outputs/final_report")
    out_plots.mkdir(parents=True, exist_ok=True)
    out_final.mkdir(parents=True, exist_ok=True)

    sums = {did: load_summary(did) for _, did in BUDGETS}
    budgets = [b for b, _ in BUDGETS]

    # Determine the per-class Dice key pattern (robust)
    mean0 = sums[BUDGETS[0][1]]["mean"]
    dice_key = {}
    for k in range(1, K + 1):
        dice_key[k] = pick_key(mean0, [
            f"dice_{k}",
            f"mean_dice_{k}",
            f"class{k}_dice",
            f"dice_class_{k}",
        ])

    # Plot
    plt.figure()
    for k in range(1, K + 1):
        y = [sums[did]["mean"][dice_key[k]] for _, did in BUDGETS]
        plt.plot(budgets, y, marker="o", label=STRUCT_NAMES.get(k, f"Structure {k}"))

    plt.xticks(budgets, [str(b) for b in budgets])
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.title("Per-Structure Dice vs Label Budget")
    plt.xlabel("Number of Labeled Training Volumes")
    plt.ylabel("Dice Similarity Coefficient (DSC)")
    plt.legend(ncol=2, fontsize=9)

    for outdir in [out_plots, out_final]:
        (outdir / "per_class_dice_vs_budget.png").parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outdir / "per_class_dice_vs_budget.png", dpi=200, bbox_inches="tight")

    print("[OK] Wrote per_class_dice_vs_budget.png to outputs/plots and outputs/final_report")

if __name__ == "__main__":
    main()
