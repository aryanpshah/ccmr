#!/usr/bin/env python3
"""
Aggregate per-case metrics into Table 1 (macro) and Table 2 (per-structure).
"""
import argparse
from pathlib import Path

import pandas as pd

STRUCT = {1: "LV", 2: "RV", 3: "LA", 4: "RA", 5: "Aorta", 6: "PA", 7: "SVC", 8: "IVC"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csvs", type=str, nargs="+", required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--num_classes", type=int, default=8)
    args = ap.parse_args()

    import re
    dfs = []
    for p in args.input_csvs:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(path)
        m = re.search(r"L(\d+)", path.stem, re.I)
        budget = "L%s" % m.group(1) if m else path.stem.split("_")[0]
        df = pd.read_csv(path)
        df["budget"] = budget
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    macro = []
    for budget in sorted(df["budget"].unique(), key=lambda x: int(x[1:])):
        bdf = df[df["budget"] == budget]
        macro.append({"Budget": budget, "Dice_mean": bdf["fg_mean_dice"].mean(), "Dice_sd": bdf["fg_mean_dice"].std(), "HD95_mean_mm": bdf["fg_mean_hd95_mm"].mean(), "HD95_sd_mm": bdf["fg_mean_hd95_mm"].std()})
    pd.DataFrame(macro).to_csv(out_dir / "budget_macro_metrics.csv", index=False)

    struct_dice = []
    for k in range(1, args.num_classes + 1):
        row = {"Structure": STRUCT.get(k, "S%d" % k)}
        for budget in sorted(df["budget"].unique(), key=lambda x: int(x[1:])):
            row[budget] = df[df["budget"] == budget]["dice_%d" % k].mean()
        struct_dice.append(row)
    pd.DataFrame(struct_dice).to_csv(out_dir / "per_structure_dice_by_budget.csv", index=False)

    struct_hd95 = []
    for k in range(1, args.num_classes + 1):
        row = {"Structure": STRUCT.get(k, "S%d" % k)}
        for budget in sorted(df["budget"].unique(), key=lambda x: int(x[1:])):
            bdf = df[df["budget"] == budget]
            col = "hd95_%d_mm" % k
            if col in bdf.columns:
                row["HD95_%s_mm" % budget] = float(bdf[col].mean())
        struct_hd95.append(row)
    pd.DataFrame(struct_hd95).to_csv(out_dir / "per_structure_hd95_by_budget.csv", index=False)
    print("[OK] Table 1 and Table 2 ->", out_dir)
