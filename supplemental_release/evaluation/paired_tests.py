#!/usr/bin/env python3
"""
Paired statistical tests (t-test, Wilcoxon) on per-case macro Dice.

Reproduces: Statistical comparisons between adjacent budgets.

Usage:
    python paired_tests.py --input_csvs artifacts/L5_per_case.csv ... --out_txt artifacts/stat_tests.txt \\
        --data_root . --splits_dir splits --out_dir .
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


def load_fg_dice(path: Path) -> dict:
    """Load case_id -> fg_mean_dice."""
    lines = path.read_text().strip().splitlines()
    if not lines:
        return {}
    hdr = lines[0].split(",")
    idx = hdr.index("fg_mean_dice") if "fg_mean_dice" in hdr else 1
    case_idx = hdr.index("case") if "case" in hdr else 0
    out = {}
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) > max(idx, case_idx):
            out[parts[case_idx]] = float(parts[idx])
    return out


def main():
    ap = argparse.ArgumentParser(description="Paired statistical tests on macro Dice")
    ap.add_argument("--input_csvs", type=str, nargs="+", required=True,
                    help="Per-case CSV files (L5, L10, L20, L40 order)")
    ap.add_argument("--out_txt", type=str, default="artifacts/stat_tests_fg_dice.txt")
    ap.add_argument("--data_root", type=str, default=".")
    ap.add_argument("--splits_dir", type=str, default="splits")
    ap.add_argument("--out_dir", type=str, default=".")
    args = ap.parse_args()

    print("Reproduces: Paired t-test and Wilcoxon on per-case macro Dice")

    data = {}
    for p in args.input_csvs:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Input not found: {path}")
        budget = path.stem.split("_")[0]
        data[budget] = load_fg_dice(path)

    common = set(data[list(data.keys())[0]])
    for k in data:
        common &= set(data[k])
    common = sorted(common)
    n = len(common)
    if n == 0:
        raise RuntimeError("No common cases across budgets")

    def vec(label):
        return np.array([data[label][c] for c in common], dtype=float)

    comparisons = [("L5", "L10"), ("L10", "L20"), ("L20", "L40"), ("L5", "L40")]
    lines = ["comparison, mean_improvement, paired_t_p, wilcoxon_p (H1: second > first)", f"n_cases: {n}"]

    for a, b in comparisons:
        if a not in data or b not in data:
            continue
        x, y = vec(a), vec(b)
        td = ttest_rel(y, x, nan_policy="omit")
        try:
            wd = wilcoxon(y, x, alternative="greater")
            wp = wd.pvalue
        except Exception:
            wp = float("nan")
        diff = float(np.mean(y - x))
        lines.append(f"{a}->{b}, {diff:.4f}, {td.pvalue:.4g}, {wp:.4g}")

    out_path = Path(args.out_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[OK] Wrote {out_path}")
