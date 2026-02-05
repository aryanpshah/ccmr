#!/usr/bin/env python3
"""
Compute per-case, per-structure Dice and HD95 from predictions and ground truth.

Reproduces: Metrics for Table 1 and Table 2.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.io import read_nii
from utils.metrics import dice, hd95_mm

K = 8


def read_case_ids(path):
    if not path.exists():
        raise FileNotFoundError(f"Test IDs file not found: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", type=str, required=True)
    ap.add_argument("--pred_dir", type=str, required=True)
    ap.add_argument("--test_ids", type=str, required=True)
    ap.add_argument("--budget", type=str, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--data_root", type=str, default=".")
    ap.add_argument("--splits_dir", type=str, default="splits")
    ap.add_argument("--out_dir", type=str, default=".")
    ap.add_argument("--num_classes", type=int, default=8)
    args = ap.parse_args()

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    test_ids = read_case_ids(Path(args.test_ids))
    rows = []

    for cid in test_ids:
        gt_path = gt_dir / (cid + ".nii.gz")
        pred_path = pred_dir / (cid + ".nii.gz")
        if not gt_path.exists():
            gt_path = gt_dir / (cid + ".nii")
        if not pred_path.exists():
            pred_path = pred_dir / (cid + ".nii")
        if not gt_path.exists() or not pred_path.exists():
            raise FileNotFoundError(f"Missing GT or pred for {cid}")

        gt, spacing = read_nii(gt_path)
        pr, _ = read_nii(pred_path)
        dice_k = {}
        hd95_k = {}
        for k in range(1, args.num_classes + 1):
            mgt = (gt == k)
            mpr = (pr == k)
            dice_k[k] = float(dice(mgt, mpr))
            hd95_k[k] = float(hd95_mm(mgt, mpr, spacing))

        fg_dice = float(sum(dice_k.values()) / args.num_classes)
        hd_vals = [v for v in hd95_k.values() if not (isinstance(v, float) and str(v) == "nan")]
        fg_hd95 = float(sum(hd_vals) / len(hd_vals)) if hd_vals else float("nan")

        row = {"case": cid, "fg_mean_dice": fg_dice, "fg_mean_hd95_mm": fg_hd95}
        for k in range(1, args.num_classes + 1):
            row["dice_%d" % k] = dice_k[k]
            row["hd95_%d_mm" % k] = hd95_k[k]
        rows.append(row)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case", "fg_mean_dice", "fg_mean_hd95_mm"] + ["dice_%d" % k for k in range(1, args.num_classes + 1)] + ["hd95_%d_mm" % k for k in range(1, args.num_classes + 1)]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("[OK] Wrote", out_csv, len(rows), "cases")
