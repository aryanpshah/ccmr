#!/usr/bin/env python3
import json
import math
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, distance_transform_edt

def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return 2.0 * inter / denom

def hd95(pred: np.ndarray, gt: np.ndarray, spacing_xyz) -> float:
    """
    95th percentile symmetric Hausdorff distance in mm.
    Returns:
      0.0 if both empty
      nan if one empty and the other not
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")

    structure = np.ones((3, 3, 3), dtype=bool)
    pred_surf = np.logical_xor(pred, binary_erosion(pred, structure=structure, iterations=1))
    gt_surf   = np.logical_xor(gt,   binary_erosion(gt,   structure=structure, iterations=1))

    # distance_transform_edt expects sampling in array axis order (z,y,x)
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])

    dt_gt = distance_transform_edt(~gt_surf, sampling=spacing_zyx)
    dt_pr = distance_transform_edt(~pred_surf, sampling=spacing_zyx)

    d_pred_to_gt = dt_gt[pred_surf]
    d_gt_to_pred = dt_pr[gt_surf]
    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred], axis=0)
    if all_d.size == 0:
        return float("nan")
    return float(np.percentile(all_d, 95))

def load_nii(path: Path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # z,y,x
    spacing = img.GetSpacing()         # x,y,z
    return arr, spacing

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", type=int, required=True)
    ap.add_argument("--pred-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default="outputs/metrics")
    ap.add_argument("--classes", type=int, default=8)
    args = ap.parse_args()

    # figure out raw path (prefer env var if set)
    nnunet_raw = Path(
        str(
            __import__("os").environ.get(
                "nnUNet_raw",
                "data/nnunet/nnUNet_raw"
            )
        )
    )

    ds = sorted(nnunet_raw.glob(f"Dataset{args.dataset_id}_*"))
    if not ds:
        raise RuntimeError(f"Could not find Dataset{args.dataset_id}_* under {nnunet_raw}")
    ds = ds[0]

    labelsTs = ds / "labelsTs"
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_files = sorted(labelsTs.glob("*.nii.gz"))
    if not gt_files:
        raise RuntimeError(f"No gt files in {labelsTs}")

    rows = []
    for gt_path in gt_files:
        cid = gt_path.name.replace(".nii.gz", "")
        pr_path = pred_dir / f"{cid}.nii.gz"
        if not pr_path.exists():
            raise RuntimeError(f"Missing prediction for {cid}: {pr_path}")

        gt, spacing = load_nii(gt_path)
        pr, _ = load_nii(pr_path)

        case = {"case_id": cid}
        fg_dices = []
        fg_hd = []

        for k in range(1, args.classes + 1):
            gt_k = (gt == k)
            pr_k = (pr == k)
            d = dice(pr_k, gt_k)
            h = hd95(pr_k, gt_k, spacing)

            case[f"dice_{k}"] = d
            case[f"hd95_{k}"] = h
            fg_dices.append(d)
            if not (isinstance(h, float) and math.isnan(h)):
                fg_hd.append(h)

        case["fg_mean_dice"] = float(np.mean(fg_dices)) if fg_dices else float("nan")
        case["fg_mean_hd95"] = float(np.mean(fg_hd)) if fg_hd else float("nan")
        case["all_background_pred"] = int(np.all(pr == 0))
        rows.append(case)

    summary = {
        "dataset_id": args.dataset_id,
        "n_test": len(rows),
        "classes": args.classes,
        "all_background_preds": int(sum(r["all_background_pred"] for r in rows)),
        "mean": {},
        "std": {},
    }

    for k in range(1, args.classes + 1):
        dvals = np.array([r[f"dice_{k}"] for r in rows], dtype=float)
        hvals = np.array([r[f"hd95_{k}"] for r in rows], dtype=float)
        summary["mean"][f"dice_{k}"] = float(np.nanmean(dvals))
        summary["std"][f"dice_{k}"]  = float(np.nanstd(dvals))
        summary["mean"][f"hd95_{k}"] = float(np.nanmean(hvals))
        summary["std"][f"hd95_{k}"]  = float(np.nanstd(hvals))

    fg_d = np.array([r["fg_mean_dice"] for r in rows], dtype=float)
    fg_h = np.array([r["fg_mean_hd95"] for r in rows], dtype=float)
    summary["mean"]["fg_mean_dice"] = float(np.nanmean(fg_d))
    summary["std"]["fg_mean_dice"]  = float(np.nanstd(fg_d))
    summary["mean"]["fg_mean_hd95"] = float(np.nanmean(fg_h))
    summary["std"]["fg_mean_hd95"]  = float(np.nanstd(fg_h))

    csv_path = out_dir / f"D{args.dataset_id}_per_case.csv"
    keys = (
        ["case_id"]
        + [f"dice_{k}" for k in range(1, args.classes + 1)]
        + [f"hd95_{k}" for k in range(1, args.classes + 1)]
        + ["fg_mean_dice", "fg_mean_hd95", "all_background_pred"]
    )

    with open(csv_path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

    summary_path = out_dir / f"D{args.dataset_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"[OK] {args.dataset_id}: wrote {csv_path}")
    print(f"[OK] {args.dataset_id}: wrote {summary_path}")
    print(f"[OK] {args.dataset_id}: fg_mean_dice={summary['mean']['fg_mean_dice']:.6f} +/- {summary['std']['fg_mean_dice']:.6f}")
    print(f"[OK] {args.dataset_id}: all_background_preds={summary['all_background_preds']}")

if __name__ == "__main__":
    main()
