#!/usr/bin/env python3
import json, csv
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, distance_transform_edt

K = 8  # foreground classes 1..8

def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return 2.0 * inter / denom

def surface(mask: np.ndarray) -> np.ndarray:
    # 1-voxel thick surface
    if mask.sum() == 0:
        return mask
    er = binary_erosion(mask, iterations=1, border_value=0)
    return np.logical_and(mask, np.logical_not(er))

def hd95_mm(gt: np.ndarray, pr: np.ndarray, spacing_xyz_mm) -> float:
    # Returns NaN if one empty and the other non-empty
    gt = gt.astype(bool); pr = pr.astype(bool)
    if gt.sum() == 0 and pr.sum() == 0:
        return 0.0
    if gt.sum() == 0 or pr.sum() == 0:
        return float("nan")

    s_gt = surface(gt)
    s_pr = surface(pr)

    # SciPy expects spacing in array axis order (z,y,x). SITK spacing is (x,y,z).
    sp_zyx = (spacing_xyz_mm[2], spacing_xyz_mm[1], spacing_xyz_mm[0])

    # distance from every voxel to nearest surface voxel
    dt_gt = distance_transform_edt(~s_gt, sampling=sp_zyx)
    dt_pr = distance_transform_edt(~s_pr, sampling=sp_zyx)

    d1 = dt_gt[s_pr]
    d2 = dt_pr[s_gt]
    if d1.size == 0 or d2.size == 0:
        return float("nan")
    all_d = np.concatenate([d1, d2])
    return float(np.percentile(all_d, 95))

def find_dataset_dir(did: int) -> Path:
    raw_root = Path("data/nnunet/nnUNet_raw")
    matches = sorted(raw_root.glob(f"Dataset{did}_*"))
    if not matches:
        raise FileNotFoundError(f"Could not find Dataset{did}_* under {raw_root}")
    return matches[0]

def main():
    out_metrics = Path("outputs/metrics")
    out_metrics.mkdir(parents=True, exist_ok=True)

    pairs = [(5,985),(10,986),(20,987),(40,988)]

    for L, did in pairs:
        ds = find_dataset_dir(did)
        gt_dir = ds / "labelsTs"
        pred_dir = Path(f"outputs/test_preds/D{did}_3d_fullres")
        if not gt_dir.exists():
            raise FileNotFoundError(f"Missing {gt_dir}")
        if not pred_dir.exists():
            raise FileNotFoundError(f"Missing {pred_dir}")

        gts = {p.name: p for p in gt_dir.glob("*.nii.gz")}
        prs = {p.name: p for p in pred_dir.glob("*.nii.gz")}
        common = sorted(set(gts.keys()) & set(prs.keys()))
        if not common:
            raise RuntimeError(f"No matching filenames between {gt_dir} and {pred_dir}")

        rows = []
        all_bg = 0

        for name in common:
            gt_img = sitk.ReadImage(str(gts[name]))
            pr_img = sitk.ReadImage(str(prs[name]))

            gt = sitk.GetArrayFromImage(gt_img).astype(np.int16)  # z,y,x
            pr = sitk.GetArrayFromImage(pr_img).astype(np.int16)

            if int((pr > 0).sum()) == 0:
                all_bg += 1

            spacing_xyz = gt_img.GetSpacing()  # (x,y,z) mm

            dice_k = {}
            hd95_k = {}
            for k in range(1, K+1):
                mgt = (gt == k)
                mpr = (pr == k)
                dice_k[k] = float(dice(mgt, mpr))
                hd95_k[k] = float(hd95_mm(mgt, mpr, spacing_xyz))

            # macro across 1..8 (ignore NaNs for HD95)
            dice_vals = [dice_k[k] for k in range(1, K+1)]
            hd_vals = [hd95_k[k] for k in range(1, K+1) if not np.isnan(hd95_k[k])]

            fg_mean_dice = float(np.mean(dice_vals))
            fg_mean_hd95 = float(np.mean(hd_vals)) if len(hd_vals) else float("nan")

            row = {
                "case": name.replace(".nii.gz",""),
                "fg_mean_dice": fg_mean_dice,
                "fg_mean_hd95_mm": fg_mean_hd95,
            }
            for k in range(1, K+1):
                row[f"dice_{k}"] = dice_k[k]
                row[f"hd95_{k}_mm"] = hd95_k[k]
            rows.append(row)

        # write per-case CSV
        per_case_csv = out_metrics / f"D{did}_per_case.csv"
        fields = ["case","fg_mean_dice","fg_mean_hd95_mm"] + \
                 [f"dice_{k}" for k in range(1,K+1)] + \
                 [f"hd95_{k}_mm" for k in range(1,K+1)]
        with per_case_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)

        # summary stats (mean/std across cases)
        def col(name):
            vals = [r[name] for r in rows]
            arr = np.array(vals, dtype=float)
            return arr

        summary = {"mean": {}, "std": {}, "n_test": len(rows), "all_background_preds": int(all_bg)}

        # fg macro
        for key in ["fg_mean_dice", "fg_mean_hd95_mm"]:
            arr = col(key)
            summary["mean"][key] = float(np.nanmean(arr))
            summary["std"][key] = float(np.nanstd(arr, ddof=1))

        # per-class
        for k in range(1, K+1):
            dk = col(f"dice_{k}")
            hk = col(f"hd95_{k}_mm")
            summary["mean"][f"dice_{k}"] = float(np.nanmean(dk))
            summary["std"][f"dice_{k}"] = float(np.nanstd(dk, ddof=1))
            summary["mean"][f"hd95_{k}_mm"] = float(np.nanmean(hk))
            summary["std"][f"hd95_{k}_mm"] = float(np.nanstd(hk, ddof=1))

        (out_metrics / f"D{did}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[OK] D{did} (L={L}): wrote {per_case_csv} and D{did}_summary.json (n_test={len(rows)}, all_bg={all_bg})")

if __name__ == "__main__":
    main()
