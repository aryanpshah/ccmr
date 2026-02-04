import argparse, csv
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, distance_transform_edt
from nnunetv2.paths import nnUNet_raw

K = 8

def dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return 1.0 if denom == 0 else float(2.0 * inter / denom)

def hd95_mm(a: np.ndarray, b: np.ndarray, spacing_zyx) -> float:
    # a, b are boolean masks for one class
    if a.sum() == 0 and b.sum() == 0:
        return 0.0
    if a.sum() == 0 or b.sum() == 0:
        return float("nan")

    # surface voxels
    a_s = np.logical_xor(a, binary_erosion(a))
    b_s = np.logical_xor(b, binary_erosion(b))

    if a_s.sum() == 0 and b_s.sum() == 0:
        return 0.0
    if a_s.sum() == 0 or b_s.sum() == 0:
        return float("nan")

    # distance to nearest surface of the other mask
    dt_b = distance_transform_edt(~b_s, sampling=spacing_zyx)
    dt_a = distance_transform_edt(~a_s, sampling=spacing_zyx)

    d_ab = dt_b[a_s]
    d_ba = dt_a[b_s]
    d = np.concatenate([d_ab, d_ba]).astype(np.float64)

    if d.size == 0:
        return float("nan")
    return float(np.nanpercentile(d, 95))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", type=int, required=True)
    ap.add_argument("--pred-dir", type=str, required=True)
    ap.add_argument("--out-csv", type=str, default=None)
    args = ap.parse_args()

    did = args.dataset_id
    pred_dir = Path(args.pred_dir)
    if not pred_dir.exists():
        raise SystemExit(f"pred dir not found: {pred_dir}")

    raw_ds = sorted(Path(nnUNet_raw).glob(f"Dataset{did}_*"))
    if not raw_ds:
        raise SystemExit(f"Could not find dataset folder in nnUNet_raw for Dataset{did}_*")
    raw_ds = raw_ds[0]
    gt_dir = raw_ds / "labelsTs"
    if not gt_dir.exists():
        raise SystemExit(f"Missing GT labelsTs: {gt_dir}")

    out_csv = Path(args.out_csv) if args.out_csv else Path("outputs/metrics") / f"D{did}_per_case.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    gt_files = sorted(gt_dir.glob("*.nii.gz"))
    if not gt_files:
        raise SystemExit(f"No GT files found in {gt_dir}")

    # Header
    header = ["case_id", "all_bg_pred", "fg_mean_dice", "fg_mean_hd95_mm"]
    header += [f"dice_{k}" for k in range(1, K+1)]
    header += [f"hd95_{k}_mm" for k in range(1, K+1)]

    rows = []
    missing_preds = 0

    for gt_path in gt_files:
        case_id = gt_path.name.replace(".nii.gz", "")
        pr_path = pred_dir / gt_path.name
        if not pr_path.exists():
            # sometimes nnU-Net writes without .nii.gz double suffix issues; try fallback
            alt = pred_dir / (case_id + ".nii.gz")
            if alt.exists():
                pr_path = alt
            else:
                missing_preds += 1
                continue

        gt_img = sitk.ReadImage(str(gt_path))
        pr_img = sitk.ReadImage(str(pr_path))

        gt = sitk.GetArrayFromImage(gt_img)
        pr = sitk.GetArrayFromImage(pr_img)

        # spacing from image header is (x,y,z); arrays are (z,y,x)
        sp_xyz = gt_img.GetSpacing()
        sp_zyx = (sp_xyz[2], sp_xyz[1], sp_xyz[0])

        all_bg = int(np.max(pr) == 0)

        d_list = []
        h_list = []
        per_d = {}
        per_h = {}

        for k in range(1, K+1):
            gk = (gt == k)
            pk = (pr == k)

            dk = dice(gk, pk)
            hk = hd95_mm(gk, pk, sp_zyx)

            per_d[k] = dk
            per_h[k] = hk
            d_list.append(dk)
            h_list.append(hk)

        fg_mean_d = float(np.nanmean(np.array(d_list, dtype=float)))
        fg_mean_h = float(np.nanmean(np.array(h_list, dtype=float)))

        row = [case_id, all_bg, f"{fg_mean_d:.6f}", f"{fg_mean_h:.6f}"]
        row += [f"{per_d[k]:.6f}" for k in range(1, K+1)]
        row += [("nan" if not np.isfinite(per_h[k]) else f"{per_h[k]:.6f}") for k in range(1, K+1)]
        rows.append(row)

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print("[OK] wrote", out_csv, "| n_cases:", len(rows), "| missing preds:", missing_preds)

if __name__ == "__main__":
    main()
