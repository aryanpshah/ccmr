#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import glob

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

def read_arr(path: Path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # z,y,x
    return arr

def pick_slice(gt):
    fg = (gt > 0).astype(np.uint8)
    sums = fg.reshape(fg.shape[0], -1).sum(axis=1)
    z = int(np.argmax(sums))
    return z

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="e.g., pat11")
    ap.add_argument("--out", default="outputs/qualitative", help="output dir")
    args = ap.parse_args()

    case = args.case

    # use any dataset raw that has labelsTs; pick L40 dataset (988) for GT
    raw = glob.glob(str(Path(os.environ.get("nnUNet_raw", ".")) / "Dataset988_*"))[0]
    gt_path = Path(raw) / "labelsTs" / f"{case}.nii.gz"
    if not gt_path.exists():
        raise SystemExit(f"Missing GT: {gt_path}")

    gt = read_arr(gt_path)
    z = pick_slice(gt)

    # predictions
    preds = {
        "L5":  Path(f"outputs/test_preds/D985_3d_fullres/{case}.nii.gz"),
        "L10": Path(f"outputs/test_preds/D986_3d_fullres/{case}.nii.gz"),
        "L20": Path(f"outputs/test_preds/D987_3d_fullres/{case}.nii.gz"),
        "L40": Path(f"outputs/test_preds/D988_3d_fullres/{case}.nii.gz"),
    }
    for k,p in preds.items():
        if not p.exists():
            raise SystemExit(f"Missing pred {k}: {p}")

    pr = {k: read_arr(p) for k,p in preds.items()}

    # build figure (2 rows x 5 cols)
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    titles = ["GT", "L5", "L10", "L20", "L40"]

    row0 = [gt, pr["L5"], pr["L10"], pr["L20"], pr["L40"]]
    for j,arr in enumerate(row0):
        axes[0, j].imshow(arr[z], vmin=0, vmax=8)
        axes[0, j].set_title(titles[j])
        axes[0, j].axis("off")

    # error maps (pred != gt, show as 0/1)
    axes[1,0].axis("off")
    axes[1,0].set_title("")

    for j,(lab,arr) in enumerate([("L5",pr["L5"]),("L10",pr["L10"]),("L20",pr["L20"]),("L40",pr["L40"])], start=1):
        err = (arr[z] != gt[z]).astype(np.uint8)
        axes[1, j].imshow(err)
        axes[1, j].set_title(f"Error vs GT ({lab})")
        axes[1, j].axis("off")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{case}_slice{z}_montage.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print("[OK] wrote", outpath)

if __name__ == "__main__":
    main()
