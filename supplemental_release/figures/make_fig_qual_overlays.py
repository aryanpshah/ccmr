#!/usr/bin/env python3
"""
Generate qualitative overlay figure (GT vs L5/L10/L20/L40 predictions).

Requires: Ground truth and prediction NIfTI files.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.io import read_nii

STRUCT = {1: "LV", 2: "RV", 3: "LA", 4: "RA", 5: "Aorta", 6: "PA", 7: "SVC", 8: "IVC"}


def make_cmap():
    colors = [(0, 0, 0, 0.0)] + [(0.9, 0.1, 0.1, 0.35), (0.1, 0.6, 0.1, 0.35), (0.1, 0.35, 0.9, 0.35),
             (0.75, 0.1, 0.75, 0.35), (0.9, 0.55, 0.1, 0.35), (0.1, 0.75, 0.75, 0.35),
             (0.55, 0.55, 0.1, 0.35), (0.45, 0.25, 0.9, 0.35)]
    return ListedColormap(colors)


def pick_slice(seg):
    fg = (seg > 0).sum(axis=(1, 2))
    return int(np.argmax(fg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=str, required=True, help="Case ID (e.g., pat11)")
    ap.add_argument("--gt_path", type=str, required=True)
    ap.add_argument("--img_path", type=str, default=None)
    ap.add_argument("--pred_l5", type=str, required=True)
    ap.add_argument("--pred_l10", type=str, required=True)
    ap.add_argument("--pred_l20", type=str, required=True)
    ap.add_argument("--pred_l40", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="artifacts")
    ap.add_argument("--data_root", type=str, default=".")
    ap.add_argument("--splits_dir", type=str, default="splits")
    args = ap.parse_args()

    cid = args.case
    gt, _ = read_nii(Path(args.gt_path))
    img_path = args.img_path
    if img_path and Path(img_path).exists():
        img, _ = read_nii(Path(img_path))
        img = np.clip((img.astype(float) - np.percentile(img, 1)) / (np.percentile(img, 99) - np.percentile(img, 1) + 1e-8), 0, 1)
    else:
        img = np.zeros_like(gt, dtype=float)

    preds = {}
    for k, p in [("L5", args.pred_l5), ("L10", args.pred_l10), ("L20", args.pred_l20), ("L40", args.pred_l40)]:
        path = Path(p) / (cid + ".nii.gz")
        if not path.exists():
            path = Path(p) / (cid + ".nii")
        if not path.exists():
            raise FileNotFoundError(path)
        preds[k], _ = read_nii(path)

    z = pick_slice(gt)
    cmap = make_cmap()
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    axes[0].imshow(img[z], cmap="gray")
    axes[0].imshow(gt[z], cmap=cmap, vmin=0, vmax=8, alpha=0.6)
    axes[0].set_title("GT (z=%d)" % z)
    axes[0].axis("off")
    for i, (lab, pr) in enumerate(preds.items(), 1):
        axes[i].imshow(img[z], cmap="gray")
        axes[i].imshow(pr[z], cmap=cmap, vmin=0, vmax=8, alpha=0.6)
        axes[i].set_title(lab)
        axes[i].axis("off")
    handles = [Patch(facecolor=cmap(k), edgecolor="black", label=STRUCT[k]) for k in range(1, 9)]
    fig.legend(handles=handles, loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.02))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / ("%s_GT_L5_L10_L20_L40.png" % cid), dpi=200, bbox_inches="tight")
    plt.close()
    print("[OK]", out_dir / ("%s_GT_L5_L10_L20_L40.png" % cid))
