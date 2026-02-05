from pathlib import Path
import argparse
import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap


STRUCT = {
    1: "LV",
    2: "RV",
    3: "LA",
    4: "RA",
    5: "Aorta",
    6: "PA",
    7: "SVC",
    8: "IVC",
}
def read_nii(p: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(p)))  # z,y,x

def normalize_img(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [1, 99])
    x = np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)
    return x

def pick_slice(seg_gt: np.ndarray) -> int:
    fg = (seg_gt > 0).sum(axis=(1,2))
    return int(np.argmax(fg))

def make_cmap():
    # background transparent
    colors = [
        (0,0,0,0.0),         # 0
        (0.90,0.10,0.10,0.35),# 1
        (0.10,0.60,0.10,0.35),# 2
        (0.10,0.35,0.90,0.35),# 3
        (0.75,0.10,0.75,0.35),# 4
        (0.90,0.55,0.10,0.35),# 5
        (0.10,0.75,0.75,0.35),# 6
        (0.55,0.55,0.10,0.35),# 7
        (0.45,0.25,0.90,0.35),# 8
    ]
    return ListedColormap(colors)

def overlay(ax, img2d, seg2d, title):
    ax.imshow(img2d, cmap="gray", interpolation="nearest")
    ax.imshow(seg2d, cmap=make_cmap(), interpolation="nearest", vmin=0, vmax=8)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def add_legend(fig):
    cmap = make_cmap()
    handles = []
    for k in range(1, 9):
        handles.append(Patch(facecolor=cmap(k), edgecolor="black", label=STRUCT[k]))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=8,
        frameon=True,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", type=int, required=True,
                    help="Used only to locate imagesTs/labelsTs for the held-out test set.")
    ap.add_argument("--case", type=str, required=True,
                    help="Case id without extension, e.g. pat11")
    ap.add_argument("--pred-l5", type=str, required=True)
    ap.add_argument("--pred-l10", type=str, required=True)
    ap.add_argument("--pred-l20", type=str, required=True)
    ap.add_argument("--pred-l40", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="outputs/qualitative/multi_budget")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--slice", type=int, default=-1,
                    help="If -1, auto-pick slice with max GT foreground. Else use this z index.")
    args = ap.parse_args()

    nnUNet_raw = Path(os.environ["nnUNet_raw"])
    ds = sorted(nnUNet_raw.glob(f"Dataset{args.dataset_id}_*"))[0]
    img_dir = ds/"imagesTs"
    gt_dir  = ds/"labelsTs"

    cid = args.case
    img_p = img_dir/f"{cid}_0000.nii.gz"
    gt_p  = gt_dir/f"{cid}.nii.gz"

    p5  = Path(args.pred_l5)/f"{cid}.nii.gz"
    p10 = Path(args.pred_l10)/f"{cid}.nii.gz"
    p20 = Path(args.pred_l20)/f"{cid}.nii.gz"
    p40 = Path(args.pred_l40)/f"{cid}.nii.gz"

    missing = [str(p) for p in [img_p, gt_p, p5, p10, p20, p40] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    img = read_nii(img_p)
    gt  = read_nii(gt_p).astype(np.int16)
    pr5  = read_nii(p5).astype(np.int16)
    pr10 = read_nii(p10).astype(np.int16)
    pr20 = read_nii(p20).astype(np.int16)
    pr40 = read_nii(p40).astype(np.int16)

    z = pick_slice(gt) if args.slice < 0 else int(args.slice)

    img2d = normalize_img(img[z])
    gt2d  = gt[z]
    pr5_2d  = pr5[z]
    pr10_2d = pr10[z]
    pr20_2d = pr20[z]
    pr40_2d = pr40[z]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 4))
    axes = [fig.add_subplot(1,5,i+1) for i in range(5)]

    overlay(axes[0], img2d, gt2d,  f"{cid} — GT (z={z})")
    overlay(axes[1], img2d, pr5_2d,  "L5")
    overlay(axes[2], img2d, pr10_2d, "L10")
    overlay(axes[3], img2d, pr20_2d, "L20")
    overlay(axes[4], img2d, pr40_2d, "L40")

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    add_legend(fig)
    out_p = outdir/f"{cid}_GT_L5_L10_L20_L40_legend.png"
    fig.savefig(out_p, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print("[OK]", out_p)

if __name__ == "__main__":
    main()
