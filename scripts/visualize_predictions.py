import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def find_image_path(images_ts_dir: Path, case_id: str) -> Path:
    """
    Find the image file in imagesTs that corresponds to this case_id.
    Typically something like pat6_0000.nii.gz.
    """
    candidates = sorted(images_ts_dir.glob(f"{case_id}*.nii.gz"))
    if not candidates:
        raise FileNotFoundError(f"No imageTs file found for case_id={case_id} in {images_ts_dir}")
    return candidates[0]

def choose_slice(label_vol: np.ndarray) -> int:
    """
    Choose a representative axial slice:
    - if there is any foreground, pick the middle of the nonzero z-slices
    - otherwise fall back to the middle slice of the volume.
    """
    assert label_vol.ndim == 3
    z_indices = np.where(label_vol > 0)[2]
    if z_indices.size > 0:
        z = int(np.median(z_indices))
    else:
        z = label_vol.shape[2] // 2
    return z

def plot_case(img_path: Path, gt_path: Path, pred_path: Path, out_path: Path, title_suffix: str = ""):
    print(f"Processing {img_path.name} -> {out_path}")

    img_nii = nib.load(str(img_path))
    gt_nii = nib.load(str(gt_path))
    pred_nii = nib.load(str(pred_path))

    img = img_nii.get_fdata()
    gt = gt_nii.get_fdata()
    pred = pred_nii.get_fdata()

    # ensure shapes compatible
    if img.shape != gt.shape:
        raise ValueError(f"Image and GT shapes differ: {img.shape} vs {gt.shape}")
    if gt.shape != pred.shape:
        raise ValueError(f"GT and prediction shapes differ: {gt.shape} vs {pred.shape}")

    z = choose_slice(gt)

    img_slice = img[:, :, z]
    gt_slice = gt[:, :, z]
    pred_slice = pred[:, :, z]

    # Normalize image for display
    img_disp = img_slice.copy()
    if img_disp.max() > img_disp.min():
        img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min())

    case_id = img_path.stem.split("_")[0]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"{case_id} {title_suffix}".strip(), fontsize=14)

    # 1. Raw image
    ax = axes[0]
    ax.imshow(img_disp.T, cmap="gray", origin="lower")
    ax.set_title("Image")
    ax.axis("off")

    # 2. GT vs Prediction overlay
    ax = axes[1]
    ax.imshow(img_disp.T, cmap="gray", origin="lower")
    # GT contour in one style
    ax.contour(gt_slice.T, levels=[0.5], linewidths=1.5)
    # Prediction contour in another style
    ax.contour(pred_slice.T, levels=[0.5], linewidths=1.5, linestyles="dashed")
    ax.set_title("GT (solid) vs Pred (dashed)")
    ax.axis("off")

    # 3. GT and Pred masks side by side
    ax = axes[2]
    # Show overlap: GT=1, Pred=2, Overlap=3 (optional)
    overlap = np.zeros_like(gt_slice)
    overlap[gt_slice > 0] = 1
    overlap[pred_slice > 0] += 2
    ax.imshow(overlap.T, origin="lower")
    ax.set_title("Mask overlay (GT/Pred)")
    ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Qualitative nnUNet prediction vs GT visualization")
    parser.add_argument("--dataset-id", type=int, required=True, help="Dataset ID, e.g. 905, 910, 920, 940")
    parser.add_argument("--tag", type=str, required=True, help="Label budget tag, e.g. L5, L10, L20, L40")
    parser.add_argument("--cases", type=str, nargs="+", required=True, help="Case IDs, e.g. pat6 pat31 pat56")
    parser.add_argument("--outdir", type=str, default="figures/qualitative", help="Output directory for PNGs")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    nnunet_raw = repo_root / "data" / "nnunet" / "nnUNet_raw"
    raw_ds = nnunet_raw / f"Dataset{args.dataset_id}_HVSMR_{args.tag}"
    images_ts_dir = raw_ds / "imagesTs"
    labels_ts_dir = raw_ds / "labelsTs"

    pred_dir = repo_root / "data" / "nnunet" / "predictions" / args.tag
    out_dir = repo_root / args.outdir

    for case_id in args.cases:
        img_path = find_image_path(images_ts_dir, case_id)
        gt_path = labels_ts_dir / f"{case_id}.nii.gz"
        pred_path = pred_dir / f"{case_id}.nii.gz"

        if not gt_path.is_file():
            raise FileNotFoundError(f"GT label not found: {gt_path}")
        if not pred_path.is_file():
            raise FileNotFoundError(f"Prediction not found: {pred_path}")

        out_path = out_dir / f"{case_id}_{args.tag}.png"
        plot_case(img_path, gt_path, pred_path, out_path, title_suffix=f"({args.tag})")

if __name__ == "__main__":
    main()
