"""
Generate quick-look plots of the preprocessed HVSMR volumes with segmentation overlays.

Loads the processed 192^3 isotropic volumes from ``data/processed/images`` and
the corresponding raw segmentation masks from ``data/raw/HVSMR2/cropped_norm``,
applies the same spacing and crop/pad transforms used during preprocessing,
and saves axial slice overlays to ``results/previews``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List, Sequence

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage

# Allow imports from the sibling preprocessing script.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from preprocess_hvsmr import TARGET_SHAPE, TARGET_SPACING, center_crop_or_pad

RAW_MASK_ROOT = Path("data/raw/HVSMR2/cropped_norm")
PROC_IMG_DIR = Path("data/processed/images")
OUTPUT_DIR = Path("results/previews")


def discover_cases(limit: int | None = None) -> List[str]:
    """Return sorted case ids available in the processed image directory."""
    ids = sorted(p.name.replace("_img_proc.nii.gz", "") for p in PROC_IMG_DIR.glob("pat*_img_proc.nii.gz"))
    return ids[:limit] if limit else ids


def load_processed_volume(case_id: str) -> np.ndarray:
    """Load the preprocessed volume in (z, y, x) order."""
    path = PROC_IMG_DIR / f"{case_id}_img_proc.nii.gz"
    if not path.exists():
        raise FileNotFoundError(f"Processed volume not found: {path}")
    img = nib.load(path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    return np.transpose(data, (2, 1, 0))  # (z, y, x)


def load_and_process_mask(case_id: str) -> np.ndarray:
    """Load the raw segmentation, resample to target spacing, and crop/pad to the processed shape."""
    base = RAW_MASK_ROOT / f"{case_id}_cropped_seg.nii"
    candidates = [
        base / base.name,  # some extracts contain a self-named subfile
        base.with_suffix(".nii.gz"),
        base,
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Segmentation mask not found for {case_id}")

    img = nib.load(path)
    data = np.asarray(img.dataobj)
    mask = np.transpose(data, (2, 1, 0))  # (z, y, x)

    zooms = img.header.get_zooms()[:3]
    current_spacing = (zooms[2], zooms[1], zooms[0])  # match (z, y, x)
    zoom_factors = tuple(cs / ts for cs, ts in zip(current_spacing, TARGET_SPACING))

    # Use nearest-neighbor interpolation to preserve integer labels.
    resampled = ndimage.zoom(mask, zoom=zoom_factors, order=0)
    processed = center_crop_or_pad(resampled, TARGET_SHAPE)
    return processed.astype(np.uint8)


def choose_slices(mask: np.ndarray, num_slices: int) -> np.ndarray:
    """Pick evenly spaced axial slices, focusing on regions where the mask is present."""
    nonzero = np.any(mask, axis=(1, 2))
    indices = np.where(nonzero)[0]
    if len(indices) == 0:
        return np.linspace(0, mask.shape[0] - 1, num_slices, dtype=int)
    start, end = int(indices.min()), int(indices.max())
    return np.linspace(start, end, num_slices, dtype=int)


def plot_case(case_id: str, num_slices: int) -> Path:
    """Create and save overlay plots for a single case."""
    volume = load_processed_volume(case_id)
    mask = load_and_process_mask(case_id)
    if volume.shape != mask.shape:
        raise ValueError(f"Volume and mask shapes differ for {case_id}: {volume.shape} vs {mask.shape}")

    slice_indices = choose_slices(mask, num_slices)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{case_id}_axial.png"

    fig, axes = plt.subplots(1, num_slices, figsize=(3.2 * num_slices, 3.2))
    for ax, idx in zip(axes, slice_indices):
        ax.imshow(volume[idx], cmap="gray")
        overlay = np.ma.masked_where(mask[idx] == 0, mask[idx])
        ax.imshow(overlay, cmap="autumn", alpha=0.35, interpolation="nearest")
        ax.set_title(f"{case_id} | z={idx}")
        ax.axis("off")

    fig.suptitle(f"{case_id}: preprocessed volume + mask (axial)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot preprocessed HVSMR volumes with segmentation overlays."
    )
    parser.add_argument(
        "--case-ids",
        nargs="+",
        help="Specific case ids to plot (e.g., pat0 pat1). Defaults to the first few available cases.",
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=3,
        help="Number of cases to plot when case ids are not provided.",
    )
    parser.add_argument(
        "--num-slices",
        type=int,
        default=4,
        help="Number of axial slices to include per case.",
    )
    args = parser.parse_args()

    if args.case_ids:
        case_ids: Sequence[str] = args.case_ids
    else:
        case_ids = discover_cases(limit=args.num_cases)
        if not case_ids:
            raise RuntimeError(f"No processed cases found under {PROC_IMG_DIR}")

    print(f"Generating overlays for: {', '.join(case_ids)}")
    for case_id in case_ids:
        out_path = plot_case(case_id, args.num_slices)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
