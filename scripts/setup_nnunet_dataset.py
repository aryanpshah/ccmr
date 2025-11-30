"""
Create an nnU-Net v2 dataset from the processed HVSMR volumes and masks.

- Copies processed images from ``data/processed/images`` into an nnU-Net raw dataset
  layout (imagesTr/imagesTs with `_0000` suffix).
- Resamples the original cropped segmentation masks with the same spacing/crop logic
  used during preprocessing and writes them into ``labelsTr``.
- Emits a ``dataset.json`` that points nnU-Net to the correct files.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Sequence

import nibabel as nib
import numpy as np
from scipy import ndimage

from preprocess_hvsmr import TARGET_SHAPE, TARGET_SPACING, center_crop_or_pad

RAW_MASK_ROOT = Path("data/raw/HVSMR2/cropped_norm")
PROC_IMG_DIR = Path("data/processed/images")
SPLIT_DIR = Path("data/splits")
NNUNET_ROOT = Path("data/nnunet")
NNUNET_RAW = NNUNET_ROOT / "nnUNet_raw"
NNUNET_PREPROCESSED = NNUNET_ROOT / "nnUNet_preprocessed"
NNUNET_RESULTS = NNUNET_ROOT / "nnUNet_results"

DEFAULT_DATASET_ID = 901
DEFAULT_DATASET_NAME = f"Dataset{DEFAULT_DATASET_ID}_HVSMRProc"


def read_ids(path: Path) -> List[str]:
    """Return a list of case ids from a text file (one per line)."""
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def discover_processed_cases() -> List[str]:
    """Find all processed images and return their case ids (patX)."""
    return sorted(p.name.replace("_img_proc.nii.gz", "") for p in PROC_IMG_DIR.glob("pat*_img_proc.nii.gz"))


def load_and_process_mask(case_id: str) -> np.ndarray:
    """
    Load the raw cropped segmentation and align it with the processed volume grid.

    Handles both flat .nii files and the awkward self-named directory layout some
    HVSMR extracts use.
    """
    base = RAW_MASK_ROOT / f"{case_id}_cropped_seg.nii"
    candidates = [
        base / base.name,  # nested file scenario
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
    current_spacing = (zooms[2], zooms[1], zooms[0])
    zoom_factors = tuple(cs / ts for cs, ts in zip(current_spacing, TARGET_SPACING))

    resampled = ndimage.zoom(mask, zoom=zoom_factors, order=0)  # nearest neighbor for labels
    processed = center_crop_or_pad(resampled, TARGET_SHAPE)
    processed = processed.astype(np.uint8)
    # Ensure labels are within expected set {0,1,2}; drop any stray labels.
    processed = np.clip(processed, 0, 2)
    return processed


def save_mask(mask_zyx: np.ndarray, out_path: Path) -> None:
    """Persist mask (z, y, x) as NIfTI with identity affine in (x, y, z)."""
    mask_xyz = np.transpose(mask_zyx, (2, 1, 0)).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask_xyz, affine=np.eye(4)), out_path)


def copy_image(case_id: str, dest_dir: Path) -> Path:
    """Copy processed image into nnU-Net naming convention."""
    src = PROC_IMG_DIR / f"{case_id}_img_proc.nii.gz"
    if not src.exists():
        raise FileNotFoundError(f"Processed image not found: {src}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{case_id}_0000.nii.gz"
    shutil.copy2(src, dest)
    return dest


def build_dataset(train_ids: Sequence[str], test_ids: Sequence[str], dataset_name: str) -> None:
    dataset_dir = NNUNET_RAW / dataset_name
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    labels_ts = dataset_dir / "labelsTs"  # not used by nnU-Net, but handy for local eval

    dataset_dir.mkdir(parents=True, exist_ok=True)
    images_ts.mkdir(parents=True, exist_ok=True)
    labels_ts.mkdir(parents=True, exist_ok=True)

    print(f"Writing nnU-Net dataset to {dataset_dir}")
    for case_id in train_ids:
        print(f"  train: {case_id}")
        copy_image(case_id, images_tr)
        mask = load_and_process_mask(case_id)
        save_mask(mask, labels_tr / f"{case_id}.nii.gz")

    for case_id in test_ids:
        print(f"  test: {case_id}")
        copy_image(case_id, images_ts)
        # Store labels for the test split so we can run local evaluation/inspection.
        # nnU-Net ignores labelsTs during training/inference.
        mask = load_and_process_mask(case_id)
        save_mask(mask, labels_ts / f"{case_id}.nii.gz")

    dataset_json = {
        "name": "HVSMR processed",
        "description": "HVSMR 2016 processed volumes (1mm, 192^3) with cropped masks",
        "reference": "HVSMR challenge",
        "licence": "HVSMR challenge terms",
        "release": "1.0",
        "tensorImageSize": "3D",
        "file_ending": ".nii.gz",
        "channel_names": {"0": "MRI"},
        # HVSMR masks: 1=blood pool, 2=myocardium
        "labels": {"background": 0, "blood_pool": 1, "myocardium": 2},
        "numTraining": len(train_ids),
        "numTest": len(test_ids),
        "training": [
            {
                "image": f"./imagesTr/{cid}_0000.nii.gz",
                "label": f"./labelsTr/{cid}.nii.gz",
            }
            for cid in train_ids
        ],
        "test": [f"./imagesTs/{cid}_0000.nii.gz" for cid in test_ids],
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
    print(f"Wrote dataset.json with {len(train_ids)} training and {len(test_ids)} test cases")

    # Ensure standard nnU-Net directory placeholders exist.
    NNUNET_PREPROCESSED.mkdir(parents=True, exist_ok=True)
    NNUNET_RESULTS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create nnU-Net dataset folders from processed HVSMR volumes.")
    parser.add_argument(
        "--label-budget",
        type=int,
        choices=[5, 10, 20, 40],
        help="Use the specified label budget subset (train_L{budget}.txt). Defaults to full train+val.",
    )
    parser.add_argument(
        "--dataset-id",
        type=int,
        default=None,
        help="Optional numeric dataset id (used in folder name). Defaults to 901 for full data or a budget-specific id if provided.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Optional dataset folder name override (e.g., Dataset905_HVSMR_L5).",
    )
    args = parser.parse_args()

    if args.label_budget:
        budget_file = SPLIT_DIR / f"train_L{args.label_budget}.txt"
        train_ids = read_ids(budget_file)
        if not train_ids:
            raise RuntimeError(f"Label budget file missing or empty: {budget_file}")
        test_ids = read_ids(SPLIT_DIR / "test_ids.txt")
        suggested_id = {5: 905, 10: 910, 20: 920, 40: 940}.get(args.label_budget, DEFAULT_DATASET_ID)
        dataset_id = args.dataset_id if args.dataset_id is not None else suggested_id
        dataset_name = args.dataset_name or f"Dataset{dataset_id:03d}_HVSMR_L{args.label_budget}"
        print(f"Using label budget L={args.label_budget}: {len(train_ids)} training cases -> {dataset_name}")
    else:
        train_ids = read_ids(SPLIT_DIR / "train_ids.txt") + read_ids(SPLIT_DIR / "val_ids.txt")
        test_ids = read_ids(SPLIT_DIR / "test_ids.txt")
        dataset_id = args.dataset_id if args.dataset_id is not None else DEFAULT_DATASET_ID
        dataset_name = args.dataset_name or DEFAULT_DATASET_NAME

    if not train_ids:
        print("train_ids not found; defaulting to all processed cases.")
        train_ids = discover_processed_cases()
    if not test_ids:
        remaining = set(discover_processed_cases()) - set(train_ids)
        test_ids = sorted(remaining)

    if not train_ids:
        raise RuntimeError("No processed cases available to build the dataset.")

    build_dataset(train_ids, test_ids, dataset_name)


if __name__ == "__main__":
    main()
