"""
Add labels for test cases into labelsTs for existing nnU-Net v2 datasets.

This does not touch train/test splits; it only writes masks for the test set
into labelsTs so you can run local evaluation/inspection. Uses the same mask
processing as setup_nnunet_dataset.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

# Reuse mask loading/saving logic to stay consistent with dataset creation.
from setup_nnunet_dataset import RAW_MASK_ROOT, load_and_process_mask, save_mask  # type: ignore  # noqa: E402

NNUNET_RAW = ROOT / "data/nnunet/nnUNet_raw"

DEFAULT_DATASETS = [
    "Dataset905_HVSMR_L5",
    "Dataset910_HVSMR_L10",
    "Dataset920_HVSMR_L20",
    "Dataset940_HVSMR_L40",
]


def extract_case_ids(test_entries: Iterable[str]) -> List[str]:
    """Strip nnU-Net image paths like ./imagesTs/pat0_0000.nii.gz -> pat0."""
    case_ids: List[str] = []
    for entry in test_entries:
        stem = Path(entry).name
        if stem.endswith("_0000.nii.gz"):
            case_ids.append(stem.replace("_0000.nii.gz", ""))
    return case_ids


def add_labels_for_dataset(dataset_name: str, overwrite: bool = False) -> None:
    ds_dir = NNUNET_RAW / dataset_name
    dataset_json = ds_dir / "dataset.json"
    if not dataset_json.exists():
        print(f"[WARN] {dataset_json} missing; skipping {dataset_name}")
        return

    labels_ts = ds_dir / "labelsTs"
    labels_ts.mkdir(parents=True, exist_ok=True)

    meta = json.loads(dataset_json.read_text())
    test_entries = meta.get("test", [])
    if not test_entries:
        print(f"[WARN] No test entries in {dataset_json}; skipping {dataset_name}")
        return

    case_ids = extract_case_ids(test_entries)
    print(f"[INFO] {dataset_name}: writing labelsTs for {len(case_ids)} test cases")

    for cid in case_ids:
        out_path = labels_ts / f"{cid}.nii.gz"
        if out_path.exists() and not overwrite:
            print(f"  [SKIP] {cid} exists (use --overwrite to replace)")
            continue
        mask = load_and_process_mask(cid)
        save_mask(mask, out_path)
        print(f"  [OK] {cid} -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create labelsTs for existing nnU-Net datasets without altering splits.")
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset folder name under data/nnunet/nnUNet_raw (can be passed multiple times).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing labelsTs files.")
    args = parser.parse_args()

    datasets = args.datasets or DEFAULT_DATASETS
    for name in datasets:
        add_labels_for_dataset(name, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
