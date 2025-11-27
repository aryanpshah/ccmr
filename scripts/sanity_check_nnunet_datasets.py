"""
Sanity-check the nnU-Net v2 label-budget datasets.

For datasets 905/910/920/940, this script:
- Loads dataset.json to report counts and ID lists.
- Verifies train IDs match the corresponding train_L*.txt file.
- Verifies test IDs match test_ids.txt and are identical across datasets.
- Samples one preprocessed .npz per dataset and reports the array shape and label values.

Run from repo root:
    python scripts/sanity_check_nnunet_datasets.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

RAW_ROOT = Path("data/nnunet/nnUNet_raw")
PREPROCESSED_ROOT = Path("data/nnunet/nnUNet_preprocessed")
SPLIT_DIR = Path("data/splits")

DATASETS = {
    905: ("Dataset905_HVSMR_L5", SPLIT_DIR / "train_L5.txt"),
    910: ("Dataset910_HVSMR_L10", SPLIT_DIR / "train_L10.txt"),
    920: ("Dataset920_HVSMR_L20", SPLIT_DIR / "train_L20.txt"),
    940: ("Dataset940_HVSMR_L40", SPLIT_DIR / "train_L40.txt"),
}


def read_ids(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"IDs file not found: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_dataset_json(dataset_name: str) -> Dict:
    path = RAW_ROOT / dataset_name / "dataset.json"
    if not path.exists():
        raise FileNotFoundError(f"dataset.json missing: {path}")
    return json.loads(path.read_text())


def get_training_and_test_ids(dataset_json: Dict) -> Tuple[List[str], List[str]]:
    train_ids = [Path(entry["image"]).name.split("_")[0] for entry in dataset_json["training"]]
    test_ids = [Path(p).name.split("_")[0] for p in dataset_json["test"]]
    return train_ids, test_ids


def sample_preprocessed_npz(dataset_name: str) -> Path | None:
    candidates = list((PREPROCESSED_ROOT / dataset_name).glob("*.npz"))
    if not candidates:
        return None
    return random.choice(candidates)


def main() -> None:
    expected_test_ids = read_ids(SPLIT_DIR / "test_ids.txt")
    all_test_id_sets = []

    for ds_id, (ds_name, train_split_path) in DATASETS.items():
        print(f"\nDataset {ds_id} ({ds_name})")
        ds_json = load_dataset_json(ds_name)
        train_ids, test_ids = get_training_and_test_ids(ds_json)

        print(f"  Training cases: {len(train_ids)}")
        print(f"  Test cases:     {len(test_ids)}")
        print(f"  Train IDs: {', '.join(train_ids)}")
        print(f"  Test  IDs: {', '.join(test_ids)}")

        # Check train IDs against split file
        expected_train_ids = read_ids(train_split_path)
        if train_ids == expected_train_ids:
            print(f"  OK: training IDs match {train_split_path.name}")
        else:
            print(f"  ERROR: training IDs differ from {train_split_path.name}")

        # Track test IDs for cross-dataset comparison
        all_test_id_sets.append(test_ids)
        if test_ids == expected_test_ids:
            print("  OK: test IDs match test_ids.txt")
        else:
            print("  ERROR: test IDs differ from test_ids.txt")

        # Sample a preprocessed npz
        npz_path = sample_preprocessed_npz(ds_name)
        if npz_path is None:
            print(f"  WARNING: no preprocessed npz found in {PREPROCESSED_ROOT / ds_name}")
        else:
            npz = np.load(npz_path)
            data = npz["data"]
            seg = npz["seg"]
            print(f"  Sample npz: {npz_path.name}")
            print(f"    data shape: {data.shape}")
            print(f"    seg unique values: {np.unique(seg)}")

    # Cross-dataset test ID consistency
    first = all_test_id_sets[0]
    if all(t == first for t in all_test_id_sets[1:]):
        print("\nOK: test IDs match across all datasets")
    else:
        print("\nERROR: test IDs differ across datasets")


if __name__ == "__main__":
    main()
