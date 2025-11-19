
"""
Preprocess the HVSMR-2.0 dataset by normalizing volumes to 1 mm isotropic
spacing, cropping/padding to 192^3, and exporting standardized split files.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAW_ROOT = "./data/raw/HVSMR2/cropped_norm"
CLINICAL_CSV = "./data/raw/HVSMR2/hvsmr_clinical.csv"
PROC_DIR = "./data/processed"
PROC_IMG_DIR = os.path.join(PROC_DIR, "images")
SPLIT_DIR = "./data/splits"

TARGET_SPACING: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # (z, y, x)
TARGET_SHAPE: Tuple[int, int, int] = (192, 192, 192)  # (z, y, x)
RANDOM_SEED = 42

TRAIN_TARGET = 39
VAL_TARGET = 10
TEST_TARGET = 10
LABEL_BUDGET_SIZES = (5, 10, 20, 40)
LABEL_BUDGET_SEED = 1337


@dataclass
class CaseRecord:
    """Container describing a single case."""

    pat_index: int
    case_id: str
    image_path: str
    severity: str


def load_severity_map(csv_path: str) -> Dict[int, str]:
    """Read the clinical CSV and map patient indices to severity labels."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Clinical CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"Pat", "Category"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {', '.join(missing)}")

    severity_map: Dict[int, str] = {}
    # HVSMR clinical metadata enumerates patients from 0 through 58 inclusive.
    for _, row in df.iterrows():
        pat = int(row["Pat"])
        severity = str(row["Category"]).strip().lower()
        severity_map[pat] = severity
    return severity_map


def extract_case_info(filepath: str) -> Tuple[int, str]:
    """Return the patient index and canonical case_id (e.g. pat0)."""
    base = os.path.basename(filepath)
    base = re.sub(r"\.nii(\.gz)?$", "", base, flags=re.IGNORECASE)
    match = re.search(r"pat(\d+)", base, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not parse patient index from filename: {filepath}")
    pat_idx = int(match.group(1))
    case_id = f"pat{pat_idx}"
    return pat_idx, case_id


def discover_cases(raw_root: str, severity_map: Dict[int, str]) -> List[CaseRecord]:
    """Walk the raw directory and collect every NIfTI case."""
    if not os.path.isdir(raw_root):
        raise FileNotFoundError(f"Raw data directory not found: {raw_root}")

    cases_map: Dict[str, CaseRecord] = {}
    for root, _, files in os.walk(raw_root):
        for fname in files:
            if not fname.lower().endswith((".nii", ".nii.gz")):
                continue
            path = os.path.join(root, fname)
            pat_idx, case_id = extract_case_info(path)
            severity = severity_map.get(pat_idx)
            severity = severity if severity is not None else "unknown"
            if case_id in cases_map:
                continue  # avoid duplicate entries for identical case folders
            cases_map[case_id] = CaseRecord(
                pat_index=pat_idx,
                case_id=case_id,
                image_path=os.path.abspath(path),
                severity=severity,
            )
    cases = sorted(cases_map.values(), key=lambda c: c.pat_index)
    if not cases:
        raise RuntimeError(f"No cases discovered in {raw_root}")
    return cases


def resample_to_spacing(
    volume: np.ndarray,
    current_spacing: Sequence[float],
    target_spacing: Sequence[float],
) -> np.ndarray:
    """Resample `volume` from current spacing to target spacing."""
    zoom_factors = tuple(cs / ts for cs, ts in zip(current_spacing, target_spacing))
    return ndimage.zoom(volume, zoom=zoom_factors, order=3)


def center_crop_or_pad(volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """Symmetrically crop or pad to reach `target_shape`."""
    result = volume
    for axis, target in enumerate(target_shape):
        current = result.shape[axis]
        if current > target:
            start = (current - target) // 2
            end = start + target
            slices = [slice(None)] * result.ndim
            slices[axis] = slice(start, end)
            result = result[tuple(slices)]
        elif current < target:
            pad_total = target - current
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            pad_widths = [(0, 0)] * result.ndim
            pad_widths[axis] = (pad_before, pad_after)
            result = np.pad(result, pad_widths, mode="constant")
    return result


def process_case(case: CaseRecord) -> np.ndarray:
    """Load, resample, and reshape a single case."""
    img = nib.load(case.image_path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    volume = np.transpose(data, (2, 1, 0))  # (z, y, x)
    zooms = img.header.get_zooms()[:3]
    current_spacing = (zooms[2], zooms[1], zooms[0])  # match (z, y, x)

    resampled = resample_to_spacing(volume, current_spacing, TARGET_SPACING)
    processed = center_crop_or_pad(resampled, TARGET_SHAPE)
    return processed


def save_nifti(volume_zyx: np.ndarray, out_path: str) -> None:
    """Save the processed (z, y, x) volume as NIfTI with identity affine."""
    volume_xyz = np.transpose(volume_zyx, (2, 1, 0))
    nifti_img = nib.Nifti1Image(volume_xyz, affine=np.eye(4))
    nib.save(nifti_img, out_path)


def compute_split_sizes(total: int) -> Tuple[int, int, int]:
    """Adjust split sizes to match `total` while preserving target ratios."""
    base_total = TRAIN_TARGET + VAL_TARGET + TEST_TARGET
    if base_total <= 0:
        raise ValueError("Target split counts must be positive.")
    scaled = [
        TRAIN_TARGET * total / base_total,
        VAL_TARGET * total / base_total,
        TEST_TARGET * total / base_total,
    ]
    sizes = [int(np.floor(x)) for x in scaled]
    remainder = total - sum(sizes)
    if remainder < 0:
        raise ValueError("Split size calculation underflowed.")
    if remainder:
        fracs = [(scaled[i] - sizes[i], i) for i in range(3)]
        for _, idx in sorted(fracs, reverse=True)[:remainder]:
            sizes[idx] += 1
    return tuple(sizes)  # type: ignore[return-value]


def create_stratified_splits(
    case_ids: Sequence[str],
    severities: Sequence[str],
    *,
    train_n: int,
    val_n: int,
    test_n: int,
    random_state: int,
) -> Tuple[List[str], List[str], List[str]]:
    """Generate stratified train/val/test splits."""
    total = len(case_ids)
    if train_n + val_n + test_n != total:
        raise ValueError("Split sizes do not match number of cases.")

    train_ids, temp_ids, train_sev, temp_sev = train_test_split(
        case_ids,
        severities,
        train_size=train_n,
        test_size=val_n + test_n,
        stratify=severities,
        random_state=random_state,
    )
    val_ids, test_ids, _, _ = train_test_split(
        temp_ids,
        temp_sev,
        train_size=val_n,
        test_size=test_n,
        stratify=temp_sev,
        random_state=random_state,
    )
    return list(train_ids), list(val_ids), list(test_ids)


def write_split_file(path: str, ids: Iterable[str]) -> None:
    """Persist a list of IDs to disk."""
    with open(path, "w", encoding="utf-8") as f:
        for case_id in ids:
            f.write(f"{case_id}\n")


def create_label_budget_subsets(
    train_ids: Sequence[str],
    budgets: Sequence[int],
    seed: int,
) -> Dict[int, List[str]]:
    """Create nested label budgets from the train IDs."""
    max_budget = max(budgets)
    if max_budget > len(train_ids):
        raise ValueError(
            f"Largest label budget ({max_budget}) exceeds available train IDs ({len(train_ids)})."
        )
    rng = np.random.default_rng(seed)
    permuted = list(train_ids)
    rng.shuffle(permuted)
    subsets: Dict[int, List[str]] = {}
    for size in sorted(budgets):
        subsets[size] = permuted[:size]
    return subsets


def log_severity_counts(label: str, case_ids: Sequence[str], severity_lookup: Dict[str, str]) -> None:
    """Print a short summary of severity distribution."""
    counts = Counter(severity_lookup[cid] for cid in case_ids)
    summary = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    print(f"{label}: {len(case_ids)} cases ({summary})")


def main() -> None:
    severity_map = load_severity_map(CLINICAL_CSV)
    cases = discover_cases(RAW_ROOT, severity_map)
    case_ids = [case.case_id for case in cases]

    labeled_cases = [case for case in cases if case.pat_index in severity_map]
    labeled_ids = [case.case_id for case in labeled_cases]
    severity_lookup = {case.case_id: case.severity for case in labeled_cases}

    print(f"Discovered {len(cases)} cases under {RAW_ROOT}")
    log_severity_counts("Clinical severity distribution", labeled_ids, severity_lookup)
    unlabeled = sorted(set(case_ids) - set(labeled_ids))
    if unlabeled:
        print(f"Cases lacking clinical severity metadata: {', '.join(unlabeled)}")

    os.makedirs(PROC_IMG_DIR, exist_ok=True)
    for case in cases:
        print(f"Processing {case.case_id} (pat {case.pat_index}, severity={case.severity})")
        processed_volume = process_case(case)
        out_path = os.path.join(PROC_IMG_DIR, f"{case.case_id}_img_proc.nii.gz")
        save_nifti(processed_volume, out_path)

    labeled_sev = [case.severity for case in labeled_cases]
    train_n, val_n, test_n = compute_split_sizes(len(labeled_ids))
    train_ids, val_ids, test_ids = create_stratified_splits(
        labeled_ids,
        labeled_sev,
        train_n=train_n,
        val_n=val_n,
        test_n=test_n,
        random_state=RANDOM_SEED,
    )

    log_severity_counts("Train split severity", train_ids, severity_lookup)
    log_severity_counts("Validation split severity", val_ids, severity_lookup)
    log_severity_counts("Test split severity", test_ids, severity_lookup)

    all_case_set = set(case_ids)
    if "pat59" in all_case_set:
        print(
            "pat59 has no clinical severity metadata; adding to train set for overall training."
        )
        if "pat59" not in train_ids:
            train_ids.append("pat59")

    label_budgets = create_label_budget_subsets(
        train_ids, LABEL_BUDGET_SIZES, LABEL_BUDGET_SEED
    )

    print(
        "Train cases: "
        f"{len(train_ids)}, Val cases: {len(val_ids)}, Test cases: {len(test_ids)}"
    )

    os.makedirs(SPLIT_DIR, exist_ok=True)
    write_split_file(os.path.join(SPLIT_DIR, "train_ids.txt"), train_ids)
    write_split_file(os.path.join(SPLIT_DIR, "val_ids.txt"), val_ids)
    write_split_file(os.path.join(SPLIT_DIR, "test_ids.txt"), test_ids)
    for size in sorted(label_budgets):
        out_path = os.path.join(SPLIT_DIR, f"train_L{size}.txt")
        write_split_file(out_path, label_budgets[size])
    print(
        "Saved label budget subsets: "
        + ", ".join(f"L{size}={len(label_budgets[size])}" for size in sorted(label_budgets))
    )

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
