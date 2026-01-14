"""
Compute per-case, per-structure segmentation metrics (Dice, HD95) for nnU-Net predictions.

Handles empty masks correctly:
- If both GT and pred empty: Dice=1, HD95=0
- If GT empty but pred non-empty OR GT non-empty but pred empty: Dice=0, HD95=NaN

Reads spacing from NIfTI headers (no assumptions about isotropy).
Validates all inputs before processing.

Usage:
    python compute_per_case_metrics.py \\
        --gt_dir data/nnunet/nnUNet_raw/Dataset905_HVSMR_L5/labelsTs \\
        --pred_dir data/nnunet/predictions/L5/seed0 \\
        --test_ids data/splits/test_ids.txt \\
        --out_csv results/L5_seed0_per_case.csv \\
        --budget L5 \\
        --seed 0
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt


def read_case_ids(path: Path) -> List[str]:
    """Read case IDs from a text file (one per line)."""
    if not path.exists():
        raise FileNotFoundError(f"Test IDs file not found: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_nifti_with_spacing(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Load NIfTI volume and return (data, spacing).
    
    Returns:
        data: (z, y, x) array
        spacing: (z_spacing, y_spacing, x_spacing) in mm
    """
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")
    
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.int16)
    
    # Convert from (x, y, z) to (z, y, x) for consistency
    data = np.transpose(data, (2, 1, 0))
    
    # Get spacing from header (x, y, z) -> convert to (z, y, x)
    zooms = img.header.get_zooms()[:3]
    spacing = (zooms[2], zooms[1], zooms[0])
    
    return data, spacing


def validate_volume(data: np.ndarray, case_id: str, name: str, max_label: int = 8) -> None:
    """Validate that a segmentation volume is valid."""
    unique_vals = np.unique(data)
    
    # Check label values are in valid range
    if np.any(unique_vals < 0) or np.any(unique_vals > max_label):
        invalid = unique_vals[(unique_vals < 0) | (unique_vals > max_label)]
        raise ValueError(
            f"{name} for {case_id} contains invalid label values: {invalid.tolist()}. "
            f"Expected range: [0, {max_label}]"
        )


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute Dice coefficient for binary masks.
    
    Handles empty cases:
    - Both empty: returns 1.0
    - One empty: returns 0.0
    """
    pred_sum = np.sum(pred_mask)
    gt_sum = np.sum(gt_mask)
    
    # Both empty
    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    
    # One empty
    if pred_sum == 0 or gt_sum == 0:
        return 0.0
    
    # Normal case
    intersection = np.sum(pred_mask & gt_mask)
    dice = 2.0 * intersection / (pred_sum + gt_sum)
    return float(dice)


def compute_hd95(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    spacing: Tuple[float, float, float]
) -> float:
    """
    Compute 95th percentile Hausdorff Distance.
    
    Returns:
        HD95 in millimeters, or NaN if either mask is empty (but not both).
        Returns 0.0 if both masks are empty.
    """
    pred_sum = np.sum(pred_mask)
    gt_sum = np.sum(gt_mask)
    
    # Both empty
    if pred_sum == 0 and gt_sum == 0:
        return 0.0
    
    # One empty (error case)
    if pred_sum == 0 or gt_sum == 0:
        return np.nan
    
    # Compute surface distances
    # Distance from pred surface to nearest GT point
    pred_border = pred_mask ^ ndi.binary_erosion(pred_mask)
    gt_border = gt_mask ^ ndi.binary_erosion(gt_mask)
    
    if not np.any(pred_border) or not np.any(gt_border):
        # If no border (single voxel), fall back to centroid distance
        return 0.0
    
    # Distance transform gives distance to nearest True pixel
    gt_dt = distance_transform_edt(~gt_mask, sampling=spacing)
    pred_dt = distance_transform_edt(~pred_mask, sampling=spacing)
    
    # Get distances from pred border to GT
    distances_pred_to_gt = gt_dt[pred_border]
    # Get distances from GT border to pred
    distances_gt_to_pred = pred_dt[gt_border]
    
    # Symmetric HD: combine both directions
    all_distances = np.concatenate([distances_pred_to_gt, distances_gt_to_pred])
    
    # 95th percentile
    hd95 = np.percentile(all_distances, 95)
    return float(hd95)


def compute_case_metrics(
    pred_path: Path,
    gt_path: Path,
    case_id: str,
    num_classes: int = 8
) -> Dict[int, Dict[str, float]]:
    """
    Compute metrics for all structures in a single case.
    
    Returns:
        Dict mapping structure_id (1..num_classes) to metrics dict containing:
        - dice, hd95, gt_voxels, pred_voxels
    """
    # Load volumes
    pred_data, pred_spacing = load_nifti_with_spacing(pred_path)
    gt_data, gt_spacing = load_nifti_with_spacing(gt_path)
    
    # Validate
    if pred_data.shape != gt_data.shape:
        raise ValueError(
            f"Shape mismatch for {case_id}: "
            f"pred {pred_data.shape} vs gt {gt_data.shape}"
        )
    
    if pred_spacing != gt_spacing:
        warnings.warn(
            f"Spacing mismatch for {case_id}: "
            f"pred {pred_spacing} vs gt {gt_spacing}. Using GT spacing."
        )
    
    spacing = gt_spacing
    
    validate_volume(pred_data, case_id, "Prediction", max_label=num_classes)
    validate_volume(gt_data, case_id, "Ground truth", max_label=num_classes)
    
    # Compute metrics per structure
    results = {}
    
    for structure_id in range(1, num_classes + 1):
        pred_mask = (pred_data == structure_id).astype(bool)
        gt_mask = (gt_data == structure_id).astype(bool)
        
        dice = compute_dice(pred_mask, gt_mask)
        hd95 = compute_hd95(pred_mask, gt_mask, spacing)
        
        results[structure_id] = {
            'dice': dice,
            'hd95': hd95,
            'gt_voxels': int(np.sum(gt_mask)),
            'pred_voxels': int(np.sum(pred_mask)),
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-case, per-structure metrics (Dice, HD95)"
    )
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Directory containing ground truth NIfTI files")
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Directory containing prediction NIfTI files")
    parser.add_argument("--test_ids", type=str, required=True,
                        help="Text file with test case IDs (one per line)")
    parser.add_argument("--out_csv", type=str, required=True,
                        help="Output CSV file path")
    parser.add_argument("--budget", type=str, required=True,
                        help="Label budget (e.g., L5, L10)")
    parser.add_argument("--seed", type=int, required=True,
                        help="Random seed for this run")
    parser.add_argument("--num_classes", type=int, default=8,
                        help="Number of anatomical structures (default: 8)")
    
    args = parser.parse_args()
    
    # Setup paths
    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    test_ids_file = Path(args.test_ids)
    out_csv = Path(args.out_csv)
    
    if not gt_dir.exists():
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    
    # Read test case IDs
    test_ids = read_case_ids(test_ids_file)
    print(f"Found {len(test_ids)} test cases")
    
    # Validate all files exist
    missing_gt = []
    missing_pred = []
    
    for case_id in test_ids:
        gt_path = gt_dir / f"{case_id}.nii.gz"
        pred_path = pred_dir / f"{case_id}.nii.gz"
        
        if not gt_path.exists():
            missing_gt.append(case_id)
        if not pred_path.exists():
            missing_pred.append(case_id)
    
    if missing_gt:
        print(f"ERROR: Missing GT files for {len(missing_gt)} cases:", file=sys.stderr)
        for cid in missing_gt:
            print(f"  {cid}", file=sys.stderr)
        sys.exit(1)
    
    if missing_pred:
        print(f"ERROR: Missing prediction files for {len(missing_pred)} cases:", file=sys.stderr)
        for cid in missing_pred:
            print(f"  {cid}", file=sys.stderr)
        sys.exit(1)
    
    print(f"All {len(test_ids)} cases have both GT and prediction files")
    
    # Compute metrics for each case
    rows = []
    
    for i, case_id in enumerate(test_ids, 1):
        print(f"[{i}/{len(test_ids)}] Processing {case_id}...")
        
        gt_path = gt_dir / f"{case_id}.nii.gz"
        pred_path = pred_dir / f"{case_id}.nii.gz"
        
        try:
            case_metrics = compute_case_metrics(
                pred_path, gt_path, case_id, num_classes=args.num_classes
            )
            
            # Create row for each structure
            for structure_id, metrics in case_metrics.items():
                row = {
                    'case_id': case_id,
                    'budget': args.budget,
                    'seed': args.seed,
                    f'struct_{structure_id}_dice': metrics['dice'],
                    f'struct_{structure_id}_hd95': metrics['hd95'],
                    f'struct_{structure_id}_gt_voxels': metrics['gt_voxels'],
                    f'struct_{structure_id}_pred_voxels': metrics['pred_voxels'],
                }
                rows.append(row)
        
        except Exception as e:
            print(f"ERROR processing {case_id}: {e}", file=sys.stderr)
            raise
    
    # Convert to wide format (one row per case)
    # Each case has columns for all structures
    df_long = pd.DataFrame(rows)
    
    # Pivot to wide format
    case_data = []
    for case_id in test_ids:
        case_rows = df_long[df_long['case_id'] == case_id]
        if len(case_rows) == 0:
            continue
        
        wide_row = {
            'case_id': case_id,
            'budget': args.budget,
            'seed': args.seed,
        }
        
        for structure_id in range(1, args.num_classes + 1):
            struct_key_prefix = f'struct_{structure_id}'
            struct_rows = case_rows[case_rows.columns[case_rows.columns.str.startswith(struct_key_prefix)]]
            
            if len(struct_rows) > 0:
                for col in struct_rows.columns:
                    wide_row[col] = struct_rows[col].iloc[0]
        
        case_data.append(wide_row)
    
    df_wide = pd.DataFrame(case_data)
    
    # Save
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_wide.to_csv(out_csv, index=False)
    
    print(f"\nMetrics saved to: {out_csv}")
    print(f"Total cases: {len(df_wide)}")
    
    # Print summary statistics
    print("\n=== SUMMARY ===")
    for structure_id in range(1, args.num_classes + 1):
        dice_col = f'struct_{structure_id}_dice'
        hd95_col = f'struct_{structure_id}_hd95'
        
        if dice_col in df_wide.columns:
            mean_dice = df_wide[dice_col].mean()
            # For HD95, exclude NaN values from mean
            mean_hd95 = df_wide[hd95_col].dropna().mean()
            num_nan = df_wide[hd95_col].isna().sum()
            
            print(f"Structure {structure_id}: Dice={mean_dice:.4f}, "
                  f"HD95={mean_hd95:.2f}mm (NaN cases: {num_nan})")


# Import scipy.ndimage as ndi for binary_erosion
import scipy.ndimage as ndi


if __name__ == "__main__":
    main()
