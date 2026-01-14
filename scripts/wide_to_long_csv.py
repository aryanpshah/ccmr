"""
Convert wide per-case CSV to long/tidy format for easier plotting and analysis.

Adds severity information from clinical CSV and reshapes data so each row represents
one structure measurement for one case.

Usage:
    python wide_to_long_csv.py \\
        --wide_csv results/L5_seed0_per_case.csv \\
        --severity_csv data/raw/HVSMR2/hvsmr_clinical.csv \\
        --out_csv results/L5_seed0_long.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import pandas as pd


def load_severity_map(severity_csv: Path) -> Dict[str, str]:
    """
    Load case_id -> severity mapping from clinical CSV.
    
    Expected columns: patient_id (or similar), severity
    """
    if not severity_csv.exists():
        raise FileNotFoundError(f"Severity CSV not found: {severity_csv}")
    
    df = pd.read_csv(severity_csv)
    
    # Find the patient ID column (may be named differently)
    id_col = None
    for col in df.columns:
        if 'patient' in col.lower() or 'id' in col.lower() or 'case' in col.lower():
            id_col = col
            break
    
    if id_col is None:
        raise ValueError(f"Could not find patient ID column in {severity_csv}")
    
    # Find severity column
    severity_col = None
    for col in df.columns:
        if 'severity' in col.lower():
            severity_col = col
            break
    
    if severity_col is None:
        raise ValueError(f"Could not find severity column in {severity_csv}")
    
    # Build map: extract patX from patient ID if needed
    severity_map = {}
    for _, row in df.iterrows():
        patient_str = str(row[id_col]).lower()
        
        # Extract patX format
        import re
        match = re.search(r'pat(\d+)', patient_str)
        if match:
            case_id = f"pat{match.group(1)}"
        else:
            # Try direct use
            case_id = patient_str.strip()
        
        severity_map[case_id] = str(row[severity_col]).strip()
    
    return severity_map


def wide_to_long(
    wide_df: pd.DataFrame,
    severity_map: Dict[str, str],
    num_classes: int = 8
) -> pd.DataFrame:
    """
    Convert wide format (one row per case, columns per structure) to long format
    (one row per case-structure-metric combination).
    
    Wide format columns example:
        case_id, budget, seed, struct_1_dice, struct_1_hd95, struct_1_gt_voxels, ...
    
    Long format columns:
        case_id, budget, seed, severity, structure_id, metric, value, gt_voxels, pred_voxels
    """
    rows = []
    
    for _, case_row in wide_df.iterrows():
        case_id = case_row['case_id']
        budget = case_row['budget']
        seed = case_row['seed']
        severity = severity_map.get(case_id, 'unknown')
        
        for structure_id in range(1, num_classes + 1):
            # Extract metrics for this structure
            dice_col = f'struct_{structure_id}_dice'
            hd95_col = f'struct_{structure_id}_hd95'
            gt_vox_col = f'struct_{structure_id}_gt_voxels'
            pred_vox_col = f'struct_{structure_id}_pred_voxels'
            
            if dice_col not in case_row:
                continue
            
            dice_val = case_row[dice_col]
            hd95_val = case_row[hd95_col]
            gt_voxels = case_row.get(gt_vox_col, 0)
            pred_voxels = case_row.get(pred_vox_col, 0)
            
            # Add two rows: one for Dice, one for HD95
            rows.append({
                'case_id': case_id,
                'budget': budget,
                'seed': seed,
                'severity': severity,
                'structure_id': structure_id,
                'metric': 'dice',
                'value': dice_val,
                'gt_voxels': gt_voxels,
                'pred_voxels': pred_voxels,
            })
            
            rows.append({
                'case_id': case_id,
                'budget': budget,
                'seed': seed,
                'severity': severity,
                'structure_id': structure_id,
                'metric': 'hd95',
                'value': hd95_val,
                'gt_voxels': gt_voxels,
                'pred_voxels': pred_voxels,
            })
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Convert wide per-case CSV to long/tidy format"
    )
    parser.add_argument("--wide_csv", type=str, required=True,
                        help="Input wide CSV file")
    parser.add_argument("--severity_csv", type=str, required=True,
                        help="Clinical CSV with severity information")
    parser.add_argument("--out_csv", type=str, required=True,
                        help="Output long/tidy CSV file")
    parser.add_argument("--num_classes", type=int, default=8,
                        help="Number of anatomical structures (default: 8)")
    
    args = parser.parse_args()
    
    wide_csv = Path(args.wide_csv)
    severity_csv = Path(args.severity_csv)
    out_csv = Path(args.out_csv)
    
    if not wide_csv.exists():
        raise FileNotFoundError(f"Wide CSV not found: {wide_csv}")
    
    print(f"Loading wide CSV: {wide_csv}")
    wide_df = pd.read_csv(wide_csv)
    print(f"  Cases: {len(wide_df)}")
    
    print(f"Loading severity map: {severity_csv}")
    severity_map = load_severity_map(severity_csv)
    print(f"  Severity entries: {len(severity_map)}")
    
    print("Converting to long format...")
    long_df = wide_to_long(wide_df, severity_map, num_classes=args.num_classes)
    print(f"  Output rows: {len(long_df)}")
    
    # Save
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out_csv, index=False)
    
    print(f"\nLong CSV saved to: {out_csv}")
    
    # Summary
    print("\n=== SUMMARY ===")
    print(f"Unique cases: {long_df['case_id'].nunique()}")
    print(f"Budgets: {long_df['budget'].unique().tolist()}")
    print(f"Seeds: {sorted(long_df['seed'].unique().tolist())}")
    print(f"Severities: {long_df['severity'].unique().tolist()}")
    print(f"Structures: {sorted(long_df['structure_id'].unique().tolist())}")
    print(f"Metrics: {long_df['metric'].unique().tolist()}")


if __name__ == "__main__":
    main()
