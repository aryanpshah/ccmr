"""
Extract GPU and compute statistics from nnU-Net training logs.

Extracts:
- Training wall-clock time (start to finish)
- Epochs trained and best epoch
- Peak GPU memory (if logged)
- Inference time per volume

Usage:
    python extract_compute_stats.py \\
        --log_file logs/nnunet/nnunet_L5_fold0_seed0.log \\
        --results_dir data/nnunet/nnUNet_results/Dataset905_HVSMR_L5/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0 \\
        --budget L5 \\
        --seed 0 \\
        --out_csv results/compute_stats.csv
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def parse_timestamp(line: str) -> Optional[datetime]:
    """
    Extract timestamp from log line.
    
    Expected formats:
    - 2025-11-28 22:07:13
    - [2025-11-28 22:07:13]
    """
    patterns = [
        r'\[?(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\]?',
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, line)
        if match:
            try:
                timestamp_str = match.group(1).replace('T', ' ')
                return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                continue
    
    return None


def extract_from_log(log_path: Path) -> Dict[str, any]:
    """Extract training statistics from log file."""
    if not log_path.exists():
        print(f"Warning: Log file not found: {log_path}")
        return {}
    
    stats = {
        'start_time': None,
        'end_time': None,
        'total_epochs': None,
        'best_epoch': None,
        'gpu_name': None,
        'peak_memory_gb': None,
    }
    
    lines = log_path.read_text(encoding='utf-8', errors='ignore').splitlines()
    
    for line in lines:
        # Extract timestamps
        timestamp = parse_timestamp(line)
        
        if timestamp:
            if stats['start_time'] is None:
                stats['start_time'] = timestamp
            stats['end_time'] = timestamp  # Keep updating to last timestamp
        
        # GPU name
        if 'Using device' in line or 'cuda' in line.lower():
            # Try to extract GPU name
            gpu_match = re.search(r'(NVIDIA|GeForce|Tesla|RTX|GTX)\s+[\w\s]+', line, re.IGNORECASE)
            if gpu_match and stats['gpu_name'] is None:
                stats['gpu_name'] = gpu_match.group(0).strip()
        
        # Peak memory
        if 'memory' in line.lower() and ('peak' in line.lower() or 'max' in line.lower()):
            # Try to extract memory value in GB or MB
            mem_match = re.search(r'(\d+\.?\d*)\s*(GB|MB)', line, re.IGNORECASE)
            if mem_match:
                value = float(mem_match.group(1))
                unit = mem_match.group(2).upper()
                if unit == 'MB':
                    value /= 1024
                stats['peak_memory_gb'] = value
        
        # Epoch information
        if 'Epoch' in line or 'epoch' in line:
            epoch_match = re.search(r'[Ee]poch\s+(\d+)', line)
            if epoch_match:
                epoch_num = int(epoch_match.group(1))
                if stats['total_epochs'] is None or epoch_num > stats['total_epochs']:
                    stats['total_epochs'] = epoch_num
        
        # Best epoch
        if 'best' in line.lower() and 'epoch' in line.lower():
            best_match = re.search(r'[Ee]poch\s+(\d+)', line)
            if best_match:
                stats['best_epoch'] = int(best_match.group(1))
    
    return stats


def extract_from_progress_pkl(results_dir: Path) -> Dict[str, any]:
    """Extract information from nnU-Net progress.pkl file."""
    progress_file = results_dir / 'progress.pkl'
    
    if not progress_file.exists():
        print(f"Warning: progress.pkl not found: {progress_file}")
        return {}
    
    try:
        with open(progress_file, 'rb') as f:
            progress = pickle.load(f)
        
        stats = {
            'total_epochs': len(progress.get('train_losses', [])),
            'best_epoch': progress.get('best_epoch'),
            'best_pseudo_dice': progress.get('best_ema_pseudo_dice'),
        }
        
        return stats
    
    except Exception as e:
        print(f"Warning: Could not load progress.pkl: {e}")
        return {}


def extract_from_training_log_json(results_dir: Path) -> Dict[str, any]:
    """Extract from training_log.txt (nnU-Net v2 format)."""
    training_log = results_dir / 'training_log.txt'
    
    if not training_log.exists():
        return {}
    
    try:
        lines = training_log.read_text(encoding='utf-8').strip().splitlines()
        
        # Last line usually has final epoch info
        if lines:
            last_line = lines[-1]
            
            # Try to parse as JSON
            try:
                data = json.loads(last_line)
                return {
                    'total_epochs': data.get('epoch'),
                }
            except json.JSONDecodeError:
                pass
    
    except Exception as e:
        print(f"Warning: Could not read training_log.txt: {e}")
    
    return {}


def estimate_inference_time(
    pred_dir: Path,
    num_cases: int,
    log_path: Optional[Path] = None
) -> Optional[float]:
    """
    Estimate inference time per volume.
    
    Can extract from:
    1. Inference log file (if available)
    2. File timestamps (time between first and last prediction)
    """
    if log_path and log_path.exists():
        # Try to extract from log
        lines = log_path.read_text(encoding='utf-8', errors='ignore').splitlines()
        
        for line in lines:
            # Look for timing information
            if 'inference' in line.lower() and 'time' in line.lower():
                # Try to extract seconds per case
                time_match = re.search(r'(\d+\.?\d*)\s*s(?:ec)?(?:/case|/volume)?', line, re.IGNORECASE)
                if time_match:
                    return float(time_match.group(1))
    
    # Fallback: use file timestamps
    if pred_dir.exists():
        nii_files = sorted(pred_dir.glob('*.nii.gz'))
        
        if len(nii_files) >= 2:
            first_time = nii_files[0].stat().st_mtime
            last_time = nii_files[-1].stat().st_mtime
            
            total_time = last_time - first_time
            
            if num_cases > 0:
                return total_time / num_cases
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract GPU and compute statistics from nnU-Net logs"
    )
    parser.add_argument("--log_file", type=str, required=True,
                        help="Training log file")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="nnU-Net results directory (fold_X)")
    parser.add_argument("--budget", type=str, required=True,
                        help="Label budget (e.g., L5)")
    parser.add_argument("--seed", type=int, required=True,
                        help="Random seed")
    parser.add_argument("--out_csv", type=str, required=True,
                        help="Output CSV file")
    parser.add_argument("--pred_dir", type=str, default=None,
                        help="Prediction directory (for inference time estimation)")
    parser.add_argument("--num_test_cases", type=int, default=10,
                        help="Number of test cases (for inference time)")
    
    args = parser.parse_args()
    
    log_path = Path(args.log_file)
    results_dir = Path(args.results_dir)
    out_csv = Path(args.out_csv)
    
    # Extract from various sources
    print(f"Extracting stats for {args.budget}, seed {args.seed}")
    
    print(f"  Reading log: {log_path}")
    log_stats = extract_from_log(log_path)
    
    print(f"  Reading progress.pkl: {results_dir}")
    pkl_stats = extract_from_progress_pkl(results_dir)
    
    training_log_stats = extract_from_training_log_json(results_dir)
    
    # Merge stats (pkl takes precedence for epoch info)
    stats = {**log_stats, **training_log_stats, **pkl_stats}
    
    # Compute wall-clock time
    if stats.get('start_time') and stats.get('end_time'):
        delta = stats['end_time'] - stats['start_time']
        stats['wall_hours'] = delta.total_seconds() / 3600
    else:
        stats['wall_hours'] = None
    
    # Inference time
    if args.pred_dir:
        pred_dir = Path(args.pred_dir)
        inference_time = estimate_inference_time(pred_dir, args.num_test_cases)
        stats['inference_sec_per_vol'] = inference_time
    else:
        stats['inference_sec_per_vol'] = None
    
    # Create row
    row = {
        'budget': args.budget,
        'seed': args.seed,
        'wall_hours': stats.get('wall_hours'),
        'total_epochs': stats.get('total_epochs'),
        'best_epoch': stats.get('best_epoch'),
        'gpu_name': stats.get('gpu_name', 'Unknown'),
        'peak_memory_gb': stats.get('peak_memory_gb'),
        'inference_sec_per_vol': stats.get('inference_sec_per_vol'),
    }
    
    # Save or append
    if out_csv.exists():
        df_existing = pd.read_csv(out_csv)
        df_new = pd.DataFrame([row])
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    
    print(f"\nStats saved to: {out_csv}")
    print("\nExtracted statistics:")
    for key, value in row.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
