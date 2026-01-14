"""
Generate comprehensive figures for label-budget analysis:
1. Label-budget scaling curves (Dice and HD95)
2. Per-structure learning curves
3. Structure ranking by learning speed (AULC, budget-to-90%)
4. Severity-stratified performance
5. Qualitative overlays (best/median/worst cases)

Usage:
    python generate_plots.py \\
        --long_csvs results/*_long.csv \\
        --pred_dirs data/nnunet/predictions \\
        --gt_dir data/nnunet/nnUNet_raw/Dataset905_HVSMR_L5/labelsTs \\
        --out_dir figures \\
        --num_classes 8
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches
from scipy import integrate

# Set style
sns.set_style("whitegrid")
sns.set_palette("colorblind")
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['font.size'] = 10


def load_all_data(csv_patterns: List[str]) -> pd.DataFrame:
    """Load and combine all long CSV files."""
    dfs = []
    for pattern in csv_patterns:
        csv_files = list(Path().glob(pattern))
        if not csv_files:
            csv_path = Path(pattern)
            if csv_path.exists():
                csv_files = [csv_path]
        
        for csv_file in csv_files:
            print(f"Loading: {csv_file}")
            df = pd.read_csv(csv_file)
            dfs.append(df)
    
    if not dfs:
        raise ValueError("No CSV files found")
    
    return pd.concat(dfs, ignore_index=True)


def extract_budget_number(budget_str: str) -> int:
    """Extract numeric value from budget string (e.g., 'L5' -> 5)."""
    import re
    match = re.search(r'(\d+)', budget_str)
    if match:
        return int(match.group(1))
    return 0


def plot_scaling_curves(df: pd.DataFrame, out_dir: Path, metric_name: str = 'dice'):
    """
    Plot label-budget scaling curves showing how performance improves with more labels.
    
    Creates:
    - Overall (foreground average) curve
    - Per-structure curves on separate subplots
    """
    df_metric = df[df['metric'] == metric_name].copy()
    df_metric['budget_num'] = df_metric['budget'].apply(extract_budget_number)
    
    # Overall scaling curve (foreground average)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Compute case-level averages, then aggregate
    case_means = df_metric.groupby(['budget', 'budget_num', 'case_id'])['value'].mean().reset_index()
    budget_stats = case_means.groupby(['budget', 'budget_num'])['value'].agg(['mean', 'std', 'count']).reset_index()
    budget_stats['sem'] = budget_stats['std'] / np.sqrt(budget_stats['count'])
    budget_stats = budget_stats.sort_values('budget_num')
    
    ax.plot(budget_stats['budget_num'], budget_stats['mean'], marker='o', linewidth=2, markersize=8)
    ax.fill_between(
        budget_stats['budget_num'],
        budget_stats['mean'] - 1.96 * budget_stats['sem'],
        budget_stats['mean'] + 1.96 * budget_stats['sem'],
        alpha=0.2
    )
    
    ax.set_xlabel('Number of Labeled Training Cases', fontsize=12)
    ylabel = 'Dice Coefficient' if metric_name == 'dice' else 'Hausdorff Distance 95 (mm)'
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'{ylabel} vs Label Budget (Foreground Average)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if metric_name == 'dice':
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(out_dir / f'scaling_curve_{metric_name}_overall.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: scaling_curve_{metric_name}_overall.png")
    
    # Per-structure scaling curves
    structures = sorted(df_metric['structure_id'].unique())
    n_structs = len(structures)
    
    ncols = 4
    nrows = int(np.ceil(n_structs / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten() if n_structs > 1 else [axes]
    
    for idx, struct_id in enumerate(structures):
        ax = axes[idx]
        
        struct_df = df_metric[df_metric['structure_id'] == struct_id]
        struct_stats = struct_df.groupby(['budget', 'budget_num'])['value'].agg(['mean', 'std', 'count']).reset_index()
        struct_stats['sem'] = struct_stats['std'] / np.sqrt(struct_stats['count'])
        struct_stats = struct_stats.sort_values('budget_num')
        
        ax.plot(struct_stats['budget_num'], struct_stats['mean'], marker='o', linewidth=2)
        ax.fill_between(
            struct_stats['budget_num'],
            struct_stats['mean'] - 1.96 * struct_stats['sem'],
            struct_stats['mean'] + 1.96 * struct_stats['sem'],
            alpha=0.2
        )
        
        ax.set_title(f'Structure {struct_id}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Label Budget')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        
        if metric_name == 'dice':
            ax.set_ylim([0, 1])
    
    # Hide empty subplots
    for idx in range(n_structs, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f'{ylabel} vs Label Budget (Per Structure)', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(out_dir / f'scaling_curve_{metric_name}_per_structure.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: scaling_curve_{metric_name}_per_structure.png")


def compute_aulc(budget_nums: np.ndarray, performance: np.ndarray) -> float:
    """Compute Area Under Learning Curve (normalized by budget range)."""
    if len(budget_nums) < 2:
        return 0.0
    
    # Sort by budget
    sorted_idx = np.argsort(budget_nums)
    x = budget_nums[sorted_idx]
    y = performance[sorted_idx]
    
    # Trapezoidal integration
    auc = integrate.trapezoid(y, x)
    
    # Normalize by budget range
    budget_range = x[-1] - x[0]
    if budget_range > 0:
        auc_normalized = auc / budget_range
    else:
        auc_normalized = y[0]
    
    return float(auc_normalized)


def compute_budget_to_threshold(
    budget_nums: np.ndarray,
    performance: np.ndarray,
    threshold_pct: float = 0.9
) -> float:
    """
    Compute budget needed to reach threshold_pct of final performance.
    
    Returns:
        Budget value, or NaN if threshold never reached.
    """
    if len(budget_nums) < 2:
        return np.nan
    
    sorted_idx = np.argsort(budget_nums)
    x = budget_nums[sorted_idx]
    y = performance[sorted_idx]
    
    final_perf = y[-1]
    threshold = threshold_pct * final_perf
    
    # Find first budget where performance >= threshold
    above_threshold = y >= threshold
    if not np.any(above_threshold):
        return np.nan
    
    first_idx = np.where(above_threshold)[0][0]
    return float(x[first_idx])


def plot_structure_rankings(df: pd.DataFrame, out_dir: Path):
    """
    Rank structures by learning speed using:
    1. AULC (Area Under Learning Curve)
    2. Budget to reach 90% of final performance
    """
    df_dice = df[df['metric'] == 'dice'].copy()
    df_dice['budget_num'] = df_dice['budget'].apply(extract_budget_number)
    
    structures = sorted(df_dice['structure_id'].unique())
    
    aulc_scores = []
    budget_to_90_scores = []
    
    for struct_id in structures:
        struct_df = df_dice[df_dice['structure_id'] == struct_id]
        
        # Get mean performance at each budget
        budget_perf = struct_df.groupby('budget_num')['value'].mean()
        budget_nums = budget_perf.index.values
        performance = budget_perf.values
        
        # AULC
        aulc = compute_aulc(budget_nums, performance)
        aulc_scores.append(aulc)
        
        # Budget to 90%
        b90 = compute_budget_to_threshold(budget_nums, performance, threshold_pct=0.9)
        budget_to_90_scores.append(b90)
    
    # Create ranking DataFrame
    ranking_df = pd.DataFrame({
        'structure_id': structures,
        'aulc': aulc_scores,
        'budget_to_90': budget_to_90_scores,
    })
    
    ranking_df = ranking_df.sort_values('aulc', ascending=False)
    
    # Save table
    ranking_df.to_csv(out_dir / 'structure_learning_rankings.csv', index=False)
    
    # Plot AULC ranking
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # AULC
    colors = sns.color_palette("RdYlGn", len(structures))
    ax1.barh(range(len(structures)), ranking_df['aulc'], color=colors[::-1])
    ax1.set_yticks(range(len(structures)))
    ax1.set_yticklabels([f"Struct {sid}" for sid in ranking_df['structure_id']])
    ax1.set_xlabel('AULC (Area Under Learning Curve)', fontsize=12)
    ax1.set_title('Structures Ranked by Learning Speed (AULC)', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Budget to 90%
    ranking_df_b90 = ranking_df.dropna(subset=['budget_to_90']).sort_values('budget_to_90')
    colors_b90 = sns.color_palette("RdYlGn_r", len(ranking_df_b90))
    
    ax2.barh(range(len(ranking_df_b90)), ranking_df_b90['budget_to_90'], color=colors_b90)
    ax2.set_yticks(range(len(ranking_df_b90)))
    ax2.set_yticklabels([f"Struct {sid}" for sid in ranking_df_b90['structure_id']])
    ax2.set_xlabel('Budget to Reach 90% of Final Performance', fontsize=12)
    ax2.set_title('Structures by Budget Efficiency', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'structure_learning_rankings.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: structure_learning_rankings.png")
    print("\nStructure Learning Rankings (fastest to slowest):")
    for _, row in ranking_df.iterrows():
        print(f"  Structure {row['structure_id']}: AULC={row['aulc']:.3f}, "
              f"Budget-to-90%={row['budget_to_90']:.1f}")


def plot_severity_stratified(df: pd.DataFrame, out_dir: Path):
    """
    Plot performance stratified by severity level.
    """
    df_dice = df[df['metric'] == 'dice'].copy()
    df_hd95 = df[df['metric'] == 'hd95'].copy()
    
    df_dice['budget_num'] = df_dice['budget'].apply(extract_budget_number)
    df_hd95['budget_num'] = df_hd95['budget'].apply(extract_budget_number)
    
    severities = sorted(df_dice['severity'].unique())
    
    # Dice by severity
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Dice
    ax = axes[0]
    for severity in severities:
        sev_df = df_dice[df_dice['severity'] == severity]
        
        # Case-level average, then budget-level stats
        case_means = sev_df.groupby(['budget_num', 'case_id'])['value'].mean().reset_index()
        budget_stats = case_means.groupby('budget_num')['value'].agg(['mean', 'std', 'count']).reset_index()
        budget_stats['sem'] = budget_stats['std'] / np.sqrt(budget_stats['count'])
        
        ax.plot(budget_stats['budget_num'], budget_stats['mean'], marker='o', label=severity, linewidth=2)
        ax.fill_between(
            budget_stats['budget_num'],
            budget_stats['mean'] - 1.96 * budget_stats['sem'],
            budget_stats['mean'] + 1.96 * budget_stats['sem'],
            alpha=0.15
        )
    
    ax.set_xlabel('Label Budget', fontsize=12)
    ax.set_ylabel('Dice Coefficient', fontsize=12)
    ax.set_title('Dice by Severity', fontsize=13, fontweight='bold')
    ax.legend(title='Severity')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # HD95
    ax = axes[1]
    for severity in severities:
        sev_df = df_hd95[df_hd95['severity'] == severity]
        sev_df = sev_df[sev_df['value'].notna()]  # Exclude NaN
        
        case_means = sev_df.groupby(['budget_num', 'case_id'])['value'].mean().reset_index()
        budget_stats = case_means.groupby('budget_num')['value'].agg(['mean', 'std', 'count']).reset_index()
        budget_stats['sem'] = budget_stats['std'] / np.sqrt(budget_stats['count'])
        
        ax.plot(budget_stats['budget_num'], budget_stats['mean'], marker='o', label=severity, linewidth=2)
        ax.fill_between(
            budget_stats['budget_num'],
            budget_stats['mean'] - 1.96 * budget_stats['sem'],
            budget_stats['mean'] + 1.96 * budget_stats['sem'],
            alpha=0.15
        )
    
    ax.set_xlabel('Label Budget', fontsize=12)
    ax.set_ylabel('Hausdorff Distance 95 (mm)', fontsize=12)
    ax.set_title('HD95 by Severity', fontsize=13, fontweight='bold')
    ax.legend(title='Severity')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'severity_stratified_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: severity_stratified_performance.png")


def create_qualitative_overlays(
    df: pd.DataFrame,
    pred_dirs: Path,
    gt_dir: Path,
    out_dir: Path,
    budget: str = 'L40',
    seed: int = 0
):
    """
    Create qualitative overlay visualizations for best/median/worst cases.
    
    Selects cases based on overall Dice score.
    """
    # Filter to specific budget and seed
    df_budget = df[(df['budget'] == budget) & (df['seed'] == seed) & (df['metric'] == 'dice')].copy()
    
    # Compute case-level average Dice
    case_dice = df_budget.groupby('case_id')['value'].mean().reset_index()
    case_dice = case_dice.sort_values('value')
    
    if len(case_dice) == 0:
        print(f"No data found for budget={budget}, seed={seed}")
        return
    
    # Select best, median, worst
    worst_case = case_dice.iloc[0]['case_id']
    median_case = case_dice.iloc[len(case_dice) // 2]['case_id']
    best_case = case_dice.iloc[-1]['case_id']
    
    selected_cases = {
        'worst': (worst_case, case_dice.iloc[0]['value']),
        'median': (median_case, case_dice.iloc[len(case_dice) // 2]['value']),
        'best': (best_case, case_dice.iloc[-1]['value']),
    }
    
    print(f"\nQualitative overlays for {budget}, seed {seed}:")
    for label, (case_id, dice) in selected_cases.items():
        print(f"  {label.capitalize()}: {case_id} (Dice={dice:.3f})")
    
    # Create overlays
    pred_dir = pred_dirs / budget / f"seed{seed}"
    if not pred_dir.exists():
        # Try without seed subdirectory
        pred_dir = pred_dirs / budget
    
    for label, (case_id, dice) in selected_cases.items():
        try:
            create_overlay_figure(
                case_id, pred_dir, gt_dir, out_dir, f"{budget}_seed{seed}_{label}", dice
            )
        except Exception as e:
            print(f"Warning: Could not create overlay for {case_id}: {e}")


def create_overlay_figure(
    case_id: str,
    pred_dir: Path,
    gt_dir: Path,
    out_dir: Path,
    filename_prefix: str,
    dice_score: float
):
    """Create a figure showing GT and prediction overlay for a single case."""
    # Load volumes
    gt_path = gt_dir / f"{case_id}.nii.gz"
    pred_path = pred_dir / f"{case_id}.nii.gz"
    
    if not gt_path.exists() or not pred_path.exists():
        print(f"Missing files for {case_id}")
        return
    
    gt_img = nib.load(str(gt_path))
    pred_img = nib.load(str(pred_path))
    
    gt_data = np.asarray(gt_img.dataobj).astype(np.int16)
    pred_data = np.asarray(pred_img.dataobj).astype(np.int16)
    
    # Transpose to (z, y, x)
    gt_data = np.transpose(gt_data, (2, 1, 0))
    pred_data = np.transpose(pred_data, (2, 1, 0))
    
    # Select middle slices
    z_mid = gt_data.shape[0] // 2
    y_mid = gt_data.shape[1] // 2
    x_mid = gt_data.shape[2] // 2
    
    # Create figure with 3 views
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    views = [
        ('Axial', gt_data[z_mid, :, :], pred_data[z_mid, :, :]),
        ('Coronal', gt_data[:, y_mid, :], pred_data[:, y_mid, :]),
        ('Sagittal', gt_data[:, :, x_mid], pred_data[:, :, x_mid]),
    ]
    
    for ax, (view_name, gt_slice, pred_slice) in zip(axes, views):
        # Create RGB overlay
        # GT = green, Pred = red, Overlap = yellow
        overlay = np.zeros((*gt_slice.shape, 3))
        
        # Red channel: prediction
        overlay[pred_slice > 0, 0] = 1.0
        # Green channel: GT
        overlay[gt_slice > 0, 1] = 1.0
        
        ax.imshow(overlay)
        ax.set_title(f"{view_name} View", fontsize=12)
        ax.axis('off')
    
    fig.suptitle(f"Case: {case_id} (Dice={dice_score:.3f})\nGreen=GT, Red=Pred, Yellow=Overlap",
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    out_path = out_dir / f"overlay_{filename_prefix}_{case_id}.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive analysis figures")
    parser.add_argument("--long_csvs", type=str, nargs='+', required=True,
                        help="Input long CSV files (can use wildcards)")
    parser.add_argument("--pred_dirs", type=str, required=True,
                        help="Root directory containing predictions (organized by budget/seed)")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Directory containing ground truth labels")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for figures")
    parser.add_argument("--num_classes", type=int, default=8,
                        help="Number of anatomical structures (default: 8)")
    
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pred_dirs = Path(args.pred_dirs)
    gt_dir = Path(args.gt_dir)
    
    # Load data
    print("Loading data...")
    df = load_all_data(args.long_csvs)
    print(f"Loaded {len(df)} rows from {df['case_id'].nunique()} cases")
    
    # 1. Scaling curves
    print("\n=== Generating scaling curves ===")
    plot_scaling_curves(df, out_dir, metric_name='dice')
    plot_scaling_curves(df, out_dir, metric_name='hd95')
    
    # 2. Structure rankings
    print("\n=== Generating structure rankings ===")
    plot_structure_rankings(df, out_dir)
    
    # 3. Severity-stratified
    print("\n=== Generating severity-stratified plots ===")
    plot_severity_stratified(df, out_dir)
    
    # 4. Qualitative overlays
    print("\n=== Generating qualitative overlays ===")
    for budget in ['L5', 'L10', 'L20', 'L40']:
        for seed in [0, 1, 2]:
            if budget in ['L20', 'L40'] and seed > 1:
                continue  # Only 2 seeds for L20/L40
            
            create_qualitative_overlays(df, pred_dirs, gt_dir, out_dir, budget=budget, seed=seed)
    
    print(f"\n=== All figures saved to {out_dir}/ ===")


if __name__ == "__main__":
    main()
