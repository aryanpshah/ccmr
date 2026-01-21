"""
Aggregate per-case metrics and compute summary statistics (mean, CI) for each budget level.

Uses bootstrap resampling for robust confidence intervals when sample sizes are small.

Outputs:
- Summary table with mean ± 95% CI for Dice and HD95 per structure per budget
- Overall (foreground average) metrics
- LaTeX-formatted tables for publication

Usage:
    python aggregate_metrics.py \\
        --input_csvs results/*_long.csv \\
        --out_dir results/tables
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy import stats


def compute_bootstrap_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000
) -> tuple[float, float, float]:
    """
    Compute mean and confidence interval using bootstrap resampling.
    
    Bootstrap is more robust for small sample sizes and doesn't assume normality.
    
    Returns:
        (mean, lower_bound, upper_bound)
    """
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    
    mean = np.mean(values)
    
    if len(values) == 1:
        return mean, mean, mean
    
    # Bootstrap resampling
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        resample = rng.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(resample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Compute percentile-based confidence intervals
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return mean, ci_lower, ci_upper


def compute_confidence_interval(
    values: np.ndarray,
    confidence: float = 0.95
) -> tuple[float, float, float]:
    """
    Compute mean and confidence interval using bootstrap resampling.
    
    Wrapper function that calls bootstrap CI computation.
    
    Returns:
        (mean, lower_bound, upper_bound)
    """
    return compute_bootstrap_ci(values, confidence=confidence)


def aggregate_by_budget_structure(
    df_long: pd.DataFrame,
    metric_name: str
) -> pd.DataFrame:
    """
    Aggregate metrics by budget and structure.
    
    Returns DataFrame with columns:
        budget, structure_id, mean, ci_lower, ci_upper, n_cases
    """
    df_metric = df_long[df_long['metric'] == metric_name].copy()
    
    # Exclude NaN values for HD95
    df_metric = df_metric[df_metric['value'].notna()]
    
    results = []
    
    for budget in sorted(df_metric['budget'].unique()):
        for structure_id in sorted(df_metric['structure_id'].unique()):
            mask = (df_metric['budget'] == budget) & (df_metric['structure_id'] == structure_id)
            values = df_metric[mask]['value'].values
            
            if len(values) == 0:
                continue
            
            mean, ci_lower, ci_upper = compute_confidence_interval(values)
            
            results.append({
                'budget': budget,
                'structure_id': structure_id,
                'mean': mean,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'std': np.std(values, ddof=1) if len(values) > 1 else 0.0,
                'n_cases': len(values),
            })
    
    return pd.DataFrame(results)


def aggregate_foreground_average(
    df_long: pd.DataFrame,
    metric_name: str
) -> pd.DataFrame:
    """
    Compute foreground-averaged metrics (average across all structures per case,
    then average across cases).
    
    Returns DataFrame with columns:
        budget, mean, ci_lower, ci_upper, n_cases
    """
    df_metric = df_long[df_long['metric'] == metric_name].copy()
    
    # Exclude NaN for HD95
    df_metric = df_metric[df_metric['value'].notna()]
    
    results = []
    
    for budget in sorted(df_metric['budget'].unique()):
        budget_df = df_metric[df_metric['budget'] == budget]
        
        # Average per case across structures
        case_means = budget_df.groupby('case_id')['value'].mean()
        
        mean, ci_lower, ci_upper = compute_confidence_interval(case_means.values)
        
        results.append({
            'budget': budget,
            'mean': mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'std': np.std(case_means.values, ddof=1) if len(case_means) > 1 else 0.0,
            'n_cases': len(case_means),
        })
    
    return pd.DataFrame(results)


def format_mean_ci(mean: float, ci_lower: float, ci_upper: float, metric: str) -> str:
    """Format mean ± CI for display."""
    if metric == 'dice':
        # Dice: 0.XXX ± 0.XXX format
        margin = (ci_upper - ci_lower) / 2
        return f"{mean:.3f} ± {margin:.3f}"
    else:
        # HD95: X.XX ± X.XX mm
        margin = (ci_upper - ci_lower) / 2
        return f"{mean:.2f} ± {margin:.2f}"


def create_latex_table(
    df_dice: pd.DataFrame,
    df_hd95: pd.DataFrame,
    num_structures: int = 8
) -> str:
    """
    Create a LaTeX table with Dice and HD95 for all budgets and structures.
    """
    budgets = sorted(df_dice['budget'].unique())
    
    # Header
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\caption{Per-structure metrics across label budgets (mean ± 95\\% CI)}\n"
    latex += "\\begin{tabular}{l" + "c" * len(budgets) + "}\n"
    latex += "\\toprule\n"
    latex += "Structure & " + " & ".join(budgets) + " \\\\\n"
    latex += "\\midrule\n"
    
    # Dice section
    latex += "\\multicolumn{" + str(len(budgets) + 1) + "}{l}{\\textbf{Dice Coefficient}} \\\\\n"
    
    for struct_id in range(1, num_structures + 1):
        row = f"Structure {struct_id}"
        
        for budget in budgets:
            mask = (df_dice['budget'] == budget) & (df_dice['structure_id'] == struct_id)
            if mask.any():
                row_data = df_dice[mask].iloc[0]
                formatted = format_mean_ci(
                    row_data['mean'], row_data['ci_lower'], row_data['ci_upper'], 'dice'
                )
                row += f" & {formatted}"
            else:
                row += " & ---"
        
        row += " \\\\\n"
        latex += row
    
    # Foreground average
    latex += "\\midrule\n"
    latex += "Foreground Avg."
    
    df_dice_fg = aggregate_foreground_average(df_dice, 'dice')  # Need to pass full data
    # This is a placeholder - would need full data passed in
    latex += " & " + " & ".join(["---"] * len(budgets)) + " \\\\\n"
    
    # HD95 section
    latex += "\\midrule\n"
    latex += "\\multicolumn{" + str(len(budgets) + 1) + "}{l}{\\textbf{Hausdorff Distance 95 (mm)}} \\\\\n"
    
    for struct_id in range(1, num_structures + 1):
        row = f"Structure {struct_id}"
        
        for budget in budgets:
            mask = (df_hd95['budget'] == budget) & (df_hd95['structure_id'] == struct_id)
            if mask.any():
                row_data = df_hd95[mask].iloc[0]
                formatted = format_mean_ci(
                    row_data['mean'], row_data['ci_lower'], row_data['ci_upper'], 'hd95'
                )
                row += f" & {formatted}"
            else:
                row += " & ---"
        
        row += " \\\\\n"
        latex += row
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-case metrics and compute summary statistics"
    )
    parser.add_argument("--input_csvs", type=str, nargs='+', required=True,
                        help="Input long CSV files (can use wildcards)")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for tables")
    parser.add_argument("--num_classes", type=int, default=8,
                        help="Number of anatomical structures (default: 8)")
    
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all CSVs
    dfs = []
    for csv_pattern in args.input_csvs:
        csv_files = list(Path().glob(csv_pattern))
        if not csv_files:
            # Try as direct path
            csv_path = Path(csv_pattern)
            if csv_path.exists():
                csv_files = [csv_path]
        
        for csv_file in csv_files:
            print(f"Loading: {csv_file}")
            df = pd.read_csv(csv_file)
            dfs.append(df)
    
    if not dfs:
        raise ValueError("No CSV files found matching the input pattern")
    
    # Combine all data
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows: {len(df_all)}")
    print(f"Unique cases: {df_all['case_id'].nunique()}")
    print(f"Budgets: {sorted(df_all['budget'].unique())}")
    print(f"Seeds: {sorted(df_all['seed'].unique())}")
    
    # Aggregate metrics
    print("\nAggregating Dice metrics...")
    df_dice_agg = aggregate_by_budget_structure(df_all, 'dice')
    
    print("Aggregating HD95 metrics...")
    df_hd95_agg = aggregate_by_budget_structure(df_all, 'hd95')
    
    print("Aggregating foreground averages...")
    df_dice_fg = aggregate_foreground_average(df_all, 'dice')
    df_hd95_fg = aggregate_foreground_average(df_all, 'hd95')
    
    # Save tables
    df_dice_agg.to_csv(out_dir / 'dice_by_structure.csv', index=False)
    df_hd95_agg.to_csv(out_dir / 'hd95_by_structure.csv', index=False)
    df_dice_fg.to_csv(out_dir / 'dice_foreground_avg.csv', index=False)
    df_hd95_fg.to_csv(out_dir / 'hd95_foreground_avg.csv', index=False)
    
    print(f"\nSaved tables to {out_dir}/")
    
    # Create human-readable summary
    print("\n=== FOREGROUND AVERAGE RESULTS ===")
    print("\nDice Coefficient:")
    for _, row in df_dice_fg.iterrows():
        formatted = format_mean_ci(row['mean'], row['ci_lower'], row['ci_upper'], 'dice')
        print(f"  {row['budget']}: {formatted}")
    
    print("\nHausdorff Distance 95 (mm):")
    for _, row in df_hd95_fg.iterrows():
        formatted = format_mean_ci(row['mean'], row['ci_lower'], row['ci_upper'], 'hd95')
        print(f"  {row['budget']}: {formatted}")
    
    # Generate LaTeX table
    print("\nGenerating LaTeX table...")
    latex_table = create_latex_table(df_dice_agg, df_hd95_agg, num_structures=args.num_classes)
    latex_path = out_dir / 'summary_table.tex'
    latex_path.write_text(latex_table)
    print(f"LaTeX table saved to: {latex_path}")


if __name__ == "__main__":
    main()
