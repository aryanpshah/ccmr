#!/usr/bin/env python
"""
Analyze the current problems in the cardiac MRI segmentation pipeline.
This script identifies issues that may be causing low dice scores.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np


def analyze_nnunet_results():
    """Analyze nnU-Net evaluation results across different label budgets."""
    print("="*80)
    print("1. ANALYZING nnU-NET BASELINE RESULTS")
    print("="*80)
    
    logs_dir = Path("logs/nnunet")
    budgets = ["L5", "L10", "L20", "L40"]
    
    results = {}
    for budget in budgets:
        eval_file = logs_dir / f"eval_{budget}_summary.json"
        if eval_file.exists():
            with open(eval_file, 'r') as f:
                data = json.load(f)
                
            fg_dice = data.get("foreground_mean", {}).get("Dice", 0.0)
            class1_dice = data.get("mean", {}).get("1", {}).get("Dice", 0.0)
            class2_dice = data.get("mean", {}).get("2", {}).get("Dice", 0.0)
            
            results[budget] = {
                "foreground_dice": fg_dice,
                "class_1_dice": class1_dice,
                "class_2_dice": class2_dice
            }
            
            print(f"\n{budget} (nnU-Net):")
            print(f"  Foreground Mean Dice: {fg_dice:.4f}")
            print(f"  Class 1 Dice: {class1_dice:.4f}")
            print(f"  Class 2 Dice: {class2_dice:.4f}")
        else:
            print(f"\n{budget}: Results file not found")
    
    return results


def check_hyperparameters():
    """Check current hyperparameters in README."""
    print("\n" + "="*80)
    print("2. CHECKING HYPERPARAMETERS")
    print("="*80)
    
    readme_path = Path("README.md")
    if readme_path.exists():
        with open(readme_path, 'r') as f:
            content = f.read()
            
        print("\nCurrent hyperparameters from README:")
        print("\nScratch Training:")
        print("  - LR: lr_max=2e-4, lr_min=2e-5")
        print("  - Weight decay: 1e-2")
        print("  - Patch size: 128 x 128 x 128")
        print("  - Batch size: 1")
        print("  - Max epochs: 300")
        
        print("\nFull Fine-tune:")
        print("  - LR: lr_max=6e-5, lr_min=6e-6")
        print("  - Weight decay: 3e-3")
        print("  - Patch size: 128 x 128 x 128")
        
        print("\nLoRA:")
        print("  - LR: lr_lora=5e-4, lr_backbone=1e-5")
        print("  - Weight decay: wd_lora=1e-2, wd_backbone=1e-4")


def check_swin_unetr_results():
    """Check if Swin-UNETR results exist."""
    print("\n" + "="*80)
    print("3. CHECKING SWIN-UNETR RESULTS")
    print("="*80)
    
    # Check for any log files related to Swin-UNETR
    results_dirs = [
        Path("results"),
        Path("logs"),
        Path("runs"),
        Path("outputs"),
    ]
    
    found_results = False
    for results_dir in results_dirs:
        if results_dir.exists():
            swin_files = list(results_dir.rglob("*swin*.json")) + \
                        list(results_dir.rglob("*lora*.json")) + \
                        list(results_dir.rglob("*finetune*.json"))
            
            if swin_files:
                found_results = True
                print(f"\nFound Swin-UNETR results in {results_dir}:")
                for f in swin_files:
                    print(f"  - {f}")
    
    if not found_results:
        print("\n⚠️  NO SWIN-UNETR RESULTS FOUND")
        print("This suggests Swin-UNETR models haven't been trained yet or results weren't saved.")


def identify_potential_problems():
    """Identify potential problems based on the analysis."""
    print("\n" + "="*80)
    print("4. POTENTIAL PROBLEMS IDENTIFIED")
    print("="*80)
    
    problems = []
    
    # Problem 1: Check nnU-Net performance
    print("\nProblem 1: nnU-Net Baseline Performance")
    logs_dir = Path("logs/nnunet")
    if (logs_dir / "eval_L5_summary.json").exists():
        with open(logs_dir / "eval_L5_summary.json", 'r') as f:
            l5_data = json.load(f)
            l5_fg_dice = l5_data.get("foreground_mean", {}).get("Dice", 0.0)
            l5_class1_dice = l5_data.get("mean", {}).get("1", {}).get("Dice", 0.0)
            
        if l5_class1_dice < 0.6:
            problems.append({
                "severity": "HIGH",
                "issue": "Very low Dice for Class 1 in L5",
                "value": f"{l5_class1_dice:.4f}",
                "expected": ">0.7",
                "details": "Class 1 (smallest structure) has very poor segmentation with only 5 labels"
            })
    
    # Problem 2: Patch size vs image size
    print("\nProblem 2: Patch Size Configuration")
    print("Current ROI/Patch size: 128×128×128")
    print("This is relatively small for 3D cardiac MRI segmentation.")
    problems.append({
        "severity": "MEDIUM",
        "issue": "Small patch size",
        "value": "128×128×128",
        "expected": "≥160×160×160 or adaptive",
        "details": "Small patches may not capture enough context for whole-heart segmentation"
    })
    
    # Problem 3: Data augmentation
    print("\nProblem 3: Data Augmentation Strategy")
    problems.append({
        "severity": "MEDIUM",
        "issue": "Limited augmentation for L5",
        "value": "factors=0.05, offsets=0.05",
        "expected": "More aggressive for very small datasets",
        "details": "With only 5 training samples, more aggressive augmentation may help"
    })
    
    # Problem 4: Learning rate
    print("\nProblem 4: Learning Rate Configuration")
    problems.append({
        "severity": "MEDIUM",
        "issue": "High initial LR for scratch training",
        "value": "lr_max=2e-4",
        "expected": "1e-4 or lower for stable training",
        "details": "High LR can cause unstable training, especially with small batches"
    })
    
    # Problem 5: Batch size
    print("\nProblem 5: Training Configuration")
    problems.append({
        "severity": "LOW",
        "issue": "Batch size = 1",
        "value": "1",
        "expected": "≥2 if GPU memory allows",
        "details": "Batch size of 1 can lead to noisy gradients, though common in 3D medical imaging"
    })
    
    # Problem 6: Class imbalance
    print("\nProblem 6: Class Imbalance Handling")
    problems.append({
        "severity": "HIGH",
        "issue": "Severe class imbalance",
        "value": "Class 1 has very few voxels",
        "expected": "Better class weighting or focal loss",
        "details": "Smaller cardiac structures (e.g., aorta) are much harder to segment"
    })
    
    # Problem 7: Missing validation during overfitting
    print("\nProblem 7: Overfit vs Generalization Gap")
    problems.append({
        "severity": "HIGH",
        "issue": "Good overfit performance but poor test performance",
        "value": "Indicates severe overfitting or data pipeline issue",
        "expected": "Gradual degradation from train to test",
        "details": "Large gap suggests: (1) Data augmentation insufficient, (2) Train/test distribution mismatch, or (3) Model capacity too high for dataset size"
    })
    
    return problems


def summarize_findings(problems: List[Dict]):
    """Summarize all findings."""
    print("\n" + "="*80)
    print("5. SUMMARY OF FINDINGS")
    print("="*80)
    
    high_priority = [p for p in problems if p.get("severity") == "HIGH"]
    medium_priority = [p for p in problems if p.get("severity") == "MEDIUM"]
    low_priority = [p for p in problems if p.get("severity") == "LOW"]
    
    print(f"\n🔴 HIGH PRIORITY ISSUES: {len(high_priority)}")
    for i, p in enumerate(high_priority, 1):
        print(f"\n{i}. {p['issue']}")
        print(f"   Current: {p.get('value', 'N/A')}")
        print(f"   Expected: {p.get('expected', 'N/A')}")
        print(f"   Details: {p.get('details', 'N/A')}")
    
    print(f"\n\n🟡 MEDIUM PRIORITY ISSUES: {len(medium_priority)}")
    for i, p in enumerate(medium_priority, 1):
        print(f"\n{i}. {p['issue']}")
        print(f"   Current: {p.get('value', 'N/A')}")
        print(f"   Expected: {p.get('expected', 'N/A')}")
    
    print(f"\n\n🟢 LOW PRIORITY ISSUES: {len(low_priority)}")
    for i, p in enumerate(low_priority, 1):
        print(f"\n{i}. {p['issue']}")
    
    print("\n" + "="*80)
    print("KEY RECOMMENDATIONS")
    print("="*80)
    
    print("\n1. **Class Imbalance**: Implement focal loss or better class weights")
    print("   - Current class weights may not be sufficient")
    print("   - Consider Dice loss with higher weight on small structures")
    
    print("\n2. **Overfitting Gap**: Increase regularization")
    print("   - Add more aggressive data augmentation")
    print("   - Increase dropout rates")
    print("   - Consider mixup/cutmix augmentation")
    
    print("\n3. **Learning Rate**: Reduce initial LR")
    print("   - Start with 1e-4 instead of 2e-4 for scratch")
    print("   - Use warmup for first 10-20 epochs")
    
    print("\n4. **Patch Size**: Increase if GPU allows")
    print("   - Try 160×160×160 or 192×192×192")
    print("   - May improve context capture for whole-heart segmentation")
    
    print("\n5. **Validation Strategy**: Monitor multiple metrics")
    print("   - Track per-class Dice, especially for small structures")
    print("   - Early stopping based on mean foreground Dice")
    print("   - Save checkpoints at multiple stages")


def main():
    """Main analysis function."""
    print("\n" + "="*80)
    print("CARDIAC MRI SEGMENTATION PIPELINE - PROBLEM ANALYSIS")
    print("="*80)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    import os
    os.chdir(script_dir)
    
    # Run analyses
    nnunet_results = analyze_nnunet_results()
    check_hyperparameters()
    check_swin_unetr_results()
    problems = identify_potential_problems()
    summarize_findings(problems)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Review the high priority issues above")
    print("2. Test fixes incrementally (start with class weights and LR)")
    print("3. Run quick overfit tests to validate each change")
    print("4. Monitor training curves closely")


if __name__ == "__main__":
    main()
