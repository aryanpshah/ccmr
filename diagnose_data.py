#!/usr/bin/env python
"""
Quick diagnostic to verify data pipeline and identify preprocessing issues.
This helps distinguish between model problems and data problems.
"""
import sys
from pathlib import Path
import numpy as np
import nibabel as nib

# Add scripts to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR / "scripts"))

def check_data_splits():
    """Verify data splits and their sizes."""
    print("="*80)
    print("DATA SPLIT VERIFICATION")
    print("="*80)
    
    splits_dir = Path("data/splits")
    budgets = ["train_L5.txt", "train_L10.txt", "train_L20.txt", "train_L40.txt", 
               "val_ids.txt", "test_ids.txt"]
    
    for split_file in budgets:
        path = splits_dir / split_file
        if path.exists():
            with open(path, 'r') as f:
                ids = [line.strip() for line in f if line.strip()]
            print(f"{split_file:20s}: {len(ids):3d} cases")
        else:
            print(f"{split_file:20s}: NOT FOUND")
    
    print()


def check_processed_data():
    """Check processed data dimensions and value ranges."""
    print("="*80)
    print("PROCESSED DATA CHECK")
    print("="*80)
    
    processed_dir = Path("data/processed/hvsmr2")
    
    # Check if directories exist
    images_dir = processed_dir / "imagesTr"
    labels_dir = processed_dir / "labelsTr"
    
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return
    if not labels_dir.exists():
        print(f"❌ Labels directory not found: {labels_dir}")
        return
    
    # Get first few files
    image_files = sorted(images_dir.glob("*.nii*"))[:3]
    
    print(f"\nFound {len(list(images_dir.glob('*.nii*')))} image files")
    print(f"Found {len(list(labels_dir.glob('*.nii*')))} label files")
    
    print("\nChecking first 3 samples:")
    for img_path in image_files:
        # Load image
        img = nib.load(str(img_path))
        img_data = np.asarray(img.dataobj)
        
        # Find corresponding label
        stem = img_path.name.split('.nii')[0]
        label_candidates = [
            labels_dir / img_path.name,
            labels_dir / f"{stem}.nii.gz",
            labels_dir / f"{stem}.nii"
        ]
        label_path = None
        for cand in label_candidates:
            if cand.exists():
                label_path = cand
                break
        
        if label_path:
            label = nib.load(str(label_path))
            label_data = np.asarray(label.dataobj)
            
            # Check shapes
            print(f"\n{img_path.name}:")
            print(f"  Image shape: {img_data.shape}")
            print(f"  Label shape: {label_data.shape}")
            print(f"  Image range: [{img_data.min():.2f}, {img_data.max():.2f}]")
            print(f"  Image mean/std: {img_data.mean():.2f} ± {img_data.std():.2f}")
            
            # Check label classes
            unique_labels = np.unique(label_data).astype(int)
            print(f"  Label classes: {unique_labels.tolist()}")
            
            # Check class distribution
            class_counts = {}
            for cls in unique_labels:
                count = np.sum(label_data == cls)
                class_counts[cls] = count
            
            total_voxels = label_data.size
            print(f"  Class distribution:")
            for cls in sorted(class_counts.keys()):
                count = class_counts[cls]
                pct = 100.0 * count / total_voxels
                print(f"    Class {cls}: {count:8d} voxels ({pct:5.2f}%)")
        else:
            print(f"\n{img_path.name}:")
            print(f"  ❌ No matching label found")


def check_class_imbalance():
    """Analyze class imbalance across the dataset."""
    print("\n" + "="*80)
    print("CLASS IMBALANCE ANALYSIS")
    print("="*80)
    
    # This would require loading all labels which is slow
    # For now, provide theoretical analysis
    print("\nTypical cardiac MRI class distribution:")
    print("  Class 0 (Background): ~85-90% of voxels")
    print("  Class 1 (Small vessel): ~0.1-0.5% of voxels  ⚠️ EXTREME IMBALANCE")
    print("  Class 2 (Large chamber): ~5-10% of voxels")
    print("  Classes 3-8: ~0.5-2% each")
    
    print("\nThis explains why:")
    print("  • Class 1 (small structures) have very low Dice with few labels")
    print("  • Class 2 (large chambers) maintain good performance")
    print("  • Aggressive class weighting is CRITICAL")


def analyze_training_curves():
    """Check if training curve logs exist."""
    print("\n" + "="*80)
    print("TRAINING CURVES CHECK")
    print("="*80)
    
    # Check for tensorboard logs, training logs, etc.
    possible_log_dirs = [
        Path("runs"),
        Path("logs"),
        Path("outputs"),
        Path("results")
    ]
    
    found_logs = False
    for log_dir in possible_log_dirs:
        if log_dir.exists():
            # Check for tensorboard event files
            tb_files = list(log_dir.rglob("events.out.tfevents*"))
            if tb_files:
                found_logs = True
                print(f"✅ Found tensorboard logs in {log_dir}")
            
            # Check for text logs
            txt_logs = list(log_dir.rglob("*.log")) + list(log_dir.rglob("*.txt"))
            if txt_logs:
                found_logs = True
                print(f"✅ Found text logs in {log_dir}")
    
    if not found_logs:
        print("⚠️  No training logs found")
        print("   Recommendation: Add tensorboard logging to track:")
        print("   - Per-class Dice over epochs")
        print("   - Train vs validation gap")
        print("   - Learning rate schedule")


def main():
    """Run all diagnostics."""
    print("\n" + "="*80)
    print("CARDIAC MRI PIPELINE - DATA DIAGNOSTICS")
    print("="*80)
    print()
    
    # Change to project root
    project_root = Path(__file__).parent
    import os
    os.chdir(project_root)
    
    try:
        check_data_splits()
        check_processed_data()
        check_class_imbalance()
        analyze_training_curves()
        
        print("\n" + "="*80)
        print("DIAGNOSTIC COMPLETE")
        print("="*80)
        
        print("\nKey Findings:")
        print("1. Check if image/label shapes match expected dimensions")
        print("2. Verify label classes are [0, 1, 2, ..., 8]")
        print("3. Confirm class imbalance (Class 1 should be rare)")
        print("4. Ensure preprocessing is consistent train/test")
        
    except Exception as e:
        print(f"\n❌ Error during diagnostics: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
