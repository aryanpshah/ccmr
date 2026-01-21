#!/usr/bin/env python
"""
Quick test to validate that all critical fixes are properly applied.
Run this before starting training to ensure configuration is correct.
"""
import sys
from pathlib import Path

# Add scripts to path
SCRIPT_DIR = Path(__file__).parent
sys.path.append(str(SCRIPT_DIR / "scripts"))

def test_augmentation():
    """Test that augmentation parameters are updated."""
    print("="*80)
    print("TEST 1: DATA AUGMENTATION")
    print("="*80)
    
    try:
        from scripts.swin_unetr_btcv_setup import DEFAULT_ROI_SIZE, DEFAULT_CLASS_WEIGHTS
        
        print(f"\n✅ DEFAULT_ROI_SIZE = {DEFAULT_ROI_SIZE}")
        expected_roi = (160, 160, 160)
        if DEFAULT_ROI_SIZE == expected_roi:
            print(f"   ✅ CORRECT: ROI size is {expected_roi}")
        else:
            print(f"   ❌ ERROR: Expected {expected_roi}, got {DEFAULT_ROI_SIZE}")
            return False
        
        print(f"\n✅ DEFAULT_CLASS_WEIGHTS = {DEFAULT_CLASS_WEIGHTS}")
        # Check key weights
        if DEFAULT_CLASS_WEIGHTS[0] == 0.01:  # Background
            print("   ✅ CORRECT: Background weight is 0.01 (was 0.05)")
        else:
            print(f"   ❌ ERROR: Background weight should be 0.01, got {DEFAULT_CLASS_WEIGHTS[0]}")
            return False
            
        if DEFAULT_CLASS_WEIGHTS[1] == 5.0:  # Class 1 (small structure)
            print("   ✅ CORRECT: Class 1 weight is 5.0 (was 1.0)")
        else:
            print(f"   ❌ ERROR: Class 1 weight should be 5.0, got {DEFAULT_CLASS_WEIGHTS[1]}")
            return False
        
        print("\n✅ TEST 1 PASSED: Augmentation and class weights updated correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scratch_hyperparams():
    """Test scratch training hyperparameters."""
    print("\n" + "="*80)
    print("TEST 2: SCRATCH TRAINING HYPERPARAMETERS")
    print("="*80)
    
    try:
        # Parse default args from scratch script
        import argparse
        from scripts.train_swin_unetr_scratch import main
        
        # Read the file to check default values
        scratch_file = Path("scripts/train_swin_unetr_scratch.py")
        with open(scratch_file, 'r') as f:
            content = f.read()
        
        # Check for critical fixes
        checks = [
            ("default=1e-4", "Learning rate reduced to 1e-4"),
            ("default=1e-5", "Min learning rate reduced to 1e-5"),
            ("default=3e-2", "Weight decay increased to 3e-2"),
            ("default=20", "Warmup epochs set to 20"),
            ("default=(160, 160, 160)", "ROI size increased to 160"),
            ("DiceFocalLoss", "Using DiceFocalLoss instead of DiceCELoss"),
            ("gamma=2.0", "Focal loss gamma set to 2.0"),
            ("LinearLR", "Warmup scheduler implemented"),
        ]
        
        all_passed = True
        for check_str, description in checks:
            if check_str in content:
                print(f"   ✅ {description}")
            else:
                print(f"   ❌ MISSING: {description}")
                all_passed = False
        
        if all_passed:
            print("\n✅ TEST 2 PASSED: All scratch training fixes applied")
        else:
            print("\n⚠️  TEST 2 INCOMPLETE: Some fixes may be missing")
        
        return all_passed
        
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_finetune_hyperparams():
    """Test fine-tuning hyperparameters."""
    print("\n" + "="*80)
    print("TEST 3: FINE-TUNING HYPERPARAMETERS")
    print("="*80)
    
    try:
        finetune_file = Path("scripts/train_swin_unetr_finetune_btcv.py")
        with open(finetune_file, 'r') as f:
            content = f.read()
        
        checks = [
            ("default=3e-5", "Learning rate reduced to 3e-5"),
            ("default=3e-6", "Min learning rate reduced to 3e-6"),
            ("default=1e-2", "Weight decay increased to 1e-2"),
            ("default=10", "Warmup epochs set to 10"),
            ("default=(160, 160, 160)", "ROI size increased to 160"),
            ("DiceFocalLoss", "Using DiceFocalLoss"),
            ("gamma=2.0", "Focal loss gamma set to 2.0"),
        ]
        
        all_passed = True
        for check_str, description in checks:
            if check_str in content:
                print(f"   ✅ {description}")
            else:
                print(f"   ❌ MISSING: {description}")
                all_passed = False
        
        if all_passed:
            print("\n✅ TEST 3 PASSED: All fine-tuning fixes applied")
        else:
            print("\n⚠️  TEST 3 INCOMPLETE: Some fixes may be missing")
        
        return all_passed
        
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lora_hyperparams():
    """Test LoRA training hyperparameters."""
    print("\n" + "="*80)
    print("TEST 4: LORA TRAINING HYPERPARAMETERS")
    print("="*80)
    
    try:
        lora_file = Path("scripts/train_swin_unetr_lora.py")
        with open(lora_file, 'r') as f:
            content = f.read()
        
        checks = [
            ("default=3e-4", "LoRA learning rate reduced to 3e-4"),
            ("default=5e-6", "Backbone learning rate reduced to 5e-6"),
            ("default=3e-2", "LoRA weight decay increased to 3e-2"),
            ("default=1e-3", "Backbone weight decay increased to 1e-3"),
            ("default=15", "Warmup epochs increased to 15"),
            ("default=(160, 160, 160)", "ROI size increased to 160"),
        ]
        
        all_passed = True
        for check_str, description in checks:
            if check_str in content:
                print(f"   ✅ {description}")
            else:
                print(f"   ❌ MISSING: {description}")
                all_passed = False
        
        if all_passed:
            print("\n✅ TEST 4 PASSED: All LoRA training fixes applied")
        else:
            print("\n⚠️  TEST 4 INCOMPLETE: Some fixes may be missing")
        
        return all_passed
        
    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_data_existence():
    """Check that data files exist."""
    print("\n" + "="*80)
    print("TEST 5: DATA FILES EXISTENCE")
    print("="*80)
    
    try:
        required_paths = [
            "data/processed/hvsmr2/imagesTr",
            "data/processed/hvsmr2/labelsTr",
            "data/splits/train_L5.txt",
            "data/splits/train_L10.txt",
            "data/splits/train_L20.txt",
            "data/splits/train_L40.txt",
            "data/splits/val_ids.txt",
            "data/splits/test_ids.txt",
        ]
        
        all_exist = True
        for path_str in required_paths:
            path = Path(path_str)
            if path.exists():
                if path.is_dir():
                    count = len(list(path.glob("*.nii*")))
                    print(f"   ✅ {path_str} ({count} files)")
                else:
                    with open(path, 'r') as f:
                        count = len([l for l in f if l.strip()])
                    print(f"   ✅ {path_str} ({count} cases)")
            else:
                print(f"   ❌ MISSING: {path_str}")
                all_exist = False
        
        if all_exist:
            print("\n✅ TEST 5 PASSED: All required data files exist")
        else:
            print("\n⚠️  TEST 5 INCOMPLETE: Some data files are missing")
        
        return all_exist
        
    except Exception as e:
        print(f"\n❌ TEST 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("\n" + "="*80)
    print("VALIDATION TEST FOR CRITICAL FIXES")
    print("="*80)
    print()
    
    # Change to project root
    project_root = Path(__file__).parent
    import os
    os.chdir(project_root)
    
    results = []
    
    # Run tests
    results.append(("Augmentation & Class Weights", test_augmentation()))
    results.append(("Scratch Training Config", test_scratch_hyperparams()))
    results.append(("Fine-tuning Config", test_finetune_hyperparams()))
    results.append(("LoRA Config", test_lora_hyperparams()))
    results.append(("Data Files", check_data_existence()))
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10s} - {test_name}")
    
    print()
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! You can proceed with training.")
        print("\nNext steps:")
        print("1. Run overfit test: python scripts/train_swin_unetr_lora.py --overfit_debug ...")
        print("2. Train L5 scratch: python scripts/train_swin_unetr_scratch.py --train_split data/splits/train_L5.txt ...")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
