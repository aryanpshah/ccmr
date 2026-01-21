# 🔍 Current Problems in Cardiac MRI Segmentation Pipeline

## Executive Summary

Your client is experiencing a **significant overfitting issue** where:

- ✅ **Overfit tests work well** with high Dice values
- ❌ **Normal testing produces low values**

Based on the analysis, here are the specific problems identified:

---

## 📊 Current nnU-Net Performance (Baseline)

| Label Budget | Foreground Dice | Class 1 (Small) | Class 2 (Large) |
| ------------ | --------------- | --------------- | --------------- |
| **L5**       | 0.667 ❌        | 0.484 ❌        | 0.851 ✅        |
| **L10**      | 0.823 ⚠️        | 0.774 ⚠️        | 0.872 ✅        |
| **L20**      | 0.892 ✅        | 0.878 ✅        | 0.906 ✅        |
| **L40**      | 0.879 ✅        | 0.853 ✅        | 0.904 ✅        |

### Observations:

- **Class 1 (smallest structure)** struggles significantly with limited labels
- Performance improves dramatically with more labels (L5 → L20)
- **L40 performance dips** compared to L20 (possible overfitting)

---

## 🔴 HIGH PRIORITY ISSUES

### 1. **Severe Overfitting / Train-Test Distribution Mismatch**

**Current State:** Good overfit performance but poor generalization

**Likely Causes:**

- Insufficient data augmentation for small training sets
- Model memorizing training samples instead of learning features
- Possible preprocessing differences between train and test

**Impact:** This is the PRIMARY issue causing low test scores

---

### 2. **Class Imbalance - Small Structures Underperforming**

**Current State:** Class 1 Dice = 0.484 for L5 (very poor)

**Current Class Weights:**

```python
(0.05, 1.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.0, 3.0)
```

**Problem:**

- Small cardiac structures (aorta, small vessels) have very few voxels
- Standard Dice loss treats all classes somewhat equally
- Class weights may not be aggressive enough

**Evidence:**

- L5 Class 1: 0.484 (fails)
- L5 Class 2: 0.851 (succeeds)
- Large structures are easier to segment even with few labels

---

### 3. **Training Instability**

**Current State:** High learning rate with small batch size

**Configuration Issues:**

- `lr_max = 2e-4` (too high for scratch training)
- `batch_size = 1` (noisy gradients)
- `patch_size = 128³` (may be too small)

**Why This Matters:**

- Unstable training → poor feature learning
- Small patches → insufficient context for whole-heart segmentation
- Noisy gradients → erratic weight updates

---

## 🟡 MEDIUM PRIORITY ISSUES

### 4. **Limited Data Augmentation for Small Datasets**

**Current L5 Augmentation:**

```python
RandScaleIntensityd(factors=0.05, prob=0.5)  # Very conservative
RandShiftIntensityd(offsets=0.05, prob=0.5)  # Very conservative
```

**Problem:** With only 5 training samples, need MORE aggressive augmentation, not less

---

### 5. **Patch Size Too Small**

**Current:** 128 × 128 × 128  
**Recommended:** ≥ 160 × 160 × 160 or 192 × 192 × 192

**Impact:**

- Whole-heart segmentation needs larger context
- Small patches may cut critical anatomical relationships
- Limits ability to see full structure geometry

---

### 6. **No Swin-UNETR Results Yet**

**Current State:** No trained Swin-UNETR models found

This means the comparison study is incomplete - only nnU-Net baseline exists.

---

## 🎯 Root Cause Analysis

### Why Overfit Works But Test Fails?

#### Scenario 1: **Memorization vs Generalization** (Most Likely)

```
Overfit test: Model sees SAME samples repeatedly
└─> Learns to perfectly predict those specific volumes
└─> Achieves high Dice scores

Test: Model sees DIFFERENT samples
└─> Hasn't learned generalizable features
└─> Poor Dice scores
```

#### Scenario 2: **Data Pipeline Issue**

```
Training: Strong augmentation during overfit test
└─> Model robust to variations
└─> High scores

Testing: Different preprocessing or no augmentation
└─> Distribution shift
└─> Low scores
```

#### Scenario 3: **Incorrect Validation Strategy**

```
Overfit: Training patches are cropped around foreground
└─> Model only sees "easy" crops with structures present
└─> High scores

Test: Full volume inference with sliding window
└─> Model sees background regions, difficult angles
└─> Struggles and scores drop
```

---

## 🔧 Critical Fixes Needed (In Priority Order)

### Fix #1: **Increase Data Augmentation** ⭐⭐⭐

```python
# For L5/L10, use MORE aggressive augmentation:
RandScaleIntensityd(factors=0.15, prob=0.8)  # 0.05 → 0.15
RandShiftIntensityd(offsets=0.15, prob=0.8)  # 0.05 → 0.15

# Add elastic deformation:
RandAffined(prob=0.7, rotate_range=0.2, scale_range=0.15)
```

### Fix #2: **Improve Class Weights / Use Focal Loss** ⭐⭐⭐

```python
# Option A: More aggressive class weights
class_weights = [0.01, 4.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.0]

# Option B: Focal Dice Loss (better for extreme imbalance)
from monai.losses import DiceFocalLoss
loss = DiceFocalLoss(gamma=2.0, lambda_dice=0.7, lambda_focal=0.3)
```

### Fix #3: **Reduce Learning Rate** ⭐⭐

```python
# Scratch training:
lr_max = 1e-4  # was 2e-4
lr_min = 1e-5  # was 2e-5

# Add warmup:
warmup_epochs = 20

# Fine-tuning:
lr_max = 3e-5  # was 6e-5
```

### Fix #4: **Increase Patch Size** ⭐⭐

```python
# If GPU memory allows:
roi_size = (160, 160, 160)  # or (192, 192, 192)
```

### Fix #5: **Better Regularization** ⭐

```python
# Increase weight decay
weight_decay = 3e-2  # was 1e-2

# Add dropout in decoder
dropout_rate = 0.2
```

---

## 🧪 Recommended Testing Strategy

### Phase 1: Validate Fixes with Overfit Test (1-2 hours)

1. Apply Fix #1 (augmentation) + Fix #2 (class weights)
2. Run overfit test on 1 case for 50 epochs
3. Should still achieve Dice > 0.95

### Phase 2: Quick Validation Run (4-6 hours)

1. Train on L5 split with fixes for 100 epochs
2. Monitor per-class Dice on validation set
3. Target: Class 1 Dice > 0.6, Foreground Dice > 0.75

### Phase 3: Full Training (3-5 days)

1. Train all configurations (L5, L10, L20, L40)
2. Train all Swin-UNETR variants (scratch, finetune, LoRA)
3. Compare against improved nnU-Net baseline

---

## 📈 Expected Improvements

### After Fixes:

| Budget | Current FG Dice | Expected FG Dice | Improvement |
| ------ | --------------- | ---------------- | ----------- |
| L5     | 0.667           | 0.75-0.80        | +12-20%     |
| L10    | 0.823           | 0.85-0.88        | +3-7%       |
| L20    | 0.892           | 0.90-0.92        | +1-3%       |
| L40    | 0.879           | 0.91-0.93        | +3-6%       |

### Key Metrics to Track:

- **Class 1 Dice** (smallest structure) - should improve most
- **Train vs Val gap** - should decrease
- **Hausdorff Distance** - should improve for small structures

---

## 🎬 Next Steps

1. **Review this analysis** with client to confirm understanding
2. **Run quick overfit test** (1-2 hrs) to establish baseline
3. **Implement fixes incrementally** - test each change
4. **Train Swin-UNETR models** after validating fixes
5. **Generate final comparison** across all methods

---

## 💡 Key Insight

The problem is **NOT** the model architecture or basic approach - nnU-Net is proven for medical imaging. The issue is:

> **"Good overfit + Poor test = Underfitting on the DISTRIBUTION, not the training samples"**

This means the model needs:

- ✅ Better regularization (augmentation)
- ✅ Better loss function (class weights/focal loss)
- ✅ More stable training (lower LR, warmup)
- ✅ Larger context (bigger patches)

All of these are **hyperparameter fixes** that can be tested quickly!
