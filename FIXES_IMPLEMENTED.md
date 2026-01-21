# ✅ Critical Fixes Implemented

## Summary of Changes

All critical fixes have been implemented to address the overfitting problem where models performed well on overfit tests but poorly on normal testing.

---

## 🔴 HIGH PRIORITY FIXES (Completed)

### 1. ✅ INCREASED DATA AUGMENTATION (3x Stronger)

**Files Modified:** `scripts/swin_unetr_btcv_setup.py`

**Changes:**

- **L5/L10 datasets** (most critical):
  - `RandScaleIntensityd`: factors 0.05 → **0.15** (3x stronger), prob 0.5 → 0.8
  - `RandShiftIntensityd`: offsets 0.05 → **0.15** (3x stronger), prob 0.5 → 0.8
- **L20/L40 datasets**:

  - `RandScaleIntensityd`: factors 0.1 → **0.12**, prob 0.5 → 0.7
  - `RandShiftIntensityd`: offsets 0.1 → **0.12**, prob 0.5 → 0.7

- **Added NEW augmentation**:
  - `RandAffined` with prob=0.7:
    - Rotation: ±11 degrees per axis
    - Scaling: ±15% per axis
    - Mode: bilinear for images, nearest for labels

**Rationale:** With only 5-10 training samples, aggressive augmentation is critical to prevent memorization and enable generalization.

---

### 2. ✅ IMPLEMENTED FOCAL LOSS FOR CLASS IMBALANCE

**Files Modified:**

- `scripts/train_swin_unetr_scratch.py`
- `scripts/train_swin_unetr_finetune_btcv.py`

**Changes:**

- Replaced `DiceCELoss` with **`DiceFocalLoss`**
- Configuration:
  ```python
  DiceFocalLoss(
      gamma=2.0,          # Focus on hard examples
      lambda_dice=0.7,    # 70% Dice contribution
      lambda_focal=0.3,   # 30% Focal CE contribution
      weight=class_weights
  )
  ```

**Rationale:** Focal loss significantly improves learning for severely imbalanced classes (Class 1 has only 0.1-0.5% of voxels).

---

### 3. ✅ IMPROVED CLASS WEIGHTS (5x Stronger for Small Structures)

**Files Modified:** `scripts/swin_unetr_btcv_setup.py`

**Changes:**

```python
OLD: [0.05, 1.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.0, 3.0]
NEW: [0.01, 5.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.0]
```

**Key Changes:**

- Background weight: 0.05 → **0.01** (even lower to focus on foreground)
- Class 1 (smallest): 1.0 → **5.0** (5x increase!)
- Class 2 (large): 1.0 → **2.0** (moderate increase)
- Classes 3-8: Progressive increase to **5.0** for smallest structures

**Rationale:** Small cardiac structures were severely underweighted. New weights force the model to pay much more attention to rare classes.

---

### 4. ✅ REDUCED LEARNING RATES + ADDED WARMUP

**Files Modified:**

- `scripts/train_swin_unetr_scratch.py`
- `scripts/train_swin_unetr_finetune_btcv.py`
- `scripts/train_swin_unetr_lora.py`

**Changes:**

| Config               | Old  | New           | Change        |
| -------------------- | ---- | ------------- | ------------- |
| **Scratch LR max**   | 2e-4 | **1e-4**      | 50% reduction |
| **Scratch LR min**   | 2e-5 | **1e-5**      | 50% reduction |
| **Scratch warmup**   | 0    | **20 epochs** | Added         |
| **Finetune LR max**  | 6e-5 | **3e-5**      | 50% reduction |
| **Finetune LR min**  | 6e-6 | **3e-6**      | 50% reduction |
| **Finetune warmup**  | 0    | **10 epochs** | Added         |
| **LoRA LR**          | 5e-4 | **3e-4**      | 40% reduction |
| **LoRA backbone LR** | 1e-5 | **5e-6**      | 50% reduction |
| **LoRA warmup**      | 5    | **15 epochs** | 3x increase   |

**Scheduler Implementation:**

```python
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs-warmup_epochs, eta_min=lr_min)
scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
```

**Rationale:** High learning rates with batch_size=1 cause unstable training. Lower LR + warmup = smoother convergence.

---

### 5. ✅ INCREASED WEIGHT DECAY (Better Regularization)

**Files Modified:** All training scripts

**Changes:**

| Config              | Old  | New      | Change        |
| ------------------- | ---- | -------- | ------------- |
| **Scratch**         | 1e-2 | **3e-2** | 3x increase   |
| **Finetune**        | 3e-3 | **1e-2** | 3.3x increase |
| **LoRA (adapters)** | 1e-2 | **3e-2** | 3x increase   |
| **LoRA (backbone)** | 1e-4 | **1e-3** | 10x increase  |

**Rationale:** Stronger weight decay acts as additional regularization to combat overfitting on small datasets.

---

## 🟡 MEDIUM PRIORITY FIXES (Completed)

### 6. ✅ INCREASED DEFAULT PATCH SIZE

**Files Modified:** All training scripts + `swin_unetr_btcv_setup.py`

**Changes:**

- Default ROI size: `(128, 128, 128)` → **`(160, 160, 160)`**

**GPU Memory Impact:**

- 128³ ≈ 2.1M voxels → ~4-6 GB
- 160³ ≈ 4.1M voxels → ~8-10 GB

**Rationale:** Larger patches capture more anatomical context for whole-heart segmentation. Critical for seeing full structures.

**Note:** Users can still override with `--roi_size 128 128 128` if GPU memory is limited.

---

## 📊 Expected Performance Improvements

### Predicted Results After Fixes:

| Budget  | Old FG Dice | Expected FG Dice | Improvement | Old Class 1 | Expected Class 1 |
| ------- | ----------- | ---------------- | ----------- | ----------- | ---------------- |
| **L5**  | 0.667 ❌    | **0.75-0.80** ✅ | +12-20%     | 0.484 ❌    | **0.60-0.70** ⚠️ |
| **L10** | 0.823 ⚠️    | **0.85-0.88** ✅ | +3-7%       | 0.774 ⚠️    | **0.82-0.86** ✅ |
| **L20** | 0.892 ✅    | **0.90-0.93** ✅ | +1-3%       | 0.878 ✅    | **0.89-0.92** ✅ |
| **L40** | 0.879 ✅    | **0.91-0.94** ✅ | +3-7%       | 0.853 ✅    | **0.88-0.91** ✅ |

---

## 🧪 How to Test the Fixes

### Step 1: Quick Overfit Test (2-3 hours)

This validates that fixes don't break the model's ability to learn:

```bash
# Test on 1 case with LoRA (fastest to train)
python scripts/train_swin_unetr_lora.py \
    --data_root data/processed/hvsmr2 \
    --train_split data/splits/train_L5.txt \
    --val_split data/splits/val_ids.txt \
    --output_dir runs/overfit_test_fixed \
    --epochs 50 \
    --batch_size 1 \
    --roi_size 160 160 160 \
    --overfit_debug
```

**Expected Result:** Should still achieve Dice > 0.95 on the overfit case.

---

### Step 2: L5 Validation Test (6-8 hours)

This tests if fixes improve generalization:

```bash
# Train on L5 split with all fixes
python scripts/train_swin_unetr_scratch.py \
    --data_root data/processed/hvsmr2 \
    --train_split data/splits/train_L5.txt \
    --val_split data/splits/val_ids.txt \
    --output_dir runs/scratch_L5_fixed \
    --epochs 150 \
    --batch_size 1 \
    --roi_size 160 160 160 \
    --seed 42
```

**Expected Results:**

- Validation Foreground Dice: **0.75-0.80** (was 0.667)
- Validation Class 1 Dice: **0.60-0.70** (was 0.484)
- Train-Val gap should be **smaller** than before

**Monitoring:**

- Watch per-class Dice in logs
- If Class 1 improves significantly → fixes are working!
- If training is more stable (no wild oscillations) → LR/warmup working

---

### Step 3: Full Training (1 week)

After validating on L5:

```bash
# Train all budgets with scratch model
for budget in L5 L10 L20 L40; do
    python scripts/train_swin_unetr_scratch.py \
        --data_root data/processed/hvsmr2 \
        --train_split data/splits/train_${budget}.txt \
        --val_split data/splits/val_ids.txt \
        --output_dir runs/scratch_${budget}_fixed \
        --epochs 300 \
        --batch_size 1 \
        --roi_size 160 160 160
done

# Train with fine-tuning
for budget in L5 L10 L20 L40; do
    python scripts/train_swin_unetr_finetune_btcv.py \
        --data_root data/processed/hvsmr2 \
        --train_split data/splits/train_${budget}.txt \
        --val_split data/splits/val_ids.txt \
        --pretrained_ckpt pretrained/model.pt \
        --output_dir runs/finetune_${budget}_fixed \
        --epochs 300 \
        --batch_size 1 \
        --roi_size 160 160 160
done

# Train with LoRA
for budget in L5 L10 L20 L40; do
    python scripts/train_swin_unetr_lora.py \
        --data_root data/processed/hvsmr2 \
        --train_split data/splits/train_${budget}.txt \
        --val_split data/splits/val_ids.txt \
        --pretrained_ckpt pretrained/model.pt \
        --output_dir runs/lora_${budget}_fixed \
        --epochs 300 \
        --batch_size 1 \
        --roi_size 160 160 160 \
        --lora_rank 4
done
```

---

## 🔧 Files Modified

1. **`scripts/swin_unetr_btcv_setup.py`**

   - Increased augmentation strength (3x for L5/L10)
   - Added RandAffined transform
   - Improved default class weights
   - Increased default ROI size to 160³

2. **`scripts/train_swin_unetr_scratch.py`**

   - Reduced learning rates (2e-4 → 1e-4)
   - Added 20-epoch warmup
   - Replaced DiceCELoss with DiceFocalLoss
   - Increased weight decay (1e-2 → 3e-2)
   - Increased default ROI size to 160³

3. **`scripts/train_swin_unetr_finetune_btcv.py`**

   - Reduced learning rates (6e-5 → 3e-5)
   - Added 10-epoch warmup
   - Replaced DiceCELoss with DiceFocalLoss
   - Increased weight decay (3e-3 → 1e-2)
   - Increased default ROI size to 160³

4. **`scripts/train_swin_unetr_lora.py`**
   - Reduced LoRA learning rates (5e-4 → 3e-4)
   - Reduced backbone learning rates (1e-5 → 5e-6)
   - Increased weight decay (1e-2 → 3e-2 for adapters, 1e-4 → 1e-3 for backbone)
   - Increased warmup epochs (5 → 15)
   - Increased default ROI size to 160³

---

## 💡 Key Insights

### Why These Fixes Work:

1. **Stronger Augmentation** → Forces model to learn invariant features instead of memorizing
2. **Focal Loss** → Makes model focus on hard examples (small structures)
3. **Better Class Weights** → Prevents model from ignoring rare structures
4. **Lower LR + Warmup** → Stable training, smoother convergence
5. **Higher Weight Decay** → Prevents overfitting on small datasets
6. **Larger Patches** → More anatomical context = better segmentation

### The Core Problem Was:

> **"The model was optimizing for the training distribution, not the underlying task"**

With only 5-10 samples:

- ❌ Old config: Model memorized those exact volumes
- ✅ New config: Model learns robust features that generalize

---

## 📋 Backward Compatibility

All changes are **backward compatible**:

- Default hyperparameters are updated, but users can still override with command-line args
- Example: `--roi_size 128 128 128` if GPU memory is limited
- Example: `--lr_max 2e-4` to use old learning rate (not recommended)

---

## 🚨 Important Notes

### GPU Memory Considerations:

**160³ patches require ~8-10 GB GPU memory**

If you encounter OOM errors:

```bash
# Option 1: Use smaller patches
--roi_size 128 128 128

# Option 2: Reduce batch size (already 1, so not applicable)

# Option 3: Use gradient checkpointing (requires code modification)
```

### Training Time Estimates:

- **Overfit test**: 2-3 hours (50 epochs, 1 case)
- **L5 training**: 6-8 hours (150 epochs, 5 cases)
- **L40 training**: 2-3 days (300 epochs, 40 cases)
- **Full study** (all budgets × 3 methods): ~1 week on 1 GPU

---

## 📞 Questions or Issues?

If you encounter problems:

1. **Check logs for**:
   - `[CRITICAL FIX]` messages confirming new hyperparameters are active
   - Per-class Dice values in validation
   - Learning rate values during warmup
2. **Verify augmentation**:
   - Look for `[DBG crop]` and `[DBG post-crop]` messages in first epoch
   - Should see augmentation messages for L5/L10
3. **Monitor training**:
   - Loss should decrease smoothly (no wild spikes)
   - Validation Dice should improve steadily
   - Class 1 Dice should be > 0.6 by epoch 100 for L5

---

## ✅ Summary

All critical fixes have been implemented to address the client's overfitting problem. The changes target the root cause: **insufficient regularization for small datasets**.

**Next Step:** Run the quick overfit test to validate the fixes, then proceed with L5 training.

Good luck! 🚀
