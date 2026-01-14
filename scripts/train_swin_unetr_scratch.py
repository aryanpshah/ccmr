#!/usr/bin/env python
"""
Train Swin-UNETR from scratch for 9-class segmentation (background + 8 structures).
Uses utilities from swin_unetr_btcv_setup.py for loaders/model configuration.
"""
import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from swin_unetr_btcv_setup import (  # noqa: E402
    DEFAULT_ROI_SIZE,
    NUM_CLASSES,
    create_model,
    create_hvsmr_loaders,
    log_and_validate_batch_shape,
    set_seed,
)
from training_utils import compute_metrics, save_checkpoint  # noqa: E402


# Global scaler for mixed precision training (essential for 6GB VRAM)
scaler = torch.amp.GradScaler('cuda')


def train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_function: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch_idx: int,
    debug_label_hist_steps: int = 3,
) -> float:
    model.train()
    epoch_loss = 0.0
    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        if epoch_idx == 0 and step <= debug_label_hist_steps:
            with torch.no_grad():
                label_hist = torch.bincount(labels.long().flatten(), minlength=NUM_CLASSES).cpu().numpy().tolist()
            print(f"  [DEBUG] label histogram (step {step:02d}): {label_hist}")
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision training - critical for RTX 3050 6GB
        with torch.amp.autocast('cuda'):
            logits = model(images)
            loss = loss_function(logits, labels)
        
        # Scaled backward pass with gradient clipping for stability
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # Unscale before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        scaler.step(optimizer)
        scaler.update()

        # Capture scalar early and clear tensors to avoid retaining graphs/activations between steps.
        loss_value = loss.item()
        epoch_loss += loss_value
        if step == 1 or step % 5 == 0:
            print(f"  train step {step:03d} - loss: {loss_value:.4f}")

        del loss, logits, images, labels
        torch.cuda.empty_cache()  # Help with memory on 6GB GPU
    return epoch_loss / max(1, len(loader))


def validate_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    roi_size: Iterable[int],
    epoch_idx: int,
    debug_interval: int = 20,
) -> Tuple[float, np.ndarray, float]:
    """
    Fast validation using direct inference on pre-cropped volumes.
    For final evaluation, use sliding window inference separately.
    """
    model.eval()
    dice_metric = DiceMetric(include_background=True, reduction="none", num_classes=NUM_CLASSES)
    should_debug = epoch_idx == 0 or ((epoch_idx + 1) % max(1, debug_interval) == 0)
    logged_debug = False

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for batch_idx, batch in enumerate(loader, start=1):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            # Process each sample in the batch individually
            batch_size = images.shape[0]
            for i in range(batch_size):
                image_sample = images[i:i+1]  # Keep batch dimension [1, C, H, W, D]
                label_sample = labels[i:i+1]  # Keep batch dimension [1, 1, H, W, D]
                
                # Ensure image and label have same spatial size
                img_spatial = image_sample.shape[2:]
                lbl_spatial = label_sample.shape[2:]
                if img_spatial != lbl_spatial:
                    label_sample = F.interpolate(
                        label_sample.float(), 
                        size=img_spatial, 
                        mode='nearest'
                    ).long()
                
                # FAST VALIDATION: Direct inference if volume fits roi_size, else sliding window
                vol_size = image_sample.shape[2:]
                roi_tuple = tuple(roi_size)
                
                # Check if volume is small enough for direct inference
                if all(v <= r * 1.5 for v, r in zip(vol_size, roi_tuple)):
                    logits = model(image_sample)
                else:
                    logits = sliding_window_inference(
                        image_sample, 
                        roi_size=roi_tuple, 
                        sw_batch_size=1,
                        predictor=model,
                        overlap=0.125,
                    )
                
                # FIX: Proper shape handling for Dice computation
                # logits: [1, NUM_CLASSES, H, W, D] -> pred: [1, NUM_CLASSES, H, W, D] (one-hot)
                # label_sample: [1, 1, H, W, D] -> label: [1, NUM_CLASSES, H, W, D] (one-hot)
                
                # Get predictions as class indices then one-hot
                pred_classes = torch.argmax(logits, dim=1, keepdim=True)  # [1, 1, H, W, D]
                pred_onehot = torch.zeros_like(logits)  # [1, NUM_CLASSES, H, W, D]
                pred_onehot.scatter_(1, pred_classes, 1)
                
                # Convert labels to one-hot (label values are class indices 0 to NUM_CLASSES-1)
                label_squeezed = label_sample.long()  # [1, 1, H, W, D]
                label_onehot = torch.zeros(1, NUM_CLASSES, *label_squeezed.shape[2:], device=device)
                # Clamp label values to valid range to avoid index errors
                label_clamped = label_squeezed.clamp(0, NUM_CLASSES - 1)
                label_onehot.scatter_(1, label_clamped, 1)
                
                if should_debug and not logged_debug and i == 0:
                    gt_unique = torch.unique(label_sample).detach().cpu().numpy().tolist()
                    pred_unique = torch.unique(pred_classes).detach().cpu().numpy().tolist()
                    print(f"  [DEBUG] val uniq labels (epoch {epoch_idx + 1}, batch {batch_idx}): GT={gt_unique}, PRED={pred_unique}")
                    print(f"  [DEBUG] shapes: image={image_sample.shape}, label={label_sample.shape}, logits={logits.shape}")
                    print(f"  [DEBUG] one-hot shapes: pred_onehot={pred_onehot.shape}, label_onehot={label_onehot.shape}")
                    logged_debug = True
                
                # Compute dice - both should be [1, NUM_CLASSES, H, W, D]
                dice_metric(y_pred=pred_onehot, y=label_onehot)
                
                del pred_onehot, label_onehot, logits, image_sample, label_sample, pred_classes
            
            del images, labels
            torch.cuda.empty_cache()  # Help with memory on 6GB GPU

    mean_dice_all, mean_dice_per_class, mean_fg_dice = compute_metrics(dice_metric, NUM_CLASSES)
    return mean_dice_all, mean_dice_per_class, mean_fg_dice


def main():
    parser = argparse.ArgumentParser(description="Training Swin-UNETR from scratch (9 classes).")
    parser.add_argument("--data_root", type=Path, default=Path("data/processed/hvsmr2"), help="Root dir containing imagesTr/ and labelsTr/")
    parser.add_argument("--train_split", type=Path, required=True, help="Path to train split txt (nnU-Net style).")
    parser.add_argument("--val_split", type=Path, required=True, help="Path to val split txt (nnU-Net style).")
    parser.add_argument("--label_root", type=Path, default=None, help="Root dir for labels if not under data_root (defaults to data_root/labelsTr).")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", "--lr_max", dest="lr_max", type=float, default=1e-4, help="Initial / max learning rate")
    parser.add_argument("--lr_min", type=float, default=1e-6, help="Final learning rate at the end of training")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay - low for stable training")
    parser.add_argument("--warmup_epochs", type=int, default=20, help="Warmup epochs - longer for stable start")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for checkpoints/logs.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--roi_size", type=int, nargs=3, default=(96, 96, 96), metavar=("X", "Y", "Z"), help="ROI size - 96 for 6GB VRAM, increase to 128 with more memory")
    parser.add_argument("--patience", type=int, default=60, help="Early stopping patience (validation epochs).")
    args = parser.parse_args()

    print("Training Swin-UNETR from scratch for 9-class segmentation.")
    set_seed(args.seed)
    set_determinism(seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    roi_size = tuple(args.roi_size)

    # The train/val splits are read from the same txt files used by nnU-Net, so Swin-UNETR sees the same images as nnU-Net for a fair comparison.
    train_loader, val_loader = create_hvsmr_loaders(
        data_root=str(args.data_root),
        train_split_file=str(args.train_split),
        val_split_file=str(args.val_split),
        label_root=str(args.label_root) if args.label_root else None,
        roi_size=roi_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    log_and_validate_batch_shape(train_loader, roi_size)
    print(f"Summary: train cases={len(train_loader.dataset)}, val cases={len(val_loader.dataset)}, roi_size={roi_size}, batch_size={args.batch_size}")

    model = create_model(device=device, roi_size=roi_size)

    # CRITICAL FIX: Use PURE DiceLoss - DiceCELoss causes model collapse (predicts only background)
    # Pure Dice loss focuses entirely on segmentation overlap, preventing background dominance
    from monai.losses import DiceLoss
    
    print("Using PURE DiceLoss (NO CrossEntropy) with include_background=False")
    print("This prevents model collapse by focusing entirely on foreground segmentation")
    
    # Pure Dice loss - no CE component which causes model to predict background
    loss_function = DiceLoss(
        include_background=False,  # CRITICAL: Exclude background from loss
        to_onehot_y=True,
        softmax=True,
        reduction="mean",
        smooth_nr=1e-5,
        smooth_dr=1e-5,
    )
    
    # Use higher learning rate for small datasets - needs to learn faster
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay)
    
    # CRITICAL FIX: Add warmup scheduler for stable training
    from torch.optim.lr_scheduler import LinearLR, SequentialLR
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=args.warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs - args.warmup_epochs),
        eta_min=args.lr_min
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs]
    )
    print(
        f"Mode: scratch | Loss: PURE DiceLoss (no CE) | Optimizer: AdamW | "
        f"lr range: {args.lr_max:.3e} -> {args.lr_min:.3e} (warmup={args.warmup_epochs}) | "
        f"weight_decay: {args.weight_decay:.3e} | batch_size: {args.batch_size} | "
        f"roi_size: {roi_size} | max_epochs: {args.epochs} | early_stopping_patience: {args.patience}"
    )

    best_dice = -1.0
    epochs_no_improve = 0
    best_path = args.output_dir / "best_model.pt"
    last_path = args.output_dir / "last_model.pt"

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, device, loss_function, optimizer, epoch_idx=epoch)
        print(f"  Mean train loss: {train_loss:.4f}")

        val_mean_all, val_per_class_mean, val_mean_fg = validate_epoch(
            model, val_loader, device, roi_size, epoch_idx=epoch
        )
        per_class_str = ", ".join(f"{i}:{float(v):.3f}" for i, v in enumerate(val_per_class_mean))
        print(f"  Val mean Dice (all): {val_mean_all:.4f}")
        print(f"  Val mean Dice (fg): {val_mean_fg:.4f}")
        print(f"  Per-class mean Dice: [{per_class_str}]")
        scheduler.step()

        save_checkpoint(model, last_path)
        if val_mean_all > best_dice:
            best_dice = val_mean_all
            epochs_no_improve = 0
            save_checkpoint(model, best_path)
            print(f"  New best model saved to {best_path} (Dice={best_dice:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"  Early stopping triggered (no improvement for {args.patience} epochs).")
                break


if __name__ == "__main__":
    main()
