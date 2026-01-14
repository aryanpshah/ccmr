#!/usr/bin/env python
"""
Fine-tune Swin-UNETR from BTCV pre-trained weights for 9-class segmentation
(background + 8 structures).
"""
import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
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
from training_utils import compute_metrics, load_checkpoint, save_checkpoint  # noqa: E402


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
        logits = model(images)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()

        # Grab scalar values early and then drop tensor references to avoid keeping graphs/activations.
        loss_value = loss.item()
        epoch_loss += loss_value
        if step == 1 or step % 5 == 0:
            print(f"  train step {step:03d} - loss: {loss_value:.4f}")

        # Explicitly free large tensors so nothing lingers past this iteration.
        del loss, logits, images, labels
    return epoch_loss / max(1, len(loader))


def validate_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    roi_size: Iterable[int],
) -> Tuple[float, np.ndarray, float]:
    model.eval()
    post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
    post_label = AsDiscrete(to_onehot=NUM_CLASSES)
    dice_metric = DiceMetric(include_background=True, reduction="none", num_classes=NUM_CLASSES)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = sliding_window_inference(images, roi_size=roi_size, sw_batch_size=1, predictor=model)

            # Compute Dice per batch and immediately release predictions/labels to avoid retaining full volumes.
            preds = [post_pred(i) for i in decollate_batch(logits)]
            labels_list = [post_label(i) for i in decollate_batch(labels)]
            dice_metric(y_pred=preds, y=labels_list)

            del preds, labels_list, logits, images, labels

    mean_dice_all, mean_dice_per_class, mean_fg_dice = compute_metrics(dice_metric, NUM_CLASSES)
    return mean_dice_all, mean_dice_per_class, mean_fg_dice


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Swin-UNETR with BTCV weights (9 classes).")
    parser.add_argument("--data_root", type=Path, default=Path("data/processed/hvsmr2"), help="Root dir containing imagesTr/ and labelsTr/")
    parser.add_argument("--train_split", type=Path, required=True, help="Path to train split txt (nnU-Net style).")
    parser.add_argument("--val_split", type=Path, required=True, help="Path to val split txt (nnU-Net style).")
    parser.add_argument("--label_root", type=Path, default=None, help="Root dir for labels if not under data_root (defaults to data_root/labelsTr).")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", "--lr_max", dest="lr_max", type=float, default=3e-5, help="Initial / max learning rate. CRITICAL FIX: Reduced from 6e-5 to 3e-5")
    parser.add_argument("--lr_min", type=float, default=3e-6, help="Final learning rate at the end of training. CRITICAL FIX: Reduced from 6e-6 to 3e-6")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay. CRITICAL FIX: Increased from 3e-3 to 1e-2")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="CRITICAL FIX: Number of warmup epochs for learning rate")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for checkpoints/logs.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--roi_size", type=int, nargs=3, default=(160, 160, 160), metavar=("X", "Y", "Z"), help="CRITICAL FIX: Increased default from 128 to 160")
    parser.add_argument("--pretrained_ckpt", type=Path, required=True, help="Path to BTCV Swin-UNETR checkpoint (.pt/.pth).")
    parser.add_argument("--patience", type=int, default=60, help="Early stopping patience (validation epochs).")
    args = parser.parse_args()

    print("Fine-tuning Swin-UNETR from BTCV pre-trained weights for 9-class segmentation.")
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
    missing, unexpected = load_checkpoint(model, args.pretrained_ckpt, filter_mismatch=True)
    print(f"Loaded BTCV Swin-UNETR weights. Missing: {missing}, Unexpected: {unexpected}")

    # Full fine-tune: ensure every parameter remains trainable (no backbone freezing, no LoRA-only params here).
    for param in model.parameters():
        param.requires_grad = True
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_samples = [name for name, p in model.named_parameters() if not p.requires_grad][:10]
    print(f"Parameter counts -> total: {total_params:,} | trainable: {trainable_params:,}")
    if frozen_samples:
        print(f"  [WARN] Found frozen params (should be none in full fine-tune): {frozen_samples}")

    # CRITICAL FIX: Use improved class weights and DiceFocalLoss
    from monai.losses import DiceFocalLoss
    class_weights = torch.tensor([0.01, 5.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.0], device=device)
    print(f"CRITICAL FIX: Using DiceFocalLoss with improved class weights: {class_weights.tolist()}")
    loss_function = DiceFocalLoss(
        include_background=True,
        to_onehot_y=True,
        softmax=True,
        gamma=2.0,  # Focus on hard examples
        lambda_dice=0.7,  # 70% Dice
        lambda_focal=0.3,  # 30% Focal cross-entropy
        weight=class_weights,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay)
    
    # CRITICAL FIX: Add warmup scheduler for stable fine-tuning
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
        f"Mode: finetune | Loss: DiceFocalLoss (gamma=2.0) | Optimizer: AdamW | "
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
        train_loss = train_epoch(
            model,
            train_loader,
            device,
            loss_function,
            optimizer,
            epoch_idx=epoch,
        )
        print(f"  Mean train loss: {train_loss:.4f}")

        val_mean_all, val_per_class_mean, val_mean_fg = validate_epoch(model, val_loader, device, roi_size)
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

    final_epoch = epoch + 1
    print(f"Training completed at epoch {final_epoch}. Best val mean Dice: {best_dice:.4f}.")


if __name__ == "__main__":
    main()
