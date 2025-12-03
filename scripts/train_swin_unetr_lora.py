#!/usr/bin/env python
"""
Train Swin-UNETR with LoRA adapters on attention Q/V projections.
Defaults freeze the backbone so only LoRA params (and any unfrozen heads) train.
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
ROOT_DIR = SCRIPT_DIR.parent
for p in (SCRIPT_DIR, ROOT_DIR):
    if str(p) not in sys.path:
        sys.path.append(str(p))

from swin_unetr_btcv_setup import (  # noqa: E402
    NUM_CLASSES,
    create_model,
    create_hvsmr_loaders,
    log_and_validate_batch_shape,
    set_seed,
)
from training_utils import compute_metrics, load_checkpoint, save_checkpoint  # noqa: E402
from models.lora_utils import (  # noqa: E402
    add_lora_to_swin_unetr,
    count_parameters,
    get_lora_params,
    log_model_params,
)


def train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_function: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    epoch_loss = 0.0
    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if step == 1 or step % 5 == 0:
            print(f"  train step {step:03d} - loss: {loss.item():.4f}")
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
            preds = [post_pred(i) for i in decollate_batch(logits)]
            labels_list = [post_label(i) for i in decollate_batch(labels)]
            dice_metric(y_pred=preds, y=labels_list)

    mean_dice_all, mean_dice_per_class, mean_fg_dice = compute_metrics(dice_metric, NUM_CLASSES)
    return mean_dice_all, mean_dice_per_class, mean_fg_dice


def main():
    parser = argparse.ArgumentParser(description="Parameter-efficient LoRA fine-tune of Swin-UNETR attention (Q/V only).")
    parser.add_argument("--data_root", type=Path, default=Path("data/processed/hvsmr2"), help="Root dir containing imagesTr/ and labelsTr/")
    parser.add_argument("--train_split", type=Path, required=True, help="Path to train split txt (nnU-Net style).")
    parser.add_argument("--val_split", type=Path, required=True, help="Path to val split txt (nnU-Net style).")
    parser.add_argument("--label_root", type=Path, default=None, help="Root dir for labels if not under data_root (defaults to data_root/labelsTr).")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", "--lr_max", dest="lr_max", type=float, default=6e-5, help="Initial / max learning rate.")
    parser.add_argument("--lr_min", type=float, default=6e-6, help="Final learning rate at the end of training.")
    parser.add_argument("--weight_decay", type=float, default=3e-3)
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for checkpoints/logs.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--roi_size", type=int, nargs=3, default=(128, 128, 128), metavar=("X", "Y", "Z"))
    parser.add_argument("--pretrained_ckpt", type=Path, default=None, help="Optional path to Swin-UNETR checkpoint (.pt/.pth).")
    parser.add_argument("--patience", type=int, default=60, help="Early stopping patience (validation epochs).")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--freeze_backbone", dest="freeze_backbone", action="store_true", default=True)
    parser.add_argument("--no_freeze_backbone", dest="freeze_backbone", action="store_false", help="Train full backbone along with LoRA.")
    args = parser.parse_args()

    print("Training Swin-UNETR with LoRA adapters (Q/V).")
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
    if args.pretrained_ckpt is not None:
        missing, unexpected = load_checkpoint(model, args.pretrained_ckpt, filter_mismatch=True)
        print(f"Loaded pretrained Swin-UNETR weights. Missing: {missing}, Unexpected: {unexpected}")

    if args.lora_rank > 0:
        model = add_lora_to_swin_unetr(
            model,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            freeze_backbone=args.freeze_backbone,
        )
    else:
        # lora_rank == 0: train as usual
        args.freeze_backbone = False

    total_params, trainable_params = count_parameters(model)
    lora_params = get_lora_params(model)
    lora_param_count = sum(p.numel() for p in lora_params)
    print(
        f"Mode: lora (rank={args.lora_rank}, alpha={args.lora_alpha}) | Optimizer: AdamW | "
        f"lr range: {args.lr_max:.3e} -> {args.lr_min:.3e} | weight_decay: {args.weight_decay:.3e} | "
        f"batch_size: {args.batch_size} | roi_size: {roi_size} | max_epochs: {args.epochs} | "
        f"early_stopping_patience: {args.patience} | freeze_backbone: {args.freeze_backbone}"
    )
    log_model_params(model)

    # Quick sanity check
    with torch.no_grad():
        dummy = torch.randn(1, 1, 32, 32, 32, device=device)
        _ = model(dummy)

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True, include_background=True)
    lora_params_for_optim = get_lora_params(model)
    if not lora_params_for_optim:
        # Fallback: if nothing is marked as LoRA (e.g., lora_rank=0), optimize all trainable params.
        lora_params_for_optim = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params_for_optim, lr=args.lr_max, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=args.lr_min
    )
    # Preview a few LR steps to verify scheduler wiring (does not affect the real scheduler).
    with torch.no_grad():
        dummy_opt = torch.optim.SGD([torch.zeros(1)], lr=args.lr_max)
        dummy_sched = torch.optim.lr_scheduler.CosineAnnealingLR(dummy_opt, T_max=max(1, args.epochs), eta_min=args.lr_min)
        lr_preview = [dummy_sched.get_last_lr()[0]]
        for _ in range(3):
            dummy_sched.step()
            lr_preview.append(dummy_sched.get_last_lr()[0])
        print(f"LR preview (first 4 steps): {[f'{lr:.3e}' for lr in lr_preview]}")

    best_dice = -1.0
    epochs_no_improve = 0
    best_path = args.output_dir / "best_model.pt"
    last_path = args.output_dir / "last_model.pt"

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, device, loss_function, optimizer)
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


if __name__ == "__main__":
    main()
