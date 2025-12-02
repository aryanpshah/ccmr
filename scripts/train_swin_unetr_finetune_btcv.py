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
    parser = argparse.ArgumentParser(description="Fine-tune Swin-UNETR with BTCV weights (9 classes).")
    parser.add_argument("--data_root", type=Path, default=Path("data/processed/hvsmr2"), help="Root dir containing imagesTr/ and labelsTr/")
    parser.add_argument("--train_split", type=Path, required=True, help="Path to train split txt (nnU-Net style).")
    parser.add_argument("--val_split", type=Path, required=True, help="Path to val split txt (nnU-Net style).")
    parser.add_argument("--label_root", type=Path, default=None, help="Root dir for labels if not under data_root (defaults to data_root/labelsTr).")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for checkpoints/logs.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--roi_size", type=int, nargs=3, default=DEFAULT_ROI_SIZE, metavar=("X", "Y", "Z"))
    parser.add_argument("--pretrained_ckpt", type=Path, required=True, help="Path to BTCV Swin-UNETR checkpoint (.pt/.pth).")
    args = parser.parse_args()

    print("Fine-tuning Swin-UNETR from BTCV pre-trained weights for 9-class segmentation.")
    set_seed(args.seed)
    set_determinism(seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # The train/val splits are read from the same txt files used by nnU-Net, so Swin-UNETR sees the same images as nnU-Net for a fair comparison.
    train_loader, val_loader = create_hvsmr_loaders(
        data_root=str(args.data_root),
        train_split_file=str(args.train_split),
        val_split_file=str(args.val_split),
        label_root=str(args.label_root) if args.label_root else None,
        roi_size=tuple(args.roi_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    roi_size = tuple(args.roi_size)
    log_and_validate_batch_shape(train_loader, roi_size)
    print(f"Summary: train cases={len(train_loader.dataset)}, val cases={len(val_loader.dataset)}, roi_size={roi_size}, batch_size={args.batch_size}")

    model = create_model(device=device, roi_size=roi_size)
    missing, unexpected = load_checkpoint(model, args.pretrained_ckpt, filter_mismatch=True)
    print(f"Loaded BTCV Swin-UNETR weights. Missing: {missing}, Unexpected: {unexpected}")

    loss_function = DiceCELoss(to_onehot_y=True, softmax=True, include_background=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_dice = -1.0
    best_path = args.output_dir / "best_model.pt"
    last_path = args.output_dir / "last_model.pt"

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, device, loss_function, optimizer)
        scheduler.step()
        print(f"  Mean train loss: {train_loss:.4f}")

        val_mean_all, val_per_class_mean, val_mean_fg = validate_epoch(model, val_loader, device, roi_size)
        per_class_str = ", ".join(f"{i}:{float(v):.3f}" for i, v in enumerate(val_per_class_mean))
        print(f"  Val mean Dice (all): {val_mean_all:.4f}")
        print(f"  Val mean Dice (fg): {val_mean_fg:.4f}")
        print(f"  Per-class mean Dice: [{per_class_str}]")

        save_checkpoint(model, last_path)
        if val_mean_all > best_dice:
            best_dice = val_mean_all
            save_checkpoint(model, best_path)
            print(f"  New best model saved to {best_path} (Dice={best_dice:.4f})")


if __name__ == "__main__":
    main()
