#!/usr/bin/env python
"""
Train Swin-UNETR with LoRA adapters on attention Q/V projections.
Defaults freeze the backbone so only LoRA params (and any unfrozen heads) train.
"""
import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot
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
from training_utils import compute_metrics, load_checkpoint  # noqa: E402
from models.lora_utils import (  # noqa: E402
    add_lora_to_swin_unetr,
    count_parameters,
    freeze_backbone_enable_lora_and_decoder,
    get_lora_params,
    log_model_params,
    summarize_lora_modules,
)


def train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_function: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    grad_accum: int,
    rare_dice_w: float,
    rare_classes: list[int],
    overfit_debug: bool = False,
    max_train_batches: Optional[int] = None,
) -> float:
    model.train()
    epoch_loss = 0.0
    steps_processed = 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        if max_train_batches is not None and step >= max_train_batches:
            break
        images = batch["image"].to(device)
        labels = batch["label"].to(device).long()
        step_display = step + 1
        if overfit_debug and epoch == 0 and step < 20:
            gt_unique = torch.unique(labels).detach().cpu().tolist()
            gt_hist = torch.bincount(labels.flatten(), minlength=NUM_CLASSES).detach().cpu().tolist()
            print(f"[TRAIN DEBUG] step={step_display} gt_unique={gt_unique} gt_hist={gt_hist}")
        assert torch.all((labels >= 0) & (labels < NUM_CLASSES)), f"Found invalid label values: {torch.unique(labels)}"
        logits = model(images)
        base_loss = loss_function(logits, labels)

        rare_dice_loss = torch.tensor(0.0, device=device, dtype=logits.dtype)
        if rare_dice_w > 0.0 and rare_classes:
            probs = torch.softmax(logits, dim=1)
            y_onehot = one_hot(labels, num_classes=NUM_CLASSES)
            y_r = y_onehot[:, rare_classes]
            rare_voxels = y_r.sum()
            if rare_voxels.item() > 0:
                p_r = probs[:, rare_classes]
                intersect = (p_r * y_r).sum()
                denom = p_r.sum() + rare_voxels
                eps = 1e-6
                dice = (2 * intersect + eps) / (denom + eps)
                rare_dice_loss = 1.0 - dice

        loss = base_loss + rare_dice_w * rare_dice_loss
        loss_to_backprop = loss / grad_accum
        loss_to_backprop.backward()
        if epoch == 0 and step == 0:
            lora_grad_norm = 0.0
            decoder_head_grad_norm = 0.0
            frozen_grad_norm = 0.0
            frozen_with_grads: list[str] = []
            for name, param in model.named_parameters():
                grad = param.grad
                if grad is None:
                    continue
                grad_norm = grad.norm().item()
                name_lower = name.lower()
                if "lora" in name_lower and param.requires_grad:
                    lora_grad_norm += grad_norm
                elif not name.startswith("swinViT") and "lora" not in name_lower:
                    decoder_head_grad_norm += grad_norm
                elif name.startswith("swinViT") and not param.requires_grad:
                    frozen_grad_norm += grad_norm
                    frozen_with_grads.append(name)
            print(
                f"[DEBUG GRADS] lora_grad_norm={lora_grad_norm:.4e}, decoder_head_grad_norm={decoder_head_grad_norm:.4e}, frozen_grad_norm={frozen_grad_norm:.4e}"
            )
            if frozen_grad_norm > 1e-8:
                print(f"  [WARN] Frozen parameters show non-zero gradients; clearing: {frozen_with_grads}")
                for name, param in model.named_parameters():
                    if name in frozen_with_grads:
                        param.grad = None
        if (step + 1) % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Cache scalar before freeing tensors to avoid keeping computation graphs/activations.
        loss_value = loss.item()
        epoch_loss += loss_value
        steps_processed += 1
        if step_display == 1 or step_display % 5 == 0:
            print(f"  train step {step_display:03d} - loss: {loss_value:.4f}")

        # Drop references so per-step allocations can be released promptly.
        del loss, loss_to_backprop, logits, images, labels
    if steps_processed % grad_accum != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return epoch_loss / max(1, steps_processed)


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
        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device).long()
            logits = sliding_window_inference(images, roi_size=roi_size, sw_batch_size=1, predictor=model)
            probs = torch.softmax(logits, dim=1)

            # Compute Dice per batch and release predictions immediately to avoid caching full volumes.
            preds = [post_pred(i) for i in decollate_batch(probs)]
            labels_list = [post_label(i) for i in decollate_batch(labels)]
            dice_metric(y_pred=preds, y=labels_list)

            if batch_idx < 2:
                gt_unique = torch.unique(labels)
                pred_unique = torch.unique(torch.argmax(probs, dim=1))
                gt_hist = torch.bincount(labels.long().flatten(), minlength=NUM_CLASSES).cpu().tolist()
                print(
                    f"[VAL DEBUG] batch={batch_idx} gt_unique={gt_unique.tolist()} pred_unique={pred_unique.tolist()} "
                    f"gt_hist={gt_hist}"
                )

            del preds, labels_list, logits, images, labels

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
    parser.add_argument("--grad_accum", type=int, default=1, help="Accumulate gradients for N steps before optimizer.step().")
    parser.add_argument("--lr", "--lr_max", dest="lr_max", type=float, default=2e-4, help="Initial / max learning rate.")
    parser.add_argument("--lr_min", type=float, default=2e-5, help="Final learning rate at the end of training.")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Linear warmup epochs before cosine decay.")
    parser.add_argument("--weight_decay", type=float, default=1e-2)
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
    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=None,
        help="If set, limit the number of training batches per epoch (debug only).",
    )
    parser.add_argument("--overfit_debug", action="store_true", help="Overfit on a tiny subset (1-2 cases) with no augmentations.")
    parser.add_argument("--rare_dice_w", type=float, default=0.2, help="Weight for rare-class Dice regularizer.")
    parser.add_argument(
        "--rare_classes",
        type=str,
        default="7,8",
        help="Comma-separated list of rare class indices to regularize (e.g., '7,8').",
    )
    args = parser.parse_args()

    print("Training Swin-UNETR with LoRA adapters (Q/V).")
    set_seed(args.seed)
    set_determinism(seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    roi_size = tuple(args.roi_size)
    print(f"Gradient accumulation: grad_accum={args.grad_accum}")

    rare_classes = sorted({int(x) for x in args.rare_classes.split(",") if x.strip()})
    print(f"Rare-class Dice regularizer -> classes={rare_classes}, weight={args.rare_dice_w}")

    # The train/val splits are read from the same txt files used by nnU-Net, so Swin-UNETR sees the same images as nnU-Net for a fair comparison.
    train_loader, val_loader = create_hvsmr_loaders(
        data_root=str(args.data_root),
        train_split_file=str(args.train_split),
        val_split_file=str(args.val_split),
        label_root=str(args.label_root) if args.label_root else None,
        roi_size=roi_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        overfit_debug=args.overfit_debug,
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
        summarize_lora_modules(model)
    else:
        # lora_rank == 0: train as usual
        args.freeze_backbone = False

    if args.freeze_backbone:
        counts = freeze_backbone_enable_lora_and_decoder(model)
        print(
            "Trainable parameter groups -> "
            f"LoRA: {counts['lora']:,} | Decoder: {counts['decoder']:,} | Head: {counts['head']:,} | Other (still frozen): {counts['other']:,}"
        )

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
        dummy = torch.randn((1, 1, *roi_size), device=device)
        _ = model(dummy)

    ce_weights = torch.tensor(
        [0.05, 0.2, 0.2, 0.2, 0.2, 1.0, 1.0, 1.2, 1.2],
        dtype=torch.float32,
        device=device,
    )  # modestly upweight rare classes (7-8) to stabilize CE
    loss_function = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=True,
        weight=ce_weights,
    )
    trainable_params_for_optim = [p for p in model.parameters() if p.requires_grad]
    opt_param_count = sum(p.numel() for p in trainable_params_for_optim)
    print(f"Optimizer will update {opt_param_count:,} parameters (matches trainable count above).")
    optimizer = torch.optim.AdamW(trainable_params_for_optim, lr=args.lr_max, weight_decay=args.weight_decay)

    def lr_lambda(epoch: int) -> float:
        if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
            return float(epoch + 1) / float(args.warmup_epochs)
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        min_scale = args.lr_min / args.lr_max
        return min_scale + (1 - min_scale) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    print(f"LR preview (first 4 epochs): {[f'{args.lr_max * lr_lambda(e):.3e}' for e in range(4)]}")

    best_dice = -1.0
    epochs_no_improve = 0
    best_path = args.output_dir / "best_model.pth"
    last_path = args.output_dir / "last_model.pth"

    def save_training_state(path: Path, epoch: int, best_metric: float) -> None:
        state = {
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_dice": best_metric,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        scheduler.step(epoch)
        current_lr = scheduler.get_last_lr()[0]
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        train_loss = train_epoch(
            model,
            train_loader,
            device,
            loss_function,
            optimizer,
            epoch,
            grad_accum=args.grad_accum,
            rare_dice_w=args.rare_dice_w,
            rare_classes=rare_classes,
            overfit_debug=args.overfit_debug,
            max_train_batches=args.max_train_batches,
        )
        print(f"  Mean train loss: {train_loss:.4f}")
        print(f"  Learning rate: {current_lr:.3e}")

        val_mean_all, val_per_class_mean, val_mean_fg = validate_epoch(model, val_loader, device, roi_size)
        per_class_str = ", ".join(f"{i}:{float(v):.3f}" for i, v in enumerate(val_per_class_mean))
        per_class_fg_str = ", ".join(f"{i}:{float(val_per_class_mean[i]):.3f}" for i in range(1, NUM_CLASSES))
        print(f"  Val mean Dice (all): {val_mean_all:.4f}")
        print(f"  Val mean Dice (fg): {val_mean_fg:.4f}")
        print(f"  Per-class mean Dice: [{per_class_str}]")
        print(f"  Per-class mean Dice (1-8): [{per_class_fg_str}]")

        save_training_state(last_path, epoch=epoch, best_metric=best_dice)
        if val_mean_all > best_dice:
            prev_best = best_dice
            best_dice = val_mean_all
            epochs_no_improve = 0
            save_training_state(best_path, epoch=epoch, best_metric=best_dice)
            print(f"  New best model saved to {best_path} (Dice={best_dice:.4f}, prev_best={prev_best:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"  Early stopping triggered (no improvement for {args.patience} epochs).")
                break


if __name__ == "__main__":
    main()
