#!/usr/bin/env python
"""
Train Swin-UNETR with LoRA adapters on attention Q/V projections.
Defaults freeze the backbone so only LoRA params (and any unfrozen heads) train.
"""
# Compile check: python -m py_compile scripts/train_swin_unetr_lora.py scripts/swin_unetr_btcv_setup.py
import argparse
import json
import math
import random
import socket
import subprocess
import sys
from datetime import datetime
from itertools import cycle
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import monai
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
    load_case_ids,
    log_and_validate_batch_shape,
    set_seed,
)
from training_utils import compute_metrics, load_checkpoint  # noqa: E402
from models.lora_utils import (  # noqa: E402
    add_lora_to_swin_unetr,
    count_parameters,
    get_lora_params,
    log_model_params,
    summarize_lora_modules,
)


def dump_run_config(args, output_dir: Path, extra: dict | None = None) -> None:
    def _format_lines(data: dict, indent: int = 0) -> list[str]:
        lines = []
        pad = "  " * indent
        for key in sorted(data):
            val = data[key]
            if isinstance(val, dict):
                lines.append(f"{pad}{key}:")
                lines.extend(_format_lines(val, indent + 1))
            else:
                lines.append(f"{pad}{key}: {val}")
        return lines

    try:
        git_info = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT_DIR), text=True).strip(), "branch": subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(ROOT_DIR), text=True).strip(), "status": subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT_DIR), text=True).strip()}
    except Exception as exc:
        git_info = {"error": str(exc)}

    output_dir.mkdir(parents=True, exist_ok=True)
    label_root_used = args.label_root if args.label_root else Path(args.data_root) / "labelsTr"
    cuda_devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    rare_classes = sorted({int(x) for x in str(args.rare_classes).split(",") if x.strip()})
    args_dict = {k: (str(v) if isinstance(v, Path) else v.item() if isinstance(v, (np.integer, np.floating)) else v) for k, v in vars(args).items()}

    config = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "args": args_dict,
        "run": {"output_dir": str(output_dir)},
        "git": git_info,
        "system": {
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "monai_version": monai.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_devices": cuda_devices,
            "hostname": socket.gethostname(),
        },
        "data": {"data_root": str(args.data_root), "label_root": str(label_root_used) if label_root_used is not None else None, "train_split": str(args.train_split), "val_split": str(args.val_split)},
        "training": {"max_epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size, "roi_size": tuple(args.roi_size), "num_workers": args.num_workers, "seed": args.seed, "determinism": True},
        "optimizer": {"name": "AdamW", "single_group": {"lr": args.lr_max, "weight_decay": args.weight_decay}, "lr_lora": args.lr_lora, "lr_backbone": args.lr_backbone, "wd_lora": args.wd_lora, "wd_backbone": args.wd_backbone, "groupA": {"lr": args.lr_lora, "weight_decay": args.wd_lora}, "groupB": {"lr": args.lr_backbone, "weight_decay": args.wd_backbone}, "freeze_backbone": args.freeze_backbone, "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha},
        "scheduler": {"type": args.sched, "warmup_epochs": args.warmup_epochs, "poly_power": args.poly_power, "decay_to_zero": True, "lr_min": 0.0},
        "loss": {"rare_classes": rare_classes, "rare_dice_w": args.rare_dice_w},
        "model": {"num_classes": NUM_CLASSES},
    }
    if extra:
        config.update(extra)

    json_path = output_dir / "run_config.json"
    txt_path = output_dir / "run_config.txt"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    lines = _format_lines(config)
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[RUN CONFIG]")
    for line in lines:
        print(line)


def _get_loss_attr(loss_fn: torch.nn.Module, dice_obj: object | None, name: str):
    if dice_obj is not None and hasattr(dice_obj, name):
        return getattr(dice_obj, name)
    if hasattr(loss_fn, name):
        return getattr(loss_fn, name)
    return None


def _summarize_loss_config(loss_fn: torch.nn.Module):
    dice_obj = getattr(loss_fn, "dice", None) or getattr(loss_fn, "dice_loss", None)
    include_background = _get_loss_attr(loss_fn, dice_obj, "include_background")
    softmax = _get_loss_attr(loss_fn, dice_obj, "softmax")
    to_onehot_y = _get_loss_attr(loss_fn, dice_obj, "to_onehot_y")
    ce_obj = getattr(loss_fn, "ce", None) or getattr(loss_fn, "ce_loss", None)
    ce_weights = None
    if ce_obj is not None and hasattr(ce_obj, "weight"):
        ce_weights = getattr(ce_obj, "weight")
        if torch.is_tensor(ce_weights):
            ce_weights = ce_weights.detach().cpu().tolist()
    loss_type = type(loss_fn).__name__
    return loss_type, include_background, softmax, to_onehot_y, ce_weights


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
    debug_steps: int = 10,
) -> Tuple[float, int]:
    model.train()
    epoch_loss = 0.0
    steps_processed = 0
    optimizer.zero_grad(set_to_none=True)
    if max_train_batches is None or max_train_batches <= len(loader):
        batch_iter = enumerate(loader)
        enforce_max = max_train_batches
    else:
        loader_iter = cycle(loader)
        batch_iter = ((step, next(loader_iter)) for step in range(max_train_batches))
        enforce_max = None
    for step, batch in batch_iter:
        if enforce_max is not None and step >= enforce_max:
            break
        images = batch["image"].to(device)
        labels = batch["label"].to(device).long()
        step_display = step + 1
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
        if overfit_debug and epoch == 0 and step < debug_steps:
            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                y_unique = torch.unique(labels).detach().cpu().tolist()
                p_unique = torch.unique(pred).detach().cpu().tolist()
                fg_vox = int((labels > 0).sum().item())
                fg_frac = fg_vox / float(labels.numel())
                p_fg_vox = int((pred > 0).sum().item())
                p_fg_frac = p_fg_vox / float(pred.numel())
                loss_value = float(loss.item())
            print(
                f"[TRAIN DBG] e={epoch} step={step_display} fg_frac={fg_frac:.4f} fg_vox={fg_vox} "
                f"yuniq={y_unique} puniq={p_unique} p_fg={p_fg_frac:.4f} loss={loss_value:.4f}"
            )
        loss_to_backprop = loss / grad_accum
        loss_to_backprop.backward()
        if epoch == 0 and step == 0:
            lora_grad_norm = 0.0
            decoder_head_grad_norm = 0.0
            backbone_grad_norm = 0.0
            frozen_backbone_with_grads: list[str] = []
            for name, param in model.named_parameters():
                grad = param.grad
                if grad is None:
                    continue
                grad_norm = grad.norm().item()
                is_lora_param = getattr(param, "_is_lora_param", False)
                if is_lora_param and param.requires_grad:
                    lora_grad_norm += grad_norm
                elif name.startswith("swinViT") and not is_lora_param:
                    backbone_grad_norm += grad_norm
                    if not param.requires_grad:
                        frozen_backbone_with_grads.append(name)
                elif not name.startswith("swinViT") and not is_lora_param:
                    decoder_head_grad_norm += grad_norm
            print(
                f"[DEBUG GRADS] lora_grad_norm={lora_grad_norm:.4e}, decoder_head_grad_norm={decoder_head_grad_norm:.4e}, backbone_grad_norm={backbone_grad_norm:.4e}"
            )
            if frozen_backbone_with_grads:
                print(f"  [WARN] Frozen backbone parameters show non-zero gradients; clearing: {frozen_backbone_with_grads}")
                for name, param in model.named_parameters():
                    if name in frozen_backbone_with_grads:
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
    return epoch_loss / max(1, steps_processed), steps_processed


def validate_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    roi_size: Iterable[int],
    debug: bool = False,
) -> Tuple[float, np.ndarray, float]:
    model.eval()
    post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
    post_label = AsDiscrete(to_onehot=NUM_CLASSES)
    dice_metric = DiceMetric(include_background=True, reduction="none", num_classes=NUM_CLASSES)
    n_seen = 0
    n_used = 0
    n_skipped = 0

    warned_missing_orig = False
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            n_seen += 1
            if batch_idx == 0:
                image_shape = tuple(batch["image"].shape)
                label_shape = tuple(batch["label"].shape)
                print(f"[VAL DEBUG] batch=0 image_shape={image_shape} label_shape={label_shape}")
                roi_size_tuple = tuple(int(x) for x in roi_size)
                label_spatial = label_shape[2:]
                label_meta = batch.get("label_meta_dict") or batch.get("label_meta")
                if isinstance(label_meta, list):
                    label_meta = label_meta[0] if label_meta else None
                orig_shape = None
                orig_key = None
                if isinstance(label_meta, dict):
                    for key in ("original_shape", "spatial_shape", "orig_shape", "original_size"):
                        value = label_meta.get(key)
                        if value is not None:
                            orig_shape = value
                            orig_key = key
                            break
                    if orig_shape is not None:
                        orig_shape = tuple(int(x) for x in orig_shape)
                if orig_shape is not None:
                    if label_spatial == roi_size_tuple and orig_shape != roi_size_tuple:
                        raise AssertionError(
                            "Validation labels appear cropped to roi_size; "
                            f"label_spatial={label_spatial}, roi_size={roi_size_tuple}, "
                            f"original_shape={orig_shape} (meta_key={orig_key})"
                        )
                elif label_spatial == roi_size_tuple and not warned_missing_orig:
                    print(
                        "[VAL WARN] original_shape missing; skipping 'cropped-to-roi' assertion for this batch. "
                        f"label_spatial={label_spatial}, roi_size={roi_size_tuple}, meta_key=None"
                    )
                    warned_missing_orig = True
            images = batch["image"].to(device)
            labels = batch["label"].to(device).long()
            gt_hist = torch.bincount(labels.long().flatten(), minlength=NUM_CLASSES).detach().cpu().tolist()
            gt_max = labels.max().item()
            if gt_max == 0 or sum(gt_hist[1:]) == 0:
                meta = batch.get("image_meta_dict") or batch.get("label_meta_dict") or batch.get("image_meta") or batch.get("label_meta")
                if isinstance(meta, list):
                    meta = meta[0] if meta else None
                filename = meta.get("filename_or_obj") if isinstance(meta, dict) else None
                if filename is None and isinstance(meta, dict):
                    filename = meta.get("filename")
                case_id = str(filename) if filename is not None else f"batch{batch_idx}"
                label_shape = tuple(labels.shape)
                print(f"[VAL SKIP] case={case_id} reason=all_zero_gt gt_hist={gt_hist} label_shape={label_shape}")
                n_skipped += 1
                continue
            logits = sliding_window_inference(images, roi_size=roi_size, sw_batch_size=1, predictor=model)
            probs = torch.softmax(logits, dim=1)
            if debug and batch_idx == 0:
                pred = torch.argmax(probs, dim=1)
                gt_unique = torch.unique(labels).detach().cpu().tolist()
                pred_unique = torch.unique(pred).detach().cpu().tolist()
                print(
                    f"[VAL DBG] img={tuple(images.shape)} lab={tuple(labels.shape)} "
                    f"yuniq={gt_unique} puniq={pred_unique}"
                )

            # Compute Dice per batch and release predictions immediately to avoid caching full volumes.
            preds = [post_pred(i) for i in decollate_batch(probs)]
            labels_list = [post_label(i) for i in decollate_batch(labels)]
            dice_metric(y_pred=preds, y=labels_list)
            n_used += 1

            if batch_idx < 2:
                gt_unique = torch.unique(labels)
                pred_unique = torch.unique(torch.argmax(probs, dim=1))
                print(
                    f"[VAL DEBUG] batch={batch_idx} gt_unique={gt_unique.tolist()} pred_unique={pred_unique.tolist()} "
                    f"gt_hist={gt_hist}"
                )

            del preds, labels_list, logits, images, labels

    print(f"[VAL SUMMARY] used={n_used} skipped={n_skipped} seen={n_seen}")
    if n_used == 0:
        raise RuntimeError("No labeled validation cases found after filtering all-zero GT volumes.")
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
    parser.add_argument("--lr_lora", type=float, default=5e-4, help="Learning rate for LoRA/decoder/head params.")
    parser.add_argument("--lr_backbone", type=float, default=1e-5, help="Learning rate for backbone params.")
    parser.add_argument("--wd_lora", type=float, default=1e-2, help="Weight decay for LoRA/decoder/head params.")
    parser.add_argument("--wd_backbone", type=float, default=1e-4, help="Weight decay for backbone params.")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Linear warmup epochs before decay.")
    parser.add_argument("--sched", choices=("cosine", "poly"), default="cosine", help="LR decay schedule after warmup.")
    parser.add_argument("--poly_power", type=float, default=1.0, help="Polynomial power for poly decay (sched=poly).")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for checkpoints/logs.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--roi_size", type=int, nargs=3, default=(128, 128, 128), metavar=("X", "Y", "Z"))
    parser.add_argument("--pretrained_ckpt", type=Path, default=None, help="Optional path to Swin-UNETR checkpoint (.pt/.pth).")
    parser.add_argument("--patience", type=int, default=60, help="Early stopping patience (validation epochs).")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--freeze_backbone", action="store_true", default=True, help="Freeze backbone parameters (default).")
    parser.add_argument("--no_freeze_backbone", action="store_true", default=False, help="Train full backbone along with LoRA.")
    parser.add_argument("--overfit_case_id", type=str, default=None, help="Case ID to overfit when --overfit_debug is set.")
    parser.add_argument("--min_fg_vox", type=int, default=2000, help="Minimum foreground voxels for crop rejection sampling.")
    parser.add_argument("--crop_max_tries", type=int, default=20, help="Max resample attempts for crop rejection sampling.")
    parser.add_argument("--num_samples_per_volume", type=int, default=None, help="Override num_samples for label-aware crops.")
    parser.add_argument("--rare_bias_78", action="store_true", default=False, help="Bias crop ratios toward classes 7/8.")
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
    parser.add_argument("--ce_bg_w", type=float, default=None, help="Override CE background class weight.")
    parser.add_argument("--ce_w_78", type=float, default=None, help="Override CE weights for classes 7 and 8.")
    parser.add_argument("--lr", "--lr_max", dest="lr_max", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--lr_min", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--weight_decay", type=float, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    do_freeze_backbone = args.freeze_backbone and not args.no_freeze_backbone
    if args.lora_rank <= 0:
        do_freeze_backbone = False
    args.freeze_backbone = do_freeze_backbone

    if args.lr_max is not None or args.lr_min is not None or args.weight_decay is not None:
        print("[WARN] Deprecated --lr/--lr_max/--lr_min/--weight_decay ignored; use --lr_lora/--lr_backbone/--wd_lora/--wd_backbone.")

    print("Training Swin-UNETR with LoRA adapters (Q/V).")
    set_seed(args.seed)
    set_determinism(seed=args.seed)
    if args.overfit_debug:
        seed = args.seed if args.seed is not None else 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if args.num_workers != 0:
            print(f"[OVERFIT DEBUG] Forcing num_workers=0 (was {args.num_workers}).")
            args.num_workers = 0
        if args.max_train_batches is None or args.max_train_batches < 128:
            print(f"[OVERFIT DEBUG] Setting max_train_batches=128 (was {args.max_train_batches}).")
            args.max_train_batches = 128
        if args.patience < args.epochs * 10:
            print(f"[OVERFIT DEBUG] Increasing patience to {args.epochs * 10}.")
            args.patience = args.epochs * 10
        if args.lr_lora != 1e-3 or args.lr_backbone != 1e-3:
            print(
                f"[OVERFIT_DEBUG] Overriding lr_lora/lr_backbone to 1e-3 (was {args.lr_lora}, {args.lr_backbone})."
            )
        args.lr_lora = 1e-3
        args.lr_backbone = 1e-3
        print(f"[OVERFIT_DEBUG] lr_lora={args.lr_lora} lr_backbone={args.lr_backbone}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    roi_size = tuple(args.roi_size)
    print(f"Gradient accumulation: grad_accum={args.grad_accum}")

    rare_classes = sorted({int(x) for x in args.rare_classes.split(",") if x.strip()})
    print(f"Rare-class Dice regularizer -> classes={rare_classes}, weight={args.rare_dice_w}")
    dump_run_config(args, args.output_dir)

    # The train/val splits are read from the same txt files used by nnU-Net, so Swin-UNETR sees the same images as nnU-Net for a fair comparison.
    train_ids = None
    val_ids = None
    overfit_case_id = None
    if args.overfit_debug:
        if args.overfit_case_id:
            overfit_case_id = args.overfit_case_id
        else:
            train_ids = load_case_ids(str(args.train_split))
            val_ids = load_case_ids(str(args.val_split))
            if train_ids:
                overfit_case_id = train_ids[0]
            elif val_ids:
                overfit_case_id = val_ids[0]
        if overfit_case_id is None:
            default_id = "pat17"
            img_root = Path(args.data_root) / "imagesTr"
            search_root = img_root if img_root.is_dir() else Path(args.data_root)
            matches = list(search_root.glob(f"{default_id}*.nii*"))
            if matches:
                overfit_case_id = default_id
            else:
                print(f"[OVERFIT DEBUG] Default case_id {default_id} not found under {search_root}; using it anyway.")
                overfit_case_id = default_id
        train_ids = [overfit_case_id]
        val_ids = [overfit_case_id]
        print(f"[OVERFIT_DEBUG] case_id={overfit_case_id} train=1 val=1")
        print(
            "[OVERFIT_DEBUG] Example command:\n"
            "  python -u scripts/train_swin_unetr_lora.py \\\n"
            "    --data_root data/processed/hvsmr2 \\\n"
            "    --label_root data/processed/hvsmr2/labelsTr_matched \\\n"
            "    --train_split data/splits/train_L40.txt \\\n"
            "    --val_split data/splits/val_ids.txt \\\n"
            "    --output_dir runs/overfit_pat17_debug \\\n"
            "    --epochs 120 \\\n"
            "    --batch_size 1 \\\n"
            "    --roi_size 128 128 128 \\\n"
            "    --num_workers 0 \\\n"
            "    --max_train_batches 128 \\\n"
            f"    --overfit_debug --overfit_case_id {overfit_case_id}"
        )
    if train_ids is None:
        train_ids = load_case_ids(str(args.train_split))
    if val_ids is None:
        val_ids = load_case_ids(str(args.val_split))
    print(f"[IDS] train_ids={train_ids} val_ids={val_ids}")

    enable_crop_rejection = args.overfit_debug or ("--min_fg_vox" in sys.argv or "--crop_max_tries" in sys.argv)
    min_fg_vox = args.min_fg_vox if enable_crop_rejection else 0
    crop_max_tries = args.crop_max_tries if enable_crop_rejection else 1
    overfit_train_steps = args.max_train_batches if args.overfit_debug else None
    print(
        "[TRAINING] "
        f"num_samples_per_volume={args.num_samples_per_volume}, "
        f"rare_bias_78={args.rare_bias_78}, "
        f"ce_bg_w={args.ce_bg_w}, "
        f"ce_w_78={args.ce_w_78}, "
        f"min_fg_vox={min_fg_vox}, "
        f"crop_max_tries={crop_max_tries}, "
        f"max_train_batches={args.max_train_batches}"
    )

    train_loader, val_loader = create_hvsmr_loaders(
        data_root=str(args.data_root),
        train_split_file=str(args.train_split),
        val_split_file=str(args.val_split),
        train_ids=train_ids,
        val_ids=val_ids,
        label_root=str(args.label_root) if args.label_root else None,
        roi_size=roi_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        overfit_debug=args.overfit_debug,
        overfit_case_id=overfit_case_id,
        overfit_train_steps=overfit_train_steps,
        min_fg_vox=min_fg_vox,
        crop_max_tries=crop_max_tries,
        enable_crop_rejection=enable_crop_rejection,
        num_samples_per_volume=args.num_samples_per_volume,
        rare_bias_78=args.rare_bias_78,
    )
    print(f"[TRAIN LOADER] len(train_loader)={len(train_loader)} max_train_batches={args.max_train_batches}")
    log_and_validate_batch_shape(train_loader, roi_size)
    print(f"Summary: train cases={len(train_loader.dataset)}, val cases={len(val_loader.dataset)}, roi_size={roi_size}, batch_size={args.batch_size}")
    if args.overfit_debug:
        train_len = len(train_loader.dataset)
        val_len = len(val_loader.dataset)
        if train_len != 1 or val_len != 1:
            raise ValueError(
                f"[OVERFIT_DEBUG] expected 1 train/val case but got train={train_len} val={val_len}; check ID override placement."
            )

    model = create_model(device=device, roi_size=roi_size)
    if args.pretrained_ckpt is not None:
        missing, unexpected = load_checkpoint(model, args.pretrained_ckpt, filter_mismatch=True)
        print(f"Loaded pretrained Swin-UNETR weights. Missing: {missing}, Unexpected: {unexpected}")

    if args.lora_rank > 0:
        model = add_lora_to_swin_unetr(
            model,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            freeze_backbone=False,
        )
        summarize_lora_modules(model)
    else:
        # lora_rank == 0: train as usual
        do_freeze_backbone = False
        args.freeze_backbone = False

    backbone_params: list[torch.nn.Parameter] = []
    decoder_head_params: list[torch.nn.Parameter] = []
    lora_param_list: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        is_lora_param = getattr(param, "_is_lora_param", False)
        if name.startswith("swinViT") and not is_lora_param:
            backbone_params.append(param)
        root = name.split(".")[0]
        if root.startswith("decoder") or root.startswith("out"):
            decoder_head_params.append(param)
        if is_lora_param:
            lora_param_list.append(param)

    for param in backbone_params:
        param.requires_grad = not do_freeze_backbone
    for param in decoder_head_params:
        param.requires_grad = True
    for param in lora_param_list:
        param.requires_grad = True

    if do_freeze_backbone:
        counts = {"lora": 0, "decoder": 0, "head": 0, "other": 0}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            root = name.split(".")[0]
            if getattr(param, "_is_lora_param", False):
                counts["lora"] += param.numel()
            elif root.startswith("decoder"):
                counts["decoder"] += param.numel()
            elif root.startswith("out"):
                counts["head"] += param.numel()
            else:
                counts["other"] += param.numel()
        print(
            "Trainable parameter groups -> "
            f"LoRA: {counts['lora']:,} | Decoder: {counts['decoder']:,} | Head: {counts['head']:,} | Other trainable: {counts['other']:,}"
        )

    if args.no_freeze_backbone:
        assert any(p.requires_grad for p in backbone_params), "Expected at least one trainable backbone parameter with --no_freeze_backbone."
    if args.freeze_backbone and not args.no_freeze_backbone:
        assert all(not p.requires_grad for p in backbone_params), "Expected all backbone parameters frozen with --freeze_backbone."

    total_params, trainable_params = count_parameters(model)
    lora_params = get_lora_params(model)
    lora_param_count = sum(p.numel() for p in lora_params)
    print(f"Params summary -> total={total_params:,} | trainable={trainable_params:,}")
    print(
        f"Mode: lora (rank={args.lora_rank}, alpha={args.lora_alpha}) | Optimizer: AdamW | "
        f"lr_lora: {args.lr_lora:.3e} | lr_backbone: {args.lr_backbone:.3e} | "
        f"wd_lora: {args.wd_lora:.3e} | wd_backbone: {args.wd_backbone:.3e} | "
        f"sched: {args.sched} | warmup_epochs: {args.warmup_epochs} | poly_power: {args.poly_power:.2f} | "
        f"batch_size: {args.batch_size} | roi_size: {roi_size} | max_epochs: {args.epochs} | "
        f"early_stopping_patience: {args.patience} | freeze_backbone: {args.freeze_backbone}"
    )
    log_model_params(model)

    # Quick sanity check
    with torch.no_grad():
        dummy = torch.randn((1, 1, *roi_size), device=device)
        _ = model(dummy)

    use_custom_ce = args.ce_bg_w is not None or args.ce_w_78 is not None
    if use_custom_ce:
        ce_weights = torch.ones(NUM_CLASSES, dtype=torch.float32, device=device)
        if args.ce_bg_w is not None:
            ce_weights[0] = float(args.ce_bg_w)
        if args.ce_w_78 is not None:
            ce_weights[7] = float(args.ce_w_78)
            ce_weights[8] = float(args.ce_w_78)
    else:
        ce_weights = torch.tensor(
            [0.02, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            dtype=torch.float32,
            device=device,
        )  # low background weight for CE
    loss_function = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
        weight=ce_weights,
    )
    loss_type, include_bg, softmax, to_onehot_y, ce_weights_print = _summarize_loss_config(loss_function)
    if use_custom_ce:
        print(
            "[LOSS] "
            f"dice(include_bg={include_bg}, softmax={softmax}, onehot={to_onehot_y}), "
            f"ce(weights={ce_weights_print})"
        )
    else:
        lambda_dice = getattr(loss_function, "lambda_dice", 1.0)
        lambda_ce = getattr(loss_function, "lambda_ce", 1.0)
        print(
            "[LOSS] "
            f"type={loss_type} "
            f"dice(include_bg={include_bg}, softmax={softmax}, onehot={to_onehot_y}) "
            f"ce(weights={ce_weights_print}) total=dice+ce, "
            f"lambda_dice={lambda_dice}, lambda_ce={lambda_ce}"
        )
    group_a_params: list[torch.nn.Parameter] = []
    group_b_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_lora_param = getattr(param, "_is_lora_param", False)
        if is_lora_param or not name.startswith("swinViT"):
            group_a_params.append(param)
        else:
            group_b_params.append(param)
    group_a_count = sum(p.numel() for p in group_a_params)
    group_b_count = sum(p.numel() for p in group_b_params)
    trainable_count = group_a_count + group_b_count
    group_a_tensors = len(group_a_params)
    group_b_tensors = len(group_b_params)
    print(
        "Optimizer parameter groups -> "
        f"trainable_total={trainable_count:,} | "
        f"groupA(lora/non-backbone)={group_a_count:,} ({group_a_tensors} tensors) | "
        f"groupB(backbone)={group_b_count:,} ({group_b_tensors} tensors)"
    )
    if trainable_count != trainable_params:
        print(f"[WARN] Trainable param mismatch: count_parameters={trainable_params:,} vs grouped={trainable_count:,}.")
    print(
        "Group hyperparams -> "
        f"groupA lr={args.lr_lora:.3e}, wd={args.wd_lora:.3e} | "
        f"groupB lr={args.lr_backbone:.3e}, wd={args.wd_backbone:.3e}"
    )
    if not group_b_params:
        print("Backbone frozen -> groupB is empty; no backbone params will be optimized.")

    optim_groups = [{"params": group_a_params, "lr": args.lr_lora, "weight_decay": args.wd_lora}]
    if group_b_params:
        optim_groups.append({"params": group_b_params, "lr": args.lr_backbone, "weight_decay": args.wd_backbone})
    optimizer = torch.optim.AdamW(optim_groups)
    group_a_lr = optimizer.param_groups[0]["lr"]
    group_a_wd = optimizer.param_groups[0].get("weight_decay", 0.0)
    group_b_lr = None
    group_b_wd = None
    if len(optimizer.param_groups) > 1:
        group_b_lr = optimizer.param_groups[1]["lr"]
        group_b_wd = optimizer.param_groups[1].get("weight_decay", 0.0)
    run_config_extra = {
        "model": {"num_classes": NUM_CLASSES, "total_params": total_params, "trainable_params": trainable_params, "lora_params": lora_param_count},
        "optimizer": {
            "name": "AdamW",
            "single_group": {"lr": args.lr_max, "weight_decay": args.weight_decay},
            "lr_lora": args.lr_lora,
            "lr_backbone": args.lr_backbone,
            "wd_lora": args.wd_lora,
            "wd_backbone": args.wd_backbone,
            "groupA": {"num_params": len(group_a_params), "numel": group_a_count, "lr": group_a_lr, "weight_decay": group_a_wd},
            "groupB": {"num_params": len(group_b_params), "numel": group_b_count, "present": bool(group_b_params), "lr": group_b_lr, "weight_decay": group_b_wd},
            "freeze_backbone": args.freeze_backbone,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "trainable_total_numel": trainable_count,
        },
        "loss": {"ce_weights": ce_weights.detach().cpu().tolist(), "rare_classes": rare_classes, "rare_dice_w": args.rare_dice_w},
    }
    dump_run_config(args, args.output_dir, extra=run_config_extra)

    def lr_scale(epoch: int) -> float:
        if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
            return float(epoch + 1) / float(args.warmup_epochs)
        decay_epochs = args.epochs - args.warmup_epochs
        if decay_epochs <= 1:
            progress = 1.0
        else:
            progress = (epoch - args.warmup_epochs) / (decay_epochs - 1)
        progress = min(max(progress, 0.0), 1.0)
        if args.sched == "cosine":
            return 0.5 * (1 + math.cos(math.pi * progress))
        return (1 - progress) ** args.poly_power

    scheduler = None
    if args.overfit_debug:
        print("[OVERFIT_DEBUG] scheduler=disabled (constant lr)")
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lr_scale] * len(optimizer.param_groups))
        preview_scales = [lr_scale(e) for e in range(4)]
        preview_lora = [f"{args.lr_lora * s:.3e}" for s in preview_scales]
        preview_backbone = [f"{args.lr_backbone * s:.3e}" for s in preview_scales]
        print(f"LR preview (first 4 epochs): groupA={preview_lora} | groupB={preview_backbone}")

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
        if scheduler is not None:
            scheduler.step(epoch)
            current_lrs = scheduler.get_last_lr()
        else:
            current_lrs = [group["lr"] for group in optimizer.param_groups]
        lr_group_a = current_lrs[0] if current_lrs else 0.0
        lr_group_b = current_lrs[1] if len(current_lrs) > 1 else 0.0
        train_loss, train_steps = train_epoch(
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
            debug_steps=10,
        )
        print(f"  Mean train loss: {train_loss:.4f}")
        print(f"  Learning rates: groupA={lr_group_a:.3e} | groupB={lr_group_b:.3e}")
        if args.overfit_debug:
            print(f"[OVERFIT_DEBUG] epoch={epoch + 1} num_train_steps={train_steps}")

        val_mean_all, val_per_class_mean, val_mean_fg = validate_epoch(
            model,
            val_loader,
            device,
            roi_size,
            debug=bool(args.overfit_debug and epoch == 0),
        )
        per_class_str = ", ".join(f"{i}:{float(v):.3f}" for i, v in enumerate(val_per_class_mean))
        per_class_fg_str = ", ".join(f"{i}:{float(val_per_class_mean[i]):.3f}" for i in range(1, NUM_CLASSES))
        print(f"  Val mean Dice (all): {val_mean_all:.4f}")
        print(f"  Val mean Dice (fg): {val_mean_fg:.4f}")
        print(f"  Per-class mean Dice: [{per_class_str}]")
        print(f"  Per-class mean Dice (1-8): [{per_class_fg_str}]")

        save_training_state(last_path, epoch=epoch, best_metric=best_dice)
        if val_mean_fg > best_dice:
            prev_best = best_dice
            best_dice = val_mean_fg
            epochs_no_improve = 0
            save_training_state(best_path, epoch=epoch, best_metric=best_dice)
            print(f"  New best model saved to {best_path} (FG Dice={best_dice:.4f}, prev_best={prev_best:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"  Early stopping triggered (no improvement for {args.patience} epochs).")
                break


if __name__ == "__main__":
    main()
