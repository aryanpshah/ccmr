#!/usr/bin/env python
"""
Evaluate a trained Swin-UNETR on the HVSMR test set using nnU-Net style splits.
Computes per-class and mean Dice, saving metrics to JSON.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from swin_unetr_btcv_setup import (  # noqa: E402
    NUM_CLASSES,
    build_dataset_dicts,
    create_model,
    load_case_ids,
)


def build_test_loader(data_root: str, test_split: str, num_workers: int) -> DataLoader:
    case_ids = load_case_ids(test_split)
    test_dicts = build_dataset_dicts(data_root, case_ids)

    test_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-1000,
                a_max=1000,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    test_ds = Dataset(data=test_dicts, transform=test_transforms)
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return test_loader


def load_model(checkpoint: Path, device: torch.device, roi_size: Tuple[int, int, int]) -> torch.nn.Module:
    model = create_model(device=device, roi_size=roi_size)
    ckpt = torch.load(checkpoint, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    result = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint from {checkpoint}")
    print(f"Missing keys: {result.missing_keys}, Unexpected keys: {result.unexpected_keys}")
    model.eval()
    return model


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, roi_size: Tuple[int, int, int]):
    dice_metric = DiceMetric(
        include_background=True,
        reduction="none",
        get_not_nans=False,
        num_classes=NUM_CLASSES,
        softmax=False,
    )
    hd_metric = HausdorffDistanceMetric(
        include_background=True,
        reduction="none",
        percentile=95,
        get_not_nans=False,
    )
    to_onehot = AsDiscrete(to_onehot=NUM_CLASSES)
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = sliding_window_inference(
                images,
                roi_size=roi_size,
                sw_batch_size=1,
                predictor=model,
                overlap=0.25,
            )
            preds = torch.argmax(logits, dim=1, keepdim=True)
            preds = [to_onehot(p) for p in decollate_batch(preds)]
            labels_oh = [to_onehot(l) for l in decollate_batch(labels)]
            dice_metric(y_pred=preds, y=labels_oh)
            hd_metric(y_pred=preds, y=labels_oh)

    dice_per_class = dice_metric.aggregate().cpu().numpy()
    hd_per_class = hd_metric.aggregate().cpu().numpy()
    dice_metric.reset()
    hd_metric.reset()
    mean_all = float(dice_per_class.mean())
    mean_fg = float(dice_per_class[1:].mean()) if dice_per_class.shape[0] > 1 else float("nan")
    mean_hd_all = float(np.nanmean(hd_per_class))
    mean_hd_fg = float(np.nanmean(hd_per_class[1:])) if hd_per_class.shape[0] > 1 else float("nan")
    return dice_per_class, mean_all, mean_fg, hd_per_class, mean_hd_all, mean_hd_fg


def main():
    parser = argparse.ArgumentParser(description="Evaluate Swin-UNETR on HVSMR test split.")
    parser.add_argument("--data_root", type=Path, default=Path("data/processed/hvsmr2"), help="Root dir with imagesTr/ and labelsTr/")
    parser.add_argument("--test_split", type=Path, required=True, help="Path to test split txt (nnU-Net style).")
    parser.add_argument("--roi_size", type=int, nargs=3, default=(96, 96, 96), metavar=("X", "Y", "Z"))
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained model checkpoint (.pt/.pth).")
    parser.add_argument("--output_json", type=Path, default=None, help="Where to save metrics JSON.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    roi_size = tuple(args.roi_size)

    test_loader = build_test_loader(str(args.data_root), str(args.test_split), args.num_workers)
    print(f"Test cases: {len(test_loader.dataset)}, roi_size={roi_size}, device={device}")

    model = load_model(args.checkpoint, device=device, roi_size=roi_size)

    dice_per_class, mean_all, mean_fg, hd_per_class, hd_mean_all, hd_mean_fg = evaluate(
        model, test_loader, device, roi_size
    )
    print(f"Mean Dice (all classes): {mean_all:.4f}")
    print(f"Mean Dice (foreground only): {mean_fg:.4f}")
    print(f"Per-class Dice: {dice_per_class}")
    print(f"Mean HD95 (all classes): {hd_mean_all:.4f}")
    print(f"Mean HD95 (foreground only): {hd_mean_fg:.4f}")
    print(f"Per-class HD95: {hd_per_class}")

    metrics = {
        "num_classes": NUM_CLASSES,
        "class_ids": list(range(NUM_CLASSES)),
        "dice_per_class": dice_per_class.tolist(),
        "mean_dice_all_classes": mean_all,
        "mean_dice_foreground": mean_fg,
        "hausdorff95_per_class": hd_per_class.tolist(),
        "mean_hausdorff95_all_classes": hd_mean_all,
        "mean_hausdorff95_foreground": hd_mean_fg,
        "num_test_cases": len(test_loader.dataset),
        "checkpoint": str(args.checkpoint),
        "test_split": str(args.test_split),
    }

    if args.output_json is None:
        out_dir = Path("logs") / "swin_unetr"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_json = out_dir / "hvsmr2_eval_metrics.json"

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {args.output_json}")


if __name__ == "__main__":
    main()
