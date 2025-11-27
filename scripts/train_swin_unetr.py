
import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from scipy import ndimage

from monai.data import DataLoader, Dataset, decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandRotate90d,
    ScaleIntensityd,
    EnsureTyped,
)

# Configuration matches preprocess_hvsmr.py
TARGET_SHAPE = (192, 192, 192)
TARGET_SPACING = (1.0, 1.0, 1.0)
RAW_MASK_ROOT = Path("data/raw/HVSMR2/cropped_norm")
PROC_IMG_DIR = Path("data/processed/images")
SPLIT_DIR = Path("data/splits")

def load_ids(split_path: Path) -> List[str]:
    """Load case IDs from a split file."""
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    return [line.strip() for line in split_path.read_text().splitlines() if line.strip()]

def center_crop_or_pad(volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """Symmetrically crop or pad to reach `target_shape`."""
    result = volume
    for axis, target in enumerate(target_shape):
        current = result.shape[axis]
        if current > target:
            start = (current - target) // 2
            end = start + target
            slices = [slice(None)] * result.ndim
            slices[axis] = slice(start, end)
            result = result[tuple(slices)]
        elif current < target:
            pad_total = target - current
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            pad_widths = [(0, 0)] * result.ndim
            pad_widths[axis] = (pad_before, pad_after)
            result = np.pad(result, pad_widths, mode="constant")
    return result

class HVSMRDataset(Dataset):
    """
    Dataset that loads preprocessed images and raw masks, 
    processing masks on-the-fly to match the images.
    """
    def __init__(self, case_ids: List[str], transforms=None):
        self.case_ids = case_ids
        self.transforms = transforms

    def __len__(self):
        return len(self.case_ids)

    def _load_and_process_mask(self, case_id: str) -> np.ndarray:
        """Load raw mask, resample to target spacing, and crop/pad."""
        base = RAW_MASK_ROOT / f"{case_id}_cropped_seg.nii"
        candidates = [
            base / base.name,
            base.with_suffix(".nii.gz"),
            base,
        ]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Mask not found for {case_id}")

        img = nib.load(path)
        data = np.asarray(img.dataobj)
        mask = np.transpose(data, (2, 1, 0)) # (z, y, x)

        zooms = img.header.get_zooms()[:3]
        current_spacing = (zooms[2], zooms[1], zooms[0])
        zoom_factors = tuple(cs / ts for cs, ts in zip(current_spacing, TARGET_SPACING))

        # Nearest neighbor interpolation for labels
        resampled = ndimage.zoom(mask, zoom=zoom_factors, order=0)
        processed = center_crop_or_pad(resampled, TARGET_SHAPE)
        return processed.astype(np.uint8)

    def _load_processed_image(self, case_id: str) -> np.ndarray:
        """Load preprocessed image."""
        path = PROC_IMG_DIR / f"{case_id}_img_proc.nii.gz"
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        img = nib.load(path)
        data = np.asarray(img.dataobj, dtype=np.float32)
        return np.transpose(data, (2, 1, 0)) # (z, y, x)

    def __getitem__(self, index):
        case_id = self.case_ids[index]
        
        img = self._load_processed_image(case_id)
        # Add channel dim
        img = img[None, ...] 
        
        mask = self._load_and_process_mask(case_id)
        # Add channel dim
        mask = mask[None, ...]

        data = {"image": img, "label": mask}

        if self.transforms:
            data = self.transforms(data)
            
        return data

def get_transforms(mode="train"):
    if mode == "train":
        return Compose([
            EnsureTyped(keys=["image", "label"]),
            ScaleIntensityd(keys=["image"]),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        ])
    else:
        return Compose([
            EnsureTyped(keys=["image", "label"]),
            ScaleIntensityd(keys=["image"]),
        ])

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data
    train_ids = load_ids(SPLIT_DIR / "train_ids.txt")
    val_ids = load_ids(SPLIT_DIR / "val_ids.txt")
    
    if args.limit:
        train_ids = train_ids[:args.limit]
        val_ids = val_ids[:args.limit]

    print(f"Training on {len(train_ids)} cases, validating on {len(val_ids)} cases.")

    train_ds = HVSMRDataset(train_ids, transforms=get_transforms("train"))
    val_ds = HVSMRDataset(val_ids, transforms=get_transforms("val"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.workers)

    # 2. Model
    # HVSMR raw data seems to have labels 0-8
    model = SwinUNETR(
        in_channels=1,
        out_channels=9, 
        feature_size=args.feature_size,
        use_checkpoint=True,
        spatial_dims=3,
    ).to(device)

    if args.resume:
        print(f"Resuming from {args.resume}")
        model.load_state_dict(torch.load(args.resume))

    # 3. Optimization
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    # 4. Training Loop
    writer = SummaryWriter(log_dir=args.log_dir)
    best_metric = -1
    best_metric_epoch = -1

    for epoch in range(args.epochs):
        print(f"-" * 10)
        print(f"Epoch {epoch + 1}/{args.epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_data in pbar:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
            # Log step loss
            writer.add_scalar("train_loss", loss.item(), epoch * len(train_loader) + step)

        epoch_loss /= step
        print(f"Average train loss: {epoch_loss:.4f}")

        # Validation
        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in tqdm(val_loader, desc="Validation"):
                    val_inputs, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                    val_outputs = model(val_inputs)
                    
                    # Compute metric
                    # Post-processing for metric
                    val_outputs = [AsDiscrete(argmax=True, to_onehot=3)(i) for i in decollate_batch(val_outputs)]
                    val_labels = [AsDiscrete(to_onehot=3)(i) for i in decollate_batch(val_labels)]
                    
                    dice_metric(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                dice_metric.reset()

                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(args.log_dir, "best_metric_model.pth"))
                    print(f"New best metric: {best_metric:.4f} at epoch {best_metric_epoch}")
                else:
                    print(f"Metric: {metric:.4f} (Best: {best_metric:.4f} at epoch {best_metric_epoch})")

    print(f"Training completed. Best metric: {best_metric:.4f} at epoch {best_metric_epoch}")
    writer.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--log_dir", type=str, default="results/swin_unetr_runs")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of training cases for debugging")
    parser.add_argument("--feature_size", type=int, default=48, help="Feature size for SwinUNETR")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    train(args)

if __name__ == "__main__":
    main()

