#!/usr/bin/env python
"""
Minimal MONAI + Swin-UNETR BTCV training/inference scaffold.
Downloads the BTCV Swin-UNETR bundle checkpoint from NGC (or a local path),
sets up deterministic data loaders, and provides train/val + single-volume
inference entry points.
"""
import argparse
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import nibabel as nib
import numpy as np
import torch
from monai.apps import download_url
from monai.data import DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    DivisiblePadd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandRotate90d,
    RandCropByPosNegLabeld,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    NormalizeIntensityd,
    Spacingd,
    ResizeWithPadOrCropd,
    SpatialPadd,
)
from monai.utils import set_determinism
from monai.data import CacheDataset
import monai.transforms.transform as monai_transform_mod
import monai.utils.misc as monai_misc_mod

# Guard against uint32 overflow in MONAI seeding on some platforms.
SAFE_MAX_SEED = np.iinfo(np.uint32).max
monai_transform_mod.MAX_SEED = SAFE_MAX_SEED
monai_misc_mod.MAX_SEED = SAFE_MAX_SEED

try:
    from monai.transforms import AddChanneld as ChannelFirstd  # preferred when available
except ImportError:
    ChannelFirstd = EnsureChannelFirstd  # fallback for current MONAI version

# BTCV bundle specifics (adapted here to 8 foreground structures + background)
NUM_CLASSES = 9  # 9 classes total: label 0 = background, labels 1-8 = structures
DEFAULT_FEATURE_SIZE = 48
DEFAULT_SPACING = (1.0, 1.0, 1.0)  # HVSMR resampled spacing used in loaders
DEFAULT_ROI_SIZE = (96, 96, 96)  # Default patch size; raise/lower depending on GPU memory
DEFAULT_PRETRAINED_URL = (
    # Public NGC bundle; may require `NGC_CLI_API_KEY` for authenticated download.
    "https://api.ngc.nvidia.com/v2/org/nvidia/teams/monaitoolkit/models/"
    "monai_swin_unetr_btcv_segmentation/versions/0.5.6/files/models/model.pt"
)


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)


class SegmentationHead(torch.nn.Module):
    """
    Simple 1x1x1 conv head producing 9-channel logits (background + 8 structures).
    Useful when you want an explicit head layer; Swin-UNETR already outputs logits,
    so we keep this available for clarity and future customization.
    """

    def __init__(self, in_channels: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.conv = torch.nn.Conv3d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def build_datalist(data_dir: Path) -> List[Dict[str, str]]:
    """
    Build MONAI-style list of dicts with "image" and "label" keys.
    Expects /imagesTr and /labelsTr with matching filenames.
    """
    image_dir = data_dir / "imagesTr"
    label_dir = data_dir / "labelsTr"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise ValueError(f"Expected {image_dir} and {label_dir} to exist.")

    image_files = sorted(image_dir.glob("*.nii*"))
    datalist: List[Dict[str, str]] = []
    for img in image_files:
        resolved_img = _resolve_nifti_path(img)
        if resolved_img is None:
            print(f"[WARN] Could not resolve image file {img}; skipping.")
            continue
        stem = resolved_img.name.split(".nii")[0]
        candidates = [
            label_dir / resolved_img.name,
            label_dir / f"{stem}.nii.gz",
            label_dir / f"{stem}.nii",
        ]
        label_path = _first_existing_nifti(candidates)
        if label_path is None:
            print(f"[WARN] Missing label for {img.name}; skipping.")
            continue
        datalist.append({"image": str(resolved_img), "label": str(label_path)})

    if not datalist:
        raise RuntimeError("No image/label pairs were found.")
    return datalist


def load_case_ids(split_file: str) -> List[str]:
    """
    Read a txt file where each line is a case ID (no extension).
    Return a list of string case IDs.
    """
    with open(split_file, "r", encoding="utf-8") as f:
        case_ids = [line.strip() for line in f.readlines() if line.strip()]
    return case_ids


def _resolve_nifti_path(path: Path | None) -> Path | None:
    """
    Return a real NIfTI file path, even if `path` is a directory wrapper (e.g., OneDrive placeholder).
    """
    if path is None:
        return None
    path = Path(path)
    if path.is_file():
        return path
    if path.exists() and path.is_dir():
        same_name = path / path.name  # OneDrive can expose files as folders with the same name
        if same_name.is_file():
            return same_name
        inner = next((p for p in path.glob("*.nii*") if p.is_file()), None)
        if inner is not None:
            return inner
    return None


def _first_existing_nifti(candidates: Iterable[Path]) -> Path | None:
    """Pick the first candidate that resolves to an on-disk NIfTI file."""
    for cand in candidates:
        resolved = _resolve_nifti_path(cand)
        if resolved is not None:
            return resolved
    return None


def build_dataset_dicts(
    data_root: str,
    case_ids: List[str],
    label_root: str | None = None,
) -> List[Dict[str, str]]:
    """
    Build MONAI dataset dicts for a list of case IDs.
    Supports nnU-Net-style (imagesTr/labelsTr) or HVSMR processed naming.
    If label_root is provided, labels are resolved there; otherwise tries
    imagesTr/labelsTr, then defaults to data/raw/HVSMR2/cropped_norm if present.
    """
    data_root = str(data_root)
    img_root = Path(data_root) / "imagesTr" if (Path(data_root) / "imagesTr").is_dir() else Path(data_root)
    # Resolve label root
    if label_root is not None:
        lbl_root = Path(label_root)
    elif (Path(data_root) / "labelsTr").is_dir():
        lbl_root = Path(data_root) / "labelsTr"
    elif Path("data/raw/HVSMR2/cropped_norm").exists():
        lbl_root = Path("data/raw/HVSMR2/cropped_norm")
    else:
        lbl_root = Path(data_root)

    # Optional fallback roots if primary search fails
    fallback_img_roots = [Path("data/processed/images"), Path("data/raw/HVSMR2/cropped_norm")]
    fallback_lbl_roots = [Path("data/raw/HVSMR2/cropped_norm"), Path("data/processed/images")]

    dicts: List[Dict[str, str]] = []
    for cid in case_ids:
        img_candidates = [
            img_root / f"{cid}_image.nii.gz",
            img_root / f"{cid}_image.nii",
            img_root / f"{cid}_img_proc.nii.gz",
            img_root / f"{cid}_img_proc.nii",
            img_root / f"{cid}.nii.gz",
            img_root / f"{cid}.nii",
        ]
        lbl_candidates = [
            lbl_root / f"{cid}_label.nii.gz",
            lbl_root / f"{cid}_label.nii",
            lbl_root / f"{cid}_seg.nii.gz",
            lbl_root / f"{cid}_seg.nii",
            lbl_root / f"{cid}_cropped_seg.nii.gz",
            lbl_root / f"{cid}_cropped_seg.nii",
        ]
        img_path = _first_existing_nifti(img_candidates)
        lbl_path = _first_existing_nifti(lbl_candidates)

        # Fallback search if not found
        if img_path is None:
            for root in fallback_img_roots:
                img_candidates_fb = [
                    root / f"{cid}_img_proc.nii.gz",
                    root / f"{cid}_img_proc.nii",
                    root / f"{cid}_image.nii.gz",
                    root / f"{cid}_image.nii",
                    root / f"{cid}.nii.gz",
                    root / f"{cid}.nii",
                ]
                img_path = _first_existing_nifti(img_candidates_fb)
                if img_path is not None:
                    break

        if lbl_path is None:
            for root in fallback_lbl_roots:
                lbl_candidates_fb = [
                    root / f"{cid}_label.nii.gz",
                    root / f"{cid}_label.nii",
                    root / f"{cid}_seg.nii.gz",
                    root / f"{cid}_seg.nii",
                    root / f"{cid}_cropped_seg.nii.gz",
                    root / f"{cid}_cropped_seg.nii",
                ]
                lbl_path = _first_existing_nifti(lbl_candidates_fb)
                if lbl_path is not None:
                    break

        if img_path is None or lbl_path is None:
            raise FileNotFoundError(
                f"Missing image/label for case {cid}. "
                f"Checked primary roots {img_root},{lbl_root} and fallbacks; "
                f"last tried img candidates {img_candidates} / lbl candidates {lbl_candidates}"
            )
        dicts.append({"image": str(img_path), "label": str(lbl_path)})
    return dicts


def create_hvsmr_loaders(
    data_root: str,
    train_split_file: str,
    val_split_file: str,
    label_root: str | None = None,
    roi_size: Tuple[int, int, int] = (96, 96, 96),
    batch_size: int = 1,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val loaders for HVSMR using nnU-Net split files and 3D patches.
    The train/val splits are read from the same txt files used by nnU-Net, so
    Swin-UNETR sees the same images as nnU-Net for a fair comparison.
    """
    train_ids = load_case_ids(train_split_file)
    val_ids = load_case_ids(val_split_file)
    train_dicts = build_dataset_dicts(data_root, train_ids, label_root=label_root)
    val_dicts = build_dataset_dicts(data_root, val_ids, label_root=label_root)

    spatial_keys = ["image", "label"]
    split_tokens = Path(train_split_file).stem.replace("-", "_").upper().split("_")
    is_l5_split = "L5" in split_tokens
    if is_l5_split:
        print("Detected L5 split -> using lighter augmentations (no elastic / heavy noise).")
    intensity_transforms = (
        [
            RandScaleIntensityd(keys=["image"], factors=0.05, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.05, prob=0.5),
        ]
        if is_l5_split
        else [
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        ]
    )
    # Use the requested crop count (8) for all splits.
    rand_crop_samples = 8

    train_transforms = Compose(
        [
            LoadImaged(keys=spatial_keys),
            ChannelFirstd(keys=spatial_keys),
            Spacingd(keys=spatial_keys, pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            Orientationd(keys=spatial_keys, axcodes="RAS"),
            # MRI intensity normalization: z-score on non-zero voxels, channel-wise.
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            SpatialPadd(keys=spatial_keys, spatial_size=roi_size, mode=("reflect", "constant")),
            RandCropByPosNegLabeld(
                keys=spatial_keys,
                label_key="label",
                spatial_size=roi_size,
                pos=3,
                neg=1,
                num_samples=rand_crop_samples,
            ),
            ResizeWithPadOrCropd(keys=spatial_keys, spatial_size=roi_size),
            RandFlipd(keys=spatial_keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=spatial_keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=spatial_keys, prob=0.5, spatial_axis=2),
            RandRotate90d(keys=spatial_keys, prob=0.5, max_k=3),
            *intensity_transforms,
            EnsureTyped(keys=spatial_keys),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            ChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # MRI intensity normalization: z-score on non-zero voxels, channel-wise.
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            SpatialPadd(keys=["image", "label"], spatial_size=roi_size, mode=("reflect", "constant")),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=roi_size),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    train_ds = CacheDataset(data=train_dicts, transform=train_transforms, cache_rate=1.0, num_workers=num_workers)
    val_ds = CacheDataset(data=val_dicts, transform=val_transforms, cache_rate=1.0, num_workers=num_workers)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    # Debug info
    print(f"Loaded {len(train_ds)} training cases and {len(val_ds)} validation cases.")
    first_batch = next(iter(train_loader))
    img, lab = first_batch["image"], first_batch["label"]
    print(f"First train batch shapes -> image: {tuple(img.shape)}, label: {tuple(lab.shape)}; roi_size={roi_size}")
    print(f"Debug shapes after transforms: img={tuple(img.shape)}, lab={tuple(lab.shape)}")
    assert img.shape[2:] == lab.shape[2:], "Image and label spatial shapes must match"

    return train_loader, val_loader


def get_transforms(spacing: Tuple[float, float, float]):
    """Create train/val/infer transforms."""
    common = [
        LoadImaged(keys=["image", "label"]),
        ChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
        DivisiblePadd(keys=["image", "label"], k=32, mode=("reflect", "constant")),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        EnsureTyped(keys=["image", "label"]),
    ]
    train_transforms = Compose(common)
    val_transforms = Compose(common)
    infer_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            ChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=spacing, mode="bilinear"),
            DivisiblePadd(keys=["image"], k=32, mode="reflect"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            EnsureTyped(keys=["image"]),
        ]
    )
    return train_transforms, val_transforms, infer_transforms


def get_loaders(
    data_dir: Path,
    batch_size: int,
    val_ratio: float,
    num_workers: int,
    spacing: Tuple[float, float, float],
) -> Tuple[DataLoader, DataLoader, Compose]:
    datalist = build_datalist(data_dir)
    split_idx = max(1, int(len(datalist) * (1.0 - val_ratio)))
    train_files = datalist[:split_idx]
    val_files = datalist[split_idx:] or datalist[-1:]

    train_transforms, val_transforms, infer_transforms = get_transforms(spacing)
    train_ds = Dataset(train_files, transform=train_transforms)
    val_ds = Dataset(val_files, transform=val_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, infer_transforms


def create_model(device: torch.device, roi_size: Iterable[int]) -> SwinUNETR:
    """
    Construct Swin-UNETR with BTCV settings.
    Note: MONAI 1.5 SwinUNETR infers patch sizes from input spatial dims; ROI/volume
    size must remain consistent with preprocessing (e.g., 96^3 or padded to a multiple of 32).
    """
    roi_size = tuple(int(v) for v in roi_size)
    model = SwinUNETR(
        in_channels=1,
        # 9 classes total: label 0 = background, labels 1-8 = structures
        out_channels=NUM_CLASSES,
        feature_size=DEFAULT_FEATURE_SIZE,
        use_checkpoint=True,
        spatial_dims=3,
        img_size=roi_size,
    )
    return model.to(device)


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def load_pretrained_weights(
    model: torch.nn.Module,
    checkpoint_path: Path,
    url: str = DEFAULT_PRETRAINED_URL,
) -> None:
    """
    Load weights from BTCV bundle.
    If checkpoint_path does not exist, attempts to download into that path.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint_path.exists():
        print(f"Downloading pretrained weights from NGC to {checkpoint_path} ...")
        try:
            download_url(url=url, filepath=str(checkpoint_path), progress=True)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to download pretrained weights. "
                "Update DEFAULT_PRETRAINED_URL or pass --checkpoint_path to a local file."
            ) from exc

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    cleaned_state = _strip_module_prefix(state_dict)
    load_result = model.load_state_dict(cleaned_state, strict=False)
    print(f"Loaded checkpoint keys. Missing: {load_result.missing_keys}, Unexpected: {load_result.unexpected_keys}")
    print("Pre-trained BTCV weights loaded successfully.")


def log_and_validate_batch_shape(
    train_loader: DataLoader,
    roi_size: Iterable[int],
) -> None:
    """
    Peek at the first batch to assert expected 3D shape [B, 1, D, H, W] and
    that spatial dims match the ROI/preprocessed volume size.
    """
    first = next(iter(train_loader))
    img = first["image"]
    spatial = tuple(img.shape[2:])
    expected = tuple(roi_size)
    print(f"First batch shape: {tuple(img.shape)}")
    if spatial != expected:
        raise ValueError(
            f"Expected spatial dims {expected} (from preprocessing/ROI), got {spatial}. "
            "Ensure img_size/ROI matches your dataset."
        )


def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    roi_size: Iterable[int],
) -> float:
    model.eval()
    dice_metric = DiceMetric(include_background=True, reduction="mean", num_classes=NUM_CLASSES)
    post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)
    post_label = AsDiscrete(to_onehot=NUM_CLASSES)

    with torch.no_grad():
        for batch_data in val_loader:
            images = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            logits = sliding_window_inference(images, roi_size=roi_size, sw_batch_size=1, predictor=model)

            # Compute Dice per batch and immediately drop tensors so validation does not cache volumes.
            preds = [post_pred(i) for i in decollate_batch(logits)]
            labels_list = [post_label(i) for i in decollate_batch(labels)]
            dice_metric(y_pred=preds, y=labels_list)

            del preds, labels_list, logits, images, labels

    dice = float(dice_metric.aggregate().item())
    dice_metric.reset()
    return dice


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    roi_size: Iterable[int],
) -> None:
    """Basic training loop skeleton."""
    loss_function = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0.0
        for step, batch_data in enumerate(train_loader, start=1):
            images = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            # Record scalar first, then release tensors to prevent lingering activations between steps.
            loss_value = loss.item()
            epoch_loss += loss_value
            if step % 5 == 0 or step == 1:
                print(f"  step {step:03d} - loss: {loss_value:.4f}")

            del loss, logits, images, labels

        scheduler.step()
        avg_loss = epoch_loss / max(1, len(train_loader))
        print(f"Average train loss: {avg_loss:.4f}")

        # TODO: increase validation frequency for real training
        val_dice = validate(model, val_loader, device=device, roi_size=roi_size)
        print(f"Validation mean Dice: {val_dice:.4f}")


def run_inference_on_volume(
    model: torch.nn.Module,
    device: torch.device,
    infer_transform: Compose,
    image_path: Path,
    output_path: Path,
    roi_size: Iterable[int],
) -> None:
    """Run sliding-window inference on a single volume and save NIfTI."""
    model.eval()
    sample = {"image": str(image_path)}
    sample = infer_transform(sample)
    image = sample["image"].unsqueeze(0).to(device)  # add batch dim

    with torch.no_grad():
        logits = sliding_window_inference(image, roi_size=roi_size, sw_batch_size=1, predictor=model)
        # Argmax over channel dim -> labels 0 (background) .. 8 (foreground classes)
        pred = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)[0]

    meta = sample.get("image_meta_dict", {})
    affine = meta.get("affine")
    if affine is None:
        affine = np.eye(4)
    else:
        affine = np.asarray(affine)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(pred, affine=affine), str(output_path))
    print(f"Saved inference mask to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Swin-UNETR BTCV setup script")
    parser.add_argument("--data_dir", type=Path, required=True, help="Root directory containing imagesTr/ and labelsTr/")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to store outputs/checkpoints")
    parser.add_argument("--checkpoint_path", type=Path, default=Path("pretrained") / "swin_unetr_btcv_0.5.6.pt")
    parser.add_argument("--pretrained_url", type=str, default=DEFAULT_PRETRAINED_URL, help="NGC URL for the BTCV bundle")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1, help="TODO: increase for real training")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--roi_size", type=int, nargs=3, default=DEFAULT_ROI_SIZE, metavar=("X", "Y", "Z"))
    parser.add_argument("--run_inference", type=str, default=None, help="Optional path to a single volume for inference")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, infer_transform = get_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        spacing=DEFAULT_SPACING,
    )

    roi_size = tuple(args.roi_size)
    # Confirm preprocessing/ROI matches the model's expected spatial dims (e.g., 96^3).
    log_and_validate_batch_shape(train_loader, roi_size)

    model = create_model(device=device, roi_size=roi_size)
    load_pretrained_weights(
        model=model,
        checkpoint_path=args.checkpoint_path,
        url=args.pretrained_url,
    )

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        roi_size=roi_size,
    )

    if args.run_inference:
        image_path = Path(args.run_inference)
        output_path = args.output_dir / f"{image_path.stem}_swin_unetr_seg.nii.gz"
        run_inference_on_volume(
            model=model,
            device=device,
            infer_transform=infer_transform,
            image_path=image_path,
            output_path=output_path,
            roi_size=roi_size,
        )


if __name__ == "__main__":
    main()
