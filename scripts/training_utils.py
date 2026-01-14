#!/usr/bin/env python
"""
Shared training utilities for Swin-UNETR experiments.
"""
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
from monai.metrics import CumulativeIterationMetric


def save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    """Save a model state_dict, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint: Path,
    filter_mismatch: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Load a checkpoint into the model. If filter_mismatch is True, drop keys whose
    shapes do not match the current model (useful for different output heads).
    Returns (missing_keys, unexpected_keys) from load_state_dict.
    """
    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location="cpu")
    state_dict = state["state_dict"] if "state_dict" in state else state
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    if filter_mismatch:
        model_state = model.state_dict()
        filtered_state = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
        dropped = sorted(set(state_dict.keys()) - set(filtered_state.keys()))
        if dropped:
            print(f"[load_checkpoint] Dropping mismatched keys: {dropped}")
        state_dict = filtered_state

    result = model.load_state_dict(state_dict, strict=False)
    return result.missing_keys, result.unexpected_keys


def compute_metrics(
    dice_metric: CumulativeIterationMetric,
    num_classes: int,
) -> Tuple[float, List[float], float]:
    """
    Aggregate MONAI DiceMetric and compute mean Dice values while ignoring NaNs.
    Returns (mean_all, per_class_mean_list, mean_foreground).
    
    CRITICAL: Handle cases where some classes may not be present in validation set.
    """
    dice = dice_metric.aggregate()
    dice_np = dice.cpu().numpy()
    
    # dice_np shape is typically (num_samples, num_classes) or (num_classes,)
    if dice_np.ndim == 1:
        # Single sample case
        per_class_mean = dice_np.tolist()
    else:
        # Multiple samples - compute per-class mean ignoring NaNs
        # Replace zeros with NaN for cases where class wasn't present
        per_class_mean = []
        for c in range(num_classes):
            class_dice = dice_np[:, c] if dice_np.ndim > 1 else [dice_np[c]]
            # Filter out NaN values
            valid_dice = [d for d in class_dice if not np.isnan(d)]
            if valid_dice:
                per_class_mean.append(float(np.mean(valid_dice)))
            else:
                per_class_mean.append(float('nan'))
    
    # Calculate overall mean (excluding NaN)
    valid_all = [d for d in np.array(per_class_mean).flatten() if not np.isnan(d)]
    mean_all = float(np.mean(valid_all)) if valid_all else 0.0
    
    # Calculate foreground mean (classes 1-8, excluding background class 0)
    fg_dice = per_class_mean[1:] if num_classes > 1 else per_class_mean
    valid_fg = [d for d in fg_dice if not np.isnan(d)]
    mean_fg = float(np.mean(valid_fg)) if valid_fg else 0.0
    
    dice_metric.reset()
    return mean_all, per_class_mean, mean_fg
