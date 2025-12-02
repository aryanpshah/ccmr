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
    """
    dice = dice_metric.aggregate()
    dice_np = dice.cpu().numpy()
    mean_all = float(np.nanmean(dice_np))
    per_class_mean = np.nanmean(dice_np.reshape(-1, num_classes), axis=0).tolist()
    mean_fg = float(np.nanmean(per_class_mean[1:])) if num_classes > 1 else mean_all
    dice_metric.reset()
    return mean_all, per_class_mean, mean_fg
