"""Metric computation for whole-heart segmentation (Dice, HD95)."""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

K = 8


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice coefficient for binary masks. Returns 1.0 if both empty."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


def surface(mask: np.ndarray) -> np.ndarray:
    """1-voxel thick surface."""
    if mask.sum() == 0:
        return mask
    er = binary_erosion(mask, iterations=1, border_value=0)
    return np.logical_and(mask, np.logical_not(er))


def hd95_mm(gt: np.ndarray, pr: np.ndarray, spacing_xyz_mm: tuple) -> float:
    """95th percentile symmetric Hausdorff distance in mm. NaN if one empty."""
    gt = gt.astype(bool)
    pr = pr.astype(bool)
    if gt.sum() == 0 and pr.sum() == 0:
        return 0.0
    if gt.sum() == 0 or pr.sum() == 0:
        return float("nan")
    s_gt = surface(gt)
    s_pr = surface(pr)
    sp_zyx = (spacing_xyz_mm[2], spacing_xyz_mm[1], spacing_xyz_mm[0])
    dt_gt = distance_transform_edt(~s_gt, sampling=sp_zyx)
    dt_pr = distance_transform_edt(~s_pr, sampling=sp_zyx)
    d1 = dt_gt[s_pr]
    d2 = dt_pr[s_gt]
    if d1.size == 0 or d2.size == 0:
        return float("nan")
    return float(np.percentile(np.concatenate([d1, d2]), 95))
