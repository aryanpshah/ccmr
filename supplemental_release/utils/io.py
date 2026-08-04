"""I/O helpers for NIfTI volumes."""
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False

try:
    import nibabel as nib
    HAS_NIB = True
except ImportError:
    HAS_NIB = False


def read_nii_sitk(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Read NIfTI; return (data zyx, spacing_xyz_mm)."""
    if not HAS_SITK:
        raise ImportError("SimpleITK required for read_nii_sitk")
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # z,y,x
    spacing = img.GetSpacing()  # x,y,z
    return arr.astype(np.int16), tuple(float(s) for s in spacing)


def read_nii_nib(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Read NIfTI via nibabel; return (data zyx, spacing_xyz_mm)."""
    if not HAS_NIB:
        raise ImportError("nibabel required for read_nii_nib")
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.int16)
    data = np.transpose(data, (2, 1, 0))
    zooms = img.header.get_zooms()[:3]
    spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    return data, spacing


def read_nii(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Read NIfTI with SimpleITK or nibabel."""
    if HAS_SITK:
        return read_nii_sitk(path)
    return read_nii_nib(path)
