"""
MONAI Volumetric Transformation Pipeline.
Provides modality-specific intensity normalization, spatial resampling,
and tensor dimension formatting for 3D medical image segmentation.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Sequence, Tuple
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from monai.data import MetaTensor
from monai.transforms import (
    CastToTyped,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    NormalizeIntensityd,
    ScaleIntensityRanged,
    Spacingd,
)

from medical_segmentation.config import parse_target_spacing

logger = logging.getLogger(__name__)

__all__ = [
    "build_volumetric_transforms",
    "prepare_tensor_from_sitk",
    "resample_mask_to_shape",
]


def build_volumetric_transforms(
    modality: str = "CT",
    target_spacing: Optional[Tuple[float, float, float] | Sequence[float] | str] = None,
    device: str | torch.device = "cpu",
    ensure_channel_first: bool = True,
) -> Compose:
    """
    Construct a MONAI dictionary-based transformation pipeline with strict input validation.

    Args:
        modality: "CT", "CBCT", or "MRI" to configure appropriate intensity scaling.
        target_spacing: Desired isotropic or anisotropic voxel spacing (dx, dy, dz) in mm.
        device: Target PyTorch execution device ('cuda', 'mps', 'cpu' or torch.device).
        ensure_channel_first: Whether to prepend a channel axis to channel-less 3D arrays.

    Returns:
        MONAI Compose transform object.
    """
    if not isinstance(modality, str):
        raise TypeError(f"modality must be a string, got {type(modality).__name__}.")
    modality_clean = modality.upper().strip()
    if not modality_clean:
        raise ValueError("modality cannot be empty.")

    parsed_spacing = parse_target_spacing(target_spacing) if target_spacing is not None else None

    if not isinstance(device, (str, torch.device)):
        raise TypeError(f"device must be a string or torch.device, got {type(device).__name__}.")
    try:
        target_device = torch.device(device)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid PyTorch execution device '{device}': {exc}") from exc

    transforms_list = []
    if ensure_channel_first:
        # Ensure 1st dimension is channel: (C, D, H, W)
        transforms_list.append(EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"))
    transforms_list.append(CastToTyped(keys=["image"], dtype=np.float32))

    # Optional 3D spatial resampling to uniform physical resolution
    if parsed_spacing is not None:
        # Volumetric arrays use (Z, Y, X) axis ordering while target_spacing is (dx, dy, dz)
        spacing_zyx = (parsed_spacing[2], parsed_spacing[1], parsed_spacing[0])
        transforms_list.append(
            Spacingd(
                keys=["image"],
                pixdim=spacing_zyx,
                mode="bilinear",
                align_corners=True,
            )
        )

    # Modality-aware intensity normalization
    if modality_clean in ("CT", "CBCT"):
        # Soft tissue / standard Hounsfield Unit window: [-1000, 1000] HU -> [0.0, 1.0]
        logger.info("Applying CT Hounsfield Unit windowing: [-1000, 1000] -> [0.0, 1.0].")
        transforms_list.append(
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-1000.0,
                a_max=1000.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            )
        )
    elif modality_clean == "MRI":
        # Nonzero voxel z-score standardization for MRI volumes
        logger.info("Applying MRI nonzero z-score intensity normalization.")
        transforms_list.append(
            NormalizeIntensityd(
                keys=["image"],
                nonzero=True,
                channel_wise=True,
            )
        )
    else:
        # Fallback min-max normalization
        transforms_list.append(
            NormalizeIntensityd(
                keys=["image"],
                nonzero=False,
            )
        )

    # Ensure output is a PyTorch tensor on the requested device
    transforms_list.append(
        EnsureTyped(
            keys=["image"],
            device=target_device,
            dtype=torch.float32,
        )
    )

    return Compose(transforms_list)


def prepare_tensor_from_sitk(
    image: sitk.Image,
    modality: str = "CT",
    target_spacing: Optional[Tuple[float, float, float] | Sequence[float] | str] = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Bridge SimpleITK image to a normalized PyTorch 3D batch tensor (1, C, D, H, W).

    Args:
        image: Standardized SimpleITK 3D image.
        modality: "CT", "CBCT", or "MRI" (or other modality for generic normalization).
        target_spacing: Optional physical voxel spacing (dx, dy, dz).
        device: PyTorch device ('cuda', 'mps', or 'cpu').

    Returns:
        torch.Tensor of shape (1, 1, Depth, Height, Width).
    """
    if not isinstance(image, sitk.Image):
        raise TypeError(f"image must be a SimpleITK Image, got {type(image).__name__}.")
    if image.GetDimension() != 3:
        raise ValueError(f"Expected 3D SimpleITK image (depth, height, width), got {image.GetDimension()}D image.")
    img_size = image.GetSize()
    if any(dim <= 0 for dim in img_size) or image.GetNumberOfPixels() == 0:
        raise ValueError(f"SimpleITK image dimensions must be strictly positive, got {img_size}.")

    array = sitk.GetArrayFromImage(image).astype(np.float32)

    # SimpleITK returns spacing in (X, Y, Z); NumPy array axes are (Z, Y, X)
    raw_spacing = image.GetSpacing()
    if (
        isinstance(raw_spacing, (tuple, list))
        and len(raw_spacing) == 3
        and all(isinstance(s, (int, float)) and math.isfinite(s) and s > 0.0 for s in raw_spacing)
    ):
        spacing_zyx = (float(raw_spacing[2]), float(raw_spacing[1]), float(raw_spacing[0]))
    else:
        spacing_zyx = (1.0, 1.0, 1.0)

    affine = np.diag([spacing_zyx[0], spacing_zyx[1], spacing_zyx[2], 1.0])
    image_meta = MetaTensor(array[None, ...], affine=affine)

    pipeline = build_volumetric_transforms(
        modality=modality,
        target_spacing=target_spacing,
        device=device,
        ensure_channel_first=False,
    )

    data_dict = {"image": image_meta}
    transformed = pipeline(data_dict)

    output_tensor = transformed["image"]
    if hasattr(output_tensor, "as_tensor"):
        output_tensor = output_tensor.as_tensor()

    # Shape: (C, D, H, W) -> add batch dimension: (1, C, D, H, W)
    tensor = output_tensor.unsqueeze(0)
    logger.info("Prepared 3D volumetric tensor for inference with shape: %s", tuple(tensor.shape))
    return tensor


def resample_mask_to_shape(
    mask_array: np.ndarray,
    target_shape: Tuple[int, int, int],
) -> np.ndarray:
    """
    Resample a discrete 3D segmentation mask array to target (Z, Y, X) dimensions
    using exact nearest-neighbor interpolation so integer class labels are preserved.

    Args:
        mask_array: 3D integer mask array of shape (Z', Y', X').
        target_shape: Desired 3D output shape (Z, Y, X).

    Returns:
        3D mask array of shape target_shape with the original dtype preserved.
    """
    if not isinstance(mask_array, np.ndarray):
        raise TypeError(f"mask_array must be a numpy ndarray, got {type(mask_array).__name__}.")
    if mask_array.ndim != 3:
        raise ValueError(
            f"mask_array must be 3-dimensional (Z, Y, X), got {mask_array.ndim} dimensions."
        )
    if (
        not isinstance(target_shape, (tuple, list))
        or len(target_shape) != 3
        or any(isinstance(d, bool) or not isinstance(d, int) or d <= 0 for d in target_shape)
    ):
        raise ValueError(
            f"target_shape must be 3 strictly positive integers (Z, Y, X), got {target_shape}."
        )

    target_tuple = (int(target_shape[0]), int(target_shape[1]), int(target_shape[2]))
    if mask_array.shape == target_tuple:
        return mask_array

    float_dtype = np.float64 if mask_array.dtype.itemsize > 2 else np.float32
    mask_tensor = torch.from_numpy(mask_array.astype(float_dtype)).unsqueeze(0).unsqueeze(0)
    resampled = F.interpolate(
        mask_tensor,
        size=target_tuple,
        mode="nearest-exact",
    )
    return resampled.squeeze(0).squeeze(0).cpu().numpy().astype(mask_array.dtype)
