"""
MONAI Volumetric Transformation Pipeline.
Provides modality-specific intensity normalization, spatial resampling,
and tensor dimension formatting for 3D medical image segmentation.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple
import numpy as np
import SimpleITK as sitk
import torch

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


def build_volumetric_transforms(
    modality: str = "CT",
    target_spacing: Optional[Tuple[float, float, float] | Sequence[float] | str] = None,
    device: str | torch.device = "cpu",
) -> Compose:
    """
    Construct a MONAI dictionary-based transformation pipeline with strict input validation.

    Args:
        modality: "CT", "CBCT", or "MRI" to configure appropriate intensity scaling.
        target_spacing: Desired isotropic or anisotropic voxel spacing (dx, dy, dz) in mm.
        device: Target PyTorch execution device ('cuda', 'mps', 'cpu' or torch.device).

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

    transforms_list = [
        # Ensure 1st dimension is channel: (C, D, H, W)
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        CastToTyped(keys=["image"], dtype=np.float32),
    ]

    # Optional 3D spatial resampling to uniform physical resolution
    if parsed_spacing is not None:
        transforms_list.append(
            Spacingd(
                keys=["image"],
                pixdim=parsed_spacing,
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

    pipeline = build_volumetric_transforms(
        modality=modality,
        target_spacing=target_spacing,
        device=device,
    )

    data_dict = {"image": array}
    transformed = pipeline(data_dict)

    # Shape: (C, D, H, W) -> add batch dimension: (1, C, D, H, W)
    tensor = transformed["image"].unsqueeze(0)
    logger.info("Prepared 3D volumetric tensor for inference with shape: %s", tuple(tensor.shape))
    return tensor
