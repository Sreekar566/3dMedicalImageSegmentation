"""
MONAI Volumetric Transformation Pipeline.
Provides modality-specific intensity normalization, spatial resampling,
and tensor dimension formatting for 3D medical image segmentation.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple
import numpy as np
import torch
import SimpleITK as sitk

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    NormalizeIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    CastToTyped,
)

logger = logging.getLogger(__name__)


def build_volumetric_transforms(
    modality: str = "CT",
    target_spacing: Optional[Tuple[float, float, float]] = None,
    device: str = "cpu",
) -> Compose:
    """
    Construct a MONAI dictionary-based transformation pipeline.

    Args:
        modality: "CT", "CBCT", or "MRI" to configure appropriate intensity scaling.
        target_spacing: Desired isotropic or anisotropic voxel spacing (e.g., (1.0, 1.0, 1.0)).
        device: Target PyTorch execution device.

    Returns:
        MONAI Compose transform object.
    """
    transforms_list = [
        # Ensure 1st dimension is channel: (C, D, H, W)
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        CastToTyped(keys=["image"], dtype=np.float32),
    ]

    # Optional 3D spatial resampling to uniform physical resolution
    if target_spacing is not None:
        transforms_list.append(
            Spacingd(
                keys=["image"],
                pixdim=target_spacing,
                mode="bilinear",
                align_corners=True,
            )
        )

    # Modality-aware intensity normalization
    modality_upper = modality.upper()
    if modality_upper in ("CT", "CBCT"):
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
    elif modality_upper == "MRI":
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
            device=torch.device(device),
            dtype=torch.float32,
        )
    )

    return Compose(transforms_list)


def prepare_tensor_from_sitk(
    image: sitk.Image,
    modality: str = "CT",
    target_spacing: Optional[Tuple[float, float, float]] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Bridge SimpleITK image to a normalized PyTorch 3D batch tensor (1, C, D, H, W).

    Args:
        image: Standardized SimpleITK 3D image.
        modality: "CT", "CBCT", or "MRI".
        target_spacing: Optional physical voxel spacing (dx, dy, dz).
        device: PyTorch device ('cuda', 'mps', or 'cpu').

    Returns:
        torch.Tensor of shape (1, 1, Depth, Height, Width).
    """
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
