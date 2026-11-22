"""
Volumetric Segmentation Mask Serialization and NIfTI Export Engine.
Preserves exact physical coordinate geometry (origin, spacing, direction matrix)
from reference SimpleITK images or VolumetricMetadata dataclasses.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import SimpleITK as sitk

if TYPE_CHECKING:
    from medical_segmentation.io.dicom_loader import VolumetricMetadata

logger = logging.getLogger(__name__)

__all__ = [
    "VolumetricMaskWriter",
    "save_mask_as_nifti",
]


def save_mask_as_nifti(
    mask_array: np.ndarray,
    reference_image: Optional[sitk.Image | VolumetricMetadata] = None,
    output_path: Optional[Path | str] = None,
    *,
    reference: Optional[sitk.Image | VolumetricMetadata] = None,
) -> Path:
    """
    Save a 3D segmentation mask (NumPy array) to a compressed NIfTI (.nii.gz) file,
    preserving the exact physical coordinate system (origin, spacing, direction matrix)
    of the reference SimpleITK image or VolumetricMetadata.

    Args:
        mask_array: Binary or multi-class integer NumPy array of shape (Z, Y, X).
        reference_image: Reference sitk.Image or VolumetricMetadata containing spatial geometry.
        output_path: Destination path for the .nii.gz file.
        reference: Keyword-only alias for reference_image.

    Returns:
        The resolved Path to the written NIfTI file.
    """
    ref = reference if reference is not None else reference_image
    if ref is None:
        raise TypeError("reference_image must be a SimpleITK Image or VolumetricMetadata, got None.")

    if not isinstance(mask_array, np.ndarray):
        raise TypeError(f"mask_array must be a numpy ndarray, got {type(mask_array).__name__}.")
    if mask_array.ndim != 3:
        raise ValueError(f"mask_array must be 3-dimensional (Z, Y, X), got {mask_array.ndim} dimensions.")

    # Extract spatial parameters from sitk.Image or VolumetricMetadata
    if isinstance(ref, sitk.Image):
        ref_size = ref.GetSize()  # (X, Y, Z)
        expected_shape = (ref_size[2], ref_size[1], ref_size[0])  # (Z, Y, X)
        spacing = ref.GetSpacing()
        origin = ref.GetOrigin()
        direction = ref.GetDirection()
    elif hasattr(ref, "dimensions") and hasattr(ref, "spacing") and hasattr(ref, "origin"):
        # VolumetricMetadata instance or duck-typed geometry provider
        ref_dims = ref.dimensions  # (X, Y, Z)
        expected_shape = (ref_dims[2], ref_dims[1], ref_dims[0])  # (Z, Y, X)
        spacing = ref.spacing
        origin = ref.origin
        direction = getattr(ref, "direction", ())
    else:
        raise TypeError(
            f"reference_image must be a SimpleITK Image or VolumetricMetadata, got {type(ref).__name__}."
        )

    if mask_array.shape != expected_shape:
        raise ValueError(
            f"mask_array shape {mask_array.shape} does not match reference image shape {expected_shape}."
        )

    if output_path is None or not str(output_path).strip():
        raise ValueError("output_path must be specified.")

    resolved_output = Path(str(output_path).strip()).resolve()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    # Use uint16 for multi-label segmentations exceeding 255 to prevent uint8 overflow
    max_label = int(np.max(mask_array)) if mask_array.size > 0 else 0
    mask_dtype = np.uint16 if max_label > 255 else np.uint8

    mask_sitk = sitk.GetImageFromArray(mask_array.astype(mask_dtype))
    mask_sitk.SetSpacing(spacing)
    mask_sitk.SetOrigin(origin)
    if direction and len(direction) == 9:
        mask_sitk.SetDirection(direction)

    logger.info("Writing 3D segmentation mask to: %s", resolved_output)
    sitk.WriteImage(mask_sitk, str(resolved_output), True)  # True = compression enabled
    return resolved_output


class VolumetricMaskWriter:
    """
    Dedicated serialization engine for 3D volumetric segmentation masks.
    Encapsulates staging output directory management and coordinate system preservation.
    """

    def __init__(self, default_output_dir: Optional[Path | str] = None):
        self.default_output_dir = Path(default_output_dir).resolve() if default_output_dir else None
        if self.default_output_dir is not None:
            self.default_output_dir.mkdir(parents=True, exist_ok=True)

    def write_mask(
        self,
        mask_array: np.ndarray,
        reference: sitk.Image | VolumetricMetadata,
        output_path: Optional[Path | str] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Save a 3D segmentation mask preserving reference geometry.

        Args:
            mask_array: 3D integer label map array (Z, Y, X).
            reference: Reference sitk.Image or VolumetricMetadata.
            output_path: Explicit target destination file path.
            filename: Target file name inside default_output_dir.

        Returns:
            Resolved Path to saved NIfTI mask.
        """
        clean_output = str(output_path).strip() if output_path is not None else ""
        clean_filename = str(filename).strip() if filename is not None else ""

        if clean_output:
            target_path = Path(clean_output).resolve()
        elif self.default_output_dir is not None and clean_filename:
            target_path = self.default_output_dir / clean_filename
        elif clean_filename:
            target_path = Path(clean_filename).resolve()
        else:
            raise ValueError("Either output_path or filename (with default_output_dir) must be specified.")

        return save_mask_as_nifti(
            mask_array=mask_array,
            reference=reference,
            output_path=target_path,
        )
