"""
DICOM Series and 3D Volumetric Loader with SimpleITK Spatial Orientation Fix.
Extracts clinical metadata via pydicom and standardizes spatial coordinates via sitk.DICOMOrient.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pydicom
import SimpleITK as sitk

logger = logging.getLogger(__name__)


@dataclass
class VolumetricMetadata:
    """Standardized physical geometry and DICOM clinical metadata for a 3D medical volume."""

    patient_id: str = "UNKNOWN"
    study_instance_uid: str = ""
    series_instance_uid: str = ""
    modality: str = "CT"
    body_part: str = "UNKNOWN"
    dimensions: Tuple[int, int, int] = (0, 0, 0)  # (X, Y, Z) in voxels
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # (dx, dy, dz) in mm
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: Tuple[float, ...] = ()
    spatial_orientation: str = "RAS"
    rescale_intercept: float = 0.0
    rescale_slope: float = 1.0
    window_center: Optional[float] = None
    window_width: Optional[float] = None
    extra_tags: Dict[str, Any] = field(default_factory=dict)

    @property
    def voxel_volume_mm3(self) -> float:
        """Physical volume of a single voxel in mm³."""
        return float(self.spacing[0] * self.spacing[1] * self.spacing[2])

    @property
    def pixel_area_axial_mm2(self) -> float:
        """Physical cross-sectional area of a single axial pixel in mm²."""
        return float(self.spacing[0] * self.spacing[1])

    def continuous_index_to_physical_point(
        self,
        continuous_index: Tuple[float, float, float] | Sequence[float],
    ) -> Tuple[float, float, float]:
        """
        Transform continuous voxel index (X, Y, Z) to physical patient coordinate (x, y, z) in mm,
        accounting for origin, voxel spacing, and 3D direction cosine matrix.

        Args:
            continuous_index: Continuous index sequence (cx, cy, cz) in voxel coordinates.

        Returns:
            Tuple of physical coordinates (px, py, pz) in millimeters.
        """
        cx, cy, cz = continuous_index
        dx, dy, dz = self.spacing
        ox, oy, oz = self.origin

        if len(self.direction) == 9:
            d = self.direction
            px = ox + d[0] * (cx * dx) + d[1] * (cy * dy) + d[2] * (cz * dz)
            py = oy + d[3] * (cx * dx) + d[4] * (cy * dy) + d[5] * (cz * dz)
            pz = oz + d[6] * (cx * dx) + d[7] * (cy * dy) + d[8] * (cz * dz)
            return (float(px), float(py), float(pz))

        return (
            float(ox + cx * dx),
            float(oy + cy * dy),
            float(oz + cz * dz),
        )


class DicomVolumeLoader:
    """Robust volumetric medical image ingestion engine supporting DICOM series and NIfTI volumes."""

    def __init__(self, target_orientation: str = "RAS"):
        """
        Initialize the volume loader with a desired target spatial orientation.

        Args:
            target_orientation: Standard 3-letter anatomical coordinate code (e.g., 'RAS' or 'LPS').
                                'RAS': Right-to-Left, Anterior-to-Posterior, Superior-to-Inferior.
        """
        self.target_orientation = target_orientation.upper()

    def load_dicom_series(
        self,
        directory_path: Path,
        series_id: Optional[str] = None,
    ) -> Tuple[sitk.Image, VolumetricMetadata]:
        """
        Read a directory containing a 3D DICOM series, extract metadata via pydicom,
        and standardize its anatomical orientation via sitk.DICOMOrient.

        Args:
            directory_path: Local folder containing .dcm slice files.
            series_id: Optional specific DICOM SeriesInstanceUID to load if multiple exist.

        Returns:
            Tuple of (Standardized sitk.Image in target orientation, VolumetricMetadata).
        """
        if not directory_path.is_dir():
            raise FileNotFoundError(f"DICOM series directory not found: {directory_path}")

        logger.info("Scanning directory '%s' for DICOM series...", directory_path)
        series_reader = sitk.ImageSeriesReader()
        series_ids = series_reader.GetGDCMSeriesIDs(str(directory_path))

        if not series_ids:
            # Fallback: check if directory contains individual DICOM files directly
            dcm_files = sorted(list(directory_path.glob("*.dcm")) + list(directory_path.glob("*.DCM")))
            if not dcm_files:
                raise ValueError(f"No valid DICOM series or .dcm files found in: {directory_path}")
            logger.info("Directly reading %d DICOM files...", len(dcm_files))
            file_names = [str(f) for f in dcm_files]
        else:
            selected_series_id = series_id or series_ids[0]
            logger.info("Found %d series. Loading series ID: '%s'", len(series_ids), selected_series_id)
            file_names = series_reader.GetGDCMSeriesFileNames(str(directory_path), selected_series_id)

        if not file_names:
            raise ValueError(f"Could not retrieve DICOM slice filenames from: {directory_path}")

        # 1. Extract clinical tags from the first slice using pydicom
        metadata = self._extract_pydicom_metadata(Path(file_names[0]))

        # 2. Read full 3D volume using SimpleITK GDCM reader
        series_reader.SetFileNames(file_names)
        series_reader.MetaDataDictionaryArrayUpdateOn()
        series_reader.LoadPrivateTagsOn()
        raw_volume: sitk.Image = series_reader.Execute()

        # 3. Standardize volumetric geometry and orientation
        oriented_volume, metadata = self._orient_and_populate_geometry(raw_volume, metadata)

        logger.info(
            "DICOM volume loaded: shape=%s, spacing=%s, orientation=%s",
            oriented_volume.GetSize(),
            oriented_volume.GetSpacing(),
            metadata.spatial_orientation,
        )
        return oriented_volume, metadata

    def load_volume_file(self, file_path: Path) -> Tuple[sitk.Image, VolumetricMetadata]:
        """
        Load a single 3D volumetric file (e.g. .nii, .nii.gz, .mha, or multi-frame .dcm).

        Args:
            file_path: Path to the 3D volume file.

        Returns:
            Tuple of (Oriented sitk.Image, VolumetricMetadata).
        """
        if not file_path.is_file():
            raise FileNotFoundError(f"Volume file not found: {file_path}")

        logger.info("Reading 3D volumetric file: '%s'", file_path)
        raw_volume = sitk.ReadImage(str(file_path))

        metadata = VolumetricMetadata(
            study_instance_uid=file_path.stem,
            series_instance_uid=file_path.name,
            modality="CT" if "ct" in file_path.name.lower() else "MRI",
        )

        # Orient volume and populate physical coordinate geometry
        oriented_volume, metadata = self._orient_and_populate_geometry(raw_volume, metadata)
        return oriented_volume, metadata

    def _orient_and_populate_geometry(
        self,
        volume: sitk.Image,
        metadata: VolumetricMetadata,
    ) -> Tuple[sitk.Image, VolumetricMetadata]:
        """
        Standardize volume orientation using sitk.DICOMOrient and update physical metadata.
        """
        logger.info("Applying sitk.DICOMOrient(target='%s')...", self.target_orientation)
        oriented_image = sitk.DICOMOrient(volume, self.target_orientation)

        # Update metadata with post-orientation physical parameters
        metadata.dimensions = oriented_image.GetSize()  # (X, Y, Z)
        metadata.spacing = oriented_image.GetSpacing()  # (dx, dy, dz)
        metadata.origin = oriented_image.GetOrigin()    # (ox, oy, oz)
        metadata.direction = oriented_image.GetDirection()
        metadata.spatial_orientation = self.target_orientation

        return oriented_image, metadata

    def _extract_pydicom_metadata(self, sample_slice_path: Path) -> VolumetricMetadata:
        """Parse DICOM header attributes from a slice using pydicom."""
        try:
            ds = pydicom.dcmread(sample_slice_path, stop_before_pixels=True)
            patient_id = str(getattr(ds, "PatientID", "ANONYMOUS"))
            study_uid = str(getattr(ds, "StudyInstanceUID", ""))
            series_uid = str(getattr(ds, "SeriesInstanceUID", ""))
            modality = str(getattr(ds, "Modality", "CT")).upper()
            body_part = str(getattr(ds, "BodyPartExamined", "HEAD_NECK"))

            rescale_intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            rescale_slope = float(getattr(ds, "RescaleSlope", 1.0))

            window_center = None
            if hasattr(ds, "WindowCenter"):
                wc = ds.WindowCenter
                window_center = float(wc[0] if isinstance(wc, (list, pydicom.multival.MultiValue)) else wc)

            window_width = None
            if hasattr(ds, "WindowWidth"):
                ww = ds.WindowWidth
                window_width = float(ww[0] if isinstance(ww, (list, pydicom.multival.MultiValue)) else ww)

            return VolumetricMetadata(
                patient_id=patient_id,
                study_instance_uid=study_uid,
                series_instance_uid=series_uid,
                modality=modality,
                body_part=body_part,
                rescale_intercept=rescale_intercept,
                rescale_slope=rescale_slope,
                window_center=window_center,
                window_width=window_width,
            )
        except Exception as exc:
            logger.warning("Could not read full DICOM tags from '%s' via pydicom: %s", sample_slice_path, exc)
            return VolumetricMetadata()

    @staticmethod
    def sitk_to_numpy(image: sitk.Image) -> np.ndarray:
        """
        Convert SimpleITK image to a float32 NumPy array.
        Note: SimpleITK GetArrayFromImage returns order (Z, Y, X).
        """
        return sitk.GetArrayFromImage(image).astype(np.float32)

    @staticmethod
    def save_mask_as_nifti(
        mask_array: np.ndarray,
        reference_image: sitk.Image,
        output_path: Path,
    ) -> Path:
        """
        Save a 3D segmentation mask (NumPy array) back to a NIfTI (.nii.gz) file,
        preserving the exact physical coordinate system (origin, spacing, direction matrix)
        of the reference oriented SimpleITK image.

        Args:
            mask_array: Binary or multi-class integer NumPy array (Z, Y, X).
            reference_image: The oriented sitk.Image used during inference.
            output_path: Path to write the .nii.gz file.

        Returns:
            The resolved Path to the saved file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mask_sitk = sitk.GetImageFromArray(mask_array.astype(np.uint8))
        mask_sitk.SetSpacing(reference_image.GetSpacing())
        mask_sitk.SetOrigin(reference_image.GetOrigin())
        mask_sitk.SetDirection(reference_image.GetDirection())

        logger.info("Writing 3D segmentation mask to: %s", output_path)
        sitk.WriteImage(mask_sitk, str(output_path), True)  # True = compression enabled
        return output_path
