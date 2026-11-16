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

from medical_segmentation.config import validate_orientation_code

logger = logging.getLogger(__name__)


__all__ = [
    "DicomVolumeLoader",
    "VolumetricMetadata",
    "safe_parse_float",
    "safe_parse_str",
]


def safe_parse_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """
    Safely parse a DICOM header value into a float.
    Handles None, empty strings, lists, tuples, pydicom MultiValue,
    backslash-delimited strings, and conversion exceptions.
    """
    if value is None:
        return default
    if isinstance(value, (list, tuple, Sequence)) and not isinstance(value, (str, bytes)):
        if len(value) == 0:
            return default
        value = value[0]
        if value is None:
            return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (str, bytes)):
        clean = (value.decode("utf-8", errors="ignore") if isinstance(value, bytes) else value).strip()
        if not clean or clean.lower() in ("none", "null"):
            return default
        if "\\" in clean:
            clean = clean.split("\\")[0].strip()
        try:
            return float(clean)
        except (ValueError, TypeError):
            return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_parse_str(value: Any, default: str = "") -> str:
    """
    Safely parse a DICOM header value into a clean string.
    Normalizes None, 'None', empty strings, and whitespace to default.
    """
    if value is None:
        return default
    s = str(value).strip()
    if not s or s.lower() in ("none", "null"):
        return default
    return s


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
        if len(self.spacing) >= 3:
            return float(self.spacing[0] * self.spacing[1] * self.spacing[2])
        return 1.0

    @property
    def pixel_area_slice_mm2(self) -> float:
        """Physical cross-sectional area of a single slice pixel (X-Y in voxel space) in mm²."""
        if len(self.spacing) >= 2:
            return float(self.spacing[0] * self.spacing[1])
        return 1.0

    @property
    def pixel_area_axial_mm2(self) -> float:
        """Backward-compatible alias for pixel_area_slice_mm2."""
        return self.pixel_area_slice_mm2

    def get_slice_plane_and_position(self, slice_index: float | int) -> Tuple[str, float]:
        """
        Determine the anatomical slice plane ('axial', 'coronal', or 'sagittal')
        and physical slice position in millimeters along the slice normal axis.

        Args:
            slice_index: Continuous or discrete slice index (z) along the volume's third dimension.

        Returns:
            Tuple of (plane_name, slice_position_mm).
        """
        pt = self.continuous_index_to_physical_point((0.0, 0.0, float(slice_index)))

        orient = self.spatial_orientation.upper() if self.spatial_orientation else ""
        if len(orient) == 3:
            last_code = orient[2]
            if last_code in ("S", "I"):
                return "axial", pt[2]
            elif last_code in ("A", "P"):
                return "coronal", pt[1]
            elif last_code in ("R", "L"):
                return "sagittal", pt[0]

        if len(self.direction) == 9:
            dominant_axis = int(np.argmax(np.abs([self.direction[2], self.direction[5], self.direction[8]])))
            if dominant_axis == 2:
                return "axial", pt[2]
            elif dominant_axis == 1:
                return "coronal", pt[1]
            else:
                return "sagittal", pt[0]

        return "axial", pt[2]

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
        if len(continuous_index) != 3:
            raise ValueError(
                f"continuous_index must contain exactly 3 coordinates (cx, cy, cz), got {len(continuous_index)}."
            )
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
        self.target_orientation = validate_orientation_code(target_orientation)

    def load_dicom_series(
        self,
        directory_path: Path | str,
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
        dir_path = Path(directory_path).resolve()
        if not dir_path.is_dir():
            raise FileNotFoundError(f"DICOM series directory not found: {dir_path}")

        logger.info("Scanning directory '%s' for DICOM series...", dir_path)
        series_reader = sitk.ImageSeriesReader()
        series_ids = series_reader.GetGDCMSeriesIDs(str(dir_path))

        if not series_ids:
            # Fallback: check if directory contains individual DICOM files directly
            # Deduplicate matching files for case-insensitive filesystems
            dcm_files = sorted(set(list(dir_path.glob("*.dcm")) + list(dir_path.glob("*.DCM"))))
            if not dcm_files:
                raise ValueError(f"No valid DICOM series or .dcm files found in: {dir_path}")
            logger.info("Directly reading %d DICOM files...", len(dcm_files))
            file_names = [str(f) for f in dcm_files]
        else:
            selected_series_id = series_id or series_ids[0]
            logger.info("Found %d series. Loading series ID: '%s'", len(series_ids), selected_series_id)
            file_names = series_reader.GetGDCMSeriesFileNames(str(dir_path), selected_series_id)

        if not file_names:
            raise ValueError(f"Could not retrieve DICOM slice filenames from: {dir_path}")

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

    def load_volume_file(self, file_path: Path | str) -> Tuple[sitk.Image, VolumetricMetadata]:
        """
        Load a single 3D volumetric file (e.g. .nii, .nii.gz, .mha, or multi-frame .dcm).

        Args:
            file_path: Path to the 3D volume file.

        Returns:
            Tuple of (Oriented sitk.Image, VolumetricMetadata).
        """
        target_file = Path(file_path).resolve()
        if not target_file.is_file():
            raise FileNotFoundError(f"Volume file not found: {target_file}")

        logger.info("Reading 3D volumetric file: '%s'", target_file)
        raw_volume = sitk.ReadImage(str(target_file))

        # Determine clean base identifier without archive extensions (.nii.gz -> base)
        file_name_lower = target_file.name.lower()
        if file_name_lower.endswith(".nii.gz"):
            study_uid = target_file.name[:-7]
        else:
            study_uid = target_file.stem

        if target_file.suffix.lower() in (".dcm", ".ima"):
            metadata = self._extract_pydicom_metadata(target_file)
            if not metadata.study_instance_uid:
                metadata.study_instance_uid = study_uid
            if not metadata.series_instance_uid:
                metadata.series_instance_uid = target_file.name
        else:
            metadata = VolumetricMetadata(
                study_instance_uid=study_uid,
                series_instance_uid=target_file.name,
                modality="CT" if "ct" in target_file.name.lower() else "MRI",
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
            patient_id = safe_parse_str(getattr(ds, "PatientID", None), "ANONYMOUS")
            study_uid = safe_parse_str(getattr(ds, "StudyInstanceUID", None), "")
            series_uid = safe_parse_str(getattr(ds, "SeriesInstanceUID", None), "")
            modality = safe_parse_str(getattr(ds, "Modality", None), "CT").upper()
            body_part = safe_parse_str(getattr(ds, "BodyPartExamined", None), "HEAD_NECK")

            val_intercept = safe_parse_float(getattr(ds, "RescaleIntercept", None), 0.0)
            rescale_intercept = 0.0 if val_intercept is None else val_intercept

            val_slope = safe_parse_float(getattr(ds, "RescaleSlope", None), 1.0)
            rescale_slope = 1.0 if val_slope is None else val_slope

            window_center = safe_parse_float(getattr(ds, "WindowCenter", None), None)
            window_width = safe_parse_float(getattr(ds, "WindowWidth", None), None)

            extra_tags: Dict[str, Any] = {}
            for tag_name in (
                "SliceThickness",
                "SpacingBetweenSlices",
                "PixelSpacing",
                "PatientSex",
                "PatientAge",
                "StudyDate",
                "SeriesDate",
                "StudyDescription",
                "SeriesDescription",
                "Manufacturer",
            ):
                if hasattr(ds, tag_name):
                    raw_val = getattr(ds, tag_name)
                    if raw_val is not None:
                        val_str = str(raw_val).strip()
                        if val_str and val_str.lower() not in ("none", "null"):
                            extra_tags[tag_name] = val_str

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
                extra_tags=extra_tags,
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
        output_path: Path | str,
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
        if not isinstance(mask_array, np.ndarray):
            raise TypeError(f"mask_array must be a numpy ndarray, got {type(mask_array).__name__}.")
        if mask_array.ndim != 3:
            raise ValueError(f"mask_array must be 3-dimensional (Z, Y, X), got {mask_array.ndim} dimensions.")
        if not isinstance(reference_image, sitk.Image):
            raise TypeError(f"reference_image must be a SimpleITK Image, got {type(reference_image).__name__}.")

        ref_size = reference_image.GetSize()  # (X, Y, Z)
        expected_shape = (ref_size[2], ref_size[1], ref_size[0])  # (Z, Y, X)
        if mask_array.shape != expected_shape:
            raise ValueError(
                f"mask_array shape {mask_array.shape} does not match reference image shape {expected_shape}."
            )

        resolved_output = Path(output_path).resolve()
        resolved_output.parent.mkdir(parents=True, exist_ok=True)

        max_label = int(np.max(mask_array)) if mask_array.size > 0 else 0
        mask_dtype = np.uint16 if max_label > 255 else np.uint8
        mask_sitk = sitk.GetImageFromArray(mask_array.astype(mask_dtype))
        mask_sitk.SetSpacing(reference_image.GetSpacing())
        mask_sitk.SetOrigin(reference_image.GetOrigin())
        mask_sitk.SetDirection(reference_image.GetDirection())

        logger.info("Writing 3D segmentation mask to: %s", resolved_output)
        sitk.WriteImage(mask_sitk, str(resolved_output), True)  # True = compression enabled
        return resolved_output
