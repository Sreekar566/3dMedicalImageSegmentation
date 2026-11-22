"""
Quantitative Volumetric and Morphological Metric Extraction Engine.
Extracts label-level physical statistics and generates memory-efficient
slice-by-slice voxel distribution profiles.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from medical_segmentation.io.dicom_loader import VolumetricMetadata

logger = logging.getLogger(__name__)

__all__ = [
    "LabelSummaryMetric",
    "SegmentationMetrics",
    "VolumetricMetricsCalculator",
    "build_slice_record",
    "compute_intensity_statistics",
    "compute_label_summary",
    "compute_label_summary_with_counts",
    "compute_slice_positions",
    "compute_spatial_bounds_and_centroid",
    "generate_slice_metrics",
    "resolve_slice_plane",
    "validate_volumetric_arrays",
]


@dataclass
class LabelSummaryMetric:
    """Summary volumetric statistics for a specific segmented label/structure."""

    label_id: int
    structure_name: str
    voxel_count: int
    volume_mm3: float
    volume_cm3: float
    mean_intensity: float
    std_intensity: float
    min_intensity: float
    max_intensity: float
    centroid_mm: Tuple[float, float, float]
    bounding_box: Tuple[int, int, int, int, int, int]  # (min_x, min_y, min_z, max_x, max_y, max_z)


@dataclass
class SegmentationMetrics:
    """Aggregated inference result and quantitative physical metrics."""

    duration_seconds: float
    labels: List[LabelSummaryMetric] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def validate_volumetric_arrays(raw_array: np.ndarray, mask_array: np.ndarray) -> None:
    """
    Validate array dimensions, types, and shape compatibility.

    Args:
        raw_array: 3D NumPy array of image voxel intensities.
        mask_array: 3D NumPy array of discrete segmentation labels.

    Raises:
        TypeError: If either array is not a numpy ndarray.
        ValueError: If arrays are not 3-dimensional, have mismatched shapes, or are empty.
    """
    if not isinstance(raw_array, np.ndarray) or not isinstance(mask_array, np.ndarray):
        raise TypeError("Both raw_array and mask_array must be numpy ndarrays.")
    if raw_array.ndim != 3 or mask_array.ndim != 3:
        raise ValueError(
            f"Expected 3D volumetric arrays (Z, Y, X), but got raw_array.ndim={raw_array.ndim} "
            f"and mask_array.ndim={mask_array.ndim}."
        )
    if raw_array.shape != mask_array.shape:
        raise ValueError(
            f"Array shape mismatch: raw_array shape {raw_array.shape} "
            f"does not match mask_array shape {mask_array.shape}."
        )
    if raw_array.size == 0:
        raise ValueError("Arrays cannot be empty (size 0).")


def compute_intensity_statistics(
    raw_array: np.ndarray,
    label_mask: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Compute (mean, std, min, max) intensity statistics for voxels matching the label mask.

    Args:
        raw_array: 3D intensity array (Z, Y, X).
        label_mask: 3D boolean mask for the current label.

    Returns:
        Tuple of (mean_intensity, std_intensity, min_intensity, max_intensity).

    Raises:
        ValueError: If label_mask contains no active voxels.
    """
    roi_intensities = raw_array[label_mask]
    if roi_intensities.size == 0:
        raise ValueError("Cannot compute intensity statistics for empty label mask.")
    return (
        float(np.mean(roi_intensities)),
        float(np.std(roi_intensities)),
        float(np.min(roi_intensities)),
        float(np.max(roi_intensities)),
    )


def compute_spatial_bounds_and_centroid(
    label_mask: np.ndarray,
    metadata: VolumetricMetadata,
    voxel_count: int,
) -> Tuple[Tuple[int, int, int, int, int, int], Tuple[float, float, float], np.ndarray]:
    """
    Compute bounding box, physical centroid, and 1D axial slice counts using
    memory-efficient 1D orthogonal projections.

    Avoids full 3D coordinate array allocations (e.g. np.where/np.nonzero)
    by projecting active voxels along each coordinate axis.

    Args:
        label_mask: 3D boolean mask for the current label (Z, Y, X).
        metadata: Geometric physical metadata for coordinate transforms.
        voxel_count: Precomputed non-zero voxel count.

    Returns:
        Tuple of:
        - bounding_box: (min_x, min_y, min_z, max_x, max_y, max_z) in voxel coordinates.
        - centroid_mm: (px, py, pz) physical coordinates in millimeters.
        - z_counts: 1D array of active voxel counts per axial slice (depth).

    Raises:
        TypeError: If metadata is None.
        ValueError: If voxel_count <= 0 or label_mask contains no active voxels.
    """
    if metadata is None:
        raise TypeError("metadata must be a VolumetricMetadata instance, got None.")
    if not isinstance(voxel_count, (int, np.integer)) or voxel_count <= 0:
        raise ValueError(f"voxel_count must be a strictly positive integer, got {voxel_count}.")

    z_counts = np.sum(label_mask, axis=(1, 2))
    y_counts = np.sum(label_mask, axis=(0, 2))
    x_counts = np.sum(label_mask, axis=(0, 1))

    z_nz = np.flatnonzero(z_counts)
    y_nz = np.flatnonzero(y_counts)
    x_nz = np.flatnonzero(x_counts)

    if z_nz.size == 0 or y_nz.size == 0 or x_nz.size == 0:
        raise ValueError("label_mask contains no active non-zero voxels.")

    bbox = (
        int(x_nz[0]),
        int(y_nz[0]),
        int(z_nz[0]),
        int(x_nz[-1]),
        int(y_nz[-1]),
        int(z_nz[-1]),
    )

    cz = float(np.sum(z_nz * z_counts[z_nz]) / voxel_count)
    cy = float(np.sum(y_nz * y_counts[y_nz]) / voxel_count)
    cx = float(np.sum(x_nz * x_counts[x_nz]) / voxel_count)

    centroid_mm = metadata.continuous_index_to_physical_point((cx, cy, cz))
    return bbox, centroid_mm, z_counts


def compute_label_summary_with_counts(
    raw_array: np.ndarray,
    label_mask: np.ndarray,
    label_id: int,
    metadata: VolumetricMetadata,
) -> Tuple[Optional[LabelSummaryMetric], Optional[np.ndarray]]:
    """
    Compute summary volumetric metrics and 1D axial slice counts for a single label mask.

    Returns:
        Tuple of (LabelSummaryMetric or None, z_counts array or None).
    """
    voxel_count = int(np.count_nonzero(label_mask))
    if voxel_count == 0:
        return None, None

    volume_mm3 = float(voxel_count * metadata.voxel_volume_mm3)
    volume_cm3 = float(volume_mm3 / 1000.0)

    mean_val, std_val, min_val, max_val = compute_intensity_statistics(raw_array, label_mask)
    bbox, centroid_mm, z_counts = compute_spatial_bounds_and_centroid(
        label_mask, metadata, voxel_count
    )

    structure_name = f"Segmented_Structure_{label_id}" if label_id > 1 else "Primary_Lesion"

    summary = LabelSummaryMetric(
        label_id=label_id,
        structure_name=structure_name,
        voxel_count=voxel_count,
        volume_mm3=volume_mm3,
        volume_cm3=volume_cm3,
        mean_intensity=mean_val,
        std_intensity=std_val,
        min_intensity=min_val,
        max_intensity=max_val,
        centroid_mm=centroid_mm,
        bounding_box=bbox,
    )
    return summary, z_counts


def compute_label_summary(
    raw_array: np.ndarray,
    label_mask: np.ndarray,
    label_id: int,
    metadata: VolumetricMetadata,
) -> Optional[LabelSummaryMetric]:
    """
    Calculate summary volumetric metrics for a specific label mask.

    Args:
        raw_array: Full 3D image array (Z, Y, X).
        label_mask: Boolean 3D mask for this label.
        label_id: Integer identifier for the structure.
        metadata: Geometric physical metadata.

    Returns:
        LabelSummaryMetric instance or None if voxel count is zero.
    """
    summary, _ = compute_label_summary_with_counts(
        raw_array=raw_array,
        label_mask=label_mask,
        label_id=label_id,
        metadata=metadata,
    )
    return summary


def resolve_slice_plane(metadata: VolumetricMetadata) -> str:
    """Resolve anatomical orientation plane ('axial', 'coronal', or 'sagittal')."""
    if hasattr(metadata, "get_slice_plane_and_position"):
        plane, _ = metadata.get_slice_plane_and_position(0)
        return plane
    return "axial"


def compute_slice_positions(metadata: VolumetricMetadata, depth: int) -> List[float]:
    """
    Precompute physical slice positions in mm along the normal axis for all slices.
    Eliminates repeated orientation lookup and coordinate transformations in inner loops.
    """
    if depth <= 0:
        return []
    if hasattr(metadata, "get_slice_plane_and_position"):
        return [float(metadata.get_slice_plane_and_position(z)[1]) for z in range(depth)]
    return [
        float(metadata.continuous_index_to_physical_point((0.0, 0.0, float(z)))[2])
        for z in range(depth)
    ]


def build_slice_record(
    label_id: int,
    slice_index: int,
    plane: str,
    slice_position_mm: float,
    slice_voxel_count: int,
    pixel_area: float,
    slice_intensities: np.ndarray,
) -> Dict[str, Any]:
    """Construct a standardized dictionary record for a single slice metric."""
    return {
        "label_id": label_id,
        "slice_index": slice_index,
        "plane": plane,
        "slice_position_mm": slice_position_mm,
        "slice_voxel_count": slice_voxel_count,
        "cross_sectional_area_mm2": float(slice_voxel_count * pixel_area),
        "mean_intensity": float(np.mean(slice_intensities)),
        "max_intensity": float(np.max(slice_intensities)),
    }


def generate_slice_metrics(
    raw_array: np.ndarray,
    mask_array: np.ndarray,
    unique_labels: List[int],
    metadata: VolumetricMetadata,
    label_bounds: Optional[Dict[int, Tuple[int, int]]] = None,
    label_slice_counts: Optional[Dict[int, np.ndarray]] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Generator yielding slice-level volumetric profile records.
    Yields dictionaries ready for SQLAlchemy chunked batched inserts.

    Optimized to restrict slice traversal to the axial extent [min_z, max_z]
    for each segmented label, skipping redundant empty slice evaluations.
    """
    depth, height, width = mask_array.shape
    pixel_area = (
        metadata.pixel_area_slice_mm2
        if hasattr(metadata, "pixel_area_slice_mm2")
        else getattr(metadata, "pixel_area_axial_mm2", 1.0)
    )

    plane = resolve_slice_plane(metadata)
    slice_positions = compute_slice_positions(metadata, depth)

    for lbl in unique_labels:
        if label_bounds and lbl in label_bounds:
            z_start, z_end = label_bounds[lbl]
            z_start = max(0, min(z_start, depth - 1))
            z_end = max(0, min(z_end, depth - 1))
        else:
            z_start, z_end = 0, depth - 1

        known_counts = label_slice_counts.get(lbl) if label_slice_counts else None

        for z in range(z_start, z_end + 1):
            if known_counts is not None:
                slice_count = int(known_counts[z])
                if slice_count == 0:
                    continue
                slice_mask = (mask_array[z, :, :] == lbl)
            else:
                slice_mask = (mask_array[z, :, :] == lbl)
                slice_count = int(np.count_nonzero(slice_mask))
                if slice_count == 0:
                    continue

            slice_intensities = raw_array[z, :, :][slice_mask]
            yield build_slice_record(
                label_id=lbl,
                slice_index=z,
                plane=plane,
                slice_position_mm=slice_positions[z],
                slice_voxel_count=slice_count,
                pixel_area=pixel_area,
                slice_intensities=slice_intensities,
            )


class VolumetricMetricsCalculator:
    """
    Dedicated engine for calculating quantitative volumetric statistics, physical centroids,
    bounding boxes, and memory-efficient slice-by-slice morphological profiles.
    """

    validate_arrays = staticmethod(validate_volumetric_arrays)
    compute_intensity_statistics = staticmethod(compute_intensity_statistics)
    compute_spatial_bounds_and_centroid = staticmethod(compute_spatial_bounds_and_centroid)
    compute_label_summary_with_counts = staticmethod(compute_label_summary_with_counts)
    compute_label_summary = staticmethod(compute_label_summary)
    resolve_slice_plane = staticmethod(resolve_slice_plane)
    compute_slice_positions = staticmethod(compute_slice_positions)
    build_slice_record = staticmethod(build_slice_record)
    generate_slice_metrics = staticmethod(generate_slice_metrics)

    # Backward-compatible private method aliases
    _compute_intensity_statistics = staticmethod(compute_intensity_statistics)
    _compute_spatial_bounds_and_centroid = staticmethod(compute_spatial_bounds_and_centroid)
    _compute_label_summary_with_counts = staticmethod(compute_label_summary_with_counts)
    _resolve_slice_plane = staticmethod(resolve_slice_plane)
    _compute_slice_positions = staticmethod(compute_slice_positions)
    _build_slice_record = staticmethod(build_slice_record)

    def compute_metrics(
        self,
        raw_array: np.ndarray,
        mask_array: np.ndarray,
        metadata: VolumetricMetadata,
        duration_seconds: float = 0.0,
    ) -> Tuple[SegmentationMetrics, Iterator[Dict[str, Any]]]:
        """
        Extract volumetric statistics and generate a memory-efficient slice-by-slice metric iterator.

        Args:
            raw_array: Original image array (Z, Y, X).
            mask_array: 3D segmentation mask (Z, Y, X).
            metadata: Physical geometry metadata (spacing, origin, direction).
            duration_seconds: Total inference execution time.

        Returns:
            Tuple of:
            - SegmentationMetrics: High-level volume summary per label.
            - Iterator[Dict[str, Any]]: Generator yielding 500-row chunkable slice metrics.
        """
        logger.info("Extracting quantitative volumetric metrics...")
        if metadata is None:
            raise TypeError("metadata must be a VolumetricMetadata instance, got None.")
        if not isinstance(duration_seconds, (int, float)) or not math.isfinite(duration_seconds) or duration_seconds < 0.0:
            raise ValueError(f"duration_seconds must be a non-negative finite number, got {duration_seconds}.")
        self.validate_arrays(raw_array, mask_array)

        unique_labels = [int(lbl) for lbl in np.unique(mask_array) if lbl != 0]
        label_summaries: List[LabelSummaryMetric] = []
        label_bounds: Dict[int, Tuple[int, int]] = {}
        label_slice_counts: Dict[int, np.ndarray] = {}

        for lbl in unique_labels:
            label_mask = (mask_array == lbl)
            summary, z_counts = self._compute_label_summary_with_counts(
                raw_array=raw_array,
                label_mask=label_mask,
                label_id=lbl,
                metadata=metadata,
            )
            if summary is not None:
                label_summaries.append(summary)
                # Bounding box is (min_x, min_y, min_z, max_x, max_y, max_z)
                label_bounds[lbl] = (summary.bounding_box[2], summary.bounding_box[5])
                if z_counts is not None:
                    label_slice_counts[lbl] = z_counts

        metrics = SegmentationMetrics(
            duration_seconds=duration_seconds,
            labels=label_summaries,
            metadata={
                "spatial_orientation": metadata.spatial_orientation,
                "voxel_spacing": metadata.spacing,
                "volume_dimensions": metadata.dimensions,
            },
        )

        slice_generator = self.generate_slice_metrics(
            raw_array=raw_array,
            mask_array=mask_array,
            unique_labels=unique_labels,
            metadata=metadata,
            label_bounds=label_bounds,
            label_slice_counts=label_slice_counts,
        )

        return metrics, slice_generator
