"""
Quantitative Volumetric and Morphological Metric Extraction Engine.
Extracts label-level physical statistics and generates memory-efficient
slice-by-slice voxel distribution profiles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from medical_segmentation.io.dicom_loader import VolumetricMetadata

logger = logging.getLogger(__name__)


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


class VolumetricMetricsCalculator:
    """
    Dedicated engine for calculating quantitative volumetric statistics, physical centroids,
    bounding boxes, and memory-efficient slice-by-slice morphological profiles.
    """

    def validate_arrays(self, raw_array: np.ndarray, mask_array: np.ndarray) -> None:
        """Validate array dimensions and compatibility."""
        if raw_array.shape != mask_array.shape:
            raise ValueError(
                f"Array shape mismatch: raw_array shape {raw_array.shape} "
                f"does not match mask_array shape {mask_array.shape}."
            )

    def compute_label_summary(
        self,
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
        voxel_count = int(np.count_nonzero(label_mask))
        if voxel_count == 0:
            return None

        volume_mm3 = float(voxel_count * metadata.voxel_volume_mm3)
        volume_cm3 = float(volume_mm3 / 1000.0)

        roi_intensities = raw_array[label_mask]
        mean_val = float(np.mean(roi_intensities))
        std_val = float(np.std(roi_intensities))
        min_val = float(np.min(roi_intensities))
        max_val = float(np.max(roi_intensities))

        z_indices, y_indices, x_indices = np.where(label_mask)
        cz = float(np.mean(z_indices))
        cy = float(np.mean(y_indices))
        cx = float(np.mean(x_indices))

        # Transform continuous voxel index (X, Y, Z) to physical patient coordinates in mm
        centroid_mm = metadata.continuous_index_to_physical_point((cx, cy, cz))

        bbox = (
            int(np.min(x_indices)),
            int(np.min(y_indices)),
            int(np.min(z_indices)),
            int(np.max(x_indices)),
            int(np.max(y_indices)),
            int(np.max(z_indices)),
        )

        structure_name = f"Segmented_Structure_{label_id}" if label_id > 1 else "Primary_Lesion"

        return LabelSummaryMetric(
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

    def generate_slice_metrics(
        self,
        raw_array: np.ndarray,
        mask_array: np.ndarray,
        unique_labels: List[int],
        metadata: VolumetricMetadata,
        label_bounds: Optional[Dict[int, Tuple[int, int]]] = None,
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
            else metadata.pixel_area_axial_mm2
        )

        for lbl in unique_labels:
            if label_bounds and lbl in label_bounds:
                z_start, z_end = label_bounds[lbl]
                z_start = max(0, min(z_start, depth - 1))
                z_end = max(0, min(z_end, depth - 1))
            else:
                z_start, z_end = 0, depth - 1

            for z in range(z_start, z_end + 1):
                slice_mask = (mask_array[z, :, :] == lbl)
                slice_count = int(np.count_nonzero(slice_mask))
                if slice_count == 0:
                    continue

                slice_intensities = raw_array[z, :, :][slice_mask]
                area_mm2 = float(slice_count * pixel_area)
                if hasattr(metadata, "get_slice_plane_and_position"):
                    plane, slice_pos_mm = metadata.get_slice_plane_and_position(z)
                else:
                    plane = "axial"
                    slice_pos_mm = metadata.continuous_index_to_physical_point((0.0, 0.0, float(z)))[2]

                yield {
                    "label_id": lbl,
                    "slice_index": z,
                    "plane": plane,
                    "slice_position_mm": slice_pos_mm,
                    "slice_voxel_count": slice_count,
                    "cross_sectional_area_mm2": area_mm2,
                    "mean_intensity": float(np.mean(slice_intensities)),
                    "max_intensity": float(np.max(slice_intensities)),
                }

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
        self.validate_arrays(raw_array, mask_array)

        unique_labels = [int(lbl) for lbl in np.unique(mask_array) if lbl != 0]
        label_summaries: List[LabelSummaryMetric] = []
        label_bounds: Dict[int, Tuple[int, int]] = {}

        for lbl in unique_labels:
            label_mask = (mask_array == lbl)
            summary = self.compute_label_summary(
                raw_array=raw_array,
                label_mask=label_mask,
                label_id=lbl,
                metadata=metadata,
            )
            if summary is not None:
                label_summaries.append(summary)
                # Bounding box is (min_x, min_y, min_z, max_x, max_y, max_z)
                label_bounds[lbl] = (summary.bounding_box[2], summary.bounding_box[5])

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
        )

        return metrics, slice_generator
