"""
Volumetric Sliding-Window Inference and Quantitative Metric Extraction Engine.
Executes MONAI sliding_window_inference with Gaussian blending and calculates
voxel-level, slice-by-slice, and organ-level physical metrics.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference

from medical_segmentation.config import InferenceSettings
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


class SlidingWindowInferenceEngine:
    """Production 3D sliding-window inference runner with Gaussian patch blending."""

    def __init__(self, model: nn.Module, settings: InferenceSettings):
        self.model = model
        self.settings = settings
        self.device = torch.device(settings.device)

    def execute_inference(self, volume_tensor: torch.Tensor) -> np.ndarray:
        """
        Execute sliding-window inference on a full 3D volumetric tensor.

        Args:
            volume_tensor: Tensor of shape (1, C, D, H, W) on target device.

        Returns:
            Discretized 3D integer label map array of shape (D, H, W).
        """
        logger.info(
            "Executing 3D sliding-window inference: roi_size=%s, sw_batch_size=%d, overlap=%.2f, mode='gaussian'",
            self.settings.roi_size,
            self.settings.sw_batch_size,
            self.settings.overlap,
        )
        self.model.eval()
        volume_tensor = volume_tensor.to(self.device)

        start_time = time.perf_counter()
        use_cuda_amp = self.device.type == "cuda"

        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type, enabled=use_cuda_amp):
                output_logits = sliding_window_inference(
                    inputs=volume_tensor,
                    roi_size=self.settings.roi_size,
                    sw_batch_size=self.settings.sw_batch_size,
                    predictor=self.model,
                    overlap=self.settings.overlap,
                    mode="gaussian",
                    device=self.device,
                )

                if self.settings.out_channels > 1:
                    probabilities = torch.softmax(output_logits, dim=1)
                    discrete_mask = torch.argmax(probabilities, dim=1).squeeze(0)
                else:
                    probabilities = torch.sigmoid(output_logits)
                    discrete_mask = (probabilities > 0.5).int().squeeze(0).squeeze(0)

        elapsed = time.perf_counter() - start_time
        logger.info("3D inference completed in %.2f seconds.", elapsed)

        mask_np = discrete_mask.cpu().numpy().astype(np.uint8)
        return mask_np

    def compute_metrics(
        self,
        raw_array: np.ndarray,
        mask_array: np.ndarray,
        metadata: VolumetricMetadata,
        duration_seconds: float,
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

        if raw_array.shape != mask_array.shape:
            raise ValueError(
                f"Array shape mismatch: raw_array shape {raw_array.shape} "
                f"does not match mask_array shape {mask_array.shape}."
            )

        dx, dy, dz = metadata.spacing
        voxel_volume_mm3 = dx * dy * dz
        pixel_area_axial_mm2 = dx * dy

        unique_labels = [int(lbl) for lbl in np.unique(mask_array) if lbl != 0]
        label_summaries: List[LabelSummaryMetric] = []

        for lbl in unique_labels:
            label_mask = (mask_array == lbl)
            voxel_count = int(np.count_nonzero(label_mask))

            if voxel_count == 0:
                continue

            volume_mm3 = float(voxel_count * voxel_volume_mm3)
            volume_cm3 = float(volume_mm3 / 1000.0)

            roi_intensities = raw_array[label_mask]
            mean_val = float(np.mean(roi_intensities))
            std_val = float(np.std(roi_intensities))
            min_val = float(np.min(roi_intensities))
            max_val = float(np.max(roi_intensities))

            z_indices, y_indices, x_indices = np.where(label_mask)
            cz, cy, cx = np.mean(z_indices), np.mean(y_indices), np.mean(x_indices)

            ox, oy, oz = metadata.origin
            if len(metadata.direction) == 9:
                d = metadata.direction
                # SimpleITK continuous index (cx, cy, cz) to physical point in patient space
                px = ox + d[0] * (cx * dx) + d[1] * (cy * dy) + d[2] * (cz * dz)
                py = oy + d[3] * (cx * dx) + d[4] * (cy * dy) + d[5] * (cz * dz)
                pz = oz + d[6] * (cx * dx) + d[7] * (cy * dy) + d[8] * (cz * dz)
                centroid_mm = (float(px), float(py), float(pz))
            else:
                centroid_mm = (
                    float(ox + cx * dx),
                    float(oy + cy * dy),
                    float(oz + cz * dz),
                )

            bbox = (
                int(np.min(x_indices)),
                int(np.min(y_indices)),
                int(np.min(z_indices)),
                int(np.max(x_indices)),
                int(np.max(y_indices)),
                int(np.max(z_indices)),
            )

            structure_name = f"Segmented_Structure_{lbl}" if lbl > 1 else "Primary_Lesion"

            label_summaries.append(
                LabelSummaryMetric(
                    label_id=lbl,
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
            )

        summary = SegmentationMetrics(
            duration_seconds=duration_seconds,
            labels=label_summaries,
            metadata={
                "spatial_orientation": metadata.spatial_orientation,
                "voxel_spacing": metadata.spacing,
                "volume_dimensions": metadata.dimensions,
            },
        )

        slice_generator = self._generate_slice_metrics(
            raw_array=raw_array,
            mask_array=mask_array,
            unique_labels=unique_labels,
            metadata=metadata,
            pixel_area_axial_mm2=pixel_area_axial_mm2,
        )

        return summary, slice_generator

    def _generate_slice_metrics(
        self,
        raw_array: np.ndarray,
        mask_array: np.ndarray,
        unique_labels: List[int],
        metadata: VolumetricMetadata,
        pixel_area_axial_mm2: float,
    ) -> Iterator[Dict[str, Any]]:
        """
        Generator yielding slice-level volumetric profile records.
        Yields dictionaries ready for SQLAlchemy chunked batched inserts.
        """
        depth, height, width = mask_array.shape
        ox, oy, oz = metadata.origin
        dx, dy, dz = metadata.spacing
        has_direction = len(metadata.direction) == 9
        d = metadata.direction if has_direction else ()

        for lbl in unique_labels:
            for z in range(depth):
                slice_mask = (mask_array[z, :, :] == lbl)
                slice_count = int(np.count_nonzero(slice_mask))
                if slice_count == 0:
                    continue

                slice_intensities = raw_array[z, :, :][slice_mask]
                area_mm2 = float(slice_count * pixel_area_axial_mm2)
                if has_direction:
                    pos_z_mm = float(oz + d[6] * 0.0 + d[7] * 0.0 + d[8] * (z * dz))
                else:
                    pos_z_mm = float(oz + z * dz)

                yield {
                    "label_id": lbl,
                    "slice_index": z,
                    "plane": "axial",
                    "slice_position_mm": pos_z_mm,
                    "slice_voxel_count": slice_count,
                    "cross_sectional_area_mm2": area_mm2,
                    "mean_intensity": float(np.mean(slice_intensities)),
                    "max_intensity": float(np.max(slice_intensities)),
                }
