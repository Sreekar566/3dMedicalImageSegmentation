"""
Volumetric Sliding-Window Inference Engine.
Executes MONAI sliding_window_inference with Gaussian blending and delegates
quantitative physical metric extraction to VolumetricMetricsCalculator.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference

from medical_segmentation.config import InferenceSettings
from medical_segmentation.io.dicom_loader import VolumetricMetadata
from medical_segmentation.pipeline.metrics import (
    LabelSummaryMetric,
    SegmentationMetrics,
    VolumetricMetricsCalculator,
)

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
__all__ = [
    "LabelSummaryMetric",
    "SegmentationMetrics",
    "SlidingWindowInferenceEngine",
    "VolumetricMetricsCalculator",
]


class SlidingWindowInferenceEngine:
    """Production 3D sliding-window inference runner with Gaussian patch blending."""

    def __init__(
        self,
        model: nn.Module,
        settings: InferenceSettings,
        metrics_calculator: Optional[VolumetricMetricsCalculator] = None,
    ):
        self.model = model
        self.settings = settings
        self.device = torch.device(settings.device)
        self.metrics_calculator = metrics_calculator or VolumetricMetricsCalculator()

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
                    discrete_mask = torch.argmax(output_logits, dim=1).squeeze(0)
                else:
                    discrete_mask = (output_logits > 0.0).to(torch.uint8).squeeze(0).squeeze(0)

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
        Delegates to VolumetricMetricsCalculator.
        """
        return self.metrics_calculator.compute_metrics(
            raw_array=raw_array,
            mask_array=mask_array,
            metadata=metadata,
            duration_seconds=duration_seconds,
        )
