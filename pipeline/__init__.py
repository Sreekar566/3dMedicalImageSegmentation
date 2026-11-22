"""
Pipeline Package for 3dMedicalImageSegmentation.
Contains MONAI volumetric transforms, 3D U-Net model loaders, sliding-window inference engines, and volumetric metric extractors.
"""

from .transforms import build_volumetric_transforms, prepare_tensor_from_sitk
from .model import build_3d_unet, extract_state_dict, load_3d_unet
from .inference import SlidingWindowInferenceEngine
from .metrics import (
    LabelSummaryMetric,
    SegmentationMetrics,
    VolumetricMetricsCalculator,
    build_slice_record,
    compute_intensity_statistics,
    compute_label_summary,
    compute_label_summary_with_counts,
    compute_slice_positions,
    compute_spatial_bounds_and_centroid,
    generate_slice_metrics,
    resolve_slice_plane,
    validate_volumetric_arrays,
)
from .orchestrator import PipelineOrchestrator, PipelineResult

__all__ = [
    "build_volumetric_transforms",
    "prepare_tensor_from_sitk",
    "build_3d_unet",
    "extract_state_dict",
    "load_3d_unet",
    "SlidingWindowInferenceEngine",
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
    "PipelineOrchestrator",
    "PipelineResult",
]
