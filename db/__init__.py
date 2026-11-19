"""
Database Package for 3dMedicalImageSegmentation.
Contains SQLAlchemy 2.0 ORM models, connection pooling, and 500-row chunked batch insert logic.
"""

from .models import Base, PatientStudy, SegmentationRun, VolumeMetric, VoxelSliceMetric
from .database import (
    DatabaseManager,
    build_patient_study,
    build_segmentation_run,
    build_volume_metric,
    build_volume_metrics,
    chunked_iterable,
)

__all__ = [
    "Base",
    "PatientStudy",
    "SegmentationRun",
    "VolumeMetric",
    "VoxelSliceMetric",
    "DatabaseManager",
    "build_patient_study",
    "build_segmentation_run",
    "build_volume_metric",
    "build_volume_metrics",
    "chunked_iterable",
]
