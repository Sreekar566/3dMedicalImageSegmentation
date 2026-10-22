"""
Database Package for 3dMedicalImageSegmentation.
Contains SQLAlchemy 2.0 ORM models, connection pooling, and 500-row chunked batch insert logic.
"""

from .models import Base, PatientStudy, SegmentationRun, VolumeMetric, VoxelSliceMetric
from .database import DatabaseManager, chunked_iterable

__all__ = [
    "Base",
    "PatientStudy",
    "SegmentationRun",
    "VolumeMetric",
    "VoxelSliceMetric",
    "DatabaseManager",
    "chunked_iterable",
]
