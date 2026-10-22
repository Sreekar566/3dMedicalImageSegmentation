"""
SQLAlchemy 2.0 ORM Schema Models for 3dMedicalImageSegmentation.
Defines tables for Patient Studies, Segmentation Runs, Volumetric Summary Metrics,
and Slice-Level Voxel Distribution Profiles.
"""

from __future__ import annotations

import datetime
from typing import List, Optional
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base class for SQLAlchemy 2.0 ORM models."""
    pass


class PatientStudy(Base):
    """Clinical study and volumetric metadata record."""

    __tablename__ = "patient_studies"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    patient_id: Mapped[str] = mapped_column(String(128), index=True, default="ANONYMOUS")
    study_instance_uid: Mapped[str] = mapped_column(String(256), index=True)
    series_instance_uid: Mapped[str] = mapped_column(String(256), index=True)
    modality: Mapped[str] = mapped_column(String(32), default="CT")
    body_part: Mapped[str] = mapped_column(String(64), default="UNKNOWN")
    spatial_orientation: Mapped[str] = mapped_column(String(16), default="RAS")

    voxel_spacing_x: Mapped[float] = mapped_column(Float, default=1.0)
    voxel_spacing_y: Mapped[float] = mapped_column(Float, default=1.0)
    voxel_spacing_z: Mapped[float] = mapped_column(Float, default=1.0)

    dimensions_x: Mapped[int] = mapped_column(Integer, default=0)
    dimensions_y: Mapped[int] = mapped_column(Integer, default=0)
    dimensions_z: Mapped[int] = mapped_column(Integer, default=0)

    blob_uri: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    segmentation_runs: Mapped[List["SegmentationRun"]] = relationship(
        "SegmentationRun",
        back_populates="study",
        cascade="all, delete-orphan",
    )


class SegmentationRun(Base):
    """Record of an executed volumetric inference run."""

    __tablename__ = "segmentation_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    study_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("patient_studies.id", ondelete="CASCADE"),
        index=True,
    )
    model_name: Mapped[str] = mapped_column(String(128), default="MONAI_3D_UNet")
    roi_size: Mapped[str] = mapped_column(String(64), default="(96, 96, 96)")
    overlap: Mapped[float] = mapped_column(Float, default=0.5)
    device: Mapped[str] = mapped_column(String(32), default="cpu")
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    output_mask_blob_uri: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    study: Mapped["PatientStudy"] = relationship("PatientStudy", back_populates="segmentation_runs")
    volume_metrics: Mapped[List["VolumeMetric"]] = relationship(
        "VolumeMetric",
        back_populates="run",
        cascade="all, delete-orphan",
    )
    slice_metrics: Mapped[List["VoxelSliceMetric"]] = relationship(
        "VoxelSliceMetric",
        back_populates="run",
        cascade="all, delete-orphan",
    )


class VolumeMetric(Base):
    """Aggregated quantitative summary metrics per segmented anatomical structure."""

    __tablename__ = "volume_metrics"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("segmentation_runs.id", ondelete="CASCADE"),
        index=True,
    )
    label_id: Mapped[int] = mapped_column(Integer, nullable=False)
    structure_name: Mapped[str] = mapped_column(String(128), default="Structure")

    voxel_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    volume_mm3: Mapped[float] = mapped_column(Float, nullable=False)
    volume_cm3: Mapped[float] = mapped_column(Float, nullable=False)

    mean_intensity: Mapped[float] = mapped_column(Float, default=0.0)
    std_intensity: Mapped[float] = mapped_column(Float, default=0.0)
    min_intensity: Mapped[float] = mapped_column(Float, default=0.0)
    max_intensity: Mapped[float] = mapped_column(Float, default=0.0)

    centroid_x_mm: Mapped[float] = mapped_column(Float, default=0.0)
    centroid_y_mm: Mapped[float] = mapped_column(Float, default=0.0)
    centroid_z_mm: Mapped[float] = mapped_column(Float, default=0.0)

    bbox_min_x: Mapped[int] = mapped_column(Integer, default=0)
    bbox_min_y: Mapped[int] = mapped_column(Integer, default=0)
    bbox_min_z: Mapped[int] = mapped_column(Integer, default=0)
    bbox_max_x: Mapped[int] = mapped_column(Integer, default=0)
    bbox_max_y: Mapped[int] = mapped_column(Integer, default=0)
    bbox_max_z: Mapped[int] = mapped_column(Integer, default=0)

    run: Mapped["SegmentationRun"] = relationship("SegmentationRun", back_populates="volume_metrics")


class VoxelSliceMetric(Base):
    """
    High-volume slice-by-slice voxel distribution metrics table.
    Designed for memory-efficient 500-row chunked batched inserts to avoid OOM errors
    and key-generation round-trip bottlenecks.
    """

    __tablename__ = "voxel_slice_metrics"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("segmentation_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    label_id: Mapped[int] = mapped_column(Integer, nullable=False)
    slice_index: Mapped[int] = mapped_column(Integer, nullable=False)
    plane: Mapped[str] = mapped_column(String(16), default="axial")
    slice_position_mm: Mapped[float] = mapped_column(Float, default=0.0)
    slice_voxel_count: Mapped[int] = mapped_column(Integer, nullable=False)
    cross_sectional_area_mm2: Mapped[float] = mapped_column(Float, default=0.0)
    mean_intensity: Mapped[float] = mapped_column(Float, default=0.0)
    max_intensity: Mapped[float] = mapped_column(Float, default=0.0)

    run: Mapped["SegmentationRun"] = relationship("SegmentationRun", back_populates="slice_metrics")

    __table_args__ = (
        Index("ix_slice_metrics_run_slice", "run_id", "slice_index"),
        Index("ix_slice_metrics_run_label", "run_id", "label_id"),
    )
