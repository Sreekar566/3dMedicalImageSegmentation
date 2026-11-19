"""
Database Management and High-Throughput Chunked Batched Insert Operations.
Configures SQLAlchemy QueuePool for Azure Database for PostgreSQL and implements
memory-efficient 500-row chunked batched inserts to avoid OOM errors and key-generation bottlenecks.
"""

from __future__ import annotations

import datetime
import itertools
import logging
from contextlib import contextmanager
from typing import Any, Dict, Generator, Iterable, Iterator, List, Optional, Tuple

from sqlalchemy import create_engine, insert, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from medical_segmentation.config import DatabaseSettings
from medical_segmentation.db.models import (
    Base,
    PatientStudy,
    SegmentationRun,
    VolumeMetric,
    VoxelSliceMetric,
)
from medical_segmentation.io.dicom_loader import VolumetricMetadata
from medical_segmentation.pipeline.metrics import LabelSummaryMetric, SegmentationMetrics

logger = logging.getLogger(__name__)

__all__ = [
    "DatabaseManager",
    "build_patient_study",
    "build_segmentation_run",
    "build_volume_metric",
    "build_volume_metrics",
    "chunked_iterable",
]


def build_patient_study(
    metadata: VolumetricMetadata,
    timestamp: Optional[int] = None,
) -> PatientStudy:
    """
    Construct a validated PatientStudy ORM model from VolumetricMetadata.
    Safely normalizes voxel spacing, volume dimensions, and clinical identifiers.

    Args:
        metadata: Volumetric clinical and geometric metadata.
        timestamp: Optional epoch timestamp for fallback UID generation.

    Returns:
        Instantiated PatientStudy model.
    """
    if metadata is None:
        raise TypeError("metadata must be a VolumetricMetadata instance, got None.")

    raw_sp = tuple(metadata.spacing) if getattr(metadata, "spacing", None) else ()
    spacing = (raw_sp + (1.0, 1.0, 1.0))[:3]

    raw_dims = tuple(metadata.dimensions) if getattr(metadata, "dimensions", None) else ()
    dims = (raw_dims + (0, 0, 0))[:3]

    ts = int(datetime.datetime.now().timestamp()) if timestamp is None else timestamp
    patient_id = metadata.patient_id or "ANONYMOUS"
    study_instance_uid = metadata.study_instance_uid or f"study_{ts}"
    series_instance_uid = metadata.series_instance_uid or f"series_{ts}"
    modality = metadata.modality or "CT"
    body_part = metadata.body_part or "UNKNOWN"
    spatial_orientation = metadata.spatial_orientation or "RAS"

    return PatientStudy(
        patient_id=patient_id,
        study_instance_uid=study_instance_uid,
        series_instance_uid=series_instance_uid,
        modality=modality,
        body_part=body_part,
        spatial_orientation=spatial_orientation,
        voxel_spacing_x=float(spacing[0]),
        voxel_spacing_y=float(spacing[1]),
        voxel_spacing_z=float(spacing[2]),
        dimensions_x=int(dims[0]),
        dimensions_y=int(dims[1]),
        dimensions_z=int(dims[2]),
    )


def build_segmentation_run(
    study_id: int,
    metrics: SegmentationMetrics,
    model_name: str,
    roi_size: str,
    overlap: float,
    device: str,
    output_mask_blob_uri: Optional[str] = None,
) -> SegmentationRun:
    """
    Construct a SegmentationRun ORM model associated with a parent study.

    Args:
        study_id: Foreign key ID of parent PatientStudy.
        metrics: Segmentation metrics containing duration.
        model_name: Identifier name of segmentation model.
        roi_size: Sliding-window patch size string.
        overlap: Sliding-window overlap fraction.
        device: Execution compute device string.
        output_mask_blob_uri: Optional cloud URI of saved mask.

    Returns:
        Instantiated SegmentationRun model.
    """
    if not isinstance(study_id, int) or study_id <= 0:
        raise ValueError(f"study_id must be a strictly positive integer, got {study_id}.")
    if metrics is None:
        raise TypeError("metrics must be a SegmentationMetrics instance, got None.")

    return SegmentationRun(
        study_id=study_id,
        model_name=model_name,
        roi_size=roi_size,
        overlap=overlap,
        device=device,
        duration_seconds=metrics.duration_seconds,
        output_mask_blob_uri=output_mask_blob_uri,
    )


def build_volume_metric(
    run_id: int,
    label: LabelSummaryMetric,
) -> VolumeMetric:
    """
    Construct a VolumeMetric ORM model from a LabelSummaryMetric.
    Safely handles missing or truncated physical centroid and bounding box coordinates.

    Args:
        run_id: Foreign key ID of parent SegmentationRun.
        label: Volumetric summary statistics for a segmented structure.

    Returns:
        Instantiated VolumeMetric model.
    """
    if not isinstance(run_id, int) or run_id <= 0:
        raise ValueError(f"run_id must be a strictly positive integer, got {run_id}.")
    if label is None or not isinstance(label, LabelSummaryMetric):
        raise TypeError(f"label must be a LabelSummaryMetric instance, got {type(label).__name__}.")

    raw_c = tuple(label.centroid_mm) if getattr(label, "centroid_mm", None) else ()
    c = (raw_c + (0.0, 0.0, 0.0))[:3]

    raw_b = tuple(label.bounding_box) if getattr(label, "bounding_box", None) else ()
    b = (raw_b + (0, 0, 0, 0, 0, 0))[:6]

    return VolumeMetric(
        run_id=run_id,
        label_id=label.label_id,
        structure_name=label.structure_name,
        voxel_count=label.voxel_count,
        volume_mm3=label.volume_mm3,
        volume_cm3=label.volume_cm3,
        mean_intensity=label.mean_intensity,
        std_intensity=label.std_intensity,
        min_intensity=label.min_intensity,
        max_intensity=label.max_intensity,
        centroid_x_mm=float(c[0]),
        centroid_y_mm=float(c[1]),
        centroid_z_mm=float(c[2]),
        bbox_min_x=int(b[0]),
        bbox_min_y=int(b[1]),
        bbox_min_z=int(b[2]),
        bbox_max_x=int(b[3]),
        bbox_max_y=int(b[4]),
        bbox_max_z=int(b[5]),
    )


def build_volume_metrics(
    run_id: int,
    labels: Iterable[LabelSummaryMetric],
) -> List[VolumeMetric]:
    """
    Construct a list of VolumeMetric ORM models from an iterable of LabelSummaryMetric.

    Args:
        run_id: Foreign key ID of parent SegmentationRun.
        labels: Iterable of LabelSummaryMetric objects.

    Returns:
        List of instantiated VolumeMetric models.
    """
    if not isinstance(run_id, int) or run_id <= 0:
        raise ValueError(f"run_id must be a strictly positive integer, got {run_id}.")
    if labels is None:
        return []
    return [build_volume_metric(run_id=run_id, label=label) for label in labels]


def chunked_iterable(iterable: Iterable[Any], chunk_size: int = 500) -> Generator[List[Any], None, None]:
    """
    Memory-efficient generator that slices an arbitrary iterable or generator
    into bounded batches of chunk_size elements.

    Args:
        iterable: Input generator or list of items.
        chunk_size: Maximum elements per yielded chunk (default 500).

    Yields:
        Lists of up to chunk_size items.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be a strictly positive integer, got {chunk_size}.")
    iterator = iter(iterable)
    while True:
        chunk = list(itertools.islice(iterator, chunk_size))
        if not chunk:
            break
        yield chunk


class DatabaseManager:
    """Manages PostgreSQL connection pooling, schema lifecycle, and bulk chunked insertion."""

    def __init__(self, settings: DatabaseSettings, engine: Optional[Any] = None):
        self.settings = settings
        if engine is not None:
            self.engine = engine
        else:
            logger.info(
                "Configuring PostgreSQL connection pool for host '%s', database '%s' (pool_size=%d, max_overflow=%d, batch_size=%d).",
                settings.host,
                settings.name,
                settings.pool_size,
                settings.max_overflow,
                settings.batch_size,
            )

            self.engine = create_engine(
                self.settings.connection_url,
                poolclass=QueuePool,
                pool_size=self.settings.pool_size,
                max_overflow=self.settings.max_overflow,
                pool_timeout=self.settings.pool_timeout,
                pool_pre_ping=True,      # Tests connection liveness before checkout
                pool_recycle=1800,        # Recycles connections every 30 minutes to avoid Azure idle timeouts
                echo=False,
            )

        self.SessionFactory = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )

    def dispose(self) -> None:
        """Dispose of the underlying SQLAlchemy engine and release pooled connections."""
        if hasattr(self, "engine") and self.engine is not None:
            try:
                self.engine.dispose()
                logger.info("DatabaseManager connection pool successfully disposed.")
            except Exception as exc:
                logger.warning("Error while disposing database engine: %s", exc)

    def close(self) -> None:
        """Alias for dispose() to conform to standard resource lifecycle protocols."""
        self.dispose()

    def __enter__(self) -> DatabaseManager:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.dispose()

    def init_tables(self) -> None:
        """Create all tables in PostgreSQL if they do not already exist."""
        logger.info("Verifying and initializing database schema tables...")
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database schema initialized successfully.")
        except Exception as exc:
            logger.error("Failed to initialize database tables: %s", exc)
            raise

    def check_connection(self) -> bool:
        """
        Verify database connectivity and responsiveness with a lightweight ping.

        Returns:
            True if database is reachable and responsive, False otherwise.
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as exc:
            logger.warning("Database connection check failed: %s", exc)
            return False

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Transactional session context manager with automatic rollback on error."""
        session: Session = self.SessionFactory()
        try:
            yield session
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.error("Database transaction rolled back due to error: %s", exc)
            raise
        finally:
            session.close()

    def record_study_and_run(
        self,
        metadata: VolumetricMetadata,
        metrics: SegmentationMetrics,
        model_name: str,
        roi_size: str,
        overlap: float,
        device: str,
        output_mask_blob_uri: Optional[str] = None,
    ) -> Tuple[int, int]:
        """
        Persist high-level Study, SegmentationRun, and VolumeMetric records.

        Returns:
            Tuple of (study_id, run_id).
        """
        with self.get_session() as session:
            study = build_patient_study(metadata)
            session.add(study)
            session.flush()

            run = build_segmentation_run(
                study_id=study.id,
                metrics=metrics,
                model_name=model_name,
                roi_size=roi_size,
                overlap=overlap,
                device=device,
                output_mask_blob_uri=output_mask_blob_uri,
            )
            session.add(run)
            session.flush()

            for vol_metric in build_volume_metrics(run.id, metrics.labels):
                session.add(vol_metric)

            study_id, run_id = study.id, run.id

        logger.info("Successfully persisted study_id=%d and run_id=%d.", study_id, run_id)
        return study_id, run_id

    build_patient_study = staticmethod(build_patient_study)
    build_segmentation_run = staticmethod(build_segmentation_run)
    build_volume_metric = staticmethod(build_volume_metric)
    build_volume_metrics = staticmethod(build_volume_metrics)

    def bulk_insert_slice_metrics(
        self,
        run_id: int,
        slice_generator: Iterator[Dict[str, Any]],
        batch_size: Optional[int] = None,
    ) -> int:
        """
        Execute memory-efficient, chunked batched inserts (500 rows/batch) into PostgreSQL.

        This design:
        1. Avoids Out-Of-Memory (OOM) by consuming records from a streaming Python generator.
        2. Avoids primary key generation and ORM unit-of-work overhead via SQLAlchemy Core direct multi-row inserts.
        3. Commits transactions in bounded chunks, preventing PostgreSQL transaction log bloat.

        Args:
            run_id: Parent SegmentationRun ID foreign key.
            slice_generator: Generator yielding dictionary records.
            batch_size: Number of rows per insert batch (defaults to settings.batch_size = 500).

        Returns:
            Total number of slice metric rows successfully persisted.
        """
        if not isinstance(run_id, int) or run_id <= 0:
            raise ValueError(f"run_id must be a strictly positive integer, got {run_id}.")
        effective_batch_size = self.settings.batch_size if batch_size is None else batch_size
        if effective_batch_size <= 0:
            raise ValueError(f"batch_size must be a strictly positive integer, got {effective_batch_size}.")
        logger.info(
            "Beginning chunked batched insert for run_id=%d (batch_size=%d rows/chunk)...",
            run_id,
            effective_batch_size,
        )

        total_inserted = 0
        batch_counter = 0

        for chunk in chunked_iterable(slice_generator, chunk_size=effective_batch_size):
            batch_counter += 1
            for record in chunk:
                record["run_id"] = run_id

            with self.get_session() as session:
                session.execute(insert(VoxelSliceMetric), chunk)

            total_inserted += len(chunk)
            logger.debug(
                "Persisted batch #%d (%d rows). Cumulative total: %d rows.",
                batch_counter,
                len(chunk),
                total_inserted,
            )

        logger.info(
            "Chunked batched insert complete: successfully committed %d rows across %d batches.",
            total_inserted,
            batch_counter,
        )
        return total_inserted
