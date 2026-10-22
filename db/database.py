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

from sqlalchemy import create_engine, insert
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
from medical_segmentation.pipeline.inference import SegmentationMetrics

logger = logging.getLogger(__name__)


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
    iterator = iter(iterable)
    while True:
        chunk = list(itertools.islice(iterator, chunk_size))
        if not chunk:
            break
        yield chunk


class DatabaseManager:
    """Manages PostgreSQL connection pooling, schema lifecycle, and bulk chunked insertion."""

    def __init__(self, settings: DatabaseSettings):
        self.settings = settings
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

    def init_tables(self) -> None:
        """Create all tables in PostgreSQL if they do not already exist."""
        logger.info("Verifying and initializing database schema tables...")
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database schema initialized successfully.")
        except Exception as exc:
            logger.error("Failed to initialize database tables: %s", exc)
            raise

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
            study = PatientStudy(
                patient_id=metadata.patient_id,
                study_instance_uid=metadata.study_instance_uid or f"study_{int(datetime.datetime.now().timestamp())}",
                series_instance_uid=metadata.series_instance_uid or f"series_{int(datetime.datetime.now().timestamp())}",
                modality=metadata.modality,
                body_part=metadata.body_part,
                spatial_orientation=metadata.spatial_orientation,
                voxel_spacing_x=metadata.spacing[0],
                voxel_spacing_y=metadata.spacing[1],
                voxel_spacing_z=metadata.spacing[2],
                dimensions_x=metadata.dimensions[0],
                dimensions_y=metadata.dimensions[1],
                dimensions_z=metadata.dimensions[2],
            )
            session.add(study)
            session.flush()

            run = SegmentationRun(
                study_id=study.id,
                model_name=model_name,
                roi_size=roi_size,
                overlap=overlap,
                device=device,
                duration_seconds=metrics.duration_seconds,
                output_mask_blob_uri=output_mask_blob_uri,
            )
            session.add(run)
            session.flush()

            for label in metrics.labels:
                vol_metric = VolumeMetric(
                    run_id=run.id,
                    label_id=label.label_id,
                    structure_name=label.structure_name,
                    voxel_count=label.voxel_count,
                    volume_mm3=label.volume_mm3,
                    volume_cm3=label.volume_cm3,
                    mean_intensity=label.mean_intensity,
                    std_intensity=label.std_intensity,
                    min_intensity=label.min_intensity,
                    max_intensity=label.max_intensity,
                    centroid_x_mm=label.centroid_mm[0],
                    centroid_y_mm=label.centroid_mm[1],
                    centroid_z_mm=label.centroid_mm[2],
                    bbox_min_x=label.bounding_box[0],
                    bbox_min_y=label.bounding_box[1],
                    bbox_min_z=label.bounding_box[2],
                    bbox_max_x=label.bounding_box[3],
                    bbox_max_y=label.bounding_box[4],
                    bbox_max_z=label.bounding_box[5],
                )
                session.add(vol_metric)

            study_id, run_id = study.id, run.id

        logger.info("Successfully persisted study_id=%d and run_id=%d.", study_id, run_id)
        return study_id, run_id

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
        effective_batch_size = batch_size or self.settings.batch_size
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
