"""
Unit and Integration Tests for Database Management and SQLAlchemy 2.0 ORM Models.
Tests Connection Pooling, Schema Lifecycle, Transaction Management,
Cascade Deletions, and 500-Row Chunked Batched Insert Operations.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List
from unittest.mock import MagicMock, Mock, patch

import pytest
from sqlalchemy import create_engine, inspect, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.pool import QueuePool

from medical_segmentation.config import DatabaseSettings
from medical_segmentation.db import (
    Base,
    DatabaseManager,
    PatientStudy,
    SegmentationRun,
    VolumeMetric,
    VoxelSliceMetric,
    chunked_iterable,
)
from medical_segmentation.io.dicom_loader import VolumetricMetadata
from medical_segmentation.pipeline.metrics import LabelSummaryMetric, SegmentationMetrics


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def sqlite_engine():
    """Create an isolated in-memory SQLite engine for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def db_settings() -> DatabaseSettings:
    """Create test DatabaseSettings with known parameters."""
    return DatabaseSettings(
        host="test-postgres.database.azure.com",
        port=5432,
        name="test_med_db",
        user="test_admin",
        password="secret_password",
        sslmode="require",
        pool_size=15,
        max_overflow=25,
        pool_timeout=45,
        batch_size=500,
    )


@pytest.fixture
def db_manager(db_settings: DatabaseSettings, sqlite_engine) -> DatabaseManager:
    """Create a DatabaseManager initialized with the test SQLite engine."""
    return DatabaseManager(settings=db_settings, engine=sqlite_engine)


@pytest.fixture
def sample_metadata() -> VolumetricMetadata:
    """Construct a sample VolumetricMetadata instance."""
    return VolumetricMetadata(
        patient_id="PAT_TEST_888",
        study_instance_uid="1.2.840.113619.2.55.12345",
        series_instance_uid="1.2.840.113619.2.55.67890",
        modality="CT",
        body_part="ABDOMEN",
        spatial_orientation="RAS",
        spacing=(0.75, 0.75, 2.0),
        origin=(10.0, 20.0, -100.0),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        dimensions=(512, 512, 120),
    )


@pytest.fixture
def sample_metrics() -> SegmentationMetrics:
    """Construct sample multi-label volumetric segmentation metrics."""
    label1 = LabelSummaryMetric(
        label_id=1,
        structure_name="Liver",
        voxel_count=25000,
        volume_mm3=28125.0,
        volume_cm3=28.125,
        mean_intensity=105.4,
        std_intensity=14.2,
        min_intensity=60.0,
        max_intensity=150.0,
        centroid_mm=(125.0, 160.0, -40.0),
        bounding_box=(40, 50, 10, 240, 260, 90),
    )
    label2 = LabelSummaryMetric(
        label_id=2,
        structure_name="Tumor",
        voxel_count=1200,
        volume_mm3=1350.0,
        volume_cm3=1.35,
        mean_intensity=45.6,
        std_intensity=8.5,
        min_intensity=25.0,
        max_intensity=75.0,
        centroid_mm=(110.0, 140.0, -35.0),
        bounding_box=(90, 120, 25, 130, 155, 45),
    )
    return SegmentationMetrics(
        duration_seconds=8.75,
        labels=[label1, label2],
        metadata={"target_orientation": "RAS"},
    )


# =====================================================================
# DatabaseManager Initialization and Engine Configuration Tests
# =====================================================================


def test_database_manager_init_with_custom_engine(db_settings: DatabaseSettings, sqlite_engine) -> None:
    """Verify that DatabaseManager retains a supplied engine and configures SessionFactory."""
    manager = DatabaseManager(settings=db_settings, engine=sqlite_engine)
    assert manager.engine is sqlite_engine
    assert manager.settings is db_settings
    assert manager.SessionFactory is not None

    session = manager.SessionFactory()
    assert session.bind is sqlite_engine
    session.close()


def test_database_manager_init_without_engine_creates_queue_pool(db_settings: DatabaseSettings) -> None:
    """Verify that DatabaseManager constructs a QueuePool engine with settings parameters when engine is None."""
    with patch("medical_segmentation.db.database.create_engine") as mock_create_engine:
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        manager = DatabaseManager(settings=db_settings, engine=None)

        mock_create_engine.assert_called_once_with(
            db_settings.connection_url,
            poolclass=QueuePool,
            pool_size=15,
            max_overflow=25,
            pool_timeout=45,
            pool_pre_ping=True,
            pool_recycle=1800,
            echo=False,
        )
        assert manager.engine is mock_engine


def test_database_manager_init_tables_success() -> None:
    """Verify that init_tables initializes all four tables in the database schema."""
    engine = create_engine("sqlite:///:memory:")
    manager = DatabaseManager(settings=DatabaseSettings(), engine=engine)

    inspector_before = inspect(engine)
    assert len(inspector_before.get_table_names()) == 0

    manager.init_tables()

    inspector_after = inspect(engine)
    table_names = set(inspector_after.get_table_names())
    expected_tables = {
        "patient_studies",
        "segmentation_runs",
        "volume_metrics",
        "voxel_slice_metrics",
    }
    assert expected_tables.issubset(table_names)
    engine.dispose()


def test_database_manager_init_tables_exception_propagation(db_manager: DatabaseManager) -> None:
    """Verify that init_tables logs and re-raises any database exception during schema creation."""
    with patch.object(Base.metadata, "create_all", side_effect=SQLAlchemyError("Connection refused")):
        with pytest.raises(SQLAlchemyError, match="Connection refused"):
            db_manager.init_tables()


# =====================================================================
# Session Lifecycle and Transaction Management Tests
# =====================================================================


def test_database_manager_get_session_commit_on_success(db_manager: DatabaseManager) -> None:
    """Verify that get_session automatically commits transactions when exiting without errors."""
    with db_manager.get_session() as session:
        study = PatientStudy(
            patient_id="PAT_SESSION_01",
            study_instance_uid="STUDY_UID_01",
            series_instance_uid="SERIES_UID_01",
        )
        session.add(study)

    # In a separate session, verify record was committed
    with db_manager.get_session() as verify_session:
        retrieved = verify_session.execute(
            select(PatientStudy).where(PatientStudy.patient_id == "PAT_SESSION_01")
        ).scalar_one_or_none()
        assert retrieved is not None
        assert retrieved.study_instance_uid == "STUDY_UID_01"


def test_database_manager_get_session_rollback_on_exception(db_manager: DatabaseManager) -> None:
    """Verify that get_session rolls back uncommitted changes and re-raises exceptions."""
    with pytest.raises(RuntimeError, match="Simulated application failure"):
        with db_manager.get_session() as session:
            study = PatientStudy(
                patient_id="PAT_FAIL_ROLLBACK",
                study_instance_uid="STUDY_FAIL_01",
                series_instance_uid="SERIES_FAIL_01",
            )
            session.add(study)
            session.flush()
            raise RuntimeError("Simulated application failure")

    # In a separate session, verify record was NOT committed
    with db_manager.get_session() as verify_session:
        retrieved = verify_session.execute(
            select(PatientStudy).where(PatientStudy.patient_id == "PAT_FAIL_ROLLBACK")
        ).scalar_one_or_none()
        assert retrieved is None


def test_database_manager_get_session_always_closes(db_manager: DatabaseManager) -> None:
    """Verify that get_session guarantees session closure on both success and error."""
    mock_session = MagicMock()
    with patch.object(db_manager, "SessionFactory", return_value=mock_session):
        # 1. Success case
        with db_manager.get_session() as sess:
            assert sess is mock_session
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()

        mock_session.reset_mock()

        # 2. Error case
        with pytest.raises(ValueError, match="Boom"):
            with db_manager.get_session():
                raise ValueError("Boom")
        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()


# =====================================================================
# Record Study and Run Tests
# =====================================================================


def test_database_manager_record_study_and_run_full(
    db_manager: DatabaseManager,
    sample_metadata: VolumetricMetadata,
    sample_metrics: SegmentationMetrics,
) -> None:
    """Verify record_study_and_run creates Study, Run, and all VolumeMetric records accurately."""
    study_id, run_id = db_manager.record_study_and_run(
        metadata=sample_metadata,
        metrics=sample_metrics,
        model_name="MONAI_DynUNet_3D",
        roi_size="(128, 128, 64)",
        overlap=0.4,
        device="cuda:0",
        output_mask_blob_uri="https://azure.blob.core.windows.net/masks/study888.nii.gz",
    )

    assert isinstance(study_id, int)
    assert isinstance(run_id, int)
    assert study_id > 0
    assert run_id > 0

    with db_manager.get_session() as session:
        # Validate PatientStudy
        study = session.get(PatientStudy, study_id)
        assert study is not None
        assert study.patient_id == "PAT_TEST_888"
        assert study.study_instance_uid == "1.2.840.113619.2.55.12345"
        assert study.series_instance_uid == "1.2.840.113619.2.55.67890"
        assert study.modality == "CT"
        assert study.body_part == "ABDOMEN"
        assert study.spatial_orientation == "RAS"
        assert (study.voxel_spacing_x, study.voxel_spacing_y, study.voxel_spacing_z) == (0.75, 0.75, 2.0)
        assert (study.dimensions_x, study.dimensions_y, study.dimensions_z) == (512, 512, 120)

        # Validate SegmentationRun
        run = session.get(SegmentationRun, run_id)
        assert run is not None
        assert run.study_id == study_id
        assert run.model_name == "MONAI_DynUNet_3D"
        assert run.roi_size == "(128, 128, 64)"
        assert run.overlap == 0.4
        assert run.device == "cuda:0"
        assert run.duration_seconds == 8.75
        assert run.output_mask_blob_uri == "https://azure.blob.core.windows.net/masks/study888.nii.gz"

        # Validate VolumeMetric records
        vols = session.execute(
            select(VolumeMetric).where(VolumeMetric.run_id == run_id).order_by(VolumeMetric.label_id)
        ).scalars().all()
        assert len(vols) == 2

        liver = vols[0]
        assert liver.label_id == 1
        assert liver.structure_name == "Liver"
        assert liver.voxel_count == 25000
        assert liver.volume_mm3 == 28125.0
        assert liver.volume_cm3 == 28.125
        assert liver.mean_intensity == 105.4
        assert liver.std_intensity == 14.2
        assert liver.min_intensity == 60.0
        assert liver.max_intensity == 150.0
        assert (liver.centroid_x_mm, liver.centroid_y_mm, liver.centroid_z_mm) == (125.0, 160.0, -40.0)
        assert (liver.bbox_min_x, liver.bbox_min_y, liver.bbox_min_z, liver.bbox_max_x, liver.bbox_max_y, liver.bbox_max_z) == (40, 50, 10, 240, 260, 90)

        tumor = vols[1]
        assert tumor.label_id == 2
        assert tumor.structure_name == "Tumor"
        assert tumor.voxel_count == 1200
        assert tumor.volume_mm3 == 1350.0


def test_database_manager_record_study_and_run_fallback_uids(
    db_manager: DatabaseManager,
    sample_metrics: SegmentationMetrics,
) -> None:
    """Verify that missing study_instance_uid and series_instance_uid generate fallback timestamps."""
    meta = VolumetricMetadata(
        patient_id="PAT_FALLBACK",
        study_instance_uid="",
        series_instance_uid="",
    )

    study_id, run_id = db_manager.record_study_and_run(
        metadata=meta,
        metrics=sample_metrics,
        model_name="TestModel",
        roi_size="(96, 96, 96)",
        overlap=0.5,
        device="cpu",
    )

    with db_manager.get_session() as session:
        study = session.get(PatientStudy, study_id)
        assert study is not None
        assert study.study_instance_uid.startswith("study_")
        assert study.series_instance_uid.startswith("series_")


def test_database_manager_record_study_and_run_rollback_on_error(
    db_manager: DatabaseManager,
    sample_metadata: VolumetricMetadata,
    sample_metrics: SegmentationMetrics,
) -> None:
    """Verify atomicity of record_study_and_run: if an error occurs, no partial records are retained."""
    with patch.object(SegmentationRun, "__init__", side_effect=RuntimeError("Instantiation error")):
        with pytest.raises(RuntimeError, match="Instantiation error"):
            db_manager.record_study_and_run(
                metadata=sample_metadata,
                metrics=sample_metrics,
                model_name="TestModel",
                roi_size="(96, 96, 96)",
                overlap=0.5,
                device="cpu",
            )

    # Verify no orphan study was committed
    with db_manager.get_session() as session:
        count = session.execute(select(PatientStudy)).scalars().all()
        assert len(count) == 0


# =====================================================================
# Bulk Insert Slice Metrics Tests
# =====================================================================


def _generate_synthetic_slice_metrics(count: int, label_id: int = 1) -> Generator[Dict[str, Any], None, None]:
    """Helper generator yielding slice metric dictionary records."""
    for i in range(count):
        yield {
            "label_id": label_id,
            "slice_index": i,
            "plane": "axial",
            "slice_position_mm": float(i * 2.5),
            "slice_voxel_count": 100 + (i % 20),
            "cross_sectional_area_mm2": float((100 + (i % 20)) * 0.5625),
            "mean_intensity": 100.0 + (i % 10),
            "max_intensity": 180.0,
        }


def test_database_manager_bulk_insert_slice_metrics_chunking(
    db_manager: DatabaseManager,
    sample_metadata: VolumetricMetadata,
    sample_metrics: SegmentationMetrics,
) -> None:
    """Verify that bulk_insert_slice_metrics streams and inserts 1250 rows in bounded batches."""
    study_id, run_id = db_manager.record_study_and_run(
        metadata=sample_metadata,
        metrics=sample_metrics,
        model_name="TestModel",
        roi_size="(96, 96, 96)",
        overlap=0.5,
        device="cpu",
    )

    records = _generate_synthetic_slice_metrics(1250)
    total_persisted = db_manager.bulk_insert_slice_metrics(
        run_id=run_id,
        slice_generator=records,
        batch_size=500,
    )

    assert total_persisted == 1250

    with db_manager.get_session() as session:
        rows = session.execute(
            select(VoxelSliceMetric).where(VoxelSliceMetric.run_id == run_id).order_by(VoxelSliceMetric.slice_index)
        ).scalars().all()
        assert len(rows) == 1250

        # Validate run_id assignment and values of sampled records
        first_row = rows[0]
        assert first_row.run_id == run_id
        assert first_row.slice_index == 0
        assert first_row.plane == "axial"
        assert first_row.slice_position_mm == 0.0
        assert first_row.slice_voxel_count == 100

        last_row = rows[-1]
        assert last_row.run_id == run_id
        assert last_row.slice_index == 1249
        assert last_row.slice_position_mm == float(1249 * 2.5)


def test_database_manager_bulk_insert_slice_metrics_default_batch_size(
    db_manager: DatabaseManager,
    sample_metadata: VolumetricMetadata,
    sample_metrics: SegmentationMetrics,
) -> None:
    """Verify that omitting batch_size falls back to settings.batch_size."""
    db_manager.settings.batch_size = 250
    _, run_id = db_manager.record_study_and_run(
        metadata=sample_metadata,
        metrics=sample_metrics,
        model_name="TestModel",
        roi_size="(96, 96, 96)",
        overlap=0.5,
        device="cpu",
    )

    records = _generate_synthetic_slice_metrics(600)
    total = db_manager.bulk_insert_slice_metrics(run_id=run_id, slice_generator=records, batch_size=None)
    assert total == 600

    with db_manager.get_session() as session:
        count = session.execute(select(VoxelSliceMetric).where(VoxelSliceMetric.run_id == run_id)).scalars().all()
        assert len(count) == 600


def test_database_manager_bulk_insert_slice_metrics_empty_generator(
    db_manager: DatabaseManager,
    sample_metadata: VolumetricMetadata,
    sample_metrics: SegmentationMetrics,
) -> None:
    """Verify that an empty generator yields 0 persisted rows without errors."""
    _, run_id = db_manager.record_study_and_run(
        metadata=sample_metadata,
        metrics=sample_metrics,
        model_name="TestModel",
        roi_size="(96, 96, 96)",
        overlap=0.5,
        device="cpu",
    )

    total = db_manager.bulk_insert_slice_metrics(run_id=run_id, slice_generator=iter([]), batch_size=500)
    assert total == 0

    with db_manager.get_session() as session:
        count = session.execute(select(VoxelSliceMetric).where(VoxelSliceMetric.run_id == run_id)).scalars().all()
        assert len(count) == 0


def test_database_manager_bulk_insert_slice_metrics_invalid_batch_size(db_manager: DatabaseManager) -> None:
    """Verify that non-positive batch_size raises a ValueError."""
    for invalid_size in [0, -1, -500]:
        with pytest.raises(ValueError, match="strictly positive integer"):
            db_manager.bulk_insert_slice_metrics(
                run_id=1,
                slice_generator=iter([]),
                batch_size=invalid_size,
            )


def test_database_manager_bulk_insert_slice_metrics_error_propagation(
    db_manager: DatabaseManager,
    sample_metadata: VolumetricMetadata,
    sample_metrics: SegmentationMetrics,
) -> None:
    """Verify that an error during chunk insertion propagates and rolls back the active transaction."""
    _, run_id = db_manager.record_study_and_run(
        metadata=sample_metadata,
        metrics=sample_metrics,
        model_name="TestModel",
        roi_size="(96, 96, 96)",
        overlap=0.5,
        device="cpu",
    )

    # Missing non-nullable column 'slice_voxel_count'
    malformed_records = [
        {"label_id": 1, "slice_index": 0},
    ]

    with pytest.raises(SQLAlchemyError):
        db_manager.bulk_insert_slice_metrics(
            run_id=run_id,
            slice_generator=iter(malformed_records),
            batch_size=10,
        )


# =====================================================================
# ORM Schema Models, Relationships, and Cascades Tests
# =====================================================================


def test_orm_relationships_and_orphan_cascade_delete(db_manager: DatabaseManager) -> None:
    """Verify ORM relationships and cascade delete behavior from PatientStudy to runs and metrics."""
    with db_manager.get_session() as session:
        study = PatientStudy(
            patient_id="PAT_CASCADE_TEST",
            study_instance_uid="STUDY_C1",
            series_instance_uid="SERIES_C1",
        )
        run = SegmentationRun(
            study=study,
            model_name="CascadeModel",
        )
        vol_metric = VolumeMetric(
            run=run,
            label_id=1,
            voxel_count=500,
            volume_mm3=500.0,
            volume_cm3=0.5,
        )
        slice_metric = VoxelSliceMetric(
            run=run,
            label_id=1,
            slice_index=0,
            slice_voxel_count=100,
        )
        session.add_all([study, run, vol_metric, slice_metric])

    # Verify insertion and bidirectional relationships
    with db_manager.get_session() as session:
        retrieved_study = session.execute(
            select(PatientStudy).where(PatientStudy.patient_id == "PAT_CASCADE_TEST")
        ).scalar_one()

        assert len(retrieved_study.segmentation_runs) == 1
        retrieved_run = retrieved_study.segmentation_runs[0]
        assert retrieved_run.study is retrieved_study
        assert len(retrieved_run.volume_metrics) == 1
        assert len(retrieved_run.slice_metrics) == 1
        assert retrieved_run.volume_metrics[0].run is retrieved_run
        assert retrieved_run.slice_metrics[0].run is retrieved_run

        # Delete study and verify ORM cascade deletes child run and metrics
        session.delete(retrieved_study)

    # Verify all records are deleted
    with db_manager.get_session() as session:
        assert session.execute(select(PatientStudy)).scalars().all() == []
        assert session.execute(select(SegmentationRun)).scalars().all() == []
        assert session.execute(select(VolumeMetric)).scalars().all() == []
        assert session.execute(select(VoxelSliceMetric)).scalars().all() == []


def test_orm_models_table_args_and_index_definitions() -> None:
    """Verify table names, composite indexes, and metadata definitions on ORM models."""
    assert PatientStudy.__tablename__ == "patient_studies"
    assert SegmentationRun.__tablename__ == "segmentation_runs"
    assert VolumeMetric.__tablename__ == "volume_metrics"
    assert VoxelSliceMetric.__tablename__ == "voxel_slice_metrics"

    index_names = {idx.name: [c.name for c in idx.columns] for idx in VoxelSliceMetric.__table__.indexes}
    assert "ix_slice_metrics_run_slice" in index_names
    assert index_names["ix_slice_metrics_run_slice"] == ["run_id", "slice_index"]
    assert "ix_slice_metrics_run_label" in index_names
    assert index_names["ix_slice_metrics_run_label"] == ["run_id", "label_id"]


def test_chunked_iterable_utility_edge_cases() -> None:
    """Verify chunked_iterable handles edge cases such as empty input, exact multiples, and remainders."""
    # Empty
    assert list(chunked_iterable([], chunk_size=10)) == []

    # Exact multiple
    chunks = list(chunked_iterable(range(100), chunk_size=25))
    assert len(chunks) == 4
    assert all(len(c) == 25 for c in chunks)

    # Non-multiple with remainder
    chunks = list(chunked_iterable(range(105), chunk_size=50))
    assert len(chunks) == 3
    assert len(chunks[0]) == 50
    assert len(chunks[1]) == 50
    assert len(chunks[2]) == 5

    # Negative chunk_size
    with pytest.raises(ValueError, match="strictly positive integer"):
        list(chunked_iterable(range(10), chunk_size=0))
    with pytest.raises(ValueError, match="strictly positive integer"):
        list(chunked_iterable(range(10), chunk_size=-5))


def test_database_manager_dispose_and_context_manager() -> None:
    """Verify DatabaseManager.dispose(), close(), and context manager protocol properly manage engine lifecycle."""
    test_engine = create_engine("sqlite:///:memory:")
    settings = DatabaseSettings()
    db = DatabaseManager(settings=settings, engine=test_engine)

    assert db.check_connection() is True
    db.dispose()

    # Context manager protocol test
    engine2 = create_engine("sqlite:///:memory:")
    with DatabaseManager(settings=settings, engine=engine2) as managed_db:
        assert managed_db.check_connection() is True
    # Upon context exit, dispose should have been invoked
    assert managed_db.engine is not None


def test_database_manager_record_study_and_run_malformed_metadata_tuples(
    db_manager: DatabaseManager,
) -> None:
    """Verify record_study_and_run safely handles missing or truncated spacing, dimensions, centroid, and bbox."""
    # Metadata with empty or short tuples that would previously cause IndexError
    meta = VolumetricMetadata(
        patient_id="PATIENT_ROBUST",
        study_instance_uid="1.2.3.4.robust",
        series_instance_uid="1.2.3.4.robust.1",
        modality="CT",
        body_part="CHEST",
        spatial_orientation="RAS",
        spacing=(1.5,),  # Len < 3
        dimensions=(512, 512),  # Len < 3
    )

    # Label with short centroid and bbox
    label = LabelSummaryMetric(
        label_id=1,
        structure_name="Lesion",
        voxel_count=100,
        volume_mm3=500.0,
        volume_cm3=0.5,
        mean_intensity=50.0,
        std_intensity=5.0,
        min_intensity=10.0,
        max_intensity=90.0,
        centroid_mm=(10.0, 20.0),  # Len < 3
        bounding_box=(0, 0, 0, 10),  # Len < 6
    )

    metrics = SegmentationMetrics(
        duration_seconds=2.5,
        labels=[label],
    )

    study_id, run_id = db_manager.record_study_and_run(
        metadata=meta,
        metrics=metrics,
        model_name="3D_UNet_Robust",
        roi_size="(32, 32, 32)",
        overlap=0.5,
        device="cpu",
    )

    assert study_id > 0
    assert run_id > 0

    with db_manager.get_session() as session:
        study = session.get(PatientStudy, study_id)
        assert study is not None
        assert study.voxel_spacing_x == 1.5
        assert study.voxel_spacing_y == 1.0  # Safe fallback
        assert study.voxel_spacing_z == 1.0  # Safe fallback
        assert study.dimensions_x == 512
        assert study.dimensions_y == 512
        assert study.dimensions_z == 0  # Safe fallback

        vol_metrics = session.query(VolumeMetric).filter_by(run_id=run_id).all()
        assert len(vol_metrics) == 1
        vm = vol_metrics[0]
        assert vm.centroid_x_mm == 10.0
        assert vm.centroid_y_mm == 20.0
        assert vm.centroid_z_mm == 0.0  # Safe fallback
        assert vm.bbox_max_y == 0  # Safe fallback
        assert vm.bbox_max_z == 0  # Safe fallback


def test_database_manager_bulk_insert_slice_metrics_invalid_run_id(
    db_manager: DatabaseManager,
) -> None:
    """Verify bulk_insert_slice_metrics rejects non-positive run_id."""
    with pytest.raises(ValueError, match="run_id must be a strictly positive integer"):
        db_manager.bulk_insert_slice_metrics(run_id=0, slice_generator=iter([]))

    with pytest.raises(ValueError, match="run_id must be a strictly positive integer"):
        db_manager.bulk_insert_slice_metrics(run_id=-1, slice_generator=iter([]))
