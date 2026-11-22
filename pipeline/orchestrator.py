"""
End-to-End Volumetric Segmentation Pipeline Orchestrator.
Coordinates volume ingestion, spatial orientation standardization,
MONAI volumetric transformation, sliding-window inference, quantitative metric
extraction, NIfTI mask serialization, and chunked database persistence.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional, Tuple

import numpy as np
import SimpleITK as sitk
import torch

from medical_segmentation.config import (
    PipelineConfig,
    config as default_config,
    validate_azure_container_name,
    validate_orientation_code,
)
from medical_segmentation.io.azure_client import AzureBlobClient
from medical_segmentation.io.dicom_loader import DicomVolumeLoader, VolumetricMetadata, sitk_to_numpy
from medical_segmentation.io.mask_writer import VolumetricMaskWriter, save_mask_as_nifti
from medical_segmentation.io.volume_ingester import VolumeIngester, validate_input_sources
from medical_segmentation.pipeline.inference import SlidingWindowInferenceEngine
from medical_segmentation.pipeline.metrics import (
    SegmentationMetrics,
    VolumetricMetricsCalculator,
)
from medical_segmentation.pipeline.model import load_3d_unet
from medical_segmentation.pipeline.transforms import prepare_tensor_from_sitk

if TYPE_CHECKING:
    from medical_segmentation.db.database import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Structured execution results from an end-to-end segmentation pipeline run."""

    summary_metrics: SegmentationMetrics
    mask_array: np.ndarray
    local_mask_path: Path
    output_blob_uri: Optional[str]
    study_id: Optional[int]
    run_id: Optional[int]
    slices_inserted: int
    duration_seconds: float
    metadata: VolumetricMetadata

    @property
    def is_persisted_to_db(self) -> bool:
        """Check whether segmentation run was successfully persisted to PostgreSQL."""
        return self.study_id is not None and self.run_id is not None

    @property
    def is_uploaded_to_cloud(self) -> bool:
        """Check whether segmentation mask was uploaded to Azure Blob Storage."""
        return bool(self.output_blob_uri)

    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline execution summary to a JSON-serializable dictionary."""
        return {
            "study_id": self.study_id,
            "run_id": self.run_id,
            "duration_seconds": self.duration_seconds,
            "local_mask_path": str(self.local_mask_path),
            "output_blob_uri": self.output_blob_uri,
            "slices_inserted": self.slices_inserted,
            "is_persisted_to_db": self.is_persisted_to_db,
            "is_uploaded_to_cloud": self.is_uploaded_to_cloud,
            "patient_id": self.metadata.patient_id,
            "study_instance_uid": self.metadata.study_instance_uid,
            "modality": self.metadata.modality,
            "dimensions": list(self.metadata.dimensions),
            "spacing": list(self.metadata.spacing),
            "spatial_orientation": self.metadata.spatial_orientation,
            "labels_count": len(self.summary_metrics.labels),
        }


class PipelineOrchestrator:
    """
    Modular orchestrator for the 3D medical image segmentation pipeline.
    Decouples volume ingestion, orientation, transforms, inference,
    metric computation, file export, and database persistence into testable stages.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        azure_client: Optional[AzureBlobClient] = None,
        volume_loader: Optional[DicomVolumeLoader] = None,
        db_manager: Optional[DatabaseManager] = None,
        inference_engine: Optional[SlidingWindowInferenceEngine] = None,
        metrics_calculator: Optional[VolumetricMetricsCalculator] = None,
        workspace: Optional[Path | str] = None,
        mask_writer: Optional[VolumetricMaskWriter] = None,
        volume_ingester: Optional[VolumeIngester] = None,
    ):
        self.config = config or default_config
        self.azure_client = azure_client or AzureBlobClient(self.config.azure)
        self.volume_loader = volume_loader or DicomVolumeLoader(
            target_orientation=self.config.inference.target_orientation
        )
        self.db_manager = db_manager
        self.inference_engine = inference_engine
        self.metrics_calculator = metrics_calculator or VolumetricMetricsCalculator()
        if workspace is not None:
            if not str(workspace).strip():
                raise ValueError("workspace cannot be an empty path.")
            self.workspace = Path(workspace).resolve()
        else:
            self.workspace = Path(self.config.work_dir).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.mask_writer = mask_writer or VolumetricMaskWriter(
            default_output_dir=self.workspace / "outputs"
        )
        self.volume_ingester = volume_ingester or VolumeIngester(
            azure_client=self.azure_client,
            workspace=self.workspace,
            default_container=self.config.azure.input_container,
        )

    def close(self) -> None:
        """
        Release allocated resources and dispose of database connection pools if active.
        """
        if self.db_manager is not None and hasattr(self.db_manager, "dispose"):
            self.db_manager.dispose()

    def __enter__(self) -> PipelineOrchestrator:
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit point; ensures resource cleanup."""
        self.close()

    def ingest_volume(
        self,
        blob_path: Optional[str] = None,
        series_prefix: Optional[str] = None,
        local_input: Optional[Path | str] = None,
        input_container: Optional[str] = None,
    ) -> Tuple[Path, bool]:
        """
        Step 1: Ingest input volume from local filesystem or Azure Blob Storage.

        Args:
            blob_path: Path to single 3D volume blob in Azure.
            series_prefix: Prefix/folder of DICOM slice blobs in Azure.
            local_input: Local file or directory path.
            input_container: Azure input container name.

        Returns:
            Tuple of (resolved_path, is_dicom_dir).
        """
        return self.volume_ingester.ingest(
            blob_path=blob_path,
            series_prefix=series_prefix,
            local_input=local_input,
            input_container=input_container,
        )

    def standardize_geometry(
        self,
        target_path: Path,
        is_dicom_dir: bool,
        modality: str = "CT",
        orientation: Optional[str] = None,
    ) -> Tuple[sitk.Image, VolumetricMetadata]:
        """
        Step 2: Load volume and standardize spatial orientation via sitk.DICOMOrient.
        """
        if not isinstance(modality, str) or modality.upper().strip() not in ("CT", "CBCT", "MRI"):
            raise ValueError(f"Invalid modality '{modality}'. Must be one of: 'CT', 'CBCT', 'MRI'.")
        modality_clean = modality.upper().strip()

        target_orientation = validate_orientation_code(
            orientation or self.config.inference.target_orientation
        )
        if target_orientation != self.volume_loader.target_orientation:
            self.volume_loader = DicomVolumeLoader(target_orientation=target_orientation)

        logger.info(
            "Standardizing volume geometry (sitk.DICOMOrient -> %s)...",
            self.volume_loader.target_orientation,
        )
        if hasattr(self.volume_loader, "load"):
            sitk_image, metadata = self.volume_loader.load(target_path, is_dicom_dir=is_dicom_dir)
        elif is_dicom_dir:
            sitk_image, metadata = self.volume_loader.load_dicom_series(target_path)
        else:
            sitk_image, metadata = self.volume_loader.load_volume_file(target_path)

        metadata.modality = modality_clean
        logger.info(
            "Geometry standardized: dimensions=%s, spacing=%s, orientation=%s",
            metadata.dimensions,
            metadata.spacing,
            metadata.spatial_orientation,
        )
        return sitk_image, metadata

    def prepare_tensor(
        self,
        sitk_image: sitk.Image,
        modality: str = "CT",
    ) -> torch.Tensor:
        """
        Step 3: MONAI volumetric transformation into normalized batch tensor (1, C, D, H, W).
        """
        logger.info("Preparing MONAI volumetric tensor for inference...")
        return prepare_tensor_from_sitk(
            image=sitk_image,
            modality=modality,
            target_spacing=self.config.inference.target_spacing,
            device=self.config.inference.device,
        )

    def execute_inference(
        self,
        volume_tensor: torch.Tensor,
        checkpoint: Optional[Path | str] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Step 4: Load 3D U-Net and execute sliding-window volumetric inference.

        Returns:
            Tuple of (discretized integer mask array, inference duration in seconds).
        """
        if self.inference_engine is None:
            model = load_3d_unet(
                checkpoint_path=checkpoint or self.config.inference.model_checkpoint_path,
                in_channels=self.config.inference.in_channels,
                out_channels=self.config.inference.out_channels,
                device=self.config.inference.device,
            )
            self.inference_engine = SlidingWindowInferenceEngine(
                model=model,
                settings=self.config.inference,
                metrics_calculator=self.metrics_calculator,
            )

        logger.info("Executing 3D sliding-window volumetric inference...")
        start_time = time.perf_counter()
        mask_array = self.inference_engine.execute_inference(volume_tensor)
        duration = time.perf_counter() - start_time
        logger.info("Volumetric inference completed in %.2f seconds.", duration)
        return mask_array, duration

    def extract_metrics(
        self,
        raw_array: np.ndarray,
        mask_array: np.ndarray,
        metadata: VolumetricMetadata,
        inference_duration: float,
    ) -> Tuple[SegmentationMetrics, Iterator[Dict[str, Any]]]:
        """
        Step 5: Extract quantitative volumetric statistics and generate slice iterator.
        """
        logger.info("Extracting quantitative volumetric metrics...")
        summary_metrics, slice_generator = self.metrics_calculator.compute_metrics(
            raw_array=raw_array,
            mask_array=mask_array,
            metadata=metadata,
            duration_seconds=inference_duration,
        )

        for label in summary_metrics.labels:
            logger.info(
                "Structure '%s' (Label %d): Volume = %.2f mm³ (%.3f cm³), Voxels = %d, Mean Intensity = %.1f",
                label.structure_name,
                label.label_id,
                label.volume_mm3,
                label.volume_cm3,
                label.voxel_count,
                label.mean_intensity,
            )

        return summary_metrics, slice_generator

    def generate_mask_filename(
        self,
        metadata: VolumetricMetadata,
        timestamp: Optional[int] = None,
    ) -> str:
        """
        Generate a standardized, filesystem-safe filename for a segmentation mask.

        Args:
            metadata: Volumetric clinical metadata containing study UID.
            timestamp: Optional epoch timestamp (defaults to current time).

        Returns:
            Formatted NIfTI filename (e.g., 'mask_1_2_840_113619_1699999999.nii.gz').
        """
        clean_uid = re.sub(r"[^\w\-.]", "_", metadata.study_instance_uid or "run")
        ts = int(time.time()) if timestamp is None else timestamp
        return f"mask_{clean_uid}_{ts}.nii.gz"

    def upload_mask(
        self,
        local_mask_path: Path,
        mask_filename: str,
        output_container: Optional[str] = None,
        dry_run: bool = False,
    ) -> Optional[str]:
        """
        Upload saved segmentation mask to Azure Blob Storage if connected and not in dry-run mode.

        Args:
            local_mask_path: Local filesystem path to the mask file.
            mask_filename: Destination filename within Azure Blob prefix.
            output_container: Azure destination container name.
            dry_run: Whether to skip cloud upload.

        Returns:
            Uploaded Azure Blob URI string, or None if skipped.
        """
        if not isinstance(mask_filename, str) or not mask_filename.strip():
            raise ValueError("mask_filename must be a non-empty string.")

        container = (
            validate_azure_container_name(output_container)
            if output_container is not None
            else self.config.azure.output_container
        )

        if dry_run or not self.azure_client.is_connected:
            logger.info("Dry-run active or Azure disconnected; skipped mask cloud upload.")
            return None

        output_blob_uri = self.azure_client.upload_file_to_blob(
            container_name=container,
            blob_name=f"segmentations/{mask_filename.strip()}",
            source_path=local_mask_path,
            content_type="application/gzip",
        )
        logger.info("Mask uploaded to Azure Blob: %s", output_blob_uri)
        return output_blob_uri

    def export_mask(
        self,
        mask_array: np.ndarray,
        reference_image: sitk.Image | VolumetricMetadata,
        metadata: VolumetricMetadata,
        output_container: Optional[str] = None,
        dry_run: bool = False,
    ) -> Tuple[Path, Optional[str]]:
        """
        Step 6: Export segmentation mask to compressed NIfTI and upload to Azure Blob Storage.

        Returns:
            Tuple of (local_mask_path, output_blob_uri).
        """
        if output_container is not None:
            validate_azure_container_name(output_container)

        mask_filename = self.generate_mask_filename(metadata)
        local_mask_path = self.mask_writer.write_mask(
            mask_array=mask_array,
            reference=reference_image,
            filename=mask_filename,
        )
        output_blob_uri = self.upload_mask(
            local_mask_path=local_mask_path,
            mask_filename=mask_filename,
            output_container=output_container,
            dry_run=dry_run,
        )
        return local_mask_path, output_blob_uri

    def persist_to_database(
        self,
        metadata: VolumetricMetadata,
        summary_metrics: SegmentationMetrics,
        slice_generator: Iterator[Dict[str, Any]],
        output_blob_uri: Optional[str] = None,
        batch_size: Optional[int] = None,
        dry_run: bool = False,
    ) -> Tuple[Optional[int], Optional[int], int]:
        """
        Step 7: Chunked batched ingestion into PostgreSQL database.

        Returns:
            Tuple of (study_id, run_id, total_slices_inserted).
        """
        if dry_run:
            logger.info("Dry-run active; skipped PostgreSQL persistence.")
            return None, None, 0

        effective_batch_size = self.config.db.batch_size if batch_size is None else batch_size
        if effective_batch_size <= 0:
            raise ValueError(f"batch_size must be a strictly positive integer, got {effective_batch_size}.")
        owns_db = False
        db: Optional[Any] = None
        try:
            if self.db_manager is not None:
                db = self.db_manager
            else:
                from medical_segmentation.db.database import DatabaseManager
                db = DatabaseManager(self.config.db)
                owns_db = True
            db.init_tables()

            study_id, run_id = db.record_study_and_run(
                metadata=metadata,
                metrics=summary_metrics,
                model_name="MONAI_3D_UNet",
                roi_size=str(self.config.inference.roi_size),
                overlap=self.config.inference.overlap,
                device=self.config.inference.device,
                output_mask_blob_uri=output_blob_uri,
            )

            total_slices_inserted = db.bulk_insert_slice_metrics(
                run_id=run_id,
                slice_generator=slice_generator,
                batch_size=effective_batch_size,
            )
            logger.info("Database ingestion finalized: %d slice rows committed.", total_slices_inserted)
            return study_id, run_id, total_slices_inserted
        except Exception as exc:
            logger.error("PostgreSQL database persistence failed: %s", exc, exc_info=True)
            logger.warning("Pipeline proceeding despite database persistence failure.")
            return None, None, 0
        finally:
            if owns_db and db is not None and hasattr(db, "dispose"):
                db.dispose()

    def run(
        self,
        blob_path: Optional[str] = None,
        series_prefix: Optional[str] = None,
        local_input: Optional[Path | str] = None,
        input_container: Optional[str] = None,
        output_container: Optional[str] = None,
        modality: Optional[str] = None,
        orientation: Optional[str] = None,
        checkpoint: Optional[Path | str] = None,
        batch_size: Optional[int] = None,
        dry_run: bool = False,
    ) -> PipelineResult:
        """
        Execute end-to-end 3D medical segmentation workflow across all 7 stages.
        """
        total_start = time.perf_counter()
        raw_modality = modality if modality is not None else self.config.inference.modality
        if not isinstance(raw_modality, str) or raw_modality.upper().strip() not in ("CT", "CBCT", "MRI"):
            raise ValueError(f"Invalid modality '{raw_modality}'. Must be one of: 'CT', 'CBCT', 'MRI'.")
        modality_clean = raw_modality.upper().strip()
        target_orientation = validate_orientation_code(
            orientation or self.config.inference.target_orientation
        )
        if batch_size is not None and batch_size <= 0:
            raise ValueError(f"batch_size must be a strictly positive integer, got {batch_size}.")
        if input_container is not None:
            validate_azure_container_name(input_container)
        if output_container is not None:
            validate_azure_container_name(output_container)

        logger.info("=== Step 1: Volume Ingestion ===")
        target_path, is_dicom_dir = self.ingest_volume(
            blob_path=blob_path,
            series_prefix=series_prefix,
            local_input=local_input,
            input_container=input_container,
        )

        logger.info("=== Step 2: Spatial Standardization (sitk.DICOMOrient -> %s) ===", target_orientation)
        sitk_image, metadata = self.standardize_geometry(
            target_path=target_path,
            is_dicom_dir=is_dicom_dir,
            modality=modality_clean,
            orientation=target_orientation,
        )

        logger.info("=== Step 3: MONAI Volumetric Transformation Pipeline ===")
        volume_tensor = self.prepare_tensor(sitk_image=sitk_image, modality=modality_clean)

        logger.info("=== Step 4: MONAI 3D Sliding-Window Volumetric Inference ===")
        mask_array, inference_duration = self.execute_inference(
            volume_tensor=volume_tensor,
            checkpoint=checkpoint,
        )
        raw_array = sitk_to_numpy(sitk_image)

        logger.info("=== Step 5: Quantitative Metric Extraction ===")
        summary_metrics, slice_generator = self.extract_metrics(
            raw_array=raw_array,
            mask_array=mask_array,
            metadata=metadata,
            inference_duration=inference_duration,
        )

        logger.info("=== Step 6: Export Segmentation Mask ===")
        local_mask_path, output_blob_uri = self.export_mask(
            mask_array=mask_array,
            reference_image=sitk_image,
            metadata=metadata,
            output_container=output_container,
            dry_run=dry_run,
        )

        logger.info("=== Step 7: PostgreSQL Database Ingestion ===")
        study_id, run_id, slices_inserted = self.persist_to_database(
            metadata=metadata,
            summary_metrics=summary_metrics,
            slice_generator=slice_generator,
            output_blob_uri=output_blob_uri,
            batch_size=batch_size,
            dry_run=dry_run,
        )

        total_duration = time.perf_counter() - total_start
        logger.info("=== Pipeline Completed Successfully in %.2f seconds ===", total_duration)

        return PipelineResult(
            summary_metrics=summary_metrics,
            mask_array=mask_array,
            local_mask_path=local_mask_path,
            output_blob_uri=output_blob_uri,
            study_id=study_id,
            run_id=run_id,
            slices_inserted=slices_inserted,
            duration_seconds=total_duration,
            metadata=metadata,
        )
