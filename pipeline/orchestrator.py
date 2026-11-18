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
    validate_orientation_code,
)
from medical_segmentation.io.azure_client import AzureBlobClient
from medical_segmentation.io.dicom_loader import DicomVolumeLoader, VolumetricMetadata, sitk_to_numpy
from medical_segmentation.io.mask_writer import save_mask_as_nifti
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
    ):
        self.config = config or default_config
        self.azure_client = azure_client or AzureBlobClient(self.config.azure)
        self.volume_loader = volume_loader or DicomVolumeLoader(
            target_orientation=self.config.inference.target_orientation
        )
        self.db_manager = db_manager
        self.inference_engine = inference_engine
        self.metrics_calculator = metrics_calculator or VolumetricMetricsCalculator()
        self.workspace = Path(workspace or self.config.work_dir).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

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
        container = input_container or self.config.azure.input_container

        if local_input:
            target_path = Path(local_input).resolve()
            if not target_path.exists():
                raise FileNotFoundError(f"Local input path not found: {target_path}")
            logger.info("Operating in local mode using path: '%s'", target_path)
            return target_path, target_path.is_dir()

        if series_prefix:
            logger.info(
                "Downloading DICOM series prefix '%s' from container '%s'...",
                series_prefix,
                container,
            )
            series_dir = self.workspace / "raw_series" / Path(series_prefix).name
            self.azure_client.download_series_prefix(
                container_name=container,
                prefix=series_prefix,
                destination_dir=series_dir,
            )
            return series_dir, True

        if blob_path:
            logger.info(
                "Downloading volume blob '%s' from container '%s'...",
                blob_path,
                container,
            )
            local_file = self.workspace / "raw_volumes" / Path(blob_path).name
            self.azure_client.download_blob_to_file(
                container_name=container,
                blob_name=blob_path,
                destination_path=local_file,
            )
            return local_file, False

        raise ValueError("No input volume source specified (provide local_input, series_prefix, or blob_path).")

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
        target_orientation = validate_orientation_code(
            orientation or self.config.inference.target_orientation
        )
        if target_orientation != self.volume_loader.target_orientation:
            self.volume_loader = DicomVolumeLoader(target_orientation=target_orientation)

        logger.info(
            "Standardizing volume geometry (sitk.DICOMOrient -> %s)...",
            self.volume_loader.target_orientation,
        )
        if is_dicom_dir:
            sitk_image, metadata = self.volume_loader.load_dicom_series(target_path)
        else:
            sitk_image, metadata = self.volume_loader.load_volume_file(target_path)

        metadata.modality = modality
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

    def export_mask(
        self,
        mask_array: np.ndarray,
        reference_image: sitk.Image,
        metadata: VolumetricMetadata,
        output_container: Optional[str] = None,
        dry_run: bool = False,
    ) -> Tuple[Path, Optional[str]]:
        """
        Step 6: Export segmentation mask to compressed NIfTI and upload to Azure Blob Storage.

        Returns:
            Tuple of (local_mask_path, output_blob_uri).
        """
        clean_uid = re.sub(r"[^\w\-.]", "_", metadata.study_instance_uid or "run")
        mask_filename = f"mask_{clean_uid}_{int(time.time())}.nii.gz"
        local_mask_path = self.workspace / "outputs" / mask_filename

        save_mask_as_nifti(
            mask_array=mask_array,
            reference_image=reference_image,
            output_path=local_mask_path,
        )

        output_blob_uri: Optional[str] = None
        container = output_container or self.config.azure.output_container
        if not dry_run and self.azure_client.is_connected:
            output_blob_uri = self.azure_client.upload_file_to_blob(
                container_name=container,
                blob_name=f"segmentations/{mask_filename}",
                source_path=local_mask_path,
                content_type="application/gzip",
            )
            logger.info("Mask uploaded to Azure Blob: %s", output_blob_uri)
        else:
            logger.info("Dry-run active or Azure disconnected; skipped mask cloud upload.")

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
        modality: str = "CT",
        orientation: Optional[str] = None,
        checkpoint: Optional[Path | str] = None,
        batch_size: Optional[int] = None,
        dry_run: bool = False,
    ) -> PipelineResult:
        """
        Execute end-to-end 3D medical segmentation workflow across all 7 stages.
        """
        total_start = time.perf_counter()
        target_orientation = validate_orientation_code(
            orientation or self.config.inference.target_orientation
        )

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
            modality=modality,
            orientation=target_orientation,
        )

        logger.info("=== Step 3: MONAI Volumetric Transformation Pipeline ===")
        volume_tensor = self.prepare_tensor(sitk_image=sitk_image, modality=modality)

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
