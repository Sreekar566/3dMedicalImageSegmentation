"""
Main CLI Orchestration Entry Point for 3dMedicalImageSegmentation.
Coordinates Azure Blob I/O, SimpleITK spatial orientation, MONAI sliding-window inference,
quantitative metric extraction, and 500-row chunked batched PostgreSQL ingestion.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from medical_segmentation.config import config
from medical_segmentation.db.database import DatabaseManager
from medical_segmentation.io.azure_client import AzureBlobClient
from medical_segmentation.io.dicom_loader import DicomVolumeLoader, VolumetricMetadata
from medical_segmentation.pipeline.inference import SlidingWindowInferenceEngine
from medical_segmentation.pipeline.model import load_3d_unet
from medical_segmentation.pipeline.transforms import prepare_tensor_from_sitk

logging.basicConfig(
    level=getattr(logging, config.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("3dMedicalImageSegmentation.main")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the segmentation pipeline."""
    parser = argparse.ArgumentParser(
        description="3D Medical Image Volumetric Segmentation & Cloud Ingestion Pipeline"
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--blob-path",
        type=str,
        help="Path to a single 3D volume blob in Azure (e.g., 'mri/patient_01.nii.gz').",
    )
    input_group.add_argument(
        "--series-prefix",
        type=str,
        help="Prefix/folder of DICOM slice blobs in Azure (e.g., 'dicom/series_001/').",
    )
    input_group.add_argument(
        "--local-input",
        type=Path,
        help="Local file or directory path for testing without Azure download.",
    )

    parser.add_argument(
        "--container",
        type=str,
        default=config.azure.input_container,
        help="Azure Blob input container name.",
    )
    parser.add_argument(
        "--output-container",
        type=str,
        default=config.azure.output_container,
        help="Azure Blob output container name.",
    )
    parser.add_argument(
        "--modality",
        choices=["CT", "CBCT", "MRI"],
        default=config.inference.modality,
        help="Imaging modality for intensity preprocessing.",
    )
    parser.add_argument(
        "--orientation",
        type=str,
        default=config.inference.target_orientation,
        help="Target anatomical orientation for SimpleITK DICOMOrient (e.g. 'RAS').",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=config.inference.model_checkpoint_path,
        help="Path to pre-trained 3D U-Net checkpoint (.pt or .pth).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.db.batch_size,
        help="Batch size for chunked PostgreSQL insertion (default: 500).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute inference and metric extraction without uploading to Azure or PostgreSQL.",
    )

    return parser.parse_args()


def run_pipeline(
    blob_path: Optional[str] = None,
    series_prefix: Optional[str] = None,
    local_input: Optional[Path] = None,
    input_container: Optional[str] = None,
    output_container: Optional[str] = None,
    modality: str = "CT",
    orientation: str = "RAS",
    checkpoint: Optional[Path] = None,
    batch_size: int = 500,
    dry_run: bool = False,
) -> int:
    """
    Execute the end-to-end 3D medical segmentation workflow.

    Returns:
        Exit code: 0 on success, non-zero on error.
    """
    total_start = time.perf_counter()
    workspace = config.work_dir
    workspace.mkdir(parents=True, exist_ok=True)

    input_container = input_container or config.azure.input_container
    output_container = output_container or config.azure.output_container

    azure_client = AzureBlobClient(config.azure)
    loader = DicomVolumeLoader(target_orientation=orientation)

    # -------------------------------------------------------------------------
    # Step 1: Volume Ingestion (Azure Blob Storage or Local Input)
    # -------------------------------------------------------------------------
    logger.info("=== Step 1: Volume Ingestion ===")
    if local_input:
        target_path = local_input
        logger.info("Operating in local mode using path: '%s'", target_path)
        is_dicom_dir = target_path.is_dir()
    elif series_prefix:
        logger.info("Downloading DICOM series prefix '%s' from container '%s'...", series_prefix, input_container)
        series_dir = workspace / "raw_series" / Path(series_prefix).name
        azure_client.download_series_prefix(
            container_name=input_container,
            prefix=series_prefix,
            destination_dir=series_dir,
        )
        target_path = series_dir
        is_dicom_dir = True
    elif blob_path:
        logger.info("Downloading volume blob '%s' from container '%s'...", blob_path, input_container)
        local_file = workspace / "raw_volumes" / Path(blob_path).name
        azure_client.download_blob_to_file(
            container_name=input_container,
            blob_name=blob_path,
            destination_path=local_file,
        )
        target_path = local_file
        is_dicom_dir = False
    else:
        logger.error("No input volume source specified.")
        return 1

    # -------------------------------------------------------------------------
    # Step 2: Spatial Orientation & Metadata Standardization
    # -------------------------------------------------------------------------
    logger.info("=== Step 2: Spatial Standardization (sitk.DICOMOrient -> %s) ===", orientation)
    if is_dicom_dir:
        sitk_image, metadata = loader.load_dicom_series(target_path)
    else:
        sitk_image, metadata = loader.load_volume_file(target_path)

    metadata.modality = modality
    logger.info(
        "Geometry standardized: dimensions=%s, spacing=%s, orientation=%s",
        metadata.dimensions,
        metadata.spacing,
        metadata.spatial_orientation,
    )

    # -------------------------------------------------------------------------
    # Step 3: MONAI Preprocessing Transforms
    # -------------------------------------------------------------------------
    logger.info("=== Step 3: MONAI Volumetric Transformation Pipeline ===")
    volume_tensor = prepare_tensor_from_sitk(
        image=sitk_image,
        modality=modality,
        target_spacing=config.inference.target_spacing,
        device=config.inference.device,
    )

    # -------------------------------------------------------------------------
    # Step 4: 3D U-Net Model Loading & Sliding-Window Inference
    # -------------------------------------------------------------------------
    logger.info("=== Step 4: MONAI 3D Sliding-Window Volumetric Inference ===")
    model = load_3d_unet(
        checkpoint_path=checkpoint,
        in_channels=config.inference.in_channels,
        out_channels=config.inference.out_channels,
        device=config.inference.device,
    )

    inferer = SlidingWindowInferenceEngine(model=model, settings=config.inference)
    inference_start = time.perf_counter()
    mask_array = inferer.execute_inference(volume_tensor)
    inference_duration = time.perf_counter() - inference_start

    raw_array = DicomVolumeLoader.sitk_to_numpy(sitk_image)

    # -------------------------------------------------------------------------
    # Step 5: Quantitative Metric Extraction
    # -------------------------------------------------------------------------
    logger.info("=== Step 5: Quantitative Metric Extraction ===")
    summary_metrics, slice_generator = inferer.compute_metrics(
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

    # -------------------------------------------------------------------------
    # Step 6: Export Segmentation Mask (NIfTI) & Upload to Azure
    # -------------------------------------------------------------------------
    logger.info("=== Step 6: Export Segmentation Mask ===")
    mask_filename = f"mask_{metadata.study_instance_uid or 'run'}_{int(time.time())}.nii.gz"
    local_mask_path = workspace / "outputs" / mask_filename

    DicomVolumeLoader.save_mask_as_nifti(
        mask_array=mask_array,
        reference_image=sitk_image,
        output_path=local_mask_path,
    )

    output_blob_uri: Optional[str] = None
    if not dry_run and azure_client.is_connected:
        output_blob_uri = azure_client.upload_file_to_blob(
            container_name=output_container,
            blob_name=f"segmentations/{mask_filename}",
            source_path=local_mask_path,
            content_type="application/gzip",
        )
        logger.info("Mask uploaded to Azure Blob: %s", output_blob_uri)
    else:
        logger.info("Dry-run active or Azure disconnected; skipped mask cloud upload.")

    # -------------------------------------------------------------------------
    # Step 7: Chunked Batched Ingestion to PostgreSQL (500 rows/batch)
    # -------------------------------------------------------------------------
    logger.info("=== Step 7: PostgreSQL Database Ingestion ===")
    if not dry_run:
        try:
            db_manager = DatabaseManager(config.db)
            db_manager.init_tables()

            study_id, run_id = db_manager.record_study_and_run(
                metadata=metadata,
                metrics=summary_metrics,
                model_name="MONAI_3D_UNet",
                roi_size=str(config.inference.roi_size),
                overlap=config.inference.overlap,
                device=config.inference.device,
                output_mask_blob_uri=output_blob_uri,
            )

            total_slices_inserted = db_manager.bulk_insert_slice_metrics(
                run_id=run_id,
                slice_generator=slice_generator,
                batch_size=batch_size,
            )
            logger.info("Database ingestion finalized: %d slice rows committed.", total_slices_inserted)
        except Exception as exc:
            logger.error("PostgreSQL database persistence failed: %s", exc, exc_info=True)
            if not dry_run:
                logger.warning("Pipeline proceeding despite database persistence failure.")
    else:
        logger.info("Dry-run active; skipped PostgreSQL persistence.")

    total_duration = time.perf_counter() - total_start
    logger.info("=== Pipeline Completed Successfully in %.2f seconds ===", total_duration)
    return 0


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    sys.exit(
        run_pipeline(
            blob_path=args.blob_path,
            series_prefix=args.series_prefix,
            local_input=args.local_input,
            input_container=args.container,
            output_container=args.output_container,
            modality=args.modality,
            orientation=args.orientation,
            checkpoint=args.checkpoint,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
