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
import types
from pathlib import Path
from typing import Optional

# Ensure 'medical_segmentation' module namespace is resolvable
if "medical_segmentation" not in sys.modules:
    med_pkg = types.ModuleType("medical_segmentation")
    med_pkg.__path__ = [str(Path(__file__).resolve().parent)]
    sys.modules["medical_segmentation"] = med_pkg

from medical_segmentation.config import config
from medical_segmentation.pipeline import PipelineOrchestrator, PipelineResult

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
    try:
        orchestrator = PipelineOrchestrator(config=config)
        orchestrator.run(
            blob_path=blob_path,
            series_prefix=series_prefix,
            local_input=local_input,
            input_container=input_container,
            output_container=output_container,
            modality=modality,
            orientation=orientation,
            checkpoint=checkpoint,
            batch_size=batch_size,
            dry_run=dry_run,
        )
        return 0
    except Exception as exc:
        logger.error("Pipeline run failed: %s", exc, exc_info=True)
        return 1


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
