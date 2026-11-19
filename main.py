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
from typing import Any, Dict, Optional, Sequence

# Ensure 'medical_segmentation' module namespace is resolvable
if "medical_segmentation" not in sys.modules:
    med_pkg = types.ModuleType("medical_segmentation")
    med_pkg.__path__ = [str(Path(__file__).resolve().parent)]
    sys.modules["medical_segmentation"] = med_pkg

from medical_segmentation.config import (
    PipelineConfig,
    config,
    validate_azure_container_name,
    validate_orientation_code,
)
default_config = config
from medical_segmentation.diagnostics import (
    check_system_health as _diagnostics_check_system_health,
    run_health_check as _diagnostics_run_health_check,
)
from medical_segmentation.io.volume_ingester import validate_input_sources
from medical_segmentation.logging_config import (
    StandardTextFormatter,
    StructuredJsonFormatter,
    configure_logging,
    get_logger,
)
from medical_segmentation.pipeline import PipelineOrchestrator, PipelineResult

__all__ = [
    "StandardTextFormatter",
    "StructuredJsonFormatter",
    "check_system_health",
    "configure_logging",
    "get_logger",
    "main",
    "parse_args",
    "run_health_check",
    "run_pipeline",
]

configure_logging(
    level=config.log_level,
    log_format=config.log_format,
    log_file=config.log_file,
)
logger = get_logger("3dMedicalImageSegmentation.main")


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
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
    input_group.add_argument(
        "--check-health",
        action="store_true",
        help="Perform diagnostic system and dependency health check and exit.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to a JSON configuration file to initialize pipeline settings.",
    )
    parser.add_argument(
        "--check-db",
        action="store_true",
        help="Include PostgreSQL database connection verification during health check.",
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
    parser.add_argument(
        "--log-level",
        type=str,
        default=config.log_level,
        choices=["DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL"],
        help="Application logging verbosity (default: %(default)s).",
    )
    parser.add_argument(
        "--log-format",
        type=str,
        default=config.log_format,
        choices=["text", "json"],
        help="Log output formatting style ('text' or 'json', default: %(default)s).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=config.log_file,
        help="Optional destination path for persistent log file.",
    )

    return parser.parse_args()


def run_pipeline(
    blob_path: Optional[str] = None,
    series_prefix: Optional[str] = None,
    local_input: Optional[Path | str] = None,
    input_container: Optional[str] = None,
    output_container: Optional[str] = None,
    modality: str = "CT",
    orientation: str = "RAS",
    checkpoint: Optional[Path | str] = None,
    batch_size: int = 500,
    dry_run: bool = False,
    config: Optional[PipelineConfig] = None,
) -> int:
    """
    Execute the end-to-end 3D medical segmentation workflow.

    Returns:
        Exit code: 0 on success, non-zero on error.
    """
    try:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be a strictly positive integer, got {batch_size}.")
        orientation = validate_orientation_code(orientation)
        if not isinstance(modality, str) or modality.upper().strip() not in ("CT", "CBCT", "MRI"):
            raise ValueError(f"Invalid modality '{modality}'. Must be one of: 'CT', 'CBCT', 'MRI'.")
        modality = modality.upper().strip()

        if input_container is not None:
            validate_azure_container_name(input_container)
        if output_container is not None:
            validate_azure_container_name(output_container)

        clean_blob, clean_prefix, clean_local = validate_input_sources(
            blob_path=blob_path,
            series_prefix=series_prefix,
            local_input=local_input,
        )

        active_config = config or default_config
        with PipelineOrchestrator(config=active_config) as orchestrator:
            orchestrator.run(
                blob_path=clean_blob,
                series_prefix=clean_prefix,
                local_input=clean_local,
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


def check_system_health(
    check_db: bool = False,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    """
    Perform diagnostic health checks across runtime dependencies and storage.
    Delegates to medical_segmentation.diagnostics.check_system_health.

    Returns:
        Dictionary detailing component statuses and overall health flag ('status': 'healthy' | 'unhealthy').
    """
    return _diagnostics_check_system_health(check_db=check_db, config=config)


def run_health_check(
    check_db: bool = False,
    config: Optional[PipelineConfig] = None,
) -> int:
    """
    CLI/probe wrapper for check_system_health.
    Logs diagnostics and returns 0 if healthy, 1 if unhealthy.
    Delegates to medical_segmentation.diagnostics.run_health_check.
    """
    return _diagnostics_run_health_check(
        check_db=check_db,
        config=config,
        health_checker=check_system_health,
    )


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    config_file = getattr(args, "config", None)
    active_config = config
    has_custom_config = False
    if isinstance(config_file, (str, Path)):
        active_config = PipelineConfig.from_file(config_file)
        has_custom_config = True

    log_level = getattr(args, "log_level", None)
    log_format = getattr(args, "log_format", None)
    log_file = getattr(args, "log_file", None)
    if isinstance(log_level, str) or isinstance(log_format, str) or isinstance(log_file, (str, Path)):
        configure_logging(
            level=log_level if isinstance(log_level, str) else active_config.log_level,
            log_format=log_format if isinstance(log_format, str) else active_config.log_format,
            log_file=log_file if isinstance(log_file, (str, Path)) else active_config.log_file,
        )

    if getattr(args, "check_health", None) is True:
        check_db = getattr(args, "check_db", None) is True
        if has_custom_config:
            sys.exit(run_health_check(check_db=check_db, config=active_config))
        sys.exit(run_health_check(check_db=check_db))

    kwargs: Dict[str, Any] = {
        "blob_path": args.blob_path,
        "series_prefix": args.series_prefix,
        "local_input": args.local_input,
        "input_container": args.container,
        "output_container": args.output_container,
        "modality": args.modality,
        "orientation": args.orientation,
        "checkpoint": args.checkpoint,
        "batch_size": args.batch_size,
        "dry_run": args.dry_run,
    }
    if has_custom_config:
        kwargs["config"] = active_config
    sys.exit(run_pipeline(**kwargs))


if __name__ == "__main__":
    main()
