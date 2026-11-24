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
    check_azure,
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
    "check_azure",
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
        "--check-azure",
        action="store_true",
        help="Include Azure Blob Storage connection verification during health check.",
    )
    parser.add_argument(
        "--check-all",
        action="store_true",
        help="Include all external services (PostgreSQL and Azure Blob Storage) during health check.",
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

    return parser.parse_args(args)


def run_pipeline(
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
    config: Optional[PipelineConfig] = None,
) -> int:
    """
    Execute the end-to-end 3D medical segmentation workflow.

    Returns:
        Exit code: 0 on success, non-zero on error.
    """
    try:
        active_config = config or default_config

        eff_batch_size = active_config.db.batch_size if batch_size is None else batch_size
        if eff_batch_size <= 0:
            raise ValueError(f"batch_size must be a strictly positive integer, got {eff_batch_size}.")

        eff_orientation = validate_orientation_code(
            orientation if orientation is not None else active_config.inference.target_orientation
        )

        raw_modality = modality if modality is not None else active_config.inference.modality
        if not isinstance(raw_modality, str) or raw_modality.upper().strip() not in ("CT", "CBCT", "MRI"):
            raise ValueError(f"Invalid modality '{raw_modality}'. Must be one of: 'CT', 'CBCT', 'MRI'.")
        eff_modality = raw_modality.upper().strip()

        eff_in_container = input_container if input_container is not None else active_config.azure.input_container
        if eff_in_container is not None:
            validate_azure_container_name(eff_in_container)

        eff_out_container = output_container if output_container is not None else active_config.azure.output_container
        if eff_out_container is not None:
            validate_azure_container_name(eff_out_container)

        eff_checkpoint = checkpoint if checkpoint is not None else active_config.inference.model_checkpoint_path

        clean_blob, clean_prefix, clean_local = validate_input_sources(
            blob_path=blob_path,
            series_prefix=series_prefix,
            local_input=local_input,
        )

        with PipelineOrchestrator(config=active_config) as orchestrator:
            orchestrator.run(
                blob_path=clean_blob,
                series_prefix=clean_prefix,
                local_input=clean_local,
                input_container=eff_in_container,
                output_container=eff_out_container,
                modality=eff_modality,
                orientation=eff_orientation,
                checkpoint=eff_checkpoint,
                batch_size=eff_batch_size,
                dry_run=dry_run,
            )
        return 0
    except Exception as exc:
        logger.error("Pipeline run failed: %s", exc, exc_info=True)
        return 1


def check_system_health(
    check_db: bool = False,
    check_azure: bool = False,
    verify_azure_containers: bool = False,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    """
    Perform diagnostic health checks across runtime dependencies and storage.
    Delegates to medical_segmentation.diagnostics.check_system_health.

    Returns:
        Dictionary detailing component statuses and overall health flag ('status': 'healthy' | 'unhealthy').
    """
    return _diagnostics_check_system_health(
        check_db=check_db,
        check_azure=check_azure,
        verify_azure_containers=verify_azure_containers,
        config=config,
    )


def run_health_check(
    check_db: bool = False,
    check_azure: bool = False,
    verify_azure_containers: bool = False,
    config: Optional[PipelineConfig] = None,
    health_checker: Optional[Callable[..., Dict[str, Any]]] = None,
) -> int:
    """
    CLI/probe wrapper for check_system_health.
    Logs diagnostics and returns 0 if healthy, 1 if unhealthy.
    Delegates to medical_segmentation.diagnostics.run_health_check.
    """
    return _diagnostics_run_health_check(
        check_db=check_db,
        check_azure=check_azure,
        verify_azure_containers=verify_azure_containers,
        config=config,
        health_checker=health_checker or check_system_health,
    )


def main(args: Optional[Sequence[str]] = None) -> None:
    """CLI entrypoint."""
    parsed_args = parse_args(args)
    config_file = getattr(parsed_args, "config", None)
    active_config = config
    has_custom_config = False
    if isinstance(config_file, (str, Path)):
        active_config = PipelineConfig.from_file(config_file)
        has_custom_config = True

    raw_cli_args = sys.argv[1:] if args is None else list(args)
    explicit_flags = {
        arg.split("=")[0]
        for arg in raw_cli_args
        if arg.startswith("--")
    }

    log_level = getattr(parsed_args, "log_level", None)
    log_format = getattr(parsed_args, "log_format", None)
    log_file = getattr(parsed_args, "log_file", None)
    if has_custom_config:
        eff_log_level = parsed_args.log_level if "--log-level" in explicit_flags else active_config.log_level
        eff_log_format = parsed_args.log_format if "--log-format" in explicit_flags else active_config.log_format
        eff_log_file = parsed_args.log_file if "--log-file" in explicit_flags else active_config.log_file
    else:
        eff_log_level = log_level if isinstance(log_level, str) else active_config.log_level
        eff_log_format = log_format if isinstance(log_format, str) else active_config.log_format
        eff_log_file = log_file if isinstance(log_file, (str, Path)) else active_config.log_file

    if isinstance(eff_log_level, str) or isinstance(eff_log_format, str) or isinstance(eff_log_file, (str, Path)):
        configure_logging(
            level=eff_log_level,
            log_format=eff_log_format,
            log_file=eff_log_file,
        )

    if getattr(parsed_args, "check_health", None) is True:
        check_all_services = getattr(parsed_args, "check_all", None) is True
        check_db = check_all_services or getattr(parsed_args, "check_db", None) is True
        check_azure = check_all_services or getattr(parsed_args, "check_azure", None) is True

        kwargs: Dict[str, Any] = {"check_db": check_db}
        if check_azure:
            kwargs["check_azure"] = True
        if has_custom_config:
            kwargs["config"] = active_config

        sys.exit(run_health_check(**kwargs))

    if has_custom_config:
        input_container = parsed_args.container if "--container" in explicit_flags else active_config.azure.input_container
        output_container = parsed_args.output_container if "--output-container" in explicit_flags else active_config.azure.output_container
        modality = parsed_args.modality if "--modality" in explicit_flags else active_config.inference.modality
        orientation = parsed_args.orientation if "--orientation" in explicit_flags else active_config.inference.target_orientation
        checkpoint = parsed_args.checkpoint if "--checkpoint" in explicit_flags else active_config.inference.model_checkpoint_path
        batch_size = parsed_args.batch_size if "--batch-size" in explicit_flags else active_config.db.batch_size
    else:
        input_container = parsed_args.container
        output_container = parsed_args.output_container
        modality = parsed_args.modality
        orientation = parsed_args.orientation
        checkpoint = parsed_args.checkpoint
        batch_size = parsed_args.batch_size

    kwargs: Dict[str, Any] = {
        "blob_path": parsed_args.blob_path,
        "series_prefix": parsed_args.series_prefix,
        "local_input": parsed_args.local_input,
        "input_container": input_container,
        "output_container": output_container,
        "modality": modality,
        "orientation": orientation,
        "checkpoint": checkpoint,
        "batch_size": batch_size,
        "dry_run": parsed_args.dry_run,
    }
    if has_custom_config:
        kwargs["config"] = active_config
    sys.exit(run_pipeline(**kwargs))


if __name__ == "__main__":
    main()
