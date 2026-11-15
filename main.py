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
from typing import Any, Dict, Optional

# Ensure 'medical_segmentation' module namespace is resolvable
if "medical_segmentation" not in sys.modules:
    med_pkg = types.ModuleType("medical_segmentation")
    med_pkg.__path__ = [str(Path(__file__).resolve().parent)]
    sys.modules["medical_segmentation"] = med_pkg

from medical_segmentation.config import PipelineConfig, config, validate_orientation_code
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
    input_group.add_argument(
        "--check-health",
        action="store_true",
        help="Perform diagnostic system and dependency health check and exit.",
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
        if batch_size <= 0:
            raise ValueError(f"batch_size must be a strictly positive integer, got {batch_size}.")
        orientation = validate_orientation_code(orientation)
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


def check_system_health(
    check_db: bool = False,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    """
    Perform diagnostic health checks across runtime dependencies and storage.
    Verifies Python, PyTorch, SimpleITK, MONAI, workspace storage, and optionally PostgreSQL.

    Returns:
        Dictionary detailing component statuses and overall health flag ('status': 'healthy' | 'unhealthy').
    """
    from medical_segmentation.config import config as default_config

    active_config = config or default_config

    results: Dict[str, Any] = {
        "status": "healthy",
        "timestamp": time.time(),
        "components": {},
    }

    # 1. Python runtime
    py_version = sys.version.split()[0]
    results["components"]["python"] = {
        "status": "ok",
        "version": py_version,
    }

    # 2. PyTorch & Compute device
    try:
        import torch

        device_name = active_config.inference.device
        device = torch.device(device_name)
        _ = torch.zeros((1, 1, 4, 4, 4), device=device)
        results["components"]["torch"] = {
            "status": "ok",
            "version": torch.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
        }
    except Exception as exc:
        results["status"] = "unhealthy"
        results["components"]["torch"] = {
            "status": "error",
            "error": str(exc),
        }

    # 3. SimpleITK runtime
    try:
        import SimpleITK as sitk

        _ = sitk.Image(4, 4, 4, sitk.sitkFloat32)
        results["components"]["simpleitk"] = {
            "status": "ok",
            "version": sitk.Version().VersionString(),
        }
    except Exception as exc:
        results["status"] = "unhealthy"
        results["components"]["simpleitk"] = {
            "status": "error",
            "error": str(exc),
        }

    # 4. MONAI runtime
    try:
        import monai
        from medical_segmentation.pipeline.model import build_3d_unet

        net = build_3d_unet(in_channels=1, out_channels=2)
        results["components"]["monai"] = {
            "status": "ok",
            "version": monai.__version__,
            "model_initialized": net is not None,
        }
    except Exception as exc:
        results["status"] = "unhealthy"
        results["components"]["monai"] = {
            "status": "error",
            "error": str(exc),
        }

    # 5. Local Workspace Staging
    try:
        work_dir = Path(active_config.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        probe_file = work_dir / f".healthcheck_probe_{int(time.time() * 1000)}"
        probe_file.write_text("ok", encoding="utf-8")
        probe_file.unlink()
        results["components"]["workspace"] = {
            "status": "ok",
            "path": str(work_dir),
            "writable": True,
        }
    except Exception as exc:
        results["status"] = "unhealthy"
        results["components"]["workspace"] = {
            "status": "error",
            "error": str(exc),
        }

    # 6. Database (optional or requested)
    if check_db:
        try:
            from medical_segmentation.db.database import DatabaseManager

            db_mgr = DatabaseManager(settings=active_config.db)
            db_healthy = db_mgr.check_connection()
            if db_healthy:
                results["components"]["database"] = {
                    "status": "ok",
                    "host": active_config.db.host,
                    "database": active_config.db.name,
                }
            else:
                results["status"] = "unhealthy"
                results["components"]["database"] = {
                    "status": "error",
                    "error": "Failed to connect to PostgreSQL database.",
                }
        except Exception as exc:
            results["status"] = "unhealthy"
            results["components"]["database"] = {
                "status": "error",
                "error": str(exc),
            }

    return results


def run_health_check(
    check_db: bool = False,
    config: Optional[PipelineConfig] = None,
) -> int:
    """
    CLI/probe wrapper for check_system_health.
    Logs diagnostics and returns 0 if healthy, 1 if unhealthy.
    """
    logger.info("Executing pipeline system health check (check_db=%s)...", check_db)
    results = check_system_health(check_db=check_db, config=config)
    status = results.get("status", "unknown")
    if status == "healthy":
        logger.info("[HEALTH] System status: HEALTHY. All checked components operational.")
        for comp, details in results.get("components", {}).items():
            logger.info(" - Component [%s]: %s", comp, details.get("status"))
        return 0
    else:
        logger.error("[HEALTH] System status: UNHEALTHY. One or more components failed.")
        for comp, details in results.get("components", {}).items():
            comp_status = details.get("status")
            if comp_status != "ok":
                logger.error(
                    " - Component [%s]: %s (error: %s)", comp, comp_status, details.get("error")
                )
            else:
                logger.info(" - Component [%s]: %s", comp, comp_status)
        return 1


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    if getattr(args, "check_health", None) is True:
        check_db = getattr(args, "check_db", None) is True
        sys.exit(run_health_check(check_db=check_db))
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
