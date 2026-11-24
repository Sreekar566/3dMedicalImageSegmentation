"""
System and Dependency Healthcheck Diagnostics Engine for 3dMedicalImageSegmentation.
Provides isolated, single-responsibility diagnostic probes for Python, PyTorch compute devices,
SimpleITK image runtime, MONAI model builder, local workspace storage, and PostgreSQL database connectivity.
"""

from __future__ import annotations

import logging
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Ensure 'medical_segmentation' module namespace is resolvable
if "medical_segmentation" not in sys.modules:
    med_pkg = types.ModuleType("medical_segmentation")
    med_pkg.__path__ = [str(Path(__file__).resolve().parent)]
    sys.modules["medical_segmentation"] = med_pkg

from medical_segmentation.config import AzureStorageSettings, DatabaseSettings, PipelineConfig

logger = logging.getLogger("3dMedicalImageSegmentation.diagnostics")

__all__ = [
    "SystemHealthChecker",
    "check_azure",
    "check_database",
    "check_monai",
    "check_python",
    "check_simpleitk",
    "check_system_health",
    "check_torch",
    "check_workspace",
    "run_health_check",
]


def check_python() -> Dict[str, Any]:
    """Verify Python runtime availability and version."""
    py_version = sys.version.split()[0]
    return {
        "status": "ok",
        "version": py_version,
    }


def check_torch(device_name: str = "cpu") -> Dict[str, Any]:
    """Verify PyTorch runtime, target execution device allocation, and CUDA availability."""
    try:
        import torch

        device = torch.device(device_name)
        _ = torch.zeros((1, 1, 4, 4, 4), device=device)
        return {
            "status": "ok",
            "version": torch.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
        }


def check_simpleitk() -> Dict[str, Any]:
    """Verify SimpleITK runtime, version, and volumetric image allocation."""
    try:
        import SimpleITK as sitk

        _ = sitk.Image(4, 4, 4, sitk.sitkFloat32)
        return {
            "status": "ok",
            "version": sitk.Version().VersionString(),
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
        }


def check_monai() -> Dict[str, Any]:
    """Verify MONAI library runtime and 3D U-Net neural network architecture initialization."""
    try:
        import monai
        from medical_segmentation.pipeline.model import build_3d_unet

        net = build_3d_unet(in_channels=1, out_channels=2)
        return {
            "status": "ok",
            "version": monai.__version__,
            "model_initialized": net is not None,
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
        }


def check_workspace(work_dir: Path | str) -> Dict[str, Any]:
    """Verify local filesystem workspace directory existence, permissions, and writability."""
    try:
        target_dir = Path(work_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        probe_file = target_dir / f".healthcheck_probe_{int(time.time() * 1000)}"
        try:
            probe_file.write_text("ok", encoding="utf-8")
        finally:
            if probe_file.exists():
                probe_file.unlink()
        return {
            "status": "ok",
            "path": str(target_dir),
            "writable": True,
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
        }


def check_database(settings: Optional[DatabaseSettings] = None) -> Dict[str, Any]:
    """Verify PostgreSQL database connectivity and responsiveness via connection pooling."""
    db_mgr: Optional[Any] = None
    try:
        from medical_segmentation.config import DatabaseSettings as DefaultDbSettings
        from medical_segmentation.db.database import DatabaseManager

        db_settings = settings or DefaultDbSettings()
        db_mgr = DatabaseManager(settings=db_settings)
        db_healthy = db_mgr.check_connection()
        if db_healthy:
            return {
                "status": "ok",
                "host": db_settings.host,
                "database": db_settings.name,
            }
        else:
            return {
                "status": "error",
                "error": "Failed to connect to PostgreSQL database.",
            }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
        }
    finally:
        if db_mgr is not None and hasattr(db_mgr, "dispose"):
            db_mgr.dispose()


def check_azure(
    settings: Optional[AzureStorageSettings] = None,
    verify_containers: bool = False,
) -> Dict[str, Any]:
    """
    Verify Azure Blob Storage connectivity, credentials, and optional container accessibility.

    Args:
        settings: Optional AzureStorageSettings instance.
        verify_containers: If True, also checks that input and output containers exist.

    Returns:
        Dictionary detailing probe status ('ok' or 'error') and metadata.
    """
    try:
        from medical_segmentation.config import AzureStorageSettings as DefaultAzureSettings
        from medical_segmentation.io.azure_client import AzureBlobClient

        azure_settings = settings or DefaultAzureSettings()
        azure_client = AzureBlobClient(settings=azure_settings)

        if not azure_client.is_connected:
            return {
                "status": "error",
                "error": "No Azure Storage credentials configured (AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_NAME/KEY required).",
            }

        client = azure_client._client
        if client is None:
            return {
                "status": "error",
                "error": "Failed to initialize Azure BlobServiceClient.",
            }

        account_info = client.get_account_information()
        account_kind = (
            account_info.get("account_kind")
            if isinstance(account_info, dict)
            else getattr(account_info, "account_kind", None)
        )
        sku_name = (
            account_info.get("sku_name")
            if isinstance(account_info, dict)
            else getattr(account_info, "sku_name", None)
        )

        details: Dict[str, Any] = {
            "status": "ok",
            "account_name": azure_settings.account_name or "connection_string",
            "account_kind": account_kind,
            "sku_name": sku_name,
            "input_container": azure_settings.input_container,
            "output_container": azure_settings.output_container,
        }

        if verify_containers:
            container_status: Dict[str, bool] = {}
            missing_containers = []
            for c_name in (azure_settings.input_container, azure_settings.output_container):
                if c_name:
                    c_client = client.get_container_client(c_name)
                    exists = bool(c_client.exists())
                    container_status[c_name] = exists
                    if not exists:
                        missing_containers.append(c_name)

            details["containers"] = container_status
            if missing_containers:
                return {
                    "status": "error",
                    "error": f"Azure container(s) not found or inaccessible: {', '.join(missing_containers)}",
                    "containers": container_status,
                }

        return details
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
        }


class SystemHealthChecker:
    """Orchestrates diagnostic health probes across subsystems."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config

    def check_all(
        self,
        check_db: bool = False,
        check_azure: bool = False,
        verify_azure_containers: bool = False,
    ) -> Dict[str, Any]:
        """Execute all subsystem probes and aggregate overall health status."""
        from medical_segmentation.config import config as default_config

        active_config = self.config or default_config

        results: Dict[str, Any] = {
            "status": "healthy",
            "timestamp": time.time(),
            "components": {},
        }

        # 1. Python runtime
        results["components"]["python"] = check_python()

        # 2. PyTorch & Compute device
        results["components"]["torch"] = check_torch(active_config.inference.device)

        # 3. SimpleITK runtime
        results["components"]["simpleitk"] = check_simpleitk()

        # 4. MONAI runtime
        results["components"]["monai"] = check_monai()

        # 5. Local Workspace Staging
        results["components"]["workspace"] = check_workspace(active_config.work_dir)

        # 6. Database (optional or requested)
        if check_db:
            results["components"]["database"] = check_database(active_config.db)

        # 7. Azure Blob Storage (optional or requested)
        if check_azure:
            azure_probe_fn = globals()["check_azure"]
            results["components"]["azure"] = azure_probe_fn(
                active_config.azure,
                verify_containers=verify_azure_containers,
            )

        # Determine overall system health
        for comp_data in results["components"].values():
            if comp_data.get("status") != "ok":
                results["status"] = "unhealthy"
                break

        return results


def check_system_health(
    check_db: bool = False,
    check_azure: bool = False,
    verify_azure_containers: bool = False,
    config: Optional[PipelineConfig] = None,
) -> Dict[str, Any]:
    """
    Perform diagnostic health checks across runtime dependencies and storage.
    Verifies Python, PyTorch, SimpleITK, MONAI, workspace storage, and optionally PostgreSQL and Azure Blob Storage.

    Returns:
        Dictionary detailing component statuses and overall health flag ('status': 'healthy' | 'unhealthy').
    """
    checker = SystemHealthChecker(config=config)
    return checker.check_all(
        check_db=check_db,
        check_azure=check_azure,
        verify_azure_containers=verify_azure_containers,
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

    Args:
        check_db: Whether to include PostgreSQL database verification.
        check_azure: Whether to include Azure Blob Storage verification.
        verify_azure_containers: Whether to verify input/output container existence during Azure check.
        config: Optional PipelineConfig instance.
        health_checker: Optional callable overriding check_system_health (useful for dependency injection and mocking).

    Returns:
        Exit code: 0 on healthy, 1 on unhealthy.
    """
    logger.info(
        "Executing pipeline system health check (check_db=%s, check_azure=%s)...",
        check_db,
        check_azure,
    )
    checker_fn = health_checker or check_system_health
    try:
        results = checker_fn(
            check_db=check_db,
            check_azure=check_azure,
            verify_azure_containers=verify_azure_containers,
            config=config,
        )
    except TypeError:
        results = checker_fn(check_db=check_db, config=config)
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
