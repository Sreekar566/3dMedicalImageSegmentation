"""
Unit and integration tests for system healthcheck diagnostic probes,
DatabaseManager connectivity verification, and CLI healthcheck entrypoints.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

from medical_segmentation.config import AzureStorageSettings, DatabaseSettings, PipelineConfig
from medical_segmentation.db.database import DatabaseManager
from medical_segmentation.diagnostics import (
    SystemHealthChecker,
    check_azure,
    check_database,
    check_monai,
    check_python,
    check_simpleitk,
    check_torch,
    check_workspace,
)
from medical_segmentation.main import (
    check_system_health,
    main,
    parse_args,
    run_health_check,
)


def test_database_manager_check_connection_success() -> None:
    """Verify DatabaseManager.check_connection returns True when engine is responsive."""
    sqlite_engine = create_engine("sqlite:///:memory:")
    settings = DatabaseSettings()
    db_mgr = DatabaseManager(settings=settings, engine=sqlite_engine)

    assert db_mgr.check_connection() is True


def test_database_manager_check_connection_failure() -> None:
    """Verify DatabaseManager.check_connection gracefully returns False on connection error."""
    mock_engine = MagicMock()
    mock_engine.connect.side_effect = OperationalError("connection refused", params=None, orig=Exception())
    settings = DatabaseSettings()
    db_mgr = DatabaseManager(settings=settings, engine=mock_engine)

    assert db_mgr.check_connection() is False


def test_check_system_health_healthy(tmp_path: Path) -> None:
    """Verify check_system_health verifies Python, PyTorch, SimpleITK, MONAI, and workspace."""
    cfg = PipelineConfig(work_dir=tmp_path / "healthcheck_workdir")
    report = check_system_health(check_db=False, config=cfg)

    assert report["status"] == "healthy"
    assert "timestamp" in report
    components = report["components"]

    assert components["python"]["status"] == "ok"
    assert components["torch"]["status"] == "ok"
    assert "version" in components["torch"]
    assert "device" in components["torch"]

    assert components["simpleitk"]["status"] == "ok"
    assert components["monai"]["status"] == "ok"
    assert components["monai"]["model_initialized"] is True

    assert components["workspace"]["status"] == "ok"
    assert components["workspace"]["writable"] is True
    assert "database" not in components


def test_check_system_health_with_database_success(tmp_path: Path) -> None:
    """Verify check_system_health includes database when check_db=True and database is reachable."""
    cfg = PipelineConfig(work_dir=tmp_path / "healthcheck_workdir")

    with patch("medical_segmentation.db.database.DatabaseManager.check_connection", return_value=True):
        report = check_system_health(check_db=True, config=cfg)

    assert report["status"] == "healthy"
    assert report["components"]["database"]["status"] == "ok"


def test_check_system_health_with_database_failure(tmp_path: Path) -> None:
    """Verify check_system_health marks system unhealthy when database check fails."""
    cfg = PipelineConfig(work_dir=tmp_path / "healthcheck_workdir")

    with patch("medical_segmentation.db.database.DatabaseManager.check_connection", return_value=False):
        report = check_system_health(check_db=True, config=cfg)

    assert report["status"] == "unhealthy"
    assert report["components"]["database"]["status"] == "error"


def test_check_system_health_component_failure(tmp_path: Path) -> None:
    """Verify check_system_health handles and reports component initialization exceptions."""
    cfg = PipelineConfig(work_dir=tmp_path / "healthcheck_workdir")

    with patch("medical_segmentation.pipeline.model.build_3d_unet", side_effect=RuntimeError("GPU OOM")):
        report = check_system_health(check_db=False, config=cfg)

    assert report["status"] == "unhealthy"
    assert report["components"]["monai"]["status"] == "error"
    assert "GPU OOM" in report["components"]["monai"]["error"]


def test_check_system_health_workspace_unwritable(tmp_path: Path) -> None:
    """Verify check_system_health catches unwritable workspace directory errors."""
    unwritable_dir = tmp_path / "read_only_dir"
    cfg = PipelineConfig(work_dir=unwritable_dir)

    with patch.object(Path, "mkdir", side_effect=PermissionError("Permission denied")):
        report = check_system_health(check_db=False, config=cfg)

    assert report["status"] == "unhealthy"
    assert report["components"]["workspace"]["status"] == "error"


def test_run_health_check_exit_codes(tmp_path: Path) -> None:
    """Verify run_health_check returns exit code 0 when healthy and 1 when unhealthy."""
    cfg = PipelineConfig(work_dir=tmp_path / "healthcheck_workdir")

    healthy_exit = run_health_check(check_db=False, config=cfg)
    assert healthy_exit == 0

    with patch("medical_segmentation.main.check_system_health", return_value={"status": "unhealthy", "components": {}}):
        unhealthy_exit = run_health_check(check_db=False, config=cfg)
        assert unhealthy_exit == 1


def test_cli_parse_args_healthcheck(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CLI argument parsing recognizes --check-health and --check-db flags."""
    monkeypatch.setattr("sys.argv", ["main.py", "--check-health"])
    args = parse_args()
    assert args.check_health is True
    assert args.check_db is False

    monkeypatch.setattr("sys.argv", ["main.py", "--check-health", "--check-db"])
    args_db = parse_args()
    assert args_db.check_health is True
    assert args_db.check_db is True


def test_cli_main_entrypoint_healthcheck(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify main() entrypoint triggers run_health_check and sys.exit when --check-health is supplied."""
    monkeypatch.setattr("sys.argv", ["main.py", "--check-health"])

    with patch("medical_segmentation.main.run_health_check", return_value=0) as mock_health:
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        mock_health.assert_called_once_with(check_db=False)


def test_diagnostics_individual_probes(tmp_path: Path) -> None:
    """Verify individual decoupled diagnostic probes execute and report status."""
    # 1. Python probe
    py_res = check_python()
    assert py_res["status"] == "ok"
    assert "version" in py_res

    # 2. PyTorch probe
    torch_res = check_torch("cpu")
    assert torch_res["status"] == "ok"
    assert torch_res["device"] == "cpu"
    assert "cuda_available" in torch_res

    torch_err = check_torch("invalid_device_name_xyz")
    assert torch_err["status"] == "error"
    assert "error" in torch_err

    # 3. SimpleITK probe
    sitk_res = check_simpleitk()
    assert sitk_res["status"] == "ok"
    assert "version" in sitk_res

    # 4. MONAI probe
    monai_res = check_monai()
    assert monai_res["status"] == "ok"
    assert monai_res["model_initialized"] is True

    # 5. Workspace probe
    ws_ok = check_workspace(tmp_path / "valid_dir")
    assert ws_ok["status"] == "ok"
    assert ws_ok["writable"] is True

    ws_fail = check_workspace(tmp_path / "read_only" / "\0invalid_path")
    assert ws_fail["status"] == "error"

    # 6. Database probe with mock
    with patch("medical_segmentation.db.database.DatabaseManager.check_connection", return_value=True):
        db_ok = check_database(DatabaseSettings())
        assert db_ok["status"] == "ok"

    with patch("medical_segmentation.db.database.DatabaseManager.check_connection", return_value=False):
        db_fail = check_database(DatabaseSettings())
        assert db_fail["status"] == "error"


def test_system_health_checker_class(tmp_path: Path) -> None:
    """Verify SystemHealthChecker orchestrates probes and produces aggregate report."""
    cfg = PipelineConfig(work_dir=tmp_path / "checker_workdir")
    checker = SystemHealthChecker(config=cfg)

    report = checker.check_all(check_db=False)
    assert report["status"] == "healthy"
    assert len(report["components"]) == 5
    assert "database" not in report["components"]

    with patch("medical_segmentation.diagnostics.check_monai", return_value={"status": "error", "error": "mock err"}):
        fail_report = checker.check_all(check_db=False)
        assert fail_report["status"] == "unhealthy"
        assert fail_report["components"]["monai"]["status"] == "error"


def test_check_database_disposes_engine() -> None:
    """Verify check_database cleans up its DatabaseManager connection pool via dispose()."""
    with patch("medical_segmentation.db.database.DatabaseManager.check_connection", return_value=True), \
         patch("medical_segmentation.db.database.DatabaseManager.dispose") as mock_dispose:
        result = check_database(DatabaseSettings())
        assert result["status"] == "ok"
        mock_dispose.assert_called_once()


def test_diagnostics_check_simpleitk_and_check_database_exceptions() -> None:
    """Verify check_simpleitk and check_database capture and report internal unhandled exceptions."""
    # 1. check_simpleitk error handling
    with patch("SimpleITK.Image", side_effect=RuntimeError("SimpleITK allocation failed")):
        sitk_res = check_simpleitk()
        assert sitk_res["status"] == "error"
        assert "SimpleITK allocation failed" in sitk_res["error"]

    # 2. check_database unhandled exception handling
    with patch("medical_segmentation.db.database.DatabaseManager", side_effect=RuntimeError("DatabaseManager crash")):
        db_res = check_database(DatabaseSettings())
        assert db_res["status"] == "error"
        assert "DatabaseManager crash" in db_res["error"]


def test_run_health_check_unhealthy_mixed_component_logging() -> None:
    """Verify run_health_check logs both passing and failing components when overall status is unhealthy."""
    mixed_results = {
        "status": "unhealthy",
        "timestamp": "2026-01-01T00:00:00Z",
        "components": {
            "python": {"status": "ok", "version": "3.12"},
            "monai": {"status": "error", "error": "Import error"},
        },
    }
    with patch("medical_segmentation.main.check_system_health", return_value=mixed_results):
        exit_code = run_health_check()
        assert exit_code == 1


def test_diagnostics_namespace_registration_bootstrap() -> None:
    """Verify diagnostics.py bootstraps 'medical_segmentation' into sys.modules when absent."""
    import runpy
    import sys
    saved = sys.modules.pop("medical_segmentation", None)
    try:
        assert "medical_segmentation" not in sys.modules
        diag_path = Path(__file__).resolve().parent.parent / "diagnostics.py"
        runpy.run_path(str(diag_path))
        assert "medical_segmentation" in sys.modules
    finally:
        if saved is not None:
            sys.modules["medical_segmentation"] = saved


def test_check_workspace_failure_before_probe_creation() -> None:
    """Verify check_workspace gracefully handles write failure where probe file was never created."""
    with patch("pathlib.Path.write_text", side_effect=PermissionError("Permission denied")):
        res = check_workspace("/tmp/test_probe_cleanup_fail")
        assert res["status"] == "error"
        assert "Permission denied" in res["error"]


def test_check_workspace_probe_unlink_skipped_when_probe_missing() -> None:
    """Verify check_workspace cleanly completes without unlink if probe_file.exists() returns False."""
    with patch.object(Path, "exists", return_value=False):
        res = check_workspace("/tmp")
        assert res["status"] == "ok"


def test_check_azure_offline_mode() -> None:
    """Verify check_azure reports error when no Azure credentials are configured."""
    settings = AzureStorageSettings(connection_string="", account_name="", account_key="")
    result = check_azure(settings=settings)
    assert result["status"] == "error"
    assert "No Azure Storage credentials configured" in result["error"]


def test_check_azure_success_connection_string() -> None:
    """Verify check_azure verifies connectivity using connection string and reports account info."""
    mock_blob_client = MagicMock()
    mock_blob_client.get_account_information.return_value = {
        "sku_name": "Standard_RAGRS",
        "account_kind": "StorageV2",
    }

    settings = AzureStorageSettings(
        connection_string="DefaultEndpointsProtocol=https;AccountName=testmedstorage;AccountKey=dGVzdGtleQ==;EndpointSuffix=core.windows.net",
        input_container="test-in",
        output_container="test-out",
    )

    with patch("azure.storage.blob.BlobServiceClient.from_connection_string", return_value=mock_blob_client):
        result = check_azure(settings=settings, verify_containers=False)

    assert result["status"] == "ok"
    assert result["sku_name"] == "Standard_RAGRS"
    assert result["account_kind"] == "StorageV2"
    assert result["input_container"] == "test-in"
    assert result["output_container"] == "test-out"


def test_check_azure_success_account_key() -> None:
    """Verify check_azure verifies connectivity using account name & key with object account info."""
    class AccountInfo:
        sku_name = "Premium_LRS"
        account_kind = "BlockBlobStorage"

    mock_blob_client = MagicMock()
    mock_blob_client.get_account_information.return_value = AccountInfo()

    settings = AzureStorageSettings(
        account_name="testmedstorage",
        account_key="dGVzdGtleQ==",
        input_container="test-in",
        output_container="test-out",
    )

    with patch("azure.storage.blob.BlobServiceClient.__init__", return_value=None), \
         patch.object(mock_blob_client, "get_account_information", return_value=AccountInfo()):
        with patch("medical_segmentation.io.azure_client.BlobServiceClient", return_value=mock_blob_client):
            result = check_azure(settings=settings, verify_containers=False)

    assert result["status"] == "ok"
    assert result["account_name"] == "testmedstorage"
    assert result["sku_name"] == "Premium_LRS"
    assert result["account_kind"] == "BlockBlobStorage"


def test_check_azure_with_verify_containers_success() -> None:
    """Verify check_azure confirms container presence when verify_containers=True."""
    mock_blob_client = MagicMock()
    mock_blob_client.get_account_information.return_value = {"sku_name": "Standard_LRS", "account_kind": "StorageV2"}
    mock_container_client = MagicMock()
    mock_container_client.exists.return_value = True
    mock_blob_client.get_container_client.return_value = mock_container_client

    settings = AzureStorageSettings(
        connection_string="DefaultEndpointsProtocol=https;AccountName=testmedstorage;AccountKey=dGVzdGtleQ==;EndpointSuffix=core.windows.net",
        input_container="test-in",
        output_container="test-out",
    )

    with patch("azure.storage.blob.BlobServiceClient.from_connection_string", return_value=mock_blob_client):
        result = check_azure(settings=settings, verify_containers=True)

    assert result["status"] == "ok"
    assert result["containers"] == {"test-in": True, "test-out": True}


def test_check_azure_with_verify_containers_missing() -> None:
    """Verify check_azure reports error when configured containers do not exist."""
    mock_blob_client = MagicMock()
    mock_blob_client.get_account_information.return_value = {"sku_name": "Standard_LRS", "account_kind": "StorageV2"}

    def get_container(name: str):
        c = MagicMock()
        c.exists.return_value = name == "test-in"
        return c

    mock_blob_client.get_container_client.side_effect = get_container

    settings = AzureStorageSettings(
        connection_string="DefaultEndpointsProtocol=https;AccountName=testmedstorage;AccountKey=dGVzdGtleQ==;EndpointSuffix=core.windows.net",
        input_container="test-in",
        output_container="missing-out",
    )

    with patch("azure.storage.blob.BlobServiceClient.from_connection_string", return_value=mock_blob_client):
        result = check_azure(settings=settings, verify_containers=True)

    assert result["status"] == "error"
    assert "missing-out" in result["error"]
    assert result["containers"]["test-in"] is True
    assert result["containers"]["missing-out"] is False


def test_check_azure_connection_exception() -> None:
    """Verify check_azure captures connection exceptions and returns structured error dict."""
    settings = AzureStorageSettings(
        connection_string="DefaultEndpointsProtocol=https;AccountName=testmedstorage;AccountKey=dGVzdGtleQ==;EndpointSuffix=core.windows.net",
    )

    mock_blob_client = MagicMock()
    mock_blob_client.get_account_information.side_effect = ConnectionError("Azure host unreachable")

    with patch("azure.storage.blob.BlobServiceClient.from_connection_string", return_value=mock_blob_client):
        result = check_azure(settings=settings)

    assert result["status"] == "error"
    assert "Azure host unreachable" in result["error"]


def test_check_azure_client_none_edge_case() -> None:
    """Verify check_azure handles edge case where AzureBlobClient._client is None."""
    settings = AzureStorageSettings(
        connection_string="DefaultEndpointsProtocol=https;AccountName=test;AccountKey=key;EndpointSuffix=core.windows.net",
    )
    with patch("medical_segmentation.io.azure_client.AzureBlobClient.is_connected", return_value=True), \
         patch("medical_segmentation.io.azure_client.AzureBlobClient._initialize_client"):
        result = check_azure(settings=settings)
    assert result["status"] == "error"
    assert "Failed to initialize Azure BlobServiceClient" in result["error"]


def test_check_system_health_with_azure_success(tmp_path: Path) -> None:
    """Verify check_system_health includes Azure probe and passes when check_azure=True."""
    cfg = PipelineConfig(work_dir=tmp_path / "healthcheck_workdir")

    with patch("medical_segmentation.diagnostics.check_azure", return_value={"status": "ok", "account_name": "medazure"}):
        report = check_system_health(check_db=False, check_azure=True, config=cfg)

    assert report["status"] == "healthy"
    assert "azure" in report["components"]
    assert report["components"]["azure"]["status"] == "ok"
    assert "database" not in report["components"]


def test_check_system_health_with_azure_failure(tmp_path: Path) -> None:
    """Verify check_system_health marks system unhealthy when Azure probe fails."""
    cfg = PipelineConfig(work_dir=tmp_path / "healthcheck_workdir")

    with patch("medical_segmentation.diagnostics.check_azure", return_value={"status": "error", "error": "Auth failed"}):
        report = check_system_health(check_db=False, check_azure=True, config=cfg)

    assert report["status"] == "unhealthy"
    assert report["components"]["azure"]["status"] == "error"


def test_check_system_health_with_check_all_services(tmp_path: Path) -> None:
    """Verify check_system_health executes both database and azure probes when both flags are active."""
    cfg = PipelineConfig(work_dir=tmp_path / "healthcheck_workdir")

    with patch("medical_segmentation.db.database.DatabaseManager.check_connection", return_value=True), \
         patch("medical_segmentation.diagnostics.check_azure", return_value={"status": "ok", "account_name": "medazure"}):
        report = check_system_health(check_db=True, check_azure=True, config=cfg)

    assert report["status"] == "healthy"
    assert len(report["components"]) == 7
    assert report["components"]["database"]["status"] == "ok"
    assert report["components"]["azure"]["status"] == "ok"


def test_cli_parse_args_check_azure_and_check_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify CLI argument parsing recognizes --check-azure and --check-all flags."""
    monkeypatch.setattr("sys.argv", ["main.py", "--check-health", "--check-azure"])
    args_azure = parse_args()
    assert args_azure.check_health is True
    assert args_azure.check_azure is True
    assert args_azure.check_db is False
    assert args_azure.check_all is False

    monkeypatch.setattr("sys.argv", ["main.py", "--check-health", "--check-all"])
    args_all = parse_args()
    assert args_all.check_health is True
    assert args_all.check_all is True


def test_cli_main_entrypoint_check_azure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify main() entrypoint triggers run_health_check with check_azure=True."""
    monkeypatch.setattr("sys.argv", ["main.py", "--check-health", "--check-azure"])
    with patch("medical_segmentation.main.run_health_check", return_value=0) as mock_health:
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        mock_health.assert_called_once_with(check_db=False, check_azure=True)


def test_cli_main_entrypoint_check_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify main() entrypoint triggers run_health_check with check_db=True and check_azure=True."""
    monkeypatch.setattr("sys.argv", ["main.py", "--check-health", "--check-all"])
    with patch("medical_segmentation.main.run_health_check", return_value=0) as mock_health:
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        mock_health.assert_called_once_with(check_db=True, check_azure=True)


def test_run_health_check_with_azure_exit_codes(tmp_path: Path) -> None:
    """Verify run_health_check returns 0 when azure is healthy and 1 when unhealthy."""
    cfg = PipelineConfig(work_dir=tmp_path / "healthcheck_workdir")

    with patch("medical_segmentation.diagnostics.check_azure", return_value={"status": "ok"}):
        healthy_exit = run_health_check(check_db=False, check_azure=True, config=cfg)
        assert healthy_exit == 0

    with patch("medical_segmentation.diagnostics.check_azure", return_value={"status": "error", "error": "Azure unreachable"}):
        unhealthy_exit = run_health_check(check_db=False, check_azure=True, config=cfg)
        assert unhealthy_exit == 1


def test_run_health_check_typeerror_fallback_handling() -> None:
    """Verify run_health_check catches TypeError when health_checker does not accept check_azure."""
    def legacy_health_checker(check_db: bool = False, config=None):
        return {"status": "healthy", "components": {"db": {"status": "ok"}}}

    exit_code = run_health_check(check_db=True, check_azure=True, health_checker=legacy_health_checker)
    assert exit_code == 0
