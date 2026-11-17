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

from medical_segmentation.config import DatabaseSettings, PipelineConfig
from medical_segmentation.db.database import DatabaseManager
from medical_segmentation.diagnostics import (
    SystemHealthChecker,
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

