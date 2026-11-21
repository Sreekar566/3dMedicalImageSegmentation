"""
Unit and integration tests for Structured JSON Logging, StandardTextFormatter,
configure_logging handler orchestration, and CLI logging arguments.
"""

from __future__ import annotations

import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from medical_segmentation.config import PipelineConfig
from medical_segmentation.logging_config import (
    StandardTextFormatter,
    StructuredJsonFormatter,
    _safe_json_serializer,
    configure_logging,
    get_logger,
)
from medical_segmentation.main import main, parse_args


def test_structured_json_formatter_basic() -> None:
    """Verify that StructuredJsonFormatter emits valid JSON containing standard fields."""
    formatter = StructuredJsonFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=42,
        msg="Processing volumetric DICOM scan %s",
        args=("series_001",),
        exc_info=None,
    )

    output = formatter.format(record)
    data = json.loads(output)

    assert data["level"] == "INFO"
    assert data["logger"] == "test_logger"
    assert data["message"] == "Processing volumetric DICOM scan series_001"
    assert data["line"] == 42
    assert "timestamp" in data
    assert data["timestamp"].endswith("Z")
    assert "process" in data
    assert "thread" in data
    assert "module" in data
    assert "function" in data
    assert "extra" not in data
    assert "exception" not in data


def test_structured_json_formatter_with_extra_context() -> None:
    """Verify that custom contextual attributes in extra are included in JSON output."""
    formatter = StructuredJsonFormatter()
    record = logging.LogRecord(
        name="medseg.pipeline",
        level=logging.WARNING,
        pathname=__file__,
        lineno=100,
        msg="Anatomical reorientation applied",
        args=(),
        exc_info=None,
    )
    record.study_uid = "1.2.840.113619"
    record.orientation = "RAS"
    record.voxel_spacing = (1.5, 1.5, 2.0)

    output = formatter.format(record)
    data = json.loads(output)

    assert data["level"] == "WARNING"
    assert "extra" in data
    assert data["extra"]["study_uid"] == "1.2.840.113619"
    assert data["extra"]["orientation"] == "RAS"
    assert data["extra"]["voxel_spacing"] == [1.5, 1.5, 2.0]


def test_structured_json_formatter_with_exception() -> None:
    """Verify that exception tracebacks are captured in a structured exception object."""
    formatter = StructuredJsonFormatter()
    try:
        raise ValueError("Invalid slice dimensions: expected 3D, got 2D")
    except ValueError:
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="medseg.error",
        level=logging.ERROR,
        pathname=__file__,
        lineno=120,
        msg="Segmentation pipeline failed",
        args=(),
        exc_info=exc_info,
    )

    output = formatter.format(record)
    # Output must be a single line of valid JSON
    assert "\n" not in output
    data = json.loads(output)

    assert data["level"] == "ERROR"
    assert "exception" in data
    assert data["exception"]["type"] == "ValueError"
    assert "Invalid slice dimensions" in data["exception"]["message"]
    assert "Traceback" in data["exception"]["traceback"]


def test_structured_json_safe_serializer_edge_cases(tmp_path: Path) -> None:
    """Verify _safe_json_serializer handles non-standard types safely."""
    now = datetime.datetime.now(datetime.timezone.utc)
    assert _safe_json_serializer(now) == now.isoformat()

    p = tmp_path / "volume.nii"
    assert _safe_json_serializer(p) == str(p)

    s = {"b", "a"}
    assert _safe_json_serializer(s) == ["a", "b"]

    class MockObject:
        def __init__(self) -> None:
            self.alpha = "one"
            self.beta = 2

    mock_obj = MockObject()
    serialized = _safe_json_serializer(mock_obj)
    assert isinstance(serialized, dict)
    assert serialized["alpha"] == "one"

    class UnserializableObject:
        def __str__(self) -> str:
            return "unserializable_custom_str"

    assert _safe_json_serializer(UnserializableObject()) == "unserializable_custom_str"


def test_standard_text_formatter_with_and_without_extra() -> None:
    """Verify StandardTextFormatter produces human-readable output and appends extra fields."""
    formatter = StandardTextFormatter()

    # Without extra
    record = logging.LogRecord(
        name="test_text",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Simple text message",
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    assert "[INFO] [test_text]: Simple text message" in formatted
    assert "[" not in formatted.split("]: ")[1]

    # With extra
    record.patient_id = "PATIENT_999"
    record.duration_sec = 3.14
    formatted_extra = formatter.format(record)
    assert "Simple text message [duration_sec=3.14, patient_id=PATIENT_999]" in formatted_extra


def test_configure_logging_stream_handler(capsys: pytest.CaptureFixture) -> None:
    """Verify configure_logging attaches stdout stream handler with specified format."""
    logger_name = "test_stream_config_logger"
    logger = configure_logging(
        level="DEBUG",
        log_format="json",
        root_logger_name=logger_name,
        force=True,
    )

    logger.debug("Debug probe message", extra={"metric": 42})
    captured = capsys.readouterr().out
    assert captured.strip() != ""

    parsed = json.loads(captured.strip())
    assert parsed["level"] == "DEBUG"
    assert parsed["message"] == "Debug probe message"
    assert parsed["extra"]["metric"] == 42


def test_configure_logging_file_handler(tmp_path: Path) -> None:
    """Verify configure_logging creates log directory and writes entries to file."""
    log_file = tmp_path / "subfolder" / "pipeline.log"
    logger_name = "test_file_config_logger"

    logger = configure_logging(
        level="INFO",
        log_format="text",
        log_file=log_file,
        root_logger_name=logger_name,
        force=True,
    )

    logger.info("File persistence test entry")

    # Flush handlers
    for h in logger.handlers:
        h.flush()

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "[INFO] [test_file_config_logger]: File persistence test entry" in content


def test_configure_logging_log_levels() -> None:
    """Verify configure_logging accepts string names, WARN alias, and integers."""
    logger = configure_logging(level="WARN", root_logger_name="test_warn_logger")
    assert logger.level == logging.WARNING

    logger_int = configure_logging(level=logging.ERROR, root_logger_name="test_int_logger")
    assert logger_int.level == logging.ERROR


def test_get_logger() -> None:
    """Verify get_logger returns logger with requested name."""
    log = get_logger("custom.component")
    assert log.name == "custom.component"


def test_pipeline_config_logging_fields(tmp_path: Path) -> None:
    """Verify PipelineConfig validates log_format and log_file."""
    # Default values
    cfg = PipelineConfig()
    assert cfg.log_format == "text"
    assert cfg.log_file is None

    # Custom valid values
    log_path = tmp_path / "logs" / "run.log"
    cfg_custom = PipelineConfig(log_format="JSON", log_file=str(log_path))
    assert cfg_custom.log_format == "json"
    assert cfg_custom.log_file == log_path

    # Invalid log_format
    with pytest.raises(ValueError, match="Invalid log_format 'xml'"):
        PipelineConfig(log_format="xml")

    # Invalid type
    with pytest.raises(ValidationError):
        PipelineConfig(log_format=123)  # type: ignore[arg-type]


def test_cli_parse_args_logging_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify parse_args supports --log-format, --log-level, and --log-file."""
    log_dest = tmp_path / "app.log"
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--local-input",
            str(tmp_path / "scan.nii"),
            "--log-format",
            "json",
            "--log-level",
            "DEBUG",
            "--log-file",
            str(log_dest),
        ],
    )
    args = parse_args()
    assert args.log_format == "json"
    assert args.log_level == "DEBUG"
    assert args.log_file == log_dest


def test_cli_main_reconfigures_logging(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify main() entrypoint triggers configure_logging with CLI-specified options."""
    log_dest = tmp_path / "cli_run.log"
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--check-health",
            "--log-format",
            "json",
            "--log-level",
            "DEBUG",
            "--log-file",
            str(log_dest),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    assert log_dest.exists()
    content = log_dest.read_text(encoding="utf-8")
    assert content.strip() != ""
    first_line = content.strip().splitlines()[0]
    parsed_record = json.loads(first_line)
    assert "timestamp" in parsed_record
    assert parsed_record["level"] in ("INFO", "DEBUG")


def test_safe_json_serializer_numpy_and_torch_types() -> None:
    """Verify _safe_json_serializer correctly serializes numpy scalars, arrays, torch tensors, and devices."""
    # Numpy scalars
    assert _safe_json_serializer(np.int32(42)) == 42
    assert _safe_json_serializer(np.int64(100)) == 100
    assert _safe_json_serializer(np.float32(3.14)) == pytest.approx(3.14, rel=1e-3)
    assert _safe_json_serializer(np.float64(2.718)) == pytest.approx(2.718, rel=1e-3)

    # Numpy arrays: small array (<= 20 elements) converts to list
    small_arr = np.array([1, 2, 3, 4])
    assert _safe_json_serializer(small_arr) == [1, 2, 3, 4]

    # Numpy arrays: large array (> 20 elements) converts to descriptor string
    large_arr = np.zeros((5, 5))
    serialized_large = _safe_json_serializer(large_arr)
    assert "<ndarray shape=(5, 5) dtype=float64>" in serialized_large

    # PyTorch device and tensor
    device = torch.device("cpu")
    assert _safe_json_serializer(device) == "cpu"

    tensor = torch.zeros((1, 2, 3), dtype=torch.float32)
    serialized_tensor = _safe_json_serializer(tensor)
    assert "<Tensor shape=(1, 2, 3) dtype=torch.float32>" == serialized_tensor

    # Fallback when item() raises an exception
    class int_scalar:
        def item(self):
            raise RuntimeError("item error")

        def __str__(self):
            return "fallback_scalar_str"

    assert _safe_json_serializer(int_scalar()) == "fallback_scalar_str"


def test_structured_json_formatter_stack_info_and_message_exception() -> None:
    """Verify StructuredJsonFormatter handles stack_info and formatting errors in record.msg."""
    formatter = StructuredJsonFormatter()

    # Stack info handling
    record_stack = logging.LogRecord(
        name="test_stack",
        level=logging.ERROR,
        pathname=__file__,
        lineno=50,
        msg="Stack dump test",
        args=(),
        exc_info=None,
    )
    record_stack.stack_info = "Traceback (most recent call last):\n  File 'test.py', line 1"
    output_stack = formatter.format(record_stack)
    data_stack = json.loads(output_stack)
    assert "stack_info" in data_stack
    assert "Traceback" in data_stack["stack_info"]

    # Formatting error in getMessage()
    record_bad_args = logging.LogRecord(
        name="test_bad_msg",
        level=logging.WARNING,
        pathname=__file__,
        lineno=60,
        msg="Expected %d integers but got %d",
        args=("only_one",),  # Type/count mismatch causes record.getMessage() to raise
        exc_info=None,
    )
    output_bad_args = formatter.format(record_bad_args)
    data_bad_args = json.loads(output_bad_args)
    assert data_bad_args["message"] == "Expected %d integers but got %d"


def test_configure_logging_error_resilience(tmp_path: Path) -> None:
    """Verify configure_logging gracefully handles handler close failures and unwritable file logs."""
    logger_name = "test_resilience_logger"
    logger = logging.getLogger(logger_name)

    # 1. Existing handler that fails on close()
    class BrokenHandler(logging.Handler):
        def close(self) -> None:
            raise RuntimeError("Close failed")

    broken_h = BrokenHandler()
    logger.addHandler(broken_h)

    # force=True should remove and attempt to close handlers without failing
    reconfigured = configure_logging(
        level="INFO",
        log_format="text",
        root_logger_name=logger_name,
        force=True,
    )
    assert reconfigured.name == logger_name

    # 2. File handler initialization failure (e.g. uncreatable path)
    with patch("pathlib.Path.mkdir", side_effect=PermissionError("Permission denied")):
        resilient_logger = configure_logging(
            level="INFO",
            log_format="text",
            log_file=tmp_path / "forbidden" / "test.log",
            root_logger_name="test_file_fail_logger",
            force=True,
        )
        assert resilient_logger.name == "test_file_fail_logger"


def test_safe_json_serializer_exception_fallbacks() -> None:
    """Verify _safe_json_serializer falls back gracefully when item(), tolist(), or __dict__ str() raise."""
    # 1. Custom float-like type with item() raising an exception
    class broken_float_scalar:
        def item(self) -> float:
            raise RuntimeError("float item failure")

        def __str__(self) -> str:
            return "broken_float_fallback"

    assert _safe_json_serializer(broken_float_scalar()) == "broken_float_fallback"

    # 2. Custom ndarray-like type with size <= 20 and tolist() raising an exception
    class broken_ndarray:
        size = 5
        shape = (5,)
        dtype = np.float32

        def tolist(self) -> list:
            raise RuntimeError("tolist conversion failure")

    serialized_nd = _safe_json_serializer(broken_ndarray())
    assert "<ndarray shape=(5,) dtype=" in serialized_nd

    # 3. Object with __dict__ containing an unstringable attribute
    class UnstringableValue:
        def __str__(self) -> str:
            raise RuntimeError("cannot stringify value")

    class ContainerWithBadAttr:
        def __init__(self) -> None:
            self.bad_field = UnstringableValue()

        def __str__(self) -> str:
            return "container_fallback_str"

    assert _safe_json_serializer(ContainerWithBadAttr()) == "container_fallback_str"


def test_logging_config_namespace_registration_bootstrap() -> None:
    """Verify logging_config.py bootstraps 'medical_segmentation' into sys.modules when absent."""
    import runpy
    import sys
    saved = sys.modules.pop("medical_segmentation", None)
    try:
        assert "medical_segmentation" not in sys.modules
        log_path = Path(__file__).resolve().parent.parent / "logging_config.py"
        runpy.run_path(str(log_path))
        assert "medical_segmentation" in sys.modules
    finally:
        if saved is not None:
            sys.modules["medical_segmentation"] = saved

