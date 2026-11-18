"""
Structured Logging Configuration Engine for 3dMedicalImageSegmentation.
Provides production-ready structured JSON log formatting for cloud log aggregators
(Azure Log Analytics, AWS CloudWatch, Datadog), human-readable text formatting,
contextual logging with safe serialization, and configurable stream/file handlers.
"""

from __future__ import annotations

import datetime
import json
import logging
import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional, Set

# Ensure 'medical_segmentation' module namespace is resolvable
if "medical_segmentation" not in sys.modules:
    med_pkg = types.ModuleType("medical_segmentation")
    med_pkg.__path__ = [str(Path(__file__).resolve().parent)]
    sys.modules["medical_segmentation"] = med_pkg

__all__ = [
    "DEFAULT_DATE_FORMAT",
    "DEFAULT_TEXT_FORMAT",
    "StandardTextFormatter",
    "StructuredJsonFormatter",
    "configure_logging",
    "get_logger",
]

DEFAULT_TEXT_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_STANDARD_LOG_RECORD_ATTRS: Set[str] = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "message",
    "module",
    "msecs",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "taskName",
    "thread",
    "threadName",
}


def _safe_json_serializer(obj: Any) -> Any:
    """Fallback JSON serializer for domain objects, timestamps, Paths, and tensors."""
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(list(obj), key=str)

    # Check numpy types without hard dependency
    obj_type = type(obj).__name__
    if "int" in obj_type and hasattr(obj, "item"):
        try:
            return int(obj.item())
        except Exception:
            pass
    if "float" in obj_type and hasattr(obj, "item"):
        try:
            return float(obj.item())
        except Exception:
            pass
    if "ndarray" in obj_type:
        if getattr(obj, "size", 0) <= 20:
            try:
                return obj.tolist()
            except Exception:
                pass
        return f"<ndarray shape={getattr(obj, 'shape', None)} dtype={getattr(obj, 'dtype', None)}>"

    # Check PyTorch device/tensor
    if "device" in obj_type:
        return str(obj)
    if "Tensor" in obj_type:
        return f"<Tensor shape={tuple(obj.shape)} dtype={obj.dtype}>"

    # Objects with dict
    if hasattr(obj, "__dict__"):
        try:
            attrs = {k: str(v) for k, v in vars(obj).items() if not k.startswith("_")}
            if attrs:
                return attrs
        except Exception:
            pass

    return str(obj)


class StructuredJsonFormatter(logging.Formatter):
    """
    Format LogRecords as single-line JSON objects for automated log ingestion.
    Serializes timestamp (UTC ISO-8601), log level, logger name, message,
    callsite info (module, function, line), PID/thread, custom contextual attributes,
    and structured exception details.
    """

    def __init__(self, ensure_ascii: bool = False):
        super().__init__()
        self.ensure_ascii = ensure_ascii

    def format(self, record: logging.LogRecord) -> str:
        """Format the specified record as a single JSON line."""
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)

        created_utc = datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc)
        iso_timestamp = created_utc.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        entry: Dict[str, Any] = {
            "timestamp": iso_timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": message,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.threadName,
        }

        # Extract extra contextual fields
        extra_fields: Dict[str, Any] = {}
        for key, val in record.__dict__.items():
            if key not in _STANDARD_LOG_RECORD_ATTRS and not key.startswith("_"):
                extra_fields[key] = val

        if extra_fields:
            entry["extra"] = extra_fields

        # Format exception traceback if present
        if record.exc_info:
            exc_type, exc_val, _ = record.exc_info
            entry["exception"] = {
                "type": getattr(exc_type, "__name__", "Exception") if exc_type else "Exception",
                "message": str(exc_val) if exc_val else "",
                "traceback": self.formatException(record.exc_info),
            }

        if record.stack_info:
            entry["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(entry, default=_safe_json_serializer, ensure_ascii=self.ensure_ascii)


class StandardTextFormatter(logging.Formatter):
    """
    Human-readable console text formatter with timestamp, level, logger name,
    message, and optional formatted extra attributes.
    """

    def __init__(
        self,
        fmt: str = DEFAULT_TEXT_FORMAT,
        datefmt: str = DEFAULT_DATE_FORMAT,
    ):
        super().__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        """Format the specified record as readable text."""
        base_formatted = super().format(record)

        extra_fields = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _STANDARD_LOG_RECORD_ATTRS and not k.startswith("_")
        }

        if extra_fields:
            extra_tokens = [f"{k}={v}" for k, v in sorted(extra_fields.items(), key=lambda x: str(x[0]))]
            return f"{base_formatted} [{', '.join(extra_tokens)}]"

        return base_formatted


def configure_logging(
    level: str | int = "INFO",
    log_format: str = "text",
    log_file: Optional[Path | str] = None,
    root_logger_name: Optional[str] = None,
    force: bool = True,
) -> logging.Logger:
    """
    Configure application and root logging handlers, levels, and formatters.

    Args:
        level: Logging verbosity level (e.g. 'DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_format: Output format ('text' for human-readable console, 'json' for structured lines).
        log_file: Optional path to append log entries on the local filesystem.
        root_logger_name: Target logger name. None configures the root logger.
        force: If True, removes all existing handlers before attaching new ones.

    Returns:
        Configured logging.Logger instance.
    """
    # 1. Resolve numeric log level
    if isinstance(level, str):
        level_upper = level.upper().strip()
        if level_upper == "WARN":
            level_upper = "WARNING"
        numeric_level = getattr(logging, level_upper, logging.INFO)
    else:
        numeric_level = int(level)

    # 2. Select formatter
    format_clean = str(log_format).lower().strip()
    if format_clean == "json":
        formatter: logging.Formatter = StructuredJsonFormatter()
    else:
        formatter = StandardTextFormatter()

    # 3. Determine target logger
    logger = logging.getLogger(root_logger_name) if root_logger_name else logging.getLogger()

    # 4. Clear existing handlers if force is set
    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    logger.setLevel(numeric_level)

    # 5. Attach stdout stream handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(numeric_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # 6. Attach optional file handler
    if log_file:
        file_path = Path(log_file)
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(file_path), encoding="utf-8")
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as exc:
            logger.warning("Failed to initialize file logging at '%s': %s", file_path, exc)

    return logger


def get_logger(name: str = "3dMedicalImageSegmentation") -> logging.Logger:
    """Obtain a logger instance by name."""
    return logging.getLogger(name)
