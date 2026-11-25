"""
Configuration Module for 3dMedicalImageSegmentation.
Defines strongly-typed settings using Pydantic Settings for Azure Blob Storage,
PostgreSQL database connection pooling, and MONAI 3D sliding-window inference.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Sequence, Tuple, TypeVar

try:
    import tomllib  # Python 3.11+ standard library
except ImportError:
    tomllib = None  # type: ignore[assignment]

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

import torch
from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_T = TypeVar("_T")

__all__ = [
    "AZURE_CONTAINER_NAME_REGEX",
    "AzureStorageSettings",
    "DatabaseSettings",
    "InferenceSettings",
    "PipelineConfig",
    "config",
    "parse_model_checkpoint_path",
    "parse_roi_size",
    "parse_target_spacing",
    "validate_azure_container_name",
    "validate_orientation_code",
]

AZURE_CONTAINER_NAME_REGEX = re.compile(r"^[a-z0-9](?!.*--)[a-z0-9-]{1,61}[a-z0-9]$")


def validate_orientation_code(v: str) -> str:
    """
    Validate standard 3-letter anatomical coordinate code for SimpleITK / ITK DICOMOrient.
    A valid code is a 3-letter permutation consisting of exactly one character from each
    orthogonal anatomical axis pair: {R, L}, {A, P}, and {S, I}.
    """
    if not isinstance(v, str):
        raise TypeError(f"Target orientation must be a string, got {type(v).__name__}.")
    v_clean = v.upper().strip()
    if len(v_clean) != 3:
        raise ValueError(
            f"Invalid 3D anatomical orientation '{v}'. Must be exactly 3 characters."
        )

    rl_count = sum(1 for c in v_clean if c in "RL")
    ap_count = sum(1 for c in v_clean if c in "AP")
    si_count = sum(1 for c in v_clean if c in "SI")

    if rl_count != 1 or ap_count != 1 or si_count != 1:
        raise ValueError(
            f"Invalid 3D anatomical orientation '{v}'. Must be a 3-letter permutation consisting of "
            f"one of R/L, one of A/P, and one of S/I (e.g., 'RAS', 'LPS', 'PIR')."
        )
    return v_clean


def validate_azure_container_name(name: str) -> str:
    """
    Validate Azure Blob Storage container naming rules according to Microsoft specifications:
    - 3 to 63 characters in length
    - Starts and ends with a lowercase alphanumeric character
    - Contains only lowercase letters, numbers, and hyphens (no consecutive hyphens)
    """
    if not isinstance(name, str):
        raise TypeError(f"Container name must be a string, got {type(name).__name__}.")
    name_clean = name.strip()
    if not name_clean:
        raise ValueError("Container name cannot be empty.")
    if not AZURE_CONTAINER_NAME_REGEX.match(name_clean):
        raise ValueError(
            f"Invalid Azure Blob container name '{name}'. Must be 3-63 lowercase alphanumeric "
            f"characters or hyphens, start/end with an alphanumeric character, and contain no consecutive hyphens."
        )
    return name_clean


def _parse_3d_triplet(
    v: Any,
    *,
    field_name: str,
    display_name: str,
    axes_label: str,
    type_plural: str,
    str_item_parser: Callable[[str], _T],
    seq_item_converter: Callable[[Any], _T],
    allow_none: bool = False,
) -> Tuple[_T, _T, _T] | None:
    """Parse a 3-element numeric tuple from a comma-separated string or 3-item sequence."""
    if allow_none and v is None:
        return None
    if isinstance(v, str):
        cleaned = v.strip().strip("()[]")
        if allow_none and (not cleaned or cleaned.lower() in ("none", "null")):
            return None
        try:
            parts = [str_item_parser(p.strip()) for p in cleaned.split(",") if p.strip()]
        except ValueError as exc:
            raise ValueError(
                f"{display_name} string must contain 3 comma-separated {type_plural} ({axes_label}), got '{v}'."
            ) from exc
        if len(parts) != 3:
            raise ValueError(
                f"{display_name} string must contain 3 comma-separated {type_plural} ({axes_label}), got '{v}'."
            )
        v = tuple(parts)
    if isinstance(v, (list, tuple)):
        if len(v) != 3:
            raise ValueError(f"{display_name} must have exactly 3 dimensions ({axes_label}), got {len(v)} elements.")
        if any(isinstance(x, bool) for x in v):
            raise TypeError(f"{display_name} dimensions cannot be boolean values.")
        try:
            converted = tuple(seq_item_converter(x) for x in v)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{display_name} values must be valid {type_plural}, got {v}.") from exc
        return converted
    raise TypeError(f"Invalid type for {field_name}: {type(v).__name__}.")


def _to_exact_int(value: Any) -> int:
    """Convert numeric value to int while rejecting non-integral floats."""
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"ROI size dimensions must be integers, got float {value}.")
    return int(value)


def _parse_optional_path(v: Any, field_name: str) -> Path | None:
    """Normalize None, empty strings, sentinel strings ('none', 'null'), or Paths to Path | None."""
    if v is None:
        return None
    if isinstance(v, str):
        cleaned = v.strip()
        if not cleaned or cleaned.lower() in ("none", "null"):
            return None
        return Path(cleaned)
    if isinstance(v, Path):
        cleaned_str = str(v).strip()
        if not cleaned_str or cleaned_str.lower() in (".", "none", "null"):
            return None
        return Path(cleaned_str)
    raise TypeError(f"Invalid type for {field_name}: {type(v).__name__}.")


def parse_target_spacing(v: Any) -> Tuple[float, float, float] | None:
    """Parse and validate 3D voxel spacing tuple (dx, dy, dz) in mm."""
    spacing = _parse_3d_triplet(
        v,
        field_name="target_spacing",
        display_name="Target spacing",
        axes_label="dx, dy, dz",
        type_plural="floats",
        str_item_parser=float,
        seq_item_converter=float,
        allow_none=True,
    )
    if spacing is None:
        return None
    if any(not math.isfinite(s) or s <= 0.0 for s in spacing):
        raise ValueError(f"Target spacing dimensions must be strictly positive (> 0.0), got {spacing}.")
    return spacing


def parse_roi_size(v: Any) -> Tuple[int, int, int]:
    """Parse and validate 3D sub-patch ROI size tuple (depth, height, width)."""
    roi = _parse_3d_triplet(
        v,
        field_name="roi_size",
        display_name="ROI size",
        axes_label="depth, height, width",
        type_plural="integers",
        str_item_parser=int,
        seq_item_converter=_to_exact_int,
        allow_none=False,
    )
    assert roi is not None
    if any(d <= 0 for d in roi):
        raise ValueError(f"ROI size dimensions must be strictly positive integers (> 0), got {roi}.")
    return roi


def parse_model_checkpoint_path(v: Any) -> Path | None:
    """Parse and clean model checkpoint path, normalizing empty strings to None."""
    return _parse_optional_path(v, "model_checkpoint_path")


class AzureStorageSettings(BaseSettings):
    """Azure Blob Storage credentials and container configuration."""

    model_config = SettingsConfigDict(
        env_prefix="AZURE_STORAGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    connection_string: str = Field(
        default="",
        description="Azure Blob Storage connection string. Leave blank if using account key or SAS token.",
    )
    account_name: str = Field(
        default="",
        description="Azure storage account name.",
    )
    account_key: str = Field(
        default="",
        description="Azure storage account access key.",
    )
    input_container: str = Field(
        default="medical-volumes-input",
        description="Azure Blob container for incoming DICOM series or NIfTI volumes.",
    )
    output_container: str = Field(
        default="medical-segmentations-output",
        description="Azure Blob container for generated 3D segmentation masks.",
    )
    max_concurrency: int = Field(
        default=4,
        ge=1,
        description="Maximum concurrent connections for blob download/upload operations.",
    )
    timeout_seconds: int = Field(
        default=300,
        ge=1,
        description="Timeout for blob transfer operations.",
    )

    @field_validator("account_name", "account_key", "connection_string", mode="before")
    @classmethod
    def strip_credentials(cls, v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, str):
            return v.strip()
        return str(v)

    @field_validator("input_container", "output_container")
    @classmethod
    def validate_container(cls, v: str) -> str:
        return validate_azure_container_name(v)


class DatabaseSettings(BaseSettings):
    """Azure Database for PostgreSQL connection and connection-pool configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = Field(
        default="localhost",
        min_length=1,
        description="PostgreSQL server hostname (e.g., your-azure-db.postgres.database.azure.com).",
    )
    port: int = Field(
        default=5432,
        ge=1,
        le=65535,
        description="PostgreSQL port number.",
    )
    name: str = Field(
        default="medical_imaging",
        min_length=1,
        description="Target database name.",
    )
    user: str = Field(
        default="postgres",
        min_length=1,
        description="Database username.",
    )
    password: str = Field(
        default="postgres",
        description="Database password.",
    )
    sslmode: Literal["disable", "allow", "prefer", "require", "verify-ca", "verify-full"] = Field(
        default="require",
        description="SSL mode for Azure Database for PostgreSQL (must be 'require' or 'verify-full').",
    )
    pool_size: int = Field(
        default=10,
        ge=1,
        description="SQLAlchemy QueuePool base connection pool size.",
    )
    max_overflow: int = Field(
        default=20,
        ge=0,
        description="SQLAlchemy QueuePool max overflow connections during peak spikes.",
    )
    pool_timeout: int = Field(
        default=30,
        ge=1,
        description="Seconds to wait before timing out on getting a connection from the pool.",
    )
    batch_size: int = Field(
        default=500,
        ge=1,
        description="Chunk size for batched bulk inserts into PostgreSQL to avoid OOM.",
    )
    pool_recycle: int = Field(
        default=1800,
        ge=1,
        description="Recycles connections older than this number of seconds (default: 1800s / 30m) to avoid Azure idle timeouts.",
    )
    pool_pre_ping: bool = Field(
        default=True,
        description="Tests connection liveness before checkout from SQLAlchemy connection pool.",
    )

    @field_validator("host", "name", "user")
    @classmethod
    def validate_non_empty_strings(cls, v: Any, info: ValidationInfo) -> str:
        if not isinstance(v, str):
            raise TypeError(f"{info.field_name} must be a string, got {type(v).__name__}.")
        cleaned = v.strip()
        if not cleaned:
            raise ValueError(f"Database {info.field_name} cannot be empty or whitespace only.")
        return cleaned

    @property
    def connection_url(self) -> str:
        """Construct the psycopg2 connection URL with required Azure SSL parameters."""
        encoded_user = urllib.parse.quote_plus(self.user.strip())
        encoded_password = urllib.parse.quote_plus(self.password)
        encoded_host = self.host.strip()
        encoded_name = urllib.parse.quote_plus(self.name.strip())
        return (
            f"postgresql+psycopg2://{encoded_user}:{encoded_password}@"
            f"{encoded_host}:{self.port}/{encoded_name}?sslmode={self.sslmode}"
        )


class InferenceSettings(BaseSettings):
    """MONAI 3D sliding-window volumetric inference and model parameters."""

    model_config = SettingsConfigDict(
        env_prefix="INFERENCE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    modality: Literal["CT", "CBCT", "MRI"] = Field(
        default="CT",
        description="Primary imaging modality (controls intensity windowing and normalization).",
    )
    target_orientation: str = Field(
        default="RAS",
        description="Target spatial orientation code for SimpleITK DICOMOrient (e.g. 'RAS', 'LPS').",
    )
    target_spacing: Tuple[float, float, float] | Sequence[float] | str | None = Field(
        default=(1.0, 1.0, 1.0),
        description="Target voxel spacing in mm (x, y, z). Set to None to keep native acquisition spacing.",
    )
    roi_size: Tuple[int, int, int] | Sequence[int] | str = Field(
        default=(96, 96, 96),
        description="Volumetric sub-patch ROI size for sliding-window inference (depth, height, width).",
    )
    sw_batch_size: int = Field(
        default=2,
        ge=1,
        description="Number of patches evaluated concurrently in sliding-window inference.",
    )
    overlap: float = Field(
        default=0.5,
        ge=0.0,
        le=0.8,
        description="Patch overlap ratio during sliding-window inference for seamless stitching.",
    )
    in_channels: int = Field(
        default=1,
        ge=1,
        description="Number of input volume channels (1 for grayscale CT/MRI).",
    )
    out_channels: int = Field(
        default=2,
        ge=1,
        description="Number of segmentation classes (e.g. 2 for binary foreground/background).",
    )
    model_checkpoint_path: Path | str | None = Field(
        default=None,
        description="Path to pre-trained PyTorch 3D U-Net checkpoint (.pt or .pth).",
    )
    device: str | torch.device = Field(
        default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        description="Execution device: 'cuda', 'mps', or 'cpu'.",
    )
    blend_mode: Literal["gaussian", "constant"] = Field(
        default="gaussian",
        description="Blending mode for sliding window inference patch overlap ('gaussian' or 'constant').",
    )

    @field_validator("modality", mode="before")
    @classmethod
    def normalize_modality(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.upper().strip()
        return v

    @field_validator("target_orientation")
    @classmethod
    def validate_orientation(cls, v: str) -> str:
        return validate_orientation_code(v)

    @field_validator("target_spacing", mode="before")
    @classmethod
    def validate_spacing(cls, v: Any) -> Tuple[float, float, float] | None:
        return parse_target_spacing(v)

    @field_validator("roi_size", mode="before")
    @classmethod
    def validate_roi(cls, v: Any) -> Tuple[int, int, int]:
        return parse_roi_size(v)

    @field_validator("model_checkpoint_path", mode="before")
    @classmethod
    def validate_checkpoint(cls, v: Any) -> Path | None:
        return parse_model_checkpoint_path(v)

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: Any) -> str:
        if isinstance(v, torch.device):
            return str(v)
        if not isinstance(v, str):
            raise TypeError(f"Execution device must be a string or torch.device, got {type(v).__name__}.")
        v_clean = v.lower().strip()
        try:
            torch.device(v_clean)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid PyTorch execution device '{v}': {exc}") from exc
        return v_clean

    @field_validator("blend_mode", mode="before")
    @classmethod
    def normalize_blend_mode(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError(f"blend_mode must be a string, got {type(v).__name__}.")
        v_clean = v.lower().strip()
        if v_clean not in ("gaussian", "constant"):
            raise ValueError(f"Invalid blend_mode '{v}'. Must be 'gaussian' or 'constant'.")
        return v_clean


class PipelineConfig(BaseSettings):
    """Master pipeline configuration combining Azure, DB, and Inference settings."""

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    azure: AzureStorageSettings = Field(default_factory=AzureStorageSettings)
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    work_dir: Path | str = Field(
        default=Path("/tmp/medical_segmentation_workspace"),
        description="Local staging directory for downloaded series and generated volumes.",
    )
    log_level: str = Field(
        default="INFO",
        description="Application logging verbosity: DEBUG, INFO, WARNING, ERROR.",
    )
    log_format: str = Field(
        default="text",
        description="Log output formatting style ('text' or 'json').",
    )
    log_file: Path | str | None = Field(
        default=None,
        description="Optional local filesystem destination for persistent log records.",
    )

    @field_validator("work_dir", mode="before")
    @classmethod
    def validate_work_dir(cls, v: Any) -> Path:
        if isinstance(v, (str, Path)):
            cleaned = str(v).strip()
            if not cleaned:
                raise ValueError("work_dir cannot be an empty path.")
            return Path(cleaned)
        raise TypeError(f"Invalid type for work_dir: {type(v).__name__}.")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL"}
        v_upper = v.upper().strip()
        if v_upper not in valid_levels:
            raise ValueError(
                f"Invalid log_level '{v}'. Must be one of: {', '.join(sorted(valid_levels))}."
            )
        return "WARNING" if v_upper == "WARN" else v_upper

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        if not isinstance(v, str):
            raise TypeError(f"log_format must be a string, got {type(v).__name__}.")
        v_clean = v.lower().strip()
        if v_clean not in {"text", "json"}:
            raise ValueError(f"Invalid log_format '{v}'. Must be 'text' or 'json'.")
        return v_clean

    @field_validator("log_file", mode="before")
    @classmethod
    def validate_log_file(cls, v: Any) -> Path | None:
        return _parse_optional_path(v, "log_file")

    @staticmethod
    def _resolve_file_path(path: Path | str, empty_message: str) -> Path:
        """Validate and resolve a configuration file path."""
        if not isinstance(path, (str, Path)):
            raise TypeError(f"path must be a string or Path, got {type(path).__name__}.")
        clean_path_str = str(path).strip()
        if not clean_path_str:
            raise ValueError(empty_message)
        return Path(clean_path_str).resolve()

    @staticmethod
    def _load_config_dict(resolved_path: Path) -> Dict[str, Any]:
        """Parse a JSON, YAML, or TOML file into a dictionary payload."""
        ext = resolved_path.suffix.lower()
        if ext in (".yaml", ".yml"):
            yaml_mod = globals().get("yaml")
            if yaml_mod is None:
                raise ImportError(
                    f"PyYAML is required to load YAML configuration from '{resolved_path}'. "
                    "Install PyYAML via 'pip install pyyaml'."
                )
            format_label = "YAML"
            open_mode = "r"
            loader = yaml_mod.safe_load
            parse_errors: tuple[type[BaseException], ...] = (Exception,)
        elif ext == ".toml":
            toml_mod = globals().get("tomllib")
            if toml_mod is None:
                raise ImportError(
                    f"tomllib is required to load TOML configuration from '{resolved_path}'."
                )
            format_label = "TOML"
            open_mode = "rb"
            loader = toml_mod.load
            parse_errors = (Exception,)
        else:
            format_label = "JSON"
            open_mode = "r"
            loader = json.load
            parse_errors = (json.JSONDecodeError,)

        open_kwargs = {"encoding": "utf-8"} if "b" not in open_mode else {}
        try:
            with open(resolved_path, open_mode, **open_kwargs) as f:
                data = loader(f)
        except parse_errors as exc:
            raise ValueError(f"Invalid {format_label} configuration in '{resolved_path}': {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError(
                f"Configuration {format_label} in '{resolved_path}' must be an object/dictionary, got {type(data).__name__}."
            )
        return data

    @classmethod
    def from_file(cls, path: Path | str) -> PipelineConfig:
        """
        Load and validate pipeline configuration from a JSON, YAML, or TOML configuration file.

        Args:
            path: Local filesystem path to a configuration file (.json, .yaml, .yml, or .toml).

        Returns:
            Validated PipelineConfig instance.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the file is not valid JSON/YAML/TOML or not a dictionary/object.
            TypeError: If the path parameter is not a string or Path.
            ImportError: If parsing requires an unavailable optional library (e.g., PyYAML).
            ValidationError: If the configuration parameters fail schema validation.
        """
        resolved_path = cls._resolve_file_path(path, "Configuration file path cannot be empty.")
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {resolved_path}")

        data = cls._load_config_dict(resolved_path)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PipelineConfig:
        """
        Construct and validate PipelineConfig from a dictionary payload.
        Supports both nested sections ('azure', 'db', 'inference') and automatically
        routes known flat configuration keys to their respective sub-models.

        Args:
            data: Configuration dictionary with optional nested 'azure', 'db', and 'inference' sections.

        Returns:
            Validated PipelineConfig instance.
        """
        if not isinstance(data, dict):
            raise TypeError(f"Configuration data must be a dictionary, got {type(data).__name__}.")

        payload: Dict[str, Any] = dict(data)
        subsection_models = {
            "azure": AzureStorageSettings,
            "db": DatabaseSettings,
            "inference": InferenceSettings,
        }

        for section_name, model_cls in subsection_models.items():
            existing = payload.get(section_name)
            section_data = dict(existing) if isinstance(existing, dict) else {}
            for field_name in model_cls.model_fields:
                if field_name in payload and field_name not in section_data:
                    section_data[field_name] = payload[field_name]
            if section_data:
                payload[section_name] = section_data

        return cls.model_validate(payload)

    def to_dict(self) -> Dict[str, Any]:
        """Export strongly-typed configuration to a JSON-serializable dictionary."""
        return self.model_dump(mode="json")

    def to_file(self, path: Path | str, format: str | None = None) -> Path:
        """
        Serialize and export configuration to a JSON or YAML file.

        Args:
            path: Destination file path.
            format: Optional format override ('json', 'yaml', or 'yml').
                    Defaults to inferring from file extension or 'json'.

        Returns:
            Resolved Path of the saved configuration file.
        """
        resolved_path = self._resolve_file_path(path, "Destination file path cannot be empty.")
        resolved_path.parent.mkdir(parents=True, exist_ok=True)

        target_format = format.lower().strip() if format else resolved_path.suffix.lower().lstrip(".")
        if target_format in ("yaml", "yml"):
            yaml_mod = globals().get("yaml")
            if yaml_mod is None:
                raise ImportError(
                    f"PyYAML is required to write YAML configuration to '{resolved_path}'. "
                    "Install PyYAML via 'pip install pyyaml'."
                )
            with open(resolved_path, "w", encoding="utf-8") as f:
                yaml_mod.safe_dump(self.to_dict(), f, sort_keys=False)
        else:
            with open(resolved_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
        return resolved_path


# Global configuration instance
config = PipelineConfig()
