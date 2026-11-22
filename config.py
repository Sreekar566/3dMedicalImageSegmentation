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
from typing import Any, Dict, Literal, Sequence, Tuple, Union

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


def parse_target_spacing(v: Any) -> Tuple[float, float, float] | None:
    """Parse and validate 3D voxel spacing tuple (dx, dy, dz) in mm."""
    if v is None:
        return None
    if isinstance(v, str):
        cleaned = v.strip().strip("()[]")
        if not cleaned or cleaned.lower() in ("none", "null"):
            return None
        try:
            parts = [float(p.strip()) for p in cleaned.split(",") if p.strip()]
        except ValueError as exc:
            raise ValueError(
                f"Target spacing string must contain 3 comma-separated floats (dx, dy, dz), got '{v}'."
            ) from exc
        if len(parts) != 3:
            raise ValueError(
                f"Target spacing string must contain 3 comma-separated floats (dx, dy, dz), got '{v}'."
            )
        v = tuple(parts)
    if isinstance(v, (list, tuple)):
        if len(v) != 3:
            raise ValueError(f"Target spacing must have exactly 3 dimensions (dx, dy, dz), got {len(v)} elements.")
        if any(isinstance(x, bool) for x in v):
            raise TypeError("Target spacing dimensions cannot be boolean values.")
        try:
            spacing = tuple(float(x) for x in v)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Target spacing values must be valid floats, got {v}.") from exc
        if any(not math.isfinite(s) or s <= 0.0 for s in spacing):
            raise ValueError(f"Target spacing dimensions must be strictly positive (> 0.0), got {spacing}.")
        return spacing
    raise TypeError(f"Invalid type for target_spacing: {type(v).__name__}.")


def parse_roi_size(v: Any) -> Tuple[int, int, int]:
    """Parse and validate 3D sub-patch ROI size tuple (depth, height, width)."""
    if isinstance(v, str):
        cleaned = v.strip().strip("()[]")
        try:
            parts = [int(p.strip()) for p in cleaned.split(",") if p.strip()]
        except ValueError as exc:
            raise ValueError(
                f"ROI size string must contain 3 comma-separated integers (depth, height, width), got '{v}'."
            ) from exc
        if len(parts) != 3:
            raise ValueError(
                f"ROI size string must contain 3 comma-separated integers (depth, height, width), got '{v}'."
            )
        v = tuple(parts)
    if isinstance(v, (list, tuple)):
        if len(v) != 3:
            raise ValueError(f"ROI size must have exactly 3 dimensions (depth, height, width), got {len(v)} elements.")
        if any(isinstance(x, bool) for x in v):
            raise TypeError("ROI size dimensions cannot be boolean values.")
        try:
            for x in v:
                if isinstance(x, float) and not x.is_integer():
                    raise ValueError(f"ROI size dimensions must be integers, got float {x}.")
            roi = tuple(int(x) for x in v)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"ROI size values must be valid integers, got {v}.") from exc
        if any(d <= 0 for d in roi):
            raise ValueError(f"ROI size dimensions must be strictly positive integers (> 0), got {roi}.")
        return roi
    raise TypeError(f"Invalid type for roi_size: {type(v).__name__}.")


def parse_model_checkpoint_path(v: Any) -> Path | None:
    """Parse and clean model checkpoint path, normalizing empty strings to None."""
    if v is None:
        return None
    if isinstance(v, str):
        cleaned = v.strip()
        if not cleaned or cleaned.lower() in ("none", "null"):
            return None
        return Path(cleaned)
    if isinstance(v, Path):
        cleaned_str = str(v).strip()
        if not cleaned_str or cleaned_str in (".", "none", "null"):
            return None
        return Path(cleaned_str)
    raise TypeError(f"Invalid type for model_checkpoint_path: {type(v).__name__}.")


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
        if v is None:
            return None
        if isinstance(v, str):
            cleaned = v.strip()
            if not cleaned or cleaned.lower() in ("none", "null"):
                return None
            return Path(cleaned)
        if isinstance(v, Path):
            return v
        raise TypeError(f"Invalid type for log_file: {type(v).__name__}.")

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
        if not isinstance(path, (str, Path)):
            raise TypeError(f"path must be a string or Path, got {type(path).__name__}.")
        clean_path_str = str(path).strip()
        if not clean_path_str:
            raise ValueError("Configuration file path cannot be empty.")
        resolved_path = Path(clean_path_str).resolve()
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {resolved_path}")

        ext = resolved_path.suffix.lower()
        if ext in (".yaml", ".yml"):
            if yaml is None:
                raise ImportError(
                    f"PyYAML is required to load YAML configuration from '{resolved_path}'. "
                    "Install PyYAML via 'pip install pyyaml'."
                )
            try:
                with open(resolved_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            except Exception as exc:
                raise ValueError(f"Invalid YAML configuration in '{resolved_path}': {exc}") from exc
            if not isinstance(data, dict):
                raise ValueError(
                    f"Configuration YAML in '{resolved_path}' must be an object/dictionary, got {type(data).__name__}."
                )
        elif ext == ".toml":
            if tomllib is None:
                raise ImportError(
                    f"tomllib is required to load TOML configuration from '{resolved_path}'."
                )
            try:
                with open(resolved_path, "rb") as f:
                    data = tomllib.load(f)
            except Exception as exc:
                raise ValueError(f"Invalid TOML configuration in '{resolved_path}': {exc}") from exc
            if not isinstance(data, dict):
                raise ValueError(
                    f"Configuration TOML in '{resolved_path}' must be an object/dictionary, got {type(data).__name__}."
                )
        else:
            try:
                with open(resolved_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON configuration in '{resolved_path}': {exc}") from exc

            if not isinstance(data, dict):
                raise ValueError(
                    f"Configuration JSON in '{resolved_path}' must be an object/dictionary, got {type(data).__name__}."
                )
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

        payload: Dict[str, Any] = {k: v for k, v in data.items()}
        azure_fields = {
            "connection_string",
            "account_name",
            "account_key",
            "input_container",
            "output_container",
            "max_concurrency",
            "timeout_seconds",
        }
        db_fields = {
            "host",
            "port",
            "name",
            "user",
            "password",
            "sslmode",
            "pool_size",
            "max_overflow",
            "pool_timeout",
            "batch_size",
            "pool_recycle",
            "pool_pre_ping",
        }
        inference_fields = {
            "modality",
            "target_orientation",
            "target_spacing",
            "roi_size",
            "sw_batch_size",
            "overlap",
            "in_channels",
            "out_channels",
            "model_checkpoint_path",
            "device",
            "blend_mode",
        }

        # Route top-level keys into nested sections if not explicitly defined inside sections
        azure_sec = dict(payload.get("azure", {})) if isinstance(payload.get("azure"), dict) else {}
        db_sec = dict(payload.get("db", {})) if isinstance(payload.get("db"), dict) else {}
        inference_sec = dict(payload.get("inference", {})) if isinstance(payload.get("inference"), dict) else {}

        for k, v in list(payload.items()):
            if k in azure_fields and k not in azure_sec:
                azure_sec[k] = v
            elif k in db_fields and k not in db_sec:
                db_sec[k] = v
            elif k in inference_fields and k not in inference_sec:
                inference_sec[k] = v

        if azure_sec:
            payload["azure"] = azure_sec
        if db_sec:
            payload["db"] = db_sec
        if inference_sec:
            payload["inference"] = inference_sec

        return cls.model_validate(payload)

    def to_dict(self) -> Dict[str, Any]:
        """Export strongly-typed configuration to a JSON-serializable dictionary."""
        return self.model_dump(mode="json")

    def to_file(self, path: Path | str, format: Optional[str] = None) -> Path:
        """
        Serialize and export configuration to a JSON or YAML file.

        Args:
            path: Destination file path.
            format: Optional format override ('json', 'yaml', or 'yml').
                    Defaults to inferring from file extension or 'json'.

        Returns:
            Resolved Path of the saved configuration file.
        """
        if not isinstance(path, (str, Path)):
            raise TypeError(f"path must be a string or Path, got {type(path).__name__}.")
        clean_path_str = str(path).strip()
        if not clean_path_str:
            raise ValueError("Destination file path cannot be empty.")
        resolved_path = Path(clean_path_str).resolve()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)

        target_format = (format.lower().strip() if format else resolved_path.suffix.lower().lstrip("."))
        if target_format in ("yaml", "yml"):
            if yaml is None:
                raise ImportError(
                    f"PyYAML is required to write YAML configuration to '{resolved_path}'. "
                    "Install PyYAML via 'pip install pyyaml'."
                )
            with open(resolved_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.to_dict(), f, sort_keys=False)
        else:
            with open(resolved_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
        return resolved_path


# Global configuration instance
config = PipelineConfig()
