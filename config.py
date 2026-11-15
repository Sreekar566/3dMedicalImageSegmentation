"""
Configuration Module for 3dMedicalImageSegmentation.
Defines strongly-typed settings using Pydantic Settings for Azure Blob Storage,
PostgreSQL database connection pooling, and MONAI 3D sliding-window inference.
"""

from __future__ import annotations

import os
import re
import urllib.parse
from pathlib import Path
from typing import Any, Literal, Tuple, Union

import torch
from pydantic import Field, field_validator
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
        try:
            spacing = tuple(float(x) for x in v)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Target spacing values must be valid floats, got {v}.") from exc
        if any(s <= 0.0 for s in spacing):
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
        try:
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
        if str(v) in ("", "."):
            return None
        return v
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

    @property
    def connection_url(self) -> str:
        """Construct the psycopg2 connection URL with required Azure SSL parameters."""
        encoded_user = urllib.parse.quote_plus(self.user)
        encoded_password = urllib.parse.quote_plus(self.password)
        return (
            f"postgresql+psycopg2://{encoded_user}:{encoded_password}@"
            f"{self.host}:{self.port}/{self.name}?sslmode={self.sslmode}"
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
    target_spacing: Tuple[float, float, float] | str | None = Field(
        default=(1.0, 1.0, 1.0),
        description="Target voxel spacing in mm (x, y, z). Set to None to keep native acquisition spacing.",
    )
    roi_size: Tuple[int, int, int] | str = Field(
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
    device: str = Field(
        default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        description="Execution device: 'cuda', 'mps', or 'cpu'.",
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
    def validate_device(cls, v: str) -> str:
        if not isinstance(v, str):
            raise TypeError(f"Execution device must be a string, got {type(v).__name__}.")
        v_clean = v.lower().strip()
        try:
            torch.device(v_clean)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid PyTorch execution device '{v}': {exc}") from exc
        return v_clean


class PipelineConfig(BaseSettings):
    """Master pipeline configuration combining Azure, DB, and Inference settings."""

    model_config = SettingsConfigDict(
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


# Global configuration instance
config = PipelineConfig()
