"""
Configuration Module for 3dMedicalImageSegmentation.
Defines strongly-typed settings using Pydantic Settings for Azure Blob Storage,
PostgreSQL database connection pooling, and MONAI 3D sliding-window inference.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Tuple
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AzureStorageSettings(BaseSettings):
    """Azure Blob Storage credentials and container configuration."""

    model_config = SettingsConfigDict(env_prefix="AZURE_STORAGE_", extra="ignore")

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
        description="Maximum concurrent connections for blob download/upload operations.",
    )
    timeout_seconds: int = Field(
        default=300,
        description="Timeout for blob transfer operations.",
    )


class DatabaseSettings(BaseSettings):
    """Azure Database for PostgreSQL connection and connection-pool configuration."""

    model_config = SettingsConfigDict(env_prefix="DB_", extra="ignore")

    host: str = Field(
        default="localhost",
        description="PostgreSQL server hostname (e.g., your-azure-db.postgres.database.azure.com).",
    )
    port: int = Field(
        default=5432,
        description="PostgreSQL port number.",
    )
    name: str = Field(
        default="medical_imaging",
        description="Target database name.",
    )
    user: str = Field(
        default="postgres",
        description="Database username.",
    )
    password: str = Field(
        default="postgres",
        description="Database password.",
    )
    sslmode: str = Field(
        default="require",
        description="SSL mode for Azure Database for PostgreSQL (must be 'require' or 'verify-full').",
    )
    pool_size: int = Field(
        default=10,
        description="SQLAlchemy QueuePool base connection pool size.",
    )
    max_overflow: int = Field(
        default=20,
        description="SQLAlchemy QueuePool max overflow connections during peak spikes.",
    )
    pool_timeout: int = Field(
        default=30,
        description="Seconds to wait before timing out on getting a connection from the pool.",
    )
    batch_size: int = Field(
        default=500,
        description="Chunk size for batched bulk inserts into PostgreSQL to avoid OOM.",
    )

    @property
    def connection_url(self) -> str:
        """Construct the psycopg2 connection URL with required Azure SSL parameters."""
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.name}?sslmode={self.sslmode}"
        )


class InferenceSettings(BaseSettings):
    """MONAI 3D sliding-window volumetric inference and model parameters."""

    model_config = SettingsConfigDict(env_prefix="INFERENCE_", extra="ignore")

    modality: Literal["CT", "CBCT", "MRI"] = Field(
        default="CT",
        description="Primary imaging modality (controls intensity windowing and normalization).",
    )
    target_orientation: str = Field(
        default="RAS",
        description="Target spatial orientation code for SimpleITK DICOMOrient (e.g. 'RAS', 'LPS').",
    )
    target_spacing: Tuple[float, float, float] | None = Field(
        default=(1.0, 1.0, 1.0),
        description="Target voxel spacing in mm (x, y, z). Set to None to keep native acquisition spacing.",
    )
    roi_size: Tuple[int, int, int] = Field(
        default=(96, 96, 96),
        description="Volumetric sub-patch ROI size for sliding-window inference (depth, height, width).",
    )
    sw_batch_size: int = Field(
        default=2,
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
        description="Number of input volume channels (1 for grayscale CT/MRI).",
    )
    out_channels: int = Field(
        default=2,
        description="Number of segmentation classes (e.g. 2 for binary foreground/background).",
    )
    model_checkpoint_path: Path | None = Field(
        default=None,
        description="Path to pre-trained PyTorch 3D U-Net checkpoint (.pt or .pth).",
    )
    device: str = Field(
        default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        description="Execution device: 'cuda', 'mps', or 'cpu'.",
    )

    @field_validator("target_orientation")
    @classmethod
    def validate_orientation(cls, v: str) -> str:
        v = v.upper().strip()
        if len(v) != 3 or any(c not in "RLAPSI" for c in v):
            raise ValueError(
                f"Invalid 3D anatomical orientation '{v}'. Must be a 3-letter permutation of R/L, A/P, S/I."
            )
        return v


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
    work_dir: Path = Field(
        default=Path("/tmp/medical_segmentation_workspace"),
        description="Local staging directory for downloaded series and generated volumes.",
    )
    log_level: str = Field(
        default="INFO",
        description="Application logging verbosity: DEBUG, INFO, WARNING, ERROR.",
    )


# Global configuration instance
config = PipelineConfig()
