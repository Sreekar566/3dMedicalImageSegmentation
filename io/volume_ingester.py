"""
Volumetric Image Ingestion and Input Source Resolution Engine.
Handles local file/directory verification, cloud-to-local staging for Azure Blob
volumes and DICOM series prefixes, and input source validation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

from medical_segmentation.config import validate_azure_container_name

if TYPE_CHECKING:
    from medical_segmentation.io.azure_client import AzureBlobClient

logger = logging.getLogger(__name__)

__all__ = [
    "VolumeIngester",
    "validate_input_sources",
]


def validate_input_sources(
    blob_path: Optional[str] = None,
    series_prefix: Optional[str] = None,
    local_input: Optional[Path | str] = None,
) -> Tuple[Optional[str], Optional[str], Optional[Path]]:
    """
    Validate that exactly one input volume source is provided among blob_path, series_prefix, and local_input.

    Args:
        blob_path: Optional single 3D volume blob path in Azure Blob Storage.
        series_prefix: Optional prefix for DICOM slice blobs in Azure Blob Storage.
        local_input: Optional local filesystem path to a file or directory.

    Returns:
        Tuple of (clean_blob, clean_prefix, clean_local_path).

    Raises:
        ValueError: If zero or more than one input source is provided.
    """
    clean_blob = blob_path.strip() if isinstance(blob_path, str) and blob_path.strip() else None
    clean_prefix = series_prefix.strip() if isinstance(series_prefix, str) and series_prefix.strip() else None
    clean_local = (
        Path(local_input).resolve()
        if local_input is not None
        and str(local_input).strip()
        and str(local_input).strip() not in (".", "none", "null")
        else None
    )

    inputs_provided = sum(1 for x in (clean_blob, clean_prefix, clean_local) if x is not None)
    if inputs_provided == 0:
        raise ValueError(
            "No input volume source specified (provide exactly one of local_input, series_prefix, or blob_path)."
        )
    if inputs_provided > 1:
        raise ValueError(
            "Conflicting input volume sources specified; provide exactly one of local_input, series_prefix, or blob_path."
        )

    return clean_blob, clean_prefix, clean_local


class VolumeIngester:
    """
    Dedicated volumetric image ingestion engine.
    Resolves input sources (local filesystem paths, Azure Blob files,
    or Azure DICOM series prefixes) and stages them in local workspace storage.
    """

    def __init__(
        self,
        azure_client: Optional[AzureBlobClient] = None,
        workspace: Optional[Path | str] = None,
        default_container: Optional[str] = None,
    ):
        self.azure_client = azure_client
        if workspace is not None:
            if not str(workspace).strip():
                raise ValueError("workspace cannot be an empty path.")
            self.workspace = Path(workspace).resolve()
        else:
            self.workspace = Path("/tmp/medical_segmentation_workspace").resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.default_container = default_container

    @property
    def raw_volumes_dir(self) -> Path:
        """Staging directory for downloaded 3D volume files."""
        return self.workspace / "raw_volumes"

    @property
    def raw_series_dir(self) -> Path:
        """Staging directory for downloaded DICOM slice series."""
        return self.workspace / "raw_series"

    def ingest(
        self,
        blob_path: Optional[str] = None,
        series_prefix: Optional[str] = None,
        local_input: Optional[Path | str] = None,
        input_container: Optional[str] = None,
    ) -> Tuple[Path, bool]:
        """
        Ingest input volume from local filesystem or Azure Blob Storage.

        Args:
            blob_path: Path to single 3D volume blob in Azure.
            series_prefix: Prefix/folder of DICOM slice blobs in Azure.
            local_input: Local file or directory path.
            input_container: Azure input container name.

        Returns:
            Tuple of (resolved_path, is_dicom_dir).
        """
        clean_blob, clean_prefix, clean_local = validate_input_sources(
            blob_path=blob_path,
            series_prefix=series_prefix,
            local_input=local_input,
        )

        container = (
            validate_azure_container_name(input_container)
            if input_container is not None
            else (
                validate_azure_container_name(self.default_container)
                if self.default_container is not None
                else "medical-volumes-input"
            )
        )

        if clean_local is not None:
            if not clean_local.exists():
                raise FileNotFoundError(f"Local input path not found: {clean_local}")
            logger.info("Operating in local mode using path: '%s'", clean_local)
            return clean_local, clean_local.is_dir()

        if self.azure_client is None:
            raise RuntimeError("AzureBlobClient is required for cloud volume ingestion.")

        if clean_prefix is not None:
            logger.info(
                "Downloading DICOM series prefix '%s' from container '%s'...",
                clean_prefix,
                container,
            )
            series_dir = self.raw_series_dir / Path(clean_prefix).name
            self.azure_client.download_series_prefix(
                container_name=container,
                prefix=clean_prefix,
                destination_dir=series_dir,
            )
            return series_dir, True

        assert clean_blob is not None
        logger.info(
            "Downloading volume blob '%s' from container '%s'...",
            clean_blob,
            container,
        )
        local_file = self.raw_volumes_dir / Path(clean_blob).name
        self.azure_client.download_blob_to_file(
            container_name=container,
            blob_name=clean_blob,
            destination_path=local_file,
        )
        return local_file, False
