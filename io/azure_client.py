"""
Azure Blob Storage Client for Ingesting and Exporting 3D Volumetric Medical Images.
Provides resilient streaming downloads, batch series directory downloads, and chunked uploads.
"""

from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List, Optional
from azure.core.exceptions import AzureError, ResourceNotFoundError
from azure.storage.blob import BlobServiceClient, ContentSettings

from medical_segmentation.config import AzureStorageSettings

logger = logging.getLogger(__name__)


class AzureBlobClient:
    """Production Azure Blob Storage manager with resilient error handling and stream I/O."""

    def __init__(self, settings: AzureStorageSettings):
        self.settings = settings
        self._client: Optional[BlobServiceClient] = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the BlobServiceClient using connection string, account key, or fallback to mock mode."""
        if self.settings.connection_string:
            logger.info("Initializing Azure BlobServiceClient via connection string.")
            self._client = BlobServiceClient.from_connection_string(
                self.settings.connection_string,
                max_single_put_size=64 * 1024 * 1024,
                max_single_get_size=64 * 1024 * 1024,
            )
        elif self.settings.account_name and self.settings.account_key:
            account_url = f"https://{self.settings.account_name}.blob.core.windows.net"
            logger.info("Initializing Azure BlobServiceClient via account URL: %s", account_url)
            self._client = BlobServiceClient(
                account_url=account_url,
                credential=self.settings.account_key,
            )
        else:
            logger.warning(
                "No Azure Storage credentials provided. AzureBlobClient operating in offline/mock mode."
            )
            self._client = None

    @property
    def is_connected(self) -> bool:
        """Check whether a live Azure client is active."""
        return self._client is not None

    def download_blob_to_stream(self, container_name: str, blob_name: str) -> io.BytesIO:
        """
        Download a single blob directly into an in-memory BytesIO stream without writing to disk.

        Args:
            container_name: Target Azure container.
            blob_name: Full blob name/path within the container.

        Returns:
            io.BytesIO containing the raw binary volume payload.
        """
        if not self.is_connected:
            raise RuntimeError("Azure Blob Client is not connected to a live storage account.")

        logger.info("Streaming blob 'az://%s/%s' into memory...", container_name, blob_name)
        stream = io.BytesIO()
        try:
            container_client = self._client.get_container_client(container_name)
            blob_client = container_client.get_blob_client(blob_name)
            download_stream = blob_client.download_blob(
                max_concurrency=self.settings.max_concurrency,
                timeout=self.settings.timeout_seconds,
            )
            download_stream.readinto(stream)
            stream.seek(0)
            logger.info(
                "Successfully streamed blob '%s' (%d bytes).", blob_name, stream.getbuffer().nbytes
            )
            return stream
        except ResourceNotFoundError:
            logger.error("Blob '%s' not found in container '%s'.", blob_name, container_name)
            raise
        except AzureError as exc:
            logger.error("Azure Storage error while streaming blob '%s': %s", blob_name, exc)
            raise

    def download_blob_to_file(self, container_name: str, blob_name: str, destination_path: Path) -> Path:
        """
        Download a blob directly to a specified local filesystem path.

        Args:
            container_name: Target container.
            blob_name: Source blob path.
            destination_path: Local destination path.
        """
        if not self.is_connected:
            raise RuntimeError("Azure Blob Client is not connected to a live storage account.")

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading 'az://%s/%s' -> '%s'...", container_name, blob_name, destination_path)

        try:
            container_client = self._client.get_container_client(container_name)
            blob_client = container_client.get_blob_client(blob_name)
            with open(destination_path, "wb") as f:
                blob_data = blob_client.download_blob(
                    max_concurrency=self.settings.max_concurrency,
                    timeout=self.settings.timeout_seconds,
                )
                blob_data.readinto(f)

            logger.info("Download complete: '%s' (%.2f MB)", destination_path, destination_path.stat().st_size / (1024 * 1024))
            return destination_path
        except AzureError as exc:
            logger.error("Failed downloading blob '%s': %s", blob_name, exc)
            raise

    def download_series_prefix(
        self,
        container_name: str,
        prefix: str,
        destination_dir: Path,
    ) -> List[Path]:
        """
        Download an entire DICOM series folder/prefix (multiple .dcm slices) from Azure Blob Storage.

        Args:
            container_name: Target Azure container.
            prefix: Prefix directory in blob storage (e.g., 'studies/123/series_1/').
            destination_dir: Local staging directory to save slices into.

        Returns:
            List of Path objects for downloaded DICOM slice files.
        """
        if not self.is_connected:
            raise RuntimeError("Azure Blob Client is not connected to a live storage account.")

        destination_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading DICOM series from prefix 'az://%s/%s' into '%s'...", container_name, prefix, destination_dir)

        downloaded_files: List[Path] = []
        try:
            container_client = self._client.get_container_client(container_name)
            blobs = list(container_client.list_blobs(name_starts_with=prefix))

            if not blobs:
                logger.warning("No blobs found under prefix '%s' in container '%s'.", prefix, container_name)
                return []

            logger.info("Found %d slice blobs under prefix '%s'. Starting concurrent download.", len(blobs), prefix)

            def _download_single_blob(blob_item: Any) -> Path:
                filename = Path(blob_item.name).name
                dest_file = destination_dir / filename
                blob_client = container_client.get_blob_client(blob_item.name)
                with open(dest_file, "wb") as f:
                    blob_data = blob_client.download_blob(
                        max_concurrency=self.settings.max_concurrency,
                        timeout=self.settings.timeout_seconds,
                    )
                    blob_data.readinto(f)
                return dest_file

            with ThreadPoolExecutor(max_workers=self.settings.max_concurrency) as executor:
                downloaded_files = list(executor.map(_download_single_blob, blobs))

            logger.info("Successfully downloaded %d DICOM slices to '%s'.", len(downloaded_files), destination_dir)
            return downloaded_files
        except AzureError as exc:
            logger.error("Failed downloading DICOM series prefix '%s': %s", prefix, exc)
            raise

    def upload_file_to_blob(
        self,
        container_name: str,
        blob_name: str,
        source_path: Path,
        content_type: str = "application/octet-stream",
        overwrite: bool = True,
    ) -> str:
        """
        Upload a local 3D segmentation mask or metadata artifact to Azure Blob Storage.

        Args:
            container_name: Target container.
            blob_name: Destination blob name.
            source_path: Local file path to upload.
            content_type: MIME content type (e.g. 'application/gzip' or 'application/octet-stream').
            overwrite: Whether to overwrite existing blob.

        Returns:
            The complete Azure Blob URI string.
        """
        if not self.is_connected:
            raise RuntimeError("Azure Blob Client is not connected to a live storage account.")

        if not source_path.is_file():
            raise FileNotFoundError(f"Source file not found for upload: {source_path}")

        logger.info("Uploading '%s' -> 'az://%s/%s'...", source_path, container_name, blob_name)
        try:
            container_client = self._client.get_container_client(container_name)
            if not container_client.exists():
                logger.info("Container '%s' does not exist. Creating container...", container_name)
                container_client.create_container()

            blob_client = container_client.get_blob_client(blob_name)
            with open(source_path, "rb") as data:
                blob_client.upload_blob(
                    data=data,
                    length=source_path.stat().st_size,
                    overwrite=overwrite,
                    content_settings=ContentSettings(content_type=content_type),
                    max_concurrency=self.settings.max_concurrency,
                    timeout=self.settings.timeout_seconds,
                )

            blob_url = blob_client.url
            logger.info("Upload successful: %s", blob_url)
            return blob_url
        except AzureError as exc:
            logger.error("Failed uploading '%s' to Azure Blob Storage: %s", source_path, exc)
            raise

    def list_blobs(self, container_name: str, prefix: str = "") -> List[str]:
        """List all blob paths under an optional prefix."""
        if not self.is_connected:
            return []
        try:
            container_client = self._client.get_container_client(container_name)
            return [b.name for b in container_client.list_blobs(name_starts_with=prefix)]
        except AzureError as exc:
            logger.error("Failed listing blobs in container '%s': %s", container_name, exc)
            raise
