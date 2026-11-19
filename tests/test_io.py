"""
Unit and Integration Tests for Azure Blob Storage Client, DICOM Series Loader,
and Cloud Ingestion Pipeline Stages.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
from azure.core.exceptions import AzureError, ResourceNotFoundError
from pydicom.dataset import Dataset, FileMetaDataset
from sqlalchemy import create_engine

from medical_segmentation.config import AzureStorageSettings, DatabaseSettings, PipelineConfig
from medical_segmentation.db.database import DatabaseManager
from medical_segmentation.io.azure_client import AzureBlobClient
from medical_segmentation.io.dicom_loader import (
    DicomVolumeLoader,
    VolumetricMetadata,
    extract_pydicom_metadata,
    safe_parse_float,
    safe_parse_str,
    sitk_to_numpy,
)
from medical_segmentation.io.mask_writer import VolumetricMaskWriter, save_mask_as_nifti
from medical_segmentation.io.volume_ingester import VolumeIngester, validate_input_sources
from medical_segmentation.pipeline.orchestrator import PipelineOrchestrator, PipelineResult


# =====================================================================
# Helpers
# =====================================================================


def _create_synthetic_dicom_slice(
    path: Path,
    patient_id: str = "PATIENT_TEST_01",
    study_uid: str = "1.2.840.10008.1.1",
    series_uid: str = "1.2.840.10008.1.2",
    sop_uid: Optional[str] = None,
    slice_idx: int = 0,
    rows: int = 16,
    cols: int = 16,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 2.5),
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    modality: str = "CT",
    window_center: Any = 40.0,
    window_width: Any = 400.0,
) -> Path:
    """Create a valid synthetic DICOM slice file on disk for testing."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid or pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = file_meta
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.PatientID = patient_id
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.Modality = modality
    ds.BodyPartExamined = "CHEST"
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelSpacing = [str(spacing[0]), str(spacing[1])]
    slice_z = origin[2] + slice_idx * spacing[2]
    ds.ImagePositionPatient = [str(origin[0]), str(origin[1]), str(slice_z)]
    ds.ImageOrientationPatient = ["1", "0", "0", "0", "1", "0"]
    ds.RescaleIntercept = "0"
    ds.RescaleSlope = "1"

    if isinstance(window_center, list):
        ds.WindowCenter = [str(wc) for wc in window_center]
    elif window_center is not None:
        ds.WindowCenter = str(window_center)

    if isinstance(window_width, list):
        ds.WindowWidth = [str(ww) for ww in window_width]
    elif window_width is not None:
        ds.WindowWidth = str(window_width)

    pixel_array = np.full((rows, cols), fill_value=slice_idx * 15 + 100, dtype=np.uint16)
    ds.PixelData = pixel_array.tobytes()

    pydicom.dcmwrite(str(path), ds, enforce_file_format=True)
    return path


def _make_connected_client(
    max_concurrency: int = 4,
    timeout_seconds: int = 300,
) -> Tuple[AzureBlobClient, MagicMock]:
    """Instantiate an AzureBlobClient with a mocked active BlobServiceClient."""
    settings = AzureStorageSettings(
        connection_string="",
        max_concurrency=max_concurrency,
        timeout_seconds=timeout_seconds,
    )
    client = AzureBlobClient(settings)
    mock_bsc = MagicMock()
    client._client = mock_bsc
    return client, mock_bsc


# =====================================================================
# AzureBlobClient Unit Tests
# =====================================================================


def test_azure_client_offline_mode_when_no_credentials() -> None:
    """Validate that AzureBlobClient initializes in offline/mock mode when no credentials are provided."""
    settings = AzureStorageSettings(connection_string="", account_name="", account_key="")
    client = AzureBlobClient(settings)

    assert not client.is_connected
    assert client._client is None
    assert client.list_blobs("any-container") == []


def test_azure_client_init_via_connection_string() -> None:
    """Validate that AzureBlobClient properly initializes BlobServiceClient from connection string."""
    conn_str = "DefaultEndpointsProtocol=https;AccountName=testacc;AccountKey=fakekey;EndpointSuffix=core.windows.net"
    settings = AzureStorageSettings(connection_string=conn_str)

    with patch("medical_segmentation.io.azure_client.BlobServiceClient") as mock_bsc:
        mock_instance = MagicMock()
        mock_bsc.from_connection_string.return_value = mock_instance

        client = AzureBlobClient(settings)

        assert client.is_connected
        mock_bsc.from_connection_string.assert_called_once_with(
            conn_str,
            max_single_put_size=64 * 1024 * 1024,
            max_single_get_size=64 * 1024 * 1024,
        )


def test_azure_client_init_via_account_key() -> None:
    """Validate that AzureBlobClient properly initializes BlobServiceClient from account name and key."""
    settings = AzureStorageSettings(
        connection_string="",
        account_name="medicalstorage",
        account_key="secretkey123",
    )

    with patch("medical_segmentation.io.azure_client.BlobServiceClient") as mock_bsc:
        mock_instance = MagicMock()
        mock_bsc.return_value = mock_instance

        client = AzureBlobClient(settings)

        assert client.is_connected
        mock_bsc.assert_called_once_with(
            account_url="https://medicalstorage.blob.core.windows.net",
            credential="secretkey123",
        )


def test_azure_client_offline_methods_raise_runtime_error(tmp_path: Path) -> None:
    """Verify that attempting blob operations on an offline client raises RuntimeError."""
    client = AzureBlobClient(AzureStorageSettings())
    assert not client.is_connected

    with pytest.raises(RuntimeError, match="not connected"):
        client.download_blob_to_stream("container", "blob.nii.gz")

    with pytest.raises(RuntimeError, match="not connected"):
        client.download_blob_to_file("container", "blob.nii.gz", tmp_path / "dest.nii.gz")

    with pytest.raises(RuntimeError, match="not connected"):
        client.download_series_prefix("container", "prefix/", tmp_path / "series")

    dummy_file = tmp_path / "dummy.txt"
    dummy_file.write_text("content")
    with pytest.raises(RuntimeError, match="not connected"):
        client.upload_file_to_blob("container", "blob.txt", dummy_file)


def test_azure_client_download_blob_to_stream() -> None:
    """Verify downloading a blob directly into an in-memory BytesIO stream."""
    client, mock_bsc = _make_connected_client()

    mock_cc = MagicMock()
    mock_bc = MagicMock()
    mock_downloader = MagicMock()

    payload = b"3D_VOLUMETRIC_IMAGE_DATA_BYTES"
    mock_downloader.readinto.side_effect = lambda buf: buf.write(payload)

    mock_bsc.get_container_client.return_value = mock_cc
    mock_cc.get_blob_client.return_value = mock_bc
    mock_bc.download_blob.return_value = mock_downloader

    stream = client.download_blob_to_stream("med-container", "study/vol.nii.gz")

    assert isinstance(stream, io.BytesIO)
    assert stream.getvalue() == payload
    assert stream.tell() == 0
    mock_bsc.get_container_client.assert_called_once_with("med-container")
    mock_cc.get_blob_client.assert_called_once_with("study/vol.nii.gz")


def test_azure_client_download_blob_to_stream_exceptions() -> None:
    """Verify download_blob_to_stream propagates ResourceNotFoundError and AzureError."""
    client, mock_bsc = _make_connected_client()

    mock_cc = MagicMock()
    mock_bc = MagicMock()

    mock_bsc.get_container_client.return_value = mock_cc
    mock_cc.get_blob_client.return_value = mock_bc

    # 1. ResourceNotFoundError
    mock_bc.download_blob.side_effect = ResourceNotFoundError("Blob not found")
    with pytest.raises(ResourceNotFoundError):
        client.download_blob_to_stream("med-container", "missing.nii.gz")

    # 2. AzureError
    mock_bc.download_blob.side_effect = AzureError("Network connection reset")
    with pytest.raises(AzureError):
        client.download_blob_to_stream("med-container", "error.nii.gz")


def test_azure_client_download_blob_to_file(tmp_path: Path) -> None:
    """Verify downloading a blob to local filesystem, ensuring directory creation and file persistence."""
    client, mock_bsc = _make_connected_client()

    mock_cc = MagicMock()
    mock_bc = MagicMock()
    mock_downloader = MagicMock()

    payload = b"RAW_NIFTI_GZ_BINARY_PAYLOAD"
    mock_downloader.readinto.side_effect = lambda f: f.write(payload)

    mock_bsc.get_container_client.return_value = mock_cc
    mock_cc.get_blob_client.return_value = mock_bc
    mock_bc.download_blob.return_value = mock_downloader

    dest_file = tmp_path / "deep" / "nested" / "folder" / "volume.nii.gz"
    result_path = client.download_blob_to_file("med-container", "volume.nii.gz", dest_file)

    assert result_path == dest_file
    assert dest_file.is_file()
    assert dest_file.read_bytes() == payload

    # Error propagation
    mock_bc.download_blob.side_effect = AzureError("Download failed")
    with pytest.raises(AzureError):
        client.download_blob_to_file("med-container", "volume.nii.gz", dest_file)


def test_azure_client_download_series_prefix(tmp_path: Path) -> None:
    """Verify multi-threaded batch download of DICOM series slices matching a blob prefix."""
    client, mock_bsc = _make_connected_client(max_concurrency=2)
    mock_cc = MagicMock()
    mock_bsc.get_container_client.return_value = mock_cc

    # 1. Empty prefix case
    mock_cc.list_blobs.return_value = []
    series_dest = tmp_path / "series_empty"
    result = client.download_series_prefix("med-container", "empty_series/", series_dest)
    assert result == []

    # 2. Multiple slice blobs case
    slice_names = ["series_01/slice_001.dcm", "series_01/slice_002.dcm", "series_01/slice_003.dcm"]
    mock_blobs = []
    for name in slice_names:
        b = MagicMock()
        b.name = name
        mock_blobs.append(b)
    mock_cc.list_blobs.return_value = mock_blobs

    def _get_blob_client_mock(blob_name: str) -> MagicMock:
        bc = MagicMock()
        dl = MagicMock()
        dl.readinto.side_effect = lambda f, bn=blob_name: f.write(f"DATA_FOR_{bn}".encode())
        bc.download_blob.return_value = dl
        return bc

    mock_cc.get_blob_client.side_effect = _get_blob_client_mock

    series_dest = tmp_path / "series_downloaded"
    downloaded_files = client.download_series_prefix("med-container", "series_01/", series_dest)

    assert len(downloaded_files) == 3
    for s_name in slice_names:
        expected_file = series_dest / Path(s_name).name
        assert expected_file in downloaded_files
        assert expected_file.is_file()
        assert expected_file.read_text() == f"DATA_FOR_{s_name}"

    # 3. Error propagation
    mock_cc.list_blobs.side_effect = AzureError("Container listing failed")
    with pytest.raises(AzureError):
        client.download_series_prefix("med-container", "error_prefix/", series_dest)


def test_azure_client_upload_file_to_blob(tmp_path: Path) -> None:
    """Verify file upload to Azure Blob Storage including container creation check and content settings."""
    client, mock_bsc = _make_connected_client()

    mock_cc = MagicMock()
    mock_bc = MagicMock()
    mock_bc.url = "https://testacc.blob.core.windows.net/med-output/mask.nii.gz"

    mock_bsc.get_container_client.return_value = mock_cc
    mock_cc.get_blob_client.return_value = mock_bc

    source_file = tmp_path / "mask.nii.gz"
    source_file.write_bytes(b"GZIPPED_MASK_DATA")

    # Case 1: Non-existent source file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        client.upload_file_to_blob("med-output", "mask.nii.gz", tmp_path / "missing.nii.gz")

    # Case 2: Container does not exist -> creates container then uploads
    mock_cc.exists.return_value = False
    url = client.upload_file_to_blob(
        container_name="med-output",
        blob_name="mask.nii.gz",
        source_path=source_file,
        content_type="application/gzip",
    )
    assert url == mock_bc.url
    mock_cc.create_container.assert_called_once()
    mock_bc.upload_blob.assert_called_once()

    # Case 3: Container already exists -> does not call create_container
    mock_cc.reset_mock()
    mock_bc.reset_mock()
    mock_cc.exists.return_value = True

    url2 = client.upload_file_to_blob(
        container_name="med-output",
        blob_name="mask.nii.gz",
        source_path=source_file,
    )
    assert url2 == mock_bc.url
    mock_cc.create_container.assert_not_called()
    mock_bc.upload_blob.assert_called_once()

    # Case 4: AzureError propagation
    mock_bc.upload_blob.side_effect = AzureError("Upload rejected by service")
    with pytest.raises(AzureError):
        client.upload_file_to_blob("med-output", "mask.nii.gz", source_file)


def test_azure_client_list_blobs() -> None:
    """Verify list_blobs enumerates blob names matching prefix and handles exceptions."""
    client, mock_bsc = _make_connected_client()
    mock_cc = MagicMock()
    mock_bsc.get_container_client.return_value = mock_cc

    b1 = MagicMock()
    b1.name = "studies/01/slice_1.dcm"
    b2 = MagicMock()
    b2.name = "studies/01/slice_2.dcm"
    mock_cc.list_blobs.return_value = [b1, b2]

    blobs = client.list_blobs("med-container", prefix="studies/01/")
    assert blobs == ["studies/01/slice_1.dcm", "studies/01/slice_2.dcm"]
    mock_cc.list_blobs.assert_called_once_with(name_starts_with="studies/01/")

    # Exception propagation
    mock_cc.list_blobs.side_effect = AzureError("Permission denied")
    with pytest.raises(AzureError):
        client.list_blobs("med-container", prefix="forbidden/")


# =====================================================================
# DicomVolumeLoader Unit Tests
# =====================================================================


def test_dicom_volume_loader_missing_dir_raises_file_not_found(tmp_path: Path) -> None:
    """Verify load_dicom_series raises FileNotFoundError on missing directory."""
    loader = DicomVolumeLoader()
    with pytest.raises(FileNotFoundError, match="DICOM series directory not found"):
        loader.load_dicom_series(tmp_path / "non_existent_dir")


def test_dicom_volume_loader_empty_dir_raises_value_error(tmp_path: Path) -> None:
    """Verify load_dicom_series raises ValueError when directory has no DICOM files."""
    loader = DicomVolumeLoader()
    empty_dir = tmp_path / "empty_dicom_dir"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="No valid DICOM series or .dcm files found"):
        loader.load_dicom_series(empty_dir)


def test_dicom_volume_loader_synthetic_series_load(tmp_path: Path) -> None:
    """Verify load_dicom_series reads synthetic DICOM series, extracts tags, and standardizes orientation."""
    series_dir = tmp_path / "dicom_series_ct"
    series_dir.mkdir()

    # Create 3 slices
    for idx in range(3):
        _create_synthetic_dicom_slice(
            path=series_dir / f"slice_{idx:03d}.dcm",
            patient_id="PATIENT_42",
            study_uid="1.2.3.4.42",
            series_uid="1.2.3.4.42.1",
            slice_idx=idx,
            rows=16,
            cols=16,
            spacing=(1.5, 1.5, 2.0),
            origin=(10.0, 20.0, 30.0),
            modality="CT",
            window_center=[40.0, 80.0],  # Multi-value WindowCenter
            window_width=[350.0, 700.0],  # Multi-value WindowWidth
        )

    loader = DicomVolumeLoader(target_orientation="RAS")
    sitk_image, metadata = loader.load_dicom_series(series_dir)

    assert isinstance(sitk_image, sitk.Image)
    assert sitk_image.GetSize() == (16, 16, 3)
    assert np.allclose(sitk_image.GetSpacing(), (1.5, 1.5, 2.0))
    assert metadata.patient_id == "PATIENT_42"
    assert metadata.study_instance_uid == "1.2.3.4.42"
    assert metadata.series_instance_uid == "1.2.3.4.42.1"
    assert metadata.modality == "CT"
    assert metadata.body_part == "CHEST"
    assert metadata.window_center == 40.0
    assert metadata.window_width == 350.0
    assert metadata.spatial_orientation == "RAS"


def test_dicom_volume_loader_direct_dcm_fallback(tmp_path: Path) -> None:
    """Verify fallback to direct *.dcm file reading when GetGDCMSeriesIDs returns empty."""
    series_dir = tmp_path / "direct_dcm_dir"
    series_dir.mkdir()

    for idx in range(2):
        _create_synthetic_dicom_slice(
            path=series_dir / f"image_{idx:02d}.dcm",
            slice_idx=idx,
            rows=8,
            cols=8,
        )

    loader = DicomVolumeLoader(target_orientation="RAS")

    # Mock GetGDCMSeriesIDs to return empty tuple, triggering the *.dcm glob fallback
    with patch.object(sitk.ImageSeriesReader, "GetGDCMSeriesIDs", return_value=()):
        sitk_image, metadata = loader.load_dicom_series(series_dir)
        assert isinstance(sitk_image, sitk.Image)
        assert sitk_image.GetSize() == (8, 8, 2)


def test_dicom_volume_loader_extract_pydicom_metadata_corrupt_file(tmp_path: Path) -> None:
    """Verify _extract_pydicom_metadata gracefully returns default VolumetricMetadata on corrupt slice."""
    corrupt_file = tmp_path / "corrupt_slice.dcm"
    corrupt_file.write_bytes(b"NOT_A_VALID_DICOM_FILE_HEADER")

    loader = DicomVolumeLoader()
    meta = loader._extract_pydicom_metadata(corrupt_file)

    assert isinstance(meta, VolumetricMetadata)
    assert meta.patient_id == "UNKNOWN"
    assert meta.study_instance_uid == ""


def test_dicom_volume_loader_load_volume_file(tmp_path: Path) -> None:
    """Verify load_volume_file reads 3D NIfTI volumes and detects CT vs MRI modality from filename."""
    loader = DicomVolumeLoader(target_orientation="RAS")

    # Missing file raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Volume file not found"):
        loader.load_volume_file(tmp_path / "missing.nii.gz")

    # Create synthetic CT volume
    arr = np.zeros((10, 12, 14), dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 2.0))
    img.SetOrigin((5.0, 10.0, 15.0))

    ct_path = tmp_path / "patient_ct_chest.nii.gz"
    sitk.WriteImage(img, str(ct_path))

    oriented_ct, meta_ct = loader.load_volume_file(ct_path)
    assert oriented_ct.GetSize() == (14, 12, 10)
    assert meta_ct.modality == "CT"

    # Create synthetic MRI volume
    mri_path = tmp_path / "patient_mri_brain.nii.gz"
    sitk.WriteImage(img, str(mri_path))

    oriented_mri, meta_mri = loader.load_volume_file(mri_path)
    assert oriented_mri.GetSize() == (14, 12, 10)
    assert meta_mri.modality == "MRI"


def test_dicom_volume_loader_save_mask_as_nifti(tmp_path: Path) -> None:
    """Verify save_mask_as_nifti persists mask preserving geometry and orientation."""
    ref_arr = np.zeros((8, 16, 24), dtype=np.float32)
    ref_img = sitk.GetImageFromArray(ref_arr)
    ref_img.SetSpacing((1.2, 1.4, 2.5))
    ref_img.SetOrigin((15.0, 25.0, 35.0))
    ref_img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    mask_arr = np.ones((8, 16, 24), dtype=np.uint8)
    mask_arr[0:2, :, :] = 0

    output_path = tmp_path / "nested" / "saved_mask.nii.gz"
    saved_path = DicomVolumeLoader.save_mask_as_nifti(
        mask_array=mask_arr,
        reference_image=ref_img,
        output_path=output_path,
    )

    assert saved_path == output_path
    assert output_path.is_file()

    # Read back and compare
    saved_img = sitk.ReadImage(str(output_path))
    assert saved_img.GetSize() == ref_img.GetSize()
    assert np.allclose(saved_img.GetSpacing(), ref_img.GetSpacing())
    assert np.allclose(saved_img.GetOrigin(), ref_img.GetOrigin())
    assert np.allclose(saved_img.GetDirection(), ref_img.GetDirection())
    loaded_mask_arr = sitk.GetArrayFromImage(saved_img)
    assert np.array_equal(loaded_mask_arr, mask_arr)


# =====================================================================
# Pipeline Orchestrator Azure Ingestion & Export Tests
# =====================================================================


def test_orchestrator_azure_volume_ingestion(tmp_path: Path) -> None:
    """Verify that ingest_volume correctly dispatches Azure blob and series prefix downloads."""
    mock_azure = MagicMock(spec=AzureBlobClient)
    mock_azure.is_connected = True

    workspace = tmp_path / "test_workspace"
    orchestrator = PipelineOrchestrator(
        azure_client=mock_azure,
        workspace=workspace,
    )

    # 1. blob_path download
    mock_azure.download_blob_to_file.return_value = workspace / "raw_volumes" / "test.nii.gz"
    target_path, is_dir = orchestrator.ingest_volume(
        blob_path="inputs/test.nii.gz",
        input_container="custom-container",
    )
    assert not is_dir
    assert target_path == workspace / "raw_volumes" / "test.nii.gz"
    mock_azure.download_blob_to_file.assert_called_once_with(
        container_name="custom-container",
        blob_name="inputs/test.nii.gz",
        destination_path=workspace / "raw_volumes" / "test.nii.gz",
    )

    # 2. series_prefix download
    mock_azure.reset_mock()
    target_dir, is_dir_series = orchestrator.ingest_volume(
        series_prefix="series/series_001",
        input_container="custom-container",
    )
    assert is_dir_series
    assert target_dir == workspace / "raw_series" / "series_001"
    mock_azure.download_series_prefix.assert_called_once_with(
        container_name="custom-container",
        prefix="series/series_001",
        destination_dir=workspace / "raw_series" / "series_001",
    )


def test_orchestrator_azure_export_mask_upload(tmp_path: Path) -> None:
    """Verify export_mask uploads to Azure Blob Storage when not in dry-run mode and client is connected."""
    mock_azure = MagicMock(spec=AzureBlobClient)
    mock_azure.is_connected = True
    mock_azure.upload_file_to_blob.return_value = "https://azure.blob/output/mask.nii.gz"

    workspace = tmp_path / "test_workspace"
    orchestrator = PipelineOrchestrator(
        azure_client=mock_azure,
        workspace=workspace,
    )

    ref_img = sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32))
    mask_arr = np.zeros((8, 8, 8), dtype=np.uint8)
    meta = VolumetricMetadata(study_instance_uid="STUDY_AZURE")

    # Case 1: Upload triggered when dry_run=False and is_connected=True
    local_path, blob_uri = orchestrator.export_mask(
        mask_array=mask_arr,
        reference_image=ref_img,
        metadata=meta,
        dry_run=False,
    )
    assert local_path.is_file()
    assert blob_uri == "https://azure.blob/output/mask.nii.gz"
    mock_azure.upload_file_to_blob.assert_called_once()

    # Case 2: Upload skipped when dry_run=True
    mock_azure.reset_mock()
    local_path_dry, blob_uri_dry = orchestrator.export_mask(
        mask_array=mask_arr,
        reference_image=ref_img,
        metadata=meta,
        dry_run=True,
    )
    assert local_path_dry.is_file()
    assert blob_uri_dry is None
    mock_azure.upload_file_to_blob.assert_not_called()


def test_orchestrator_end_to_end_azure_blob_run(tmp_path: Path) -> None:
    """Verify complete end-to-end pipeline run ingesting a volume blob from Azure and exporting mask."""
    workspace = tmp_path / "workspace_azure_blob"
    raw_vols = workspace / "raw_volumes"
    raw_vols.mkdir(parents=True, exist_ok=True)

    arr = np.random.uniform(low=-100.0, high=200.0, size=(32, 32, 32)).astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    vol_path = raw_vols / "patient_ct_volume.nii.gz"
    sitk.WriteImage(img, str(vol_path))

    mock_azure = MagicMock(spec=AzureBlobClient)
    mock_azure.is_connected = True
    mock_azure.download_blob_to_file.return_value = vol_path
    mock_azure.upload_file_to_blob.return_value = "https://medicalstorage.blob.core.windows.net/output/mask.nii.gz"

    engine = create_engine("sqlite:///:memory:")
    db_mgr = DatabaseManager(DatabaseSettings(), engine=engine)

    cfg = PipelineConfig()
    cfg.work_dir = workspace
    cfg.inference.roi_size = (32, 32, 32)
    cfg.inference.target_spacing = None
    cfg.inference.device = "cpu"

    orchestrator = PipelineOrchestrator(
        config=cfg,
        azure_client=mock_azure,
        db_manager=db_mgr,
        workspace=workspace,
    )

    result = orchestrator.run(
        blob_path="inputs/patient_ct_volume.nii.gz",
        modality="CT",
        dry_run=False,
    )

    assert isinstance(result, PipelineResult)
    assert result.output_blob_uri == "https://medicalstorage.blob.core.windows.net/output/mask.nii.gz"
    assert result.local_mask_path.is_file()
    assert result.study_id is not None
    assert result.run_id is not None
    assert result.mask_array.shape == (32, 32, 32)
    mock_azure.download_blob_to_file.assert_called_once()
    mock_azure.upload_file_to_blob.assert_called_once()


def test_orchestrator_end_to_end_azure_series_prefix_run(tmp_path: Path) -> None:
    """Verify complete end-to-end pipeline run downloading a DICOM series from Azure prefix and exporting mask."""
    workspace = tmp_path / "workspace_azure_series"
    series_dir = workspace / "raw_series" / "series_ct_001"
    series_dir.mkdir(parents=True, exist_ok=True)

    downloaded_slices = []
    for i in range(32):
        slice_path = _create_synthetic_dicom_slice(
            path=series_dir / f"slice_{i:03d}.dcm",
            slice_idx=i,
            rows=32,
            cols=32,
            spacing=(1.0, 1.0, 1.0),
        )
        downloaded_slices.append(slice_path)

    mock_azure = MagicMock(spec=AzureBlobClient)
    mock_azure.is_connected = True
    mock_azure.download_series_prefix.return_value = downloaded_slices
    mock_azure.upload_file_to_blob.return_value = "https://medicalstorage.blob.core.windows.net/output/mask_series.nii.gz"

    engine = create_engine("sqlite:///:memory:")
    db_mgr = DatabaseManager(DatabaseSettings(), engine=engine)

    cfg = PipelineConfig()
    cfg.work_dir = workspace
    cfg.inference.roi_size = (32, 32, 32)
    cfg.inference.target_spacing = None
    cfg.inference.device = "cpu"

    orchestrator = PipelineOrchestrator(
        config=cfg,
        azure_client=mock_azure,
        db_manager=db_mgr,
        workspace=workspace,
    )

    result = orchestrator.run(
        series_prefix="dicom_studies/series_ct_001",
        modality="CT",
        dry_run=False,
    )

    assert isinstance(result, PipelineResult)
    assert result.output_blob_uri == "https://medicalstorage.blob.core.windows.net/output/mask_series.nii.gz"
    assert result.local_mask_path.is_file()
    assert result.study_id is not None
    assert result.run_id is not None
    assert result.mask_array.shape == (32, 32, 32)
    mock_azure.download_series_prefix.assert_called_once()
    mock_azure.upload_file_to_blob.assert_called_once()


def test_safe_parse_float_and_str_edge_cases() -> None:
    """Verify safe_parse_float and safe_parse_str handle edge cases robustly."""
    # safe_parse_float
    assert safe_parse_float(None, 0.0) == 0.0
    assert safe_parse_float("", 0.0) == 0.0
    assert safe_parse_float("   ", 0.0) == 0.0
    assert safe_parse_float("none", 0.0) == 0.0
    assert safe_parse_float("null", 0.0) == 0.0
    assert safe_parse_float([], 0.0) == 0.0
    assert safe_parse_float([None], 0.0) == 0.0
    assert safe_parse_float([""], 0.0) == 0.0
    assert safe_parse_float(["40.0", "80.0"], 0.0) == 40.0
    assert safe_parse_float("40.0\\80.0", 0.0) == 40.0
    assert safe_parse_float(40) == 40.0
    assert safe_parse_float(40.5) == 40.5
    assert safe_parse_float("not_a_float", 0.0) == 0.0
    assert safe_parse_float(None) is None

    # safe_parse_str
    assert safe_parse_str(None, "ANONYMOUS") == "ANONYMOUS"
    assert safe_parse_str("", "ANONYMOUS") == "ANONYMOUS"
    assert safe_parse_str("   ", "ANONYMOUS") == "ANONYMOUS"
    assert safe_parse_str("None", "ANONYMOUS") == "ANONYMOUS"
    assert safe_parse_str("null", "ANONYMOUS") == "ANONYMOUS"
    assert safe_parse_str("  PATIENT_123  ", "ANONYMOUS") == "PATIENT_123"
    assert safe_parse_str("CT", "CT") == "CT"


def test_dicom_volume_loader_extract_pydicom_metadata_empty_and_malformed_tags(tmp_path: Path) -> None:
    """Verify _extract_pydicom_metadata preserves clinical metadata when window/rescale tags are empty or malformed."""
    slice_path = tmp_path / "slice_empty_tags.dcm"

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = file_meta
    ds.PatientID = "REAL_PATIENT_123"
    ds.StudyInstanceUID = "1.2.3.4.5"
    ds.SeriesInstanceUID = "1.2.3.4.5.6"
    ds.Modality = "CT"
    ds.BodyPartExamined = "CHEST"
    ds.SliceThickness = "2.5"
    ds.PatientSex = "M"
    # Empty tags that previously triggered TypeError / IndexError and wiped out all metadata
    ds.WindowCenter = []
    ds.WindowWidth = ""

    pydicom.dcmwrite(str(slice_path), ds, enforce_file_format=True)

    loader = DicomVolumeLoader()
    meta = loader._extract_pydicom_metadata(slice_path)

    assert meta.patient_id == "REAL_PATIENT_123"
    assert meta.study_instance_uid == "1.2.3.4.5"
    assert meta.series_instance_uid == "1.2.3.4.5.6"
    assert meta.modality == "CT"
    assert meta.body_part == "CHEST"
    assert meta.window_center is None
    assert meta.window_width is None
    assert meta.rescale_intercept == 0.0
    assert meta.rescale_slope == 1.0
    assert meta.extra_tags.get("SliceThickness") == "2.5"
    assert meta.extra_tags.get("PatientSex") == "M"


def test_dicom_volume_loader_load_volume_file_clean_stem(tmp_path: Path) -> None:
    """Verify load_volume_file strips .nii.gz properly so study_instance_uid has no .nii suffix."""
    arr = np.zeros((8, 8, 8), dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    nii_path = tmp_path / "scan_patient_99.nii.gz"
    sitk.WriteImage(img, str(nii_path))

    loader = DicomVolumeLoader()
    _, meta = loader.load_volume_file(nii_path)

    assert meta.study_instance_uid == "scan_patient_99"
    assert not meta.study_instance_uid.endswith(".nii")


def test_dicom_volume_loader_load_volume_file_single_dcm(tmp_path: Path) -> None:
    """Verify load_volume_file extracts DICOM tags when given a standalone .dcm file."""
    slice_path = tmp_path / "single_slice.dcm"
    _create_synthetic_dicom_slice(
        path=slice_path,
        patient_id="PATIENT_STANDALONE",
        study_uid="1.2.999.1",
        series_uid="1.2.999.2",
    )

    loader = DicomVolumeLoader()
    sitk_img, meta = loader.load_volume_file(slice_path)

    assert isinstance(sitk_img, sitk.Image)
    assert meta.patient_id == "PATIENT_STANDALONE"
    assert meta.study_instance_uid == "1.2.999.1"
    assert meta.series_instance_uid == "1.2.999.2"


def test_dicom_volume_loader_save_mask_as_nifti_large_label_dtype(tmp_path: Path) -> None:
    """Verify save_mask_as_nifti supports multi-label masks > 255 without uint8 overflow."""
    ref_img = sitk.GetImageFromArray(np.zeros((4, 4, 4), dtype=np.float32))
    ref_img.SetSpacing((1.0, 1.0, 1.0))

    # Mask with label 300 (would wrap to 44 with uint8)
    mask_arr = np.zeros((4, 4, 4), dtype=np.int32)
    mask_arr[1, 1, 1] = 300

    out_file = tmp_path / "mask_uint16.nii.gz"
    DicomVolumeLoader.save_mask_as_nifti(mask_arr, ref_img, out_file)

    loaded_img = sitk.ReadImage(str(out_file))
    loaded_arr = sitk.GetArrayFromImage(loaded_img)
    assert loaded_arr[1, 1, 1] == 300
    assert np.array_equal(loaded_arr, mask_arr)


def test_azure_client_string_path_support(tmp_path: Path) -> None:
    """Verify download_blob_to_file, download_series_prefix, and upload_file_to_blob accept str paths."""
    client, mock_bsc = _make_connected_client()
    mock_cc = MagicMock()
    mock_bc = MagicMock()
    mock_bsc.get_container_client.return_value = mock_cc
    mock_cc.get_blob_client.return_value = mock_bc

    # download_blob_to_file with str path
    str_dest = str(tmp_path / "str_dest" / "vol.nii.gz")
    res = client.download_blob_to_file("med-container", "vol.nii.gz", str_dest)
    assert isinstance(res, Path)
    assert res == Path(str_dest).resolve()

    # download_series_prefix with str dir
    mock_blob = MagicMock()
    mock_blob.name = "series/slice_01.dcm"
    mock_cc.list_blobs.return_value = [mock_blob]
    str_dir = str(tmp_path / "str_series")
    downloaded = client.download_series_prefix("med-container", "series/", str_dir)
    assert len(downloaded) == 1
    assert isinstance(downloaded[0], Path)

    # upload_file_to_blob with str path
    dummy_file = tmp_path / "upload_target.nii.gz"
    dummy_file.write_bytes(b"DATA")
    mock_bc.url = "https://mock.blob.core.windows.net/med-container/upload_target.nii.gz"
    url = client.upload_file_to_blob("med-container", "upload_target.nii.gz", str(dummy_file))
    assert url == mock_bc.url


def test_mask_writer_standalone_save_mask_as_nifti_with_sitk_image(tmp_path: Path) -> None:
    """Verify standalone save_mask_as_nifti saves mask accurately with sitk.Image reference."""
    ref_arr = np.zeros((6, 12, 18), dtype=np.float32)
    ref_img = sitk.GetImageFromArray(ref_arr)
    ref_img.SetSpacing((1.1, 1.2, 2.0))
    ref_img.SetOrigin((5.0, 15.0, 25.0))

    mask_arr = np.ones((6, 12, 18), dtype=np.uint8)
    out_file = tmp_path / "mask_sitk_ref.nii.gz"

    result_path = save_mask_as_nifti(
        mask_array=mask_arr,
        reference_image=ref_img,
        output_path=out_file,
    )
    assert result_path == out_file
    assert out_file.is_file()

    loaded = sitk.ReadImage(str(out_file))
    assert loaded.GetSize() == (18, 12, 6)
    assert np.allclose(loaded.GetSpacing(), (1.1, 1.2, 2.0))
    assert np.allclose(loaded.GetOrigin(), (5.0, 15.0, 25.0))
    assert np.array_equal(sitk.GetArrayFromImage(loaded), mask_arr)


def test_mask_writer_standalone_save_mask_as_nifti_with_volumetric_metadata(tmp_path: Path) -> None:
    """Verify save_mask_as_nifti saves mask accurately using VolumetricMetadata reference geometry."""
    metadata = VolumetricMetadata(
        dimensions=(15, 10, 5),  # (X, Y, Z)
        spacing=(0.8, 0.8, 1.5),
        origin=(10.0, 20.0, 30.0),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        spatial_orientation="RAS",
    )

    mask_arr = np.zeros((5, 10, 15), dtype=np.uint8)  # (Z, Y, X)
    mask_arr[2, 5, 7] = 1

    out_file = tmp_path / "mask_meta_ref.nii.gz"
    result_path = save_mask_as_nifti(
        mask_array=mask_arr,
        reference=metadata,
        output_path=out_file,
    )
    assert result_path == out_file
    assert out_file.is_file()

    loaded = sitk.ReadImage(str(out_file))
    assert loaded.GetSize() == (15, 10, 5)
    assert np.allclose(loaded.GetSpacing(), (0.8, 0.8, 1.5))
    assert np.allclose(loaded.GetOrigin(), (10.0, 20.0, 30.0))
    assert np.allclose(loaded.GetDirection(), (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    assert np.array_equal(sitk.GetArrayFromImage(loaded), mask_arr)


def test_volumetric_mask_writer_class(tmp_path: Path) -> None:
    """Verify VolumetricMaskWriter class manages default output directory and filename routing."""
    out_dir = tmp_path / "staging_masks"
    writer = VolumetricMaskWriter(default_output_dir=out_dir)
    assert out_dir.is_dir()

    ref_img = sitk.GetImageFromArray(np.zeros((4, 8, 8), dtype=np.float32))
    ref_img.SetSpacing((1.0, 1.0, 1.0))
    mask_arr = np.zeros((4, 8, 8), dtype=np.uint8)

    # 1. Write via filename relative to default_output_dir
    saved = writer.write_mask(mask_arr, reference=ref_img, filename="pred.nii.gz")
    assert saved == out_dir / "pred.nii.gz"
    assert saved.is_file()

    # 2. Write via explicit output_path
    custom_path = tmp_path / "custom" / "custom_mask.nii.gz"
    saved_custom = writer.write_mask(mask_arr, reference=ref_img, output_path=custom_path)
    assert saved_custom == custom_path
    assert custom_path.is_file()

    # 3. Missing both output_path and filename raises ValueError
    writer_no_dir = VolumetricMaskWriter()
    with pytest.raises(ValueError, match="Either output_path or filename"):
        writer_no_dir.write_mask(mask_arr, reference=ref_img)


def test_mask_writer_validation_errors(tmp_path: Path) -> None:
    """Verify mask writer input validation for reference types and output paths."""
    mask_arr = np.zeros((4, 4, 4), dtype=np.uint8)
    ref_img = sitk.GetImageFromArray(np.zeros((4, 4, 4), dtype=np.float32))

    # Missing reference
    with pytest.raises(TypeError, match="reference_image must be a SimpleITK Image or VolumetricMetadata, got None"):
        save_mask_as_nifti(mask_arr, reference=None, output_path=tmp_path / "out.nii.gz")

    # Invalid reference type
    with pytest.raises(TypeError, match="reference_image must be a SimpleITK Image or VolumetricMetadata, got int"):
        save_mask_as_nifti(mask_arr, reference=12345, output_path=tmp_path / "out.nii.gz")  # type: ignore[arg-type]

    # Missing output_path
    with pytest.raises(ValueError, match="output_path must be specified"):
        save_mask_as_nifti(mask_arr, reference=ref_img, output_path=None)


def test_extract_pydicom_metadata_and_sitk_to_numpy_standalone(tmp_path: Path) -> None:
    """Verify standalone extract_pydicom_metadata and sitk_to_numpy functions."""
    slice_path = tmp_path / "slice_standalone.dcm"
    _create_synthetic_dicom_slice(
        path=slice_path,
        patient_id="STANDALONE_PAT",
        study_uid="1.2.3.99",
        series_uid="1.2.3.99.1",
        modality="CT",
    )

    metadata = extract_pydicom_metadata(slice_path)
    assert metadata.patient_id == "STANDALONE_PAT"
    assert metadata.study_instance_uid == "1.2.3.99"
    assert metadata.series_instance_uid == "1.2.3.99.1"
    assert metadata.modality == "CT"

    # sitk_to_numpy
    img = sitk.GetImageFromArray(np.full((3, 4, 5), fill_value=42.0, dtype=np.float32))
    arr = sitk_to_numpy(img)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3, 4, 5)
    assert arr.dtype == np.float32
    assert np.allclose(arr, 42.0)


def test_azure_client_enhanced_input_validation(tmp_path: Path) -> None:
    """Verify AzureBlobClient validates container names, blob names, prefixes, and paths."""
    client, mock_bsc = _make_connected_client()
    dummy_file = tmp_path / "dummy.txt"
    dummy_file.write_text("sample")

    # 1. download_blob_to_stream validation
    with pytest.raises(ValueError, match="Invalid Azure Blob container name"):
        client.download_blob_to_stream("INVALID_CONTAINER", "vol.nii.gz")
    with pytest.raises(ValueError, match="blob_name must be a non-empty string"):
        client.download_blob_to_stream("valid-container", "")
    with pytest.raises(ValueError, match="blob_name must be a non-empty string"):
        client.download_blob_to_stream("valid-container", "   ")
    with pytest.raises(ValueError, match="blob_name must be a non-empty string"):
        client.download_blob_to_stream("valid-container", None)  # type: ignore[arg-type]

    # 2. download_blob_to_file validation
    with pytest.raises(ValueError, match="Invalid Azure Blob container name"):
        client.download_blob_to_file("in--valid", "vol.nii.gz", tmp_path / "dest.nii.gz")
    with pytest.raises(ValueError, match="blob_name must be a non-empty string"):
        client.download_blob_to_file("valid-container", "", tmp_path / "dest.nii.gz")
    with pytest.raises(ValueError, match="destination_path cannot be empty"):
        client.download_blob_to_file("valid-container", "vol.nii.gz", "")

    # 3. download_series_prefix validation
    with pytest.raises(ValueError, match="Invalid Azure Blob container name"):
        client.download_series_prefix("-invalid", "series/", tmp_path / "dest")
    with pytest.raises(ValueError, match="prefix must be a non-empty string"):
        client.download_series_prefix("valid-container", "", tmp_path / "dest")
    with pytest.raises(ValueError, match="destination_dir cannot be empty"):
        client.download_series_prefix("valid-container", "series/", "")

    # 4. upload_file_to_blob validation
    with pytest.raises(ValueError, match="Invalid Azure Blob container name"):
        client.upload_file_to_blob("invalid-", "blob.txt", dummy_file)
    with pytest.raises(ValueError, match="blob_name must be a non-empty string"):
        client.upload_file_to_blob("valid-container", "", dummy_file)
    with pytest.raises(ValueError, match="source_path cannot be empty"):
        client.upload_file_to_blob("valid-container", "blob.txt", "")

    # 5. list_blobs validation
    with pytest.raises(ValueError, match="Invalid Azure Blob container name"):
        client.list_blobs("INVALID_UPPER")
    with pytest.raises(TypeError, match="prefix must be a string"):
        client.list_blobs("valid-container", prefix=123)  # type: ignore[arg-type]


def test_validate_input_sources(tmp_path: Path) -> None:
    """Verify validate_input_sources correctly parses single sources and rejects missing/conflicting sources."""
    local_file = tmp_path / "volume.nii.gz"
    local_file.write_text("dummy")

    # Valid cases
    b, p, l = validate_input_sources(local_input=local_file)
    assert b is None and p is None and l == local_file.resolve()

    b, p, l = validate_input_sources(blob_path="volumes/study1.nii.gz")
    assert b == "volumes/study1.nii.gz" and p is None and l is None

    b, p, l = validate_input_sources(series_prefix="series/series_001/")
    assert b is None and p == "series/series_001/" and l is None

    # Missing sources
    with pytest.raises(ValueError, match="No input volume source specified"):
        validate_input_sources()

    with pytest.raises(ValueError, match="No input volume source specified"):
        validate_input_sources(blob_path="", series_prefix="  ", local_input=None)

    # Conflicting sources
    with pytest.raises(ValueError, match="Conflicting input volume sources"):
        validate_input_sources(blob_path="a.nii.gz", series_prefix="b/")

    with pytest.raises(ValueError, match="Conflicting input volume sources"):
        validate_input_sources(blob_path="a.nii.gz", local_input=local_file)

    with pytest.raises(ValueError, match="Conflicting input volume sources"):
        validate_input_sources(series_prefix="b/", local_input=local_file)

    with pytest.raises(ValueError, match="Conflicting input volume sources"):
        validate_input_sources(blob_path="a.nii.gz", series_prefix="b/", local_input=local_file)


def test_volume_ingester_local_resolution(tmp_path: Path) -> None:
    """Verify VolumeIngester resolves local files and directories accurately."""
    ingester = VolumeIngester(workspace=tmp_path / "workspace")

    # 1. Local file
    local_file = tmp_path / "study.nii.gz"
    local_file.write_text("data")
    path, is_dir = ingester.ingest(local_input=local_file)
    assert path == local_file.resolve()
    assert is_dir is False

    # 2. Local directory
    local_dir = tmp_path / "dicom_series"
    local_dir.mkdir()
    path_dir, is_dir_true = ingester.ingest(local_input=local_dir)
    assert path_dir == local_dir.resolve()
    assert is_dir_true is True

    # 3. Missing local path raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Local input path not found"):
        ingester.ingest(local_input=tmp_path / "nonexistent.nii.gz")


def test_volume_ingester_azure_downloads(tmp_path: Path) -> None:
    """Verify VolumeIngester coordinates Azure blob and DICOM series prefix downloads."""
    mock_azure = MagicMock(spec=AzureBlobClient)
    workspace = tmp_path / "workspace"
    ingester = VolumeIngester(azure_client=mock_azure, workspace=workspace)

    assert ingester.raw_volumes_dir == workspace / "raw_volumes"
    assert ingester.raw_series_dir == workspace / "raw_series"

    # 1. Azure blob download
    dest_file = workspace / "raw_volumes" / "patient.nii.gz"
    mock_azure.download_blob_to_file.return_value = dest_file
    res_path, res_is_dir = ingester.ingest(
        blob_path="inputs/patient.nii.gz",
        input_container="custom-container",
    )
    assert res_path == dest_file
    assert res_is_dir is False
    mock_azure.download_blob_to_file.assert_called_once_with(
        container_name="custom-container",
        blob_name="inputs/patient.nii.gz",
        destination_path=dest_file,
    )

    # 2. Azure series prefix download
    mock_azure.reset_mock()
    dest_series = workspace / "raw_series" / "series_01"
    mock_azure.download_series_prefix.return_value = [dest_series / "slice1.dcm"]
    res_series_path, res_series_is_dir = ingester.ingest(
        series_prefix="dicom/series_01",
        input_container="custom-container",
    )
    assert res_series_path == dest_series
    assert res_series_is_dir is True
    mock_azure.download_series_prefix.assert_called_once_with(
        container_name="custom-container",
        prefix="dicom/series_01",
        destination_dir=dest_series,
    )


def test_volume_ingester_validation_errors(tmp_path: Path) -> None:
    """Verify VolumeIngester error handling for missing client, invalid containers, and empty workspace."""
    # 1. Empty workspace path validation
    with pytest.raises(ValueError, match="workspace cannot be an empty path"):
        VolumeIngester(workspace="")
    with pytest.raises(ValueError, match="workspace cannot be an empty path"):
        VolumeIngester(workspace="   ")

    # 2. Cloud ingestion requires azure_client
    offline_ingester = VolumeIngester(azure_client=None, workspace=tmp_path)
    with pytest.raises(RuntimeError, match="AzureBlobClient is required"):
        offline_ingester.ingest(blob_path="vol.nii.gz")
    with pytest.raises(RuntimeError, match="AzureBlobClient is required"):
        offline_ingester.ingest(series_prefix="series/")

    # 3. Invalid container name validation
    mock_azure = MagicMock(spec=AzureBlobClient)
    ingester = VolumeIngester(azure_client=mock_azure, workspace=tmp_path)
    with pytest.raises(ValueError, match="Invalid Azure Blob container name"):
        ingester.ingest(blob_path="vol.nii.gz", input_container="INVALID_CONTAINER")




