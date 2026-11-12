"""
Comprehensive Unit Tests for 3dMedicalImageSegmentation Pipeline.
Tests spatial orientation standardization, MONAI transforms, 3D U-Net forward execution,
sliding-window inference, metric extraction, and 500-row chunked batch generation.
"""

from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk
import torch

from medical_segmentation.config import InferenceSettings, PipelineConfig
from medical_segmentation.db.database import chunked_iterable
from medical_segmentation.io.dicom_loader import DicomVolumeLoader, VolumetricMetadata
from medical_segmentation.pipeline.inference import SlidingWindowInferenceEngine
from medical_segmentation.pipeline.model import build_3d_unet
from medical_segmentation.pipeline.transforms import build_volumetric_transforms, prepare_tensor_from_sitk


def test_chunked_iterable_exact_500_row_batches() -> None:
    """Validate that chunked_iterable partitions streaming data into exact 500-row batches."""
    items = list(range(1250))
    chunks = list(chunked_iterable(items, chunk_size=500))

    assert len(chunks) == 3
    assert len(chunks[0]) == 500
    assert len(chunks[1]) == 500
    assert len(chunks[2]) == 250

    flattened = [x for c in chunks for x in c]
    assert flattened == items

    gen = (x * 2 for x in range(1001))
    gen_chunks = list(chunked_iterable(gen, chunk_size=500))
    assert len(gen_chunks) == 3
    assert len(gen_chunks[0]) == 500
    assert len(gen_chunks[1]) == 500
    assert len(gen_chunks[2]) == 1

    assert list(chunked_iterable([], chunk_size=500)) == []


def test_sitk_dicom_orient_standardization() -> None:
    """Validate that SimpleITK DICOMOrient correctly standardizes spatial orientation."""
    arr = np.zeros((16, 32, 32), dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.5, 1.5, 2.0))
    img.SetOrigin((10.0, 20.0, 30.0))

    img_lps = sitk.DICOMOrient(img, "LPS")
    assert sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(img_lps.GetDirection()) == "LPS"

    img_ras = sitk.DICOMOrient(img_lps, "RAS")
    assert sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(img_ras.GetDirection()) == "RAS"


def test_monai_transforms_and_tensor_preparation() -> None:
    """Verify that MONAI transformation pipeline produces a valid (1, 1, D, H, W) tensor."""
    arr = np.random.uniform(low=-1000.0, high=1000.0, size=(16, 32, 32)).astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))

    tensor = prepare_tensor_from_sitk(
        image=img,
        modality="CT",
        target_spacing=None,
        device="cpu",
    )

    assert tensor.ndim == 5
    assert tensor.shape == (1, 1, 16, 32, 32)
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_3d_unet_forward_pass() -> None:
    """Verify that 3D U-Net computes forward pass with appropriate tensor shape."""
    model = build_3d_unet(
        in_channels=1,
        out_channels=2,
        channels=(8, 16, 32),
        strides=(2, 2),
        num_res_units=1,
        norm="instance",
    )
    model.eval()

    dummy_input = torch.randn(1, 1, 16, 32, 32)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (1, 2, 16, 32, 32)


def test_sliding_window_inference_and_metric_calculation() -> None:
    """Validate sliding window inference execution and physical metric extraction."""
    settings = InferenceSettings(
        roi_size=(16, 16, 16),
        sw_batch_size=1,
        overlap=0.25,
        in_channels=1,
        out_channels=2,
        device="cpu",
    )
    model = build_3d_unet(
        in_channels=1,
        out_channels=2,
        channels=(8, 16),
        strides=(2,),
        num_res_units=1,
    )
    inferer = SlidingWindowInferenceEngine(model=model, settings=settings)

    volume_tensor = torch.zeros(1, 1, 24, 24, 24)
    mask = inferer.execute_inference(volume_tensor)

    assert mask.shape == (24, 24, 24)
    assert mask.dtype == np.uint8

    test_mask = np.zeros((24, 24, 24), dtype=np.uint8)
    test_mask[4:14, 4:14, 4:14] = 1  # 10x10x10 = 1000 voxels

    raw_intensities = np.full((24, 24, 24), fill_value=150.0, dtype=np.float32)
    metadata = VolumetricMetadata(
        dimensions=(24, 24, 24),
        spacing=(2.0, 2.0, 2.0),
        origin=(0.0, 0.0, 0.0),
        spatial_orientation="RAS",
    )

    summary, slice_gen = inferer.compute_metrics(
        raw_array=raw_intensities,
        mask_array=test_mask,
        metadata=metadata,
        duration_seconds=1.25,
    )

    assert len(summary.labels) == 1
    label_info = summary.labels[0]
    assert label_info.voxel_count == 1000
    assert label_info.volume_mm3 == 1000 * 8.0  # 8000 mm³
    assert label_info.volume_cm3 == 8.0          # 8.0 cm³
    assert label_info.mean_intensity == 150.0

    slice_records = list(slice_gen)
    assert len(slice_records) == 10
    for rec in slice_records:
        assert rec["label_id"] == 1
        assert rec["slice_voxel_count"] == 100
        assert rec["cross_sectional_area_mm2"] == 100 * (2.0 * 2.0)
        assert rec["plane"] == "axial"


def test_config_orientation_validator() -> None:
    """Verify that configuration validates anatomical orientation strings."""
    cfg = InferenceSettings(target_orientation="RAS")
    assert cfg.target_orientation == "RAS"

    cfg_lps = InferenceSettings(target_orientation="lps")
    assert cfg_lps.target_orientation == "LPS"

    with pytest.raises(ValueError):
        InferenceSettings(target_orientation="INVALID")

    with pytest.raises(ValueError):
        InferenceSettings(target_orientation="XYZ")


def test_direction_matrix_physical_centroid_calculation() -> None:
    """Validate that physical centroid coordinates correctly apply the 3D direction cosine matrix."""
    arr = np.zeros((20, 30, 40), dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.5, 2.0, 2.5))
    img.SetOrigin((100.0, -50.0, 20.0))

    img_ras = sitk.DICOMOrient(img, "RAS")

    metadata = VolumetricMetadata(
        dimensions=img_ras.GetSize(),
        spacing=img_ras.GetSpacing(),
        origin=img_ras.GetOrigin(),
        direction=img_ras.GetDirection(),
        spatial_orientation="RAS",
    )

    # SimpleITK GetArrayFromImage returns (Z, Y, X)
    raw_array = sitk.GetArrayFromImage(img_ras)
    mask_array = np.zeros_like(raw_array, dtype=np.uint8)

    # Set a 4x6x8 region as label 1
    # Z indices: 5..8 (mean = 6.5)
    # Y indices: 10..15 (mean = 12.5)
    # X indices: 20..27 (mean = 23.5)
    mask_array[5:9, 10:16, 20:28] = 1

    settings = InferenceSettings(roi_size=(16, 16, 16), device="cpu")
    model = build_3d_unet(in_channels=1, out_channels=2, channels=(8, 16), strides=(2,), num_res_units=1)
    inferer = SlidingWindowInferenceEngine(model=model, settings=settings)

    summary, slice_gen = inferer.compute_metrics(
        raw_array=raw_array,
        mask_array=mask_array,
        metadata=metadata,
        duration_seconds=0.5,
    )

    assert len(summary.labels) == 1
    label = summary.labels[0]

    # Expected centroid continuous index (X, Y, Z)
    expected_cx = 23.5
    expected_cy = 12.5
    expected_cz = 6.5
    expected_physical = img_ras.TransformContinuousIndexToPhysicalPoint((expected_cx, expected_cy, expected_cz))

    assert np.allclose(label.centroid_mm, expected_physical, atol=1e-4)


def test_compute_metrics_shape_mismatch_raises_value_error() -> None:
    """Verify that compute_metrics raises ValueError on mismatched raw and mask shapes."""
    settings = InferenceSettings(roi_size=(16, 16, 16), device="cpu")
    model = build_3d_unet(in_channels=1, out_channels=2, channels=(8, 16), strides=(2,), num_res_units=1)
    inferer = SlidingWindowInferenceEngine(model=model, settings=settings)

    raw_array = np.zeros((24, 24, 24), dtype=np.float32)
    mask_array = np.zeros((16, 16, 16), dtype=np.uint8)
    metadata = VolumetricMetadata(dimensions=(24, 24, 24), spacing=(1.0, 1.0, 1.0))

    with pytest.raises(ValueError, match="Array shape mismatch"):
        inferer.compute_metrics(
            raw_array=raw_array,
            mask_array=mask_array,
            metadata=metadata,
            duration_seconds=0.1,
        )

