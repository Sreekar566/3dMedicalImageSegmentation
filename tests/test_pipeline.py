"""
Comprehensive Unit Tests for 3dMedicalImageSegmentation Pipeline.
Tests spatial orientation standardization, MONAI transforms, 3D U-Net forward execution,
sliding-window inference, metric extraction, and 500-row chunked batch generation.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import SimpleITK as sitk
import torch
from pydantic import ValidationError
from sqlalchemy import create_engine

from medical_segmentation.config import (
    AzureStorageSettings,
    DatabaseSettings,
    InferenceSettings,
    PipelineConfig,
    parse_model_checkpoint_path,
    parse_roi_size,
    parse_target_spacing,
    validate_azure_container_name,
    validate_orientation_code,
)
from medical_segmentation.db.database import DatabaseManager, chunked_iterable
from medical_segmentation.io.dicom_loader import DicomVolumeLoader, VolumetricMetadata
from medical_segmentation.io.mask_writer import VolumetricMaskWriter
from medical_segmentation.io.volume_ingester import VolumeIngester
from medical_segmentation.main import main, parse_args, run_pipeline
from medical_segmentation.pipeline import (
    PipelineOrchestrator,
    PipelineResult,
    SegmentationMetrics,
)
from medical_segmentation.pipeline.metrics import LabelSummaryMetric
from medical_segmentation.pipeline.inference import (
    SlidingWindowInferenceEngine,
    VolumetricMetricsCalculator,
)
from medical_segmentation.pipeline.model import build_3d_unet, extract_state_dict, load_3d_unet
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


def test_monai_transforms_mri_and_fallback_modalities() -> None:
    """Verify MONAI transformations for MRI nonzero z-score normalization and fallback modalities."""
    arr = np.random.uniform(low=0.0, high=500.0, size=(16, 32, 32)).astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))

    # MRI modality: nonzero z-score normalization
    tensor_mri = prepare_tensor_from_sitk(
        image=img,
        modality="MRI",
        target_spacing=None,
        device="cpu",
    )
    assert tensor_mri.ndim == 5
    assert tensor_mri.shape == (1, 1, 16, 32, 32)

    # CBCT modality
    cbct_transform = build_volumetric_transforms(modality="CBCT")
    assert cbct_transform is not None

    # Fallback/other modality
    fallback_transform = build_volumetric_transforms(modality="PET")
    assert fallback_transform is not None


def test_monai_transforms_spatial_resampling() -> None:
    """Verify that Spacingd resamples input volumes to the requested target voxel spacing."""
    arr = np.random.uniform(low=-100.0, high=100.0, size=(16, 32, 32)).astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))  # 1mm isotropic

    # Resample to 2mm isotropic -> spatial dimensions should be halved: 16->8, 32->16, 32->16
    tensor_resampled = prepare_tensor_from_sitk(
        image=img,
        modality="CT",
        target_spacing=(2.0, 2.0, 2.0),
        device="cpu",
    )
    assert tensor_resampled.ndim == 5
    assert tensor_resampled.shape == (1, 1, 8, 16, 16)


def test_load_3d_unet_from_checkpoint_and_fallback(tmp_path: Path) -> None:
    """Verify load_3d_unet loads raw and wrapped state_dict weights and handles missing checkpoints gracefully."""
    model = build_3d_unet(in_channels=1, out_channels=2)
    for p in model.parameters():
        p.data.fill_(42.0)

    # 1. Raw state dict
    raw_ckpt_path = tmp_path / "raw_weights.pt"
    torch.save(model.state_dict(), str(raw_ckpt_path))
    loaded_raw = load_3d_unet(checkpoint_path=raw_ckpt_path, in_channels=1, out_channels=2)
    assert isinstance(loaded_raw, torch.nn.Module)
    assert torch.allclose(next(loaded_raw.parameters()), torch.full_like(next(loaded_raw.parameters()), 42.0))

    # 2. Wrapped state dict ({'state_dict': ...})
    wrapped_ckpt_path = tmp_path / "wrapped_weights.pt"
    torch.save({"state_dict": model.state_dict()}, str(wrapped_ckpt_path))
    loaded_wrapped = load_3d_unet(checkpoint_path=wrapped_ckpt_path, in_channels=1, out_channels=2)
    assert isinstance(loaded_wrapped, torch.nn.Module)
    assert torch.allclose(next(loaded_wrapped.parameters()), torch.full_like(next(loaded_wrapped.parameters()), 42.0))

    # 3. Missing checkpoint path falls back to freshly initialized model
    missing_ckpt = tmp_path / "non_existent_weights.pt"
    fallback_model = load_3d_unet(checkpoint_path=missing_ckpt, in_channels=1, out_channels=2)
    assert isinstance(fallback_model, torch.nn.Module)


def test_load_3d_unet_ddp_and_model_state_dict_variants(tmp_path: Path) -> None:
    """Verify load_3d_unet loads DDP ('module.'), compile ('_orig_mod.'), and wrapped ('model_state_dict') checkpoints."""
    model = build_3d_unet(in_channels=1, out_channels=2)
    for p in model.parameters():
        p.data.fill_(99.5)

    # 1. DDP multi-GPU prefix ('module.')
    ddp_ckpt = tmp_path / "ddp_weights.pt"
    torch.save({"module." + k: v for k, v in model.state_dict().items()}, str(ddp_ckpt))
    loaded_ddp = load_3d_unet(checkpoint_path=ddp_ckpt, in_channels=1, out_channels=2)
    assert torch.allclose(next(loaded_ddp.parameters()), torch.full_like(next(loaded_ddp.parameters()), 99.5))

    # 2. torch.compile prefix ('_orig_mod.')
    compile_ckpt = tmp_path / "compile_weights.pt"
    torch.save({"_orig_mod." + k: v for k, v in model.state_dict().items()}, str(compile_ckpt))
    loaded_compile = load_3d_unet(checkpoint_path=compile_ckpt, in_channels=1, out_channels=2)
    assert torch.allclose(next(loaded_compile.parameters()), torch.full_like(next(loaded_compile.parameters()), 99.5))

    # 3. Wrapped in 'model_state_dict' (standard PyTorch idiom)
    msd_ckpt = tmp_path / "msd_weights.pt"
    torch.save({"model_state_dict": model.state_dict(), "epoch": 10}, str(msd_ckpt))
    loaded_msd = load_3d_unet(checkpoint_path=msd_ckpt, in_channels=1, out_channels=2)
    assert torch.allclose(next(loaded_msd.parameters()), torch.full_like(next(loaded_msd.parameters()), 99.5))

    # 4. DDP inside wrapped dictionary
    nested_ckpt = tmp_path / "nested_ddp.pt"
    torch.save({"model_state_dict": {"module." + k: v for k, v in model.state_dict().items()}}, str(nested_ckpt))
    loaded_nested = load_3d_unet(checkpoint_path=nested_ckpt, in_channels=1, out_channels=2)
    assert torch.allclose(next(loaded_nested.parameters()), torch.full_like(next(loaded_nested.parameters()), 99.5))

    # 5. Full serialized nn.Module instance
    full_ckpt = tmp_path / "full_module.pt"
    torch.save(model, str(full_ckpt))
    loaded_full = load_3d_unet(checkpoint_path=full_ckpt, in_channels=1, out_channels=2)
    assert torch.allclose(next(loaded_full.parameters()), torch.full_like(next(loaded_full.parameters()), 99.5))


def test_load_3d_unet_incompatible_checkpoint_raises_value_error(tmp_path: Path) -> None:
    """Verify that attempting to load a checkpoint with zero matching parameters raises ValueError."""
    bad_ckpt = tmp_path / "bad_checkpoint.pt"
    torch.save({"irrelevant_tensor": torch.randn(4, 4)}, str(bad_ckpt))

    with pytest.raises(ValueError, match="no matching weights found for 3D U-Net"):
        load_3d_unet(checkpoint_path=bad_ckpt, in_channels=1, out_channels=2)


def test_extract_state_dict_helper() -> None:
    """Validate extract_state_dict normalizes various dictionary keys and handles errors."""
    dummy_model = build_3d_unet(1, 2)
    target_keys = list(dummy_model.state_dict().keys())

    # Invalid payload type
    with pytest.raises(ValueError, match="Unexpected checkpoint payload type"):
        extract_state_dict("not_a_valid_checkpoint", target_keys)

    # Bare tensor dictionary without model. prefix
    stripped_dict = {k.removeprefix("model."): v for k, v in dummy_model.state_dict().items()}
    cleaned = extract_state_dict(stripped_dict, target_keys)
    assert set(cleaned.keys()) == set(target_keys)

    # Sub-dict with non-tensor values ignored
    metadata_dict = {"state_dict": dummy_model.state_dict(), "epoch": 5, "optimizer": {}}
    cleaned_meta = extract_state_dict(metadata_dict, target_keys)
    assert set(cleaned_meta.keys()) == set(target_keys)



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


def test_volumetric_metadata_properties_and_transforms() -> None:
    """Validate VolumetricMetadata properties and physical point coordinate transformations."""
    meta = VolumetricMetadata(
        spacing=(2.0, 3.0, 4.0),
        origin=(10.0, 20.0, 30.0),
    )
    assert meta.voxel_volume_mm3 == 24.0
    assert meta.pixel_area_axial_mm2 == 6.0

    # Without direction cosine matrix
    pt = meta.continuous_index_to_physical_point((1.0, 2.0, 3.0))
    # 10 + 1*2 = 12, 20 + 2*3 = 26, 30 + 3*4 = 42
    assert pt == (12.0, 26.0, 42.0)


def test_volumetric_metrics_calculator_standalone() -> None:
    """Verify that VolumetricMetricsCalculator operates independently without requiring model instantiation."""
    calc = VolumetricMetricsCalculator()
    raw = np.full((10, 10, 10), fill_value=50.0, dtype=np.float32)
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[2:6, 2:6, 2:6] = 1  # 4x4x4 = 64 voxels

    meta = VolumetricMetadata(
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )

    metrics, slice_gen = calc.compute_metrics(raw, mask, meta, duration_seconds=0.25)
    assert len(metrics.labels) == 1
    assert metrics.labels[0].voxel_count == 64
    assert metrics.labels[0].mean_intensity == 50.0
    assert metrics.labels[0].bounding_box == (2, 2, 2, 5, 5, 5)

    slices = list(slice_gen)
    assert len(slices) == 4
    for s in slices:
        assert s["label_id"] == 1
        assert s["slice_voxel_count"] == 16


def test_volumetric_metrics_calculator_empty_mask() -> None:
    """Verify that an all-zero mask returns clean empty metric summaries without raising errors."""
    calc = VolumetricMetricsCalculator()
    raw = np.zeros((8, 8, 8), dtype=np.float32)
    mask = np.zeros((8, 8, 8), dtype=np.uint8)
    meta = VolumetricMetadata()

    metrics, slice_gen = calc.compute_metrics(raw, mask, meta)
    assert len(metrics.labels) == 0
    assert list(slice_gen) == []


def test_volumetric_metrics_calculator_multilabel() -> None:
    """Verify that multi-class masks extract distinct metrics for each segmented structure."""
    calc = VolumetricMetricsCalculator()
    raw = np.zeros((12, 12, 12), dtype=np.float32)
    mask = np.zeros((12, 12, 12), dtype=np.uint8)

    # Label 1 at z=1..3
    mask[1:4, 2:5, 2:5] = 1
    raw[1:4, 2:5, 2:5] = 100.0

    # Label 2 at z=7..9
    mask[7:10, 6:9, 6:9] = 2
    raw[7:10, 6:9, 6:9] = 200.0

    meta = VolumetricMetadata(spacing=(1.0, 1.0, 2.0), origin=(0.0, 0.0, 0.0))
    metrics, slice_gen = calc.compute_metrics(raw, mask, meta)

    assert len(metrics.labels) == 2
    label1 = next(lbl for lbl in metrics.labels if lbl.label_id == 1)
    label2 = next(lbl for lbl in metrics.labels if lbl.label_id == 2)

    assert label1.voxel_count == 27
    assert label1.mean_intensity == 100.0
    assert label2.voxel_count == 27
    assert label2.mean_intensity == 200.0

    slices = list(slice_gen)
    assert len(slices) == 6  # 3 slices for label 1, 3 slices for label 2
    lbl1_slices = [s for s in slices if s["label_id"] == 1]
    lbl2_slices = [s for s in slices if s["label_id"] == 2]
    assert [s["slice_index"] for s in lbl1_slices] == [1, 2, 3]
    assert [s["slice_index"] for s in lbl2_slices] == [7, 8, 9]


def test_execute_inference_binary_channel() -> None:
    """Validate SlidingWindowInferenceEngine execution with single binary output channel."""
    settings = InferenceSettings(
        roi_size=(16, 16, 16),
        sw_batch_size=1,
        overlap=0.25,
        in_channels=1,
        out_channels=1,
        device="cpu",
    )
    model = build_3d_unet(
        in_channels=1,
        out_channels=1,
        channels=(8, 16),
        strides=(2,),
        num_res_units=1,
    )
    inferer = SlidingWindowInferenceEngine(model=model, settings=settings)

    volume_tensor = torch.zeros(1, 1, 20, 20, 20)
    mask = inferer.execute_inference(volume_tensor)

    assert mask.shape == (20, 20, 20)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1})


def test_execute_inference_large_label_uint16() -> None:
    """Validate SlidingWindowInferenceEngine uses uint16 to prevent overflow when out_channels > 256."""
    settings = InferenceSettings(
        roi_size=(16, 16, 16),
        sw_batch_size=1,
        overlap=0.25,
        in_channels=1,
        out_channels=300,
        device="cpu",
    )
    # Mock model returning logits of shape (1, 300, D, H, W) where channel 260 has highest logit
    class LargeChannelMockModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch, _, d, h, w = x.shape
            out = torch.zeros(batch, 300, d, h, w, device=x.device, dtype=x.dtype)
            out[:, 260, :, :, :] = 10.0
            return out

    model = LargeChannelMockModel()
    inferer = SlidingWindowInferenceEngine(model=model, settings=settings)

    volume_tensor = torch.zeros(1, 1, 16, 16, 16)
    mask = inferer.execute_inference(volume_tensor)

    assert mask.shape == (16, 16, 16)
    assert mask.dtype == np.uint16
    assert (mask == 260).all()


def test_volumetric_metadata_slice_plane_and_position_all_anatomical_planes() -> None:
    """Validate that VolumetricMetadata determines correct anatomical planes and positions for axial, coronal, and sagittal orientations."""
    # 1. Axial orientation (RAS)
    img_ras = sitk.Image(10, 20, 30, sitk.sitkFloat32)
    img_ras.SetSpacing((1.0, 1.0, 2.0))
    img_ras.SetOrigin((10.0, 20.0, 30.0))
    img_ras = sitk.DICOMOrient(img_ras, "RAS")
    meta_ras = VolumetricMetadata(
        spacing=img_ras.GetSpacing(),
        origin=img_ras.GetOrigin(),
        direction=img_ras.GetDirection(),
        spatial_orientation="RAS",
    )
    plane_ras, pos_ras_0 = meta_ras.get_slice_plane_and_position(0)
    _, pos_ras_5 = meta_ras.get_slice_plane_and_position(5)
    assert plane_ras == "axial"
    assert pos_ras_5 != pos_ras_0
    assert np.isclose(pos_ras_5 - pos_ras_0, 5 * 2.0)

    # 2. Coronal orientation (RSP: X=Right, Y=Superior, Z=Posterior)
    img_rsp = sitk.Image(10, 20, 30, sitk.sitkFloat32)
    img_rsp.SetSpacing((1.0, 1.0, 2.0))
    img_rsp.SetOrigin((10.0, 20.0, 30.0))
    img_rsp = sitk.DICOMOrient(img_rsp, "RSP")
    meta_rsp = VolumetricMetadata(
        spacing=img_rsp.GetSpacing(),
        origin=img_rsp.GetOrigin(),
        direction=img_rsp.GetDirection(),
        spatial_orientation="RSP",
    )
    plane_rsp, pos_rsp_0 = meta_rsp.get_slice_plane_and_position(0)
    _, pos_rsp_5 = meta_rsp.get_slice_plane_and_position(5)
    assert plane_rsp == "coronal"
    assert pos_rsp_5 != pos_rsp_0

    # 3. Sagittal orientation (ASR: X=Anterior, Y=Superior, Z=Right)
    img_asr = sitk.Image(10, 20, 30, sitk.sitkFloat32)
    img_asr.SetSpacing((1.0, 1.0, 2.0))
    img_asr.SetOrigin((10.0, 20.0, 30.0))
    img_asr = sitk.DICOMOrient(img_asr, "ASR")
    meta_asr = VolumetricMetadata(
        spacing=img_asr.GetSpacing(),
        origin=img_asr.GetOrigin(),
        direction=img_asr.GetDirection(),
        spatial_orientation="ASR",
    )
    plane_asr, pos_asr_0 = meta_asr.get_slice_plane_and_position(0)
    _, pos_asr_5 = meta_asr.get_slice_plane_and_position(5)
    assert plane_asr == "sagittal"
    assert pos_asr_5 != pos_asr_0

    # 4. Fallback using direction cosine matrix when spatial_orientation is empty
    meta_fallback = VolumetricMetadata(
        spacing=img_rsp.GetSpacing(),
        origin=img_rsp.GetOrigin(),
        direction=img_rsp.GetDirection(),
        spatial_orientation="",
    )
    plane_fb, pos_fb = meta_fallback.get_slice_plane_and_position(0)
    assert plane_fb == "coronal"


def test_generate_slice_metrics_coronal_and_sagittal_planes() -> None:
    """Validate that generate_slice_metrics correctly assigns coronal and sagittal plane labels and varying positions."""
    calc = VolumetricMetricsCalculator()
    raw = np.full((10, 10, 10), fill_value=100.0, dtype=np.float32)
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[2:5, :, :] = 1  # Slices 2, 3, 4

    # Test Coronal
    img_coronal = sitk.Image(10, 10, 10, sitk.sitkFloat32)
    img_coronal.SetSpacing((1.0, 1.0, 2.0))
    img_coronal.SetOrigin((0.0, 0.0, 0.0))
    img_coronal = sitk.DICOMOrient(img_coronal, "RSP")
    meta_coronal = VolumetricMetadata(
        dimensions=(10, 10, 10),
        spacing=img_coronal.GetSpacing(),
        origin=img_coronal.GetOrigin(),
        direction=img_coronal.GetDirection(),
        spatial_orientation="RSP",
    )

    _, slice_gen_coronal = calc.compute_metrics(raw, mask, meta_coronal)
    coronal_slices = list(slice_gen_coronal)
    assert len(coronal_slices) == 3
    for s in coronal_slices:
        assert s["plane"] == "coronal"
    positions = [s["slice_position_mm"] for s in coronal_slices]
    # Slices must have distinct physical positions
    assert len(set(positions)) == 3
    assert positions[0] != positions[1]

    # Test Sagittal
    img_sagittal = sitk.DICOMOrient(img_coronal, "ASR")
    meta_sagittal = VolumetricMetadata(
        dimensions=(10, 10, 10),
        spacing=img_sagittal.GetSpacing(),
        origin=img_sagittal.GetOrigin(),
        direction=img_sagittal.GetDirection(),
        spatial_orientation="ASR",
    )

    _, slice_gen_sagittal = calc.compute_metrics(raw, mask, meta_sagittal)
    sagittal_slices = list(slice_gen_sagittal)
    assert len(sagittal_slices) == 3
    for s in sagittal_slices:
        assert s["plane"] == "sagittal"
    positions_sag = [s["slice_position_mm"] for s in sagittal_slices]
    assert len(set(positions_sag)) == 3


def test_pipeline_orchestrator_local_dry_run(tmp_path: Path) -> None:
    """Verify that PipelineOrchestrator successfully executes an end-to-end run in local dry-run mode."""
    arr = np.random.uniform(low=-100.0, high=200.0, size=(32, 32, 32)).astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.5, 1.5, 2.0))
    img.SetOrigin((0.0, 0.0, 0.0))

    volume_path = tmp_path / "test_volume.nii.gz"
    sitk.WriteImage(img, str(volume_path))

    workspace_dir = tmp_path / "workspace"
    cfg = PipelineConfig()
    cfg.work_dir = workspace_dir
    cfg.inference.roi_size = (32, 32, 32)
    cfg.inference.target_spacing = None
    cfg.inference.device = "cpu"

    orchestrator = PipelineOrchestrator(config=cfg, workspace=workspace_dir)
    result = orchestrator.run(
        local_input=volume_path,
        modality="CT",
        dry_run=True,
    )

    assert isinstance(result, PipelineResult)
    assert result.duration_seconds > 0.0
    assert result.mask_array.shape == (32, 32, 32)
    assert result.local_mask_path.is_file()
    assert result.output_blob_uri is None
    assert result.study_id is None
    assert result.run_id is None
    assert result.slices_inserted == 0
    assert result.metadata.dimensions == (32, 32, 32)


def test_pipeline_orchestrator_ingest_volume_validation(tmp_path: Path) -> None:
    """Verify that ingest_volume enforces input source presence and file existence."""
    orchestrator = PipelineOrchestrator()

    with pytest.raises(ValueError, match="No input volume source specified"):
        orchestrator.ingest_volume()

    missing_path = tmp_path / "non_existent.nii.gz"
    with pytest.raises(FileNotFoundError, match="Local input path not found"):
        orchestrator.ingest_volume(local_input=missing_path)


def test_pipeline_orchestrator_database_persistence() -> None:
    """Verify that persist_to_database records study, run, and chunked slice metrics."""
    engine = create_engine("sqlite:///:memory:")
    db_manager = DatabaseManager(DatabaseSettings(), engine=engine)

    orchestrator = PipelineOrchestrator(db_manager=db_manager)

    meta = VolumetricMetadata(
        patient_id="PATIENT_99",
        study_instance_uid="STUDY_99",
        series_instance_uid="SERIES_99",
        dimensions=(16, 16, 16),
        spacing=(1.0, 1.0, 1.0),
    )

    calc = VolumetricMetricsCalculator()
    raw = np.full((16, 16, 16), fill_value=120.0, dtype=np.float32)
    mask = np.zeros((16, 16, 16), dtype=np.uint8)
    mask[4:8, 4:8, 4:8] = 1

    summary_metrics, slice_gen = calc.compute_metrics(raw, mask, meta, duration_seconds=0.5)

    study_id, run_id, inserted_count = orchestrator.persist_to_database(
        metadata=meta,
        summary_metrics=summary_metrics,
        slice_generator=slice_gen,
        batch_size=2,
        dry_run=False,
    )

    assert study_id is not None
    assert run_id is not None
    assert inserted_count == 4


def test_pipeline_orchestrator_database_persistence_disposes_local_manager() -> None:
    """Verify persist_to_database properly disposes of internally created DatabaseManager."""
    orchestrator = PipelineOrchestrator(db_manager=None)

    meta = VolumetricMetadata(
        patient_id="PATIENT_LIFECYCLE",
        study_instance_uid="STUDY_LIFECYCLE",
        series_instance_uid="SERIES_LIFECYCLE",
        dimensions=(8, 8, 8),
        spacing=(1.0, 1.0, 1.0),
    )
    summary = SegmentationMetrics(duration_seconds=0.1)

    mock_db = MagicMock(spec=DatabaseManager)
    mock_db.record_study_and_run.return_value = (1, 1)
    mock_db.bulk_insert_slice_metrics.return_value = 0

    with patch("medical_segmentation.db.database.DatabaseManager", return_value=mock_db):
        study_id, run_id, count = orchestrator.persist_to_database(
            metadata=meta,
            summary_metrics=summary,
            slice_generator=iter([]),
            dry_run=False,
        )

        assert study_id == 1
        assert run_id == 1
        mock_db.dispose.assert_called_once()


def test_main_run_pipeline_cli_wrapper(tmp_path: Path) -> None:
    """Validate that run_pipeline CLI wrapper executes cleanly on local inputs and handles failures."""
    arr = np.zeros((32, 32, 32), dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    vol_path = tmp_path / "cli_vol.nii.gz"
    sitk.WriteImage(img, str(vol_path))

    # Success case
    exit_code = run_pipeline(local_input=vol_path, dry_run=True)
    assert exit_code == 0

    # Failure case (missing input)
    err_code = run_pipeline()
    assert err_code == 1


def test_volumetric_metrics_calculator_decoupled_helpers() -> None:
    """Validate extracted helper methods in VolumetricMetricsCalculator."""
    calc = VolumetricMetricsCalculator()

    # 1. Intensity statistics helper
    raw = np.array([[[10.0, 20.0], [30.0, 40.0]]], dtype=np.float32)
    mask = np.array([[[True, False], [True, True]]], dtype=bool)
    mean_val, std_val, min_val, max_val = calc._compute_intensity_statistics(raw, mask)
    expected_vals = [10.0, 30.0, 40.0]
    assert np.isclose(mean_val, np.mean(expected_vals))
    assert np.isclose(std_val, np.std(expected_vals))
    assert min_val == 10.0
    assert max_val == 40.0

    # 2. 1D orthogonal projection spatial bounds and centroid
    label_mask = np.zeros((10, 12, 14), dtype=bool)
    label_mask[2:6, 3:8, 4:10] = True  # Z: 2..5, Y: 3..7, X: 4..9
    voxel_count = int(np.count_nonzero(label_mask))
    meta = VolumetricMetadata(
        spacing=(2.0, 3.0, 4.0),
        origin=(10.0, 20.0, 30.0),
        spatial_orientation="RAS",
    )

    bbox, centroid_mm, z_counts = calc._compute_spatial_bounds_and_centroid(label_mask, meta, voxel_count)
    assert bbox == (4, 3, 2, 9, 7, 5)
    # Centroid continuous index: mean of [4..9] is 6.5, mean of [3..7] is 5.0, mean of [2..5] is 3.5
    expected_physical = meta.continuous_index_to_physical_point((6.5, 5.0, 3.5))
    assert np.allclose(centroid_mm, expected_physical)
    assert len(z_counts) == 10
    assert z_counts[0] == 0
    assert z_counts[2] == 5 * 6  # Y len * X len = 30

    # 3. Slice plane and positions helper
    plane = calc._resolve_slice_plane(meta)
    assert plane == "axial"
    positions = calc._compute_slice_positions(meta, depth=10)
    assert len(positions) == 10
    assert np.isclose(positions[1] - positions[0], 4.0)

    # 4. Slice record builder
    slice_intensities = np.array([100.0, 200.0], dtype=np.float32)
    record = calc._build_slice_record(
        label_id=1,
        slice_index=3,
        plane="axial",
        slice_position_mm=42.0,
        slice_voxel_count=2,
        pixel_area=6.0,
        slice_intensities=slice_intensities,
    )
    assert record["label_id"] == 1
    assert record["slice_index"] == 3
    assert record["plane"] == "axial"
    assert record["slice_position_mm"] == 42.0
    assert record["slice_voxel_count"] == 2
    assert record["cross_sectional_area_mm2"] == 12.0
    assert record["mean_intensity"] == 150.0
    assert record["max_intensity"] == 200.0

    # 5. Empty mask returns None, None from _compute_label_summary_with_counts
    empty_mask = np.zeros((5, 5, 5), dtype=bool)
    summary, counts = calc._compute_label_summary_with_counts(np.zeros((5, 5, 5), dtype=np.float32), empty_mask, 1, meta)
    assert summary is None
    assert counts is None


def test_config_settings_validation_and_parsing(tmp_path: Path) -> None:
    """Verify input validation and configuration parsing across Inference, DB, and Azure settings."""
    # 1. InferenceSettings: roi_size parsing and validation
    cfg = InferenceSettings(roi_size="(96, 96, 96)")
    assert cfg.roi_size == (96, 96, 96)
    assert InferenceSettings(roi_size="[64, 64, 64]").roi_size == (64, 64, 64)
    assert InferenceSettings(roi_size="32, 32, 32").roi_size == (32, 32, 32)
    assert InferenceSettings(roi_size=(48, 48, 48)).roi_size == (48, 48, 48)

    with pytest.raises(ValueError, match="strictly positive integers"):
        InferenceSettings(roi_size=(0, 96, 96))
    with pytest.raises(ValueError, match="strictly positive integers"):
        InferenceSettings(roi_size=(-5, 96, 96))
    with pytest.raises(ValueError, match="exactly 3 dimensions"):
        InferenceSettings(roi_size=(96, 96))
    with pytest.raises(ValueError, match="comma-separated integers"):
        InferenceSettings(roi_size="invalid")

    # 2. InferenceSettings: target_spacing parsing and validation
    assert InferenceSettings(target_spacing=None).target_spacing is None
    assert InferenceSettings(target_spacing="").target_spacing is None
    assert InferenceSettings(target_spacing="none").target_spacing is None
    assert InferenceSettings(target_spacing="null").target_spacing is None
    assert InferenceSettings(target_spacing="(1.0, 1.0, 1.0)").target_spacing == (1.0, 1.0, 1.0)
    assert InferenceSettings(target_spacing="1.5, 1.5, 2.0").target_spacing == (1.5, 1.5, 2.0)

    with pytest.raises(ValueError, match="strictly positive"):
        InferenceSettings(target_spacing=(1.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="strictly positive"):
        InferenceSettings(target_spacing=(-1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="exactly 3 dimensions"):
        InferenceSettings(target_spacing=(1.0, 1.0))
    with pytest.raises(ValueError, match="comma-separated floats"):
        InferenceSettings(target_spacing="invalid")

    # 3. InferenceSettings: model_checkpoint_path normalization
    assert InferenceSettings(model_checkpoint_path=None).model_checkpoint_path is None
    assert InferenceSettings(model_checkpoint_path="").model_checkpoint_path is None
    assert InferenceSettings(model_checkpoint_path="none").model_checkpoint_path is None
    assert InferenceSettings(model_checkpoint_path="null").model_checkpoint_path is None
    assert InferenceSettings(model_checkpoint_path=Path("")).model_checkpoint_path is None
    assert InferenceSettings(model_checkpoint_path=Path(".")).model_checkpoint_path is None
    assert InferenceSettings(model_checkpoint_path=Path("ckpt.pt")).model_checkpoint_path == Path("ckpt.pt")
    assert InferenceSettings(model_checkpoint_path="/tmp/ckpt.pt").model_checkpoint_path == Path("/tmp/ckpt.pt")

    # 4. InferenceSettings: modality normalization & validation
    assert InferenceSettings(modality="ct").modality == "CT"
    assert InferenceSettings(modality="mri").modality == "MRI"
    assert InferenceSettings(modality="cbct").modality == "CBCT"
    with pytest.raises(ValidationError):
        InferenceSettings(modality="PET")

    # 5. InferenceSettings: device validation
    assert InferenceSettings(device="cpu").device == "cpu"
    with pytest.raises(ValueError, match="Invalid PyTorch execution device"):
        InferenceSettings(device="not_a_device_type")
    with pytest.raises((TypeError, ValidationError)):
        InferenceSettings(device=123)

    # 6. InferenceSettings: numeric range validation
    with pytest.raises(ValidationError):
        InferenceSettings(sw_batch_size=0)
    with pytest.raises(ValidationError):
        InferenceSettings(in_channels=0)
    with pytest.raises(ValidationError):
        InferenceSettings(out_channels=0)
    with pytest.raises(ValidationError):
        InferenceSettings(overlap=-0.1)
    with pytest.raises(ValidationError):
        InferenceSettings(overlap=0.9)

    # 7. DatabaseSettings: port and batch size validation
    assert DatabaseSettings(port=5432).port == 5432
    with pytest.raises(ValidationError):
        DatabaseSettings(port=0)
    with pytest.raises(ValidationError):
        DatabaseSettings(port=70000)
    with pytest.raises(ValidationError):
        DatabaseSettings(batch_size=0)
    with pytest.raises(ValidationError):
        DatabaseSettings(pool_size=0)
    with pytest.raises(ValidationError):
        DatabaseSettings(max_overflow=-1)
    with pytest.raises(ValidationError):
        DatabaseSettings(pool_timeout=0)

    # 8. DatabaseSettings: connection_url special characters encoding
    db_special = DatabaseSettings(
        user="special@user",
        password="p@ss:w/ord#1",
        host="myhost",
        port=5432,
        name="test_db",
        sslmode="require",
    )
    url = db_special.connection_url
    assert "special%40user" in url
    assert "p%40ss%3Aw%2Ford%231" in url
    assert url.startswith("postgresql+psycopg2://special%40user:p%40ss%3Aw%2Ford%231@myhost:5432/test_db?sslmode=require")

    # 9. AzureStorageSettings: container name and concurrency validation
    assert AzureStorageSettings(input_container="valid-input-1").input_container == "valid-input-1"
    with pytest.raises(ValidationError):
        AzureStorageSettings(input_container="INVALID_UPPERCASE")
    with pytest.raises(ValidationError):
        AzureStorageSettings(input_container="in--valid")
    with pytest.raises(ValidationError):
        AzureStorageSettings(input_container="-invalid")
    with pytest.raises(ValidationError):
        AzureStorageSettings(input_container="invalid-")
    with pytest.raises(ValidationError):
        AzureStorageSettings(input_container="ab")
    with pytest.raises(ValidationError):
        AzureStorageSettings(max_concurrency=0)
    with pytest.raises(ValidationError):
        AzureStorageSettings(timeout_seconds=0)

    # 10. PipelineConfig: log level and work_dir validation
    cfg_pipe = PipelineConfig(log_level="debug", work_dir="/tmp/test_dir")
    assert cfg_pipe.log_level == "DEBUG"
    assert cfg_pipe.work_dir == Path("/tmp/test_dir")
    cfg_warn = PipelineConfig(log_level="warn")
    assert cfg_warn.log_level == "WARNING"
    with pytest.raises(ValidationError):
        PipelineConfig(log_level="VERBOSE")
    with pytest.raises(ValidationError):
        PipelineConfig(work_dir="")


def test_strict_anatomical_orientation_validation() -> None:
    """Verify strict 3-orthogonal-axis permutation validation for DICOM spatial orientations."""
    # Valid 3D orientations (permutations of {R, L}, {A, P}, {S, I})
    valid_codes = ["RAS", "LPS", "PIR", "ASR", "RSP", "LPI", "IPL", "SRA"]
    for code in valid_codes:
        assert validate_orientation_code(code) == code
        assert validate_orientation_code(code.lower()) == code
        cfg = InferenceSettings(target_orientation=code.lower())
        assert cfg.target_orientation == code
        loader = DicomVolumeLoader(target_orientation=code.lower())
        assert loader.target_orientation == code

    # Invalid orientations: duplicate axes or opposing directions
    invalid_codes = ["RRR", "RLP", "SSS", "AAP", "XYZ", "INVALID", "RA", "RASS", "123", ""]
    for inv in invalid_codes:
        with pytest.raises(ValueError):
            validate_orientation_code(inv)
        with pytest.raises(ValueError):
            InferenceSettings(target_orientation=inv)
        with pytest.raises(ValueError):
            DicomVolumeLoader(target_orientation=inv)

    with pytest.raises(TypeError):
        validate_orientation_code(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        DicomVolumeLoader(target_orientation=123)  # type: ignore[arg-type]


def test_continuous_index_validation() -> None:
    """Verify continuous_index_to_physical_point validates coordinate tuple length."""
    meta = VolumetricMetadata(spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    assert meta.continuous_index_to_physical_point((1.0, 2.0, 3.0)) == (1.0, 2.0, 3.0)

    with pytest.raises(ValueError, match="must contain exactly 3 coordinates"):
        meta.continuous_index_to_physical_point((1.0, 2.0))

    with pytest.raises(ValueError, match="must contain exactly 3 coordinates"):
        meta.continuous_index_to_physical_point((1.0, 2.0, 3.0, 4.0))


def test_save_mask_as_nifti_input_validation(tmp_path: Path) -> None:
    """Verify save_mask_as_nifti validates mask array type, dimensions, and reference shape."""
    ref_arr = np.zeros((10, 20, 30), dtype=np.float32)
    ref_img = sitk.GetImageFromArray(ref_arr)
    valid_mask = np.zeros((10, 20, 30), dtype=np.uint8)
    out_file = tmp_path / "valid_mask.nii.gz"

    # Valid save
    saved = DicomVolumeLoader.save_mask_as_nifti(valid_mask, ref_img, out_file)
    assert saved.is_file()

    # Mismatched shape
    mismatched_mask = np.zeros((10, 20, 25), dtype=np.uint8)
    with pytest.raises(ValueError, match="does not match reference image shape"):
        DicomVolumeLoader.save_mask_as_nifti(mismatched_mask, ref_img, tmp_path / "err.nii.gz")

    # Non-3D mask
    with pytest.raises(ValueError, match="must be 3-dimensional"):
        DicomVolumeLoader.save_mask_as_nifti(np.zeros((10, 20), dtype=np.uint8), ref_img, tmp_path / "err.nii.gz")

    # Non-ndarray mask
    with pytest.raises(TypeError, match="must be a numpy ndarray"):
        DicomVolumeLoader.save_mask_as_nifti([1, 2, 3], ref_img, tmp_path / "err.nii.gz")  # type: ignore[arg-type]

    # Non-sitk reference image
    with pytest.raises(TypeError, match="must be a SimpleITK Image"):
        DicomVolumeLoader.save_mask_as_nifti(valid_mask, "not_an_image", tmp_path / "err.nii.gz")  # type: ignore[arg-type]


def test_volumetric_metrics_calculator_array_validation() -> None:
    """Verify VolumetricMetricsCalculator.validate_arrays catches dimension and empty array errors."""
    calc = VolumetricMetricsCalculator()
    valid_arr = np.ones((8, 8, 8), dtype=np.float32)
    calc.validate_arrays(valid_arr, valid_arr)

    # 2D array
    with pytest.raises(ValueError, match="Expected 3D volumetric arrays"):
        calc.validate_arrays(np.ones((8, 8)), np.ones((8, 8)))

    # 4D array
    with pytest.raises(ValueError, match="Expected 3D volumetric arrays"):
        calc.validate_arrays(np.ones((1, 8, 8, 8)), np.ones((1, 8, 8, 8)))

    # Empty array
    with pytest.raises(ValueError, match="Arrays cannot be empty"):
        calc.validate_arrays(np.zeros((0, 0, 0)), np.zeros((0, 0, 0)))

    # Non-ndarray
    with pytest.raises(TypeError, match="must be numpy ndarrays"):
        calc.validate_arrays([1, 2], [1, 2])  # type: ignore[arg-type]


def test_sliding_window_inference_tensor_validation() -> None:
    """Verify SlidingWindowInferenceEngine.execute_inference validates tensor rank, batch size, and dimensions."""
    settings = InferenceSettings(roi_size=(8, 8, 8), device="cpu")
    model = build_3d_unet(in_channels=1, out_channels=2, channels=(4, 8), strides=(2,), num_res_units=1)
    engine = SlidingWindowInferenceEngine(model=model, settings=settings)

    # Non-tensor
    with pytest.raises(TypeError, match="volume_tensor must be a torch.Tensor"):
        engine.execute_inference(np.zeros((1, 1, 8, 8, 8)))  # type: ignore[arg-type]

    # 4D tensor (missing batch or channel)
    with pytest.raises(ValueError, match="Expected 5D volume tensor"):
        engine.execute_inference(torch.zeros((1, 8, 8, 8)))

    # Batch size > 1
    with pytest.raises(ValueError, match="Expected batch size 1"):
        engine.execute_inference(torch.zeros((2, 1, 8, 8, 8)))

    # Non-positive dimension
    with pytest.raises(ValueError, match="dimensions must be strictly positive"):
        engine.execute_inference(torch.zeros((1, 1, 0, 8, 8)))


def test_chunked_iterable_negative_or_zero_chunk_size() -> None:
    """Verify chunked_iterable raises ValueError for non-positive chunk sizes."""
    with pytest.raises(ValueError, match="strictly positive integer"):
        list(chunked_iterable([1, 2, 3], chunk_size=0))

    with pytest.raises(ValueError, match="strictly positive integer"):
        list(chunked_iterable([1, 2, 3], chunk_size=-5))


def test_main_run_pipeline_input_validation() -> None:
    """Verify run_pipeline rejects invalid batch_size and orientation arguments with non-zero exit code."""
    # Negative batch size
    exit_code_batch = run_pipeline(batch_size=0)
    assert exit_code_batch != 0

    # Invalid orientation
    exit_code_orient = run_pipeline(orientation="RRR")
    assert exit_code_orient != 0


def test_pipeline_config_dotenv_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that .env configuration file seamlessly populates nested settings and parses complex types."""
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text(
        """
AZURE_STORAGE_MAX_CONCURRENCY=8
AZURE_STORAGE_TIMEOUT_SECONDS=120
DB_PORT=5433
DB_HOST=my-custom-pg.postgres.database.azure.com
INFERENCE_MODALITY="MRI"
INFERENCE_ROI_SIZE="(64, 64, 64)"
INFERENCE_TARGET_SPACING="(1.5, 1.5, 2.0)"
INFERENCE_MODEL_CHECKPOINT_PATH=""
LOG_LEVEL="DEBUG"
"""
    )
    monkeypatch.chdir(tmp_path)
    pipe_cfg = PipelineConfig()
    assert pipe_cfg.azure.max_concurrency == 8
    assert pipe_cfg.azure.timeout_seconds == 120
    assert pipe_cfg.db.port == 5433
    assert pipe_cfg.db.host == "my-custom-pg.postgres.database.azure.com"
    assert pipe_cfg.inference.modality == "MRI"
    assert pipe_cfg.inference.roi_size == (64, 64, 64)
    assert pipe_cfg.inference.target_spacing == (1.5, 1.5, 2.0)
    assert pipe_cfg.inference.model_checkpoint_path is None
    assert pipe_cfg.log_level == "DEBUG"


def test_main_parse_args_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Validate argument parsing for local-input, blob-path, series-prefix, and optional flags."""
    vol_file = tmp_path / "sample.nii.gz"
    ckpt_file = tmp_path / "model.pt"

    # 1. Local input with defaults
    monkeypatch.setattr("sys.argv", ["main.py", "--local-input", str(vol_file)])
    args = parse_args()
    assert args.local_input == vol_file
    assert args.blob_path is None
    assert args.series_prefix is None
    assert args.modality == "CT"
    assert args.orientation == "RAS"
    assert args.dry_run is False

    # 2. Blob path with custom options
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--blob-path",
            "scans/patient_01.nii.gz",
            "--container",
            "custom-in",
            "--output-container",
            "custom-out",
            "--modality",
            "MRI",
            "--orientation",
            "LPS",
            "--checkpoint",
            str(ckpt_file),
            "--batch-size",
            "250",
            "--dry-run",
        ],
    )
    args = parse_args()
    assert args.blob_path == "scans/patient_01.nii.gz"
    assert args.local_input is None
    assert args.container == "custom-in"
    assert args.output_container == "custom-out"
    assert args.modality == "MRI"
    assert args.orientation == "LPS"
    assert args.checkpoint == ckpt_file
    assert args.batch_size == 250
    assert args.dry_run is True

    # 3. Series prefix
    monkeypatch.setattr("sys.argv", ["main.py", "--series-prefix", "dicom/series_001/"])
    args = parse_args()
    assert args.series_prefix == "dicom/series_001/"
    assert args.blob_path is None
    assert args.local_input is None


def test_main_parse_args_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify parse_args rejects missing inputs, conflicting inputs, and invalid choices."""
    # Missing input
    monkeypatch.setattr("sys.argv", ["main.py"])
    with pytest.raises(SystemExit) as exc_info:
        parse_args()
    assert exc_info.value.code != 0

    # Conflicting mutually exclusive inputs
    monkeypatch.setattr("sys.argv", ["main.py", "--local-input", "vol.nii", "--blob-path", "blob.nii"])
    with pytest.raises(SystemExit) as exc_info:
        parse_args()
    assert exc_info.value.code != 0

    # Invalid modality choice
    monkeypatch.setattr("sys.argv", ["main.py", "--local-input", "vol.nii", "--modality", "XRAY"])
    with pytest.raises(SystemExit) as exc_info:
        parse_args()
    assert exc_info.value.code != 0


def test_main_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify main() entrypoint parses arguments, executes run_pipeline, and invokes sys.exit."""
    from unittest.mock import MagicMock, patch

    mock_args = MagicMock(
        blob_path="blob.nii",
        series_prefix=None,
        local_input=None,
        container="in-container",
        output_container="out-container",
        modality="CT",
        orientation="RAS",
        checkpoint=None,
        batch_size=500,
        dry_run=True,
    )

    with patch("medical_segmentation.main.parse_args", return_value=mock_args):
        with patch("medical_segmentation.main.run_pipeline", return_value=0) as mock_run:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
            mock_run.assert_called_once_with(
                blob_path="blob.nii",
                series_prefix=None,
                local_input=None,
                input_container="in-container",
                output_container="out-container",
                modality="CT",
                orientation="RAS",
                checkpoint=None,
                batch_size=500,
                dry_run=True,
            )


def test_prepare_tensor_from_sitk_input_validation() -> None:
    """Verify prepare_tensor_from_sitk validates image type, dimensionality, empty voxels, modality, spacing, and device."""
    valid_arr = np.zeros((8, 8, 8), dtype=np.float32)
    valid_img = sitk.GetImageFromArray(valid_arr)

    # 1. Non-sitk image
    with pytest.raises(TypeError, match="image must be a SimpleITK Image"):
        prepare_tensor_from_sitk(image="not_an_image")  # type: ignore[arg-type]

    # 2. 2D SimpleITK image
    img_2d = sitk.GetImageFromArray(np.zeros((8, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="Expected 3D SimpleITK image"):
        prepare_tensor_from_sitk(image=img_2d)

    # 3. Non-string modality
    with pytest.raises(TypeError, match="modality must be a string"):
        prepare_tensor_from_sitk(image=valid_img, modality=123)  # type: ignore[arg-type]

    # 4. Empty string modality
    with pytest.raises(ValueError, match="modality cannot be empty"):
        prepare_tensor_from_sitk(image=valid_img, modality="   ")

    # 5. Invalid target_spacing (non-positive or wrong dimension)
    with pytest.raises(ValueError, match="Target spacing dimensions must be strictly positive"):
        prepare_tensor_from_sitk(image=valid_img, target_spacing=(1.0, -1.0, 1.0))

    # 6. Invalid PyTorch device
    with pytest.raises(ValueError, match="Invalid PyTorch execution device"):
        prepare_tensor_from_sitk(image=valid_img, device="invalid_device_name_xyz")

    with pytest.raises(TypeError, match="device must be a string or torch.device"):
        prepare_tensor_from_sitk(image=valid_img, device=12345)  # type: ignore[arg-type]


def test_build_volumetric_transforms_input_validation() -> None:
    """Verify build_volumetric_transforms rejects invalid modality, spacing, and device inputs."""
    # Non-string modality
    with pytest.raises(TypeError, match="modality must be a string"):
        build_volumetric_transforms(modality=None)  # type: ignore[arg-type]

    # Empty modality
    with pytest.raises(ValueError, match="modality cannot be empty"):
        build_volumetric_transforms(modality="")

    # Target spacing string parsing and validation
    transform_from_str = build_volumetric_transforms(target_spacing="(1.5, 1.5, 2.0)")
    assert transform_from_str is not None

    with pytest.raises(ValueError, match="Target spacing"):
        build_volumetric_transforms(target_spacing="invalid,spacing")

    # Invalid device
    with pytest.raises(ValueError, match="Invalid PyTorch execution device"):
        build_volumetric_transforms(device="non_existent_device")


def test_build_3d_unet_input_validation() -> None:
    """Verify build_3d_unet validates hyperparameters including channels, strides, in/out channels, and dropout."""
    # in_channels <= 0
    with pytest.raises(ValueError, match="in_channels must be a strictly positive integer"):
        build_3d_unet(in_channels=0)

    # out_channels <= 0
    with pytest.raises(ValueError, match="out_channels must be a strictly positive integer"):
        build_3d_unet(out_channels=-1)

    # num_res_units < 0
    with pytest.raises(ValueError, match="num_res_units must be a non-negative integer"):
        build_3d_unet(num_res_units=-2)

    # dropout out of range [0.0, 1.0]
    with pytest.raises(ValueError, match="dropout must be a float between 0.0 and 1.0"):
        build_3d_unet(dropout=1.5)
    with pytest.raises(ValueError, match="dropout must be a float between 0.0 and 1.0"):
        build_3d_unet(dropout=-0.1)

    # channels length < 2
    with pytest.raises(ValueError, match="channels must contain at least 2 resolution levels"):
        build_3d_unet(channels=[16], strides=[])

    # non-positive channel dimensions
    with pytest.raises(ValueError, match="All channel dimensions must be strictly positive"):
        build_3d_unet(channels=[16, 0, 32], strides=[2, 2])

    # strides length mismatch
    with pytest.raises(ValueError, match="strides length"):
        build_3d_unet(channels=[16, 32, 64], strides=[2])

    # non-positive stride values
    with pytest.raises(ValueError, match="All stride values must be strictly positive"):
        build_3d_unet(channels=[16, 32], strides=[0])

    # empty norm
    with pytest.raises(ValueError, match="norm must be a non-empty string"):
        build_3d_unet(norm="   ")


def test_load_3d_unet_input_validation() -> None:
    """Verify load_3d_unet validates device, normalizes empty string checkpoint paths, and catches invalid types."""
    # Invalid device
    with pytest.raises(ValueError, match="Invalid PyTorch execution device"):
        load_3d_unet(device="invalid_dev")

    with pytest.raises(TypeError, match="device must be a string or torch.device"):
        load_3d_unet(device=999)  # type: ignore[arg-type]

    # Empty string or "none" checkpoint path normalizes to None (returns fresh initialized model)
    model_empty_str = load_3d_unet(checkpoint_path="")
    assert model_empty_str is not None

    model_none_str = load_3d_unet(checkpoint_path="none")
    assert model_none_str is not None

    # Invalid type for checkpoint_path
    with pytest.raises(TypeError, match="checkpoint_path must be a Path, str, or None"):
        load_3d_unet(checkpoint_path=[1, 2, 3])  # type: ignore[arg-type]


def test_sliding_window_inference_channel_validation() -> None:
    """Verify SlidingWindowInferenceEngine catches channel mismatches between input tensor and model configuration."""
    settings = InferenceSettings(roi_size=(8, 8, 8), in_channels=1, out_channels=2, device="cpu")
    model = build_3d_unet(in_channels=1, out_channels=2, channels=(4, 8), strides=(2,), num_res_units=1)
    engine = SlidingWindowInferenceEngine(model=model, settings=settings)

    # Input tensor has 3 channels but model expects 1
    mismatched_tensor = torch.zeros((1, 3, 8, 8, 8), dtype=torch.float32)
    with pytest.raises(ValueError, match="Expected volume tensor with 1 channel"):
        engine.execute_inference(mismatched_tensor)


def test_pipeline_orchestrator_workspace_str_handling(tmp_path: Path) -> None:
    """Verify PipelineOrchestrator accepts workspace as a string and correctly creates staging directory."""
    str_workspace = str(tmp_path / "custom_str_workspace")
    orchestrator = PipelineOrchestrator(workspace=str_workspace)
    assert isinstance(orchestrator.workspace, Path)
    assert orchestrator.workspace.is_dir()


def test_pipeline_orchestrator_mask_writer_injection(tmp_path: Path) -> None:
    """Verify that a custom VolumetricMaskWriter can be injected into PipelineOrchestrator."""
    custom_output_dir = tmp_path / "custom_masks"
    custom_writer = VolumetricMaskWriter(default_output_dir=custom_output_dir)

    orchestrator = PipelineOrchestrator(
        workspace=tmp_path / "ws",
        mask_writer=custom_writer,
    )
    assert orchestrator.mask_writer is custom_writer

    ref_img = sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32))
    ref_img.SetSpacing((1.0, 1.0, 1.0))
    mask_arr = np.zeros((8, 8, 8), dtype=np.uint8)
    meta = VolumetricMetadata(study_instance_uid="STUDY_INJECT")

    saved_path, blob_uri = orchestrator.export_mask(
        mask_array=mask_arr,
        reference_image=ref_img,
        metadata=meta,
        dry_run=True,
    )
    assert saved_path.parent == custom_output_dir
    assert saved_path.is_file()
    assert blob_uri is None


def test_pipeline_orchestrator_generate_mask_filename() -> None:
    """Verify generate_mask_filename properly formats clean UIDs and incorporates timestamps."""
    orchestrator = PipelineOrchestrator()
    meta = VolumetricMetadata(study_instance_uid="1.2.840.113619.2.55/test:run")

    filename = orchestrator.generate_mask_filename(meta, timestamp=1700000000)
    assert filename == "mask_1.2.840.113619.2.55_test_run_1700000000.nii.gz"

    # Fallback to 'run' when study_instance_uid is empty
    empty_meta = VolumetricMetadata(study_instance_uid="")
    fallback_filename = orchestrator.generate_mask_filename(empty_meta, timestamp=12345)
    assert fallback_filename == "mask_run_12345.nii.gz"


def test_pipeline_orchestrator_upload_mask_helper(tmp_path: Path) -> None:
    """Verify upload_mask helper handles dry-run, disconnected client, and successful upload."""
    mock_azure = MagicMock()
    mock_azure.is_connected = True
    mock_azure.upload_file_to_blob.return_value = "https://azure.blob/segmentations/mask.nii.gz"

    orchestrator = PipelineOrchestrator(azure_client=mock_azure, workspace=tmp_path)
    mask_file = tmp_path / "mask_test.nii.gz"
    mask_file.write_bytes(b"dummy_nifti")

    # 1. Dry run returns None and skips upload
    uri_dry = orchestrator.upload_mask(mask_file, "mask_test.nii.gz", dry_run=True)
    assert uri_dry is None
    mock_azure.upload_file_to_blob.assert_not_called()

    # 2. Live upload returns blob URI
    uri_live = orchestrator.upload_mask(mask_file, "mask_test.nii.gz", dry_run=False)
    assert uri_live == "https://azure.blob/segmentations/mask.nii.gz"
    mock_azure.upload_file_to_blob.assert_called_once()

    # 3. Disconnected client skips upload
    mock_azure.reset_mock()
    mock_azure.is_connected = False
    uri_offline = orchestrator.upload_mask(mask_file, "mask_test.nii.gz", dry_run=False)
    assert uri_offline is None
    mock_azure.upload_file_to_blob.assert_not_called()


def test_pipeline_orchestrator_context_manager_and_close() -> None:
    """Verify PipelineOrchestrator close() and context manager protocol properly dispose db_manager."""
    mock_db = MagicMock(spec=DatabaseManager)
    orchestrator = PipelineOrchestrator(db_manager=mock_db)

    # Context manager invocation
    with orchestrator as ctx:
        assert ctx is orchestrator
    mock_db.dispose.assert_called_once()

    # Explicit close invocation
    mock_db.reset_mock()
    orchestrator.close()
    mock_db.dispose.assert_called_once()


def test_pipeline_result_properties_and_to_dict(tmp_path: Path) -> None:
    """Verify PipelineResult boolean properties and to_dict serialization."""
    from medical_segmentation.pipeline.metrics import LabelSummaryMetric

    meta = VolumetricMetadata(
        patient_id="P_001",
        study_instance_uid="1.2.3.4",
        modality="CT",
        dimensions=(10, 20, 30),
        spacing=(1.0, 1.0, 2.0),
        spatial_orientation="RAS",
    )
    label = LabelSummaryMetric(
        label_id=1,
        structure_name="Lesion",
        voxel_count=50,
        volume_mm3=100.0,
        volume_cm3=0.1,
        mean_intensity=50.0,
        std_intensity=5.0,
        min_intensity=40.0,
        max_intensity=60.0,
        centroid_mm=(5.0, 10.0, 15.0),
        bounding_box=(0, 0, 0, 5, 5, 5),
    )
    metrics = SegmentationMetrics(duration_seconds=1.23, labels=[label])

    # Case 1: Fully persisted and uploaded
    res_full = PipelineResult(
        summary_metrics=metrics,
        mask_array=np.zeros((30, 20, 10), dtype=np.uint8),
        local_mask_path=tmp_path / "mask.nii.gz",
        output_blob_uri="https://blob/mask.nii.gz",
        study_id=10,
        run_id=20,
        slices_inserted=30,
        duration_seconds=2.5,
        metadata=meta,
    )
    assert res_full.is_persisted_to_db is True
    assert res_full.is_uploaded_to_cloud is True

    d = res_full.to_dict()
    assert d["study_id"] == 10
    assert d["run_id"] == 20
    assert d["duration_seconds"] == 2.5
    assert d["is_persisted_to_db"] is True
    assert d["is_uploaded_to_cloud"] is True
    assert d["patient_id"] == "P_001"
    assert d["study_instance_uid"] == "1.2.3.4"
    assert d["modality"] == "CT"
    assert d["dimensions"] == [10, 20, 30]
    assert d["spacing"] == [1.0, 1.0, 2.0]
    assert d["spatial_orientation"] == "RAS"
    assert d["labels_count"] == 1

    # Case 2: Dry run (not persisted, not uploaded)
    res_dry = PipelineResult(
        summary_metrics=metrics,
        mask_array=np.zeros((30, 20, 10), dtype=np.uint8),
        local_mask_path=tmp_path / "mask.nii.gz",
        output_blob_uri=None,
        study_id=None,
        run_id=None,
        slices_inserted=0,
        duration_seconds=1.0,
        metadata=meta,
    )
    assert res_dry.is_persisted_to_db is False
    assert res_dry.is_uploaded_to_cloud is False


def test_pipeline_orchestrator_enhanced_validation(tmp_path: Path) -> None:
    """Verify enhanced input validation across orchestrator lifecycle, ingestion, and export."""
    # 1. Empty workspace path validation
    with pytest.raises(ValueError, match="workspace cannot be an empty path"):
        PipelineOrchestrator(workspace="")
    with pytest.raises(ValueError, match="workspace cannot be an empty path"):
        PipelineOrchestrator(workspace="   ")

    orchestrator = PipelineOrchestrator(workspace=tmp_path)

    # 2. Ingest volume conflicting sources
    with pytest.raises(ValueError, match="Conflicting input volume sources"):
        orchestrator.ingest_volume(local_input=tmp_path, blob_path="vol.nii.gz")
    with pytest.raises(ValueError, match="Conflicting input volume sources"):
        orchestrator.ingest_volume(blob_path="vol.nii.gz", series_prefix="prefix/")

    # 3. Ingest volume invalid container name
    with pytest.raises(ValueError, match="Invalid Azure Blob container name"):
        orchestrator.ingest_volume(blob_path="vol.nii.gz", input_container="INVALID_CONTAINER")

    # 4. Standardize geometry invalid modality
    with pytest.raises(ValueError, match="Invalid modality 'INVALID'"):
        orchestrator.standardize_geometry(target_path=tmp_path, is_dicom_dir=False, modality="INVALID")

    # 5. Upload mask validation
    with pytest.raises(ValueError, match="mask_filename must be a non-empty string"):
        orchestrator.upload_mask(local_mask_path=tmp_path / "mask.nii.gz", mask_filename="")
    with pytest.raises(ValueError, match="Invalid Azure Blob container name"):
        orchestrator.upload_mask(
            local_mask_path=tmp_path / "mask.nii.gz",
            mask_filename="mask.nii.gz",
            output_container="in--valid",
        )

    # 6. Export mask invalid output container
    meta = VolumetricMetadata(study_instance_uid="STUDY_VAL")
    ref_img = sitk.GetImageFromArray(np.zeros((4, 4, 4), dtype=np.float32))
    mask_arr = np.zeros((4, 4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="Invalid Azure Blob container name"):
        orchestrator.export_mask(
            mask_array=mask_arr,
            reference_image=ref_img,
            metadata=meta,
            output_container="INVALID_OUT",
        )

    # 7. Run invalid arguments
    with pytest.raises(ValueError, match="Invalid modality 'INVALID'"):
        orchestrator.run(local_input=tmp_path, modality="INVALID")
    with pytest.raises(ValueError, match="batch_size must be a strictly positive integer"):
        orchestrator.run(local_input=tmp_path, batch_size=0)
    with pytest.raises(ValueError, match="Invalid Azure Blob container name"):
        orchestrator.run(local_input=tmp_path, input_container="INVALID_IN")
    with pytest.raises(ValueError, match="Invalid Azure Blob container name"):
        orchestrator.run(local_input=tmp_path, output_container="INVALID_OUT")


def test_main_run_pipeline_enhanced_validation(tmp_path: Path) -> None:
    """Verify run_pipeline handles custom config injection, conflicting inputs, and argument validation."""
    arr = np.zeros((32, 32, 32), dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    vol_path = tmp_path / "test_vol.nii.gz"
    sitk.WriteImage(img, str(vol_path))

    # 1. Custom config injection
    custom_cfg = PipelineConfig(
        inference=InferenceSettings(roi_size=(32, 32, 32), target_spacing=None, device="cpu"),
        work_dir=tmp_path / "custom_work",
    )
    exit_code = run_pipeline(local_input=vol_path, config=custom_cfg, dry_run=True)
    assert exit_code == 0

    # 2. Conflicting input sources
    assert run_pipeline(blob_path="a.nii.gz", local_input=vol_path) != 0

    # 3. Invalid modality
    assert run_pipeline(local_input=vol_path, modality="ULTRASOUND") != 0

    # 4. Invalid input/output container names
    assert run_pipeline(local_input=vol_path, input_container="INVALID_CONTAINER") != 0
    assert run_pipeline(local_input=vol_path, output_container="INVALID_CONTAINER") != 0


def test_config_enhanced_spacing_and_device() -> None:
    """Verify NaN/Inf detection in spacing and torch.device support in InferenceSettings."""
    # 1. NaN and Inf spacing validation
    with pytest.raises(ValueError, match="strictly positive"):
        InferenceSettings(target_spacing=(float("nan"), 1.0, 1.0))
    with pytest.raises(ValueError, match="strictly positive"):
        InferenceSettings(target_spacing=(float("inf"), 1.0, 1.0))

    # 2. torch.device object support in InferenceSettings
    dev = torch.device("cpu")
    cfg = InferenceSettings(device=dev)
    assert cfg.device == "cpu"

    # 3. validate_azure_container_name empty check
    with pytest.raises(ValueError, match="Container name cannot be empty"):
        validate_azure_container_name("")
    with pytest.raises(ValueError, match="Container name cannot be empty"):
        validate_azure_container_name("   ")


def test_pipeline_orchestrator_volume_ingester_injection(tmp_path: Path) -> None:
    """Verify that a custom VolumeIngester can be injected into PipelineOrchestrator."""
    mock_ingester = MagicMock(spec=VolumeIngester)
    expected_path = tmp_path / "custom_injected.nii.gz"
    mock_ingester.ingest.return_value = (expected_path, False)

    orchestrator = PipelineOrchestrator(
        workspace=tmp_path,
        volume_ingester=mock_ingester,
    )

    result_path, is_dir = orchestrator.ingest_volume(blob_path="test_blob.nii.gz")
    assert result_path == expected_path
    assert is_dir is False
    mock_ingester.ingest.assert_called_once_with(
        blob_path="test_blob.nii.gz",
        series_prefix=None,
        local_input=None,
        input_container=None,
    )


def test_pipeline_config_from_file_and_dict_lifecycle(tmp_path: Path) -> None:
    """Verify PipelineConfig from_file, from_dict, to_dict, and to_file methods."""
    # 1. to_dict serialization
    cfg = PipelineConfig()
    cfg_dict = cfg.to_dict()
    assert isinstance(cfg_dict, dict)
    assert "azure" in cfg_dict
    assert "db" in cfg_dict
    assert "inference" in cfg_dict
    assert list(cfg_dict["inference"]["roi_size"]) == [96, 96, 96]

    # 2. to_file and from_file roundtrip
    out_json = tmp_path / "saved_config.json"
    written_path = cfg.to_file(out_json)
    assert written_path == out_json
    assert out_json.is_file()

    loaded_cfg = PipelineConfig.from_file(out_json)
    assert loaded_cfg.log_level == cfg.log_level
    assert loaded_cfg.inference.modality == cfg.inference.modality
    assert loaded_cfg.work_dir == cfg.work_dir

    # 3. from_dict with partial nested configurations
    custom_dict = {
        "inference": {
            "modality": "MRI",
            "target_orientation": "LPS",
            "target_spacing": [1.5, 1.5, 2.5],
            "roi_size": [64, 64, 64],
        },
        "log_level": "DEBUG",
    }
    dict_cfg = PipelineConfig.from_dict(custom_dict)
    assert dict_cfg.inference.modality == "MRI"
    assert dict_cfg.inference.target_orientation == "LPS"
    assert dict_cfg.inference.target_spacing == (1.5, 1.5, 2.5)
    assert dict_cfg.inference.roi_size == (64, 64, 64)
    assert dict_cfg.log_level == "DEBUG"

    # 4. Error handling: non-existent file
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        PipelineConfig.from_file(tmp_path / "missing.json")

    # 5. Error handling: empty path
    with pytest.raises(ValueError, match="Configuration file path cannot be empty"):
        PipelineConfig.from_file("")
    with pytest.raises(TypeError, match="path must be a string or Path"):
        PipelineConfig.from_file(12345)  # type: ignore[arg-type]

    # 6. Error handling: to_file empty path
    with pytest.raises(ValueError, match="Destination file path cannot be empty"):
        cfg.to_file("")
    with pytest.raises(TypeError, match="path must be a string or Path"):
        cfg.to_file(None)  # type: ignore[arg-type]

    # 7. Error handling: malformed JSON syntax
    bad_json = tmp_path / "corrupted.json"
    bad_json.write_text("{invalid_json: true", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON configuration"):
        PipelineConfig.from_file(bad_json)

    # 8. Error handling: JSON root is not an object/dictionary
    array_json = tmp_path / "array.json"
    array_json.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object/dictionary"):
        PipelineConfig.from_file(array_json)

    # 9. Error handling: non-dict data in from_dict
    with pytest.raises(TypeError, match="Configuration data must be a dictionary"):
        PipelineConfig.from_dict(["not", "a", "dict"])  # type: ignore[arg-type]


def test_enhanced_input_validation_and_type_checks() -> None:
    """Verify input validation rules for target spacing, ROI size, checkpoint path, and DB credentials."""
    # 1. Target spacing rejects boolean elements
    with pytest.raises(TypeError, match="Target spacing dimensions cannot be boolean"):
        parse_target_spacing((True, 1.0, 1.0))
    with pytest.raises(TypeError, match="Target spacing dimensions cannot be boolean"):
        parse_target_spacing([1.0, False, 1.0])

    # 2. ROI size rejects boolean elements and non-integer floats
    with pytest.raises(TypeError, match="ROI size dimensions cannot be boolean"):
        parse_roi_size((True, 96, 96))
    with pytest.raises(TypeError, match="ROI size dimensions cannot be boolean"):
        parse_roi_size([96, False, 96])
    with pytest.raises(ValueError, match="ROI size values must be valid integers"):
        parse_roi_size((96.5, 96.0, 96.0))
    # Valid float representations of integers are accepted
    assert parse_roi_size((96.0, 96.0, 96.0)) == (96, 96, 96)

    # 3. Model checkpoint path normalization
    assert parse_model_checkpoint_path(Path("   ")) is None
    assert parse_model_checkpoint_path("   ") is None
    assert parse_model_checkpoint_path(Path("valid_model.pt")) == Path("valid_model.pt")

    # 4. DatabaseSettings rejects empty or whitespace-only host, name, user
    with pytest.raises(ValidationError):
        DatabaseSettings(host="   ")
    with pytest.raises(ValidationError):
        DatabaseSettings(name="   ")
    with pytest.raises(ValidationError):
        DatabaseSettings(user="   ")

    # 5. AzureStorageSettings strips credentials
    az_cfg = AzureStorageSettings(
        account_name="  myaccount  ",
        account_key="  mysecretkey  ",
        connection_string="  AccountName=myaccount  ",
    )
    assert az_cfg.account_name == "myaccount"
    assert az_cfg.account_key == "mysecretkey"
    assert az_cfg.connection_string == "AccountName=myaccount"


def test_cli_config_argument_and_main_dispatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify CLI --config option is parsed and loaded in parse_args and main()."""
    cfg_file = tmp_path / "custom_cli_config.json"
    custom_cfg = PipelineConfig(
        inference=InferenceSettings(modality="MRI", roi_size=(64, 64, 64)),
        log_level="DEBUG",
    )
    custom_cfg.to_file(cfg_file)

    vol_file = tmp_path / "vol.nii.gz"
    vol_file.touch()

    # 1. parse_args with --config
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--local-input",
            str(vol_file),
            "--config",
            str(cfg_file),
        ],
    )
    parsed = parse_args()
    assert parsed.config == cfg_file
    assert parsed.local_input == vol_file

    # 2. main() with --config dispatches to run_pipeline with custom config
    with patch("medical_segmentation.main.run_pipeline", return_value=0) as mock_run:
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert "config" in kwargs
        assert kwargs["config"].inference.modality == "MRI"
        assert kwargs["config"].inference.roi_size == (64, 64, 64)

    # 3. main() with --config and --check-health
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "--check-health",
            "--config",
            str(cfg_file),
        ],
    )
    with patch("medical_segmentation.main.run_health_check", return_value=0) as mock_health:
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
        mock_health.assert_called_once()
        _, health_kwargs = mock_health.call_args
        assert health_kwargs.get("config") is not None
        assert health_kwargs["config"].inference.modality == "MRI"


def test_volumetric_metrics_calculator_compute_label_summary_direct() -> None:
    """Verify compute_label_summary directly computes volume, intensity, and geometric statistics."""
    calc = VolumetricMetricsCalculator()
    meta = VolumetricMetadata(
        spacing=(1.0, 1.0, 2.0),
        origin=(0.0, 0.0, 0.0),
        spatial_orientation="RAS",
    )
    raw_array = np.full((10, 10, 10), 100.0, dtype=np.float32)

    # 1. Zero voxel mask returns None
    empty_mask = np.zeros((10, 10, 10), dtype=bool)
    assert calc.compute_label_summary(raw_array, empty_mask, 1, meta) is None

    # 2. Non-empty mask returns valid LabelSummaryMetric
    active_mask = np.zeros((10, 10, 10), dtype=bool)
    active_mask[2:6, 2:6, 2:6] = True  # 4 * 4 * 4 = 64 voxels
    summary = calc.compute_label_summary(raw_array, active_mask, 1, meta)
    assert summary is not None
    assert isinstance(summary, LabelSummaryMetric)
    assert summary.label_id == 1
    assert summary.voxel_count == 64
    assert summary.volume_mm3 == pytest.approx(64 * (1.0 * 1.0 * 2.0))
    assert summary.volume_cm3 == pytest.approx((64 * 2.0) / 1000.0)
    assert summary.mean_intensity == pytest.approx(100.0)
    assert summary.min_intensity == pytest.approx(100.0)
    assert summary.max_intensity == pytest.approx(100.0)


def test_volumetric_metrics_calculator_generate_slice_metrics_defaults_and_fallback_metadata() -> None:
    """Verify generate_slice_metrics handles default None bounds/counts and metadata without plane resolver."""
    calc = VolumetricMetricsCalculator()
    raw = np.full((6, 6, 6), 50.0, dtype=np.float32)
    mask = np.zeros((6, 6, 6), dtype=np.int32)
    mask[2, 1:4, 1:4] = 1  # Slice 2
    mask[4, 1:4, 1:4] = 1  # Slice 4

    # 1. Default None bounds and slice counts (traverses entire depth and computes counts per slice)
    meta = VolumetricMetadata(
        spacing=(1.5, 1.5, 2.0),
        origin=(0.0, 0.0, 0.0),
        spatial_orientation="RAS",
    )
    records = list(
        calc.generate_slice_metrics(
            raw_array=raw,
            mask_array=mask,
            unique_labels=[1],
            metadata=meta,
            label_bounds=None,
            label_slice_counts=None,
        )
    )
    assert len(records) == 2
    assert records[0]["slice_index"] == 2
    assert records[1]["slice_index"] == 4
    assert records[0]["slice_voxel_count"] == 9
    assert records[0]["cross_sectional_area_mm2"] == pytest.approx(9 * (1.5 * 1.5))
    assert records[0]["plane"] == "axial"

    # 2. Metadata lacking get_slice_plane_and_position method falls back to axial and coordinate transform
    class MinimalMetadata:
        pixel_area_axial_mm2 = 2.25

        def continuous_index_to_physical_point(self, idx: Tuple[float, float, float]) -> Tuple[float, float, float]:
            return (0.0, 0.0, float(idx[2]) * 2.0)

    min_meta = MinimalMetadata()
    fallback_records = list(
        calc.generate_slice_metrics(
            raw_array=raw,
            mask_array=mask,
            unique_labels=[1],
            metadata=min_meta,  # type: ignore[arg-type]
        )
    )
    assert len(fallback_records) == 2
    assert fallback_records[0]["plane"] == "axial"
    assert fallback_records[0]["slice_position_mm"] == pytest.approx(4.0)  # index 2 * 2.0

    # 3. Known slice counts with zero-count slices skipped
    known_counts_records = list(
        calc.generate_slice_metrics(
            raw_array=raw,
            mask_array=mask,
            unique_labels=[1],
            metadata=meta,
            label_bounds={1: (0, 5)},
            label_slice_counts={1: np.array([0, 0, 9, 0, 9, 0])},
        )
    )
    assert len(known_counts_records) == 2
    assert known_counts_records[0]["slice_index"] == 2
    assert known_counts_records[1]["slice_index"] == 4


def test_pipeline_orchestrator_dynamic_orientation_change() -> None:
    """Verify standardize_geometry re-instantiates volume_loader when a different orientation is requested."""
    config = PipelineConfig(inference=InferenceSettings(target_orientation="RAS"))
    orchestrator = PipelineOrchestrator(config=config)
    assert orchestrator.volume_loader.target_orientation == "RAS"

    sitk_img = sitk.GetImageFromArray(np.zeros((4, 4, 4), dtype=np.float32))
    meta = VolumetricMetadata(spatial_orientation="LPS")
    with patch.object(DicomVolumeLoader, "load", return_value=(sitk_img, meta)):
        _, returned_meta = orchestrator.standardize_geometry(Path("/mock/scan.nii"), is_dicom_dir=False, orientation="LPS")
        assert orchestrator.volume_loader.target_orientation == "LPS"
        assert returned_meta.spatial_orientation == "LPS"


def test_pipeline_orchestrator_custom_loader_dispatch_fallbacks() -> None:
    """Verify standardize_geometry falls back to load_dicom_series or load_volume_file if loader lacks load()."""
    sitk_img = sitk.GetImageFromArray(np.zeros((4, 4, 4), dtype=np.float32))
    meta = VolumetricMetadata()

    custom_loader = MagicMock(spec=["load_dicom_series", "load_volume_file", "target_orientation"])
    custom_loader.target_orientation = "RAS"
    custom_loader.load_dicom_series.return_value = (sitk_img, meta)
    custom_loader.load_volume_file.return_value = (sitk_img, meta)

    orchestrator = PipelineOrchestrator(volume_loader=custom_loader)

    # 1. is_dicom_dir=True -> load_dicom_series
    orchestrator.standardize_geometry(Path("/mock/dicom_dir"), is_dicom_dir=True)
    custom_loader.load_dicom_series.assert_called_once_with(Path("/mock/dicom_dir"))

    # 2. is_dicom_dir=False -> load_volume_file
    orchestrator.standardize_geometry(Path("/mock/file.nii"), is_dicom_dir=False)
    custom_loader.load_volume_file.assert_called_once_with(Path("/mock/file.nii"))


def test_pipeline_orchestrator_db_persistence_resilience_and_batch_size_validation() -> None:
    """Verify persist_to_database validates batch_size and gracefully logs and survives db exceptions."""
    orchestrator = PipelineOrchestrator()
    meta = VolumetricMetadata()
    metrics = SegmentationMetrics(duration_seconds=1.0)

    # 1. batch_size <= 0 raises ValueError
    with pytest.raises(ValueError, match="batch_size must be a strictly positive integer"):
        orchestrator.persist_to_database(meta, metrics, iter([]), batch_size=0)

    # 2. Database exception handling: returns (None, None, 0) and does not raise
    mock_db = MagicMock()
    mock_db.init_tables.side_effect = RuntimeError("PostgreSQL connection refused")
    orchestrator_with_mock = PipelineOrchestrator(db_manager=mock_db)

    study_id, run_id, inserted = orchestrator_with_mock.persist_to_database(meta, metrics, iter([]))
    assert study_id is None
    assert run_id is None
    assert inserted == 0


def test_extract_state_dict_non_tensor_and_nested_prefix() -> None:
    """Verify extract_state_dict ignores non-tensor entries and correctly strips nested prefixes."""
    model = build_3d_unet(in_channels=1, out_channels=2)
    target_keys = list(model.state_dict().keys())

    # Build checkpoint with non-tensor entries and nested prefixes
    first_key = target_keys[0]
    raw_payload = {
        "epoch": 42,
        "loss": 0.05,
        "metadata": {"version": "2.0"},
        first_key: torch.zeros_like(model.state_dict()[first_key]),
    }
    cleaned = extract_state_dict(raw_payload, target_keys=target_keys)
    assert first_key in cleaned
    assert "epoch" not in cleaned
    assert "loss" not in cleaned
    assert "metadata" not in cleaned

    # Test "model.model." prefix stripping: first_key is "model.0.conv...", so "model." + first_key starts with "model.model."
    nested_payload = {
        f"model.{first_key}": torch.zeros_like(model.state_dict()[first_key]),
    }
    cleaned_nested = extract_state_dict(nested_payload, target_keys=target_keys)
    assert first_key in cleaned_nested


def test_load_3d_unet_string_path_and_partial_checkpoint_warning(tmp_path: Path) -> None:
    """Verify load_3d_unet supports string checkpoint paths and logs partial-match warning on non-strict load."""
    model = build_3d_unet(in_channels=1, out_channels=2)
    state = model.state_dict()
    target_keys = list(state.keys())

    # Create partial checkpoint with only 1 parameter
    partial_state = {target_keys[0]: state[target_keys[0]]}
    ckpt_file = tmp_path / "partial_unet.pt"
    torch.save(partial_state, ckpt_file)

    # Load using string path and strict=False
    loaded_model = load_3d_unet(
        checkpoint_path=str(ckpt_file),
        in_channels=1,
        out_channels=2,
        strict=False,
    )
    assert isinstance(loaded_model, torch.nn.Module)


def test_config_settings_string_parsers_and_edge_cases() -> None:
    """Verify string parsers and validators for target_spacing, roi_size, and container names."""
    # 1. parse_target_spacing edge cases
    with pytest.raises(ValueError, match="Target spacing string must contain 3 comma-separated floats"):
        parse_target_spacing("1.0, 2.0")
    with pytest.raises(ValueError, match="Target spacing string must contain 3 comma-separated floats"):
        parse_target_spacing("1.0, 2.0, abc")
    with pytest.raises(ValueError, match="Target spacing values must be valid floats"):
        parse_target_spacing(["1.0", "not_a_float", "2.0"])
    with pytest.raises(ValueError, match="Target spacing dimensions must be strictly positive"):
        parse_target_spacing([1.0, 2.0, -0.5])
    with pytest.raises(ValueError, match="Target spacing dimensions must be strictly positive"):
        parse_target_spacing([1.0, 2.0, float("nan")])
    with pytest.raises(TypeError, match="Target spacing dimensions cannot be boolean values"):
        parse_target_spacing([1.0, 2.0, True])
    with pytest.raises(TypeError, match="Invalid type for target_spacing"):
        parse_target_spacing(12345)

    # 2. parse_roi_size edge cases
    with pytest.raises(ValueError, match="ROI size string must contain 3 comma-separated integers"):
        parse_roi_size("64, 64")
    with pytest.raises(ValueError, match="ROI size string must contain 3 comma-separated integers"):
        parse_roi_size("64, 64, invalid_int")
    with pytest.raises(TypeError, match="Invalid type for roi_size"):
        parse_roi_size(9999)

    # 3. parse_model_checkpoint_path and InferenceSettings type edge cases
    with pytest.raises(TypeError, match="Invalid type for model_checkpoint_path"):
        parse_model_checkpoint_path(12345)
    assert InferenceSettings.normalize_modality(12345) == 12345
    with pytest.raises(TypeError, match="Execution device must be a string or torch.device"):
        InferenceSettings.validate_device(12345)

    # 4. validate_azure_container_name type error
    with pytest.raises(TypeError, match="Container name must be a string"):
        validate_azure_container_name(12345)  # type: ignore[arg-type]

    # 5. AzureStorageSettings credentials stripping
    assert AzureStorageSettings.strip_credentials(None) == ""
    assert AzureStorageSettings.strip_credentials(123) == "123"

    # 6. DatabaseSettings non-string field validator
    with pytest.raises(TypeError, match="host must be a string"):
        DatabaseSettings.validate_non_empty_strings(12345, MagicMock(field_name="host"))

    # 7. PipelineConfig work_dir, log_format, and log_file validators
    with pytest.raises((TypeError, ValidationError)):
        PipelineConfig(work_dir=12345)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="log_format must be a string"):
        PipelineConfig.validate_log_format(123)  # type: ignore[arg-type]
    assert PipelineConfig.validate_log_file("none") is None
    assert PipelineConfig.validate_log_file("null") is None
    assert PipelineConfig.validate_log_file("   ") is None
    assert PipelineConfig.validate_log_file(Path("/tmp/run.log")) == Path("/tmp/run.log")
    with pytest.raises(TypeError, match="Invalid type for log_file"):
        PipelineConfig.validate_log_file(12345)


def test_prepare_tensor_from_sitk_empty_image_validation() -> None:
    """Verify prepare_tensor_from_sitk raises ValueError when SimpleITK image has 0 pixels."""
    empty_img = sitk.Image(0, 0, 0, sitk.sitkFloat32)
    with pytest.raises(ValueError, match="SimpleITK image dimensions must be strictly positive"):
        prepare_tensor_from_sitk(image=empty_img)


def test_main_script_entrypoint_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify executing main.py as a script (__main__) triggers main() entrypoint."""
    import runpy
    import sys
    monkeypatch.setattr(sys, "argv", ["main.py", "--check-health"])
    with patch("medical_segmentation.diagnostics.run_health_check", return_value=0) as mock_health:
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_path(str(Path(__file__).resolve().parent.parent / "main.py"), run_name="__main__")
        assert exc_info.value.code == 0
        mock_health.assert_called_once()



def test_package_init_and_main_namespace_registration_bootstrap() -> None:
    """Verify __init__.py and main.py bootstrap 'medical_segmentation' into sys.modules when absent."""
    import runpy
    import sys
    repo_root = Path(__file__).resolve().parent.parent

    # 1. __init__.py bootstrap
    saved = sys.modules.pop("medical_segmentation", None)
    try:
        assert "medical_segmentation" not in sys.modules
        init_path = repo_root / "__init__.py"
        runpy.run_path(str(init_path))
        assert "medical_segmentation" in sys.modules
    finally:
        if saved is not None:
            sys.modules["medical_segmentation"] = saved

    # 2. main.py bootstrap without running __main__
    saved = sys.modules.pop("medical_segmentation", None)
    try:
        assert "medical_segmentation" not in sys.modules
        main_path = repo_root / "main.py"
        runpy.run_path(str(main_path), run_name="medical_segmentation.main")
        assert "medical_segmentation" in sys.modules
    finally:
        if saved is not None:
            sys.modules["medical_segmentation"] = saved


def test_extract_state_dict_non_tensor_subdict_wrapper_fallback() -> None:
    """Verify extract_state_dict skips nested wrapper dicts that contain no torch.Tensor and continues searching."""
    target_keys = ["layer1.weight", "layer1.bias"]
    ckpt_payload = {
        "state_dict": {"meta": "version_1", "info": {"foo": "bar"}},
        "model": {
            "layer1.weight": torch.randn(2, 2),
            "layer1.bias": torch.randn(2),
        },
    }
    cleaned = extract_state_dict(ckpt_payload, target_keys=target_keys)
    assert set(cleaned.keys()) == set(target_keys)


def test_load_3d_unet_empty_path_resilience() -> None:
    """Verify load_3d_unet treats Path('') and Path('.') as None without error."""
    net1 = load_3d_unet(checkpoint_path=Path(""))
    assert isinstance(net1, torch.nn.Module)

    net2 = load_3d_unet(checkpoint_path=Path("."))
    assert isinstance(net2, torch.nn.Module)


def test_orchestrator_execute_inference_engine_reuse() -> None:
    """Verify PipelineOrchestrator reuses its existing inference_engine across multiple calls to execute_inference."""
    orchestrator = PipelineOrchestrator()
    vol_tensor = torch.zeros((1, 1, 8, 8, 8), dtype=torch.float32)

    mask1, dur1 = orchestrator.execute_inference(vol_tensor)
    assert orchestrator.inference_engine is not None
    engine_ref = orchestrator.inference_engine

    mask2, dur2 = orchestrator.execute_inference(vol_tensor)
    assert orchestrator.inference_engine is engine_ref


def test_metrics_calculator_duck_typed_metadata_and_empty_summaries() -> None:
    """Verify VolumetricMetricsCalculator handles duck-typed metadata and None summaries/counts."""
    calc = VolumetricMetricsCalculator()
    raw_arr = np.ones((4, 8, 8), dtype=np.float32)
    mask_arr = np.zeros((4, 8, 8), dtype=np.uint8)
    mask_arr[1:3, 2:6, 2:6] = 1

    class DuckTypedMeta:
        def __init__(self) -> None:
            self.spatial_orientation = "RAS"
            self.spacing = (1.0, 1.0, 2.0)
            self.dimensions = (8, 8, 4)
            self.voxel_volume_mm3 = 2.0
            self.pixel_area_axial_mm2 = 1.0

        def continuous_index_to_physical_point(self, pt: Tuple[float, float, float]) -> Tuple[float, float, float]:
            return (pt[0] * 1.0, pt[1] * 1.0, pt[2] * 2.0)

    duck_meta = DuckTypedMeta()
    plane = calc._resolve_slice_plane(duck_meta)  # type: ignore[arg-type]
    assert plane == "axial"

    positions = calc._compute_slice_positions(duck_meta, depth=4)  # type: ignore[arg-type]
    assert len(positions) == 4
    assert positions == [0.0, 2.0, 4.0, 6.0]

    slice_records = list(calc.generate_slice_metrics(
        raw_array=raw_arr,
        mask_array=mask_arr,
        unique_labels=[1],
        metadata=duck_meta,  # type: ignore[arg-type]
    ))
    assert len(slice_records) == 2
    assert slice_records[0]["plane"] == "axial"
    assert slice_records[0]["cross_sectional_area_mm2"] == 16.0

    with patch.object(calc, "_compute_label_summary_with_counts", return_value=(None, None)):
        res_none = calc.compute_metrics(raw_arr, mask_arr, duck_meta, duration_seconds=1.0)  # type: ignore[arg-type]
        assert len(res_none[0].labels) == 0

    dummy_summary = LabelSummaryMetric(
        label_id=1,
        structure_name="Test",
        voxel_count=32,
        volume_mm3=64.0,
        volume_cm3=0.064,
        mean_intensity=1.0,
        std_intensity=0.0,
        min_intensity=1.0,
        max_intensity=1.0,
        centroid_mm=(0.0, 0.0, 0.0),
        bounding_box=(2, 2, 1, 5, 5, 2),
    )
    with patch.object(calc, "_compute_label_summary_with_counts", return_value=(dummy_summary, None)):
        res_summary_only = calc.compute_metrics(raw_arr, mask_arr, duck_meta, duration_seconds=1.0)  # type: ignore[arg-type]
        assert len(res_summary_only[0].labels) == 1


def test_conftest_namespace_bootstrap_already_registered() -> None:
    """Verify conftest.py executes cleanly when 'medical_segmentation' is already in sys.modules."""
    import runpy
    import sys
    assert "medical_segmentation" in sys.modules
    conftest_path = Path(__file__).resolve().parent / "conftest.py"
    runpy.run_path(str(conftest_path))
    assert "medical_segmentation" in sys.modules


def test_pipeline_config_yaml_and_toml_support(tmp_path: Path) -> None:
    """Verify PipelineConfig multi-format configuration file loading and serialization."""
    # 1. YAML round-trip (.yaml and .yml)
    cfg = PipelineConfig(
        inference=InferenceSettings(modality="MRI", roi_size=(64, 64, 64), blend_mode="constant"),
        db=DatabaseSettings(batch_size=250, pool_recycle=900, pool_pre_ping=False),
        log_level="DEBUG",
    )
    yaml_file = tmp_path / "config.yaml"
    cfg.to_file(yaml_file)
    assert yaml_file.is_file()

    loaded_yaml = PipelineConfig.from_file(yaml_file)
    assert loaded_yaml.inference.modality == "MRI"
    assert loaded_yaml.inference.roi_size == (64, 64, 64)
    assert loaded_yaml.inference.blend_mode == "constant"
    assert loaded_yaml.db.batch_size == 250
    assert loaded_yaml.db.pool_recycle == 900
    assert loaded_yaml.db.pool_pre_ping is False
    assert loaded_yaml.log_level == "DEBUG"

    # Also test .yml extension
    yml_file = tmp_path / "config.yml"
    loaded_yaml.to_file(yml_file)
    assert yml_file.is_file()
    loaded_yml = PipelineConfig.from_file(yml_file)
    assert loaded_yml.inference.modality == "MRI"

    # 2. TOML file loading
    toml_file = tmp_path / "config.toml"
    toml_content = """
    log_level = "WARNING"
    log_format = "json"

    [inference]
    modality = "CBCT"
    target_orientation = "LPS"
    roi_size = [48, 48, 48]
    blend_mode = "gaussian"

    [db]
    batch_size = 100
    pool_recycle = 1200
    pool_pre_ping = true
    """
    toml_file.write_text(toml_content, encoding="utf-8")
    loaded_toml = PipelineConfig.from_file(toml_file)
    assert loaded_toml.log_level == "WARNING"
    assert loaded_toml.log_format == "json"
    assert loaded_toml.inference.modality == "CBCT"
    assert loaded_toml.inference.target_orientation == "LPS"
    assert loaded_toml.inference.roi_size == (48, 48, 48)
    assert loaded_toml.db.batch_size == 100
    assert loaded_toml.db.pool_recycle == 1200
    assert loaded_toml.db.pool_pre_ping is True

    # 3. Corrupted YAML error handling
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("inference: [unclosed list", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid YAML configuration"):
        PipelineConfig.from_file(bad_yaml)

    # 4. Non-dict YAML error handling
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object/dictionary"):
        PipelineConfig.from_file(list_yaml)

    # 5. Corrupted TOML error handling
    bad_toml = tmp_path / "bad.toml"
    bad_toml.write_text("invalid_toml = [", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid TOML configuration"):
        PipelineConfig.from_file(bad_toml)

    # 6. Non-dict TOML / empty table handling (empty TOML yields empty dict -> valid default config)
    empty_toml = tmp_path / "empty.toml"
    empty_toml.write_text("", encoding="utf-8")
    empty_cfg = PipelineConfig.from_file(empty_toml)
    assert isinstance(empty_cfg, PipelineConfig)


def test_pipeline_config_from_dict_flat_normalization() -> None:
    """Verify PipelineConfig.from_dict routes top-level/flat keys into nested sections."""
    flat_data = {
        "modality": "MRI",
        "target_orientation": "LPS",
        "target_spacing": [1.5, 1.5, 2.0],
        "roi_size": [64, 64, 64],
        "blend_mode": "constant",
        "batch_size": 250,
        "pool_recycle": 1500,
        "pool_pre_ping": True,
        "input_container": "custom-input-blob",
        "account_name": "medimagingstore",
        "work_dir": "/tmp/custom_work",
        "log_level": "DEBUG",
    }
    cfg = PipelineConfig.from_dict(flat_data)
    assert cfg.inference.modality == "MRI"
    assert cfg.inference.target_orientation == "LPS"
    assert cfg.inference.target_spacing == (1.5, 1.5, 2.0)
    assert cfg.inference.roi_size == (64, 64, 64)
    assert cfg.inference.blend_mode == "constant"
    assert cfg.db.batch_size == 250
    assert cfg.db.pool_recycle == 1500
    assert cfg.db.pool_pre_ping is True
    assert cfg.azure.input_container == "custom-input-blob"
    assert cfg.azure.account_name == "medimagingstore"
    assert cfg.work_dir == Path("/tmp/custom_work")
    assert cfg.log_level == "DEBUG"


def test_database_settings_pool_recycle_and_pre_ping() -> None:
    """Verify DatabaseSettings pool_recycle and pool_pre_ping configuration and validation."""
    db_default = DatabaseSettings()
    assert db_default.pool_recycle == 1800
    assert db_default.pool_pre_ping is True

    custom_db = DatabaseSettings(pool_recycle=600, pool_pre_ping=False)
    assert custom_db.pool_recycle == 600
    assert custom_db.pool_pre_ping is False

    with pytest.raises(ValidationError):
        DatabaseSettings(pool_recycle=0)
    with pytest.raises(ValidationError):
        DatabaseSettings(pool_recycle=-10)

    # Verify DatabaseManager configures engine with pool_recycle and pool_pre_ping
    with patch("medical_segmentation.db.database.create_engine") as mock_create_engine:
        DatabaseManager(settings=custom_db)
        mock_create_engine.assert_called_once()
        _, kwargs = mock_create_engine.call_args
        assert kwargs.get("pool_recycle") == 600
        assert kwargs.get("pool_pre_ping") is False


def test_inference_settings_blend_mode_validation_and_usage() -> None:
    """Verify InferenceSettings blend_mode validation and propagation to sliding_window_inference."""
    inf_default = InferenceSettings()
    assert inf_default.blend_mode == "gaussian"

    inf_constant = InferenceSettings(blend_mode="constant")
    assert inf_constant.blend_mode == "constant"

    inf_normalized = InferenceSettings(blend_mode=" GAUSSIAN ")
    assert inf_normalized.blend_mode == "gaussian"

    with pytest.raises(ValidationError):
        InferenceSettings(blend_mode="linear")
    with pytest.raises(ValidationError):
        InferenceSettings(blend_mode=123)  # type: ignore[arg-type]

    # Verify SlidingWindowInferenceEngine uses configured blend_mode
    model = build_3d_unet(in_channels=1, out_channels=2, channels=(4, 8), strides=(2,), num_res_units=1)
    engine = SlidingWindowInferenceEngine(model=model, settings=inf_constant)
    vol_tensor = torch.zeros((1, 1, 8, 8, 8))

    with patch("medical_segmentation.pipeline.inference.sliding_window_inference") as mock_sw:
        mock_sw.return_value = torch.zeros((1, 2, 8, 8, 8))
        _ = engine.execute_inference(vol_tensor)
        mock_sw.assert_called_once()
        _, sw_kwargs = mock_sw.call_args
        assert sw_kwargs.get("mode") == "constant"


def test_main_cli_configuration_precedence_and_explicit_args(tmp_path: Path) -> None:
    """Verify main() CLI configuration precedence when --config is supplied."""
    cfg_file = tmp_path / "custom_precedence.json"
    custom_cfg = PipelineConfig(
        inference=InferenceSettings(modality="MRI", target_orientation="LPS"),
        db=DatabaseSettings(batch_size=250),
        log_level="DEBUG",
    )
    custom_cfg.to_file(cfg_file)

    vol_file = tmp_path / "vol.nii.gz"
    vol_file.touch()

    # 1. Calling parse_args directly with explicit sequence
    parsed = parse_args(["--local-input", str(vol_file), "--config", str(cfg_file)])
    assert parsed.config == cfg_file
    assert parsed.local_input == vol_file

    # 2. main() with explicit sequence: unspecified CLI flags fall back to custom config
    with patch("medical_segmentation.main.run_pipeline", return_value=0) as mock_run:
        with pytest.raises(SystemExit) as exc:
            main(["--local-input", str(vol_file), "--config", str(cfg_file)])
        assert exc.value.code == 0
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs["modality"] == "MRI"
        assert kwargs["orientation"] == "LPS"
        assert kwargs["batch_size"] == 250
        assert kwargs["config"].inference.modality == "MRI"

    # 3. main() with explicit sequence: explicit CLI flag overrides custom config
    with patch("medical_segmentation.main.run_pipeline", return_value=0) as mock_run:
        with pytest.raises(SystemExit) as exc:
            main(["--local-input", str(vol_file), "--config", str(cfg_file), "--modality", "CBCT", "--batch-size", "150"])
        assert exc.value.code == 0
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        assert kwargs["modality"] == "CBCT"
        assert kwargs["batch_size"] == 150
        assert kwargs["orientation"] == "LPS"


def test_mask_writer_input_validation(tmp_path: Path) -> None:
    """Verify save_mask_as_nifti and VolumetricMaskWriter reject empty/whitespace paths."""
    from medical_segmentation.io.mask_writer import VolumetricMaskWriter, save_mask_as_nifti

    ref = sitk.Image(8, 8, 8, sitk.sitkFloat32)
    mask = np.zeros((8, 8, 8), dtype=np.uint8)

    with pytest.raises(ValueError, match="output_path must be specified"):
        save_mask_as_nifti(mask, reference_image=ref, output_path="")
    with pytest.raises(ValueError, match="output_path must be specified"):
        save_mask_as_nifti(mask, reference_image=ref, output_path="   ")

    writer = VolumetricMaskWriter()
    with pytest.raises(ValueError, match="Either output_path or filename"):
        writer.write_mask(mask, reference=ref, output_path="")
    with pytest.raises(ValueError, match="Either output_path or filename"):
        writer.write_mask(mask, reference=ref, filename="   ")


def test_metrics_calculator_metadata_and_duration_validation() -> None:
    """Verify VolumetricMetricsCalculator rejects None metadata and invalid duration_seconds."""
    calc = VolumetricMetricsCalculator()
    arr = np.ones((8, 8, 8), dtype=np.float32)
    mask = np.zeros((8, 8, 8), dtype=np.uint8)

    with pytest.raises(TypeError, match="metadata must be a VolumetricMetadata instance"):
        calc.compute_metrics(arr, mask, metadata=None)  # type: ignore[arg-type]

    meta = VolumetricMetadata()
    with pytest.raises(ValueError, match="duration_seconds must be a non-negative finite number"):
        calc.compute_metrics(arr, mask, metadata=meta, duration_seconds=-1.0)
    with pytest.raises(ValueError, match="duration_seconds must be a non-negative finite number"):
        calc.compute_metrics(arr, mask, metadata=meta, duration_seconds=float("nan"))
    with pytest.raises(ValueError, match="duration_seconds must be a non-negative finite number"):
        calc.compute_metrics(arr, mask, metadata=meta, duration_seconds=float("inf"))


def test_extracted_metrics_helpers_standalone() -> None:
    """Verify extracted standalone metrics functions and class staticmethod bindings."""
    from medical_segmentation.pipeline.metrics import (
        build_slice_record,
        compute_intensity_statistics,
        compute_label_summary,
        compute_label_summary_with_counts,
        compute_slice_positions,
        compute_spatial_bounds_and_centroid,
        generate_slice_metrics,
        resolve_slice_plane,
        validate_volumetric_arrays,
    )

    # 1. validate_volumetric_arrays
    valid_raw = np.zeros((10, 10, 10), dtype=np.float32)
    valid_mask = np.zeros((10, 10, 10), dtype=np.uint8)
    validate_volumetric_arrays(valid_raw, valid_mask)  # Should not raise

    with pytest.raises(TypeError, match="Both raw_array and mask_array must be numpy ndarrays"):
        validate_volumetric_arrays([1, 2, 3], valid_mask)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Expected 3D volumetric arrays"):
        validate_volumetric_arrays(np.zeros((10, 10)), valid_mask)
    with pytest.raises(ValueError, match="Array shape mismatch"):
        validate_volumetric_arrays(np.zeros((10, 10, 10)), np.zeros((10, 10, 5)))
    with pytest.raises(ValueError, match="Arrays cannot be empty"):
        validate_volumetric_arrays(np.zeros((0, 10, 10)), np.zeros((0, 10, 10)))

    # 2. compute_intensity_statistics
    raw_cube = np.arange(27, dtype=np.float32).reshape((3, 3, 3))
    label_mask = np.zeros((3, 3, 3), dtype=bool)
    label_mask[1, 1, 1] = True
    mean_val, std_val, min_val, max_val = compute_intensity_statistics(raw_cube, label_mask)
    assert mean_val == 13.0
    assert std_val == 0.0
    assert min_val == 13.0
    assert max_val == 13.0

    empty_mask = np.zeros((3, 3, 3), dtype=bool)
    with pytest.raises(ValueError, match="Cannot compute intensity statistics for empty label mask"):
        compute_intensity_statistics(raw_cube, empty_mask)

    # 3. compute_spatial_bounds_and_centroid
    meta = VolumetricMetadata(
        spacing=(1.0, 2.0, 3.0),
        origin=(0.0, 0.0, 0.0),
        spatial_orientation="RAS",
    )
    mask_box = np.zeros((6, 6, 6), dtype=bool)
    mask_box[1:4, 2:5, 3:6] = True  # Z=[1, 2, 3], Y=[2, 3, 4], X=[3, 4, 5]
    bbox, centroid_mm, z_counts = compute_spatial_bounds_and_centroid(mask_box, meta, voxel_count=27)
    assert bbox == (3, 2, 1, 5, 4, 3)  # (min_x, min_y, min_z, max_x, max_y, max_z)
    assert np.allclose(centroid_mm, meta.continuous_index_to_physical_point((4.0, 3.0, 2.0)))
    assert len(z_counts) == 6
    assert z_counts[1] == 9

    with pytest.raises(TypeError, match="metadata must be a VolumetricMetadata instance"):
        compute_spatial_bounds_and_centroid(mask_box, None, voxel_count=27)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="voxel_count must be a strictly positive integer"):
        compute_spatial_bounds_and_centroid(mask_box, meta, voxel_count=0)
    with pytest.raises(ValueError, match="label_mask contains no active non-zero voxels"):
        compute_spatial_bounds_and_centroid(empty_mask, meta, voxel_count=1)

    # 4. resolve_slice_plane and compute_slice_positions
    assert resolve_slice_plane(meta) == "axial"
    positions = compute_slice_positions(meta, depth=5)
    assert len(positions) == 5
    assert np.isclose(positions[1] - positions[0], 3.0)
    assert compute_slice_positions(meta, depth=0) == []
    assert compute_slice_positions(meta, depth=-2) == []

    # 5. build_slice_record
    record = build_slice_record(
        label_id=2,
        slice_index=4,
        plane="axial",
        slice_position_mm=12.0,
        slice_voxel_count=10,
        pixel_area=2.5,
        slice_intensities=np.array([10.0, 20.0, 30.0], dtype=np.float32),
    )
    assert record["label_id"] == 2
    assert record["slice_index"] == 4
    assert record["plane"] == "axial"
    assert record["slice_position_mm"] == 12.0
    assert record["slice_voxel_count"] == 10
    assert record["cross_sectional_area_mm2"] == 25.0
    assert record["mean_intensity"] == 20.0
    assert record["max_intensity"] == 30.0

    # 6. compute_label_summary and compute_label_summary_with_counts
    summary_empty, counts_empty = compute_label_summary_with_counts(raw_cube, empty_mask, 1, meta)
    assert summary_empty is None
    assert counts_empty is None
    assert compute_label_summary(raw_cube, empty_mask, 1, meta) is None

    summary_active = compute_label_summary(raw_cube, label_mask, 1, meta)
    assert summary_active is not None
    assert summary_active.voxel_count == 1
    assert summary_active.structure_name == "Primary_Lesion"

    summary_multi = compute_label_summary(raw_cube, label_mask, 2, meta)
    assert summary_multi is not None
    assert summary_multi.structure_name == "Segmented_Structure_2"

    # 7. generate_slice_metrics standalone generator
    raw_vol = np.ones((5, 5, 5), dtype=np.float32) * 50.0
    mask_vol = np.zeros((5, 5, 5), dtype=np.uint8)
    mask_vol[2, 1:3, 1:3] = 1
    slice_records = list(
        generate_slice_metrics(
            raw_array=raw_vol,
            mask_array=mask_vol,
            unique_labels=[1],
            metadata=meta,
        )
    )
    assert len(slice_records) == 1
    assert slice_records[0]["slice_index"] == 2
    assert slice_records[0]["slice_voxel_count"] == 4

    # 8. VolumetricMetricsCalculator staticmethod bindings check
    calc = VolumetricMetricsCalculator()
    assert calc.validate_arrays is validate_volumetric_arrays
    assert calc.compute_intensity_statistics is compute_intensity_statistics
    assert calc.compute_spatial_bounds_and_centroid is compute_spatial_bounds_and_centroid
    assert calc.resolve_slice_plane is resolve_slice_plane
    assert calc.compute_slice_positions is compute_slice_positions
    assert calc.build_slice_record is build_slice_record
    assert calc.generate_slice_metrics is generate_slice_metrics
    assert calc.compute_label_summary is compute_label_summary
    assert calc.compute_label_summary_with_counts is compute_label_summary_with_counts




