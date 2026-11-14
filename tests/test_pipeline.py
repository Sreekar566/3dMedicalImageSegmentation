"""
Comprehensive Unit Tests for 3dMedicalImageSegmentation Pipeline.
Tests spatial orientation standardization, MONAI transforms, 3D U-Net forward execution,
sliding-window inference, metric extraction, and 500-row chunked batch generation.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import SimpleITK as sitk
import torch
from sqlalchemy import create_engine

from medical_segmentation.config import DatabaseSettings, InferenceSettings, PipelineConfig
from medical_segmentation.db.database import DatabaseManager, chunked_iterable
from medical_segmentation.io.dicom_loader import DicomVolumeLoader, VolumetricMetadata
from medical_segmentation.main import run_pipeline
from medical_segmentation.pipeline import PipelineOrchestrator, PipelineResult
from medical_segmentation.pipeline.inference import (
    SlidingWindowInferenceEngine,
    VolumetricMetricsCalculator,
)
from medical_segmentation.pipeline.model import build_3d_unet, load_3d_unet
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

    # 1. Raw state dict
    raw_ckpt_path = tmp_path / "raw_weights.pt"
    torch.save(model.state_dict(), str(raw_ckpt_path))
    loaded_raw = load_3d_unet(checkpoint_path=raw_ckpt_path, in_channels=1, out_channels=2)
    assert isinstance(loaded_raw, torch.nn.Module)

    # 2. Wrapped state dict ({'state_dict': ...})
    wrapped_ckpt_path = tmp_path / "wrapped_weights.pt"
    torch.save({"state_dict": model.state_dict()}, str(wrapped_ckpt_path))
    loaded_wrapped = load_3d_unet(checkpoint_path=wrapped_ckpt_path, in_channels=1, out_channels=2)
    assert isinstance(loaded_wrapped, torch.nn.Module)

    # 3. Missing checkpoint path falls back to freshly initialized model
    missing_ckpt = tmp_path / "non_existent_weights.pt"
    fallback_model = load_3d_unet(checkpoint_path=missing_ckpt, in_channels=1, out_channels=2)
    assert isinstance(fallback_model, torch.nn.Module)



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




