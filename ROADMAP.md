# 3dMedicalImageSegmentation

An automated 3D medical image segmentation engine for CBCT, CT, and MRI volumes with Azure Blob Storage ingestion, SimpleITK spatial orientation (`sitk.DICOMOrient`), MONAI sliding-window volumetric inference, and memory-efficient 500-row chunked batched PostgreSQL persistence.

## Architecture & Roadmap

- [x] **Phase 1: Project Foundation & Configuration**
  - Dependency specification (`requirements.txt`, `.gitignore`, `.env.example`)
  - Strongly-typed Pydantic Settings (`config.py`)
- [x] **Phase 2: Azure Blob & DICOM/NIfTI I/O Engine**
  - Azure Blob transfer and streaming client (`io/azure_client.py`)
  - SimpleITK DICOM series reader & `sitk.DICOMOrient` orientation fix (`io/dicom_loader.py`)
- [x] **Phase 3: MONAI Volumetric Transformation Pipeline**
  - Modality-aware intensity normalization and spatial resampling (`pipeline/transforms.py`)
- [x] **Phase 4: 3D U-Net Model Architecture**
  - MONAI 3D residual U-Net and checkpoint loader (`pipeline/model.py`)
- [x] **Phase 5: Sliding-Window Volumetric Inference & Metric Engine**
  - Gaussian patch blending inference and physical metric extraction (`pipeline/inference.py`)
- [x] **Phase 6: PostgreSQL Database & 500-Row Chunked Batch Ingestion**
  - SQLAlchemy 2.0 ORM schemas (`db/models.py`)
  - Connection pooling and memory-bounded batch insert generator (`db/database.py`)
- [x] **Phase 7: CLI Orchestrator & Test Suite**
  - CLI workflow runner (`main.py`)
  - Comprehensive unit test suite (`tests/test_pipeline.py`)
  - Complete documentation & setup instructions (`README.md`)
