# 3dMedicalImageSegmentation

A clean, modular, production-grade 3D medical image segmentation engine built with PyTorch, MONAI, SimpleITK, Azure Blob Storage, and PostgreSQL.

Designed for automated, high-throughput volumetric processing of 3D CBCT, CT, and MRI scans with spatial orientation standardization, sliding-window volumetric inference, and memory-efficient chunked database persistence.

---

## Architecture Overview

```text
                                  ┌─────────────────────────────┐
                                  │      Azure Blob Storage     │
                                  │  (DICOM Series / NIfTI 3D)  │
                                  └──────────────┬──────────────┘
                                                 │
                                     [Stream / Multi-Slice I/O]
                                                 ▼
                                  ┌─────────────────────────────┐
                                  │    SimpleITK Ingestion      │
                                  │ (sitk.DICOMOrient -> 'RAS') │
                                  └──────────────┬──────────────┘
                                                 │
                                        [Normalized Tensor]
                                                 ▼
                                  ┌─────────────────────────────┐
                                  │      MONAI 3D U-Net         │
                                  │  Sliding-Window Inference   │
                                  │    (Gaussian Patch Blend)   │
                                  └──────────────┬──────────────┘
                                                 │
                                                 ├──> [Export NIfTI Mask -> Azure Blob Output]
                                                 │
                                                 ▼
                                  ┌─────────────────────────────┐
                                  │ Quantitative Metric Engine  │
                                  │ (Volumes, HU Stats, Slices) │
                                  └──────────────┬──────────────┘
                                                 │
                                     [Streaming Generator Yield]
                                                 ▼
                                  ┌─────────────────────────────┐
                                  │   Chunked Batched Insert    │
                                  │      (500 rows / batch)     │
                                  └──────────────┬──────────────┘
                                                 │
                                                 ▼
                                  ┌─────────────────────────────┐
                                  │ Azure Database (PostgreSQL) │
                                  │ (QueuePool + SSL Required)  │
                                  └─────────────────────────────┘
```

---

## Directory Structure

```text
3dMedicalImageSegmentation/
├── config.py                 # Azure credentials, inference params, DB configs via Pydantic Settings
├── io/
│   ├── __init__.py
│   ├── azure_client.py       # Blob download/upload handlers & stream readers
│   └── dicom_loader.py       # pydicom + SimpleITK DICOM series reader & orientation fix
├── pipeline/
│   ├── __init__.py
│   ├── transforms.py         # MONAI intensity normalization, resampling, tensor conversion
│   ├── model.py              # 3D U-Net instantiation and checkpoint loading
│   └── inference.py          # Sliding window inference execution & post-processing
├── db/
│   ├── __init__.py
│   ├── models.py             # Schema definitions for segmentation results and metrics
│   └── database.py           # Connection pool & 500-row chunked batched insert logic
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py      # Unit tests for transforms, models, metrics, and batching
├── main.py                   # CLI/pipeline orchestration entry point
├── requirements.txt          # Pinned production dependencies
├── .env.example              # Sample environment configuration
├── ROADMAP.md                # Multi-phase roadmap and milestones
└── README.md                 # Complete documentation and architecture guide
```

---

## Core Engineering Features

### 1. Ingestion & Spatial Orientation Standardization
- **Full 3D Ingestion**: Supports multi-slice DICOM series directories (via `sitk.ImageSeriesReader`) and single 3D volumetric files (`.nii`, `.nii.gz`, `.mha`).
- **Clinical Header Parsing**: Uses `pydicom` to extract PatientID, StudyInstanceUID, SeriesInstanceUID, Modality, BodyPart, Windowing, and Rescale Intercept/Slope.
- **Orientation Standardization**: Enforces anatomical consistency across patient cohorts using `sitk.DICOMOrient(image, "RAS")`, guaranteeing consistent coordinate systems regardless of acquisition scanner convention.

### 2. Volumetric Preprocessing & Sliding-Window Inference
- **Modality-Aware Normalization**:
  - **CT / CBCT**: Hounsfield Unit soft-tissue windowing `[-1000, 1000] HU` scaled to `[0.0, 1.0]`.
  - **MRI**: Nonzero voxel z-score standardization (`NormalizeIntensityd(nonzero=True)`).
- **MONAI 3D U-Net**: Deep residual 3D U-Net (`monai.networks.nets.UNet`) with Instance Normalization (`InstanceNorm3d`).
- **Memory-Bounded Sliding Window**: Implements `monai.inferers.sliding_window_inference` with Gaussian importance weighting (`mode="gaussian"`, `overlap=0.5`) to stitch patch predictions without seam artifacts or GPU Out-Of-Memory (OOM) errors.

### 3. Quantitative Physical Metrics Extraction
- **Organ & Lesion Level**:
  - Voxel count $N$ and physical volume $V = N \times s_x \times s_y \times s_z$ in $\text{mm}^3$ and $\text{cm}^3$.
  - Intensity statistics within the segmented volume (Mean, Std, Min, Max).
  - Physical 3D Centroid $(x_c, y_c, z_c)$ in patient world space (mm).
  - 3D Bounding box limits.
- **Slice-by-Slice Voxel Profile**:
  - Computes axial cross-sectional area ($\text{mm}^2$), slice voxel count, and slice intensity distributions.
  - Emits records via a streaming Python generator (`yield`), keeping memory usage strictly constant.

### 4. Database Resilience: 500-Row Chunked Batched Inserts
- **SQLAlchemy 2.0 Connection Pool**: `QueuePool` with `pool_pre_ping=True`, `pool_recycle=1800`, and SSL enforcement (`sslmode=require`) designed for Azure Database for PostgreSQL.
- **500-Row Chunked Insert Generator**:
  - Consumes records from the streaming metric generator.
  - Slices records into batches of exactly 500 rows (`chunked_iterable`).
  - Executes direct multi-row `session.execute(insert(VoxelSliceMetric), chunk)`.
  - Commits each batch individually, eliminating key-generation bottlenecks, identity-map overhead, and transaction log bloat.

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- Azure Blob Storage Account
- Azure Database for PostgreSQL

### 1. Install Dependencies

```bash
cd 3dMedicalImageSegmentation
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy `.env.example` to `.env` and fill in your connection details:

```bash
cp .env.example .env
```

```ini
AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;"
DB_HOST="your-azure-db.postgres.database.azure.com"
DB_NAME="medical_imaging"
DB_USER="postgres"
DB_PASSWORD="your_password"
DB_SSLMODE="require"
INFERENCE_DEVICE="cuda"  # or 'cpu' / 'mps'
```

---

## Usage

### Ingest DICOM Series from Azure Blob Storage

```bash
python -m 3dMedicalImageSegmentation.main \
    --series-prefix "studies/patient_101/series_002/" \
    --container "medical-volumes-input" \
    --modality "CT" \
    --orientation "RAS"
```

### Ingest Single NIfTI Volume from Azure Blob Storage

```bash
python -m 3dMedicalImageSegmentation.main \
    --blob-path "mri_scans/patient_42_brain.nii.gz" \
    --container "medical-volumes-input" \
    --modality "MRI"
```

### Local Testing / Development Mode (Offline)

```bash
python -m 3dMedicalImageSegmentation.main \
    --local-input "/path/to/local/dicom_folder" \
    --dry-run
```

---

## Running Unit Tests

Run the test suite verifying orientation transforms, sliding-window inference, metric extraction, and 500-row batching:

```bash
pytest 3dMedicalImageSegmentation/tests/ -v
```
