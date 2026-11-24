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
├── .github/
│   └── workflows/
│       └── ci.yml            # Automated CI/CD test and healthcheck workflow
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
├── diagnostics.py            # Isolated diagnostic probes for Python, Torch, SimpleITK, MONAI, and DB
├── logging_config.py         # Production structured JSON / text logging & contextual formatter
├── tests/
│   ├── __init__.py
│   ├── test_db.py            # Unit & integration tests for database models and pool
│   ├── test_healthcheck.py   # System health probe and connectivity tests
│   ├── test_io.py            # Ingestion and Azure client tests
│   ├── test_logging.py       # Structured JSON formatting, file logging, and CLI log arguments
│   └── test_pipeline.py      # Unit tests for transforms, models, metrics, and batching
├── Dockerfile                # Production-hardened container definition with healthcheck
├── docker-compose.yml        # Orchestrated PostgreSQL 16 & pipeline services
├── Makefile                  # Developer tooling commands (test, healthcheck, compose)
├── main.py                   # CLI/pipeline orchestration and healthcheck entry point
├── requirements.txt          # Pinned production dependencies
├── .dockerignore             # Docker build context exclusions
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

### 5. Structured JSON & Cloud Logging Telemetry
- **Cloud-Ready JSON Logs**: Produces single-line JSON log events for cloud ingestion (Azure Monitor, AWS CloudWatch, Datadog, Elastic) with ISO-8601 UTC timestamps, log level, caller module/function/line, process/thread IDs, and contextual metadata.
- **Traceback Ingestion**: Formats exceptions into a structured `exception` object (`type`, `message`, `traceback`), preventing multi-line stack trace corruption in container log aggregators.
- **Configurable Outputs**: Supports console stdout (`--log-format text` or `--log-format json`) and persistent file logging (`--log-file /path/to/log.jsonl` or `LOG_FILE` environment variable).

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

### System Diagnostics & Health Check

Run automated health diagnostics verifying Python environment, PyTorch execution device, SimpleITK C++ bindings, MONAI network initialization, and workspace volume permissions:

```bash
python -m main --check-health
```

To additionally verify PostgreSQL database connectivity and responsiveness:

```bash
python -m main --check-health --check-db
```

To verify Azure Blob Storage connectivity, credentials, and container accessibility:

```bash
python -m main --check-health --check-azure
```

To execute a comprehensive diagnostic health check verifying all runtime components, PostgreSQL, and Azure Blob Storage:

```bash
python -m main --check-health --check-all
```

### Structured JSON Logging & File Persistence

Run health checks or pipeline execution with structured JSON logs formatted for automated ingestion by cloud aggregators (Azure Monitor, CloudWatch, Datadog):

```bash
python -m main --check-health --log-format json
```

Direct log output to a persistent local log file:

```bash
python -m main --local-input "/path/to/scan.nii.gz" --dry-run --log-file "pipeline.log" --log-format json
```

---

## Docker & Container Deployment

A production-hardened `Dockerfile` and `docker-compose.yml` are provided for containerized execution and local database orchestration.

### 1. Launch Stack with Docker Compose

Spin up PostgreSQL 16 Alpine and the pipeline container with integrated service healthchecks:

```bash
docker compose up -d
```

Compose ensures the segmentation pipeline container waits for PostgreSQL to pass its `pg_isready` healthcheck before initializing.

### 2. Check Service Health & Logs

```bash
docker compose ps
docker compose logs -f
```

### 3. Build Container Image Manually

```bash
docker build -t medseg-pipeline:latest .
```

---

## Developer Tooling

A `Makefile` is provided for rapid local development workflows:

```bash
make help          # List all available targets
make install       # Install pinned requirements
make test          # Execute pytest test suite
make lint          # Run code verification and syntax compilation checks
make coverage      # Run pytest test suite with code coverage reporting
make healthcheck         # Run system diagnostics probe
make healthcheck-db      # Run health check including PostgreSQL database
make healthcheck-azure   # Run health check including Azure Blob Storage
make healthcheck-all     # Run comprehensive health check with DB and Azure
make docker-up           # Launch PostgreSQL and pipeline via compose
make docker-down   # Tear down compose services and networks
make clean         # Remove python bytecode and test caches
```

---

## Running Unit Tests

Run the test suite verifying orientation transforms, sliding-window inference, metric extraction, and 500-row batching:

```bash
pytest tests/ -v
```
