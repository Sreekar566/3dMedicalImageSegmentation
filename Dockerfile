# Production-grade container image for 3dMedicalImageSegmentation
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    WORK_DIR=/tmp/medical_segmentation_workspace \
    PYTHONPATH=/app

# Install minimal OS runtime dependencies required for C/C++ extensions, SimpleITK, and PostgreSQL
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root application user for clinical/HIPAA compliance
RUN groupadd --gid 10001 medseg && \
    useradd --uid 10001 --gid medseg --shell /bin/bash --create-home appuser

# Set up application working directory and staging workspace
WORKDIR /app
RUN mkdir -p /tmp/medical_segmentation_workspace && \
    chown -R appuser:medseg /tmp/medical_segmentation_workspace /app

# Install Python dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY --chown=appuser:medseg __init__.py config.py diagnostics.py logging_config.py main.py ./
COPY --chown=appuser:medseg io/ ./io/
COPY --chown=appuser:medseg pipeline/ ./pipeline/
COPY --chown=appuser:medseg db/ ./db/

# Switch to non-root user
USER appuser

# Container healthcheck using the built-in diagnostic probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -m main --check-health || exit 1

# Default entrypoint
ENTRYPOINT ["python", "-m", "main"]
CMD ["--help"]
