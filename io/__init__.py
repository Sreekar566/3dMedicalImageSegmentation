"""
I/O Package for 3dMedicalImageSegmentation.
Contains Azure Blob Storage client, DICOM/NIfTI volume loaders, and volumetric mask writers.
"""

from .azure_client import AzureBlobClient
from .dicom_loader import (
    DicomVolumeLoader,
    VolumetricMetadata,
    extract_pydicom_metadata,
    safe_parse_float,
    safe_parse_str,
    sitk_to_numpy,
)
from .mask_writer import VolumetricMaskWriter, save_mask_as_nifti
from .volume_ingester import VolumeIngester, validate_input_sources

__all__ = [
    "AzureBlobClient",
    "DicomVolumeLoader",
    "VolumeIngester",
    "VolumetricMaskWriter",
    "VolumetricMetadata",
    "extract_pydicom_metadata",
    "safe_parse_float",
    "safe_parse_str",
    "save_mask_as_nifti",
    "sitk_to_numpy",
    "validate_input_sources",
]
