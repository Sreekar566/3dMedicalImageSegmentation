"""
3dMedicalImageSegmentation: Automated 3D medical image segmentation engine.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

__version__ = "0.1.0"

# Register 'medical_segmentation' module namespace alias in sys.modules
if "medical_segmentation" not in sys.modules:
    med_pkg = types.ModuleType("medical_segmentation")
    med_pkg.__path__ = [str(Path(__file__).resolve().parent)]
    sys.modules["medical_segmentation"] = med_pkg
