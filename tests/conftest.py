"""
Pytest configuration for 3dMedicalImageSegmentation test suite.
Registers the repository root as the 'medical_segmentation' module namespace
so tests can import from medical_segmentation without requiring installation.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if "medical_segmentation" not in sys.modules:
    med_pkg = types.ModuleType("medical_segmentation")
    med_pkg.__path__ = [str(repo_root)]
    sys.modules["medical_segmentation"] = med_pkg
