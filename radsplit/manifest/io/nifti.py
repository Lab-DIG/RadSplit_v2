"""NIfTI header reader (voxel data are not loaded)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


def read_nifti(path: Path) -> dict[str, Any]:
    image = nib.load(str(path))
    shape = image.shape
    zooms = image.header.get_zooms()
    return {
        "file_type": "nifti",
        "rows": int(shape[0]) if len(shape) > 0 else None,
        "columns": int(shape[1]) if len(shape) > 1 else None,
        "depth": int(shape[2]) if len(shape) > 2 else 1,
        "spacing_x": float(zooms[0]) if len(zooms) > 0 else None,
        "spacing_y": float(zooms[1]) if len(zooms) > 1 else None,
        "spacing_z": float(zooms[2]) if len(zooms) > 2 else None,
        "orientation": "".join(nib.aff2axcodes(image.affine)),
        "affine": json.dumps(np.round(image.affine, 6).tolist()),
    }
