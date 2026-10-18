"""MetaImage (.mha, .mhd) and NRRD (.nrrd, .nhdr) header reader.

Requires the optional SimpleITK dependency. Voxel data are not loaded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def read_itk(path: Path) -> dict[str, Any]:
    try:
        import SimpleITK as sitk
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "Reading MetaImage and NRRD files needs SimpleITK: pip install 'radsplit[manifest]'"
        ) from error
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    size = reader.GetSize()
    spacing = reader.GetSpacing()
    dimension = reader.GetDimension()
    direction = np.asarray(reader.GetDirection()).reshape(dimension, dimension)
    affine = np.eye(4)
    d = min(dimension, 3)
    affine[:d, :d] = direction[:d, :d] * np.asarray(spacing[:d])
    affine[:d, 3] = reader.GetOrigin()[:d]
    orientation = None
    if dimension == 3:
        orientation = sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(
            reader.GetDirection())
    return {
        "file_type": "nrrd" if path.suffix.lower() in {".nrrd", ".nhdr"} else "metaimage",
        "rows": int(size[0]) if dimension > 0 else None,
        "columns": int(size[1]) if dimension > 1 else None,
        "depth": int(size[2]) if dimension > 2 else 1,
        "spacing_x": float(spacing[0]) if dimension > 0 else None,
        "spacing_y": float(spacing[1]) if dimension > 1 else None,
        "spacing_z": float(spacing[2]) if dimension > 2 else None,
        "orientation": orientation,
        "affine": json.dumps(np.round(affine, 6).tolist()),
    }
