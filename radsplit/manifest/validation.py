"""Structured integrity checks on a manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

GEOMETRY = ["rows", "columns", "depth", "spacing_x", "spacing_y", "spacing_z", "orientation"]


def _affine_differs(a: Any, b: Any, tolerance: float) -> bool:
    if a is None or b is None or pd.isna(a) or pd.isna(b):
        return False
    return not np.allclose(np.asarray(json.loads(a)), np.asarray(json.loads(b)), atol=tolerance)


def validate_manifest(
    manifest: pd.DataFrame,
    scan_errors: list[dict[str, str]],
    *,
    check_mask_content: bool = True,
    affine_tolerance: float = 1e-3,
) -> dict[str, Any]:
    """Return ``{"valid", "counts", "issues"}`` for a manifest.

    Errors: unreadable, missing or empty files, non positive spacing, and
    image and mask pairs that disagree in matrix size, spacing, orientation
    or affine. Warnings: missing patient identifiers, duplicate checksums,
    duplicate SOP instance identifiers, unpaired masks and empty masks.
    """
    issues: list[dict[str, Any]] = [
        {"severity": "error", "check": "unreadable_file", **error} for error in scan_errors
    ]
    for index, row in manifest.iterrows():
        path = Path(str(row["path"]))
        if not path.exists():
            issues.append({"severity": "error", "check": "missing_file", "row": int(index)})
        elif path.stat().st_size == 0:
            issues.append({"severity": "error", "check": "zero_byte_file", "row": int(index)})
        identifier = row.get("patient_id")
        if identifier is None or pd.isna(identifier) or not str(identifier).strip():
            issues.append({"severity": "warning", "check": "missing_patient_id", "row": int(index)})
        if any(pd.notna(row.get(c)) and float(row[c]) <= 0
               for c in ("spacing_x", "spacing_y", "spacing_z")):
            issues.append({"severity": "error", "check": "invalid_spacing", "row": int(index)})
    for column, check in (("checksum", "duplicate_checksum"),
                          ("sop_instance_uid", "duplicate_dicom_uid")):
        if column not in manifest.columns:
            continue
        repeated = manifest[manifest[column].notna() & manifest.duplicated(column, keep=False)]
        for value, group in repeated.groupby(column):
            issues.append({"severity": "warning", "check": check, "value": value,
                           "rows": [int(x) for x in group.index]})
    masks = manifest[manifest["role"] == "mask"]
    by_path = manifest.set_index("relative_path", drop=False)
    for index, mask in masks.iterrows():
        if pd.isna(mask["mask_for"]):
            issues.append({"severity": "warning", "check": "unpaired_mask", "row": int(index)})
            continue
        if mask["mask_for"] not in by_path.index:
            issues.append({"severity": "error", "check": "missing_paired_image", "row": int(index)})
            continue
        image = by_path.loc[mask["mask_for"]]
        fields = [c for c in GEOMETRY
                  if pd.notna(mask.get(c)) and pd.notna(image.get(c)) and mask[c] != image[c]]
        if _affine_differs(mask.get("affine"), image.get("affine"), affine_tolerance):
            fields.append("affine")
        if fields:
            issues.append({"severity": "error", "check": "image_mask_geometry_mismatch",
                           "row": int(index), "fields": fields})
        if check_mask_content and mask.get("file_type") == "nifti" and Path(mask["path"]).exists():
            import nibabel as nib

            if not np.any(np.asanyarray(nib.load(str(mask["path"])).dataobj)):
                issues.append({"severity": "warning", "check": "empty_mask", "row": int(index)})
    counts = pd.Series([x["check"] for x in issues]).value_counts().to_dict() if issues else {}
    return {
        "valid": not any(x["severity"] == "error" for x in issues),
        "counts": counts,
        "issues": issues,
    }
