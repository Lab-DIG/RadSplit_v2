"""DICOM header reader (pixel data are not loaded)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pydicom


def _text(dataset: pydicom.Dataset, name: str) -> Any:
    value = getattr(dataset, name, None)
    return str(value) if value is not None else None


def _number(dataset: pydicom.Dataset, name: str) -> float | None:
    value = getattr(dataset, name, None)
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _age(dataset: pydicom.Dataset) -> float | None:
    """PatientAge (for example ``063Y``) in years."""
    value = str(getattr(dataset, "PatientAge", "") or "")
    if len(value) == 4 and value[:3].isdigit():
        return float(value[:3]) / {"Y": 1, "M": 12, "W": 52.18, "D": 365.25}.get(value[3], 1)
    return None


def read_dicom(path: Path) -> dict[str, Any]:
    ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    spacing = getattr(ds, "PixelSpacing", None) or getattr(ds, "ImagerPixelSpacing", None) or []
    orientation = getattr(ds, "ImageOrientationPatient", None)
    thickness = getattr(ds, "SliceThickness", None)
    return {
        "patient_id": _text(ds, "PatientID"),
        "study_id": _text(ds, "StudyInstanceUID"),
        "series_id": _text(ds, "SeriesInstanceUID"),
        "instance_id": _text(ds, "SOPInstanceUID"),
        "file_type": "dicom",
        "modality": _text(ds, "Modality"),
        "manufacturer": _text(ds, "Manufacturer"),
        "scanner_model": _text(ds, "ManufacturerModelName"),
        "institution": _text(ds, "InstitutionName"),
        "station_name": _text(ds, "StationName"),
        "software_version": _text(ds, "SoftwareVersions"),
        "convolution_kernel": _text(ds, "ConvolutionKernel"),
        "slice_thickness": _number(ds, "SliceThickness"),
        "kvp": _number(ds, "KVP"),
        "field_strength": _number(ds, "MagneticFieldStrength"),
        "contrast_agent": _text(ds, "ContrastBolusAgent"),
        "body_part": _text(ds, "BodyPartExamined"),
        "series_description": _text(ds, "SeriesDescription"),
        "patient_sex": _text(ds, "PatientSex"),
        "patient_age": _age(ds),
        "study_date": _text(ds, "StudyDate"),
        "series_date": _text(ds, "SeriesDate"),
        "rows": getattr(ds, "Rows", None),
        "columns": getattr(ds, "Columns", None),
        "depth": 1,
        "spacing_x": float(spacing[0]) if len(spacing) > 0 else None,
        "spacing_y": float(spacing[1]) if len(spacing) > 1 else None,
        "spacing_z": float(thickness) if thickness not in (None, "") else None,
        "orientation": "\\".join(map(str, orientation)) if orientation else None,
        "frame_of_reference_uid": _text(ds, "FrameOfReferenceUID"),
        "study_instance_uid": _text(ds, "StudyInstanceUID"),
        "series_instance_uid": _text(ds, "SeriesInstanceUID"),
        "sop_instance_uid": _text(ds, "SOPInstanceUID"),
    }
