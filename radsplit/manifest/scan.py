"""Recursive scan of a directory into a canonical manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from .dataset import Manifest
from .schema import MANIFEST_COLUMNS, SCHEMA_VERSION

MASK_TOKENS = ("mask", "seg", "labelmap", "labelstr", "labelsts")
MASK_SUFFIXES = ("_mask", "_seg", "_label", "-mask", "-seg", "-label")
ITK_SUFFIXES = (".mha", ".mhd", ".nrrd", ".nhdr")
DECATHLON_FOLDERS = {"imagestr", "imagests", "labelstr", "labelsts"}


def _has_dicom_signature(path: Path) -> bool:
    """True for files without a known suffix that carry the DICOM preamble."""
    if path.suffix and not path.suffix[1:].isdigit():
        return False
    try:
        with path.open("rb") as stream:
            stream.seek(128)
            return stream.read(4) == b"DICM"
    except OSError:
        return False


def _checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _role(relative: Path) -> str:
    text = relative.as_posix().lower()
    return "mask" if any(token in text for token in MASK_TOKENS) else "image"


def _case_from_path(relative: Path) -> str:
    name = relative.name
    stem = name[:-7] if name.lower().endswith(".nii.gz") else Path(name).stem
    if name.lower().endswith(".raw") or name.lower().endswith(".zraw"):
        stem = Path(name).stem
    for suffix in MASK_SUFFIXES:
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if relative.parts and relative.parts[0].lower() in DECATHLON_FOLDERS:
        return stem
    return relative.parts[0] if len(relative.parts) > 1 else stem


def _pair_masks(manifest: pd.DataFrame) -> None:
    """Pair a mask with an image only when the pairing is unambiguous."""
    images = manifest[manifest["role"] == "image"].groupby("patient_id")["relative_path"].agg(list)
    for index, mask in manifest[manifest["role"] == "mask"].iterrows():
        candidates = images.get(mask["patient_id"], [])
        if len(candidates) == 1:
            manifest.at[index, "mask_for"] = candidates[0]


def scan(
    root: str | Path,
    *,
    recursive: bool = True,
    detect_masks: bool = True,
) -> Manifest:
    """Scan imaging files below ``root`` into a manifest.

    Supported: DICOM (``.dcm``, ``.dicom``, ``.ima`` and files without a suffix
    that carry the DICOM signature), NIfTI (``.nii``, ``.nii.gz``), MetaImage
    (``.mha``, ``.mhd``) and NRRD (``.nrrd``, ``.nhdr``).


    Operating system packaging artifacts (``.DS_Store`` and AppleDouble ``._*``
    files) are skipped and listed in the data set metadata. Files that cannot
    be read are recorded as scan errors and reported by ``validate``.
    """
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise NotADirectoryError(base)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    ignored: list[str] = []
    paths = base.rglob("*") if recursive else base.glob("*")
    for path in sorted(p for p in paths if p.is_file()):
        relative = path.relative_to(base)
        if any(part == ".DS_Store" or part.startswith("._") for part in relative.parts):
            ignored.append(relative.as_posix())
            continue
        lower = path.name.lower()
        try:
            if lower.endswith((".nii", ".nii.gz")):
                from .io.nifti import read_nifti

                metadata = read_nifti(path)
                metadata.setdefault("patient_id", _case_from_path(relative))
            elif lower.endswith(ITK_SUFFIXES):
                from .io.itk import read_itk

                metadata = read_itk(path)
                metadata.setdefault("patient_id", _case_from_path(relative))
            elif lower.endswith((".dcm", ".dicom", ".ima")) or _has_dicom_signature(path):
                from .io.dicom import read_dicom

                metadata = read_dicom(path)
            else:
                continue
        except Exception as exc:  # recorded and reported, never silently dropped
            errors.append({"path": relative.as_posix(), "error": f"{type(exc).__name__}: {exc}"})
            continue
        row: dict[str, Any] = {column: None for column in MANIFEST_COLUMNS}
        row.update(metadata)
        row.update(
            {
                "path": str(path),
                "relative_path": relative.as_posix(),
                "role": _role(relative) if detect_masks else "image",
                "checksum": _checksum(path),
                "source": str(base),
            }
        )
        rows.append(row)
    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    _pair_masks(manifest)
    dataset = Manifest(
        manifest,
        metadata={
            "source_root": str(base),
            "scan_errors": errors,
            "ignored_packaging_artifacts": ignored,
            "schema_version": SCHEMA_VERSION,
        },
    )
    dataset.record("scan", {"recursive": recursive, "files": len(rows), "ignored": len(ignored)})
    return dataset
