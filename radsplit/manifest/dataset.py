"""In memory data set: manifest, metadata and provenance."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .validation import validate_manifest

VERSION = "1.0.0"


class Manifest:
    def __init__(
        self,
        manifest: pd.DataFrame,
        metadata: dict[str, Any] | None = None,
        provenance: list[dict[str, Any]] | None = None,
    ) -> None:
        self.table = manifest.copy()
        self.metadata = deepcopy(metadata or {})
        self.provenance = deepcopy(provenance or [])

    @property
    def images(self) -> pd.DataFrame:
        return self.table[self.table["role"] == "image"]

    @property
    def masks(self) -> pd.DataFrame:
        return self.table[self.table["role"] == "mask"]

    def record(self, operation: str, parameters: dict[str, Any]) -> None:
        self.provenance.append(
            {
                "operation": operation,
                "parameters": parameters,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "software": {"radsplit": VERSION},
            }
        )

    def summary(self) -> dict[str, Any]:
        def unique(column: str) -> int:
            return int(self.table[column].nunique(dropna=True))

        return {
            "files": int(len(self.table)),
            "patients": unique("patient_id"),
            "studies": unique("study_id"),
            "series": unique("series_id"),
            "images": int(len(self.images)),
            "masks": int(len(self.masks)),
            "modalities": self.table["modality"].value_counts(dropna=True).to_dict(),
            "dataset_id": self.fingerprint(),
        }

    def series(self) -> pd.DataFrame:
        """One row per image series (or per file for formats without series).

        DICOM instances that share a series identifier are combined; ``depth``
        becomes the number of instances and ``files`` counts them. Use this
        table for partitioning so that one record is one series, not one slice.
        """
        frame = self.table.copy()
        key = frame["series_instance_uid"].where(
            frame["series_instance_uid"].notna(), frame["relative_path"])
        frame["series_key"] = key
        first = frame.groupby("series_key", sort=False).first()
        first["files"] = frame.groupby("series_key", sort=False).size()
        is_dicom = first["file_type"] == "dicom"
        first.loc[is_dicom, "depth"] = first.loc[is_dicom, "files"]
        return first.reset_index(drop=True).drop(columns=["instance_id", "sop_instance_uid"])

    def join(self, table: pd.DataFrame, on: str = "patient_id") -> Manifest:
        """Add columns (labels, clinical variables) from ``table`` by a shared key."""
        if table[on].duplicated().any():
            raise ValueError(f"Values of {on!r} must be unique in the joined table")
        left = self.table
        shared = [c for c in table.columns if c != on and c in left.columns]
        conflicts = [c for c in shared if left[c].notna().any()]
        if conflicts:
            raise ValueError(
                f"Columns already in the manifest: {conflicts}; rename them before joining")
        left = left.drop(columns=shared)  # empty placeholder columns are filled by the join
        right = table.copy()
        right[on] = right[on].astype(str)
        merged = left.assign(**{on: left[on].astype(str)}).merge(right, on=on, how="left")
        result = Manifest(merged, self.metadata, self.provenance)
        matched = int(merged[on].isin(right[on]).sum())
        result.record("join", {"on": on, "columns": [c for c in table.columns if c != on],
                               "matched_rows": matched, "rows": len(merged)})
        return result

    def query(self, expression: str) -> Manifest:
        result = Manifest(self.table.query(expression), self.metadata, self.provenance)
        result.record("query", {"expression": expression})
        return result

    def validate(self, **options: Any) -> dict[str, Any]:
        return validate_manifest(self.table, self.metadata.get("scan_errors", []), **options)

    def fingerprint(self) -> str:
        """SHA-256 digest of the manifest content, independent of absolute paths."""
        excluded = {"path", "source", "parent_dataset_id", "split"}
        columns = sorted(c for c in self.table.columns if c not in excluded)
        records = (
            self.table[columns].astype(object).where(self.table[columns].notna(), "<null>")
            .sort_values("relative_path").to_dict(orient="records")
        )
        blob = json.dumps({"schema": columns, "records": records}, sort_keys=True, default=str)
        return "mf:sha256:" + hashlib.sha256(blob.encode()).hexdigest()

    def split(self, *, patient: str = "patient_id", **options: Any):
        """Partition the manifest with :func:`radsplit.split` and record the result."""
        from radsplit import split

        result = split(self.table, patient=patient, **options)
        self.table["split"] = result.data["split"].to_numpy()
        self.record("split", {"split_hash": result.split_hash, **result.config})
        return result

    def save(self, destination: str | Path) -> Path:
        output = Path(destination)
        output.mkdir(parents=True, exist_ok=True)
        self.table.to_parquet(output / "manifest.parquet", index=False)
        (output / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, default=str), encoding="utf-8")
        (output / "provenance.json").write_text(
            json.dumps(self.provenance, indent=2, default=str), encoding="utf-8")
        return output

    @classmethod
    def load(cls, source: str | Path) -> Manifest:
        root = Path(source)
        return cls(
            pd.read_parquet(root / "manifest.parquet"),
            json.loads((root / "metadata.json").read_text(encoding="utf-8")),
            json.loads((root / "provenance.json").read_text(encoding="utf-8")),
        )
