"""Manifest integrity checks on public NIfTI and DICOM files (supplement).

Usage: python integrity.py NAME=DIRECTORY [NAME=DIRECTORY ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from radsplit import manifest
from radsplit.manifest import Manifest

RESULTS = Path(__file__).resolve().parents[1] / "results"


def check(name: str, source: Path) -> dict[str, object]:
    started = time.perf_counter()
    dataset = manifest.scan(source)
    seconds = time.perf_counter() - started
    report = dataset.validate()
    fingerprint = dataset.fingerprint()
    saved = RESULTS / "manifests" / f"{name}.manifest"
    dataset.save(saved)
    return {
        "name": name,
        "summary": dataset.summary(),
        "valid": report["valid"],
        "issue_counts": report["counts"],
        "paired_masks": int(dataset.table["mask_for"].notna().sum()),
        "ignored_packaging_artifacts": len(dataset.metadata["ignored_packaging_artifacts"]),
        "scan_seconds": seconds,
        "fingerprint": fingerprint,
        "same_after_reload": Manifest.load(saved).fingerprint() == fingerprint,
        "same_after_rescan": manifest.scan(source).fingerprint() == fingerprint,
    }


def main() -> None:
    out = [check(*argument.split("=", 1)[:1], Path(argument.split("=", 1)[1]))
           for argument in sys.argv[1:]]
    (RESULTS / "integrity.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
