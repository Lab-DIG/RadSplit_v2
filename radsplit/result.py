"""Container for a generated partition."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .audit import AuditReport


@dataclass(frozen=True)
class SplitResult:
    """Outcome of :func:`radsplit.split`.

    Attributes
    ----------
    data
        The input records with an added partition column.
    assignments
        One row per patient with the assigned partition.
    audit
        Balance and overlap report for the generated partition.
    config
        Every argument needed to regenerate the partition.
    split_hash
        SHA-256 digest of the patient assignments and configuration.
    """

    data: pd.DataFrame
    assignments: pd.DataFrame
    audit: AuditReport
    config: dict[str, Any]
    split_hash: str
    column: str = "split"

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.data.loc[self.data[self.column] == name]

    def __iter__(self):
        for name in self.audit.partitions:
            yield self[name]

    @property
    def parts(self) -> dict[str, pd.DataFrame]:
        return {name: self[name] for name in self.audit.partitions}

    def report(self) -> str:
        s = self.audit.summary
        lines = ["RadSplit report", "===============", ""]
        for _, row in self.audit.sizes.iterrows():
            lines.append(
                f"{row['partition']:<12}{row['patients']:>8} patients ({row['patient_fraction']:.1%})"
                f"{row['records']:>9} records ({row['record_fraction']:.1%})"
            )
        lines += [
            "",
            f"Patients in more than one partition  {s['patients_in_multiple_partitions']}",
            f"Mean absolute SMD                    {s['mean_abs_smd']:.4f}",
            f"Maximum absolute SMD                 {s['max_abs_smd']:.4f}",
            f"Terms with absolute SMD above {self.audit.threshold:<4}   "
            f"{s['terms_above_threshold']} of {s['terms']}",
            "",
            f"Split hash  {self.split_hash}",
        ]
        return "\n".join(lines)

    def save(self, directory: str | Path) -> Path:
        """Write assignments, configuration, and audit report to ``directory``."""
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        self.assignments.to_csv(output / "assignments.csv", index=False)
        self.audit.balance.to_csv(output / "balance.csv", index=False)
        payload = {
            "config": self.config,
            "summary": self.audit.summary,
            "split_hash": self.split_hash,
        }
        (output / "split.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        (output / "audit.md").write_text(self.audit.to_markdown(), encoding="utf-8")
        (output / "audit.html").write_text(self.audit.to_html(), encoding="utf-8")
        return output


def deterministic_hash(assignments: pd.DataFrame, config: dict[str, Any]) -> str:
    ordered = assignments.astype(str).sort_values(assignments.columns.tolist())
    blob = json.dumps(
        {"assignments": ordered.to_dict(orient="records"), "config": config},
        sort_keys=True,
        default=str,
    )
    return "rs:sha256:" + hashlib.sha256(blob.encode()).hexdigest()
