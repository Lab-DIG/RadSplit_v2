"""Audit of an existing partition: patient overlap, sizes and covariate balance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .design import Design, build_design

THRESHOLD = 0.1


@dataclass
class AuditReport:
    """Result of :func:`audit`.

    Attributes
    ----------
    sizes
        Patients and records per partition.
    overlap
        Patient overlap between each pair of partitions and the share of
        records in the later partition whose patient also appears in the
        earlier one.
    balance
        One row per balance term with pooled and per partition means,
        standardized differences against the pooled cohort, and the largest
        absolute pairwise standardized mean difference (SMD).
    summary
        Scalar summaries used throughout the documentation and manuscript.
    """

    partitions: list[str]
    sizes: pd.DataFrame
    overlap: pd.DataFrame
    balance: pd.DataFrame
    summary: dict[str, Any]
    keys: pd.DataFrame | None = None
    notes: list[str] = field(default_factory=list)
    threshold: float = THRESHOLD

    @property
    def disjoint(self) -> bool:
        return self.summary["patients_in_multiple_partitions"] == 0

    @property
    def balanced(self) -> bool:
        return self.summary["max_abs_smd"] <= self.threshold

    def table(self, digits: int = 1) -> pd.DataFrame:
        """Characteristics by partition, formatted for a manuscript table."""
        rows = []
        for _, row in self.balance.iterrows():
            entry: dict[str, Any] = {"Characteristic": row["term"]}
            for name in ["pooled", *self.partitions]:
                value = row[f"mean_{name}"]
                if row["kind"] == "mean":
                    entry[name] = f"{value:.{digits}f}"
                else:
                    entry[name] = f"{100 * value:.{digits}f}%"
            entry["max |SMD|"] = f"{row['max_abs_smd']:.3f}"
            rows.append(entry)
        return pd.DataFrame(rows)

    def to_markdown(self) -> str:
        s = self.summary
        lines = [
            "# Partition audit",
            "",
            f"- Patients: {s['patients']}; records: {s['records']}",
            f"- Patients in more than one partition: {s['patients_in_multiple_partitions']}"
            f" ({100 * s['patients_in_multiple_partitions_fraction']:.1f}%)",
            f"- Mean absolute SMD: {s['mean_abs_smd']:.4f}",
            f"- Maximum absolute SMD: {s['max_abs_smd']:.4f}",
            f"- Terms with absolute SMD above {self.threshold}: "
            f"{s['terms_above_threshold']} of {s['terms']}",
            "",
            "## Sizes",
            "",
            self.sizes.to_markdown(index=False, floatfmt=".3f"),
            "",
            "## Patient overlap",
            "",
            self.overlap.to_markdown(index=False, floatfmt=".3f"),
            "",
            "## Characteristics by partition",
            "",
            self.table().to_markdown(index=False),
        ]
        if self.keys is not None and len(self.keys):
            lines += ["", "## Other keys shared across partitions", "",
                      self.keys.to_markdown(index=False)]
        if self.notes:
            lines += ["", "## Notes", ""] + [f"- {note}" for note in self.notes]
        return "\n".join(lines) + "\n"

    def to_html(self) -> str:
        s = self.summary
        verdict = lambda ok: "pass" if ok else "review"  # noqa: E731
        style = (
            "<style>body{font-family:Arial,Helvetica,sans-serif;margin:2rem;color:#1b1f24}"
            "table{border-collapse:collapse;margin:1rem 0}th,td{border-bottom:1px solid #d0d7de;"
            "padding:4px 10px;text-align:right}th:first-child,td:first-child{text-align:left}"
            ".pass{color:#1a7f37;font-weight:bold}.review{color:#b35900;font-weight:bold}</style>"
        )
        parts = [
            "<!doctype html><html><head><meta charset='utf-8'><title>Partition audit</title>",
            style,
            "</head><body><h1>Partition audit</h1>",
            f"<p>Patient disjoint: <span class='{verdict(self.disjoint)}'>"
            f"{verdict(self.disjoint)}</span> ({s['patients_in_multiple_partitions']} patients in "
            "more than one partition).<br>",
            f"Covariate balance: <span class='{verdict(self.balanced)}'>{verdict(self.balanced)}"
            f"</span> (maximum absolute SMD {s['max_abs_smd']:.3f}, mean {s['mean_abs_smd']:.3f}; "
            f"{s['terms_above_threshold']} of {s['terms']} terms above {self.threshold}).</p>",
            "<h2>Sizes</h2>",
            self.sizes.to_html(index=False, float_format=lambda x: f"{x:.3f}"),
            "<h2>Patient overlap</h2>",
            self.overlap.to_html(index=False, float_format=lambda x: f"{x:.3f}"),
            "<h2>Characteristics by partition</h2>",
            self.table().to_html(index=False),
            "</body></html>",
        ]
        return "".join(parts)

    def loveplot(self, ax=None, *, max_terms: int = 40):
        """Dot plot of the largest absolute pairwise SMD for every term."""
        import matplotlib.pyplot as plt

        frame = self.balance.sort_values("max_abs_smd").tail(max_terms)
        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 0.22 * len(frame) + 1.2))
        ax.scatter(frame["max_abs_smd"], frame["term"], s=22, color="#1f5a94", zorder=3)
        ax.axvline(self.threshold, color="#8a8a8a", linestyle="--", linewidth=0.9)
        ax.set_xlabel("Largest absolute standardized mean difference")
        ax.set_xlim(left=0)
        ax.grid(axis="x", alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        return ax


def _summaries(
    design: Design, means: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Standardized differences against the pooled cohort and between pairs."""
    k = means.shape[0]
    versus_pooled = (means - design.pooled_mean) / design.pooled_sd
    pairs = list(combinations(range(k), 2))
    pairwise = np.vstack([(means[a] - means[b]) / design.pooled_sd for a, b in pairs])
    primary = np.array([not term.auxiliary for term in design.terms])
    absolute = np.abs(pairwise[:, primary])
    worst = np.nanmax(np.abs(pairwise), axis=0)
    summary = {
        "mean_abs_smd": float(np.nanmean(absolute)),
        "max_abs_smd": float(np.nanmax(absolute)),
        "terms": int(primary.sum()),
        "terms_above_threshold": int((worst[primary] > threshold).sum()),
    }
    return versus_pooled, worst, summary


def evaluate(
    design: Design, record_split: np.ndarray, k: int, threshold: float = THRESHOLD
) -> dict[str, float]:
    """Fast scalar balance summary for benchmarking."""
    means = design.partition_means(record_split, k)
    return _summaries(design, means, threshold)[2]


def overlap_table(
    patient_codes: np.ndarray, record_split: np.ndarray, partitions: Sequence[str]
) -> tuple[pd.DataFrame, int]:
    k = len(partitions)
    n = int(patient_codes.max()) + 1
    present = np.zeros((n, k), dtype=bool)
    present[patient_codes, record_split] = True
    rows = []
    for a, b in combinations(range(k), 2):
        shared = present[:, a] & present[:, b]
        in_b = record_split == b
        leaked = shared[patient_codes] & in_b
        rows.append(
            {
                "partition_a": partitions[a],
                "partition_b": partitions[b],
                "shared_patients": int(shared.sum()),
                "patients_b": int(present[:, b].sum()),
                "shared_fraction_of_b_patients": float(shared.sum() / max(present[:, b].sum(), 1)),
                "records_b_with_patient_in_a": int(leaked.sum()),
                "records_b": int(in_b.sum()),
                "record_fraction_of_b": float(leaked.sum() / max(in_b.sum(), 1)),
            }
        )
    multiple = int((present.sum(axis=1) > 1).sum())
    return pd.DataFrame(rows), multiple


def audit(
    data: pd.DataFrame,
    *,
    patient: str,
    split: str,
    variables: Sequence[str],
    partitions: Sequence[str] | None = None,
    also_disjoint: Sequence[str] | None = None,
    continuous: Sequence[str] | None = None,
    categorical: Sequence[str] | None = None,
    weights: Mapping[str, float] | None = None,
    rare: float = 0.01,
    threshold: float = THRESHOLD,
) -> AuditReport:
    """Audit an existing partition stored in column ``split``.

    Parameters
    ----------
    data
        One row per record (image, lesion, examination or patient).
    patient
        Column with the patient identifier.
    split
        Column with the partition name of every record.
    variables
        Columns to describe and compare across partitions.
    partitions
        Order of partition names. Defaults to training, tuning and test names
        when they are recognized, otherwise sorted order.
    also_disjoint
        Other identifier columns (for example study, lesion or site) that are
        checked for values shared across partitions.
    """
    frame = data.loc[data[split].notna()].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No records with a partition label")
    observed = frame[split].astype(str)
    if partitions is None:
        preferred = ["train", "training", "tune", "tuning", "val", "valid", "validation", "test"]
        names = sorted(observed.unique(), key=lambda x: (
            preferred.index(x.lower()) if x.lower() in preferred else len(preferred), x))
    else:
        names = [str(x) for x in partitions]
    lookup = {name: index for index, name in enumerate(names)}
    unknown = sorted(set(observed.unique()) - set(lookup))
    if unknown:
        raise ValueError(f"Partition labels not listed in partitions: {unknown}")
    record_split = observed.map(lookup).to_numpy()
    k = len(names)
    design = build_design(
        frame, patient=patient, variables=variables, continuous=continuous,
        categorical=categorical, weights=weights, rare=rare,
    )
    means = design.partition_means(record_split, k)
    versus_pooled, worst, summary = _summaries(design, means, threshold)

    balance = pd.DataFrame(
        {
            "term": [term.label for term in design.terms],
            "variable": [term.variable for term in design.terms],
            "level": [term.level for term in design.terms],
            "kind": [term.kind for term in design.terms],
            "unit": [term.unit for term in design.terms],
            "mean_pooled": design.pooled_mean,
            "sd_pooled": design.pooled_sd,
        }
    )
    for index, name in enumerate(names):
        balance[f"mean_{name}"] = means[index]
    for index, name in enumerate(names):
        balance[f"smd_{name}"] = versus_pooled[index]
    balance["max_abs_smd"] = worst

    overlap, multiple = overlap_table(design.record_patient, record_split, names)
    present = np.zeros((design.n_patients, k), dtype=bool)
    present[design.record_patient, record_split] = True
    records = np.bincount(record_split, minlength=k)
    sizes = pd.DataFrame(
        {
            "partition": names,
            "patients": present.sum(axis=0),
            "patient_fraction": present.sum(axis=0) / design.n_patients,
            "records": records,
            "record_fraction": records / len(frame),
        }
    )
    keys = None
    if also_disjoint:
        rows = []
        for column in also_disjoint:
            spread = frame.groupby(column, dropna=True)[split].nunique()
            rows.append(
                {
                    "key": column,
                    "values": int(len(spread)),
                    "values_in_multiple_partitions": int((spread > 1).sum()),
                }
            )
        keys = pd.DataFrame(rows)
    test_like = overlap.iloc[-1] if k >= 2 else None
    summary.update(
        {
            "patients": design.n_patients,
            "records": int(len(frame)),
            "patients_in_multiple_partitions": multiple,
            "patients_in_multiple_partitions_fraction": multiple / design.n_patients,
            "max_record_fraction_shared": float(overlap["record_fraction_of_b"].max()),
        }
    )
    del test_like
    return AuditReport(
        partitions=names,
        sizes=sizes,
        overlap=overlap,
        balance=balance,
        summary=summary,
        keys=keys,
        notes=list(design.notes),
        threshold=threshold,
    )
