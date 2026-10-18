"""Held out group partitions for testing under distribution shift.

A balanced random partition asks whether a model works on new patients from
the same population. A held out group partition asks whether it works on a
scanner, reconstruction kernel, site or visit that it has not seen. Both are
patient disjoint.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .api import split
from .audit import audit
from .result import SplitResult, deterministic_hash

EXCLUDED = "excluded"
MIXED_POLICIES = ("exclude", "test", "train")


def holdout(
    data: pd.DataFrame,
    *,
    patient: str,
    by: str,
    values: Sequence[Any],
    balance: Sequence[str] | None = None,
    target: str | Sequence[str] | None = None,
    development: Mapping[str, float] | None = None,
    test: str = "test",
    mixed: str = "exclude",
    seed: int = 0,
    column: str = "split",
    **options: Any,
) -> SplitResult:
    """Hold out every record whose ``by`` value is in ``values`` as the test set.

    The remaining patients are divided into the ``development`` partitions
    (default 87.5% training and 12.5% tuning) with balanced covariates.

    A patient can have records in both the held out group and other groups
    (for example examinations on two scanners). ``mixed`` decides what happens
    to such patients:

    ``"exclude"`` (default)
        The patient goes to the test set and the records from other groups
        are marked ``excluded``. The test set then contains only held out
        group records and no patient is shared.
    ``"test"``
        All records of the patient go to the test set. The test set then also
        contains records from other groups.
    ``"train"``
        The patient stays in development and the held out group records of
        that patient are marked ``excluded``. Development then contains no
        held out group record, and the test set is smaller.

    The audit of the result compares the partitions on ``by``, ``target`` and
    ``balance``; differences between development and test partitions describe
    the shift and are expected.
    """
    if mixed not in MIXED_POLICIES:
        raise ValueError(f"mixed must be one of {MIXED_POLICIES}")
    if by not in data.columns:
        raise KeyError(f"Missing column: {by}")
    frame = data.reset_index(drop=True)
    wanted = {str(v) for v in values}
    held_record = frame[by].astype(str).isin(wanted).to_numpy() & frame[by].notna().to_numpy()
    if not held_record.any():
        raise ValueError(f"No record has {by} in {sorted(wanted)}")
    any_held = pd.Series(held_record).groupby(frame[patient].to_numpy()).transform("max")
    all_held = pd.Series(held_record).groupby(frame[patient].to_numpy()).transform("min")
    any_held, all_held = any_held.to_numpy(bool), all_held.to_numpy(bool)
    mixed_patient = any_held & ~all_held

    label = np.full(len(frame), None, dtype=object)
    if mixed == "exclude":
        label[any_held & held_record] = test
        label[mixed_patient & ~held_record] = EXCLUDED
    elif mixed == "test":
        label[any_held] = test
    else:
        label[all_held] = test
        label[mixed_patient & held_record] = EXCLUDED
    free = label == None  # noqa: E711
    if not free.any():
        raise ValueError("No patient remains for the development partitions")

    development = dict(development or {"train": 0.875, "tune": 0.125})
    total = sum(development.values())
    development = {name: value / total for name, value in development.items()}
    targets = [target] if isinstance(target, str) else list(target or [])
    variables = list(dict.fromkeys([*targets, *(balance or [])]))
    rest = frame.loc[free]
    if variables:
        inner = split(rest, patient=patient, target=targets or None, balance=balance,
                      ratios=development, seed=seed, column=column, **options)
        label[np.flatnonzero(free)] = inner.data[column].to_numpy()
        exchanges = inner.audit.summary.get("exchanges", 0)
    else:
        rng = np.random.default_rng(seed)
        patients = np.unique(rest[patient].to_numpy())
        names = list(development)
        cuts = np.cumsum([development[n] for n in names])[:-1] * len(patients)
        order = rng.permutation(len(patients))
        codes = np.searchsorted(cuts, np.arange(len(patients)), side="right")
        lookup = dict(zip(patients[order], np.asarray(names, dtype=object)[codes]))
        label[np.flatnonzero(free)] = rest[patient].map(lookup).to_numpy()
        exchanges = 0

    out = frame.copy()
    out[column] = label
    names = [*development, test]
    kept = out[out[column] != EXCLUDED]
    report = audit(kept, patient=patient, split=column,
                   variables=list(dict.fromkeys([by, *variables])), partitions=names)
    shift = {
        "by": by,
        "values": sorted(wanted),
        "mixed_policy": mixed,
        "mixed_patients": int(pd.Series(mixed_patient).groupby(frame[patient].to_numpy()).max().sum()),
        "excluded_records": int((label == EXCLUDED).sum()),
        "test_patients": int(kept.loc[kept[column] == test, patient].nunique()),
        "test_records": int((kept[column] == test).sum()),
    }
    report.summary.update({"exchanges": int(exchanges), "holdout": shift})
    report.notes.append(
        f"Held out {by} in {sorted(wanted)}: differences between development and "
        f"{test} partitions are the intended shift.")
    if shift["mixed_patients"]:
        report.notes.append(
            f"{shift['mixed_patients']} patients had records inside and outside the held out "
            f"group (policy: {mixed}; {shift['excluded_records']} records excluded).")
    assignments = (out.loc[out[column] != EXCLUDED, [patient, column]]
                   .drop_duplicates().reset_index(drop=True))
    config = {"method": "held out group with balanced development partitions", "patient": patient,
              "by": by, "values": sorted(wanted), "mixed": mixed, "target": targets,
              "balance": list(balance or []), "development": development, "seed": seed}
    return SplitResult(data=out, assignments=assignments, audit=report, config=config,
                       split_hash=deterministic_hash(assignments, config), column=column)


def leave_one_group_out(
    data: pd.DataFrame,
    *,
    patient: str,
    by: str,
    min_patients: int = 20,
    **options: Any,
) -> Iterator[tuple[Any, SplitResult]]:
    """Yield ``(value, result)`` with each value of ``by`` held out in turn.

    Values with fewer than ``min_patients`` patients are skipped. Other
    arguments are passed to :func:`holdout`.
    """
    sizes = data.groupby(by)[patient].nunique().sort_values(ascending=False)
    for value, size in sizes.items():
        if size >= min_patients:
            yield value, holdout(data, patient=patient, by=by, values=[value], **options)


def shift_table(result: SplitResult, development: str = "train", test: str = "test") -> pd.DataFrame:
    """Standardized mean difference between a development partition and the test set."""
    table = result.audit.balance
    out = table[["term", "variable", "level", f"mean_{development}", f"mean_{test}"]].copy()
    out["smd"] = (table[f"mean_{test}"] - table[f"mean_{development}"]) / table["sd_pooled"]
    return out.sort_values("smd", key=np.abs, ascending=False).reset_index(drop=True)
