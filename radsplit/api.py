"""Public entry point: patient disjoint, covariate balanced partitioning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .audit import audit
from .design import build_design
from .optimizer import ExchangeOptimizer
from .result import SplitResult, deterministic_hash

DEFAULT_RATIOS: dict[str, float] = {"train": 0.7, "tune": 0.1, "test": 0.2}
RECORDS = "records per patient"


def split(
    data: pd.DataFrame,
    *,
    patient: str,
    balance: Sequence[str] | None = None,
    target: str | Sequence[str] | None = None,
    ratios: Mapping[str, float] | Sequence[float] | None = None,
    folds: int | None = None,
    continuous: Sequence[str] | None = None,
    categorical: Sequence[str] | None = None,
    weights: Mapping[str, float] | None = None,
    fixed: str | None = None,
    balance_records: bool = True,
    shape_bins: int = 4,
    rare: float = 0.01,
    seed: int = 0,
    restarts: int = 1,
    max_swaps: int = 20000,
    column: str = "split",
) -> SplitResult:
    """Assign patients to partitions with balanced covariates.

    All records of a patient are assigned to the same partition. Partition
    sizes are exact in patients (largest remainder rounding). Among the
    assignments that satisfy both constraints, the function searches for one
    that minimizes standardized differences between each partition and the
    pooled cohort for every variable in ``target`` and ``balance``.

    Parameters
    ----------
    data
        One row per record: an image, a lesion, an examination, or a patient.
    patient
        Column with the patient identifier.
    balance
        Columns to balance. Columns that vary within a patient (for example
        view position) are balanced per record; columns that are constant
        within a patient (for example sex) are balanced per patient.
    target
        Label column or columns. They are balanced in the same way as
        ``balance`` and are listed first in the report.
    ratios
        Partition names and fractions, for example
        ``{"train": 0.7, "tune": 0.1, "test": 0.2}`` (the default). A plain
        sequence of three fractions uses the default names.
    folds
        If given, create this many equally sized cross validation folds named
        ``fold1`` to ``foldK`` instead of using ``ratios``.
    continuous, categorical
        Override automatic type detection (numeric columns with more than 10
        distinct values are treated as continuous).
    weights
        Optional relative weight per variable in the objective.
    fixed
        Column with a partition name for patients whose assignment must be
        kept (missing for patients that are free to move). Useful to keep an
        official test set or to place a whole site in the test partition.
    balance_records
        Also balance the number of records per patient, which keeps record
        fractions close to the patient fractions.
    shape_bins
        Number of quantile bins used to balance the distribution shape of
        continuous variables. Set to 0 to balance means only.
    rare
        Levels observed in fewer than this fraction of units are pooled.
    seed
        Seed of the random number generator. The same data, arguments, and
        seed give the same assignment and hash.
    restarts
        Number of random initializations; the best result is kept.
    max_swaps
        Upper bound on accepted patient exchanges per initialization.
    column
        Name of the partition column added to the returned records.
    """
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("data must be a non-empty pandas DataFrame")
    targets = [target] if isinstance(target, str) else list(target or [])
    variables = list(dict.fromkeys([*targets, *(balance or [])]))
    if not variables:
        raise ValueError("Provide at least one column in target or balance")
    needed = [patient, *variables, *([fixed] if fixed else [])]
    missing = sorted(set(needed) - set(data.columns))
    if missing:
        raise KeyError(f"Missing columns: {', '.join(missing)}")
    if data[patient].isna().any():
        raise ValueError("Patient identifiers cannot be missing")

    if folds is not None:
        if folds < 2:
            raise ValueError("folds must be at least 2")
        names = [f"fold{index + 1}" for index in range(folds)]
        fractions = np.full(folds, 1.0 / folds)
    else:
        if ratios is None:
            ratios = DEFAULT_RATIOS
        if isinstance(ratios, Mapping):
            names = [str(name) for name in ratios]
            fractions = np.asarray(list(ratios.values()), dtype=float)
        else:
            fractions = np.asarray(list(ratios), dtype=float)
            if len(fractions) != len(DEFAULT_RATIOS):
                raise ValueError("Give ratios as a mapping of partition name to fraction")
            names = list(DEFAULT_RATIOS)
        if len(fractions) < 2 or (fractions <= 0).any() or not np.isclose(fractions.sum(), 1.0):
            raise ValueError("ratios must be positive and sum to 1")

    frame = data.reset_index(drop=True)
    work = frame
    design_variables = list(variables)
    per_patient = work.groupby(patient)[patient].transform("size")
    if balance_records and per_patient.nunique() > 1:
        work = work.assign(**{RECORDS: per_patient.astype(float)})
        design_variables.append(RECORDS)
        continuous = [*(continuous or []), RECORDS]
    design = build_design(
        work, patient=patient, variables=design_variables, continuous=continuous,
        categorical=categorical, weights=weights, rare=rare, shape_bins=shape_bins,
    )
    num, den = design.patient_matrix()

    fixed_codes = None
    if fixed:
        per = frame.groupby(patient)[fixed].agg(lambda s: s.dropna().astype(str).unique().tolist())
        bad = [str(pid) for pid, values in per.items() if len(values) > 1]
        if bad:
            raise ValueError(f"Conflicting fixed partitions for patients: {bad[:5]}")
        lookup = {name: index for index, name in enumerate(names)}
        fixed_codes = np.full(design.n_patients, -1)
        for position, pid in enumerate(design.patient_ids):
            values = per.loc[pid]
            if values:
                if values[0] not in lookup:
                    raise ValueError(f"Unknown partition in fixed column: {values[0]}")
                fixed_codes[position] = lookup[values[0]]

    optimizer = ExchangeOptimizer(fractions, seed=seed, restarts=restarts, max_swaps=max_swaps)
    codes, info = optimizer.fit(
        num, den, design.pooled_mean, design.pooled_sd, design.weights, fixed_codes
    )
    labels = np.asarray(names, dtype=object)[codes]
    assignments = pd.DataFrame({patient: design.patient_ids, column: labels})
    out = frame.copy()
    out[column] = labels[design.record_patient]
    report = audit(
        out, patient=patient, split=column, variables=variables, partitions=names,
        continuous=[c for c in (continuous or []) if c != RECORDS], categorical=categorical,
        rare=rare,
    )
    config: dict[str, Any] = {
        "patient": patient,
        "target": targets,
        "balance": list(balance or []),
        "partitions": {name: float(value) for name, value in zip(names, fractions)},
        "continuous": [c for c in (continuous or []) if c != RECORDS],
        "categorical": list(categorical or []),
        "weights": dict(weights or {}),
        "fixed": fixed,
        "balance_records": balance_records,
        "shape_bins": shape_bins,
        "rare": rare,
        "seed": seed,
        "restarts": restarts,
        "max_swaps": max_swaps,
        "method": "random start with size constrained pairwise exchange",
    }
    report.summary["exchanges"] = int(info["swaps"])
    return SplitResult(
        data=out,
        assignments=assignments,
        audit=report,
        config=config,
        split_hash=deterministic_hash(assignments, config),
        column=column,
    )
