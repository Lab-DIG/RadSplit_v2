"""Adapters for scikit-learn and PyTorch workflows.

Neither library is imported here, so the adapters add no dependency.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import pandas as pd

from .api import split
from .result import SplitResult

GROUP = "__group__"
LABEL = "__label__"


class BalancedGroupKFold:
    """Patient disjoint, covariate balanced K fold splitter with the scikit-learn interface.

    It can be passed as ``cv=`` to ``cross_validate``, ``cross_val_score``,
    ``GridSearchCV`` and any other tool that accepts a scikit-learn splitter.

    Parameters
    ----------
    n_splits
        Number of folds.
    covariates
        Optional table with one row per sample (same order as ``X``) holding
        additional columns to balance, for example site, scanner, sex or age.
    seed
        Random seed.

    Examples
    --------
    >>> cv = BalancedGroupKFold(n_splits=5, covariates=table[["site", "age"]])
    >>> cross_validate(model, X, y, groups=table["patient_id"], cv=cv)
    """

    def __init__(self, n_splits: int = 5, *, covariates: pd.DataFrame | None = None,
                 seed: int = 0, **options) -> None:
        self.n_splits = n_splits
        self.covariates = covariates
        self.seed = seed
        self.options = options

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if groups is None:
            raise ValueError("groups (patient identifiers) are required")
        frame = pd.DataFrame({GROUP: np.asarray(groups)})
        balance: list[str] = []
        if self.covariates is not None:
            if len(self.covariates) != len(frame):
                raise ValueError("covariates must have one row per sample")
            for column in self.covariates.columns:
                frame[str(column)] = self.covariates[column].to_numpy()
                balance.append(str(column))
        target = None
        if y is not None:
            frame[LABEL] = np.asarray(y)
            target = LABEL
        result = split(frame, patient=GROUP, target=target, balance=balance,
                       folds=self.n_splits, seed=self.seed, **self.options)
        fold = result.data[result.column].to_numpy()
        for index in range(self.n_splits):
            held = fold == f"fold{index + 1}"
            yield np.flatnonzero(~held), np.flatnonzero(held)


def indices(result: SplitResult) -> dict[str, np.ndarray]:
    """Row positions of every partition, in the row order of the input table.

    Use with ``torch.utils.data.Subset(dataset, indices(result)["train"])`` or
    ``SubsetRandomSampler`` when the data set is indexed like the input table.
    """
    labels = result.data[result.column].to_numpy()
    return {name: np.flatnonzero(labels == name) for name in result.audit.partitions}


def file_lists(result: SplitResult, column: str) -> dict[str, list[str]]:
    """Values of ``column`` (for example image paths) per partition."""
    return {name: part[column].astype(str).tolist() for name, part in result.parts.items()}


def audit_indices(table: pd.DataFrame, *, patient: str, partitions: dict[str, Sequence[int]],
                  variables: Sequence[str], **options):
    """Audit a partition given as index lists (for example from scikit-learn or PyTorch)."""
    from .audit import audit

    label = np.full(len(table), None, dtype=object)
    for name, rows in partitions.items():
        label[np.asarray(rows, dtype=int)] = name
    frame = table.reset_index(drop=True).assign(__split__=label)
    return audit(frame, patient=patient, split="__split__", variables=variables,
                 partitions=list(partitions), **options)
