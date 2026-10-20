"""Partitioning methods compared in the benchmark.

Every method returns an integer partition code per record (0 training,
1 tuning, 2 test) for a 70%/10%/20% allocation.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from radsplit import split as radsplit
from radsplit.design import Design
from radsplit.optimizer import target_counts

RATIOS = np.array([0.7, 0.1, 0.2])
NAMES = ["train", "tune", "test"]
METHODS = ["Record random", "Group random", "Stratified group", "Rerandomization", "RadSplit"]
DRAWS = 100


def record_random(n_records: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    counts = target_counts(n_records, RATIOS)
    codes = np.repeat(np.arange(3), counts)
    return codes[rng.permutation(n_records)]


def group_random(groups: np.ndarray, seed: int) -> np.ndarray:
    """Two sequential scikit-learn GroupShuffleSplit calls."""
    index = np.arange(len(groups))
    outer = GroupShuffleSplit(n_splits=1, train_size=RATIOS[0], random_state=seed)
    train, held = next(outer.split(index, groups=groups))
    inner = GroupShuffleSplit(n_splits=1, train_size=RATIOS[1] / (RATIOS[1] + RATIOS[2]),
                              random_state=seed)
    tune, test = next(inner.split(held, groups=groups[held]))
    codes = np.zeros(len(groups), dtype=int)
    codes[held[tune]] = 1
    codes[held[test]] = 2
    return codes


def stratified_group(groups: np.ndarray, label: np.ndarray, seed: int) -> np.ndarray:
    """Nested scikit-learn StratifiedGroupKFold: one of five folds for testing,
    then one of eight folds of the remainder for tuning."""
    index = np.arange(len(groups))
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    rest, test = next(outer.split(index, label, groups))
    inner = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=seed)
    _, tune = next(inner.split(rest, label[rest], groups[rest]))
    codes = np.zeros(len(groups), dtype=int)
    codes[rest[tune]] = 1
    codes[test] = 2
    return codes


class Rerandomizer:
    """Best of ``draws`` random patient level assignments.

    The retained draw is the one with the lowest mean absolute pairwise
    standardized mean difference, which is the primary benchmark outcome.
    """

    def __init__(self, design: Design, draws: int = DRAWS) -> None:
        self.design = design
        self.draws = draws
        self.num, self.den = design.patient_matrix()
        self.primary = np.array([not term.auxiliary for term in design.terms])
        self.counts = target_counts(design.n_patients, RATIOS)
        self.labels = np.repeat(np.arange(3), self.counts)

    def score(self, patient_codes: np.ndarray) -> float:
        onehot = np.eye(3)[patient_codes]
        with np.errstate(invalid="ignore", divide="ignore"):
            means = (onehot.T @ self.num) / (onehot.T @ self.den)
        diffs = [np.abs(means[a] - means[b]) / self.design.pooled_sd
                 for a, b in combinations(range(3), 2)]
        return float(np.nanmean(np.vstack(diffs)[:, self.primary]))

    def __call__(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        best_score, best = np.inf, None
        for _ in range(self.draws):
            codes = self.labels[rng.permutation(self.design.n_patients)]
            value = self.score(codes)
            if value < best_score:
                best_score, best = value, codes
        assert best is not None
        return best[self.design.record_patient]


def radsplit_codes(records: pd.DataFrame, variables: list[str], seed: int, **kwargs) -> np.ndarray:
    result = radsplit(records, patient="patient_id", balance=variables,
                      ratios=dict(zip(NAMES, RATIOS)), seed=seed, **kwargs)
    return result.data["split"].map({name: i for i, name in enumerate(NAMES)}).to_numpy()
