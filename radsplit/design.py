"""Covariate design: turns a record table into balance terms.

A *term* is one column of a characteristics table: the mean of a continuous
variable or the proportion of one level of a categorical variable. Terms are
measured either per record (for example per image) or per patient.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MISSING = "(missing)"
OTHER = "(other)"


@dataclass(frozen=True)
class Term:
    """One balance term."""

    variable: str
    level: str | None
    kind: str  # "mean", "level", "missing" or "shape"
    unit: str  # "patient" or "record"
    auxiliary: bool = False

    @property
    def label(self) -> str:
        if self.kind == "mean":
            return self.variable
        return f"{self.variable}: {self.level}"


@dataclass
class Design:
    """Record level and patient level term values for one cohort."""

    patient_ids: np.ndarray
    record_patient: np.ndarray
    terms: list[Term]
    values: list[np.ndarray]  # per term; length n_records or n_patients, NaN = missing
    pooled_mean: np.ndarray
    pooled_sd: np.ndarray
    weights: np.ndarray
    notes: list[str] = field(default_factory=list)

    @property
    def n_patients(self) -> int:
        return int(len(self.patient_ids))

    @property
    def n_records(self) -> int:
        return int(len(self.record_patient))

    def patient_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        """Per patient numerators and denominators for every term."""
        n, d = self.n_patients, len(self.terms)
        num = np.zeros((n, d))
        den = np.zeros((n, d))
        for j, (term, value) in enumerate(zip(self.terms, self.values)):
            valid = ~np.isnan(value)
            filled = np.where(valid, value, 0.0)
            if term.unit == "record":
                num[:, j] = np.bincount(self.record_patient, weights=filled, minlength=n)
                den[:, j] = np.bincount(self.record_patient, weights=valid, minlength=n)
            else:
                num[:, j] = filled
                den[:, j] = valid
        return num, den

    def partition_means(self, record_split: np.ndarray, k: int) -> np.ndarray:
        """Term means within each partition, shape (k, n_terms).

        ``record_split`` gives the partition code of every record. Patient
        level terms are computed over the unique patients present in each
        partition, so the function also works for partitions that are not
        patient disjoint.
        """
        d = len(self.terms)
        out = np.full((k, d), np.nan)
        pair = np.unique(np.stack([self.record_patient, record_split]), axis=1)
        pat_of_pair, split_of_pair = pair[0], pair[1]
        for j, (term, value) in enumerate(zip(self.terms, self.values)):
            if term.unit == "record":
                v, s = value, record_split
            else:
                v, s = value[pat_of_pair], split_of_pair
            valid = ~np.isnan(v)
            total = np.bincount(s[valid], weights=v[valid], minlength=k)
            count = np.bincount(s[valid], minlength=k)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[:, j] = total / count
        return out


def _is_continuous(series: pd.Series) -> bool:
    return (
        pd.api.types.is_numeric_dtype(series)
        and not pd.api.types.is_bool_dtype(series)
        and series.nunique(dropna=True) > 10
    )


def _level_name(value: object) -> str:
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def build_design(
    data: pd.DataFrame,
    *,
    patient: str,
    variables: Sequence[str],
    continuous: Sequence[str] | None = None,
    categorical: Sequence[str] | None = None,
    weights: Mapping[str, float] | None = None,
    rare: float = 0.01,
    shape_bins: int = 0,
) -> Design:
    """Build balance terms for ``variables``.

    Parameters
    ----------
    rare
        Levels (and missing value indicators) observed in less than this
        fraction of units are pooled into ``(other)``; a pooled level that is
        still below the threshold is not used as a term.
    shape_bins
        When greater than 1, quantile bin indicators are added for each
        continuous variable as auxiliary terms. They are used by the optimizer
        to balance distribution shape and are excluded from summaries.
    """
    if data[patient].isna().any():
        raise ValueError("Patient identifiers cannot be missing")
    patient_ids, record_patient = np.unique(data[patient].to_numpy(), return_inverse=True)
    n_pat = len(patient_ids)
    first_row = np.full(n_pat, -1, dtype=np.int64)
    order = np.arange(len(data))[::-1]
    first_row[record_patient[order]] = order
    forced_cont = set(continuous or [])
    forced_cat = set(categorical or [])
    weights = dict(weights or {})

    terms: list[Term] = []
    values: list[np.ndarray] = []
    means: list[float] = []
    sds: list[float] = []
    term_weights: list[float] = []
    notes: list[str] = []

    def add(term: Term, value: np.ndarray, weight: float) -> None:
        valid = ~np.isnan(value)
        mean = float(value[valid].mean()) if valid.any() else np.nan
        sd = float(value[valid].std()) if valid.any() else np.nan
        if not np.isfinite(sd) or sd <= 1e-12:
            notes.append(f"{term.label}: no variation, not used")
            return
        terms.append(term)
        values.append(value.astype(float))
        means.append(mean)
        sds.append(sd)
        term_weights.append(weight)

    for variable in dict.fromkeys(variables):
        if variable not in data.columns:
            raise KeyError(f"Missing column: {variable}")
        series = data[variable]
        per_patient = series.groupby(record_patient).nunique(dropna=True)
        unit = "patient" if int(per_patient.max()) <= 1 else "record"
        if unit == "patient":
            # first non missing value per patient
            filled = series.groupby(record_patient).first()
            series_u = filled.reindex(range(n_pat))
        else:
            series_u = series.reset_index(drop=True)
        weight = float(weights.get(variable, 1.0))
        is_cont = variable in forced_cont or (
            variable not in forced_cat and _is_continuous(series_u)
        )
        missing = series_u.isna().to_numpy()
        n_units = len(series_u)
        if is_cont:
            numeric = pd.to_numeric(series_u, errors="coerce").to_numpy(dtype=float)
            add(Term(variable, None, "mean", unit), numeric, weight)
            if missing.mean() >= rare:
                add(Term(variable, MISSING, "missing", unit), missing.astype(float), weight)
            if shape_bins and shape_bins > 1:
                valid = ~np.isnan(numeric)
                edges = np.unique(
                    np.quantile(numeric[valid], np.linspace(0, 1, shape_bins + 1)[1:-1])
                )
                for edge in edges:
                    indicator = np.where(valid, (numeric <= edge).astype(float), np.nan)
                    add(
                        Term(variable, f"<= {edge:g}", "shape", unit, auxiliary=True),
                        indicator,
                        weight / max(len(edges), 1),
                    )
            continue
        labels = series_u.map(_level_name, na_action="ignore").astype(object)
        labels[missing] = MISSING
        counts = labels.value_counts()
        if len(counts) <= 2:
            # binary variable: one term, kept even when the minority level is rare
            levels = sorted(counts.index, key=str)[-1:] if len(counts) == 2 else []
            if levels == [MISSING]:
                levels = [level for level in counts.index if level != MISSING]
        else:
            levels = counts[counts >= max(rare * n_units, 1)].index.tolist()
            pooled = [level for level in counts.index if level not in levels]
            if pooled:
                labels = labels.where(labels.isin(levels), OTHER)
                notes.append(f"{variable}: {len(pooled)} rare levels pooled")
                if float((labels == OTHER).mean()) >= rare:
                    levels.append(OTHER)
            levels = sorted(levels, key=str)
            if len(levels) == 2 and not pooled:
                levels = levels[-1:]
            if len(levels) == 2 and pooled and OTHER not in levels:
                levels = levels[-1:]
        array = labels.to_numpy()
        for level in levels:
            kind = "missing" if level == MISSING else "level"
            add(Term(variable, level, kind, unit), (array == level).astype(float), weight)

    if not terms:
        raise ValueError("No usable balance terms were found")
    return Design(
        patient_ids=patient_ids,
        record_patient=record_patient,
        terms=terms,
        values=values,
        pooled_mean=np.asarray(means),
        pooled_sd=np.asarray(sds),
        weights=np.asarray(term_weights),
        notes=notes,
    )
