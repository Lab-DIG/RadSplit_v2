"""Repeated seed benchmark of partitioning methods on four public cohorts."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cohorts
import methods
import numpy as np
import pandas as pd

from radsplit.audit import evaluate, overlap_table
from radsplit.design import build_design

RESULTS = Path(__file__).resolve().parents[1] / "results"


def run_cohort(cohort: cohorts.Cohort, seeds: int) -> pd.DataFrame:
    records = cohort.records
    design = build_design(records, patient="patient_id", variables=cohort.variables)
    groups = design.record_patient
    label = records[cohort.stratify].to_numpy()
    rerandomizer = methods.Rerandomizer(design)
    rows = []
    for seed in range(seeds):
        for method in methods.METHODS:
            started = time.perf_counter()
            if method == "Record random":
                codes = methods.record_random(len(records), seed)
            elif method == "Group random":
                codes = methods.group_random(groups, seed)
            elif method == "Stratified group":
                codes = methods.stratified_group(groups, label, seed)
            elif method == "Rerandomization":
                codes = rerandomizer(seed)
            else:
                codes = methods.radsplit_codes(records, cohort.variables, seed)
            runtime = time.perf_counter() - started
            summary = evaluate(design, codes, 3)
            overlap, multiple = overlap_table(groups, codes, methods.NAMES)
            present = np.zeros((design.n_patients, 3), dtype=bool)
            present[groups, codes] = True
            patient_fraction = present.sum(axis=0) / design.n_patients
            record_fraction = np.bincount(codes, minlength=3) / len(codes)
            prevalence = [label[codes == s].mean() for s in range(3)]
            train_test = overlap.iloc[1]
            rows.append(
                {
                    "cohort": cohort.key,
                    "seed": seed,
                    "method": method,
                    "mean_abs_smd": summary["mean_abs_smd"],
                    "max_abs_smd": summary["max_abs_smd"],
                    "terms": summary["terms"],
                    "terms_above_threshold": summary["terms_above_threshold"],
                    "patients_shared": multiple,
                    "test_patients_in_train": int(train_test["shared_patients"]),
                    "test_patients": int(train_test["patients_b"]),
                    "test_records_patient_in_train": int(train_test["records_b_with_patient_in_a"]),
                    "test_records": int(train_test["records_b"]),
                    "max_patient_fraction_error": float(
                        np.abs(patient_fraction - methods.RATIOS).max()),
                    "max_record_fraction_error": float(
                        np.abs(record_fraction - methods.RATIOS).max()),
                    "label_prevalence_range": float(max(prevalence) - min(prevalence)),
                    "runtime_seconds": runtime,
                }
            )
        print(f"{cohort.key}: seed {seed + 1}/{seeds}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=list(cohorts.LOADERS), action="append")
    parser.add_argument("--seeds", type=int, default=100)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    for key in args.cohort or list(cohorts.LOADERS):
        frame = run_cohort(cohorts.load(key), args.seeds)
        frame.to_csv(RESULTS / f"benchmark{key}.csv", index=False)
        print(frame.groupby("method", sort=False)[
            ["mean_abs_smd", "max_abs_smd", "terms_above_threshold", "runtime_seconds"]
        ].median())


if __name__ == "__main__":
    main()
