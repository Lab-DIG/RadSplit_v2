"""Covariate imbalance and run time as a function of cohort size.

Patients are sampled without replacement from NIH ChestX-ray14, and each
sample is partitioned 70%/10%/20% with every patient level method.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cohorts
import methods
import numpy as np
import pandas as pd

from radsplit.audit import evaluate
from radsplit.design import build_design

RESULTS = Path(__file__).resolve().parents[1] / "results"
SIZES = [100, 200, 500, 1000, 2000, 5000, 10000]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=30)
    args = parser.parse_args()
    cohort = cohorts.load("nih")
    patients = cohort.records["patient_id"].unique()
    rows = []
    for size in SIZES:
        for seed in range(args.seeds):
            rng = np.random.default_rng(1000 + seed)
            keep = patients if size >= len(patients) else rng.choice(patients, size, replace=False)
            records = cohort.records[cohort.records["patient_id"].isin(keep)].reset_index(drop=True)
            design = build_design(records, patient="patient_id", variables=cohort.variables)
            label = records[cohort.stratify].to_numpy()
            rerandomizer = methods.Rerandomizer(design)
            for method in methods.METHODS[1:]:
                started = time.perf_counter()
                if method == "Group random":
                    codes = methods.group_random(design.record_patient, seed)
                elif method == "Stratified group":
                    codes = methods.stratified_group(design.record_patient, label, seed)
                elif method == "Rerandomization":
                    codes = rerandomizer(seed)
                else:
                    codes = methods.radsplit_codes(records, cohort.variables, seed)
                runtime = time.perf_counter() - started
                summary = evaluate(design, codes, 3)
                rows.append({"patients": size, "records": len(records), "seed": seed,
                             "method": method, "runtime_seconds": runtime, **summary})
        print(f"size {size} done", flush=True)
        pd.DataFrame(rows).to_csv(RESULTS / "cohortsize.csv", index=False)


if __name__ == "__main__":
    main()
