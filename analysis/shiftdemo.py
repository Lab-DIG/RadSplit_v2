"""Held out group partitions on two public chest CT cohorts.

For NLST and LIDC-IDRI, each manufacturer, each common scanner model and each
common reconstruction kernel is held out in turn as the test set. The
remaining patients are divided into balanced training and tuning partitions.
The script records partition sizes, the handling of patients with records
inside and outside the held out group, the balance between training and
tuning partitions, and the standardized mean difference between training and
test partitions for the covariates that are not nested in the held out
variable.
"""

from __future__ import annotations

from pathlib import Path

import cohorts
import numpy as np
import pandas as pd

from radsplit import leave_one_group_out
from radsplit.shift import shift_table

RESULTS = Path(__file__).resolve().parents[1] / "results"
NESTED = {  # covariates defined by the held out variable; not informative as shift terms
    "manufacturer": {"manufacturer", "scanner_model", "kernel"},
    "scanner_model": {"manufacturer", "scanner_model", "kernel"},
    "kernel": {"manufacturer", "scanner_model", "kernel"},
}


def run(key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort = cohorts.load(key)
    records = cohort.records
    variables = [v for v in cohort.variables]
    summary, terms = [], []
    for by in ("manufacturer", "scanner_model", "kernel"):
        free = [v for v in variables if v not in NESTED[by]]
        minimum = max(20, int(0.02 * cohort.n_patients))
        for value, result in leave_one_group_out(
                records, patient="patient_id", by=by, min_patients=minimum, balance=free,
                mixed="exclude", seed=0):
            info = result.audit.summary["holdout"]
            sizes = result.audit.sizes.set_index("partition")
            table = result.audit.balance
            inner = ((table["mean_train"] - table["mean_tune"]).abs() / table["sd_pooled"])
            shift = shift_table(result)
            shift = shift[shift["variable"].isin(free)]
            summary.append(
                {
                    "cohort": key, "held_out_variable": by, "held_out_value": value,
                    "test_patients": info["test_patients"], "test_records": info["test_records"],
                    "train_patients": int(sizes.loc["train", "patients"]),
                    "tune_patients": int(sizes.loc["tune", "patients"]),
                    "mixed_patients": info["mixed_patients"],
                    "excluded_records": info["excluded_records"],
                    "patients_in_multiple_partitions":
                        result.audit.summary["patients_in_multiple_partitions"],
                    "train_tune_max_abs_smd": float(
                        inner[table["variable"].isin(free)].max()),
                    "train_test_max_abs_smd": float(shift["smd"].abs().max()),
                    "train_test_mean_abs_smd": float(shift["smd"].abs().mean()),
                    "terms_above_threshold": int((shift["smd"].abs() > 0.1).sum()),
                    "terms": int(len(shift)),
                }
            )
            for _, row in shift.iterrows():
                terms.append({"cohort": key, "held_out_variable": by, "held_out_value": value,
                              "term": row["term"], "mean_train": row["mean_train"],
                              "mean_test": row["mean_test"], "smd": row["smd"]})
            print(key, by, value, info, flush=True)
    return pd.DataFrame(summary), pd.DataFrame(terms)


def main() -> None:
    parts = [run(key) for key in ("lidc", "nlst")]
    pd.concat([p[0] for p in parts]).to_csv(RESULTS / "shiftsummary.csv", index=False)
    pd.concat([p[1] for p in parts]).to_csv(RESULTS / "shiftterms.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(pd.concat([p[0] for p in parts]).round(3))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
