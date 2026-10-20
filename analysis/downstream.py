"""Downstream consequences of the partitioning method for model evaluation.

Two classification tasks are refit for every seed and method:

* covid: COVID-19 versus other findings on frontal radiographs, logistic
  regression on fixed embeddings of a pretrained DenseNet-121.
* cbisddsm: malignant versus benign lesion records, random forest on the
  radiologist assigned descriptors in the CBIS-DDSM case tables.

For each fit the script stores the area under the receiver operating
characteristic curve (AUC) in the tuning and test partitions. For record
level random partitions the test AUC is also computed separately for test
records whose patient does or does not appear in the training partition.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cohorts
import methods
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from radsplit.design import build_design

RESULTS = Path(__file__).resolve().parents[1] / "results"


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan")


def covid_task() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], str]:
    table = pd.read_parquet(RESULTS / "covidembeddings.parquet")
    features = table[[c for c in table.columns if c.startswith("f") and c[1:].isdigit()]]
    records = table[["patient_id", "covid19", "finding", "view", "sex", "age", "country"]]
    return (records, features.to_numpy(), records["covid19"].to_numpy(),
            ["finding", "view", "sex", "age", "country"], "covid19")


def cbis_task() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], str]:
    cohort = cohorts.load("cbisddsm")
    records = cohort.records
    descriptors = ["abnormality", "mass_shape", "mass_margins", "calc_type", "calc_distribution",
                   "breast_density", "assessment", "subtlety"]
    features = pd.get_dummies(records[descriptors].fillna("none"), dtype=float)
    return records, features.to_numpy(), records["malignant"].to_numpy(), cohort.variables, "malignant"


def fit(task: str, x: np.ndarray, y: np.ndarray, codes: np.ndarray, seed: int) -> np.ndarray:
    train = codes == 0
    if task == "covid":
        best, best_auc = None, -np.inf
        for c in (0.001, 0.01, 0.1):
            model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
            model.fit(x[train], y[train])
            score = model.predict_proba(x)[:, 1]
            value = safe_auc(y[codes == 1], score[codes == 1])
            if np.isnan(value) or value > best_auc:
                best, best_auc = score, value
        return best
    model = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
    model.fit(x[train], y[train])
    return model.predict_proba(x)[:, 1]


def run(task: str, seeds: int) -> pd.DataFrame:
    records, x, y, variables, label = covid_task() if task == "covid" else cbis_task()
    records = records.reset_index(drop=True)
    design = build_design(records, patient="patient_id", variables=variables)
    groups = design.record_patient
    rerandomizer = methods.Rerandomizer(design)
    rows = []
    for seed in range(seeds):
        for method in methods.METHODS:
            if method == "Record random":
                codes = methods.record_random(len(records), seed)
            elif method == "Group random":
                codes = methods.group_random(groups, seed)
            elif method == "Stratified group":
                codes = methods.stratified_group(groups, y, seed)
            elif method == "Rerandomization":
                codes = rerandomizer(seed)
            else:
                codes = methods.radsplit_codes(records, variables, seed)
            score = fit(task, x, y, codes, seed)
            test, tune = codes == 2, codes == 1
            in_train = np.isin(groups, groups[codes == 0])
            row = {
                "task": task, "seed": seed, "method": method,
                "auc_test": safe_auc(y[test], score[test]),
                "auc_tune": safe_auc(y[tune], score[tune]),
                "test_records": int(test.sum()),
                "test_records_patient_in_train": int((test & in_train).sum()),
                "prevalence_test": float(y[test].mean()),
            }
            if method == "Record random":
                row["auc_test_patient_in_train"] = safe_auc(y[test & in_train], score[test & in_train])
                row["auc_test_patient_not_in_train"] = safe_auc(
                    y[test & ~in_train], score[test & ~in_train])
            rows.append(row)
        if (seed + 1) % 10 == 0:
            print(f"{task}: seed {seed + 1}/{seeds}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--task", choices=["covid", "cbisddsm"], action="append")
    args = parser.parse_args()
    for task in args.task or ["covid", "cbisddsm"]:
        frame = run(task, args.seeds)
        frame.to_csv(RESULTS / f"downstream{task}.csv", index=False)
        print(frame.groupby("method", sort=False)[["auc_test", "auc_tune"]].agg(["median", "std"]))


if __name__ == "__main__":
    main()
