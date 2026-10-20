"""Summary statistics and manuscript tables from saved results."""

from __future__ import annotations

import json
from pathlib import Path

import cohorts
import numpy as np
import pandas as pd
from scipy.stats import levene, wilcoxon

RESULTS = Path(__file__).resolve().parents[1] / "results"
METHODS = ["Record random", "Group random", "Stratified group", "Rerandomization", "RadSplit"]
COMPARATORS = ["Group random", "Stratified group", "Rerandomization"]


def iqr(values: pd.Series, digits: int = 3) -> str:
    q1, median, q3 = np.nanpercentile(values, [25, 50, 75])
    return f"{median:.{digits}f} ({q1:.{digits}f}, {q3:.{digits}f})"


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def cohort_table() -> tuple[pd.DataFrame, dict]:
    rows, info = [], {}
    for cohort in cohorts.load_all():
        records = cohort.records
        per_patient = records.groupby("patient_id").size()
        label = records[cohort.stratify]
        row = {
            "Cohort": cohort.name,
            "Modality": cohort.modality,
            "Record unit": cohort.record_unit,
            "Patients": f"{cohort.n_patients:,}",
            "Records": f"{len(records):,}",
            "Records per patient, median (IQR), maximum": (
                f"{per_patient.median():.0f} ({per_patient.quantile(0.25):.0f}, "
                f"{per_patient.quantile(0.75):.0f}), {per_patient.max()}"),
            "Patients with more than one record": (
                f"{(per_patient > 1).sum():,} ({100 * (per_patient > 1).mean():.1f}%)"),
            "Primary label": cohort.stratify.replace("_", " "),
            "Primary label prevalence (records)": (
                f"{int(label.sum()):,}/{len(label):,} ({100 * label.mean():.1f}%)"),
            "Balance variables": len(cohort.variables),
        }
        info[cohort.key] = {
            "name": cohort.name, "patients": cohort.n_patients, "records": int(len(records)),
            "unit": cohort.record_unit + "s",
            "patients_multiple_records": int((per_patient > 1).sum()),
            "records_per_patient_max": int(per_patient.max()),
            "label_positive": int(label.sum()),
            "variables": cohort.variables,
        }
        for column in ("sex", "age"):
            if column in records:
                unique = records.drop_duplicates("patient_id")
                if column == "sex":
                    info[cohort.key]["female_patients"] = int((unique["sex"] == "F").sum())
                    info[cohort.key]["male_patients"] = int((unique["sex"] == "M").sum())
                    info[cohort.key]["sex_missing_patients"] = int(unique["sex"].isna().sum())
                else:
                    info[cohort.key]["age_mean_records"] = float(records["age"].mean())
                    info[cohort.key]["age_sd_records"] = float(records["age"].std())
                    info[cohort.key]["age_missing_records"] = int(records["age"].isna().sum())
        rows.append(row)
    return pd.DataFrame(rows), info


def benchmark_tables() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    bench = pd.concat([pd.read_csv(RESULTS / f"benchmark{key}.csv") for key in cohorts.LOADERS],
                      ignore_index=True)
    bench["test_share"] = bench["test_records_patient_in_train"] / bench["test_records"]
    rows, stats, tests = [], {}, []
    for key in cohorts.LOADERS:
        part = bench[bench["cohort"] == key]
        stats[key] = {"terms": int(part["terms"].iloc[0]), "seeds": int(part["seed"].nunique())}
        for method in METHODS:
            m = part[part["method"] == method]
            rows.append(
                {
                    "Cohort": key, "Method": method,
                    "Test records with patient in training, %": iqr(100 * m["test_share"], 1),
                    "Mean absolute SMD": iqr(m["mean_abs_smd"], 4),
                    "Maximum absolute SMD": iqr(m["max_abs_smd"], 3),
                    "Seeds with any term above 0.1": (
                        f"{int((m['terms_above_threshold'] > 0).sum())}/{len(m)}"),
                    "Run time, s": iqr(m["runtime_seconds"], 2),
                }
            )
            stats[key][method] = {
                "mean_abs_smd_median": float(m["mean_abs_smd"].median()),
                "mean_abs_smd_q1": float(m["mean_abs_smd"].quantile(0.25)),
                "mean_abs_smd_q3": float(m["mean_abs_smd"].quantile(0.75)),
                "max_abs_smd_median": float(m["max_abs_smd"].median()),
                "max_abs_smd_q1": float(m["max_abs_smd"].quantile(0.25)),
                "max_abs_smd_q3": float(m["max_abs_smd"].quantile(0.75)),
                "seeds_any_term_above": int((m["terms_above_threshold"] > 0).sum()),
                "terms_above_median": float(m["terms_above_threshold"].median()),
                "test_share_median": float(m["test_share"].median()),
                "test_share_min": float(m["test_share"].min()),
                "test_share_max": float(m["test_share"].max()),
                "test_records_leaked_median": float(m["test_records_patient_in_train"].median()),
                "test_records_median": float(m["test_records"].median()),
                "patients_shared_median": float(m["patients_shared"].median()),
                "patients_shared_max": int(m["patients_shared"].max()),
                "max_patient_fraction_error_max": float(m["max_patient_fraction_error"].max()),
                "max_record_fraction_error_median": float(m["max_record_fraction_error"].median()),
                "label_prevalence_range_median": float(m["label_prevalence_range"].median()),
                "runtime_median": float(m["runtime_seconds"].median()),
            }
        pivot_mean = part.pivot(index="seed", columns="method", values="mean_abs_smd")
        pivot_max = part.pivot(index="seed", columns="method", values="max_abs_smd")
        for comparator in COMPARATORS:
            ratio = pivot_mean[comparator] / pivot_mean["RadSplit"]
            tests.append(
                {
                    "cohort": key, "comparator": comparator,
                    "p_mean": float(wilcoxon(pivot_mean["RadSplit"], pivot_mean[comparator]).pvalue),
                    "p_max": float(wilcoxon(pivot_max["RadSplit"], pivot_max[comparator]).pvalue),
                    "seeds_radsplit_lower_mean": int((pivot_mean["RadSplit"] < pivot_mean[comparator]).sum()),
                    "seeds_radsplit_lower_max": int((pivot_max["RadSplit"] < pivot_max[comparator]).sum()),
                    "median_fold_reduction_mean": float(ratio.median()),
                    "median_relative_reduction_percent": float(
                        (100 * (1 - pivot_mean["RadSplit"] / pivot_mean[comparator])).median()),
                }
            )
    tests = pd.DataFrame(tests)
    tests["p_mean_holm"] = holm(tests["p_mean"].tolist())
    tests["p_max_holm"] = holm(tests["p_max"].tolist())
    return pd.DataFrame(rows), tests, stats


def downstream_table() -> tuple[pd.DataFrame, dict]:
    rows, stats = [], {}
    for task in ("covid", "cbisddsm"):
        data = pd.read_csv(RESULTS / f"downstream{task}.csv")
        data["gap"] = (data["auc_tune"] - data["auc_test"]).abs()
        stats[task] = {"test_records_median": float(data["test_records"].median())}
        pivot = data.pivot(index="seed", columns="method", values="auc_test")
        gap = data.pivot(index="seed", columns="method", values="gap")
        for method in METHODS:
            m = data[data["method"] == method]
            entry = {
                "auc_test_median": float(m["auc_test"].median()),
                "auc_test_q1": float(m["auc_test"].quantile(0.25)),
                "auc_test_q3": float(m["auc_test"].quantile(0.75)),
                "auc_test_sd": float(m["auc_test"].std()),
                "auc_test_min": float(m["auc_test"].min()),
                "auc_test_max": float(m["auc_test"].max()),
                "gap_median": float(m["gap"].median()),
                "gap_q1": float(m["gap"].quantile(0.25)),
                "gap_q3": float(m["gap"].quantile(0.75)),
            }
            if method != "Record random":
                diff = pivot["Record random"] - pivot[method]
                entry["record_random_minus_method_median"] = float(diff.median())
                entry["p_vs_record_random"] = float(wilcoxon(pivot["Record random"], pivot[method]).pvalue)
            if method not in ("Record random", "RadSplit"):
                entry["p_variance_vs_radsplit"] = float(
                    levene(pivot[method], pivot["RadSplit"], center="median").pvalue)
                entry["p_gap_vs_radsplit"] = float(wilcoxon(gap[method], gap["RadSplit"]).pvalue)
            stats[task][method] = entry
            rows.append(
                {
                    "Task": task, "Method": method,
                    "Test AUC, median (IQR)": iqr(m["auc_test"], 3),
                    "Test AUC, SD across seeds": f"{m['auc_test'].std():.3f}",
                    "Absolute tuning to test AUC difference": iqr(m["gap"], 3),
                }
            )
        random = data[data["method"] == "Record random"]
        stats[task]["record_random_exposed_auc_median"] = float(
            random["auc_test_patient_in_train"].median())
        stats[task]["record_random_unexposed_auc_median"] = float(
            random["auc_test_patient_not_in_train"].median())
        both = random.dropna(subset=["auc_test_patient_in_train", "auc_test_patient_not_in_train"])
        stats[task]["p_exposed_vs_unexposed"] = float(
            wilcoxon(both["auc_test_patient_in_train"], both["auc_test_patient_not_in_train"]).pvalue)
        stats[task]["record_random_test_share_median"] = float(
            (random["test_records_patient_in_train"] / random["test_records"]).median())
    return pd.DataFrame(rows), stats


def size_stats() -> dict:
    data = pd.read_csv(RESULTS / "cohortsize.csv")
    out = {}
    for (patients, method), part in data.groupby(["patients", "method"]):
        out.setdefault(str(patients), {})[method] = {
            "mean_abs_smd_median": float(part["mean_abs_smd"].median()),
            "max_abs_smd_median": float(part["max_abs_smd"].median()),
            "samples_any_term_above": int((part["terms_above_threshold"] > 0).sum()),
            "samples": int(len(part)),
            "runtime_median": float(part["runtime_seconds"].median()),
        }
    return out


def sensitivity_stats() -> dict:
    data = pd.read_csv(RESULTS / "sensitivity.csv")
    out = {}
    keys = ["analysis", "cohort", "method", "terms_evaluated"]
    data["terms_evaluated"] = data["terms_evaluated"].fillna("all")
    for name, part in data.groupby(keys):
        out[" | ".join(name)] = {
            "mean_abs_smd_median": float(part["mean_abs_smd"].median()),
            "max_abs_smd_median": float(part["max_abs_smd"].median()),
            "runs": int(len(part)),
        }
    return out


def main() -> None:
    table1, cohort_info = cohort_table()
    table3, tests, bench_stats = benchmark_tables()
    table4, down_stats = downstream_table()
    table1.to_csv(RESULTS / "table1cohorts.csv", index=False)
    table3.to_csv(RESULTS / "table3benchmark.csv", index=False)
    tests.to_csv(RESULTS / "benchmarktests.csv", index=False)
    table4.to_csv(RESULTS / "table4downstream.csv", index=False)
    summary = {
        "cohorts": cohort_info,
        "benchmark": bench_stats,
        "benchmark_tests": tests.to_dict(orient="records"),
        "downstream": down_stats,
        "cohort_size": size_stats() if (RESULTS / "cohortsize.csv").exists() else {},
        "sensitivity": sensitivity_stats() if (RESULTS / "sensitivity.csv").exists() else {},
        "official": json.loads((RESULTS / "officialaudit.json").read_text()),
    }
    if (RESULTS / "shiftsummary.csv").exists():
        summary["shift"] = pd.read_csv(RESULTS / "shiftsummary.csv").to_dict(orient="records")
    if (RESULTS / "integrity.json").exists():
        summary["integrity"] = json.loads((RESULTS / "integrity.json").read_text())
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.set_option("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 60)
    print(table1.T, table3, tests, table4, sep="\n\n")


if __name__ == "__main__":
    main()
