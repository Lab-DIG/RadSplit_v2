"""Check internal consistency of frozen results without downloading datasets.

This checks saved outputs, not their provenance or independent reproducibility.
Run from any directory with the package's development dependencies installed.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def main():
    root = Path(__file__).resolve().parents[1] / "results"
    summary = json.loads((root / "summary.json").read_text())
    saved_tests = pd.read_csv(root / "benchmarktests.csv")
    cohorts = ["covid", "lidc", "cbisddsm", "nlst", "nih", "padchest"]
    comparators = ["Group random", "Stratified group", "Rerandomization"]
    pvalues = []
    for cohort in cohorts:
        data = pd.read_csv(root / f"benchmark{cohort}.csv")
        for method in [*comparators, "RadSplit"]:
            part = data[data.method == method]
            target = summary["benchmark"][cohort][method]
            assert len(part) == (30 if cohort == "nlst" else 100)
            assert part.seed.is_unique
            assert (part.patients_shared == 0).all()
            np.testing.assert_allclose(part.mean_abs_smd.median(),
                                       target["mean_abs_smd_median"], rtol=1e-10)
            assert (part.max_abs_smd > .1).sum() == target["seeds_any_term_above"]
        pivot = data.pivot(index="seed", columns="method", values="mean_abs_smd")
        for comparator in comparators:
            assert (pivot.RadSplit < pivot[comparator]).all()
            p = float(wilcoxon(pivot.RadSplit, pivot[comparator]).pvalue)
            pvalues.append(p)
            stored = saved_tests[(saved_tests.cohort == cohort)
                                 & (saved_tests.comparator == comparator)].iloc[0]
            np.testing.assert_allclose(np.median(pivot[comparator] / pivot.RadSplit),
                                       stored.median_fold_reduction_mean, rtol=1e-10)
    order = np.argsort(pvalues)
    adjusted = np.empty(len(order))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        np.asarray(pvalues)[order] * np.arange(len(order), 0, -1)))
    assert (adjusted < .001).all()
    assert sum(x["patients"] for x in summary["cohorts"].values()) == 127677
    assert sum(x["records"] for x in summary["cohorts"].values()) == 416224
    print("PASS: 24 cohort-method summaries, seed uniqueness, zero patient overlap,")
    print("18 paired fold reductions and Holm-adjusted significance conclusions,")
    print("and cohort totals. No source-data experiments were rerun.")


if __name__ == "__main__":
    main()
