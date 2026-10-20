"""Audit of distributed (official) partitions of two public data sets."""

from __future__ import annotations

import json
from pathlib import Path

import cohorts
import pandas as pd

from radsplit import audit

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    summary = {}
    for key, extra in (("cbisddsm", ["lesion_id"]), ("nih", None)):
        cohort = cohorts.load(key)
        records = cohort.records.copy()
        variables = list(cohort.variables)
        records["records_per_patient"] = records.groupby("patient_id")["patient_id"].transform(
            "size").astype(float)
        report = audit(records, patient="patient_id", split="official_split",
                       variables=[*variables, "records_per_patient"],
                       continuous=["records_per_patient"], also_disjoint=extra)
        report.balance.to_csv(RESULTS / f"officialbalance{key}.csv", index=False)
        report.overlap.to_csv(RESULTS / f"officialoverlap{key}.csv", index=False)
        (RESULTS / f"officialaudit{key}.md").write_text(report.to_markdown(), encoding="utf-8")
        entry = dict(report.summary)
        entry["sizes"] = report.sizes.to_dict(orient="records")
        entry["overlap"] = report.overlap.to_dict(orient="records")
        if report.keys is not None:
            entry["keys"] = report.keys.to_dict(orient="records")
        worst = report.balance.sort_values("max_abs_smd", ascending=False).head(5)
        entry["largest_differences"] = worst[["term", "mean_train", "mean_test",
                                              "max_abs_smd"]].to_dict(orient="records")
        if key == "cbisddsm":
            within = {}
            for kind, part in records.groupby("abnormality"):
                spread = part.groupby("patient_id")["official_split"].nunique()
                within[kind] = int((spread > 1).sum())
            entry["patients_in_both_within_abnormality_type"] = within
            image = records["patient_id"] + "|" + records["laterality"] + "|" + records["view"]
            spread = records.assign(image=image).groupby("image")["official_split"].nunique()
            entry["mammograms"] = int(len(spread))
            entry["mammograms_in_both"] = int((spread > 1).sum())
        summary[key] = entry
        print(report.to_markdown()[:1500])
    (RESULTS / "officialaudit.json").write_text(json.dumps(summary, indent=2, default=float),
                                                encoding="utf-8")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
