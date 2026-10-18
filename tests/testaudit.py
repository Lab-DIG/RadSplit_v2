import numpy as np
import pandas as pd

from radsplit import audit
from radsplit.synthetic import make_cohort


def leaky(seed: int = 0) -> pd.DataFrame:
    frame = make_cohort(500, seed=seed)
    rng = np.random.default_rng(seed)
    frame["split"] = rng.choice(["train", "tune", "test"], len(frame), p=[0.7, 0.1, 0.2])
    return frame


def test_record_level_random_split_is_flagged_as_leaky():
    report = audit(leaky(), patient="patient_id", split="split", variables=["label", "site"])
    assert not report.disjoint
    assert report.summary["patients_in_multiple_partitions"] > 0
    train_test = report.overlap.iloc[1]
    assert train_test["partition_a"] == "train" and train_test["partition_b"] == "test"
    assert 0 < train_test["record_fraction_of_b"] <= 1


def test_overlap_counts_match_a_hand_computed_example():
    frame = pd.DataFrame(
        {
            "patient_id": ["a", "a", "b", "b", "c", "d", "e", "e"],
            "split": ["train", "test", "train", "train", "test", "tune", "tune", "test"],
            "label": [1, 1, 0, 0, 1, 0, 1, 1],
        }
    )
    report = audit(frame, patient="patient_id", split="split", variables=["label"])
    overlap = report.overlap.set_index(["partition_a", "partition_b"])
    assert overlap.loc[("train", "test"), "shared_patients"] == 1
    assert overlap.loc[("train", "test"), "records_b_with_patient_in_a"] == 1
    assert overlap.loc[("tune", "test"), "shared_patients"] == 1
    assert report.summary["patients_in_multiple_partitions"] == 2


def test_standardized_difference_matches_manual_calculation():
    frame = pd.DataFrame(
        {
            "patient_id": range(8),
            "split": ["train"] * 4 + ["test"] * 4,
            "sex": ["F", "F", "F", "M", "F", "M", "M", "M"],
        }
    )
    report = audit(frame, patient="patient_id", split="split", variables=["sex"])
    row = report.balance.iloc[0]
    assert row["term"] == "sex: M"
    expected = abs(0.25 - 0.75) / 0.5  # pooled SD of a 50% proportion
    assert np.isclose(row["max_abs_smd"], expected)
    assert not report.balanced


def test_also_disjoint_reports_shared_keys():
    frame = leaky()
    report = audit(frame, patient="patient_id", split="split", variables=["label"],
                   also_disjoint=["site"])
    assert report.keys.iloc[0]["values_in_multiple_partitions"] == frame["site"].nunique()


def test_reports_render():
    report = audit(leaky(), patient="patient_id", split="split",
                   variables=["label", "site", "age"])
    assert "Partition audit" in report.to_markdown()
    assert report.to_html().startswith("<!doctype html>")
    table = report.table()
    assert list(table.columns) == ["Characteristic", "pooled", "train", "tune", "test", "max |SMD|"]


def test_rare_levels_are_pooled():
    frame = make_cohort(400, seed=2)
    frame["split"] = np.where(frame["patient_id"].str[-1].isin(list("01")), "test", "train")
    frame["code"] = "common"
    frame.loc[frame.index[:150], "code"] = "second"
    frame.loc[frame.index[150:300], "code"] = "third"
    frame.loc[frame.index[300:302], "code"] = ["rare one", "rare two"]
    report = audit(frame, patient="patient_id", split="split", variables=["code"])
    assert not report.balance["term"].str.contains("rare").any()
    assert any("pooled" in note for note in report.notes)
