import numpy as np
import pandas as pd
import pytest

from radsplit import audit, split
from radsplit.synthetic import make_cohort

BALANCE = ["site", "scanner", "sex", "age"]


@pytest.fixture(scope="module")
def cohort() -> pd.DataFrame:
    return make_cohort(600, seed=3)


def test_partitions_are_patient_disjoint_and_complete(cohort):
    result = split(cohort, patient="patient_id", target="label", balance=BALANCE, seed=1)
    assert result.audit.disjoint
    assert len(result.data) == len(cohort)
    assert result.data["split"].notna().all()
    per_patient = result.data.groupby("patient_id")["split"].nunique()
    assert (per_patient == 1).all()


def test_sizes_follow_largest_remainder_rounding(cohort):
    result = split(cohort, patient="patient_id", target="label", balance=BALANCE)
    sizes = result.audit.sizes.set_index("partition")["patients"].to_dict()
    assert sizes == {"train": 420, "tune": 60, "test": 120}


def test_same_seed_same_hash_different_seed_different_assignment(cohort):
    kwargs = dict(patient="patient_id", target="label", balance=BALANCE)
    first = split(cohort, seed=7, **kwargs)
    second = split(cohort, seed=7, **kwargs)
    third = split(cohort, seed=8, **kwargs)
    assert first.split_hash == second.split_hash
    assert first.assignments.equals(second.assignments)
    assert not first.assignments.equals(third.assignments)


def test_row_order_does_not_change_assignment(cohort):
    kwargs = dict(patient="patient_id", target="label", balance=BALANCE, seed=5)
    shuffled = cohort.sample(frac=1.0, random_state=0)
    assert split(cohort, **kwargs).split_hash == split(shuffled, **kwargs).split_hash


def test_balance_is_better_than_random_assignment(cohort):
    result = split(cohort, patient="patient_id", target="label", balance=BALANCE, seed=0)
    rng = np.random.default_rng(0)
    patients = cohort["patient_id"].unique()
    random_map = dict(zip(patients, rng.choice(["train", "tune", "test"], len(patients),
                                              p=[0.7, 0.1, 0.2])))
    random = cohort.assign(split=cohort["patient_id"].map(random_map))
    reference = audit(random, patient="patient_id", split="split", variables=["label", *BALANCE])
    assert result.audit.summary["mean_abs_smd"] < 0.5 * reference.summary["mean_abs_smd"]
    assert result.audit.balanced


def test_cross_validation_folds(cohort):
    result = split(cohort, patient="patient_id", target="label", balance=["site", "age"], folds=5)
    assert result.audit.partitions == [f"fold{i}" for i in range(1, 6)]
    assert result.audit.sizes["patients"].tolist() == [120] * 5
    assert result.audit.disjoint


def test_custom_partition_names_and_two_way_split(cohort):
    result = split(cohort, patient="patient_id", target="label",
                   ratios={"development": 0.8, "holdout": 0.2})
    assert set(result.parts) == {"development", "holdout"}
    assert len(result["holdout"]["patient_id"].unique()) == 120


def test_fixed_assignments_are_kept(cohort):
    frame = cohort.copy()
    frame["official"] = np.where(frame["site"] == "site 4", "test", None)
    result = split(frame, patient="patient_id", target="label", balance=["sex", "age"],
                   fixed="official")
    kept = result.data.loc[frame["official"].eq("test").to_numpy(), "split"]
    assert (kept == "test").all()
    assert result.audit.disjoint


def test_record_level_variable_is_balanced_per_record(cohort):
    result = split(cohort, patient="patient_id", target="label", balance=["scanner"])
    rows = result.audit.balance
    assert (rows.loc[rows["variable"] == "scanner", "unit"] == "record").all()
    assert (rows.loc[rows["variable"] == "label", "unit"] == "patient").all()


def test_missing_values_are_handled():
    frame = make_cohort(300, seed=1)
    rng = np.random.default_rng(1)
    frame.loc[rng.random(len(frame)) < 0.2, "sex"] = None
    frame.loc[rng.random(len(frame)) < 0.2, "age"] = np.nan
    result = split(frame, patient="patient_id", target="label", balance=["sex", "age"])
    terms = result.audit.balance["term"].tolist()
    assert "sex: (missing)" in terms and "age: (missing)" in terms
    assert result.audit.disjoint


def test_tiny_cohort_gets_three_non_empty_partitions():
    frame = pd.DataFrame({"patient_id": list("abcde"), "label": [0, 1, 0, 1, 0]})
    result = split(frame, patient="patient_id", target="label")
    assert (result.audit.sizes["patients"] >= 1).all()


@pytest.mark.parametrize(
    "kwargs, error",
    [
        (dict(patient="nope", target="label"), KeyError),
        (dict(patient="patient_id"), ValueError),
        (dict(patient="patient_id", target="label", ratios={"a": 0.5, "b": 0.6}), ValueError),
        (dict(patient="patient_id", target="label", folds=1), ValueError),
    ],
)
def test_invalid_arguments(cohort, kwargs, error):
    with pytest.raises(error):
        split(cohort, **kwargs)


def test_missing_patient_identifier_is_rejected(cohort):
    frame = cohort.copy()
    frame.loc[0, "patient_id"] = None
    with pytest.raises(ValueError):
        split(frame, patient="patient_id", target="label")


def test_save_writes_all_outputs(cohort, tmp_path):
    result = split(cohort, patient="patient_id", target="label", balance=BALANCE)
    result.save(tmp_path / "out")
    names = {p.name for p in (tmp_path / "out").iterdir()}
    assert names == {"assignments.csv", "balance.csv", "split.json", "audit.md", "audit.html"}
    assert "RadSplit report" in result.report()
