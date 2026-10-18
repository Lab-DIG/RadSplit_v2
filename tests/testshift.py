import numpy as np
import pandas as pd
import pytest

from radsplit import holdout, leave_one_group_out
from radsplit.shift import EXCLUDED, shift_table
from radsplit.synthetic import make_cohort


@pytest.fixture(scope="module")
def cohort() -> pd.DataFrame:
    return make_cohort(600, seed=7)  # scanner varies within patient, site does not


def test_holdout_of_patient_level_group(cohort):
    result = holdout(cohort, patient="patient_id", by="site", values=["site 4"],
                     target="label", balance=["sex", "age"])
    data = result.data
    assert set(data.loc[data["split"] == "test", "site"]) == {"site 4"}
    assert "site 4" not in set(data.loc[data["split"] != "test", "site"])
    assert result.audit.disjoint
    assert result.audit.summary["holdout"]["mixed_patients"] == 0
    assert (data["split"] != EXCLUDED).all()


@pytest.mark.parametrize("policy", ["exclude", "test", "train"])
def test_mixed_patients_follow_the_policy(cohort, policy):
    result = holdout(cohort, patient="patient_id", by="scanner", values=["scanner 1"],
                     balance=["sex", "age"], mixed=policy)
    data = result.data
    kept = data[data["split"] != EXCLUDED]
    assert (kept.groupby("patient_id")["split"].nunique() == 1).all()
    info = result.audit.summary["holdout"]
    assert info["mixed_patients"] > 0
    test_scanners = set(kept.loc[kept["split"] == "test", "scanner"])
    development_scanners = set(kept.loc[kept["split"] != "test", "scanner"])
    if policy == "exclude":
        assert test_scanners == {"scanner 1"} and "scanner 1" not in development_scanners
        assert info["excluded_records"] > 0
    elif policy == "test":
        assert "scanner 1" not in development_scanners and len(test_scanners) > 1
        assert info["excluded_records"] == 0
    else:
        assert test_scanners == {"scanner 1"} and "scanner 1" not in development_scanners


def test_development_partitions_are_balanced(cohort):
    result = holdout(cohort, patient="patient_id", by="site", values=["site 1"],
                     target="label", balance=["sex", "age"])
    table = result.audit.balance.set_index("term")
    gap = (table["mean_train"] - table["mean_tune"]).abs() / table["sd_pooled"]
    assert gap.drop(index=[i for i in gap.index if i.startswith("site")]).max() < 0.1


def test_leave_one_group_out_and_shift_table(cohort):
    seen = []
    for value, result in leave_one_group_out(cohort, patient="patient_id", by="site",
                                             target="label", balance=["age"]):
        seen.append(value)
        table = shift_table(result)
        assert {"term", "smd"} <= set(table.columns)
        assert np.isfinite(table["smd"]).any()
    assert sorted(seen) == sorted(cohort["site"].unique())


def test_holdout_without_covariates_and_bad_arguments(cohort):
    result = holdout(cohort, patient="patient_id", by="site", values=["site 2"])
    assert result.audit.disjoint
    with pytest.raises(ValueError):
        holdout(cohort, patient="patient_id", by="site", values=["nowhere"])
    with pytest.raises(ValueError):
        holdout(cohort, patient="patient_id", by="site", values=["site 2"], mixed="other")
