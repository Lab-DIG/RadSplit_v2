import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

from radsplit import split
from radsplit.frameworks import BalancedGroupKFold, audit_indices, file_lists, indices
from radsplit.synthetic import make_cohort


@pytest.fixture(scope="module")
def cohort() -> pd.DataFrame:
    return make_cohort(400, seed=5)


def test_scikit_learn_splitter_is_patient_disjoint_and_usable_in_cross_validate(cohort):
    cv = BalancedGroupKFold(n_splits=4, covariates=cohort[["site", "sex"]], seed=1)
    groups = cohort["patient_id"].to_numpy()
    seen = np.zeros(len(cohort), dtype=int)
    for train, test in cv.split(cohort, cohort["label"], groups):
        assert not set(groups[train]) & set(groups[test])
        seen[test] += 1
    assert (seen == 1).all()
    scores = cross_validate(LogisticRegression(), cohort[["age"]], cohort["label"],
                            groups=groups, cv=cv)
    assert len(scores["test_score"]) == 4


def test_splitter_requires_groups(cohort):
    with pytest.raises(ValueError):
        next(BalancedGroupKFold().split(cohort))


def test_indices_and_file_lists_cover_every_row(cohort):
    result = split(cohort, patient="patient_id", target="label", balance=["site"])
    index = indices(result)
    assert sorted(np.concatenate(list(index.values()))) == list(range(len(cohort)))
    lists = file_lists(result, "examination_id")
    assert sum(len(v) for v in lists.values()) == len(cohort)


def test_audit_of_index_lists_from_another_tool(cohort):
    rng = np.random.default_rng(0)
    order = rng.permutation(len(cohort))
    parts = {"train": order[:800], "test": order[800:]}
    report = audit_indices(cohort, patient="patient_id", partitions=parts,
                           variables=["label", "site"])
    assert not report.disjoint
    assert report.partitions == ["train", "test"]
