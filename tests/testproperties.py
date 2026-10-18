from hypothesis import given, settings
from hypothesis import strategies as st

from radsplit import split
from radsplit.synthetic import make_cohort


@settings(max_examples=25, deadline=None)
@given(
    n=st.integers(min_value=3, max_value=120),
    seed=st.integers(min_value=0, max_value=10_000),
    folds=st.one_of(st.none(), st.integers(min_value=2, max_value=3)),
)
def test_every_patient_is_in_exactly_one_non_empty_partition(n, seed, folds):
    frame = make_cohort(n, seed=seed)
    result = split(frame, patient="patient_id", target="label", balance=["site", "age"],
                   folds=folds, seed=seed)
    assert result.audit.disjoint
    assert result.assignments["patient_id"].is_unique
    assert len(result.assignments) == n
    assert (result.audit.sizes["patients"] >= 1).all()
    assert result.audit.sizes["patients"].sum() == n
