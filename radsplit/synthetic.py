"""Synthetic imaging cohorts for tests, examples, and tutorials."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_cohort(
    n_patients: int = 1000,
    *,
    sites: int = 4,
    scanners: int = 3,
    positive_rate: float = 0.3,
    max_examinations: int = 4,
    site_label_correlation: float = 0.4,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate a de-identified synthetic cohort with repeated examinations.

    The label prevalence differs by site, scanner varies between examinations
    of the same patient, and age and sex are constant within a patient.
    """
    if n_patients < 3:
        raise ValueError("n_patients must be at least 3")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for index in range(n_patients):
        site = int(rng.integers(sites))
        shifted = positive_rate + site_label_correlation * (site / max(sites - 1, 1) - 0.5)
        label = int(rng.random() < np.clip(shifted, 0.02, 0.98))
        age = float(np.clip(rng.normal(58 + 5 * label, 14), 18, 95))
        sex = str(rng.choice(["F", "M"]))
        for examination in range(int(rng.integers(1, max_examinations + 1))):
            rows.append(
                {
                    "patient_id": f"P{index:06d}",
                    "examination_id": f"P{index:06d}E{examination}",
                    "label": label,
                    "site": f"site {site + 1}",
                    "scanner": f"scanner {int(rng.integers(scanners)) + 1}",
                    "age": round(age, 1),
                    "sex": sex,
                }
            )
    return pd.DataFrame(rows)
