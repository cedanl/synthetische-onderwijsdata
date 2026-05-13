import numpy as np
import pandas as pd
import pytest
from relsynth.logic.degree import DegreeModel


def _student_data() -> pd.DataFrame:
    rows = []
    for sid in range(1, 51):
        n = np.random.default_rng(sid).integers(1, 8)
        for yr in range(n):
            rows.append({"student_id": sid, "year": 2020 + yr})
    return pd.DataFrame(rows)


def test_sample_respects_min_max():
    df = _student_data()
    model = DegreeModel(random_state=0).fit(df, "student_id")
    counts = model.sample(200)
    assert counts.min() >= 1
    assert counts.max() <= df.groupby("student_id").size().max()


def test_sample_length():
    model = DegreeModel(random_state=1).fit(_student_data(), "student_id")
    assert len(model.sample(100)) == 100


def test_default_sample_without_fit():
    model = DegreeModel(random_state=0)
    counts = model.sample(50)
    assert len(counts) == 50
    assert all(c >= 1 for c in counts)
