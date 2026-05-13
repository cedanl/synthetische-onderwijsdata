import pandas as pd
import numpy as np
from relsynth.logic.dbn import TransitionModel


def _make_sequences(n: int = 20, length: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    seqs = []
    for _ in range(n):
        cat = rng.choice(["A", "B", "C"], size=length).tolist()
        num = rng.uniform(0, 10, size=length).tolist()
        seqs.append(pd.DataFrame({"status": cat, "score": num}))
    return seqs


def test_categorical_transition_stays_in_categories():
    seqs = _make_sequences()
    model = TransitionModel(random_state=0).fit(seqs, ["status"], ["score"])
    state = seqs[0].iloc[0]
    for _ in range(50):
        state = model.step(state)
    assert state["status"] in {"A", "B", "C"}


def test_numeric_transition_produces_finite_values():
    seqs = _make_sequences()
    model = TransitionModel(random_state=0).fit(seqs, ["status"], ["score"])
    state = seqs[0].iloc[0].copy()
    for _ in range(100):
        state = model.step(state)
    assert np.isfinite(float(state["score"]))
