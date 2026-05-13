import numpy as np
import pandas as pd
import pytest
from relsynth.core.gcm import GCMEngine


def _correlated_df(rho: float = 0.8, n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = rho * x + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    return pd.DataFrame({"x": x, "y": y})


def test_gcm_preserves_correlation():
    df = _correlated_df(rho=0.8)
    engine = GCMEngine(random_state=0).fit(df)
    synthetic = engine.generate(2000)
    orig_r = float(df.corr().iloc[0, 1])
    synth_r = float(synthetic.corr().iloc[0, 1])
    assert abs(orig_r - synth_r) < 0.1, f"Correlation drift too large: {orig_r:.2f} vs {synth_r:.2f}"


def test_gcm_single_column():
    df = pd.DataFrame({"val": np.random.randn(200)})
    out = GCMEngine(random_state=1).fit(df).generate(100)
    assert list(out.columns) == ["val"]
    assert len(out) == 100


def test_gcm_integer_dtype_preserved():
    df = pd.DataFrame({"n": np.arange(1, 101, dtype=np.int64)})
    out = GCMEngine(random_state=2).fit(df).generate(50)
    assert np.issubdtype(out["n"].dtype, np.integer)


def test_gcm_requires_fit():
    engine = GCMEngine()
    with pytest.raises(RuntimeError, match="fit"):
        engine.generate(10)
