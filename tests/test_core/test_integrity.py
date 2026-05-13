import numpy as np
import pytest
from synthetische_onderwijsdata.engine.integrity import IntegrityRegistry


def test_fk_values_within_pk_pool():
    reg = IntegrityRegistry(random_state=42)
    pks = np.arange(1, 11, dtype=np.int64)
    reg.register_pk("dim_student", "studentnummer", pks)

    fks = reg.sample_fk("dim_student", "studentnummer", 500)
    assert set(fks.tolist()).issubset(set(pks.tolist()))


def test_validate_fk():
    reg = IntegrityRegistry()
    reg.register_pk("t", "id", np.array([1, 2, 3]))
    assert reg.validate_fk(np.array([1, 2, 3]), "t", "id")
    assert not reg.validate_fk(np.array([1, 99]), "t", "id")


def test_missing_parent_raises():
    reg = IntegrityRegistry()
    with pytest.raises(KeyError):
        reg.sample_fk("nonexistent", "id", 5)
