import pandas as pd
import numpy as np
from relsynth.presets.hooks import _apply_rule


def test_greater_than_rule_fixes_violations():
    df = pd.DataFrame({
        "diploma": [100, 50, 200],
        "inschrijf": [80, 60, 150],
    })
    result = _apply_rule(df.copy(), "diploma > inschrijf", condition=None)
    assert (result["diploma"] > result["inschrijf"]).all()


def test_conditional_rule_only_applies_to_non_null():
    df = pd.DataFrame({
        "diploma": [None, 50, 200],
        "inschrijf": [80, 60, 150],
    })
    # Row 0 has None diploma — should not be touched
    result = _apply_rule(df.copy(), "diploma > inschrijf", condition="diploma is not null")
    # Row 1: diploma=50 < inschrijf=60 → should be swapped
    assert result.loc[1, "diploma"] >= result.loc[1, "inschrijf"]
