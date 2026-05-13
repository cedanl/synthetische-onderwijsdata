import pytest
from relsynth.core.topology import SchemaGraph


def test_single_table():
    g = SchemaGraph()
    g.add_table("dim_student")
    assert g.topological_order() == ["dim_student"]


def test_two_level_dependency():
    g = SchemaGraph()
    g.add_dependency("dim_student", "fac_inschrijving")
    g.add_dependency("dim_opleiding", "fac_inschrijving")
    order = g.topological_order()
    assert order.index("dim_student") < order.index("fac_inschrijving")
    assert order.index("dim_opleiding") < order.index("fac_inschrijving")


def test_multi_level_chain():
    g = SchemaGraph()
    g.add_dependency("A", "B")
    g.add_dependency("B", "C")
    order = g.topological_order()
    assert order == ["A", "B", "C"]


def test_cycle_raises():
    g = SchemaGraph()
    g.add_dependency("A", "B")
    g.add_dependency("B", "A")
    with pytest.raises(ValueError, match="cycle"):
        g.topological_order()
