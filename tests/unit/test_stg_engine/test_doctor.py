"""stg doctor — graph hygiene report (components / zero-degree / virtual-only / shell edges / new islands)."""
import time

from stg_engine import STGEngine
from stg_engine.cli import _doctor_report


def _engine():
    e = STGEngine()
    # component 1: A-B-C (real edges with semantic fields)
    e.add_edge("A", "B", role="r1")
    e.add_edge("B", "C", role="r2")
    # component 2: island D-E created "now"
    e.add_edge("D", "E", role="r3")
    # shell edge (no semantic field) inside component 1
    e.add_edge("C", "A", description="no meta field")
    # node F attached only by a virtual edge
    e.add_edge("F", "A", edge_class="virtual", role="xref")
    # zero-degree node G
    e.add_node("G")
    return e


def test_doctor_counts_components_and_residue():
    e = _engine()
    r = _doctor_report(e)
    assert r["components"] >= 2               # {A,B,C} and {D,E} (+ isolated nodes as singletons)
    assert r["largest"] == 3
    assert "g" in r["zero_degree"]
    assert "f" in r["virtual_only"]
    assert len(r["shell_edges"]) == 1


def test_doctor_new_islands_since():
    e = _engine()
    # everything in this fixture was created "now"; a cutoff in the past → islands that contain no older node
    r = _doctor_report(e, since_ts=time.time() - 3600)
    # both real components are "new" and contain no pre-cutoff node → both reported
    assert len(r["new_islands"]) >= 2
    r2 = _doctor_report(e, since_ts=time.time() + 3600)
    assert r2["new_islands"] == []
