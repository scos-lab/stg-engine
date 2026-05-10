"""Tests for intrinsic-property self-loop semantics (STL Protocol §9.4).

§9.4 contract:
- Self-loops with action="intrinsic_properties" are surface syntax for
  node-level attributes.
- ingest_stl materializes such statements into nodes.metadata_json and
  does NOT create a graph edge.
- The defensive propagate / gravity filter (1A) remains, protecting
  against historical or manually-created edges of this kind.
- CLI _render_node_detail reads node.metadata for a Properties: section.
"""

import pytest
import os
import tempfile

from stg_engine.engine import STGEngine
from stg_engine.gravity import build_gravity_map
from stg_engine.types import INTRINSIC_PROPERTIES_ACTION, STGEdge


def _ingest_intrinsic(engine: STGEngine, name: str, **attrs) -> None:
    """Helper: ingest a self-loop intrinsic-property STL statement."""
    mods = ", ".join(f'{k}="{v}"' for k, v in attrs.items())
    engine.ingest_stl(
        f'[{name}] -> [{name}] ::mod(action="{INTRINSIC_PROPERTIES_ACTION}", '
        f'{mods}, confidence=0.99, rule="definitional")'
    )


# ─── STGEdge.is_intrinsic_property() helper (defensive detection) ──────────
#
# Per §9.4, ingest_stl no longer creates these edges, so most graphs will
# never contain one. The helper still exists for: (1) loading legacy .stg
# files that may have such edges, (2) manually-created edges via add_edge,
# (3) the propagate / gravity filter logic.

def test_is_intrinsic_property_true_for_self_loop_with_action():
    edge = STGEdge(
        source="Foo", target="Foo",
        modifiers={"action": INTRINSIC_PROPERTIES_ACTION, "appid": "123"},
        confidence=0.99,
    )
    assert edge.is_intrinsic_property() is True


def test_is_intrinsic_property_false_for_normal_edge():
    """Even with the magic action value, source != target → not intrinsic."""
    edge = STGEdge(
        source="A", target="B",
        modifiers={"action": INTRINSIC_PROPERTIES_ACTION},
        confidence=0.99,
    )
    assert edge.is_intrinsic_property() is False


def test_is_intrinsic_property_false_for_self_loop_without_action():
    edge = STGEdge(
        source="A", target="A",
        modifiers={"action": "reflects_on_itself"},
        confidence=0.9,
    )
    assert edge.is_intrinsic_property() is False


def test_is_intrinsic_property_self_loop_case_insensitive():
    """Self-loop detection mirrors engine _nk normalization (lower + hyphen→_)."""
    edge = STGEdge(
        source="Elden-Ring",
        target="elden_ring",
        modifiers={"action": INTRINSIC_PROPERTIES_ACTION},
        confidence=0.99,
    )
    assert edge.is_intrinsic_property() is True


# ─── ingest_stl materialization behavior (§9.4) ────────────────────────────

def test_intrinsic_ingest_writes_to_node_metadata():
    e = STGEngine()
    _ingest_intrinsic(e, "Elden_Ring",
                      appid="1245620", release_year="2022", price_usd="59.99")

    node = e._nodes["elden_ring"]
    assert node.metadata["appid"] == "1245620"
    assert node.metadata["release_year"] == "2022"
    assert node.metadata["price_usd"] == "59.99"


def test_intrinsic_ingest_does_not_create_edge():
    """No edge structure should be created — _edges, _edges_lookup, _graph all clean."""
    e = STGEngine()
    _ingest_intrinsic(e, "Elden_Ring", appid="1245620")

    self_loop_edges = [
        ed for ed in e._edges
        if ed.source.lower() == "elden_ring" and ed.target.lower() == "elden_ring"
    ]
    assert self_loop_edges == [], "self-loop edge was created"
    assert ("elden_ring", "elden_ring") not in e._edges_lookup
    assert not e._graph.has_edge("elden_ring", "elden_ring")


def test_intrinsic_ingest_strips_carrier_keys():
    """`action` and `edge_class` are carrier-internal — must not pollute metadata."""
    e = STGEngine()
    _ingest_intrinsic(e, "Foo", real_attr="value")

    node = e._nodes["foo"]
    assert node.metadata.get("real_attr") == "value"
    # Carrier keys must not leak into node attributes
    assert "action" not in node.metadata
    assert "edge_class" not in node.metadata


def test_intrinsic_ingest_merges_on_update():
    """Subsequent ingest of the same node merges new attrs with existing."""
    e = STGEngine()
    _ingest_intrinsic(e, "Foo", appid="123", year="2022")
    _ingest_intrinsic(e, "Foo", price="59.99", year="2023")  # year updated

    node = e._nodes["foo"]
    assert node.metadata["appid"] == "123"           # preserved
    assert node.metadata["price"] == "59.99"         # added
    assert node.metadata["year"] == "2023"           # overwritten


def test_intrinsic_ingest_does_not_affect_normal_edges():
    """A real out-edge in the same ingest batch is created normally."""
    e = STGEngine()
    e.ingest_stl(
        '[Elden_Ring] -> [Elden_Ring] ::mod(action="intrinsic_properties", appid="1245620", confidence=0.99)\n'
        '[Elden_Ring] -> [Souls_Like] ::mod(action="has_tag", confidence=0.95)\n'
    )

    # node has metadata
    assert e._nodes["elden_ring"].metadata.get("appid") == "1245620"
    # but the real edge to Souls_Like exists
    assert ("elden_ring", "souls_like") in e._edges_lookup


# ─── SQLite persistence round-trip ─────────────────────────────────────────

def test_node_metadata_persisted_to_sqlite():
    """Save + load round-trip — metadata survives in nodes.metadata_json column."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.stg")

        e1 = STGEngine()
        _ingest_intrinsic(e1, "Elden_Ring",
                          appid="1245620", release_year="2022")
        e1.save(path)

        e2 = STGEngine.load(path)
        assert e2._nodes["elden_ring"].metadata["appid"] == "1245620"
        assert e2._nodes["elden_ring"].metadata["release_year"] == "2022"


# ─── Defensive propagate / gravity filter (1A behavior preserved) ──────────
#
# These tests construct intrinsic self-loop edges manually (bypassing
# ingest_stl) to verify the engine remains robust against legacy data
# or unusual ingest paths.

def test_propagate_skips_manually_created_intrinsic_edge():
    """Even if a self-loop intrinsic edge is added directly, propagate skips it."""
    e = STGEngine()
    e.add_node("A")
    # Manually create the edge that ingest_stl would not create
    e.add_edge("A", "A", confidence=0.99,
               modifiers={"action": INTRINSIC_PROPERTIES_ACTION, "appid": "X"})
    # Reference baseline — same node, no self-loop
    e2 = STGEngine()
    e2.add_node("A")
    e.propagate("A")
    e2.propagate("A")
    a_with = e._nodes["a"].activation
    a_baseline = e2._nodes["a"].activation
    # Self-loop must not amplify activation
    assert abs(a_with - a_baseline) < 0.01


def test_propagate_normal_path_unaffected_by_intrinsic_metadata():
    """Materialized intrinsic attrs do not interfere with downstream edge propagation."""
    e = STGEngine()
    _ingest_intrinsic(e, "Elden_Ring", appid="1245620", year="2022")
    e.ingest_stl('[Elden_Ring] -> [Souls_Like] ::mod(action="has_tag", confidence=0.95)')
    e.ingest_stl('[Elden_Ring] -> [Action_RPG] ::mod(action="has_tag", confidence=0.95)')

    activated_names = e.propagate("Elden_Ring")
    assert "Souls_Like" in activated_names
    assert "Action_RPG" in activated_names


def test_gravity_unaffected_by_intrinsic_ingest():
    """Materialized intrinsic attrs do not pollute community detection."""
    e = STGEngine()
    e.add_edge("A1", "A2", confidence=0.9)
    e.add_edge("A2", "A3", confidence=0.9)
    e.add_edge("A3", "A1", confidence=0.9)

    _ingest_intrinsic(e, "A1", attr="x")  # adds node metadata, no edge

    gravity = build_gravity_map(e)
    # Communities still detectable, no spurious singletons from a self-loop
    assert gravity.community_counts["medium"] >= 1


# ─── CLI node detail rendering (Properties: section reads metadata) ────────

def test_cli_node_renders_properties_section(capsys):
    """`stg node <name>` shows node.metadata as a Properties: section."""
    from stg_engine.cli import _render_node_detail

    e = STGEngine()
    _ingest_intrinsic(e, "Elden_Ring",
                      appid="1245620", release_year="2022", price_usd="59.99")
    e.ingest_stl('[Elden_Ring] -> [Souls_Like] ::mod(action="has_tag", confidence=0.95)')

    _render_node_detail(e, "Elden_Ring")
    out = capsys.readouterr().out

    assert "Properties:" in out
    assert "appid: 1245620" in out
    assert "release_year: 2022" in out
    assert "price_usd: 59.99" in out


def test_cli_node_no_self_loop_in_outgoing(capsys):
    """Outgoing should only count real edges — intrinsic was never an edge."""
    from stg_engine.cli import _render_node_detail

    e = STGEngine()
    _ingest_intrinsic(e, "Elden_Ring", appid="1245620")
    e.ingest_stl('[Elden_Ring] -> [Souls_Like] ::mod(action="has_tag", confidence=0.95)')
    e.ingest_stl('[Elden_Ring] -> [Action_RPG] ::mod(action="has_tag", confidence=0.95)')

    _render_node_detail(e, "Elden_Ring")
    out = capsys.readouterr().out

    assert "Outgoing (2)" in out
    assert "→ [Elden_Ring]" not in out


def test_cli_node_properties_excludes_carrier_keys(capsys):
    """Properties section displays only user-facing attrs (carrier keys stripped at ingest)."""
    from stg_engine.cli import _render_node_detail

    e = STGEngine()
    _ingest_intrinsic(e, "Foo", visible_attr="should_show")

    _render_node_detail(e, "Foo")
    out = capsys.readouterr().out

    assert "visible_attr: should_show" in out
    # Carrier keys never reached metadata, so they're absent here too
    assert "action: intrinsic_properties" not in out
    assert "edge_class:" not in out


def test_cli_node_no_properties_section_when_metadata_empty(capsys):
    """Nodes with no metadata render no Properties: section."""
    from stg_engine.cli import _render_node_detail

    e = STGEngine()
    e.ingest_stl('[A] -> [B] ::mod(action="related", confidence=0.9)')

    _render_node_detail(e, "A")
    out = capsys.readouterr().out

    assert "Properties:" not in out


# ─── Edge attribute display thresholds (unchanged from acd2118) ────────────

def test_cli_edge_attrs_hidden_when_default(capsys):
    """Default c=0.95, s=0.5, sal≈c values should not render — clean output."""
    from stg_engine.cli import _render_node_detail

    e = STGEngine()
    e.ingest_stl('[A] -> [B] ::mod(action="related", confidence=0.95)')

    _render_node_detail(e, "A")
    out = capsys.readouterr().out

    assert "→ [B]" in out
    assert "c=0.95" not in out
    assert "s=0.5" not in out
    assert "sal=" not in out


def test_cli_edge_attrs_show_low_confidence(capsys):
    """confidence < 0.5 is an outlier — flag it visually."""
    from stg_engine.cli import _render_node_detail

    e = STGEngine()
    e.ingest_stl('[A] -> [B] ::mod(action="guesses", confidence=0.40)')

    _render_node_detail(e, "A")
    out = capsys.readouterr().out

    assert "c=0.4" in out


def test_cli_edge_attrs_hide_moderate_confidence(capsys):
    """confidence in 0.5-0.85 range is normal LLM territory — don't display."""
    from stg_engine.cli import _render_node_detail

    e = STGEngine()
    e.ingest_stl('[A] -> [B] ::mod(action="related", confidence=0.70)')
    e.ingest_stl('[A] -> [C] ::mod(action="related", confidence=0.85)')

    _render_node_detail(e, "A")
    out = capsys.readouterr().out

    assert "c=0.7" not in out
    assert "c=0.85" not in out


def test_cli_edge_attrs_show_nondefault_strength(capsys):
    """strength != 0.5 is non-default — render it."""
    from stg_engine.cli import _render_node_detail

    e = STGEngine()
    e.ingest_stl('[A] -> [B] ::mod(action="triggers", confidence=0.95, rule="causal", strength=0.85)')

    _render_node_detail(e, "A")
    out = capsys.readouterr().out

    assert "s=0.85" in out
    assert "c=0.95" not in out


def test_cli_edge_attrs_show_modified_salience(capsys):
    """Salience moved significantly away from confidence by Hebbian → render."""
    from stg_engine.cli import _render_node_detail

    e = STGEngine()
    e.ingest_stl('[A] -> [B] ::mod(action="related", confidence=0.95)')
    edge = next(ed for ed in e._edges if ed.source == "A" and ed.target == "B")
    edge.salience = 1.50

    _render_node_detail(e, "A")
    out = capsys.readouterr().out

    assert "sal=1.50" in out


def test_cli_edge_attrs_hide_micro_salience_drift(capsys):
    """Background Hebbian micro-adjustment (1-2 strengthen steps) → don't show."""
    from stg_engine.cli import _render_node_detail

    e = STGEngine()
    e.ingest_stl('[A] -> [B] ::mod(action="related", confidence=0.95)')
    edge = next(ed for ed in e._edges if ed.source == "A" and ed.target == "B")
    edge.salience = 0.97  # +0.02 — within tolerance

    _render_node_detail(e, "A")
    out = capsys.readouterr().out

    assert "sal=" not in out


def test_cli_edge_attrs_show_rule(capsys):
    """Rule, when present, is metadata worth keeping visible."""
    from stg_engine.cli import _render_node_detail

    e = STGEngine()
    e.ingest_stl('[A] -> [B] ::mod(action="caused", confidence=0.95, rule="empirical")')

    _render_node_detail(e, "A")
    out = capsys.readouterr().out

    assert 'rule="empirical"' in out


def test_cli_edge_attrs_combined_outliers():
    """Multiple outliers compose into a single parenthetical."""
    from stg_engine.cli import _format_edge_attrs

    edge = STGEdge(
        source="A", target="B",
        confidence=0.4, strength=0.85, salience=1.20,
        rule="causal", modifiers={},
    )
    out = _format_edge_attrs(edge)
    assert out == ' (c=0.4, s=0.85, sal=1.20, rule="causal")'


def test_cli_edge_attrs_no_signals_returns_empty():
    """Edge at all defaults → empty string (nothing to print)."""
    from stg_engine.cli import _format_edge_attrs

    edge = STGEdge(
        source="A", target="B",
        confidence=0.95, strength=0.5, salience=0.95,
        rule=None, modifiers={},
    )
    assert _format_edge_attrs(edge) == ""
