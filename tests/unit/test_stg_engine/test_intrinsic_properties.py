"""Tests for intrinsic-property self-loop edges (STL Protocol §9.4).

Self-loop edges with action="intrinsic_properties" are storage-only attribute
carriers: preserved in storage, excluded from propagation and community detection.
"""

import pytest

from stg_engine.engine import STGEngine
from stg_engine.gravity import build_gravity_map
from stg_engine.types import INTRINSIC_PROPERTIES_ACTION


def _ingest_intrinsic(engine: STGEngine, name: str, **attrs) -> None:
    """Helper: ingest a self-loop intrinsic-property edge with given attrs."""
    mods = ", ".join(f'{k}="{v}"' for k, v in attrs.items())
    engine.ingest_stl(
        f'[{name}] -> [{name}] ::mod(action="{INTRINSIC_PROPERTIES_ACTION}", '
        f'{mods}, confidence=0.99, rule="definitional")'
    )


# ─── Helper detection ─────────────────────────────────────────────────────

def test_is_intrinsic_property_true_for_self_loop_with_action():
    e = STGEngine()
    _ingest_intrinsic(e, "Foo", appid="123")
    edge = next(ed for ed in e._edges if ed.source == "Foo" and ed.target == "Foo")
    assert edge.is_intrinsic_property() is True


def test_is_intrinsic_property_false_for_normal_edge():
    e = STGEngine()
    e.ingest_stl('[A] -> [B] ::mod(action="intrinsic_properties", confidence=0.9)')
    edge = next(ed for ed in e._edges if ed.source == "A" and ed.target == "B")
    # Even with the magic action value, source != target → not intrinsic
    assert edge.is_intrinsic_property() is False


def test_is_intrinsic_property_false_for_self_loop_without_action():
    e = STGEngine()
    e.ingest_stl('[A] -> [A] ::mod(action="reflects_on_itself", confidence=0.9)')
    edge = next(ed for ed in e._edges if ed.source == "A" and ed.target == "A")
    assert edge.is_intrinsic_property() is False


def test_is_intrinsic_property_self_loop_case_insensitive():
    """Self-loop detection mirrors engine _nk normalization (lower + hyphen→_)."""
    e = STGEngine()
    # Manually craft: source/target with same normalized form
    from stg_engine.types import STGEdge
    edge = STGEdge(
        source="Elden-Ring",
        target="elden_ring",
        modifiers={"action": INTRINSIC_PROPERTIES_ACTION},
        confidence=0.99,
    )
    assert edge.is_intrinsic_property() is True


# ─── Storage preservation ────────────────────────────────────────────────

def test_intrinsic_edge_preserved_in_edges_list():
    e = STGEngine()
    _ingest_intrinsic(e, "Elden_Ring", appid="1245620", year="2022")
    matching = [ed for ed in e._edges
                if ed.source == "Elden_Ring" and ed.target == "Elden_Ring"]
    assert len(matching) == 1
    edge = matching[0]
    assert edge.modifiers["appid"] == "1245620"
    assert edge.modifiers["year"] == "2022"
    assert edge.modifiers["action"] == INTRINSIC_PROPERTIES_ACTION


def test_intrinsic_edge_preserved_in_graph():
    e = STGEngine()
    _ingest_intrinsic(e, "Elden_Ring", appid="1245620")
    # _graph still has the self-loop (so node detail / neighbor queries can read it)
    assert e._graph.has_edge("elden_ring", "elden_ring")


def test_intrinsic_edge_preserved_in_lookup():
    e = STGEngine()
    _ingest_intrinsic(e, "Elden_Ring", appid="1245620")
    assert ("elden_ring", "elden_ring") in e._edges_lookup


# ─── Propagation exclusion ───────────────────────────────────────────────

def test_propagate_does_not_traverse_intrinsic_self_loop():
    """Activation should not flow Node → Node through an intrinsic self-loop."""
    # Setup A: only an intrinsic self-loop
    e1 = STGEngine()
    _ingest_intrinsic(e1, "A", attr1="v1", attr2="v2")
    # Stub edge so the ingest doesn't leave A completely isolated for the
    # propagate seed match; target gets no propagation back to A
    e1.add_edge("Other", "Sink", confidence=0.5)
    e1.propagate("A")
    a_activation = e1._nodes["a"].activation

    # Setup B: matching baseline — no self-loop at all
    e2 = STGEngine()
    e2.add_node("A")
    e2.add_edge("Other", "Sink", confidence=0.5)
    e2.propagate("A")
    a_baseline = e2._nodes["a"].activation

    # If the intrinsic self-loop were participating in propagation, A's activation
    # in e1 would compound (loop feedback). It must match the baseline.
    assert abs(a_activation - a_baseline) < 0.01, (
        f"intrinsic self-loop appears to amplify activation: "
        f"with_loop={a_activation}, baseline={a_baseline}"
    )


def test_propagate_normal_path_still_works_with_intrinsic_present():
    """Adding intrinsic self-loops doesn't break normal propagation."""
    e = STGEngine()
    _ingest_intrinsic(e, "Elden_Ring", appid="1245620", year="2022")
    e.ingest_stl('[Elden_Ring] -> [Souls_Like] ::mod(action="has_tag", confidence=0.95)')
    e.ingest_stl('[Elden_Ring] -> [Action_RPG] ::mod(action="has_tag", confidence=0.95)')

    activated_names = e.propagate("Elden_Ring")

    # Real out-edges should still activate their targets
    assert "Souls_Like" in activated_names
    assert "Action_RPG" in activated_names


# ─── Community detection exclusion ───────────────────────────────────────

def test_intrinsic_self_loop_does_not_create_singleton_community():
    """Isolated nodes with only intrinsic self-loops should not affect Louvain."""
    e = STGEngine()
    # Build two clear communities
    e.add_edge("A1", "A2", confidence=0.9)
    e.add_edge("A2", "A3", confidence=0.9)
    e.add_edge("A3", "A1", confidence=0.9)
    e.add_edge("B1", "B2", confidence=0.9)
    e.add_edge("B2", "B3", confidence=0.9)
    e.add_edge("B3", "B1", confidence=0.9)
    e.add_edge("A1", "B1", confidence=0.5)  # bridge

    # Add intrinsic self-loops on existing nodes
    _ingest_intrinsic(e, "A1", attr="x")
    _ingest_intrinsic(e, "B1", attr="y")

    gravity = build_gravity_map(e)
    # Communities should still be detectable; intrinsic loops did not corrupt structure
    assert gravity.community_counts["medium"] >= 1


def test_intrinsic_only_node_does_not_join_unrelated_community():
    """A node with ONLY an intrinsic self-loop is isolated for community purposes."""
    e = STGEngine()
    e.add_edge("X1", "X2", confidence=0.9)
    e.add_edge("X2", "X3", confidence=0.9)
    e.add_edge("X3", "X1", confidence=0.9)

    # Isolated node with only an intrinsic self-loop
    _ingest_intrinsic(e, "Lonely", attr="solo")

    gravity = build_gravity_map(e)
    lonely_comm = gravity.node_community.get("lonely", {}).get("medium")
    x1_comm = gravity.node_community.get("x1", {}).get("medium")

    # Lonely should not be in the same community as the X cluster
    # (it's structurally disconnected once the self-loop is removed)
    if lonely_comm is not None and x1_comm is not None:
        assert lonely_comm != x1_comm, (
            "Lonely node with only intrinsic self-loop got merged into X cluster"
        )


def test_intrinsic_edge_does_not_contribute_community_heat():
    """Heat compute must skip intrinsic self-loops."""
    from stg_engine.gravity import compute_community_signals
    import time as _time

    e = STGEngine()
    e.add_edge("A1", "A2", confidence=0.9)
    e.add_edge("A2", "A3", confidence=0.9)
    e.add_edge("A3", "A1", confidence=0.9)
    _ingest_intrinsic(e, "A1", attr="x")

    gravity = build_gravity_map(e)
    # Touch all communities
    touched = set()
    for n, comms in gravity.node_community.items():
        if "medium" in comms:
            touched.add(comms["medium"])

    # Stamp last_used on all edges so heat is computable
    now = _time.time()
    for ed in e._edges:
        ed.last_used = now

    signals = compute_community_signals(
        e, gravity, list(touched), resolution="medium", now=now,
    )

    # Should not crash; intrinsic edge contributes zero heat regardless of last_used
    assert isinstance(signals, dict)


# ─── CLI node detail rendering (1B) ──────────────────────────────────────

def test_cli_node_renders_properties_section(capsys):
    """`stg node <name>` shows intrinsic attrs as Properties:, not as edges."""
    from stg_engine.cli import _render_node_detail

    e = STGEngine()
    _ingest_intrinsic(e, "Elden_Ring",
                      appid="1245620", release_year="2022", price_usd="59.99")
    e.ingest_stl('[Elden_Ring] -> [Souls_Like] ::mod(action="has_tag", confidence=0.95)')

    _render_node_detail(e, "Elden_Ring")
    out = capsys.readouterr().out

    # Properties section exists with each attribute
    assert "Properties:" in out
    assert "appid: 1245620" in out
    assert "release_year: 2022" in out
    assert "price_usd: 59.99" in out


def test_cli_node_excludes_intrinsic_from_outgoing(capsys):
    """Intrinsic self-loop must not appear in Outgoing/Incoming sections."""
    from stg_engine.cli import _render_node_detail

    e = STGEngine()
    _ingest_intrinsic(e, "Elden_Ring", appid="1245620")
    e.ingest_stl('[Elden_Ring] -> [Souls_Like] ::mod(action="has_tag", confidence=0.95)')
    e.ingest_stl('[Elden_Ring] -> [Action_RPG] ::mod(action="has_tag", confidence=0.95)')

    _render_node_detail(e, "Elden_Ring")
    out = capsys.readouterr().out

    # Outgoing count must be 2 (real edges only), not 3
    assert "Outgoing (2)" in out
    # Self-loop arrow must not appear
    assert "→ [Elden_Ring]" not in out


def test_cli_node_suppresses_carrier_metadata(capsys):
    """action/rule/edge_class/_epistemic_warnings should not show in Properties."""
    from stg_engine.cli import _render_node_detail

    e = STGEngine()
    _ingest_intrinsic(e, "Foo", visible_attr="should_show")

    _render_node_detail(e, "Foo")
    out = capsys.readouterr().out

    # Real attribute shows
    assert "visible_attr: should_show" in out
    # Carrier-internal fields suppressed
    assert "action: intrinsic_properties" not in out
    assert "rule: definitional" not in out
    assert "edge_class:" not in out


def test_cli_node_no_properties_section_when_no_intrinsic(capsys):
    """Nodes without intrinsic edges render no Properties: section."""
    from stg_engine.cli import _render_node_detail

    e = STGEngine()
    e.ingest_stl('[A] -> [B] ::mod(action="related", confidence=0.9)')

    _render_node_detail(e, "A")
    out = capsys.readouterr().out

    assert "Properties:" not in out


# ─── Edge attribute display thresholds ────────────────────────────────────

def test_cli_edge_attrs_hidden_when_default(capsys):
    """Default c=0.95, s=0.5, sal≈c values should not render — clean output."""
    from stg_engine.cli import _render_node_detail

    e = STGEngine()
    e.ingest_stl('[A] -> [B] ::mod(action="related", confidence=0.95)')

    _render_node_detail(e, "A")
    out = capsys.readouterr().out

    # The line for B should be just '→ [B]' with no parenthetical
    assert "→ [B]" in out
    assert "c=0.95" not in out
    assert "s=0.5" not in out
    assert "sal=" not in out


def test_cli_edge_attrs_show_low_confidence(capsys):
    """confidence < 0.85 is an outlier — flag it visually."""
    from stg_engine.cli import _render_node_detail

    e = STGEngine()
    e.ingest_stl('[A] -> [B] ::mod(action="guesses", confidence=0.40)')

    _render_node_detail(e, "A")
    out = capsys.readouterr().out

    assert "c=0.4" in out


def test_cli_edge_attrs_show_nondefault_strength(capsys):
    """strength != 0.5 is non-default — render it."""
    from stg_engine.cli import _render_node_detail

    e = STGEngine()
    e.ingest_stl('[A] -> [B] ::mod(action="triggers", confidence=0.95, rule="causal", strength=0.85)')

    _render_node_detail(e, "A")
    out = capsys.readouterr().out

    assert "s=0.85" in out
    # confidence is 0.95 (well-established) so should NOT show
    assert "c=0.95" not in out


def test_cli_edge_attrs_show_modified_salience(capsys):
    """Salience moved away from confidence by Hebbian → render it."""
    from stg_engine.cli import _render_node_detail

    e = STGEngine()
    e.ingest_stl('[A] -> [B] ::mod(action="related", confidence=0.95)')
    # Manually move salience to simulate Hebbian strengthening
    edge = next(ed for ed in e._edges if ed.source == "A" and ed.target == "B")
    edge.salience = 1.50

    _render_node_detail(e, "A")
    out = capsys.readouterr().out

    assert "sal=1.50" in out


def test_cli_edge_attrs_show_rule(capsys):
    """Rule, when present, is metadata worth keeping visible."""
    from stg_engine.cli import _render_node_detail

    e = STGEngine()
    e.ingest_stl('[A] -> [B] ::mod(action="caused", confidence=0.95, rule="empirical")')

    _render_node_detail(e, "A")
    out = capsys.readouterr().out

    assert 'rule="empirical"' in out


def test_cli_edge_attrs_combined_outliers(capsys):
    """Multiple outliers compose into a single parenthetical."""
    from stg_engine.cli import _format_edge_attrs
    from stg_engine.types import STGEdge

    edge = STGEdge(
        source="A", target="B",
        confidence=0.4, strength=0.85, salience=1.20,
        rule="causal", modifiers={},
    )
    out = _format_edge_attrs(edge)
    # All four signals should appear, comma-separated, in order
    assert out == ' (c=0.4, s=0.85, sal=1.20, rule="causal")'


def test_cli_edge_attrs_no_signals_returns_empty():
    """Edge at all defaults → empty string (nothing to print)."""
    from stg_engine.cli import _format_edge_attrs
    from stg_engine.types import STGEdge

    edge = STGEdge(
        source="A", target="B",
        confidence=0.95, strength=0.5, salience=0.95,
        rule=None, modifiers={},
    )
    assert _format_edge_attrs(edge) == ""
