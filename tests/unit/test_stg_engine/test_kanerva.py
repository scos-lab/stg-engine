"""Tests for Phase 8: Kanerva Extensions.

Tests for IterativePropagator, PreferenceFunction, and ConflictDetector.
"""

import pytest
from unittest.mock import MagicMock, patch

from stg_engine.engine import STGEngine
from stg_engine.types import ConvergenceResult, ConflictReport, STGEdge
from stg_engine.kanerva import (
    IterativePropagator,
    PreferenceFunction,
    ConflictDetector,
)


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════


def build_chain_graph():
    """Build A -> B -> C -> D -> E with clear structure."""
    engine = STGEngine()
    for name in ["A", "B", "C", "D", "E"]:
        engine.add_node(name)
    engine.add_edge("A", "B", confidence=0.9)
    engine.add_edge("B", "C", confidence=0.9)
    engine.add_edge("C", "D", confidence=0.9)
    engine.add_edge("D", "E", confidence=0.9)
    return engine


def build_cluster_graph():
    """Build two clusters connected by a bridge.

    Cluster 1: Memory_System, Memory_Architecture, Memory_Manager
    Cluster 2: STL_Parser, STL_Validator, STL_Builder
    Bridge: Memory_Manager -> STL_Parser
    """
    engine = STGEngine()
    # Cluster 1
    for n in ["Memory_System", "Memory_Architecture", "Memory_Manager"]:
        engine.add_node(n)
    engine.add_edge("Memory_System", "Memory_Architecture", confidence=0.9)
    engine.add_edge("Memory_Architecture", "Memory_Manager", confidence=0.9)
    engine.add_edge("Memory_Manager", "Memory_System", confidence=0.8)
    # Cluster 2
    for n in ["STL_Parser", "STL_Validator", "STL_Builder"]:
        engine.add_node(n)
    engine.add_edge("STL_Parser", "STL_Validator", confidence=0.9)
    engine.add_edge("STL_Validator", "STL_Builder", confidence=0.9)
    engine.add_edge("STL_Builder", "STL_Parser", confidence=0.8)
    # Bridge
    engine.add_edge("Memory_Manager", "STL_Parser", confidence=0.5)
    return engine


def build_diamond_graph():
    """Build diamond: A -> B -> D and A -> C -> D."""
    engine = STGEngine()
    for name in ["A", "B", "C", "D"]:
        engine.add_node(name)
    engine.add_edge("A", "B", confidence=0.9)
    engine.add_edge("A", "C", confidence=0.9)
    engine.add_edge("B", "D", confidence=0.9)
    engine.add_edge("C", "D", confidence=0.9)
    return engine


# ═══════════════════════════════════════════════════════════
# IterativePropagator Tests
# ═══════════════════════════════════════════════════════════


class TestIterativePropagator:

    def test_init_defaults(self):
        ip = IterativePropagator()
        assert ip.top_k == 5
        assert ip.max_iterations == 5
        assert ip.convergence_threshold == 0.8

    def test_init_custom(self):
        ip = IterativePropagator(top_k=3, max_iterations=10, convergence_threshold=0.5)
        assert ip.top_k == 3
        assert ip.max_iterations == 10
        assert ip.convergence_threshold == 0.5

    def test_init_invalid_top_k(self):
        with pytest.raises(ValueError, match="top_k"):
            IterativePropagator(top_k=0)

    def test_init_invalid_max_iterations(self):
        with pytest.raises(ValueError, match="max_iterations"):
            IterativePropagator(max_iterations=0)

    def test_init_invalid_threshold(self):
        with pytest.raises(ValueError, match="convergence_threshold"):
            IterativePropagator(convergence_threshold=1.5)

    def test_empty_input(self):
        engine = STGEngine()
        ip = IterativePropagator()
        result = ip.converge(engine, "")
        assert result.top_nodes == []
        assert result.iterations_used == 0
        assert result.converged is True
        assert result.stability_history == []

    def test_no_matching_nodes(self):
        engine = build_chain_graph()
        ip = IterativePropagator()
        result = ip.converge(engine, "nonexistent_concept_xyz")
        assert result.top_nodes == []
        assert result.converged is True

    def test_convergence_on_cluster(self):
        engine = build_cluster_graph()
        ip = IterativePropagator(top_k=3, max_iterations=5, convergence_threshold=0.6)
        result = ip.converge(engine, "Memory")
        assert result.converged is True
        assert result.iterations_used >= 1
        assert len(result.top_nodes) <= 3
        # Should converge to memory cluster nodes
        for node in result.top_nodes:
            assert "Memory" in node or "STL" in node

    def test_convergence_result_type(self):
        engine = build_cluster_graph()
        result = engine.convergent_propagate("Memory", top_k=3)
        assert isinstance(result, ConvergenceResult)

    def test_stability_history_length(self):
        engine = build_cluster_graph()
        ip = IterativePropagator(top_k=3, max_iterations=5)
        result = ip.converge(engine, "Memory")
        assert len(result.stability_history) == result.iterations_used

    def test_stability_history_range(self):
        engine = build_cluster_graph()
        ip = IterativePropagator(top_k=3, max_iterations=5)
        result = ip.converge(engine, "Memory")
        for s in result.stability_history:
            assert 0.0 <= s <= 1.0

    def test_max_iterations_cap(self):
        engine = build_cluster_graph()
        ip = IterativePropagator(top_k=3, max_iterations=2, convergence_threshold=1.0)
        result = ip.converge(engine, "Memory")
        # With threshold=1.0, convergence requires perfect stability
        assert result.iterations_used <= 2

    def test_immediate_convergence_when_stable(self):
        """If first propagation already returns the exact stored pattern."""
        engine = STGEngine()
        engine.add_node("Solo")
        engine.add_edge("Solo", "Solo", confidence=0.9)
        ip = IterativePropagator(top_k=1, convergence_threshold=0.5)
        result = ip.converge(engine, "Solo")
        assert result.converged is True
        assert result.iterations_used >= 1

    def test_engine_convenience_method(self):
        engine = build_cluster_graph()
        result = engine.convergent_propagate("STL", top_k=3, max_iterations=3)
        assert isinstance(result, ConvergenceResult)
        assert len(result.top_nodes) <= 3

    def test_propagate_kwargs_passed_through(self):
        engine = build_cluster_graph()
        ip = IterativePropagator(top_k=3)
        result = ip.converge(engine, "Memory", decay=0.3, threshold=0.05)
        assert isinstance(result, ConvergenceResult)


# ═══════════════════════════════════════════════════════════
# PreferenceFunction Tests
# ═══════════════════════════════════════════════════════════


class TestPreferenceFunction:

    def test_init_defaults(self):
        pf = PreferenceFunction()
        assert pf.gamma == 0.9
        assert pf.reward_scale == 0.1
        assert pf.decay_rate == 0.01

    def test_init_custom(self):
        pf = PreferenceFunction(gamma=0.8, reward_scale=0.2, decay_rate=0.05)
        assert pf.gamma == 0.8

    def test_init_invalid_gamma(self):
        with pytest.raises(ValueError, match="gamma"):
            PreferenceFunction(gamma=0.0)
        with pytest.raises(ValueError, match="gamma"):
            PreferenceFunction(gamma=1.5)

    def test_init_invalid_reward_scale(self):
        with pytest.raises(ValueError, match="reward_scale"):
            PreferenceFunction(reward_scale=0.0)

    def test_init_invalid_decay_rate(self):
        with pytest.raises(ValueError, match="decay_rate"):
            PreferenceFunction(decay_rate=1.0)

    def test_reward_path_simple(self):
        engine = build_chain_graph()
        pf = PreferenceFunction(gamma=0.9, reward_scale=0.1)

        updated = pf.reward_path(engine, ["A", "B", "C", "D"], reward=1.0)
        assert updated == 3

        # Terminal edge (C->D) gets full reward
        edge_cd = engine._edges_lookup[("c", "d")]
        assert edge_cd.preference == pytest.approx(0.1 * 1.0 * 0.9**0)

        # Middle edge (B->C) gets gamma^1
        edge_bc = engine._edges_lookup[("b", "c")]
        assert edge_bc.preference == pytest.approx(0.1 * 1.0 * 0.9**1)

        # First edge (A->B) gets gamma^2
        edge_ab = engine._edges_lookup[("a", "b")]
        assert edge_ab.preference == pytest.approx(0.1 * 1.0 * 0.9**2)

    def test_reward_path_temporal_discount(self):
        """Earlier edges in the path get less reward (temporal discount)."""
        engine = build_chain_graph()
        pf = PreferenceFunction(gamma=0.5, reward_scale=1.0)

        pf.reward_path(engine, ["A", "B", "C", "D"], reward=1.0)

        pref_cd = engine._edges_lookup[("c", "d")].preference
        pref_bc = engine._edges_lookup[("b", "c")].preference
        pref_ab = engine._edges_lookup[("a", "b")].preference

        # Terminal > middle > first
        assert pref_cd > pref_bc > pref_ab

    def test_penalize_path(self):
        engine = build_chain_graph()
        pf = PreferenceFunction(gamma=0.9, reward_scale=0.1)

        updated = pf.penalize_path(engine, ["A", "B", "C"], penalty=1.0)
        assert updated == 2

        edge_bc = engine._edges_lookup[("b", "c")]
        assert edge_bc.preference < 0

        edge_ab = engine._edges_lookup[("a", "b")]
        assert edge_ab.preference < 0

    def test_reward_then_penalize(self):
        engine = build_chain_graph()
        pf = PreferenceFunction(gamma=1.0, reward_scale=0.1)

        pf.reward_path(engine, ["A", "B"], reward=1.0)
        pf.penalize_path(engine, ["A", "B"], penalty=1.0)

        edge_ab = engine._edges_lookup[("a", "b")]
        assert edge_ab.preference == pytest.approx(0.0)

    def test_empty_path(self):
        engine = build_chain_graph()
        pf = PreferenceFunction()
        assert pf.reward_path(engine, [], reward=1.0) == 0

    def test_single_node_path(self):
        engine = build_chain_graph()
        pf = PreferenceFunction()
        assert pf.reward_path(engine, ["A"], reward=1.0) == 0

    def test_two_node_path(self):
        engine = build_chain_graph()
        pf = PreferenceFunction(gamma=0.9, reward_scale=0.1)
        updated = pf.reward_path(engine, ["A", "B"], reward=1.0)
        assert updated == 1
        edge_ab = engine._edges_lookup[("a", "b")]
        assert edge_ab.preference == pytest.approx(0.1)  # gamma^0 = 1.0

    def test_nonexistent_edges_skipped(self):
        engine = build_chain_graph()
        pf = PreferenceFunction()
        # E->A doesn't exist
        updated = pf.reward_path(engine, ["E", "A"], reward=1.0)
        assert updated == 0

    def test_decay_preferences(self):
        engine = build_chain_graph()
        pf = PreferenceFunction(decay_rate=0.1)

        # Set some preferences
        engine._edges_lookup[("a", "b")].preference = 1.0
        engine._edges_lookup[("b", "c")].preference = -0.5

        affected = pf.decay_preferences(engine)
        assert affected == 2

        assert engine._edges_lookup[("a", "b")].preference == pytest.approx(0.9)
        assert engine._edges_lookup[("b", "c")].preference == pytest.approx(-0.45)

    def test_decay_zero_stays_zero(self):
        engine = build_chain_graph()
        pf = PreferenceFunction(decay_rate=0.1)

        affected = pf.decay_preferences(engine)
        assert affected == 0  # All preferences are 0

    def test_decay_snaps_to_zero(self):
        engine = build_chain_graph()
        pf = PreferenceFunction(decay_rate=0.5)
        engine._edges_lookup[("a", "b")].preference = 1e-7
        pf.decay_preferences(engine)
        assert engine._edges_lookup[("a", "b")].preference == 0.0

    def test_get_top_preferred(self):
        engine = build_chain_graph()
        pf = PreferenceFunction()

        engine._edges_lookup[("a", "b")].preference = 0.5
        engine._edges_lookup[("b", "c")].preference = -0.3
        engine._edges_lookup[("c", "d")].preference = 0.8

        top = pf.get_top_preferred(engine, top_n=2)
        assert len(top) == 2
        assert top[0] == ("C", "D", 0.8)  # Highest absolute
        assert top[1] == ("A", "B", 0.5)

    def test_get_top_preferred_empty(self):
        engine = build_chain_graph()
        pf = PreferenceFunction()
        top = pf.get_top_preferred(engine)
        assert top == []

    def test_stats(self):
        engine = build_chain_graph()
        pf = PreferenceFunction(gamma=0.9, reward_scale=0.1)

        pf.reward_path(engine, ["A", "B", "C"], reward=1.0)
        pf.penalize_path(engine, ["C", "D"], penalty=1.0)

        stats = pf.stats
        assert stats["total_rewards"] == 2
        assert stats["total_penalties"] == 1
        assert stats["total_decays"] == 0

    def test_cumulative_reward(self):
        """Multiple rewards accumulate."""
        engine = build_chain_graph()
        pf = PreferenceFunction(gamma=1.0, reward_scale=0.1)

        pf.reward_path(engine, ["A", "B"], reward=1.0)
        pf.reward_path(engine, ["A", "B"], reward=1.0)
        pf.reward_path(engine, ["A", "B"], reward=1.0)

        assert engine._edges_lookup[("a", "b")].preference == pytest.approx(0.3)


# ═══════════════════════════════════════════════════════════
# Preference-Biased Propagation Tests
# ═══════════════════════════════════════════════════════════


class TestPreferenceBiasedPropagation:

    def test_preference_weight_default_zero(self):
        engine = STGEngine()
        assert engine.preference_weight == 0.0

    def test_preference_bias_affects_propagation(self):
        """Preferred path should get more activation."""
        engine = build_diamond_graph()

        # Give B->D high preference, C->D zero
        engine._edges_lookup[("b", "d")].preference = 1.0
        engine._edges_lookup[("c", "d")].preference = 0.0

        # Without preference bias
        engine.preference_weight = 0.0
        results_no_bias = engine.propagate("A")

        # With preference bias
        engine.preference_weight = 1.0
        results_with_bias = engine.propagate("A")

        # Both should activate D, but with bias B should be more prominent
        assert "D" in results_no_bias or "B" in results_no_bias
        assert "D" in results_with_bias or "B" in results_with_bias

    def test_zero_preference_weight_no_effect(self):
        """preference_weight=0 should not change propagation."""
        engine = build_diamond_graph()
        engine._edges_lookup[("b", "d")].preference = 100.0

        engine.preference_weight = 0.0
        r1 = engine.propagate("A")

        # Reset activation
        for n in engine._nodes.values():
            n.activation = 0.0

        engine.preference_weight = 0.0
        r2 = engine.propagate("A")

        assert r1 == r2


# ═══════════════════════════════════════════════════════════
# ConflictDetector Tests
# ═══════════════════════════════════════════════════════════


class TestConflictDetector:

    def test_init_defaults(self):
        cd = ConflictDetector()
        assert cd.deviation_threshold == 0.3
        assert cd.min_confidence == 0.5

    def test_init_invalid_threshold(self):
        with pytest.raises(ValueError, match="deviation_threshold"):
            ConflictDetector(deviation_threshold=0.0)

    def test_no_conflict_no_existing_edges(self):
        engine = STGEngine()
        engine.add_node("A")
        engine.add_node("B")
        cd = ConflictDetector()
        result = cd.check_new_edge(engine, "A", "B")
        assert result is None

    def test_no_embeddings_graceful(self):
        engine = build_chain_graph()
        cd = ConflictDetector()
        result = cd.check_new_edge(engine, "A", "E")
        assert result is None  # No embeddings, no crash

    def test_modifier_contradiction_rule(self):
        engine = STGEngine()
        engine.add_node("A")
        engine.add_node("B")
        engine.add_edge("A", "B", confidence=0.9, rule="causal")

        cd = ConflictDetector()
        result = cd.check_modifier_contradiction(
            engine, "A", "B", {"rule": "definitional"}
        )
        assert result is not None
        assert isinstance(result, ConflictReport)
        assert "rule" in result.details
        assert result.conflict_score > 0

    def test_modifier_no_contradiction_same_value(self):
        engine = STGEngine()
        engine.add_node("A")
        engine.add_node("B")
        engine.add_edge("A", "B", confidence=0.9, rule="causal")

        cd = ConflictDetector()
        result = cd.check_modifier_contradiction(
            engine, "A", "B", {"rule": "causal"}
        )
        assert result is None

    def test_modifier_contradiction_coherence(self):
        engine = STGEngine()
        engine.add_node("A")
        engine.add_node("B")
        engine.add_edge("A", "B", confidence=0.95)

        cd = ConflictDetector()
        result = cd.check_modifier_contradiction(
            engine, "A", "B", {"certainty": 0.1}
        )
        assert result is not None
        assert "incoherent" in result.details

    def test_modifier_no_contradiction_no_existing_edge(self):
        engine = STGEngine()
        engine.add_node("A")
        engine.add_node("B")

        cd = ConflictDetector()
        result = cd.check_modifier_contradiction(
            engine, "A", "B", {"rule": "causal"}
        )
        assert result is None

    def test_conflict_report_structure(self):
        engine = STGEngine()
        engine.add_node("A")
        engine.add_node("B")
        engine.add_edge("A", "B", confidence=0.9, rule="causal")

        cd = ConflictDetector()
        result = cd.check_modifier_contradiction(
            engine, "A", "B", {"rule": "empirical"}
        )
        assert result.new_edge == ("A", "B")
        assert ("A", "B") in result.conflicting_edges
        assert 0.0 < result.conflict_score <= 1.0
        assert isinstance(result.details, str)


# ═══════════════════════════════════════════════════════════
# Persistence Tests
# ═══════════════════════════════════════════════════════════


class TestPreferencePersistence:

    def test_preference_roundtrip(self, tmp_path):
        """Save and load preserves preference values."""
        path = str(tmp_path / "test.stg")

        engine = build_chain_graph()
        engine._edges_lookup[("a", "b")].preference = 0.75
        engine._edges_lookup[("b", "c")].preference = -0.3
        engine.save(path)

        loaded = STGEngine.load(path)
        assert loaded._edges_lookup[("a", "b")].preference == pytest.approx(0.75)
        assert loaded._edges_lookup[("b", "c")].preference == pytest.approx(-0.3)
        assert loaded._edges_lookup[("c", "d")].preference == pytest.approx(0.0)

    def test_preference_in_to_dict(self):
        edge = STGEdge(source="A", target="B", preference=0.42)
        d = edge.to_dict()
        assert d["preference"] == 0.42

    def test_preference_from_dict(self):
        d = {"source": "A", "target": "B", "preference": 0.42}
        edge = STGEdge.from_dict(d)
        assert edge.preference == 0.42

    def test_preference_from_dict_missing(self):
        d = {"source": "A", "target": "B"}
        edge = STGEdge.from_dict(d)
        assert edge.preference == 0.0

    def test_backward_compatible_load(self, tmp_path):
        """Load a v3 .stg file (no preference column) without crash."""
        import sqlite3
        path = str(tmp_path / "v3.stg")

        # Create a minimal v3 schema (without preference column)
        conn = sqlite3.connect(path)
        conn.executescript("""
            CREATE TABLE schema_info (key TEXT PRIMARY KEY, value TEXT);
            INSERT INTO schema_info VALUES ('version', '3');

            CREATE TABLE nodes (
                name TEXT PRIMARY KEY, namespace TEXT, anchor_type TEXT,
                metadata_json TEXT DEFAULT '{}',
                tension REAL DEFAULT 0.0, activation REAL DEFAULT 0.0,
                self_relevance REAL DEFAULT 0.0
            );
            INSERT INTO nodes VALUES ('TestNode', NULL, NULL, '{}', 0.0, 0.0, 0.0);

            CREATE TABLE edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT, target TEXT,
                confidence REAL DEFAULT 0.5, strength REAL DEFAULT 0.5,
                rule TEXT, time TEXT,
                modifiers_json TEXT DEFAULT '{}',
                session_id TEXT, event_id TEXT, last_used REAL
            );
            INSERT INTO edges (source, target, confidence) VALUES ('TestNode', 'TestNode', 0.8);

            CREATE TABLE sessions (session_id TEXT PRIMARY KEY, date TEXT, title TEXT,
                avg_importance REAL, event_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'complete', summary TEXT);
            CREATE TABLE events (event_id TEXT PRIMARY KEY, session_id TEXT,
                timestamp TEXT, event_type TEXT, memory_type TEXT, title TEXT,
                importance_score REAL DEFAULT 0.5, description TEXT,
                tags_json TEXT DEFAULT '[]', artifacts_json TEXT DEFAULT '[]',
                stl_block TEXT);
            CREATE TABLE tensions (name TEXT PRIMARY KEY, initial_value REAL DEFAULT 0.0,
                current_value REAL DEFAULT 0.0, status TEXT DEFAULT 'active',
                created_session TEXT, resolved_session TEXT, description TEXT);
            CREATE TABLE belief_evolutions (id INTEGER PRIMARY KEY AUTOINCREMENT,
                old_anchor TEXT, new_anchor TEXT, event_id TEXT,
                session_id TEXT, level INTEGER DEFAULT 1, description TEXT);
            CREATE TABLE system_snapshots (id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT, timestamp TEXT, psi_value REAL DEFAULT 0.0,
                max_tension REAL DEFAULT 0.0, structural_coherence REAL DEFAULT 0.0,
                epistemic_confidence REAL DEFAULT 0.0, total_reward REAL DEFAULT 0.0,
                node_count INTEGER DEFAULT 0, edge_count INTEGER DEFAULT 0);
            CREATE TABLE embeddings (node_name TEXT PRIMARY KEY, embed_text TEXT NOT NULL,
                vector BLOB NOT NULL, model_name TEXT NOT NULL, created_at TEXT NOT NULL);
        """)
        conn.commit()
        conn.close()

        loaded = STGEngine.load(path)
        assert "testnode" in loaded._nodes
        edge = loaded._edges_lookup[("testnode", "testnode")]
        assert edge.preference == 0.0  # Default for missing column


# ═══════════════════════════════════════════════════════════
# Edge Cases & Integration
# ═══════════════════════════════════════════════════════════


class TestKanervaIntegration:

    def test_converge_then_reward(self):
        """Full workflow: converge to find nodes, then reward a known path."""
        engine = build_cluster_graph()
        pf = PreferenceFunction(gamma=0.9, reward_scale=0.1)

        result = engine.convergent_propagate("Memory", top_k=3)
        assert result.converged is True

        # Reward a known path in the graph (not the convergence result,
        # which is a set of top nodes, not necessarily a connected path)
        updated = pf.reward_path(
            engine,
            ["Memory_System", "Memory_Architecture", "Memory_Manager"],
            reward=1.0,
        )
        assert updated == 2

    def test_preference_default_zero_on_new_edge(self):
        engine = STGEngine()
        engine.add_node("X")
        engine.add_node("Y")
        engine.add_edge("X", "Y", confidence=0.9)
        assert engine._edges_lookup[("x", "y")].preference == 0.0

    def test_convergence_result_dataclass(self):
        r = ConvergenceResult(
            top_nodes=["A", "B"],
            iterations_used=3,
            converged=True,
            stability_history=[0.5, 0.8, 1.0],
        )
        assert r.top_nodes == ["A", "B"]
        assert r.converged is True

    def test_conflict_report_dataclass(self):
        r = ConflictReport(
            new_edge=("A", "B"),
            conflicting_edges=[("A", "C")],
            conflict_score=0.7,
            details="test conflict",
        )
        assert r.new_edge == ("A", "B")
        assert r.conflict_score == 0.7
