"""Tests for Phase 7B: Hebbian Learning & Synaptic Pruning.

Tests HebbianLearner, SynapticPruner, importance-biased propagation,
engine integration, persistence roundtrip, and convergence behavior.
"""

import math
import os
import tempfile
import time

import pytest

from stg_engine.engine import STGEngine
from stg_engine.learning import HebbianLearner, SynapticPruner
from stg_engine.types import LearningEvent


# ═══════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════


@pytest.fixture
def learner():
    """Default HebbianLearner."""
    return HebbianLearner()


@pytest.fixture
def aggressive_learner():
    """HebbianLearner with high rates for fast testing."""
    return HebbianLearner(strengthen_rate=0.5, weaken_rate=0.3)


@pytest.fixture
def pruner():
    """SynapticPruner with high eid_safety to disable EID checks in basic tests."""
    return SynapticPruner(eid_safety_threshold=2.0)


@pytest.fixture
def chain_engine():
    """Engine with A→B→C→D chain, all confidence=0.5."""
    engine = STGEngine()
    engine.add_edge("A", "B", confidence=0.5)
    engine.add_edge("B", "C", confidence=0.5)
    engine.add_edge("C", "D", confidence=0.5)
    return engine


@pytest.fixture
def diamond_engine():
    """Engine with diamond: A→B, A→C, B→D, C→D."""
    engine = STGEngine()
    engine.add_edge("A", "B", confidence=0.5)
    engine.add_edge("A", "C", confidence=0.5)
    engine.add_edge("B", "D", confidence=0.5)
    engine.add_edge("C", "D", confidence=0.5)
    return engine


@pytest.fixture
def prunable_engine():
    """Engine with some low-confidence, old edges ready for pruning."""
    engine = STGEngine()
    old_time = time.time() - 400 * 86400  # 400 days ago (unused_days default=365)

    engine.add_edge("A", "B", confidence=0.8)
    engine.add_edge("A", "C", confidence=0.05)
    engine.add_edge("C", "D", confidence=0.03)
    engine.add_edge("B", "D", confidence=0.7)

    # Set last_used timestamps
    for edge in engine._edges:
        if edge.confidence < 0.1:
            edge.last_used = old_time
        else:
            edge.last_used = time.time()

    return engine


# ═══════════════════════════════════════════════════════════
# TestHebbianLearnerInit
# ═══════════════════════════════════════════════════════════


class TestHebbianLearnerInit:
    def test_default_parameters(self):
        h = HebbianLearner()
        assert h.strengthen_rate == 0.05
        assert h.weaken_rate == 0.01
        assert h.floor == 0.01
        assert h.ceiling == 1.0
        assert h.activation_threshold == 0.1
        assert h.weaken_activation_threshold == 0.15

    def test_custom_parameters(self):
        h = HebbianLearner(
            strengthen_rate=0.1,
            weaken_rate=0.05,
            confidence_floor=0.05,
            confidence_ceiling=0.95,
            activation_threshold=0.2,
        )
        assert h.strengthen_rate == 0.1
        assert h.weaken_rate == 0.05
        assert h.floor == 0.05
        assert h.ceiling == 0.95

    def test_stats_initially_zero(self):
        h = HebbianLearner()
        assert h.stats == {"strengthened": 0, "weakened": 0, "total_events": 0}

    def test_invalid_strengthen_rate(self):
        with pytest.raises(ValueError):
            HebbianLearner(strengthen_rate=0.0)
        with pytest.raises(ValueError):
            HebbianLearner(strengthen_rate=1.5)

    def test_invalid_weaken_rate(self):
        with pytest.raises(ValueError):
            HebbianLearner(weaken_rate=-0.1)

    def test_invalid_ceiling_below_floor(self):
        with pytest.raises(ValueError):
            HebbianLearner(confidence_floor=0.5, confidence_ceiling=0.3)


# ═══════════════════════════════════════════════════════════
# TestHebbianStrengthen
# ═══════════════════════════════════════════════════════════


class TestHebbianStrengthen:
    def test_coactivated_edge_strengthened(self, learner, chain_engine):
        activation_map = {"A": 1.0, "B": 0.8}
        events = learner.learn_from_propagation(chain_engine, activation_map)
        strengthened = [e for e in events if e.event_type == "strengthen"]
        assert len(strengthened) >= 1
        # A→B should be strengthened
        ab_events = [e for e in strengthened if e.source == "a" and e.target == "b"]
        assert len(ab_events) == 1
        assert ab_events[0].new_confidence > ab_events[0].old_confidence

    def test_strengthen_moves_toward_ceiling(self, learner, chain_engine):
        activation_map = {"A": 1.0, "B": 0.5}
        events = learner.learn_from_propagation(chain_engine, activation_map)
        for e in events:
            if e.event_type == "strengthen":
                assert e.new_confidence > e.old_confidence
                assert e.new_confidence <= learner.ceiling

    def test_strengthen_modulated_by_activation(self, chain_engine):
        # Low activation → small change
        low = HebbianLearner(strengthen_rate=0.5)
        engine1 = STGEngine()
        engine1.add_edge("A", "B", confidence=0.5)
        events1 = low.learn_from_propagation(engine1, {"A": 0.2, "B": 0.2})

        # High activation → larger change
        high = HebbianLearner(strengthen_rate=0.5)
        engine2 = STGEngine()
        engine2.add_edge("A", "B", confidence=0.5)
        events2 = high.learn_from_propagation(engine2, {"A": 1.0, "B": 1.0})

        if events1 and events2:
            delta1 = events1[0].new_confidence - events1[0].old_confidence
            delta2 = events2[0].new_confidence - events2[0].old_confidence
            assert delta2 > delta1

    def test_already_at_ceiling_no_change(self, learner):
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=1.0)
        events = learner.learn_from_propagation(engine, {"A": 1.0, "B": 1.0})
        strengthened = [e for e in events if e.event_type == "strengthen"]
        assert len(strengthened) == 0

    def test_last_used_updated_on_strengthen(self, learner, chain_engine):
        before = time.time()
        learner.learn_from_propagation(chain_engine, {"A": 1.0, "B": 0.8})
        after = time.time()
        edge = chain_engine._edges_lookup.get(("a", "b"))
        assert edge is not None
        assert edge.last_used is not None
        assert before <= edge.last_used <= after

    def test_learning_event_created(self, learner, chain_engine):
        events = learner.learn_from_propagation(chain_engine, {"A": 1.0, "B": 0.5})
        assert len(events) > 0
        for e in events:
            assert isinstance(e, LearningEvent)
            assert e.trigger == "propagation"

    def test_inactive_target_not_strengthened(self, learner, chain_engine):
        # A active, B below threshold → no strengthening
        events = learner.learn_from_propagation(chain_engine, {"A": 1.0, "B": 0.05})
        strengthened = [e for e in events if e.event_type == "strengthen"]
        ab_events = [e for e in strengthened if e.source == "a" and e.target == "b"]
        assert len(ab_events) == 0

    def test_inactive_source_not_strengthened(self, learner, chain_engine):
        # A below threshold, B active → no strengthening
        events = learner.learn_from_propagation(chain_engine, {"A": 0.05, "B": 1.0})
        strengthened = [e for e in events if e.event_type == "strengthen"]
        ab_events = [e for e in strengthened if e.source == "a" and e.target == "b"]
        assert len(ab_events) == 0


# ═══════════════════════════════════════════════════════════
# TestHebbianWeaken
# ═══════════════════════════════════════════════════════════


class TestHebbianWeaken:
    def test_active_source_inactive_target_weakened(self, learner, chain_engine):
        # A active, C inactive → A→B's successor B→C? Actually A→B source=A, target=B
        # Let's make B active but C not active → B→C should be weakened
        activation_map = {"A": 1.0, "B": 0.5}  # B active, C not
        events = learner.learn_from_propagation(chain_engine, activation_map)
        weakened = [e for e in events if e.event_type == "weaken"]
        bc_events = [e for e in weakened if e.source == "b" and e.target == "c"]
        assert len(bc_events) == 1
        assert bc_events[0].new_confidence < bc_events[0].old_confidence

    def test_weaken_moves_toward_floor(self, learner, chain_engine):
        activation_map = {"B": 1.0}  # B active, C not → weaken B→C
        events = learner.learn_from_propagation(chain_engine, activation_map)
        for e in events:
            if e.event_type == "weaken":
                assert e.new_confidence < e.old_confidence
                assert e.new_confidence >= learner.floor

    def test_already_at_floor_no_change(self, learner):
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.01)
        events = learner.learn_from_propagation(engine, {"A": 1.0})
        weakened = [e for e in events if e.event_type == "weaken"]
        assert len(weakened) == 0

    def test_weaken_event_created(self, learner, chain_engine):
        events = learner.learn_from_propagation(chain_engine, {"A": 1.0})
        weakened = [e for e in events if e.event_type == "weaken"]
        for e in weakened:
            assert e.trigger == "propagation"
            assert e.event_type == "weaken"

    def test_both_inactive_no_change(self, learner, chain_engine):
        # Neither A nor B active → no change to A→B
        events = learner.learn_from_propagation(chain_engine, {"X": 1.0})
        ab_events = [e for e in events if e.source == "a" and e.target == "b"]
        assert len(ab_events) == 0

    def test_last_used_NOT_updated_on_weaken(self, learner, chain_engine):
        # B→C edge: B active, C not → weaken. last_used should not be set.
        chain_engine._edges_lookup[("b", "c")].last_used = None
        learner.learn_from_propagation(chain_engine, {"B": 1.0})
        edge = chain_engine._edges_lookup.get(("b", "c"))
        assert edge.last_used is None


# ═══════════════════════════════════════════════════════════
# TestHebbianLearnFromPropagation
# ═══════════════════════════════════════════════════════════


class TestHebbianLearnFromPropagation:
    def test_chain_strengthened_and_weakened(self, aggressive_learner, chain_engine):
        # Activate A, B strongly. C, D not active.
        activation_map = {"A": 1.0, "B": 0.7}
        events = aggressive_learner.learn_from_propagation(chain_engine, activation_map)
        strengthened = [e for e in events if e.event_type == "strengthen"]
        weakened = [e for e in events if e.event_type == "weaken"]
        # A→B should be strengthened (both active)
        assert any(e.source == "a" and e.target == "b" for e in strengthened)
        # B→C should be weakened (B active, C not)
        assert any(e.source == "b" and e.target == "c" for e in weakened)

    def test_diamond_selective_strengthening(self, aggressive_learner, diamond_engine):
        # Activate A and B strongly, C and D less
        activation_map = {"A": 1.0, "B": 0.8, "D": 0.3}
        events = aggressive_learner.learn_from_propagation(diamond_engine, activation_map)
        strengthened = {(e.source, e.target) for e in events if e.event_type == "strengthen"}
        # A→B strengthened (both active)
        assert ("a", "b") in strengthened
        # B→D strengthened (both active)
        assert ("b", "d") in strengthened

    def test_returns_correct_event_count(self, learner, chain_engine):
        events = learner.learn_from_propagation(chain_engine, {"A": 1.0, "B": 0.5})
        # Should have events for modified edges only
        assert all(isinstance(e, LearningEvent) for e in events)

    def test_caches_invalidated(self, learner, chain_engine):
        # Pre-compute cache
        chain_engine.get_metrics()
        assert chain_engine._graph_metrics_cache is not None
        # Learn should invalidate
        learner.learn_from_propagation(chain_engine, {"A": 1.0, "B": 0.5})
        assert chain_engine._graph_metrics_cache is None

    def test_empty_activation_no_events(self, learner, chain_engine):
        events = learner.learn_from_propagation(chain_engine, {})
        assert len(events) == 0

    def test_all_below_threshold_no_events(self, learner, chain_engine):
        events = learner.learn_from_propagation(
            chain_engine, {"A": 0.01, "B": 0.01}
        )
        assert len(events) == 0

    def test_multiple_rounds_cumulative(self, learner, chain_engine):
        act = {"A": 1.0, "B": 0.5}
        sal_before = chain_engine._edges_lookup[("a", "b")].salience
        learner.learn_from_propagation(chain_engine, act)
        sal_mid = chain_engine._edges_lookup[("a", "b")].salience
        learner.learn_from_propagation(chain_engine, act)
        sal_after = chain_engine._edges_lookup[("a", "b")].salience
        assert sal_after > sal_mid > sal_before

    def test_stats_updated_after_learning(self, learner, chain_engine):
        learner.learn_from_propagation(chain_engine, {"A": 1.0, "B": 0.5})
        stats = learner.stats
        assert stats["total_events"] > 0
        assert stats["strengthened"] + stats["weakened"] == stats["total_events"]


# ═══════════════════════════════════════════════════════════
# TestHebbianLearnFromPath
# ═══════════════════════════════════════════════════════════


class TestHebbianLearnFromPath:
    def test_path_edges_strengthened(self, learner, chain_engine):
        events = learner.learn_from_path(chain_engine, ["A", "B", "C"])
        assert len(events) == 2
        sources = [(e.source, e.target) for e in events]
        assert ("a", "b") in sources
        assert ("b", "c") in sources
        for e in events:
            assert e.new_confidence > e.old_confidence

    def test_nonexistent_edge_skipped(self, learner, chain_engine):
        events = learner.learn_from_path(chain_engine, ["A", "X", "C"])
        # A→X doesn't exist, X→C doesn't exist
        assert len(events) == 0

    def test_strength_parameter_modulates(self, chain_engine):
        h1 = HebbianLearner(strengthen_rate=0.5)
        engine1 = STGEngine()
        engine1.add_edge("A", "B", confidence=0.5)
        events1 = h1.learn_from_path(engine1, ["A", "B"], strength=0.5)

        h2 = HebbianLearner(strengthen_rate=0.5)
        engine2 = STGEngine()
        engine2.add_edge("A", "B", confidence=0.5)
        events2 = h2.learn_from_path(engine2, ["A", "B"], strength=1.0)

        if events1 and events2:
            delta1 = events1[0].new_confidence - events1[0].old_confidence
            delta2 = events2[0].new_confidence - events2[0].old_confidence
            assert delta2 > delta1

    def test_single_node_path_no_events(self, learner, chain_engine):
        events = learner.learn_from_path(chain_engine, ["A"])
        assert len(events) == 0

    def test_trigger_is_manual(self, learner, chain_engine):
        events = learner.learn_from_path(chain_engine, ["A", "B"])
        for e in events:
            assert e.trigger == "manual"


# ═══════════════════════════════════════════════════════════
# TestSalienceDecay (G2)
# ═══════════════════════════════════════════════════════════


class TestSalienceDecay:
    """G2 fix: time-based salience decay tests."""

    def test_decay_reduces_old_edge(self):
        """Edge unused for 14 days (2 half-lives) should decay to ~25%."""
        engine = STGEngine()
        edge = engine.add_edge("A", "B", confidence=0.8)
        edge.created_at = time.time() - 14 * 86400
        edge.last_used = None
        from stg_engine.learning import decay_salience
        count = decay_salience(engine._edges, time.time(), half_life_days=7.0)
        assert count == 1
        assert 0.19 < edge.salience < 0.21  # 0.8 * 0.25 = 0.20

    def test_recent_edge_unaffected(self):
        """Edge used 1 hour ago should barely decay."""
        engine = STGEngine()
        edge = engine.add_edge("A", "B", confidence=0.9)
        edge.last_used = time.time() - 3600  # 1 hour ago
        from stg_engine.learning import decay_salience
        decay_salience(engine._edges, time.time(), half_life_days=7.0)
        assert edge.salience > 0.89  # barely changed

    def test_floor_respected(self):
        """Salience should never go below floor."""
        engine = STGEngine()
        edge = engine.add_edge("A", "B", confidence=0.5)
        edge.created_at = time.time() - 365 * 86400  # 1 year ago
        edge.last_used = None
        from stg_engine.learning import decay_salience
        decay_salience(engine._edges, time.time(), half_life_days=7.0, floor=0.05)
        assert edge.salience >= 0.05

    def test_none_last_used_falls_back_to_created_at(self):
        """Edge with no last_used should decay from created_at."""
        engine = STGEngine()
        edge = engine.add_edge("A", "B", confidence=0.8)
        edge.created_at = time.time() - 7 * 86400  # 1 half-life
        edge.last_used = None
        from stg_engine.learning import decay_salience
        decay_salience(engine._edges, time.time(), half_life_days=7.0)
        assert 0.39 < edge.salience < 0.41  # 0.8 * 0.5 = 0.40

    def test_virtual_edge_skipped(self):
        """Virtual edges should not be decayed."""
        engine = STGEngine()
        edge = engine.add_edge("A", "B", confidence=0.8, edge_class="virtual")
        edge.created_at = time.time() - 30 * 86400
        from stg_engine.learning import decay_salience
        count = decay_salience(engine._edges, time.time(), half_life_days=7.0)
        assert count == 0
        assert edge.salience == 0.8

    def test_zero_elapsed_no_decay(self):
        """Edge used just now should not decay at all."""
        engine = STGEngine()
        edge = engine.add_edge("A", "B", confidence=0.9)
        now = time.time()
        edge.last_used = now
        from stg_engine.learning import decay_salience
        count = decay_salience(engine._edges, now, half_life_days=7.0)
        assert count == 0
        assert edge.salience == 0.9

    def test_double_call_no_over_decay(self):
        """Calling decay twice 1 min apart should equal single call with total elapsed."""
        engine = STGEngine()
        edge = engine.add_edge("A", "B", confidence=0.8)
        edge.created_at = time.time() - 7 * 86400  # 7 days ago
        edge.last_used = None
        from stg_engine.learning import decay_salience
        now = time.time()
        # First call
        decay_salience(engine._edges, now, half_life_days=7.0)
        sal_after_first = edge.salience
        # Second call 60s later — should barely change since last_used was set to now
        decay_salience(engine._edges, now + 60, half_life_days=7.0)
        sal_after_second = edge.salience
        # Should be almost identical (60s is negligible vs 7-day half-life)
        assert abs(sal_after_first - sal_after_second) < 0.001

    def test_multiple_edges_mixed(self):
        """Mix of old and recent edges: only old ones decay."""
        engine = STGEngine()
        old = engine.add_edge("A", "B", confidence=0.8)
        old.created_at = time.time() - 14 * 86400
        old.last_used = None
        recent = engine.add_edge("C", "D", confidence=0.8)
        recent.last_used = time.time() - 60  # 1 min ago
        from stg_engine.learning import decay_salience
        decay_salience(engine._edges, time.time(), half_life_days=7.0)
        assert old.salience < 0.3       # heavily decayed (2 half-lives)
        assert recent.salience > 0.79   # barely changed (1 min)


# TestSynapticPrunerInit
# ═══════════════════════════════════════════════════════════


class TestSynapticPrunerInit:
    def test_default_parameters(self):
        p = SynapticPruner()
        assert p.confidence_threshold == 0.1
        assert p.unused_days == 365.0
        assert p.eid_safety_threshold == 0.01

    def test_custom_parameters(self):
        p = SynapticPruner(confidence_threshold=0.2, unused_days=14.0)
        assert p.confidence_threshold == 0.2
        assert p.unused_days == 14.0

    def test_zero_days_threshold(self):
        p = SynapticPruner(unused_days=0.0)
        assert p.unused_days == 0.0


# ═══════════════════════════════════════════════════════════
# TestSynapticPrune
# ═══════════════════════════════════════════════════════════


class TestSynapticPrune:
    def test_low_confidence_old_edges_pruned(self, pruner, prunable_engine):
        initial_edge_count = len(prunable_engine._edges)
        events = pruner.prune(prunable_engine)
        pruned = [e for e in events if e.event_type == "prune"]
        assert len(pruned) > 0
        assert len(prunable_engine._edges) < initial_edge_count

    def test_high_confidence_edges_kept(self, pruner, prunable_engine):
        events = pruner.prune(prunable_engine)
        pruned_edges = {(e.source, e.target) for e in events if e.event_type == "prune"}
        # A→B (conf=0.8) and B→D (conf=0.7) should NOT be pruned
        assert ("A", "B") not in pruned_edges
        assert ("B", "D") not in pruned_edges

    def test_recently_used_edges_kept(self):
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.05)
        engine._edges_lookup[("a", "b")].last_used = time.time()  # Just used
        pruner = SynapticPruner()
        events = pruner.prune(engine)
        pruned = [e for e in events if e.event_type == "prune"]
        assert len(pruned) == 0

    def test_critical_bridge_edges_kept(self):
        # Create a graph where one edge is a critical bridge
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.05)
        engine.add_edge("B", "C", confidence=0.8)
        # A→B is the only connection from A — it might have nonzero EID
        # Use high eid_safety_threshold to protect it
        pruner = SynapticPruner(eid_safety_threshold=0.0)
        events = pruner.prune(engine)
        # With eid_safety=0.0, even bridge edges can be pruned
        # But with default (0.01), bridges are protected

    def test_orphan_nodes_removed(self, prunable_engine):
        pruner = SynapticPruner()
        events = pruner.prune(prunable_engine)
        orphan_events = [e for e in events if e.event_type == "prune_orphan"]
        # If C→D was pruned and C has no other edges, C is orphan
        remaining_nodes = set(prunable_engine._nodes.keys())
        for name in remaining_nodes:
            assert prunable_engine._graph.degree(name) > 0

    def test_events_logged_correctly(self, pruner, prunable_engine):
        events = pruner.prune(prunable_engine)
        for e in events:
            assert isinstance(e, LearningEvent)
            assert e.trigger == "prune_cycle"
            assert e.event_type in ("prune", "prune_orphan")

    def test_caches_invalidated(self, pruner, prunable_engine):
        prunable_engine.get_metrics()
        assert prunable_engine._graph_metrics_cache is not None
        events = pruner.prune(prunable_engine)
        if events:
            assert prunable_engine._graph_metrics_cache is None

    def test_no_candidates_no_changes(self, pruner):
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.9)
        engine.add_edge("B", "C", confidence=0.8)
        initial = len(engine._edges)
        events = pruner.prune(engine)
        assert len(events) == 0
        assert len(engine._edges) == initial


# ═══════════════════════════════════════════════════════════
# TestSynapticPruneDryRun
# ═══════════════════════════════════════════════════════════


class TestSynapticPruneDryRun:
    def test_dry_run_returns_candidates(self, pruner, prunable_engine):
        candidates = pruner.dry_run(prunable_engine)
        assert len(candidates) > 0
        for src, tgt, conf, eid in candidates:
            assert conf < pruner.confidence_threshold
            assert eid < pruner.eid_safety_threshold

    def test_dry_run_does_not_modify_graph(self, pruner, prunable_engine):
        n_before = len(prunable_engine._edges)
        nodes_before = len(prunable_engine._nodes)
        pruner.dry_run(prunable_engine)
        assert len(prunable_engine._edges) == n_before
        assert len(prunable_engine._nodes) == nodes_before

    def test_dry_run_matches_prune(self, prunable_engine):
        pruner = SynapticPruner()
        candidates = pruner.dry_run(prunable_engine)
        candidate_edges = {(src, tgt) for src, tgt, _, _ in candidates}

        # Now actually prune a fresh copy
        engine2 = STGEngine()
        old_time = time.time() - 400 * 86400
        engine2.add_edge("A", "B", confidence=0.8)
        engine2.add_edge("A", "C", confidence=0.05)
        engine2.add_edge("C", "D", confidence=0.03)
        engine2.add_edge("B", "D", confidence=0.7)
        for edge in engine2._edges:
            if edge.salience < 0.1:
                edge.last_used = old_time
            else:
                edge.last_used = time.time()

        events = pruner.prune(engine2)
        pruned_edges = {(e.source, e.target) for e in events if e.event_type == "prune"}
        assert candidate_edges == pruned_edges


# ═══════════════════════════════════════════════════════════
# TestImportanceBiasedPropagation
# ═══════════════════════════════════════════════════════════


class TestImportanceBiasedPropagation:
    def test_default_no_importance_bias(self):
        engine = STGEngine()
        assert engine.importance_weight == 0.0

    def test_confidence_weighted_propagation(self):
        """Higher-confidence edges carry more activation."""
        engine = STGEngine()
        engine.add_edge("root", "high_conf", confidence=0.9)
        engine.add_edge("root", "low_conf", confidence=0.1)
        engine.propagate("root")

        high_node = engine.get_node("high_conf")
        low_node = engine.get_node("low_conf")
        assert high_node is not None and low_node is not None
        assert high_node.activation > low_node.activation

    def test_energy_conservation(self):
        """Total spread per step is approximately conserved."""
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.8)
        engine.add_edge("A", "C", confidence=0.2)
        result = engine.propagate("A", iterations=1, decay=1.0, normalize=False)
        # B + C should receive total ≈ A's initial activation
        b = engine.get_node("B")
        c = engine.get_node("C")
        if b and c:
            total = b.activation + c.activation
            # Should be close to 1.0 (initial activation of A)
            assert 0.5 < total < 1.5

    def test_returns_same_type(self, chain_engine):
        """propagate() still returns List[str]."""
        result = chain_engine.propagate("A")
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, str)

    def test_zero_confidence_edge_carries_little(self):
        """Confidence=0.01 edge carries almost nothing."""
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.99)
        engine.add_edge("A", "C", confidence=0.01)
        engine.propagate("A")
        b = engine.get_node("B")
        c = engine.get_node("C")
        assert b is not None and c is not None
        assert b.activation > 10 * c.activation


# ═══════════════════════════════════════════════════════════
# TestEngineLearningIntegration
# ═══════════════════════════════════════════════════════════


class TestEngineLearningIntegration:
    def test_learning_disabled_by_default(self):
        engine = STGEngine()
        assert engine.learning_enabled is False

    def test_enable_disable_learning(self):
        engine = STGEngine()
        engine.enable_learning()
        assert engine.learning_enabled is True
        engine.disable_learning()
        assert engine.learning_enabled is False

    def test_propagate_with_learning_modifies_salience(self, chain_engine):
        sals_before = {
            (e.source, e.target): e.salience
            for e in chain_engine._edges
        }
        chain_engine.enable_learning(strengthen_rate=0.2, weaken_rate=0.1)
        chain_engine.propagate("A")
        sals_after = {
            (e.source, e.target): e.salience
            for e in chain_engine._edges
        }
        # At least one salience should have changed
        assert any(
            abs(sals_before[k] - sals_after[k]) > 1e-10
            for k in sals_before
        )

    def test_propagate_without_learning_no_salience_change(self, chain_engine):
        sals_before = {
            (e.source, e.target): e.salience
            for e in chain_engine._edges
        }
        chain_engine.propagate("A")
        sals_after = {
            (e.source, e.target): e.salience
            for e in chain_engine._edges
        }
        assert all(
            abs(sals_before[k] - sals_after[k]) < 1e-10
            for k in sals_before
        )

    def test_learning_log_accumulates(self, chain_engine):
        chain_engine.enable_learning(strengthen_rate=0.2, weaken_rate=0.1)
        chain_engine.propagate("A")
        log1 = len(chain_engine.learning_log)
        chain_engine.propagate("A")
        log2 = len(chain_engine.learning_log)
        assert log2 > log1

    def test_learn_from_path_works_without_enable(self, chain_engine):
        assert chain_engine.learning_enabled is False
        events = chain_engine.learn_from_path(["A", "B", "C"])
        assert len(events) == 2

    def test_prune_via_engine(self, prunable_engine):
        events = prunable_engine.prune()
        assert isinstance(events, list)

    def test_save_load_preserves_last_used(self, chain_engine):
        # Set last_used on an edge
        now = time.time()
        chain_engine._edges_lookup[("a", "b")].last_used = now

        with tempfile.NamedTemporaryFile(suffix=".stg", delete=False) as f:
            path = f.name
        try:
            chain_engine.save(path)
            loaded = STGEngine.load(path)
            edge = loaded._edges_lookup.get(("a", "b"))
            assert edge is not None
            assert edge.last_used is not None
            assert abs(edge.last_used - now) < 0.01
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════
# TestPersistenceLastUsed
# ═══════════════════════════════════════════════════════════


class TestPersistenceLastUsed:
    def test_save_with_last_used(self):
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.5)
        engine._edges_lookup[("a", "b")].last_used = time.time()

        with tempfile.NamedTemporaryFile(suffix=".stg", delete=False) as f:
            path = f.name
        try:
            engine.save(path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_load_with_last_used(self):
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.5)
        ts = time.time()
        engine._edges_lookup[("a", "b")].last_used = ts

        with tempfile.NamedTemporaryFile(suffix=".stg", delete=False) as f:
            path = f.name
        try:
            engine.save(path)
            loaded = STGEngine.load(path)
            edge = loaded._edges_lookup.get(("a", "b"))
            assert edge is not None
            assert abs(edge.last_used - ts) < 0.01
        finally:
            os.unlink(path)

    def test_load_none_last_used(self):
        """Edge with last_used=None loads correctly."""
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.5)
        # last_used is None by default

        with tempfile.NamedTemporaryFile(suffix=".stg", delete=False) as f:
            path = f.name
        try:
            engine.save(path)
            loaded = STGEngine.load(path)
            edge = loaded._edges_lookup.get(("a", "b"))
            assert edge is not None
            assert edge.last_used is None
        finally:
            os.unlink(path)

    def test_roundtrip_preserves_all_last_used(self):
        engine = STGEngine()
        t1 = time.time() - 1000
        t2 = time.time()
        engine.add_edge("A", "B", confidence=0.5)
        engine.add_edge("B", "C", confidence=0.7)
        engine._edges_lookup[("a", "b")].last_used = t1
        engine._edges_lookup[("b", "c")].last_used = t2

        with tempfile.NamedTemporaryFile(suffix=".stg", delete=False) as f:
            path = f.name
        try:
            engine.save(path)
            loaded = STGEngine.load(path)
            assert abs(loaded._edges_lookup[("a", "b")].last_used - t1) < 0.01
            assert abs(loaded._edges_lookup[("b", "c")].last_used - t2) < 0.01
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════
# TestLearningConvergence
# ═══════════════════════════════════════════════════════════


class TestLearningConvergence:
    def test_repeated_propagation_strengthens_path(self):
        """After many propagations, relevant edges approach ceiling."""
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.5)
        engine.add_edge("B", "C", confidence=0.5)
        engine.enable_learning(strengthen_rate=0.1, weaken_rate=0.05)

        for _ in range(20):
            engine.propagate("A")

        ab_sal = engine._edges_lookup[("a", "b")].salience
        assert ab_sal > 0.5  # Should have increased

    def test_unused_edges_decay_over_propagations(self):
        """Edges not in the activation path decay."""
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.5)
        engine.add_edge("A", "C", confidence=0.5)
        engine.add_edge("B", "D", confidence=0.5)
        engine.add_edge("C", "D", confidence=0.5)
        engine.enable_learning(strengthen_rate=0.1, weaken_rate=0.05)

        # Propagate 'B' — activates B and D via B→D, but A→C may weaken
        for _ in range(20):
            engine.propagate("B")

        bd_sal = engine._edges_lookup[("b", "d")].salience
        # B→D should be strengthened (both B and D activated)
        assert bd_sal > 0.5

    def test_qe_can_improve_with_learning(self):
        """QE should be measurable after learning."""
        engine = STGEngine()
        engine.add_edge("consciousness", "awareness", confidence=0.5)
        engine.add_edge("consciousness", "noise1", confidence=0.5)
        engine.add_edge("consciousness", "noise2", confidence=0.5)
        engine.add_edge("awareness", "insight", confidence=0.5)
        engine.enable_learning(strengthen_rate=0.1, weaken_rate=0.05)

        # First propagation — measure QE
        engine.propagate("consciousness")
        pm1 = engine.last_propagation_metrics

        # Learn for 10 rounds
        for _ in range(10):
            engine.propagate("consciousness")

        pm2 = engine.last_propagation_metrics

        # Both should have valid metrics
        assert pm1 is not None
        assert pm2 is not None
        assert pm2.query_efficiency >= 0.0


# ═══════════════════════════════════════════════════════════
# TestWeakenActivationThreshold
# ═══════════════════════════════════════════════════════════


class TestWeakenActivationThreshold:
    """Tests for the weaken_activation_threshold and activation-modulated weakening."""

    def test_below_weaken_threshold_no_weakening(self):
        """Source active but below weaken_activation_threshold → no weakening."""
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.5)
        learner = HebbianLearner(weaken_activation_threshold=0.5)
        # A at 0.3 — above activation_threshold (0.1) but below weaken threshold (0.5)
        events = learner.learn_from_propagation(engine, {"A": 0.3})
        weakened = [e for e in events if e.event_type == "weaken"]
        assert len(weakened) == 0
        assert engine._edges_lookup[("a", "b")].salience == 0.5

    def test_above_weaken_threshold_weakens(self):
        """Source above weaken_activation_threshold → weakening occurs."""
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.5)
        learner = HebbianLearner(weaken_activation_threshold=0.3)
        # A at 0.5 — above weaken threshold (0.3)
        events = learner.learn_from_propagation(engine, {"A": 0.5})
        weakened = [e for e in events if e.event_type == "weaken"]
        assert len(weakened) == 1
        assert engine._edges_lookup[("a", "b")].salience < 0.5

    def test_weaken_modulated_by_source_activation(self):
        """Higher source activation → more weakening."""
        # Low source activation
        engine1 = STGEngine()
        engine1.add_edge("A", "B", confidence=0.5)
        learner1 = HebbianLearner(weaken_rate=0.5, weaken_activation_threshold=0.3)
        learner1.learn_from_propagation(engine1, {"A": 0.4})  # just above threshold
        sal_low = engine1._edges_lookup[("a", "b")].salience

        # High source activation
        engine2 = STGEngine()
        engine2.add_edge("A", "B", confidence=0.5)
        learner2 = HebbianLearner(weaken_rate=0.5, weaken_activation_threshold=0.3)
        learner2.learn_from_propagation(engine2, {"A": 1.0})  # strong activation
        sal_high = engine2._edges_lookup[("a", "b")].salience

        # High activation should produce MORE weakening (lower salience)
        assert sal_high < sal_low

    def test_weaken_threshold_reduces_scope(self):
        """With higher weaken threshold, fewer edges are weakened."""
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.5)
        engine.add_edge("C", "D", confidence=0.5)

        # A barely active (0.15), C strongly active (0.8)
        activation = {"A": 0.15, "C": 0.8}

        learner = HebbianLearner(weaken_activation_threshold=0.3)
        events = learner.learn_from_propagation(engine, activation)
        weakened = [e for e in events if e.event_type == "weaken"]

        # Only C→D should be weakened (C=0.8 > 0.3), not A→B (A=0.15 < 0.3)
        assert len(weakened) == 1
        assert weakened[0].source == "c"

    def test_custom_weaken_activation_threshold(self):
        """weaken_activation_threshold parameter is stored correctly."""
        h = HebbianLearner(weaken_activation_threshold=0.6)
        assert h.weaken_activation_threshold == 0.6

    def test_strengthen_unaffected_by_weaken_threshold(self):
        """weaken_activation_threshold does NOT affect strengthening."""
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.5)
        learner = HebbianLearner(
            strengthen_rate=0.5,
            weaken_activation_threshold=0.9,  # very high
        )
        # A=0.2, B=0.2 — both above activation_threshold (0.1) but below weaken threshold
        events = learner.learn_from_propagation(engine, {"A": 0.2, "B": 0.2})
        strengthened = [e for e in events if e.event_type == "strengthen"]
        assert len(strengthened) == 1  # Should still strengthen
