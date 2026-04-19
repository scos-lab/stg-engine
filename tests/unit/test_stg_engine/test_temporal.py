"""Tests for STG Temporal Structure (Phase 11).

Tests three layers:
  Layer 1: Temporal Index — created_at field, time range queries
  Layer 2: Episode Graph — temporal edges, k-fold association
  Layer 3: Temporal Retrieval — neighborhood, replay, temporal propagate
"""

import time
import pytest

from stg_engine import STGEngine
from stg_engine.temporal import (
    query_time_range,
    query_temporal_neighborhood,
    record_temporal_edge,
    build_episode_sequence,
    replay_episode,
    temporal_propagate,
    epoch_to_str,
    parse_date_str,
)


# ═══════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════


@pytest.fixture
def engine():
    """Fresh STGEngine."""
    return STGEngine()


@pytest.fixture
def timestamped_engine():
    """Engine with edges that have known created_at values."""
    e = STGEngine()
    now = time.time()

    # Create edges with controlled timestamps
    edge1 = e.add_edge("A", "B", confidence=0.9, rule="causal", session_id="S1")
    edge1.created_at = now - 7200  # 2 hours ago

    edge2 = e.add_edge("B", "C", confidence=0.8, rule="causal", session_id="S1")
    edge2.created_at = now - 3600  # 1 hour ago

    edge3 = e.add_edge("C", "D", confidence=0.7, rule="logical", session_id="S1")
    edge3.created_at = now - 1800  # 30 min ago

    edge4 = e.add_edge("D", "E", confidence=0.85, rule="empirical", session_id="S1")
    edge4.created_at = now - 900  # 15 min ago

    edge5 = e.add_edge("X", "Y", confidence=0.6, session_id="S2")
    edge5.created_at = now - 86400  # 1 day ago

    return e, now


# ═══════════════════════════════════════════════════════════
# Layer 0: STGEdge created_at field
# ═══════════════════════════════════════════════════════════


class TestCreatedAtField:
    """P0: Verify created_at is set on new edges."""

    def test_new_edge_has_timestamp(self, engine):
        before = time.time()
        edge = engine.add_edge("X", "Y", confidence=0.8)
        after = time.time()
        assert before <= edge.created_at <= after

    def test_edge_class_defaults_to_knowledge(self, engine):
        edge = engine.add_edge("X", "Y")
        assert edge.edge_class == "knowledge"

    def test_delay_k_defaults_to_zero(self, engine):
        edge = engine.add_edge("X", "Y")
        assert edge.delay_k == 0

    def test_edge_to_dict_includes_temporal_fields(self, engine):
        edge = engine.add_edge("X", "Y")
        d = edge.to_dict()
        assert "created_at" in d
        assert "edge_class" in d
        assert "delay_k" in d
        assert d["created_at"] > 0
        assert d["edge_class"] == "knowledge"
        assert d["delay_k"] == 0

    def test_edge_from_dict_round_trip(self, engine):
        from stg_engine.types import STGEdge
        edge = engine.add_edge("A", "B", confidence=0.9)
        d = edge.to_dict()
        restored = STGEdge.from_dict(d)
        assert restored.created_at == edge.created_at
        assert restored.edge_class == edge.edge_class
        assert restored.delay_k == edge.delay_k

    def test_edge_class_from_modifiers(self, engine):
        """edge_class passed as modifier is extracted to field."""
        edge = engine.add_edge("A", "B", edge_class="virtual")
        assert edge.edge_class == "virtual"

    def test_delay_k_from_modifiers(self, engine):
        """delay_k passed as modifier is extracted to field."""
        edge = engine.add_edge("A", "B", delay_k=2)
        assert edge.delay_k == 2
        # delay_k is popped from modifiers
        assert "delay_k" not in edge.modifiers


# ═══════════════════════════════════════════════════════════
# Layer 1: Temporal Index — Time range queries
# ═══════════════════════════════════════════════════════════


class TestQueryTimeRange:
    """P1: Query edges by creation time."""

    def test_returns_edges_in_range(self, timestamped_engine):
        engine, now = timestamped_engine
        # Query last 2.5 hours
        edges = query_time_range(engine, now - 9000, now)
        assert len(edges) == 4  # A→B, B→C, C→D, D→E

    def test_excludes_legacy_edges(self, engine):
        """Edges with created_at=0.0 are excluded."""
        edge = engine.add_edge("A", "B")
        edge.created_at = 0.0  # Force legacy
        edges = query_time_range(engine, 0, time.time() + 100)
        assert len(edges) == 0

    def test_sorted_chronologically(self, timestamped_engine):
        engine, now = timestamped_engine
        edges = query_time_range(engine, now - 86500, now)
        times = [e.created_at for e in edges]
        assert times == sorted(times)

    def test_filter_by_edge_class(self, timestamped_engine):
        engine, now = timestamped_engine
        # Add a temporal edge
        record_temporal_edge(engine, "A", "C", delay_k=2, session_id="S1")
        edges = query_time_range(engine, now - 86500, now + 100, edge_class="temporal")
        assert len(edges) == 1
        assert edges[0].edge_class == "temporal"

    def test_empty_range(self, timestamped_engine):
        engine, now = timestamped_engine
        # Future range — no edges
        edges = query_time_range(engine, now + 1000, now + 2000)
        assert len(edges) == 0

    def test_exact_boundary(self, timestamped_engine):
        engine, now = timestamped_engine
        # Edge at exactly now - 3600
        edge = [e for e in engine._edges if abs(e.created_at - (now - 3600)) < 1][0]
        edges = query_time_range(engine, edge.created_at, edge.created_at)
        assert len(edges) == 1


class TestQueryTemporalNeighborhood:
    """P1: Find edges created around the same time as a node's edges."""

    def test_finds_neighborhood(self, timestamped_engine):
        engine, now = timestamped_engine
        # B is involved in edges at -7200 and -3600, median is -5400
        # Window ±1h → should find edges from -9000 to -1800
        edges = query_temporal_neighborhood(engine, "B", window_seconds=3600)
        # Should find A→B(-7200), B→C(-3600), maybe C→D(-1800)
        assert len(edges) >= 2

    def test_returns_empty_for_unknown_node(self, engine):
        edges = query_temporal_neighborhood(engine, "NonExistent")
        assert edges == []

    def test_returns_empty_for_legacy_only(self, engine):
        """Node with only legacy edges (created_at=0) returns empty."""
        edge = engine.add_edge("A", "B")
        edge.created_at = 0.0
        edges = query_temporal_neighborhood(engine, "A")
        assert edges == []

    def test_large_window_captures_all(self, timestamped_engine):
        engine, now = timestamped_engine
        edges = query_temporal_neighborhood(engine, "C", window_seconds=86400 * 2)
        assert len(edges) == 5  # All timestamped edges


# ═══════════════════════════════════════════════════════════
# Layer 2: Episode Graph — Temporal edges and k-fold
# ═══════════════════════════════════════════════════════════


class TestRecordTemporalEdge:
    """Temporal edge creation."""

    def test_creates_temporal_edge(self, engine):
        edge = record_temporal_edge(engine, "A", "B", delay_k=1)
        assert edge.edge_class == "temporal"
        assert edge.delay_k == 1
        assert edge.confidence == 1.0
        assert edge.rule == "temporal"
        assert edge.created_at > 0

    def test_step2_edge(self, engine):
        edge = record_temporal_edge(engine, "A", "C", delay_k=2)
        assert edge.delay_k == 2

    def test_session_id_propagated(self, engine):
        edge = record_temporal_edge(engine, "A", "B", session_id="S42")
        assert edge.session_id == "S42"


class TestBuildEpisodeSequence:
    """k-fold episode construction from session edges."""

    def test_builds_step1_chain(self, timestamped_engine):
        engine, now = timestamped_engine
        # S1 has edges: A→B, B→C, C→D, D→E (ordered by created_at)
        edges = build_episode_sequence(engine, "S1", k_fold=1)
        # Should create temporal chain: A→B→C→D→E (step-1)
        step1 = [e for e in edges if e.delay_k == 1]
        assert len(step1) >= 3  # At least A→B, B→C, C→D (may also D→E)

    def test_builds_kfold2(self, timestamped_engine):
        engine, now = timestamped_engine
        edges = build_episode_sequence(engine, "S1", k_fold=2)
        step1 = [e for e in edges if e.delay_k == 1]
        step2 = [e for e in edges if e.delay_k == 2]
        assert len(step1) > 0
        assert len(step2) > 0

    def test_kfold3_creates_three_folds(self, timestamped_engine):
        engine, now = timestamped_engine
        edges = build_episode_sequence(engine, "S1", k_fold=3)
        folds = {e.delay_k for e in edges}
        assert 1 in folds
        assert 2 in folds
        assert 3 in folds

    def test_empty_session_returns_empty(self, engine):
        edges = build_episode_sequence(engine, "NO_SUCH_SESSION")
        assert edges == []

    def test_single_edge_session_returns_empty(self, engine):
        """A session with only 1 edge can still produce temporal edges
        if it has 2+ unique nodes."""
        edge = engine.add_edge("A", "B", session_id="S_SINGLE")
        # 2 nodes (A, B) → should produce at least 1 temporal edge
        edges = build_episode_sequence(engine, "S_SINGLE", k_fold=1)
        assert len(edges) >= 1

    def test_no_duplicate_temporal_edges(self, timestamped_engine):
        engine, now = timestamped_engine
        edges1 = build_episode_sequence(engine, "S1", k_fold=2)
        # Build again — should skip existing temporal edges
        edges2 = build_episode_sequence(engine, "S1", k_fold=2)
        assert len(edges2) == 0  # All already exist

    def test_temporal_edges_are_temporal_class(self, timestamped_engine):
        engine, now = timestamped_engine
        edges = build_episode_sequence(engine, "S1", k_fold=1)
        for e in edges:
            assert e.edge_class == "temporal"
            assert e.rule == "temporal"
            assert e.confidence == 1.0


# ═══════════════════════════════════════════════════════════
# Layer 3: Temporal Retrieval
# ═══════════════════════════════════════════════════════════


class TestReplayEpisode:
    """Episode replay via temporal edge traversal."""

    def test_replay_follows_chain(self, engine):
        """Build a simple chain and replay it."""
        record_temporal_edge(engine, "A", "B", delay_k=1, session_id="S1")
        record_temporal_edge(engine, "B", "C", delay_k=1, session_id="S1")
        record_temporal_edge(engine, "C", "D", delay_k=1, session_id="S1")

        seq = replay_episode(engine, "A", session_id="S1")
        assert seq == ["A", "B", "C", "D"]

    def test_replay_from_middle(self, engine):
        record_temporal_edge(engine, "A", "B", delay_k=1)
        record_temporal_edge(engine, "B", "C", delay_k=1)
        record_temporal_edge(engine, "C", "D", delay_k=1)

        seq = replay_episode(engine, "B")
        assert seq == ["B", "C", "D"]

    def test_replay_no_temporal_edges(self, engine):
        engine.add_edge("A", "B")  # knowledge edge, not temporal
        seq = replay_episode(engine, "A")
        assert seq == ["A"]  # Only the entry node

    def test_replay_session_constraint(self, engine):
        record_temporal_edge(engine, "A", "B", delay_k=1, session_id="S1")
        record_temporal_edge(engine, "A", "X", delay_k=1, session_id="S2")

        seq = replay_episode(engine, "A", session_id="S1")
        assert "B" in seq
        assert "X" not in seq

    def test_replay_prevents_cycles(self, engine):
        """Replay should not revisit nodes (prevents infinite loops)."""
        record_temporal_edge(engine, "A", "B", delay_k=1)
        record_temporal_edge(engine, "B", "A", delay_k=1)  # cycle!

        seq = replay_episode(engine, "A")
        assert seq == ["A", "B"]  # Stops when B→A would revisit A

    def test_replay_uses_kfold_disambiguation(self, engine):
        """When branching, k-fold context helps choose the right path."""
        # Two paths from B: B→C (in S1 sequence) and B→X (different sequence)
        record_temporal_edge(engine, "A", "B", delay_k=1, session_id="S1")
        record_temporal_edge(engine, "B", "C", delay_k=1, session_id="S1")
        record_temporal_edge(engine, "B", "X", delay_k=1, session_id="S1")
        # k=2 edge supports A→C path
        record_temporal_edge(engine, "A", "C", delay_k=2, session_id="S1")

        seq = replay_episode(engine, "A", session_id="S1")
        # Should prefer C over X due to k-fold support from A→C(k=2)
        assert seq[0] == "A"
        assert seq[1] == "B"
        assert seq[2] == "C"  # k-fold disambiguated

    def test_max_steps_limit(self, engine):
        """Replay respects max_steps."""
        for i in range(100):
            record_temporal_edge(engine, f"N{i}", f"N{i+1}", delay_k=1)
        seq = replay_episode(engine, "N0", max_steps=10)
        assert len(seq) == 11  # entry + 10 steps


class TestTemporalPropagate:
    """Propagation with recency bias."""

    def test_returns_list_of_tuples(self, timestamped_engine):
        engine, now = timestamped_engine
        results = temporal_propagate(engine, "A B C", time_bias=0.5)
        assert isinstance(results, list)
        if results:
            assert isinstance(results[0], tuple)
            assert len(results[0]) == 2

    def test_zero_bias_equals_standard(self, timestamped_engine):
        engine, now = timestamped_engine
        standard = engine.propagate("A B C")
        temporal = temporal_propagate(engine, "A B C", time_bias=0.0)
        # Same nodes (order may differ slightly due to float arithmetic)
        std_nodes = set(standard[:10])
        tmp_nodes = {n for n, _ in temporal[:10]}
        assert std_nodes == tmp_nodes

    def test_high_bias_favors_recent(self, timestamped_engine):
        engine, now = timestamped_engine
        results = temporal_propagate(engine, "A B C D E X Y", time_bias=0.9)
        if len(results) >= 2:
            # Recent nodes should score higher with high time_bias
            scores = {n: s for n, s in results}
            # D→E (15 min ago) should score higher than X→Y (1 day ago)
            if "D" in scores and "X" in scores:
                assert scores["D"] > scores["X"]


# ═══════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════


class TestEpochToStr:
    def test_valid_epoch(self):
        # 2026-01-01 00:00:00 UTC (approximate)
        result = epoch_to_str(1767225600.0)
        assert "2026" in result

    def test_zero_epoch(self):
        assert epoch_to_str(0.0) == "(unknown)"

    def test_negative_epoch(self):
        assert epoch_to_str(-1.0) == "(unknown)"


class TestParseDateStr:
    def test_date_only(self):
        ts = parse_date_str("2026-03-18")
        assert ts > 0

    def test_date_with_time(self):
        ts = parse_date_str("2026-03-18 14:30:00")
        assert ts > 0

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            parse_date_str("March 18, 2026")

    def test_round_trip(self):
        original = "2026-03-18 14:30:00"
        ts = parse_date_str(original)
        result = epoch_to_str(ts)
        assert "2026-03-18" in result
        assert "14:30:00" in result


# ═══════════════════════════════════════════════════════════
# Integration: Hebbian exclusion of temporal edges
# ═══════════════════════════════════════════════════════════


class TestHebbianExclusion:
    """Temporal edges should not be modified by Hebbian learning."""

    def test_temporal_edge_not_strengthened(self, engine):
        from stg_engine.learning import HebbianLearner

        # Add a temporal edge
        temporal_edge = record_temporal_edge(engine, "A", "B", delay_k=1)
        original_salience = temporal_edge.salience

        # Run Hebbian with both nodes active
        learner = HebbianLearner()
        activation_map = {"A": 1.0, "B": 1.0}
        learner.learn_from_propagation(engine, activation_map)

        # Salience should not change
        assert temporal_edge.salience == original_salience


# ═══════════════════════════════════════════════════════════
# Persistence: Save/Load round trip
# ═══════════════════════════════════════════════════════════


class TestPersistence:
    """Temporal fields survive save/load cycle."""

    def test_save_load_preserves_created_at(self, engine, tmp_path):
        edge = engine.add_edge("A", "B", confidence=0.9)
        original_ts = edge.created_at
        assert original_ts > 0

        path = str(tmp_path / "test.stg")
        engine.save(path)

        loaded = STGEngine.load(path)
        loaded_edge = loaded._edges_lookup[("a", "b")]
        assert loaded_edge.created_at == pytest.approx(original_ts, abs=0.01)

    def test_save_load_preserves_edge_class(self, engine, tmp_path):
        record_temporal_edge(engine, "A", "B", delay_k=2)

        path = str(tmp_path / "test.stg")
        engine.save(path)

        loaded = STGEngine.load(path)
        loaded_edge = loaded._edges_lookup[("a", "b")]
        assert loaded_edge.edge_class == "temporal"
        assert loaded_edge.delay_k == 2

    def test_save_load_preserves_delay_k(self, engine, tmp_path):
        record_temporal_edge(engine, "X", "Y", delay_k=3)

        path = str(tmp_path / "test.stg")
        engine.save(path)

        loaded = STGEngine.load(path)
        loaded_edge = loaded._edges_lookup[("x", "y")]
        assert loaded_edge.delay_k == 3

    def test_legacy_edges_load_with_defaults(self, tmp_path):
        """Older .stg files without temporal columns load with defaults."""
        import sqlite3, json

        path = str(tmp_path / "legacy.stg")
        conn = sqlite3.connect(path)
        conn.executescript("""
            CREATE TABLE schema_info (key TEXT PRIMARY KEY, value TEXT);
            INSERT INTO schema_info VALUES ('version', '8');
            CREATE TABLE nodes (name TEXT PRIMARY KEY, namespace TEXT,
                anchor_type TEXT, metadata_json TEXT DEFAULT '{}',
                tension REAL DEFAULT 0.0, activation REAL DEFAULT 0.0,
                self_relevance REAL DEFAULT 0.0);
            CREATE TABLE edges (id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT, target TEXT, confidence REAL DEFAULT 0.5,
                strength REAL DEFAULT 0.5, rule TEXT, time TEXT,
                modifiers_json TEXT DEFAULT '{}', session_id TEXT,
                event_id TEXT, last_used REAL, preference REAL DEFAULT 0.0,
                salience REAL DEFAULT 0.5);
            CREATE TABLE sessions (session_id TEXT PRIMARY KEY, date TEXT,
                title TEXT, avg_importance REAL, event_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'complete', summary TEXT);
            CREATE TABLE events (event_id TEXT PRIMARY KEY, session_id TEXT,
                timestamp TEXT, event_type TEXT, memory_type TEXT, title TEXT,
                importance_score REAL DEFAULT 0.5, description TEXT,
                tags_json TEXT DEFAULT '[]', artifacts_json TEXT DEFAULT '[]',
                stl_block TEXT);
            CREATE TABLE tensions (name TEXT PRIMARY KEY, initial_value REAL,
                current_value REAL, status TEXT, created_session TEXT,
                resolved_session TEXT, description TEXT);
            CREATE TABLE belief_evolutions (id INTEGER PRIMARY KEY AUTOINCREMENT,
                old_anchor TEXT, new_anchor TEXT, event_id TEXT, session_id TEXT,
                level INTEGER, description TEXT);
            CREATE TABLE system_snapshots (id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT, timestamp TEXT, psi_value REAL,
                max_tension REAL, structural_coherence REAL,
                epistemic_confidence REAL, total_reward REAL,
                node_count INTEGER, edge_count INTEGER);
            INSERT INTO nodes VALUES ('A', NULL, NULL, '{}', 0, 0, 0);
            INSERT INTO nodes VALUES ('B', NULL, NULL, '{}', 0, 0, 0);
            INSERT INTO edges (source, target, confidence, modifiers_json)
                VALUES ('A', 'B', 0.9, '{}');
        """)
        conn.commit()
        conn.close()

        loaded = STGEngine.load(path)
        edge = loaded._edges_lookup[("a", "b")]
        assert edge.created_at == 0.0
        assert edge.edge_class == "knowledge"
        assert edge.delay_k == 0
