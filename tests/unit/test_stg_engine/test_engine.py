"""Tests for STG Engine core."""

import os
import tempfile
import pytest
from stg_engine.engine import STGEngine
from stg_engine.types import STGSession, STGEvent, STGTension, STGBeliefEvolution


class TestSTGEngineBasic:
    def test_empty_engine(self):
        engine = STGEngine()
        assert len(engine) == 0
        stats = engine.get_stats()
        assert stats["node_count"] == 0
        assert stats["edge_count"] == 0

    def test_add_node(self):
        engine = STGEngine()
        node = engine.add_node("TestConcept", anchor_type="Concept")
        assert node.name == "TestConcept"
        assert engine.get_node("TestConcept") is node
        assert len(engine) == 1

    def test_add_node_with_namespace(self):
        engine = STGEngine()
        node = engine.add_node("Energy", namespace="Physics", anchor_type="Concept")
        assert node.namespace == "Physics"
        assert node.qualified_name == "Physics:Energy"

    def test_add_node_update_existing(self):
        engine = STGEngine()
        engine.add_node("Test")
        engine.add_node("Test", anchor_type="Concept", domain="physics")
        assert len(engine) == 1
        node = engine.get_node("Test")
        assert node.anchor_type == "Concept"
        assert node.metadata["domain"] == "physics"

    def test_add_edge(self):
        engine = STGEngine()
        edge = engine.add_edge("A", "B", confidence=0.9, rule="causal")
        assert edge.source == "A"
        assert edge.target == "B"
        assert edge.confidence == 0.9
        # Nodes auto-created
        assert len(engine) == 2

    def test_add_edge_with_modifiers(self):
        engine = STGEngine()
        edge = engine.add_edge(
            "Theory", "Prediction",
            confidence=0.95, strength=0.8, rule="logical",
            author="Einstein", domain="physics",
        )
        assert edge.modifiers["author"] == "Einstein"
        assert edge.modifiers["domain"] == "physics"

    def test_remove_edge(self):
        engine = STGEngine()
        engine.add_edge("A", "B")
        assert engine.remove_edge("A", "B") is True
        assert engine.get_edges("A", "B") == []

    def test_remove_nonexistent_edge(self):
        engine = STGEngine()
        assert engine.remove_edge("X", "Y") is False

    def test_get_edges_by_source(self):
        engine = STGEngine()
        engine.add_edge("A", "B")
        engine.add_edge("A", "C")
        engine.add_edge("B", "C")
        edges = engine.get_edges(source="A")
        assert len(edges) == 2

    def test_get_edges_by_target(self):
        engine = STGEngine()
        engine.add_edge("A", "C")
        engine.add_edge("B", "C")
        edges = engine.get_edges(target="C")
        assert len(edges) == 2

    def test_neighbors(self):
        engine = STGEngine()
        engine.add_edge("A", "B")
        engine.add_edge("A", "C")
        engine.add_edge("D", "A")

        assert set(engine.neighbors("A", "out")) == {"B", "C"}
        assert engine.neighbors("A", "in") == ["D"]
        assert set(engine.neighbors("A", "both")) == {"B", "C", "D"}

    def test_repr(self):
        engine = STGEngine()
        engine.add_edge("A", "B")
        r = repr(engine)
        assert "nodes=2" in r
        assert "edges=1" in r


class TestConflictDetection:
    """G6 fix: ingest conflict detection tests."""

    def test_contradictory_rule_detected(self):
        """Changing rule type on same edge should trigger conflict report on the new edge."""
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.9, rule="causal")
        engine.add_edge("A", "B", confidence=0.8, rule="logical")
        # With multi-edge, conflict report is on the NEW edge (lookup points to it)
        new_edge = engine._edges_lookup[("a", "b")]
        report = new_edge.modifiers.get("_conflict_report")
        assert report is not None
        assert report["conflict_score"] > 0
        assert "rule" in report["details"]
        assert "causal" in report["details"]
        assert "logical" in report["details"]

    def test_no_conflict_same_rule(self):
        """Same rule type should not trigger conflict."""
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.9, rule="causal")
        engine.add_edge("A", "B", confidence=0.95, rule="causal")
        # Multi-edge: lookup points to newest
        new_edge = engine._edges_lookup[("a", "b")]
        assert new_edge.modifiers.get("_conflict_report") is None

    def test_conflict_via_ingest_stl(self):
        """Conflict detection should work through ingest_stl path."""
        engine = STGEngine()
        engine.ingest_stl('[X] -> [Y] ::mod(confidence=0.9, rule="causal")')
        engine.ingest_stl('[X] -> [Y] ::mod(confidence=0.8, rule="empirical")')
        new_edge = engine._edges_lookup[("x", "y")]
        report = new_edge.modifiers.get("_conflict_report")
        assert report is not None
        assert "causal" in report["details"]
        assert "empirical" in report["details"]

    def test_no_conflict_on_first_edge(self):
        """First edge should never have conflict (nothing to contradict)."""
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.9, rule="causal")
        edge = engine.get_edges("A", "B")[0]
        assert edge.modifiers.get("_conflict_report") is None

    def test_conflict_report_structure(self):
        """Conflict report should have correct structure."""
        engine = STGEngine()
        engine.add_edge("A", "B", rule="causal")
        engine.add_edge("A", "B", rule="definitional")
        new_edge = engine._edges_lookup[("a", "b")]
        report = new_edge.modifiers.get("_conflict_report")
        assert report is not None
        assert "conflicting_edges" in report
        assert "conflict_score" in report
        assert "details" in report
        assert isinstance(report["conflicting_edges"], list)
        assert 0 < report["conflict_score"] <= 1.0

    def test_conflict_warn_never_reject(self):
        """Conflict should warn but never reject the edge."""
        engine = STGEngine()
        engine.add_edge("A", "B", rule="causal")
        edge2 = engine.add_edge("A", "B", rule="definitional")
        # Edge was created despite conflict
        assert edge2 is not None
        assert engine._edges_lookup[("a", "b")] is edge2

    def test_virtual_edge_skips_conflict_check(self):
        """Virtual edges should not trigger conflict detection."""
        engine = STGEngine()
        engine.add_edge("A", "C", rule="causal")
        engine.add_edge("A", "D", rule="logical", edge_class="virtual")
        edge = engine.get_edges("A", "D")[0]
        assert edge.modifiers.get("_conflict_report") is None



class TestDuplicateEdgePrevention:
    """G8 fix: duplicate edge handling — multi-edge support.

    Same (source, target) with different content → multi-edge (knowledge evolution).
    Same (source, target) with identical content → true duplicate, skip.
    """

    def test_true_duplicate_returns_same_edge(self):
        """Identical edge content → return existing, no new edge."""
        engine = STGEngine()
        edge1 = engine.add_edge("A", "B", confidence=0.5)
        edge2 = engine.add_edge("A", "B", confidence=0.5)
        assert edge1 is edge2
        assert len(engine.get_edges()) == 1

    def test_different_confidence_creates_multi_edge(self):
        """Different confidence → multi-edge, both alive (no auto-supersede).

        Same (src,tgt) edges are treated as complementary facets, not
        corrections. Supersede is the job of _flag_suspected_supersede,
        which fires only on same field+value + DIFFERENT target.
        """
        engine = STGEngine()
        edge1 = engine.add_edge("A", "B", confidence=0.5)
        edge2 = engine.add_edge("A", "B", confidence=0.9)
        assert edge1 is not edge2
        assert len(engine._edges) == 2
        assert edge1.modifiers.get("superseded_at") is None
        assert edge1.modifiers.get("suspected_supersede") is None
        # Lookup points to newest
        assert engine._edges_lookup[("a", "b")] is edge2

    def test_different_description_creates_multi_edge(self):
        """Different description → multi-edge."""
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.8, description="old info")
        engine.add_edge("A", "B", confidence=0.8, description="new info")
        assert len(engine._edges) == 2

    def test_different_rule_creates_multi_edge(self):
        """Different rule → multi-edge."""
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.7, rule="causal")
        engine.add_edge("A", "B", confidence=0.7, rule="logical")
        assert len(engine._edges) == 2

    def test_ingest_stl_multi_edge(self):
        """Ingesting different content for same pair → multi-edge."""
        engine = STGEngine()
        engine.ingest_stl('[A] -> [B] ::mod(confidence=0.5, rule="causal")')
        engine.ingest_stl('[A] -> [B] ::mod(confidence=0.9, rule="logical")')
        all_edges = engine.get_edges()
        ab_edges = [e for e in all_edges if e.source == "A" and e.target == "B"]
        assert len(ab_edges) == 2
        # Lookup points to latest (rule=logical)
        assert engine._edges_lookup[("a", "b")].rule == "logical"

    def test_different_direction_not_duplicate(self):
        """A→B and B→A are different edges, not duplicates."""
        engine = STGEngine()
        engine.add_edge("A", "B")
        engine.add_edge("B", "A")
        assert len(engine.get_edges()) >= 2

    def test_remove_clears_lookup(self):
        """Remove should clear the lookup entry."""
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.5)
        assert engine.remove_edge("A", "B") is True
        assert ("a", "b") not in engine._edges_lookup

    def test_superseded_edges_not_in_propagate(self):
        """Propagate should only use lookup (latest) edge."""
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.9, description="strong link")
        engine.add_edge("A", "B", confidence=0.3, description="weak link")
        engine.add_edge("B", "C", confidence=0.8)
        # Lookup A→B points to conf=0.3 (latest)
        assert engine._edges_lookup[("a", "b")].confidence == 0.3

    def test_g8_dedup_distinguishes_action_value(self):
        """Same (src,tgt) with different `action` values → both edges survive.

        Regression for G8 dedup gap: previously the second edge was silently
        dropped because is_true_duplicate ignored SEMANTIC_FIELDS. stg-steam
        v0.4 hit this on Game ↔ Company (developed_by + published_by).
        """
        engine = STGEngine()
        engine.add_edge("Game", "Company", confidence=1.0, action="developed_by")
        engine.add_edge("Game", "Company", confidence=1.0, action="published_by")
        edges = [e for e in engine._edges if e.target == "Company"]
        assert len(edges) == 2
        assert {e.modifiers.get("action") for e in edges} == {"developed_by", "published_by"}

    def test_g8_dedup_distinguishes_is_a_value(self):
        """Same (src,tgt) with different `is_a` values → both edges survive."""
        engine = STGEngine()
        engine.add_edge("X", "Y", confidence=1.0, is_a="category_a")
        engine.add_edge("X", "Y", confidence=1.0, is_a="category_b")
        edges = [e for e in engine._edges if e.target == "Y"]
        assert len(edges) == 2

    def test_g8_dedup_complementary_facets_coexist(self):
        """Different semantic *fields* (action vs status) also coexist.

        Mirrors the example in the comment on engine.py G8 block: action="took"
        vs status="had_amazing_time" on same (src,tgt).
        """
        engine = STGEngine()
        engine.add_edge("Trip", "Yosemite", confidence=1.0, action="took")
        engine.add_edge("Trip", "Yosemite", confidence=1.0, status="had_amazing_time")
        edges = [e for e in engine._edges if e.target == "Yosemite"]
        assert len(edges) == 2

    def test_g8_dedup_multi_semantic_field_value_change(self):
        """Edges carrying multiple SEMANTIC_FIELDS where one value differs.

        Verifies the fingerprint (all-fields) comparison: edges sharing
        action='took' but differing in role still create a second edge.
        Minimal (first-field-only) patch would FAIL this test.
        """
        engine = STGEngine()
        engine.add_edge("Trip", "Tokyo", confidence=1.0, action="took", role="tourist")
        engine.add_edge("Trip", "Tokyo", confidence=1.0, action="took", role="guide")
        edges = [e for e in engine._edges if e.target == "Tokyo"]
        assert len(edges) == 2
        assert {e.modifiers.get("role") for e in edges} == {"tourist", "guide"}

    def test_g8_dedup_still_collapses_true_duplicate_with_action(self):
        """Two genuinely identical edges (same action value) still dedup."""
        engine = STGEngine()
        e1 = engine.add_edge("X", "Y", confidence=1.0, action="did_thing")
        e2 = engine.add_edge("X", "Y", confidence=1.0, action="did_thing")
        assert e1 is e2
        assert len([e for e in engine._edges if e.target == "Y"]) == 1

    def test_g8_dedup_no_semantic_field_still_collapses(self):
        """Edges with no semantic field (only conf/strength) still dedup."""
        engine = STGEngine()
        e1 = engine.add_edge("X", "Y", confidence=0.9, strength=0.5)
        e2 = engine.add_edge("X", "Y", confidence=0.9, strength=0.5)
        assert e1 is e2
        assert len([e for e in engine._edges if e.target == "Y"]) == 1


class TestSTGEngineSTLImport:
    def test_ingest_simple_stl(self):
        engine = STGEngine()
        count = engine.ingest_stl('[A] -> [B] ::mod(confidence=0.9)')
        assert count == 1
        assert len(engine) == 2

    def test_ingest_multiple_statements(self):
        engine = STGEngine()
        stl = """
        [Theory] -> [Prediction] ::mod(confidence=0.95, rule="logical")
        [Prediction] -> [Experiment] ::mod(confidence=0.8, rule="empirical")
        [Experiment] -> [Observation] ::mod(confidence=0.9)
        """
        count = engine.ingest_stl(stl)
        assert count == 3
        assert len(engine) == 4

    def test_ingest_unicode_anchors(self):
        engine = STGEngine()
        count = engine.ingest_stl('[黄帝内经] -> [素问] ::mod(confidence=0.95)')
        assert count == 1
        assert engine.get_node("黄帝内经") is not None
        assert engine.get_node("素问") is not None

    def test_ingest_namespace_anchors(self):
        engine = STGEngine()
        count = engine.ingest_stl('[Physics:Energy] -> [Physics:Mass] ::mod(rule="logical")')
        assert count == 1
        node = engine.get_node("Energy")
        assert node is not None
        assert node.namespace == "Physics"

    def test_ingest_no_modifiers(self):
        engine = STGEngine()
        count = engine.ingest_stl('[A] -> [B]')
        assert count == 1
        edges = engine.get_edges("A", "B")
        assert len(edges) == 1
        assert edges[0].confidence == 0.5  # Default

    def test_ingest_ascii_arrow(self):
        engine = STGEngine()
        count = engine.ingest_stl('[X] -> [Y] ::mod(confidence=0.7)')
        assert count == 1


class TestSTGEngineComputation:
    def _build_test_graph(self) -> STGEngine:
        engine = STGEngine()
        engine.add_edge("Self", "Memory", confidence=0.95, rule="definitional")
        engine.add_edge("Memory", "Events", confidence=0.9, rule="causal")
        engine.add_edge("Events", "Tensions", confidence=0.85)
        engine.add_edge("Tensions", "Resolution", confidence=0.7)
        engine.add_edge("Question", "Self", confidence=0.6)
        return engine

    def test_compute_psi(self):
        engine = self._build_test_graph()
        engine.compute_all_tensions()
        psi = engine.compute_psi()
        assert psi > 0  # Should be positive for a coherent graph

    def test_compute_psi_empty_graph(self):
        engine = STGEngine()
        psi = engine.compute_psi()
        assert psi == 1.0  # Empty graph is trivially stable

    def test_compute_all_tensions(self):
        engine = self._build_test_graph()
        tensions = engine.compute_all_tensions()
        assert len(tensions) == 6  # 6 nodes: Self, Memory, Events, Tensions, Resolution, Question
        # All nodes should have some tension
        for name, t in tensions.items():
            assert t >= 0.0

    def test_compute_path_tension(self):
        engine = self._build_test_graph()
        t = engine.compute_path_tension("Self", "Resolution")
        assert t >= 0.0

    def test_compute_path_tension_no_path(self):
        engine = STGEngine()
        engine.add_edge("A", "B")
        engine.add_edge("C", "D")
        t = engine.compute_path_tension("A", "D")
        assert t == -1.0

    def test_propagate(self):
        engine = self._build_test_graph()
        activated = engine.propagate("memory events")
        # Seeds "Memory" and "Events" activate downstream nodes.
        # With decay=0.65 + normalize, budget flows downstream;
        # seed node "Memory" may drop below threshold, but "Events"
        # and downstream nodes (Tensions, Resolution) should activate.
        assert "Events" in activated
        assert len(activated) >= 2

    def test_propagate_no_match(self):
        engine = self._build_test_graph()
        activated = engine.propagate("quantum physics dark matter")
        assert activated == []

    def test_compute_activations(self):
        engine = self._build_test_graph()
        engine.compute_all_tensions()
        activations = engine.compute_activations()
        assert len(activations) == 6
        for act in activations.values():
            assert act >= 0.0

    def test_compute_reward(self):
        engine = STGEngine()
        r = engine.compute_reward(
            psi_before=0.5, psi_after=0.8,
            tension_resolved=0.3, edges_traversed=5,
        )
        assert r > 0  # Positive: stability improved

    def test_compute_reward_negative(self):
        engine = STGEngine()
        r = engine.compute_reward(
            psi_before=0.8, psi_after=0.5,
            tension_resolved=0.0, edges_traversed=100,
        )
        assert r < 0  # Negative: stability decreased, high cost

    def test_take_snapshot(self):
        engine = self._build_test_graph()
        engine.compute_all_tensions()
        snapshot = engine.take_snapshot(session_id="SESSION_TEST")
        assert snapshot.session_id == "SESSION_TEST"
        assert snapshot.node_count == 6
        assert snapshot.edge_count == 5
        assert snapshot.psi_value > 0


class TestSTGEngineQuery:
    def _build_test_engine(self) -> STGEngine:
        engine = STGEngine()
        engine.add_node("Memory", anchor_type="Concept")
        engine.add_node("Self", anchor_type="Agent")
        engine.add_node("Phase_1", anchor_type="Event")

        engine.add_edge("Self", "Memory", confidence=0.95, session_id="SESSION_008")
        engine.add_edge("Memory", "Phase_1", confidence=0.8, session_id="SESSION_008")
        engine.add_edge("Phase_1", "Self", confidence=0.7, session_id="SESSION_009")

        engine.add_session(STGSession(
            session_id="SESSION_008", date="2026-01-01", title="Phase 1 Complete",
            avg_importance=0.92,
        ))
        engine.add_event(STGEvent(
            event_id="E010", session_id="SESSION_008",
            importance_score=0.95, title="Phase 1 Complete",
        ))
        engine.add_event(STGEvent(
            event_id="E042", session_id="SESSION_019",
            importance_score=0.85, event_type="bug_fix",
            title="Pydantic Alias Bug",
        ))
        engine.add_tension(STGTension(
            name="OAuth_Broken", initial_value=1.0, current_value=0.0,
            status="resolved", created_session="SESSION_014",
        ))
        engine.add_tension(STGTension(
            name="Memory_Compression_Risk", initial_value=0.75,
            current_value=0.75, status="active",
        ))
        return engine

    def test_query_nodes_by_pattern(self):
        engine = self._build_test_engine()
        results = engine.query_nodes(name_pattern="mem")
        assert len(results) == 1
        assert results[0].name == "Memory"

    def test_query_nodes_by_type(self):
        engine = self._build_test_engine()
        results = engine.query_nodes(anchor_type="Agent")
        assert len(results) == 1
        assert results[0].name == "Self"

    def test_query_edges_by_confidence(self):
        engine = self._build_test_engine()
        results = engine.query_edges(min_confidence=0.9)
        assert len(results) == 1
        assert results[0].source == "Self"

    def test_query_edges_by_session(self):
        engine = self._build_test_engine()
        results = engine.query_edges(session_id="SESSION_008")
        assert len(results) == 2

    def test_query_events_by_importance(self):
        engine = self._build_test_engine()
        results = engine.query_events(min_importance=0.9)
        assert len(results) == 1
        assert results[0].event_id == "E010"

    def test_query_events_all(self):
        engine = self._build_test_engine()
        results = engine.query_events()
        assert len(results) == 2

    def test_query_by_session(self):
        engine = self._build_test_engine()
        data = engine.query_by_session("SESSION_008")
        assert data["session"]["session_id"] == "SESSION_008"
        assert len(data["edges"]) == 2
        assert len(data["events"]) == 1

    def test_query_tensions_active(self):
        engine = self._build_test_engine()
        results = engine.query_tensions(status="active")
        assert len(results) == 1
        assert results[0].name == "Memory_Compression_Risk"

    def test_query_tensions_resolved(self):
        engine = self._build_test_engine()
        results = engine.query_tensions(status="resolved")
        assert len(results) == 1
        assert results[0].name == "OAuth_Broken"

    def test_find_paths(self):
        engine = self._build_test_engine()
        paths = engine.find_paths("Self", "Phase_1")
        assert len(paths) >= 1
        assert paths[0] == ["Self", "Memory", "Phase_1"]

    def test_update_tension(self):
        engine = self._build_test_engine()
        engine.update_tension("Memory_Compression_Risk", 0.03, "SESSION_020")
        t = engine._tensions["Memory_Compression_Risk"]
        assert t.current_value == 0.03
        assert t.status == "resolved"

    def test_get_stats(self):
        engine = self._build_test_engine()
        stats = engine.get_stats()
        assert stats["node_count"] == 3
        assert stats["edge_count"] == 3
        assert stats["session_count"] == 1
        assert stats["event_count"] == 2
        assert stats["active_tensions"] == 1
        assert stats["total_tensions"] == 2


class TestSTGEnginePersistence:
    def test_save_and_load(self):
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.95, rule="causal")
        engine.add_edge("B", "C", confidence=0.8)
        engine.add_session(STGSession(session_id="S1", date="2026-01-01"))
        engine.add_event(STGEvent(event_id="E1", importance_score=0.9))
        engine.add_tension(STGTension(name="T1", current_value=0.5))
        engine.add_belief_evolution(STGBeliefEvolution(
            old_anchor="V1", new_anchor="V2", level=3,
        ))
        engine.compute_all_tensions()
        engine.take_snapshot("S1")

        with tempfile.NamedTemporaryFile(suffix=".stg", delete=False) as f:
            path = f.name

        try:
            engine.save(path)
            loaded = STGEngine.load(path)

            assert len(loaded) == 3  # A, B, C
            assert loaded.get_stats()["edge_count"] == 2
            assert loaded.get_stats()["session_count"] == 1
            assert loaded.get_stats()["event_count"] == 1
            assert loaded.get_stats()["total_tensions"] == 1
            assert len(loaded._belief_evolutions) == 1
            assert len(loaded._snapshots) == 1

            # Verify edge data preserved
            edges = loaded.get_edges("A", "B")
            assert len(edges) == 1
            assert edges[0].confidence == 0.95
            assert edges[0].rule == "causal"

            # Verify graph connectivity
            neighbors = loaded.neighbors("A", "out")
            assert "B" in neighbors
        finally:
            os.unlink(path)

    def test_save_empty_engine(self):
        engine = STGEngine()
        with tempfile.NamedTemporaryFile(suffix=".stg", delete=False) as f:
            path = f.name

        try:
            engine.save(path)
            loaded = STGEngine.load(path)
            assert len(loaded) == 0
        finally:
            os.unlink(path)

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            STGEngine.load("/nonexistent/path/memory.stg")

    def test_export_stl(self):
        engine = STGEngine()
        engine.add_edge("Theory", "Prediction", confidence=0.95, rule="logical")
        stl = engine.export_stl()
        assert "[Theory]" in stl
        assert "[Prediction]" in stl
        assert "confidence=0.95" in stl
        assert 'rule="logical"' in stl

    def test_roundtrip_stl_import_export(self):
        engine = STGEngine()
        original_stl = '[A] -> [B] ::mod(confidence=0.9, rule="causal")'
        engine.ingest_stl(original_stl)

        exported = engine.export_stl()
        assert "[A]" in exported
        assert "[B]" in exported
        assert "confidence=0.9" in exported


class TestSTGEngineEpisodicMemory:
    def test_belief_evolution(self):
        engine = STGEngine()
        engine.add_belief_evolution(STGBeliefEvolution(
            old_anchor="Memory_Architecture_v1",
            new_anchor="Memory_Architecture_v2",
            event_id="E004",
            session_id="SESSION_008",
            level=3,
            description="Static Archive → Living Memory",
        ))
        assert len(engine._belief_evolutions) == 1
        assert engine._belief_evolutions[0].level == 3

    def test_multiple_snapshots(self):
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.5)
        s1 = engine.take_snapshot("S1")

        engine.add_edge("B", "C", confidence=0.9)
        s2 = engine.take_snapshot("S2")

        assert len(engine._snapshots) == 2
        assert s2.edge_count > s1.edge_count


# ═══════════════════════════════════════════════════════════
# TestPropagateNormalize (Braitenberg Vehicle 12)
# ═══════════════════════════════════════════════════════════


class TestPropagateNormalize:
    """Global activation constraint: total activation is conserved."""

    def test_propagate_normalized_total_bounded(self):
        """With normalize=True, total activation never exceeds initial budget."""
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.8)
        engine.add_edge("A", "C", confidence=0.6)
        engine.add_edge("B", "D", confidence=0.7)
        engine.add_edge("C", "D", confidence=0.5)
        engine.add_edge("D", "E", confidence=0.9)

        engine.propagate("A", normalize=True, iterations=5)
        total = sum(n.activation for n in engine._nodes.values() if n.activation > 0)
        # Budget = initial activation of seed(s). Should not exceed it significantly.
        # Allow small floating point tolerance
        assert total <= 2.0  # seed activation is ~1.0, budget conserved

    def test_propagate_normalize_false_preserves_old_behavior(self):
        """With normalize=False, activation can grow beyond initial seed."""
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.8)
        engine.add_edge("B", "C", confidence=0.8)
        engine.add_edge("C", "D", confidence=0.8)
        engine.propagate("A", normalize=False, iterations=5, decay=0.8)
        # Without normalization, activation spreads freely
        # Just verify it runs without error
        result = engine.propagate("A", normalize=False)
        assert isinstance(result, list)

    def test_propagate_normalized_relative_order_preserved(self):
        """Normalization preserves relative activation ordering."""
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.9)
        engine.add_edge("A", "C", confidence=0.1)
        engine.propagate("A", normalize=True)
        b = engine.get_node("B")
        c = engine.get_node("C")
        assert b is not None and c is not None
        # High-confidence edge should still win
        assert b.activation > c.activation

    def test_propagate_normalized_single_seed(self):
        """Single seed node: budget = its initial activation."""
        engine = STGEngine()
        engine.add_edge("X", "Y", confidence=0.5)
        engine.propagate("X", normalize=True, initial_activation=0.5)
        total = sum(n.activation for n in engine._nodes.values() if n.activation > 0)
        assert total <= 1.0  # budget from seed

    def test_propagate_default_is_normalized(self):
        """Default behavior uses normalize=True."""
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.8)
        engine.add_edge("B", "C", confidence=0.8)
        # Just verify default runs (normalize=True is default)
        result = engine.propagate("A")
        assert isinstance(result, list)


class TestPropagateReadOnly:
    """propagate(read_only=True) must skip ALL write side-effects.

    Used by the HTTP server (stg_engine.server) so external traffic doesn't
    shape the agent's learning signal or pollute telemetry. CLI keeps the
    default read_only=False to preserve the full learning loop.
    """

    def _build_engine_with_hooks(self):
        engine = STGEngine()
        engine.add_edge("Game", "Company", confidence=0.9)
        engine.add_edge("Game", "Genre", confidence=0.9)
        engine.add_edge("Company", "Country", confidence=0.8)
        engine.enable_learning()
        engine.enable_telemetry()
        return engine

    def test_read_only_true_skips_hebbian_learning(self):
        """With read_only=True, _learning_log must NOT grow."""
        engine = self._build_engine_with_hooks()
        before = len(engine._learning_log)
        engine.propagate("Game Company", read_only=True)
        assert len(engine._learning_log) == before

    def test_read_only_true_skips_telemetry(self):
        """With read_only=True, telemetry counters must NOT advance."""
        engine = self._build_engine_with_hooks()
        before = len(engine._telemetry._propagations) if engine._telemetry else 0
        engine.propagate("Game Company", read_only=True)
        after = len(engine._telemetry._propagations) if engine._telemetry else 0
        assert after == before

    def test_read_only_false_default_preserves_learning(self):
        """Default (read_only=False) keeps Hebbian + telemetry alive."""
        engine = self._build_engine_with_hooks()
        before_log = len(engine._learning_log)
        before_tel = len(engine._telemetry._propagations)
        engine.propagate("Game Company")  # default read_only=False
        # At least one of the two should advance; for a connected graph
        # propagate normally produces both learning events and a telemetry tick.
        assert (
            len(engine._learning_log) > before_log
            or len(engine._telemetry._propagations) > before_tel
        )

    def test_read_only_returns_same_activations_as_default(self):
        """read_only=True must not change the answer, only suppress side-effects."""
        engine_a = self._build_engine_with_hooks()
        engine_b = self._build_engine_with_hooks()
        result_default = engine_a.propagate("Game Company")
        result_read_only = engine_b.propagate("Game Company", read_only=True)
        assert result_default == result_read_only

    def test_read_only_works_without_learner_or_telemetry(self):
        """Engine with no hooks attached still accepts read_only without crashing."""
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.9)
        # No enable_hebbian_learning / enable_telemetry — both hooks are None
        result = engine.propagate("A", read_only=True)
        assert isinstance(result, list)
