"""Tests for Phase 7C.2: Edge Merge — memory consolidation."""

import time

import pytest

from stg_engine.engine import STGEngine
from stg_engine.merge import EdgeMerger, MergeResult
from stg_engine.types import STGEdge


@pytest.fixture
def engine():
    """Fresh STGEngine for each test."""
    return STGEngine()


# ═══════════════════════════════════════════════════════════
# TestMergeEdge
# ═══════════════════════════════════════════════════════════


class TestMergeEdge:

    def test_merge_overwrites_confidence(self, engine):
        engine.add_edge("A", "B", confidence=0.5)
        EdgeMerger.merge_edge(engine, "A", "B", confidence=0.9)
        edge = engine._edges_lookup[("a", "b")]
        assert edge.confidence == 0.9

    def test_merge_overwrites_strength(self, engine):
        engine.add_edge("A", "B", strength=0.3)
        EdgeMerger.merge_edge(engine, "A", "B", strength=0.8)
        edge = engine._edges_lookup[("a", "b")]
        assert edge.strength == 0.8

    def test_merge_overwrites_rule(self, engine):
        engine.add_edge("A", "B", rule="causal")
        EdgeMerger.merge_edge(engine, "A", "B", rule="definitional")
        edge = engine._edges_lookup[("a", "b")]
        assert edge.rule == "definitional"

    def test_merge_overwrites_description(self, engine):
        engine.add_edge("A", "B", confidence=0.5, description="old")
        EdgeMerger.merge_edge(engine, "A", "B", description="new")
        edge = engine._edges_lookup[("a", "b")]
        assert edge.modifiers["description"] == "new"

    def test_merge_preserves_unmentioned_fields(self, engine):
        engine.add_edge("A", "B", confidence=0.5, description="original", lesson="keep me")
        EdgeMerger.merge_edge(engine, "A", "B", confidence=0.9)
        edge = engine._edges_lookup[("a", "b")]
        assert edge.confidence == 0.9
        assert edge.modifiers["description"] == "original"
        assert edge.modifiers["lesson"] == "keep me"

    def test_merge_accumulates_path(self, engine):
        engine.add_edge("A", "B", confidence=0.5, path="file1.py")
        EdgeMerger.merge_edge(engine, "A", "B", path="file2.py")
        edge = engine._edges_lookup[("a", "b")]
        assert "file1.py" in edge.modifiers["path"]
        assert "file2.py" in edge.modifiers["path"]
        assert ";" in edge.modifiers["path"]

    def test_merge_accumulates_source_modifier(self, engine):
        engine.add_edge("A", "B", confidence=0.5)
        # Set source modifier directly (can't pass 'source' as kwarg — collides with positional)
        engine._edges_lookup[("a", "b")].modifiers["source"] = "doi:A"
        # Use the internal _accumulate to test accumulation logic
        from stg_engine.merge import _accumulate
        mods = engine._edges_lookup[("a", "b")].modifiers
        _accumulate(mods, "source", "doi:B")
        assert "doi:A" in mods["source"]
        assert "doi:B" in mods["source"]

    def test_merge_deduplicates_accumulated_values(self, engine):
        engine.add_edge("A", "B", confidence=0.5, path="file1.py")
        EdgeMerger.merge_edge(engine, "A", "B", path="file1.py")
        edge = engine._edges_lookup[("a", "b")]
        assert edge.modifiers["path"].count("file1.py") == 1

    def test_merge_nonexistent_edge_raises_keyerror(self, engine):
        with pytest.raises(KeyError, match="No edge"):
            EdgeMerger.merge_edge(engine, "X", "Y", confidence=0.9)

    def test_merge_does_not_change_created_at(self, engine):
        engine.add_edge("A", "B", confidence=0.5, created_at=1000.0)
        EdgeMerger.merge_edge(engine, "A", "B", confidence=0.9)
        edge = engine._edges_lookup[("a", "b")]
        assert edge.created_at == 1000.0

    def test_merge_ignores_immutable_fields(self, engine):
        engine.add_edge("A", "B", confidence=0.5)
        EdgeMerger.merge_edge(engine, "A", "B", edge_class="temporal", delay_k=5)
        edge = engine._edges_lookup[("a", "b")]
        assert edge.edge_class == "knowledge"  # unchanged
        assert edge.delay_k == 0  # unchanged

    def test_merge_arbitrary_modifiers(self, engine):
        engine.add_edge("A", "B", confidence=0.5)
        EdgeMerger.merge_edge(engine, "A", "B", custom_field="hello")
        edge = engine._edges_lookup[("a", "b")]
        assert edge.modifiers["custom_field"] == "hello"

    def test_merge_returns_edge(self, engine):
        engine.add_edge("A", "B", confidence=0.5)
        result = EdgeMerger.merge_edge(engine, "A", "B", confidence=0.9)
        assert isinstance(result, STGEdge)
        assert result.confidence == 0.9

    def test_merge_none_values_are_skipped(self, engine):
        engine.add_edge("A", "B", confidence=0.5, description="keep")
        EdgeMerger.merge_edge(engine, "A", "B", description=None)
        edge = engine._edges_lookup[("a", "b")]
        assert edge.modifiers["description"] == "keep"


# ═══════════════════════════════════════════════════════════
# TestConsolidateEdges
# ═══════════════════════════════════════════════════════════


class TestConsolidateEdges:

    def _add_multi_edge(self, engine, source, target, **mods):
        """Directly append an edge to _edges (bypass G8 dedup)."""
        edge = STGEdge(
            source=source,
            target=target,
            confidence=mods.pop("confidence", 0.5),
            strength=mods.pop("strength", 0.5),
            rule=mods.pop("rule", None),
            modifiers=mods,
            salience=mods.pop("salience", 0.5),
            created_at=mods.pop("created_at", time.time()),
        )
        engine._edges.append(edge)
        engine._edges_lookup[(source.lower(), target.lower())] = edge
        # Ensure nodes and graph edge exist
        engine.add_node(source)
        engine.add_node(target)
        if not engine._graph.has_edge(source, target):
            engine._graph.add_edge(source, target)
        return edge

    def test_consolidate_two_edges_into_one(self, engine):
        engine.add_edge("A", "B", confidence=0.5, description="first")
        self._add_multi_edge(engine, "A", "B", confidence=0.9, description="second")

        result = EdgeMerger.consolidate_edges(engine, "A", "B")
        assert result is not None
        assert result.edges_merged == 2
        # Only one edge should remain
        remaining = [e for e in engine._edges if e.source == "A" and e.target == "B"]
        assert len(remaining) == 1
        assert remaining[0].confidence == 0.9
        assert remaining[0].modifiers["description"] == "second"

    def test_consolidate_preserves_latest_overwrite_fields(self, engine):
        self._add_multi_edge(engine, "A", "B",
                             confidence=0.5, description="old", created_at=1000.0)
        self._add_multi_edge(engine, "A", "B",
                             confidence=0.9, created_at=2000.0)

        result = EdgeMerger.consolidate_edges(engine, "A", "B")
        assert result.merged_edge.confidence == 0.9
        # description from first edge preserved (second didn't override)
        assert result.merged_edge.modifiers.get("description") == "old"

    def test_consolidate_accumulates_paths(self, engine):
        self._add_multi_edge(engine, "A", "B",
                             confidence=0.5, path="file1.py", created_at=1000.0)
        self._add_multi_edge(engine, "A", "B",
                             confidence=0.5, path="file2.py", created_at=2000.0)

        result = EdgeMerger.consolidate_edges(engine, "A", "B")
        path_val = result.merged_edge.modifiers["path"]
        assert "file1.py" in path_val
        assert "file2.py" in path_val

    def test_consolidate_created_at_is_now(self, engine):
        self._add_multi_edge(engine, "A", "B",
                             confidence=0.5, created_at=1000.0)
        self._add_multi_edge(engine, "A", "B",
                             confidence=0.9, created_at=2000.0)

        before = time.time()
        result = EdgeMerger.consolidate_edges(engine, "A", "B")
        after = time.time()

        assert before <= result.merged_edge.created_at <= after

    def test_consolidate_removes_old_edges(self, engine):
        self._add_multi_edge(engine, "A", "B",
                             confidence=0.5, description="e1", created_at=1000.0)
        self._add_multi_edge(engine, "A", "B",
                             confidence=0.9, description="e2", created_at=2000.0)

        before_count = len(engine._edges)
        result = EdgeMerger.consolidate_edges(engine, "A", "B")

        # 2 removed, 1 created = net -1
        assert len(engine._edges) == before_count - 1

    def test_consolidate_single_edge_returns_none(self, engine):
        engine.add_edge("A", "B", confidence=0.5)
        result = EdgeMerger.consolidate_edges(engine, "A", "B")
        assert result is None

    def test_consolidate_no_edge_returns_none(self, engine):
        result = EdgeMerger.consolidate_edges(engine, "X", "Y")
        assert result is None

    def test_consolidate_conflicting_timestamps_raises(self, engine):
        self._add_multi_edge(engine, "A", "B",
                             confidence=0.5, timestamp="1905", created_at=1000.0)
        self._add_multi_edge(engine, "A", "B",
                             confidence=0.8, timestamp="1915", created_at=2000.0)

        with pytest.raises(ValueError, match="Conflicting timestamps"):
            EdgeMerger.consolidate_edges(engine, "A", "B")

    def test_consolidate_compatible_timestamps(self, engine):
        self._add_multi_edge(engine, "A", "B",
                             confidence=0.5, timestamp="1905", created_at=1000.0)
        self._add_multi_edge(engine, "A", "B",
                             confidence=0.9, timestamp="1905", created_at=2000.0)

        result = EdgeMerger.consolidate_edges(engine, "A", "B")
        assert result is not None
        assert result.merged_edge.modifiers["timestamp"] == "1905"

    def test_consolidate_one_with_timestamp_one_without(self, engine):
        self._add_multi_edge(engine, "A", "B",
                             confidence=0.5, timestamp="1905", created_at=1000.0)
        self._add_multi_edge(engine, "A", "B",
                             confidence=0.9, created_at=2000.0)

        result = EdgeMerger.consolidate_edges(engine, "A", "B")
        assert result is not None
        assert result.merged_edge.modifiers["timestamp"] == "1905"

    def test_consolidate_preserves_graph_connectivity(self, engine):
        self._add_multi_edge(engine, "A", "B",
                             confidence=0.5, created_at=1000.0)
        self._add_multi_edge(engine, "A", "B",
                             confidence=0.9, created_at=2000.0)

        EdgeMerger.consolidate_edges(engine, "A", "B")
        assert engine._graph.has_edge("a", "b")
        assert ("a", "b") in engine._edges_lookup

    def test_consolidate_skips_virtual_edges(self, engine):
        engine.add_edge("A", "B", confidence=0.5, description="real")
        # Add a virtual edge
        self._add_multi_edge(engine, "A", "B",
                             confidence=0.15, edge_class="virtual",
                             created_at=2000.0)

        # Should return None (only 1 non-virtual edge)
        result = EdgeMerger.consolidate_edges(engine, "A", "B")
        assert result is None

    def test_consolidate_takes_max_salience(self, engine):
        self._add_multi_edge(engine, "A", "B",
                             confidence=0.5, salience=0.3, created_at=1000.0)
        self._add_multi_edge(engine, "A", "B",
                             confidence=0.9, salience=0.8, created_at=2000.0)

        result = EdgeMerger.consolidate_edges(engine, "A", "B")
        assert result.merged_edge.salience == 0.8


# ═══════════════════════════════════════════════════════════
# TestFindMergeablePairs
# ═══════════════════════════════════════════════════════════


class TestFindMergeablePairs:

    def _add_multi_edge(self, engine, source, target, **mods):
        """Directly append an edge to _edges (bypass G8 dedup)."""
        edge = STGEdge(
            source=source,
            target=target,
            confidence=mods.pop("confidence", 0.5),
            strength=mods.pop("strength", 0.5),
            rule=mods.pop("rule", None),
            modifiers=mods,
            created_at=mods.pop("created_at", time.time()),
        )
        engine._edges.append(edge)
        engine._edges_lookup[(source.lower(), target.lower())] = edge
        engine.add_node(source)
        engine.add_node(target)
        if not engine._graph.has_edge(source, target):
            engine._graph.add_edge(source, target)
        return edge

    def test_find_no_duplicates_returns_empty(self, engine):
        engine.add_edge("A", "B", confidence=0.5)
        engine.add_edge("C", "D", confidence=0.5)
        result = EdgeMerger.find_mergeable_pairs(engine)
        assert result == []

    def test_find_multi_edges_returns_pairs(self, engine):
        self._add_multi_edge(engine, "A", "B", confidence=0.5, created_at=1000.0)
        self._add_multi_edge(engine, "A", "B", confidence=0.9, created_at=2000.0)

        result = EdgeMerger.find_mergeable_pairs(engine)
        assert len(result) == 1
        assert result[0] == ("a", "b", 2)

    def test_find_excludes_timestamp_conflicts(self, engine):
        self._add_multi_edge(engine, "A", "B",
                             confidence=0.5, timestamp="1905", created_at=1000.0)
        self._add_multi_edge(engine, "A", "B",
                             confidence=0.9, timestamp="1915", created_at=2000.0)

        result = EdgeMerger.find_mergeable_pairs(engine)
        assert result == []

    def test_find_sorted_by_count_descending(self, engine):
        # Pair with 3 edges
        self._add_multi_edge(engine, "A", "B", confidence=0.5, created_at=1000.0)
        self._add_multi_edge(engine, "A", "B", confidence=0.6, created_at=2000.0)
        self._add_multi_edge(engine, "A", "B", confidence=0.7, created_at=3000.0)
        # Pair with 2 edges
        self._add_multi_edge(engine, "C", "D", confidence=0.5, created_at=1000.0)
        self._add_multi_edge(engine, "C", "D", confidence=0.9, created_at=2000.0)

        result = EdgeMerger.find_mergeable_pairs(engine)
        assert len(result) == 2
        assert result[0][2] >= result[1][2]  # sorted descending


# ═══════════════════════════════════════════════════════════
# TestEngineIntegration
# ═══════════════════════════════════════════════════════════


class TestEngineIntegration:

    def test_engine_merge_edge_delegates(self, engine):
        engine.add_edge("A", "B", confidence=0.5)
        result = engine.merge_edge("A", "B", confidence=0.9)
        assert result.confidence == 0.9

    def test_engine_consolidate_edges_delegates(self, engine):
        engine.add_edge("A", "B", confidence=0.5, description="first")
        # Directly add second edge
        edge2 = STGEdge(
            source="A", target="B", confidence=0.9,
            modifiers={"description": "second"},
            created_at=time.time(),
        )
        engine._edges.append(edge2)
        engine._edges_lookup[("a", "b")] = edge2

        result = engine.consolidate_edges("A", "B")
        assert result is not None
        assert result.edges_merged == 2
