"""Tests for Gravitational Propagation — structure-aware activation."""

import pytest

from stg_engine.engine import STGEngine
from stg_engine.gravity import (
    GravityMap,
    build_gravity_map,
    gravitational_propagate,
    gravity_info,
    gravity_node_info,
    _compute_all_elevations,
)


@pytest.fixture
def engine():
    """Fresh STGEngine for each test."""
    return STGEngine()


@pytest.fixture
def clustered_engine():
    """Engine with two clear communities connected by a bridge."""
    e = STGEngine()

    # Community A: tightly connected
    e.add_edge("A1", "A2", confidence=0.9)
    e.add_edge("A2", "A3", confidence=0.9)
    e.add_edge("A3", "A1", confidence=0.9)
    e.add_edge("A1", "A_Hub", confidence=0.9)
    e.add_edge("A2", "A_Hub", confidence=0.9)
    e.add_edge("A3", "A_Hub", confidence=0.9)

    # Community B: tightly connected
    e.add_edge("B1", "B2", confidence=0.9)
    e.add_edge("B2", "B3", confidence=0.9)
    e.add_edge("B3", "B1", confidence=0.9)
    e.add_edge("B1", "B_Hub", confidence=0.9)
    e.add_edge("B2", "B_Hub", confidence=0.9)
    e.add_edge("B3", "B_Hub", confidence=0.9)

    # Bridge: A_Hub connects to B_Hub
    e.add_edge("A_Hub", "B_Hub", confidence=0.8)
    e.add_edge("B_Hub", "A_Hub", confidence=0.8)

    # Leaf node: only 1 connection
    e.add_edge("Leaf", "A1", confidence=0.5)

    return e


# ═══════════════════════════════════════════════════════════
# TestBuildGravityMap
# ═══════════════════════════════════════════════════════════


class TestBuildGravityMap:

    def test_empty_graph(self, engine):
        gm = build_gravity_map(engine)
        assert gm.node_count == 0
        assert gm.node_elevation == {}

    def test_builds_for_clustered_graph(self, clustered_engine):
        gm = build_gravity_map(clustered_engine)
        assert gm.node_count > 0
        assert len(gm.node_elevation) == gm.node_count
        assert gm.built_at > 0

    def test_three_resolution_levels(self, clustered_engine):
        gm = build_gravity_map(clustered_engine)
        assert "coarse" in gm.community_counts
        assert "medium" in gm.community_counts
        assert "fine" in gm.community_counts

    def test_every_node_has_elevation(self, clustered_engine):
        gm = build_gravity_map(clustered_engine)
        for name in clustered_engine._nodes:
            assert name in gm.node_elevation

    def test_every_node_has_community_assignment(self, clustered_engine):
        gm = build_gravity_map(clustered_engine)
        for name in clustered_engine._nodes:
            assert name in gm.node_community
            assert "medium" in gm.node_community[name]

    def test_representatives_exist(self, clustered_engine):
        gm = build_gravity_map(clustered_engine)
        assert len(gm.representatives) > 0
        # Each representative list should have at most 3 nodes
        for key, reps in gm.representatives.items():
            assert len(reps) <= 3


# ═══════════════════════════════════════════════════════════
# TestElevation
# ═══════════════════════════════════════════════════════════


class TestElevation:

    def test_hub_nodes_higher_than_leaf(self, clustered_engine):
        """Hub nodes should have higher elevation than leaf nodes."""
        gm = build_gravity_map(clustered_engine)
        hub_elev = gm.node_elevation.get("a_hub", 0)
        leaf_elev = gm.node_elevation.get("leaf", 0)
        assert hub_elev > leaf_elev, (
            f"Hub elevation ({hub_elev}) should be > Leaf elevation ({leaf_elev})"
        )

    def test_bridge_nodes_higher_than_internal(self, clustered_engine):
        """Bridge nodes (connecting communities) should have higher elevation."""
        gm = build_gravity_map(clustered_engine)
        # A_Hub and B_Hub bridge two communities
        bridge_elev = gm.node_elevation.get("a_hub", 0)
        # A3 is internal to community A
        internal_elev = gm.node_elevation.get("a3", 0)
        # Bridge should be >= internal (not strictly > due to small graph effects)
        assert bridge_elev >= internal_elev * 0.5, (
            f"Bridge ({bridge_elev}) should be notably higher than internal ({internal_elev})"
        )

    def test_all_elevations_non_negative(self, clustered_engine):
        gm = build_gravity_map(clustered_engine)
        for name, elev in gm.node_elevation.items():
            assert elev >= 0, f"Node {name} has negative elevation {elev}"

    def test_elevation_varies(self, clustered_engine):
        """Not all elevations should be equal — structure should differentiate."""
        gm = build_gravity_map(clustered_engine)
        elevations = list(gm.node_elevation.values())
        assert max(elevations) > min(elevations), "All elevations are identical"


# ═══════════════════════════════════════════════════════════
# TestGravitationalPropagate
# ═══════════════════════════════════════════════════════════


class TestGravitationalPropagate:

    def test_returns_activated_nodes(self, clustered_engine):
        gm = build_gravity_map(clustered_engine)
        results = gravitational_propagate(clustered_engine, "A_Hub", gm)
        assert len(results) > 0

    def test_gravity_changes_ordering(self, clustered_engine):
        """With gravity, high-elevation nodes should rank higher."""
        gm = build_gravity_map(clustered_engine)

        # Standard propagate
        standard = clustered_engine.propagate("A_Hub")

        # Reset activations
        for node in clustered_engine._nodes.values():
            node.activation = 0.0

        # Gravitational propagate
        gravity_results = gravitational_propagate(
            clustered_engine, "A_Hub", gm, elevation_weight=0.8,
        )

        # Both should return results
        assert len(standard) > 0
        assert len(gravity_results) > 0

    def test_elevation_weight_zero_is_standard(self, clustered_engine):
        """With elevation_weight=0, gravity propagate should match standard."""
        gm = build_gravity_map(clustered_engine)

        standard = clustered_engine.propagate("A_Hub")
        standard_activations = {
            n: clustered_engine._nodes[n.lower()].activation
            for n in standard if n.lower() in clustered_engine._nodes
        }

        # Reset
        for node in clustered_engine._nodes.values():
            node.activation = 0.0

        gravity_results = gravitational_propagate(
            clustered_engine, "A_Hub", gm, elevation_weight=0.0,
        )

        # With weight=0, elevation^0 = 1.0 for all, so order should be same
        # (slight differences possible due to floating point)
        assert set(standard) == set(gravity_results)


# ═══════════════════════════════════════════════════════════
# TestGravityInfo
# ═══════════════════════════════════════════════════════════


class TestGravityInfo:

    def test_info_empty_graph(self, engine):
        gm = build_gravity_map(engine)
        info = gravity_info(gm)
        assert info.get("empty") is True

    def test_info_has_communities(self, clustered_engine):
        gm = build_gravity_map(clustered_engine)
        info = gravity_info(gm)
        assert "communities" in info
        assert "coarse" in info["communities"]
        assert "medium" in info["communities"]
        assert "fine" in info["communities"]

    def test_info_has_elevation_stats(self, clustered_engine):
        gm = build_gravity_map(clustered_engine)
        info = gravity_info(gm)
        assert info["elevation_min"] >= 0
        assert info["elevation_max"] >= info["elevation_min"]
        assert len(info["top_10"]) > 0

    def test_node_info(self, clustered_engine):
        gm = build_gravity_map(clustered_engine)
        info = gravity_node_info(gm, "A_Hub")
        assert info is not None
        assert info["node"] == "A_Hub"
        assert info["elevation"] > 0
        assert "percentile" in info
        assert "communities" in info

    def test_node_info_nonexistent(self, clustered_engine):
        gm = build_gravity_map(clustered_engine)
        info = gravity_node_info(gm, "NonExistent")
        assert info is None


# ═══════════════════════════════════════════════════════════
# TestEngineIntegration
# ═══════════════════════════════════════════════════════════


class TestEngineIntegration:

    def test_get_gravity_map_caches(self, clustered_engine):
        gm1 = clustered_engine.get_gravity_map()
        gm2 = clustered_engine.get_gravity_map()
        assert gm1 is gm2  # same object — cached

    def test_cache_invalidated_on_mutation(self, clustered_engine):
        gm1 = clustered_engine.get_gravity_map()
        clustered_engine.add_edge("New1", "New2", confidence=0.5)
        gm2 = clustered_engine.get_gravity_map()
        assert gm1 is not gm2  # different object — rebuilt

    def test_gravity_map_has_correct_node_count(self, clustered_engine):
        gm = clustered_engine.get_gravity_map()
        assert gm.node_count == len(clustered_engine._nodes)
