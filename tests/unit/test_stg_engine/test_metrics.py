"""Tests for STG Metrics System (Phase 7A)."""

import math
import pytest
import networkx as nx

from stg_engine.engine import STGEngine
from stg_engine.types import PropagationMetrics, GraphMetrics
from stg_engine.metrics import (
    query_efficiency,
    resonance_score,
    graph_entropy,
    graph_criticality,
    compute_importance_field,
    edge_information_density,
    compute_all_eid,
    compute_confidence_distribution,
    compute_namespace_coverage,
    compute_degree_distribution,
    compute_graph_metrics,
)


# ═══════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════


@pytest.fixture
def empty_engine():
    """Empty STGEngine."""
    return STGEngine()


@pytest.fixture
def simple_engine():
    """Engine with a small known graph.

    A -> B (0.9)
    A -> C (0.5)
    B -> D (0.8)
    C -> D (0.3)
    D -> E (0.7)
    """
    engine = STGEngine()
    engine.add_edge("A", "B", confidence=0.9)
    engine.add_edge("A", "C", confidence=0.5)
    engine.add_edge("B", "D", confidence=0.8)
    engine.add_edge("C", "D", confidence=0.3)
    engine.add_edge("D", "E", confidence=0.7)
    return engine


@pytest.fixture
def star_engine():
    """Star topology: Hub -> Leaf_0..Leaf_9."""
    engine = STGEngine()
    for i in range(10):
        engine.add_edge("Hub", f"Leaf_{i}", confidence=0.8)
    return engine


@pytest.fixture
def namespaced_engine():
    """Engine with multiple namespaces."""
    engine = STGEngine()
    engine.add_node("Store", namespace="Memory")
    engine.add_node("Retrieve", namespace="Memory")
    engine.add_node("Parser", namespace="STL")
    engine.add_node("Validator", namespace="STL")
    engine.add_node("General_Node")
    engine.add_edge("Store", "Retrieve", confidence=0.9)
    engine.add_edge("Parser", "Validator", confidence=0.8)
    engine.add_edge("Store", "Parser", confidence=0.5)
    return engine


@pytest.fixture
def simple_digraph():
    """Plain NetworkX DiGraph for pure function tests."""
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("D", "E")])
    return g


@pytest.fixture
def star_digraph():
    """Star topology DiGraph."""
    g = nx.DiGraph()
    for i in range(10):
        g.add_edge("Hub", f"Leaf_{i}")
    return g


@pytest.fixture
def cycle_digraph():
    """Cycle: A -> B -> C -> D -> A."""
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")])
    return g


# ═══════════════════════════════════════════════════════════
# Test: Query Efficiency
# ═══════════════════════════════════════════════════════════


class TestQueryEfficiency:
    def test_perfect_efficiency(self):
        """seeds == activated → QE = 1.0."""
        assert query_efficiency(5, 5, 100) == 1.0

    def test_zero_activated(self):
        """No activation → QE = 0.0."""
        assert query_efficiency(3, 0, 100) == 0.0

    def test_spreading(self):
        """activated > seeds → QE < 1.0."""
        qe = query_efficiency(2, 10, 100)
        assert 0.0 < qe < 1.0
        assert qe == pytest.approx(0.2)

    def test_bounded_to_one(self):
        """QE never exceeds 1.0, even if seeds > activated (edge case)."""
        assert query_efficiency(10, 5, 100) == 1.0


# ═══════════════════════════════════════════════════════════
# Test: Resonance Score
# ═══════════════════════════════════════════════════════════


class TestResonanceScore:
    def test_perfect_resonance(self):
        """All activation on one node → RS = 1.0."""
        assert resonance_score(5.0, 5.0) == 1.0

    def test_zero_total(self):
        """No activation → RS = 0.0."""
        assert resonance_score(0.0, 0.0) == 0.0

    def test_uniform_low_resonance(self):
        """Equal activation across many nodes → low RS."""
        # 10 nodes each with 1.0 → max=1.0, total=10.0
        rs = resonance_score(1.0, 10.0)
        assert rs == pytest.approx(0.1)

    def test_bounded(self):
        """RS is always in [0.0, 1.0]."""
        assert 0.0 <= resonance_score(3.0, 10.0) <= 1.0
        assert 0.0 <= resonance_score(0.0, 0.0) <= 1.0
        assert 0.0 <= resonance_score(10.0, 10.0) <= 1.0


# ═══════════════════════════════════════════════════════════
# Test: Graph Entropy
# ═══════════════════════════════════════════════════════════


class TestGraphEntropy:
    def test_empty_graph(self):
        g = nx.DiGraph()
        assert graph_entropy(g) == 0.0

    def test_single_node(self):
        g = nx.DiGraph()
        g.add_node("A")
        assert graph_entropy(g) == 0.0

    def test_two_nodes_one_edge(self):
        g = nx.DiGraph()
        g.add_edge("A", "B")
        h = graph_entropy(g)
        assert h >= 0.0

    def test_star_topology(self, star_digraph):
        """Star has low diversity: hub has high degree, leaves have degree 1."""
        h = graph_entropy(star_digraph)
        assert h > 0.0
        # Star with 11 nodes: 10 leaves (degree 1) + 1 hub (degree 10)
        # Only 2 distinct degree values → relatively low entropy

    def test_cycle_topology(self, cycle_digraph):
        """Cycle: every node has degree 2 → only 1 distinct degree → entropy = 0."""
        h = graph_entropy(cycle_digraph)
        assert h == 0.0  # All nodes same degree → zero entropy

    def test_entropy_increases_with_diversity(self):
        """More diverse degree distribution → higher entropy."""
        # Graph 1: uniform degree (chain)
        g1 = nx.DiGraph()
        g1.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])

        # Graph 2: varied degree (hub + chain)
        g2 = nx.DiGraph()
        g2.add_edges_from([
            ("A", "B"), ("A", "C"), ("A", "D"),  # hub
            ("B", "E"),                            # chain
        ])

        h1 = graph_entropy(g1)
        h2 = graph_entropy(g2)
        # g2 has more diverse degrees, so higher entropy
        assert h2 > h1


# ═══════════════════════════════════════════════════════════
# Test: Graph Criticality
# ═══════════════════════════════════════════════════════════


class TestGraphCriticality:
    def test_empty_graph(self):
        g = nx.DiGraph()
        assert graph_criticality(g) == 0.0

    def test_single_node(self):
        g = nx.DiGraph()
        g.add_node("A")
        assert graph_criticality(g) == 0.0

    def test_range_bounded(self, simple_digraph):
        gc = graph_criticality(simple_digraph)
        assert 0.0 <= gc <= 1.0

    def test_star_lower_than_diverse(self):
        """Star (2 distinct degrees) has lower criticality than diverse graph."""
        star = nx.DiGraph()
        for i in range(20):
            star.add_edge("Hub", f"L{i}")

        diverse = nx.DiGraph()
        # Create varied topology
        for i in range(20):
            diverse.add_edge(f"N{i}", f"N{(i+1) % 20}")
            if i % 3 == 0:
                diverse.add_edge(f"N{i}", f"N{(i+5) % 20}")
            if i % 5 == 0:
                diverse.add_edge(f"N{i}", f"N{(i+7) % 20}")

        gc_star = graph_criticality(star)
        gc_diverse = graph_criticality(diverse)
        assert gc_star < gc_diverse


# ═══════════════════════════════════════════════════════════
# Test: Importance Field
# ═══════════════════════════════════════════════════════════


class TestImportanceField:
    def test_empty_graph(self):
        g = nx.DiGraph()
        assert compute_importance_field(g, {}) == {}

    def test_single_node(self):
        g = nx.DiGraph()
        g.add_node("A")
        imp = compute_importance_field(g, {})
        assert "A" in imp
        assert imp["A"] == pytest.approx(1.0, abs=0.01)

    def test_leaves_receive_importance_from_hub(self, star_engine):
        """In Hub→Leaf star, leaves receive importance from hub.

        In directed PageRank, nodes that RECEIVE links get importance.
        Hub has only out-edges, so leaves get hub's importance via edges.
        Leaves are also dangling (no out-edges), so they redistribute
        importance evenly, giving hub indirect importance too.
        """
        imp = compute_importance_field(
            star_engine._graph, star_engine._edges_lookup
        )
        # All leaves should have similar importance (symmetric)
        leaf_values = [imp[f"leaf_{i}"] for i in range(10)]
        assert max(leaf_values) - min(leaf_values) < 0.01

    def test_confidence_weights_matter(self):
        """Higher confidence edge should transfer more importance."""
        engine = STGEngine()
        # A -> Target with high confidence
        engine.add_edge("A", "Target", confidence=0.95)
        # B -> Target with low confidence
        engine.add_edge("B", "Target", confidence=0.1)

        imp = compute_importance_field(
            engine._graph, engine._edges_lookup
        )
        # Target receives more from A than B, but both contribute
        assert imp["target"] > imp["a"]
        assert imp["target"] > imp["b"]

    def test_sums_to_approximately_one(self, simple_engine):
        imp = compute_importance_field(
            simple_engine._graph, simple_engine._edges_lookup
        )
        total = sum(imp.values())
        assert total == pytest.approx(1.0, abs=0.05)

    def test_convergence(self, simple_engine):
        """50 iterations and 100 iterations should give similar results."""
        imp50 = compute_importance_field(
            simple_engine._graph, simple_engine._edges_lookup,
            iterations=50,
        )
        imp100 = compute_importance_field(
            simple_engine._graph, simple_engine._edges_lookup,
            iterations=100,
        )
        for node in imp50:
            assert imp50[node] == pytest.approx(imp100[node], abs=0.001)

    def test_damping_factor_effect(self, simple_engine):
        """Different damping factors produce different distributions."""
        imp_high = compute_importance_field(
            simple_engine._graph, simple_engine._edges_lookup,
            damping=0.95,
        )
        imp_low = compute_importance_field(
            simple_engine._graph, simple_engine._edges_lookup,
            damping=0.5,
        )
        # Both should sum to ~1.0
        assert sum(imp_high.values()) == pytest.approx(1.0, abs=0.05)
        assert sum(imp_low.values()) == pytest.approx(1.0, abs=0.05)
        # With lower damping, base_rank dominates → more uniform
        vals_high = sorted(imp_high.values())
        vals_low = sorted(imp_low.values())
        # Range (max-min) should be smaller for low damping
        range_high = vals_high[-1] - vals_high[0]
        range_low = vals_low[-1] - vals_low[0]
        assert range_low < range_high


# ═══════════════════════════════════════════════════════════
# Test: Edge Information Density
# ═══════════════════════════════════════════════════════════


class TestEdgeInformationDensity:
    def test_edge_creating_unique_degree_has_eid(self):
        """Edge that creates a unique degree value has measurable EID.

        EID measures degree-distribution entropy change. An edge whose
        removal creates or destroys a unique degree value changes entropy.
        """
        g = nx.DiGraph()
        # All degree-2 nodes: A→B→C→D (chain)
        g.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])
        # Add extra edge from A to make A degree-2 (asymmetric)
        g.add_edge("A", "C")
        # Now removing A→C changes A's out-degree from 2→1
        eid = edge_information_density(g, "A", "C")
        assert eid > 0.0

    def test_nonexistent_edge(self, simple_digraph):
        """EID for non-existent edge returns 0.0."""
        assert edge_information_density(simple_digraph, "A", "E") == 0.0

    def test_non_negative(self, simple_digraph):
        """EID is always >= 0."""
        for src, tgt in list(simple_digraph.edges()):
            eid = edge_information_density(simple_digraph, src, tgt)
            assert eid >= 0.0, f"EID({src}->{tgt}) = {eid} < 0"

    def test_edge_restored(self, simple_digraph):
        """Graph should be unchanged after EID computation."""
        edges_before = set(simple_digraph.edges())
        edge_information_density(simple_digraph, "A", "B")
        edges_after = set(simple_digraph.edges())
        assert edges_before == edges_after


# ═══════════════════════════════════════════════════════════
# Test: Compute All EID
# ═══════════════════════════════════════════════════════════


class TestComputeAllEid:
    def test_empty_graph(self):
        g = nx.DiGraph()
        assert compute_all_eid(g) == {}

    def test_all_edges(self, simple_digraph):
        result = compute_all_eid(simple_digraph)
        assert len(result) == simple_digraph.number_of_edges()

    def test_sampled(self, simple_digraph):
        result = compute_all_eid(simple_digraph, sample_size=2)
        assert len(result) == 2


# ═══════════════════════════════════════════════════════════
# Test: Confidence Distribution
# ═══════════════════════════════════════════════════════════


class TestConfidenceDistribution:
    def test_empty_edges(self):
        dist = compute_confidence_distribution([])
        assert dist["mean"] == 0.0
        assert dist["high_ratio"] == 0.0

    def test_all_high(self, simple_engine):
        """Create engine with all high-confidence edges."""
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.9)
        engine.add_edge("B", "C", confidence=0.85)
        engine.add_edge("C", "D", confidence=0.95)
        dist = compute_confidence_distribution(engine._edges)
        assert dist["high_ratio"] == 1.0
        assert dist["low_ratio"] == 0.0
        assert dist["mean"] == pytest.approx(0.9, abs=0.05)

    def test_mixed(self, simple_engine):
        dist = compute_confidence_distribution(simple_engine._edges)
        assert 0.0 < dist["mean"] < 1.0
        assert dist["min"] <= dist["median"] <= dist["max"]

    def test_statistics_correct(self):
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.2)  # low
        engine.add_edge("B", "C", confidence=0.5)  # mid
        engine.add_edge("C", "D", confidence=0.9)  # high
        dist = compute_confidence_distribution(engine._edges)
        assert dist["low_ratio"] == pytest.approx(1 / 3, abs=0.01)
        assert dist["mid_ratio"] == pytest.approx(1 / 3, abs=0.01)
        assert dist["high_ratio"] == pytest.approx(1 / 3, abs=0.01)


# ═══════════════════════════════════════════════════════════
# Test: Namespace Coverage
# ═══════════════════════════════════════════════════════════


class TestNamespaceCoverage:
    def test_empty(self, empty_engine):
        cov = compute_namespace_coverage(empty_engine._nodes)
        assert cov == {}

    def test_single_namespace(self):
        engine = STGEngine()
        engine.add_node("A", namespace="Test")
        engine.add_node("B", namespace="Test")
        cov = compute_namespace_coverage(engine._nodes)
        assert cov == {"Test": 2}

    def test_mixed_namespaces(self, namespaced_engine):
        cov = compute_namespace_coverage(namespaced_engine._nodes)
        assert cov["Memory"] == 2
        assert cov["STL"] == 2
        assert cov["General"] == 1

    def test_no_namespace_is_general(self):
        engine = STGEngine()
        engine.add_node("Orphan")
        cov = compute_namespace_coverage(engine._nodes)
        assert "General" in cov
        assert cov["General"] == 1


# ═══════════════════════════════════════════════════════════
# Test: Degree Distribution
# ═══════════════════════════════════════════════════════════


class TestDegreeDistribution:
    def test_empty(self):
        g = nx.DiGraph()
        dist = compute_degree_distribution(g)
        assert dist["avg_total"] == 0.0
        assert dist["isolated_count"] == 0

    def test_simple_graph(self, simple_digraph):
        dist = compute_degree_distribution(simple_digraph)
        assert dist["avg_total"] > 0
        assert dist["max_total"] > 0
        assert dist["max_total_node"] in simple_digraph.nodes()

    def test_isolated_count(self):
        g = nx.DiGraph()
        g.add_node("Isolated")
        g.add_edge("A", "B")
        dist = compute_degree_distribution(g)
        assert dist["isolated_count"] == 1


# ═══════════════════════════════════════════════════════════
# Test: Graph Metrics (aggregate)
# ═══════════════════════════════════════════════════════════


class TestGraphMetrics:
    def test_all_fields_populated(self, simple_engine):
        gm = compute_graph_metrics(
            simple_engine._graph, simple_engine._nodes,
            simple_engine._edges, simple_engine._edges_lookup,
        )
        assert isinstance(gm, GraphMetrics)
        assert gm.node_count == 5
        assert gm.edge_count == 5
        assert gm.density > 0
        assert gm.entropy >= 0
        assert 0.0 <= gm.criticality <= 1.0
        assert 0.0 <= gm.confidence_mean <= 1.0
        assert gm.weakly_connected_components >= 1
        assert gm.namespace_count >= 1

    def test_empty_graph(self, empty_engine):
        gm = compute_graph_metrics(
            empty_engine._graph, empty_engine._nodes,
            empty_engine._edges, empty_engine._edges_lookup,
        )
        assert gm.node_count == 0
        assert gm.edge_count == 0
        assert gm.entropy == 0.0
        assert gm.criticality == 0.0

    def test_consistency(self, simple_engine):
        """Aggregate metrics should agree with individual functions."""
        gm = compute_graph_metrics(
            simple_engine._graph, simple_engine._nodes,
            simple_engine._edges, simple_engine._edges_lookup,
        )
        h = graph_entropy(simple_engine._graph)
        gc = graph_criticality(simple_engine._graph)
        assert gm.entropy == pytest.approx(h)
        assert gm.criticality == pytest.approx(gc)


# ═══════════════════════════════════════════════════════════
# Test: Engine Integration
# ═══════════════════════════════════════════════════════════


class TestEngineMetricsIntegration:
    def test_propagation_metrics_none_before_propagate(self, simple_engine):
        assert simple_engine.last_propagation_metrics is None

    def test_propagation_metrics_stored(self, simple_engine):
        simple_engine.propagate("A")
        pm = simple_engine.last_propagation_metrics
        assert pm is not None
        assert isinstance(pm, PropagationMetrics)
        assert pm.input_text == "A"

    def test_propagation_metrics_qe_and_rs(self, simple_engine):
        simple_engine.propagate("A")
        pm = simple_engine.last_propagation_metrics
        assert 0.0 <= pm.query_efficiency <= 1.0
        assert 0.0 <= pm.resonance_score <= 1.0
        assert pm.seed_node_count > 0
        assert pm.activated_node_count > 0

    def test_propagation_metrics_coverage(self, simple_engine):
        simple_engine.propagate("A")
        pm = simple_engine.last_propagation_metrics
        assert 0.0 <= pm.coverage <= 1.0

    def test_propagation_metrics_top_nodes(self, simple_engine):
        simple_engine.propagate("A")
        pm = simple_engine.last_propagation_metrics
        assert len(pm.top_nodes) > 0
        # Top nodes should be sorted by activation descending
        activations = [a for _, a in pm.top_nodes]
        assert activations == sorted(activations, reverse=True)

    def test_get_metrics_returns_graph_metrics(self, simple_engine):
        gm = simple_engine.get_metrics()
        assert isinstance(gm, GraphMetrics)
        assert gm.node_count == 5

    def test_get_metrics_cached(self, simple_engine):
        gm1 = simple_engine.get_metrics()
        gm2 = simple_engine.get_metrics()
        assert gm1 is gm2  # Same object — cached

    def test_cache_invalidated_on_add_edge(self, simple_engine):
        gm1 = simple_engine.get_metrics()
        simple_engine.add_edge("E", "F", confidence=0.6)
        gm2 = simple_engine.get_metrics()
        assert gm1 is not gm2
        assert gm2.node_count == 6

    def test_cache_invalidated_on_add_node(self, simple_engine):
        gm1 = simple_engine.get_metrics()
        simple_engine.add_node("NewNode")
        gm2 = simple_engine.get_metrics()
        assert gm1 is not gm2

    def test_cache_invalidated_on_ingest(self, simple_engine):
        gm1 = simple_engine.get_metrics()
        simple_engine.ingest_stl("[X] -> [Y] ::mod(confidence=0.7)")
        gm2 = simple_engine.get_metrics()
        assert gm1 is not gm2

    def test_importance_field_cached(self, simple_engine):
        imp1 = simple_engine.get_importance_field()
        imp2 = simple_engine.get_importance_field()
        assert imp1 is imp2

    def test_importance_field_invalidated(self, simple_engine):
        imp1 = simple_engine.get_importance_field()
        simple_engine.add_edge("E", "F", confidence=0.5)
        imp2 = simple_engine.get_importance_field()
        assert imp1 is not imp2

    def test_no_propagation_empty_input(self, simple_engine):
        """Empty input → no metrics stored."""
        simple_engine.propagate("")
        # propagate("") returns [] and should not crash metrics
        # Metrics might be None or have 0 values
        pm = simple_engine.last_propagation_metrics
        # Empty input returns early from propagate, so metrics unchanged
        assert pm is None
