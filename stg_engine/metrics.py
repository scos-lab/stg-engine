"""STG Metrics — Making the circuit board observable.

Phase 7A: Measurement instruments for the Semantic Tension Graph.
All functions are pure — graph data in, numbers out. No side effects.

Metrics hierarchy:
  - Per-propagation: query_efficiency, resonance_score
  - Graph topology: graph_entropy, graph_criticality
  - Importance: compute_importance_field (PageRank-style)
  - Edge analysis: edge_information_density, compute_all_eid
  - Distributions: confidence, namespace, degree
  - Aggregate: compute_graph_metrics
"""

import math
import random
import statistics
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from stg_engine.types import STGNode, STGEdge, GraphMetrics


# ═══════════════════════════════════════════════════════════
# Propagation Metrics (per-query, inline)
# ═══════════════════════════════════════════════════════════


def query_efficiency(
    seed_count: int,
    activated_count: int,
    total_nodes: int,
) -> float:
    """Measure query routing precision.

    QE = seed_count / max(activated_count, 1)

    QE → 1.0: precise routing (minimal spread beyond seeds)
    QE → 0.0: shotgun activation (massive spread)

    Args:
        seed_count: Number of initial matching nodes
        activated_count: Nodes above threshold after propagation
        total_nodes: Total nodes in graph (context, not used in formula)

    Returns:
        Query efficiency [0.0, 1.0].
    """
    if activated_count <= 0:
        return 0.0
    return min(seed_count / activated_count, 1.0)


def resonance_score(
    max_activation: float,
    total_activation: float,
) -> float:
    """Measure activation concentration (signal vs noise).

    RS = max_activation / total_activation

    RS → 1.0: perfect resonance (all energy on one node)
    RS → 0.0: pure noise (uniform spread)

    Args:
        max_activation: Highest single-node activation
        total_activation: Sum of all activations

    Returns:
        Resonance score [0.0, 1.0].
    """
    if total_activation <= 0.0:
        return 0.0
    return min(max_activation / total_activation, 1.0)


# ═══════════════════════════════════════════════════════════
# Information Theory Metrics (graph-level)
# ═══════════════════════════════════════════════════════════


def graph_entropy(graph: nx.DiGraph) -> float:
    """Shannon entropy of the degree distribution.

    H = -Σ p(d) * log2(p(d))

    Uses total degree (in + out) for each node.
    Groups nodes by degree value and computes probability distribution.

    Low H: dominated by a few degree values (rigid structure)
    High H: many different degree values (diverse structure)

    Args:
        graph: NetworkX DiGraph

    Returns:
        Entropy in bits. 0.0 for empty or single-node graphs.
    """
    n = graph.number_of_nodes()
    if n <= 1:
        return 0.0

    # Count frequency of each degree value
    degree_counts: Dict[int, int] = {}
    for _, deg in graph.degree():
        degree_counts[deg] = degree_counts.get(deg, 0) + 1

    # Compute Shannon entropy from degree frequency distribution
    entropy = 0.0
    for count in degree_counts.values():
        p = count / n
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def graph_criticality(graph: nx.DiGraph) -> float:
    """Position on the order-chaos spectrum.

    GC = H(graph) / H_max where H_max = log2(n)

    GC → 0: too rigid (few distinct degrees)
    GC → 1: too chaotic (every node has unique degree)
    GC ≈ 0.5-0.7: edge of chaos (most adaptive)

    Args:
        graph: NetworkX DiGraph

    Returns:
        Criticality [0.0, 1.0]. 0.0 for trivial graphs.
    """
    n = graph.number_of_nodes()
    if n <= 1:
        return 0.0

    h = graph_entropy(graph)
    h_max = math.log2(n)

    if h_max <= 0:
        return 0.0

    return min(h / h_max, 1.0)


# ═══════════════════════════════════════════════════════════
# Importance Field (PageRank-style)
# ═══════════════════════════════════════════════════════════


def compute_importance_field(
    graph: nx.DiGraph,
    edges_lookup: Dict[Tuple[str, str], "STGEdge"],
    iterations: int = 50,
    damping: float = 0.85,
) -> Dict[str, float]:
    """PageRank-style importance propagation weighted by edge confidence.

    I(n) = (1-d)/N + d * Σ(confidence(e) * I(pred) / out_degree(pred))

    Unlike standard PageRank, edge confidence modulates influence:
    high-confidence edges transfer more importance.

    Args:
        graph: NetworkX DiGraph
        edges_lookup: (source, target) -> STGEdge for confidence lookup
        iterations: Convergence iterations (50 is sufficient)
        damping: Damping factor (0.85 = standard)

    Returns:
        Dict mapping node_name -> importance (sums to ~1.0).
        Empty dict for empty graph.
    """
    n = graph.number_of_nodes()
    if n == 0:
        return {}

    nodes = list(graph.nodes())
    importance = {node: 1.0 / n for node in nodes}

    # Identify dangling nodes (no outgoing edges) — they "leak" importance.
    # Standard PageRank redistributes dangling importance uniformly.
    dangling_nodes = [node for node in nodes if graph.out_degree(node) == 0]

    base_rank = (1.0 - damping) / n

    for _ in range(iterations):
        # Dangling node contribution: redistribute evenly
        dangling_sum = sum(importance[node] for node in dangling_nodes)
        dangling_contrib = damping * dangling_sum / n

        new_importance = {}
        for node in nodes:
            rank = base_rank + dangling_contrib
            for pred in graph.predecessors(node):
                edge = edges_lookup.get((pred, node))
                conf = edge.confidence if edge else 0.5
                out_deg = graph.out_degree(pred)
                if out_deg > 0:
                    rank += damping * conf * importance[pred] / out_deg
            new_importance[node] = rank
        importance = new_importance

    # Normalize: confidence weighting can cause total < 1.0.
    # Normalization preserves relative ranking (which is what matters).
    total = sum(importance.values())
    if total > 0:
        importance = {k: v / total for k, v in importance.items()}

    return importance


# ═══════════════════════════════════════════════════════════
# Edge Information Density
# ═══════════════════════════════════════════════════════════


def edge_information_density(
    graph: nx.DiGraph,
    source: str,
    target: str,
) -> float:
    """Information contribution of a single edge.

    EID(e) = |H(graph) - H(graph without e)|

    High EID: critical bridge (removing changes structure significantly)
    Low EID: redundant (alternatives exist)

    WARNING: Temporarily removes and restores the edge. Not thread-safe.
    O(n) per call. Use compute_all_eid() for batch.

    Args:
        graph: NetworkX DiGraph
        source: Edge source node name
        target: Edge target node name

    Returns:
        Information density >= 0.0. 0.0 if edge doesn't exist.
    """
    if not graph.has_edge(source, target):
        return 0.0

    h_with = graph_entropy(graph)

    # Temporarily remove edge, preserving data
    edge_data = graph.edges[source, target].copy()
    graph.remove_edge(source, target)

    h_without = graph_entropy(graph)

    # Restore edge
    graph.add_edge(source, target, **edge_data)

    return abs(h_with - h_without)


def compute_all_eid(
    graph: nx.DiGraph,
    sample_size: Optional[int] = None,
) -> Dict[Tuple[str, str], float]:
    """Batch compute EID for all edges or a random sample.

    Args:
        graph: NetworkX DiGraph
        sample_size: If set, randomly sample this many edges.
                     If None, compute for all edges.

    Returns:
        Dict mapping (source, target) -> EID value.
    """
    edges = list(graph.edges())
    if not edges:
        return {}

    if sample_size is not None and sample_size < len(edges):
        edges = random.sample(edges, sample_size)

    result = {}
    for src, tgt in edges:
        result[(src, tgt)] = edge_information_density(graph, src, tgt)

    return result


# ═══════════════════════════════════════════════════════════
# Distribution Metrics
# ═══════════════════════════════════════════════════════════


def compute_confidence_distribution(
    edges: List["STGEdge"],
    exclude_virtual: bool = True,
) -> Dict[str, float]:
    """Statistical distribution of edge confidence values.

    Args:
        edges: All STGEdge objects
        exclude_virtual: If True, exclude virtual edges from distribution

    Returns:
        Dict with: mean, median, stdev, min, max,
        high_ratio (>= 0.8), mid_ratio (0.3-0.8), low_ratio (< 0.3).
        All zeros for empty edge list.
    """
    if exclude_virtual:
        edges = [e for e in edges if e.modifiers.get("edge_class") != "virtual"]
    if not edges:
        return {
            "mean": 0.0, "median": 0.0, "stdev": 0.0,
            "min": 0.0, "max": 0.0,
            "high_ratio": 0.0, "mid_ratio": 0.0, "low_ratio": 0.0,
        }

    values = [e.confidence for e in edges]
    n = len(values)

    high = sum(1 for v in values if v >= 0.8)
    low = sum(1 for v in values if v < 0.3)
    mid = n - high - low

    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if n >= 2 else 0.0,
        "min": min(values),
        "max": max(values),
        "high_ratio": high / n,
        "mid_ratio": mid / n,
        "low_ratio": low / n,
    }


def compute_namespace_coverage(
    nodes: Dict[str, "STGNode"],
) -> Dict[str, int]:
    """Node distribution across namespaces.

    Args:
        nodes: All STGNode objects keyed by name

    Returns:
        Dict mapping namespace -> node count.
        Nodes without namespace are grouped under 'General'.
    """
    coverage: Dict[str, int] = {}
    for node in nodes.values():
        ns = node.namespace if node.namespace else "General"
        coverage[ns] = coverage.get(ns, 0) + 1

    return coverage


def compute_degree_distribution(
    graph: nx.DiGraph,
) -> Dict[str, Any]:
    """Degree distribution statistics.

    Args:
        graph: NetworkX DiGraph

    Returns:
        Dict with: avg_in, avg_out, avg_total, max_in, max_out,
        max_total, max_in_node, max_out_node, max_total_node,
        isolated_count.
    """
    n = graph.number_of_nodes()
    if n == 0:
        return {
            "avg_in": 0.0, "avg_out": 0.0, "avg_total": 0.0,
            "max_in": 0, "max_out": 0, "max_total": 0,
            "max_in_node": "", "max_out_node": "", "max_total_node": "",
            "isolated_count": 0,
        }

    in_degrees = dict(graph.in_degree())
    out_degrees = dict(graph.out_degree())
    total_degrees = dict(graph.degree())

    max_in_node = max(in_degrees, key=in_degrees.get)
    max_out_node = max(out_degrees, key=out_degrees.get)
    max_total_node = max(total_degrees, key=total_degrees.get)

    isolated = sum(1 for d in total_degrees.values() if d == 0)

    return {
        "avg_in": sum(in_degrees.values()) / n,
        "avg_out": sum(out_degrees.values()) / n,
        "avg_total": sum(total_degrees.values()) / n,
        "max_in": in_degrees[max_in_node],
        "max_out": out_degrees[max_out_node],
        "max_total": total_degrees[max_total_node],
        "max_in_node": max_in_node,
        "max_out_node": max_out_node,
        "max_total_node": max_total_node,
        "isolated_count": isolated,
    }


# ═══════════════════════════════════════════════════════════
# Aggregate Graph Metrics
# ═══════════════════════════════════════════════════════════


def compute_graph_metrics(
    graph: nx.DiGraph,
    nodes: Dict[str, "STGNode"],
    edges: List["STGEdge"],
    edges_lookup: Dict[Tuple[str, str], "STGEdge"],
) -> "GraphMetrics":
    """Compute all graph-level health metrics.

    Calls entropy, criticality, confidence distribution, namespace,
    degree, and connectivity analysis. Does NOT compute importance
    field or EID (too expensive for routine use).

    Args:
        graph: NetworkX DiGraph
        nodes: All STGNode objects
        edges: All STGEdge objects
        edges_lookup: Edge lookup dict

    Returns:
        GraphMetrics dataclass with all fields populated.
    """
    from stg_engine.types import GraphMetrics as GM

    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    # Topology
    density = nx.density(graph) if n > 0 else 0.0
    deg_dist = compute_degree_distribution(graph)

    # Information theory
    h = graph_entropy(graph)
    gc = graph_criticality(graph)

    # Confidence distribution
    conf_dist = compute_confidence_distribution(edges)

    # Connectivity
    if n > 0:
        wcc = list(nx.weakly_connected_components(graph))
        num_wcc = len(wcc)
        largest = max(len(c) for c in wcc) if wcc else 0
        largest_ratio = largest / n
    else:
        num_wcc = 0
        largest_ratio = 0.0

    # Namespace coverage
    ns_coverage = compute_namespace_coverage(nodes)

    return GM(
        node_count=n,
        edge_count=m,
        density=density,
        avg_degree=deg_dist["avg_total"],
        max_degree=deg_dist["max_total"],
        max_degree_node=deg_dist["max_total_node"],
        entropy=h,
        criticality=gc,
        confidence_mean=conf_dist["mean"],
        confidence_median=conf_dist["median"],
        confidence_stdev=conf_dist["stdev"],
        high_confidence_ratio=conf_dist["high_ratio"],
        low_confidence_ratio=conf_dist["low_ratio"],
        weakly_connected_components=num_wcc,
        largest_component_ratio=largest_ratio,
        namespace_count=len(ns_coverage),
        namespaces=ns_coverage,
    )
