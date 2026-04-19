"""STG computation formulas.

Implements the 4 core formulas from STG_FORMULAS_REFERENCE.md:
  3.1: Ψ (Psi) — Mental Stability Metric
  3.2: Discrete Tension Calculus
  3.4: Activation Function for Attention Allocation
  3.5: Intrinsic Reward for Self-Learning

These formulas operate directly on NetworkX graph + STGNode/STGEdge data.
They are called by STGEngine methods — the engine IS the computation.
"""

import random
from typing import Dict, List, Optional, Tuple

import networkx as nx

from stg_engine.types import STGNode, STGEdge


# ═══════════════════════════════════════════════════════════════
# Formula 3.1: Ψ (Psi) — Mental Stability Metric
# ═══════════════════════════════════════════════════════════════
#
# Ψ_SKC = Structural_Coherence / max(Max_Tension, ε) * Epistemic_Confidence
#
# Where:
#   Structural_Coherence = (edges with confidence >= 0.8) / total_edges
#   Max_Tension = maximum tension value in current graph
#   Epistemic_Confidence = weighted average confidence across edges


def compute_psi(
    graph: nx.DiGraph,
    nodes: Dict[str, STGNode],
    edges: List[STGEdge],
) -> float:
    """Calculate Ψ (Mental Stability).

    High Ψ = coherent, low-tension, confident cognitive state.
    Low Ψ = incoherent, high-tension, uncertain state.

    Args:
        graph: NetworkX directed graph
        nodes: All STGNodes
        edges: All STGEdges

    Returns:
        Ψ value (0.0+). Higher is more stable.
    """
    # Filter out virtual edges — Ψ measures real knowledge stability only
    real_edges = [e for e in edges if e.modifiers.get("edge_class") != "virtual"]

    if not real_edges:
        return 1.0  # Empty graph is trivially stable

    # Structural Coherence: ratio of validated (high-confidence) edges
    high_conf_count = sum(1 for e in real_edges if e.confidence >= 0.8)
    structural_coherence = high_conf_count / len(real_edges)

    # Max Tension: highest tension in current graph
    max_tension = max((n.tension for n in nodes.values()), default=0.0)

    # Epistemic Confidence: average confidence across all edges
    epistemic_confidence = sum(e.confidence for e in real_edges) / len(real_edges)

    # Ψ = Coherence / max(Tension, ε) * Confidence
    # Use ε=0.01 to avoid division by zero
    psi = structural_coherence / max(max_tension, 0.01) * epistemic_confidence

    return psi


# ═══════════════════════════════════════════════════════════════
# Formula 3.2: Discrete Tension Calculus
# ═══════════════════════════════════════════════════════════════
#
# Tension_path(A→B) = Σ [Conflict(i,j) * Uncertainty(i,j)] over path
#
# Where:
#   Conflict(i,j) = semantic incompatibility measure
#   Uncertainty(i,j) = 1 - confidence_ij


def compute_edge_tension(edge: STGEdge) -> float:
    """Calculate tension for a single edge.

    Tension = Conflict * Uncertainty.
    For now, conflict is derived from modifier inconsistencies
    and low confidence. Future versions may use semantic distance.

    Args:
        edge: The STGEdge to evaluate

    Returns:
        Tension value (0.0 - 1.0)
    """
    uncertainty = edge.uncertainty  # 1 - confidence

    # Conflict detection (simple version):
    # - Missing rule type → slight conflict (0.2)
    # - Very low confidence → higher conflict
    # - Contradictory modifiers would be 1.0 (future)
    conflict = 0.0

    if edge.rule is None:
        conflict = 0.2  # Untyped relation has mild conflict
    elif edge.confidence < 0.3:
        conflict = 0.8  # Very uncertain claims have high conflict

    # Base conflict from uncertainty itself
    conflict = max(conflict, uncertainty * 0.5)

    return conflict * uncertainty


def compute_path_tension(
    graph: nx.DiGraph,
    edges_lookup: Dict[Tuple[str, str], STGEdge],
    source: str,
    target: str,
    max_depth: int = 5,
) -> float:
    """Calculate total tension along shortest path from source to target.

    T_path(A→B) = Σ [Conflict(i,j) * Uncertainty(i,j)] over path.

    Args:
        graph: NetworkX graph
        edges_lookup: Map of (source, target) -> STGEdge
        source: Source node name
        target: Target node name
        max_depth: Maximum path length

    Returns:
        Total path tension. -1.0 if no path exists.
    """
    if source not in graph or target not in graph:
        return -1.0

    try:
        path = nx.shortest_path(graph, source, target)
    except nx.NetworkXNoPath:
        return -1.0

    if len(path) - 1 > max_depth:
        return -1.0

    total_tension = 0.0
    for i in range(len(path) - 1):
        edge_key = (path[i], path[i + 1])
        edge = edges_lookup.get(edge_key)
        if edge:
            total_tension += compute_edge_tension(edge)
        else:
            # Edge exists in graph but not in our lookup — mild tension
            total_tension += 0.3

    return total_tension


def compute_node_tension(
    graph: nx.DiGraph,
    edges_lookup: Dict[Tuple[str, str], STGEdge],
    node_name: str,
) -> float:
    """Calculate tension for a single node based on its edges.

    Node tension = average tension of all connected edges.

    Args:
        graph: NetworkX graph
        edges_lookup: Map of (source, target) -> STGEdge
        node_name: Node to calculate for

    Returns:
        Node tension value (0.0+)
    """
    if node_name not in graph:
        return 0.0

    tensions = []

    # Incoming edges
    for pred in graph.predecessors(node_name):
        edge = edges_lookup.get((pred, node_name))
        if edge:
            tensions.append(compute_edge_tension(edge))

    # Outgoing edges
    for succ in graph.successors(node_name):
        edge = edges_lookup.get((node_name, succ))
        if edge:
            tensions.append(compute_edge_tension(edge))

    if not tensions:
        return 0.0

    return sum(tensions) / len(tensions)


# ═══════════════════════════════════════════════════════════════
# Formula 3.4: Activation Function for Attention Allocation
# ═══════════════════════════════════════════════════════════════
#
# Attention(node_i) = α * Self_Relevance(i)
#                   + β * Tension(i)
#                   + γ * Σ Influence(j → i)
#                   + δ * Explore_Noise


def compute_self_relevance(
    graph: nx.DiGraph,
    node_name: str,
    self_anchor: str = "Self",
) -> float:
    """Calculate how relevant a node is to [Self].

    Self_Relevance = 1 / (1 + shortest_path_distance_to_Self).
    Direct connections to [Self] get highest relevance.

    Args:
        graph: NetworkX graph
        node_name: Node to evaluate
        self_anchor: Name of the Self anchor

    Returns:
        Self-relevance (0.0 - 1.0)
    """
    if node_name == self_anchor:
        return 1.0

    if self_anchor not in graph or node_name not in graph:
        return 0.0

    try:
        # Try both directions (Self → node or node → Self)
        try:
            dist_from_self = nx.shortest_path_length(graph, self_anchor, node_name)
        except nx.NetworkXNoPath:
            dist_from_self = float("inf")

        try:
            dist_to_self = nx.shortest_path_length(graph, node_name, self_anchor)
        except nx.NetworkXNoPath:
            dist_to_self = float("inf")

        min_dist = min(dist_from_self, dist_to_self)
        if min_dist == float("inf"):
            return 0.0

        return 1.0 / (1.0 + min_dist)

    except nx.NodeNotFound:
        return 0.0


def compute_influence(
    graph: nx.DiGraph,
    edges_lookup: Dict[Tuple[str, str], STGEdge],
    node_name: str,
) -> float:
    """Calculate incoming influence on a node.

    Σ Influence(j → i) = sum of (strength * confidence) for incoming edges.

    Args:
        graph: NetworkX graph
        edges_lookup: Edge lookup
        node_name: Target node

    Returns:
        Total influence value
    """
    if node_name not in graph:
        return 0.0

    total = 0.0
    for pred in graph.predecessors(node_name):
        edge = edges_lookup.get((pred, node_name))
        if edge:
            total += edge.strength * edge.confidence
        else:
            total += 0.25  # Default influence

    return total


def compute_activation(
    graph: nx.DiGraph,
    edges_lookup: Dict[Tuple[str, str], STGEdge],
    node: STGNode,
    alpha: float = 0.3,
    beta: float = 0.4,
    gamma: float = 0.2,
    delta: float = 0.1,
    self_anchor: str = "Self",
) -> float:
    """Compute attention/activation score for a node.

    Attention = α*Self_Relevance + β*Tension + γ*Influence + δ*Noise

    Args:
        graph: NetworkX graph
        edges_lookup: Edge lookup
        node: The STGNode to evaluate
        alpha: Self-relevance weight (default 0.3)
        beta: Tension weight (default 0.4, primary driver)
        gamma: Influence weight (default 0.2)
        delta: Exploration noise (default 0.1)
        self_anchor: Name of Self anchor

    Returns:
        Activation score (0.0+)
    """
    self_rel = compute_self_relevance(graph, node.name, self_anchor)
    tension = node.tension
    influence = compute_influence(graph, edges_lookup, node.name)
    noise = random.random()

    return alpha * self_rel + beta * tension + gamma * influence + delta * noise


# ═══════════════════════════════════════════════════════════════
# Formula 3.5: Intrinsic Reward for Self-Learning
# ═══════════════════════════════════════════════════════════════
#
# R_intrinsic = ΔΨ + α * Tension_Resolved - β * Computation_Cost


def compute_intrinsic_reward(
    psi_before: float,
    psi_after: float,
    tension_resolved: float,
    computation_cost: float,
    alpha: float = 1.5,
    beta: float = 0.1,
) -> float:
    """Calculate intrinsic reward for a reasoning action.

    R = ΔΨ + α*Tension_Resolved - β*Cost

    Positive reward = action improved cognitive state.
    Negative reward = action degraded state or was wasteful.

    Args:
        psi_before: Ψ value before action
        psi_after: Ψ value after action
        tension_resolved: Total tension reduction from action
        computation_cost: Resources expended (edge count, depth, etc.)
        alpha: Reward multiplier for tension resolution
        beta: Penalty multiplier for cost

    Returns:
        Intrinsic reward value (can be negative)
    """
    delta_psi = psi_after - psi_before
    return delta_psi + alpha * tension_resolved - beta * computation_cost
