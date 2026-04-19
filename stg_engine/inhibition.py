"""STG Inhibition — Competitive dynamics for activation spreading.

Phase 9: Biologically-inspired inhibition mechanisms for the Semantic Tension Graph.

Implements three phases of inhibition:
  Phase 1 (Core): Softmax WTA + Divisive Normalization
  Phase 2: Adaptive Threshold + Refractory Period
  Phase 3: Inhibitory Edges + Community Inhibition

All functions are pure — they take activation maps and return modified maps.
The engine integrates these via InhibitionConfig (disabled by default).

Neuroscience basis: Extracted from ASTREN reading program (8 books, 26 mechanisms).
Key references:
  - Softmax WTA: Maass (2000) "On the computational power of WTA"
  - Divisive normalization: Carandini & Heeger (2012) "Normalization as canonical"
  - Adaptive threshold: Turrigiano (2008) "Homeostatic plasticity"
  - Refractory period: Hodgkin & Huxley (1952) ionic conductance model
"""

import math
import time as _time
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from stg_engine.engine import STGEngine


# ═══════════════════════════════════════════════════════════
# Phase 1: Softmax WTA + Divisive Normalization
# ═══════════════════════════════════════════════════════════


def softmax_wta(
    activation_map: Dict[str, float],
    budget: float,
    eta: float = 2.0,
) -> Dict[str, float]:
    """Winner-Take-All budget redistribution via softmax competition.

    Replaces linear budget rescaling (Vehicle 12) with nonlinear competition.
    Strong activations capture disproportionately more of the budget.

    Math: a_i_new = (a_i^η / Σ_j a_j^η) * budget

    Args:
        activation_map: Current node activations (modified in-place).
        budget: Total activation budget to redistribute.
        eta: Competition sharpness. η=1.0 = linear (no WTA), η=2.0 = quadratic
             (moderate WTA), η→∞ = hard WTA (winner takes all).

    Returns:
        Modified activation_map (same dict, modified in-place).
    """
    if not activation_map or budget <= 0:
        return activation_map

    # Compute powered activations
    powered: Dict[str, float] = {}
    for name, act in activation_map.items():
        if act > 0:
            powered[name] = act ** eta
        else:
            powered[name] = 0.0

    total_powered = sum(powered.values())
    if total_powered <= 0:
        return activation_map

    # Redistribute budget proportionally to powered activations
    for name in activation_map:
        if powered[name] > 0:
            activation_map[name] = (powered[name] / total_powered) * budget
        else:
            activation_map[name] = 0.0

    return activation_map


def divisive_normalize(
    activation_map: Dict[str, float],
    engine: "STGEngine",
    sigma: float = 0.5,
) -> Dict[str, float]:
    """Per-node divisive normalization by neighbor activity.

    Each node's activation is divided by (1 + σ * Σ neighbor_activations).
    Nodes in active neighborhoods are suppressed relative to isolated nodes.
    This implements local lateral inhibition.

    Math: a_i_new = a_i / (1 + σ * Σ_{j∈N(i)} a_j)

    Args:
        activation_map: Current node activations (modified in-place).
        engine: STGEngine for graph topology lookup.
        sigma: Normalization strength. σ=0 disables, σ=1.0 strong suppression.

    Returns:
        Modified activation_map.
    """
    if not activation_map or sigma <= 0:
        return activation_map

    # Compute neighbor sums first (snapshot before modification)
    # Build case-insensitive lookup for activation values
    _act_lower = {k.lower(): v for k, v in activation_map.items()}
    neighbor_sums: Dict[str, float] = {}
    for name in activation_map:
        _key = name.lower()
        if _key not in engine._nodes:
            neighbor_sums[name] = 0.0
            continue
        neighbors: Set[str] = set()
        if engine._graph.has_node(_key):
            neighbors.update(engine._graph.successors(_key))
            neighbors.update(engine._graph.predecessors(_key))
        neighbor_sum = sum(
            _act_lower.get(n, 0.0) for n in neighbors
        )
        neighbor_sums[name] = neighbor_sum

    # Apply divisive normalization
    for name in activation_map:
        divisor = 1.0 + sigma * neighbor_sums.get(name, 0.0)
        activation_map[name] = activation_map[name] / divisor

    return activation_map


# ═══════════════════════════════════════════════════════════
# Phase 2: Adaptive Threshold + Refractory Period
# ═══════════════════════════════════════════════════════════


def adaptive_threshold(
    activation_map: Dict[str, float],
    base_threshold: float = 0.15,
    gain: float = 1.0,
) -> float:
    """Compute adaptive activation threshold based on network activity.

    Threshold rises when the network is highly active (homeostatic plasticity).
    This prevents runaway excitation during high-activity propagation.

    Math: threshold = base + gain * mean(activations)

    Args:
        activation_map: Current activations.
        base_threshold: Minimum threshold (same as static threshold default).
        gain: How much mean activity raises the threshold.

    Returns:
        Computed threshold value.
    """
    if not activation_map:
        return base_threshold

    values = [v for v in activation_map.values() if v > 0]
    if not values:
        return base_threshold

    mean_act = sum(values) / len(values)
    return base_threshold + gain * mean_act


def apply_refractory(
    activation_map: Dict[str, float],
    refractory_set: Dict[str, float],
    decay_rate: float = 0.5,
    suppression: float = 0.3,
) -> Dict[str, float]:
    """Apply refractory period suppression to recently-activated nodes.

    Nodes that were strongly activated in recent propagations get reduced
    re-activation, preventing the same nodes from dominating repeatedly.

    Args:
        activation_map: Current activations (modified in-place).
        refractory_set: Dict of {node_name: last_activation_level}.
            Updated by the caller after propagation.
        decay_rate: How fast refractory state decays (0=no decay, 1=instant).
        suppression: Maximum suppression factor (0.3 = reduce by up to 30%).

    Returns:
        Modified activation_map.
    """
    if not refractory_set:
        return activation_map

    for name in activation_map:
        if name in refractory_set:
            prior = refractory_set[name]
            # Suppression proportional to prior activation strength
            factor = 1.0 - suppression * min(1.0, prior)
            activation_map[name] *= max(0.0, factor)

    # Decay refractory state for next round
    to_remove = []
    for name in refractory_set:
        refractory_set[name] *= (1.0 - decay_rate)
        if refractory_set[name] < 0.01:
            to_remove.append(name)
    for name in to_remove:
        del refractory_set[name]

    return activation_map


# ═══════════════════════════════════════════════════════════
# Phase 3: Inhibitory Edges + Community Inhibition
# ═══════════════════════════════════════════════════════════


def apply_inhibitory_edges(
    delta_map: Dict[str, float],
    engine: "STGEngine",
    activation_map: Dict[str, float],
    inhibitory_strength: float = 1.0,
) -> Dict[str, float]:
    """Subtract activation via inhibitory edges during propagation.

    Edges with edge_class="inhibitory" subtract rather than add activation
    to their targets. This enables explicit suppression pathways.

    Args:
        delta_map: Per-iteration delta accumulator (modified in-place).
        engine: STGEngine for edge lookup.
        activation_map: Current activations (read-only, for source strength).
        inhibitory_strength: Scaling factor for inhibitory effect.

    Returns:
        Modified delta_map.
    """
    for edge in engine._edges:
        if edge.modifiers.get("edge_class") != "inhibitory":
            continue
        source_act = activation_map.get(edge.source, 0.0)
        if source_act <= 0.01:
            continue
        # Inhibitory contribution: subtract from target
        inhibition = source_act * edge.confidence * inhibitory_strength
        delta_map[edge.target] = delta_map.get(edge.target, 0.0) - inhibition

    return delta_map


def community_inhibition(
    activation_map: Dict[str, float],
    engine: "STGEngine",
    suppression_factor: float = 0.2,
) -> Dict[str, float]:
    """Cross-community lateral inhibition via Louvain communities.

    When a community becomes highly active, it suppresses activation in
    other communities. This focuses propagation within relevant clusters.

    Args:
        activation_map: Current activations (modified in-place).
        engine: STGEngine (must have topology analyzer available).
        suppression_factor: How much dominant communities suppress others.

    Returns:
        Modified activation_map.
    """
    # Get community assignments (lazy — only compute if needed)
    try:
        from stg_engine.topology import TopologyAnalyzer
        topo = TopologyAnalyzer(engine)
        communities = topo.detect_communities()
    except Exception:
        return activation_map  # Graceful degradation

    if not communities:
        return activation_map

    # Build node→community mapping
    node_community: Dict[str, int] = {}
    for comm in communities:
        for member in comm.members:
            node_community[member] = comm.community_id

    # Compute per-community activation
    community_activation: Dict[int, float] = {}
    for name, act in activation_map.items():
        cid = node_community.get(name, -1)
        if cid >= 0:
            community_activation[cid] = community_activation.get(cid, 0.0) + act

    if not community_activation:
        return activation_map

    # Find dominant community
    max_community = max(community_activation, key=community_activation.get)  # type: ignore
    max_act = community_activation[max_community]

    if max_act <= 0:
        return activation_map

    # Suppress non-dominant communities
    for name in activation_map:
        cid = node_community.get(name, -1)
        if cid >= 0 and cid != max_community:
            community_act = community_activation.get(cid, 0.0)
            # Suppression proportional to how much weaker this community is
            ratio = community_act / max_act if max_act > 0 else 0.0
            suppression = suppression_factor * (1.0 - ratio)
            activation_map[name] *= max(0.0, 1.0 - suppression)

    return activation_map
