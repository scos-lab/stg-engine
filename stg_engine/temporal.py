"""STG Temporal Structure — Time as a dimension of the edge space.

Phase 11: Temporal indexing, episode sequencing, and time-aware retrieval
for the Semantic Tension Graph.

Implements three layers of temporal capability:
  Layer 1: Temporal Index — created_at on every edge, range queries
  Layer 2: Episode Graph — temporal sequence edges with k-fold association
  Layer 3: Temporal Retrieval — time range, neighborhood, episode replay

All functions are pure — they take engine data and return results.
The engine integrates these via add_edge(created_at=...) and CLI commands.

Theoretical basis:
  - Kanerva SDM Ch.8: Pointer chains ("data is address"), k-fold prediction
  - Eliasmith SPA Ch.6: Ordinal Serial Encoding (position binding)
  - Triangulation: §2.4 temporal mechanisms, §5.2 missing parts diagnosis

Design principle: Time is not a separate module — it is a new dimension
of the existing edge space. Temporal edges (edge_class="temporal") are
stored in the same graph, using delay_k to encode step distance.
"""

import math
import time as _time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from stg_engine.engine import STGEngine

from stg_engine.types import STGEdge


# ═══════════════════════════════════════════════════════════
# Layer 1: Temporal Index — Query edges by creation time
# ═══════════════════════════════════════════════════════════


def query_time_range(
    engine: "STGEngine",
    start: float,
    end: float,
    edge_class: Optional[str] = None,
) -> List[STGEdge]:
    """Return all edges with created_at in [start, end], ordered chronologically.

    Args:
        engine: STGEngine instance
        start: Unix epoch start time (inclusive)
        end: Unix epoch end time (inclusive)
        edge_class: Optional filter by edge class ("knowledge", "temporal", etc.)

    Returns:
        Edges sorted by created_at ascending. Edges with created_at=0.0
        (legacy, unknown creation time) are excluded.
    """
    results = []
    for edge in engine._edges:
        if edge.created_at <= 0.0:
            continue
        if edge.created_at < start or edge.created_at > end:
            continue
        if edge_class is not None and edge.edge_class != edge_class:
            continue
        results.append(edge)

    results.sort(key=lambda e: e.created_at)
    return results


def query_temporal_neighborhood(
    engine: "STGEngine",
    node: str,
    window_seconds: float = 3600.0,
) -> List[STGEdge]:
    """Find edges created within ±window of the node's edges' creation times.

    Reconstructs 'what was happening around the time this concept was active'.
    Uses the median created_at of the node's edges as the center point.

    Args:
        engine: STGEngine instance
        node: Node name to center the query on
        window_seconds: Time window in seconds (default 1 hour)

    Returns:
        Edges within the time window, sorted chronologically.
        Empty list if node has no timestamped edges.
    """
    # Find all edges involving this node (as source or target)
    node_times = []
    for edge in engine._edges:
        if edge.created_at <= 0.0:
            continue
        if edge.source == node or edge.target == node:
            node_times.append(edge.created_at)

    if not node_times:
        return []

    # Use median time as center
    node_times.sort()
    center = node_times[len(node_times) // 2]

    return query_time_range(engine, center - window_seconds, center + window_seconds)


# ═══════════════════════════════════════════════════════════
# Layer 2: Episode Graph — Temporal sequence edges
# ═══════════════════════════════════════════════════════════


def record_temporal_edge(
    engine: "STGEngine",
    source: str,
    target: str,
    delay_k: int = 1,
    session_id: Optional[str] = None,
) -> STGEdge:
    """Create a temporal sequence edge between two nodes.

    Temporal edges are distinct from knowledge edges:
    - edge_class = "temporal"
    - delay_k indicates the step distance in the sequence
    - confidence = 1.0 (temporal order is factual, not learnable)
    - rule = "temporal"

    Args:
        engine: STGEngine instance
        source: Source node (earlier in sequence)
        target: Target node (later in sequence)
        delay_k: Step distance (1 = adjacent, 2 = skip-one, etc.)
        session_id: Session that created this edge

    Returns:
        The created temporal STGEdge
    """
    edge = engine.add_edge(
        source=source,
        target=target,
        confidence=1.0,
        strength=1.0,
        rule="temporal",
        session_id=session_id,
        edge_class="temporal",
        delay_k=delay_k,
    )
    return edge


def build_episode_sequence(
    engine: "STGEngine",
    session_id: Optional[str] = None,
    k_fold: int = 2,
    time_start: float = 0.0,
    time_end: float = 0.0,
) -> List[STGEdge]:
    """Extract knowledge edges, order by created_at, create temporal chain.

    Can select edges by session_id OR by time range. At least one must be given.

    For k_fold=2, a sequence of edges [e0, e1, e2, e3] creates:
      step-1: node(e0) → node(e1), node(e1) → node(e2), node(e2) → node(e3)
      step-2: node(e0) → node(e2), node(e1) → node(e3)

    This implements Kanerva's k-fold association (SDM Ch.8):
    multiple delay channels prevent sequence crossover at shared nodes.

    Args:
        engine: STGEngine instance
        session_id: Session to build episode from (optional if time range given)
        k_fold: Number of delay folds (1=simple chain, 2=bigram, 3=trigram)
        time_start: Unix epoch start (used when session_id is None)
        time_end: Unix epoch end (used when session_id is None)

    Returns:
        List of created temporal edges
    """
    # Collect knowledge edges, ordered by creation time
    session_edges = []
    for e in engine._edges:
        if e.edge_class == "temporal" or e.created_at <= 0.0:
            continue
        if session_id is not None:
            if e.session_id != session_id:
                continue
        elif time_start > 0.0 and time_end > 0.0:
            if e.created_at < time_start or e.created_at > time_end:
                continue
        else:
            continue  # No filter specified
        session_edges.append(e)
    session_edges.sort(key=lambda e: e.created_at)

    if not session_edges:
        return []

    # Extract unique ordered node sequence (use source nodes, then last target)
    # Each edge represents a concept introduced at that time
    node_sequence: List[str] = []
    seen = set()
    for edge in session_edges:
        if edge.source not in seen:
            node_sequence.append(edge.source)
            seen.add(edge.source)
        if edge.target not in seen:
            node_sequence.append(edge.target)
            seen.add(edge.target)

    if len(node_sequence) < 2:
        return []

    # Create temporal edges for each fold
    created_edges: List[STGEdge] = []
    for k in range(1, k_fold + 1):
        for i in range(len(node_sequence) - k):
            src = node_sequence[i]
            tgt = node_sequence[i + k]
            # Avoid duplicate temporal edges
            existing = engine._edges_lookup.get((src.lower(), tgt.lower()))
            if existing and existing.edge_class == "temporal" and existing.delay_k == k:
                continue
            edge = record_temporal_edge(engine, src, tgt, delay_k=k, session_id=session_id)
            created_edges.append(edge)

    return created_edges


# ═══════════════════════════════════════════════════════════
# Layer 3: Temporal Retrieval — Episode replay & recency
# ═══════════════════════════════════════════════════════════


def replay_episode(
    engine: "STGEngine",
    entry_node: str,
    session_id: Optional[str] = None,
    max_steps: int = 50,
) -> List[str]:
    """Follow temporal edges from entry_node to reconstruct episode sequence.

    Like SDM iterative read: start from cue, follow pointer chain.
    Uses step-1 temporal edges (delay_k=1) for the main chain,
    with higher-k edges available for disambiguation if needed.

    Args:
        engine: STGEngine instance
        entry_node: Starting node for replay
        session_id: Optional session constraint
        max_steps: Maximum sequence length to prevent infinite loops

    Returns:
        Ordered list of node names forming the episode sequence.
    """
    sequence = [entry_node]
    visited = {entry_node}
    current = entry_node

    for _ in range(max_steps):
        # Find step-1 temporal edges from current node
        candidates = []
        for edge in engine._edges:
            if edge.edge_class != "temporal":
                continue
            if edge.delay_k != 1:
                continue
            if edge.source != current:
                continue
            if session_id and edge.session_id != session_id:
                continue
            if edge.target in visited:
                continue
            candidates.append(edge)

        if not candidates:
            break

        # If multiple candidates, use k-fold context for disambiguation
        if len(candidates) > 1:
            # Score each candidate by how many higher-k edges support it
            scored = []
            for cand in candidates:
                support = 0
                # Check if any previous nodes have higher-k edges pointing to this target
                for prev_node in sequence[-3:]:  # look back up to 3 steps
                    for edge in engine._edges:
                        if (edge.edge_class == "temporal"
                                and edge.source == prev_node
                                and edge.target == cand.target
                                and edge.delay_k > 1):
                            if session_id is None or edge.session_id == session_id:
                                support += 1
                scored.append((support, cand))
            scored.sort(key=lambda x: x[0], reverse=True)
            next_node = scored[0][1].target
        else:
            next_node = candidates[0].target

        sequence.append(next_node)
        visited.add(next_node)
        current = next_node

    return sequence


def temporal_propagate(
    engine: "STGEngine",
    query: str,
    time_bias: float = 0.5,
    decay_lambda: float = 0.01,
) -> List[Tuple[str, float]]:
    """Propagate with temporal weighting — favor recently created edges.

    Combines standard activation with recency score:
      final_score = (1 - time_bias) * activation + time_bias * recency_score

    recency_score = exp(-decay_lambda * hours_since_creation)

    This enables 'what was I thinking about recently?' queries.

    Args:
        engine: STGEngine instance
        query: Query text for standard propagation
        time_bias: Weight for recency vs semantic relevance (0.0-1.0)
        decay_lambda: Decay rate per hour (default 0.01 ≈ half-life ~69 hours)

    Returns:
        List of (node_name, blended_score) sorted by score descending.
    """
    # Standard propagation — returns List[str] (node names)
    # Activation values are stored on engine._nodes[name].activation
    activated_names = engine.propagate(query)

    if not activated_names:
        return []

    # Build (name, activation) pairs
    results = [
        (name, engine._nodes[name].activation if name in engine._nodes else 0.0)
        for name in activated_names
    ]

    if time_bias <= 0.0:
        return results

    now = _time.time()

    # Compute recency scores per node
    # For each node, use the most recent created_at of its edges
    node_recency: Dict[str, float] = {}
    for edge in engine._edges:
        if edge.created_at <= 0.0:
            continue
        hours_ago = (now - edge.created_at) / 3600.0
        recency = math.exp(-decay_lambda * hours_ago)
        for node_name in (edge.source, edge.target):
            if node_name not in node_recency or recency > node_recency[node_name]:
                node_recency[node_name] = recency

    # Blend scores
    blended = []
    for node_name, activation in results:
        recency = node_recency.get(node_name, 0.0)
        score = (1.0 - time_bias) * activation + time_bias * recency
        blended.append((node_name, score))

    blended.sort(key=lambda x: x[1], reverse=True)
    return blended


# ═══════════════════════════════════════════════════════════
# Utility: Human-readable time formatting
# ═══════════════════════════════════════════════════════════


def epoch_to_str(epoch: float) -> str:
    """Convert Unix epoch to human-readable local time string."""
    if epoch <= 0.0:
        return "(unknown)"
    dt = datetime.fromtimestamp(epoch)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def parse_date_str(date_str: str) -> float:
    """Parse a date string to Unix epoch. Supports YYYY-MM-DD and YYYY-MM-DD HH:MM:SS."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.timestamp()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {date_str!r}. Use YYYY-MM-DD or YYYY-MM-DD HH:MM:SS")
