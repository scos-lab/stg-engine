"""Thin read-only wrapper layer over STGEngine.

All HTTP handlers go through this module so that the `read_only=True`
discipline is enforced in one place. A future audit asks "who calls
engine.propagate(read_only=False)?" — the answer should be CLI + tests
only, never the HTTP server.

The functions here intentionally accept an STGEngine and return raw
engine types (List[str], STGNode, List[STGEdge]). Schema serialization
is the responsibility of handlers.py / schemas.py — this layer just
calls the engine.
"""

from typing import Dict, List, Optional, Tuple

from stg_engine.engine import STGEngine
from stg_engine.types import STGNode, STGEdge


def propagate_read_only(
    engine: STGEngine,
    query: str,
    max_nodes: int = 20,
) -> List[str]:
    """Run engine.propagate with read_only=True and slice to max_nodes."""
    activated = engine.propagate(query, read_only=True)
    return activated[:max_nodes]


def get_node_read_only(engine: STGEngine, name: str) -> Optional[STGNode]:
    """Look up a node by name. Already side-effect-free; flag is for symmetry."""
    return engine.get_node(name)


def get_edges_read_only(
    engine: STGEngine,
    source: Optional[str] = None,
    target: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[STGEdge]:
    """Fetch edges (already side-effect-free) with optional truncation."""
    edges = engine.get_edges(source=source, target=target)
    if limit is not None:
        edges = edges[:limit]
    return edges


def query_nodes_read_only(
    engine: STGEngine,
    pattern: str,
    namespace: Optional[str] = None,
    limit: int = 50,
) -> List[STGNode]:
    """Fuzzy substring search by node name, optionally filtered by namespace."""
    return engine.query_nodes(
        name_pattern=pattern, namespace=namespace, limit=limit
    )


def find_paths_read_only(
    engine: STGEngine,
    source: str,
    target: str,
    max_depth: int = 5,
) -> List[List[str]]:
    """All simple paths source→target (already side-effect-free)."""
    return engine.find_paths(source, target, max_depth=max_depth)


def metadata_keys_read_only(
    engine: STGEngine,
    namespace: Optional[str] = None,
    node_name: Optional[str] = None,
) -> List[Tuple[str, int, int]]:
    """Metadata key universe (already side-effect-free)."""
    return engine.query_metadata_keys(namespace=namespace, node_name=node_name)


def _edge_weight(edge: STGEdge) -> float:
    """Relevance weight of an edge for browse scoring.

    Prefer an explicit 'weight' modifier (e.g. SteamSpy vote counts on
    tagged_as edges); otherwise fall back to 1.0 so weightless edges still
    count toward intersection membership.
    """
    w = edge.modifiers.get("weight")
    if w is None:
        return 1.0
    try:
        return float(w)
    except (TypeError, ValueError):
        return 1.0


def reverse_intersect_read_only(
    engine: STGEngine,
    targets: List[str],
    mode: str = "intersection",
    namespace: Optional[str] = None,
) -> List[Tuple[STGNode, float, int]]:
    """Find source nodes linking to a set of target anchors (reverse-hub).

    For each target, collect its incoming edges and the best source→target
    weight. A node's score is the sum of those weights across the targets it
    matches; `matched` is how many targets it links to.

    Args:
        targets: Anchor node names (e.g. ['Tag:FPS', 'Tag:Co-op']).
        mode: 'intersection' keeps sources linking to ALL targets; 'union'
            keeps sources linking to ANY target.
        namespace: Optional result filter (e.g. 'Game').

    Returns:
        (node, score, matched) tuples sorted by score descending.
    """
    ns_lower = namespace.lower() if namespace else None
    per_target: List[Dict[str, float]] = []
    for t in targets:
        best: Dict[str, float] = {}
        for e in engine.get_edges(target=t):
            w = _edge_weight(e)
            if w > best.get(e.source, float("-inf")):
                best[e.source] = w
        per_target.append(best)

    if not per_target:
        return []

    if mode == "union":
        keys = set().union(*[set(m.keys()) for m in per_target])
    else:  # intersection (default)
        keys = set(per_target[0].keys())
        for m in per_target[1:]:
            keys &= set(m.keys())

    scored: List[Tuple[STGNode, float, int]] = []
    for k in keys:
        node = engine.get_node(k)
        if node is None:
            continue
        if ns_lower is not None and (node.namespace or "").lower() != ns_lower:
            continue
        score = sum(m[k] for m in per_target if k in m)
        matched = sum(1 for m in per_target if k in m)
        scored.append((node, score, matched))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
