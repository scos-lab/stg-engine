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

from typing import List, Optional

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
