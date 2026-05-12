"""HTTP route handlers — one function per endpoint.

Engine access goes through stg_engine.server.engine_wrap so that
read_only=True is enforced uniformly. Handlers shape engine return
types into Pydantic response models and never call engine.* directly
for state-touching operations.
"""

import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from stg_engine import PROVENANCE_FIELDS
from stg_engine.types import STGEdge, STGNode

from stg_engine.server import engine_wrap
from stg_engine.server.schemas import (
    ActivatedNodeOut,
    EdgeOut,
    HealthResponse,
    MatchOut,
    NodeDetailResponse,
    NodeOut,
    PropagateRequest,
    PropagateResponse,
    QueryResponse,
    StatsResponse,
)


router = APIRouter(prefix="/v1")


# ─── helpers ───────────────────────────────────────────────────────────────


def _edge_to_out(edge: STGEdge) -> EdgeOut:
    """Convert engine STGEdge to wire EdgeOut. Modifier values all stringified."""
    return EdgeOut(
        source=edge.source,
        target=edge.target,
        confidence=edge.confidence,
        strength=edge.strength,
        rule=edge.rule,
        modifiers={k: str(v) for k, v in edge.modifiers.items() if v is not None},
    )


def _fold_metadata(metadata: dict, full: bool) -> tuple[dict, int]:
    """Return (visible_metadata_as_str_dict, total_key_count_before_folding)."""
    total = len(metadata)
    if full:
        visible = {k: str(v) for k, v in metadata.items()}
    else:
        visible = {
            k: str(v) for k, v in metadata.items() if k not in PROVENANCE_FIELDS
        }
    return visible, total


def _node_to_out(
    engine,
    node: STGNode,
    full: bool = False,
    edge_limit: int = 10,
) -> NodeOut:
    """Convert engine STGNode to wire NodeOut (full detail w/ edges)."""
    visible_meta, total_count = _fold_metadata(node.metadata, full)
    outgoing = engine_wrap.get_edges_read_only(
        engine, source=node.name, limit=edge_limit
    )
    incoming = engine_wrap.get_edges_read_only(
        engine, target=node.name, limit=edge_limit
    )
    return NodeOut(
        name=node.name,
        namespace=node.namespace,
        anchor_type=node.anchor_type,
        activation=node.activation,
        tension=node.tension,
        self_relevance=node.self_relevance,
        metadata=visible_meta,
        metadata_count=total_count,
        outgoing=[_edge_to_out(e) for e in outgoing],
        incoming=[_edge_to_out(e) for e in incoming],
    )


# ─── /v1/health, /v1/stats (M1) ────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse, tags=["meta"])
def health(request: Request) -> HealthResponse:
    """Liveness + agent identity + staleness signal."""
    state = request.app.state.server_state
    return HealthResponse(
        status="ok",
        agent=state.agent_name,
        node_count=len(state.engine._nodes),
        edge_count=len(state.engine._edges),
        server_version=state.server_version,
        engine_mtime=state.engine_mtime,
        uptime_seconds=time.time() - state.server_start_time,
    )


@router.get("/stats", response_model=StatsResponse, tags=["meta"])
def stats(request: Request) -> StatsResponse:
    """Full graph stats — passes engine.get_stats() through unmodified."""
    state = request.app.state.server_state
    return StatsResponse(agent=state.agent_name, stats=state.engine.get_stats())


# ─── /v1/propagate (M3) ────────────────────────────────────────────────────


@router.post("/propagate", response_model=PropagateResponse, tags=["query"])
def propagate(request: Request, body: PropagateRequest) -> PropagateResponse:
    """Activation propagation with read_only=True (no side-effects on engine)."""
    state = request.app.state.server_state
    engine = state.engine
    t0 = time.perf_counter()
    activated = engine.propagate(body.query, read_only=True)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    total = len(activated)
    truncated = total > body.max_nodes
    activated = activated[: body.max_nodes]

    nodes_out = []
    for name in activated:
        node = engine.get_node(name)
        if node is None:
            continue
        outgoing: list[EdgeOut] = []
        if body.include_edges and body.edge_limit_per_node > 0:
            edges = engine_wrap.get_edges_read_only(
                engine, source=name, limit=body.edge_limit_per_node
            )
            outgoing = [_edge_to_out(e) for e in edges]
        _, meta_count = _fold_metadata(node.metadata, body.full)
        nodes_out.append(
            ActivatedNodeOut(
                name=node.name,
                namespace=node.namespace,
                activation=node.activation,
                metadata_count=meta_count,
                outgoing=outgoing,
            )
        )

    metrics = engine._last_propagation_metrics
    seed_count = metrics.seed_node_count if metrics is not None else 0

    return PropagateResponse(
        agent=state.agent_name,
        query=body.query,
        elapsed_ms=elapsed_ms,
        seed_count=seed_count,
        activated_count=total,
        nodes=nodes_out,
        truncated=truncated,
    )


# ─── /v1/node/{name} (M3) ──────────────────────────────────────────────────


@router.get("/node/{name}", response_model=NodeDetailResponse, tags=["query"])
def node_detail(
    request: Request,
    name: str,
    full: bool = Query(default=False, description="Include provenance fields in metadata."),
    edge_limit: int = Query(default=50, ge=0, le=500),
) -> NodeDetailResponse:
    """Single-node detail with incoming/outgoing edges."""
    state = request.app.state.server_state
    node = engine_wrap.get_node_read_only(state.engine, name)
    if node is None:
        raise HTTPException(
            status_code=404,
            detail=f"Node '{name}' not in agent '{state.agent_name}'.",
        )
    return NodeDetailResponse(
        agent=state.agent_name,
        node=_node_to_out(state.engine, node, full=full, edge_limit=edge_limit),
    )


# ─── /v1/query (M3) ────────────────────────────────────────────────────────


@router.get("/query", response_model=QueryResponse, tags=["query"])
def query(
    request: Request,
    pattern: str = Query(default="", max_length=200,
                          description="Substring to search for. Empty when listing by namespace alone."),
    namespace: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
) -> QueryResponse:
    """Fuzzy substring search of node names, optionally filtered by namespace."""
    state = request.app.state.server_state
    matches = engine_wrap.query_nodes_read_only(
        state.engine, pattern, namespace=namespace, limit=limit + 1
    )
    truncated = len(matches) > limit
    matches = matches[:limit]

    matches_out = []
    for node in matches:
        out_count = len(state.engine.get_edges(source=node.name))
        in_count = len(state.engine.get_edges(target=node.name))
        matches_out.append(
            MatchOut(
                name=node.name,
                namespace=node.namespace,
                edge_count_out=out_count,
                edge_count_in=in_count,
            )
        )

    # total_matched: if we hit the limit+1 probe, we know there are >limit. We
    # don't have a cheap way to count all matches without re-running the
    # search unbounded, so report a conservative count: len(matches_out) plus
    # an "at least one more" marker via truncated=True.
    total_matched = len(matches_out) + (1 if truncated else 0)

    return QueryResponse(
        agent=state.agent_name,
        pattern=pattern,
        namespace_filter=namespace,
        matches=matches_out,
        total_matched=total_matched,
        truncated=truncated,
    )
