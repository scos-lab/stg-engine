"""Pydantic request/response models for the v1 HTTP API.

All response models carry an `agent` field so multi-server-per-agent
deployments can be inspected without out-of-band knowledge of which port
serves which knowledge base.

Modifier values on the wire are string-only (Decision 12.3): keeps the
schema simple, matches STL ingest contract, lets clients cast on receipt.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


# ─── Health / Stats (M1) ────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Liveness probe + agent identity + freshness signal."""

    status: str = Field(default="ok", description="Always 'ok' when the server can answer at all.")
    agent: str = Field(description="Agent name (the --agent flag the server was started with).")
    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    server_version: str
    engine_mtime: float = Field(description="Epoch float — mtime of the .stg file at load time.")
    uptime_seconds: float = Field(ge=0.0)


class StatsResponse(BaseModel):
    """Full graph statistics — mirrors `engine.get_stats()` output."""

    model_config = ConfigDict(extra="forbid")

    agent: str
    stats: Dict[str, Any]


class ErrorResponse(BaseModel):
    """Standard error envelope: {error, detail}."""

    error: str
    detail: str


# ─── Edge + Node (shared building blocks) ───────────────────────────────────


class EdgeOut(BaseModel):
    """One edge on the wire. Modifiers are string-only per Decision 12.3."""

    source: str
    target: str
    confidence: float
    strength: float
    rule: Optional[str] = None
    modifiers: Dict[str, str] = Field(
        default_factory=dict,
        description="All edge modifiers (semantic + provenance) flattened to strings.",
    )


class NodeOut(BaseModel):
    """Full node detail — used by /v1/node/{name}."""

    name: str
    namespace: Optional[str] = None
    anchor_type: Optional[str] = None
    activation: float = 0.0
    tension: float = 0.0
    self_relevance: float = 0.0
    metadata: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Intrinsic node attributes. Provenance fields (source / created_at "
            "/ ingested_at / batch_id / superseded_at / recorded_at) are folded "
            "out unless the endpoint was called with ?full=true."
        ),
    )
    metadata_count: int = Field(description="Total metadata key count BEFORE provenance folding.")
    incoming: List[EdgeOut] = Field(default_factory=list)
    outgoing: List[EdgeOut] = Field(default_factory=list)


# ─── /v1/propagate ──────────────────────────────────────────────────────────


class PropagateRequest(BaseModel):
    """Request body for POST /v1/propagate.

    `query` is free-text (anchor names or natural language). The engine's
    own tokenization decides what to seed.
    """

    query: str = Field(min_length=1, max_length=2000)
    max_nodes: int = Field(default=20, ge=1, le=200, description="Cap on returned activated nodes.")
    include_edges: bool = Field(default=True, description="When false, omit per-node edge lists for size reduction.")
    edge_limit_per_node: int = Field(default=10, ge=0, le=100)
    full: bool = Field(default=False, description="When true, include provenance fields in any embedded metadata.")


class ActivatedNodeOut(BaseModel):
    """Compact node entry inside a PropagateResponse.

    Trimmer than NodeOut: no metadata dump (would be huge for 20 nodes).
    Clients that need full metadata follow up with /v1/node/{name}.
    """

    name: str
    namespace: Optional[str] = None
    activation: float = 0.0
    metadata_count: int = 0
    outgoing: List[EdgeOut] = Field(default_factory=list)


class PropagateResponse(BaseModel):
    agent: str
    query: str
    elapsed_ms: int
    seed_count: int = Field(description="Number of nodes matched by the query tokens before propagation.")
    activated_count: int = Field(description="Total activated node count BEFORE max_nodes truncation.")
    nodes: List[ActivatedNodeOut]
    truncated: bool = Field(description="True when activated_count > max_nodes.")


# ─── /v1/node/{name} ────────────────────────────────────────────────────────


class NodeDetailResponse(BaseModel):
    """Full single-node detail."""

    agent: str
    node: NodeOut


# ─── /v1/query ──────────────────────────────────────────────────────────────


class MatchOut(BaseModel):
    """Lightweight match entry in a QueryResponse."""

    name: str
    namespace: Optional[str] = None
    edge_count_out: int = 0
    edge_count_in: int = 0


class QueryResponse(BaseModel):
    agent: str
    pattern: str
    namespace_filter: Optional[str] = None
    matches: List[MatchOut]
    total_matched: int = Field(description="Total matches BEFORE limit truncation.")
    truncated: bool


# ─── /v1/paths (M4) ──────────────────────────────────────────────────────────


class PathsResponse(BaseModel):
    """All simple paths between two nodes, shortest first."""

    agent: str
    source: str
    target: str
    max_depth: int
    paths: List[List[str]] = Field(
        default_factory=list,
        description="Each path is a list of display node names from source to target.",
    )
    path_count: int = Field(description="Total paths found BEFORE limit truncation.")
    truncated: bool = False


# ─── /v1/attrs (M4) ──────────────────────────────────────────────────────────


class AttrKeyOut(BaseModel):
    """One metadata key and its coverage within the queried scope."""

    key: str
    count: int = Field(description="In-scope nodes carrying this key.")
    total: int = Field(description="Total in-scope nodes.")


class AttrsResponse(BaseModel):
    """Metadata key universe for a node / namespace / whole graph.

    Schema-less metadata means clients must discover which attributes exist
    before they can filter on them. This is that discovery endpoint.
    """

    agent: str
    scope: str = Field(description="'node:<name>' | 'namespace:<ns>' | 'graph'.")
    keys: List[AttrKeyOut] = Field(default_factory=list)


# ─── /v1/browse (M4) — reverse-hub intersection/union ─────────────────────────


class BrowseItemOut(BaseModel):
    """One node in a browse result, with relevance score against the targets."""

    name: str
    namespace: Optional[str] = None
    score: float = Field(
        description="Sum of edge weights (modifier 'weight', else 1.0) to matched targets.",
    )
    matched: int = Field(description="How many of the requested targets this node links to.")


class BrowseResponse(BaseModel):
    """Catalog browse: nodes linking to a set of target anchors.

    Generic reverse-hub. stg-steam's 'browse games by tags' is this endpoint
    with namespace='Game' and targets=['Tag:FPS', 'Tag:Co-op', ...].
    """

    agent: str
    targets: List[str]
    mode: str = Field(description="'intersection' (link ALL targets) or 'union' (link ANY).")
    namespace_filter: Optional[str] = None
    items: List[BrowseItemOut] = Field(default_factory=list)
    total_matched: int = Field(description="Matching nodes BEFORE limit truncation.")
    truncated: bool = False
