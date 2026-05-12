"""Pydantic request/response models for the v1 HTTP API.

All response models carry an `agent` field so multi-server-per-agent
deployments can be inspected without out-of-band knowledge of which port
serves which knowledge base.
"""

from typing import Any, Dict
from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Liveness probe + agent identity + freshness signal.

    `engine_mtime` is the mtime of the underlying .stg file at server-load
    time. Clients comparing this against an expected value can detect that
    the server is serving a stale snapshot (post-ingest restart required).
    """

    status: str = Field(default="ok", description="Always 'ok' when the server can answer at all.")
    agent: str = Field(description="Agent name (the --agent flag the server was started with).")
    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    server_version: str
    engine_mtime: float = Field(description="Epoch float — mtime of the .stg file at load time.")
    uptime_seconds: float = Field(ge=0.0)


class StatsResponse(BaseModel):
    """Full graph statistics — mirrors `engine.get_stats()` output.

    `stats` is intentionally an open string-keyed dict rather than a fixed
    schema: engine.get_stats() evolves and we don't want every new metric
    to require a server release.
    """

    model_config = ConfigDict(extra="forbid")

    agent: str
    stats: Dict[str, Any]


class ErrorResponse(BaseModel):
    """Standard error envelope: {error, detail}.

    `error` is a stable machine-routable code; `detail` is the human message.
    """

    error: str
    detail: str
