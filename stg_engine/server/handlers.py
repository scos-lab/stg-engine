"""HTTP route handlers — one async function per endpoint.

Each handler reads the engine through `request.app.state.server_state` to
keep the handler signature minimal and to avoid global module state.
"""

import time
from fastapi import APIRouter, Request

from stg_engine.server.schemas import HealthResponse, StatsResponse


router = APIRouter(prefix="/v1")


@router.get("/health", response_model=HealthResponse, tags=["meta"])
async def health(request: Request) -> HealthResponse:
    """Liveness + agent identity + staleness signal. Designed for sub-10ms response."""
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
async def stats(request: Request) -> StatsResponse:
    """Full graph stats — passes engine.get_stats() through unmodified."""
    state = request.app.state.server_state
    engine_stats = state.engine.get_stats()
    return StatsResponse(agent=state.agent_name, stats=engine_stats)
