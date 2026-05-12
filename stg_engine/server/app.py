"""FastAPI application factory.

`create_app(state)` is the only public entry point. It produces a
configured FastAPI instance that can be:
- run by uvicorn from `stg_engine.server.main:main`
- mounted into a larger application
- driven by `fastapi.testclient.TestClient` in tests
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from stg_engine.server.handlers import router as v1_router
from stg_engine.server.state import ServerState


def create_app(state: ServerState, allow_cors: bool = False) -> FastAPI:
    """Build the FastAPI app bound to `state`.

    Args:
        state: server-wide state (loaded engine + identity).
        allow_cors: when True, enables permissive CORS
            (Access-Control-Allow-Origin: *). Reserved for localhost binds
            — main.py decides based on --bind value, never enable for
            external binds without an explicit allow-list.
    """
    app = FastAPI(
        title=f"STG HTTP Server — {state.agent_name}",
        version=state.server_version,
        description=(
            "Read-side HTTP API for STG agent knowledge bases. "
            "All responses include an `agent` field identifying which "
            "knowledge base served the request."
        ),
    )
    app.state.server_state = state

    if allow_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    app.include_router(v1_router)
    return app
