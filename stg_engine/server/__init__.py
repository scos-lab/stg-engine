"""STG HTTP Server — read-side JSON API for AI agent knowledge bases.

Optional subpackage. Requires `pip install stg-engine[server]`.

Entry points:
    stg-server --agent <name> [--port 8765]
    from stg_engine.server.app import create_app
"""

from stg_engine.server.app import create_app
from stg_engine.server.state import ServerState

__all__ = ["create_app", "ServerState"]
