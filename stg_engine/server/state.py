"""Server-wide singleton state: loaded engine, agent identity, runtime knobs."""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from stg_engine.engine import STGEngine


@dataclass
class ServerState:
    """Holds the loaded engine and per-process metadata.

    Lifecycle: created once in main.py before uvicorn starts, attached to
    FastAPI app via app.state.server_state. Handlers read it via Request.
    Never mutated after startup (engine itself may mutate on read paths
    until M2 lands read_only=True flag).
    """

    engine: "STGEngine"
    agent_name: str
    stg_path: str            # absolute path to the loaded .stg file
    engine_mtime: float      # mtime of stg_path at load time
    server_version: str
    server_start_time: float = field(default_factory=time.time)

    # Concurrency knob — semaphore cap on simultaneous propagate calls.
    # Wired through in M4 once /v1/propagate exists. M1 ignores it.
    max_concurrent_propagate: int = 4
