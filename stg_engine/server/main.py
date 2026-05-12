"""CLI entry point: `stg-server --agent <name>`.

Resolves the agent's .stg file, loads the engine once, then hands off to
uvicorn. Never returns (uvicorn's loop owns the process).
"""

import argparse
import os
import sys
import time

LOCALHOST_BINDS = {"127.0.0.1", "localhost", "::1"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="stg-server",
        description=(
            "Run an HTTP server exposing one STG agent's knowledge base "
            "via a versioned JSON API. Read-only in v1; mutation stays on "
            "the `stg` CLI."
        ),
    )
    parser.add_argument(
        "--agent",
        required=True,
        help="Agent name. Resolves to ~/.stg/<name>/memory.stg unless --path overrides.",
    )
    parser.add_argument(
        "--path",
        help="Override .stg file path directly (debug / testing).",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
    )
    parser.add_argument(
        "--max-concurrent-propagate",
        type=int,
        default=4,
        help="Cap on simultaneous propagate calls (M4+ enforces; M1 ignored).",
    )
    parser.add_argument(
        "--allow-external-bind",
        action="store_true",
        help=(
            "Required when --bind is not localhost. "
            "Defensive — forces explicit opt-in for non-loopback exposure."
        ),
    )
    args = parser.parse_args(argv)

    # Defensive bind check — refuse to expose externally without explicit opt-in.
    is_localhost = args.bind in LOCALHOST_BINDS
    if not is_localhost and not args.allow_external_bind:
        print(
            f"stg-server: refusing to bind {args.bind!r} without --allow-external-bind. "
            "Pass the flag to confirm you intend external exposure (no auth in v1).",
            file=sys.stderr,
        )
        return 2

    # Resolve .stg path.
    if args.path:
        stg_path = os.path.abspath(os.path.expanduser(args.path))
    else:
        stg_path = os.path.expanduser(f"~/.stg/{args.agent}/memory.stg")

    if not os.path.exists(stg_path):
        print(
            f"stg-server: agent file not found at {stg_path!r}. "
            "Check the --agent name or pass --path to specify a file directly.",
            file=sys.stderr,
        )
        return 3

    # Load engine.
    from stg_engine import STGEngine, __version__
    engine = STGEngine.load(stg_path)
    engine_mtime = os.path.getmtime(stg_path)

    # Build state + app.
    from stg_engine.server.state import ServerState
    from stg_engine.server.app import create_app

    state = ServerState(
        engine=engine,
        agent_name=args.agent,
        stg_path=stg_path,
        engine_mtime=engine_mtime,
        server_version=__version__,
        server_start_time=time.time(),
        max_concurrent_propagate=args.max_concurrent_propagate,
    )
    app = create_app(state, allow_cors=is_localhost)

    # Startup banner — single line, ops-friendly.
    print(
        f"[stg-server] agent={args.agent} "
        f"nodes={len(engine._nodes)} edges={len(engine._edges)} "
        f"bind={args.bind}:{args.port} "
        f"cors={'on' if is_localhost else 'off'} "
        f"docs=http://{args.bind}:{args.port}/docs",
        flush=True,
    )

    import uvicorn
    uvicorn.run(
        app,
        host=args.bind,
        port=args.port,
        log_level=args.log_level,
        workers=1,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
