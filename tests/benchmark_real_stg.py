"""Benchmark Rust acceleration on a real STG database.

Compares engine.propagate() with and without Rust by toggling _HAS_RUST.

By default loads the active agent's database at
    ~/.stg/<default_agent>/memory.stg
(where <default_agent> is "default" unless overridden by STG_AGENT or
~/.stg/config.json). Override with STG_PATH env var:

    STG_PATH=/path/to/memory.stg python -m tests.benchmark_real_stg
"""
import os
import sys
import time
from pathlib import Path

from stg_engine.engine import STGEngine
import stg_engine.engine as engine_mod
import stg_engine.learning as learning_mod
import stg_engine.gravity as gravity_mod


def time_propagate(engine, query, runs=10):
    # Warm up
    engine.propagate(query)
    t0 = time.perf_counter()
    for _ in range(runs):
        result = engine.propagate(query)
    elapsed = (time.perf_counter() - t0) / runs * 1000
    return elapsed, len(result)


def _resolve_stg_path() -> Path:
    """Resolve which .stg to benchmark, without depending on a specific agent name."""
    env_path = os.environ.get("STG_PATH")
    if env_path:
        return Path(env_path).expanduser()

    stg_root = Path.home() / ".stg"
    # Prefer STG_AGENT env, then ~/.stg/config.json, then "default".
    agent = os.environ.get("STG_AGENT")
    if not agent:
        config_path = stg_root / "config.json"
        if config_path.exists():
            try:
                import json
                with open(config_path) as f:
                    agent = json.load(f).get("default_agent")
            except Exception:
                agent = None
    if not agent:
        agent = "default"
    return stg_root / agent / "memory.stg"


def main():
    stg_path = _resolve_stg_path()
    if not stg_path.exists():
        print(f"STG database not found at {stg_path}")
        print("Set STG_PATH=/path/to/memory.stg to pick a different file,")
        print("or run `stg ingest ...` to populate the default agent first.")
        sys.exit(1)

    print("\n  ── Real STG Database Benchmark ──\n")
    print(f"  Loading {stg_path}")

    t0 = time.perf_counter()
    engine = STGEngine.load(stg_path)
    load_time = (time.perf_counter() - t0) * 1000

    print(f"  Loaded {len(engine._nodes)} nodes, {len(engine._edges)} edges in {load_time:.1f} ms")
    print(f"  Rust acceleration: {engine_mod._HAS_RUST}")
    print()

    # Generic query set — strings that should activate something in any
    # non-trivial knowledge graph. Override via BENCHMARK_QUERIES env var
    # (semicolon-separated) if your graph's vocabulary differs.
    default_queries = [
        "memory",
        "learning",
        "graph structure",
        "activation propagation",
        "hebbian plasticity",
    ]
    queries_env = os.environ.get("BENCHMARK_QUERIES")
    queries = [q.strip() for q in queries_env.split(";")] if queries_env else default_queries

    # ── Phase 1: With Rust ──────────────────────────────────────
    print("  Phase 1: Rust acceleration ENABLED")
    print(f"  {'query':<45} {'time':>10} {'nodes':>8}")
    print(f"  {'-'*45} {'-'*10} {'-'*8}")
    rust_times = []
    for q in queries:
        elapsed, n_nodes = time_propagate(engine, q, runs=10)
        rust_times.append(elapsed)
        print(f"  {q:<45} {elapsed:>8.2f}ms {n_nodes:>8}")

    avg_rust = sum(rust_times) / len(rust_times)
    print(f"  {'AVERAGE':<45} {avg_rust:>8.2f}ms")
    print()

    # ── Phase 2: Disable Rust, force Python fallback ────────────
    print("  Phase 2: Rust acceleration DISABLED (forcing Python fallback)")
    engine_mod._HAS_RUST = False
    learning_mod._HAS_RUST = False
    gravity_mod._HAS_RUST = False

    print(f"  {'query':<45} {'time':>10} {'nodes':>8}")
    print(f"  {'-'*45} {'-'*10} {'-'*8}")
    py_times = []
    for q in queries:
        elapsed, n_nodes = time_propagate(engine, q, runs=10)
        py_times.append(elapsed)
        print(f"  {q:<45} {elapsed:>8.2f}ms {n_nodes:>8}")

    avg_py = sum(py_times) / len(py_times)
    print(f"  {'AVERAGE':<45} {avg_py:>8.2f}ms")
    print()

    # ── Summary ─────────────────────────────────────────────────
    print("  ── Summary ──")
    print(f"  Rust:    {avg_rust:7.2f} ms / propagate")
    print(f"  Python:  {avg_py:7.2f} ms / propagate")
    print(f"  Speedup: {avg_py / avg_rust:6.2f}x")
    print()

    # Re-enable for cleanliness
    engine_mod._HAS_RUST = True
    learning_mod._HAS_RUST = True
    gravity_mod._HAS_RUST = True


if __name__ == "__main__":
    main()
