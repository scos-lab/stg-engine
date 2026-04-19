"""Verify that the Rust propagate_inner_loop produces results identical
(within floating-point tolerance) to the original Python implementation
in skc.stg.engine._propagate_from_seeds.

Run with:
    python -m pytest tests/test_propagate_parity.py -v
or directly:
    python tests/test_propagate_parity.py
"""

import sys
import time
from typing import Dict, List, Tuple


def py_propagate_reference(
    activation_map: Dict[str, float],
    edges: List[Tuple[str, str, float, float, bool]],
    decay: float = 0.65,
    iterations: int = 5,
    normalize: bool = True,
) -> Dict[str, float]:
    """Pure-Python reference implementation of propagate_inner_loop.

    Mirrors the algorithm in skc.stg.engine._propagate_from_seeds
    (lines 1325-1416), with importance/preference/inhibition disabled
    (the v0.2.0a1 fast path).
    """
    activations = dict(activation_map)

    # Build adjacency: source -> [(target, weight)]
    adjacency: Dict[str, List[Tuple[str, float]]] = {}
    for src, tgt, conf, sal, is_virtual in edges:
        w = conf * sal
        if is_virtual:
            w *= 0.5
        adjacency.setdefault(src, []).append((tgt, w))

    activation_budget = sum(activations.values()) if normalize else 0.0

    for _ in range(iterations):
        delta_map: Dict[str, float] = {}
        for node, activation in list(activations.items()):
            if activation <= 0.01:
                continue
            successors = adjacency.get(node, [])
            if not successors:
                continue
            total_weight = sum(w for _, w in successors)
            if total_weight <= 0:
                continue
            for tgt, w in successors:
                spread = activation * decay * w / total_weight
                delta_map[tgt] = delta_map.get(tgt, 0.0) + spread

        for node, delta in delta_map.items():
            activations[node] = activations.get(node, 0.0) + delta
            if activations[node] < 0:
                activations[node] = 0.0

        if normalize and activation_budget > 0:
            current_total = sum(activations.values())
            if current_total > activation_budget and current_total > 0:
                scale = activation_budget / current_total
                for k in activations:
                    activations[k] *= scale

    return activations


def assert_dicts_close(
    rust_result: Dict[str, float],
    py_result: Dict[str, float],
    tol: float = 1e-5,
) -> None:
    """Assert two activation dicts are equal within tolerance."""
    assert set(rust_result.keys()) == set(py_result.keys()), (
        f"Key sets differ:\n"
        f"  Rust only: {set(rust_result) - set(py_result)}\n"
        f"  Python only: {set(py_result) - set(rust_result)}"
    )
    for k in rust_result:
        rust_v = rust_result[k]
        py_v = py_result[k]
        diff = abs(rust_v - py_v)
        assert diff < tol, (
            f"Mismatch for {k!r}: rust={rust_v:.10f}, py={py_v:.10f}, diff={diff:.2e}"
        )


# ─────────────────────────────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────────────────────────────


def test_import_rust_core():
    """Smoke test: the Rust extension imports cleanly."""
    from stg_engine import _rust_core
    assert _rust_core.__version__ == "0.2.0-alpha.1"
    assert _rust_core.__license__ == "BUSL-1.1"
    assert hasattr(_rust_core, "propagate_inner_loop")
    print(f"  ✓ _rust_core v{_rust_core.__version__} loaded")


def test_empty_inputs():
    """Empty graph + empty seeds → empty result."""
    from stg_engine import _rust_core

    rust = _rust_core.propagate_inner_loop({}, [], 0.65, 5, True)
    py = py_propagate_reference({}, [], 0.65, 5, True)

    assert rust == py == {}


def test_single_edge():
    """a -> b, seed a=1.0, 1 iteration, no normalize."""
    from stg_engine import _rust_core

    seeds = {"a": 1.0}
    edges = [("a", "b", 1.0, 1.0, False)]

    rust = _rust_core.propagate_inner_loop(seeds, edges, 0.65, 1, False)
    py = py_propagate_reference(seeds, edges, 0.65, 1, False)

    assert_dicts_close(rust, py)
    assert abs(rust["b"] - 0.65) < 1e-6


def test_chain_propagation():
    """a -> b -> c -> d, 3 iterations."""
    from stg_engine import _rust_core

    seeds = {"a": 1.0}
    edges = [
        ("a", "b", 1.0, 1.0, False),
        ("b", "c", 1.0, 1.0, False),
        ("c", "d", 1.0, 1.0, False),
    ]

    rust = _rust_core.propagate_inner_loop(seeds, edges, 0.65, 3, False)
    py = py_propagate_reference(seeds, edges, 0.65, 3, False)

    assert_dicts_close(rust, py)
    assert rust["d"] > 0  # Should have reached d after 3 iterations


def test_branching():
    """Star graph: a -> {b, c, d, e}, weights compete."""
    from stg_engine import _rust_core

    seeds = {"a": 1.0}
    edges = [
        ("a", "b", 0.9, 0.8, False),
        ("a", "c", 0.7, 0.5, False),
        ("a", "d", 0.5, 1.0, True),  # virtual
        ("a", "e", 0.3, 0.3, False),
    ]

    rust = _rust_core.propagate_inner_loop(seeds, edges, 0.65, 1, False)
    py = py_propagate_reference(seeds, edges, 0.65, 1, False)

    assert_dicts_close(rust, py)


def test_budget_normalization():
    """Cyclic graph that would explode without normalization."""
    from stg_engine import _rust_core

    seeds = {"a": 1.0, "b": 1.0, "c": 1.0}
    edges = [
        ("a", "b", 1.0, 1.0, False),
        ("b", "c", 1.0, 1.0, False),
        ("c", "a", 1.0, 1.0, False),
        ("a", "c", 1.0, 1.0, False),
        ("b", "a", 1.0, 1.0, False),
    ]

    rust = _rust_core.propagate_inner_loop(seeds, edges, 0.65, 10, True)
    py = py_propagate_reference(seeds, edges, 0.65, 10, True)

    assert_dicts_close(rust, py)
    # Total should not exceed initial budget (3.0)
    assert sum(rust.values()) <= 3.0 + 1e-5


def test_realistic_medium_graph():
    """100 nodes, ~300 edges, 5 iterations — realistic STG-scale test."""
    from stg_engine import _rust_core
    import random

    random.seed(42)
    n = 100
    nodes = [f"node_{i}" for i in range(n)]
    seeds = {nodes[0]: 1.0, nodes[5]: 0.8, nodes[20]: 0.6}

    edges = []
    for i in range(n):
        # Each node connects to 3 random successors
        for _ in range(3):
            j = random.randint(0, n - 1)
            if j != i:
                conf = random.uniform(0.5, 1.0)
                sal = random.uniform(0.3, 1.0)
                is_virtual = random.random() < 0.1
                edges.append((nodes[i], nodes[j], conf, sal, is_virtual))

    rust = _rust_core.propagate_inner_loop(seeds, edges, 0.65, 5, True)
    py = py_propagate_reference(seeds, edges, 0.65, 5, True)

    assert_dicts_close(rust, py, tol=1e-4)  # Slightly looser for f32 vs f64
    print(f"  ✓ {len(rust)} nodes activated, parity OK")


def benchmark_rust_vs_python():
    """Benchmark Rust vs Python on a realistic-size graph."""
    from stg_engine import _rust_core
    import random

    random.seed(123)
    n = 8500  # STG current scale
    edge_count = 12000

    nodes = [f"node_{i}" for i in range(n)]
    seeds = {nodes[i]: random.uniform(0.5, 1.0) for i in range(20)}

    edges = []
    for _ in range(edge_count):
        src = random.randint(0, n - 1)
        tgt = random.randint(0, n - 1)
        if src != tgt:
            conf = random.uniform(0.5, 1.0)
            sal = random.uniform(0.3, 1.0)
            is_virtual = random.random() < 0.1
            edges.append((nodes[src], nodes[tgt], conf, sal, is_virtual))

    print(f"\n  Graph: {n} nodes, {len(edges)} edges, {len(seeds)} seeds")
    print(f"  Iterations: 5, decay: 0.65, normalize: True")
    print()

    # Warm up
    _ = _rust_core.propagate_inner_loop(dict(seeds), edges, 0.65, 5, True)
    _ = py_propagate_reference(dict(seeds), edges, 0.65, 5, True)

    # Time Rust
    runs = 20
    t0 = time.perf_counter()
    for _ in range(runs):
        rust_result = _rust_core.propagate_inner_loop(dict(seeds), edges, 0.65, 5, True)
    rust_time = (time.perf_counter() - t0) / runs

    # Time Python
    runs_py = 5  # Python is much slower; fewer runs
    t0 = time.perf_counter()
    for _ in range(runs_py):
        py_result = py_propagate_reference(dict(seeds), edges, 0.65, 5, True)
    py_time = (time.perf_counter() - t0) / runs_py

    print(f"  Python:  {py_time*1000:8.2f} ms / call")
    print(f"  Rust:    {rust_time*1000:8.2f} ms / call")
    print(f"  Speedup: {py_time / rust_time:8.1f}x")
    print()

    # Verify parity
    assert_dicts_close(rust_result, py_result, tol=1e-3)
    print(f"  ✓ Parity verified ({len(rust_result)} nodes match)")


# ─────────────────────────────────────────────────────────────────────
# Run as a script
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("import_rust_core", test_import_rust_core),
        ("empty_inputs", test_empty_inputs),
        ("single_edge", test_single_edge),
        ("chain_propagation", test_chain_propagation),
        ("branching", test_branching),
        ("budget_normalization", test_budget_normalization),
        ("realistic_medium_graph", test_realistic_medium_graph),
    ]

    failed = 0
    for name, fn in tests:
        try:
            print(f"  test_{name}...")
            fn()
            print(f"  ✓ PASS")
        except Exception as e:
            print(f"  ✗ FAIL: {e}")
            failed += 1

    print()
    print(f"  {len(tests) - failed}/{len(tests)} parity tests passed")

    if failed == 0:
        print()
        print("  ── BENCHMARK ──")
        benchmark_rust_vs_python()

    sys.exit(failed)
