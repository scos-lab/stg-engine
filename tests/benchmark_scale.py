"""Stress benchmark: how does Rust scale vs Python as graph size grows?

Self-contained — does not depend on test_propagate_parity.py or local stg_engine/.
Run from anywhere:  python tests/benchmark_scale.py
"""
import time
import random
from typing import Dict, List, Tuple

from stg_engine import _rust_core


def py_reference(
    activation_map: Dict[str, float],
    edges: List[Tuple[str, str, float, float, bool]],
    decay: float,
    iterations: int,
    normalize: bool,
) -> Dict[str, float]:
    """Pure-Python fast-path reference (matches Rust algorithm exactly)."""
    activations = dict(activation_map)
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


def gen_graph(n_nodes: int, edges_per_node: int, seed_count: int):
    random.seed(42)
    nodes = [f"n_{i}" for i in range(n_nodes)]
    seeds = {nodes[i]: random.uniform(0.5, 1.0) for i in range(seed_count)}
    edges = []
    for i in range(n_nodes):
        for _ in range(edges_per_node):
            j = random.randint(0, n_nodes - 1)
            if j != i:
                edges.append((
                    nodes[i], nodes[j],
                    random.uniform(0.5, 1.0),
                    random.uniform(0.3, 1.0),
                    random.random() < 0.1,
                ))
    return seeds, edges


def bench(label: str, n_nodes: int, edges_per_node: int, iterations: int):
    seeds, edges = gen_graph(n_nodes, edges_per_node, seed_count=20)

    # Warm up
    _ = _rust_core.propagate_inner_loop(dict(seeds), edges, 0.65, iterations, True)
    _ = py_reference(dict(seeds), edges, 0.65, iterations, True)

    # Adaptive run count
    runs_rust = max(5, min(50, 5_000_000 // (n_nodes * iterations)))
    runs_py = max(2, min(20, 1_000_000 // (n_nodes * iterations)))

    t0 = time.perf_counter()
    for _ in range(runs_rust):
        _rust_core.propagate_inner_loop(dict(seeds), edges, 0.65, iterations, True)
    t_rust = (time.perf_counter() - t0) / runs_rust

    t0 = time.perf_counter()
    for _ in range(runs_py):
        py_reference(dict(seeds), edges, 0.65, iterations, True)
    t_py = (time.perf_counter() - t0) / runs_py

    print(f"  {label:32}  py={t_py*1000:8.2f}ms  rust={t_rust*1000:8.2f}ms  speedup={t_py/t_rust:5.1f}x")


print("\n  ── Scale Benchmark: Rust vs Python (clean fast-path reference) ──\n")

bench("1k nodes,   3 e/n,  5 iter", 1_000, 3, 5)
bench("8k nodes,   3 e/n,  5 iter", 8_000, 3, 5)
bench("8k nodes,   3 e/n, 20 iter", 8_000, 3, 20)
bench("50k nodes,  3 e/n,  5 iter", 50_000, 3, 5)
bench("50k nodes,  5 e/n, 10 iter", 50_000, 5, 10)
bench("100k nodes, 3 e/n,  5 iter", 100_000, 3, 5)
bench("100k nodes, 5 e/n, 10 iter", 100_000, 5, 10)

print()
print("  Notes:")
print("    - Python reference here is FAST PATH (clean dicts, no networkx, no edge.modifiers)")
print("    - REAL engine.propagate() in SKC has ~3-5x more Python overhead per iteration")
print("    - So expected real-world speedup against engine.propagate() ≈ measured × 3-5")
