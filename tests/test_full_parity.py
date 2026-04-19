"""Full parity test suite for v0.2.0a1 — verifies all 3 Rust algorithms
match their pure-Python reference implementations within FP tolerance.

Algorithms tested:
  1. propagate_inner_loop  (vs Python reference of engine._propagate_from_seeds fast path)
  2. hebbian_update         (vs Python reference of HebbianLearner.learn_from_propagation)
  3. compute_elevations     (vs Python reference of gravity._compute_all_elevations)

Run:  python tests/test_full_parity.py
"""
import math
import random
import sys
import time
from typing import Dict, List, Set, Tuple

from stg_engine import _rust_core


# ═══════════════════════════════════════════════════════════════════
# Python reference implementations (must mirror Rust exactly)
# ═══════════════════════════════════════════════════════════════════


def py_propagate(
    activation_map: Dict[str, float],
    edges: List[Tuple[str, str, float, float, bool]],
    decay: float,
    iterations: int,
    normalize: bool,
) -> Dict[str, float]:
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


def py_hebbian(
    activation_map: Dict[str, float],
    edges: List[Tuple[str, str, float, bool]],
    strengthen_rate: float = 0.05,
    weaken_rate: float = 0.01,
    ceiling: float = 1.0,
    floor: float = 0.01,
    activation_threshold: float = 0.1,
    weaken_activation_threshold: float = 0.15,
) -> List[Tuple[int, float, int]]:
    """Returns list of (edge_index, new_salience, action_code)."""
    updates: List[Tuple[int, float, int]] = []

    active_set = {n for n, a in activation_map.items() if a >= activation_threshold}
    if not active_set:
        return updates
    weaken_set = {n for n, a in activation_map.items() if a >= weaken_activation_threshold}

    for idx, (src, tgt, salience, skip) in enumerate(edges):
        if skip:
            continue
        src_active = src in active_set
        tgt_active = tgt in active_set

        if src_active and tgt_active:
            old_sal = salience
            if old_sal >= ceiling:
                continue
            act_src = activation_map.get(src, 0.0)
            act_tgt = activation_map.get(tgt, 0.0)
            modulation = min(math.sqrt(act_src * act_tgt), 1.0)
            effective_alpha = strengthen_rate * modulation
            new_sal = min(old_sal + effective_alpha * (ceiling - old_sal), ceiling)
            if abs(new_sal - old_sal) > 1e-10:
                updates.append((idx, new_sal, 1))

        elif src in weaken_set and not tgt_active:
            old_sal = salience
            if old_sal <= floor:
                continue
            act_src = activation_map.get(src, 0.0)
            effective_alpha = weaken_rate * act_src
            new_sal = max(old_sal - effective_alpha * (old_sal - floor), floor)
            if abs(new_sal - old_sal) > 1e-10:
                updates.append((idx, new_sal, 2))

    return updates


def py_elevations(
    nodes: List[str],
    communities: List[List[str]],
    importance: Dict[str, float],
    neighbors: Dict[str, List[str]],
) -> Dict[str, float]:
    if not nodes:
        return {}

    node_to_comm: Dict[str, int] = {}
    comm_sizes: Dict[int, int] = {}
    for cid, members in enumerate(communities):
        comm_sizes[cid] = len(members)
        for m in members:
            node_to_comm[m] = cid

    cross_edges: Dict[str, int] = {}
    for node in nodes:
        node_comm = node_to_comm.get(node, -1)
        seen: Set[str] = set()
        for neighbor in neighbors.get(node, []):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            neighbor_comm = node_to_comm.get(neighbor, -1)
            if neighbor_comm != node_comm and neighbor_comm >= 0:
                cross_edges[node] = cross_edges.get(node, 0) + 1

    comm_max: Dict[int, float] = {}
    for cid, members in enumerate(communities):
        max_imp = max((importance.get(m, 0.0) for m in members), default=0.0)
        comm_max[cid] = max_imp if max_imp > 0 else 1.0

    elevation: Dict[str, float] = {}
    for node in nodes:
        cid = node_to_comm.get(node, -1)
        if cid < 0:
            elevation[node] = 0.01
            continue
        raw_imp = importance.get(node, 0.0)
        local_imp = raw_imp / comm_max[cid]
        bridge = 1.0 + math.log1p(cross_edges.get(node, 0))
        comm_size = comm_sizes.get(cid, 1)
        comm_weight = math.log1p(comm_size)
        elevation[node] = local_imp * bridge * comm_weight

    return elevation


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def assert_dicts_close(a, b, tol=1e-5, label=""):
    assert set(a.keys()) == set(b.keys()), f"{label}: key set mismatch"
    for k in a:
        diff = abs(a[k] - b[k])
        assert diff < tol, f"{label}: {k!r} rust={a[k]:.10f} py={b[k]:.10f} diff={diff:.2e}"


def assert_updates_match(rust_updates, py_updates, tol=1e-5):
    rust_dict = {(idx, action): sal for idx, sal, action in rust_updates}
    py_dict = {(idx, action): sal for idx, sal, action in py_updates}
    assert set(rust_dict.keys()) == set(py_dict.keys()), (
        f"Update sets differ: rust={set(rust_dict)}, py={set(py_dict)}"
    )
    for k in rust_dict:
        diff = abs(rust_dict[k] - py_dict[k])
        assert diff < tol, f"Mismatch at {k}: rust={rust_dict[k]} py={py_dict[k]}"


# ═══════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════


def test_module_loads():
    print("\n[1] Module load + metadata")
    assert _rust_core.__version__ == "0.2.0-alpha.1"
    assert _rust_core.__license__ == "BUSL-1.1"
    assert hasattr(_rust_core, "propagate_inner_loop")
    assert hasattr(_rust_core, "hebbian_update")
    assert hasattr(_rust_core, "compute_elevations")
    print("    ✓ all 3 functions exposed")
    print(f"    ✓ version {_rust_core.__version__}, license {_rust_core.__license__}")


def test_propagate_realistic():
    print("\n[2] propagate_inner_loop — realistic 1k node graph")
    random.seed(42)
    n = 1000
    nodes = [f"n_{i}" for i in range(n)]
    seeds = {nodes[i]: random.uniform(0.5, 1.0) for i in range(10)}
    edges = []
    for i in range(n):
        for _ in range(3):
            j = random.randint(0, n - 1)
            if j != i:
                edges.append((
                    nodes[i], nodes[j],
                    random.uniform(0.5, 1.0),
                    random.uniform(0.3, 1.0),
                    random.random() < 0.1,
                ))

    rust = _rust_core.propagate_inner_loop(dict(seeds), edges, 0.65, 5, True)
    py = py_propagate(dict(seeds), edges, 0.65, 5, True)
    assert_dicts_close(rust, py, tol=1e-3, label="propagate")
    print(f"    ✓ {len(rust)} nodes, parity verified")


def test_hebbian_realistic():
    print("\n[3] hebbian_update — realistic 500 edge update")
    random.seed(42)
    n = 100
    nodes = [f"n_{i}" for i in range(n)]

    # Build activation map (simulate post-propagate)
    activations = {nodes[i]: random.uniform(0.0, 1.0) for i in range(n)}

    # Build edges (some with skip flag)
    edges = []
    for _ in range(500):
        src = nodes[random.randint(0, n - 1)]
        tgt = nodes[random.randint(0, n - 1)]
        if src == tgt:
            continue
        salience = random.uniform(0.05, 0.95)
        skip = random.random() < 0.1
        edges.append((src, tgt, salience, skip))

    rust = _rust_core.hebbian_update(
        dict(activations), edges,
        strengthen_rate=0.05,
        weaken_rate=0.01,
        ceiling=1.0,
        floor=0.01,
        activation_threshold=0.1,
        weaken_activation_threshold=0.15,
    )
    py = py_hebbian(dict(activations), edges)

    assert_updates_match(rust, py, tol=1e-5)
    strengthen_count = sum(1 for _, _, a in rust if a == 1)
    weaken_count = sum(1 for _, _, a in rust if a == 2)
    print(f"    ✓ {strengthen_count} strengthen + {weaken_count} weaken updates, parity verified")


def test_elevations_realistic():
    print("\n[4] compute_elevations — 200-node graph with 5 communities")
    random.seed(42)
    n = 200
    nodes = [f"n_{i}" for i in range(n)]

    # Partition nodes into 5 communities
    communities: List[List[str]] = [[] for _ in range(5)]
    for i, node in enumerate(nodes):
        communities[i % 5].append(node)

    # Random importance scores
    importance = {node: random.uniform(0.0, 1.0) for node in nodes}

    # Build neighbors (undirected, deduplicated)
    raw_edges = []
    for _ in range(600):
        i, j = random.randint(0, n - 1), random.randint(0, n - 1)
        if i != j:
            raw_edges.append((nodes[i], nodes[j]))

    neighbors_set: Dict[str, Set[str]] = {}
    for src, tgt in raw_edges:
        neighbors_set.setdefault(src, set()).add(tgt)
        neighbors_set.setdefault(tgt, set()).add(src)
    neighbors = {k: list(v) for k, v in neighbors_set.items()}

    rust = _rust_core.compute_elevations(nodes, communities, importance, neighbors)
    py = py_elevations(nodes, communities, importance, neighbors)

    assert_dicts_close(rust, py, tol=1e-5, label="elevations")
    print(f"    ✓ {len(rust)} elevations computed, parity verified")


def test_combined_workflow():
    print("\n[5] Combined: propagate → hebbian → elevations")
    random.seed(42)
    n = 50
    nodes = [f"n_{i}" for i in range(n)]
    seeds = {nodes[0]: 1.0, nodes[5]: 0.8}
    edges_propagate = []
    for i in range(n):
        for _ in range(2):
            j = random.randint(0, n - 1)
            if j != i:
                edges_propagate.append((
                    nodes[i], nodes[j],
                    random.uniform(0.5, 1.0),
                    random.uniform(0.3, 1.0),
                    False,
                ))

    # Step 1: propagate
    activations = _rust_core.propagate_inner_loop(seeds, edges_propagate, 0.65, 5, True)
    print(f"    propagate → {len(activations)} active nodes")

    # Step 2: hebbian using post-propagate activations
    edges_hebbian = [(s, t, c * sal, False) for s, t, c, sal, _ in edges_propagate]
    updates = _rust_core.hebbian_update(activations, edges_hebbian)
    print(f"    hebbian   → {len(updates)} edges updated")

    # Step 3: elevations using activations as importance
    communities = [[nodes[i] for i in range(n) if i % 3 == c] for c in range(3)]
    neighbors_set: Dict[str, Set[str]] = {}
    for src, tgt, *_ in edges_propagate:
        neighbors_set.setdefault(src, set()).add(tgt)
        neighbors_set.setdefault(tgt, set()).add(src)
    neighbors = {k: list(v) for k, v in neighbors_set.items()}

    elevations = _rust_core.compute_elevations(nodes, communities, activations, neighbors)
    print(f"    elevations → {len(elevations)} nodes")
    print(f"    ✓ all 3 algorithms chain together")


def benchmark_all():
    print("\n[BENCH] Hot-path performance on 8500-node graph")
    random.seed(42)
    n = 8500
    edge_count = 12000
    nodes = [f"n_{i}" for i in range(n)]
    seeds = {nodes[i]: random.uniform(0.5, 1.0) for i in range(20)}
    edges_propagate = []
    for _ in range(edge_count):
        src = random.randint(0, n - 1)
        tgt = random.randint(0, n - 1)
        if src != tgt:
            edges_propagate.append((
                nodes[src], nodes[tgt],
                random.uniform(0.5, 1.0),
                random.uniform(0.3, 1.0),
                random.random() < 0.1,
            ))

    edges_hebbian = [(s, t, c * sal, False) for s, t, c, sal, _ in edges_propagate]
    activations_for_hebbian = {nodes[i]: random.uniform(0.0, 1.0) for i in range(n)}

    # Build neighbors for elevations
    neighbors_set: Dict[str, Set[str]] = {}
    for src, tgt, *_ in edges_propagate:
        neighbors_set.setdefault(src, set()).add(tgt)
        neighbors_set.setdefault(tgt, set()).add(src)
    neighbors = {k: list(v) for k, v in neighbors_set.items()}
    communities = [[nodes[i] for i in range(n) if i % 5 == c] for c in range(5)]
    importance = {nodes[i]: random.uniform(0.0, 1.0) for i in range(n)}

    def time_it(label, fn, runs):
        # Warm up
        fn()
        t0 = time.perf_counter()
        for _ in range(runs):
            fn()
        elapsed = (time.perf_counter() - t0) / runs * 1000
        print(f"    {label:30}  {elapsed:8.2f} ms")

    print()
    time_it("propagate (Rust)", lambda: _rust_core.propagate_inner_loop(dict(seeds), edges_propagate, 0.65, 5, True), 20)
    time_it("propagate (Python)", lambda: py_propagate(dict(seeds), edges_propagate, 0.65, 5, True), 5)
    print()
    time_it("hebbian (Rust)", lambda: _rust_core.hebbian_update(dict(activations_for_hebbian), edges_hebbian), 20)
    time_it("hebbian (Python)", lambda: py_hebbian(dict(activations_for_hebbian), edges_hebbian), 5)
    print()
    time_it("elevations (Rust)", lambda: _rust_core.compute_elevations(nodes, communities, importance, neighbors), 20)
    time_it("elevations (Python)", lambda: py_elevations(nodes, communities, importance, neighbors), 5)


if __name__ == "__main__":
    tests = [
        test_module_loads,
        test_propagate_realistic,
        test_hebbian_realistic,
        test_elevations_realistic,
        test_combined_workflow,
    ]

    failed = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as e:
            print(f"    ✗ FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"    ✗ ERROR: {type(e).__name__}: {e}")
            failed += 1

    print()
    print(f"  ── {len(tests) - failed}/{len(tests)} parity tests passed ──")

    if failed == 0:
        benchmark_all()

    sys.exit(failed)
