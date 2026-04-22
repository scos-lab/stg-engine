"""Pure-Python fallback for the three hot-path algorithms.

Exposes the same three functions as `_rust_core` (the optional Rust
extension) with identical signatures, return types, and semantics.
Used automatically when the Rust extension is unavailable — e.g. when
stg-engine is installed from a pure-Python wheel, or when the user has
no Rust toolchain to compile the sdist.

Reference: `rust/src/{propagate,hebbian,gravity}.rs`.

Performance note: this is ~30-100x slower than the Rust implementation
for large graphs (~8500 nodes), but correctness is identical within
float32 rounding tolerance (tests use 1e-5 / 1e-6).
"""

from __future__ import annotations

from math import log1p, sqrt
from typing import Dict, List, Tuple

__version__ = "fallback-pure-python"
__license__ = "BUSL-1.1"


# ═══════════════════════════════════════════════════════════════════
# propagate_inner_loop — spreading activation iteration
# Reference: rust/src/propagate.rs
# ═══════════════════════════════════════════════════════════════════


def propagate_inner_loop(
    activation_map: Dict[str, float],
    edges: List[Tuple[str, str, float, float, bool]],
    decay: float = 0.65,
    iterations: int = 5,
    normalize: bool = True,
) -> Dict[str, float]:
    """Spreading activation. See rust/src/propagate.rs for algorithm spec."""
    # Copy to avoid mutating caller's dict
    activations: Dict[str, float] = dict(activation_map)

    # Pre-compute adjacency: source → list of (target, weight)
    adjacency: Dict[str, List[Tuple[str, float]]] = {}
    for src, tgt, conf, sal, is_virtual in edges:
        w = conf * sal
        if is_virtual:
            w *= 0.5
        adjacency.setdefault(src, []).append((tgt, w))

    activation_budget = sum(activations.values()) if normalize else 0.0

    for _ in range(iterations):
        # Snapshot of currently-active nodes (activation > 0.01), sorted for
        # determinism — matches Rust line 129.
        active_nodes = sorted(
            ((k, v) for k, v in activations.items() if v > 0.01),
            key=lambda kv: kv[0],
        )

        delta_map: Dict[str, float] = {}

        for node, activation in active_nodes:
            successors = adjacency.get(node)
            if not successors:
                continue

            total_weight = sum(w for _, w in successors)
            if total_weight <= 0.0:
                continue

            activation_decayed = activation * decay
            for tgt, w in successors:
                spread = activation_decayed * w / total_weight
                delta_map[tgt] = delta_map.get(tgt, 0.0) + spread

        # Apply deltas in sorted order (mirrors Rust line 161 for determinism)
        for node, delta in sorted(delta_map.items(), key=lambda kv: kv[0]):
            new_val = activations.get(node, 0.0) + delta
            activations[node] = new_val if new_val >= 0.0 else 0.0

        # Linear budget rescale (Vehicle 12)
        if normalize and activation_budget > 0.0:
            current_total = sum(activations.values())
            if current_total > activation_budget and current_total > 0.0:
                scale = activation_budget / current_total
                for k in activations:
                    activations[k] *= scale

    return activations


# ═══════════════════════════════════════════════════════════════════
# hebbian_update — co-activation-driven salience update
# Reference: rust/src/hebbian.rs
# ═══════════════════════════════════════════════════════════════════


def hebbian_update(
    activation_map: Dict[str, float],
    edges: List[Tuple[str, str, float, bool]],
    strengthen_rate: float = 0.05,
    weaken_rate: float = 0.01,
    ceiling: float = 1.0,
    floor: float = 0.01,
    activation_threshold: float = 0.1,
    weaken_activation_threshold: float = 0.15,
) -> List[Tuple[int, float, int]]:
    """Hebbian salience update. See rust/src/hebbian.rs for algorithm spec.

    Returns list of (edge_index, new_salience, action_code).
    action_code: 1=strengthen, 2=weaken.
    """
    updates: List[Tuple[int, float, int]] = []

    active_set = {k for k, a in activation_map.items() if a >= activation_threshold}
    if not active_set:
        return updates

    weaken_set = {
        k for k, a in activation_map.items() if a >= weaken_activation_threshold
    }

    for idx, (src, tgt, salience, skip) in enumerate(edges):
        if skip:
            continue

        src_active = src in active_set
        tgt_active = tgt in active_set

        if src_active and tgt_active:
            # Strengthen — Hebbian co-activation
            old_sal = salience
            if old_sal >= ceiling:
                continue

            act_src = activation_map.get(src, 0.0)
            act_tgt = activation_map.get(tgt, 0.0)

            modulation = min(sqrt(act_src * act_tgt), 1.0)
            effective_alpha = strengthen_rate * modulation
            new_sal = min(old_sal + effective_alpha * (ceiling - old_sal), ceiling)

            if abs(new_sal - old_sal) > 1e-10:
                updates.append((idx, new_sal, 1))

        elif src in weaken_set and not tgt_active:
            # Weaken — lateral inhibition (strong src, inactive tgt)
            old_sal = salience
            if old_sal <= floor:
                continue

            act_src = activation_map.get(src, 0.0)
            effective_alpha = weaken_rate * act_src
            new_sal = max(old_sal - effective_alpha * (old_sal - floor), floor)

            if abs(new_sal - old_sal) > 1e-10:
                updates.append((idx, new_sal, 2))

    return updates


# ═══════════════════════════════════════════════════════════════════
# compute_elevations — gravity-based structural importance
# Reference: rust/src/gravity.rs
# ═══════════════════════════════════════════════════════════════════


def compute_elevations(
    nodes: List[str],
    communities: List[List[str]],
    importance: Dict[str, float],
    neighbors: Dict[str, List[str]],
) -> Dict[str, float]:
    """Structural elevation per node. See rust/src/gravity.rs for spec."""
    if not nodes:
        return {}

    # Community membership lookup
    node_to_comm: Dict[str, int] = {}
    comm_sizes: Dict[int, int] = {}
    for cid, members in enumerate(communities):
        comm_sizes[cid] = len(members)
        for m in members:
            node_to_comm[m] = cid

    # Cross-community edge counts (undirected, deduplicated)
    cross_edges: Dict[str, int] = {}
    for node in nodes:
        node_comm = node_to_comm.get(node, -1)
        neigh_list = neighbors.get(node)
        if not neigh_list:
            continue
        seen = set()
        for n in neigh_list:
            if n in seen:
                continue
            seen.add(n)
            neighbor_comm = node_to_comm.get(n, -1)
            if neighbor_comm != node_comm and neighbor_comm >= 0:
                cross_edges[node] = cross_edges.get(node, 0) + 1

    # Max importance per community (for local normalization)
    comm_max_importance: Dict[int, float] = {}
    for cid, members in enumerate(communities):
        max_imp = 0.0
        for m in members:
            imp = importance.get(m, 0.0)
            if imp > max_imp:
                max_imp = imp
        comm_max_importance[cid] = max_imp if max_imp > 0.0 else 1.0

    elevation: Dict[str, float] = {}
    for node in nodes:
        comm_id = node_to_comm.get(node, -1)
        if comm_id < 0:
            elevation[node] = 0.01
            continue

        raw_imp = importance.get(node, 0.0)
        max_imp = comm_max_importance.get(comm_id, 1.0)
        local_imp = raw_imp / max_imp

        cross_count = cross_edges.get(node, 0)
        bridge = 1.0 + log1p(cross_count)

        comm_size = comm_sizes.get(comm_id, 1)
        comm_weight = log1p(comm_size)

        elevation[node] = local_imp * bridge * comm_weight

    return elevation
