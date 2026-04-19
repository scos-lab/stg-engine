"""STG Co-Activation Edge Creation — Closing the Hebbian loop.

Phase 11B: Nodes that propagate together, edge together.

Theoretical basis:
  - Braitenberg Mnemotrix: co-occurrence → wiring (Vehicles Ch.12-14)
  - Kanerva SDM: distributed write to all activated locations (Ch.2)
  - Dayan & Abbott STDP: zero-delay Hebbian (TNS Ch.8)

Four pure functions — no state, operates on engine + telemetry data.
Follows inhibition.py and temporal.py pattern.

Safeguards (8 layers):
  1. Top-K limit (20) — O(190) pairs per propagation, not O(n^2)
  2. Min cooccurrence (5) — noise never creates edges
  3. Max edges per session (10) — prevents runaway growth
  4. Weak edges — confidence ≤ 0.5, salience = 0.3
  5. Pruner removes unused — self-cleaning
  6. Degree cap (5) — per-node limit on coactivation edges [Kanerva capacity]
  7. Hub exclusion (top-10) — hubs already connect to everything [Braitenberg V12]
  8. Short lifespan (14 days) — faster pruning than knowledge edges [anti-epilepsy]
"""

import time as _time
from collections import Counter
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from stg_engine.engine import STGEngine
    from stg_engine.types import LearningEvent


# ═══════════════════════════════════════════════════════════
# Data Collection
# ═══════════════════════════════════════════════════════════


def record_cooccurrence(
    activation_map: Dict[str, float],
    top_k: int = 20,
    activation_threshold: float = 0.1,
) -> Counter:
    """Extract co-activated pairs from a single propagation result.

    Only tracks pairs among top-K activated nodes to prevent O(n^2) explosion.
    8000 nodes → 32M pairs. Top-20 → max 190 pairs.

    Args:
        activation_map: Node name → activation level from propagate()
        top_k: Number of top nodes to track pairs for
        activation_threshold: Minimum activation to count as active

    Returns:
        Counter of sorted (nodeA, nodeB) pairs where A < B lexicographically.
    """
    # Get top-K activated nodes above threshold
    candidates = sorted(
        ((n, a) for n, a in activation_map.items() if a >= activation_threshold),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    names = [n for n, _ in candidates]
    pairs: Counter = Counter()

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pair = (a, b) if a < b else (b, a)
            pairs[pair] += 1

    return pairs


# ═══════════════════════════════════════════════════════════
# Candidate Discovery
# ═══════════════════════════════════════════════════════════


def find_coactivation_candidates(
    engine: "STGEngine",
    stg_path: str,
    min_cooccurrence: int = 5,
    max_candidates: int = 10,
    max_coactivation_per_node: int = 5,
    exclude_top_hubs: int = 10,
) -> List[Tuple[str, str, int]]:
    """Find node pairs with high co-activation but no existing edge.

    Applies safeguards:
      - Excludes pairs where an edge already exists
      - Excludes top-N hub nodes by degree (Braitenberg V12 threshold control)
      - Enforces per-node degree cap on coactivation edges (Kanerva capacity)

    Args:
        engine: Current STGEngine (to check existing edges)
        stg_path: Path to .stg file (to read telemetry cooccurrence)
        min_cooccurrence: Minimum co-activation count to consider
        max_candidates: Maximum candidates to return
        max_coactivation_per_node: Max coactivation edges per node (degree cap)
        exclude_top_hubs: Skip nodes in top-N by total degree

    Returns:
        List of (node_a, node_b, count) sorted by count descending
    """
    import sqlite3
    from pathlib import Path

    path = Path(stg_path)
    if not path.exists():
        return []

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row

    # Check if cooccurrence table exists
    tables = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "telemetry_cooccurrence" not in tables:
        conn.close()
        return []

    rows = conn.execute(
        "SELECT node_a, node_b, cooccurrence_count "
        "FROM telemetry_cooccurrence "
        "WHERE cooccurrence_count >= ? "
        "ORDER BY cooccurrence_count DESC",
        (min_cooccurrence,),
    ).fetchall()
    conn.close()

    if not rows:
        return []

    # Build hub exclusion set (top-N by degree)
    hub_set: set = set()
    if exclude_top_hubs > 0 and hasattr(engine, '_graph'):
        degree_sorted = sorted(
            engine._graph.degree(),
            key=lambda x: x[1],
            reverse=True,
        )
        hub_set = {name for name, _ in degree_sorted[:exclude_top_hubs]}

    # Count existing coactivation edges per node
    coact_degree: Counter = Counter()
    for edge in engine._edges:
        if edge.modifiers.get("edge_class") == "coactivation":
            coact_degree[edge.source] += 1
            coact_degree[edge.target] += 1

    candidates: List[Tuple[str, str, int]] = []

    _nk = engine._nk

    for row in rows:
        a, b, count = row["node_a"], row["node_b"], row["cooccurrence_count"]
        nk_a, nk_b = _nk(a), _nk(b)

        # Skip if either node is a hub
        if nk_a in hub_set or nk_b in hub_set:
            continue

        # Skip if edge already exists (either direction)
        if engine._edges_lookup.get((nk_a, nk_b)) is not None:
            continue
        if engine._edges_lookup.get((nk_b, nk_a)) is not None:
            continue

        # Skip if either node doesn't exist in graph
        if nk_a not in engine._nodes or nk_b not in engine._nodes:
            continue

        # Degree cap: skip if either node already has max coactivation edges
        if coact_degree[a] >= max_coactivation_per_node:
            continue
        if coact_degree[b] >= max_coactivation_per_node:
            continue

        candidates.append((a, b, count))

        # Track that we'd add edges (for subsequent cap checks in this batch)
        coact_degree[a] += 1
        coact_degree[b] += 1

        if len(candidates) >= max_candidates:
            break

    return candidates


# ═══════════════════════════════════════════════════════════
# Edge Creation
# ═══════════════════════════════════════════════════════════


def record_coactivation_event(
    stg_path: str,
    candidates_found: int,
    edges_created: int,
    candidates_detail: List[Tuple[str, str, int]],
    skipped_hub: int = 0,
    skipped_cap: int = 0,
) -> None:
    """Record co-activation edge creation event to telemetry.

    Persists metadata about each creation cycle for parameter tuning:
      - How many candidates were found vs created
      - What cooccurrence counts triggered creation
      - How many were skipped by hub exclusion and degree cap

    Args:
        stg_path: Path to .stg file
        candidates_found: Total candidates before safeguards
        edges_created: Actually created edges
        candidates_detail: List of (node_a, node_b, count) that were created
        skipped_hub: Candidates skipped due to hub exclusion
        skipped_cap: Candidates skipped due to degree cap
    """
    import json
    import sqlite3
    from pathlib import Path

    path = Path(stg_path)
    if not path.exists():
        return

    conn = sqlite3.connect(str(path))

    # Ensure table exists
    conn.execute(
        "CREATE TABLE IF NOT EXISTS telemetry_coactivation_events ("
        "    id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "    timestamp REAL NOT NULL,"
        "    candidates_found INTEGER DEFAULT 0,"
        "    edges_created INTEGER DEFAULT 0,"
        "    skipped_hub INTEGER DEFAULT 0,"
        "    skipped_cap INTEGER DEFAULT 0,"
        "    avg_cooccurrence REAL DEFAULT 0.0,"
        "    min_cooccurrence INTEGER DEFAULT 0,"
        "    max_cooccurrence INTEGER DEFAULT 0,"
        "    created_pairs_json TEXT DEFAULT '[]'"
        ")"
    )

    now = _time.time()
    counts = [c for _, _, c in candidates_detail] if candidates_detail else []
    avg_co = sum(counts) / len(counts) if counts else 0.0
    min_co = min(counts) if counts else 0
    max_co = max(counts) if counts else 0
    pairs_json = json.dumps(
        [{"a": a, "b": b, "count": c} for a, b, c in candidates_detail],
        ensure_ascii=False,
    )

    conn.execute(
        "INSERT INTO telemetry_coactivation_events "
        "(timestamp, candidates_found, edges_created, skipped_hub, skipped_cap, "
        "avg_cooccurrence, min_cooccurrence, max_cooccurrence, created_pairs_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (now, candidates_found, edges_created, skipped_hub, skipped_cap,
         avg_co, min_co, max_co, pairs_json),
    )
    conn.commit()
    conn.close()


def create_coactivation_edges(
    engine: "STGEngine",
    candidates: List[Tuple[str, str, int]],
) -> List["LearningEvent"]:
    """Create weak edges from co-activation candidates.

    Each edge:
      - confidence = min(0.5, count × 0.05)  [conservative]
      - salience = 0.3  [low — must earn salience through use]
      - rule = "empirical"
      - edge_class = "coactivation"  [in modifiers, for tracking]
      - coactivation_created_at = timestamp  [for 14-day pruning window]

    Args:
        engine: STGEngine instance (modified in place)
        candidates: List of (node_a, node_b, cooccurrence_count)

    Returns:
        List of LearningEvent records for audit trail
    """
    from stg_engine.types import LearningEvent

    now = _time.time()
    events: List[LearningEvent] = []

    _nk = engine._nk

    for node_a, node_b, count in candidates:
        # Double-check edge doesn't exist (race condition guard)
        nk_a, nk_b = _nk(node_a), _nk(node_b)
        if engine._edges_lookup.get((nk_a, nk_b)) is not None:
            continue
        if engine._edges_lookup.get((nk_b, nk_a)) is not None:
            continue

        confidence = min(0.5, count * 0.05)

        edge = engine.add_edge(
            source=node_a,
            target=node_b,
            confidence=confidence,
            strength=0.5,
            rule="empirical",
            edge_class="coactivation",
            coactivation_created_at=now,
            cooccurrence_count=count,
            description=f"Auto-created from {count} co-activations",
        )

        # Override salience (add_edge uses default 0.5)
        edge.salience = 0.3
        edge.last_used = now

        events.append(LearningEvent(
            event_type="coactivation_create",
            source=node_a,
            target=node_b,
            old_confidence=0.0,
            new_confidence=confidence,
            timestamp=now,
            trigger="coactivation",
        ))

    if events:
        engine._invalidate_caches()

    return events


# ═══════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════


def coactivation_report(stg_path: str, top_n: int = 20) -> str:
    """Human-readable report of top co-occurring pairs and their edge status.

    Args:
        stg_path: Path to .stg file
        top_n: Number of top pairs to show

    Returns:
        Formatted report string
    """
    import sqlite3
    from pathlib import Path

    path = Path(stg_path)
    if not path.exists():
        return "No .stg file found."

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row

    tables = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "telemetry_cooccurrence" not in tables:
        conn.close()
        return "No co-occurrence data. Run some propagations first."

    # Total pairs tracked
    total = conn.execute(
        "SELECT COUNT(*) FROM telemetry_cooccurrence"
    ).fetchone()[0]

    total_above = conn.execute(
        "SELECT COUNT(*) FROM telemetry_cooccurrence WHERE cooccurrence_count >= 5"
    ).fetchone()[0]

    rows = conn.execute(
        "SELECT node_a, node_b, cooccurrence_count, first_seen, last_seen "
        "FROM telemetry_cooccurrence "
        "ORDER BY cooccurrence_count DESC LIMIT ?",
        (top_n,),
    ).fetchall()
    conn.close()

    if not rows:
        return "No co-occurrence data recorded yet."

    lines = []
    lines.append("=" * 70)
    lines.append("Co-Activation Report")
    lines.append("=" * 70)
    lines.append(f"Total pairs tracked: {total}")
    lines.append(f"Pairs above threshold (≥5): {total_above}")
    lines.append("")
    lines.append(f"  {'Node A':<30} {'Node B':<30} {'Count':>6}")
    lines.append(f"  {'─' * 30} {'─' * 30} {'─' * 6}")

    for r in rows:
        lines.append(
            f"  {r['node_a']:<30} {r['node_b']:<30} {r['cooccurrence_count']:>6}"
        )

    lines.append("")
    return "\n".join(lines)
