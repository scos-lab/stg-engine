"""STG Telemetry — Real usage data collection for parameter calibration.

Collects propagation patterns, node activation frequencies, edge mutations,
and session summaries. All data stays in memory during active use; flushed
to the .stg SQLite file at session_end.

Design goals:
  - Zero I/O during propagate() — memory buffer only (~55μs overhead)
  - Batch write at session_end (~10ms)
  - Rolling windows prevent unbounded growth (~1.5MB total)
  - Generates calibrated query sets for the simulator
"""

import json
import time
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from stg_engine.engine import STGEngine
    from stg_engine.types import LearningEvent, PropagationMetrics


# ═══════════════════════════════════════════════════════════
# Telemetry Collector (in-memory buffer)
# ═══════════════════════════════════════════════════════════


class TelemetryCollector:
    """In-memory buffer for telemetry data. Zero I/O during propagation.

    Records are accumulated in lists/dicts, then batch-written to SQLite
    via flush(). Rolling window limits are enforced at flush time.
    """

    def __init__(
        self,
        max_propagations: int = 500,
        max_sessions: int = 200,
        max_mutations: int = 2000,
        mutation_threshold: float = 0.01,
    ) -> None:
        self.max_propagations = max_propagations
        self.max_sessions = max_sessions
        self.max_mutations = max_mutations
        self.mutation_threshold = mutation_threshold

        # Buffers
        self._propagations: List[Dict[str, Any]] = []
        self._node_freq: Counter = Counter()
        self._node_seed_freq: Counter = Counter()
        self._node_total_activation: Dict[str, float] = {}
        self._node_first_seen: Dict[str, float] = {}
        self._node_last_seen: Dict[str, float] = {}
        self._session_summary: Optional[Dict[str, Any]] = None
        self._edge_mutations: List[Dict[str, Any]] = []
        # Co-occurrence tracking (Phase 11B: co-activation edge creation)
        self._cooccurrence: Counter = Counter()  # (nodeA, nodeB) sorted → count

    def record_propagation(
        self,
        metrics: "PropagationMetrics",
        activation_map: Dict[str, float],
        strengthen_count: int,
        weaken_count: int,
        strategy: Optional[str] = None,
    ) -> None:
        """Record a single propagation event. Called from engine.propagate().

        Args:
            metrics: PropagationMetrics from the propagation
            activation_map: Full activation map {node_name: activation}
            strengthen_count: Number of edges strengthened
            weaken_count: Number of edges weakened
            strategy: Optional routing strategy used
        """
        now = time.time()

        # Top 5 activated nodes
        top5 = sorted(
            ((n, a) for n, a in activation_map.items() if a > 0),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        top5_json = json.dumps(
            [{n: round(a, 4)} for n, a in top5], ensure_ascii=False
        )

        # Truncate input text
        input_text = (metrics.input_text or "")[:200]

        self._propagations.append({
            "timestamp": now,
            "input_text": input_text,
            "seed_count": metrics.seed_node_count,
            "activated_count": metrics.activated_node_count,
            "qe": round(metrics.query_efficiency, 4),
            "rs": round(metrics.resonance_score, 4),
            "coverage": round(metrics.coverage, 6),
            "strengthen_count": strengthen_count,
            "weaken_count": weaken_count,
            "top5_nodes": top5_json,
        })

        # Update node frequency counters
        for name, act in activation_map.items():
            if act > 0.01:  # Only count meaningfully activated nodes
                self._node_freq[name] += 1
                self._node_total_activation[name] = (
                    self._node_total_activation.get(name, 0.0) + act
                )
                if name not in self._node_first_seen:
                    self._node_first_seen[name] = now
                self._node_last_seen[name] = now

        # Track seed nodes separately
        for name, act in top5[:metrics.seed_node_count]:
            self._node_seed_freq[name] += 1

        # Track co-occurrence pairs (Phase 11B)
        from stg_engine.coactivation import record_cooccurrence
        pairs = record_cooccurrence(activation_map)
        self._cooccurrence.update(pairs)

    def record_edge_mutations(self, events: List["LearningEvent"]) -> None:
        """Record edge salience changes from Hebbian learning.

        Only records mutations exceeding mutation_threshold.
        """
        for ev in events:
            if ev.event_type not in ("strengthen", "weaken"):
                continue
            delta = ev.new_confidence - ev.old_confidence
            if abs(delta) < self.mutation_threshold:
                continue
            self._edge_mutations.append({
                "timestamp": ev.timestamp,
                "source": ev.source,
                "target": ev.target,
                "event_type": ev.event_type,
                "old_salience": round(ev.old_confidence, 4),
                "new_salience": round(ev.new_confidence, 4),
                "delta": round(delta, 4),
            })

    def record_session_summary(self, engine: "STGEngine") -> None:
        """Capture session-level statistics. Called at session_end."""
        now = time.time()
        stats = engine.get_stats()

        # Compute salience distribution from edges
        saliences = sorted(e.salience for e in engine._edges)
        n = len(saliences)
        if n > 0:
            p25 = saliences[n // 4]
            p50 = saliences[n // 2]
            p75 = saliences[3 * n // 4]
            mean = sum(saliences) / n
        else:
            p25 = p50 = p75 = mean = 0.0

        # Count learning events
        total_strengthen = sum(
            1 for p in self._propagations
            if p.get("strengthen_count", 0) > 0
        )
        total_weaken = sum(
            1 for p in self._propagations
            if p.get("weaken_count", 0) > 0
        )
        total_strengthen_count = sum(
            p.get("strengthen_count", 0) for p in self._propagations
        )
        total_weaken_count = sum(
            p.get("weaken_count", 0) for p in self._propagations
        )

        self._session_summary = {
            "timestamp": now,
            "propagation_count": len(self._propagations),
            "total_strengthen": total_strengthen_count,
            "total_weaken": total_weaken_count,
            "salience_p25": round(p25, 4),
            "salience_p50": round(p50, 4),
            "salience_p75": round(p75, 4),
            "salience_mean": round(mean, 4),
            "node_count": stats["node_count"],
            "edge_count": stats["edge_count"],
            "psi": round(stats["psi"], 4),
        }

    def flush(self, stg_path: str) -> int:
        """Batch write all buffered telemetry to the .stg SQLite file.

        Enforces rolling window limits. Returns total records written.

        Args:
            stg_path: Path to the .stg file
        """
        from stg_engine.persistence import _migrate_schema

        path = Path(stg_path)
        if not path.exists():
            return 0

        conn = sqlite3.connect(str(path))
        _migrate_schema(conn)
        written = 0

        try:
            # 1. Propagations (rolling window)
            if self._propagations:
                conn.executemany(
                    "INSERT INTO telemetry_propagations "
                    "(timestamp, input_text, seed_count, activated_count, "
                    "qe, rs, coverage, strengthen_count, weaken_count, top5_nodes) "
                    "VALUES (:timestamp, :input_text, :seed_count, :activated_count, "
                    ":qe, :rs, :coverage, :strengthen_count, :weaken_count, :top5_nodes)",
                    self._propagations,
                )
                written += len(self._propagations)

                # Enforce rolling window
                count = conn.execute(
                    "SELECT COUNT(*) FROM telemetry_propagations"
                ).fetchone()[0]
                if count > self.max_propagations:
                    excess = count - self.max_propagations
                    conn.execute(
                        "DELETE FROM telemetry_propagations WHERE rowid IN "
                        "(SELECT rowid FROM telemetry_propagations "
                        "ORDER BY timestamp ASC LIMIT ?)",
                        (excess,),
                    )

            # 2. Node frequencies (upsert — cumulative)
            if self._node_freq:
                for name, count in self._node_freq.items():
                    seed_count = self._node_seed_freq.get(name, 0)
                    total_act = self._node_total_activation.get(name, 0.0)
                    first = self._node_first_seen.get(name, 0.0)
                    last = self._node_last_seen.get(name, 0.0)

                    conn.execute(
                        "INSERT INTO telemetry_node_freq "
                        "(node_name, activation_count, seed_count, "
                        "total_activation, first_activated, last_activated) "
                        "VALUES (?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(node_name) DO UPDATE SET "
                        "activation_count = activation_count + excluded.activation_count, "
                        "seed_count = seed_count + excluded.seed_count, "
                        "total_activation = total_activation + excluded.total_activation, "
                        "first_activated = MIN(first_activated, excluded.first_activated), "
                        "last_activated = MAX(last_activated, excluded.last_activated)",
                        (name, count, seed_count, total_act, first, last),
                    )
                written += len(self._node_freq)

            # 3. Session summary (rolling window)
            if self._session_summary:
                conn.execute(
                    "INSERT INTO telemetry_sessions "
                    "(timestamp, propagation_count, total_strengthen, total_weaken, "
                    "salience_p25, salience_p50, salience_p75, salience_mean, "
                    "node_count, edge_count, psi) "
                    "VALUES (:timestamp, :propagation_count, :total_strengthen, "
                    ":total_weaken, :salience_p25, :salience_p50, :salience_p75, "
                    ":salience_mean, :node_count, :edge_count, :psi)",
                    self._session_summary,
                )
                written += 1

                # Enforce rolling window
                count = conn.execute(
                    "SELECT COUNT(*) FROM telemetry_sessions"
                ).fetchone()[0]
                if count > self.max_sessions:
                    excess = count - self.max_sessions
                    conn.execute(
                        "DELETE FROM telemetry_sessions WHERE rowid IN "
                        "(SELECT rowid FROM telemetry_sessions "
                        "ORDER BY timestamp ASC LIMIT ?)",
                        (excess,),
                    )

            # 4. Co-occurrence pairs (cumulative upsert, Phase 11B)
            if self._cooccurrence:
                now_ts = time.time()
                for (a, b), count in self._cooccurrence.items():
                    conn.execute(
                        "INSERT INTO telemetry_cooccurrence "
                        "(node_a, node_b, cooccurrence_count, first_seen, last_seen) "
                        "VALUES (?, ?, ?, ?, ?) "
                        "ON CONFLICT(node_a, node_b) DO UPDATE SET "
                        "cooccurrence_count = cooccurrence_count + excluded.cooccurrence_count, "
                        "first_seen = MIN(first_seen, excluded.first_seen), "
                        "last_seen = MAX(last_seen, excluded.last_seen)",
                        (a, b, count, now_ts, now_ts),
                    )
                written += len(self._cooccurrence)

            # 5. Edge mutations (rolling window)
            if self._edge_mutations:
                conn.executemany(
                    "INSERT INTO telemetry_edge_mutations "
                    "(timestamp, source, target, event_type, "
                    "old_salience, new_salience, delta) "
                    "VALUES (:timestamp, :source, :target, :event_type, "
                    ":old_salience, :new_salience, :delta)",
                    self._edge_mutations,
                )
                written += len(self._edge_mutations)

                # Enforce rolling window
                count = conn.execute(
                    "SELECT COUNT(*) FROM telemetry_edge_mutations"
                ).fetchone()[0]
                if count > self.max_mutations:
                    excess = count - self.max_mutations
                    conn.execute(
                        "DELETE FROM telemetry_edge_mutations WHERE rowid IN "
                        "(SELECT rowid FROM telemetry_edge_mutations "
                        "ORDER BY timestamp ASC LIMIT ?)",
                        (excess,),
                    )

            conn.commit()
        finally:
            conn.close()

        return written

    def reset(self) -> None:
        """Clear all in-memory buffers."""
        self._propagations.clear()
        self._node_freq.clear()
        self._node_seed_freq.clear()
        self._node_total_activation.clear()
        self._node_first_seen.clear()
        self._node_last_seen.clear()
        self._session_summary = None
        self._edge_mutations.clear()
        self._cooccurrence.clear()


# ═══════════════════════════════════════════════════════════
# Analysis Functions (read from SQLite)
# ═══════════════════════════════════════════════════════════


def _open_telemetry_db(stg_path: str) -> Optional[sqlite3.Connection]:
    """Open .stg file and verify telemetry tables exist."""
    path = Path(stg_path)
    if not path.exists():
        return None
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    tables = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "telemetry_propagations" not in tables:
        conn.close()
        return None
    return conn


def telemetry_status(stg_path: str) -> Dict[str, Any]:
    """Overview of telemetry data volume and date range."""
    conn = _open_telemetry_db(stg_path)
    if conn is None:
        return {"available": False}

    result: Dict[str, Any] = {"available": True}

    for table in ["telemetry_propagations", "telemetry_node_freq",
                   "telemetry_sessions", "telemetry_edge_mutations"]:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        result[f"{table}_count"] = row[0]

    # Date range from propagations
    row = conn.execute(
        "SELECT MIN(timestamp), MAX(timestamp) FROM telemetry_propagations"
    ).fetchone()
    if row[0]:
        result["first_propagation"] = row[0]
        result["last_propagation"] = row[1]

    # Session count
    row = conn.execute(
        "SELECT COUNT(*) FROM telemetry_sessions"
    ).fetchone()
    result["session_count"] = row[0]

    conn.close()
    return result


def telemetry_frequency(stg_path: str, top_n: int = 20) -> Dict[str, Any]:
    """Node activation frequency ranking.

    Returns:
        Dict with 'nodes' list sorted by activation_count desc,
        and 'total_propagations' count.
    """
    conn = _open_telemetry_db(stg_path)
    if conn is None:
        return {"nodes": [], "total_propagations": 0}

    total = conn.execute(
        "SELECT COUNT(*) FROM telemetry_propagations"
    ).fetchone()[0]

    rows = conn.execute(
        "SELECT node_name, activation_count, seed_count, total_activation, "
        "first_activated, last_activated "
        "FROM telemetry_node_freq ORDER BY activation_count DESC LIMIT ?",
        (top_n,),
    ).fetchall()

    nodes = []
    for r in rows:
        avg_act = r["total_activation"] / r["activation_count"] if r["activation_count"] else 0
        nodes.append({
            "name": r["node_name"],
            "activation_count": r["activation_count"],
            "seed_count": r["seed_count"],
            "avg_activation": round(avg_act, 4),
            "first_activated": r["first_activated"],
            "last_activated": r["last_activated"],
        })

    conn.close()
    return {"nodes": nodes, "total_propagations": total}


def telemetry_salience(stg_path: str) -> Dict[str, Any]:
    """Salience distribution trends across sessions."""
    conn = _open_telemetry_db(stg_path)
    if conn is None:
        return {"sessions": []}

    rows = conn.execute(
        "SELECT timestamp, salience_p25, salience_p50, salience_p75, "
        "salience_mean, node_count, edge_count, psi "
        "FROM telemetry_sessions ORDER BY timestamp ASC"
    ).fetchall()

    sessions = [dict(r) for r in rows]
    conn.close()
    return {"sessions": sessions}


def telemetry_learning(stg_path: str) -> Dict[str, Any]:
    """Learning event analysis — strengthen/weaken ratios and trends."""
    conn = _open_telemetry_db(stg_path)
    if conn is None:
        return {"mutations": [], "summary": {}}

    # Edge mutation stats
    rows = conn.execute(
        "SELECT event_type, COUNT(*) as cnt, "
        "AVG(ABS(delta)) as avg_delta, "
        "MAX(ABS(delta)) as max_delta "
        "FROM telemetry_edge_mutations GROUP BY event_type"
    ).fetchall()

    summary = {}
    for r in rows:
        summary[r["event_type"]] = {
            "count": r["cnt"],
            "avg_delta": round(r["avg_delta"], 4),
            "max_delta": round(r["max_delta"], 4),
        }

    # Most frequently mutated edges
    rows = conn.execute(
        "SELECT source, target, event_type, COUNT(*) as cnt, "
        "AVG(delta) as avg_delta "
        "FROM telemetry_edge_mutations "
        "GROUP BY source, target, event_type "
        "ORDER BY cnt DESC LIMIT 20"
    ).fetchall()

    mutations = [dict(r) for r in rows]
    conn.close()
    return {"mutations": mutations, "summary": summary}


def telemetry_report(stg_path: str) -> str:
    """Generate a comprehensive telemetry report as text."""
    status = telemetry_status(stg_path)
    if not status.get("available"):
        return "No telemetry data available. Enable telemetry and run some propagations first."

    import datetime
    lines = []
    lines.append("=" * 60)
    lines.append("STG Telemetry Report")
    lines.append("=" * 60)

    # Status
    lines.append(f"\nData Volume:")
    lines.append(f"  Propagations: {status.get('telemetry_propagations_count', 0)}")
    lines.append(f"  Tracked nodes: {status.get('telemetry_node_freq_count', 0)}")
    lines.append(f"  Sessions: {status.get('telemetry_sessions_count', 0)}")
    lines.append(f"  Edge mutations: {status.get('telemetry_edge_mutations_count', 0)}")

    if status.get("first_propagation"):
        first = datetime.datetime.fromtimestamp(status["first_propagation"])
        last = datetime.datetime.fromtimestamp(status["last_propagation"])
        lines.append(f"  Date range: {first.strftime('%Y-%m-%d')} → {last.strftime('%Y-%m-%d')}")

    # Frequency
    freq = telemetry_frequency(stg_path, top_n=15)
    if freq["nodes"]:
        lines.append(f"\nTop Activated Nodes (of {freq['total_propagations']} propagations):")
        lines.append(f"  {'Node':<40} {'Count':>6} {'Seeds':>6} {'AvgAct':>8}")
        lines.append(f"  {'─'*40} {'─'*6} {'─'*6} {'─'*8}")
        for n in freq["nodes"]:
            lines.append(
                f"  {n['name']:<40} {n['activation_count']:>6} "
                f"{n['seed_count']:>6} {n['avg_activation']:>8.4f}"
            )

    # Salience trends
    sal = telemetry_salience(stg_path)
    if sal["sessions"]:
        lines.append(f"\nSalience Distribution Trend:")
        lines.append(f"  {'Session':>20} {'P25':>7} {'P50':>7} {'P75':>7} {'Mean':>7} {'Ψ':>7}")
        lines.append(f"  {'─'*20} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
        for s in sal["sessions"][-10:]:  # Last 10 sessions
            ts = datetime.datetime.fromtimestamp(s["timestamp"]).strftime("%Y-%m-%d %H:%M")
            lines.append(
                f"  {ts:>20} {s['salience_p25']:>7.4f} {s['salience_p50']:>7.4f} "
                f"{s['salience_p75']:>7.4f} {s['salience_mean']:>7.4f} {s['psi']:>7.4f}"
            )

    # Learning
    learn = telemetry_learning(stg_path)
    if learn["summary"]:
        lines.append(f"\nLearning Events:")
        for etype, info in learn["summary"].items():
            lines.append(
                f"  {etype}: {info['count']} events, "
                f"avg Δ={info['avg_delta']:.4f}, max Δ={info['max_delta']:.4f}"
            )

    lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# Calibration — Generate queries from real usage data
# ═══════════════════════════════════════════════════════════


def generate_calibrated_queries(stg_path: str) -> List[Dict[str, Any]]:
    """Generate SimQuery-compatible queries from real telemetry data.

    Uses node activation frequency to classify nodes as high/medium/low
    frequency, then builds queries that mirror actual usage patterns.

    Returns:
        List of dicts with keys: text, expected, frequency
        Compatible with simulator.SimQuery constructor.
    """
    conn = _open_telemetry_db(stg_path)
    if conn is None:
        return []

    # Get all tracked nodes with their frequencies
    rows = conn.execute(
        "SELECT node_name, activation_count, seed_count "
        "FROM telemetry_node_freq "
        "WHERE activation_count > 0 "
        "ORDER BY activation_count DESC"
    ).fetchall()

    if not rows:
        conn.close()
        return []

    # Get actual query texts for reference
    query_rows = conn.execute(
        "SELECT input_text, top5_nodes FROM telemetry_propagations "
        "ORDER BY timestamp DESC LIMIT 100"
    ).fetchall()
    conn.close()

    # Classify nodes by frequency percentile
    counts = [r["activation_count"] for r in rows]
    total = sum(counts)
    if total == 0:
        return []

    p75 = counts[len(counts) // 4] if len(counts) > 4 else counts[0]
    p25 = counts[3 * len(counts) // 4] if len(counts) > 4 else counts[-1]

    queries = []
    seen_nodes = set()

    for r in rows:
        name = r["node_name"]
        count = r["activation_count"]
        if name in seen_nodes:
            continue

        # Determine frequency tier
        if count >= p75:
            freq = "high"
        elif count >= p25:
            freq = "medium"
        else:
            freq = "low"

        # Build query text from node name
        # Convert PascalCase/underscore to search terms
        import re
        terms = re.split(r'[_:\-]', name)
        query_text = " ".join(t for t in terms if t)

        if not query_text.strip():
            continue

        queries.append({
            "text": query_text,
            "expected": [name],
            "frequency": freq,
        })
        seen_nodes.add(name)

        # Limit total queries
        if len(queries) >= 30:
            break

    return queries
