"""STG Engine persistence layer.

Serializes/deserializes STGEngine state to .stg files (SQLite format).
Like .safetensors for neural network weights — the .stg file is a
serialization format, not a live database.

The SQLite format has the bonus of being queryable by external tools
(e.g., MCP tools in TypeScript) without loading the full engine.
"""

import json
import logging
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

_save_log = logging.getLogger("stg.audit")

from stg_engine.types import (
    STGNode, STGEdge, STGSession, STGEvent,
    STGTension, STGBeliefEvolution, SystemSnapshot,
)

# Schema version for migration support
SCHEMA_VERSION = 10

_SCHEMA_SQL = """
-- Schema metadata
CREATE TABLE IF NOT EXISTS schema_info (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- Anchor nodes
CREATE TABLE IF NOT EXISTS nodes (
    name TEXT PRIMARY KEY,
    namespace TEXT,
    anchor_type TEXT,
    metadata_json TEXT DEFAULT '{}',
    tension REAL DEFAULT 0.0,
    activation REAL DEFAULT 0.0,
    self_relevance REAL DEFAULT 0.0
);

-- Semantic relations (edges)
CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    strength REAL DEFAULT 0.5,
    rule TEXT,
    time TEXT,
    modifiers_json TEXT DEFAULT '{}',
    session_id TEXT,
    event_id TEXT,
    last_used REAL,
    preference REAL DEFAULT 0.0,
    salience REAL DEFAULT 0.5,
    created_at REAL DEFAULT 0.0,
    edge_class TEXT DEFAULT 'knowledge',
    delay_k INTEGER DEFAULT 0
);

-- Sessions
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    date TEXT,
    title TEXT,
    avg_importance REAL,
    event_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'complete',
    summary TEXT
);

-- Episodic events
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    session_id TEXT,
    timestamp TEXT,
    event_type TEXT,
    memory_type TEXT,
    title TEXT,
    importance_score REAL DEFAULT 0.5,
    description TEXT,
    tags_json TEXT DEFAULT '[]',
    artifacts_json TEXT DEFAULT '[]',
    stl_block TEXT
);

-- Tension tracking
CREATE TABLE IF NOT EXISTS tensions (
    name TEXT PRIMARY KEY,
    initial_value REAL DEFAULT 0.0,
    current_value REAL DEFAULT 0.0,
    status TEXT DEFAULT 'active',
    created_session TEXT,
    resolved_session TEXT,
    description TEXT
);

-- Tension magnitude history
CREATE TABLE IF NOT EXISTS tension_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tension_name TEXT NOT NULL,
    session_id TEXT,
    value REAL,
    context TEXT,
    timestamp TEXT
);

-- Belief evolution chains
CREATE TABLE IF NOT EXISTS belief_evolutions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    old_anchor TEXT NOT NULL,
    new_anchor TEXT NOT NULL,
    event_id TEXT,
    session_id TEXT,
    level INTEGER DEFAULT 1,
    description TEXT
);

-- System state snapshots (for ΔΨ calculation)
CREATE TABLE IF NOT EXISTS system_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    timestamp TEXT,
    psi_value REAL DEFAULT 0.0,
    max_tension REAL DEFAULT 0.0,
    structural_coherence REAL DEFAULT 0.0,
    epistemic_confidence REAL DEFAULT 0.0,
    total_reward REAL DEFAULT 0.0,
    node_count INTEGER DEFAULT 0,
    edge_count INTEGER DEFAULT 0
);

-- Embedding vectors for semantic search (Phase 7G)
CREATE TABLE IF NOT EXISTS embeddings (
    node_name TEXT PRIMARY KEY,
    embed_text TEXT NOT NULL,
    vector BLOB NOT NULL,
    model_name TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Pruning audit log (deleted nodes/edges)
CREATE TABLE IF NOT EXISTS pruned_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pruned_at REAL NOT NULL,
    item_type TEXT NOT NULL,
    source TEXT NOT NULL,
    target TEXT DEFAULT '',
    confidence REAL DEFAULT 0.0,
    salience REAL DEFAULT 0.0,
    last_used REAL,
    modifiers_json TEXT DEFAULT '{}',
    reason TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_pruned_log_time ON pruned_log(pruned_at);
CREATE INDEX IF NOT EXISTS idx_pruned_log_type ON pruned_log(item_type);

-- Telemetry: per-propagation summary (rolling window 500)
CREATE TABLE IF NOT EXISTS telemetry_propagations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    input_text TEXT,
    seed_count INTEGER DEFAULT 0,
    activated_count INTEGER DEFAULT 0,
    qe REAL DEFAULT 0.0,
    rs REAL DEFAULT 0.0,
    coverage REAL DEFAULT 0.0,
    strengthen_count INTEGER DEFAULT 0,
    weaken_count INTEGER DEFAULT 0,
    top5_nodes TEXT DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_telemetry_prop_time ON telemetry_propagations(timestamp);

-- Telemetry: cumulative node activation frequency
CREATE TABLE IF NOT EXISTS telemetry_node_freq (
    node_name TEXT PRIMARY KEY,
    activation_count INTEGER DEFAULT 0,
    seed_count INTEGER DEFAULT 0,
    total_activation REAL DEFAULT 0.0,
    first_activated REAL,
    last_activated REAL
);

-- Telemetry: per-session summary (rolling window 200)
CREATE TABLE IF NOT EXISTS telemetry_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    propagation_count INTEGER DEFAULT 0,
    total_strengthen INTEGER DEFAULT 0,
    total_weaken INTEGER DEFAULT 0,
    salience_p25 REAL DEFAULT 0.0,
    salience_p50 REAL DEFAULT 0.0,
    salience_p75 REAL DEFAULT 0.0,
    salience_mean REAL DEFAULT 0.0,
    node_count INTEGER DEFAULT 0,
    edge_count INTEGER DEFAULT 0,
    psi REAL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_telemetry_sess_time ON telemetry_sessions(timestamp);

-- Telemetry: sampled edge mutations (rolling window 2000)
CREATE TABLE IF NOT EXISTS telemetry_edge_mutations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    event_type TEXT NOT NULL,
    old_salience REAL DEFAULT 0.0,
    new_salience REAL DEFAULT 0.0,
    delta REAL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_telemetry_mut_time ON telemetry_edge_mutations(timestamp);

-- Temporal indexes (Phase 11)
CREATE INDEX IF NOT EXISTS idx_edges_created_at ON edges(created_at);
CREATE INDEX IF NOT EXISTS idx_edges_class ON edges(edge_class);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
CREATE INDEX IF NOT EXISTS idx_edges_session ON edges(session_id);
CREATE INDEX IF NOT EXISTS idx_edges_confidence ON edges(confidence);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_importance ON events(importance_score);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_tensions_status ON tensions(status);
CREATE INDEX IF NOT EXISTS idx_tension_history_name ON tension_history(tension_name);

-- Entity Resolution aliases (G7)
CREATE TABLE IF NOT EXISTS aliases (
    alias TEXT PRIMARY KEY,
    canonical TEXT NOT NULL,
    created_at REAL NOT NULL
);

-- Skill executor audit log (v0.3.1+): every `stg use` invocation records one row.
-- Rolling cap enforced by skill_runner (keep last 10000).
CREATE TABLE IF NOT EXISTS skill_invocations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    invocation_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    skill_name TEXT NOT NULL,
    target TEXT,
    path TEXT,
    interpreter TEXT,
    args_hash TEXT,
    args_preview TEXT,
    exit_code INTEGER NOT NULL,
    elapsed_s REAL NOT NULL,
    bytes_out INTEGER DEFAULT 0,
    bytes_err INTEGER DEFAULT 0,
    truncated INTEGER DEFAULT 0,
    timed_out INTEGER DEFAULT 0,
    error_msg TEXT
);
CREATE INDEX IF NOT EXISTS idx_skill_inv_time ON skill_invocations(timestamp);
CREATE INDEX IF NOT EXISTS idx_skill_inv_name ON skill_invocations(skill_name);
"""


_SKILL_INVOCATIONS_MIGRATION_SQL = """
CREATE TABLE IF NOT EXISTS skill_invocations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    invocation_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    skill_name TEXT NOT NULL,
    target TEXT,
    path TEXT,
    interpreter TEXT,
    args_hash TEXT,
    args_preview TEXT,
    exit_code INTEGER NOT NULL,
    elapsed_s REAL NOT NULL,
    bytes_out INTEGER DEFAULT 0,
    bytes_err INTEGER DEFAULT 0,
    truncated INTEGER DEFAULT 0,
    timed_out INTEGER DEFAULT 0,
    error_msg TEXT
);
CREATE INDEX IF NOT EXISTS idx_skill_inv_time ON skill_invocations(timestamp);
CREATE INDEX IF NOT EXISTS idx_skill_inv_name ON skill_invocations(skill_name);
"""


def _migrate_skill_invocations(conn: sqlite3.Connection) -> None:
    """Idempotent migration: add skill_invocations table if missing.

    Called both on fresh init and on every load of an existing .stg that
    predates v0.3.1. The CREATE TABLE IF NOT EXISTS makes this a no-op when
    already present.
    """
    conn.executescript(_SKILL_INVOCATIONS_MIGRATION_SQL)


def _init_db(conn: sqlite3.Connection) -> None:
    """Initialize database schema."""
    conn.executescript(_SCHEMA_SQL)
    conn.execute(
        "INSERT OR REPLACE INTO schema_info (key, value) VALUES (?, ?)",
        ("version", str(SCHEMA_VERSION)),
    )
    conn.commit()


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Migrate older schema versions to current.

    v2 → v3: Add embeddings table for semantic search (Phase 7G).
    v3 → v4: Add preference column to edges (Phase 8 Kanerva).
    """
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "embeddings" not in tables:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "    node_name TEXT PRIMARY KEY,"
            "    embed_text TEXT NOT NULL,"
            "    vector BLOB NOT NULL,"
            "    model_name TEXT NOT NULL,"
            "    created_at TEXT NOT NULL"
            ")"
        )

    # v3 → v4: Add preference column
    # v4 → v5: Add salience column (confidence/salience split)
    if "edges" in tables:
        edge_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(edges)").fetchall()
        }
        if "preference" not in edge_columns:
            conn.execute("ALTER TABLE edges ADD COLUMN preference REAL DEFAULT 0.0")
        if "salience" not in edge_columns:
            # Migration: inherit confidence as initial salience
            conn.execute("ALTER TABLE edges ADD COLUMN salience REAL DEFAULT 0.5")
            conn.execute("UPDATE edges SET salience = confidence")

    # v5 → v6: Add pruned_log table
    if "pruned_log" not in tables:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS pruned_log ("
            "    id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "    pruned_at REAL NOT NULL,"
            "    item_type TEXT NOT NULL,"
            "    source TEXT NOT NULL,"
            "    target TEXT DEFAULT '',"
            "    confidence REAL DEFAULT 0.0,"
            "    salience REAL DEFAULT 0.0,"
            "    last_used REAL,"
            "    modifiers_json TEXT DEFAULT '{}',"
            "    reason TEXT DEFAULT ''"
            ")"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pruned_log_time ON pruned_log(pruned_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pruned_log_type ON pruned_log(item_type)"
        )

    # v6 → v7: Add telemetry tables
    for tbl_name, tbl_sql in [
        ("telemetry_propagations",
         "CREATE TABLE IF NOT EXISTS telemetry_propagations ("
         "    id INTEGER PRIMARY KEY AUTOINCREMENT,"
         "    timestamp REAL NOT NULL,"
         "    input_text TEXT,"
         "    seed_count INTEGER DEFAULT 0,"
         "    activated_count INTEGER DEFAULT 0,"
         "    qe REAL DEFAULT 0.0,"
         "    rs REAL DEFAULT 0.0,"
         "    coverage REAL DEFAULT 0.0,"
         "    strengthen_count INTEGER DEFAULT 0,"
         "    weaken_count INTEGER DEFAULT 0,"
         "    top5_nodes TEXT DEFAULT '[]'"
         ")"),
        ("telemetry_node_freq",
         "CREATE TABLE IF NOT EXISTS telemetry_node_freq ("
         "    node_name TEXT PRIMARY KEY,"
         "    activation_count INTEGER DEFAULT 0,"
         "    seed_count INTEGER DEFAULT 0,"
         "    total_activation REAL DEFAULT 0.0,"
         "    first_activated REAL,"
         "    last_activated REAL"
         ")"),
        ("telemetry_sessions",
         "CREATE TABLE IF NOT EXISTS telemetry_sessions ("
         "    id INTEGER PRIMARY KEY AUTOINCREMENT,"
         "    timestamp REAL NOT NULL,"
         "    propagation_count INTEGER DEFAULT 0,"
         "    total_strengthen INTEGER DEFAULT 0,"
         "    total_weaken INTEGER DEFAULT 0,"
         "    salience_p25 REAL DEFAULT 0.0,"
         "    salience_p50 REAL DEFAULT 0.0,"
         "    salience_p75 REAL DEFAULT 0.0,"
         "    salience_mean REAL DEFAULT 0.0,"
         "    node_count INTEGER DEFAULT 0,"
         "    edge_count INTEGER DEFAULT 0,"
         "    psi REAL DEFAULT 0.0"
         ")"),
        ("telemetry_edge_mutations",
         "CREATE TABLE IF NOT EXISTS telemetry_edge_mutations ("
         "    id INTEGER PRIMARY KEY AUTOINCREMENT,"
         "    timestamp REAL NOT NULL,"
         "    source TEXT NOT NULL,"
         "    target TEXT NOT NULL,"
         "    event_type TEXT NOT NULL,"
         "    old_salience REAL DEFAULT 0.0,"
         "    new_salience REAL DEFAULT 0.0,"
         "    delta REAL DEFAULT 0.0"
         ")"),
    ]:
        if tbl_name not in tables:
            conn.execute(tbl_sql)

    # v7 → v8: Add co-occurrence telemetry table (Phase 11B)
    if "telemetry_cooccurrence" not in tables:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS telemetry_cooccurrence ("
            "    node_a TEXT NOT NULL,"
            "    node_b TEXT NOT NULL,"
            "    cooccurrence_count INTEGER NOT NULL DEFAULT 0,"
            "    first_seen REAL,"
            "    last_seen REAL,"
            "    PRIMARY KEY (node_a, node_b)"
            ")"
        )

    # v8 → v9: Add temporal columns to edges (Phase 11 Temporal)
    if "edges" in tables:
        edge_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(edges)").fetchall()
        }
        if "created_at" not in edge_columns:
            conn.execute("ALTER TABLE edges ADD COLUMN created_at REAL DEFAULT 0.0")
        if "edge_class" not in edge_columns:
            conn.execute("ALTER TABLE edges ADD COLUMN edge_class TEXT DEFAULT 'knowledge'")
        if "delay_k" not in edge_columns:
            conn.execute("ALTER TABLE edges ADD COLUMN delay_k INTEGER DEFAULT 0")
        # Temporal indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_created_at ON edges(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_class ON edges(edge_class)")

    # v9 → v10: Add perception tables (Phase 12)
    if "perception_frames" not in tables:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS perception_frames ("
            "    frame_hash TEXT PRIMARY KEY,"
            "    feature_vector BLOB NOT NULL,"
            "    grid_blob BLOB,"
            "    grid_width INTEGER NOT NULL,"
            "    grid_height INTEGER NOT NULL,"
            "    n_colors INTEGER DEFAULT 0,"
            "    color_histogram TEXT DEFAULT '{}',"
            "    game_id TEXT,"
            "    step_number INTEGER DEFAULT 0,"
            "    level INTEGER DEFAULT 0,"
            "    node_name TEXT,"
            "    created_at REAL NOT NULL"
            ")"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_perc_game ON perception_frames(game_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_perc_node ON perception_frames(node_name)")

    if "perception_filters" not in tables:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS perception_filters ("
            "    filter_id INTEGER PRIMARY KEY,"
            "    weights BLOB NOT NULL,"
            "    update_count INTEGER DEFAULT 0,"
            "    last_updated REAL"
            ")"
        )

    # v10 → v11: Add aliases table (G7 Entity Resolution)
    if "aliases" not in tables:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS aliases ("
            "    alias TEXT PRIMARY KEY,"
            "    canonical TEXT NOT NULL,"
            "    created_at REAL NOT NULL"
            ")"
        )

    # Telemetry indexes
    for idx_sql in [
        "CREATE INDEX IF NOT EXISTS idx_telemetry_prop_time ON telemetry_propagations(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_telemetry_sess_time ON telemetry_sessions(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_telemetry_mut_time ON telemetry_edge_mutations(timestamp)",
    ]:
        conn.execute(idx_sql)

    conn.execute(
        "INSERT OR REPLACE INTO schema_info (key, value) VALUES (?, ?)",
        ("version", str(SCHEMA_VERSION)),
    )
    conn.commit()


def save_engine_state(
    path: str,
    nodes: Dict[str, STGNode],
    edges: List[STGEdge],
    sessions: Dict[str, STGSession],
    events: Dict[str, STGEvent],
    tensions: Dict[str, STGTension],
    belief_evolutions: List[STGBeliefEvolution],
    snapshots: List[SystemSnapshot],
    force_save: bool = False,
    aliases: Optional[Dict[str, str]] = None,
) -> None:
    """Serialize full engine state to .stg file.

    Overwrites the file atomically (write to temp, then rename).
    Safety: refuses to save if node count drops >50% (prevents accidental wipe).

    Args:
        path: Path to .stg file
        nodes: All nodes in the engine
        edges: All edges
        sessions: Session records
        events: Episodic events
        tensions: Tension records
        belief_evolutions: Belief evolution chains
        snapshots: System state snapshots
        force_save: If True, skip the node count safety check
    """
    stg_path = Path(path)

    # --- Safety check: prevent catastrophic data loss ---
    new_node_count = len(nodes)
    new_edge_count = len(edges)
    old_node_count = 0
    old_edge_count = 0
    if stg_path.exists():
        old_conn = sqlite3.connect(str(stg_path))
        try:
            old_node_count = old_conn.execute("SELECT count(*) FROM nodes").fetchone()[0]
            old_edge_count = old_conn.execute("SELECT count(*) FROM edges").fetchone()[0]
        except sqlite3.OperationalError:
            pass  # File exists but has no tables (e.g., empty or non-STG file)
        finally:
            old_conn.close()
        if not force_save and old_node_count > 100 and new_node_count < old_node_count * 0.5:
            _save_log.warning(
                f"SAVE BLOCKED | nodes: {old_node_count}→{new_node_count} | "
                f"edges: {old_edge_count}→{new_edge_count} | path={path}"
            )
            raise RuntimeError(
                f"SAVE ABORTED: node count would drop from {old_node_count} to {new_node_count} "
                f"({new_node_count / old_node_count * 100:.1f}%). "
                f"This looks like accidental data loss. "
                f"Use force_save=True or fix the engine state before saving."
            )

    # Write to temp file for atomic save
    tmp_path = stg_path.with_suffix(".stg.tmp")

    # Pre-cleanup: remove stale tmp/WAL/SHM from previous crashed saves.
    # A leftover tmp with committed data causes UNIQUE constraint errors
    # because _init_db's CREATE IF NOT EXISTS becomes a no-op on existing tables.
    for suffix in ("", "-wal", "-shm"):
        stale = Path(str(tmp_path) + suffix)
        if stale.exists():
            stale.unlink()

    conn = None
    try:
        conn = sqlite3.connect(str(tmp_path))
        # DELETE journal mode (not WAL) — avoids orphaned WAL/SHM files that
        # can't be renamed atomically with the main database file.
        conn.execute("PRAGMA journal_mode=DELETE")
        _init_db(conn)

        # --- Nodes ---
        conn.executemany(
            "INSERT INTO nodes (name, namespace, anchor_type, metadata_json, tension, activation, self_relevance) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (n.name, n.namespace, n.anchor_type,
                 json.dumps(n.metadata, ensure_ascii=False),
                 n.tension, n.activation, n.self_relevance)
                for n in nodes.values()
            ],
        )

        # --- Edges ---
        conn.executemany(
            "INSERT INTO edges (source, target, confidence, strength, rule, time, "
            "modifiers_json, session_id, event_id, last_used, preference, salience, "
            "created_at, edge_class, delay_k) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (e.source, e.target, e.confidence, e.strength, e.rule, e.time,
                 json.dumps(
                     {k: v for k, v in e.modifiers.items() if not k.startswith("_")},
                     ensure_ascii=False,
                 ),
                 e.session_id, e.event_id, e.last_used, e.preference, e.salience,
                 e.created_at, e.edge_class, e.delay_k)
                for e in edges
            ],
        )

        # --- Sessions ---
        conn.executemany(
            "INSERT INTO sessions (session_id, date, title, avg_importance, event_count, status, summary) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (s.session_id, s.date, s.title, s.avg_importance,
                 s.event_count, s.status, s.summary)
                for s in sessions.values()
            ],
        )

        # --- Events ---
        conn.executemany(
            "INSERT INTO events (event_id, session_id, timestamp, event_type, memory_type, title, "
            "importance_score, description, tags_json, artifacts_json, stl_block) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (ev.event_id, ev.session_id, ev.timestamp, ev.event_type,
                 ev.memory_type, ev.title, ev.importance_score, ev.description,
                 json.dumps(ev.tags, ensure_ascii=False),
                 json.dumps(ev.artifacts, ensure_ascii=False),
                 ev.stl_block)
                for ev in events.values()
            ],
        )

        # --- Tensions ---
        conn.executemany(
            "INSERT INTO tensions (name, initial_value, current_value, status, "
            "created_session, resolved_session, description) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (t.name, t.initial_value, t.current_value, t.status,
                 t.created_session, t.resolved_session, t.description)
                for t in tensions.values()
            ],
        )

        # --- Belief Evolutions ---
        conn.executemany(
            "INSERT INTO belief_evolutions (old_anchor, new_anchor, event_id, session_id, level, description) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (be.old_anchor, be.new_anchor, be.event_id, be.session_id,
                 be.level, be.description)
                for be in belief_evolutions
            ],
        )

        # --- Snapshots ---
        conn.executemany(
            "INSERT INTO system_snapshots (session_id, timestamp, psi_value, max_tension, "
            "structural_coherence, epistemic_confidence, total_reward, node_count, edge_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (ss.session_id, ss.timestamp, ss.psi_value, ss.max_tension,
                 ss.structural_coherence, ss.epistemic_confidence,
                 ss.total_reward, ss.node_count, ss.edge_count)
                for ss in snapshots
            ],
        )

        # --- Aliases (G7 Entity Resolution) ---
        if aliases:
            import time as _time
            now = _time.time()
            conn.executemany(
                "INSERT INTO aliases (alias, canonical, created_at) VALUES (?, ?, ?)",
                [(a, c, now) for a, c in aliases.items()],
            )

        # --- Preserve append-only tables from old file ---
        if stg_path.exists():
            old_conn = sqlite3.connect(str(stg_path))
            old_tables = {
                row[0] for row in old_conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }

            # Pruned log
            if "pruned_log" in old_tables:
                old_rows = old_conn.execute(
                    "SELECT pruned_at, item_type, source, target, confidence, "
                    "salience, last_used, modifiers_json, reason FROM pruned_log"
                ).fetchall()
                if old_rows:
                    conn.executemany(
                        "INSERT INTO pruned_log (pruned_at, item_type, source, target, "
                        "confidence, salience, last_used, modifiers_json, reason) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        old_rows,
                    )

            # Telemetry tables (all append-only)
            _telemetry_tables = [
                ("telemetry_propagations",
                 "timestamp, input_text, seed_count, activated_count, "
                 "qe, rs, coverage, strengthen_count, weaken_count, top5_nodes"),
                ("telemetry_node_freq",
                 "node_name, activation_count, seed_count, "
                 "total_activation, first_activated, last_activated"),
                ("telemetry_sessions",
                 "timestamp, propagation_count, total_strengthen, total_weaken, "
                 "salience_p25, salience_p50, salience_p75, salience_mean, "
                 "node_count, edge_count, psi"),
                ("telemetry_edge_mutations",
                 "timestamp, source, target, event_type, "
                 "old_salience, new_salience, delta"),
            ]
            for tbl_name, cols in _telemetry_tables:
                if tbl_name in old_tables:
                    old_rows = old_conn.execute(
                        f"SELECT {cols} FROM {tbl_name}"
                    ).fetchall()
                    if old_rows:
                        placeholders = ", ".join("?" * len(old_rows[0]))
                        conn.executemany(
                            f"INSERT INTO {tbl_name} ({cols}) VALUES ({placeholders})",
                            old_rows,
                        )

            old_conn.close()

        conn.commit()
        conn.close()
        conn = None

        # Atomic rename
        if stg_path.exists():
            stg_path.unlink()
        tmp_path.rename(stg_path)

        # Post-rename cleanup: remove any WAL/SHM left by prior operations
        # on the old stg_path (e.g. from telemetry.flush() or safety check).
        for suffix in ("-wal", "-shm"):
            leftover = Path(str(stg_path) + suffix)
            if leftover.exists():
                leftover.unlink()

        _save_log.info(
            f"SAVE OK | nodes: {old_node_count}→{new_node_count} ({new_node_count - old_node_count:+d}) | "
            f"edges: {old_edge_count if old_node_count else 0}→{new_edge_count} | path={path}"
        )

    except Exception as exc:
        _save_log.error(f"SAVE FAILED | nodes_attempted={new_node_count} | error={exc} | path={path}")
        # Close connection before cleanup to release file handles
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        # Clean up temp file and any associated WAL/SHM
        for suffix in ("", "-wal", "-shm"):
            f = Path(str(tmp_path) + suffix)
            if f.exists():
                f.unlink()
        raise


def backup_database(path: str, backup_dir: Optional[str] = None, keep: int = 7) -> str:
    """Create a timestamped backup of the .stg database.

    Args:
        path: Path to the .stg file to back up
        backup_dir: Directory to store backups (default: <stg_dir>/backups/)
        keep: Number of recent backups to retain (default: 7, oldest auto-deleted)

    Returns:
        Path to the created backup file

    Raises:
        FileNotFoundError: If the source .stg file doesn't exist
    """
    stg_path = Path(path)
    if not stg_path.exists():
        raise FileNotFoundError(f"STG file not found: {path}")

    # Determine backup directory
    if backup_dir:
        bak_dir = Path(backup_dir)
    else:
        bak_dir = stg_path.parent / "backups"
    bak_dir.mkdir(parents=True, exist_ok=True)

    # Verify source has data (don't back up empty databases)
    conn = sqlite3.connect(str(stg_path))
    node_count = conn.execute("SELECT count(*) FROM nodes").fetchone()[0]
    edge_count = conn.execute("SELECT count(*) FROM edges").fetchone()[0]
    conn.close()

    if node_count == 0:
        raise RuntimeError(
            f"BACKUP ABORTED: source has 0 nodes. "
            f"Refusing to create a backup of an empty database."
        )

    # Create timestamped backup
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_name = f"memory-{timestamp}.stg"
    backup_path = bak_dir / backup_name
    shutil.copy2(str(stg_path), str(backup_path))

    # Rotate: keep only the most recent N backups
    existing = sorted(bak_dir.glob("memory-*.stg"), key=lambda p: p.name)
    while len(existing) > keep:
        oldest = existing.pop(0)
        oldest.unlink()

    return str(backup_path)


def load_engine_state(path: str) -> Dict[str, Any]:
    """Deserialize engine state from .stg file.

    Args:
        path: Path to .stg file

    Returns:
        Dictionary with keys: nodes, edges, sessions, events,
        tensions, belief_evolutions, snapshots

    Raises:
        FileNotFoundError: If .stg file doesn't exist
    """
    stg_path = Path(path)
    if not stg_path.exists():
        raise FileNotFoundError(f"STG file not found: {path}")

    conn = sqlite3.connect(str(stg_path))
    conn.row_factory = sqlite3.Row

    # Apply idempotent migrations (v0.3.1+: skill_invocations)
    _migrate_skill_invocations(conn)
    conn.commit()

    result: Dict[str, Any] = {}

    # --- Nodes ---
    rows = conn.execute("SELECT * FROM nodes").fetchall()
    nodes = {}
    for row in rows:
        node = STGNode(
            name=row["name"],
            namespace=row["namespace"],
            anchor_type=row["anchor_type"],
            metadata=json.loads(row["metadata_json"] or "{}"),
            tension=row["tension"] or 0.0,
            activation=row["activation"] or 0.0,
            self_relevance=row["self_relevance"] or 0.0,
        )
        nodes[node.name] = node
    result["nodes"] = nodes

    # --- Edges ---
    rows = conn.execute("SELECT * FROM edges").fetchall()
    # Check column existence (backward compat with older schemas)
    edge_col_names = {
        row[1] for row in conn.execute("PRAGMA table_info(edges)").fetchall()
    }
    has_last_used = "last_used" in edge_col_names
    has_preference = "preference" in edge_col_names
    has_salience = "salience" in edge_col_names
    has_created_at = "created_at" in edge_col_names
    has_edge_class = "edge_class" in edge_col_names
    has_delay_k = "delay_k" in edge_col_names
    edges = []
    for row in rows:
        last_used = None
        if has_last_used:
            try:
                last_used = row["last_used"]
            except (IndexError, KeyError):
                last_used = None
        preference = 0.0
        if has_preference:
            try:
                preference = row["preference"] or 0.0
            except (IndexError, KeyError):
                preference = 0.0
        # Salience: if column exists, read it; otherwise inherit from confidence
        conf_val = row["confidence"] or 0.5
        salience = conf_val  # default: inherit confidence for old schemas
        if has_salience:
            try:
                salience = row["salience"] if row["salience"] is not None else conf_val
            except (IndexError, KeyError):
                salience = conf_val
        # Temporal fields (Phase 11)
        created_at = 0.0
        if has_created_at:
            try:
                created_at = row["created_at"] or 0.0
            except (IndexError, KeyError):
                created_at = 0.0
        edge_class = "knowledge"
        if has_edge_class:
            try:
                edge_class = row["edge_class"] or "knowledge"
            except (IndexError, KeyError):
                edge_class = "knowledge"
        delay_k = 0
        if has_delay_k:
            try:
                delay_k = row["delay_k"] or 0
            except (IndexError, KeyError):
                delay_k = 0
        edge = STGEdge(
            source=row["source"],
            target=row["target"],
            confidence=conf_val,
            strength=row["strength"] or 0.5,
            rule=row["rule"],
            time=row["time"],
            modifiers=json.loads(row["modifiers_json"] or "{}"),
            session_id=row["session_id"],
            event_id=row["event_id"],
            last_used=last_used,
            preference=preference,
            salience=salience,
            created_at=created_at,
            edge_class=edge_class,
            delay_k=delay_k,
        )
        edges.append(edge)
    result["edges"] = edges

    # --- Sessions ---
    rows = conn.execute("SELECT * FROM sessions").fetchall()
    sessions = {}
    for row in rows:
        s = STGSession(
            session_id=row["session_id"],
            date=row["date"],
            title=row["title"],
            avg_importance=row["avg_importance"],
            event_count=row["event_count"] or 0,
            status=row["status"] or "complete",
            summary=row["summary"],
        )
        sessions[s.session_id] = s
    result["sessions"] = sessions

    # --- Events ---
    rows = conn.execute("SELECT * FROM events").fetchall()
    events = {}
    for row in rows:
        ev = STGEvent(
            event_id=row["event_id"],
            session_id=row["session_id"],
            timestamp=row["timestamp"],
            event_type=row["event_type"],
            memory_type=row["memory_type"],
            title=row["title"],
            importance_score=row["importance_score"] or 0.5,
            description=row["description"],
            tags=json.loads(row["tags_json"] or "[]"),
            artifacts=json.loads(row["artifacts_json"] or "[]"),
            stl_block=row["stl_block"],
        )
        events[ev.event_id] = ev
    result["events"] = events

    # --- Tensions ---
    rows = conn.execute("SELECT * FROM tensions").fetchall()
    tensions = {}
    for row in rows:
        t = STGTension(
            name=row["name"],
            initial_value=row["initial_value"] or 0.0,
            current_value=row["current_value"] or 0.0,
            status=row["status"] or "active",
            created_session=row["created_session"],
            resolved_session=row["resolved_session"],
            description=row["description"],
        )
        tensions[t.name] = t
    result["tensions"] = tensions

    # --- Belief Evolutions ---
    rows = conn.execute("SELECT * FROM belief_evolutions").fetchall()
    belief_evolutions = []
    for row in rows:
        be = STGBeliefEvolution(
            old_anchor=row["old_anchor"],
            new_anchor=row["new_anchor"],
            event_id=row["event_id"],
            session_id=row["session_id"],
            level=row["level"] or 1,
            description=row["description"],
        )
        belief_evolutions.append(be)
    result["belief_evolutions"] = belief_evolutions

    # --- Snapshots ---
    rows = conn.execute("SELECT * FROM system_snapshots ORDER BY id").fetchall()
    snapshots = []
    for row in rows:
        ss = SystemSnapshot(
            session_id=row["session_id"],
            timestamp=row["timestamp"],
            psi_value=row["psi_value"] or 0.0,
            max_tension=row["max_tension"] or 0.0,
            structural_coherence=row["structural_coherence"] or 0.0,
            epistemic_confidence=row["epistemic_confidence"] or 0.0,
            total_reward=row["total_reward"] or 0.0,
            node_count=row["node_count"] or 0,
            edge_count=row["edge_count"] or 0,
        )
        snapshots.append(ss)
    result["snapshots"] = snapshots

    # --- Aliases (G7 Entity Resolution) ---
    tables = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    aliases = {}
    if "aliases" in tables:
        rows = conn.execute("SELECT alias, canonical FROM aliases").fetchall()
        for row in rows:
            aliases[row["alias"]] = row["canonical"]
    result["aliases"] = aliases

    # --- Migration for older schema ---
    _migrate_schema(conn)

    conn.close()
    return result


# ═══════════════════════════════════════════════════════════
# Phase 7G: Embedding Persistence
# ═══════════════════════════════════════════════════════════


def save_embeddings(
    path: str,
    names: List[str],
    vectors: "numpy.ndarray",
    embed_texts: Dict[str, str],
    model_name: str,
) -> None:
    """Persist embedding vectors to .stg file.

    Appends to existing .stg file (does not overwrite other tables).

    Args:
        path: Path to .stg file
        names: Node names aligned with vector rows
        vectors: (N, dim) float32 numpy array
        embed_texts: Dict mapping node_name → embed text
        model_name: Model name for compatibility checking
    """
    import numpy as np
    from datetime import datetime

    conn = sqlite3.connect(str(path))
    _migrate_schema(conn)

    now = datetime.utcnow().isoformat()

    # Clear existing embeddings
    conn.execute("DELETE FROM embeddings")

    # Insert all embeddings
    rows = []
    for i, name in enumerate(names):
        vector_blob = vectors[i].astype(np.float32).tobytes()
        embed_text = embed_texts.get(name, name)
        rows.append((name, embed_text, vector_blob, model_name, now))

    conn.executemany(
        "INSERT INTO embeddings (node_name, embed_text, vector, model_name, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def load_embeddings(
    path: str,
    expected_model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Load embedding vectors from .stg file.

    Args:
        path: Path to .stg file
        expected_model: If set, returns None when stored model differs

    Returns:
        Dict with keys: names, vectors, embed_texts, model_name
        Returns None if no embeddings found or model mismatch.
    """
    import numpy as np

    stg_path = Path(path)
    if not stg_path.exists():
        return None

    conn = sqlite3.connect(str(stg_path))
    conn.row_factory = sqlite3.Row

    # Check if embeddings table exists
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "embeddings" not in tables:
        conn.close()
        return None

    rows = conn.execute("SELECT * FROM embeddings").fetchall()
    conn.close()

    if not rows:
        return None

    # Check model compatibility
    stored_model = rows[0]["model_name"]
    if expected_model and stored_model != expected_model:
        return None

    names = []
    vectors = []
    embed_texts = {}

    for row in rows:
        name = row["node_name"]
        names.append(name)
        embed_texts[name] = row["embed_text"]
        vec = np.frombuffer(row["vector"], dtype=np.float32)
        vectors.append(vec)

    return {
        "names": names,
        "vectors": np.vstack(vectors),
        "embed_texts": embed_texts,
        "model_name": stored_model,
    }


# ═══════════════════════════════════════════════════════════
# Pruning Audit Log
# ═══════════════════════════════════════════════════════════


def append_pruned_log(
    path: str,
    entries: List[Dict[str, Any]],
) -> int:
    """Append pruning records to the .stg file's pruned_log table.

    Called directly on the live .stg file (not through save_engine_state)
    so records persist even if save isn't called afterwards.

    Args:
        path: Path to .stg file
        entries: List of dicts with keys: pruned_at, item_type, source,
                 target, confidence, salience, last_used, modifiers_json, reason

    Returns:
        Number of records written.
    """
    stg_path = Path(path)
    if not stg_path.exists():
        return 0

    conn = sqlite3.connect(str(stg_path))
    _migrate_schema(conn)

    conn.executemany(
        "INSERT INTO pruned_log (pruned_at, item_type, source, target, "
        "confidence, salience, last_used, modifiers_json, reason) "
        "VALUES (:pruned_at, :item_type, :source, :target, "
        ":confidence, :salience, :last_used, :modifiers_json, :reason)",
        entries,
    )
    conn.commit()
    conn.close()
    return len(entries)


def read_pruned_log(
    path: str,
    limit: int = 50,
    item_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Read pruning audit log from .stg file.

    Args:
        path: Path to .stg file
        limit: Max records to return (most recent first)
        item_type: Filter by type ("edge", "virtual_edge", "orphan_node")

    Returns:
        List of pruning log records, most recent first.
    """
    stg_path = Path(path)
    if not stg_path.exists():
        return []

    conn = sqlite3.connect(str(stg_path))
    conn.row_factory = sqlite3.Row

    tables = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "pruned_log" not in tables:
        conn.close()
        return []

    if item_type:
        rows = conn.execute(
            "SELECT * FROM pruned_log WHERE item_type = ? "
            "ORDER BY pruned_at DESC LIMIT ?",
            (item_type, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM pruned_log ORDER BY pruned_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

    conn.close()
    return [dict(row) for row in rows]


# ═══════════════════════════════════════════════════════════
# Perception Persistence (Phase 12)
# ═══════════════════════════════════════════════════════════


def save_perception_frame(
    path: str,
    frame_hash: str,
    feature_vector: "np.ndarray",
    grid_width: int,
    grid_height: int,
    n_colors: int = 0,
    color_histogram: str = "{}",
    game_id: Optional[str] = None,
    step_number: int = 0,
    level: int = 0,
    node_name: Optional[str] = None,
    grid_blob: Optional[bytes] = None,
) -> None:
    """Save a single perception frame to the .stg database.

    Uses INSERT OR REPLACE to handle duplicates by frame_hash.
    """
    import time as _time
    import numpy as np

    stg_path = Path(path)
    if not stg_path.exists():
        return

    conn = sqlite3.connect(str(stg_path))
    _migrate_schema(conn)

    vec_bytes = feature_vector.astype(np.float32).tobytes()
    conn.execute(
        "INSERT OR REPLACE INTO perception_frames "
        "(frame_hash, feature_vector, grid_blob, grid_width, grid_height, "
        "n_colors, color_histogram, game_id, step_number, level, node_name, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            frame_hash, vec_bytes, grid_blob, grid_width, grid_height,
            n_colors, color_histogram, game_id or "", step_number, level,
            node_name or "", _time.time(),
        ),
    )
    conn.commit()
    conn.close()


def load_perception_frames(
    path: str,
    game_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load perception frames from .stg database.

    Args:
        path: Path to .stg file
        game_id: Optional filter by game_id

    Returns:
        List of dicts with frame_hash, feature_vector (np.ndarray), metadata
    """
    import numpy as np

    stg_path = Path(path)
    if not stg_path.exists():
        return []

    conn = sqlite3.connect(str(stg_path))
    conn.row_factory = sqlite3.Row

    tables = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "perception_frames" not in tables:
        conn.close()
        return []

    if game_id:
        rows = conn.execute(
            "SELECT * FROM perception_frames WHERE game_id = ? ORDER BY step_number",
            (game_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM perception_frames ORDER BY created_at"
        ).fetchall()

    conn.close()

    frames = []
    for row in rows:
        vec = np.frombuffer(row["feature_vector"], dtype=np.float32).copy()
        frames.append({
            "frame_hash": row["frame_hash"],
            "feature_vector": vec,
            "grid_width": row["grid_width"],
            "grid_height": row["grid_height"],
            "n_colors": row["n_colors"],
            "color_histogram": row["color_histogram"],
            "game_id": row["game_id"],
            "step_number": row["step_number"],
            "level": row["level"],
            "node_name": row["node_name"],
            "created_at": row["created_at"],
        })
    return frames


def save_perception_filters(
    path: str,
    filters: "np.ndarray",
) -> None:
    """Save learnable filter weights to .stg database."""
    import time as _time
    import numpy as np

    stg_path = Path(path)
    if not stg_path.exists():
        return

    conn = sqlite3.connect(str(stg_path))
    _migrate_schema(conn)

    for i in range(filters.shape[0]):
        weight_bytes = filters[i].astype(np.float32).tobytes()
        conn.execute(
            "INSERT OR REPLACE INTO perception_filters "
            "(filter_id, weights, update_count, last_updated) "
            "VALUES (?, ?, COALESCE("
            "  (SELECT update_count FROM perception_filters WHERE filter_id = ?), 0"
            ") + 1, ?)",
            (i, weight_bytes, i, _time.time()),
        )
    conn.commit()
    conn.close()


def load_perception_filters(path: str) -> Optional["np.ndarray"]:
    """Load learnable filter weights from .stg database.

    Returns None if no filters stored yet.
    """
    import numpy as np

    stg_path = Path(path)
    if not stg_path.exists():
        return None

    conn = sqlite3.connect(str(stg_path))
    conn.row_factory = sqlite3.Row

    tables = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "perception_filters" not in tables:
        conn.close()
        return None

    rows = conn.execute(
        "SELECT * FROM perception_filters ORDER BY filter_id"
    ).fetchall()
    conn.close()

    if not rows:
        return None

    filters = []
    for row in rows:
        f = np.frombuffer(row["weights"], dtype=np.float32).copy()
        size = int(np.sqrt(f.size))
        filters.append(f.reshape(size, size))

    return np.stack(filters)
