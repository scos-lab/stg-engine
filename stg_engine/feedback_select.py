"""STG User Feedback Selection — reward/penalize propagate results.

After propagate, user selects which results are useful.
Selected nodes' incoming edges get salience boost.
Unselected nodes' incoming edges get salience penalty.
Selected nodes become the active_context for subsequent ingest.

This is the missing feedback signal: STG learns what's useful
from user choices, not from activation frequency.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from stg_engine.engine import STGEngine


@dataclass
class SelectionResult:
    """Result of a user selection operation."""
    selected_nodes: List[str]
    rejected_nodes: List[str]
    edges_rewarded: int
    edges_penalized: int


def apply_selection(
    engine: "STGEngine",
    all_results: List[str],
    selected_indices: List[int],
    reward_delta: float = 0.05,
    penalty_delta: float = 0.02,
) -> SelectionResult:
    """Apply user selection feedback to the graph.

    Selected nodes: incoming edges get salience += reward_delta
    Unselected nodes: incoming edges get salience -= penalty_delta

    Args:
        engine: STGEngine instance
        all_results: full list of propagate result node names
        selected_indices: 1-based indices of user-selected nodes
        reward_delta: salience boost for selected nodes' edges
        penalty_delta: salience penalty for unselected nodes' edges

    Returns:
        SelectionResult with counts
    """
    # Convert 1-based to 0-based
    selected_set = set(i - 1 for i in selected_indices if 1 <= i <= len(all_results))

    selected_nodes = [all_results[i] for i in range(len(all_results)) if i in selected_set]
    rejected_nodes = [all_results[i] for i in range(len(all_results)) if i not in selected_set]

    edges_rewarded = 0
    edges_penalized = 0

    # Reward: boost salience on edges leading TO selected nodes
    for node_name in selected_nodes:
        for edge in engine._edges:
            if edge.target == node_name:
                edge.salience = min(1.0, edge.salience + reward_delta)
                edges_rewarded += 1

    # Penalize: reduce salience on edges leading TO unselected nodes
    for node_name in rejected_nodes:
        for edge in engine._edges:
            if edge.target == node_name:
                edge.salience = max(0.01, edge.salience - penalty_delta)
                edges_penalized += 1

    return SelectionResult(
        selected_nodes=selected_nodes,
        rejected_nodes=rejected_nodes,
        edges_rewarded=edges_rewarded,
        edges_penalized=edges_penalized,
    )


def save_active_context(
    engine: "STGEngine",
    selected_nodes: List[str],
    stg_path: str,
) -> None:
    """Persist selected nodes as active_context to .stg SQLite.

    Active context is used by subsequent ingest to auto-link new nodes.
    """
    import sqlite3

    with sqlite3.connect(stg_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS active_context (
                session_id TEXT DEFAULT 'default',
                node_name TEXT,
                activation REAL,
                updated_at REAL
            )
        """)
        # Clear previous context for this session
        conn.execute("DELETE FROM active_context WHERE session_id = 'default'")
        # Write new context
        now = _time.time()
        for node_name in selected_nodes:
            node = engine._nodes.get(node_name)
            activation = node.activation if node else 0.0
            conn.execute(
                "INSERT INTO active_context (session_id, node_name, activation, updated_at) "
                "VALUES (?, ?, ?, ?)",
                ("default", node_name, activation, now),
            )
        conn.commit()


def load_active_context(stg_path: str, session_id: str = "default") -> List[Tuple[str, float]]:
    """Load active context from .stg SQLite.

    Returns:
        List of (node_name, activation) sorted by activation descending
    """
    import sqlite3
    import os

    if not os.path.exists(stg_path):
        return []

    try:
        with sqlite3.connect(stg_path) as conn:
            # Check if table exists
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='active_context'"
            ).fetchall()
            if not tables:
                return []

            rows = conn.execute(
                "SELECT node_name, activation FROM active_context "
                "WHERE session_id = ? ORDER BY activation DESC",
                (session_id,),
            ).fetchall()
            return [(name, act) for name, act in rows]
    except Exception:
        return []


def clear_active_context(stg_path: str, session_id: str = "default") -> None:
    """Clear active context (e.g., at session-end)."""
    import sqlite3
    import os

    if not os.path.exists(stg_path):
        return

    try:
        with sqlite3.connect(stg_path) as conn:
            conn.execute(
                "DELETE FROM active_context WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()
    except Exception:
        pass
