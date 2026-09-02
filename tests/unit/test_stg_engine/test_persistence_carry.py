"""Side tables must survive an in-place save; pruned_log purge is reversible.

Background (2026-09-02): save_engine_state rebuilt the .stg from engine state
and carried only an explicit list of extra tables. skill_invocations was not
on it — every `stg use` audit row was wiped by the next write command, and
telemetry_cooccurrence / perception_* were dropped outright. The carry is now
generic: every table the save does not rebuild is copied verbatim.
"""
import json
import sqlite3
import time

from stg_engine.engine import STGEngine
from stg_engine.learning import SynapticPruner
from stg_engine.persistence import (
    _REBUILT_TABLES,
    append_pruned_log,
    purge_pruned_log,
    read_pruned_log,
)


def _graph(tmp_path, name="g.stg"):
    e = STGEngine()
    for i in range(6):
        e.add_edge(f"Node_{i}", f"Node_{(i + 1) % 6}", is_a="x")
    p = str(tmp_path / name)
    e.save(p)
    return e, p


def _tables(p):
    conn = sqlite3.connect(p)
    names = {
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'"
        )
    }
    conn.close()
    return names


def _count(p, table):
    conn = sqlite3.connect(p)
    n = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    conn.close()
    return n


# ── generic carry ────────────────────────────────────────────────────────


def test_skill_invocations_survive_inplace_save(tmp_path):
    _, p = _graph(tmp_path)
    conn = sqlite3.connect(p)
    conn.execute(
        "INSERT INTO skill_invocations "
        "(invocation_id, timestamp, skill_name, exit_code, elapsed_s) "
        "VALUES ('i1', 1.0, 'MemCk', 0, 0.1)"
    )
    conn.commit()
    conn.close()
    assert _count(p, "skill_invocations") == 1

    e2 = STGEngine.load(p)
    e2.add_edge("Node_0", "Node_99", is_a="x")   # a real mutation, then save
    e2.save(p)

    conn = sqlite3.connect(p)
    rows = conn.execute(
        "SELECT invocation_id, skill_name FROM skill_invocations"
    ).fetchall()
    conn.close()
    assert rows == [("i1", "MemCk")]


def test_every_schema_table_survives_inplace_save(tmp_path):
    """No table present before the save may be missing after it."""
    _, p = _graph(tmp_path)
    # Lazy subsystem tables appear when a subsystem first touches the file
    # (schema migration), not on a fresh save — simulate that touch.
    from stg_engine.persistence import _migrate_schema
    conn = sqlite3.connect(p)
    _migrate_schema(conn)
    conn.commit()
    conn.close()
    before = _tables(p)
    assert {"telemetry_cooccurrence", "perception_frames",
            "perception_filters", "skill_invocations"} <= before
    e2 = STGEngine.load(p)
    e2.save(p)
    after = _tables(p)
    assert before <= after, f"dropped: {before - after}"


def test_unknown_side_table_with_index_survives(tmp_path):
    """A table the engine has never heard of is carried with its index."""
    _, p = _graph(tmp_path)
    conn = sqlite3.connect(p)
    conn.execute("CREATE TABLE plugin_state (k TEXT PRIMARY KEY, v TEXT, n INTEGER)")
    conn.execute("CREATE INDEX idx_plugin_state_n ON plugin_state(n)")
    conn.executemany(
        "INSERT INTO plugin_state VALUES (?, ?, ?)", [("a", "1", 1), ("b", "2", 2)]
    )
    conn.commit()
    conn.close()

    e2 = STGEngine.load(p)
    e2.save(p)

    conn = sqlite3.connect(p)
    rows = conn.execute("SELECT k, v, n FROM plugin_state ORDER BY k").fetchall()
    idx = {
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND tbl_name='plugin_state'"
        )
    }
    conn.close()
    assert rows == [("a", "1", 1), ("b", "2", 2)]
    assert "idx_plugin_state_n" in idx


def test_column_drift_extra_old_column_is_ignored(tmp_path):
    """Old file has a column the current schema lacks: rows still carry."""
    _, p = _graph(tmp_path)
    conn = sqlite3.connect(p)
    conn.execute("ALTER TABLE skill_invocations ADD COLUMN future_col TEXT")
    conn.execute(
        "INSERT INTO skill_invocations "
        "(invocation_id, timestamp, skill_name, exit_code, elapsed_s, future_col) "
        "VALUES ('i1', 1.0, 'X', 0, 0.1, 'zzz')"
    )
    conn.commit()
    conn.close()

    e2 = STGEngine.load(p)
    e2.save(p)

    conn = sqlite3.connect(p)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(skill_invocations)")}
    n = conn.execute("SELECT count(*) FROM skill_invocations").fetchone()[0]
    conn.close()
    assert n == 1
    assert "future_col" not in cols     # current schema wins, data carried


def test_pruned_log_and_embeddings_carry_across_save(tmp_path):
    _, p = _graph(tmp_path)
    append_pruned_log(p, [_entry("edge", "A", "B", 1.0)])
    conn = sqlite3.connect(p)
    conn.execute(
        "INSERT INTO embeddings (node_name, embed_text, vector, model_name, created_at) "
        "VALUES ('Node_0', 'Node 0', X'00', 'm', 1.0)"
    )
    conn.commit()
    conn.close()
    e2 = STGEngine.load(p)
    e2.save(p)
    assert _count(p, "pruned_log") == 1
    assert _count(p, "embeddings") == 1


def test_rebuilt_tables_are_not_double_written(tmp_path):
    """Nodes/edges come from engine state only — never appended from old."""
    e, p = _graph(tmp_path)
    n_edges = len(e.get_edges())
    e2 = STGEngine.load(p)
    e2.save(p)
    e3 = STGEngine.load(p)
    assert len(e3.get_edges()) == n_edges
    assert "edges" in _REBUILT_TABLES and "nodes" in _REBUILT_TABLES


def test_save_to_fresh_path_has_no_old_to_carry(tmp_path):
    e, _ = _graph(tmp_path)
    p2 = str(tmp_path / "fresh.stg")
    e.save(p2)                      # no existing file — must not raise
    assert _count(p2, "skill_invocations") == 0


# ── pruned_log purge ─────────────────────────────────────────────────────


def _entry(item_type, src, tgt, at):
    return {
        "pruned_at": at, "item_type": item_type, "source": src, "target": tgt,
        "confidence": 0.5, "salience": 0.1, "last_used": None,
        "modifiers_json": "{}", "reason": "test",
    }


def test_purge_dumps_then_deletes_matching_rows(tmp_path):
    _, p = _graph(tmp_path)
    append_pruned_log(p, [
        _entry("virtual_edge", "A", "B", 10.0),
        _entry("virtual_edge", "C", "D", 20.0),
        _entry("edge", "E", "F", 30.0),
    ])
    dump = str(tmp_path / "dump.jsonl")
    r = purge_pruned_log(p, item_type="virtual_edge", dump_path=dump)
    assert (r["matched"], r["deleted"], r["dump_path"]) == (2, 2, dump)
    lines = [json.loads(l) for l in open(dump, encoding="utf-8")]
    assert [l["item_type"] for l in lines] == ["virtual_edge", "virtual_edge"]
    assert [l["source"] for l in lines] == ["A", "C"]
    left = read_pruned_log(p)
    assert [(x["item_type"], x["source"]) for x in left] == [("edge", "E")]
    assert r["bytes_after"] is not None


def test_purge_dry_run_touches_nothing(tmp_path):
    _, p = _graph(tmp_path)
    append_pruned_log(p, [_entry("virtual_edge", "A", "B", 10.0)])
    r = purge_pruned_log(p, item_type="virtual_edge", dry_run=True)
    assert r["matched"] == 1 and r["deleted"] == 0 and r["dump_path"] is None
    assert _count(p, "pruned_log") == 1


def test_purge_before_cutoff_and_default_dump_location(tmp_path):
    _, p = _graph(tmp_path)
    append_pruned_log(p, [
        _entry("edge", "A", "B", 10.0),
        _entry("edge", "C", "D", 20.0),
        _entry("edge", "E", "F", 30.0),
    ])
    r = purge_pruned_log(p, before=25.0)
    assert r["deleted"] == 2
    assert r["dump_path"].startswith(str(tmp_path / "ARCHIVED" / "pruned_log-"))
    assert r["dump_path"].endswith(".jsonl")
    assert [x["source"] for x in read_pruned_log(p)] == ["E"]


def test_purge_survives_a_later_inplace_save(tmp_path):
    """The purge is a direct file mutation; the next save must not resurrect rows."""
    _, p = _graph(tmp_path)
    append_pruned_log(p, [_entry("virtual_edge", "A", "B", 10.0)])
    purge_pruned_log(p, item_type="virtual_edge", dump_path=str(tmp_path / "d.jsonl"))
    e2 = STGEngine.load(p)
    e2.save(p)
    assert _count(p, "pruned_log") == 0


# ── pruner: virtual edges are removed but not logged by default ──────────


def _engine_with_stale_virtual(tmp_path):
    e = STGEngine()
    e.add_edge("Real_A", "Real_B", is_a="x")
    long_ago = time.time() - 400 * 86400
    e.add_edge("Real_A", "Virt_C", edge_class="virtual", virtual_created_at=long_ago)
    p = str(tmp_path / "v.stg")
    e.save(p)
    return e, p


def test_prune_removes_stale_virtual_edge_without_logging(tmp_path):
    e, p = _engine_with_stale_virtual(tmp_path)
    pruner = SynapticPruner()
    pruner.virtual_unused_days = 30
    events = pruner.prune(e, stg_path=p)
    assert any(ev.event_type == "prune_virtual" for ev in events)
    assert read_pruned_log(p, item_type="virtual_edge") == []


def test_prune_logs_virtual_edge_when_asked(tmp_path):
    e, p = _engine_with_stale_virtual(tmp_path)
    pruner = SynapticPruner()
    pruner.virtual_unused_days = 30
    pruner.prune(e, stg_path=p, log_virtual=True)
    rows = read_pruned_log(p, item_type="virtual_edge")
    assert [(r["source"], r["target"]) for r in rows] == [("Real_A", "Virt_C")]


# ── CLI: --help never runs the command ───────────────────────────────────


def test_cli_help_flag_does_not_touch_the_graph(tmp_path, monkeypatch, capsys):
    from stg_engine import cli as cli_mod
    ghost = tmp_path / "never-created.stg"
    monkeypatch.setattr(cli_mod, "STG_PATH", str(ghost))
    monkeypatch.setattr(cli_mod.sys, "argv", ["stg", "propagate", "--help"])
    cli_mod.main()
    out = capsys.readouterr().out
    assert "propagate <text>" in out
    assert not ghost.exists()
    assert not (tmp_path / "never-created.stg.lock").exists()


def test_cli_help_unknown_command_points_to_guide(monkeypatch, capsys):
    from stg_engine import cli as cli_mod
    monkeypatch.setattr(cli_mod.sys, "argv", ["stg", "no-such-cmd", "-h"])
    cli_mod.main()
    assert "No help entry" in capsys.readouterr().out
