"""Concurrency test for the CLI write lock (F1/F4).

Two subprocesses ingest DIFFERENT edges into the same .stg with a forced
load->save overlap. With the lock, writers serialize and BOTH edges survive
cleanly. Without the lock (STG_NO_INTERLOCK=1) the concurrent full-graph saves
collide — in practice either a writer crashes (tmp-file / rename collision) or
the second save clobbers the first (lost update). Either way the result is not
clean, which proves the lock in the positive test is doing real work (if the
"control" ever produced a clean result, the positive test would be vacuous).
"""
import os
import subprocess
import sys
from pathlib import Path

from stg_engine.engine import STGEngine
from stg_engine import interlock

HELPER = Path(__file__).parent / "_interlock_writer.py"
DELAY = "1.0"  # seconds; generously exceeds subprocess import+load jitter


def _make_base_stg(path):
    STGEngine().save(str(path))


def _run_two_writers(stg_path, use_lock):
    """Run two concurrent writers; return (all_exited_zero, edge_set)."""
    env = dict(os.environ)
    if use_lock:
        env.pop("STG_NO_INTERLOCK", None)
    else:
        env["STG_NO_INTERLOCK"] = "1"
    procs = [
        subprocess.Popen([sys.executable, str(HELPER), str(stg_path), tag, DELAY], env=env)
        for tag in ("E1", "E2")
    ]
    codes = [p.wait(timeout=60) for p in procs]
    engine = STGEngine.load(str(stg_path))
    edges = {(e.source, e.target) for e in engine._edges}
    return all(c == 0 for c in codes), edges


def _is_clean(ok, edges):
    targets = {t for _, t in edges}
    return ok and {"E1", "E2"} <= targets and len(edges) == 2


def test_write_lock_prevents_lost_update(tmp_path):
    stg = tmp_path / "concurrent.stg"
    _make_base_stg(stg)
    ok, edges = _run_two_writers(stg, use_lock=True)
    assert _is_clean(ok, edges), f"lock failed to serialize writers: ok={ok} edges={edges}"


def test_without_lock_demonstrates_race(tmp_path):
    stg = tmp_path / "concurrent_nolock.stg"
    _make_base_stg(stg)
    ok, edges = _run_two_writers(stg, use_lock=False)
    assert not _is_clean(ok, edges), (
        f"expected the race to crash a writer or lose an edge, but the result "
        f"was clean: ok={ok} edges={edges}"
    )


def test_lock_file_is_sidecar(tmp_path):
    stg = tmp_path / "x.stg"
    assert interlock.lock_path(str(stg)) == str(stg) + ".lock"


def test_timeout_when_held(tmp_path):
    import pytest
    stg = tmp_path / "held.stg"
    _make_base_stg(stg)
    held = interlock.StgFileLock(str(stg), timeout=15.0)
    held.acquire()
    try:
        contender = interlock.StgFileLock(str(stg), timeout=0.3, poll=0.05)
        with pytest.raises(interlock.LockTimeout):
            contender.acquire()
    finally:
        held.release()


# --- optimistic mtime/counts guard in cli._cli_save (F1/F4 belt-and-braces) ---

def test_cli_save_refuses_external_graph_change(tmp_path, monkeypatch):
    """A non-cooperating writer rewrote the graph since load -> refuse to save."""
    import time
    import pytest
    from stg_engine import cli

    stg = tmp_path / "guard.stg"
    eng = STGEngine()
    eng.add_edge("A", "B", confidence=1.0, is_a="x")
    eng.save(str(stg))
    monkeypatch.setattr(cli, "STG_PATH", str(stg))
    cli._note_loaded_baseline(eng)

    # Simulate an external writer changing node/edge counts + mtime.
    time.sleep(0.01)
    ext = STGEngine.load(str(stg))
    ext.add_edge("C", "D", confidence=1.0, is_a="x")
    ext.save(str(stg))

    with pytest.raises(SystemExit) as ei:
        cli._cli_save(eng)
    assert ei.value.code == 3


def test_cli_save_allows_own_incremental_write(tmp_path, monkeypatch):
    """An incremental side-write (mtime changes, counts do not) -> save proceeds.

    This is the propagate case: telemetry.flush/active_context bump the file
    mtime before engine.save, and the guard must NOT false-refuse.
    """
    import sqlite3
    import time
    from stg_engine import cli

    stg = tmp_path / "aux.stg"
    eng = STGEngine()
    eng.add_edge("A", "B", confidence=1.0, is_a="x")
    eng.save(str(stg))
    monkeypatch.setattr(cli, "STG_PATH", str(stg))
    cli._note_loaded_baseline(eng)

    time.sleep(0.01)
    conn = sqlite3.connect(str(stg))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS active_context "
        "(session_id TEXT, node_name TEXT, activation REAL, updated_at REAL)"
    )
    conn.execute("INSERT INTO active_context VALUES ('default','A',1.0,0.0)")
    conn.commit()
    conn.close()

    cli._cli_save(eng)  # must not raise
    assert len(STGEngine.load(str(stg))._edges) == 1
