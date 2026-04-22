# Bug Report: `stg propagate` can corrupt `memory.stg` after save failure

Date observed: 2026-04-16

Agent/database affected:

- Agent: `linux`
- Database path: `~/.stg/linux/memory.stg`
- CLI used: `stg`
- Source checkout referenced by traceback: `<stg-engine checkout>`

## Summary

While using `stg propagate` normally against the `linux` agent, some propagation commands printed valid results and Hebbian updates, but then failed during the final save step with:

```text
sqlite3.IntegrityError: UNIQUE constraint failed: nodes.name
```

After that failure, subsequent `stg --agent linux ...` commands could no longer load the database:

```text
sqlite3.OperationalError: no such table: nodes
```

The main `memory.stg` file had been reduced to a 4096-byte SQLite shell, while the actual data was still recoverable from a leftover WAL file named `memory.stg.tmp-wal`.

This suggests that the save path is not failure-atomic. A failed `save_engine_state()` can leave the live database replaced or truncated and dependent on temporary WAL files.

## Commands run before failure

The CLI was used in the normal supported way:

```bash
stg --agent linux stats
stg --agent linux query scheduler --limit 20
stg --agent linux query cfs --limit 20
stg --agent linux topology communities
stg --agent linux importance --top 20
stg --agent linux query eevdf --limit 20
stg --agent linux query rcu --limit 20
stg --agent linux node CFS_Scheduler
stg --agent linux paths CFS_Scheduler EEVDF_Two_Criteria_Selection
stg --agent linux query workqueue --limit 15
stg --agent linux query tcp --limit 15
```

Then these `propagate` commands were run:

```bash
stg --agent linux propagate "CFS scheduler why design"
stg --agent linux propagate "why CFS changed to EEVDF scheduler design" --expand 2
stg --agent linux propagate "why workqueue cmwq design worker pool" --expand 2
stg --agent linux propagate "tcp retransmission timeout memory pressure design" --expand 2
stg --agent linux propagate "eBPF verifier jit attach hooks security design" --expand 2
```

The workqueue propagation completed successfully. The CFS/EEVDF, TCP, and eBPF propagation commands printed useful propagation output and then failed during save.

## First failure trace

One failing command was:

```bash
stg --agent linux propagate "why CFS changed to EEVDF scheduler design" --expand 2
```

It printed propagation results and:

```text
Hebbian: +11 strengthen, -0 weaken
QE=1.000  RS=0.213  coverage=0.0250  seeds=42
```

Then it failed:

```text
Traceback (most recent call last):
  File "stg", line 8, in <module>
    sys.exit(main())
             ^^^^^^
  File "<stg-engine checkout>/stg_engine/cli.py", line 3179, in main
    cmd_propagate(engine, " ".join(args), use_gravity=use_gravity, resolution=resolution,
                      all_chains=all_chains, all_modifiers=all_modifiers, expand_top=expand_top)
  File "<stg-engine checkout>/stg_engine/cli.py", line 748, in cmd_propagate
    engine.save(STG_PATH)
  File "<stg-engine checkout>/stg_engine/engine.py", line 2344, in save
    save_engine_state(
  File "<stg-engine checkout>/stg_engine/persistence.py", line 517, in save_engine_state
    conn.executemany(
sqlite3.IntegrityError: UNIQUE constraint failed: nodes.name
```

Similar save failures occurred after the TCP and eBPF propagation commands.

## Load failure after corruption

After the failed saves, subsequent commands could not load the agent database:

```text
Traceback (most recent call last):
  File "stg", line 8, in <module>
    sys.exit(main())
             ^^^^^^
  File "<stg-engine checkout>/stg_engine/cli.py", line 3076, in main
    engine = load_engine()
             ^^^^^^^^^^^^^
  File "<stg-engine checkout>/stg_engine/cli.py", line 195, in load_engine
    return STGEngine.load(STG_PATH)
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "<stg-engine checkout>/stg_engine/engine.py", line 2367, in load
    state = load_engine_state(path)
            ^^^^^^^^^^^^^^^^^^^^^^^
  File "<stg-engine checkout>/stg_engine/persistence.py", line 765, in load_engine_state
    rows = conn.execute("SELECT * FROM nodes").fetchall()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
sqlite3.OperationalError: no such table: nodes
```

## On-disk state after failure

Directory listing showed:

```text
~/.stg/linux/memory.stg         4096 bytes
~/.stg/linux/memory.stg.tmp-shm 32768 bytes
~/.stg/linux/memory.stg.tmp-wal 1108312 bytes
~/.stg/linux/audit.log          10250 bytes
```

`file` output:

```text
~/.stg/linux/memory.stg:
  SQLite 3.x database, last written using SQLite version 3045001, writer version 2, read version 2, file counter 1, database pages 1, cookie 0, schema 0, unknown 0 encoding, version-valid-for 1

~/.stg/linux/memory.stg.tmp-wal:
  SQLite Write-Ahead Log, version 3007000

~/.stg/linux/memory.stg.tmp-shm:
  SQLite Write-Ahead Log shared memory, counter 37, page size 4096, 269 frames, 196 pages
```

The live main database file had no schema, while the WAL still contained the complete data.

## Recovery performed

No manual edits were made to database content.

The damaged files were copied to `/tmp` first:

```bash
mkdir -p /tmp/stg-linux-recover
cp ~/.stg/linux/memory.stg /tmp/stg-linux-recover/memory.stg.tmp
cp ~/.stg/linux/memory.stg.tmp-wal /tmp/stg-linux-recover/memory.stg.tmp-wal
cp ~/.stg/linux/memory.stg.tmp-shm /tmp/stg-linux-recover/memory.stg.tmp-shm
```

Then Python `sqlite3` was used to open the copied DB and checkpoint the WAL:

```python
import sqlite3, os

p = "/tmp/stg-linux-recover/memory.stg.tmp"
con = sqlite3.connect(p)
print(con.execute("select name from sqlite_master where type='table' order by name").fetchall())
print("nodes", con.execute("select count(*) from nodes").fetchone())
print("edges", con.execute("select count(*) from edges").fetchone())
print("wal_checkpoint", con.execute("pragma wal_checkpoint(full)").fetchall())
con.close()
print("size", os.path.getsize(p))
```

Recovery output:

```text
nodes (1000,)
edges (936,)
wal_checkpoint [(0, 269, 269)]
size 802816
```

The original broken files were backed up, the recovered database was copied back to `memory.stg`, and stale tmp WAL/SHM files were moved aside.

After recovery:

```bash
stg --agent linux stats
```

returned:

```text
Nodes: 1000
Edges: 936 (634 real + 302 virtual)
Sessions: 0
Events: 0
Tensions: 0 (0 active)
Belief Evolutions: 0
Psi: 97.9022
Density: 0.000908
```

## Files left for inspection

The following files were left in `~/.stg/linux/`:

```text
memory.stg
memory.stg.broken-20260416-1254
memory.stg.tmp-wal.broken-20260416-1254
memory.stg.tmp-shm.broken-20260416-1254
memory.stg.tmp-wal.restored-20260416-1254
memory.stg.tmp-shm.restored-20260416-1254
audit.log
```

## Initial technical hypothesis

There appear to be two related problems:

1. During `propagate`, Hebbian learning or related save-time mutation can produce duplicate node names in the in-memory engine state. The immediate exception is:

   ```text
   sqlite3.IntegrityError: UNIQUE constraint failed: nodes.name
   ```

   Relevant traceback points:

   - `stg_engine/cli.py`, `cmd_propagate()`, around line 748: `engine.save(STG_PATH)`
   - `stg_engine/engine.py`, `STGEngine.save()`, around line 2344
   - `stg_engine/persistence.py`, `save_engine_state()`, around line 517, `conn.executemany(...)`

2. `save_engine_state()` is not failure-atomic. When the insert into the temporary SQLite database fails, the live `memory.stg` can be left as a 4096-byte DB with no `nodes` table, while `memory.stg.tmp-wal` contains the actual data.

## Suggested fix areas

- Before saving, validate that the in-memory node collection has unique canonical node names and report duplicate names with enough detail to debug their source.
- Investigate whether virtual edges, case normalization, alias resolution, or propagation/Hebbian updates can introduce duplicate `STGNode.name` values.
- Make `save_engine_state()` failure-atomic:
  - Never replace/truncate the live `memory.stg` until the new database is fully written and validated.
  - Use a uniquely named temp DB, not a stable `memory.stg.tmp` if concurrent or repeated commands are possible.
  - Ensure WAL mode is checkpointed before rename.
  - Clean up temp `-wal` and `-shm` files on both success and failure.
  - On any exception, preserve the original live DB unchanged.
- Add a regression test that simulates a mid-save `IntegrityError` and asserts that the original `.stg` file remains loadable with its original schema and counts.

## Notes for reproduction safety

Do not reproduce this directly on `~/.stg/linux/memory.stg` unless it is copied first. Use a copied agent database or `--path` pointing to a scratch `.stg` file.
