"""Cross-platform advisory file lock for STG CLI write serialization (F1/F4).

WHY. Every `stg` write command does load-whole-graph -> mutate in memory ->
save-whole-graph (persistence.save_engine_state writes a tmp file then renames
it over the target = atomic full replacement). Two concurrent writers — the
normal solo setup runs an exec session, a scribe session and the main
conversation, any of which may invoke `stg` — would each load the same base
snapshot and the second save would silently overwrite the first: a *graph-level*
lost update (a whole session's ingests gone). This module serializes writers
with an OS advisory lock taken on a sidecar file ``<stg>.lock``.

SCOPE / BOUNDARY (deliberate — read before extending):

* **CLI-layer only.** The library API (``STGEngine.load`` / ``STGEngine.save``)
  is intentionally NOT wrapped. The FastAPI server uses the engine read-only,
  and embedded SKC usage manages its own lifecycle; forcing a lock inside
  ``save`` would change the library contract and could deadlock or surprise
  those callers. Locking is a policy the *CLI* opts into. A non-cooperating
  writer (anything that bypasses this lock) is still caught by the optimistic
  mtime/counts guard in ``cli._cli_save``.

* **Reads take NO lock.** ``STGEngine.load`` reads the file within a single
  connection and ``save`` replaces it atomically, so a reader always observes a
  consistent old-or-new snapshot. A *shared* lock is deliberately avoided: long
  read or ``stg use <skill>`` commands (skills may run for a very long time)
  would otherwise hold a shared lock and starve writers. Only whole-graph
  writers take a lock, and it is always exclusive.

* **Sidecar file.** The lock lives on ``<stg>.lock``, never on the ``.stg``
  itself, so it can never interfere with save's tmp+rename.

PLATFORMS. POSIX uses ``fcntl.flock``; Windows uses ``msvcrt.locking``
(exclusive only — msvcrt has no shared mode, which is fine because we only ever
take exclusive locks). Standard library only; no third-party dependency.

ESCAPE HATCH. Set ``STG_NO_INTERLOCK=1`` to disable locking (used by the
no-lock control in the concurrency test, and available for emergency recovery).
"""
import os
import time

try:
    import fcntl  # POSIX
    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover - platform dependent
    _HAVE_FCNTL = False

try:
    import msvcrt  # Windows
    _HAVE_MSVCRT = True
except ImportError:  # pragma: no cover - platform dependent
    _HAVE_MSVCRT = False


class LockTimeout(RuntimeError):
    """Raised when the STG write lock cannot be acquired within the timeout."""


def lock_path(stg_path: str) -> str:
    """Sidecar lock-file path for a given .stg path."""
    return str(stg_path) + ".lock"


class StgFileLock:
    """Exclusive advisory lock on ``<stg_path>.lock`` (also a context manager)."""

    def __init__(self, stg_path: str, timeout: float = 15.0, poll: float = 0.05):
        self.stg_path = str(stg_path)
        self.lock_file = lock_path(stg_path)
        self.timeout = timeout
        self.poll = poll
        self._fd = None

    def acquire(self) -> "StgFileLock":
        parent = os.path.dirname(self.lock_file)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # Keep the fd open for the lock's lifetime; closing it releases the lock.
        self._fd = open(self.lock_file, "a+")
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self._try_lock()
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    self._fd.close()
                    self._fd = None
                    raise LockTimeout(
                        f"could not acquire STG write lock within "
                        f"{self.timeout:.0f}s: {self.lock_file}\n"
                        f"Another `stg` write command is probably holding it "
                        f"(look for a running stg / ingest / propagate process). "
                        f"Retry shortly; if you are certain no stg process is "
                        f"running, the lock is stale — delete the file above."
                    )
                time.sleep(self.poll)
                continue
            # Acquired — record holder info for diagnostics (best effort).
            try:
                self._fd.seek(0)
                self._fd.truncate()
                self._fd.write(f"pid={os.getpid()} ts={time.time():.3f}\n")
                self._fd.flush()
            except OSError:
                pass
            return self

    def _try_lock(self):
        if _HAVE_FCNTL:
            fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        elif _HAVE_MSVCRT:  # pragma: no cover - Windows only
            self._fd.seek(0)
            msvcrt.locking(self._fd.fileno(), msvcrt.LK_NBLCK, 1)
        # else: no known primitive on this platform -> degrade to no-op.

    def release(self):
        if self._fd is None:
            return
        try:
            if _HAVE_FCNTL:
                fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
            elif _HAVE_MSVCRT:  # pragma: no cover - Windows only
                try:
                    self._fd.seek(0)
                    msvcrt.locking(self._fd.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
        finally:
            try:
                self._fd.close()
            except OSError:
                pass
            self._fd = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()


# Locks a CLI process holds until it exits (kept referenced so the fd stays open).
_HELD = []


def hold_stg_lock(stg_path: str, timeout: float = 15.0):
    """Acquire an exclusive lock and hold it for the rest of the process.

    A ``stg`` process runs exactly one command then exits, so holding the lock
    from command dispatch to process exit cleanly spans the load->mutate->save
    window (process exit also releases the OS lock as a backstop). Returns the
    ``StgFileLock`` (or ``None`` when disabled via ``STG_NO_INTERLOCK``).
    """
    if os.environ.get("STG_NO_INTERLOCK"):
        return None
    lock = StgFileLock(stg_path, timeout=timeout)
    lock.acquire()
    _HELD.append(lock)
    import atexit
    atexit.register(lock.release)
    return lock
