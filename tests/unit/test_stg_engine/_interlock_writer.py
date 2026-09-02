"""Subprocess helper for test_interlock.py.

Load a .stg, wait (to force a load->save overlap window), add one edge, save.
Uses the CLI write lock via interlock.hold_stg_lock unless STG_NO_INTERLOCK=1
is set in the environment. Not a test module (leading underscore) so pytest
does not collect it.

    python _interlock_writer.py <stg_path> <edge_target> <delay_seconds>
"""
import sys
import time

from stg_engine.engine import STGEngine
from stg_engine import interlock


def main():
    stg_path, target, delay = sys.argv[1], sys.argv[2], float(sys.argv[3])
    lock = interlock.hold_stg_lock(stg_path, timeout=30.0)  # None if disabled
    engine = STGEngine.load(stg_path)
    time.sleep(delay)  # widen load->save window so the race is deterministic
    engine.add_edge("A", target, confidence=1.0, is_a="concurrency_probe")
    engine.save(stg_path)
    if lock is not None:
        lock.release()


if __name__ == "__main__":
    main()
