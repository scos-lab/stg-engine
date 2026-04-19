"""Example 2: Persistence — save and load STG state.

This example shows how to:
  1. Build an engine
  2. Save it to a .stg file (SQLite-backed)
  3. Load it back in a fresh process
  4. Verify state is preserved

Run:
    python examples/02_persistence.py
"""
import os
import tempfile

from stg_engine import STGEngine


def main() -> None:
    print("=" * 60)
    print("STG Engine — Persistence Example")
    print("=" * 60)
    print()

    # ── Build an engine in process A ────────────────────────────
    engine_a = STGEngine()
    facts = [
        '[Brain] -> [Memory] ::mod(rule="contains", confidence=0.99)',
        '[Memory] -> [Hippocampus] ::mod(rule="located_in", confidence=0.92)',
        '[Memory] -> [LongTerm] ::mod(rule="has_kind", confidence=0.95)',
        '[Memory] -> [ShortTerm] ::mod(rule="has_kind", confidence=0.95)',
        '[Hippocampus] -> [Encoding] ::mod(rule="performs", confidence=0.9)',
    ]
    for stl in facts:
        engine_a.ingest_stl(stl)

    print(f"Process A: built engine with {len(engine_a._nodes)} nodes, {len(engine_a._edges)} edges")

    # Run a propagation to give some nodes activation
    activated_a = engine_a.propagate("Brain Memory")
    print(f"Process A: propagate('Brain Memory') -> {activated_a}")

    # ── Save ────────────────────────────────────────────────────
    tmpdir = tempfile.mkdtemp()
    save_path = os.path.join(tmpdir, "memory.stg")
    engine_a.save(save_path)
    file_size = os.path.getsize(save_path)
    print(f"Process A: saved to {save_path} ({file_size} bytes)")
    print()

    # ── Load in a fresh "process" ───────────────────────────────
    # Discard the original engine to prove the file contains everything
    del engine_a

    engine_b = STGEngine.load(save_path)
    print(f"Process B: loaded from disk")
    print(f"Process B: {len(engine_b._nodes)} nodes, {len(engine_b._edges)} edges")

    # Verify the same query produces the same results
    activated_b = engine_b.propagate("Brain Memory")
    print(f"Process B: propagate('Brain Memory') -> {activated_b}")

    # Compare
    if activated_a == activated_b:
        print("\n  ✓ Loaded state produces identical query results")
    else:
        print(f"\n  ! Mismatch:\n    A: {activated_a}\n    B: {activated_b}")

    # Cleanup
    os.remove(save_path)
    os.rmdir(tmpdir)
    print("\nDone. The .stg file format is portable across processes and machines.")


if __name__ == "__main__":
    main()
