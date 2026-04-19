"""Example 1: Basic STG Engine usage.

This example shows the simplest possible STG workflow:
  1. Create an empty engine
  2. Ingest a few facts as STL statements
  3. Run a query (spreading activation)
  4. Inspect the results

Run:
    python examples/01_basic_usage.py
"""
from stg_engine import STGEngine
from stg_engine.engine import _HAS_RUST


def main() -> None:
    print("=" * 60)
    print("STG Engine — Basic Usage Example")
    print("=" * 60)
    print(f"Rust acceleration: {'ENABLED' if _HAS_RUST else 'disabled (pure Python)'}")
    print()

    # ── Create an empty engine ──────────────────────────────────
    engine = STGEngine()

    # ── Ingest facts as STL statements ─────────────────────────
    # Format: [Source] -> [Target] ::mod(key=value, ...)
    facts = [
        '[Newton] -> [Calculus] ::mod(rule="historical", confidence=0.95, source="Principia 1687")',
        '[Leibniz] -> [Calculus] ::mod(rule="historical", confidence=0.95, source="Acta Eruditorum 1684")',
        '[Calculus] -> [Physics] ::mod(rule="enables", confidence=0.92, strength=0.9)',
        '[Calculus] -> [Engineering] ::mod(rule="enables", confidence=0.88, strength=0.85)',
        '[Physics] -> [Engineering] ::mod(rule="enables", confidence=0.85)',
        '[Newton] -> [Optics] ::mod(rule="historical", confidence=0.92)',
        '[Newton] -> [LawsOfMotion] ::mod(rule="historical", confidence=0.99)',
    ]

    print(f"Ingesting {len(facts)} STL statements...")
    for stl in facts:
        engine.ingest_stl(stl)

    print(f"  -> {len(engine._nodes)} nodes, {len(engine._edges)} edges")
    print()

    # ── Spreading activation: ask "what does Newton connect to?" ──
    print('Query: propagate("Newton")')
    results = engine.propagate("Newton")
    print(f"  Activated {len(results)} nodes:")
    for i, name in enumerate(results, 1):
        node = engine._nodes[name.lower()]
        print(f"    {i}. {name:20}  activation={node.activation:.3f}")
    print()

    # ── Query a deeper concept ──────────────────────────────────
    print('Query: propagate("Engineering")')
    results = engine.propagate("Engineering")
    print(f"  Top results: {results[:5]}")
    print()

    # ── Inspect a single node ───────────────────────────────────
    print("Node detail: Calculus")
    node = engine._nodes["calculus"]
    print(f"  name={node.name}")
    print(f"  activation={node.activation:.3f}")
    print(f"  tension={node.tension:.3f}")

    # ── Compute system stability (Ψ) ────────────────────────────
    psi = engine.compute_psi()
    print(f"\nSystem stability Ψ = {psi:.3f}")
    print()
    print("Done. Try modifying the facts above and re-running.")


if __name__ == "__main__":
    main()
