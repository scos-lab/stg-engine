"""Example 3: Hebbian learning — edges strengthen with use.

This example demonstrates the core distinction between *confidence*
(how true a fact is) and *salience* (how easily it is recalled):

  - Confidence is set when you ingest a fact and never auto-decays.
  - Salience grows when an edge is co-activated (Hebbian) and decays
    when its endpoints are not used together.

After repeating the same query 20 times, the salience of edges along
that query path should increase visibly, while edges off-path stay flat.

Run:
    python examples/03_hebbian_learning.py
"""
from stg_engine import STGEngine
from stg_engine.learning import HebbianLearner


def main() -> None:
    print("=" * 60)
    print("STG Engine — Hebbian Learning Example")
    print("=" * 60)
    print()

    # ── Build a small graph with two paths from A to D ──────────
    engine = STGEngine()
    facts = [
        # Path 1: A -> B -> D  (will be used)
        '[A] -> [B] ::mod(confidence=0.5)',
        '[B] -> [D] ::mod(confidence=0.5)',
        # Path 2: A -> C -> D  (will not be used)
        '[A] -> [C] ::mod(confidence=0.5)',
        '[C] -> [D] ::mod(confidence=0.5)',
        # Some unrelated edges
        '[X] -> [Y] ::mod(confidence=0.5)',
        '[Y] -> [Z] ::mod(confidence=0.5)',
    ]
    for stl in facts:
        engine.ingest_stl(stl)

    # Manually reset all saliences to a known starting point so the
    # learning effect is easy to see in the output.
    for edge in engine._edges:
        edge.salience = 0.5

    # Helper to read salience of a specific edge
    def sal(src: str, tgt: str) -> float:
        edge = engine._edges_lookup.get((src.lower(), tgt.lower()))
        return edge.salience if edge else float("nan")

    print("Initial salience:")
    print(f"  A -> B  =  {sal('A','B'):.3f}")
    print(f"  B -> D  =  {sal('B','D'):.3f}")
    print(f"  A -> C  =  {sal('A','C'):.3f}  (off-path)")
    print(f"  C -> D  =  {sal('C','D'):.3f}  (off-path)")
    print(f"  X -> Y  =  {sal('X','Y'):.3f}  (unrelated)")
    print()

    # ── Repeatedly query "A B D" so the path A -> B -> D fires together ──
    # weaken_rate is set very low (close to no weakening) so we focus on
    # the strengthen effect in this demo.
    learner = HebbianLearner(strengthen_rate=0.1, weaken_rate=0.001)
    print("Learning loop: 20 propagate('A B D') calls with Hebbian update")

    for i in range(20):
        # Build initial activation map from query
        results = engine.propagate("A B D")
        # Extract activation values for the learner
        activation_map = {
            name.lower(): engine._nodes[name.lower()].activation
            for name in results
            if name.lower() in engine._nodes
        }
        learner.learn_from_propagation(engine, activation_map)

    print()
    print("Final salience:")
    print(f"  A -> B  =  {sal('A','B'):.3f}  (should be > 0.5)")
    print(f"  B -> D  =  {sal('B','D'):.3f}  (should be > 0.5)")
    print(f"  A -> C  =  {sal('A','C'):.3f}  (off-path, less change)")
    print(f"  C -> D  =  {sal('C','D'):.3f}  (off-path, less change)")
    print(f"  X -> Y  =  {sal('X','Y'):.3f}  (unrelated, unchanged)")
    print()

    print(f"Learner stats: {learner.stats}")
    print()
    print("Note: confidence values are unchanged. Hebbian only modifies")
    print("salience (retrievability), never confidence (truth value).")
    print("This is the confidence/salience split — a core STG design choice.")


if __name__ == "__main__":
    main()
