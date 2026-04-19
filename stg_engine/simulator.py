"""STG Session Simulator — Accelerated parameter tuning through simulated usage.

Simulates N sessions on a copy of the real graph to observe long-term effects
of Hebbian learning, pruning, and temporal decay parameters.

Usage:
    simulator = SessionSimulator(engine, queries=QUERIES)
    report = simulator.run(num_sessions=50, params={"strengthen_rate": 0.05})
    simulator.print_report(report)

The simulator NEVER modifies the original engine — it works on a deep copy.
"""

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from stg_engine.learning import HebbianLearner, SynapticPruner


# ─── Query Definitions ──────────────────────────────────────

@dataclass
class SimQuery:
    """A simulated query with frequency and expected targets."""
    text: str
    expected: List[str]
    frequency: str = "high"  # "high" = every session, "low" = every 5th


# Default queries derived from benchmark + real usage patterns
DEFAULT_QUERIES: List[SimQuery] = [
    # High frequency — core concepts queried almost every session
    SimQuery("STG engine", ["STG", "STG_Engine", "STGEngine"], "high"),
    SimQuery("What is STL?", ["STL", "Semantic_Tension_Language"], "high"),
    SimQuery("Hebbian learning", ["Hebbian", "HebbianLearner"], "high"),
    SimQuery("propagation activation", ["Propagation", "Activation"], "high"),
    SimQuery("memory system", ["Memory", "MemoryManager"], "high"),
    SimQuery("consciousness foundations", ["Consciousness", "CONSCIOUSNESS_FOUNDATIONS"], "high"),
    # Medium frequency — queried regularly but not every session
    SimQuery("tension calculus", ["Tension", "Tension_Calculus"], "medium"),
    SimQuery("topology optimization", ["Topology", "TopologyOptimizer"], "medium"),
    SimQuery("belief evolution", ["BeliefEvolution", "Belief"], "medium"),
    SimQuery("agentic loop", ["Agentic", "AgenticLoop"], "medium"),
    SimQuery("Who is Syn-claude?", ["Syn-claude", "Syn_claude"], "medium"),
    SimQuery("importance PageRank", ["Importance", "PageRank"], "medium"),
    # Low frequency — rarely queried
    SimQuery("cognitive shifts", ["Cognitive_Shift", "COGNITIVE_SHIFT"], "low"),
    SimQuery("self-model", ["SelfModel", "Self"], "low"),
    SimQuery("What is Psi?", ["Psi", "Mental_Stability"], "low"),
    SimQuery("Cerebellum cortex", ["Cerebellum"], "low"),
    SimQuery("stl_parser", ["stl_parser"], "low"),
]


# ─── Metrics Snapshot ────────────────────────────────────────

@dataclass
class SessionSnapshot:
    """Metrics captured at a single point in the simulation."""
    session_num: int
    node_count: int
    edge_count: int
    edges_pruned_total: int
    nodes_pruned_total: int
    avg_salience_high_freq: float  # average salience of high-freq query paths
    avg_salience_low_freq: float   # average salience of low-freq query paths
    propagation_accuracy: float    # fraction of expected nodes found
    qe: float                     # query efficiency from metrics
    rs: float                     # resonance score


@dataclass
class SimulationReport:
    """Full report from a simulation run."""
    params: Dict[str, Any]
    num_sessions: int
    snapshots: List[SessionSnapshot]
    elapsed_ms: float
    # Summary stats
    final_node_count: int = 0
    final_edge_count: int = 0
    total_pruned_edges: int = 0
    total_pruned_nodes: int = 0
    final_accuracy: float = 0.0
    high_freq_salience_trend: List[float] = field(default_factory=list)
    low_freq_salience_trend: List[float] = field(default_factory=list)


# ─── Simulator ───────────────────────────────────────────────

class SessionSimulator:
    """Simulate N sessions of STG usage to observe parameter effects.

    Works on a deep copy of the engine. The original is never modified.
    """

    def __init__(
        self,
        engine,
        queries: Optional[List[SimQuery]] = None,
    ) -> None:
        self._original_engine = engine
        self._queries = queries or DEFAULT_QUERIES

    def run(
        self,
        num_sessions: int = 50,
        params: Optional[Dict[str, Any]] = None,
        snapshot_interval: int = 5,
        prune_interval: int = 7,
        days_per_session: float = 1.0,
    ) -> SimulationReport:
        """Run the simulation.

        Args:
            num_sessions: Number of sessions to simulate
            params: Parameter overrides. Keys:
                - strengthen_rate (float): Hebbian strengthen rate
                - weaken_rate (float): Hebbian weaken rate
                - activation_threshold (float): Min activation for learning
                - prune_salience_threshold (float): Prune edges below this
                - prune_unused_days (float): Prune after N days unused
                - eid_safety_threshold (float): EID bridge protection
            snapshot_interval: Record metrics every N sessions
            prune_interval: Run pruning every N sessions
            days_per_session: Simulated days between sessions

        Returns:
            SimulationReport with all snapshots and summary.
        """
        t0 = time.perf_counter()
        params = params or {}

        # Deep copy the engine
        engine = copy.deepcopy(self._original_engine)

        # Create learner and pruner with params
        learner = HebbianLearner(
            strengthen_rate=params.get("strengthen_rate", 0.05),
            weaken_rate=params.get("weaken_rate", 0.01),
            activation_threshold=params.get("activation_threshold", 0.1),
            weaken_activation_threshold=params.get("weaken_activation_threshold", 0.3),
        )
        pruner = SynapticPruner(
            confidence_threshold=params.get("prune_salience_threshold", 0.1),
            unused_days=params.get("prune_unused_days", 30.0),
            eid_safety_threshold=params.get("eid_safety_threshold", 0.01),
        )

        snapshots: List[SessionSnapshot] = []
        total_pruned_edges = 0
        total_pruned_nodes = 0
        sim_time = time.time()  # simulated clock

        # Initial snapshot
        snapshots.append(self._take_snapshot(
            engine, 0, total_pruned_edges, total_pruned_nodes,
        ))

        for session in range(1, num_sessions + 1):
            # Advance simulated time
            sim_time += days_per_session * 86400

            # Select queries for this session
            active_queries = self._select_queries(session)

            # Run each query: propagate + learn
            for q in active_queries:
                activated_names = engine.propagate(q.text)

                # Build activation_map from node state (propagate stores it)
                activation_map = {
                    name: node.activation
                    for name, node in engine._nodes.items()
                    if node.activation > 0
                }
                learner.learn_from_propagation(engine, activation_map)

                # Update last_used timestamps to simulated time
                for edge in engine._edges:
                    if edge.last_used and edge.last_used > sim_time - 86400:
                        edge.last_used = sim_time

            # Age all edges: set last_used relative to sim_time for untouched edges
            # (This happens naturally — edges not touched keep their old last_used)

            # Periodic pruning
            if session % prune_interval == 0:
                events = pruner.prune(engine)
                pe = sum(1 for e in events if e.event_type in ("prune", "prune_virtual"))
                pn = sum(1 for e in events if e.event_type == "prune_orphan")
                total_pruned_edges += pe
                total_pruned_nodes += pn

            # Snapshot
            if session % snapshot_interval == 0 or session == num_sessions:
                snapshots.append(self._take_snapshot(
                    engine, session, total_pruned_edges, total_pruned_nodes,
                ))

        elapsed = (time.perf_counter() - t0) * 1000

        # Build report
        report = SimulationReport(
            params=params,
            num_sessions=num_sessions,
            snapshots=snapshots,
            elapsed_ms=elapsed,
            final_node_count=snapshots[-1].node_count,
            final_edge_count=snapshots[-1].edge_count,
            total_pruned_edges=total_pruned_edges,
            total_pruned_nodes=total_pruned_nodes,
            final_accuracy=snapshots[-1].propagation_accuracy,
            high_freq_salience_trend=[s.avg_salience_high_freq for s in snapshots],
            low_freq_salience_trend=[s.avg_salience_low_freq for s in snapshots],
        )

        return report

    def run_calibrated(
        self,
        stg_path: str,
        num_sessions: int = 50,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SimulationReport:
        """Run simulation using calibrated queries from real telemetry data.

        Falls back to DEFAULT_QUERIES if no telemetry data is available.

        Args:
            stg_path: Path to .stg file with telemetry data
            num_sessions: Number of sessions to simulate
            params: Parameter overrides
            **kwargs: Passed to run()

        Returns:
            SimulationReport
        """
        from stg_engine.telemetry import generate_calibrated_queries

        calibrated = generate_calibrated_queries(stg_path)
        if calibrated:
            self._queries = [
                SimQuery(text=q["text"], expected=q["expected"], frequency=q["frequency"])
                for q in calibrated
            ]

        return self.run(num_sessions=num_sessions, params=params, **kwargs)

    def compare(
        self,
        param_name: str,
        values: List[Any],
        num_sessions: int = 50,
        base_params: Optional[Dict[str, Any]] = None,
    ) -> List[SimulationReport]:
        """Run multiple simulations varying one parameter.

        Args:
            param_name: Parameter to vary
            values: List of values to test
            num_sessions: Sessions per run
            base_params: Base parameters (other params held constant)

        Returns:
            List of SimulationReports, one per value.
        """
        base = base_params or {}
        reports = []

        for val in values:
            params = {**base, param_name: val}
            report = self.run(num_sessions=num_sessions, params=params)
            reports.append(report)

        return reports

    def _select_queries(self, session_num: int) -> List[SimQuery]:
        """Select queries based on frequency and session number."""
        active = []
        for q in self._queries:
            if q.frequency == "high":
                active.append(q)
            elif q.frequency == "medium" and session_num % 3 == 0:
                active.append(q)
            elif q.frequency == "low" and session_num % 5 == 0:
                active.append(q)
        return active

    def _take_snapshot(
        self,
        engine,
        session_num: int,
        pruned_edges: int,
        pruned_nodes: int,
    ) -> SessionSnapshot:
        """Capture current graph metrics."""
        stats = engine.get_stats()

        # Measure salience of high-freq and low-freq query paths
        high_sal = self._measure_path_salience(engine, "high")
        low_sal = self._measure_path_salience(engine, "low")

        # Measure propagation accuracy
        accuracy = self._measure_accuracy(engine)

        # Get QE and RS from a representative query
        qe, rs = self._measure_qe_rs(engine)

        return SessionSnapshot(
            session_num=session_num,
            node_count=stats["node_count"],
            edge_count=stats["edge_count"],
            edges_pruned_total=pruned_edges,
            nodes_pruned_total=pruned_nodes,
            avg_salience_high_freq=high_sal,
            avg_salience_low_freq=low_sal,
            propagation_accuracy=accuracy,
            qe=qe,
            rs=rs,
        )

    def _measure_path_salience(self, engine, frequency: str) -> float:
        """Average salience of edges activated by queries of given frequency."""
        queries = [q for q in self._queries if q.frequency == frequency]
        if not queries:
            return 0.0

        saliences = []
        for q in queries:
            for expected_name in q.expected:
                # Find edges touching expected nodes
                for edge in engine._edges:
                    if (expected_name.lower() in edge.source.lower() or
                            expected_name.lower() in edge.target.lower()):
                        saliences.append(edge.salience)
                        break  # one per expected node

        return sum(saliences) / len(saliences) if saliences else 0.0

    def _measure_accuracy(self, engine) -> float:
        """Fraction of benchmark queries that find at least one expected node."""
        hits = 0
        total = 0
        for q in self._queries:
            total += 1
            activated_names = engine.propagate(q.text)
            found = any(
                any(exp.lower() in node.lower() for node in activated_names)
                for exp in q.expected
            )
            if found:
                hits += 1
        return hits / total if total else 0.0

    def _measure_qe_rs(self, engine) -> Tuple[float, float]:
        """Get QE and RS from the last propagation metrics."""
        engine.propagate("STG engine knowledge")
        metrics = getattr(engine, "_last_propagation_metrics", None)
        if metrics is None:
            return 0.0, 0.0
        return metrics.query_efficiency, metrics.resonance_score


# ─── Report Formatting ───────────────────────────────────────

def print_report(report: SimulationReport) -> None:
    """Print a human-readable simulation report."""
    print(f"\n{'='*70}")
    print(f"STG Session Simulation Report")
    print(f"{'='*70}")
    print(f"Sessions: {report.num_sessions}  |  Elapsed: {report.elapsed_ms:.0f}ms")
    print(f"Parameters: {report.params or '(defaults)'}")
    print()

    # Summary
    s0 = report.snapshots[0]
    sf = report.snapshots[-1]
    print(f"  {'':>12}  {'Start':>10}  {'End':>10}  {'Delta':>10}")
    print(f"  {'Nodes':>12}  {s0.node_count:>10}  {sf.node_count:>10}  {sf.node_count - s0.node_count:>+10}")
    print(f"  {'Edges':>12}  {s0.edge_count:>10}  {sf.edge_count:>10}  {sf.edge_count - s0.edge_count:>+10}")
    print(f"  {'Accuracy':>12}  {s0.propagation_accuracy:>10.1%}  {sf.propagation_accuracy:>10.1%}  {sf.propagation_accuracy - s0.propagation_accuracy:>+10.1%}")
    print(f"  {'Hi-freq sal':>12}  {s0.avg_salience_high_freq:>10.4f}  {sf.avg_salience_high_freq:>10.4f}  {sf.avg_salience_high_freq - s0.avg_salience_high_freq:>+10.4f}")
    print(f"  {'Lo-freq sal':>12}  {s0.avg_salience_low_freq:>10.4f}  {sf.avg_salience_low_freq:>10.4f}  {sf.avg_salience_low_freq - s0.avg_salience_low_freq:>+10.4f}")
    print(f"  {'Pruned edges':>12}  {'':>10}  {report.total_pruned_edges:>10}")
    print(f"  {'Pruned nodes':>12}  {'':>10}  {report.total_pruned_nodes:>10}")
    print()

    # Timeline
    print(f"  {'Session':>8}  {'Nodes':>6}  {'Edges':>6}  {'Acc':>6}  {'HiSal':>7}  {'LoSal':>7}  {'QE':>6}  {'RS':>6}  {'Pruned':>6}")
    print(f"  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*6}")
    for s in report.snapshots:
        print(
            f"  {s.session_num:>8}  {s.node_count:>6}  {s.edge_count:>6}  "
            f"{s.propagation_accuracy:>5.0%}  {s.avg_salience_high_freq:>7.4f}  "
            f"{s.avg_salience_low_freq:>7.4f}  {s.qe:>5.3f}  {s.rs:>5.3f}  "
            f"{s.edges_pruned_total:>6}"
        )
    print()


def print_comparison(param_name: str, values: List, reports: List[SimulationReport]) -> None:
    """Print a comparison table across parameter values."""
    print(f"\n{'='*70}")
    print(f"Parameter Sweep: {param_name}")
    print(f"{'='*70}")
    print(
        f"  {'Value':>10}  {'Nodes':>6}  {'Edges':>6}  {'Acc':>6}  "
        f"{'HiSal':>7}  {'LoSal':>7}  {'Pruned':>7}  {'Time':>8}"
    )
    print(
        f"  {'─'*10}  {'─'*6}  {'─'*6}  {'─'*6}  "
        f"{'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}"
    )
    for val, r in zip(values, reports):
        sf = r.snapshots[-1]
        print(
            f"  {str(val):>10}  {sf.node_count:>6}  {sf.edge_count:>6}  "
            f"{sf.propagation_accuracy:>5.0%}  {sf.avg_salience_high_freq:>7.4f}  "
            f"{sf.avg_salience_low_freq:>7.4f}  {r.total_pruned_edges:>7}  "
            f"{r.elapsed_ms:>7.0f}ms"
        )

    # Recommendation
    best_idx = max(
        range(len(reports)),
        key=lambda i: (
            reports[i].final_accuracy,
            reports[i].snapshots[-1].avg_salience_high_freq
            - reports[i].snapshots[-1].avg_salience_low_freq,  # differentiation
        )
    )
    print(f"\n  Recommended: {param_name}={values[best_idx]} "
          f"(best accuracy + salience differentiation)")
    print()
