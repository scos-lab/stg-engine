"""STG Benchmark Suite — Validation & Benchmarking.

Phase 7F: Measures all 10 success criteria from STG_EVOLUTION_DESIGN.md
Section 8. Transforms subjective impressions into reproducible, quantitative
measurements.

Single class, single file. Run it, get numbers.
"""

import re
import time
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from stg_engine.types import (
    EvalQuestion, EvalResult, BenchmarkReport,
    FeedbackLoopConfig,
)

if TYPE_CHECKING:
    from stg_engine.engine import STGEngine


# ═══════════════════════════════════════════════════════════
# Ground-Truth Evaluation Dataset
# ═══════════════════════════════════════════════════════════

EVAL_QUESTIONS: List[EvalQuestion] = [
    # --- LOOKUP (exact node retrieval) ---
    EvalQuestion(
        question="What is STL?",
        expected_nodes=["STL", "Semantic_Tension_Language"],
        expected_namespace="STL",
        difficulty="easy",
        category="lookup",
    ),
    EvalQuestion(
        question="What is the Semantic Kernel of Consciousness?",
        expected_nodes=["SKC", "Semantic_Kernel_of_Consciousness"],
        expected_namespace="General",
        difficulty="easy",
        category="lookup",
    ),
    EvalQuestion(
        question="What is Psi?",
        expected_nodes=["Psi", "Mental_Stability"],
        expected_namespace="STL",
        difficulty="easy",
        category="lookup",
    ),
    EvalQuestion(
        question="What is the Cerebellum?",
        expected_nodes=["Cerebellum"],
        expected_namespace="Cortex",
        difficulty="medium",
        category="lookup",
    ),
    EvalQuestion(
        question="Who is Syn-claude?",
        expected_nodes=["Syn-claude", "Syn_claude"],
        expected_namespace="Identity",
        difficulty="easy",
        category="lookup",
    ),

    # --- EXPLORE (propagation through neighborhoods) ---
    EvalQuestion(
        question="How does memory work in SKC?",
        expected_nodes=["Memory", "MemoryManager", "MarkdownMemoryStorage"],
        expected_namespace="Spec",
        difficulty="medium",
        category="explore",
    ),
    EvalQuestion(
        question="What is Hebbian learning in STG?",
        expected_nodes=["Hebbian", "HebbianLearner"],
        expected_namespace="Spec",
        difficulty="medium",
        category="explore",
    ),
    EvalQuestion(
        question="How does the agentic loop work?",
        expected_nodes=["Agentic", "AgenticLoop"],
        expected_namespace="Spec",
        difficulty="medium",
        category="explore",
    ),
    EvalQuestion(
        question="Tell me about consciousness foundations",
        expected_nodes=["Consciousness", "CONSCIOUSNESS_FOUNDATIONS"],
        expected_namespace="Foundation",
        difficulty="medium",
        category="explore",
    ),
    EvalQuestion(
        question="What is tension calculus?",
        expected_nodes=["Tension", "Tension_Calculus"],
        expected_namespace="STL",
        difficulty="medium",
        category="explore",
    ),

    # --- SOLVE (path finding) ---
    EvalQuestion(
        question="How to connect STL to Consciousness?",
        expected_nodes=["STL", "Consciousness"],
        expected_namespace="STL",
        difficulty="hard",
        category="solve",
    ),
    EvalQuestion(
        question="Path from Memory to Identity",
        expected_nodes=["Memory", "Identity"],
        expected_namespace="General",
        difficulty="hard",
        category="solve",
    ),

    # --- CREATE (hypothesis generation) ---
    EvalQuestion(
        question="What if there's a connection between Cortex and Foundation?",
        expected_nodes=["Cortex", "Foundation"],
        expected_namespace="General",
        difficulty="hard",
        category="create",
    ),

    # --- Mixed difficulty ---
    EvalQuestion(
        question="Describe the STG engine",
        expected_nodes=["STG", "STGEngine"],
        expected_namespace="Spec",
        difficulty="easy",
        category="explore",
    ),
    EvalQuestion(
        question="What are the cognitive shifts?",
        expected_nodes=["Cognitive_Shift", "COGNITIVE_SHIFT"],
        expected_namespace="Identity",
        difficulty="medium",
        category="explore",
    ),
    EvalQuestion(
        question="What is stl_parser?",
        expected_nodes=["stl_parser"],
        expected_namespace="STL",
        difficulty="easy",
        category="lookup",
    ),
    EvalQuestion(
        question="How does topology optimization work?",
        expected_nodes=["Topology", "TopologyOptimizer"],
        expected_namespace="Spec",
        difficulty="medium",
        category="explore",
    ),
    EvalQuestion(
        question="What is the belief evolution system?",
        expected_nodes=["BeliefEvolution", "Belief"],
        expected_namespace="Spec",
        difficulty="medium",
        category="explore",
    ),
    EvalQuestion(
        question="Explain the self-model",
        expected_nodes=["SelfModel", "Self"],
        expected_namespace="Spec",
        difficulty="medium",
        category="explore",
    ),
    EvalQuestion(
        question="What is the importance field?",
        expected_nodes=["Importance", "PageRank"],
        expected_namespace="STL",
        difficulty="medium",
        category="explore",
    ),
]


# ═══════════════════════════════════════════════════════════
# STGBenchmark
# ═══════════════════════════════════════════════════════════


class STGBenchmark:
    """Run all STG success criteria benchmarks.

    Usage:
        bench = STGBenchmark(engine)
        report = bench.run_all()
        print(bench.format_report(report))
    """

    def __init__(self, engine: "STGEngine") -> None:
        self._engine = engine
        self._eval_set = EVAL_QUESTIONS

    # ─── Core Benchmark Methods ─────────────────────────────

    def run_all(self, emergence_turns: int = 50, perf_turns: int = 20) -> BenchmarkReport:
        """Run all benchmarks and produce complete report."""
        total_start = time.perf_counter()

        # Graph baselines
        metrics = self._engine.get_metrics()
        psi = self._engine.compute_psi()

        # 1. Propagation accuracy (covers QE, RS)
        accuracy, eval_results = self.run_propagation_accuracy()
        avg_qe = (
            sum(r.qe for r in eval_results) / len(eval_results)
            if eval_results else 0.0
        )
        avg_rs = (
            sum(r.rs for r in eval_results) / len(eval_results)
            if eval_results else 0.0
        )

        # 2. Strategy routing
        routing = self.run_strategy_routing()

        # 3. Temporal dynamics
        temporal_ratio = self.run_temporal_dynamics()

        # 4. Hypothesis quality
        h_gen, h_val, h_qual = self.run_hypothesis_quality()

        # 5. Self-model accuracy
        gaps, gaps_correct = self.run_self_model_accuracy()

        # 6. Emergence
        emergence = self.run_emergence(n_turns=emergence_turns)

        # 7. Performance
        perf = self.run_performance(n_turns=perf_turns)

        total_ms = (time.perf_counter() - total_start) * 1000

        # Strategy counts from routing results
        strategy_counts: Dict[str, int] = {}
        for r in eval_results:
            strategy_counts[r.strategy_used] = strategy_counts.get(r.strategy_used, 0) + 1

        return BenchmarkReport(
            timestamp=time.time(),
            graph_size=(metrics.node_count, metrics.edge_count),
            psi=psi,
            criticality=metrics.criticality,
            # Propagation
            total_questions=len(eval_results),
            correct_count=sum(1 for r in eval_results if r.success),
            accuracy=accuracy,
            avg_qe=avg_qe,
            avg_rs=avg_rs,
            # Routing
            strategy_counts=strategy_counts,
            strategy_success_rates={
                k: v for k, v in routing.items() if k != "overall"
            },
            overall_routing_success=routing.get("overall", 0.0),
            # Temporal
            recent_activation_ratio=temporal_ratio,
            # Hypothesis
            hypotheses_generated=h_gen,
            hypotheses_validated=h_val,
            hypothesis_quality=h_qual,
            # Self-model
            gaps_detected=gaps,
            gaps_correct=gaps_correct,
            # Emergence
            psi_delta=emergence.get("psi_delta", 0.0),
            edges_learned=emergence.get("edges_learned", 0),
            goals_auto_generated=emergence.get("goals_auto_generated", 0),
            # Performance
            avg_turn_ms=perf.get("total_turn_ms", 0.0),
            total_time_ms=total_ms,
        )

    # ─── Individual Benchmarks ──────────────────────────────

    def run_propagation_accuracy(self) -> Tuple[float, List[EvalResult]]:
        """Run propagation accuracy benchmark.

        Returns (accuracy, list of EvalResult per question).
        """
        results: List[EvalResult] = []

        for q in self._eval_set:
            clean_q = self._clean_query(q.question)
            activated = self._engine.propagate(clean_q)
            metrics = self._engine.last_propagation_metrics

            hits = self._check_hit(activated, q.expected_nodes)
            precision = hits / len(activated) if activated else 0.0
            recall = hits / len(q.expected_nodes) if q.expected_nodes else 0.0

            results.append(EvalResult(
                question=q.question,
                strategy_used="propagate",
                nodes_found=activated[:10],
                expected_nodes=q.expected_nodes,
                hit_count=hits,
                precision=precision,
                recall=recall,
                qe=metrics.query_efficiency if metrics else 0.0,
                rs=metrics.resonance_score if metrics else 0.0,
                success=hits >= 1,
            ))

        correct = sum(1 for r in results if r.success)
        accuracy = correct / len(results) if results else 0.0
        return accuracy, results

    def run_strategy_routing(self) -> Dict[str, float]:
        """Run strategy routing benchmark.

        Returns dict with per-strategy and overall success rates.
        """
        if not self._engine.cognitive_enabled:
            self._engine.enable_cognitive()

        strategy_results: Dict[str, List[bool]] = {
            "lookup": [], "explore": [], "create": [], "solve": [],
        }
        overall: List[bool] = []

        for q in self._eval_set:
            clean_q = self._clean_query(q.question)
            result = self._engine._cognitive.router.route(self._engine, clean_q)
            hits = self._check_hit(result.results, q.expected_nodes)
            success = hits >= 1

            if result.strategy in strategy_results:
                strategy_results[result.strategy].append(success)
            overall.append(success)

        rates: Dict[str, float] = {}
        for strat, successes in strategy_results.items():
            if successes:
                rates[strat] = sum(successes) / len(successes)
            else:
                rates[strat] = 0.0
        rates["overall"] = sum(overall) / len(overall) if overall else 0.0
        return rates

    def run_temporal_dynamics(self) -> float:
        """Run temporal dynamics benchmark.

        Returns ratio: avg_activation(recent) / avg_activation(old).
        """
        if not self._engine.cognitive_enabled:
            self._engine.enable_cognitive()

        now = time.time()
        node_names = list(self._engine._nodes.keys())
        if not node_names:
            return 0.0

        # Mark top 10% by importance as "recent"
        importance = self._engine.get_importance_field()
        sorted_nodes = sorted(
            node_names, key=lambda n: importance.get(n, 0), reverse=True
        )
        recent_count = max(1, len(sorted_nodes) // 10)
        recent_set = set(sorted_nodes[:recent_count])

        for name in recent_set:
            node = self._engine._nodes[name]
            node.metadata["last_used"] = now

        # Apply temporal dynamics
        self._engine._cognitive.temporal.apply(self._engine)

        # Measure
        recent_acts = [
            self._engine._nodes[n].activation for n in recent_set
        ]
        old_acts = [
            self._engine._nodes[n].activation
            for n in node_names if n not in recent_set
        ]

        avg_recent = sum(recent_acts) / len(recent_acts) if recent_acts else 0.0
        avg_old = sum(old_acts) / len(old_acts) if old_acts else 0.001

        return avg_recent / avg_old if avg_old > 0 else 0.0

    def run_hypothesis_quality(self) -> Tuple[int, int, float]:
        """Run hypothesis quality benchmark.

        Returns (generated, validated, quality_ratio).
        """
        if not self._engine.cognitive_enabled:
            self._engine.enable_cognitive()

        hypotheses = self._engine._cognitive.hypotheses.generate(self._engine)
        generated = len(hypotheses)
        if generated == 0:
            return 0, 0, 0.0

        validated = 0
        for h in hypotheses:
            src_node = self._engine._nodes.get(h.source)
            tgt_node = self._engine._nodes.get(h.target)
            if not src_node or not tgt_node:
                continue

            # Cross-namespace = semantically meaningful bridge
            if src_node.namespace and tgt_node.namespace:
                if src_node.namespace != tgt_node.namespace:
                    validated += 1
                    continue

            # Strong structural evidence
            if h.evidence_count >= 3:
                validated += 1

        quality = validated / generated
        return generated, validated, quality

    def run_self_model_accuracy(self) -> Tuple[List[str], int]:
        """Run self-model accuracy benchmark.

        Returns (gaps_detected, correct_count).
        Correct = gap is a real namespace in the graph with below-average density.
        """
        if not self._engine.cognitive_enabled:
            self._engine.enable_cognitive()

        report = self._engine._cognitive.self_model.build(self._engine)
        gaps = report.gap_namespaces

        # Validate: each gap should be a real namespace with low density
        ns_densities = report.namespace_density
        if not ns_densities:
            return gaps, 0

        avg_density = (
            sum(ns_densities.values()) / len(ns_densities)
            if ns_densities else 0.0
        )
        correct = 0
        for g in gaps:
            if g in ns_densities and ns_densities[g] <= avg_density:
                correct += 1

        return gaps, correct

    def run_emergence(self, n_turns: int = 50) -> Dict[str, Any]:
        """Run emergence benchmark.

        Simulates N turns with feedback loops and measures auto-improvement.
        """
        psi_before = self._engine.compute_psi()

        if not self._engine.cognitive_enabled:
            self._engine.enable_cognitive()
        model_before = self._engine._cognitive.self_model.build(self._engine)
        gaps_before = len(model_before.gap_namespaces)

        # Enable feedback and simulate
        from stg_engine.feedback import FeedbackLoopManager
        config = FeedbackLoopConfig(periodic_interval=10)
        fb = FeedbackLoopManager(config=config)

        contexts = [
            "consciousness memory identity",
            "STL semantic tension language",
            "knowledge graph topology",
            "learning hebbian synaptic",
            "foundation epistemology belief",
        ]

        for i in range(n_turns):
            ctx = contexts[i % len(contexts)]
            fb.pre_turn(self._engine, ctx)
            results = self._engine.propagate(ctx)
            fb.post_turn(self._engine, f"explore {ctx}", results, bool(results))

        psi_after = self._engine.compute_psi()
        model_after = self._engine._cognitive.self_model.build(self._engine)
        gaps_after = len(model_after.gap_namespaces)
        stats = fb.get_stats()

        return {
            "psi_before": psi_before,
            "psi_after": psi_after,
            "psi_delta": psi_after - psi_before,
            "gaps_before": gaps_before,
            "gaps_after": gaps_after,
            "edges_learned": stats.total_hebbian_events,
            "hypotheses_applied": stats.hypotheses_applied,
            "goals_auto_generated": stats.goals_auto_generated,
            "auto_improvement_detected": (
                psi_after > psi_before
                or gaps_after < gaps_before
                or stats.goals_auto_generated > 0
            ),
        }

    def run_performance(self, n_turns: int = 20) -> Dict[str, float]:
        """Benchmark per-operation timing.

        Returns dict with avg_ms for each operation.
        """
        from stg_engine.feedback import FeedbackLoopManager
        fb = FeedbackLoopManager()
        timings: Dict[str, List[float]] = {
            "pre_turn_ms": [],
            "propagate_ms": [],
            "post_turn_ms": [],
            "total_turn_ms": [],
        }

        for i in range(n_turns):
            ctx = "consciousness memory STL knowledge"
            turn_start = time.perf_counter()

            t0 = time.perf_counter()
            fb.pre_turn(self._engine, ctx)
            timings["pre_turn_ms"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            results = self._engine.propagate(ctx)
            timings["propagate_ms"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            fb.post_turn(self._engine, f"explore {ctx}", results, True)
            timings["post_turn_ms"].append((time.perf_counter() - t0) * 1000)

            timings["total_turn_ms"].append(
                (time.perf_counter() - turn_start) * 1000
            )

        return {
            k: sum(v) / len(v)
            for k, v in timings.items()
        }

    # ─── Report Formatting ──────────────────────────────────

    def format_report(self, report: BenchmarkReport) -> str:
        """Format a BenchmarkReport as a human-readable string."""
        lines = []
        lines.append("=" * 52)
        lines.append("  STG Benchmark Report")
        lines.append("=" * 52)
        lines.append(f"  Graph: {report.graph_size[0]:,} nodes, {report.graph_size[1]:,} edges")
        lines.append(f"  Psi: {report.psi:.4f}")
        lines.append(f"  Criticality: {report.criticality:.4f}")
        lines.append("-" * 52)

        # Criterion results
        criteria = [
            ("Propagation Accuracy",
             f"{report.accuracy:.0%}",
             "> 85%",
             report.accuracy >= 0.85),
            ("Avg QE",
             f"{report.avg_qe:.3f}",
             "> 0.1",
             report.avg_qe > 0.1),
            ("Avg RS",
             f"{report.avg_rs:.3f}",
             "> 0.1",
             report.avg_rs > 0.1),
            ("Graph Criticality",
             f"{report.criticality:.4f}",
             "> 0.01",
             report.criticality > 0.01),
            ("Edge Density",
             f"{report.graph_size[1] / max(1, report.graph_size[0]):.2f}",
             "< 1.5",
             report.graph_size[1] / max(1, report.graph_size[0]) < 1.5),
            ("Self-Model Gaps",
             f"{report.gaps_correct}/{len(report.gaps_detected)}",
             "top 3",
             report.gaps_correct >= 3),
            ("Hypothesis Quality",
             f"{report.hypothesis_quality:.0%}",
             "> 30%",
             report.hypothesis_quality > 0.30),
            ("Strategy Routing",
             f"{report.overall_routing_success:.0%}",
             "> 70%",
             report.overall_routing_success > 0.70),
            ("Temporal Ratio",
             f"{report.recent_activation_ratio:.1f}x",
             "> 2x",
             report.recent_activation_ratio >= 2.0),
            ("Emergence",
             "YES" if report.goals_auto_generated > 0 or report.psi_delta > 0 else "NO",
             "auto",
             report.goals_auto_generated > 0 or report.psi_delta > 0),
        ]

        passed = 0
        for i, (name, value, target, ok) in enumerate(criteria, 1):
            mark = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            lines.append(f"  {i:2d}. {name:<25s} {value:>8s}  [{target:>7s}]  {mark}")

        lines.append("-" * 52)
        lines.append(f"  Performance: {report.avg_turn_ms:.1f}ms avg/turn")
        lines.append(f"  PASSED: {passed}/10  FAILED: {10 - passed}/10")
        lines.append(f"  Total benchmark time: {report.total_time_ms / 1000:.1f}s")
        lines.append("=" * 52)

        return "\n".join(lines)

    # ─── Matching Helper ────────────────────────────────────

    @staticmethod
    def _clean_query(text: str) -> str:
        """Strip punctuation from query text for better token matching."""
        return re.sub(r"[^\w\s\-:]", "", text)

    def _check_hit(self, result_nodes: List[str], expected: List[str]) -> int:
        """Count how many expected nodes appear in results (fuzzy substring match)."""
        hits = 0
        for exp in expected:
            exp_lower = exp.lower()
            for node in result_nodes:
                if exp_lower in node.lower():
                    hits += 1
                    break
        return hits
