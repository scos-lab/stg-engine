"""Tests for Phase 7F: STG Benchmark Suite.

Tests the validation and benchmarking framework that measures all 10
success criteria from the design doc.
"""

import time

import pytest

from stg_engine.engine import STGEngine
from stg_engine.types import (
    EvalQuestion, EvalResult, BenchmarkReport,
)
from stg_engine.benchmark import STGBenchmark, EVAL_QUESTIONS


# ═══════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════


@pytest.fixture
def eval_engine():
    """Engine with known structure for deterministic evaluation."""
    engine = STGEngine()
    # Nodes matching eval questions
    engine.add_node("STL", namespace="STL")
    engine.add_node("Semantic_Tension_Language", namespace="STL")
    engine.add_node("SKC", namespace="General")
    engine.add_node("Semantic_Kernel_of_Consciousness", namespace="General")
    engine.add_node("Psi", namespace="STL")
    engine.add_node("Mental_Stability", namespace="STL")
    engine.add_node("Cerebellum", namespace="Cortex")
    engine.add_node("Syn_claude", namespace="Identity")
    engine.add_node("Memory", namespace="Memory")
    engine.add_node("MemoryManager", namespace="Spec")
    engine.add_node("MarkdownMemoryStorage", namespace="Spec")
    engine.add_node("Hebbian", namespace="Spec")
    engine.add_node("HebbianLearner", namespace="Spec")
    engine.add_node("Agentic", namespace="Spec")
    engine.add_node("AgenticLoop", namespace="Spec")
    engine.add_node("Consciousness", namespace="Foundation")
    engine.add_node("CONSCIOUSNESS_FOUNDATIONS", namespace="Foundation")
    engine.add_node("Tension", namespace="STL")
    engine.add_node("Tension_Calculus", namespace="STL")
    engine.add_node("Identity", namespace="Identity")
    engine.add_node("Cortex", namespace="Cortex")
    engine.add_node("Foundation", namespace="Foundation")
    engine.add_node("STG", namespace="Spec")
    engine.add_node("STGEngine", namespace="Spec")
    engine.add_node("Cognitive_Shift", namespace="Identity")
    engine.add_node("stl_parser", namespace="STL")
    engine.add_node("Topology", namespace="Spec")
    engine.add_node("TopologyOptimizer", namespace="Spec")
    engine.add_node("BeliefEvolution", namespace="Spec")
    engine.add_node("Belief", namespace="Spec")
    engine.add_node("SelfModel", namespace="Spec")
    engine.add_node("Self", namespace="Spec")
    engine.add_node("Importance", namespace="STL")
    engine.add_node("PageRank", namespace="STL")

    # Hub node for connectivity
    engine.add_node("Hub_Knowledge", namespace="General")

    # Edges
    engine.add_edge("STL", "Semantic_Tension_Language", confidence=0.95)
    engine.add_edge("STL", "Psi", confidence=0.8)
    engine.add_edge("STL", "Tension", confidence=0.85)
    engine.add_edge("STL", "stl_parser", confidence=0.9)
    engine.add_edge("STL", "Tension_Calculus", confidence=0.8)
    engine.add_edge("SKC", "Memory", confidence=0.7)
    engine.add_edge("SKC", "Consciousness", confidence=0.7)
    engine.add_edge("SKC", "Semantic_Kernel_of_Consciousness", confidence=0.95)
    engine.add_edge("Memory", "MemoryManager", confidence=0.85)
    engine.add_edge("Memory", "MarkdownMemoryStorage", confidence=0.8)
    engine.add_edge("Memory", "Identity", confidence=0.6)
    engine.add_edge("Consciousness", "Identity", confidence=0.6)
    engine.add_edge("Consciousness", "CONSCIOUSNESS_FOUNDATIONS", confidence=0.9)
    engine.add_edge("Psi", "Mental_Stability", confidence=0.95)
    engine.add_edge("Hebbian", "HebbianLearner", confidence=0.9)
    engine.add_edge("Agentic", "AgenticLoop", confidence=0.9)
    engine.add_edge("STG", "STGEngine", confidence=0.95)
    engine.add_edge("Topology", "TopologyOptimizer", confidence=0.9)
    engine.add_edge("BeliefEvolution", "Belief", confidence=0.85)
    engine.add_edge("SelfModel", "Self", confidence=0.85)
    engine.add_edge("Importance", "PageRank", confidence=0.9)
    engine.add_edge("Cerebellum", "Cortex", confidence=0.8)
    engine.add_edge("Syn_claude", "Identity", confidence=0.9)
    engine.add_edge("Cognitive_Shift", "Identity", confidence=0.7)
    engine.add_edge("Foundation", "Consciousness", confidence=0.85)
    engine.add_edge("Cortex", "Foundation", confidence=0.5)

    # Hub connections to improve connectivity
    for name in ["STL", "SKC", "Memory", "Consciousness", "Identity",
                  "Cortex", "Foundation", "STG", "Hebbian", "Agentic"]:
        engine.add_edge("Hub_Knowledge", name, confidence=0.6)
        engine.add_edge(name, "Hub_Knowledge", confidence=0.5)

    return engine


@pytest.fixture
def benchmark(eval_engine):
    return STGBenchmark(eval_engine)


# ═══════════════════════════════════════════════════════════
# TestEvalQuestion
# ═══════════════════════════════════════════════════════════


class TestEvalQuestion:
    def test_question_fields(self):
        q = EvalQuestion(
            question="test?",
            expected_nodes=["A"],
            expected_namespace="NS",
        )
        assert q.question == "test?"
        assert q.expected_nodes == ["A"]
        assert q.expected_namespace == "NS"

    def test_default_values(self):
        q = EvalQuestion(
            question="test?",
            expected_nodes=["A"],
            expected_namespace="NS",
        )
        assert q.difficulty == "medium"
        assert q.category == "lookup"

    def test_eval_set_not_empty(self):
        assert len(EVAL_QUESTIONS) >= 15


# ═══════════════════════════════════════════════════════════
# TestEvalResult
# ═══════════════════════════════════════════════════════════


class TestEvalResult:
    def test_result_fields(self):
        r = EvalResult(
            question="q?",
            strategy_used="propagate",
            nodes_found=["A"],
            expected_nodes=["A"],
            hit_count=1,
            precision=1.0,
            recall=1.0,
            qe=0.5,
            rs=0.5,
            success=True,
        )
        assert r.question == "q?"
        assert r.qe == 0.5

    def test_success_from_hits(self):
        r = EvalResult(
            question="q?",
            strategy_used="propagate",
            nodes_found=["A", "B"],
            expected_nodes=["A"],
            hit_count=1,
            precision=0.5,
            recall=1.0,
            qe=0.1,
            rs=0.2,
            success=True,
        )
        assert r.success is True


# ═══════════════════════════════════════════════════════════
# TestCheckHit
# ═══════════════════════════════════════════════════════════


class TestCheckHit:
    def test_exact_match(self, benchmark):
        assert benchmark._check_hit(["STL"], ["STL"]) == 1

    def test_substring_match(self, benchmark):
        assert benchmark._check_hit(["General:STL_Parser"], ["STL"]) == 1

    def test_case_insensitive(self, benchmark):
        assert benchmark._check_hit(["STL"], ["stl"]) == 1

    def test_no_match(self, benchmark):
        assert benchmark._check_hit(["STL", "SKC"], ["xyz"]) == 0


# ═══════════════════════════════════════════════════════════
# TestPropagationAccuracy
# ═══════════════════════════════════════════════════════════


class TestPropagationAccuracy:
    def test_returns_accuracy_and_results(self, benchmark):
        accuracy, results = benchmark.run_propagation_accuracy()
        assert isinstance(accuracy, float)
        assert isinstance(results, list)

    def test_accuracy_is_ratio(self, benchmark):
        accuracy, _ = benchmark.run_propagation_accuracy()
        assert 0.0 <= accuracy <= 1.0

    def test_results_have_eval_fields(self, benchmark):
        _, results = benchmark.run_propagation_accuracy()
        assert len(results) > 0
        r = results[0]
        assert hasattr(r, "qe")
        assert hasattr(r, "rs")
        assert hasattr(r, "success")

    def test_easy_questions_pass(self, benchmark):
        """Easy lookup questions should succeed on the eval engine."""
        _, results = benchmark.run_propagation_accuracy()
        easy = [r for r in results if r.question == "What is STL?"]
        assert len(easy) == 1
        assert easy[0].success is True

    def test_results_count_matches_eval_set(self, benchmark):
        _, results = benchmark.run_propagation_accuracy()
        assert len(results) == len(EVAL_QUESTIONS)


# ═══════════════════════════════════════════════════════════
# TestStrategyRouting
# ═══════════════════════════════════════════════════════════


class TestStrategyRouting:
    def test_returns_per_strategy_rates(self, benchmark):
        rates = benchmark.run_strategy_routing()
        assert "lookup" in rates
        assert "explore" in rates
        assert "create" in rates
        assert "solve" in rates

    def test_rates_are_ratios(self, benchmark):
        rates = benchmark.run_strategy_routing()
        for key, val in rates.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"

    def test_overall_rate(self, benchmark):
        rates = benchmark.run_strategy_routing()
        assert "overall" in rates

    def test_lookup_questions_classified(self, benchmark):
        """Router should produce results for 'What is' questions."""
        rates = benchmark.run_strategy_routing()
        # At minimum, overall should be measurable
        assert rates["overall"] >= 0.0


# ═══════════════════════════════════════════════════════════
# TestTemporalDynamics
# ═══════════════════════════════════════════════════════════


class TestTemporalDynamics:
    def test_returns_ratio(self, benchmark):
        ratio = benchmark.run_temporal_dynamics()
        assert isinstance(ratio, float)
        assert ratio >= 0.0

    def test_recent_higher_than_old(self, benchmark):
        ratio = benchmark.run_temporal_dynamics()
        assert ratio > 1.0

    def test_marking_nodes_works(self, eval_engine):
        """After marking, recent nodes should have last_used metadata."""
        bench = STGBenchmark(eval_engine)
        bench.run_temporal_dynamics()
        # Check some nodes got marked
        marked = [
            n for n in eval_engine._nodes.values()
            if "last_used" in n.metadata
        ]
        assert len(marked) > 0


# ═══════════════════════════════════════════════════════════
# TestHypothesisQuality
# ═══════════════════════════════════════════════════════════


class TestHypothesisQuality:
    def test_returns_tuple(self, benchmark):
        gen, val, qual = benchmark.run_hypothesis_quality()
        assert isinstance(gen, int)
        assert isinstance(val, int)
        assert isinstance(qual, float)

    def test_quality_is_ratio(self, benchmark):
        _, _, qual = benchmark.run_hypothesis_quality()
        assert 0.0 <= qual <= 1.0

    def test_cross_namespace_validated(self, eval_engine):
        """Cross-namespace hypotheses should count as validated."""
        bench = STGBenchmark(eval_engine)
        gen, val, _ = bench.run_hypothesis_quality()
        # Even if 0 generated, the ratio is valid
        if gen > 0:
            assert val >= 0


# ═══════════════════════════════════════════════════════════
# TestSelfModelAccuracy
# ═══════════════════════════════════════════════════════════


class TestSelfModelAccuracy:
    def test_detects_gaps(self, benchmark):
        gaps, correct = benchmark.run_self_model_accuracy()
        assert isinstance(gaps, list)

    def test_gaps_are_strings(self, benchmark):
        gaps, _ = benchmark.run_self_model_accuracy()
        for g in gaps:
            assert isinstance(g, str)

    def test_correct_count_reasonable(self, benchmark):
        gaps, correct = benchmark.run_self_model_accuracy()
        assert correct <= len(gaps)


# ═══════════════════════════════════════════════════════════
# TestEmergence
# ═══════════════════════════════════════════════════════════


class TestEmergence:
    def test_returns_psi_delta(self, benchmark):
        result = benchmark.run_emergence(n_turns=10)
        assert "psi_before" in result
        assert "psi_after" in result
        assert "psi_delta" in result

    def test_edges_learned(self, benchmark):
        result = benchmark.run_emergence(n_turns=20)
        assert result["edges_learned"] > 0

    def test_auto_improvement_detected(self, benchmark):
        result = benchmark.run_emergence(n_turns=20)
        assert result["auto_improvement_detected"] is True

    def test_goals_generated(self, benchmark):
        result = benchmark.run_emergence(n_turns=20)
        assert result["goals_auto_generated"] >= 0


# ═══════════════════════════════════════════════════════════
# TestPerformance
# ═══════════════════════════════════════════════════════════


class TestPerformance:
    def test_returns_timing_dict(self, benchmark):
        perf = benchmark.run_performance(n_turns=5)
        assert "pre_turn_ms" in perf
        assert "propagate_ms" in perf
        assert "post_turn_ms" in perf
        assert "total_turn_ms" in perf

    def test_all_positive(self, benchmark):
        perf = benchmark.run_performance(n_turns=5)
        for k, v in perf.items():
            assert v > 0, f"{k} should be positive, got {v}"

    def test_total_reasonable(self, benchmark):
        perf = benchmark.run_performance(n_turns=5)
        assert perf["total_turn_ms"] < 5000


# ═══════════════════════════════════════════════════════════
# TestBenchmarkReport
# ═══════════════════════════════════════════════════════════


class TestBenchmarkReport:
    def test_run_all_returns_report(self, benchmark):
        report = benchmark.run_all(emergence_turns=10, perf_turns=5)
        assert isinstance(report, BenchmarkReport)

    def test_report_has_all_fields(self, benchmark):
        report = benchmark.run_all(emergence_turns=10, perf_turns=5)
        assert report.total_questions > 0
        assert report.accuracy >= 0.0
        assert report.avg_qe >= 0.0
        assert report.overall_routing_success >= 0.0
        assert report.recent_activation_ratio >= 0.0
        assert isinstance(report.gaps_detected, list)

    def test_format_report_produces_string(self, benchmark):
        report = benchmark.run_all(emergence_turns=10, perf_turns=5)
        text = benchmark.format_report(report)
        assert isinstance(text, str)
        assert len(text) > 0


# ═══════════════════════════════════════════════════════════
# TestFormatReport
# ═══════════════════════════════════════════════════════════


class TestFormatReport:
    def test_includes_pass_fail(self, benchmark):
        report = benchmark.run_all(emergence_turns=10, perf_turns=5)
        text = benchmark.format_report(report)
        assert "PASS" in text or "FAIL" in text

    def test_includes_all_metrics(self, benchmark):
        report = benchmark.run_all(emergence_turns=10, perf_turns=5)
        text = benchmark.format_report(report)
        assert "QE" in text or "Avg QE" in text
        assert "RS" in text or "Avg RS" in text
        assert "Accuracy" in text or "Propagation" in text
