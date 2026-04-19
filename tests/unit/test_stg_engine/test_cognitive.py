"""Tests for Phase 7D: Cognitive Architecture.

Tests GoalRegister, PredictiveWarmer, HypothesisGenerator,
SelfModel, MultiStrategyRouter, TemporalDynamics, and
CognitiveArchitecture facade + engine integration.
"""

import math
import time

import pytest

from stg_engine.engine import STGEngine
from stg_engine.types import (
    GoalEntry, Hypothesis, SelfModelReport, StrategyResult,
)
from stg_engine.cognitive import (
    GoalRegister,
    PredictiveWarmer,
    HypothesisGenerator,
    SelfModel,
    MultiStrategyRouter,
    TemporalDynamics,
    CognitiveArchitecture,
)


# ═══════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════


@pytest.fixture
def goal_register():
    return GoalRegister()


@pytest.fixture
def warmer():
    return PredictiveWarmer()


@pytest.fixture
def hypothesis_gen():
    return HypothesisGenerator()


@pytest.fixture
def self_model():
    return SelfModel()


@pytest.fixture
def router():
    return MultiStrategyRouter()


@pytest.fixture
def temporal():
    return TemporalDynamics()


@pytest.fixture
def cognitive_engine():
    """Engine with ~20 nodes across 3 namespaces.

    Alpha: A1-A6 (dense triangle cluster)
    Beta:  B1-B6 (dense triangle cluster)
    Gamma: G1-G4 (sparse chain)
    Hubs:  Hub1-Hub4 connecting clusters
    """
    engine = STGEngine()
    # Alpha cluster
    for i in range(1, 7):
        engine.add_node(f"Alpha_{i}", namespace="Alpha", anchor_type="Concept")
    engine.add_edge("Alpha_1", "Alpha_2", confidence=0.9)
    engine.add_edge("Alpha_2", "Alpha_3", confidence=0.9)
    engine.add_edge("Alpha_3", "Alpha_1", confidence=0.9)
    engine.add_edge("Alpha_4", "Alpha_5", confidence=0.8)
    engine.add_edge("Alpha_5", "Alpha_6", confidence=0.8)
    engine.add_edge("Alpha_6", "Alpha_4", confidence=0.8)
    engine.add_edge("Alpha_1", "Alpha_4", confidence=0.7)

    # Beta cluster
    for i in range(1, 7):
        engine.add_node(f"Beta_{i}", namespace="Beta", anchor_type="Concept")
    engine.add_edge("Beta_1", "Beta_2", confidence=0.9)
    engine.add_edge("Beta_2", "Beta_3", confidence=0.9)
    engine.add_edge("Beta_3", "Beta_1", confidence=0.9)
    engine.add_edge("Beta_4", "Beta_5", confidence=0.8)
    engine.add_edge("Beta_5", "Beta_6", confidence=0.8)
    engine.add_edge("Beta_6", "Beta_4", confidence=0.8)
    engine.add_edge("Beta_1", "Beta_4", confidence=0.7)

    # Gamma (sparse)
    for i in range(1, 5):
        engine.add_node(f"Gamma_{i}", namespace="Gamma", anchor_type="Concept")
    engine.add_edge("Gamma_1", "Gamma_2", confidence=0.6)
    engine.add_edge("Gamma_2", "Gamma_3", confidence=0.6)
    engine.add_edge("Gamma_3", "Gamma_4", confidence=0.6)

    # Hub nodes connecting clusters
    engine.add_node("Hub_Memory", namespace="Alpha", anchor_type="Concept")
    engine.add_node("Hub_Self", namespace="Beta", anchor_type="Concept")
    engine.add_node("Hub_Knowledge", namespace="Gamma", anchor_type="Concept")
    engine.add_node("Hub_Consciousness", namespace="Alpha", anchor_type="Concept")

    engine.add_edge("Hub_Memory", "Alpha_1", confidence=0.8)
    engine.add_edge("Hub_Memory", "Beta_1", confidence=0.7)
    engine.add_edge("Hub_Self", "Beta_1", confidence=0.8)
    engine.add_edge("Hub_Self", "Gamma_1", confidence=0.6)
    engine.add_edge("Hub_Knowledge", "Gamma_1", confidence=0.8)
    engine.add_edge("Hub_Knowledge", "Alpha_3", confidence=0.6)
    engine.add_edge("Hub_Consciousness", "Hub_Memory", confidence=0.9)
    engine.add_edge("Hub_Consciousness", "Hub_Self", confidence=0.9)
    engine.add_edge("Hub_Consciousness", "Hub_Knowledge", confidence=0.8)

    return engine


@pytest.fixture
def rich_engine():
    """Larger engine with ~30 nodes for hypothesis testing.

    Creates a graph where common neighbors pattern is detectable.
    Nodes A, B share neighbors N1, N2, N3 but are NOT connected.
    """
    engine = STGEngine()
    # Core nodes
    engine.add_node("NodeA", namespace="Core", anchor_type="Concept")
    engine.add_node("NodeB", namespace="Core", anchor_type="Concept")
    # Shared neighbors
    for i in range(1, 4):
        engine.add_node(f"Shared_{i}", namespace="Core", anchor_type="Concept")
        engine.add_edge("NodeA", f"Shared_{i}", confidence=0.8)
        engine.add_edge("NodeB", f"Shared_{i}", confidence=0.8)
    # Additional nodes to fill graph
    for i in range(1, 20):
        engine.add_node(f"Filler_{i}", namespace="Fill", anchor_type="Concept")
        # Connect to random core nodes
        if i % 3 == 0:
            engine.add_edge(f"Filler_{i}", "NodeA", confidence=0.5)
        if i % 4 == 0:
            engine.add_edge(f"Filler_{i}", "NodeB", confidence=0.5)
        if i > 1:
            engine.add_edge(f"Filler_{i-1}", f"Filler_{i}", confidence=0.4)
    return engine


# ═══════════════════════════════════════════════════════════
# TestGoalRegisterInit
# ═══════════════════════════════════════════════════════════


class TestGoalRegisterInit:
    def test_default_parameters(self, goal_register):
        assert goal_register.max_goals == 3
        assert goal_register.default_priority == 1.0

    def test_custom_parameters(self):
        g = GoalRegister(max_goals=5, default_priority=1.5)
        assert g.max_goals == 5
        assert g.default_priority == 1.5

    def test_starts_empty(self, goal_register):
        assert len(goal_register.current_goals) == 0


# ═══════════════════════════════════════════════════════════
# TestGoalRegisterAddRemove
# ═══════════════════════════════════════════════════════════


class TestGoalRegisterAddRemove:
    def test_add_goal_returns_entry(self, goal_register):
        entry = goal_register.add_goal("test", ["memory", "recall"])
        assert isinstance(entry, GoalEntry)
        assert entry.name == "test"
        assert entry.keywords == ["memory", "recall"]

    def test_add_goal_stores(self, goal_register):
        goal_register.add_goal("test", ["memory"])
        assert len(goal_register.current_goals) == 1

    def test_add_duplicate_updates(self, goal_register):
        goal_register.add_goal("test", ["old"])
        goal_register.add_goal("test", ["new"], priority=1.5)
        goals = goal_register.current_goals
        assert len(goals) == 1
        assert goals[0].keywords == ["new"]
        assert goals[0].priority == 1.5

    def test_add_over_capacity_evicts(self, goal_register):
        goal_register.add_goal("g1", ["a"])
        goal_register.add_goal("g2", ["b"])
        goal_register.add_goal("g3", ["c"])
        goal_register.add_goal("g4", ["d"])
        goals = goal_register.current_goals
        assert len(goals) == 3
        names = {g.name for g in goals}
        assert "g1" not in names  # oldest evicted

    def test_remove_existing(self, goal_register):
        goal_register.add_goal("test", ["a"])
        assert goal_register.remove_goal("test") is True
        assert len(goal_register.current_goals) == 0

    def test_remove_nonexistent(self, goal_register):
        assert goal_register.remove_goal("nope") is False

    def test_current_goals_sorted(self, goal_register):
        goal_register.add_goal("low", ["a"], priority=0.5)
        goal_register.add_goal("high", ["b"], priority=2.0)
        goal_register.add_goal("mid", ["c"], priority=1.0)
        goals = goal_register.current_goals
        assert goals[0].name == "high"
        assert goals[-1].name == "low"

    def test_goal_timestamp_set(self, goal_register):
        before = time.time()
        entry = goal_register.add_goal("test", ["a"])
        assert entry.created_at >= before


# ═══════════════════════════════════════════════════════════
# TestGoalRegisterBias
# ═══════════════════════════════════════════════════════════


class TestGoalRegisterBias:
    def test_no_goals_no_bias(self, goal_register):
        assert goal_register.compute_bias("Memory_Architecture") == 1.0

    def test_matching_keyword_increases_bias(self, goal_register):
        goal_register.add_goal("mem", ["memory"])
        bias = goal_register.compute_bias("Hub_Memory")
        assert bias > 1.0

    def test_non_matching_no_bias(self, goal_register):
        goal_register.add_goal("mem", ["memory"])
        bias = goal_register.compute_bias("Beta_Process")
        assert bias == 1.0

    def test_multiple_goals_stack(self, goal_register):
        goal_register.add_goal("g1", ["alpha"])
        goal_register.add_goal("g2", ["alpha"])
        bias = goal_register.compute_bias("Alpha_Node")
        assert bias > 1.15  # two goals matching

    def test_bias_capped_at_1_5(self, goal_register):
        goal_register.add_goal("g1", ["node"], priority=2.0)
        goal_register.add_goal("g2", ["node"], priority=2.0)
        goal_register.add_goal("g3", ["node"], priority=2.0)
        bias = goal_register.compute_bias("Node_Test")
        assert bias <= 1.5


# ═══════════════════════════════════════════════════════════
# TestPredictiveWarmerInit
# ═══════════════════════════════════════════════════════════


class TestPredictiveWarmerInit:
    def test_default_parameters(self, warmer):
        assert warmer.max_keywords == 10
        assert warmer.warmup_factor == 0.3

    def test_custom_parameters(self):
        w = PredictiveWarmer(max_keywords=5, warmup_factor=0.5)
        assert w.max_keywords == 5
        assert w.warmup_factor == 0.5

    def test_include_neighbors_default(self, warmer):
        assert warmer.include_neighbors is True


# ═══════════════════════════════════════════════════════════
# TestPredictiveWarmerWarmup
# ═══════════════════════════════════════════════════════════


class TestPredictiveWarmerWarmup:
    def test_warms_matching_nodes(self, warmer, cognitive_engine):
        count = warmer.warmup(cognitive_engine, "alpha cluster analysis")
        assert count > 0
        # At least one Alpha node should have activation > 0
        alpha_acts = [
            cognitive_engine._nodes[n].activation
            for n in cognitive_engine._nodes
            if "alpha" in n.lower()
        ]
        assert any(a > 0 for a in alpha_acts)

    def test_warms_neighbors(self, warmer, cognitive_engine):
        # Hub_Memory is neighbor of Alpha_1
        count_with = warmer.warmup(cognitive_engine, "hub memory concept")
        assert count_with > 1  # Should warm hub + neighbors

    def test_no_neighbors_flag(self, cognitive_engine):
        w = PredictiveWarmer(include_neighbors=False)
        count = w.warmup(cognitive_engine, "hub memory concept")
        # Without neighbors, only direct matches
        w2 = PredictiveWarmer(include_neighbors=True)
        # Reset activations
        for n in cognitive_engine._nodes.values():
            n.activation = 0.0
        count2 = w2.warmup(cognitive_engine, "hub memory concept")
        assert count2 >= count

    def test_returns_count(self, warmer, cognitive_engine):
        count = warmer.warmup(cognitive_engine, "alpha node")
        assert isinstance(count, int)
        assert count >= 0

    def test_activation_capped(self, warmer, cognitive_engine):
        # Set high initial activation
        for n in cognitive_engine._nodes.values():
            n.activation = 0.9
        warmer.warmup(cognitive_engine, "alpha beta gamma hub")
        for n in cognitive_engine._nodes.values():
            assert n.activation <= 1.0

    def test_no_match_zero_warmed(self, warmer, cognitive_engine):
        count = warmer.warmup(cognitive_engine, "xyzzy qwerty nonsense")
        assert count == 0


# ═══════════════════════════════════════════════════════════
# TestHypothesisGeneratorInit
# ═══════════════════════════════════════════════════════════


class TestHypothesisGeneratorInit:
    def test_default_parameters(self, hypothesis_gen):
        assert hypothesis_gen.min_common == 2
        assert hypothesis_gen.importance_pct == 0.5
        assert hypothesis_gen.max_hypotheses == 10
        assert hypothesis_gen.max_confidence == 0.8

    def test_custom_parameters(self):
        h = HypothesisGenerator(
            min_common_neighbors=3,
            importance_percentile=0.3,
            max_hypotheses=5,
            max_confidence=0.6,
        )
        assert h.min_common == 3
        assert h.max_hypotheses == 5
        assert h.max_confidence == 0.6

    def test_confidence_cap_clamped(self):
        h = HypothesisGenerator(max_confidence=1.5)
        assert h.max_confidence == 1.0


# ═══════════════════════════════════════════════════════════
# TestHypothesisGenerate
# ═══════════════════════════════════════════════════════════


class TestHypothesisGenerate:
    def test_finds_common_neighbor_hypotheses(self, rich_engine):
        gen = HypothesisGenerator(
            min_common_neighbors=2,
            importance_percentile=0.9,  # include most nodes
        )
        hypotheses = gen.generate(rich_engine)
        # NodeA and NodeB share 3 neighbors, should get hypothesis
        pairs = {(h.source, h.target) for h in hypotheses}
        pairs |= {(h.target, h.source) for h in hypotheses}
        assert ("nodea", "nodeb") in pairs or ("nodeb", "nodea") in pairs

    def test_skips_existing_edges(self):
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.9)
        engine.add_edge("A", "C", confidence=0.9)
        engine.add_edge("B", "C", confidence=0.9)  # Already connected
        gen = HypothesisGenerator(min_common_neighbors=1, importance_percentile=1.0)
        hypotheses = gen.generate(engine)
        # All pairs already connected
        assert len(hypotheses) == 0

    def test_respects_importance_filter(self, rich_engine):
        # With strict filter, fewer hypotheses
        strict = HypothesisGenerator(importance_percentile=0.1)
        h_strict = strict.generate(rich_engine)
        loose = HypothesisGenerator(importance_percentile=0.9)
        h_loose = loose.generate(rich_engine)
        assert len(h_strict) <= len(h_loose)

    def test_confidence_formula(self, rich_engine):
        gen = HypothesisGenerator(
            min_common_neighbors=2,
            importance_percentile=0.9,
            max_confidence=0.8,
        )
        hypotheses = gen.generate(rich_engine)
        for h in hypotheses:
            expected_max = min(h.evidence_count * 0.15, 0.8)
            assert h.confidence <= expected_max + 0.001

    def test_sorted_by_confidence(self, rich_engine):
        gen = HypothesisGenerator(importance_percentile=0.9)
        hypotheses = gen.generate(rich_engine)
        if len(hypotheses) > 1:
            for i in range(len(hypotheses) - 1):
                assert hypotheses[i].confidence >= hypotheses[i + 1].confidence

    def test_capped_at_max(self, rich_engine):
        gen = HypothesisGenerator(
            max_hypotheses=2,
            importance_percentile=0.9,
        )
        hypotheses = gen.generate(rich_engine)
        assert len(hypotheses) <= 2

    def test_empty_graph_returns_empty(self):
        engine = STGEngine()
        gen = HypothesisGenerator()
        assert gen.generate(engine) == []


# ═══════════════════════════════════════════════════════════
# TestHypothesisApply
# ═══════════════════════════════════════════════════════════


class TestHypothesisApply:
    def test_creates_edges(self):
        engine = STGEngine()
        engine.add_node("A")
        engine.add_node("B")
        gen = HypothesisGenerator()
        hypotheses = [
            Hypothesis("A", "B", 0.6, 3, "common_neighbors"),
        ]
        created = gen.apply_hypotheses(engine, hypotheses)
        assert created == 1
        assert engine._edges_lookup.get(("a", "b")) is not None

    def test_skips_existing(self):
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.9)
        gen = HypothesisGenerator()
        hypotheses = [
            Hypothesis("A", "B", 0.6, 3, "common_neighbors"),
        ]
        created = gen.apply_hypotheses(engine, hypotheses)
        assert created == 0

    def test_returns_count(self):
        engine = STGEngine()
        engine.add_node("A")
        engine.add_node("B")
        engine.add_node("C")
        gen = HypothesisGenerator()
        hypotheses = [
            Hypothesis("A", "B", 0.5, 2, "common_neighbors"),
            Hypothesis("B", "C", 0.4, 2, "common_neighbors"),
        ]
        created = gen.apply_hypotheses(engine, hypotheses)
        assert created == 2


# ═══════════════════════════════════════════════════════════
# TestSelfModelInit
# ═══════════════════════════════════════════════════════════


class TestSelfModelInit:
    def test_default_parameters(self, self_model):
        assert self_model.top_hubs_count == 10
        assert self_model.fragile_threshold == 1

    def test_custom_parameters(self):
        s = SelfModel(top_hubs_count=5, fragile_threshold=2)
        assert s.top_hubs_count == 5
        assert s.fragile_threshold == 2


# ═══════════════════════════════════════════════════════════
# TestSelfModelBuild
# ═══════════════════════════════════════════════════════════


class TestSelfModelBuild:
    def test_namespace_density_computed(self, self_model, cognitive_engine):
        report = self_model.build(cognitive_engine)
        assert len(report.namespace_density) > 0
        assert "Alpha" in report.namespace_density
        assert "Beta" in report.namespace_density

    def test_connectivity_health(self, self_model, cognitive_engine):
        report = self_model.build(cognitive_engine)
        assert 0.0 <= report.connectivity_health <= 1.0

    def test_isolation_count(self, self_model, cognitive_engine):
        report = self_model.build(cognitive_engine)
        assert isinstance(report.isolation_count, int)
        assert report.isolation_count >= 0

    def test_cross_namespace_score(self, self_model, cognitive_engine):
        report = self_model.build(cognitive_engine)
        assert 0.0 <= report.cross_namespace_score <= 1.0
        # Should have some cross-namespace edges (hub connections)
        assert report.cross_namespace_score > 0

    def test_top_hubs_populated(self, self_model, cognitive_engine):
        report = self_model.build(cognitive_engine)
        assert len(report.top_hubs) > 0
        # Each hub is (name, importance_score)
        name, score = report.top_hubs[0]
        assert isinstance(name, str)
        assert isinstance(score, float)

    def test_gap_detection(self, self_model, cognitive_engine):
        report = self_model.build(cognitive_engine)
        # Should detect gaps (some namespaces below median)
        assert isinstance(report.gap_namespaces, list)

    def test_assessment_generated(self, self_model, cognitive_engine):
        report = self_model.build(cognitive_engine)
        assert len(report.assessment) > 0


# ═══════════════════════════════════════════════════════════
# TestMultiStrategyRouterInit
# ═══════════════════════════════════════════════════════════


class TestMultiStrategyRouterInit:
    def test_default_strategies(self, router):
        stats = router.get_stats()
        assert set(stats.keys()) == {"lookup", "explore", "create", "solve"}

    def test_empty_stats(self, router):
        stats = router.get_stats()
        for s in stats.values():
            assert s["total"] == 0


# ═══════════════════════════════════════════════════════════
# TestMultiStrategyRouterRoute
# ═══════════════════════════════════════════════════════════


class TestMultiStrategyRouterRoute:
    def test_what_is_routes_to_lookup(self, router, cognitive_engine):
        result = router.route(cognitive_engine, "what is memory architecture")
        assert result.strategy == "lookup"

    def test_how_does_routes_to_explore(self, router, cognitive_engine):
        result = router.route(cognitive_engine, "how does consciousness work")
        assert result.strategy == "explore"

    def test_what_if_routes_to_create(self, router, cognitive_engine):
        result = router.route(cognitive_engine, "what if we connect alpha to gamma")
        assert result.strategy == "create"

    def test_path_from_routes_to_solve(self, router, cognitive_engine):
        result = router.route(cognitive_engine, "path from Alpha_1 to Beta_1")
        assert result.strategy == "solve"

    def test_unknown_defaults_to_explore(self, router, cognitive_engine):
        result = router.route(cognitive_engine, "tell me something random")
        assert result.strategy == "explore"

    def test_returns_strategy_result(self, router, cognitive_engine):
        result = router.route(cognitive_engine, "what is Hub_Memory")
        assert isinstance(result, StrategyResult)
        assert result.query == "what is Hub_Memory"


# ═══════════════════════════════════════════════════════════
# TestRouterStats
# ═══════════════════════════════════════════════════════════


class TestRouterStats:
    def test_report_success(self, router):
        router._stats["lookup"]["total"] = 1
        router.report_success("lookup", True)
        assert router._stats["lookup"]["successes"] == 1

    def test_report_failure(self, router):
        router._stats["lookup"]["total"] = 1
        router.report_success("lookup", False)
        assert router._stats["lookup"]["successes"] == 0

    def test_success_rate(self, router):
        router._stats["explore"]["total"] = 10
        router._stats["explore"]["successes"] = 7
        stats = router.get_stats()
        assert abs(stats["explore"]["rate"] - 0.7) < 0.01


# ═══════════════════════════════════════════════════════════
# TestTemporalDynamicsInit
# ═══════════════════════════════════════════════════════════


class TestTemporalDynamicsInit:
    def test_default_parameters(self, temporal):
        assert temporal.half_life_days == 7.0
        assert temporal.permanence_scale == 1.0

    def test_custom_parameters(self):
        t = TemporalDynamics(half_life_days=14.0, permanence_scale=2.0)
        assert t.half_life_days == 14.0
        assert t.permanence_scale == 2.0

    def test_min_max_permanence(self, temporal):
        assert temporal.min_permanence == 0.01
        assert temporal.max_permanence == 0.8


# ═══════════════════════════════════════════════════════════
# TestTemporalDynamicsApply
# ═══════════════════════════════════════════════════════════


class TestTemporalDynamicsApply:
    def test_updates_all_nodes(self, temporal, cognitive_engine):
        count = temporal.apply(cognitive_engine)
        assert count == len(cognitive_engine._nodes)

    def test_recent_nodes_higher_activation(self, temporal):
        engine = STGEngine()
        now = time.time()
        # Pass last_used as direct kwargs (add_node uses **metadata)
        engine.add_node("Recent", last_used=now)
        engine.add_node("Old", last_used=now - 86400 * 60)
        # Add edges so both nodes have similar importance
        engine.add_edge("Recent", "Old", confidence=0.8)
        engine.add_edge("Old", "Recent", confidence=0.8)
        temporal.apply(engine)
        assert engine._nodes["recent"].activation > engine._nodes["old"].activation

    def test_important_nodes_have_floor(self, cognitive_engine):
        # Hub_Consciousness has high degree → high importance → high permanence floor
        t = TemporalDynamics(permanence_scale=50.0)  # Amplify importance
        t.apply(cognitive_engine)
        hub = cognitive_engine._nodes["hub_consciousness"]
        assert hub.activation >= 0.01  # At least min permanence

    def test_activation_bounded(self, temporal, cognitive_engine):
        temporal.apply(cognitive_engine)
        for node in cognitive_engine._nodes.values():
            assert 0.0 <= node.activation <= 1.0

    def test_no_timestamp_uses_default(self, temporal):
        engine = STGEngine()
        engine.add_node("NoTime")
        engine.add_edge("NoTime", "NoTime", confidence=0.5)
        temporal.apply(engine)
        # Should use default recency=0.5, not crash
        assert engine._nodes["notime"].activation > 0

    def test_zero_half_life_handled(self):
        t = TemporalDynamics(half_life_days=0.0)
        engine = STGEngine()
        engine.add_node("X")
        engine.add_edge("X", "X", confidence=0.5)
        # Should not crash
        t.apply(engine)
        assert 0.0 <= engine._nodes["x"].activation <= 1.0


# ═══════════════════════════════════════════════════════════
# TestTemporalRecency
# ═══════════════════════════════════════════════════════════


class TestTemporalRecency:
    def test_just_used_recency_near_1(self, temporal):
        now = time.time()
        r = temporal.compute_recency(now, now)
        assert abs(r - 1.0) < 0.01

    def test_half_life_recency_near_half(self, temporal):
        now = time.time()
        half_life_ago = now - 7 * 86400  # 7 days
        r = temporal.compute_recency(half_life_ago, now)
        assert abs(r - 0.5) < 0.05

    def test_old_node_low_recency(self, temporal):
        now = time.time()
        very_old = now - 365 * 86400  # 1 year
        r = temporal.compute_recency(very_old, now)
        assert r < 0.01

    def test_pure_function(self, temporal):
        now = time.time()
        r1 = temporal.compute_recency(now - 86400, now)
        r2 = temporal.compute_recency(now - 86400, now)
        assert r1 == r2


# ═══════════════════════════════════════════════════════════
# TestCognitiveArchitectureInit
# ═══════════════════════════════════════════════════════════


class TestCognitiveArchitectureInit:
    def test_creates_all_modules(self):
        ca = CognitiveArchitecture()
        assert isinstance(ca.goals, GoalRegister)
        assert isinstance(ca.warmer, PredictiveWarmer)
        assert isinstance(ca.hypotheses, HypothesisGenerator)
        assert isinstance(ca.self_model, SelfModel)
        assert isinstance(ca.router, MultiStrategyRouter)
        assert isinstance(ca.temporal, TemporalDynamics)

    def test_custom_parameters_forwarded(self):
        ca = CognitiveArchitecture(
            max_goals=5,
            warmup_factor=0.5,
            max_hypotheses=20,
            half_life_days=14.0,
        )
        assert ca.goals.max_goals == 5
        assert ca.warmer.warmup_factor == 0.5
        assert ca.hypotheses.max_hypotheses == 20
        assert ca.temporal.half_life_days == 14.0

    def test_modules_accessible(self):
        ca = CognitiveArchitecture()
        assert hasattr(ca, "goals")
        assert hasattr(ca, "warmer")
        assert hasattr(ca, "hypotheses")
        assert hasattr(ca, "self_model")
        assert hasattr(ca, "router")
        assert hasattr(ca, "temporal")


# ═══════════════════════════════════════════════════════════
# TestCognitivePreTurn
# ═══════════════════════════════════════════════════════════


class TestCognitivePreTurn:
    def test_applies_temporal(self, cognitive_engine):
        ca = CognitiveArchitecture()
        result = ca.pre_turn(cognitive_engine, "some context")
        assert result["temporal_updates"] == len(cognitive_engine._nodes)

    def test_warms_context(self, cognitive_engine):
        ca = CognitiveArchitecture()
        result = ca.pre_turn(cognitive_engine, "alpha hub memory")
        assert result["warmup_count"] > 0

    def test_returns_summary(self, cognitive_engine):
        ca = CognitiveArchitecture()
        result = ca.pre_turn(cognitive_engine, "test context")
        assert "temporal_updates" in result
        assert "warmup_count" in result
        assert "active_goals" in result


# ═══════════════════════════════════════════════════════════
# TestCognitiveReflect
# ═══════════════════════════════════════════════════════════


class TestCognitiveReflect:
    def test_builds_self_model(self, cognitive_engine):
        ca = CognitiveArchitecture()
        report = ca.reflect(cognitive_engine)
        assert isinstance(report, SelfModelReport)

    def test_report_has_assessment(self, cognitive_engine):
        ca = CognitiveArchitecture()
        report = ca.reflect(cognitive_engine)
        assert len(report.assessment) > 0

    def test_report_has_hubs(self, cognitive_engine):
        ca = CognitiveArchitecture()
        report = ca.reflect(cognitive_engine)
        assert len(report.top_hubs) > 0


# ═══════════════════════════════════════════════════════════
# TestEngineIntegration
# ═══════════════════════════════════════════════════════════


class TestEngineIntegration:
    def test_enable_cognitive(self):
        engine = STGEngine()
        assert not engine.cognitive_enabled
        engine.enable_cognitive()
        assert engine.cognitive_enabled

    def test_add_goal_auto_enables(self):
        engine = STGEngine()
        assert not engine.cognitive_enabled
        engine.add_goal("test", ["memory"])
        assert engine.cognitive_enabled

    def test_get_self_model_works(self, cognitive_engine):
        report = cognitive_engine.get_self_model()
        assert isinstance(report, SelfModelReport)
        assert report.connectivity_health > 0

    def test_generate_hypotheses_works(self, cognitive_engine):
        result = cognitive_engine.generate_hypotheses()
        assert isinstance(result, list)

    def test_apply_temporal_works(self, cognitive_engine):
        count = cognitive_engine.apply_temporal()
        assert count == len(cognitive_engine._nodes)
