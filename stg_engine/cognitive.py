"""STG Cognitive Architecture — From Circuit Board to Thinking System.

Phase 7D: Six functional modules that transform a learning graph into
a cognitive system. Each module is independently testable and deployable.

Modules:
  - GoalRegister: Active goals biasing propagation (prefrontal cortex)
  - PredictiveWarmer: Pre-activate nodes from context (thalamic relay)
  - HypothesisGenerator: Link prediction via common neighbors (hippocampus)
  - SelfModel: Knowledge landscape assessment (default mode network)
  - MultiStrategyRouter: Query routing to strategies (thalamus)
  - TemporalDynamics: Time-aware activation (LTP/circadian)
  - CognitiveArchitecture: Facade composing all six modules
"""

import math
import re
import time
from collections import Counter, defaultdict
from typing import (
    Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING,
)

from stg_engine.types import (
    GoalEntry, Hypothesis, SelfModelReport, StrategyResult,
)

if TYPE_CHECKING:
    from stg_engine.engine import STGEngine


# ═══════════════════════════════════════════════════════════
# D.1: Goal Register (Direction)
# ═══════════════════════════════════════════════════════════


class GoalRegister:
    """Active goals biasing STG propagation.

    Brain analogy: Prefrontal cortex maintaining task-relevant
    representations that bias attention and processing.
    Maximum concurrent goals limited (working memory constraint).
    """

    def __init__(
        self,
        max_goals: int = 3,
        default_priority: float = 1.0,
    ) -> None:
        self.max_goals = max(1, max_goals)
        self.default_priority = max(0.5, min(default_priority, 2.0))
        self._goals: Dict[str, GoalEntry] = {}

    def add_goal(
        self,
        name: str,
        keywords: List[str],
        priority: Optional[float] = None,
    ) -> GoalEntry:
        """Register a new active goal.

        If goal with same name exists, updates it.
        If at capacity, evicts the oldest goal.
        """
        if not name or not keywords:
            raise ValueError("Goal name and keywords must be non-empty")

        p = priority if priority is not None else self.default_priority
        p = max(0.5, min(p, 2.0))

        # Update existing
        if name in self._goals:
            self._goals[name].keywords = keywords
            self._goals[name].priority = p
            return self._goals[name]

        # Evict oldest if at capacity
        if len(self._goals) >= self.max_goals:
            oldest = min(self._goals.values(), key=lambda g: g.created_at)
            del self._goals[oldest.name]

        entry = GoalEntry(
            name=name,
            keywords=keywords,
            priority=p,
            created_at=time.time(),
        )
        self._goals[name] = entry
        return entry

    def remove_goal(self, name: str) -> bool:
        """Remove a goal by name. Returns True if removed."""
        if name in self._goals:
            del self._goals[name]
            return True
        return False

    @property
    def current_goals(self) -> List[GoalEntry]:
        """Active goals sorted by priority descending."""
        return sorted(
            self._goals.values(),
            key=lambda g: g.priority,
            reverse=True,
        )

    def compute_bias(self, node_name: str) -> float:
        """Compute goal-bias multiplier for a node.

        Returns 1.0 (no bias) if no goal keywords match.
        Returns up to 1.5 if keywords match active goals.
        """
        if not self._goals:
            return 1.0

        tokens = set(node_name.lower().replace(":", "_").split("_"))
        tokens = {t for t in tokens if len(t) > 1}
        bias = 1.0

        for goal in self._goals.values():
            goal_kws = {k.lower() for k in goal.keywords}
            overlap = tokens & goal_kws
            if overlap:
                bias += goal.priority * 0.15 * len(overlap)

        return min(bias, 1.5)


# ═══════════════════════════════════════════════════════════
# D.2: Predictive Pre-activation (Anticipation)
# ═══════════════════════════════════════════════════════════


class PredictiveWarmer:
    """Pre-activate STG regions from context.

    Brain analogy: Thalamic relay — pre-conscious routing that
    primes relevant cortical areas before conscious processing.
    """

    def __init__(
        self,
        max_keywords: int = 10,
        warmup_factor: float = 0.3,
        include_neighbors: bool = True,
    ) -> None:
        self.max_keywords = max(1, max_keywords)
        self.warmup_factor = max(0.1, min(warmup_factor, 0.8))
        self.include_neighbors = include_neighbors
        self._last_warmed: Set[str] = set()

    def warmup(self, engine: "STGEngine", context: str) -> int:
        """Pre-activate nodes matching context keywords.

        Returns count of nodes warmed.
        """
        keywords = self._extract_keywords(context)
        if not keywords:
            return 0

        warmed: Set[str] = set()

        for kw in keywords:
            for name in engine._nodes:
                if kw in name.lower():
                    warmed.add(name)

        if self.include_neighbors:
            neighbors: Set[str] = set()
            for name in warmed:
                neighbors.update(n.lower() for n in engine.neighbors(name, "out"))
                neighbors.update(n.lower() for n in engine.neighbors(name, "in"))
            warmed.update(neighbors)

        for name in warmed:
            node = engine._nodes.get(name)
            if node:
                node.activation = min(node.activation + self.warmup_factor, 1.0)

        self._last_warmed = set(warmed)
        return len(warmed)

    @property
    def last_warmed_nodes(self) -> Set[str]:
        """Nodes warmed in the last warmup() call."""
        return self._last_warmed

    def _extract_keywords(self, context: str) -> List[str]:
        """Extract top keywords from context text."""
        tokens = re.findall(r'[a-zA-Z_]{3,}', context.lower())
        freq = Counter(tokens)
        return [t for t, _ in freq.most_common(self.max_keywords)]


# ═══════════════════════════════════════════════════════════
# D.3: Hypothesis Generator (Creativity)
# ═══════════════════════════════════════════════════════════


class HypothesisGenerator:
    """Link prediction for missing connections.

    Brain analogy: Hippocampal pattern completion — filling in
    gaps based on partial structural patterns.

    Algorithm: Common neighbors — if A and C share >= N neighbors
    but are not directly connected, hypothesize A→C.
    """

    def __init__(
        self,
        min_common_neighbors: int = 2,
        importance_percentile: float = 0.5,
        max_hypotheses: int = 10,
        max_confidence: float = 0.8,
    ) -> None:
        self.min_common = max(1, min_common_neighbors)
        self.importance_pct = max(0.0, min(importance_percentile, 1.0))
        self.max_hypotheses = max(1, max_hypotheses)
        self.max_confidence = max(0.1, min(max_confidence, 1.0))

    def generate(self, engine: "STGEngine") -> List[Hypothesis]:
        """Scan important nodes for common-neighbor patterns.

        Returns hypotheses sorted by confidence descending,
        capped at max_hypotheses.
        """
        importance = engine.get_importance_field()
        if not importance:
            return []

        # Filter to top N% by importance
        sorted_vals = sorted(importance.values())
        if not sorted_vals:
            return []
        cutoff_idx = int(len(sorted_vals) * (1 - self.importance_pct))
        cutoff_idx = max(0, min(cutoff_idx, len(sorted_vals) - 1))
        threshold = sorted_vals[cutoff_idx]
        important_nodes = {n for n, v in importance.items() if v >= threshold}

        if len(important_nodes) < 2:
            return []

        # Build neighbor sets once
        neighbor_cache: Dict[str, Set[str]] = {}
        for node in important_nodes:
            neighbor_cache[node] = (
                set(engine._graph.successors(node))
                | set(engine._graph.predecessors(node))
            )

        hypotheses: List[Hypothesis] = []
        important_list = sorted(important_nodes)

        for i, node_a in enumerate(important_list):
            for node_c in important_list[i + 1:]:
                # Skip if already connected
                if engine._edges_lookup.get((node_a, node_c)) is not None:
                    continue
                if engine._edges_lookup.get((node_c, node_a)) is not None:
                    continue

                common = neighbor_cache[node_a] & neighbor_cache[node_c]
                if len(common) >= self.min_common:
                    conf = min(len(common) * 0.15, self.max_confidence)
                    hypotheses.append(Hypothesis(
                        source=node_a,
                        target=node_c,
                        confidence=conf,
                        evidence_count=len(common),
                        rationale="common_neighbors",
                        timestamp=time.time(),
                    ))

        hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        return hypotheses[:self.max_hypotheses]

    def apply_hypotheses(
        self,
        engine: "STGEngine",
        hypotheses: List[Hypothesis],
    ) -> int:
        """Commit hypotheses as new edges. Returns count created."""
        created = 0
        for h in hypotheses:
            if engine._edges_lookup.get((h.source.lower(), h.target.lower())) is not None:
                continue
            engine.add_edge(
                h.source, h.target,
                confidence=h.confidence,
                rule="logical",
            )
            created += 1
        return created


# ═══════════════════════════════════════════════════════════
# D.4: Self-Model / Meta-Graph (Reflection)
# ═══════════════════════════════════════════════════════════


class SelfModel:
    """STG self-assessment and gap identification.

    Brain analogy: Default Mode Network (DMN) — the brain's
    self-referential introspection system, active during rest.
    """

    def __init__(
        self,
        top_hubs_count: int = 10,
        fragile_threshold: int = 1,
        max_fragile_reported: int = 20,
    ) -> None:
        self.top_hubs_count = max(1, top_hubs_count)
        self.fragile_threshold = max(0, fragile_threshold)
        self.max_fragile_reported = max(1, max_fragile_reported)

    def build(self, engine: "STGEngine") -> SelfModelReport:
        """Analyze knowledge landscape and produce report."""
        importance = engine.get_importance_field()
        metrics = engine.get_metrics()

        # Namespace density (average importance per namespace)
        ns_vals: Dict[str, List[float]] = defaultdict(list)
        for name, node in engine._nodes.items():
            ns = node.namespace or "?"
            ns_vals[ns].append(importance.get(name, 0.0))
        ns_density = {
            ns: sum(vals) / len(vals) if vals else 0.0
            for ns, vals in ns_vals.items()
        }

        # Connectivity health
        conn_health = metrics.largest_component_ratio

        # Isolation
        fragile = [
            n for n in engine._graph.nodes()
            if engine._graph.degree(n) <= self.fragile_threshold
        ]

        # Cross-namespace integration
        cross = 0
        total = len(engine._edges)
        for e in engine._edges:
            src_node = engine._nodes.get(e.source.lower())
            tgt_node = engine._nodes.get(e.target.lower())
            src_ns = src_node.namespace if src_node else None
            tgt_ns = tgt_node.namespace if tgt_node else None
            if src_ns and tgt_ns and src_ns != tgt_ns:
                cross += 1
        cross_score = cross / total if total > 0 else 0.0

        # Top hubs
        top_hubs = sorted(
            importance.items(), key=lambda x: x[1], reverse=True,
        )[:self.top_hubs_count]

        # Gap detection
        if ns_density:
            sorted_densities = sorted(ns_density.values())
            median_d = sorted_densities[len(sorted_densities) // 2]
            gaps = sorted(ns for ns, d in ns_density.items() if d < median_d)
        else:
            gaps = []

        # Assessment
        assessment = self._generate_assessment(
            conn_health, len(fragile), cross_score, gaps, metrics,
        )

        return SelfModelReport(
            namespace_density=ns_density,
            connectivity_health=conn_health,
            isolation_count=len(fragile),
            cross_namespace_score=cross_score,
            top_hubs=top_hubs,
            gap_namespaces=gaps,
            fragile_nodes=fragile[:self.max_fragile_reported],
            assessment=assessment,
            timestamp=time.time(),
        )

    def _generate_assessment(
        self,
        connectivity: float,
        isolation_count: int,
        cross_score: float,
        gaps: List[str],
        metrics: Any,
    ) -> str:
        """Build human-readable assessment string."""
        lines: List[str] = []

        # Connectivity
        if connectivity >= 0.95:
            lines.append(f"Connectivity: {connectivity:.0%} (excellent)")
        elif connectivity >= 0.8:
            lines.append(f"Connectivity: {connectivity:.0%} (healthy)")
        else:
            lines.append(
                f"WARNING: Low connectivity ({connectivity:.0%}). "
                "Consider adding bridge edges."
            )

        # Isolation
        total = metrics.node_count or 1
        frag_pct = isolation_count / total
        if frag_pct > 0.05:
            lines.append(
                f"WARNING: {isolation_count} fragile nodes "
                f"({frag_pct:.1%} of graph, degree <= 1)."
            )

        # Integration
        if cross_score < 0.3:
            lines.append(
                f"WARNING: Low cross-namespace integration ({cross_score:.0%}). "
                "Knowledge silos detected."
            )
        else:
            lines.append(f"Cross-namespace integration: {cross_score:.0%}")

        # Gaps
        if gaps:
            lines.append(f"Knowledge gaps: {', '.join(gaps)}")

        return " | ".join(lines)


# ═══════════════════════════════════════════════════════════
# D.5: Multi-Strategy Router (Adaptation)
# ═══════════════════════════════════════════════════════════


# Strategy trigger patterns
_STRATEGY_TRIGGERS: Dict[str, List[str]] = {
    "lookup": ["what is", "define", "who is", "describe", "what are"],
    "explore": ["how does", "related to", "context of", "explain", "tell me about"],
    "create": ["what if", "could", "imagine", "suggest", "hypothesize"],
    "solve": ["how to", "path from", "connect", "bridge", "solve"],
}


class MultiStrategyRouter:
    """Query routing hub.

    Brain analogy: Thalamus — routes different types of input
    to the appropriate processing regions.

    Four strategies: lookup, explore, create, solve.
    """

    def __init__(self) -> None:
        self._stats: Dict[str, Dict[str, int]] = {
            s: {"total": 0, "successes": 0}
            for s in _STRATEGY_TRIGGERS
        }

    def route(
        self,
        engine: "STGEngine",
        query: str,
    ) -> StrategyResult:
        """Classify query and execute selected strategy.

        Defaults to 'explore' if no trigger matched.
        """
        strategy = self._classify(query)
        self._stats[strategy]["total"] += 1

        if strategy == "lookup":
            results = self._exec_lookup(engine, query)
        elif strategy == "explore":
            results = self._exec_explore(engine, query)
        elif strategy == "create":
            results = self._exec_create(engine, query)
        elif strategy == "solve":
            results = self._exec_solve(engine, query)
        else:
            results = self._exec_explore(engine, query)

        return StrategyResult(
            strategy=strategy,
            query=query,
            results=results,
            confidence=0.5 if not results else 0.8,
        )

    def report_success(self, strategy: str, success: bool) -> None:
        """Record strategy success/failure for tracking."""
        if strategy in self._stats:
            if success:
                self._stats[strategy]["successes"] += 1

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Per-strategy success rate statistics."""
        result: Dict[str, Dict[str, Any]] = {}
        for s, data in self._stats.items():
            total = data["total"]
            successes = data["successes"]
            result[s] = {
                "total": total,
                "successes": successes,
                "rate": successes / total if total > 0 else 0.0,
            }
        return result

    def _classify(self, query: str) -> str:
        """Match query against trigger patterns."""
        q = query.lower()
        for strategy, triggers in _STRATEGY_TRIGGERS.items():
            for trigger in triggers:
                if trigger in q:
                    return strategy
        return "explore"

    def _exec_lookup(self, engine: "STGEngine", query: str) -> List[str]:
        """Direct node query with propagation fallback."""
        words = self._content_words(query)
        results: List[str] = []
        for w in words:
            nodes = engine.query_nodes(name_pattern=w, limit=5)
            for n in nodes:
                if n.qualified_name not in results:
                    results.append(n.qualified_name)
        # Fallback: propagation if direct lookup found too few
        if len(results) < 3:
            activated = engine.propagate(query)
            for a in activated[:10]:
                if a not in results:
                    results.append(a)
        return results[:10]

    def _exec_explore(self, engine: "STGEngine", query: str) -> List[str]:
        """Propagation-based exploration."""
        activated = engine.propagate(query)
        return activated[:10]

    def _exec_create(self, engine: "STGEngine", query: str) -> List[str]:
        """Hypothesis generation + propagation context."""
        # First get query-relevant nodes via propagation
        activated = engine.propagate(query)
        results = list(activated[:5])
        # Then add hypotheses
        gen = HypothesisGenerator(max_hypotheses=5)
        hypotheses = gen.generate(engine)
        for h in hypotheses:
            entry = f"{h.source}"
            if entry not in results:
                results.append(entry)
            entry2 = f"{h.target}"
            if entry2 not in results:
                results.append(entry2)
        return results[:10]

    def _exec_solve(self, engine: "STGEngine", query: str) -> List[str]:
        """Path finding between concepts."""
        words = self._content_words(query)
        if len(words) >= 2:
            # Try to find actual nodes matching the words
            src_name = self._resolve_node(engine, words[0])
            tgt_name = self._resolve_node(engine, words[-1])
            if src_name and tgt_name:
                paths = engine.find_paths(src_name, tgt_name, max_depth=5)
                if paths:
                    return [" -> ".join(p) for p in paths[:5]]
        # Fallback to propagation
        activated = engine.propagate(query)
        return activated[:10]

    @staticmethod
    def _content_words(query: str) -> List[str]:
        """Extract content words from query, filtering stop words."""
        _stop = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "am", "do", "does", "did", "has", "have", "had", "it", "its",
            "what", "who", "how", "why", "when", "where", "which",
            "to", "of", "in", "on", "at", "by", "for", "with", "from",
            "and", "or", "not", "no", "if", "but", "so", "as", "than",
            "me", "my", "we", "us", "you", "he", "she", "they", "them",
            "this", "that", "these", "those", "about", "tell", "describe",
            "explain", "can", "could", "would", "should", "will",
            "theres",
        }
        words = re.findall(r'[A-Za-z_:]{2,}', query)
        content = [w for w in words if w.lower() not in _stop]
        return content if content else words

    @staticmethod
    def _resolve_node(engine: "STGEngine", word: str) -> Optional[str]:
        """Try to find a real node matching this word."""
        nodes = engine.query_nodes(name_pattern=word, limit=1)
        return nodes[0].qualified_name if nodes else None


# ═══════════════════════════════════════════════════════════
# D.6: Temporal Dynamics (Time Awareness)
# ═══════════════════════════════════════════════════════════


class TemporalDynamics:
    """Time-aware activation management.

    Brain analogy: Long-term potentiation + synaptic depression
    combined with circadian activity cycles.

    Formula:
      activation = permanence_floor + (1 - permanence_floor) * recency * confidence

    Where:
      recency = exp(-time_since_last_access * ln(2) / half_life)
      permanence_floor = clamp(importance * permanence_scale, min, max)
    """

    def __init__(
        self,
        half_life_days: float = 7.0,
        permanence_scale: float = 1.0,
        min_permanence: float = 0.01,
        max_permanence: float = 0.8,
    ) -> None:
        self.half_life_days = max(0.0, half_life_days)
        self.permanence_scale = max(0.0, permanence_scale)
        self.min_permanence = max(0.0, min(min_permanence, 1.0))
        self.max_permanence = max(self.min_permanence, min(max_permanence, 1.0))

    def apply(self, engine: "STGEngine") -> int:
        """Apply temporal dynamics to all nodes.

        Sets activation based on recency, importance-derived permanence,
        and average edge confidence. Called at session start.
        Returns count of nodes updated.
        """
        importance = engine.get_importance_field()
        now = time.time()
        updated = 0

        for name, node in engine._nodes.items():
            # Permanence from importance
            imp = importance.get(name, 0.0)
            permanence = max(
                self.min_permanence,
                min(imp * self.permanence_scale, self.max_permanence),
            )

            # Recency from last_used
            last_used = self._get_last_used(engine, name)
            recency = self.compute_recency(last_used, now)

            # Average confidence of connected edges
            avg_conf = self._avg_confidence(engine, name)

            # Formula
            activation = permanence + (1 - permanence) * recency * avg_conf
            node.activation = max(0.0, min(activation, 1.0))
            updated += 1

        return updated

    def compute_recency(
        self,
        last_used: float,
        now: Optional[float] = None,
    ) -> float:
        """Pure function: exponential recency decay.

        Returns float in [0.0, 1.0].
        """
        if now is None:
            now = time.time()

        if last_used <= 0:
            return 0.15  # Low default — no usage evidence = low recency

        half_life_seconds = self.half_life_days * 86400.0
        if half_life_seconds <= 0:
            return 0.5  # Avoid division by zero

        elapsed = max(0.0, now - last_used)
        return math.exp(-elapsed * math.log(2) / half_life_seconds)

    def _get_last_used(self, engine: "STGEngine", name: str) -> float:
        """Get last_used timestamp for a node.

        Checks edges for last_used, node metadata, falls back to 0.
        """
        best = 0.0

        # Check edges touching this node
        for edge in engine._edges:
            if (edge.source == name or edge.target == name) and edge.last_used:
                if edge.last_used > best:
                    best = edge.last_used

        # Check node metadata
        node = engine._nodes.get(name)
        if node:
            meta_ts = node.metadata.get("last_used", 0.0)
            if isinstance(meta_ts, (int, float)) and meta_ts > best:
                best = meta_ts

        return best

    def _avg_confidence(self, engine: "STGEngine", name: str) -> float:
        """Average effective weight (confidence × salience) of edges touching this node."""
        weights: List[float] = []
        for edge in engine._edges:
            if edge.source == name or edge.target == name:
                weights.append(edge.confidence * edge.salience)
        return sum(weights) / len(weights) if weights else 0.5


# ═══════════════════════════════════════════════════════════
# Cognitive Architecture (Facade)
# ═══════════════════════════════════════════════════════════


class CognitiveArchitecture:
    """Facade over six cognitive modules.

    Provides high-level operations that coordinate multiple modules.
    Single entry point for engine integration.
    """

    def __init__(
        self,
        max_goals: int = 3,
        warmup_factor: float = 0.3,
        max_hypotheses: int = 10,
        half_life_days: float = 7.0,
    ) -> None:
        self.goals = GoalRegister(max_goals=max_goals)
        self.warmer = PredictiveWarmer(warmup_factor=warmup_factor)
        self.hypotheses = HypothesisGenerator(max_hypotheses=max_hypotheses)
        self.self_model = SelfModel()
        self.router = MultiStrategyRouter()
        self.temporal = TemporalDynamics(half_life_days=half_life_days)

    def pre_turn(
        self,
        engine: "STGEngine",
        context: str,
    ) -> Dict[str, Any]:
        """Called at turn start.

        Applies temporal dynamics and warms context-relevant nodes.
        Returns summary dict.
        """
        temporal_updates = self.temporal.apply(engine)
        warmup_count = self.warmer.warmup(engine, context)

        return {
            "temporal_updates": temporal_updates,
            "warmup_count": warmup_count,
            "active_goals": len(self.goals.current_goals),
        }

    def route_query(
        self,
        engine: "STGEngine",
        query: str,
    ) -> StrategyResult:
        """Route a query through the MultiStrategyRouter."""
        return self.router.route(engine, query)

    def reflect(self, engine: "STGEngine") -> SelfModelReport:
        """Build self-model. Intended for periodic use."""
        return self.self_model.build(engine)
