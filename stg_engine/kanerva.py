"""STG Kanerva Extensions — From Static Graph to Adaptive Memory.

Phase 8: Three extensions grounded in Kanerva's SDM theory (1988),
validated by triangulation with Eliasmith's SPA (2013).

  - IterativePropagator (F5): Iterative convergence for precise retrieval
  - PreferenceFunction (F7): Utility tracking + temporal discount backprop
  - ConflictDetector (F6): Third-edge distance-based conflict detection

Each component is independently usable and testable.
"""

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from stg_engine.types import ConvergenceResult, ConflictReport

if TYPE_CHECKING:
    from stg_engine.engine import STGEngine


# ═══════════════════════════════════════════════════════════
# F5: Iterative Propagation Convergence
# ═══════════════════════════════════════════════════════════


class IterativePropagator:
    """Kanerva F5: Iterative propagation until convergence.

    Feeds propagation output back as input, refining until the top-k
    activated nodes stabilize. Mathematically equivalent to SDM's
    iterative read: approximate query -> read -> refine -> converge
    to stored pattern.

    Convergence condition: d(W(z), target) < d(z, target)
    Kanerva proves convergence when initial distance < n/3.
    """

    def __init__(
        self,
        top_k: int = 5,
        max_iterations: int = 5,
        convergence_threshold: float = 0.8,
    ) -> None:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
        if not 0.0 <= convergence_threshold <= 1.0:
            raise ValueError(
                f"convergence_threshold must be in [0, 1], got {convergence_threshold}"
            )
        self.top_k = top_k
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    def converge(
        self,
        engine: "STGEngine",
        input_text: str,
        **propagate_kwargs,
    ) -> ConvergenceResult:
        """Run iterative propagation until convergence.

        Algorithm:
          1. results = engine.propagate(input_text)
          2. top = set(results[:top_k])
          3. new_query = " ".join(top)
          4. new_results = engine.propagate(new_query)
          5. new_top = set(new_results[:top_k])
          6. jaccard = |top & new_top| / |top | new_top|
          7. if jaccard >= threshold: converged
          8. else: top = new_top, goto 3

        Returns:
            ConvergenceResult with final top nodes and convergence info
        """
        # First propagation from original input
        results = engine.propagate(input_text, **propagate_kwargs)

        if not results:
            return ConvergenceResult(
                top_nodes=[],
                iterations_used=0,
                converged=True,
                stability_history=[],
            )

        current_top = set(results[:self.top_k])
        stability_history: List[float] = []

        for i in range(self.max_iterations):
            # Build new query from current top nodes
            new_query = " ".join(current_top)
            new_results = engine.propagate(new_query, **propagate_kwargs)

            if not new_results:
                # Propagation returned nothing — treat as converged
                stability_history.append(0.0)
                return ConvergenceResult(
                    top_nodes=sorted(current_top),
                    iterations_used=i + 1,
                    converged=True,
                    stability_history=stability_history,
                )

            new_top = set(new_results[:self.top_k])

            # Jaccard similarity
            intersection = len(current_top & new_top)
            union = len(current_top | new_top)
            jaccard = intersection / union if union > 0 else 1.0
            stability_history.append(jaccard)

            if jaccard >= self.convergence_threshold:
                # Use the new results as final (they're the refined version)
                return ConvergenceResult(
                    top_nodes=new_results[:self.top_k],
                    iterations_used=i + 1,
                    converged=True,
                    stability_history=stability_history,
                )

            current_top = new_top

        # Hit max iterations without convergence
        return ConvergenceResult(
            top_nodes=sorted(current_top),
            iterations_used=self.max_iterations,
            converged=False,
            stability_history=stability_history,
        )


# ═══════════════════════════════════════════════════════════
# F7: Preference Function
# ═══════════════════════════════════════════════════════════


class PreferenceFunction:
    """Kanerva F7: Preference extension for STG edges.

    Tracks utility of edges — which paths through the graph lead to
    successful outcomes? Reward propagates backward along paths with
    temporal discounting, like TD(0) learning.

    Formula: pref(s_t) += gamma^(T-t) * reward_scale * reward(s_T)

    Preference is orthogonal to confidence:
      - confidence = epistemics (how certain is this relation?)
      - preference = pragmatics (how useful is this relation?)
    """

    def __init__(
        self,
        gamma: float = 0.9,
        reward_scale: float = 0.1,
        decay_rate: float = 0.01,
    ) -> None:
        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        if reward_scale <= 0.0:
            raise ValueError(f"reward_scale must be > 0, got {reward_scale}")
        if not 0.0 <= decay_rate < 1.0:
            raise ValueError(f"decay_rate must be in [0, 1), got {decay_rate}")
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.decay_rate = decay_rate

        # Stats
        self._total_rewards: int = 0
        self._total_penalties: int = 0
        self._total_decays: int = 0

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "total_rewards": self._total_rewards,
            "total_penalties": self._total_penalties,
            "total_decays": self._total_decays,
        }

    def reward_path(
        self,
        engine: "STGEngine",
        path: List[str],
        reward: float = 1.0,
    ) -> int:
        """Apply positive reward along a path of nodes.

        For path [A, B, C, D] with reward R:
          edge(C,D).preference += reward_scale * R * gamma^0
          edge(B,C).preference += reward_scale * R * gamma^1
          edge(A,B).preference += reward_scale * R * gamma^2

        Reward flows backward: terminal edge gets full reward,
        earlier edges get discounted reward.

        Returns:
            Number of edges updated
        """
        return self._apply_along_path(engine, path, reward)

    def penalize_path(
        self,
        engine: "STGEngine",
        path: List[str],
        penalty: float = 1.0,
    ) -> int:
        """Apply negative reward (penalty) along a path.

        Returns:
            Number of edges updated
        """
        return self._apply_along_path(engine, path, -abs(penalty))

    def _apply_along_path(
        self,
        engine: "STGEngine",
        path: List[str],
        reward: float,
    ) -> int:
        if len(path) < 2:
            return 0

        updated = 0
        # Walk backward: last edge gets gamma^0, second-to-last gets gamma^1, etc.
        edges_reversed = []
        for i in range(len(path) - 1):
            src, tgt = path[i].lower(), path[i + 1].lower()
            edge = engine._edges_lookup.get((src, tgt))
            if edge is not None:
                edges_reversed.append(edge)

        for step, edge in enumerate(reversed(edges_reversed)):
            discount = self.gamma ** step
            delta = self.reward_scale * reward * discount
            edge.preference += delta
            updated += 1

        if reward > 0:
            self._total_rewards += updated
        else:
            self._total_penalties += updated

        return updated

    def decay_preferences(
        self,
        engine: "STGEngine",
    ) -> int:
        """Decay all preferences toward zero.

        Called at session-end. Each edge's preference shrinks:
        pref *= (1 - decay_rate).

        Returns:
            Number of edges affected (preference != 0 before decay)
        """
        affected = 0
        for edge in engine._edges:
            if edge.preference != 0.0:
                edge.preference *= (1.0 - self.decay_rate)
                # Snap to zero if tiny
                if abs(edge.preference) < 1e-6:
                    edge.preference = 0.0
                affected += 1
        self._total_decays += affected
        return affected

    def get_top_preferred(
        self,
        engine: "STGEngine",
        top_n: int = 20,
    ) -> List[Tuple[str, str, float]]:
        """Return top-N edges by absolute preference value.

        Returns:
            List of (source, target, preference) sorted by |preference| desc
        """
        edges_with_pref = [
            (e.source, e.target, e.preference)
            for e in engine._edges
            if e.preference != 0.0
        ]
        edges_with_pref.sort(key=lambda x: abs(x[2]), reverse=True)
        return edges_with_pref[:top_n]


# ═══════════════════════════════════════════════════════════
# F6: Conflict Detection
# ═══════════════════════════════════════════════════════════


# Modifier keys where different values indicate contradiction
_ENUM_KEYS = {"rule", "tense", "necessity", "modality"}
# Modifier keys that are contradictory if one is high and other is low
_COHERENCE_PAIRS = [("confidence", "certainty")]


class ConflictDetector:
    """Kanerva F6: Third-edge distance-based conflict detection.

    When a new edge [A]->[B] is ingested, check existing edges from A:
      - For each existing [A]->[C], predict expected distance d(B,C)
      - Compare with actual embedding distance
      - Large deviation = potential conflict

    Also checks modifier contradictions on same (source, target) pair.

    This is a programmatic pre-filter. Final judgment requires
    LLM or human review. Principle: warn, never reject.
    """

    def __init__(
        self,
        deviation_threshold: float = 0.3,
        min_confidence: float = 0.5,
    ) -> None:
        if not 0.0 < deviation_threshold <= 1.0:
            raise ValueError(
                f"deviation_threshold must be in (0, 1], got {deviation_threshold}"
            )
        self.deviation_threshold = deviation_threshold
        self.min_confidence = min_confidence

    def check_new_edge(
        self,
        engine: "STGEngine",
        source: str,
        target: str,
        new_modifiers: Optional[Dict] = None,
    ) -> Optional[ConflictReport]:
        """Check a new edge against existing graph for conflicts.

        Checks two things:
          1. Modifier contradictions on same (source, target) if edge exists
          2. Third-edge distance anomaly using embeddings (if available)

        Returns:
            ConflictReport if conflict found, None otherwise
        """
        conflicts: List[Tuple[str, str]] = []
        details_parts: List[str] = []

        # --- Check 1: Modifier contradiction ---
        if new_modifiers:
            mod_report = self.check_modifier_contradiction(
                engine, source, target, new_modifiers
            )
            if mod_report:
                return mod_report

        # --- Check 2: Third-edge distance anomaly ---
        # Requires embeddings (Phase 7G)
        if engine._vector_index is None or engine._embed_texts is None:
            return None

        # Get all existing edges from source with sufficient confidence
        existing_targets = []
        for edge in engine._edges:
            if (edge.source == source
                    and edge.target != target
                    and edge.confidence >= self.min_confidence):
                existing_targets.append(edge.target)

        if not existing_targets:
            return None

        # Get embedding for the new target
        try:
            target_vec = engine._vector_index.get_vector(target)
            source_vec = engine._vector_index.get_vector(source)
        except (KeyError, AttributeError):
            return None

        if target_vec is None or source_vec is None:
            return None

        # Cosine distance: 1 - cosine_similarity, in [0, 2]
        # Normalize to [0, 1] by dividing by 2
        import numpy as np

        def _cosine_dist(a, b):
            dot = np.dot(a, b)
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na == 0 or nb == 0:
                return 1.0
            return float(1.0 - dot / (na * nb))

        dist_source_target = _cosine_dist(source_vec, target_vec)

        for existing_tgt in existing_targets:
            try:
                existing_vec = engine._vector_index.get_vector(existing_tgt)
            except (KeyError, AttributeError):
                continue
            if existing_vec is None:
                continue

            dist_source_existing = _cosine_dist(source_vec, existing_vec)

            # F6: E{d(B,C)} = a + b - 2ab  (for distances in [0,1])
            predicted_dist = (
                dist_source_target + dist_source_existing
                - 2.0 * dist_source_target * dist_source_existing
            )

            actual_dist = _cosine_dist(target_vec, existing_vec)
            deviation = abs(actual_dist - predicted_dist)

            if deviation > self.deviation_threshold:
                conflicts.append((source, existing_tgt))
                details_parts.append(
                    f"[{source}]->[{existing_tgt}]: predicted d({target},{existing_tgt})="
                    f"{predicted_dist:.3f}, actual={actual_dist:.3f}, "
                    f"deviation={deviation:.3f}"
                )

        if not conflicts:
            return None

        max_deviation = max(
            abs(
                _cosine_dist(target_vec, engine._vector_index.get_vector(c[1]))
                - (dist_source_target
                   + _cosine_dist(source_vec, engine._vector_index.get_vector(c[1]))
                   - 2.0 * dist_source_target
                   * _cosine_dist(source_vec, engine._vector_index.get_vector(c[1])))
            )
            for c in conflicts
            if engine._vector_index.get_vector(c[1]) is not None
        ) if conflicts else 0.0

        return ConflictReport(
            new_edge=(source, target),
            conflicting_edges=conflicts,
            conflict_score=min(1.0, max_deviation / self.deviation_threshold),
            details="; ".join(details_parts),
        )

    def check_modifier_contradiction(
        self,
        engine: "STGEngine",
        source: str,
        target: str,
        new_modifiers: Dict,
    ) -> Optional[ConflictReport]:
        """Check if new modifiers contradict existing edge modifiers.

        Contradiction rules:
          - Same enum key (rule, tense, etc.), different value
          - confidence/certainty incoherence (one near 0, other near 1)

        Returns:
            ConflictReport if contradiction found, None otherwise
        """
        existing_edge = engine._edges_lookup.get((source.lower(), target.lower()))
        if existing_edge is None:
            return None

        existing_mods = existing_edge.modifiers
        contradictions: List[str] = []

        # Check enum key contradictions
        for key in _ENUM_KEYS:
            old_val = existing_mods.get(key) or getattr(existing_edge, key, None)
            new_val = new_modifiers.get(key)
            if old_val is not None and new_val is not None and old_val != new_val:
                contradictions.append(
                    f"{key}: existing='{old_val}' vs new='{new_val}'"
                )

        # Check coherence pairs
        for key_a, key_b in _COHERENCE_PAIRS:
            old_a = existing_mods.get(key_a, existing_edge.confidence if key_a == "confidence" else None)
            new_b = new_modifiers.get(key_b)
            if old_a is not None and new_b is not None:
                if isinstance(old_a, (int, float)) and isinstance(new_b, (int, float)):
                    if abs(old_a - new_b) > 0.7:
                        contradictions.append(
                            f"{key_a}={old_a:.2f} vs {key_b}={new_b:.2f} (incoherent)"
                        )

        if not contradictions:
            return None

        return ConflictReport(
            new_edge=(source, target),
            conflicting_edges=[(source, target)],
            conflict_score=min(1.0, len(contradictions) * 0.4),
            details="Modifier contradictions: " + "; ".join(contradictions),
        )
