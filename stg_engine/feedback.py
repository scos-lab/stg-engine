"""STG Feedback Loop Architecture — Emergence Through Interconnection.

Phase 7E: Wires three feedback loops between Phase 7B-7D modules to
produce emergent cognitive behavior. Provides turn-lifecycle hooks
(pre_turn, post_turn, run_periodic, session_end).

No new algorithms — only compositions of existing modules:
  - HebbianLearner + SynapticPruner (Phase 7B)
  - CognitiveArchitecture (Phase 7D): goals, warmer, hypotheses,
    self_model, router, temporal

Three feedback loops:
  1. Self-Improvement: SelfModel → GoalRegister → Router → Learning → SelfModel
  2. Predictive: TemporalDynamics → Warmup → Query → Hebbian → TemporalDynamics
  3. Creative: HypothesisGenerator → Validation → New Edges → Pruning → HypothesisGenerator
"""

import time
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

from stg_engine.learning import HebbianLearner, SynapticPruner
from stg_engine.types import FeedbackLoopConfig, LoopStats, STGTension, TurnRecord

if TYPE_CHECKING:
    from stg_engine.engine import STGEngine


class FeedbackLoopManager:
    """Orchestrate feedback loops and turn lifecycle.

    Composes:
      - CognitiveArchitecture (Phase 7D) via engine._cognitive
      - HebbianLearner (Phase 7B): edge weight learning
      - SynapticPruner (Phase 7B): periodic pruning

    Does NOT own cognitive modules — accesses them through engine.
    Creates its own HebbianLearner and SynapticPruner instances.
    """

    def __init__(
        self,
        config: Optional[FeedbackLoopConfig] = None,
        hebbian_learner: Optional[HebbianLearner] = None,
        synaptic_pruner: Optional[SynapticPruner] = None,
        stg_path: Optional[str] = None,
    ) -> None:
        self.config = config or FeedbackLoopConfig()
        self._hebbian = hebbian_learner or HebbianLearner()
        self._pruner = synaptic_pruner or SynapticPruner()
        self._stg_path = stg_path
        self._turn_count: int = 0
        self._stats = LoopStats()
        self._history: List[TurnRecord] = []
        self._predicted_nodes: Set[str] = set()

    # ─── Turn Lifecycle ──────────────────────────────────────

    def pre_turn(self, engine: "STGEngine", context: str) -> Dict[str, Any]:
        """Prepare STG for a new turn.

        Steps:
          1. Apply temporal dynamics (time-based activation)
          2. Predictive warmup (context-based pre-activation)
          3. Goal bias is implicit in subsequent propagation
        """
        self._ensure_cognitive(engine)
        cognitive = engine._cognitive

        temporal_updates = cognitive.temporal.apply(engine)
        warmup_count = 0
        if self.config.warmup_on_pre_turn and context:
            warmup_count = cognitive.warmer.warmup(engine, context)

        self._stats.total_warmups += 1

        # Save predictions for post_turn comparison (Vehicle 13)
        if self.config.tension_on_prediction_error and warmup_count > 0:
            self._predicted_nodes = cognitive.warmer.last_warmed_nodes.copy()
        else:
            self._predicted_nodes = set()

        return {
            "temporal_updates": temporal_updates,
            "warmup_count": warmup_count,
            "active_goals": len(cognitive.goals.current_goals),
        }

    def post_turn(
        self,
        engine: "STGEngine",
        query: str,
        results: List[str],
        success: bool,
    ) -> Dict[str, Any]:
        """Consolidate learning after a turn.

        Steps:
          1. Report strategy success to router
          2. Hebbian learning from current activation state
          3. Increment turn counter
          4. Auto-run periodic if interval reached
        """
        self._ensure_cognitive(engine)
        cognitive = engine._cognitive
        summary: Dict[str, Any] = {}

        # Step 1: Router feedback
        strategy = cognitive.router._classify(query)
        cognitive.router.report_success(strategy, success)

        # Step 2: Hebbian learning
        hebbian_events = 0
        if self.config.learn_on_post_turn and results:
            activation_map = {
                name: node.activation
                for name, node in engine._nodes.items()
            }
            events = self._hebbian.learn_from_propagation(engine, activation_map)
            hebbian_events = len(events)
            self._stats.total_hebbian_events += hebbian_events

        # Step 2.5: Prediction error → auto tension (Braitenberg Vehicle 13)
        prediction_tensions = 0
        if self.config.tension_on_prediction_error and self._predicted_nodes:
            actual = {n for n, nd in engine._nodes.items() if nd.activation >= 0.1}
            missed = self._predicted_nodes - actual
            surprised = actual - self._predicted_nodes

            if len(missed) > len(self._predicted_nodes) * 0.3:
                engine.add_tension(STGTension(
                    name=f"prediction_miss_turn_{self._turn_count + 1}",
                    initial_value=len(missed) / max(len(self._predicted_nodes), 1),
                    current_value=len(missed) / max(len(self._predicted_nodes), 1),
                    status="active",
                    description=f"Predicted {len(self._predicted_nodes)}, {len(missed)} missed",
                ))
                prediction_tensions += 1

            if len(surprised) > 5:
                engine.add_tension(STGTension(
                    name=f"prediction_surprise_turn_{self._turn_count + 1}",
                    initial_value=len(surprised) / max(len(actual), 1),
                    current_value=len(surprised) / max(len(actual), 1),
                    status="active",
                    description=f"Surprise: {len(surprised)} unpredicted nodes activated",
                ))
                prediction_tensions += 1

            self._predicted_nodes = set()
        summary["prediction_tensions"] = prediction_tensions

        # Step 3: Turn counter
        self._turn_count += 1
        self._stats.total_turns += 1
        summary["hebbian_events"] = hebbian_events
        summary["turn_number"] = self._turn_count

        # Step 4: Auto-periodic
        periodic_summary: Dict[str, Any] = {}
        if self._turn_count % self.config.periodic_interval == 0:
            periodic_summary = self.run_periodic(engine)

        # Record turn
        record = TurnRecord(
            turn_number=self._turn_count,
            post_turn_summary=summary,
            periodic_ran=bool(periodic_summary),
            periodic_summary=periodic_summary,
            timestamp=time.time(),
        )
        self._history.append(record)
        if len(self._history) > 100:
            self._history = self._history[-100:]

        summary["periodic"] = periodic_summary
        return summary

    def run_periodic(self, engine: "STGEngine") -> Dict[str, Any]:
        """Run periodic maintenance tasks.

        Includes self-improvement loop (self-model -> goals) and
        creative loop (hypotheses -> apply -> prune).
        """
        self._ensure_cognitive(engine)
        self._stats.total_periodic_runs += 1

        improvement = self._run_self_improvement(engine)
        creative = self._run_creative(engine)

        return {
            "self_improvement": improvement,
            "creative": creative,
            "periodic_run_number": self._stats.total_periodic_runs,
        }

    def session_end(self, engine: "STGEngine") -> Dict[str, Any]:
        """Clean up at session end. Call before engine.save().

        Sequence:
          1. Flush telemetry (writes cooccurrence data to SQLite)
          2. Co-activation edge creation (reads cooccurrence, creates edges)
          3. Prune (removes weak unused edges, including stale coactivation edges)
        """
        summary: Dict[str, Any] = {}

        # Step 1: Flush telemetry data (Phase 10)
        telemetry_written = 0
        if engine._telemetry is not None and self._stg_path:
            engine._telemetry.record_session_summary(engine)
            telemetry_written = engine._telemetry.flush(self._stg_path)
            engine._telemetry.reset()
        summary["telemetry_written"] = telemetry_written

        # Step 2: Co-activation edge creation (Phase 11B)
        coactivation_created = 0
        if self.config.coactivation_on_session_end and self._stg_path:
            from stg_engine.coactivation import (
                find_coactivation_candidates,
                create_coactivation_edges,
                record_coactivation_event,
            )
            candidates = find_coactivation_candidates(
                engine,
                self._stg_path,
                min_cooccurrence=self.config.coactivation_min_count,
                max_candidates=self.config.coactivation_max_per_session,
                max_coactivation_per_node=self.config.coactivation_max_per_node,
                exclude_top_hubs=self.config.coactivation_exclude_top_hubs,
            )
            if candidates:
                events = create_coactivation_edges(engine, candidates)
                coactivation_created = len(events)
                record_coactivation_event(
                    self._stg_path,
                    candidates_found=len(candidates),
                    edges_created=coactivation_created,
                    candidates_detail=candidates,
                )
        summary["coactivation_created"] = coactivation_created

        # Step 2.5: Salience time decay (G2) — before pruning so decayed edges
        # can naturally fall into pruning range
        from stg_engine.learning import decay_salience
        decayed = decay_salience(engine._edges, time.time())
        summary["salience_decayed"] = decayed

        # Step 3: Prune (removes stale edges including unused coactivation edges)
        if self.config.auto_prune_on_session_end:
            events = self._pruner.prune(engine, stg_path=self._stg_path)
            pruned = len(events)
            self._stats.edges_pruned += pruned
            summary["edges_pruned"] = pruned
            summary["orphans_removed"] = sum(
                1 for e in events if e.event_type == "prune_orphan"
            )

        summary["final_stats"] = {
            "total_turns": self._stats.total_turns,
            "total_periodic_runs": self._stats.total_periodic_runs,
            "hypotheses_applied": self._stats.hypotheses_applied,
            "total_hebbian_events": self._stats.total_hebbian_events,
            "edges_pruned": self._stats.edges_pruned,
        }
        return summary

    # ─── Accessors ───────────────────────────────────────────

    def get_stats(self) -> LoopStats:
        """Return cumulative statistics."""
        return self._stats

    def get_history(self, n: int = 10) -> List[TurnRecord]:
        """Return last N turn records."""
        return self._history[-n:]

    # ─── Self-Improvement Loop ───────────────────────────────

    def _run_self_improvement(self, engine: "STGEngine") -> Dict[str, Any]:
        """Execute one self-improvement cycle.

        SelfModel -> GoalRegister -> (bias future turns)
        """
        self._ensure_cognitive(engine)
        cognitive = engine._cognitive
        if cognitive is None:
            return {}

        # Step 1: Build self-model
        report = cognitive.self_model.build(engine)
        self._stats.self_models_built += 1

        # Step 2: Auto-update goals from gaps
        goals_added = 0
        if self.config.auto_goal_update and report.gap_namespaces:
            for ns in report.gap_namespaces[:self.config.max_auto_goals]:
                keywords = [ns.lower()]
                # Add top node tokens from that namespace
                for name, node in engine._nodes.items():
                    if node.namespace == ns:
                        keywords.extend(
                            t for t in name.lower().replace(":", "_").split("_")
                            if len(t) > 2
                        )
                        if len(keywords) >= 5:
                            break
                try:
                    cognitive.goals.add_goal(
                        name=f"strengthen_{ns}",
                        keywords=keywords[:5],
                        priority=1.2,
                    )
                    goals_added += 1
                    self._stats.goals_auto_generated += 1
                except ValueError:
                    pass

        return {
            "self_model_connectivity": report.connectivity_health,
            "gap_namespaces": report.gap_namespaces,
            "goals_added": goals_added,
        }

    # ─── Creative Loop ───────────────────────────────────────

    def _run_creative(self, engine: "STGEngine") -> Dict[str, Any]:
        """Execute one creative cycle.

        HypothesisGenerator -> filter -> apply -> optional prune
        """
        self._ensure_cognitive(engine)
        cognitive = engine._cognitive
        if cognitive is None:
            return {}

        # Step 1: Generate hypotheses
        hypotheses = cognitive.hypotheses.generate(engine)
        self._stats.hypotheses_generated += len(hypotheses)

        # Step 2: Filter by confidence
        qualified = [
            h for h in hypotheses
            if h.confidence >= self.config.hypothesis_min_confidence
        ]
        to_apply = qualified[:self.config.hypothesis_max_apply]
        rejected = len(hypotheses) - len(to_apply)
        self._stats.hypotheses_rejected += rejected

        # Step 3: Apply
        applied = 0
        if to_apply:
            applied = cognitive.hypotheses.apply_hypotheses(engine, to_apply)
            self._stats.hypotheses_applied += applied

        # Step 4: Optional prune
        pruned = 0
        if self.config.prune_after_creative and applied > 0:
            prune_events = self._pruner.prune(engine, stg_path=self._stg_path)
            pruned = len(prune_events)
            self._stats.edges_pruned += pruned

        return {
            "hypotheses_found": len(hypotheses),
            "hypotheses_applied": applied,
            "hypotheses_rejected": rejected,
            "edges_pruned": pruned,
        }

    # ─── Internal ────────────────────────────────────────────

    def _ensure_cognitive(self, engine: "STGEngine") -> None:
        """Ensure engine has cognitive architecture enabled."""
        if not engine.cognitive_enabled:
            engine.enable_cognitive()
