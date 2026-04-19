"""STG Learning — Making the circuit board alive.

Phase 7B: Hebbian learning and synaptic pruning for the Semantic Tension Graph.

HebbianLearner: Strengthens co-activated edges, weakens noise edges.
SynapticPruner: Removes low-confidence, unused, non-critical edges.

Both are pure operators — they take an engine reference and modify it.
All actions are logged as LearningEvent for audit trail.
"""

import json
import math
import time
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from stg_engine.engine import STGEngine

from stg_engine.types import LearningEvent

# Rust core (required)
from stg_engine import _rust_core as _rust


# ═══════════════════════════════════════════════════════════
# Hebbian Learning
# ═══════════════════════════════════════════════════════════


class HebbianLearner:
    """Hebbian learning engine for STG.

    "Neurons that fire together wire together" (Hebb, 1949).

    After propagation, edges between co-activated nodes are strengthened.
    Edges from strongly active to inactive nodes are weakened (lateral
    inhibition). Both strengthen and weaken are modulated by source
    activation — weakly activated sources cause minimal change.

    IMPORTANT: Modifies edge.salience (retrieval accessibility), NOT
    edge.confidence (truth value). Confidence never auto-decays.
    See CONFIDENCE_SALIENCE_SPLIT.md for rationale.

    Weakening uses a separate, higher activation threshold
    (weaken_activation_threshold, default 0.3) to prevent the
    broad-scope problem where 99% of edges get weakened every
    propagation cycle. Only edges from strongly activated source
    nodes are candidates for weakening, and the effective rate
    is scaled by source activation level.
    """

    def __init__(
        self,
        strengthen_rate: float = 0.05,
        weaken_rate: float = 0.01,
        confidence_floor: float = 0.01,
        confidence_ceiling: float = 1.0,
        activation_threshold: float = 0.1,
        weaken_activation_threshold: float = 0.15,
        rule_modulation: Optional[Dict[str, float]] = None,
    ) -> None:
        if not 0.0 < strengthen_rate <= 1.0:
            raise ValueError(f"strengthen_rate must be in (0, 1], got {strengthen_rate}")
        if not 0.0 < weaken_rate <= 1.0:
            raise ValueError(f"weaken_rate must be in (0, 1], got {weaken_rate}")
        if confidence_floor < 0.0:
            raise ValueError(f"confidence_floor must be >= 0, got {confidence_floor}")
        if confidence_ceiling > 1.0 or confidence_ceiling <= confidence_floor:
            raise ValueError(
                f"confidence_ceiling must be > floor and <= 1.0, "
                f"got ceiling={confidence_ceiling}, floor={confidence_floor}"
            )

        self.strengthen_rate = strengthen_rate
        self.weaken_rate = weaken_rate
        self.floor = confidence_floor
        self.ceiling = confidence_ceiling
        self.activation_threshold = activation_threshold
        self.weaken_activation_threshold = weaken_activation_threshold
        self.rule_modulation: Dict[str, float] = rule_modulation or {}

        # Cumulative stats
        self._strengthened: int = 0
        self._weakened: int = 0

    @property
    def stats(self) -> Dict[str, int]:
        """Cumulative learning statistics."""
        return {
            "strengthened": self._strengthened,
            "weakened": self._weakened,
            "total_events": self._strengthened + self._weakened,
        }

    def learn_from_propagation(
        self,
        engine: "STGEngine",
        activation_map: Dict[str, float],
    ) -> List[LearningEvent]:
        """Learn from a propagation result.

        For each edge (A→B):
        - If both A and B are active: strengthen (Hebbian co-activation)
        - If A is active but B is not: weaken (lateral inhibition)
        - If neither is active: no change

        Args:
            engine: STGEngine instance (edges modified in place)
            activation_map: Node name → activation level from propagate()

        Returns:
            List of LearningEvent records for audit trail.
        """
        now = time.time()
        events: List[LearningEvent] = []
        # Normalize activation_map keys to lowercase (case-insensitive matching)
        _act_map = {k.lower(): v for k, v in activation_map.items()}

        # Build flat edge list with skip flag for virtual/temporal/superseded
        _rust_edges: List[Tuple[str, str, float, bool]] = []
        for edge in engine._edges:
            _ec = edge.modifiers.get("edge_class", edge.edge_class)
            _skip = (
                _ec in ("virtual", "temporal")
                or bool(edge.modifiers.get("superseded_at"))
            )
            _rust_edges.append((
                edge.source.lower(),
                edge.target.lower(),
                float(edge.salience),
                _skip,
            ))

        updates = _rust.hebbian_update(
            _act_map,
            _rust_edges,
            self.strengthen_rate,
            self.weaken_rate,
            self.ceiling,
            self.floor,
            self.activation_threshold,
            self.weaken_activation_threshold,
        )

        for idx, new_sal, action_code in updates:
            edge = engine._edges[idx]
            old_sal = edge.salience
            edge.salience = new_sal
            if action_code == 1:
                edge.last_used = now
                self._strengthened += 1
                event_type = "strengthen"
            else:
                self._weakened += 1
                event_type = "weaken"
            events.append(LearningEvent(
                event_type=event_type,
                source=edge.source.lower(),
                target=edge.target.lower(),
                old_confidence=old_sal,
                new_confidence=new_sal,
                timestamp=now,
                trigger="propagation",
            ))

        if events:
            engine._invalidate_caches()

        return events

    def learn_from_path(
        self,
        engine: "STGEngine",
        path: List[str],
        strength: float = 1.0,
    ) -> List[LearningEvent]:
        """Explicitly strengthen a known-good path.

        Each consecutive edge in the path is strengthened.
        Does NOT weaken other edges (no lateral inhibition).

        Args:
            engine: STGEngine instance
            path: List of node names forming a path
            strength: Modulation factor (0.0-1.0). Lower = less strengthening.

        Returns:
            List of LearningEvent records.
        """
        now = time.time()
        events: List[LearningEvent] = []
        effective_alpha = self.strengthen_rate * max(min(strength, 1.0), 0.0)

        for i in range(len(path) - 1):
            src, tgt = path[i].lower(), path[i + 1].lower()
            edge = engine._edges_lookup.get((src, tgt))
            if edge is None:
                continue
            # Skip virtual/temporal edges — only strengthen real knowledge paths
            _ec = edge.modifiers.get("edge_class", edge.edge_class)
            if _ec in ("virtual", "temporal"):
                continue

            old_sal = edge.salience
            if old_sal >= self.ceiling:
                continue

            new_sal = old_sal + effective_alpha * (self.ceiling - old_sal)
            new_sal = min(new_sal, self.ceiling)

            if abs(new_sal - old_sal) > 1e-10:
                edge.salience = new_sal
                edge.last_used = now
                self._strengthened += 1
                events.append(LearningEvent(
                    event_type="strengthen",
                    source=src,
                    target=tgt,
                    old_confidence=old_sal,
                    new_confidence=new_sal,
                    timestamp=now,
                    trigger="manual",
                ))

        if events:
            engine._invalidate_caches()

        return events


# ═══════════════════════════════════════════════════════════
# Salience Time Decay (G2)
# ═══════════════════════════════════════════════════════════


def decay_salience(
    edges: list,
    now: float,
    half_life_days: float = 365.0,
    floor: float = 0.01,
) -> int:
    """Apply time-based exponential decay to edge salience.

    Edges that haven't been used recently have their salience reduced,
    making them harder to retrieve via propagation and closer to pruning.

    Formula: new_sal = sal × exp(-elapsed × ln2 / half_life_seconds)

    Args:
        edges: List of STGEdge objects (modified in place)
        now: Current timestamp (epoch seconds)
        half_life_days: Half-life in days (default 365)
        floor: Minimum salience floor (default 0.01)

    Returns:
        Number of edges whose salience was decayed
    """
    if half_life_days <= 0:
        return 0

    half_life_seconds = half_life_days * 86400.0
    decay_count = 0

    for edge in edges:
        # Skip virtual edges (they have their own cleanup)
        if edge.modifiers.get("edge_class") == "virtual":
            continue
        # Skip superseded edges (historical, frozen)
        if edge.modifiers.get("superseded_at"):
            continue

        # Determine last activity time
        last_active = edge.last_used or edge.created_at or 0.0
        if last_active <= 0:
            continue  # No timestamp info, can't decay

        elapsed = max(0.0, now - last_active)
        if elapsed <= 0:
            continue

        # Exponential decay
        decay_factor = math.exp(-elapsed * math.log(2) / half_life_seconds)
        new_salience = max(floor, edge.salience * decay_factor)

        if new_salience < edge.salience:
            edge.salience = new_salience
            edge.last_used = now  # prevent over-decay on repeated calls
            decay_count += 1

    return decay_count


# ═══════════════════════════════════════════════════════════
# Synaptic Pruning
# ═══════════════════════════════════════════════════════════


class SynapticPruner:
    """Periodic graph cleanup — remove low-salience, unused, non-critical edges.

    Brain analogy: synaptic pruning during sleep.

    An edge is pruned only if ALL three conditions are true:
    1. Salience below threshold (low retrieval accessibility)
    2. Not used recently (stale)
    3. Low EID (not a critical bridge)

    Note: confidence (truth value) is NOT a pruning criterion.
    A true but forgotten fact should not be deleted.

    After edge removal, orphan nodes (degree=0) are also removed.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.1,
        unused_days: float = 365.0,
        eid_safety_threshold: float = 0.01,
        virtual_unused_days: float = 90.0,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.unused_days = unused_days
        self.eid_safety_threshold = eid_safety_threshold
        self.virtual_unused_days = virtual_unused_days

    def prune(
        self,
        engine: "STGEngine",
        stg_path: Optional[str] = None,
    ) -> List[LearningEvent]:
        """Execute one pruning cycle.

        Removes edges that are weak + unused + non-critical.
        Then removes orphan nodes.
        All deletions are logged to the pruned_log table if stg_path is provided.

        Args:
            engine: STGEngine instance (modified in place)
            stg_path: Path to .stg file for persistent audit logging.
                      If None, deletions are still returned as LearningEvents
                      but not persisted to the pruned_log table.

        Returns:
            List of LearningEvent records for audit trail.
        """
        from stg_engine.metrics import edge_information_density

        now = time.time()
        events: List[LearningEvent] = []
        pruned_entries: List[Dict] = []

        # Pass 0: Prune stale virtual edges (shorter lifespan than real edges)
        for edge in list(engine._edges):
            if edge.modifiers.get("edge_class") != "virtual":
                continue
            created_at = edge.modifiers.get("virtual_created_at") or edge.created_at or 0.0
            last_check = edge.last_used or created_at
            days_unused = (now - last_check) / 86400 if last_check else float("inf")
            if days_unused >= self.virtual_unused_days:
                old_conf = edge.confidence
                pruned_entries.append({
                    "pruned_at": now,
                    "item_type": "virtual_edge",
                    "source": edge.source,
                    "target": edge.target,
                    "confidence": edge.confidence,
                    "salience": edge.salience,
                    "last_used": edge.last_used,
                    "modifiers_json": json.dumps(edge.modifiers, ensure_ascii=False),
                    "reason": f"virtual_unused_{days_unused:.0f}d",
                })
                engine.remove_edge(edge.source, edge.target)
                events.append(LearningEvent(
                    event_type="prune_virtual",
                    source=edge.source,
                    target=edge.target,
                    old_confidence=old_conf,
                    new_confidence=0.0,
                    timestamp=now,
                    trigger="prune_cycle",
                ))

        # Step 1: Identify candidates (real edges only, by salience)
        candidates = []
        for edge in list(engine._edges):
            if edge.salience >= self.confidence_threshold:
                continue

            # Check staleness
            if edge.last_used is not None:
                days_unused = (now - edge.last_used) / 86400
                if days_unused < self.unused_days:
                    continue

            candidates.append(edge)

        # Step 2 & 3: EID safety check + remove
        for edge in candidates:
            eid = edge_information_density(
                engine._graph, edge.source, edge.target
            )
            if eid >= self.eid_safety_threshold:
                continue  # Critical bridge — do not prune

            old_conf = edge.confidence
            days = (now - edge.last_used) / 86400 if edge.last_used else -1
            pruned_entries.append({
                "pruned_at": now,
                "item_type": "edge",
                "source": edge.source,
                "target": edge.target,
                "confidence": edge.confidence,
                "salience": edge.salience,
                "last_used": edge.last_used,
                "modifiers_json": json.dumps(edge.modifiers, ensure_ascii=False),
                "reason": f"low_salience={edge.salience:.3f}_unused_{days:.0f}d_eid={eid:.4f}",
            })
            engine.remove_edge(edge.source, edge.target)
            events.append(LearningEvent(
                event_type="prune",
                source=edge.source,
                target=edge.target,
                old_confidence=old_conf,
                new_confidence=0.0,
                timestamp=now,
                trigger="prune_cycle",
            ))

        # Step 4: Remove orphan nodes
        orphans = [
            name for name in list(engine._nodes)
            if name in engine._graph and engine._graph.degree(name) == 0
        ]
        for name in orphans:
            node = engine._nodes[name]
            pruned_entries.append({
                "pruned_at": now,
                "item_type": "orphan_node",
                "source": name,
                "target": "",
                "confidence": 0.0,
                "salience": 0.0,
                "last_used": None,
                "modifiers_json": json.dumps({
                    "namespace": node.namespace,
                    "anchor_type": node.anchor_type,
                }, ensure_ascii=False),
                "reason": "orphan_after_edge_pruning",
            })
            del engine._nodes[name]
            engine._graph.remove_node(name)
            events.append(LearningEvent(
                event_type="prune_orphan",
                source=name,
                target="",
                old_confidence=0.0,
                new_confidence=0.0,
                timestamp=now,
                trigger="prune_cycle",
            ))

        if events:
            engine._invalidate_caches()

        # Persist pruning audit log
        if pruned_entries and stg_path:
            from stg_engine.persistence import append_pruned_log
            append_pruned_log(stg_path, pruned_entries)

        return events

    def dry_run(
        self, engine: "STGEngine"
    ) -> List[Tuple[str, str, float, float]]:
        """Preview what would be pruned without modifying the graph.

        Returns:
            List of (source, target, salience, eid) tuples.
        """
        from stg_engine.metrics import edge_information_density

        now = time.time()
        results: List[Tuple[str, str, float, float]] = []

        for edge in engine._edges:
            if edge.salience >= self.confidence_threshold:
                continue

            if edge.last_used is not None:
                days_unused = (now - edge.last_used) / 86400
                if days_unused < self.unused_days:
                    continue

            eid = edge_information_density(
                engine._graph, edge.source, edge.target
            )
            if eid >= self.eid_safety_threshold:
                continue

            results.append((
                edge.source,
                edge.target,
                edge.salience,
                eid,
            ))

        return results
