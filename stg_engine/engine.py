"""STG Engine — Self-contained computation graph for Semantic Tension.

The STG Engine is to semantic knowledge what PyTorch is to neural networks:
a unified substrate where storage and computation are one.

Nodes carry their computed state. Edges carry their modifiers.
Formulas execute directly on the graph. Persistence is serialization.
"""

import logging
import re
import time as _time
from typing import Dict, FrozenSet, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# Universal semantic-carrying modifier fields.
# Used by supersede detection: when two edges share (source, field, value)
# but differ in target, the older one is flagged as suspected_supersede.
SEMANTIC_FIELDS: Tuple[str, ...] = (
    "relation",
    "status",
    "role",
    "type",
    "kind",
    "is_a",
    "action",
    "predicate",
    "phase",
)


def _get_semantic_field(modifiers: Optional[dict]) -> Tuple[Optional[str], Optional[str]]:
    """Return (field_name, value) of the first semantic field present, or (None, None)."""
    if not modifiers:
        return None, None
    for f in SEMANTIC_FIELDS:
        v = modifiers.get(f)
        if v:
            return f, str(v)
    return None, None


# Skill invocation modifiers — orthogonal to SEMANTIC_FIELDS. These describe HOW
# to invoke a Skill node (path, interpreter, timeout) rather than the semantic
# relation between nodes. Used by the skill_runner module; see
# development/design/STG_SKILL_EXECUTOR_DESIGN.md.
SKILL_INVOCATION_FIELDS: Tuple[str, ...] = (
    "executable",
    "interpreter",
    "args_template",
    "stl_io",
    "timeout_s",
    "allow_root_override",
)

# The Skill namespace is reserved: only nodes in this namespace are considered
# by `stg use` and the `stg skill` subcommand family.
SKILL_NAMESPACE: str = "Skill"


def _truthy(val) -> bool:
    """Parse STL modifier strings as booleans. STL stores everything as strings."""
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes", "on")


def _get_skill_invocation(modifiers: Optional[dict]) -> dict:
    """Extract the skill invocation subset from an edge's modifiers.

    Returns a dict with only the SKILL_INVOCATION_FIELDS that are present.
    `executable` and `stl_io` are coerced to bool; `timeout_s` to int.
    """
    if not modifiers:
        return {}
    out: dict = {}
    for f in SKILL_INVOCATION_FIELDS:
        if f not in modifiers:
            continue
        v = modifiers[f]
        if f in ("executable", "stl_io"):
            out[f] = _truthy(v)
        elif f == "timeout_s":
            try:
                out[f] = int(str(v))
            except (TypeError, ValueError):
                continue
        else:
            out[f] = str(v)
    return out

import networkx as nx

from stg_engine.types import (
    STGNode, STGEdge, STGSession, STGEvent,
    STGTension, STGBeliefEvolution, SystemSnapshot,
    PropagationMetrics, GraphMetrics, SearchResult,
    ConvergenceResult, InhibitionConfig,
)
from stg_engine.formulas import (
    compute_psi,
    compute_node_tension,
    compute_path_tension as _compute_path_tension,
    compute_edge_tension,
    compute_activation as _compute_activation,
    compute_self_relevance,
    compute_influence,
    compute_intrinsic_reward,
)
from stg_engine.persistence import save_engine_state, load_engine_state

# ─── Hot-path core: optional Rust, pure-Python fallback ──────────
try:
    from stg_engine import _rust_core as _rust
except ImportError:
    from stg_engine import _core_fallback as _rust


class STGEngine:
    """Self-contained computation graph for Semantic Tension.

    Storage and computation are unified. The graph IS the computation
    substrate — like tensors in a neural network.

    Usage:
        engine = STGEngine()
        engine.ingest_stl('[A] -> [B] ::mod(confidence=0.9)')
        engine.compute_all_tensions()
        psi = engine.compute_psi()
        engine.save("memory.stg")

        # Later...
        engine = STGEngine.load("memory.stg")
    """

    def __init__(self) -> None:
        """Initialize empty STG Engine."""
        # Internal graph (implementation detail — not exposed)
        self._graph: nx.DiGraph = nx.DiGraph()

        # Primary data stores
        self._nodes: Dict[str, STGNode] = {}
        self._edges: List[STGEdge] = []
        self._edges_lookup: Dict[Tuple[str, str], STGEdge] = {}

        # Episodic memory
        self._sessions: Dict[str, STGSession] = {}
        self._events: Dict[str, STGEvent] = {}

        # Tension tracking
        self._tensions: Dict[str, STGTension] = {}

        # Belief evolution
        self._belief_evolutions: List[STGBeliefEvolution] = []

        # System state history
        self._snapshots: List[SystemSnapshot] = []

        # Metrics caches (Phase 7A)
        self._last_propagation_metrics: Optional[PropagationMetrics] = None
        self._importance_cache: Optional[Dict[str, float]] = None
        self._graph_metrics_cache: Optional[GraphMetrics] = None
        self._gravity_map = None  # Optional[GravityMap] — cached, invalidated on mutation

        # Learning (Phase 7B)
        self._learner = None  # Optional[HebbianLearner]
        self._learning_log: List = []  # List[LearningEvent]
        self.importance_weight: float = 0.0  # Importance bias for propagation

        # Cognitive architecture (Phase 7D)
        self._cognitive = None  # Optional[CognitiveArchitecture]

        # Feedback loops (Phase 7E)
        self._feedback = None  # Optional[FeedbackLoopManager]

        # Semantic search (Phase 7G)
        self._embed_model = None       # Loaded lazily on first search()
        self._vector_index = None      # VectorIndex instance
        self._embed_texts = None       # Dict[str, str] from EmbeddingBuilder
        self._model_name = None        # Name of loaded model

        # Kanerva extensions (Phase 8)
        self.preference_weight: float = 0.0  # Preference bias for propagation

        # Conflict detection (G6) — lazy init, always on
        self._conflict_detector = None  # Optional[ConflictDetector]

        # Entity Resolution (G7) — candidate detection + alias registry
        self._node_tokens: Dict[str, FrozenSet[str]] = {}  # nk → token set
        self._aliases: Dict[str, str] = {}  # normalized_alias → normalized_canonical
        self._last_entity_candidates: List[Tuple[str, float, str]] = []

        # Telemetry (Phase 10)
        self._telemetry = None  # Optional[TelemetryCollector]

        # Inhibition (Phase 9) — disabled by default
        self._inhibition_config: InhibitionConfig = InhibitionConfig()
        self._refractory_set: Dict[str, float] = {}  # node → prior activation

    # ═══════════════════════════════════════════════════════════
    # Inhibition Configuration (Phase 9)
    # ═══════════════════════════════════════════════════════════

    def enable_inhibition(self, **kwargs) -> "InhibitionConfig":
        """Enable and configure the inhibition system.

        Args:
            **kwargs: Override any InhibitionConfig field.
                E.g. enable_inhibition(eta=3.0, sigma=0.8)

        Returns:
            The active InhibitionConfig.
        """
        self._inhibition_config = InhibitionConfig(enabled=True, **kwargs)
        return self._inhibition_config

    def disable_inhibition(self) -> None:
        """Disable all inhibition (restore default linear rescaling)."""
        self._inhibition_config = InhibitionConfig(enabled=False)
        self._refractory_set.clear()

    @property
    def inhibition_config(self) -> "InhibitionConfig":
        """Current inhibition configuration (read-only access)."""
        return self._inhibition_config

    # ═══════════════════════════════════════════════════════════
    # Cache Management (Phase 7A)
    # ═══════════════════════════════════════════════════════════

    def _invalidate_caches(self) -> None:
        """Invalidate cached metrics after graph mutation."""
        self._importance_cache = None
        self._graph_metrics_cache = None
        self._gravity_map = None

    # ═══════════════════════════════════════════════════════════
    # Case-insensitive node key normalization
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _nk(name: str) -> str:
        """Normalize node key — case-insensitive, separator-insensitive storage.

        All dict keys, edge source/target, and graph node IDs use
        the normalized form. STGNode.name preserves the original casing
        from the first add_node() call (display name).

        Normalization: lowercase + hyphens treated as underscores.
        """
        return name.lower().replace("-", "_")

    def _dn(self, key: str) -> str:
        """Display name — map normalized key back to original casing.

        Returns STGNode.name (the casing from the first add_node() call).
        Falls back to the key itself if node not found.
        """
        node = self._nodes.get(key)
        return node.name if node else key

    # ═══════════════════════════════════════════════════════════
    # Entity Resolution (G7)
    # ═══════════════════════════════════════════════════════════

    _ER_STOP_WORDS = frozenset({
        "the", "a", "an", "of", "and", "or", "in", "on", "to",
        "for", "is", "at", "by", "with", "from", "as", "not",
    })

    @staticmethod
    def _tokenize_for_er(name: str) -> FrozenSet[str]:
        """Extract meaningful tokens from a node name for entity resolution."""
        normalized = name.lower().replace("-", "_")
        tokens = normalized.split("_")
        tokens = [t for t in tokens if t and t not in STGEngine._ER_STOP_WORDS]
        return frozenset(tokens)

    def _find_similar_nodes(self, name: str) -> List[Tuple[str, float, str]]:
        """Find existing nodes similar to a new name.

        Returns:
            List of (existing_display_name, similarity_score, reason)
            sorted by score descending. Empty if no candidates.
        """
        new_tokens = self._tokenize_for_er(name)
        if len(new_tokens) < 2:
            return []  # Single-token names: too ambiguous to match

        candidates = []
        sorted_new = sorted(new_tokens)

        for nk, existing_tokens in self._node_tokens.items():
            if len(existing_tokens) < 2:
                continue

            shared = new_tokens & existing_tokens
            if len(shared) < 2:
                continue

            # Strategy A: word-order variant (same tokens, different order)
            sorted_existing = sorted(existing_tokens)
            if sorted_new == sorted_existing:
                candidates.append((self._dn(nk), 1.0, "word-order variant"))
                continue

            # Strategy B: Jaccard overlap
            union = new_tokens | existing_tokens
            jaccard = len(shared) / len(union) if union else 0

            if jaccard < 2 / 3:
                continue

            # Exclude parent→child specialization:
            # If one is a strict subset AND token count differs by ≥2, skip
            smaller_len = min(len(new_tokens), len(existing_tokens))
            larger_len = max(len(new_tokens), len(existing_tokens))
            if len(shared) == smaller_len and larger_len - smaller_len >= 2:
                continue

            candidates.append((
                self._dn(nk), jaccard,
                "token overlap {%s}" % ", ".join(sorted(shared)),
            ))

        candidates.sort(key=lambda x: -x[1])
        return candidates

    def register_alias(self, alias: str, canonical: str) -> bool:
        """Register an alias for an existing node.

        After registration, add_node(alias) will resolve to the canonical node.

        Args:
            alias: The alternative name
            canonical: The existing node name (must exist in graph)

        Returns:
            True if registered, False if canonical doesn't exist.
        """
        nk_alias = self._nk(alias)
        nk_canonical = self._nk(canonical)
        if nk_canonical not in self._nodes:
            return False
        self._aliases[nk_alias] = nk_canonical
        return True

    def remove_alias(self, alias: str) -> bool:
        """Remove an alias. Returns True if it existed."""
        return self._aliases.pop(self._nk(alias), None) is not None

    def list_aliases(self) -> List[Tuple[str, str]]:
        """Return all (alias_key, canonical_display_name) pairs."""
        return [
            (alias, self._dn(canonical))
            for alias, canonical in self._aliases.items()
        ]

    def resolve_name(self, name: str) -> str:
        """Resolve a name through alias table.

        Returns canonical display name if alias exists,
        existing display name if node exists, or name as-is.
        """
        nk = self._nk(name)
        if nk in self._aliases:
            return self._dn(self._aliases[nk])
        return self._dn(nk) if nk in self._nodes else name

    # ═══════════════════════════════════════════════════════════
    # Graph Manipulation
    # ═══════════════════════════════════════════════════════════

    def add_node(
        self,
        name: str,
        namespace: Optional[str] = None,
        anchor_type: Optional[str] = None,
        **metadata: Any,
    ) -> STGNode:
        """Add or update a node in the graph.

        Args:
            name: Anchor name (e.g., "Memory_Architecture")
            namespace: Optional namespace (e.g., "Physics")
            anchor_type: Anchor type (Concept/Event/Entity/Agent/...)
            **metadata: Additional metadata key-value pairs

        Returns:
            The created or updated STGNode
        """
        key = self._nk(name)

        # G7: Alias resolution — redirect to canonical node if alias exists
        if key in self._aliases:
            key = self._aliases[key]
            name = self._dn(key)  # use canonical display name

        if key in self._nodes:
            # Update existing node metadata
            node = self._nodes[key]
            if namespace is not None:
                node.namespace = namespace
            if anchor_type is not None:
                node.anchor_type = anchor_type
            node.metadata.update(metadata)
        else:
            # G7: Check for similar existing nodes before creating
            candidates = self._find_similar_nodes(name)
            if candidates:
                self._last_entity_candidates = candidates
                top_name, top_score, top_reason = candidates[0]
                logger.info(
                    "Entity resolution candidate: '%s' ~ '%s' (%.2f, %s)",
                    name, top_name, top_score, top_reason,
                )

            node = STGNode(
                name=name,  # preserve original casing as display name
                namespace=namespace,
                anchor_type=anchor_type,
                metadata=metadata,
            )
            self._nodes[key] = node
            self._graph.add_node(key)
            self._node_tokens[key] = self._tokenize_for_er(name)
            self._invalidate_caches()

        return node

    def add_edge(
        self,
        source: str,
        target: str,
        confidence: float = 0.5,
        strength: float = 0.5,
        rule: Optional[str] = None,
        time: Optional[str] = None,
        session_id: Optional[str] = None,
        event_id: Optional[str] = None,
        created_at: Optional[float] = None,
        **modifiers: Any,
    ) -> STGEdge:
        """Add an edge (semantic relation) to the graph.

        Automatically creates source/target nodes if they don't exist.

        Args:
            source: Source anchor name
            target: Target anchor name
            confidence: Confidence value (0.0-1.0)
            strength: Relation strength (0.0-1.0)
            rule: Rule type (causal/logical/empirical/definitional)
            time: Temporal context
            session_id: Session that created this edge
            event_id: Event that created this edge
            created_at: Custom creation timestamp (epoch float). Defaults to current time.
            **modifiers: Additional STL modifiers

        Returns:
            The created STGEdge
        """
        # Ensure nodes exist (add_node handles alias resolution internally)
        self.add_node(source)
        self.add_node(target)

        # Resolve aliases before computing normalized keys
        _src = self._nk(source)
        _tgt = self._nk(target)
        if _src in self._aliases:
            _src = self._aliases[_src]
        if _tgt in self._aliases:
            _tgt = self._aliases[_tgt]
        # Display names from the first add_node() call
        _src_dn = self._dn(_src)
        _tgt_dn = self._dn(_tgt)

        # Virtual edge absorption: if adding a real edge where a virtual edge
        # exists (either direction), remove the virtual edge first
        if modifiers.get("edge_class") != "virtual":
            for pair in [(_src, _tgt), (_tgt, _src)]:
                existing = self._edges_lookup.get(pair)
                if existing and existing.modifiers.get("edge_class") == "virtual":
                    self.remove_edge(pair[0], pair[1])

        # G6 fix: conflict detection — check for contradictions before writing
        if modifiers.get("edge_class") != "virtual":
            if self._conflict_detector is None:
                from stg_engine.kanerva import ConflictDetector
                self._conflict_detector = ConflictDetector()
            # Build full modifier dict including named params for contradiction check
            _check_mods = dict(modifiers)
            if rule is not None:
                _check_mods["rule"] = rule
            if confidence is not None:
                _check_mods["confidence"] = confidence
            conflict = self._conflict_detector.check_new_edge(
                self, _src, _tgt, _check_mods
            )
            if conflict:
                # Warn, never reject — log warning and store report
                logger.warning(
                    "Conflict detected on [%s]->[%s]: %s",
                    _src_dn, _tgt_dn, conflict.details,
                )
                modifiers["_conflict_report"] = {
                    "conflicting_edges": conflict.conflicting_edges,
                    "conflict_score": conflict.conflict_score,
                    "details": conflict.details,
                }

        # G8 fix: duplicate edge handling
        # If (source, target) already exists:
        #   - Semantically identical: true duplicate, skip (return existing)
        #   - Different content: allow multi-edge (knowledge evolution)
        #     Old edge stays in _edges list, lookup points to newest
        existing_edge = self._edges_lookup.get((_src, _tgt))
        if existing_edge and existing_edge.modifiers.get("edge_class") != "virtual":
            # Check if new edge is semantically identical to existing
            same_conf = abs(existing_edge.confidence - confidence) < 0.001
            same_strength = abs(existing_edge.strength - strength) < 0.001
            same_rule = (existing_edge.rule or "") == (rule or "")
            same_desc = (existing_edge.modifiers.get("description", "")
                         == modifiers.get("description", ""))
            is_true_duplicate = same_conf and same_strength and same_rule and same_desc

            if is_true_duplicate:
                # True duplicate — no new information, skip
                return existing_edge
            else:
                # Knowledge evolution — allow multi-edge
                # Mark old edge as superseded, keep in _edges for history
                existing_edge.modifiers["superseded_at"] = created_at or _time.time()
                # Fall through to create new edge; lookup will point to it

        edge = STGEdge(
            source=_src_dn,  # display name on edge (user-facing)
            target=_tgt_dn,
            confidence=confidence,
            strength=strength,
            rule=rule,
            time=time,
            modifiers=modifiers,
            salience=confidence,  # new edges: salience starts at confidence
            session_id=session_id,
            event_id=event_id,
            created_at=created_at if created_at is not None else _time.time(),
            edge_class=modifiers.get("edge_class", "knowledge"),
            delay_k=modifiers.pop("delay_k", 0),
        )

        # Phase 7H: epistemic validation (warnings only, never rejects)
        from stg_engine.epistemic import validate_epistemic_metadata
        warnings = validate_epistemic_metadata(confidence, modifiers)
        if warnings:
            edge.modifiers["_epistemic_warnings"] = warnings

        self._edges.append(edge)
        self._edges_lookup[(_src, _tgt)] = edge
        self._graph.add_edge(_src, _tgt)

        # Supersede detection: flag prior edges that share
        # (source, semantic_field, semantic_value) but have a different target.
        if modifiers.get("edge_class") != "virtual":
            self._flag_suspected_supersede(edge)

        self._invalidate_caches()

        return edge

    def _flag_suspected_supersede(self, new_edge: STGEdge) -> int:
        """Flag prior edges as suspected_supersede when the new edge looks
        like a correction.

        Heuristic: same source, same (semantic_field, semantic_value) pair
        but DIFFERENT target. The older edge is flagged, not deleted.

        Returns the number of edges flagged.
        """
        new_field, new_value = _get_semantic_field(new_edge.modifiers)
        if not new_field or not new_value:
            return 0

        nk = self._nk
        new_src = nk(new_edge.source)
        new_tgt = nk(new_edge.target)
        new_ts = float(getattr(new_edge, "created_at", 0.0) or 0.0)

        flagged = 0
        for other in self._edges:
            if other is new_edge:
                continue
            if other.modifiers.get("edge_class") == "virtual":
                continue
            if other.modifiers.get("virtual_reason"):
                continue
            if nk(other.source) != new_src:
                continue
            if nk(other.target) == new_tgt:
                continue
            other_field, other_value = _get_semantic_field(other.modifiers)
            if other_field != new_field:
                continue
            if str(other_value) != str(new_value):
                continue
            other.modifiers["suspected_supersede"] = True
            other.modifiers["superseded_by"] = str(new_edge.target)
            if new_ts > 0.0:
                other.modifiers["superseded_at"] = new_ts
            flagged += 1
        return flagged

    def merge_edge(self, source: str, target: str, **patch) -> "STGEdge":
        """Patch an existing edge with new modifiers. Memory consolidation."""
        from stg_engine.merge import EdgeMerger
        return EdgeMerger.merge_edge(self, self._nk(source), self._nk(target), **patch)

    def consolidate_edges(self, source: str, target: str):
        """Consolidate multi-edges on (source, target) into one."""
        from stg_engine.merge import EdgeMerger
        return EdgeMerger.consolidate_edges(self, self._nk(source), self._nk(target))

    def remove_edge(self, source: str, target: str) -> bool:
        """Remove an edge from the graph (case-insensitive).

        Args:
            source: Source anchor name
            target: Target anchor name

        Returns:
            True if edge was found and removed
        """
        src, tgt = self._nk(source), self._nk(target)
        key = (src, tgt)
        if key in self._edges_lookup:
            edge = self._edges_lookup.pop(key)
            self._edges.remove(edge)
            if self._graph.has_edge(src, tgt):
                self._graph.remove_edge(src, tgt)
            self._invalidate_caches()
            return True
        return False

    def get_node(self, name: str) -> Optional[STGNode]:
        """Get a node by name (case-insensitive)."""
        return self._nodes.get(self._nk(name))

    def get_edges(
        self,
        source: Optional[str] = None,
        target: Optional[str] = None,
    ) -> List[STGEdge]:
        """Get edges, optionally filtered by source and/or target (case-insensitive)."""
        _nk = self._nk
        src = _nk(source) if source is not None else None
        tgt = _nk(target) if target is not None else None
        if src is not None and tgt is not None:
            return [e for e in self._edges if _nk(e.source) == src and _nk(e.target) == tgt]
        elif src is not None:
            return [e for e in self._edges if _nk(e.source) == src]
        elif tgt is not None:
            return [e for e in self._edges if _nk(e.target) == tgt]
        return list(self._edges)

    def neighbors(self, name: str, direction: str = "out") -> List[str]:
        """Get neighbor node names (case-insensitive lookup).

        Args:
            name: Node name
            direction: 'out' (successors), 'in' (predecessors), 'both'
        """
        key = self._nk(name)
        if key not in self._graph:
            return []

        if direction == "out":
            return [self._dn(n) for n in self._graph.successors(key)]
        elif direction == "in":
            return [self._dn(n) for n in self._graph.predecessors(key)]
        else:
            out = set(self._graph.successors(key))
            inp = set(self._graph.predecessors(key))
            return [self._dn(n) for n in out | inp]

    # ═══════════════════════════════════════════════════════════
    # Episodic Memory
    # ═══════════════════════════════════════════════════════════

    def add_session(self, session: STGSession) -> None:
        """Register a session."""
        self._sessions[session.session_id] = session

    def add_event(self, event: STGEvent) -> None:
        """Register an episodic event."""
        self._events[event.event_id] = event

    def add_tension(self, tension: STGTension) -> None:
        """Register or update a tension."""
        self._tensions[tension.name] = tension

    def update_tension(
        self,
        name: str,
        new_value: float,
        session_id: Optional[str] = None,
        context: Optional[str] = None,
    ) -> None:
        """Update a tension's current value.

        Args:
            name: Tension name
            new_value: New magnitude value
            session_id: Session making the update
            context: Description of why value changed
        """
        if name in self._tensions:
            tension = self._tensions[name]
            tension.current_value = new_value
            if new_value <= 0.05:
                tension.status = "resolved"
                tension.resolved_session = session_id

    def add_belief_evolution(self, evolution: STGBeliefEvolution) -> None:
        """Record a belief evolution (M → M')."""
        self._belief_evolutions.append(evolution)

    # ═══════════════════════════════════════════════════════════
    # STL Import
    # ═══════════════════════════════════════════════════════════

    def ingest_stl(
        self,
        stl_text: str,
        session_id: Optional[str] = None,
        auto_virtual: bool = True,
        created_at: Optional[float] = None,
    ) -> int:
        """Parse STL text and add all statements to the graph.

        Args:
            stl_text: Raw STL text containing statements
            session_id: Optional session to associate edges with
            auto_virtual: If True, auto-create virtual edges between siblings
            created_at: Custom creation timestamp (epoch float). Defaults to current time.

        Returns:
            Number of edges added
        """
        try:
            from stl_parser import validate_llm_output
        except ImportError:
            # Fallback: basic regex parser for simple statements
            return self._ingest_stl_regex(stl_text, session_id, auto_virtual)

        result = validate_llm_output(stl_text)
        # If formal parser returns nothing, fall back to regex
        if not result.statements:
            return self._ingest_stl_regex(stl_text, session_id, auto_virtual, created_at=created_at)

        count = 0
        new_edges: List[STGEdge] = []

        for stmt in result.statements:
            source = str(stmt.source).strip("[]")
            target = str(stmt.target).strip("[]")

            # Parse namespace from "Namespace:Name" format
            src_ns, src_name = self._parse_anchor_name(source)
            tgt_ns, tgt_name = self._parse_anchor_name(target)

            # Extract modifiers
            modifiers = {}
            confidence = 0.5
            strength = 0.5
            rule = None
            time_val = None

            if stmt.modifiers:
                mod_dict = stmt.modifiers.model_dump(exclude_none=True)
                # Pop custom dict and merge
                custom = mod_dict.pop("custom", {})
                modifiers = {**mod_dict, **custom}

                confidence = modifiers.pop("confidence", 0.5)
                strength = modifiers.pop("strength", 0.5)
                rule = modifiers.pop("rule", None)
                time_val = modifiers.pop("time", None)
                # Avoid collision with add_edge() positional params
                modifiers.pop("source", None)
                modifiers.pop("target", None)
                modifiers.pop("session_id", None)
                modifiers.pop("event_id", None)
                # Allow created_at from modifier to override if no explicit param
                mod_created_at = modifiers.pop("created_at", None)
                if mod_created_at is not None and created_at is None:
                    created_at = float(mod_created_at)

                # Auto-parse timestamp modifier into created_at
                timestamp_str = modifiers.get("timestamp")
                if timestamp_str and created_at is None and mod_created_at is None:
                    try:
                        from dateutil.parser import parse as _parse_dt
                        created_at = _parse_dt(timestamp_str).timestamp()
                    except (ValueError, ImportError):
                        pass  # keep timestamp as modifier, don't set created_at

            self.add_node(src_name, namespace=src_ns)
            self.add_node(tgt_name, namespace=tgt_ns)

            edge = self.add_edge(
                source=src_name,
                target=tgt_name,
                confidence=confidence,
                strength=strength,
                rule=rule,
                time=time_val,
                session_id=session_id,
                created_at=created_at,
                **modifiers,
            )
            new_edges.append(edge)
            count += 1

        if auto_virtual and new_edges:
            self._create_virtual_edges_for_siblings(new_edges)

        return count

    def ingest_stl_file(
        self,
        path: str,
        session_id: Optional[str] = None,
        created_at: Optional[float] = None,
        auto_virtual: bool = True,
    ) -> int:
        """Parse a text file containing STL statements and add to graph.

        Accepts any text file (.md, .txt, .stl, etc.) — extension is ignored.

        Args:
            path: Path to file containing STL statements
            session_id: Optional session to associate with
            created_at: Custom creation timestamp (epoch float). Defaults to current time.
            auto_virtual: If True, auto-create virtual edges between siblings.

        Returns:
            Number of edges added
        """
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return self.ingest_stl(content, session_id, auto_virtual=auto_virtual, created_at=created_at)

    def _ingest_stl_regex(
        self,
        stl_text: str,
        session_id: Optional[str] = None,
        auto_virtual: bool = True,
        created_at: Optional[float] = None,
    ) -> int:
        """Fallback STL parser using regex for simple statements.

        Handles: [Source] -> [Target] ::mod(key=value, ...)
        """
        # Pattern for STL statements
        pattern = r'\[([^\]]+)\]\s*(?:→|->)\s*\[([^\]]+)\]'
        mod_pattern = r'::mod\(([^)]*)\)'

        count = 0
        new_edges: List[STGEdge] = []
        for line in stl_text.split("\n"):
            line = line.strip()
            match = re.search(pattern, line)
            if not match:
                continue

            source_raw = match.group(1).strip()
            target_raw = match.group(2).strip()

            src_ns, src_name = self._parse_anchor_name(source_raw)
            tgt_ns, tgt_name = self._parse_anchor_name(target_raw)

            # Extract modifiers
            modifiers = {}
            confidence = 0.5
            strength = 0.5
            rule = None
            time_val = None
            edge_created_at = created_at

            mod_match = re.search(mod_pattern, line)
            if mod_match:
                mod_text = mod_match.group(1)
                modifiers = self._parse_modifier_text(mod_text)
                confidence = float(modifiers.pop("confidence", 0.5))
                strength = float(modifiers.pop("strength", 0.5))
                rule = modifiers.pop("rule", None)
                time_val = modifiers.pop("time", None)
                modifiers.pop("source", None)
                modifiers.pop("target", None)
                modifiers.pop("session_id", None)
                modifiers.pop("event_id", None)
                # Handle created_at from modifier
                mod_created_at = modifiers.pop("created_at", None)
                if mod_created_at is not None and created_at is None:
                    edge_created_at = float(mod_created_at)
                else:
                    edge_created_at = created_at
                # Handle timestamp auto-parse
                timestamp_str = modifiers.get("timestamp")
                if timestamp_str and edge_created_at is None:
                    try:
                        from dateutil.parser import parse as _parse_dt
                        edge_created_at = _parse_dt(timestamp_str).timestamp()
                    except (ValueError, ImportError):
                        pass

            self.add_node(src_name, namespace=src_ns)
            self.add_node(tgt_name, namespace=tgt_ns)

            edge = self.add_edge(
                source=src_name,
                target=tgt_name,
                confidence=confidence,
                strength=strength,
                rule=rule,
                time=time_val,
                session_id=session_id,
                created_at=edge_created_at,
                **modifiers,
            )
            new_edges.append(edge)
            count += 1

        if auto_virtual and new_edges:
            self._create_virtual_edges_for_siblings(new_edges)

        return count

    @staticmethod
    def _parse_anchor_name(raw: str) -> Tuple[Optional[str], str]:
        """Parse 'Namespace:Name' into (namespace, name)."""
        if ":" in raw:
            parts = raw.split(":", 1)
            return parts[0], parts[1]
        return None, raw

    @staticmethod
    def _parse_modifier_text(text: str) -> Dict[str, Any]:
        """Parse modifier key=value pairs from text."""
        result = {}
        # Match key=value or key="value"
        kv_pattern = r'(\w+)\s*=\s*(?:"([^"]*)"|([\d.]+)|(\w+))'
        for match in re.finditer(kv_pattern, text):
            key = match.group(1)
            if match.group(2) is not None:
                result[key] = match.group(2)  # Quoted string
            elif match.group(3) is not None:
                # Try float, fall back to string
                val = match.group(3)
                try:
                    result[key] = float(val)
                except ValueError:
                    result[key] = val
            elif match.group(4) is not None:
                result[key] = match.group(4)  # Bare word
        return result

    # ═══════════════════════════════════════════════════════════
    # Virtual Edges (sibling proximity detection)
    # ═══════════════════════════════════════════════════════════

    def _create_virtual_edges_for_siblings(
        self,
        new_edges: List[STGEdge],
        max_siblings: int = 20,
    ) -> int:
        """Auto-create virtual edges between sibling nodes.

        Siblings = nodes that share a common parent (source) or common
        target. Virtual edges are low-confidence proximity hints, not
        knowledge assertions.

        Args:
            new_edges: Edges just ingested (used to find parents/targets)
            max_siblings: Max children per parent before skipping (防爆)

        Returns:
            Number of virtual edges created
        """
        now = _time.time()
        created = 0

        # Collect parents (sources) and targets from new edges
        parents = set()
        targets = set()
        for e in new_edges:
            if e.modifiers.get("edge_class") == "virtual":
                continue
            parents.add(e.source)
            targets.add(e.target)

        # Pattern 1: Sibling detection — nodes sharing a common parent
        for parent in parents:
            children = [
                e.target for e in self._edges
                if e.source == parent
                and e.modifiers.get("edge_class") != "virtual"
            ]
            if len(children) < 2 or len(children) > max_siblings:
                continue

            # Create virtual edges between siblings (alphabetical order, single direction)
            children_sorted = sorted(set(children))
            for i in range(len(children_sorted)):
                for j in range(i + 1, len(children_sorted)):
                    a, b = children_sorted[i], children_sorted[j]
                    # Skip if any real edge already exists between them
                    if (a, b) in self._edges_lookup or (b, a) in self._edges_lookup:
                        continue
                    self._add_virtual_edge(a, b, "sibling", parent, now)
                    created += 1

        # Pattern 2: Co-source detection — nodes sharing a common target
        for target in targets:
            sources = [
                e.source for e in self._edges
                if e.target == target
                and e.modifiers.get("edge_class") != "virtual"
            ]
            if len(sources) < 2 or len(sources) > max_siblings:
                continue

            sources_sorted = sorted(set(sources))
            for i in range(len(sources_sorted)):
                for j in range(i + 1, len(sources_sorted)):
                    a, b = sources_sorted[i], sources_sorted[j]
                    if (a, b) in self._edges_lookup or (b, a) in self._edges_lookup:
                        continue
                    self._add_virtual_edge(a, b, "co_source", target, now)
                    created += 1

        return created

    def _add_virtual_edge(
        self,
        source: str,
        target: str,
        reason: str,
        parent: str,
        timestamp: float,
    ) -> STGEdge:
        """Create a single virtual edge with standard metadata."""
        return self.add_edge(
            source=source,
            target=target,
            confidence=0.15,
            strength=0.1,
            rule=None,
            edge_class="virtual",
            virtual_reason=reason,
            virtual_parent=parent,
            virtual_created_at=timestamp,
        )

    def rebuild_virtual_edges(self, max_siblings: int = 20) -> int:
        """Clear all virtual edges and rebuild from current graph topology.

        Returns:
            Number of virtual edges created
        """
        self.clear_virtual_edges()

        now = _time.time()
        created = 0

        # Scan all parents (nodes with multiple outgoing real edges)
        parent_children: Dict[str, List[str]] = {}
        target_sources: Dict[str, List[str]] = {}
        for e in self._edges:
            if e.modifiers.get("edge_class") == "virtual":
                continue
            parent_children.setdefault(e.source, []).append(e.target)
            target_sources.setdefault(e.target, []).append(e.source)

        # Sibling pattern
        for parent, children in parent_children.items():
            children_uniq = sorted(set(children))
            if len(children_uniq) < 2 or len(children_uniq) > max_siblings:
                continue
            for i in range(len(children_uniq)):
                for j in range(i + 1, len(children_uniq)):
                    a, b = children_uniq[i], children_uniq[j]
                    if (a, b) in self._edges_lookup or (b, a) in self._edges_lookup:
                        continue
                    self._add_virtual_edge(a, b, "sibling", parent, now)
                    created += 1

        # Co-source pattern
        for target, sources in target_sources.items():
            sources_uniq = sorted(set(sources))
            if len(sources_uniq) < 2 or len(sources_uniq) > max_siblings:
                continue
            for i in range(len(sources_uniq)):
                for j in range(i + 1, len(sources_uniq)):
                    a, b = sources_uniq[i], sources_uniq[j]
                    if (a, b) in self._edges_lookup or (b, a) in self._edges_lookup:
                        continue
                    self._add_virtual_edge(a, b, "co_source", target, now)
                    created += 1

        return created

    def clear_virtual_edges(self) -> int:
        """Remove all virtual edges from the graph.

        Returns:
            Number of virtual edges removed
        """
        to_remove = [
            (e.source, e.target) for e in self._edges
            if e.modifiers.get("edge_class") == "virtual"
        ]
        for src, tgt in to_remove:
            self.remove_edge(src, tgt)
        return len(to_remove)

    def get_virtual_edge_stats(self) -> Dict[str, Any]:
        """Return statistics about virtual edges."""
        virtual_edges = [
            e for e in self._edges
            if e.modifiers.get("edge_class") == "virtual"
        ]
        real_edges = [
            e for e in self._edges
            if e.modifiers.get("edge_class") != "virtual"
        ]

        reason_dist: Dict[str, int] = {}
        for e in virtual_edges:
            reason = e.modifiers.get("virtual_reason", "unknown")
            reason_dist[reason] = reason_dist.get(reason, 0) + 1

        return {
            "total_edges": len(self._edges),
            "real_edges": len(real_edges),
            "virtual_edges": len(virtual_edges),
            "reason_distribution": reason_dist,
        }

    # ═══════════════════════════════════════════════════════════
    # Built-in Computations
    # ═══════════════════════════════════════════════════════════

    def compute_psi(self) -> float:
        """Calculate Ψ (Mental Stability).

        Ψ = Structural_Coherence / max(Max_Tension, ε) * Epistemic_Confidence

        Returns:
            Ψ value (higher = more stable)
        """
        return compute_psi(self._graph, self._nodes, self._edges)

    def compute_path_tension(self, source: str, target: str) -> float:
        """Calculate total tension along path from source to target.

        Args:
            source: Source node name
            target: Target node name

        Returns:
            Path tension value. -1.0 if no path exists.
        """
        return _compute_path_tension(
            self._graph, self._edges_lookup, self._nk(source), self._nk(target)
        )

    def compute_all_tensions(self) -> Dict[str, float]:
        """Recompute tension for every node in the graph.

        Updates each node's .tension attribute in-place.

        Returns:
            Map of node_name -> tension_value
        """
        tensions = {}
        for name, node in self._nodes.items():
            t = compute_node_tension(self._graph, self._edges_lookup, name)
            node.tension = t
            tensions[name] = t
        return tensions

    def compute_activations(
        self,
        alpha: float = 0.3,
        beta: float = 0.4,
        gamma: float = 0.2,
        delta: float = 0.1,
    ) -> Dict[str, float]:
        """Compute activation scores for all nodes.

        Updates each node's .activation attribute in-place.

        Returns:
            Map of node_name -> activation_value
        """
        activations = {}
        for name, node in self._nodes.items():
            a = _compute_activation(
                self._graph, self._edges_lookup, node,
                alpha, beta, gamma, delta,
            )
            node.activation = a
            activations[name] = a
        return activations

    def compute_reward(
        self,
        psi_before: float,
        psi_after: float,
        tension_resolved: float,
        edges_traversed: int,
    ) -> float:
        """Calculate intrinsic reward for a reasoning action.

        Args:
            psi_before: Ψ before the action
            psi_after: Ψ after the action
            tension_resolved: Total tension reduction
            edges_traversed: Number of edges used

        Returns:
            Reward value (can be negative)
        """
        return compute_intrinsic_reward(
            psi_before, psi_after, tension_resolved, float(edges_traversed)
        )

    def propagate(
        self,
        input_text: str,
        initial_activation: float = 1.0,
        decay: float = 0.65,
        iterations: int = 5,
        threshold: float = 0.10,
        normalize: bool = True,
    ) -> List[str]:
        """Activate nodes matching input and propagate through graph.

        Args:
            input_text: User input to match against node names
            initial_activation: Starting activation level
            decay: Decay factor per iteration
            iterations: Number of propagation iterations
            threshold: Minimum activation to include in results
            normalize: If True, enforce global activation budget (Braitenberg
                Vehicle 12 threshold control). Total activation is conserved
                so nodes compete for limited activation resources.

        Returns:
            List of activated node names, sorted by activation descending
        """
        # Tokenize with stop word and short token filtering
        _stop = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "am", "do", "does", "did", "has", "have", "had", "it", "its",
            "what", "who", "how", "why", "when", "where", "which",
            "to", "of", "in", "on", "at", "by", "for", "with", "from",
            "and", "or", "not", "no", "if", "but", "so", "as", "than",
            "me", "my", "we", "us", "you", "he", "she", "they", "them",
            "this", "that", "these", "those", "about", "tell", "describe",
            "explain", "can", "could", "would", "should", "will",
            # Common generic words that cause false seed matches
            "first", "last", "second", "third", "next", "new", "old",
            "set", "sets", "get", "gets", "got", "put", "take", "took",
            "make", "made", "give", "gave", "come", "came", "go", "went",
            "one", "two", "three", "also", "just", "even", "still",
            "most", "more", "much", "many", "some", "any", "all", "each",
            "very", "own", "same", "other", "such", "only", "back",
            "after", "before", "between", "through", "over", "under",
            "into", "out", "up", "down", "off", "then", "now", "here",
            "there", "way", "well", "part", "like", "being", "both",
            "may", "might", "must", "shall", "his", "her", "our", "your",
            "their", "its", "him", "itself", "never", "always", "often",
        }
        raw = input_text.lower().split()
        # Split compound tokens (hyphen, underscore) into parts too
        # and strip trailing punctuation from each part
        expanded = []
        for t in raw:
            parts = re.split(r'[_\-:]', t)
            expanded.extend(
                re.sub(r'[^a-z0-9\u4e00-\u9fff\u3400-\u4dbf]+$', '', p)
                for p in parts if p
            )
            # Also extract CJK sequences from mixed text
            cjk_seqs = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]+', t)
            for seq in cjk_seqs:
                if seq not in expanded:
                    expanded.append(seq)
        # CJK chars are semantically meaningful at len=1, so only filter len<2 for ASCII
        _has_cjk = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')
        tokens = [t for t in expanded if (len(t) >= 2 or _has_cjk.search(t)) and t not in _stop]
        if not tokens:
            # Fallback 1: non-stop words of any length
            tokens = [t for t in expanded if t not in _stop]
        if not tokens:
            # Fallback 2: all words (for single-char node names in tests)
            tokens = expanded if expanded else raw
        if not tokens:
            return []

        # Find matching nodes
        # Long tokens (>= 2 chars): word boundary matching
        # Short tokens (1 char): substring matching (fallback for A/B/C nodes)
        long_tokens = [t for t in tokens if len(t) >= 2]
        short_tokens = [t for t in tokens if len(t) < 2]
        # Track (node_name, hit_count, matched_tokens) for IDF scoring
        matching_hits: List[Tuple[str, int, List[str]]] = []
        for name in self._nodes:
            name_lower = name.lower()
            hit_count = 0
            # Short token: substring match (backward compat for single-char nodes)
            if short_tokens:
                short_matched = [t for t in short_tokens if t in name_lower]
                if short_matched:
                    matching_hits.append((name, len(short_matched), short_matched))
                    continue
            # Long token: word boundary match with hit counting
            if long_tokens:
                name_parts = re.split(r'[_:\-]', name_lower)
                words = []
                for p in name_parts:
                    # Latin/digit words
                    words.extend(
                        w.lower() for w in re.findall(r'[a-z]+|[A-Z][a-z]*|\d+', p)
                        if len(w) >= 2
                    )
                    # CJK characters: each char is a word, also keep full string
                    cjk_chars = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]+', p)
                    for cjk in cjk_chars:
                        words.append(cjk)  # full string (e.g. "贾宝玉")
                        words.extend(cjk)  # individual chars (e.g. "贾","宝","玉")
                # Morphological prefix matching: only allow prefix match when
                # the suffix is a known English morphological ending.
                # This prevents false matches like "attic" → "atticus"
                # while preserving valid ones like "mad" → "madness".
                _morph_suffixes = (
                    "s", "es", "ed", "ing", "er", "est", "ly",
                    "ness", "ment", "tion", "sion", "ation",
                    "ous", "ious", "ful", "less", "able", "ible",
                    "ive", "al", "ial", "ical", "ity", "ty",
                    "ence", "ance", "dom", "ship", "ism", "ist",
                    "ize", "ise", "ify", "en",
                )

                def _is_morph_prefix(shorter: str, longer: str) -> bool:
                    """Check if shorter is a morphological prefix of longer."""
                    if not longer.startswith(shorter):
                        return False
                    suffix = longer[len(shorter):]
                    if not suffix:
                        return True  # exact match
                    # Short stems (<=3 chars) are too ambiguous for prefix
                    # matching (e.g. "mr" → "mrs", "set" → "setting")
                    if len(shorter) <= 3:
                        return False
                    return suffix in _morph_suffixes

                # Per-token matching with IDF tracking
                matched_tokens = []
                for token in long_tokens:
                    if token in words or any(
                        _is_morph_prefix(token, w) or _is_morph_prefix(w, token)
                        for w in words
                    ):
                        matched_tokens.append(token)
                hit_count = len(matched_tokens)
                # CJK substring match: "宝玉" should match node "贾宝玉"
                if hit_count == 0:
                    cjk_tokens = [t for t in long_tokens if _has_cjk.search(t)]
                    for ct in cjk_tokens:
                        if ct in name_lower:
                            hit_count += 1
                            matched_tokens.append(ct)
                if hit_count > 0:
                    matching_hits.append((name, hit_count, matched_tokens))

        # Compute IDF: token → log(N / (1 + df)) where df = nodes matching token
        _N = max(1, len(self._nodes))
        _token_df: Dict[str, int] = {}
        for _, _, mtokens in matching_hits:
            for tk in mtokens:
                _token_df[tk] = _token_df.get(tk, 0) + 1
        _log = __import__("math").log
        _token_idf = {tk: _log(_N / (1 + df)) for tk, df in _token_df.items()}

        # IDF-weighted hit score: sum of IDF weights of matched tokens
        matching_scored: List[Tuple[str, float, int]] = []
        for name, hit_count, mtokens in matching_hits:
            idf_score = sum(_token_idf.get(tk, 0.0) for tk in mtokens)
            matching_scored.append((name, idf_score, hit_count))

        # Sort by IDF score descending — discriminative matches first
        matching_scored.sort(key=lambda x: x[1], reverse=True)

        # Cap seeds: if too many matches, keep top N by IDF score (then importance)
        max_seeds = min(50, max(20, len(self._nodes) // 100))
        if len(matching_scored) > max_seeds:
            imp = self.get_importance_field()
            matching_scored.sort(
                key=lambda x: (x[1], imp.get(x[0], 0.0)), reverse=True
            )
            matching_scored = matching_scored[:max_seeds]

        if not matching_scored:
            return []

        # Initialize activation with IDF-weighted scoring
        # idf_ratio (primary): IDF score relative to max — discriminative matches rank higher
        # name_precision (secondary): shorter/simpler names get minor boost
        # gravity (optional): elevation factor suppresses low-elevation seeds
        max_idf = max(s for _, s, _ in matching_scored) if matching_scored else 1.0
        max_idf = max(max_idf, 0.01)  # avoid div by zero
        gravity_elevations = None
        if self._gravity_map is not None:
            gravity_elevations = self._gravity_map.node_elevation
        activation_map: Dict[str, float] = {}
        for name, idf_score, hit_count in matching_scored:
            parts = re.split(r'[_:\-]', name.lower())
            word_count = max(1, len(parts))
            name_precision = min(1.0, 1.0 / word_count)
            idf_ratio = idf_score / max_idf
            # IDF ratio dominates (80%), name_precision is tiebreaker (20%)
            # Use idf_ratio^2 for steeper differentiation
            score = max(0.1, (idf_ratio ** 2) * 0.8 + name_precision * 0.2)
            # Gravity: scale seed activation by sqrt(elevation)
            if gravity_elevations is not None:
                elev = gravity_elevations.get(name, 0.01)
                score *= max(0.05, elev ** 0.5)
            activation_map[name] = initial_activation * score

        return self._propagate_from_seeds(
            activation_map=activation_map,
            decay=decay,
            iterations=iterations,
            threshold=threshold,
            normalize=normalize,
            input_text=input_text,
            token_count=len(tokens),
            seed_count=len(matching_hits),
        )

    def _propagate_from_seeds(
        self,
        activation_map: Dict[str, float],
        decay: float = 0.65,
        iterations: int = 5,
        threshold: float = 0.10,
        normalize: bool = True,
        input_text: str = "",
        token_count: int = 0,
        seed_count: int = 0,
    ) -> List[str]:
        """Core propagation loop from pre-built activation seeds.

        Extracted from propagate() for reuse by perceive_and_propagate().
        All parameters and behavior identical to the original inner loop.
        """
        # Get importance field if importance bias is enabled
        importance = None
        if self.importance_weight > 0:
            importance = self.get_importance_field()

        # Global activation budget (Braitenberg Vehicle 12 threshold control)
        activation_budget = sum(activation_map.values()) if normalize else None

        # ─── Rust fast path ───────────────────────────────────────
        # Use the compiled Rust core when:
        #   - It's available (stg-engine wheel installed)
        #   - importance/preference biases are off (default)
        inh = self._inhibition_config

        # Build flat edges list for Rust
        _rust_edges: List[Tuple[str, str, float, float, bool]] = []
        for (src, tgt), edge in self._edges_lookup.items():
            conf = edge.confidence
            sal = edge.salience
            is_virtual = edge.modifiers.get("edge_class") == "virtual"
            _rust_edges.append((src, tgt, conf, sal, is_virtual))

        activation_map = _rust.propagate_inner_loop(
            activation_map,
            _rust_edges,
            decay,
            iterations,
            bool(normalize),
        )

        # Phase 9: Refractory period (post-loop)
        if inh.enabled and inh.refractory:
            from stg_engine.inhibition import apply_refractory
            apply_refractory(
                activation_map, self._refractory_set,
                inh.refractory_decay, inh.refractory_suppression,
            )

        # Phase 9: Community inhibition (post-loop)
        if inh.enabled and inh.community_inhibition:
            from stg_engine.inhibition import community_inhibition
            community_inhibition(
                activation_map, self, inh.community_suppression
            )

        # Phase 9: Adaptive threshold
        effective_threshold = threshold
        if inh.enabled and inh.adaptive_threshold:
            from stg_engine.inhibition import adaptive_threshold as _adapt_thresh
            effective_threshold = _adapt_thresh(
                activation_map, threshold, inh.threshold_gain
            )

        # Update node activation state
        for name, act in activation_map.items():
            if name in self._nodes:
                self._nodes[name].activation = act

        # Phase 9: Update refractory set for next propagation
        if inh.enabled and inh.refractory:
            for name, act in activation_map.items():
                if act > threshold:
                    self._refractory_set[name] = act

        # Return activated nodes above threshold
        activated = [
            (name, act) for name, act in activation_map.items()
            if act > effective_threshold
        ]
        activated.sort(key=lambda x: x[1], reverse=True)

        # Compute and store propagation metrics (Phase 7A)
        from stg_engine.metrics import (
            query_efficiency as _qe,
            resonance_score as _rs,
        )
        # RS uses only above-threshold nodes (signal, not noise)
        total_act = sum(act for _, act in activated)
        max_act = activated[0][1] if activated else 0.0
        self._last_propagation_metrics = PropagationMetrics(
            input_text=input_text,
            token_count=token_count,
            seed_node_count=seed_count,
            activated_node_count=len(activated),
            total_activation=total_act,
            max_activation=max_act,
            iterations_used=iterations,
            query_efficiency=_qe(seed_count, len(activated), len(self._nodes)),
            resonance_score=_rs(max_act, total_act),
            coverage=len(activated) / len(self._nodes) if self._nodes else 0.0,
            top_nodes=[(self._dn(n), a) for n, a in activated[:10]],
        )

        # Hebbian learning hook (Phase 7B)
        learning_events = []
        if self._learner is not None:
            learning_events = self._learner.learn_from_propagation(
                self, activation_map
            )
            self._learning_log.extend(learning_events)

        # Telemetry hook (Phase 10)
        if self._telemetry is not None:
            strengthen_count = sum(
                1 for e in learning_events if e.event_type == "strengthen"
            )
            weaken_count = sum(
                1 for e in learning_events if e.event_type == "weaken"
            )
            self._telemetry.record_propagation(
                self._last_propagation_metrics, activation_map,
                strengthen_count, weaken_count,
            )
            if learning_events:
                self._telemetry.record_edge_mutations(learning_events)

        return [self._dn(name) for name, _ in activated]

    # ═══════════════════════════════════════════════════════════
    # Perception (Phase 12)
    # ═══════════════════════════════════════════════════════════

    def perceive(
        self,
        grid: List[List[int]],
        game_id: Optional[str] = None,
        step_number: int = 0,
        level: int = 0,
    ) -> Tuple[str, List[str]]:
        """Perceive a grid frame and find similar past states.

        Creates a Visual:frame_{hash} node in the graph and indexes its
        feature vector for similarity search.

        Args:
            grid: HxW grid with integer color values (e.g., 64x64, 0-15)
            game_id: Optional game identifier
            step_number: Step within the game
            level: Current game level

        Returns:
            (frame_hash, list of similar node names)
        """
        from stg_engine.perception import perceive_frame, find_similar_states
        fhash, _ = perceive_frame(self, grid, game_id, step_number, level)
        similar = find_similar_states(self, grid, top_k=5)
        # Exclude self from similar results
        similar_names = [name for name, sim in similar if not name.endswith(fhash)]
        return fhash, similar_names

    def perceive_and_propagate(
        self,
        grid: List[List[int]],
        top_k: int = 5,
        initial_activation: float = 1.0,
        decay: float = 0.65,
        iterations: int = 5,
        threshold: float = 0.10,
        normalize: bool = True,
    ) -> List[str]:
        """Perceive grid, seed similar past states, propagate activation.

        Combines visual perception with graph propagation:
        1. Extract visual features from grid
        2. Find top-k most similar past frames
        3. Seed those frames' nodes with activation (weighted by similarity)
        4. Run standard propagation from those seeds

        Args:
            grid: HxW grid with integer color values
            top_k: Number of similar past frames to seed
            initial_activation: Base activation for seeds
            decay: Per-iteration decay
            iterations: Propagation rounds
            threshold: Minimum activation for results
            normalize: Enforce activation budget

        Returns:
            List of activated node names, sorted by activation descending
        """
        from stg_engine.perception import (
            extract_features, find_similar_states,
            build_fixed_filters, perceive_frame,
        )

        # Ensure frame is indexed
        perceive_frame(self, grid)

        # Find similar past frames
        similar = find_similar_states(self, grid, top_k=top_k)

        if not similar:
            return []

        # Build activation map: seed similar frames weighted by similarity
        activation_map: Dict[str, float] = {}
        for node_name, similarity in similar:
            activation_map[node_name] = initial_activation * similarity

        return self._propagate_from_seeds(
            activation_map=activation_map,
            decay=decay,
            iterations=iterations,
            threshold=threshold,
            normalize=normalize,
            input_text=f"[visual:{len(grid)}x{len(grid[0]) if grid else 0}]",
            token_count=0,
            seed_count=len(similar),
        )

    def convergent_propagate(
        self,
        input_text: str,
        top_k: int = 5,
        max_iterations: int = 5,
        convergence_threshold: float = 0.8,
        **propagate_kwargs,
    ) -> "ConvergenceResult":
        """Iterative propagation until convergence (Kanerva F5).

        Feeds propagation output back as input, refining until the top-k
        nodes stabilize. Like SDM's iterative read operation.

        Args:
            input_text: Query text
            top_k: Number of top nodes to track for convergence
            max_iterations: Safety cap on iterations
            convergence_threshold: Jaccard similarity for convergence (0-1)
            **propagate_kwargs: Passed to propagate()

        Returns:
            ConvergenceResult with final top nodes and convergence info
        """
        from stg_engine.kanerva import IterativePropagator
        propagator = IterativePropagator(
            top_k=top_k,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
        )
        return propagator.converge(self, input_text, **propagate_kwargs)

    def take_snapshot(self, session_id: Optional[str] = None) -> SystemSnapshot:
        """Take a snapshot of current system state.

        Used for ΔΨ tracking and intrinsic reward computation.
        """
        snapshot = SystemSnapshot(
            session_id=session_id,
            psi_value=self.compute_psi(),
            max_tension=max(
                (n.tension for n in self._nodes.values()), default=0.0
            ),
            structural_coherence=self._compute_structural_coherence(),
            epistemic_confidence=self._compute_epistemic_confidence(),
            node_count=len(self._nodes),
            edge_count=len(self._edges),
        )
        self._snapshots.append(snapshot)
        return snapshot

    def _compute_structural_coherence(self) -> float:
        """Ratio of high-confidence edges to total real edges."""
        real = [e for e in self._edges if e.modifiers.get("edge_class") != "virtual"]
        if not real:
            return 1.0
        high = sum(1 for e in real if e.confidence >= 0.8)
        return high / len(real)

    def _compute_epistemic_confidence(self) -> float:
        """Average confidence across all real edges."""
        real = [e for e in self._edges if e.modifiers.get("edge_class") != "virtual"]
        if not real:
            return 1.0
        return sum(e.confidence for e in real) / len(real)

    # ═══════════════════════════════════════════════════════════
    # Query Interface
    # ═══════════════════════════════════════════════════════════

    def query_nodes(
        self,
        name_pattern: Optional[str] = None,
        anchor_type: Optional[str] = None,
        min_tension: Optional[float] = None,
        limit: int = 50,
    ) -> List[STGNode]:
        """Query nodes with filters.

        Args:
            name_pattern: Substring match on node name (case-insensitive)
            anchor_type: Filter by anchor type
            min_tension: Minimum tension value
            limit: Max results

        Returns:
            Matching nodes
        """
        results = list(self._nodes.values())

        if name_pattern:
            pattern_lower = name_pattern.lower()
            results = [n for n in results if pattern_lower in n.name.lower()]

        if anchor_type:
            results = [n for n in results if n.anchor_type == anchor_type]

        if min_tension is not None:
            results = [n for n in results if n.tension >= min_tension]

        return results[:limit]

    def query_edges(
        self,
        source: Optional[str] = None,
        target: Optional[str] = None,
        min_confidence: Optional[float] = None,
        rule: Optional[str] = None,
        session_id: Optional[str] = None,
        # Phase 7H: epistemic filters
        edge_class: Optional[str] = None,
        trace_type: Optional[str] = None,
        verification_status: Optional[str] = None,
        epistemic_status: Optional[str] = None,
        scope: Optional[str] = None,
        min_structural_coherence: Optional[float] = None,
        limit: int = 100,
    ) -> List[STGEdge]:
        """Query edges with filters.

        Args:
            source: Filter by source anchor name
            target: Filter by target anchor name
            min_confidence: Minimum confidence threshold
            rule: Filter by rule type
            session_id: Filter by session ID
            edge_class: Filter by edge class (cognitive/knowledge/structural)
            trace_type: Filter by trace type (EarthTrace/CosmicTrace/UserClaimed)
            verification_status: Filter by verification status
            epistemic_status: Filter by epistemic status
            scope: Filter by scope
            min_structural_coherence: Minimum structural coherence threshold
            limit: Maximum results to return
        """
        results = list(self._edges)

        if source:
            results = [e for e in results if e.source == source]
        if target:
            results = [e for e in results if e.target == target]
        if min_confidence is not None:
            results = [e for e in results if e.confidence >= min_confidence]
        if rule:
            results = [e for e in results if e.rule == rule]
        if session_id:
            results = [e for e in results if e.session_id == session_id]

        # Phase 7H: epistemic filters — all read from modifiers dict
        if edge_class:
            results = [e for e in results
                       if e.modifiers.get("edge_class", "structural") == edge_class]
        if trace_type:
            results = [e for e in results
                       if e.modifiers.get("trace_type") == trace_type]
        if verification_status:
            results = [e for e in results
                       if e.modifiers.get("verification_status") == verification_status]
        if epistemic_status:
            results = [e for e in results
                       if e.modifiers.get("epistemic_status") == epistemic_status]
        if scope:
            results = [e for e in results
                       if e.modifiers.get("scope") == scope]
        if min_structural_coherence is not None:
            results = [e for e in results
                       if (e.modifiers.get("structural_coherence") or 0.0)
                       >= min_structural_coherence]

        return results[:limit]

    def epistemic_summary(self) -> Dict[str, Any]:
        """Return epistemic composition of the graph.

        Returns:
            Dict with distribution counts for edge_class, trace_type,
            verification_status, epistemic_status, scope.
        """
        from stg_engine.epistemic import epistemic_summary as _epistemic_summary
        return _epistemic_summary(self._edges)

    def query_by_session(self, session_id: str) -> Dict[str, Any]:
        """Get all data associated with a session.

        Returns dict with edges, events, session info.
        """
        session = self._sessions.get(session_id)
        edges = [e for e in self._edges if e.session_id == session_id]
        events = [
            ev for ev in self._events.values()
            if ev.session_id == session_id
        ]
        return {
            "session": session.to_dict() if session else None,
            "edges": [e.to_dict() for e in edges],
            "events": [ev.to_dict() for ev in events],
        }

    def query_events(
        self,
        session_id: Optional[str] = None,
        min_importance: Optional[float] = None,
        event_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[STGEvent]:
        """Query episodic events."""
        results = list(self._events.values())

        if session_id:
            results = [e for e in results if e.session_id == session_id]
        if min_importance is not None:
            results = [
                e for e in results if e.importance_score >= min_importance
            ]
        if event_type:
            results = [e for e in results if e.event_type == event_type]

        # Sort by importance descending
        results.sort(key=lambda e: e.importance_score, reverse=True)
        return results[:limit]

    def query_tensions(self, status: Optional[str] = None) -> List[STGTension]:
        """Query tensions, optionally filtered by status.

        Args:
            status: 'active', 'resolved', 'persisting', or None for all
        """
        results = list(self._tensions.values())
        if status:
            results = [t for t in results if t.status == status]
        # Sort by current_value descending
        results.sort(key=lambda t: t.current_value, reverse=True)
        return results

    def find_paths(
        self,
        source: str,
        target: str,
        max_depth: int = 5,
    ) -> List[List[str]]:
        """Find all simple paths between two nodes.

        Args:
            source: Source node name
            target: Target node name
            max_depth: Maximum path length

        Returns:
            List of paths, where each path is a list of node names
        """
        src, tgt = self._nk(source), self._nk(target)
        if src not in self._graph or tgt not in self._graph:
            return []

        try:
            paths = list(
                nx.all_simple_paths(self._graph, src, tgt, cutoff=max_depth)
            )
            # Return display names
            return [[self._dn(n) for n in path] for path in paths]
        except nx.NodeNotFound:
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        virtual_count = sum(
            1 for e in self._edges
            if e.modifiers.get("edge_class") == "virtual"
        )
        total_edges = len(self._edges)
        return {
            "node_count": len(self._nodes),
            "edge_count": total_edges,
            "real_edge_count": total_edges - virtual_count,
            "virtual_edge_count": virtual_count,
            "session_count": len(self._sessions),
            "event_count": len(self._events),
            "active_tensions": sum(
                1 for t in self._tensions.values() if t.status == "active"
            ),
            "total_tensions": len(self._tensions),
            "belief_evolutions": len(self._belief_evolutions),
            "snapshots": len(self._snapshots),
            "psi": self.compute_psi(),
            "graph_density": nx.density(self._graph) if self._nodes else 0.0,
        }

    # ═══════════════════════════════════════════════════════════
    # Metrics (Phase 7A)
    # ═══════════════════════════════════════════════════════════

    @property
    def last_propagation_metrics(self) -> Optional[PropagationMetrics]:
        """Metrics from the most recent propagate() call.

        Returns None if propagate() has not been called yet.
        """
        return self._last_propagation_metrics

    def get_metrics(self) -> GraphMetrics:
        """Compute graph-level health metrics.

        Cached until graph is mutated (add/remove node/edge).

        Returns:
            GraphMetrics with topology, entropy, confidence, connectivity stats.
        """
        if self._graph_metrics_cache is None:
            from stg_engine.metrics import compute_graph_metrics
            self._graph_metrics_cache = compute_graph_metrics(
                self._graph, self._nodes, self._edges, self._edges_lookup
            )
        return self._graph_metrics_cache

    def get_importance_field(self) -> Dict[str, float]:
        """Compute PageRank-style importance for all nodes.

        Cached until graph is mutated.

        Returns:
            Dict mapping node_name -> importance score (sums to ~1.0).
        """
        if self._importance_cache is None:
            from stg_engine.metrics import compute_importance_field
            self._importance_cache = compute_importance_field(
                self._graph, self._edges_lookup
            )
        return self._importance_cache

    def get_gravity_map(self):
        """Get or build the gravity map (multi-resolution community structure).

        Cached until graph is mutated.
        """
        if self._gravity_map is None:
            from stg_engine.gravity import build_gravity_map
            self._gravity_map = build_gravity_map(self)
        return self._gravity_map

    # ═══════════════════════════════════════════════════════════
    # Learning (Phase 7B)
    # ═══════════════════════════════════════════════════════════

    def enable_learning(self, **kwargs) -> None:
        """Enable Hebbian learning after each propagate() call.

        Args:
            **kwargs: Passed to HebbianLearner constructor
                (strengthen_rate, weaken_rate, confidence_floor,
                 confidence_ceiling, activation_threshold)
        """
        from stg_engine.learning import HebbianLearner
        self._learner = HebbianLearner(**kwargs)

    def disable_learning(self) -> None:
        """Disable auto-learning after propagate()."""
        self._learner = None

    @property
    def learning_enabled(self) -> bool:
        """True if Hebbian learning is active."""
        return self._learner is not None

    def enable_telemetry(self, **kwargs) -> None:
        """Enable telemetry collection during propagate().

        Args:
            **kwargs: Passed to TelemetryCollector constructor
                (max_propagations, max_sessions, max_mutations, mutation_threshold)
        """
        from stg_engine.telemetry import TelemetryCollector
        self._telemetry = TelemetryCollector(**kwargs)

    def disable_telemetry(self) -> None:
        """Disable telemetry collection."""
        self._telemetry = None

    @property
    def telemetry_enabled(self) -> bool:
        """True if telemetry collection is active."""
        return self._telemetry is not None

    @property
    def learning_log(self) -> List:
        """All learning events since last clear."""
        return self._learning_log

    def learn_from_path(
        self,
        path: List[str],
        strength: float = 1.0,
        **kwargs,
    ) -> List:
        """Explicitly strengthen a path.

        Works even if auto-learning is disabled.

        Args:
            path: List of node names forming a path
            strength: Modulation factor (0.0-1.0)
            **kwargs: HebbianLearner params if no learner attached

        Returns:
            List of LearningEvent records.
        """
        from stg_engine.learning import HebbianLearner
        learner = self._learner or HebbianLearner(**kwargs)
        events = learner.learn_from_path(self, path, strength)
        self._learning_log.extend(events)
        return events

    def prune(self, stg_path: str = None, **kwargs) -> List:
        """Run one synaptic pruning cycle.

        Args:
            stg_path: Path to .stg file for persistent audit logging.
            **kwargs: Passed to SynapticPruner constructor
                (confidence_threshold, unused_days, eid_safety_threshold)

        Returns:
            List of LearningEvent records.
        """
        from stg_engine.learning import SynapticPruner
        pruner = SynapticPruner(**kwargs)
        events = pruner.prune(self, stg_path=stg_path)
        self._learning_log.extend(events)
        return events

    # ═══════════════════════════════════════════════════════════
    # Topology Optimization (Phase 7C)
    # ═══════════════════════════════════════════════════════════

    def analyze_topology(self, **kwargs):
        """Analyze graph topology without modification.

        Args:
            **kwargs: Passed to TopologyOptimizer constructor.

        Returns:
            TopologyReport with communities, bridges, redundancy analysis.
        """
        from stg_engine.topology import TopologyOptimizer
        optimizer = TopologyOptimizer(**kwargs)
        return optimizer.analyze(self)

    def optimize_topology(self, **kwargs):
        """Analyze and optimize graph topology.

        Args:
            **kwargs: Passed to TopologyOptimizer constructor.
                Also accepts apply_bridges (bool) and apply_redundancy (bool).

        Returns:
            TopologyReport (pre-optimization snapshot).
        """
        from stg_engine.topology import TopologyOptimizer
        apply_bridges = kwargs.pop("apply_bridges", True)
        apply_redundancy = kwargs.pop("apply_redundancy", True)
        optimizer = TopologyOptimizer(**kwargs)
        return optimizer.optimize(
            self,
            apply_bridges=apply_bridges,
            apply_redundancy=apply_redundancy,
        )

    # ═══════════════════════════════════════════════════════════
    # Cognitive Architecture (Phase 7D)
    # ═══════════════════════════════════════════════════════════

    def enable_cognitive(self, **kwargs) -> None:
        """Enable cognitive architecture modules."""
        from stg_engine.cognitive import CognitiveArchitecture
        self._cognitive = CognitiveArchitecture(**kwargs)

    @property
    def cognitive_enabled(self) -> bool:
        """Whether cognitive architecture is active."""
        return self._cognitive is not None

    def add_goal(self, name: str, keywords: List[str], priority: float = 1.0):
        """Add an active goal biasing propagation."""
        if not self._cognitive:
            self.enable_cognitive()
        return self._cognitive.goals.add_goal(name, keywords, priority)

    def get_self_model(self):
        """Build self-model report of knowledge landscape."""
        if not self._cognitive:
            self.enable_cognitive()
        return self._cognitive.self_model.build(self)

    def generate_hypotheses(self, **kwargs):
        """Generate link-prediction hypotheses."""
        if not self._cognitive:
            self.enable_cognitive()
        return self._cognitive.hypotheses.generate(self)

    def apply_temporal(self, **kwargs):
        """Apply temporal dynamics to all nodes."""
        if not self._cognitive:
            self.enable_cognitive()
        return self._cognitive.temporal.apply(self)

    # ═══════════════════════════════════════════════════════════
    # Feedback Loops (Phase 7E)
    # ═══════════════════════════════════════════════════════════

    def enable_feedback(self, config=None, stg_path: str = None, **kwargs) -> None:
        """Enable feedback loop manager."""
        from stg_engine.feedback import FeedbackLoopManager
        if not self.cognitive_enabled:
            self.enable_cognitive()
        self._feedback = FeedbackLoopManager(config=config, stg_path=stg_path, **kwargs)

    @property
    def feedback_enabled(self) -> bool:
        """Whether feedback loop manager is active."""
        return self._feedback is not None

    def pre_turn(self, context: str) -> dict:
        """Prepare STG for a new turn (temporal + warmup)."""
        if not self._feedback:
            self.enable_feedback()
        return self._feedback.pre_turn(self, context)

    def post_turn(self, query: str, results: list, success: bool) -> dict:
        """Consolidate learning after a turn (hebbian + router feedback)."""
        if not self._feedback:
            self.enable_feedback()
        return self._feedback.post_turn(self, query, results, success)

    def session_end(self) -> dict:
        """Session-end cleanup (pruning). Call before save()."""
        if not self._feedback:
            return {}
        return self._feedback.session_end(self)

    @property
    def feedback_stats(self):
        """Feedback loop cumulative statistics."""
        if self._feedback:
            return self._feedback.get_stats()
        return None

    # ═══════════════════════════════════════════════════════════
    # Phase 7F: Validation & Benchmarking
    # ═══════════════════════════════════════════════════════════

    def run_benchmark(self, **kwargs) -> "BenchmarkReport":
        """Run full benchmark suite against success criteria."""
        from stg_engine.benchmark import STGBenchmark
        bench = STGBenchmark(self)
        return bench.run_all(**kwargs)

    # ═══════════════════════════════════════════════════════════
    # Semantic Search (Phase 7G)
    # ═══════════════════════════════════════════════════════════

    def search(
        self,
        query: str,
        top_k: int = 10,
        propagate: bool = True,
        min_similarity: float = 0.3,
        similarity_weight: float = 0.6,
        activation_weight: float = 0.4,
    ) -> SearchResult:
        """Semantic search: Flash (vector similarity) + Unfold (graph propagation).

        Requires sentence-transformers to be installed. The embedding model
        is loaded lazily on first call.

        Args:
            query: Natural language query (any language supported by the model)
            top_k: Number of seed nodes from vector similarity
            propagate: Whether to expand results via graph propagation
            min_similarity: Minimum cosine similarity threshold for seeds
            similarity_weight: Weight for similarity in combined ranking
            activation_weight: Weight for activation in combined ranking

        Returns:
            SearchResult with seeds, propagated nodes, and combined ranking

        Raises:
            ImportError: If sentence-transformers is not installed
        """
        import time as _time
        start = _time.time()

        self._ensure_search_ready()

        # Phase 1: Flash — vector similarity
        query_vec = self._embed_model.encode(
            [query], normalize_embeddings=True
        )[0]
        seeds = self._vector_index.query(query_vec, top_k=top_k)
        seeds = [(name, score) for name, score in seeds if score >= min_similarity]

        # Phase 2: Unfold — graph propagation from seed nodes
        propagated_list = []
        if propagate and seeds:
            seed_names = [name for name, _ in seeds]
            seed_set = set(seed_names)

            # Run propagation using seed names as input
            seed_text = " ".join(seed_names)
            all_activated = self.propagate(seed_text)

            # Collect propagated nodes (excluding seeds)
            for node_name in all_activated:
                if node_name not in seed_set:
                    act = self._nodes[node_name].activation if node_name in self._nodes else 0.0
                    if act > 0:
                        propagated_list.append((node_name, act))
            propagated_list.sort(key=lambda x: x[1], reverse=True)

        # Phase 3: Rank — combine similarity + activation
        score_map: Dict[str, float] = {}

        # Seeds get similarity + activation
        for name, sim in seeds:
            act = self._nodes[name].activation if name in self._nodes else 0.0
            score_map[name] = similarity_weight * sim + activation_weight * act

        # Propagated nodes get activation only
        for name, act in propagated_list:
            if name not in score_map:
                score_map[name] = activation_weight * act

        combined = sorted(score_map.items(), key=lambda x: x[1], reverse=True)

        elapsed = (_time.time() - start) * 1000

        return SearchResult(
            query=query,
            seeds=seeds,
            propagated=propagated_list[:top_k],
            combined=combined[:top_k * 2],
            search_time_ms=elapsed,
        )

    def build_search_index(self, model_name: str = None) -> int:
        """Build or rebuild the embedding index for all nodes.

        Args:
            model_name: Override default model name

        Returns:
            Number of nodes indexed
        """
        from stg_engine.semantic import (
            EmbeddingBuilder, VectorIndex, load_embedding_model,
            DEFAULT_MODEL_NAME,
        )

        model_name = model_name or DEFAULT_MODEL_NAME
        self._model_name = model_name
        self._embed_model = load_embedding_model(model_name)

        builder = EmbeddingBuilder()
        self._embed_texts = builder.build_all(self)

        if not self._embed_texts:
            self._vector_index = VectorIndex()
            return 0

        # Encode all texts
        texts = list(self._embed_texts.values())
        names = list(self._embed_texts.keys())
        vectors = self._embed_model.encode(texts, normalize_embeddings=True)

        # Build index
        self._vector_index = VectorIndex()
        embeddings_dict = {name: vectors[i] for i, name in enumerate(names)}
        self._vector_index.build(embeddings_dict)

        return self._vector_index.size

    def _ensure_search_ready(self) -> None:
        """Load model and build index if not already done."""
        if self._embed_model is None:
            from stg_engine.semantic import load_embedding_model, DEFAULT_MODEL_NAME
            self._model_name = DEFAULT_MODEL_NAME
            self._embed_model = load_embedding_model(DEFAULT_MODEL_NAME)

        if self._vector_index is None or self._vector_index.size == 0:
            self._build_vector_index()

    def _build_vector_index(self) -> None:
        """Build vector index from current graph state."""
        from stg_engine.semantic import EmbeddingBuilder, VectorIndex
        import numpy as np

        builder = EmbeddingBuilder()
        self._embed_texts = builder.build_all(self)

        if not self._embed_texts:
            self._vector_index = VectorIndex()
            return

        texts = list(self._embed_texts.values())
        names = list(self._embed_texts.keys())
        vectors = self._embed_model.encode(texts, normalize_embeddings=True)

        self._vector_index = VectorIndex()
        embeddings_dict = {name: vectors[i] for i, name in enumerate(names)}
        self._vector_index.build(embeddings_dict)

    # ═══════════════════════════════════════════════════════════
    # Persistence (.stg format)
    # ═══════════════════════════════════════════════════════════

    def save(self, path: str, force_save: bool = False) -> None:
        """Serialize entire engine state to .stg file.

        The .stg file uses SQLite as serialization format.
        Safety: refuses to save if node count drops >50% unless force_save=True.
        """
        save_engine_state(
            path=path,
            nodes=self._nodes,
            edges=self._edges,
            sessions=self._sessions,
            events=self._events,
            tensions=self._tensions,
            belief_evolutions=self._belief_evolutions,
            snapshots=self._snapshots,
            force_save=force_save,
            aliases=self._aliases if self._aliases else None,
        )

    @classmethod
    def load(cls, path: str) -> "STGEngine":
        """Deserialize engine from .stg file.

        Args:
            path: Path to .stg file

        Returns:
            Fully loaded STGEngine
        """
        state = load_engine_state(path)

        engine = cls()
        _nk = cls._nk

        # Restore nodes — normalize keys, merge case duplicates
        for old_key, node in state["nodes"].items():
            nkey = _nk(old_key)
            if nkey in engine._nodes:
                # Merge: keep first display name, merge metadata
                engine._nodes[nkey].metadata.update(node.metadata)
            else:
                node.name = node.name  # preserve original display name
                engine._nodes[nkey] = node
            engine._graph.add_node(nkey)

        # Restore edges — use normalized keys for lookup, preserve display names
        engine._edges = state["edges"]
        for edge in engine._edges:
            sk, tk = _nk(edge.source), _nk(edge.target)
            engine._edges_lookup[(sk, tk)] = edge
            engine._graph.add_edge(sk, tk)

        # Restore episodic data
        engine._sessions = state["sessions"]
        engine._events = state["events"]
        engine._tensions = state["tensions"]
        engine._belief_evolutions = state["belief_evolutions"]
        engine._snapshots = state["snapshots"]

        # G7: Build token index for entity resolution
        for nkey, node in engine._nodes.items():
            engine._node_tokens[nkey] = cls._tokenize_for_er(node.name)

        # G7: Load aliases if present
        engine._aliases = state.get("aliases", {})

        return engine

    def export_stl(self) -> str:
        """Export all edges back to STL text format.

        Uses stl_parser's Builder for type-safe STL generation.
        Falls back to manual formatting if stl_parser is unavailable.

        Returns:
            STL text representation of the graph
        """
        try:
            from stl_parser import stl as stl_builder
        except ImportError:
            return self._export_stl_manual()

        lines = []
        for edge in self._edges:
            mods = {}
            if edge.confidence != 0.5:
                mods["confidence"] = edge.confidence
            if edge.strength != 0.5:
                mods["strength"] = edge.strength
            if edge.rule:
                mods["rule"] = edge.rule
            if edge.time:
                mods["time"] = edge.time
            mods.update(edge.modifiers)

            builder = stl_builder(f"[{edge.source}]", f"[{edge.target}]")
            if mods:
                builder = builder.mod(**mods)
            lines.append(str(builder.no_validate().build()))

        return "\n".join(lines)

    def _export_stl_manual(self) -> str:
        """Fallback STL export using manual string formatting."""
        lines = []
        for edge in self._edges:
            mod_parts = []
            if edge.confidence != 0.5:
                mod_parts.append(f"confidence={edge.confidence}")
            if edge.strength != 0.5:
                mod_parts.append(f"strength={edge.strength}")
            if edge.rule:
                mod_parts.append(f'rule="{edge.rule}"')
            if edge.time:
                mod_parts.append(f'time="{edge.time}"')
            for k, v in edge.modifiers.items():
                if isinstance(v, str):
                    mod_parts.append(f'{k}="{v}"')
                else:
                    mod_parts.append(f"{k}={v}")

            line = f"[{edge.source}] -> [{edge.target}]"
            if mod_parts:
                line += f" ::mod({', '.join(mod_parts)})"
            lines.append(line)

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════
    # Dunder methods
    # ═══════════════════════════════════════════════════════════

    def __repr__(self) -> str:
        return (
            f"STGEngine(nodes={len(self._nodes)}, edges={len(self._edges)}, "
            f"events={len(self._events)}, tensions={len(self._tensions)})"
        )

    def __len__(self) -> int:
        """Number of nodes in the graph."""
        return len(self._nodes)
