"""STG Engine data types.

Core data structures for the Semantic Tension Graph computation engine.
Nodes carry computed state (tension, activation) as part of their structure —
storage and computation are unified, not separate layers.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple


@dataclass
class STGNode:
    """A semantic anchor in the computation graph.

    Each node represents a concept, entity, event, agent, or other
    semantic element from STL. Computed state (tension, activation,
    self_relevance) lives directly on the node — the data IS the
    computation substrate.
    """
    name: str
    namespace: Optional[str] = None
    anchor_type: Optional[str] = None  # Concept/Event/Entity/Agent/Name/Question/Verifier/Relational/PathSegment
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Computed state — part of the node, not a separate cache
    tension: float = 0.0
    activation: float = 0.0
    self_relevance: float = 0.0

    @property
    def qualified_name(self) -> str:
        """Full name with namespace if present. E.g. 'Physics:Energy'."""
        if self.namespace:
            return f"{self.namespace}:{self.name}"
        return self.name

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "namespace": self.namespace,
            "anchor_type": self.anchor_type,
            "metadata": self.metadata,
            "tension": self.tension,
            "activation": self.activation,
            "self_relevance": self.self_relevance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "STGNode":
        """Deserialize from dictionary."""
        return cls(
            name=data["name"],
            namespace=data.get("namespace"),
            anchor_type=data.get("anchor_type"),
            metadata=data.get("metadata", {}),
            tension=data.get("tension", 0.0),
            activation=data.get("activation", 0.0),
            self_relevance=data.get("self_relevance", 0.0),
        )


@dataclass
class STGEdge:
    """A semantic relation with modifiers — the computation unit.

    Each edge connects two anchors with directional semantics.
    Key modifier fields are extracted for fast computation;
    the full modifier set is preserved for completeness.
    """
    source: str
    target: str

    # Extracted key fields — hot path for computation
    confidence: float = 0.5
    strength: float = 0.5
    rule: Optional[str] = None        # causal / logical / empirical / definitional
    time: Optional[str] = None

    # Full modifier set — all STL modifiers preserved
    modifiers: Dict[str, Any] = field(default_factory=dict)

    # Provenance — which session/event created this edge
    session_id: Optional[str] = None
    event_id: Optional[str] = None

    # Retrieval salience — how easily recalled (modified by Hebbian learning, decays)
    # Distinct from confidence (truth value, never auto-decays)
    # Propagation weight = confidence × salience
    salience: float = 0.5

    # Usage tracking — updated by HebbianLearner (Phase 7B)
    last_used: Optional[float] = None  # Unix timestamp of last co-activation

    # Utility tracking — updated by PreferenceFunction (Phase 8)
    preference: float = 0.0  # positive=useful, negative=anti-useful, zero=neutral

    # Temporal structure — Phase 11 (Temporal)
    created_at: float = 0.0            # Unix epoch when edge was created (0.0 = unknown/legacy)
    edge_class: str = "knowledge"      # "knowledge" | "temporal" | "virtual"
    delay_k: int = 0                   # k-fold delay (0 = not a sequence edge, 1+ = step distance)

    @property
    def uncertainty(self) -> float:
        """Uncertainty = 1 - confidence. Used in tension calculus."""
        return 1.0 - self.confidence

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "confidence": self.confidence,
            "strength": self.strength,
            "rule": self.rule,
            "time": self.time,
            "modifiers": self.modifiers,
            "salience": self.salience,
            "session_id": self.session_id,
            "event_id": self.event_id,
            "last_used": self.last_used,
            "preference": self.preference,
            "created_at": self.created_at,
            "edge_class": self.edge_class,
            "delay_k": self.delay_k,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "STGEdge":
        """Deserialize from dictionary."""
        return cls(
            source=data["source"],
            target=data["target"],
            confidence=data.get("confidence", 0.5),
            strength=data.get("strength", 0.5),
            rule=data.get("rule"),
            time=data.get("time"),
            modifiers=data.get("modifiers", {}),
            salience=data.get("salience", data.get("confidence", 0.5)),
            session_id=data.get("session_id"),
            event_id=data.get("event_id"),
            last_used=data.get("last_used"),
            preference=data.get("preference", 0.0),
            created_at=data.get("created_at", 0.0),
            edge_class=data.get("edge_class", "knowledge"),
            delay_k=data.get("delay_k", 0),
        )


@dataclass
class STGSession:
    """Metadata for a work session."""
    session_id: str
    date: Optional[str] = None
    title: Optional[str] = None
    avg_importance: Optional[float] = None
    event_count: int = 0
    status: str = "complete"
    summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "date": self.date,
            "title": self.title,
            "avg_importance": self.avg_importance,
            "event_count": self.event_count,
            "status": self.status,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "STGSession":
        return cls(
            session_id=data["session_id"],
            date=data.get("date"),
            title=data.get("title"),
            avg_importance=data.get("avg_importance"),
            event_count=data.get("event_count", 0),
            status=data.get("status", "complete"),
            summary=data.get("summary"),
        )


@dataclass
class STGEvent:
    """An episodic memory event."""
    event_id: str
    session_id: Optional[str] = None
    timestamp: Optional[str] = None
    event_type: Optional[str] = None   # bug_fix, milestone, reconsolidation, etc.
    memory_type: Optional[str] = None  # fact, belief, insight, tension
    title: Optional[str] = None
    importance_score: float = 0.5
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    stl_block: Optional[str] = None    # Original STL statements

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "memory_type": self.memory_type,
            "title": self.title,
            "importance_score": self.importance_score,
            "description": self.description,
            "tags": self.tags,
            "artifacts": self.artifacts,
            "stl_block": self.stl_block,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "STGEvent":
        return cls(
            event_id=data["event_id"],
            session_id=data.get("session_id"),
            timestamp=data.get("timestamp"),
            event_type=data.get("event_type"),
            memory_type=data.get("memory_type"),
            title=data.get("title"),
            importance_score=data.get("importance_score", 0.5),
            description=data.get("description"),
            tags=data.get("tags", []),
            artifacts=data.get("artifacts", []),
            stl_block=data.get("stl_block"),
        )


@dataclass
class STGTension:
    """A tracked semantic tension (unresolved question/conflict)."""
    name: str
    initial_value: float = 0.0
    current_value: float = 0.0
    status: str = "active"  # active / resolved / persisting
    created_session: Optional[str] = None
    resolved_session: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "initial_value": self.initial_value,
            "current_value": self.current_value,
            "status": self.status,
            "created_session": self.created_session,
            "resolved_session": self.resolved_session,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "STGTension":
        return cls(
            name=data["name"],
            initial_value=data.get("initial_value", 0.0),
            current_value=data.get("current_value", 0.0),
            status=data.get("status", "active"),
            created_session=data.get("created_session"),
            resolved_session=data.get("resolved_session"),
            description=data.get("description"),
        )


@dataclass
class STGBeliefEvolution:
    """A belief evolution record: M -> M'."""
    old_anchor: str
    new_anchor: str
    event_id: Optional[str] = None
    session_id: Optional[str] = None
    level: int = 1  # 1=Reinforcement, 2=Integration, 3=Reconceptualization
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "old_anchor": self.old_anchor,
            "new_anchor": self.new_anchor,
            "event_id": self.event_id,
            "session_id": self.session_id,
            "level": self.level,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "STGBeliefEvolution":
        return cls(
            old_anchor=data["old_anchor"],
            new_anchor=data["new_anchor"],
            event_id=data.get("event_id"),
            session_id=data.get("session_id"),
            level=data.get("level", 1),
            description=data.get("description"),
        )


@dataclass
class SystemSnapshot:
    """A point-in-time snapshot of system state for ΔΨ calculation."""
    session_id: Optional[str] = None
    timestamp: Optional[str] = None
    psi_value: float = 0.0
    max_tension: float = 0.0
    structural_coherence: float = 0.0
    epistemic_confidence: float = 0.0
    total_reward: float = 0.0
    node_count: int = 0
    edge_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "psi_value": self.psi_value,
            "max_tension": self.max_tension,
            "structural_coherence": self.structural_coherence,
            "epistemic_confidence": self.epistemic_confidence,
            "total_reward": self.total_reward,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SystemSnapshot":
        return cls(
            session_id=data.get("session_id"),
            timestamp=data.get("timestamp"),
            psi_value=data.get("psi_value", 0.0),
            max_tension=data.get("max_tension", 0.0),
            structural_coherence=data.get("structural_coherence", 0.0),
            epistemic_confidence=data.get("epistemic_confidence", 0.0),
            total_reward=data.get("total_reward", 0.0),
            node_count=data.get("node_count", 0),
            edge_count=data.get("edge_count", 0),
        )


# ═══════════════════════════════════════════════════════════
# Phase 7A: Metrics Types
# ═══════════════════════════════════════════════════════════


@dataclass
class PropagationMetrics:
    """Metrics captured from a single propagate() call.

    Stored on STGEngine._last_propagation_metrics after each propagate().
    Enables post-hoc analysis of query routing quality.
    """
    # Input characteristics
    input_text: str
    token_count: int
    seed_node_count: int

    # Propagation results
    activated_node_count: int
    total_activation: float
    max_activation: float
    iterations_used: int

    # Computed metrics
    query_efficiency: float     # seeds / activated (precision of routing)
    resonance_score: float      # max_activation / total_activation (signal vs noise)
    coverage: float             # activated / total_nodes

    # Top results for analysis
    top_nodes: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class GraphMetrics:
    """Aggregate health metrics for the STG graph.

    Computed by engine.get_metrics(). Cached until graph mutation.
    """
    # Topology
    node_count: int = 0
    edge_count: int = 0
    density: float = 0.0
    avg_degree: float = 0.0
    max_degree: int = 0
    max_degree_node: str = ""

    # Information theory
    entropy: float = 0.0          # Shannon entropy of degree distribution
    criticality: float = 0.0      # entropy / max_entropy (edge of chaos)

    # Confidence distribution
    confidence_mean: float = 0.0
    confidence_median: float = 0.0
    confidence_stdev: float = 0.0
    high_confidence_ratio: float = 0.0   # >= 0.8
    low_confidence_ratio: float = 0.0    # < 0.3

    # Connectivity
    weakly_connected_components: int = 0
    largest_component_ratio: float = 0.0

    # Namespace coverage
    namespace_count: int = 0
    namespaces: Dict[str, int] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════
# Phase 7B: Learning Types
# ═══════════════════════════════════════════════════════════


@dataclass
class LearningEvent:
    """Record of a single learning or pruning action.

    Created by HebbianLearner and SynapticPruner for audit trail.
    """
    event_type: str       # "strengthen", "weaken", "prune", "prune_orphan"
    source: str           # Edge source (or node name for prune_orphan)
    target: str           # Edge target (or "" for prune_orphan)
    old_confidence: float
    new_confidence: float  # 0.0 for pruned edges
    timestamp: float
    trigger: str          # "propagation", "manual", "prune_cycle"


# ═══════════════════════════════════════════════════════════
# Phase 7C: Topology Types
# ═══════════════════════════════════════════════════════════


@dataclass
class CommunityInfo:
    """A detected community (cluster) in the graph."""
    community_id: int
    members: List[str]
    size: int
    dominant_namespace: Optional[str] = None
    namespace_purity: float = 0.0
    internal_density: float = 0.0


@dataclass
class BridgeSuggestion:
    """A suggested bridge edge between two communities."""
    source: str
    target: str
    source_community: int
    target_community: int
    confidence: float = 0.5
    rationale: str = ""


@dataclass
class TopologyReport:
    """Complete topology analysis result."""
    # Community analysis
    communities: List[CommunityInfo] = field(default_factory=list)
    community_count: int = 0
    modularity: float = 0.0
    namespace_alignment: float = 0.0

    # Bridge analysis
    bridge_suggestions: List[BridgeSuggestion] = field(default_factory=list)
    disconnected_pairs: int = 0

    # Redundancy analysis
    redundant_edges: List[Tuple[str, str, float]] = field(default_factory=list)
    redundant_count: int = 0

    # Summary
    node_count: int = 0
    edge_count: int = 0
    timestamp: float = 0.0


# ═══════════════════════════════════════════════════════════
# Phase 7D: Cognitive Architecture Types
# ═══════════════════════════════════════════════════════════


@dataclass
class GoalEntry:
    """An active goal biasing propagation."""
    name: str
    keywords: List[str]
    priority: float = 1.0
    created_at: float = 0.0


@dataclass
class Hypothesis:
    """A predicted missing connection."""
    source: str
    target: str
    confidence: float
    evidence_count: int
    rationale: str
    timestamp: float = 0.0


@dataclass
class SelfModelReport:
    """STG's self-assessment of its knowledge state."""
    namespace_density: Dict[str, float] = field(default_factory=dict)
    connectivity_health: float = 0.0
    isolation_count: int = 0
    cross_namespace_score: float = 0.0
    top_hubs: List[Tuple[str, float]] = field(default_factory=list)
    gap_namespaces: List[str] = field(default_factory=list)
    fragile_nodes: List[str] = field(default_factory=list)
    assessment: str = ""
    timestamp: float = 0.0


@dataclass
class StrategyResult:
    """Output from query routing."""
    strategy: str
    query: str
    results: List[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════
# Phase 7E: Feedback Loop Types
# ═══════════════════════════════════════════════════════════


@dataclass
class FeedbackLoopConfig:
    """Configuration for feedback loop cadence and behavior."""
    # Periodic cadence
    periodic_interval: int = 10           # Run periodic tasks every N turns
    hypothesis_max_apply: int = 3         # Max hypotheses to auto-apply per cycle
    hypothesis_min_confidence: float = 0.5  # Only auto-apply above this
    # Self-improvement
    auto_goal_update: bool = True         # Auto-update goals from self-model gaps
    max_auto_goals: int = 2              # Max goals auto-generated from gaps
    # Creative loop
    auto_prune_on_session_end: bool = True
    prune_after_creative: bool = False    # Prune immediately after adding hypotheses
    # Predictive loop
    warmup_on_pre_turn: bool = True
    learn_on_post_turn: bool = True
    tension_on_prediction_error: bool = False  # Auto-create tension from prediction mismatch
    # Co-activation edge creation (Phase 11B)
    coactivation_on_session_end: bool = False  # Auto-create edges from co-activation data
    coactivation_min_count: int = 5            # Min co-occurrences to consider
    coactivation_max_per_session: int = 10     # Max edges created per session-end
    coactivation_max_per_node: int = 5         # Per-node degree cap on coactivation edges
    coactivation_exclude_top_hubs: int = 10    # Skip top-N nodes by degree


@dataclass
class TurnRecord:
    """Record of a single turn's processing summary."""
    turn_number: int
    pre_turn_summary: Dict[str, Any] = field(default_factory=dict)
    post_turn_summary: Dict[str, Any] = field(default_factory=dict)
    periodic_ran: bool = False
    periodic_summary: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class LoopStats:
    """Cumulative statistics for feedback loop execution."""
    total_turns: int = 0
    total_periodic_runs: int = 0
    # Self-improvement loop
    goals_auto_generated: int = 0
    self_models_built: int = 0
    # Predictive loop
    total_warmups: int = 0
    total_hebbian_events: int = 0
    # Creative loop
    hypotheses_generated: int = 0
    hypotheses_applied: int = 0
    hypotheses_rejected: int = 0
    edges_pruned: int = 0


# ═══════════════════════════════════════════════════════════
# Phase 7F: Validation & Benchmarking Types
# ═══════════════════════════════════════════════════════════


@dataclass
class EvalQuestion:
    """A ground-truth evaluation question."""
    question: str
    expected_nodes: List[str]     # Nodes that should appear in results
    expected_namespace: str        # Primary namespace of the answer
    difficulty: str = "medium"     # easy / medium / hard
    category: str = "lookup"       # lookup / explore / create / solve


@dataclass
class EvalResult:
    """Result of evaluating a single question."""
    question: str
    strategy_used: str
    nodes_found: List[str]
    expected_nodes: List[str]
    hit_count: int                 # How many expected nodes were found
    precision: float               # hit_count / len(nodes_found)
    recall: float                  # hit_count / len(expected_nodes)
    qe: float                     # Query efficiency
    rs: float                     # Resonance score
    success: bool                  # recall >= threshold


@dataclass
class BenchmarkReport:
    """Complete benchmark run result."""
    timestamp: float
    graph_size: Tuple[int, int]    # (nodes, edges)
    psi: float
    criticality: float
    # Propagation accuracy
    total_questions: int
    correct_count: int
    accuracy: float
    avg_qe: float
    avg_rs: float
    # Strategy routing
    strategy_counts: Dict[str, int]
    strategy_success_rates: Dict[str, float]
    overall_routing_success: float
    # Temporal
    recent_activation_ratio: float  # avg(recent) / avg(old)
    # Hypothesis
    hypotheses_generated: int
    hypotheses_validated: int       # against ground truth
    hypothesis_quality: float       # validated / generated
    # Self-model
    gaps_detected: List[str]
    gaps_correct: int               # matched expected gaps
    # Emergence
    psi_delta: float                # Change after N turns
    edges_learned: int
    goals_auto_generated: int
    # Performance
    avg_turn_ms: float
    total_time_ms: float


# ═══════════════════════════════════════════════════════════
# Phase 7.5: Turn Integration Types
# ═══════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════
# Phase 7G: Semantic Query Layer Types
# ═══════════════════════════════════════════════════════════


@dataclass
class SearchResult:
    """Result of a semantic search operation (Phase 7G)."""
    query: str
    seeds: List[Tuple[str, float]]        # (node_name, similarity_score)
    propagated: List[Tuple[str, float]]   # (node_name, activation_score)
    combined: List[Tuple[str, float]]     # (node_name, final_score) — ranked
    search_time_ms: float = 0.0


@dataclass
class STGTurnContext:
    """Context passed between STG hooks within a single turn.

    Created by STGBridge.pre_turn(), enriched by route_query(),
    finalized by post_turn(). Exposed as client.last_stg_context.
    """
    turn_number: int = 0
    user_message: str = ""
    pre_turn: Dict[str, Any] = field(default_factory=dict)
    route: Dict[str, Any] = field(default_factory=dict)
    post_turn: Dict[str, Any] = field(default_factory=dict)
    activated_nodes: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════
# Phase 8: Kanerva Extensions Types
# ═══════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════
# Phase 9: Inhibition Types
# ═══════════════════════════════════════════════════════════


@dataclass
class InhibitionConfig:
    """Configuration for the inhibition system (Phase 9).

    All inhibition is disabled by default (enabled=False).
    When enabled, each mechanism can be individually toggled.

    Three phases of inhibition:
      Phase 1: Softmax WTA + Divisive Normalization (core)
      Phase 2: Adaptive Threshold + Refractory Period
      Phase 3: Inhibitory Edges + Community Inhibition
    """
    # Master switch
    enabled: bool = False

    # Phase 1: Softmax WTA
    softmax_wta: bool = True        # Use softmax WTA instead of linear rescaling
    eta: float = 2.0                # WTA sharpness (1.0=linear, 2.0=quadratic, higher=sharper)

    # Phase 1: Divisive Normalization
    divisive_normalization: bool = True
    sigma: float = 0.5             # Normalization strength (0=off, 1.0=strong)

    # Phase 2: Adaptive Threshold
    adaptive_threshold: bool = False
    threshold_gain: float = 1.0    # How much mean activity raises threshold

    # Phase 2: Refractory Period
    refractory: bool = False
    refractory_decay: float = 0.5  # Refractory decay rate per propagation
    refractory_suppression: float = 0.3  # Max suppression (0.3 = 30%)

    # Phase 3: Inhibitory Edges
    inhibitory_edges: bool = False
    inhibitory_strength: float = 1.0

    # Phase 3: Community Inhibition
    community_inhibition: bool = False
    community_suppression: float = 0.2


@dataclass
class PerceptionConfig:
    """Configuration for the perception system (Phase 12).

    Enables CNN-based visual perception within STG.
    Disabled by default. When enabled, perceive() and
    perceive_and_propagate() methods become available on STGEngine.
    """
    # Master switch
    enabled: bool = False

    # Feature extraction
    feature_dim: int = 128
    fixed_filter_count: int = 16
    learnable_filter_count: int = 8
    learnable_filter_size: int = 5
    max_colors: int = 16

    # Hebbian learning for learnable filters
    learning_rate: float = 0.01

    # Similarity search
    default_top_k: int = 5


@dataclass
class ConvergenceResult:
    """Result of iterative propagation convergence (Kanerva F5)."""
    top_nodes: List[str]              # Final stabilized top-N nodes
    iterations_used: int              # How many rounds before convergence
    converged: bool                   # True if stabilized before max_iterations
    stability_history: List[float]    # Jaccard similarity between consecutive rounds


@dataclass
class ConflictReport:
    """Report of a potential semantic conflict detected during ingest (Kanerva F6)."""
    new_edge: Tuple[str, str]                      # (source, target) of newly ingested edge
    conflicting_edges: List[Tuple[str, str]]        # Existing edges that conflict
    conflict_score: float                           # 0.0-1.0, severity
    details: str                                    # Human-readable explanation


# Phase 7I — Community-Centric Propagation
@dataclass
class RepresentativeEntry:
    """One representative node within a community (Phase 7I)."""
    node_name: str
    activation: float
    elevation: float


@dataclass
class CommunityPropagateResult:
    """A community surfaced by community-mode propagate (Phase 7I).

    score = name_boost * rep_activation * (1 + alpha * heat + beta * recency)
    In M1: heat=1.0, recency=1.0 placeholders (battery arrives in M2).
    Aggregation uses top-k representatives — no full-community sum/mean.
    """
    community_key: str                       # e.g. "medium_3"
    community_name: str                      # human-readable name (top representative)
    score: float                             # final ranking score
    rep_activation: float                    # mean activation over top-k representatives
    heat: float = 1.0                        # M1 placeholder; M2 wires battery
    recency: float = 1.0                     # M1 placeholder; M2 wires battery
    baseline_heat: float = 0.0               # M2: structural baseline floor
    name_matched: bool = False               # query string found in community_name
    representatives: List[RepresentativeEntry] = field(default_factory=list)
    # Query-matching nodes inside this community that are NOT top-k reps.
    # Surfaces precise hits that would otherwise vanish when changing
    # granularity from node to community (see Phase 7I usability fix 2026-04-19).
    query_seeds: List[RepresentativeEntry] = field(default_factory=list)
    recent_events: List["EventEntry"] = field(default_factory=list)  # M3 populates


@dataclass
class EventEntry:
    """One recent event within a community (Phase 7I, populated in M3)."""
    node_name: str
    created_at: float
    description: Optional[str] = None
