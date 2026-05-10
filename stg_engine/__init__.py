"""STG (Semantic Tension Graph) Engine.

A self-contained computation graph for semantic knowledge.
Storage and computation are unified — the graph IS the computation substrate.

License: BUSL-1.1 (Business Source License 1.1)
Free for personal, academic, educational, non-profit, government,
freelancer, and open source use. Commercial use by for-profit
companies requires a separate commercial license.
Contact: licensing@scos-lab.org

Usage:
    from stg_engine import STGEngine

    engine = STGEngine()
    engine.ingest_stl('[A] -> [B] ::mod(confidence=0.9)')
    engine.compute_all_tensions()
    psi = engine.compute_psi()
    engine.save("memory.stg")
"""

__version__ = "0.5.0a7"
__author__ = "wuko / scos-lab"
__license__ = "BUSL-1.1"
__url__ = "https://github.com/scos-lab/stg-engine"

# Hot-path algorithms: optional Rust extension, pure-Python fallback.
# The Rust core provides ~30-100x speedup but is optional — if it is
# unavailable (e.g. installing a pure-Python wheel or an environment
# without a Rust toolchain for sdist), the fallback module provides
# semantically-identical Python implementations.
try:
    from stg_engine import _rust_core  # noqa: F401
    HAS_RUST_CORE = True
except ImportError:
    HAS_RUST_CORE = False

from stg_engine.engine import STGEngine, SEMANTIC_FIELDS
from stg_engine.types import (
    STGNode, STGEdge, STGSession, STGEvent,
    STGTension, STGBeliefEvolution, SystemSnapshot,
    PropagationMetrics, GraphMetrics, LearningEvent,
    CommunityInfo, BridgeSuggestion, TopologyReport,
    GoalEntry, Hypothesis, SelfModelReport, StrategyResult,
    FeedbackLoopConfig, TurnRecord, LoopStats,
    EvalQuestion, EvalResult, BenchmarkReport,
    SearchResult, ConvergenceResult, ConflictReport,
    InhibitionConfig,
    PerceptionConfig,
)
from stg_engine.importers import import_memory_matrix
from stg_engine.universal_importer import import_knowledge_base
from stg_engine.epistemic import (
    get_edge_class, validate_epistemic_metadata, epistemic_summary,
    VALID_EDGE_CLASSES, VALID_TRACE_TYPES, VALID_VERIFICATION_STATUSES,
    VALID_EPISTEMIC_STATUSES, VALID_SCOPES, EPISTEMIC_KEYS,
)
from stg_engine.kanerva import (
    IterativePropagator, PreferenceFunction, ConflictDetector,
)
from stg_engine.temporal import (
    query_time_range, query_temporal_neighborhood,
    record_temporal_edge, build_episode_sequence,
    replay_episode, temporal_propagate,
    epoch_to_str, parse_date_str,
)

__all__ = [
    "STGEngine",
    "STGNode",
    "STGEdge",
    "STGSession",
    "STGEvent",
    "STGTension",
    "STGBeliefEvolution",
    "SystemSnapshot",
    "PropagationMetrics",
    "GraphMetrics",
    "LearningEvent",
    "CommunityInfo",
    "BridgeSuggestion",
    "TopologyReport",
    "GoalEntry",
    "Hypothesis",
    "SelfModelReport",
    "StrategyResult",
    "FeedbackLoopConfig",
    "TurnRecord",
    "LoopStats",
    "EvalQuestion",
    "EvalResult",
    "BenchmarkReport",
    "SearchResult",
    "ConvergenceResult",
    "ConflictReport",
    "InhibitionConfig",
    "PerceptionConfig",
    # Phase 8: Kanerva Extensions
    "IterativePropagator",
    "PreferenceFunction",
    "ConflictDetector",
    "import_memory_matrix",
    "import_knowledge_base",
    # Phase 7H: Epistemic metadata
    "get_edge_class",
    "validate_epistemic_metadata",
    "epistemic_summary",
    "VALID_EDGE_CLASSES",
    "VALID_TRACE_TYPES",
    "VALID_VERIFICATION_STATUSES",
    "VALID_EPISTEMIC_STATUSES",
    "VALID_SCOPES",
    "EPISTEMIC_KEYS",
    # Semantic fields + supersede detection
    "SEMANTIC_FIELDS",
    # Phase 11: Temporal Structure
    "query_time_range",
    "query_temporal_neighborhood",
    "record_temporal_edge",
    "build_episode_sequence",
    "replay_episode",
    "temporal_propagate",
    "epoch_to_str",
    "parse_date_str",
]
