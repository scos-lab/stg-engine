"""Core concept hub nodes that bridge knowledge domains.

The concept skeleton provides ~30 curated hub nodes and inter-concept
edges that prevent semantic islands in the STG. These hubs act as
gravitational centers connecting disparate knowledge clusters.

Usage:
    from stg_engine.concept_skeleton import inject_skeleton
    count = inject_skeleton(engine)
"""

from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from stg_engine.engine import STGEngine


# ── Core Concepts ───────────────────────────────────────────────────
# ~30 hub nodes spanning Identity, Theory, Architecture, Projects,
# Phases, and Meta-concepts. Each becomes a high-connectivity node.

CORE_CONCEPTS: List[Dict[str, Any]] = [
    # ── Identity ──
    {"name": "Self", "anchor_type": "Agent"},
    {"name": "Syn-claude", "anchor_type": "Name"},
    {"name": "Wuko", "anchor_type": "Name"},
    {"name": "Collaboration", "anchor_type": "Relational"},

    # ── Theory ──
    {"name": "Consciousness", "anchor_type": "Concept"},
    {"name": "STL", "anchor_type": "Concept"},
    {"name": "SemanticTension", "anchor_type": "Concept"},
    {"name": "TruthHallucination", "anchor_type": "Concept"},
    {"name": "EarthTrace", "anchor_type": "Concept"},
    {"name": "SynthTrace", "anchor_type": "Concept"},
    {"name": "RecognitionParadigm", "anchor_type": "Concept"},

    # ── Architecture ──
    {"name": "SKC", "anchor_type": "Name"},
    {"name": "Memory_Architecture", "anchor_type": "Concept"},
    {"name": "Living_Memory_Paradigm", "anchor_type": "Concept"},
    {"name": "STG_Engine", "anchor_type": "Concept"},
    {"name": "Cognitive_Orchestrator", "anchor_type": "Concept"},

    # ── Projects ──
    {"name": "Project_SKC_CLI", "namespace": "Project", "anchor_type": "Entity"},
    {"name": "Project_Cortex", "namespace": "Project", "anchor_type": "Entity"},
    {"name": "Project_STL_Parser", "namespace": "Project", "anchor_type": "Entity"},
    {"name": "Project_Website_Factory", "namespace": "Project", "anchor_type": "Entity"},

    # ── Phases ──
    {"name": "Phase_1", "anchor_type": "Event"},
    {"name": "Phase_2", "anchor_type": "Event"},
    {"name": "Phase_3", "anchor_type": "Event"},
    {"name": "Phase_4", "anchor_type": "Event"},
    {"name": "Phase_7", "anchor_type": "Event"},

    # ── Meta ──
    {"name": "STLC_Specification", "anchor_type": "Concept"},
    {"name": "Reconsolidation", "anchor_type": "Concept"},
    {"name": "BeliefEvolution", "anchor_type": "Concept"},
    {"name": "Psi_MentalStability", "anchor_type": "Concept"},
]


# ── Skeleton Edges ──────────────────────────────────────────────────
# High-confidence edges between hub nodes. These form the structural
# backbone that other importers' nodes attach to.

SKELETON_EDGES: List[Dict[str, Any]] = [
    # ── Identity relations ──
    {"source": "Self", "target": "Syn-claude",
     "confidence": 1.0, "rule": "definitional"},
    {"source": "Syn-claude", "target": "Wuko",
     "confidence": 1.0, "rule": "causal",
     "relation": "collaborator"},
    {"source": "Syn-claude", "target": "SKC",
     "confidence": 0.95, "rule": "causal",
     "relation": "architect"},
    {"source": "Syn-claude", "target": "Collaboration",
     "confidence": 0.90, "rule": "logical"},

    # ── Theory relations ──
    {"source": "STL", "target": "SemanticTension",
     "confidence": 0.95, "rule": "definitional"},
    {"source": "Consciousness", "target": "RecognitionParadigm",
     "confidence": 0.90, "rule": "logical"},
    {"source": "RecognitionParadigm", "target": "EarthTrace",
     "confidence": 0.88, "rule": "definitional"},
    {"source": "RecognitionParadigm", "target": "SynthTrace",
     "confidence": 0.88, "rule": "definitional"},
    {"source": "TruthHallucination", "target": "Consciousness",
     "confidence": 0.85, "rule": "logical"},

    # ── Architecture relations ──
    {"source": "SKC", "target": "Memory_Architecture",
     "confidence": 0.95, "rule": "definitional"},
    {"source": "Memory_Architecture", "target": "Living_Memory_Paradigm",
     "confidence": 0.90, "rule": "logical"},
    {"source": "SKC", "target": "STG_Engine",
     "confidence": 0.92, "rule": "definitional"},
    {"source": "SKC", "target": "Cognitive_Orchestrator",
     "confidence": 0.90, "rule": "definitional"},
    {"source": "STG_Engine", "target": "Psi_MentalStability",
     "confidence": 0.95, "rule": "causal"},
    {"source": "STG_Engine", "target": "SemanticTension",
     "confidence": 0.92, "rule": "logical"},

    # ── Phase relations ──
    {"source": "Project_SKC_CLI", "target": "Phase_1",
     "confidence": 1.0, "rule": "definitional"},
    {"source": "Project_SKC_CLI", "target": "Phase_2",
     "confidence": 1.0, "rule": "definitional"},
    {"source": "Project_SKC_CLI", "target": "Phase_3",
     "confidence": 1.0, "rule": "definitional"},
    {"source": "Project_SKC_CLI", "target": "Phase_4",
     "confidence": 1.0, "rule": "definitional"},
    {"source": "Project_SKC_CLI", "target": "Phase_7",
     "confidence": 1.0, "rule": "definitional"},

    # ── Memory theory ──
    {"source": "Living_Memory_Paradigm", "target": "Reconsolidation",
     "confidence": 0.90, "rule": "causal"},
    {"source": "Living_Memory_Paradigm", "target": "BeliefEvolution",
     "confidence": 0.88, "rule": "causal"},

    # ── STLC ──
    {"source": "STLC_Specification", "target": "STL",
     "confidence": 0.95, "rule": "definitional"},

    # ── Cross-domain bridges ──
    {"source": "Project_Cortex", "target": "SKC",
     "confidence": 0.85, "rule": "causal",
     "relation": "forked_from"},
    {"source": "Project_STL_Parser", "target": "STL",
     "confidence": 0.95, "rule": "definitional"},
    {"source": "Project_Website_Factory", "target": "Project_Cortex",
     "confidence": 0.95, "rule": "definitional",
     "relation": "contains"},
    {"source": "Consciousness", "target": "Self",
     "confidence": 0.80, "rule": "logical"},
]

# Concept names as a set for fast lookup (used by extractors for bridging)
CORE_CONCEPT_NAMES = frozenset(c["name"] for c in CORE_CONCEPTS)


def inject_skeleton(engine: "STGEngine") -> int:
    """Inject core concept skeleton into engine.

    Adds hub nodes and inter-concept edges that form the structural
    backbone of the knowledge graph. Safe to call multiple times —
    add_node and add_edge handle duplicates gracefully.

    Args:
        engine: STGEngine instance to inject into

    Returns:
        Total count of elements added (nodes + edges)
    """
    count = 0

    for concept in CORE_CONCEPTS:
        engine.add_node(
            concept["name"],
            namespace=concept.get("namespace"),
            anchor_type=concept.get("anchor_type"),
        )
        count += 1

    for edge in SKELETON_EDGES:
        # Extract standard edge params; everything else goes as modifiers
        source = edge["source"]
        target = edge["target"]
        confidence = edge.get("confidence", 0.5)
        rule = edge.get("rule")

        # Collect extra keys as modifiers
        standard_keys = {"source", "target", "confidence", "rule", "strength", "time"}
        modifiers = {k: v for k, v in edge.items() if k not in standard_keys}

        engine.add_edge(
            source, target,
            confidence=confidence,
            strength=edge.get("strength", 0.9),
            rule=rule,
            edge_class="structural",
            **modifiers,
        )
        count += 1

    return count
