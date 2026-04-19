"""Epistemic Metadata Layer for STG Engine (Phase 7H).

Adds multi-dimensional epistemic metadata to STG edges:
- edge_class: cognitive / knowledge / structural (source monitoring)
- trace_type: EarthTrace / CosmicTrace / UserClaimed (TH v1.2)
- structural_coherence: 0.0-1.0 (logical consistency)
- verification_status: consensus / reproducible / documented / beyond_current_paradigm / subjective / contested
- epistemic_status: established / provisional / speculative / metaphorical
- scope: universal / physical / biological / cultural / personal / cosmic / metaphysical

All metadata is stored in STGEdge.modifiers (zero-invasive).
"""

from typing import Any, Dict, List, Literal

# ─── Type Definitions ─────────────────────────────────────────────

EdgeClass = Literal["cognitive", "knowledge", "structural", "virtual"]
TraceType = Literal["EarthTrace", "CosmicTrace", "UserClaimed"]
VerificationStatus = Literal[
    "consensus", "reproducible", "documented",
    "beyond_current_paradigm", "subjective", "contested",
]
EpistemicStatus = Literal["established", "provisional", "speculative", "metaphorical"]
Scope = Literal[
    "universal", "physical", "biological", "cultural",
    "personal", "cosmic", "metaphysical",
]

# ─── Valid Value Sets ─────────────────────────────────────────────

VALID_EDGE_CLASSES = {"cognitive", "knowledge", "structural", "virtual"}
VALID_TRACE_TYPES = {"EarthTrace", "CosmicTrace", "UserClaimed"}
VALID_VERIFICATION_STATUSES = {
    "consensus", "reproducible", "documented",
    "beyond_current_paradigm", "subjective", "contested",
}
VALID_EPISTEMIC_STATUSES = {"established", "provisional", "speculative", "metaphorical"}
VALID_SCOPES = {
    "universal", "physical", "biological", "cultural",
    "personal", "cosmic", "metaphysical",
}

# Keys that constitute epistemic metadata (only meaningful on knowledge edges)
EPISTEMIC_KEYS = {
    "trace_type", "structural_coherence", "verification_status",
    "epistemic_status", "scope",
}

# Typical confidence ranges per trace_type (advisory, from TH v1.2)
CONFIDENCE_RANGES = {
    "EarthTrace": (0.60, 0.98),
    "CosmicTrace": (0.50, 0.75),
    "UserClaimed": (0.35, 0.70),
}


# ─── Helper Functions ─────────────────────────────────────────────

def get_edge_class(modifiers: Dict[str, Any]) -> str:
    """Get edge class from modifiers, defaulting to 'structural'."""
    return modifiers.get("edge_class", "structural")


def is_virtual_edge(edge) -> bool:
    """Check if an edge is a virtual (auto-generated proximity) edge."""
    return edge.modifiers.get("edge_class") == "virtual"


def is_real_edge(edge) -> bool:
    """Check if an edge is a real (non-virtual) edge."""
    return edge.modifiers.get("edge_class") != "virtual"


def validate_epistemic_metadata(
    confidence: float,
    modifiers: Dict[str, Any],
) -> List[str]:
    """Validate epistemic metadata in edge modifiers.

    Returns list of warning strings. Empty list = fully valid.
    Warnings are advisory — edges are always stored regardless.
    """
    warnings: List[str] = []

    # Rule 0: Validate edge_class
    edge_class = modifiers.get("edge_class")
    if edge_class is not None and edge_class not in VALID_EDGE_CLASSES:
        warnings.append(
            f"Unknown edge_class '{edge_class}'. "
            f"Expected one of: {sorted(VALID_EDGE_CLASSES)}"
        )

    # Rule 3: Epistemic metadata only meaningful on knowledge edges
    resolved_class = edge_class if edge_class else "structural"
    has_epistemic = bool(EPISTEMIC_KEYS & set(modifiers.keys()))
    if resolved_class != "knowledge" and has_epistemic:
        warnings.append(
            f"Epistemic metadata on '{resolved_class}' edge. "
            f"Epistemic metadata is designed for knowledge edges."
        )

    # Rules below only apply when trace_type is present
    trace_type = modifiers.get("trace_type")
    if trace_type is None:
        return warnings

    # Rule: Valid trace_type
    if trace_type not in VALID_TRACE_TYPES:
        warnings.append(f"Unknown trace_type '{trace_type}'.")

    # Rule 1: CosmicTrace requires structural_coherence >= 0.70
    if trace_type == "CosmicTrace":
        sc = modifiers.get("structural_coherence")
        if sc is not None and sc < 0.70:
            warnings.append(
                f"CosmicTrace requires structural_coherence >= 0.70, got {sc}. "
                f"Consider reclassifying as UserClaimed."
            )
        if sc is None:
            warnings.append(
                "CosmicTrace should have structural_coherence specified."
            )

    # Rule 2: Confidence range matches trace_type
    if trace_type in CONFIDENCE_RANGES:
        lo, hi = CONFIDENCE_RANGES[trace_type]
        if not (lo <= confidence <= hi):
            warnings.append(
                f"{trace_type} confidence typically in [{lo}, {hi}], got {confidence}."
            )

    # Rule 4: Valid enum values for other fields
    vs = modifiers.get("verification_status")
    if vs is not None and vs not in VALID_VERIFICATION_STATUSES:
        warnings.append(f"Unknown verification_status '{vs}'.")

    es = modifiers.get("epistemic_status")
    if es is not None and es not in VALID_EPISTEMIC_STATUSES:
        warnings.append(f"Unknown epistemic_status '{es}'.")

    scope = modifiers.get("scope")
    if scope is not None and scope not in VALID_SCOPES:
        warnings.append(f"Unknown scope '{scope}'.")

    # Rule 5: structural_coherence range
    sc = modifiers.get("structural_coherence")
    if sc is not None and not (0.0 <= sc <= 1.0):
        warnings.append(f"structural_coherence must be in [0.0, 1.0], got {sc}.")

    return warnings


def epistemic_summary(edges) -> Dict[str, Any]:
    """Compute epistemic composition summary from a list of STGEdge objects.

    Args:
        edges: Iterable of STGEdge objects

    Returns:
        Dict with distribution counts for edge_class, trace_type,
        verification_status, epistemic_status, scope.
    """
    edge_class_dist: Dict[str, int] = {"cognitive": 0, "knowledge": 0, "structural": 0}
    trace_type_dist: Dict[str, int] = {
        "EarthTrace": 0, "CosmicTrace": 0, "UserClaimed": 0, "untagged": 0,
    }
    verification_dist: Dict[str, int] = {}
    epistemic_status_dist: Dict[str, int] = {}
    scope_dist: Dict[str, int] = {}

    total = 0
    knowledge_count = 0

    for edge in edges:
        total += 1
        ec = get_edge_class(edge.modifiers)
        edge_class_dist[ec] = edge_class_dist.get(ec, 0) + 1

        if ec == "knowledge":
            knowledge_count += 1

        tt = edge.modifiers.get("trace_type")
        if tt:
            trace_type_dist[tt] = trace_type_dist.get(tt, 0) + 1
        elif ec == "knowledge":
            trace_type_dist["untagged"] += 1

        vs = edge.modifiers.get("verification_status")
        if vs:
            verification_dist[vs] = verification_dist.get(vs, 0) + 1

        es = edge.modifiers.get("epistemic_status")
        if es:
            epistemic_status_dist[es] = epistemic_status_dist.get(es, 0) + 1

        sc = edge.modifiers.get("scope")
        if sc:
            scope_dist[sc] = scope_dist.get(sc, 0) + 1

    return {
        "total_edge_count": total,
        "knowledge_edge_count": knowledge_count,
        "edge_class_distribution": edge_class_dist,
        "trace_type_distribution": trace_type_dist,
        "verification_status_distribution": verification_dist,
        "epistemic_status_distribution": epistemic_status_dist,
        "scope_distribution": scope_dist,
    }
