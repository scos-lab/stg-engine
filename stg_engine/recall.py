"""Precision recall postprocessing — Phase 1.

Pure postprocessing pass applied AFTER propagate + aggregate_to_communities.
Does not touch core propagate, Rust core, gravity, aggregate_to_communities,
or any existing pipeline component.

Implements:
- R1: recency soft weight (created_at exponential decay)
- R1: supersede soft decay (superseded_at edges down-weighted, NOT filtered)
- R7: community dominance ratio (collapse weak communities)

Memory-never-vanishes principle (STG_PRECISION_RECALL_DESIGN §2.4):
  All edges remain in propagate output. Only their effective weight is
  softly attenuated. CLI continues to render (superseded) markers on
  display so reader can apply semantic judgment. No edge or node is
  ever filtered out by recall postprocessing.

Reference: development/design/STG_PRECISION_RECALL_DESIGN.md
"""

from __future__ import annotations

import math
import time as _time
from collections import defaultdict
from contextlib import contextmanager
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

if TYPE_CHECKING:
    from stg_engine.engine import STGEngine
    from stg_engine.gravity import GravityMap
    from stg_engine.types import STGEdge, CommunityPropagateResult


DEFAULT_RECENCY_HALFLIFE_DAYS = 30.0
DEFAULT_SUPERSEDE_DECAY_FACTOR = 0.3
DEFAULT_DOMINANCE_RATIO = 3.0
DEFAULT_ACTIVE_CONTEXT_BOOST = 5.0
DEFAULT_ACTIVE_CONTEXT_TTL_SECONDS = 1800.0  # 30 minutes
DEFAULT_MIN_CHAIN_LENGTH = 2

# R6 — edge content scan (fallback for tokens not matching any node name).
# Scans these 6 meta semantic / informational fields on edges. Substring
# match (no morphological prefix). See STG_R6_EDGE_FALLBACK_SEED_DESIGN.md
EDGE_SCAN_FIELDS: Tuple[str, ...] = (
    "description", "lesson", "action", "role", "status", "is_a",
)
DEFAULT_MAX_EDGE_HITS = 50
DEFAULT_MIN_EDGE_TOKEN_LENGTH = 3

# Lightweight tokenizer for multi-seed dispatch decision.
# Mirrors the stop-word/short-token filter from engine.propagate() at a
# coarse level. We do NOT replicate morphological prefix matching here —
# each per-token sub-propagate runs the full tokenizer internally.
_MULTI_SEED_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "am", "do", "does", "did", "has", "have", "had", "it", "its",
    "what", "who", "how", "why", "when", "where", "which",
    "to", "of", "in", "on", "at", "by", "for", "with", "from",
    "and", "or", "not", "no", "if", "but", "so", "as", "than",
    "me", "my", "we", "us", "you", "he", "she", "they", "them",
    "this", "that", "these", "those", "about", "tell", "describe",
    "explain", "can", "could", "would", "should", "will",
})


def _split_tokens(text: str) -> List[str]:
    """Split a query string into stop-word-filtered tokens.

    Used only to decide whether to dispatch to single-seed or multi-seed
    pipeline. The tokens returned here are also fed back as individual
    sub-queries to engine.propagate(); each call internally re-tokenizes
    with the full propagate tokenizer.
    """
    import re
    raw = text.lower().split()
    expanded: List[str] = []
    cjk_re = re.compile(r'[一-鿿㐀-䶿]')
    for t in raw:
        parts = re.split(r'[_\-:]', t)
        for p in parts:
            if not p:
                continue
            cleaned = re.sub(r'[^a-z0-9一-鿿㐀-䶿]+$', '', p)
            if cleaned:
                expanded.append(cleaned)
    return [
        t for t in expanded
        if (len(t) >= 2 or cjk_re.search(t)) and t not in _MULTI_SEED_STOP_WORDS
    ]


def _edge_weight(
    edge: "STGEdge",
    now: float,
    halflife_days: float,
    supersede_factor: float,
) -> float:
    """Soft weight for one edge: salience × recency × supersede_decay.

    - salience: existing edge field (Hebbian-modified, base 0.5)
    - recency: exp(-age_days / halflife). created_at <= 0 means legacy/unknown;
               returns 1.0 (no decay).
    - supersede_decay: superseded_at != null → factor (default 0.3); else 1.0
    """
    if edge.created_at > 0.0:
        age_days = max(0.0, (now - edge.created_at) / 86400.0)
        # True half-life: weight halves every halflife_days
        # exp(-age * ln(2) / halflife) ⇒ 0.5 when age = halflife
        recency = math.exp(-age_days * math.log(2) / halflife_days)
    else:
        recency = 1.0

    if edge.modifiers.get("superseded_at") is not None:
        supersede = supersede_factor
    else:
        supersede = 1.0

    return edge.salience * recency * supersede


def apply_recency_weight(
    engine: "STGEngine",
    activated: List[str],
    halflife_days: float = DEFAULT_RECENCY_HALFLIFE_DAYS,
    supersede_factor: float = DEFAULT_SUPERSEDE_DECAY_FACTOR,
    now: Optional[float] = None,
) -> List[str]:
    """Multiply each activated node's activation by its max in-subgraph edge weight.

    For each activated node, compute weight = max(edge_weight) over edges
    where both endpoints are in the activated set. Apply weight as a multiplier
    to the node's stored activation. Re-sort by adjusted activation.

    Mutates engine._nodes[name].activation in place — same pattern as
    gravitational_propagate (gravity.py:250-252).

    No node is ever removed. Even isolated activated nodes (no in-subgraph edges)
    keep their original activation (factor = 1.0).

    Args:
        engine: STGEngine
        activated: list of activated node names (display case from propagate)
        halflife_days: recency exponential halflife (default 30 days)
        supersede_factor: weight multiplier for superseded edges (default 0.3)
        now: timestamp for recency calc (default = current time)

    Returns:
        Re-sorted list of node names by adjusted activation descending.
        Length is preserved — the function never drops a node.
    """
    if not activated:
        return []

    if now is None:
        now = _time.time()

    activated_lower = {n.lower() for n in activated}

    # Single-pass scan: collect in-subgraph edges per node
    edges_per_node: Dict[str, List["STGEdge"]] = defaultdict(list)
    for edge in engine._edges:
        src_l = edge.source.lower()
        tgt_l = edge.target.lower()
        if src_l in activated_lower and tgt_l in activated_lower:
            edges_per_node[src_l].append(edge)
            if src_l != tgt_l:
                edges_per_node[tgt_l].append(edge)

    # Apply weight; mutate node.activation
    adjusted: Dict[str, float] = {}
    for name in activated:
        key = name.lower()
        node = engine._nodes.get(key)
        if not node or node.activation <= 0:
            adjusted[name] = node.activation if node else 0.0
            continue
        edges = edges_per_node.get(key, [])
        if edges:
            factor = max(
                _edge_weight(e, now, halflife_days, supersede_factor)
                for e in edges
            )
        else:
            factor = 1.0  # isolated activated node — preserve as-is
        node.activation = node.activation * factor
        adjusted[name] = node.activation

    # Re-sort by adjusted activation descending
    return sorted(adjusted, key=lambda n: adjusted[n], reverse=True)


@contextmanager
def context_anchor_boost(
    gravity_map: "GravityMap",
    anchor_names: Iterable[str],
    boost: float = DEFAULT_ACTIVE_CONTEXT_BOOST,
):
    """Temporarily boost elevation of active_context nodes (R5 anchoring).

    For each anchor node present in the GravityMap, add `boost` to its
    elevation across all resolution layers and the default node_elevation
    table. On context exit, original values are restored — even if an
    exception is raised inside the with-block.

    Anchor nodes not present in the GravityMap are silently skipped.

    Usage:
        with context_anchor_boost(gravity_map, ['User_Dog_Max']):
            activated = engine.propagate(query)  # Max-side gets pulled

    Args:
        gravity_map: pre-built GravityMap (mutated and restored)
        anchor_names: iterable of node names to boost (case-insensitive)
        boost: elevation increment (default +5.0)

    See STG_PRECISION_RECALL_DESIGN.md §4.5.
    """
    backup: List[Tuple[Dict[str, float], str, float]] = []
    boosted_keys = {n.lower() for n in anchor_names}

    if gravity_map is not None and boost > 0 and boosted_keys:
        # Boost default elevation map
        for key in boosted_keys:
            if key in gravity_map.node_elevation:
                backup.append((gravity_map.node_elevation, key,
                               gravity_map.node_elevation[key]))
                gravity_map.node_elevation[key] = (
                    gravity_map.node_elevation[key] + boost
                )
        # Boost each per-resolution layer
        for layer in gravity_map.elevation_by_resolution.values():
            for key in boosted_keys:
                if key in layer:
                    backup.append((layer, key, layer[key]))
                    layer[key] = layer[key] + boost

    try:
        yield
    finally:
        # Restore in reverse insertion order so layered backups unwind
        # correctly even if a key appears more than once.
        for d, key, original in reversed(backup):
            d[key] = original


def match_exact_anchors(
    engine: "STGEngine",
    query: str,
) -> Tuple[List[str], List[str]]:
    """A: greedy N-gram match against canonical node names (case-insensitive).

    Whitespace tokens are scanned left-to-right; at each position we try
    the longest contiguous N-gram first (joined with `_`). If that matches
    a node name, those N tokens are consumed as a single anchor and the
    scan advances by N. This means:

        "User food_for_thought_charity_gala"   → underscored chunk matches directly
        "User Food for thought charity gala"   → 5-gram joined as
                                                 'food_for_thought_charity_gala' matches
        "user volunteered at gala"             → 'gala' alone may match

    Tokens that don't participate in any match flow back as
    `remaining_tokens` for the regular tokenizer.

    Returns:
        (anchor_names, remaining_tokens)
    """
    raw = [t for t in query.split() if t]
    if not raw:
        return [], []

    anchors: List[str] = []
    remaining: List[str] = []
    n = len(raw)
    i = 0
    while i < n:
        matched = False
        # Try longest N-gram first (greedy)
        for k in range(n - i, 0, -1):
            chunk = raw[i:i + k]
            # Skip single-token attempts that are just stop-like noise:
            # the underscore-joined form is what we test against node names.
            candidate = "_".join(chunk).lower()
            node = engine._nodes.get(candidate)
            if node is not None:
                anchors.append(getattr(node, "name", candidate))
                i += k
                matched = True
                break
        if not matched:
            remaining.append(raw[i])
            i += 1
    return anchors, remaining


def find_edges_between(
    engine: "STGEngine",
    anchor_names: List[str],
) -> List["STGEdge"]:
    """C / R8: edges with both endpoints in the anchor set (bidirectional).

    Used when a query contains ≥2 exact anchor matches — the user has
    pointed at specific nodes and almost certainly wants to know how they
    connect. We list every edge whose `source` and `target` are both in
    the anchor set (regardless of direction), so the user sees the
    relationship modifiers (action / occurred_time / description) directly.

    Self-loops (source == target) are skipped.
    """
    if len(anchor_names) < 2:
        return []
    anchor_lower = {n.lower() for n in anchor_names}
    matched: List["STGEdge"] = []
    for edge in engine._edges:
        src_l = edge.source.lower()
        tgt_l = edge.target.lower()
        if src_l == tgt_l:
            continue
        if src_l in anchor_lower and tgt_l in anchor_lower:
            matched.append(edge)
    return matched


def classify_tokens(
    engine: "STGEngine",
    tokens: List[str],
) -> Tuple[List[str], List[str]]:
    """R6 token routing — split tokens into node-name vs edge-content groups.

    For each token, check if it matches any node name in the graph using the
    same morphological prefix logic as engine.propagate(). Tokens that match
    at least one node go to `node_tokens`; tokens that don't go to
    `edge_tokens` (will be looked up in edge content via `scan_edges_by_content`).

    Returns:
        (node_tokens, edge_tokens) — both ordered as in input.
    """
    import re
    _morph_suffixes = (
        "s", "es", "ed", "ing", "er", "est", "ly",
        "ness", "ment", "tion", "sion", "ation",
        "ous", "ious", "ful", "less", "able", "ible",
        "ive", "al", "ial", "ical", "ity", "ty",
        "ence", "ance", "dom", "ship", "ism", "ist",
        "ize", "ise", "ify", "en",
    )

    def _is_morph_prefix(shorter: str, longer: str) -> bool:
        if not longer.startswith(shorter):
            return False
        suffix = longer[len(shorter):]
        if not suffix:
            return True
        if len(shorter) <= 3:
            return False
        return suffix in _morph_suffixes

    cjk_re = re.compile(r'[一-鿿㐀-䶿]')
    node_tokens: List[str] = []
    edge_tokens: List[str] = []

    for token in tokens:
        token_l = token.lower()
        is_cjk = bool(cjk_re.search(token))
        matched = False
        for name in engine._nodes:
            name_lower = name.lower()
            # CJK substring
            if is_cjk and token_l in name_lower:
                matched = True
                break
            # Word-boundary / morphological match for ASCII
            name_parts = re.split(r'[_:\-]', name_lower)
            words: List[str] = []
            for p in name_parts:
                words.extend(
                    w.lower() for w in re.findall(r'[a-z]+|[A-Z][a-z]*|\d+', p)
                    if len(w) >= 2
                )
            if token_l in words or any(
                _is_morph_prefix(token_l, w) or _is_morph_prefix(w, token_l)
                for w in words
            ):
                matched = True
                break

        if matched:
            node_tokens.append(token)
        else:
            edge_tokens.append(token)

    return node_tokens, edge_tokens


def scan_edges_by_content(
    engine: "STGEngine",
    edge_tokens: List[str],
    fields: Tuple[str, ...] = EDGE_SCAN_FIELDS,
    max_hits: int = DEFAULT_MAX_EDGE_HITS,
    min_token_length: int = DEFAULT_MIN_EDGE_TOKEN_LENGTH,
) -> List[Tuple["STGEdge", List[str], float]]:
    """R6 edge content scan — find edges whose meta semantic fields contain
    any of the given tokens.

    For each edge, concatenate the values of `fields` (description, lesson,
    action, role, status, is_a) and substring-match against `edge_tokens`.
    Edges with at least one matched token are scored by IDF (tokens that
    appear in fewer edges score higher) and capped at `max_hits`.

    Memory-never-vanishes principle: this function adds candidate edges to
    the recall result, never removes anything from the propagate output.

    Args:
        engine: STGEngine
        edge_tokens: tokens that did NOT match any node name (from classify_tokens)
        fields: edge modifier keys to scan (default: EDGE_SCAN_FIELDS)
        max_hits: cap on returned hits (default 50)
        min_token_length: drop tokens shorter than this (avoid noise)

    Returns:
        List of (edge, matched_tokens, idf_score), sorted by score desc.
        Empty if no matches.
    """
    # Filter short tokens
    effective_tokens = [t.lower() for t in edge_tokens if len(t) >= min_token_length]
    if not effective_tokens:
        return []

    # Single pass: collect hits and accumulate document frequencies
    raw_hits: List[Tuple["STGEdge", List[str]]] = []
    token_df: Dict[str, int] = {t: 0 for t in effective_tokens}

    for edge in engine._edges:
        text_parts: List[str] = []
        for f in fields:
            v = edge.modifiers.get(f)
            if v is None:
                continue
            text_parts.append(str(v))
        if not text_parts:
            continue
        text = " ".join(text_parts).lower()

        matched: List[str] = []
        for token in effective_tokens:
            if token in text:
                matched.append(token)
        if matched:
            raw_hits.append((edge, matched))
            for t in matched:
                token_df[t] = token_df.get(t, 0) + 1

    if not raw_hits:
        return []

    # IDF: tokens appearing in fewer edges score higher
    N = max(1, len(engine._edges))
    token_idf = {t: math.log(N / (1 + df)) for t, df in token_df.items()}

    # Score = sum of matched-token IDFs (capped at >= 0; log of small N
    # over large df can be negative — clamp to 0 to avoid penalizing edges
    # that match very-common tokens)
    scored: List[Tuple["STGEdge", List[str], float]] = []
    for edge, matched in raw_hits:
        score = sum(max(0.0, token_idf.get(t, 0.0)) for t in matched)
        scored.append((edge, matched, score))

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:max_hits]


def multi_seed_propagate(
    engine: "STGEngine",
    query: str,
    tokens: List[str],
    min_chain_length: int = DEFAULT_MIN_CHAIN_LENGTH,
    use_gravity: bool = True,
    resolution: str = "medium",
) -> Tuple[List[str], List[Tuple[str, List[str], List[List[str]]]]]:
    """R2 chain intersection wrapper for multi-token queries.

    For each query token: runs engine.propagate(token) as a black box,
    reconstructs chains via stl_parser STLGraph.extract_chains, collects
    the set of nodes appearing in those chains. Returns the intersection
    across all tokens (nodes reachable from every seed).

    Falls back to the union if intersection is empty — preserves recall
    completeness per §2.4 memory-never-vanishes.

    Does not modify propagate / Rust core / gravity. Pure wrapper.

    Args:
        engine: STGEngine
        query: original query string (used for diagnostic / display)
        tokens: pre-tokenized list (e.g. from _split_tokens(query))
        min_chain_length: passed to STLGraph.extract_chains (default 2)
        use_gravity: route per-token sub-propagate through gravitational
            propagate when True (default — matches CLI default)
        resolution: gravity resolution layer

    Returns:
        Tuple of (intersection_nodes, per_token_data) where:
        - intersection_nodes: list of node names sorted by chain
          appearance count descending. Display-cased (preserves the
          casing returned by propagate).
        - per_token_data: list of (token, activated, chains) tuples
          carrying enough context for downstream display of which
          chains converged.

    See STG_PRECISION_RECALL_DESIGN.md §4.3.
    """
    from stl_parser.graph import STLGraph

    per_token_data: List[Tuple[str, List[str], List[List[str]]]] = []
    nodes_per_token: List[set] = []
    name_lookup: Dict[str, str] = {}  # lower → display case

    for token in tokens:
        if use_gravity:
            from stg_engine.gravity import gravitational_propagate
            gravity_map = engine.get_gravity_map()
            activated = gravitational_propagate(
                engine, token, gravity_map, resolution=resolution,
            )
        else:
            activated = engine.propagate(token)

        if not activated or len(activated) < 2:
            continue

        for n in activated:
            name_lookup.setdefault(n.lower(), n)

        activated_lower = {n.lower() for n in activated}
        try:
            subgraph = engine._graph.subgraph(activated_lower).copy()
        except Exception:
            subgraph = None

        chains: List[List[str]] = []
        if subgraph is not None and subgraph.number_of_edges() > 0:
            try:
                stl_g = STLGraph.from_networkx(subgraph)
                chains = stl_g.extract_chains(min_length=min_chain_length)
            except Exception:
                chains = []

        # Set of nodes participating in this token's reachable subgraph.
        # Includes activated singletons (no-chain leaf nodes) so they
        # are not lost from the intersection.
        nodes_in_subgraph = set(activated_lower)
        for chain in chains:
            nodes_in_subgraph.update(n.lower() for n in chain)

        nodes_per_token.append(nodes_in_subgraph)
        per_token_data.append((token, list(activated), chains))

    if not nodes_per_token:
        return [], []

    intersection = set.intersection(*nodes_per_token)
    if not intersection:
        # Fallback: union (memory-never-vanishes — never empty if any
        # token had hits).
        intersection = set.union(*nodes_per_token)

    # Rank by chain appearance count across all per-token chain sets.
    appearance: Dict[str, int] = defaultdict(int)
    for _, _, chains in per_token_data:
        for chain in chains:
            seen_in_chain: set = set()
            for n in chain:
                k = n.lower()
                if k in intersection and k not in seen_in_chain:
                    appearance[k] += 1
                    seen_in_chain.add(k)
        # Activated-but-no-chain singletons get appearance 0; ranked last.

    sorted_keys = sorted(
        intersection,
        key=lambda k: (appearance.get(k, 0), -len(k)),  # tiebreaker: shorter name first
        reverse=True,
    )
    display_names = [name_lookup.get(k, k) for k in sorted_keys]

    return display_names, per_token_data


def community_dominance_filter(
    community_results: List["CommunityPropagateResult"],
    ratio: float = DEFAULT_DOMINANCE_RATIO,
) -> List["CommunityPropagateResult"]:
    """Fold weak communities below dominance threshold.

    Given communities sorted by score descending (output of
    aggregate_to_communities), keep all whose score >= dominant_score / ratio.
    Once one drops below threshold, stop — subsequent communities are weaker.

    The dominant community is always returned (even if degenerate score=0).

    Args:
        community_results: list sorted by score descending
        ratio: dominant/weak threshold (default 3.0 — weak community must be
            at least 1/3 of dominant's score to survive)

    Returns:
        Filtered list. Length >= 1 if input non-empty.
    """
    if not community_results:
        return []

    dominant = community_results[0]
    if dominant.score <= 0:
        return community_results  # degenerate — pass through

    threshold = dominant.score / ratio
    kept = [dominant]
    for c in community_results[1:]:
        # Keep if score above threshold, OR if community contains query_seeds
        # (precise query hits that should never be hidden — protects against
        # R7 collapsing a community whose top representatives happen to have
        # zero activation but whose internal query-matching nodes are real).
        if c.score >= threshold or c.query_seeds:
            kept.append(c)
        elif c.score < threshold and not c.query_seeds:
            # Once a community drops below threshold AND has no query seeds,
            # subsequent ones (sorted by score desc) are even weaker — but
            # they may still have query_seeds, so we have to keep scanning.
            continue
    return kept
