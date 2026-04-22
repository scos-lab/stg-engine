"""STG Gravitational Propagation — structure-aware activation.

The graph's community structure IS the gravity.
Nodes in hub positions (high elevation) attract activation.
Nodes buried inside communities (low elevation) are suppressed.

No manual labeling needed — elevation emerges from topology.

Core idea: good structure = intelligence.
Not designing how water flows, but designing gravity itself.

Usage:
    gravity = build_gravity_map(engine)
    results = gravitational_propagate(engine, "query", gravity)
"""

from __future__ import annotations

import math
import time as _time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import networkx as nx

if TYPE_CHECKING:
    from stg_engine.engine import STGEngine

# Hot-path core: optional Rust, pure-Python fallback
try:
    from stg_engine import _rust_core as _rust
except ImportError:
    from stg_engine import _core_fallback as _rust


@dataclass
class GravityMap:
    """Cached multi-resolution community structure.

    Three resolution levels capture different event granularities:
    - coarse: project-level (3-10 communities)
    - medium: phase-level (20-50 communities)
    - fine: detail-level (100+ communities)
    """

    # Node → community ID at each resolution
    node_community: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Community ID → representative nodes at each resolution
    representatives: Dict[str, List[str]] = field(default_factory=dict)

    # Node → elevation (structural importance) — default (medium) layer
    node_elevation: Dict[str, float] = field(default_factory=dict)

    # Node → elevation at each resolution layer
    elevation_by_resolution: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Community counts per resolution
    community_counts: Dict[str, int] = field(default_factory=dict)

    # Community names: "medium_0" → "Spreading_Activation" (top representative)
    community_names: Dict[str, str] = field(default_factory=dict)

    # Build metadata
    built_at: float = 0.0
    node_count: int = 0
    edge_count: int = 0


def build_gravity_map(
    engine: "STGEngine",
    resolutions: Tuple[float, float, float] = (0.3, 1.0, 2.0),
    representatives_per_community: int = 3,
) -> GravityMap:
    """Build the gravity map from graph topology.

    1. Run Louvain community detection at three resolutions
    2. Identify representative nodes per community (highest local PageRank)
    3. Compute elevation for every node

    Args:
        engine: STGEngine instance
        resolutions: (coarse, medium, fine) resolution parameters
        representatives_per_community: max representatives per community

    Returns:
        GravityMap with elevation scores for all nodes
    """
    from networkx.algorithms.community import louvain_communities

    graph = engine._graph
    if graph.number_of_nodes() == 0:
        return GravityMap(built_at=_time.time())

    undirected = graph.to_undirected()
    resolution_names = ("coarse", "medium", "fine")

    # Step 1: Community detection at three resolutions
    node_community: Dict[str, Dict[str, int]] = defaultdict(dict)
    all_communities: Dict[str, List[Set[str]]] = {}
    community_counts: Dict[str, int] = {}

    for res_name, res_val in zip(resolution_names, resolutions):
        try:
            raw = louvain_communities(undirected, resolution=res_val, seed=42)
        except Exception:
            raw = [set(graph.nodes())]

        communities = sorted(raw, key=len, reverse=True)
        all_communities[res_name] = communities
        community_counts[res_name] = len(communities)

        for comm_id, members in enumerate(communities):
            for member in members:
                node_community[member][res_name] = comm_id

    # Step 2: Compute local importance within each medium-resolution community
    # Use medium resolution as the primary level for elevation
    importance = engine.get_importance_field()
    representatives: Dict[str, List[str]] = {}

    for res_name in resolution_names:
        for comm_id, members in enumerate(all_communities[res_name]):
            key = f"{res_name}_{comm_id}"
            # Sort members by global importance (proxy for local importance)
            ranked = sorted(members, key=lambda n: importance.get(n, 0.0), reverse=True)
            representatives[key] = ranked[:representatives_per_community]

    # Step 3: Compute elevation at each resolution
    elevation_by_resolution: Dict[str, Dict[str, float]] = {}
    for res_name in resolution_names:
        elevation_by_resolution[res_name] = _compute_all_elevations(
            engine, all_communities[res_name], importance,
        )

    # Default elevation uses medium layer
    node_elevation = elevation_by_resolution["medium"]

    # Step 4: Name each community by its top representative
    community_names: Dict[str, str] = {}
    for key, reps in representatives.items():
        if reps:
            community_names[key] = reps[0]

    return GravityMap(
        node_community=dict(node_community),
        representatives=representatives,
        node_elevation=node_elevation,
        elevation_by_resolution=elevation_by_resolution,
        community_counts=community_counts,
        community_names=community_names,
        built_at=_time.time(),
        node_count=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
    )


def _compute_all_elevations(
    engine: "STGEngine",
    communities: List[Set[str]],
    importance: Dict[str, float],
) -> Dict[str, float]:
    """Compute elevation for every node.

    elevation = local_importance × bridge_factor × community_weight

    local_importance: node's PageRank normalized within its community (0-1)
    bridge_factor: 1 + log(1 + cross_community_edges)
    community_weight: log(community_size) — being representative of a
        3-node community is less meaningful than a 300-node community
    """
    graph = engine._graph
    total_nodes = graph.number_of_nodes()
    if total_nodes == 0:
        return {}

    nodes_list = list(graph.nodes())
    neighbors_dict: Dict[str, List[str]] = {}
    for node in nodes_list:
        neigh = set(graph.successors(node)) | set(graph.predecessors(node))
        if neigh:
            neighbors_dict[node] = list(neigh)

    communities_list = [list(c) for c in communities]

    return _rust.compute_elevations(
        nodes_list,
        communities_list,
        importance,
        neighbors_dict,
    )


def gravitational_propagate(
    engine: "STGEngine",
    query: str,
    gravity: GravityMap,
    elevation_weight: float = 0.5,
    resolution: str = "medium",
    **propagate_kwargs,
) -> List[str]:
    """Run propagate with gravity-aware activation weighting.

    After standard propagation, adjust each node's activation by its elevation:
        adjusted = raw_activation × (elevation ^ elevation_weight)

    High-elevation nodes (community representatives, bridge nodes) are amplified.
    Low-elevation nodes (internal fragments) are suppressed.

    Args:
        engine: STGEngine instance
        query: propagate query string
        gravity: pre-built GravityMap
        elevation_weight: 0=ignore elevation (standard), 1=full gravity effect
        resolution: "coarse" (project-level), "medium" (phase-level), "fine" (detail-level)
        **propagate_kwargs: passed to engine.propagate()

    Returns:
        List of activated node names, sorted by gravity-adjusted activation
    """
    # Run standard propagation — returns only above-threshold nodes
    activated = engine.propagate(query, **propagate_kwargs)

    if not activated:
        return []

    # Read activation values from the activated nodes only
    raw_activations: Dict[str, float] = {}
    for name in activated:
        node = engine._nodes.get(name.lower())
        if node and node.activation > 0:
            raw_activations[name] = node.activation

    if not raw_activations:
        return []

    # Select elevation map by resolution
    elevations = gravity.elevation_by_resolution.get(resolution, gravity.node_elevation)

    # Apply gravity: adjust activation by elevation
    adjusted: Dict[str, float] = {}
    for name, raw_act in raw_activations.items():
        elev = elevations.get(name.lower(), 0.01)
        # elevation^weight: weight=0 → all 1.0 (no effect), weight=1 → full elevation
        gravity_factor = max(0.01, elev) ** elevation_weight
        adjusted[name] = raw_act * gravity_factor

    # Update node activations with adjusted values
    for name, adj_act in adjusted.items():
        if name.lower() in engine._nodes:
            engine._nodes[name.lower()].activation = adj_act

    # Sort and return
    results = sorted(adjusted.items(), key=lambda x: x[1], reverse=True)

    # Update propagation metrics top_nodes with adjusted values
    if engine._last_propagation_metrics:
        engine._last_propagation_metrics.top_nodes = [
            (name, act) for name, act in results[:10]
        ]
        engine._last_propagation_metrics.activated_node_count = len(results)

    return [name for name, _ in results]


def gravity_info(gravity: GravityMap) -> Dict:
    """Summary statistics of a GravityMap."""
    elevations = list(gravity.node_elevation.values())
    if not elevations:
        return {"empty": True}

    sorted_by_elev = sorted(
        gravity.node_elevation.items(), key=lambda x: x[1], reverse=True,
    )

    # Build community listing with names and sizes
    community_listing: Dict[str, List[Tuple[str, str, int]]] = {}
    for res_name in ("coarse", "medium", "fine"):
        entries = []
        count = gravity.community_counts.get(res_name, 0)
        for i in range(count):
            key = f"{res_name}_{i}"
            name = gravity.community_names.get(key, f"#{i}")
            # Count members
            size = sum(
                1 for comms in gravity.node_community.values()
                if comms.get(res_name) == i
            )
            entries.append((key, name, size))
        community_listing[res_name] = entries

    return {
        "communities": gravity.community_counts,
        "community_listing": community_listing,
        "total_nodes": gravity.node_count,
        "total_edges": gravity.edge_count,
        "elevation_min": min(elevations),
        "elevation_max": max(elevations),
        "elevation_mean": sum(elevations) / len(elevations),
        "top_10": [(name, f"{elev:.3f}") for name, elev in sorted_by_elev[:10]],
        "bottom_10": [(name, f"{elev:.3f}") for name, elev in sorted_by_elev[-10:]],
        "built_at": gravity.built_at,
    }


def community_name_for(
    gravity: GravityMap,
    node_name: str,
    resolution: str = "medium",
) -> Optional[str]:
    """Return the community name for a node at a given resolution.

    Community name = the top representative node's name.
    Returns None if node is not in the gravity map.
    """
    _key = node_name.lower()
    communities = gravity.node_community.get(_key, {})
    comm_id = communities.get(resolution)
    if comm_id is None:
        return None
    key = f"{resolution}_{comm_id}"
    return gravity.community_names.get(key)


DEFAULT_HALFLIFE_DAYS = 30.0
DEFAULT_BASELINE_SCALE = 1.0
# Heat sigmoid half-saturation (see usage below).
# Heat sigmoid half-saturation: effective_heat value at which normalized_heat = 0.5.
# Without this, raw heat is unbounded (Σ salience·exp(-λΔt)) and dominates the
# score formula, drowning out rep_activation differences. Sigmoid saturates at 1
# so heat becomes a bounded "temperature multiplier" rather than a primary factor.
DEFAULT_HEAT_HALF_SATURATION = 5.0


def _normalize_for_match(s: str) -> str:
    """Normalize separators for query↔community_name matching.

    Users type 'website factory' (spaces) but node names use 'website_factory'
    (underscores). Without normalization these don't match, and the community
    misses its name_boost. Treats _ - . as equivalent to space, lowercases.
    """
    if not s:
        return ""
    out = s.lower()
    for ch in ("_", "-", "."):
        out = out.replace(ch, " ")
    return " ".join(out.split())  # collapse runs of whitespace


def compute_community_signals(
    engine: "STGEngine",
    gravity: GravityMap,
    touched_community_ids: List[int],
    resolution: str = "medium",
    now: Optional[float] = None,
    halflife_days: float = DEFAULT_HALFLIFE_DAYS,
    baseline_scale: float = DEFAULT_BASELINE_SCALE,
    heat_half_saturation: float = DEFAULT_HEAT_HALF_SATURATION,
    k: int = 3,
) -> Dict[int, Dict[str, float]]:
    """Compute community heat / recency / baseline purely from existing edge state.

    No writes. No stored community state. All values derived at query time from:
      - edge.salience   (maintained by Hebbian learning during propagate)
      - edge.last_used  (maintained by Hebbian learning during propagate)
      - gravity.node_elevation (maintained by GravityMap build)

    For each touched community c:
        heat(c)     = Σ_{e ∈ internal(c)} salience(e) · exp(-λ · (now - last_used(e)))
        recency(c)  = exp(-λ · (now - max_{e ∈ internal(c)} last_used(e)))
        baseline(c) = mean_elevation(top_k reps of c) / max_elevation_globally
        effective_heat(c) = max(heat(c), baseline(c) · baseline_scale)

    An edge is "internal to c" iff both its source and target map to c at the
    chosen resolution. Cross-community bridge edges are excluded — they are not
    part of any single community's internal activity.

    Args:
        engine: STGEngine holding edge state
        gravity: GravityMap (source of node_community + node_elevation)
        touched_community_ids: communities to compute (at `resolution`)
        resolution: which gravity layer to use
        now: reference time (default: time.time())
        halflife_days: decay half-life in days (λ = ln(2) / halflife_seconds)
        baseline_scale: scale factor for structural baseline floor
        k: number of top representatives used for baseline mean

    Returns:
        {community_id: {"heat": float, "recency": float,
                        "baseline": float, "effective_heat": float}}
        Communities in touched_community_ids with no internal edges or no
        reps still get an entry — heat=0, recency=0, baseline computed.
    """
    if now is None:
        now = _time.time()
    if not touched_community_ids:
        return {}

    lam = math.log(2.0) / (halflife_days * 86400.0)
    touched_set = set(touched_community_ids)

    # Initialize per-community accumulators
    heat: Dict[int, float] = {c: 0.0 for c in touched_set}
    max_last: Dict[int, float] = {c: 0.0 for c in touched_set}

    # Single pass over all edges; charge communities where source+target both
    # land in the same touched community at the chosen resolution.
    for e in engine._edges:
        src_comms = gravity.node_community.get(e.source.lower(), {})
        tgt_comms = gravity.node_community.get(e.target.lower(), {})
        src_c = src_comms.get(resolution)
        if src_c is None or src_c not in touched_set:
            continue
        tgt_c = tgt_comms.get(resolution)
        if tgt_c != src_c:
            continue
        lu = e.last_used
        if lu is None or lu <= 0.0:
            continue
        dt = now - lu
        if dt < 0:
            dt = 0.0
        weight = (e.salience if e.salience is not None else 0.0) * math.exp(-lam * dt)
        heat[src_c] += weight
        if lu > max_last[src_c]:
            max_last[src_c] = lu

    # Baseline from elevation (structural importance)
    all_elevs = gravity.elevation_by_resolution.get(resolution, gravity.node_elevation)
    max_elev = max(all_elevs.values()) if all_elevs else 0.0

    out: Dict[int, Dict[str, float]] = {}
    for c in touched_set:
        # Recency
        if max_last[c] > 0:
            recency = math.exp(-lam * (now - max_last[c]))
        else:
            recency = 0.0

        # Baseline: mean elevation of top-k reps, normalized by global max
        comm_key = f"{resolution}_{c}"
        reps = gravity.representatives.get(comm_key, [])[:k]
        if reps and max_elev > 0:
            mean_rep_elev = sum(all_elevs.get(r.lower(), 0.0) for r in reps) / len(reps)
            baseline = mean_rep_elev / max_elev
        else:
            baseline = 0.0

        effective = max(heat[c], baseline * baseline_scale)
        # Sigmoid normalization: unbounded raw heat → saturated [0, 1) temperature.
        # Keeps heat as a mild temperature modifier rather than a dominant factor.
        if heat_half_saturation > 0:
            normalized = effective / (effective + heat_half_saturation)
        else:
            normalized = effective
        out[c] = {
            "heat": heat[c],
            "recency": recency,
            "baseline": baseline,
            "effective_heat": effective,
            "normalized_heat": normalized,
        }
    return out


def aggregate_to_communities(
    engine: "STGEngine",
    activated: List[str],
    gravity: GravityMap,
    resolution: str = "medium",
    k: int = 3,
    query: str = "",
    name_boost: float = 2.0,
    top_m: int = 5,
    halflife_days: float = DEFAULT_HALFLIFE_DAYS,
    baseline_scale: float = DEFAULT_BASELINE_SCALE,
    alpha_heat: float = 0.5,
    beta_recency: float = 0.3,
    now: Optional[float] = None,
) -> List:
    """Aggregate node activations into community-level results (Phase 7I, M1).

    For each community touched by the activation set, the community's signal
    is read from its top-k representatives' activations — NOT summed over all
    member nodes. Gravity has already concentrated activation on structural
    peaks (representatives), so reading the peaks IS the measurement.
    Integrating the full community would re-do gravity's work.

    Args:
        engine: STGEngine with active node activations
        activated: list of node names with nonzero activation
        gravity: pre-built GravityMap
        resolution: "coarse" | "medium" | "fine"
        k: representatives per community to read
        query: query string for name-match boost
        name_boost: multiplier when query substring matches community_name
        top_m: return at most this many communities

    Returns:
        List[CommunityPropagateResult] sorted by score descending.
        Communities with zero representative activation are skipped.
    """
    from stg_engine.types import CommunityPropagateResult, RepresentativeEntry

    if not activated:
        return []

    activated_set = {n.lower() for n in activated}
    query_lower = query.lower().strip()
    # Normalized form handles space/underscore/hyphen as equivalent.
    query_norm = _normalize_for_match(query)

    # Group activated nodes by community at the chosen resolution
    touched: Dict[int, List[str]] = defaultdict(list)
    for name in activated:
        key = name.lower()
        comms = gravity.node_community.get(key, {})
        comm_id = comms.get(resolution)
        if comm_id is None:
            continue
        touched[comm_id].append(name)

    if not touched:
        return []

    elevations = gravity.elevation_by_resolution.get(resolution, gravity.node_elevation)

    # M2: compute heat/recency/baseline for every touched community in a single pass
    signals = compute_community_signals(
        engine, gravity,
        list(touched.keys()),
        resolution=resolution,
        now=now,
        halflife_days=halflife_days,
        baseline_scale=baseline_scale,
        k=k,
    )

    results: List[CommunityPropagateResult] = []

    for comm_id, member_names in touched.items():
        comm_key = f"{resolution}_{comm_id}"
        reps_full = gravity.representatives.get(comm_key, [])

        # Read activation only from the top-k representatives (structural peaks).
        # If a representative has no activation (was not reached by propagation),
        # its contribution is 0 — preserving the "peak signal" semantics.
        rep_entries: List[RepresentativeEntry] = []
        rep_acts: List[float] = []
        for rep_name in reps_full[:k]:
            rep_key = rep_name.lower()
            node = engine._nodes.get(rep_key)
            act = node.activation if (node and rep_key in activated_set) else 0.0
            elev = elevations.get(rep_key, 0.0)
            rep_entries.append(
                RepresentativeEntry(node_name=rep_name, activation=act, elevation=elev)
            )
            rep_acts.append(act)

        rep_activation = sum(rep_acts) / len(rep_acts) if rep_acts else 0.0

        # Identify query-matching nodes inside this community that are NOT
        # top-k reps. These are precise hits that would otherwise vanish
        # at the community level but users explicitly asked about them.
        # Use normalized match so "website factory" finds "website_factory_deploy" etc.
        rep_key_set = {r.lower() for r in reps_full[:k]}
        query_seeds_entries: List[RepresentativeEntry] = []
        if query_norm:
            for mname in member_names:
                m_key = mname.lower()
                if m_key in rep_key_set:
                    continue
                if query_norm not in _normalize_for_match(mname):
                    continue
                m_node = engine._nodes.get(m_key)
                if not m_node:
                    continue
                query_seeds_entries.append(
                    RepresentativeEntry(
                        node_name=mname,
                        activation=m_node.activation,
                        elevation=elevations.get(m_key, 0.0),
                    )
                )
            query_seeds_entries.sort(key=lambda r: r.activation, reverse=True)
            query_seeds_entries = query_seeds_entries[:3]

        # Skip communities that were only touched through non-representative
        # members WITHOUT a query match — they did not truly "light up" at
        # the peak and have no precise hit either. Keep communities with
        # query_seeds even if reps are cold — user explicitly asked about them.
        if rep_activation <= 0.0 and not query_seeds_entries:
            continue

        comm_name = gravity.community_names.get(comm_key, comm_key)
        # Normalized comparison: "website factory" matches "website_factory" community.
        matched = bool(query_norm) and query_norm in _normalize_for_match(comm_name)
        name_mult = name_boost if matched else 1.0

        # M2: read derived signals instead of 1.0 placeholders
        sig = signals.get(comm_id, {})
        heat = sig.get("heat", 0.0)
        recency = sig.get("recency", 0.0)
        baseline = sig.get("baseline", 0.0)
        effective_heat = sig.get("effective_heat", 0.0)
        normalized_heat = sig.get("normalized_heat", 0.0)

        # Score uses NORMALIZED heat (bounded [0,1]). Without normalization,
        # a hot community would dominate score ~20x over query-matched cold ones.
        score = name_mult * rep_activation * (
            1.0 + alpha_heat * normalized_heat + beta_recency * recency
        )

        # Boost score a bit when the community contains direct query matches
        # that aren't reps — ensures such communities can't be drowned out by
        # merely-hot ones.
        if query_seeds_entries:
            seed_boost = 1.0 + 0.5 * len(query_seeds_entries)
            score *= seed_boost

        results.append(
            CommunityPropagateResult(
                community_key=comm_key,
                community_name=comm_name,
                score=score,
                rep_activation=rep_activation,
                heat=heat,
                recency=recency,
                baseline_heat=baseline,
                name_matched=matched,
                representatives=rep_entries,
                query_seeds=query_seeds_entries,
            )
        )

    # Two-tier sort (Phase 7I usability P2, 2026-04-19):
    #   Tier 0: community_name matched the query AND has rep activation > 0
    #   Tier 1: everything else
    # Within each tier, sort by score descending.
    # Prevents structural hub communities from burying explicit name-match hits
    # (e.g. "website factory" → website_factory community must surface even
    # though skc has higher rep_activation through cross-community bridges).
    results.sort(key=lambda r: (
        0 if (r.name_matched and r.rep_activation > 0) else 1,
        -r.score,
    ))
    return results[:top_m]


def gravity_node_info(
    gravity: GravityMap,
    node_name: str,
) -> Optional[Dict]:
    """Detailed gravity info for a specific node."""
    _key = node_name.lower()
    if _key not in gravity.node_elevation:
        return None

    elev = gravity.node_elevation[_key]
    communities = gravity.node_community.get(_key, {})

    # Find if this node is a representative at any level
    rep_roles = []
    for key, reps in gravity.representatives.items():
        if _key in reps:
            rep_roles.append(key)

    # Elevation percentile
    all_elevations = sorted(gravity.node_elevation.values())
    rank = sum(1 for e in all_elevations if e <= elev)
    percentile = rank / len(all_elevations) * 100 if all_elevations else 0

    # Resolve community names for each resolution
    community_labels = {}
    for res, comm_id in communities.items():
        key = f"{res}_{comm_id}"
        name = gravity.community_names.get(key)
        community_labels[res] = name or f"#{comm_id}"

    return {
        "node": node_name,
        "elevation": elev,
        "percentile": percentile,
        "communities": communities,
        "community_names": community_labels,
        "representative_of": rep_roles,
    }
