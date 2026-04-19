"""STG Topology Optimization — Restructuring the circuit board.

Phase 7C: Community detection, bridge discovery, and redundancy elimination.

CommunityDetector: Finds natural clusters via Louvain, compares with namespaces.
BridgeDiscoverer:  Suggests missing inter-community connections.
RedundancyEliminator: Removes informationally redundant edges (low EID).
TopologyOptimizer: Facade composing all three operators.

All operators are read-only by default — analyze first, apply on explicit request.
"""

import time
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import networkx as nx
from networkx.algorithms.community import louvain_communities, modularity

if TYPE_CHECKING:
    from stg_engine.engine import STGEngine

from stg_engine.types import CommunityInfo, BridgeSuggestion, TopologyReport


# ═══════════════════════════════════════════════════════════
# Community Detection
# ═══════════════════════════════════════════════════════════


class CommunityDetector:
    """Discover natural community structure using Louvain algorithm.

    Runs on the undirected projection of the STG graph.
    Compares detected communities against STG namespaces to measure
    structural alignment with human-curated categorization.
    """

    def __init__(
        self,
        resolution: float = 1.0,
        min_community_size: int = 3,
    ) -> None:
        if resolution <= 0:
            raise ValueError(f"resolution must be > 0, got {resolution}")
        self.resolution = resolution
        self.min_community_size = max(1, min_community_size)

    def detect(self, engine: "STGEngine") -> List[CommunityInfo]:
        """Run community detection.

        Returns list of CommunityInfo sorted by size descending.
        Small communities below min_community_size are merged into
        a single 'misc' bucket at the end.
        """
        if engine._graph.number_of_nodes() == 0:
            return []

        undirected = engine._graph.to_undirected()
        raw = louvain_communities(undirected, resolution=self.resolution)

        # Split significant vs small
        significant: List[List[str]] = []
        misc_members: List[str] = []
        for comm in raw:
            members = sorted(comm)
            if len(members) >= self.min_community_size:
                significant.append(members)
            else:
                misc_members.extend(members)

        # Sort by size descending
        significant.sort(key=len, reverse=True)

        results: List[CommunityInfo] = []
        for idx, members in enumerate(significant):
            results.append(self._analyze_community(engine, idx, members))

        if misc_members:
            results.append(
                self._analyze_community(engine, len(results), misc_members)
            )

        return results

    def compute_modularity(
        self, engine: "STGEngine", communities: List[CommunityInfo]
    ) -> float:
        """Compute modularity Q for a community partition.

        Returns modularity score. Higher = stronger community structure.
        """
        if not communities:
            return 0.0

        undirected = engine._graph.to_undirected()
        partition = [set(c.members) for c in communities]

        try:
            return modularity(undirected, partition)
        except (nx.NetworkXError, ZeroDivisionError):
            return 0.0

    def find_misplaced_nodes(
        self, engine: "STGEngine", communities: List[CommunityInfo]
    ) -> List[Tuple[str, str, str]]:
        """Find nodes whose namespace doesn't match community's dominant namespace.

        Returns list of (node_name, actual_namespace, expected_namespace).
        Skips nodes with no namespace and communities with no dominant namespace.
        """
        results: List[Tuple[str, str, str]] = []
        for comm in communities:
            if not comm.dominant_namespace:
                continue
            for name in comm.members:
                node = engine._nodes.get(name)
                if not node or not node.namespace:
                    continue
                if node.namespace != comm.dominant_namespace:
                    results.append((name, node.namespace, comm.dominant_namespace))
        return results

    def _analyze_community(
        self, engine: "STGEngine", idx: int, members: List[str]
    ) -> CommunityInfo:
        """Analyze a single community's properties."""
        size = len(members)
        member_set = set(members)

        # Dominant namespace
        ns_counts: Counter = Counter()
        for name in members:
            node = engine._nodes.get(name)
            if node and node.namespace:
                ns_counts[node.namespace] += 1

        dominant_ns = ns_counts.most_common(1)[0][0] if ns_counts else None
        purity = ns_counts.most_common(1)[0][1] / size if ns_counts else 0.0

        # Internal density: actual internal edges / max possible
        internal_edges = 0
        for name in members:
            for succ in engine._graph.successors(name):
                if succ in member_set:
                    internal_edges += 1
        max_possible = size * (size - 1) if size > 1 else 1
        density = internal_edges / max_possible

        return CommunityInfo(
            community_id=idx,
            members=members,
            size=size,
            dominant_namespace=dominant_ns,
            namespace_purity=purity,
            internal_density=density,
        )


# ═══════════════════════════════════════════════════════════
# Bridge Discovery
# ═══════════════════════════════════════════════════════════


class BridgeDiscoverer:
    """Discover missing inter-community connections.

    For each pair of communities with no strong edge between them,
    suggests bridge candidates using three strategies:
    1. Shared neighbor (highest confidence)
    2. Hub connection (medium confidence)
    3. Name similarity (lowest confidence)
    """

    def __init__(
        self,
        weak_edge_threshold: float = 0.3,
        max_suggestions_per_pair: int = 2,
        max_total_suggestions: int = 20,
    ) -> None:
        self.weak_edge_threshold = weak_edge_threshold
        self.max_per_pair = max(1, max_suggestions_per_pair)
        self.max_total = max(1, max_total_suggestions)

    def discover(
        self,
        engine: "STGEngine",
        communities: List[CommunityInfo],
    ) -> List[BridgeSuggestion]:
        """Analyze all community pairs and suggest bridges.

        Returns list of BridgeSuggestion sorted by confidence descending,
        capped at max_total_suggestions.
        """
        if len(communities) < 2:
            return []

        # Build membership map: node → community_id
        membership: Dict[str, int] = {}
        for comm in communities:
            for name in comm.members:
                membership[name] = comm.community_id

        # Load hub concept names for strategy 2
        from stg_engine.concept_skeleton import CORE_CONCEPT_NAMES

        suggestions: List[BridgeSuggestion] = []

        for i in range(len(communities)):
            for j in range(i + 1, len(communities)):
                # Count strong existing bridges
                strong_bridges = self._count_strong_bridges(
                    engine, communities[i], communities[j]
                )
                if strong_bridges > 0:
                    continue

                # Generate suggestions for this pair
                pair_suggestions = self._suggest_for_pair(
                    engine, communities[i], communities[j],
                    membership, CORE_CONCEPT_NAMES,
                )
                suggestions.extend(pair_suggestions[:self.max_per_pair])

        # Sort by confidence descending, cap
        suggestions.sort(key=lambda s: s.confidence, reverse=True)
        return suggestions[:self.max_total]

    def apply_suggestions(
        self,
        engine: "STGEngine",
        suggestions: List[BridgeSuggestion],
    ) -> int:
        """Apply bridge suggestions to the engine.

        Creates new edges with suggested confidence and rule='logical'.
        Skips edges that already exist. Returns count of edges created.
        """
        created = 0
        for s in suggestions:
            if engine._edges_lookup.get((s.source.lower(), s.target.lower())) is not None:
                continue
            engine.add_edge(
                s.source, s.target,
                confidence=s.confidence,
                rule="logical",
            )
            created += 1
        return created

    def _count_strong_bridges(
        self,
        engine: "STGEngine",
        comm_a: CommunityInfo,
        comm_b: CommunityInfo,
    ) -> int:
        """Count edges between two communities above the weak threshold."""
        set_a = set(comm_a.members)
        set_b = set(comm_b.members)
        count = 0
        for name in comm_a.members:
            for succ in engine._graph.successors(name):
                if succ in set_b:
                    edge = engine._edges_lookup.get((name, succ))
                    if edge and edge.confidence >= self.weak_edge_threshold:
                        count += 1
        for name in comm_b.members:
            for succ in engine._graph.successors(name):
                if succ in set_a:
                    edge = engine._edges_lookup.get((name, succ))
                    if edge and edge.confidence >= self.weak_edge_threshold:
                        count += 1
        return count

    def _suggest_for_pair(
        self,
        engine: "STGEngine",
        comm_a: CommunityInfo,
        comm_b: CommunityInfo,
        membership: Dict[str, int],
        hub_names: frozenset,
    ) -> List[BridgeSuggestion]:
        """Generate bridge suggestions for one community pair."""
        suggestions: List[BridgeSuggestion] = []
        seen: Set[Tuple[str, str]] = set()

        # Strategy 1: Shared neighbor
        for s in self._strategy_shared_neighbor(
            engine, comm_a, comm_b, membership
        ):
            if (s.source, s.target) not in seen:
                seen.add((s.source, s.target))
                suggestions.append(s)

        # Strategy 2: Hub connection
        for s in self._strategy_hub_connection(
            engine, comm_a, comm_b, hub_names
        ):
            if (s.source, s.target) not in seen:
                seen.add((s.source, s.target))
                suggestions.append(s)

        # Strategy 3: Name similarity
        for s in self._strategy_name_similarity(comm_a, comm_b):
            if (s.source, s.target) not in seen:
                seen.add((s.source, s.target))
                suggestions.append(s)

        # Fallback: if no strategy yielded results, connect highest-degree
        # nodes from each community (ensures disconnected islands get linked)
        if not suggestions:
            best_a = max(
                comm_a.members,
                key=lambda n: engine._graph.degree(n),
            )
            best_b = max(
                comm_b.members,
                key=lambda n: engine._graph.degree(n),
            )
            if not engine._edges_lookup.get((best_a, best_b)):
                suggestions.append(BridgeSuggestion(
                    source=best_a,
                    target=best_b,
                    source_community=comm_a.community_id,
                    target_community=comm_b.community_id,
                    confidence=0.3,
                    rationale="fallback_degree",
                ))

        suggestions.sort(key=lambda s: s.confidence, reverse=True)
        return suggestions

    def _strategy_shared_neighbor(
        self,
        engine: "STGEngine",
        comm_a: CommunityInfo,
        comm_b: CommunityInfo,
        membership: Dict[str, int],
    ) -> List[BridgeSuggestion]:
        """Find nodes from different communities sharing a common neighbor."""
        results: List[BridgeSuggestion] = []
        set_b = set(comm_b.members)

        for node_a in comm_a.members:
            neighbors_a = set(engine._graph.successors(node_a)) | set(
                engine._graph.predecessors(node_a)
            )
            for node_b in comm_b.members:
                neighbors_b = set(engine._graph.successors(node_b)) | set(
                    engine._graph.predecessors(node_b)
                )
                common = neighbors_a & neighbors_b
                if common:
                    results.append(BridgeSuggestion(
                        source=node_a,
                        target=node_b,
                        source_community=comm_a.community_id,
                        target_community=comm_b.community_id,
                        confidence=0.6,
                        rationale="shared_neighbor",
                    ))
                    if len(results) >= self.max_per_pair:
                        return results
        return results

    def _strategy_hub_connection(
        self,
        engine: "STGEngine",
        comm_a: CommunityInfo,
        comm_b: CommunityInfo,
        hub_names: frozenset,
    ) -> List[BridgeSuggestion]:
        """Connect to skeleton hub concepts in the other community."""
        results: List[BridgeSuggestion] = []

        # Find hubs in community B, connect from highest-degree node in A
        hubs_b = [n for n in comm_b.members if n in hub_names]
        if hubs_b:
            # Pick highest-degree node from A
            best_a = max(
                comm_a.members,
                key=lambda n: engine._graph.degree(n),
            )
            results.append(BridgeSuggestion(
                source=best_a,
                target=hubs_b[0],
                source_community=comm_a.community_id,
                target_community=comm_b.community_id,
                confidence=0.5,
                rationale="hub_connection",
            ))

        # Reverse: hubs in A, connect from B
        hubs_a = [n for n in comm_a.members if n in hub_names]
        if hubs_a and len(results) < self.max_per_pair:
            best_b = max(
                comm_b.members,
                key=lambda n: engine._graph.degree(n),
            )
            results.append(BridgeSuggestion(
                source=best_b,
                target=hubs_a[0],
                source_community=comm_b.community_id,
                target_community=comm_a.community_id,
                confidence=0.5,
                rationale="hub_connection",
            ))

        return results

    def _strategy_name_similarity(
        self,
        comm_a: CommunityInfo,
        comm_b: CommunityInfo,
    ) -> List[BridgeSuggestion]:
        """Find nodes with overlapping name tokens."""
        results: List[BridgeSuggestion] = []

        def tokenize(name: str) -> Set[str]:
            tokens = set()
            for part in name.replace(":", "_").split("_"):
                if len(part) > 2:  # Skip tiny fragments
                    tokens.add(part.lower())
            return tokens

        # Pre-tokenize community B
        tokens_b = {name: tokenize(name) for name in comm_b.members}

        for node_a in comm_a.members:
            toks_a = tokenize(node_a)
            if len(toks_a) < 2:
                continue
            for node_b, toks_b in tokens_b.items():
                overlap = toks_a & toks_b
                if len(overlap) >= 2:
                    results.append(BridgeSuggestion(
                        source=node_a,
                        target=node_b,
                        source_community=comm_a.community_id,
                        target_community=comm_b.community_id,
                        confidence=0.4,
                        rationale="name_similarity",
                    ))
                    if len(results) >= self.max_per_pair:
                        return results
        return results


# ═══════════════════════════════════════════════════════════
# Redundancy Elimination
# ═══════════════════════════════════════════════════════════


class RedundancyEliminator:
    """Remove informationally redundant edges based on EID.

    An edge with EID near zero means its removal doesn't change the
    graph's entropy — alternative paths carry the same information.

    Safety: skeleton edges are protected, max_removals caps changes.
    """

    def __init__(
        self,
        eid_threshold: float = 0.001,
        max_removals: int = 200,
        protect_skeleton: bool = True,
    ) -> None:
        self.eid_threshold = eid_threshold
        self.max_removals = max(0, max_removals)
        self.protect_skeleton = protect_skeleton

    def find_redundant(
        self, engine: "STGEngine"
    ) -> List[Tuple[str, str, float]]:
        """Identify redundant edges without modifying the graph.

        Returns list of (source, target, eid) sorted by EID ascending.
        """
        from stg_engine.metrics import compute_all_eid

        if engine._graph.number_of_edges() == 0:
            return []

        # Build protected set
        protected: Set[Tuple[str, str]] = set()
        if self.protect_skeleton:
            from stg_engine.concept_skeleton import SKELETON_EDGES
            for edge_def in SKELETON_EDGES:
                protected.add((edge_def["source"], edge_def["target"]))

        # Compute all EID values
        all_eid = compute_all_eid(engine._graph)

        # Filter and sort
        candidates: List[Tuple[str, str, float]] = []
        for (src, tgt), eid in all_eid.items():
            if eid >= self.eid_threshold:
                continue
            if (src, tgt) in protected:
                continue
            candidates.append((src, tgt, eid))

        candidates.sort(key=lambda x: x[2])
        return candidates[:self.max_removals]

    def eliminate(
        self, engine: "STGEngine"
    ) -> List[Tuple[str, str, float]]:
        """Find and remove redundant edges.

        Returns list of (source, target, eid) for edges actually removed.
        """
        candidates = self.find_redundant(engine)
        removed: List[Tuple[str, str, float]] = []

        for src, tgt, eid in candidates:
            if engine.remove_edge(src, tgt):
                removed.append((src, tgt, eid))

        return removed


# ═══════════════════════════════════════════════════════════
# Topology Optimizer (Facade)
# ═══════════════════════════════════════════════════════════


class TopologyOptimizer:
    """Full topology optimization pipeline.

    Composes CommunityDetector, BridgeDiscoverer, and RedundancyEliminator
    into a single analysis/optimization pass.

    Usage:
        optimizer = TopologyOptimizer()
        report = optimizer.analyze(engine)       # Read-only
        report = optimizer.optimize(engine)      # Apply changes
    """

    def __init__(
        self,
        resolution: float = 1.0,
        min_community_size: int = 3,
        weak_edge_threshold: float = 0.3,
        max_bridge_suggestions: int = 20,
        eid_threshold: float = 0.001,
        max_redundancy_removals: int = 200,
        protect_skeleton: bool = True,
    ) -> None:
        self.detector = CommunityDetector(
            resolution=resolution,
            min_community_size=min_community_size,
        )
        self.discoverer = BridgeDiscoverer(
            weak_edge_threshold=weak_edge_threshold,
            max_total_suggestions=max_bridge_suggestions,
        )
        self.eliminator = RedundancyEliminator(
            eid_threshold=eid_threshold,
            max_removals=max_redundancy_removals,
            protect_skeleton=protect_skeleton,
        )

    def analyze(self, engine: "STGEngine") -> TopologyReport:
        """Run full topology analysis without modifying the graph.

        Returns a TopologyReport with communities, bridge suggestions,
        and redundant edge candidates.
        """
        now = time.time()

        # Step 1: Community detection
        communities = self.detector.detect(engine)
        mod_score = self.detector.compute_modularity(engine, communities)

        # Namespace alignment = average purity
        if communities:
            ns_align = sum(c.namespace_purity for c in communities) / len(communities)
        else:
            ns_align = 0.0

        # Step 2: Bridge discovery
        suggestions = self.discoverer.discover(engine, communities)

        # Count disconnected pairs
        disconnected = 0
        for i in range(len(communities)):
            for j in range(i + 1, len(communities)):
                bridges = self.discoverer._count_strong_bridges(
                    engine, communities[i], communities[j]
                )
                if bridges == 0:
                    disconnected += 1

        # Step 3: Redundancy analysis
        redundant = self.eliminator.find_redundant(engine)

        return TopologyReport(
            communities=communities,
            community_count=len(communities),
            modularity=mod_score,
            namespace_alignment=ns_align,
            bridge_suggestions=suggestions,
            disconnected_pairs=disconnected,
            redundant_edges=redundant,
            redundant_count=len(redundant),
            node_count=len(engine._nodes),
            edge_count=len(engine._edges),
            timestamp=now,
        )

    def optimize(
        self,
        engine: "STGEngine",
        apply_bridges: bool = True,
        apply_redundancy: bool = True,
    ) -> TopologyReport:
        """Run analysis AND apply changes.

        Returns the report (pre-optimization snapshot).
        Callers should save() after optimize() to persist.
        """
        report = self.analyze(engine)

        if apply_bridges and report.bridge_suggestions:
            self.discoverer.apply_suggestions(
                engine, report.bridge_suggestions
            )

        if apply_redundancy and report.redundant_edges:
            self.eliminator.eliminate(engine)

        return report
