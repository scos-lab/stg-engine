"""STG XRef — Cross-Reference Resolution for memory consolidation.

Phase 7C.3: Scan edge descriptions for implicit references to existing
nodes and create virtual edges to surface latent connections.

This is a memory consolidation operation — it reorganizes existing
information, not adds new information. The raw material (description text)
already exists in the graph.

Entry points:
  resolve()      — scan all edges, create virtual edges for cross-references
  resolve_node() — scan edges of one node only (incremental)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from stg_engine.engine import STGEngine


# --- Shared tokenization (same pipeline as propagate) ---

STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "am", "do", "does", "did", "has", "have", "had", "it", "its",
    "what", "who", "how", "why", "when", "where", "which",
    "to", "of", "in", "on", "at", "by", "for", "with", "from",
    "and", "or", "not", "no", "if", "but", "so", "as", "than",
    "me", "my", "we", "us", "you", "he", "she", "they", "them",
    "this", "that", "these", "those", "about", "tell", "describe",
    "explain", "can", "could", "would", "should", "will",
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
    # Programming/technical terms too generic for XRef
    "test", "file", "check", "load", "save", "return", "call",
    "run", "start", "stop", "create", "update", "delete", "add",
    "remove", "send", "read", "write", "open", "close", "handle",
    "error", "result", "value", "type", "name", "list", "data",
    "key", "node", "edge", "path", "source", "target", "input",
    "output", "state", "event", "action", "config", "model",
    "user", "turn", "cache", "token", "parse", "format",
    "true", "false", "null", "none", "default", "auto",
    "based", "using", "used", "use", "without", "within",
    "during", "while", "when", "where", "need", "needs",
    "specific", "current", "available", "support", "process",
})

MORPH_SUFFIXES = frozenset({
    "s", "es", "ed", "ing", "er", "est", "ly",
    "ness", "ment", "tion", "sion", "ation",
    "ous", "ious", "ful", "less", "able", "ible",
    "ive", "al", "ial", "ical", "ity", "ty",
    "ence", "ance", "dom", "ship", "ism", "ist",
    "ize", "ise", "ify", "en",
})

_PUNCT_STRIP = re.compile(r"[^a-z0-9\u4e00-\u9fff\u3400-\u4dbf]")
_HAS_CJK = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')


def tokenize(text: str) -> List[str]:
    """Tokenize text using propagate's pipeline.

    Splits on whitespace, underscores, hyphens, colons.
    Strips trailing punctuation. Filters stop words.
    Returns lowercase tokens (len >= 2, or CJK).
    """
    raw = text.lower().split()
    expanded = []
    for t in raw:
        parts = re.split(r'[_\-:]', t)
        for p in parts:
            if not p:
                continue
            # Strip ALL non-alphanumeric chars (not just trailing)
            # to handle possessives ("rochester's" → "rochester")
            # and quoted text ("'the" → "the")
            cleaned = _PUNCT_STRIP.sub('', p)
            if cleaned:
                expanded.append(cleaned)
        cjk_seqs = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]+', t)
        for seq in cjk_seqs:
            if seq not in expanded:
                expanded.append(seq)
    tokens = [
        t for t in expanded
        if (len(t) >= 2 or _HAS_CJK.search(t)) and t not in STOP_WORDS
    ]
    return tokens


def is_morph_prefix(shorter: str, longer: str) -> bool:
    """Check if shorter is a morphological prefix of longer."""
    if not longer.startswith(shorter):
        return False
    suffix = longer[len(shorter):]
    if not suffix:
        return True  # exact match
    # Short stems (<=3 chars): only allow long, unambiguous suffixes
    # "mad"→"madness" OK (suffix "ness"), "mr"→"mrs" blocked (suffix "s")
    if len(shorter) <= 3:
        return len(suffix) >= 3 and suffix in MORPH_SUFFIXES
    return suffix in MORPH_SUFFIXES


def node_name_words(name: str) -> List[str]:
    """Extract matchable words from a node name."""
    name_lower = name.lower()
    parts = re.split(r'[_:\-]', name_lower)
    words = []
    for p in parts:
        words.extend(
            w.lower() for w in re.findall(r'[a-z]+|[A-Z][a-z]*|\d+', p)
            if len(w) >= 2
        )
        cjk_chars = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]+', p)
        for cjk in cjk_chars:
            words.append(cjk)
            words.extend(cjk)
    return words


def token_matches_node(token: str, node_words: List[str]) -> bool:
    """Check if a token matches any word in a node name."""
    if token in node_words:
        return True
    return any(
        is_morph_prefix(token, w) or is_morph_prefix(w, token)
        for w in node_words
    )


# --- XRef types ---

@dataclass
class XRefResult:
    """One candidate cross-reference."""
    source: str
    target: str
    matched_token: str
    idf_score: float
    edge_created: bool


@dataclass
class XRefReport:
    """Summary of an XRef resolution run."""
    edges_scanned: int = 0
    descriptions_found: int = 0
    candidates_found: int = 0
    edges_created: int = 0
    edges_skipped_existing: int = 0
    edges_skipped_idf: int = 0
    results: List[XRefResult] = field(default_factory=list)


# --- XRefResolver ---

class XRefResolver:
    """Resolve implicit cross-references in edge descriptions."""

    @staticmethod
    def resolve(
        engine: "STGEngine",
        dry_run: bool = False,
        idf_max_ratio: float = 0.01,
    ) -> XRefReport:
        """Scan all real edges and create virtual edges for cross-references.

        Args:
            engine: STG engine instance
            dry_run: If True, report candidates without creating edges
            idf_max_ratio: Max fraction of nodes a token can match before
                being filtered (default 1% of total nodes)

        Returns:
            XRefReport with statistics and results
        """
        return XRefResolver._resolve_impl(
            engine, dry_run=dry_run, idf_max_ratio=idf_max_ratio,
            node_filter=None,
        )

    @staticmethod
    def resolve_node(
        engine: "STGEngine",
        node_name: str,
        dry_run: bool = False,
        idf_max_ratio: float = 0.01,
    ) -> XRefReport:
        """Scan edges of one node and create virtual XRef edges.

        Args:
            engine: STG engine instance
            node_name: Only process edges where source == node_name
            dry_run: If True, report candidates without creating edges
            idf_max_ratio: Max fraction of nodes a token can match

        Returns:
            XRefReport with statistics and results
        """
        return XRefResolver._resolve_impl(
            engine, dry_run=dry_run, idf_max_ratio=idf_max_ratio,
            node_filter=node_name,
        )

    @staticmethod
    def _resolve_impl(
        engine: "STGEngine",
        dry_run: bool,
        idf_max_ratio: float,
        node_filter: Optional[str],
    ) -> XRefReport:
        report = XRefReport()

        # Pre-compute node name words for all nodes
        all_nodes = set(engine._nodes.keys())
        node_words_cache: Dict[str, List[str]] = {}
        for name in all_nodes:
            node_words_cache[name] = node_name_words(name)

        # Pre-compute community membership for same-community filtering.
        # XRef only connects nodes within the same community (coarse level).
        # Cross-community bridges are left to LLM bind judgment.
        # Community filter only applied for graphs with meaningful community
        # structure (20+ nodes). Small graphs have unreliable gravity clustering.
        node_community: Dict[str, int] = {}
        if len(all_nodes) >= 20:
            try:
                gmap = engine.get_gravity_map()
                if gmap is not None and hasattr(gmap, "node_community"):
                    for node_name, levels in gmap.node_community.items():
                        if isinstance(levels, dict) and "coarse" in levels:
                            node_community[node_name] = levels["coarse"]
            except Exception:
                pass  # No gravity → no community filter, allow all

        # Pre-compute neighbor sets (distance <= 1) for fast lookup
        # Keys and values are lowercased for case-insensitive matching
        neighbor_cache: Dict[str, Set[str]] = {}
        for e in engine._edges:
            _sk, _tk = e.source.lower(), e.target.lower()
            neighbor_cache.setdefault(_sk, set()).add(_tk)
            neighbor_cache.setdefault(_tk, set()).add(_sk)

        # Compute document frequency: for each possible token,
        # count how many nodes it can MATCH (including morphological).
        # This is the true "how noisy is this token" metric.
        token_match_count: Dict[str, int] = {}
        all_node_words: Dict[str, List[str]] = node_words_cache
        # Build inverted index: word → set of node names
        word_to_nodes: Dict[str, Set[str]] = {}
        for name, words in all_node_words.items():
            for w in set(words):
                word_to_nodes.setdefault(w, set()).add(name)

        N = max(1, len(all_nodes))
        idf_max_df = max(3, min(int(N * idf_max_ratio), 50))

        # Track created pairs to avoid duplicates within same run
        created_pairs: Set[Tuple[str, str]] = set()

        # Scan edges
        for e in engine._edges:
            # Skip virtual edges
            if (e.modifiers or {}).get("edge_class") == "virtual":
                continue
            # Node filter
            if node_filter is not None and e.source.lower() != node_filter.lower():
                continue

            report.edges_scanned += 1

            desc = (e.modifiers or {}).get("description", "")
            if not desc:
                continue
            report.descriptions_found += 1

            # Tokenize description
            tokens = tokenize(desc)
            if not tokens:
                continue

            # Match tokens against all nodes
            source_node = e.source.lower()
            source_neighbors = neighbor_cache.get(source_node, set())
            edges_this_source = 0
            max_edges_per_source = 3  # limit XRef edges per description

            for token in tokens:
                # IDF filter: count how many nodes this token matches.
                # Uses inverted index for exact matches + morphological scan.
                if token not in token_match_count:
                    matched_nodes: Set[str] = set()
                    # Exact word match
                    if token in word_to_nodes:
                        matched_nodes.update(word_to_nodes[token])
                    # Morphological matches
                    for w, nodes in word_to_nodes.items():
                        if w != token and (
                            is_morph_prefix(token, w)
                            or is_morph_prefix(w, token)
                        ):
                            matched_nodes.update(nodes)
                    token_match_count[token] = len(matched_nodes)
                df = token_match_count[token]
                if df > idf_max_df:
                    report.edges_skipped_idf += 1
                    continue

                if edges_this_source >= max_edges_per_source:
                    break

                idf_score = math.log(N / (1 + df)) if df > 0 else 0.0

                # Find matching nodes (same community only)
                for name, words in node_words_cache.items():
                    # Skip self
                    if name == source_node:
                        continue
                    # Skip existing neighbors
                    if name in source_neighbors:
                        continue
                    # Same-community filter: only connect within same
                    # coarse community. Cross-community left to LLM bind.
                    if node_community:
                        src_comm = node_community.get(source_node, -1)
                        tgt_comm = node_community.get(name, -2)
                        if src_comm != tgt_comm:
                            continue

                    if token_matches_node(token, words):
                        report.candidates_found += 1

                        # Edge direction: referenced_node → source_node
                        # so propagate can flow FROM the referenced concept
                        # TO the node whose description mentions it.
                        # e.g. Bertha's desc mentions "mad" → Madness_Theme
                        # edge: [Madness_Theme] → [Bertha_Mason]
                        xref_src = name
                        xref_tgt = source_node
                        pair = (xref_src, xref_tgt)

                        # Skip if already created in this run
                        if pair in created_pairs:
                            continue

                        # Check if edge already exists in graph
                        ref_neighbors = neighbor_cache.get(xref_src, set())
                        if xref_tgt in ref_neighbors:
                            report.edges_skipped_existing += 1
                            continue

                        # Create virtual edge
                        created_pairs.add(pair)
                        if not dry_run:
                            import time as _time
                            engine.add_edge(
                                xref_src, xref_tgt,
                                confidence=0.10,
                                edge_class="virtual",
                                virtual_reason="xref",
                                xref_token=token,
                                virtual_created_at=_time.time(),
                            )
                            # Update neighbor cache
                            ref_neighbors.add(xref_tgt)
                            neighbor_cache.setdefault(xref_tgt, set()).add(xref_src)

                        report.edges_created += 1
                        edges_this_source += 1
                        result = XRefResult(
                            source=xref_src, target=xref_tgt,
                            matched_token=token, idf_score=idf_score,
                            edge_created=not dry_run,
                        )
                        report.results.append(result)

                        if edges_this_source >= max_edges_per_source:
                            break

        return report
