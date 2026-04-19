"""STG Edge Merge — memory consolidation for multi-edges.

Phase 7C.2: Merge multiple edges on the same (source, target) pair
into one consolidated edge. This is a memory tidying operation,
not part of the ingest flow.

Two entry points:
  merge_edge()           — patch a single existing edge
  consolidate_edges()    — merge all multi-edges on a pair into one
  find_mergeable_pairs() — scan graph for consolidation candidates
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from stg_engine.engine import STGEngine
    from stg_engine.types import STGEdge


# Field classification for merge behavior
OVERWRITE_FIELDS = {"confidence", "strength", "rule", "description", "lesson",
                    "domain", "author", "certainty"}
ACCUMULATE_FIELDS = {"source", "path"}
IMMUTABLE_FIELDS = {"edge_class", "delay_k"}

# Top-level STGEdge attributes (not in modifiers dict)
EDGE_ATTRS = {"confidence", "strength", "rule"}


@dataclass
class MergeResult:
    """Result of a consolidation operation."""
    source: str
    target: str
    edges_merged: int
    merged_edge: "STGEdge"
    skipped_timestamps: int = 0


class EdgeMerger:
    """Stateless utility for edge merge operations."""

    @staticmethod
    def merge_edge(
        engine: "STGEngine",
        source: str,
        target: str,
        confidence: Optional[float] = None,
        strength: Optional[float] = None,
        rule: Optional[str] = None,
        **modifiers: Any,
    ) -> "STGEdge":
        """Patch an existing edge with new modifier values.

        Overwrite fields: new value replaces old (if provided).
        Accumulate fields (source, path): old + new, semicolon-separated.
        Immutable fields (edge_class, delay_k): ignored.

        Args:
            engine: STGEngine instance
            source: Source anchor name
            target: Target anchor name
            confidence: New confidence (overwrites)
            strength: New strength (overwrites)
            rule: New rule type (overwrites)
            **modifiers: Additional modifier patches

        Returns:
            The patched edge

        Raises:
            KeyError: if no edge exists for (source, target)
        """
        edge = engine._edges_lookup.get((source.lower(), target.lower()))
        if edge is None:
            raise KeyError(f"No edge [{source}] -> [{target}] to merge into")

        # Overwrite top-level attrs if provided
        if confidence is not None:
            edge.confidence = confidence
        if strength is not None:
            edge.strength = strength
        if rule is not None:
            edge.rule = rule

        # Process modifiers
        for key, value in modifiers.items():
            if value is None:
                continue
            if key in IMMUTABLE_FIELDS:
                continue
            if key in ACCUMULATE_FIELDS:
                _accumulate(edge.modifiers, key, str(value))
            elif key in OVERWRITE_FIELDS - EDGE_ATTRS:
                edge.modifiers[key] = value
            else:
                # Unknown fields: overwrite
                edge.modifiers[key] = value

        engine._invalidate_caches()
        return edge

    @staticmethod
    def consolidate_edges(
        engine: "STGEngine",
        source: str,
        target: str,
    ) -> Optional[MergeResult]:
        """Consolidate all multi-edges on (source, target) into one.

        Merges edges in chronological order (oldest first).
        The resulting edge gets created_at = now.
        All original edges are removed.

        Returns:
            MergeResult if edges were consolidated, None if <=1 edge exists

        Raises:
            ValueError: if edges have conflicting timestamps
        """
        all_edges = [e for e in engine._edges
                     if e.source.lower() == source.lower()
                     and e.target.lower() == target.lower()
                     and e.modifiers.get("edge_class") != "virtual"]
        if len(all_edges) <= 1:
            return None

        # Timestamp compatibility check
        timestamps = set()
        for e in all_edges:
            ts = e.modifiers.get("timestamp")
            if ts is not None:
                timestamps.add(ts)
        if len(timestamps) > 1:
            raise ValueError(
                f"Conflicting timestamps on [{source}] -> [{target}]: "
                f"{timestamps}. Cannot merge edges with different event times."
            )

        # Sort by created_at ascending (oldest first)
        all_edges.sort(key=lambda e: e.created_at)

        # Build consolidated fields by applying edges in order
        base = all_edges[0]
        merged_conf = base.confidence
        merged_strength = base.strength
        merged_rule = base.rule
        merged_salience = base.salience
        merged_mods: Dict[str, Any] = dict(base.modifiers)

        for edge in all_edges[1:]:
            # Overwrite fields: later value wins if non-default
            if edge.confidence != 0.5:  # non-default
                merged_conf = edge.confidence
            if edge.strength != 0.5:
                merged_strength = edge.strength
            if edge.rule is not None:
                merged_rule = edge.rule
            merged_salience = max(merged_salience, edge.salience)

            for key, value in edge.modifiers.items():
                if value is None:
                    continue
                if key in IMMUTABLE_FIELDS:
                    continue
                if key.startswith("_"):
                    # Internal fields (superseded_at, conflict_report, etc.)
                    # Don't carry over
                    continue
                if key in ACCUMULATE_FIELDS:
                    _accumulate(merged_mods, key, str(value))
                else:
                    # Overwrite
                    merged_mods[key] = value

        # Remove all original edges
        for edge in all_edges:
            if edge in engine._edges:
                engine._edges.remove(edge)
        _sk, _tk = source.lower(), target.lower()
        if (_sk, _tk) in engine._edges_lookup:
            del engine._edges_lookup[(_sk, _tk)]

        # Remove internal fields from merged mods
        for internal_key in list(merged_mods.keys()):
            if internal_key.startswith("_"):
                del merged_mods[internal_key]

        # Create new consolidated edge via add_edge
        # (gets fresh created_at, goes through normal validation)
        new_edge = engine.add_edge(
            source=source,
            target=target,
            confidence=merged_conf,
            strength=merged_strength,
            rule=merged_rule,
            **merged_mods,
        )
        new_edge.salience = merged_salience

        return MergeResult(
            source=source,
            target=target,
            edges_merged=len(all_edges),
            merged_edge=new_edge,
        )

    @staticmethod
    def find_mergeable_pairs(
        engine: "STGEngine",
    ) -> List[Tuple[str, str, int]]:
        """Find all (source, target) pairs with mergeable multi-edges.

        Excludes pairs with conflicting timestamps.

        Returns:
            List of (source, target, edge_count) sorted by count descending
        """
        # Count edges per (source, target)
        pair_counts: Dict[Tuple[str, str], List["STGEdge"]] = {}
        for edge in engine._edges:
            if edge.modifiers.get("edge_class") == "virtual":
                continue
            key = (edge.source.lower(), edge.target.lower())
            if key not in pair_counts:
                pair_counts[key] = []
            pair_counts[key].append(edge)

        # Filter to multi-edge pairs with compatible timestamps
        candidates = []
        for (src, tgt), edges in pair_counts.items():
            if len(edges) <= 1:
                continue
            timestamps = {e.modifiers.get("timestamp")
                         for e in edges} - {None}
            if len(timestamps) > 1:
                continue  # timestamp conflict, skip
            candidates.append((src, tgt, len(edges)))

        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates


def _accumulate(mods: Dict[str, Any], key: str, new_value: str) -> None:
    """Accumulate a value into a modifier field with semicolon separation."""
    old = mods.get(key, "")
    if not old:
        mods[key] = new_value
    elif new_value not in old:
        mods[key] = f"{old}; {new_value}"
    # else: already contains this value, skip
