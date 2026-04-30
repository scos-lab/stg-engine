"""Unit tests for stg_engine.recall — Phase 1 precision recall postprocessing.

Covers:
- _edge_weight: salience × recency × supersede_decay math
- apply_recency_weight: subgraph max-edge-weight aggregation, recall completeness
- community_dominance_filter: ratio threshold, edge cases

Reference: development/design/STG_PRECISION_RECALL_DESIGN.md
"""

from __future__ import annotations

import math
import time

import pytest

from stg_engine.gravity import GravityMap
from stg_engine.recall import (
    DEFAULT_ACTIVE_CONTEXT_BOOST,
    DEFAULT_DOMINANCE_RATIO,
    DEFAULT_MAX_EDGE_HITS,
    DEFAULT_MIN_CHAIN_LENGTH,
    DEFAULT_MIN_EDGE_TOKEN_LENGTH,
    DEFAULT_RECENCY_HALFLIFE_DAYS,
    DEFAULT_SUPERSEDE_DECAY_FACTOR,
    EDGE_SCAN_FIELDS,
    _edge_weight,
    _split_tokens,
    apply_recency_weight,
    community_dominance_filter,
    context_anchor_boost,
    find_edges_between,
    match_exact_anchors,
    scan_edges_by_content,
)
from stg_engine.types import CommunityPropagateResult, STGEdge, STGNode


# ─── _edge_weight ─────────────────────────────────────────────────────


def test_edge_weight_fresh_no_supersede():
    """Fresh edge (created_at = now) → weight = salience."""
    now = time.time()
    e = STGEdge(source="A", target="B", salience=0.5, created_at=now)
    w = _edge_weight(e, now, halflife_days=30, supersede_factor=0.3)
    assert abs(w - 0.5) < 0.01


def test_edge_weight_one_halflife():
    """Edge aged exactly one halflife → weight halved."""
    now = time.time()
    e = STGEdge(source="A", target="B", salience=0.5, created_at=now - 30 * 86400)
    w = _edge_weight(e, now, halflife_days=30, supersede_factor=0.3)
    assert abs(w - 0.25) < 0.01


def test_edge_weight_two_halflives():
    """Edge aged two halflives → weight quartered."""
    now = time.time()
    e = STGEdge(source="A", target="B", salience=0.5, created_at=now - 60 * 86400)
    w = _edge_weight(e, now, halflife_days=30, supersede_factor=0.3)
    assert abs(w - 0.125) < 0.01


def test_edge_weight_superseded_fresh():
    """Superseded fresh edge: salience × 1 × supersede_factor."""
    now = time.time()
    e = STGEdge(
        source="A", target="B", salience=0.5, created_at=now,
        modifiers={"superseded_at": now - 3600},
    )
    w = _edge_weight(e, now, halflife_days=30, supersede_factor=0.3)
    assert abs(w - 0.15) < 0.01


def test_edge_weight_legacy_zero_created_at():
    """Legacy edge (created_at = 0.0) → no recency decay applied."""
    now = time.time()
    e = STGEdge(source="A", target="B", salience=0.5, created_at=0.0)
    w = _edge_weight(e, now, halflife_days=30, supersede_factor=0.3)
    assert abs(w - 0.5) < 0.01


def test_edge_weight_negative_age_clamped():
    """Edge from the future (created_at > now) → age clamped to 0, no boost."""
    now = time.time()
    e = STGEdge(source="A", target="B", salience=0.5, created_at=now + 86400)
    w = _edge_weight(e, now, halflife_days=30, supersede_factor=0.3)
    assert abs(w - 0.5) < 0.01


# ─── community_dominance_filter ──────────────────────────────────────


def _mk_comm(name: str, score: float) -> CommunityPropagateResult:
    return CommunityPropagateResult(
        community_key=f"medium_{name}",
        community_name=name,
        score=score,
        rep_activation=score / 2.0,
    )


def test_dominance_filter_default_ratio():
    """Default ratio=3.0: keep communities >= dominant/3."""
    c1 = _mk_comm("dominant", 10.0)
    c2 = _mk_comm("medium", 4.0)   # 4.0 >= 10/3=3.33 → kept
    c3 = _mk_comm("weak", 1.0)     # 1.0 < 3.33 → folded
    kept = community_dominance_filter([c1, c2, c3], ratio=3.0)
    assert [c.community_name for c in kept] == ["dominant", "medium"]


def test_dominance_filter_tight_ratio():
    """Ratio=2.0: only dominant (medium 4.0 < 10/2=5.0)."""
    c1 = _mk_comm("dominant", 10.0)
    c2 = _mk_comm("medium", 4.0)
    c3 = _mk_comm("weak", 1.0)
    kept = community_dominance_filter([c1, c2, c3], ratio=2.0)
    assert [c.community_name for c in kept] == ["dominant"]


def test_dominance_filter_loose_ratio():
    """Ratio=10.0: all communities pass."""
    c1 = _mk_comm("dominant", 10.0)
    c2 = _mk_comm("medium", 4.0)
    c3 = _mk_comm("weak", 1.0)
    kept = community_dominance_filter([c1, c2, c3], ratio=10.0)
    assert len(kept) == 3


def test_dominance_filter_empty_input():
    """Empty input → empty output."""
    assert community_dominance_filter([]) == []


def test_dominance_filter_single_community():
    """Single community is always kept."""
    c1 = _mk_comm("only", 5.0)
    kept = community_dominance_filter([c1], ratio=3.0)
    assert len(kept) == 1


def test_dominance_filter_degenerate_zero_score():
    """Degenerate dominant score=0.0 → pass through unchanged."""
    c0 = _mk_comm("zero", 0.0)
    c1 = _mk_comm("a", 0.0)
    c2 = _mk_comm("b", 0.0)
    kept = community_dominance_filter([c0, c1, c2], ratio=3.0)
    assert len(kept) == 3


def test_dominance_filter_folds_below_threshold():
    """Communities below threshold without query_seeds are folded;
    those above are kept. Scan continues across all input (no early
    break) so query_seeds protection works regardless of sort position."""
    c1 = _mk_comm("top", 10.0)
    c2 = _mk_comm("a", 3.5)   # >= 3.33 → kept
    c3 = _mk_comm("b", 3.4)   # >= 3.33 → kept
    c4 = _mk_comm("c", 3.0)   # < 3.33, no query_seeds → folded
    kept = community_dominance_filter([c1, c2, c3, c4], ratio=3.0)
    assert [c.community_name for c in kept] == ["top", "a", "b"]


def test_dominance_filter_keeps_low_score_with_query_seeds():
    """Community with query_seeds is kept even if score below threshold.

    Protects against R7 collapsing a community whose top representatives
    have zero activation but whose internal query-matching nodes are real
    precise hits (e.g. User_Bike_Repair in a low-rep community).
    """
    from stg_engine.types import RepresentativeEntry
    seed = RepresentativeEntry(node_name="User_Bike_Repair", activation=0.21, elevation=5.57)
    c1 = _mk_comm("top", 10.0)
    c2 = CommunityPropagateResult(
        community_key="medium_irvine",
        community_name="irvine_ca",
        score=0.0,  # below threshold
        rep_activation=0.0,
        query_seeds=[seed],  # but has precise query hit
    )
    kept = community_dominance_filter([c1, c2], ratio=3.0)
    names = [c.community_name for c in kept]
    assert "irvine_ca" in names


# ─── apply_recency_weight (memory-never-vanishes principle) ─────────


def test_apply_recency_weight_empty_activated():
    """Empty activated list → empty output."""
    # No engine needed since early-return on empty
    result = apply_recency_weight(engine=None, activated=[])  # type: ignore
    assert result == []


# ─── Default parameter sanity ───────────────────────────────────────


def test_default_halflife_30_days():
    assert DEFAULT_RECENCY_HALFLIFE_DAYS == 30.0


def test_default_supersede_factor_0_3():
    assert DEFAULT_SUPERSEDE_DECAY_FACTOR == 0.3


def test_default_dominance_ratio_3():
    assert DEFAULT_DOMINANCE_RATIO == 3.0


def test_default_active_context_boost_5():
    assert DEFAULT_ACTIVE_CONTEXT_BOOST == 5.0


# ─── context_anchor_boost (R5) ──────────────────────────────────────


def _mk_gravity_map() -> GravityMap:
    """Build a minimal GravityMap for testing."""
    return GravityMap(
        node_elevation={"user_dog_max": 2.86, "food_recovery": 1.20},
        elevation_by_resolution={
            "medium": {"user_dog_max": 2.86, "food_recovery": 1.20},
            "coarse": {"user_dog_max": 1.50},
        },
    )


def test_context_anchor_boost_lifts_elevation():
    """Inside the with-block, anchor node elevation is boost higher."""
    gm = _mk_gravity_map()
    with context_anchor_boost(gm, ["User_Dog_Max"], boost=5.0):
        assert gm.node_elevation["user_dog_max"] == pytest.approx(7.86)
        assert gm.elevation_by_resolution["medium"]["user_dog_max"] == pytest.approx(7.86)
        assert gm.elevation_by_resolution["coarse"]["user_dog_max"] == pytest.approx(6.50)


def test_context_anchor_boost_restores_on_exit():
    """After with-block, all elevations are restored to original values."""
    gm = _mk_gravity_map()
    original_node = gm.node_elevation["user_dog_max"]
    original_medium = gm.elevation_by_resolution["medium"]["user_dog_max"]
    original_coarse = gm.elevation_by_resolution["coarse"]["user_dog_max"]
    with context_anchor_boost(gm, ["User_Dog_Max"], boost=5.0):
        pass
    assert gm.node_elevation["user_dog_max"] == original_node
    assert gm.elevation_by_resolution["medium"]["user_dog_max"] == original_medium
    assert gm.elevation_by_resolution["coarse"]["user_dog_max"] == original_coarse


def test_context_anchor_boost_restores_on_exception():
    """Restoration happens even if an exception is raised inside the block."""
    gm = _mk_gravity_map()
    original = gm.node_elevation["user_dog_max"]
    with pytest.raises(RuntimeError):
        with context_anchor_boost(gm, ["User_Dog_Max"], boost=5.0):
            raise RuntimeError("boom")
    assert gm.node_elevation["user_dog_max"] == original


def test_context_anchor_boost_unknown_node_skipped():
    """Anchor name not in GravityMap is silently skipped (no error)."""
    gm = _mk_gravity_map()
    with context_anchor_boost(gm, ["Nonexistent_Node"], boost=5.0):
        # No exception, no mutation
        assert "nonexistent_node" not in gm.node_elevation


def test_context_anchor_boost_does_not_affect_non_anchor():
    """Non-anchor nodes' elevations are unchanged inside the block."""
    gm = _mk_gravity_map()
    with context_anchor_boost(gm, ["User_Dog_Max"], boost=5.0):
        assert gm.node_elevation["food_recovery"] == 1.20  # unchanged


def test_context_anchor_boost_empty_anchor_list():
    """Empty anchor list → no-op."""
    gm = _mk_gravity_map()
    snapshot = dict(gm.node_elevation)
    with context_anchor_boost(gm, [], boost=5.0):
        pass
    assert gm.node_elevation == snapshot


def test_context_anchor_boost_zero_boost_no_op():
    """boost=0.0 → no-op (no backup, no mutation)."""
    gm = _mk_gravity_map()
    snapshot = dict(gm.node_elevation)
    with context_anchor_boost(gm, ["User_Dog_Max"], boost=0.0):
        assert gm.node_elevation == snapshot
    assert gm.node_elevation == snapshot


def test_context_anchor_boost_case_insensitive():
    """Anchor name matching is case-insensitive (lower-case keys)."""
    gm = _mk_gravity_map()
    with context_anchor_boost(gm, ["USER_DOG_MAX"], boost=5.0):
        assert gm.node_elevation["user_dog_max"] == pytest.approx(7.86)
    assert gm.node_elevation["user_dog_max"] == pytest.approx(2.86)


def test_context_anchor_boost_multiple_anchors():
    """Multiple anchor nodes all get boosted."""
    gm = _mk_gravity_map()
    with context_anchor_boost(gm, ["User_Dog_Max", "Food_Recovery"], boost=5.0):
        assert gm.node_elevation["user_dog_max"] == pytest.approx(7.86)
        assert gm.node_elevation["food_recovery"] == pytest.approx(6.20)
    # Both restored
    assert gm.node_elevation["user_dog_max"] == pytest.approx(2.86)
    assert gm.node_elevation["food_recovery"] == pytest.approx(1.20)


# ─── _split_tokens (R2 dispatch) ────────────────────────────────────


def test_split_tokens_two_words():
    """Two-word query → two tokens."""
    assert _split_tokens("dog food") == ["dog", "food"]


def test_split_tokens_with_stop_words():
    """Stop words filtered out."""
    assert _split_tokens("the dog and food") == ["dog", "food"]


def test_split_tokens_short_word_dropped():
    """Single-char ASCII tokens dropped (length < 2)."""
    assert _split_tokens("a b c dog") == ["dog"]


def test_split_tokens_underscore_split():
    """Underscore-joined identifier splits into parts."""
    assert _split_tokens("STG_Engine") == ["stg", "engine"]


def test_split_tokens_hyphen_split():
    """Hyphen-joined identifier splits into parts."""
    assert _split_tokens("dog-food") == ["dog", "food"]


def test_split_tokens_cjk_kept():
    """Single-char CJK tokens are kept (semantically meaningful)."""
    tokens = _split_tokens("食物")
    assert "食物" in tokens


def test_split_tokens_punctuation_stripped():
    """Trailing punctuation stripped."""
    assert _split_tokens("dog! food.") == ["dog", "food"]


def test_split_tokens_empty():
    """Empty string → empty list."""
    assert _split_tokens("") == []


def test_split_tokens_only_stop_words():
    """All-stop-word query → empty list (single-seed fallback)."""
    assert _split_tokens("the and of") == []


def test_default_min_chain_length():
    assert DEFAULT_MIN_CHAIN_LENGTH == 2


# ─── R6: scan_edges_by_content ──────────────────────────────────────


class _FakeEngine:
    """Minimal stand-in for STGEngine in scan_edges_by_content tests."""
    def __init__(self, edges):
        self._edges = edges


def _mk_edge(source, target, **mods):
    return STGEdge(source=source, target=target, modifiers=dict(mods))


def test_scan_edges_finds_token_in_description():
    e = _mk_edge("User", "Event_X", description="User volunteered at the gala")
    hits = scan_edges_by_content(_FakeEngine([e]), ["volunteered"])
    assert len(hits) == 1
    edge, matched, score = hits[0]
    assert edge.target == "Event_X"
    assert matched == ["volunteered"]
    assert score >= 0.0


def test_scan_edges_finds_token_in_action():
    e = _mk_edge("User", "Event_Y", action="participated_in")
    hits = scan_edges_by_content(_FakeEngine([e]), ["participated"])
    assert len(hits) == 1


def test_scan_edges_substring_match_no_morphological_prefix():
    """Substring match means 'volunteer' hits 'volunteered_at'."""
    e = _mk_edge("User", "Event", action="volunteered_at")
    hits = scan_edges_by_content(_FakeEngine([e]), ["volunteer"])
    assert len(hits) == 1


def test_scan_edges_short_token_dropped():
    """Tokens shorter than DEFAULT_MIN_EDGE_TOKEN_LENGTH are skipped."""
    e = _mk_edge("User", "Event", description="User did A B C")
    # default min length is 3; "ab" should be dropped
    hits = scan_edges_by_content(_FakeEngine([e]), ["ab"])
    assert hits == []


def test_scan_edges_idf_ranks_rare_token_higher():
    """Token appearing in fewer edges scores higher than common token."""
    edges = [
        _mk_edge("U", "X1", description="user volunteered rare-keyword-zzz"),
        _mk_edge("U", "X2", description="user attended"),
        _mk_edge("U", "X3", description="user attended"),
        _mk_edge("U", "X4", description="user attended"),
    ]
    hits = scan_edges_by_content(_FakeEngine(edges), ["rare-keyword-zzz", "user"])
    # First hit should be X1 since "rare-keyword-zzz" is in only 1 edge,
    # giving it a much higher IDF than "user"
    assert hits[0][0].target == "X1"


def test_scan_edges_cap_to_max_hits():
    edges = [_mk_edge("U", f"X{i}", description="match here") for i in range(100)]
    hits = scan_edges_by_content(_FakeEngine(edges), ["match"], max_hits=10)
    assert len(hits) == 10


def test_scan_edges_empty_input():
    hits = scan_edges_by_content(_FakeEngine([]), ["foo"])
    assert hits == []


def test_scan_edges_no_match():
    e = _mk_edge("U", "X", description="something unrelated")
    hits = scan_edges_by_content(_FakeEngine([e]), ["nonexistent"])
    assert hits == []


def test_scan_edges_field_coverage():
    """All 6 EDGE_SCAN_FIELDS are scanned."""
    for field in EDGE_SCAN_FIELDS:
        kwargs = {field: "trigger_token"}
        e = _mk_edge("U", "X", **kwargs)
        hits = scan_edges_by_content(_FakeEngine([e]), ["trigger_token"])
        assert len(hits) == 1, f"Field {field} did not get scanned"


def test_scan_edges_skips_unscanned_fields():
    """Fields outside EDGE_SCAN_FIELDS are NOT scanned (e.g. confidence)."""
    e = _mk_edge("U", "X", confidence=0.99)
    # 'confidence' value 0.99 is numeric — token "0.99" wouldn't match anyway.
    # Use a string field outside the scan set:
    e2 = _mk_edge("U", "Y", source_url="trigger_token_url")
    hits = scan_edges_by_content(_FakeEngine([e2]), ["trigger_token"])
    assert hits == []


def test_default_max_edge_hits_50():
    assert DEFAULT_MAX_EDGE_HITS == 50


def test_default_min_edge_token_length_3():
    assert DEFAULT_MIN_EDGE_TOKEN_LENGTH == 3


def test_edge_scan_fields_count():
    assert EDGE_SCAN_FIELDS == ("description", "lesson", "action", "role", "status", "is_a")


# ─── A: match_exact_anchors ─────────────────────────────────────────


class _AnchorEngine:
    """Stand-in for STGEngine carrying just _nodes and _edges."""
    def __init__(self, nodes, edges=None):
        self._nodes = {n.name.lower(): n for n in nodes}
        self._edges = edges or []


def test_exact_anchor_matches_full_node_name():
    engine = _AnchorEngine([
        STGNode(name="User"),
        STGNode(name="Food_For_Thought_Charity_Gala"),
    ])
    anchors, remaining = match_exact_anchors(
        engine, "User food_for_thought_charity_gala"
    )
    assert anchors == ["User", "Food_For_Thought_Charity_Gala"]
    assert remaining == []


def test_exact_anchor_case_insensitive():
    engine = _AnchorEngine([STGNode(name="MyNode")])
    anchors, remaining = match_exact_anchors(engine, "MYNODE mynode")
    assert anchors == ["MyNode", "MyNode"]


def test_exact_anchor_unmatched_falls_through():
    engine = _AnchorEngine([STGNode(name="User")])
    anchors, remaining = match_exact_anchors(engine, "User volunteered charity")
    assert anchors == ["User"]
    assert remaining == ["volunteered", "charity"]


def test_exact_anchor_empty_query():
    engine = _AnchorEngine([STGNode(name="User")])
    anchors, remaining = match_exact_anchors(engine, "")
    assert anchors == []
    assert remaining == []


def test_exact_anchor_preserves_display_case():
    """Even if user types lowercase, return display case from node."""
    engine = _AnchorEngine([STGNode(name="STG_Engine")])
    anchors, _ = match_exact_anchors(engine, "stg_engine")
    assert anchors == ["STG_Engine"]


def test_exact_anchor_ngram_greedy_match_space_separated():
    """Multi-word node name typed with spaces → matched via N-gram."""
    engine = _AnchorEngine([
        STGNode(name="User"),
        STGNode(name="Food_For_Thought_Charity_Gala"),
    ])
    anchors, remaining = match_exact_anchors(
        engine, "user Food for thought charity gala"
    )
    assert anchors == ["User", "Food_For_Thought_Charity_Gala"]
    assert remaining == []


def test_exact_anchor_ngram_picks_longest():
    """When both 'User_Bike' and 'User' exist, N-gram greedy picks longer."""
    engine = _AnchorEngine([
        STGNode(name="User"),
        STGNode(name="User_Bike"),
    ])
    anchors, _ = match_exact_anchors(engine, "User Bike")
    # Greedy 2-gram first → User_Bike (consumes both tokens)
    assert anchors == ["User_Bike"]


def test_exact_anchor_ngram_consumes_no_overlap():
    """Tokens consumed by an N-gram match are not used by other matches."""
    engine = _AnchorEngine([
        STGNode(name="A_B_C"),
        STGNode(name="C_D"),
    ])
    # Greedy from left: A_B_C consumes [A, B, C]; D remains → no match
    anchors, remaining = match_exact_anchors(engine, "A B C D")
    assert anchors == ["A_B_C"]
    assert remaining == ["D"]


def test_exact_anchor_ngram_falls_back_to_unigram():
    """Unmatched longer N-grams still allow unigram match for the same start."""
    engine = _AnchorEngine([STGNode(name="X")])
    anchors, _ = match_exact_anchors(engine, "X Y Z")
    # 3-gram X_Y_Z fails, 2-gram X_Y fails, 1-gram X matches
    assert anchors == ["X"]


# ─── C: find_edges_between ─────────────────────────────────────────


def test_find_edges_between_directed_match():
    edge = STGEdge(source="User", target="Charity_Event", modifiers={"action": "attended"})
    engine = _AnchorEngine(
        nodes=[STGNode(name="User"), STGNode(name="Charity_Event")],
        edges=[edge],
    )
    matched = find_edges_between(engine, ["User", "Charity_Event"])
    assert len(matched) == 1
    assert matched[0] is edge


def test_find_edges_between_bidirectional():
    """Order in anchor list should not matter — search is symmetric."""
    edge = STGEdge(source="User", target="Charity_Event")
    engine = _AnchorEngine(
        nodes=[STGNode(name="User"), STGNode(name="Charity_Event")],
        edges=[edge],
    )
    matched_a = find_edges_between(engine, ["User", "Charity_Event"])
    matched_b = find_edges_between(engine, ["Charity_Event", "User"])
    assert matched_a == matched_b


def test_find_edges_between_skips_self_loop():
    edge = STGEdge(source="A", target="A")
    engine = _AnchorEngine(nodes=[STGNode(name="A")], edges=[edge])
    matched = find_edges_between(engine, ["A", "A"])
    assert matched == []


def test_find_edges_between_excludes_non_anchor_endpoints():
    e_inside = STGEdge(source="User", target="Charity_Event")
    e_outside = STGEdge(source="User", target="Other_Node")
    engine = _AnchorEngine(
        nodes=[STGNode(name="User"), STGNode(name="Charity_Event"), STGNode(name="Other_Node")],
        edges=[e_inside, e_outside],
    )
    matched = find_edges_between(engine, ["User", "Charity_Event"])
    assert len(matched) == 1
    assert matched[0] is e_inside


def test_find_edges_between_requires_two_anchors():
    edge = STGEdge(source="A", target="B")
    engine = _AnchorEngine(nodes=[STGNode(name="A"), STGNode(name="B")], edges=[edge])
    assert find_edges_between(engine, []) == []
    assert find_edges_between(engine, ["A"]) == []


def test_find_edges_between_three_anchors_returns_all_internal():
    e_ab = STGEdge(source="A", target="B")
    e_bc = STGEdge(source="B", target="C")
    e_ac = STGEdge(source="A", target="C")
    e_ax = STGEdge(source="A", target="X")
    engine = _AnchorEngine(
        nodes=[STGNode(name="A"), STGNode(name="B"), STGNode(name="C"), STGNode(name="X")],
        edges=[e_ab, e_bc, e_ac, e_ax],
    )
    matched = find_edges_between(engine, ["A", "B", "C"])
    assert len(matched) == 3
    assert e_ax not in matched
