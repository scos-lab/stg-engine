"""Tests for Phase 7I — Community-Centric Propagation (M1).

M1 scope: representative-based community aggregation (no heat battery yet).
Heat/recency are placeholders=1.0 in M1; M2 adds the battery model.
"""

from __future__ import annotations

import pytest

from stg_engine.engine import STGEngine
from stg_engine.gravity import aggregate_to_communities, build_gravity_map
from stg_engine.types import CommunityPropagateResult, RepresentativeEntry


@pytest.fixture
def clustered_engine():
    """Engine with two clear communities connected by a bridge."""
    e = STGEngine()

    # Community A: tightly connected
    e.add_edge("A1", "A2", confidence=0.9)
    e.add_edge("A2", "A3", confidence=0.9)
    e.add_edge("A3", "A1", confidence=0.9)
    e.add_edge("A1", "A_Hub", confidence=0.9)
    e.add_edge("A2", "A_Hub", confidence=0.9)
    e.add_edge("A3", "A_Hub", confidence=0.9)

    # Community B: tightly connected
    e.add_edge("B1", "B2", confidence=0.9)
    e.add_edge("B2", "B3", confidence=0.9)
    e.add_edge("B3", "B1", confidence=0.9)
    e.add_edge("B1", "B_Hub", confidence=0.9)
    e.add_edge("B2", "B_Hub", confidence=0.9)
    e.add_edge("B3", "B_Hub", confidence=0.9)

    # Bridge between communities
    e.add_edge("A_Hub", "B_Hub", confidence=0.8)
    e.add_edge("B_Hub", "A_Hub", confidence=0.8)

    return e


class TestAggregateToCommunities:
    """M1: aggregation correctness."""

    def test_empty_activated_returns_empty(self, clustered_engine):
        gravity = build_gravity_map(clustered_engine)
        result = aggregate_to_communities(clustered_engine, [], gravity)
        assert result == []

    def test_returns_community_results(self, clustered_engine):
        """Test 13 prerequisite: aggregation produces CommunityPropagateResult."""
        engine = clustered_engine
        gravity = engine.get_gravity_map()
        # Manually set activations on community A members
        for name in ("A1", "A2", "A3", "A_Hub"):
            key = name.lower()
            if key in engine._nodes:
                engine._nodes[key].activation = 0.5
        activated = ["A1", "A2", "A3", "A_Hub"]
        results = aggregate_to_communities(engine, activated, gravity)
        assert isinstance(results, list)
        assert all(isinstance(r, CommunityPropagateResult) for r in results)
        assert len(results) >= 1

    def test_rep_activation_equals_top_k_mean(self, clustered_engine):
        """Test 3: community score reads top-k rep activations as mean — no full sum."""
        engine = clustered_engine
        gravity = engine.get_gravity_map()
        # Set known activations
        known_acts = {"A1": 0.9, "A2": 0.6, "A3": 0.3, "A_Hub": 0.8}
        for name, act in known_acts.items():
            key = name.lower()
            if key in engine._nodes:
                engine._nodes[key].activation = act
        activated = list(known_acts.keys())
        results = aggregate_to_communities(engine, activated, gravity, k=3)

        # Find the community containing these nodes — one of them should surface
        assert len(results) >= 1
        top = results[0]
        # Mean of the top-k rep activations should equal rep_activation
        rep_acts = [r.activation for r in top.representatives]
        assert rep_acts  # at least one rep
        expected_mean = sum(rep_acts) / len(rep_acts)
        assert abs(top.rep_activation - expected_mean) < 1e-6

    def test_zero_activation_community_not_returned(self, clustered_engine):
        """Tests 8 + 10: community whose representatives all have act=0 is skipped."""
        engine = clustered_engine
        gravity = engine.get_gravity_map()
        # Activate only non-representative nodes — community A leaf members, not hubs
        # Actually: set one community's members to 0, another to nonzero
        for name in ("A1", "A2", "A3", "A_Hub"):
            key = name.lower()
            if key in engine._nodes:
                engine._nodes[key].activation = 0.7
        # B community: zero activation
        for name in ("B1", "B2", "B3", "B_Hub"):
            key = name.lower()
            if key in engine._nodes:
                engine._nodes[key].activation = 0.0

        activated = ["A1", "A2", "A3", "A_Hub"]  # B not in activated list
        results = aggregate_to_communities(engine, activated, gravity)

        # No result should have B_Hub or B-community nodes as representatives
        for r in results:
            for rep in r.representatives:
                assert not rep.node_name.lower().startswith("b")

    def test_single_rep_community_works(self):
        """Test 9: community with only 1 representative still aggregates."""
        e = STGEngine()
        # Trivial two-node graph → likely becomes 1 community with 1-2 reps
        e.add_edge("Lone1", "Lone2", confidence=0.9)
        e._nodes["lone1"].activation = 0.8
        e._nodes["lone2"].activation = 0.6
        gravity = e.get_gravity_map()
        activated = ["Lone1", "Lone2"]
        results = aggregate_to_communities(e, activated, gravity, k=3)
        assert len(results) >= 1
        # At least one rep should be present regardless of k
        assert len(results[0].representatives) >= 1
        assert results[0].rep_activation > 0

    def test_sorted_by_score_desc_within_tier(self, clustered_engine):
        """Within each tier (matched/unmatched), score must be descending.
        Note: two-tier sort puts name_matched+active first, so global order may
        differ from pure score desc. But within each tier it stays sorted."""
        engine = clustered_engine
        gravity = engine.get_gravity_map()
        for name, act in (("A1", 0.9), ("A2", 0.8), ("A3", 0.7), ("A_Hub", 0.85),
                         ("B1", 0.3), ("B2", 0.2), ("B3", 0.1), ("B_Hub", 0.25)):
            key = name.lower()
            if key in engine._nodes:
                engine._nodes[key].activation = act
        activated = ["A1", "A2", "A3", "A_Hub", "B1", "B2", "B3", "B_Hub"]
        results = aggregate_to_communities(engine, activated, gravity, query="")
        # With empty query, nothing is name_matched → all in tier 1, pure score desc
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_name_matched_community_priority_over_structural_hub(self):
        """Two-tier sort: a name-matched community with rep_act>0 ranks above
        a higher-scoring unmatched structural hub."""
        e = STGEngine()
        # Structural hub: HUB_node with many connections + high importance
        for i in range(8):
            e.add_edge("HUB_node", f"HubMember{i}", confidence=0.9)
        # Tiny named community matching query
        e.add_edge("Website_Factory_Thing", "WF_Member", confidence=0.9)

        gravity = e.get_gravity_map()

        # Force reps/communities deterministically
        gravity.representatives["medium_99"] = ["HUB_node", "HubMember0", "HubMember1"]
        gravity.representatives["medium_100"] = ["Website_Factory_Thing", "WF_Member"]
        for n in ("hub_node", "hubmember0", "hubmember1", "hubmember2", "hubmember3",
                  "hubmember4", "hubmember5", "hubmember6", "hubmember7"):
            gravity.node_community[n] = {"medium": 99}
        for n in ("website_factory_thing", "wf_member"):
            gravity.node_community[n] = {"medium": 100}
        gravity.community_names["medium_99"] = "Structural_Hub"
        gravity.community_names["medium_100"] = "Website_Factory_Thing"
        gravity.community_counts["medium"] = 200

        # Hub gets big activation, named community gets small but nonzero
        e._nodes["hub_node"].activation = 5.0
        e._nodes["hubmember0"].activation = 2.0
        e._nodes["hubmember1"].activation = 1.5
        e._nodes["website_factory_thing"].activation = 0.3
        e._nodes["wf_member"].activation = 0.2

        activated = ["HUB_node", "HubMember0", "HubMember1",
                    "Website_Factory_Thing", "WF_Member"]
        results = aggregate_to_communities(
            e, activated, gravity, query="website factory", top_m=10,
        )
        # Website_Factory_Thing community must be first despite lower rep_activation
        assert len(results) >= 1
        first = results[0]
        assert first.community_key == "medium_100"
        assert first.name_matched is True

    def test_top_m_caps_results(self, clustered_engine):
        """top_m parameter limits output count."""
        engine = clustered_engine
        gravity = engine.get_gravity_map()
        for name in ("A1", "A2", "A3", "A_Hub", "B1", "B2", "B3", "B_Hub"):
            key = name.lower()
            if key in engine._nodes:
                engine._nodes[key].activation = 0.5
        activated = ["A1", "A2", "A3", "A_Hub", "B1", "B2", "B3", "B_Hub"]
        results = aggregate_to_communities(engine, activated, gravity, top_m=1)
        assert len(results) <= 1

    def test_name_match_boost_applied(self, clustered_engine):
        """Test 5: query matching community_name applies name_boost."""
        engine = clustered_engine
        gravity = engine.get_gravity_map()
        for name in ("A1", "A2", "A3", "A_Hub"):
            key = name.lower()
            if key in engine._nodes:
                engine._nodes[key].activation = 0.5
        activated = ["A1", "A2", "A3", "A_Hub"]

        # Without query — no boost
        no_query = aggregate_to_communities(engine, activated, gravity, query="")
        # With query matching rep name — score should be higher
        rep_name = no_query[0].community_name if no_query else ""
        if not rep_name:
            pytest.skip("No community name available for boost test")
        with_query = aggregate_to_communities(
            engine, activated, gravity, query=rep_name, name_boost=3.0,
        )
        # Match community_key across runs
        baseline = {r.community_key: r.score for r in no_query}
        for r in with_query:
            if r.name_matched and r.community_key in baseline:
                assert r.score > baseline[r.community_key]
                assert r.score == pytest.approx(baseline[r.community_key] * 3.0, rel=0.01)
                return
        pytest.skip("No community name_matched in boosted run")

    def test_signals_populated_not_placeholder(self, clustered_engine):
        """M2: heat/recency/baseline are derived from edge state, no longer 1.0 placeholders."""
        import time as _time
        engine = clustered_engine
        gravity = engine.get_gravity_map()
        # Activate some nodes + mark some edges as recently used with salience
        for name in ("A1", "A2", "A_Hub"):
            key = name.lower()
            if key in engine._nodes:
                engine._nodes[key].activation = 0.5
        now = _time.time()
        for e in engine._edges:
            if e.source.lower() in ("a1", "a2", "a_hub") and e.target.lower() in ("a1", "a2", "a_hub"):
                e.last_used = now
                e.salience = 0.8
        activated = ["A1", "A2", "A_Hub"]
        results = aggregate_to_communities(engine, activated, gravity, now=now)
        # At least one community should have non-placeholder values
        assert any(r.heat > 0 for r in results)
        # baseline_heat is now derived, can be nonzero for structural hubs
        assert all(r.baseline_heat >= 0 for r in results)


class TestComputeCommunitySignals:
    """M2: pure signal derivation from edge state."""

    def test_empty_touched_returns_empty(self, clustered_engine):
        from stg_engine.gravity import compute_community_signals
        gravity = clustered_engine.get_gravity_map()
        assert compute_community_signals(clustered_engine, gravity, []) == {}

    def test_heat_decays_with_age(self, clustered_engine):
        """Recent edges contribute more heat than old ones at same salience."""
        import time as _time
        from stg_engine.gravity import compute_community_signals
        engine = clustered_engine
        gravity = engine.get_gravity_map()
        now = _time.time()

        # Set identical salience on all A-community internal edges, but vary last_used
        for e in engine._edges:
            src_lower = e.source.lower()
            tgt_lower = e.target.lower()
            if src_lower.startswith("a") and tgt_lower.startswith("a"):
                e.salience = 1.0
                e.last_used = now  # fresh
            elif src_lower.startswith("b") and tgt_lower.startswith("b"):
                e.salience = 1.0
                e.last_used = now - 365 * 86400  # 1 year old

        # Find the community id for an A-member and a B-member
        a_comms = gravity.node_community.get("a1", {})
        b_comms = gravity.node_community.get("b1", {})
        a_cid = a_comms.get("medium")
        b_cid = b_comms.get("medium")
        if a_cid is None or b_cid is None or a_cid == b_cid:
            pytest.skip("A and B didn't resolve into distinct medium communities")

        sigs = compute_community_signals(
            engine, gravity, [a_cid, b_cid], now=now, halflife_days=30.0,
        )
        assert sigs[a_cid]["heat"] > sigs[b_cid]["heat"]
        # 12 halflives of decay ≈ 2^-12 = 1/4096 ratio
        ratio = sigs[b_cid]["heat"] / max(sigs[a_cid]["heat"], 1e-9)
        assert ratio < 0.01

    def test_heat_accumulates_from_multiple_edges(self, clustered_engine):
        """More touched edges → more heat, at same salience and age."""
        import time as _time
        from stg_engine.gravity import compute_community_signals
        engine = clustered_engine
        gravity = engine.get_gravity_map()
        now = _time.time()

        # Set all A-community internal edges fresh and salient
        touched_edges = 0
        for e in engine._edges:
            if e.source.lower().startswith("a") and e.target.lower().startswith("a"):
                e.salience = 0.5
                e.last_used = now
                touched_edges += 1

        a_cid = gravity.node_community.get("a1", {}).get("medium")
        if a_cid is None:
            pytest.skip("A nodes not in medium community")

        sigs = compute_community_signals(engine, gravity, [a_cid], now=now)
        # heat = Σ salience · e^0 = Σ salience = 0.5 · n_internal_edges
        # We don't know exact n (depends on community boundary), but should be > 0
        # and roughly proportional to touched_edges
        assert sigs[a_cid]["heat"] > 0
        # At worst half the edges are "internal" to the discovered community
        assert sigs[a_cid]["heat"] <= 0.5 * touched_edges + 1e-6

    def test_cross_community_edge_not_counted(self, clustered_engine):
        """Edge whose endpoints are in different communities is NOT internal."""
        import time as _time
        from stg_engine.gravity import compute_community_signals
        engine = clustered_engine
        gravity = engine.get_gravity_map()
        now = _time.time()

        # Set ONLY the bridge edge (A_Hub ↔ B_Hub) recently used, everything else cold
        for e in engine._edges:
            e.salience = 0.0
            e.last_used = None
        bridge_found = False
        for e in engine._edges:
            s, t = e.source.lower(), e.target.lower()
            if {s, t} == {"a_hub", "b_hub"}:
                e.salience = 1.0
                e.last_used = now
                bridge_found = True
        assert bridge_found

        a_cid = gravity.node_community.get("a_hub", {}).get("medium")
        b_cid = gravity.node_community.get("b_hub", {}).get("medium")
        if a_cid is None or b_cid is None:
            pytest.skip("Hub nodes not assigned to medium communities")

        sigs = compute_community_signals(engine, gravity, [a_cid, b_cid], now=now)
        # Neither community should register heat — the hot edge is cross-community
        if a_cid != b_cid:
            assert sigs[a_cid]["heat"] == 0.0
            assert sigs[b_cid]["heat"] == 0.0

    def test_null_last_used_ignored(self, clustered_engine):
        """Edges with last_used=None contribute zero heat."""
        import time as _time
        from stg_engine.gravity import compute_community_signals
        engine = clustered_engine
        gravity = engine.get_gravity_map()
        now = _time.time()
        for e in engine._edges:
            e.salience = 1.0
            e.last_used = None  # never activated
        a_cid = gravity.node_community.get("a1", {}).get("medium")
        if a_cid is None:
            pytest.skip()
        sigs = compute_community_signals(engine, gravity, [a_cid], now=now)
        assert sigs[a_cid]["heat"] == 0.0
        assert sigs[a_cid]["recency"] == 0.0

    def test_recency_from_max_last_used(self, clustered_engine):
        """recency reflects the most recent internal edge touch."""
        import time as _time, math
        from stg_engine.gravity import compute_community_signals
        engine = clustered_engine
        gravity = engine.get_gravity_map()
        now = _time.time()
        # One recent edge, one old edge, both A-internal
        a_internal = [e for e in engine._edges
                      if e.source.lower().startswith("a") and e.target.lower().startswith("a")]
        if len(a_internal) < 2:
            pytest.skip("Not enough internal A edges")
        for e in engine._edges:
            e.salience = 0.0
            e.last_used = None
        a_internal[0].salience = 0.5
        a_internal[0].last_used = now - 100 * 86400  # 100 days old
        a_internal[1].salience = 0.5
        a_internal[1].last_used = now - 1 * 86400    # 1 day old (the max)
        a_cid = gravity.node_community.get(a_internal[0].source.lower(), {}).get("medium")
        sigs = compute_community_signals(engine, gravity, [a_cid], now=now, halflife_days=30.0)
        # recency should correspond to 1-day-old edge, not 100-day-old
        expected = math.exp(-math.log(2) * (1 / 30))
        assert abs(sigs[a_cid]["recency"] - expected) < 0.02

    def test_baseline_derived_from_elevation(self, clustered_engine):
        """baseline = mean_elev(reps) / max_elev, purely from gravity.node_elevation."""
        from stg_engine.gravity import compute_community_signals
        engine = clustered_engine
        gravity = engine.get_gravity_map()
        # Manually override node_elevation to known values
        # Set one node very high, so max_elev is predictable
        all_nodes = list(gravity.node_elevation.keys())
        if not all_nodes:
            pytest.skip()
        # Pick a node for which we can target its community
        target_node = "a_hub" if "a_hub" in gravity.node_elevation else all_nodes[0]
        gravity.node_elevation[target_node] = 10.0
        for n in all_nodes:
            if n != target_node and gravity.node_elevation.get(n, 0) > 10.0:
                gravity.node_elevation[n] = 1.0
        # Also propagate to elevation_by_resolution medium for the function to read
        if "medium" in gravity.elevation_by_resolution:
            gravity.elevation_by_resolution["medium"] = dict(gravity.node_elevation)
        cid = gravity.node_community.get(target_node, {}).get("medium")
        if cid is None:
            pytest.skip()
        sigs = compute_community_signals(engine, gravity, [cid])
        # baseline should be in [0, 1]
        assert 0.0 <= sigs[cid]["baseline"] <= 1.0 + 1e-9

    def test_effective_heat_equals_max_heat_baseline(self, clustered_engine):
        """effective_heat = max(heat, baseline * scale)."""
        import time as _time
        from stg_engine.gravity import compute_community_signals
        engine = clustered_engine
        gravity = engine.get_gravity_map()
        now = _time.time()
        # Case 1: all edges cold (heat=0) — effective should equal baseline*scale
        for e in engine._edges:
            e.salience = 0.0
            e.last_used = None
        a_cid = gravity.node_community.get("a1", {}).get("medium")
        if a_cid is None:
            pytest.skip()
        sigs = compute_community_signals(engine, gravity, [a_cid], now=now, baseline_scale=2.0)
        assert sigs[a_cid]["heat"] == 0.0
        # effective = max(0, baseline*2)
        assert abs(sigs[a_cid]["effective_heat"] - sigs[a_cid]["baseline"] * 2.0) < 1e-9

        # Case 2: set hot edges to push heat far above baseline
        for e in engine._edges:
            if e.source.lower().startswith("a") and e.target.lower().startswith("a"):
                e.salience = 100.0  # huge
                e.last_used = now
        sigs2 = compute_community_signals(engine, gravity, [a_cid], now=now, baseline_scale=1.0)
        assert sigs2[a_cid]["heat"] > sigs2[a_cid]["baseline"]
        assert sigs2[a_cid]["effective_heat"] == sigs2[a_cid]["heat"]

    def test_normalized_heat_bounded_in_unit_interval(self, clustered_engine):
        """Sigmoid normalization keeps heat multiplier bounded, preventing it
        from drowning out rep_activation in score."""
        import time as _time
        from stg_engine.gravity import compute_community_signals
        engine = clustered_engine
        gravity = engine.get_gravity_map()
        now = _time.time()
        # Make heat unbounded-large
        for e in engine._edges:
            if e.source.lower().startswith("a") and e.target.lower().startswith("a"):
                e.salience = 1000.0
                e.last_used = now
        a_cid = gravity.node_community.get("a1", {}).get("medium")
        if a_cid is None:
            pytest.skip()
        sigs = compute_community_signals(engine, gravity, [a_cid], now=now, heat_half_saturation=5.0)
        # Raw heat is huge, normalized stays < 1
        assert sigs[a_cid]["effective_heat"] > 100
        assert 0 <= sigs[a_cid]["normalized_heat"] < 1.0
        # At half_saturation=5 and huge heat, normalized should approach 1
        assert sigs[a_cid]["normalized_heat"] > 0.95

    def test_normalized_heat_zero_when_cold(self, clustered_engine):
        """Cold communities have normalized_heat = 0 (no false warmth)."""
        import time as _time
        from stg_engine.gravity import compute_community_signals
        engine = clustered_engine
        gravity = engine.get_gravity_map()
        now = _time.time()
        for e in engine._edges:
            e.salience = 0.0
            e.last_used = None
        # Override elevation so baseline is also zero
        for n in list(gravity.node_elevation.keys()):
            gravity.node_elevation[n] = 0.0
        if "medium" in gravity.elevation_by_resolution:
            for n in list(gravity.elevation_by_resolution["medium"].keys()):
                gravity.elevation_by_resolution["medium"][n] = 0.0
        a_cid = gravity.node_community.get("a1", {}).get("medium")
        if a_cid is None:
            pytest.skip()
        sigs = compute_community_signals(engine, gravity, [a_cid], now=now)
        assert sigs[a_cid]["normalized_heat"] == 0.0


class TestQuerySeedsInCommunity:
    """P1: query-matching nodes that aren't top-k reps must still be surfaced.

    These tests directly control gravity.representatives to isolate the
    seed-detection logic from Louvain/PageRank determinism.
    """

    def test_query_seed_surfaced_when_not_a_rep(self):
        """A node matching the query substring, not in top-k reps, appears in
        query_seeds and still triggers the community to show."""
        e = STGEngine()
        # Graph: Hub connects to Member0/1/2 and to Obscure_MatchTarget_Leaf.
        e.add_edge("Hub", "Member0", confidence=0.9)
        e.add_edge("Hub", "Member1", confidence=0.9)
        e.add_edge("Hub", "Member2", confidence=0.9)
        e.add_edge("Hub", "Obscure_MatchTarget_Leaf", confidence=0.5)

        gravity = e.get_gravity_map()
        # Force reps to be Hub/Member0/Member1 (NOT the match target).
        # Pick a community key that has at least one member we control.
        # Use a synthetic key and route all our nodes there.
        comm_key = "medium_99"
        gravity.representatives[comm_key] = ["Hub", "Member0", "Member1"]
        for n in ("hub", "member0", "member1", "member2", "obscure_matchtarget_leaf"):
            gravity.node_community[n] = {"medium": 99}
        gravity.community_counts["medium"] = max(gravity.community_counts.get("medium", 0), 100)

        # Activate the leaf that matches query
        e._nodes["obscure_matchtarget_leaf"].activation = 0.8
        for name in ("hub", "member0", "member1", "member2"):
            if name in e._nodes:
                e._nodes[name].activation = 0.0

        activated = ["Obscure_MatchTarget_Leaf"]
        results = aggregate_to_communities(
            e, activated, gravity, query="MatchTarget", k=3,
        )
        # Community with no rep activation but a query match should still show
        our = [r for r in results if r.community_key == comm_key]
        assert len(our) == 1
        seed_names = [s.node_name.lower() for s in our[0].query_seeds]
        assert "obscure_matchtarget_leaf" in seed_names

    def test_query_seed_excludes_reps(self):
        """If a rep's name matches the query, it stays in reps and does NOT
        duplicate into query_seeds."""
        e = STGEngine()
        e.add_edge("MatchHub", "Member0", confidence=0.9)
        e.add_edge("MatchHub", "Member1", confidence=0.9)

        gravity = e.get_gravity_map()
        comm_key = "medium_99"
        # Force MatchHub to be a rep
        gravity.representatives[comm_key] = ["MatchHub", "Member0", "Member1"]
        for n in ("matchhub", "member0", "member1"):
            gravity.node_community[n] = {"medium": 99}
        gravity.community_counts["medium"] = max(gravity.community_counts.get("medium", 0), 100)

        e._nodes["matchhub"].activation = 0.9
        e._nodes["member0"].activation = 0.3
        e._nodes["member1"].activation = 0.3

        activated = ["MatchHub", "Member0", "Member1"]
        results = aggregate_to_communities(
            e, activated, gravity, query="MatchHub", k=3,
        )
        our = [r for r in results if r.community_key == comm_key]
        assert len(our) == 1
        seed_names = [s.node_name.lower() for s in our[0].query_seeds]
        assert "matchhub" not in seed_names

    def test_empty_query_means_no_seeds(self):
        """query='' → no seed detection runs."""
        e = STGEngine()
        e.add_edge("X", "Y", confidence=0.9)
        e._nodes["x"].activation = 0.5
        gravity = e.get_gravity_map()
        results = aggregate_to_communities(e, ["X"], gravity, query="")
        for r in results:
            assert r.query_seeds == []


class TestQueryNormalization:
    """P2 fix 2026-04-19: 'website factory' should match 'website_factory'."""

    def test_normalize_helper(self):
        from stg_engine.gravity import _normalize_for_match
        assert _normalize_for_match("website_factory") == "website factory"
        assert _normalize_for_match("Website-Factory") == "website factory"
        assert _normalize_for_match("Stg.Engine") == "stg engine"
        assert _normalize_for_match("UPPER_CASE") == "upper case"
        assert _normalize_for_match("") == ""
        assert _normalize_for_match("  multi   space  ") == "multi space"

    def test_space_query_matches_underscore_community_name(self):
        """User types 'website factory', community is named 'website_factory'."""
        e = STGEngine()
        e.add_edge("WebsiteFactory_Thing", "WF_Component", confidence=0.9)
        gravity = e.get_gravity_map()
        gravity.representatives["medium_42"] = ["WebsiteFactory_Thing"]
        gravity.node_community["websitefactory_thing"] = {"medium": 42}
        gravity.node_community["wf_component"] = {"medium": 42}
        gravity.community_names["medium_42"] = "Website_Factory"
        gravity.community_counts["medium"] = 100

        e._nodes["websitefactory_thing"].activation = 0.5
        activated = ["WebsiteFactory_Thing"]

        # Space query — should match underscore community name
        r = aggregate_to_communities(e, activated, gravity, query="website factory")
        our = [x for x in r if x.community_key == "medium_42"]
        assert len(our) == 1
        assert our[0].name_matched is True

    def test_space_query_matches_underscore_node_seed(self):
        """Space query should surface underscore-named non-rep nodes as seeds."""
        e = STGEngine()
        e.add_edge("Parent", "Website_Factory_Deploy", confidence=0.9)
        e.add_edge("Parent", "Other_Child", confidence=0.9)
        gravity = e.get_gravity_map()
        gravity.representatives["medium_42"] = ["Parent", "Other_Child"]
        gravity.node_community["parent"] = {"medium": 42}
        gravity.node_community["other_child"] = {"medium": 42}
        gravity.node_community["website_factory_deploy"] = {"medium": 42}
        gravity.community_names["medium_42"] = "Parent"
        gravity.community_counts["medium"] = 100

        e._nodes["parent"].activation = 0.5
        e._nodes["website_factory_deploy"].activation = 0.4
        activated = ["Parent", "Website_Factory_Deploy"]

        r = aggregate_to_communities(e, activated, gravity, query="website factory")
        our = [x for x in r if x.community_key == "medium_42"]
        assert len(our) == 1
        seed_names = [s.node_name.lower() for s in our[0].query_seeds]
        assert "website_factory_deploy" in seed_names
