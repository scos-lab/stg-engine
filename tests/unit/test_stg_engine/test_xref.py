"""Tests for Phase 7C.3: XRef — Cross-Reference Resolution."""

import pytest

from stg_engine.engine import STGEngine
from stg_engine.xref import (
    XRefReport,
    XRefResolver,
    XRefResult,
    is_morph_prefix,
    node_name_words,
    token_matches_node,
    tokenize,
)


@pytest.fixture
def engine():
    """Fresh STGEngine for each test."""
    return STGEngine()


@pytest.fixture
def book_engine():
    """Engine with two 'book' communities (20+ nodes) for cross-community tests.

    Needs >= 20 nodes for community filter to activate.
    """
    e = STGEngine()
    # Community 1: Hamlet (~10 nodes)
    e.ingest_stl('[Hamlet_Prince] -> [Hamlet] ::mod(confidence=0.99, rule="definitional", description="Prince of Denmark. Feigns madness.")')
    e.ingest_stl('[Ophelia] -> [Hamlet] ::mod(confidence=0.97, rule="definitional", description="Driven to madness by cruelty.")')
    e.ingest_stl('[Madness_Theme] -> [Hamlet] ::mod(confidence=0.93, rule="definitional", description="Real madness vs performed madness.")')
    e.ingest_stl('[Claudius] -> [Hamlet] ::mod(confidence=0.95, rule="definitional", description="Hamlets uncle. Murdered King Hamlet.")')
    e.ingest_stl('[Gertrude] -> [Hamlet] ::mod(confidence=0.93, rule="definitional", description="Hamlets mother. Married Claudius.")')
    e.ingest_stl('[Polonius] -> [Hamlet] ::mod(confidence=0.90, rule="definitional", description="Lord Chamberlain. Killed by Hamlet.")')
    e.ingest_stl('[Horatio] -> [Hamlet] ::mod(confidence=0.90, rule="definitional", description="Hamlets loyal friend.")')
    e.ingest_stl('[Laertes] -> [Hamlet] ::mod(confidence=0.90, rule="definitional", description="Ophelias brother.")')
    e.ingest_stl('[Ghost_Appearance] -> [Hamlet_Prince] ::mod(confidence=0.95, rule="causal", description="Ghost reveals murder.")')
    e.ingest_stl('[Final_Duel] -> [Hamlet_Prince] ::mod(confidence=0.98, rule="causal", description="Poisoned sword duel.")')
    # Community 2: Jane Eyre (~10 nodes)
    e.ingest_stl('[Jane_Eyre_Character] -> [Jane_Eyre] ::mod(confidence=0.99, rule="definitional", description="Orphan, plain, independent.")')
    e.ingest_stl('[Bertha_Mason] -> [Jane_Eyre] ::mod(confidence=0.97, rule="definitional", description="Rochester first wife. Locked in attic. Violent, mad. The madwoman.")')
    e.ingest_stl('[Edward_Rochester] -> [Jane_Eyre] ::mod(confidence=0.98, rule="definitional", description="Master of Thornfield. Hides mad wife.")')
    e.ingest_stl('[St_John_Rivers] -> [Jane_Eyre] ::mod(confidence=0.90, rule="definitional", description="Clergyman. Janes cousin.")')
    e.ingest_stl('[Mrs_Reed] -> [Jane_Eyre] ::mod(confidence=0.88, rule="definitional", description="Janes cruel aunt.")')
    e.ingest_stl('[Helen_Burns] -> [Jane_Eyre] ::mod(confidence=0.88, rule="definitional", description="Janes friend at Lowood.")')
    e.ingest_stl('[Red_Room] -> [Jane_Eyre_Character] ::mod(confidence=0.93, rule="causal", description="Jane locked in red room.")')
    e.ingest_stl('[Wedding_Interrupted] -> [Edward_Rochester] ::mod(confidence=0.98, rule="causal", description="Bertha revealed at altar.")')
    e.ingest_stl('[Thornfield_Fire] -> [Bertha_Mason] ::mod(confidence=0.97, rule="causal", description="Bertha sets fire to Thornfield.")')
    e.ingest_stl('[Independence_Theme] -> [Jane_Eyre] ::mod(confidence=0.95, rule="definitional", description="Jane insists on equality.")')
    return e


# ═══════════════════════════════════════════════════════════
# TestTokenize
# ═══════════════════════════════════════════════════════════


class TestTokenize:

    def test_basic_tokenize(self):
        tokens = tokenize("The quick brown fox")
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens
        assert "the" not in tokens  # stop word

    def test_strips_punctuation(self):
        tokens = tokenize("Rochester's wife. Violent, mad.")
        assert "rochesters" in tokens or "rochester" in tokens
        assert "wife" in tokens
        assert "violent" in tokens
        assert "mad" in tokens

    def test_stop_words_filtered(self):
        tokens = tokenize("the first new old set get make")
        assert len(tokens) == 0

    def test_underscore_split(self):
        tokens = tokenize("Madness_Theme is important")
        assert "madness" in tokens
        assert "theme" in tokens

    def test_cjk_preserved(self):
        tokens = tokenize("贾宝玉 is a character")
        assert "贾宝玉" in tokens


# ═══════════════════════════════════════════════════════════
# TestMorphPrefix
# ═══════════════════════════════════════════════════════════


class TestMorphPrefix:

    def test_exact_match(self):
        assert is_morph_prefix("mad", "mad") is True

    def test_valid_suffix_ness(self):
        # "mad" (3 chars) + "ness" (3+ chars) → allowed
        assert is_morph_prefix("mad", "madness") is True

    def test_short_stem_blocks_short_suffix(self):
        # "mr" (2 chars) + "s" (1 char) → blocked
        assert is_morph_prefix("mr", "mrs") is False

    def test_invalid_suffix(self):
        # "attic" + "us" is not a valid morphological suffix
        assert is_morph_prefix("attic", "atticus") is False

    def test_valid_suffix_ing(self):
        assert is_morph_prefix("model", "modeling") is True

    def test_no_prefix_relation(self):
        assert is_morph_prefix("cat", "dog") is False


# ═══════════════════════════════════════════════════════════
# TestNodeNameWords
# ═══════════════════════════════════════════════════════════


class TestNodeNameWords:

    def test_underscore_split(self):
        words = node_name_words("Madness_Theme")
        assert "madness" in words
        assert "theme" in words

    def test_colon_namespace(self):
        words = node_name_words("Spec:Phase_1")
        assert "spec" in words
        assert "phase" in words


# ═══════════════════════════════════════════════════════════
# TestTokenMatchesNode
# ═══════════════════════════════════════════════════════════


class TestTokenMatchesNode:

    def test_exact_match(self):
        words = node_name_words("Madness_Theme")
        assert token_matches_node("madness", words) is True

    def test_morph_match(self):
        words = node_name_words("Madness_Theme")
        assert token_matches_node("mad", words) is True  # mad → madness

    def test_no_match(self):
        words = node_name_words("Madness_Theme")
        assert token_matches_node("coffee", words) is False

    def test_attic_does_not_match_atticus(self):
        words = node_name_words("Atticus_Finch")
        assert token_matches_node("attic", words) is False


# ═══════════════════════════════════════════════════════════
# TestXRefBasic
# ═══════════════════════════════════════════════════════════


class TestXRefBasic:

    def test_creates_virtual_edge(self, engine):
        """Description mentions existing node → XRef creates edge."""
        engine.add_node("Volcano")
        engine.add_edge("Eruption", "Mountain", description="The volcano erupted violently")
        report = XRefResolver.resolve(engine, dry_run=False)
        assert report.edges_created >= 1
        # Check edge exists: Volcano → Eruption (referenced node → source)
        found = any(
            e.source == "Volcano" and e.target == "Eruption"
            and (e.modifiers or {}).get("virtual_reason") == "xref"
            for e in engine._edges
        )
        assert found, "XRef edge Volcano → Eruption should exist"

    def test_dry_run_no_edges(self, engine):
        """dry_run=True reports candidates but creates no edges."""
        engine.add_node("Volcano")
        engine.add_edge("Eruption", "Mountain", description="The volcano erupted violently")
        edges_before = len(engine._edges)
        report = XRefResolver.resolve(engine, dry_run=True)
        assert report.edges_created >= 1
        assert len(engine._edges) == edges_before  # no actual edges added

    def test_skip_existing_neighbor(self, engine):
        """If target is already a neighbor, skip."""
        engine.add_edge("Alpha", "Beta", description="mentions Beta directly")
        # Beta is already a neighbor of Alpha → no XRef needed
        report = XRefResolver.resolve(engine, dry_run=True)
        # Should not create Beta→Alpha (already neighbors)
        xref = [r for r in report.results if r.source == "Beta" and r.target == "Alpha"]
        assert len(xref) == 0

    def test_no_duplicate_within_run(self, engine):
        """Same (source, target) pair not created twice."""
        engine.add_node("Volcano")
        engine.add_edge("Eruption", "Mountain", description="volcano erupted, the volcano was huge")
        report = XRefResolver.resolve(engine, dry_run=True)
        pairs = [(r.source, r.target) for r in report.results]
        assert len(pairs) == len(set(pairs))

    def test_idempotent_single_run(self, engine):
        """Running resolve twice: second run creates zero new edges."""
        engine.add_node("Volcano")
        engine.add_edge("Eruption", "Mountain", description="The volcano erupted")
        r1 = XRefResolver.resolve(engine, dry_run=False)
        assert r1.edges_created >= 1
        r2 = XRefResolver.resolve(engine, dry_run=False)
        assert r2.edges_created == 0

    def test_stop_words_no_edges(self, engine):
        """Description with only stop words creates no XRef edges."""
        engine.add_node("Volcano")
        engine.add_edge("Alpha", "Beta", description="the first new old set")
        report = XRefResolver.resolve(engine, dry_run=True)
        assert report.edges_created == 0

    def test_morph_match_mad_madness(self, engine):
        """'mad' in description matches node Madness_Theme."""
        engine.add_node("Madness_Theme")
        engine.add_edge("Bertha", "Rochester", description="Violent and mad behavior")
        report = XRefResolver.resolve(engine, dry_run=True)
        found = any(r.target == "bertha" and "madness" in r.source for r in report.results)
        assert found, "Should find Madness_Theme → Bertha via 'mad' morph match"

    def test_attic_does_not_match_atticus(self, engine):
        """'attic' in description should NOT match Atticus_Finch."""
        engine.add_node("Atticus_Finch")
        engine.add_edge("Bertha", "Rochester", description="Locked in the attic")
        report = XRefResolver.resolve(engine, dry_run=True)
        atticus = [r for r in report.results if "Atticus" in r.source or "Atticus" in r.target]
        assert len(atticus) == 0


# ═══════════════════════════════════════════════════════════
# TestXRefIDF
# ═══════════════════════════════════════════════════════════


class TestXRefIDF:

    def test_high_frequency_token_filtered(self, engine):
        """Token matching >idf_max_df nodes gets filtered."""
        # Create 10 nodes with "Alpha" in the name
        for i in range(10):
            engine.add_node(f"Alpha_{i:03d}")
        engine.add_edge("Source", "Target", description="something about alpha concept")
        # "alpha" matches 10 nodes. With idf_max_ratio=0.01 and ~12 nodes,
        # idf_max_df=max(3, 0.12)=3. 10 > 3 → should be filtered.
        report = XRefResolver.resolve(engine, dry_run=True, idf_max_ratio=0.01)
        assert report.edges_skipped_idf >= 1


# ═══════════════════════════════════════════════════════════
# TestXRefCommunity
# ═══════════════════════════════════════════════════════════


class TestXRefCommunity:

    def test_same_community_only(self, book_engine):
        """XRef only creates edges within the same community."""
        report = XRefResolver.resolve(book_engine, dry_run=True)
        # Build community map
        gmap = book_engine.get_gravity_map()
        nc = gmap.node_community

        for r in report.results:
            src_comm = nc.get(r.source, {}).get("coarse", -1)
            tgt_comm = nc.get(r.target, {}).get("coarse", -2)
            assert src_comm == tgt_comm, (
                f"Cross-community XRef: [{r.source}](comm={src_comm}) → "
                f"[{r.target}](comm={tgt_comm}) via token '{r.matched_token}'"
            )

    def test_mad_does_not_cross_community(self, book_engine):
        """'mad' in Bertha's description should NOT create edge to Hamlet's Madness_Theme
        if they are in different communities."""
        report = XRefResolver.resolve(book_engine, dry_run=True)
        gmap = book_engine.get_gravity_map()
        nc = gmap.node_community

        bertha_comm = nc.get("Bertha_Mason", {}).get("coarse", -1)
        madness_comm = nc.get("Madness_Theme", {}).get("coarse", -2)

        if bertha_comm != madness_comm:
            # Different communities → should NOT have XRef between them
            cross = [
                r for r in report.results
                if ("Bertha" in r.source and "Madness" in r.target)
                or ("Madness" in r.source and "Bertha" in r.target)
            ]
            assert len(cross) == 0, "Should not create cross-community Bertha↔Madness edge"


# ═══════════════════════════════════════════════════════════
# TestXRefResolveNode
# ═══════════════════════════════════════════════════════════


class TestXRefResolveNode:

    def test_resolve_node_scoped(self, book_engine):
        """resolve_node only processes edges of the specified node."""
        report_full = XRefResolver.resolve(book_engine, dry_run=True)
        report_node = XRefResolver.resolve_node(
            book_engine, "Bertha_Mason", dry_run=True
        )
        # Node-scoped should scan fewer edges
        assert report_node.edges_scanned <= report_full.edges_scanned
        # All results should involve Bertha_Mason as target
        for r in report_node.results:
            assert r.target == "Bertha_Mason", (
                f"resolve_node('Bertha_Mason') returned edge to {r.target}"
            )


# ═══════════════════════════════════════════════════════════
# TestXRefReport
# ═══════════════════════════════════════════════════════════


class TestXRefReport:

    def test_report_fields(self, engine):
        """Report contains expected fields."""
        engine.add_node("Volcano")
        engine.add_edge("Eruption", "Mountain", description="The volcano erupted")
        report = XRefResolver.resolve(engine, dry_run=True)
        assert isinstance(report, XRefReport)
        assert report.edges_scanned >= 1
        assert report.descriptions_found >= 1
        assert isinstance(report.results, list)

    def test_result_fields(self, engine):
        """Each result has expected fields."""
        engine.add_node("Volcano")
        engine.add_edge("Eruption", "Mountain", description="The volcano erupted")
        report = XRefResolver.resolve(engine, dry_run=True)
        assert len(report.results) >= 1
        r = report.results[0]
        assert isinstance(r, XRefResult)
        assert isinstance(r.source, str)
        assert isinstance(r.target, str)
        assert isinstance(r.matched_token, str)
        assert isinstance(r.idf_score, float)
        assert isinstance(r.edge_created, bool)
