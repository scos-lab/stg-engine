"""Unit tests for the propagate inverted word index (F5).

Focus: the lazily-built index (word -> nodes, first-char buckets, CJK-node set,
per-node word cache) stays consistent with a full rebuild across graph
mutations, and candidate-filtered propagate yields the same result as a fresh
engine. End-to-end set-equality on the real 9.4K-node graph is covered by the
golden-verify harness; these tests pin the maintenance invariants.
"""
from stg_engine.engine import STGEngine, _name_words, _is_morph_prefix, _HAS_CJK


def _full_word_index(engine):
    exp = {}
    for nk in engine._nodes:
        for w in _name_words(nk):
            exp.setdefault(w, set()).add(nk)
    return exp


def test_index_equals_full_rebuild():
    e = STGEngine()
    e.add_edge("Density_Monism", "Rho_Field", is_a="x")
    e.add_edge("Memory_System", "Recall_Path", is_a="x")
    e._ensure_word_index()
    assert e._word_index == _full_word_index(e)
    # first-char buckets and node cache are consistent
    for w, nodes in e._word_index.items():
        assert w in e._word_first_char[w[0]]
    for nk in e._nodes:
        assert e._node_words[nk] == _name_words(nk)


def test_index_invalidated_and_rebuilt_after_add():
    e = STGEngine()
    e.add_edge("Density_Monism", "Rho_Field", is_a="x")
    e.propagate("density", read_only=True)              # builds index
    assert e._word_index is not None
    e.add_edge("Koide_Formula", "Tau_Mass", is_a="x")  # mutation -> invalidate
    assert e._word_index is None
    e.propagate("koide", read_only=True)               # lazy rebuild
    assert e._word_index == _full_word_index(e)
    assert "koide" in e._word_index                    # newly added word present


def test_index_consistent_after_remove_edge():
    e = STGEngine()
    e.add_edge("Alpha_One", "Beta_Two", is_a="x")
    e.add_edge("Gamma_Three", "Delta_Four", is_a="x")
    e._ensure_word_index()
    e.remove_edge("Gamma_Three", "Delta_Four")         # invalidates
    e._ensure_word_index()                             # rebuild
    assert e._word_index == _full_word_index(e)


def test_register_alias_does_not_corrupt_index():
    e = STGEngine()
    e.add_edge("Semantic_Tension", "Graph_Node", is_a="x")
    e._ensure_word_index()
    before = {w: set(s) for w, s in e._word_index.items()}
    e.register_alias("STG", "Semantic_Tension")        # no node-name change
    e._word_index = None                               # force rebuild
    e._ensure_word_index()
    assert e._word_index == before == _full_word_index(e)


def test_incremental_propagate_matches_fresh_engine():
    e = STGEngine()
    e.add_edge("Consciousness_Field", "Density_Monism", is_a="x")
    e.propagate("consciousness", read_only=True)       # build over first graph
    e.add_edge("Consciousness_Act", "Awareness", is_a="x")
    after = sorted(e.propagate("consciousness", read_only=True))
    fresh = STGEngine()
    for s, t in [("Consciousness_Field", "Density_Monism"),
                 ("Consciousness_Act", "Awareness")]:
        fresh.add_edge(s, t, is_a="x")
    assert after == sorted(fresh.propagate("consciousness", read_only=True))


def test_cjk_node_indexed_and_candidate_substring():
    e = STGEngine()
    e.add_edge("贾宝玉", "红楼梦", is_a="x")
    e.add_edge("Latin_Only", "Node", is_a="x")
    e._ensure_word_index()
    jia = e._nk("贾宝玉")
    assert jia in e._cjk_name_nodes
    assert e._nk("latin_only") not in e._cjk_name_nodes
    # CJK substring "宝玉" is not a word unit but must still surface the node
    # via the CJK-substring candidate path.
    cand = e._seed_candidates(["宝玉"], [])
    assert jia in cand
    # and end-to-end the node is recalled
    assert "贾宝玉" in e.propagate("宝玉", read_only=True)


def test_candidate_superset_covers_morph_matches():
    # A morphological match (token "consciousness" vs node word "conscious")
    # must appear in the candidate set, not just exact matches.
    e = STGEngine()
    e.add_edge("Conscious_Mind", "Field", is_a="x")     # word "conscious"
    e.add_edge("Unrelated_Thing", "Other", is_a="x")
    e._ensure_word_index()
    cand = e._seed_candidates(["consciousness"], [])
    # brute-force full scan with the exact matching predicate
    full = set()
    for nk in e._nodes:
        words = _name_words(nk)
        if "consciousness" in words or any(
            _is_morph_prefix("consciousness", w) or _is_morph_prefix(w, "consciousness")
            for w in words
        ):
            full.add(nk)
    assert full and full <= cand
