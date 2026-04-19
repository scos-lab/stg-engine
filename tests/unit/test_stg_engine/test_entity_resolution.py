"""Tests for Entity Resolution (G7).

Three layers:
  - Layer 0: Separator normalization (_nk treats - as _)
  - Layer 1: Candidate detection (similar node warnings)
  - Layer 2: Alias registry (persistent name mappings)
"""

import time
import pytest
from stg_engine.engine import STGEngine


# ═══════════════════════════════════════════════════════════
# Layer 0: Separator Normalization
# ═══════════════════════════════════════════════════════════


class TestSeparatorNormalization:
    def test_hyphen_underscore_same_node(self):
        engine = STGEngine()
        engine.add_node("A-B")
        engine.add_node("A_B")
        assert len(engine._nodes) == 1

    def test_hyphen_underscore_edge_same_source(self):
        engine = STGEngine()
        engine.add_edge("A-B", "C", confidence=0.8)
        engine.add_edge("A_B", "D", confidence=0.7)
        # A-B and A_B are same node, should have 2 outgoing edges
        assert len(engine._nodes) == 3  # A-B, C, D

    def test_display_name_preserved(self):
        engine = STGEngine()
        engine.add_node("Syn-claude")
        engine.add_node("Syn_Claude")
        # First add_node sets the display name
        node = engine.get_node("Syn-claude")
        assert node is not None
        assert node.name == "Syn-claude"

    def test_nk_idempotent(self):
        assert STGEngine._nk("A-B-C") == STGEngine._nk("A_B_C")
        assert STGEngine._nk("a-b") == STGEngine._nk("A_B")

    def test_get_node_either_separator(self):
        engine = STGEngine()
        engine.add_node("My-Node")
        assert engine.get_node("My_Node") is not None
        assert engine.get_node("my-node") is not None


# ═══════════════════════════════════════════════════════════
# Layer 1: Candidate Detection
# ═══════════════════════════════════════════════════════════


class TestCandidateDetection:
    def test_word_order_variant_detected(self):
        engine = STGEngine()
        engine.add_node("Exit_Cleanup")
        engine.add_node("Cleanup_Exit")  # word-order variant
        candidates = engine._last_entity_candidates
        assert len(candidates) >= 1
        assert candidates[0][0] == "Exit_Cleanup"
        assert candidates[0][1] == 1.0
        assert "word-order" in candidates[0][2]

    def test_high_jaccard_detected(self):
        engine = STGEngine()
        engine.add_node("Koide_Formula")
        engine.add_node("Koide_Mass_Formula")
        candidates = engine._last_entity_candidates
        assert len(candidates) >= 1
        assert candidates[0][0] == "Koide_Formula"
        assert candidates[0][1] >= 2 / 3

    def test_parent_child_not_reported(self):
        """Specialization (token count differs by ≥2) should not trigger."""
        engine = STGEngine()
        engine.add_node("Handle_API_Error")
        engine._last_entity_candidates = []
        engine.add_node("Handle_API_Error_Gemini_OAuth")  # +2 tokens
        assert len(engine._last_entity_candidates) == 0

    def test_low_overlap_not_reported(self):
        engine = STGEngine()
        engine.add_node("Energy")
        engine._last_entity_candidates = []
        engine.add_node("Dark_Energy")
        # Single-token "Energy" has < 2 tokens, should not match
        assert len(engine._last_entity_candidates) == 0

    def test_single_token_no_candidates(self):
        """Single-token names should never trigger candidates."""
        engine = STGEngine()
        engine.add_node("Alpha")
        engine._last_entity_candidates = []
        engine.add_node("Beta")
        assert len(engine._last_entity_candidates) == 0

    def test_existing_node_no_candidates(self):
        """Re-adding an existing node should not trigger candidates."""
        engine = STGEngine()
        engine.add_node("My_Node")
        engine._last_entity_candidates = []
        engine.add_node("My_Node")  # same node, just update
        assert len(engine._last_entity_candidates) == 0

    def test_candidates_sorted_by_score(self):
        engine = STGEngine()
        engine.add_node("Alpha_Beta_Gamma")
        engine.add_node("Alpha_Beta_Delta")
        # Now add a node similar to both
        engine.add_node("Gamma_Beta_Alpha")  # word-order variant of first
        candidates = engine._last_entity_candidates
        assert len(candidates) >= 1
        # Word-order variant (score=1.0) should be first
        assert candidates[0][1] == 1.0


# ═══════════════════════════════════════════════════════════
# Layer 2: Alias Registry
# ═══════════════════════════════════════════════════════════


class TestAliasRegistry:
    def test_register_alias(self):
        engine = STGEngine()
        engine.add_node("Psi_Formula")
        ok = engine.register_alias("Formula_Psi", "Psi_Formula")
        assert ok is True

    def test_register_alias_nonexistent_canonical(self):
        engine = STGEngine()
        ok = engine.register_alias("Alias", "Nonexistent")
        assert ok is False

    def test_alias_resolves_in_add_node(self):
        engine = STGEngine()
        engine.add_node("Psi_Formula")
        engine.register_alias("Formula_Psi", "Psi_Formula")
        engine.add_node("Formula_Psi")
        # Should resolve to existing node, not create new one
        assert len(engine._nodes) == 1

    def test_alias_resolves_in_add_edge(self):
        engine = STGEngine()
        engine.add_node("Psi_Formula")
        engine.register_alias("Formula_Psi", "Psi_Formula")
        engine.add_edge("Formula_Psi", "Target", confidence=0.9)
        # Edge source should be the canonical name
        edge = engine._edges[0]
        assert edge.source == "Psi_Formula"

    def test_remove_alias(self):
        engine = STGEngine()
        engine.add_node("Psi_Formula")
        engine.register_alias("Formula_Psi", "Psi_Formula")
        removed = engine.remove_alias("Formula_Psi")
        assert removed is True
        # After removal, add_node creates a new node
        engine.add_node("Formula_Psi")
        assert len(engine._nodes) == 2

    def test_remove_nonexistent_alias(self):
        engine = STGEngine()
        assert engine.remove_alias("Nonexistent") is False

    def test_list_aliases(self):
        engine = STGEngine()
        engine.add_node("A_B")
        engine.add_node("C_D")
        engine.register_alias("B_A", "A_B")
        engine.register_alias("D_C", "C_D")
        aliases = engine.list_aliases()
        assert len(aliases) == 2

    def test_resolve_name_with_alias(self):
        engine = STGEngine()
        engine.add_node("Psi_Formula")
        engine.register_alias("Formula_Psi", "Psi_Formula")
        assert engine.resolve_name("Formula_Psi") == "Psi_Formula"

    def test_resolve_name_without_alias(self):
        engine = STGEngine()
        engine.add_node("Psi_Formula")
        assert engine.resolve_name("Psi_Formula") == "Psi_Formula"

    def test_resolve_name_unknown(self):
        engine = STGEngine()
        assert engine.resolve_name("Unknown") == "Unknown"

    def test_alias_case_insensitive(self):
        engine = STGEngine()
        engine.add_node("Psi_Formula")
        engine.register_alias("FORMULA_PSI", "Psi_Formula")
        engine.add_node("formula_psi")
        assert len(engine._nodes) == 1

    def test_alias_separator_insensitive(self):
        engine = STGEngine()
        engine.add_node("Psi_Formula")
        engine.register_alias("Formula-Psi", "Psi_Formula")
        engine.add_node("Formula_Psi")
        assert len(engine._nodes) == 1


# ═══════════════════════════════════════════════════════════
# Persistence
# ═══════════════════════════════════════════════════════════


class TestAliasPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "test.stg")
        engine = STGEngine()
        engine.add_node("Psi_Formula")
        engine.add_node("Target")
        engine.add_edge("Psi_Formula", "Target", confidence=0.9)
        engine.register_alias("Formula_Psi", "Psi_Formula")
        engine.save(path)

        engine2 = STGEngine.load(path)
        assert len(engine2._aliases) == 1
        # Alias should work after load
        engine2.add_node("Formula_Psi")
        assert len(engine2._nodes) == 2  # Psi_Formula + Target

    def test_load_no_aliases_table(self, tmp_path):
        """Old .stg files without aliases table should load fine."""
        path = str(tmp_path / "test.stg")
        engine = STGEngine()
        engine.add_node("A")
        engine.save(path)
        # Load should not fail even if aliases table doesn't exist
        engine2 = STGEngine.load(path)
        assert engine2._aliases == {}


# ═══════════════════════════════════════════════════════════
# Tokenizer
# ═══════════════════════════════════════════════════════════


class TestTokenizer:
    def test_basic_tokenization(self):
        tokens = STGEngine._tokenize_for_er("Hello_World")
        assert tokens == frozenset({"hello", "world"})

    def test_stop_words_removed(self):
        tokens = STGEngine._tokenize_for_er("Path_To_The_Goal")
        assert "to" not in tokens
        assert "the" not in tokens
        assert "path" in tokens
        assert "goal" in tokens

    def test_hyphen_underscore_equivalent(self):
        t1 = STGEngine._tokenize_for_er("A-B-C")
        t2 = STGEngine._tokenize_for_er("A_B_C")
        assert t1 == t2

    def test_empty_name(self):
        tokens = STGEngine._tokenize_for_er("")
        assert tokens == frozenset()
