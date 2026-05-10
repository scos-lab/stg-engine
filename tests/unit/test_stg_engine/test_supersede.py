"""Tests for SEMANTIC_FIELDS and supersede detection."""

import pytest
from stg_engine import STGEngine, SEMANTIC_FIELDS


class TestSemanticFields:

    def test_constant_is_tuple(self):
        assert isinstance(SEMANTIC_FIELDS, tuple)

    def test_contains_expected_fields(self):
        for f in ("relation", "status", "role", "type", "kind",
                   "is_a", "action", "predicate", "phase"):
            assert f in SEMANTIC_FIELDS

    def test_importable_from_engine(self):
        from stg_engine.engine import SEMANTIC_FIELDS as SF, _get_semantic_field
        assert SF is SEMANTIC_FIELDS
        assert callable(_get_semantic_field)


class TestGetSemanticField:

    def test_returns_first_match(self):
        from stg_engine.engine import _get_semantic_field
        field, value = _get_semantic_field({"status": "dead", "role": "leader"})
        # "status" comes after "relation" but before "role" in tuple order
        assert field == "status"
        assert value == "dead"

    def test_returns_none_for_empty(self):
        from stg_engine.engine import _get_semantic_field
        assert _get_semantic_field({}) == (None, None)
        assert _get_semantic_field(None) == (None, None)

    def test_skips_falsy_values(self):
        from stg_engine.engine import _get_semantic_field
        field, value = _get_semantic_field({"relation": "", "role": "mentor"})
        assert field == "role"
        assert value == "mentor"


class TestSupersedeDetection:

    def _engine(self):
        return STGEngine()

    def test_flags_same_source_field_value_different_target(self):
        e = self._engine()
        # Single-value field (status) + >60s gap → genuine correction
        e.add_edge("Project_X", "Active", confidence=0.9, rule="empirical",
                    status="phase", description="old phase",
                    created_at=1000.0)
        e.add_edge("Project_X", "Done", confidence=0.9, rule="empirical",
                    status="phase", description="new phase",
                    created_at=2000.0)

        old = [x for x in e._edges
               if x.target.lower() == "active"
               and x.modifiers.get("status") == "phase"]
        assert len(old) == 1
        assert old[0].modifiers.get("suspected_supersede") is True
        assert old[0].modifiers.get("superseded_by") == "Done"

    def test_different_field_no_flag(self):
        e = self._engine()
        e.add_edge("王婆", "李逍遥", confidence=0.9, relation="师傅")
        e.add_edge("王婆", "张三丰", confidence=0.9, status="师傅")

        old = [x for x in e._edges
               if x.target.lower() == "李逍遥"]
        assert not any(x.modifiers.get("suspected_supersede") for x in old)

    def test_different_value_no_flag(self):
        e = self._engine()
        e.add_edge("王婆", "李逍遥", confidence=0.9, relation="师傅")
        e.add_edge("王婆", "张三丰", confidence=0.9, relation="朋友")

        old = [x for x in e._edges
               if x.target.lower() == "李逍遥"]
        assert not any(x.modifiers.get("suspected_supersede") for x in old)

    def test_same_target_no_flag(self):
        e = self._engine()
        e.add_edge("王婆", "李逍遥", confidence=0.9, relation="师傅",
                    description="first")
        e.add_edge("王婆", "李逍遥", confidence=0.95, relation="师傅",
                    description="updated")

        edges = [x for x in e._edges
                 if x.target.lower() == "李逍遥"]
        assert not any(x.modifiers.get("suspected_supersede") for x in edges)

    def test_virtual_edges_ignored(self):
        e = self._engine()
        e.add_edge("王婆", "李逍遥", confidence=0.9, relation="师傅",
                    edge_class="virtual", virtual_reason="co_source")
        e.add_edge("王婆", "张三丰", confidence=0.9, relation="师傅",
                    description="new")

        virtual = [x for x in e._edges
                   if x.target.lower() == "李逍遥"
                   and x.modifiers.get("relation") == "师傅"]
        assert not any(x.modifiers.get("suspected_supersede") for x in virtual)

    def test_chain_correction_re_flags(self):
        """A→B then A→C then A→D: both B and C should be flagged by D.

        Uses single-value field + spaced timestamps so both guards pass.
        """
        e = self._engine()
        e.add_edge("NPC", "B", confidence=0.9, status="phase",
                    description="first", created_at=1000.0)
        e.add_edge("NPC", "C", confidence=0.9, status="phase",
                    description="second", created_at=2000.0)
        e.add_edge("NPC", "D", confidence=0.9, status="phase",
                    description="third", created_at=3000.0)

        for name in ("b", "c"):
            flagged = [x for x in e._edges
                       if x.target.lower() == name
                       and x.modifiers.get("status") == "phase"]
            assert len(flagged) == 1
            assert flagged[0].modifiers.get("suspected_supersede") is True
            assert flagged[0].modifiers.get("superseded_by") == "D"

    def test_no_semantic_field_no_flag(self):
        e = self._engine()
        e.add_edge("A", "B", confidence=0.9, description="edge 1")
        e.add_edge("A", "C", confidence=0.9, description="edge 2")

        all_edges = list(e._edges)
        assert not any(x.modifiers.get("suspected_supersede") for x in all_edges)

    def test_multi_value_field_no_flag(self):
        """Guard A: is_a / action / role are cardinality=many — must not flag.

        Regression for stg-steam: every Game→Feature edge was being
        falsely marked superseded by whatever feature was ingested last.
        """
        e = self._engine()
        e.add_edge("Game", "Single_player", confidence=0.99, is_a="game_feature",
                    created_at=1000.0)
        e.add_edge("Game", "Multi_player", confidence=0.99, is_a="game_feature",
                    created_at=2000.0)
        e.add_edge("Game", "PvP", confidence=0.99, is_a="game_feature",
                    created_at=3000.0)

        all_edges = [x for x in e._edges
                     if x.modifiers.get("is_a") == "game_feature"]
        assert len(all_edges) == 3
        assert not any(x.modifiers.get("suspected_supersede") for x in all_edges)

    def test_same_batch_no_flag(self):
        """Guard B: edges within SUPERSEDE_MIN_GAP_SECONDS = co-declared, not correction."""
        e = self._engine()
        # Same single-value field, but ingested 1 second apart (within 60s window)
        e.add_edge("Project_Y", "Active", confidence=0.9, status="phase",
                    created_at=1000.0)
        e.add_edge("Project_Y", "Pending", confidence=0.9, status="phase",
                    created_at=1001.0)

        edges = [x for x in e._edges if x.modifiers.get("status") == "phase"]
        assert len(edges) == 2
        assert not any(x.modifiers.get("suspected_supersede") for x in edges)

    def test_superseded_at_set(self):
        e = self._engine()
        e.add_edge("Project_X", "Active", confidence=0.9, status="phase",
                    description="old", created_at=500.0)
        e.add_edge("Project_X", "Done", confidence=0.9, status="phase",
                    description="new", created_at=1000.0)

        old = [x for x in e._edges
               if x.target.lower() == "active"
               and x.modifiers.get("status") == "phase"]
        assert old[0].modifiers.get("superseded_at") == 1000.0
