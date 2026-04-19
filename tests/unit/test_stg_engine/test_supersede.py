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
        e.add_edge("王婆", "李逍遥", confidence=0.9, rule="empirical",
                    relation="师傅", description="old teacher")
        e.add_edge("王婆", "张三丰", confidence=0.9, rule="empirical",
                    relation="师傅", description="new teacher")

        old = [x for x in e._edges
               if x.target.lower() == "李逍遥"
               and x.modifiers.get("relation") == "师傅"]
        assert len(old) == 1
        assert old[0].modifiers.get("suspected_supersede") is True
        assert old[0].modifiers.get("superseded_by") == "张三丰"

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
        """A→B then A→C then A→D: both B and C should be flagged by D."""
        e = self._engine()
        e.add_edge("NPC", "B", confidence=0.9, relation="师傅",
                    description="first")
        e.add_edge("NPC", "C", confidence=0.9, relation="师傅",
                    description="second")
        e.add_edge("NPC", "D", confidence=0.9, relation="师傅",
                    description="third")

        for name in ("b", "c"):
            flagged = [x for x in e._edges
                       if x.target.lower() == name
                       and x.modifiers.get("relation") == "师傅"]
            assert len(flagged) == 1
            assert flagged[0].modifiers.get("suspected_supersede") is True
            assert flagged[0].modifiers.get("superseded_by") == "D"

    def test_no_semantic_field_no_flag(self):
        e = self._engine()
        e.add_edge("A", "B", confidence=0.9, description="edge 1")
        e.add_edge("A", "C", confidence=0.9, description="edge 2")

        all_edges = list(e._edges)
        assert not any(x.modifiers.get("suspected_supersede") for x in all_edges)

    def test_superseded_at_set(self):
        e = self._engine()
        e.add_edge("王婆", "李逍遥", confidence=0.9, relation="师傅",
                    description="old")
        e.add_edge("王婆", "张三丰", confidence=0.9, relation="师傅",
                    description="new", created_at=1000.0)

        old = [x for x in e._edges
               if x.target.lower() == "李逍遥"
               and x.modifiers.get("relation") == "师傅"]
        assert old[0].modifiers.get("superseded_at") == 1000.0
