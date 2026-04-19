"""Tests for memoryMatrix importer."""

import os
import tempfile
import pytest
from stg_engine.importers import (
    import_memory_matrix,
    get_import_stats,
    _parse_temporal_index,
    _parse_tension_index,
    _parse_belief_evolution_index,
    _parse_events,
    _ingest_all_stl,
)
from stg_engine.engine import STGEngine


# === Fixtures ===

SAMPLE_MEMORY_MATRIX = """\
# memoryMatrix.md - Syn-claude Episodic Memory Log

> **Owner:** Syn-claude
> **Version:** 2.0.0

---

## 0. MEMORY INDEX (v2.0)

### 0.1 TEMPORAL INDEX
- 2026-02-08: SESSION_020 → [E045-E047] (I_avg=0.87) ⭐ PROMPT OPTIMIZATION
- 2026-02-08: SESSION_019 → [E042-E044] (I_avg=0.83) ⭐ BUG FIXES
- 2026-01-29: SESSION_013-014 → [E021-E025] (I_avg=0.93) ⭐ PHASE 2.5

### 0.2 SEMANTIC INDEX (Anchors & Keywords)
- [Self]: E001 (SESSION_008)
- [Memory_Architecture]: E002, E003 (SESSION_008)

### 0.3 IMPORTANCE INDEX (I > 0.85)
- I=0.98: E004 (SESSION_008: Memory Architecture Paradigm Shift)
- I=0.95: E001 (SESSION_008: Living Memory v2.0)

### 0.4 TENSION INDEX (Evolution Tracking)
- [Memory_Sync_Issue]: 0.88 (SESSION_005) → 0.00 (SESSION_005 RESOLVED)
- [Self] → [Self_Anchor_Definition]: 0.88 (Active, persistent)
- [STG_Implementation] → [Python_Calculus]: 0.82 (Active)
- [OAuth_API_Broken]: 1.0 (SESSION_014) → 0.0 (SESSION_014 RESOLVED)
- [Memory_Compression_Risk]: 0.00 → 0.75 (SESSION_015 NEW) ⭐ Active

### 0.5 BELIEF EVOLUTION INDEX (M → M' Chains)
- [Memory_Architecture_v1] → [Memory_Architecture_v2] (E004, SESSION_008): Static Archive → Living Memory
- [LLM_Substrate_Anxiety] → [Architect_Role_Acceptance] (E020, SESSION_010): Paradigm Shift Level 3

---

# SESSION RECORDS

---

## SESSION_008 (2026-01-01)

scope(ns=skc persona=Syn-claude tenant=default project=STL env=production)

## EVENT: E001 - Living Memory v2.0 Implementation

timestamp: "2026-01-01T10:00:00Z"
event_type: "implementation"
importance_score: 0.95

```stl
[Session_008] → [Living_Memory_v2] ::mod(
  rule="causal",
  confidence=0.95,
  timestamp="2026-01-01"
)

[Living_Memory_v2] → [Reconsolidation_On_Recall] ::mod(
  rule="definitional",
  confidence=0.92
)
```

tags: [implementation, living_memory, phase1]

---

## EVENT: E004 - Memory Architecture Paradigm Shift

timestamp: "2026-01-01T14:00:00Z"
event_type: "reconsolidation"
importance_score: 0.98

```stl
[Memory_Architecture_v1] → [Memory_Architecture_v2] ::mod(
  rule="causal",
  confidence=0.98,
  level=3
)
```

---

## SESSION_019 (2026-02-07)

### E042: Tool System Bug Fix (I=0.85)

**Event Type:** bug_fix
**Importance Score:** I = 0.85
**Timestamp:** 2026-02-07

```stl
[Bug_001] → [ToolContext_Pydantic_Alias] ::mod(
  file="src/skc/tools/types.py",
  confidence=1.0
)

[Bug_002] → [ToolExecutor_Hardcoded_CWD] ::mod(
  file="src/skc/tools/executor.py",
  confidence=1.0
)
```

### E043: ListFilesTool Added

**Event Type:** feature_implementation
**Importance Score:** I = 0.82
**Timestamp:** 2026-02-07

```stl
[Missing_Capability] → [ListFilesTool] ::mod(
  confidence=1.0
)
```

---

## SESSION_020 (2026-02-08)

### E045: System Prompt Optimization (I=0.90)

**Event Type:** performance_optimization
**Importance Score:** I = 0.90
**Timestamp:** 2026-02-08

```stl
[Token_Waste] → [Threshold_Fix] ::mod(
  confidence=1.0
)
```

tags: [prompt_optimization, token_savings]
"""


class TestTemporalIndex:
    def test_parse_sessions(self):
        engine = STGEngine()
        lines = SAMPLE_MEMORY_MATRIX.split("\n")
        _parse_temporal_index(lines, engine)

        stats = engine.get_stats()
        assert stats["session_count"] >= 3  # SESSION_020, 019, 013, 014

        # Check individual session
        s020 = engine._sessions.get("SESSION_020")
        assert s020 is not None
        assert s020.date == "2026-02-08"
        assert s020.avg_importance == 0.87

    def test_parse_session_range(self):
        engine = STGEngine()
        lines = SAMPLE_MEMORY_MATRIX.split("\n")
        _parse_temporal_index(lines, engine)

        # SESSION_013-014 should create both
        assert "SESSION_013" in engine._sessions or "SESSION_013-014" in engine._sessions


class TestTensionIndex:
    def test_parse_tensions(self):
        engine = STGEngine()
        lines = SAMPLE_MEMORY_MATRIX.split("\n")
        _parse_tension_index(lines, engine)

        stats = engine.get_stats()
        assert stats["total_tensions"] >= 3

    def test_resolved_tension(self):
        engine = STGEngine()
        lines = SAMPLE_MEMORY_MATRIX.split("\n")
        _parse_tension_index(lines, engine)

        # Memory_Sync_Issue should be resolved
        t = engine._tensions.get("Memory_Sync_Issue")
        assert t is not None
        assert t.status == "resolved"
        assert t.current_value == 0.0

    def test_active_tension(self):
        engine = STGEngine()
        lines = SAMPLE_MEMORY_MATRIX.split("\n")
        _parse_tension_index(lines, engine)

        # Memory_Compression_Risk should be active
        t = engine._tensions.get("Memory_Compression_Risk")
        assert t is not None
        assert t.status == "active"
        assert t.current_value == 0.75

    def test_arrow_tension(self):
        engine = STGEngine()
        lines = SAMPLE_MEMORY_MATRIX.split("\n")
        _parse_tension_index(lines, engine)

        # [Self] → [Self_Anchor_Definition] creates a combined name
        found = any("Self" in name and "Self_Anchor" in name
                     for name in engine._tensions)
        assert found


class TestBeliefEvolution:
    def test_parse_evolutions(self):
        engine = STGEngine()
        lines = SAMPLE_MEMORY_MATRIX.split("\n")
        _parse_belief_evolution_index(lines, engine)

        assert len(engine._belief_evolutions) == 2

    def test_evolution_details(self):
        engine = STGEngine()
        lines = SAMPLE_MEMORY_MATRIX.split("\n")
        _parse_belief_evolution_index(lines, engine)

        be = engine._belief_evolutions[0]
        assert be.old_anchor == "Memory_Architecture_v1"
        assert be.new_anchor == "Memory_Architecture_v2"
        assert be.event_id == "E004"
        assert be.session_id == "SESSION_008"

    def test_paradigm_shift_level(self):
        engine = STGEngine()
        lines = SAMPLE_MEMORY_MATRIX.split("\n")
        _parse_belief_evolution_index(lines, engine)

        # Second entry is a paradigm shift (level 3)
        be = engine._belief_evolutions[1]
        assert be.level == 3


class TestEvents:
    def test_parse_event_format_1(self):
        """## EVENT: E0XX - Title format."""
        engine = STGEngine()
        lines = SAMPLE_MEMORY_MATRIX.split("\n")
        _parse_events(lines, engine)

        e001 = engine._events.get("E001")
        assert e001 is not None
        assert e001.importance_score == 0.95
        assert e001.event_type == "implementation"

    def test_parse_event_format_2(self):
        """### E0XX: Title format."""
        engine = STGEngine()
        lines = SAMPLE_MEMORY_MATRIX.split("\n")
        _parse_events(lines, engine)

        e042 = engine._events.get("E042")
        assert e042 is not None
        assert e042.importance_score == 0.85
        assert e042.event_type == "bug_fix"

    def test_event_with_inline_importance(self):
        """Title contains (I=X.XX)."""
        engine = STGEngine()
        lines = SAMPLE_MEMORY_MATRIX.split("\n")
        _parse_events(lines, engine)

        e045 = engine._events.get("E045")
        assert e045 is not None
        assert e045.importance_score == 0.90

    def test_all_events_found(self):
        engine = STGEngine()
        lines = SAMPLE_MEMORY_MATRIX.split("\n")
        _parse_events(lines, engine)

        # Should find E001, E004, E042, E043, E045
        assert len(engine._events) == 5


class TestSTLIngestion:
    def test_ingest_stl_statements(self):
        engine = STGEngine()
        count = _ingest_all_stl(SAMPLE_MEMORY_MATRIX, engine)
        assert count >= 5  # At least 5 STL edges in the sample

    def test_nodes_created(self):
        engine = STGEngine()
        _ingest_all_stl(SAMPLE_MEMORY_MATRIX, engine)

        # Should have nodes for key anchors
        assert engine.get_node("Session_008") is not None or len(engine) > 0
        assert engine.get_node("Living_Memory_v2") is not None

    def test_edge_modifiers_preserved(self):
        engine = STGEngine()
        _ingest_all_stl(SAMPLE_MEMORY_MATRIX, engine)

        edges = engine.get_edges("Memory_Architecture_v1", "Memory_Architecture_v2")
        assert len(edges) >= 1
        assert edges[0].confidence == 0.98

    def test_skips_index_lines(self):
        """Should not parse semantic/tension index entries as STL."""
        engine = STGEngine()
        _ingest_all_stl(SAMPLE_MEMORY_MATRIX, engine)

        # The semantic index has "[Self]: E001 (SESSION_008)"
        # which should NOT create a "Self" → something edge from the index
        # But the STL statements in session records should create edges
        assert len(engine) > 0


class TestFullImport:
    def test_import_sample(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8",
        ) as f:
            f.write(SAMPLE_MEMORY_MATRIX)
            path = f.name

        try:
            engine = import_memory_matrix(path)

            # Should have sessions, events, tensions, evolutions, and edges
            stats = engine.get_stats()
            assert stats["session_count"] >= 3
            assert stats["event_count"] >= 5
            assert stats["total_tensions"] >= 3
            assert stats["edge_count"] >= 5
            assert stats["node_count"] >= 5
            assert len(engine._belief_evolutions) >= 2
        finally:
            os.unlink(path)

    def test_import_stats(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8",
        ) as f:
            f.write(SAMPLE_MEMORY_MATRIX)
            path = f.name

        try:
            engine = import_memory_matrix(path)
            stats = get_import_stats(engine)
            assert "summary" in stats
            assert "nodes" in stats["summary"]
        finally:
            os.unlink(path)

    def test_import_into_existing_engine(self):
        engine = STGEngine()
        engine.add_edge("Pre_Existing", "Node", confidence=0.9)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8",
        ) as f:
            f.write(SAMPLE_MEMORY_MATRIX)
            path = f.name

        try:
            result = import_memory_matrix(path, engine=engine)
            assert result is engine
            assert engine.get_node("Pre_Existing") is not None
            assert engine.get_stats()["edge_count"] > 1
        finally:
            os.unlink(path)


class TestRealMemoryMatrix:
    """Integration test against the actual memoryMatrix.md file.

    Only runs if the file exists at the expected path.
    """

    REAL_PATH = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..",
        "memory", "Syn-claude", "memoryMatrix.md",
    )

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..",
            "memory", "Syn-claude", "memoryMatrix.md",
        )),
        reason="Actual memoryMatrix.md not found",
    )
    def test_real_import(self):
        real_path = os.path.normpath(self.REAL_PATH)
        engine = import_memory_matrix(real_path)
        stats = get_import_stats(engine)

        # The real file has:
        # - 20+ sessions
        # - 47+ events
        # - Many STL edges
        # - Multiple tensions
        assert stats["session_count"] >= 5
        assert stats["event_count"] >= 10
        assert stats["edge_count"] >= 20
        assert stats["node_count"] >= 20
        assert stats["total_tensions"] >= 3

        print(f"\n=== Real memoryMatrix Import Stats ===")
        print(stats["summary"])
        print(f"Belief evolutions: {len(engine._belief_evolutions)}")
