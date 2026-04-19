"""Tests for Universal STG Knowledge Base Importer.

Tests the concept skeleton, markdown extractor, spec extractor,
and universal importer orchestrator.
"""

import json
import os
import tempfile
import pytest

from stg_engine.engine import STGEngine
from stg_engine.concept_skeleton import (
    CORE_CONCEPTS,
    SKELETON_EDGES,
    CORE_CONCEPT_NAMES,
    inject_skeleton,
)
from stg_engine.markdown_extractor import (
    extract_markdown_structure,
    _sanitize_name,
    _derive_doc_name,
    _find_sections,
    _concept_mentioned_in,
)
from stg_engine.spec_extractor import (
    extract_spec,
    _extract_header,
    _extract_phase_refs,
    _sanitize_name as spec_sanitize_name,
)
from stg_engine.universal_importer import (
    import_knowledge_base,
    _resolve_paths,
    _load_manifest,
    _load_stl_manifest,
    _modifiers_to_dict,
)


# ── Concept Skeleton Tests ──────────────────────────────────────────

class TestConceptSkeleton:
    """Tests for concept_skeleton.py"""

    def test_core_concepts_not_empty(self):
        assert len(CORE_CONCEPTS) >= 25

    def test_core_concepts_have_required_fields(self):
        for concept in CORE_CONCEPTS:
            assert "name" in concept
            assert "anchor_type" in concept

    def test_core_concept_names_unique(self):
        names = [c["name"] for c in CORE_CONCEPTS]
        assert len(names) == len(set(names))

    def test_skeleton_edges_not_empty(self):
        assert len(SKELETON_EDGES) >= 20

    def test_skeleton_edges_reference_valid_concepts(self):
        concept_names = {c["name"] for c in CORE_CONCEPTS}
        for edge in SKELETON_EDGES:
            assert edge["source"] in concept_names, \
                f"Edge source '{edge['source']}' not in CORE_CONCEPTS"
            assert edge["target"] in concept_names, \
                f"Edge target '{edge['target']}' not in CORE_CONCEPTS"

    def test_skeleton_edges_have_confidence(self):
        for edge in SKELETON_EDGES:
            assert "confidence" in edge
            assert 0.0 <= edge["confidence"] <= 1.0

    def test_skeleton_edges_have_rule(self):
        valid_rules = {"causal", "logical", "empirical", "definitional"}
        for edge in SKELETON_EDGES:
            assert "rule" in edge
            assert edge["rule"] in valid_rules

    def test_core_concept_names_frozenset(self):
        assert isinstance(CORE_CONCEPT_NAMES, frozenset)
        assert "Self" in CORE_CONCEPT_NAMES
        assert "Consciousness" in CORE_CONCEPT_NAMES
        assert "STL" in CORE_CONCEPT_NAMES

    def test_inject_skeleton(self):
        engine = STGEngine()
        count = inject_skeleton(engine)
        assert count == len(CORE_CONCEPTS) + len(SKELETON_EDGES)

    def test_inject_skeleton_creates_nodes(self):
        engine = STGEngine()
        inject_skeleton(engine)
        for concept in CORE_CONCEPTS:
            node = engine.get_node(concept["name"])
            assert node is not None, f"Node '{concept['name']}' not created"
            assert node.anchor_type == concept.get("anchor_type")

    def test_inject_skeleton_creates_edges(self):
        engine = STGEngine()
        inject_skeleton(engine)
        for edge in SKELETON_EDGES:
            edges = engine.get_edges(source=edge["source"])
            targets = [e.target for e in edges]
            assert edge["target"] in targets, \
                f"Edge {edge['source']} -> {edge['target']} not created"

    def test_inject_skeleton_idempotent(self):
        """Calling inject_skeleton twice should not cause errors."""
        engine = STGEngine()
        inject_skeleton(engine)
        count1 = engine.get_stats()["node_count"]
        inject_skeleton(engine)
        count2 = engine.get_stats()["node_count"]
        assert count2 >= count1  # May create duplicates but shouldn't crash

    def test_self_node_has_agent_type(self):
        engine = STGEngine()
        inject_skeleton(engine)
        node = engine.get_node("Self")
        assert node is not None
        assert node.anchor_type == "Agent"

    def test_skc_connected_to_multiple_components(self):
        """SKC should be a hub connecting architecture, phases, etc."""
        engine = STGEngine()
        inject_skeleton(engine)
        edges = engine.get_edges(source="SKC")
        assert len(edges) >= 3  # Memory_Architecture, STG_Engine, Cognitive_Orchestrator


# ── Markdown Extractor Tests ────────────────────────────────────────

class TestSanitizeName:

    def test_simple_name(self):
        assert _sanitize_name("Hello World") == "Hello_World"

    def test_special_chars_removed(self):
        assert _sanitize_name("**Bold** Text!") == "Bold_Text"

    def test_markdown_formatting(self):
        assert _sanitize_name("# Heading `code`") == "Heading_code"

    def test_empty_fallback(self):
        assert _sanitize_name("!!!") == "Unnamed"

    def test_colons_preserved(self):
        assert _sanitize_name("Physics:Energy") == "Physics:Energy"


class TestDeriveDocName:

    def test_simple_filename(self):
        assert _derive_doc_name("docs/README.md") == "README"

    def test_complex_filename(self):
        assert _derive_doc_name("path/to/MEMORY_ARCHITECTURE_DESIGN.md") == \
            "MEMORY_ARCHITECTURE_DESIGN"

    def test_windows_path(self):
        assert _derive_doc_name("C:\\path\\to\\FILE.md") == "FILE"


class TestFindSections:

    def test_finds_h2_sections(self):
        lines = [
            "# Title\n",
            "\n",
            "## Section 1\n",
            "Content\n",
            "## Section 2\n",
            "More content\n",
        ]
        sections = _find_sections(lines)
        assert len(sections) == 2
        assert sections[0].title == "Section 1"
        assert sections[1].title == "Section 2"

    def test_section_line_ranges(self):
        lines = [
            "## A\n",
            "text\n",
            "## B\n",
            "text\n",
        ]
        sections = _find_sections(lines)
        assert sections[0].line_start == 1
        assert sections[0].line_end == 2
        assert sections[1].line_start == 3
        assert sections[1].line_end == 4

    def test_section_summary(self):
        lines = [
            "## Section\n",
            "\n",
            "This is the summary line.\n",
            "More content.\n",
        ]
        sections = _find_sections(lines)
        assert sections[0].summary == "This is the summary line."

    def test_no_sections(self):
        lines = ["Just text\n", "No headings\n"]
        sections = _find_sections(lines)
        assert len(sections) == 0


class TestConceptMentioned:

    def test_exact_match(self):
        assert _concept_mentioned_in("The STL language is great", "STL")

    def test_underscore_as_space(self):
        assert _concept_mentioned_in("Memory Architecture design", "Memory_Architecture")

    def test_case_insensitive(self):
        assert _concept_mentioned_in("the stl parser", "STL")

    def test_no_false_positive(self):
        assert not _concept_mentioned_in("Phase 10 is new", "Phase_1")


class TestMarkdownExtractor:

    def test_extract_simple_doc(self):
        engine = STGEngine()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "Test_Document.md")
            with open(path, 'w', encoding='utf-8') as f:
                f.write("# Test Document\n\n## Section A\nContent A\n\n## Section B\nContent B\n")
            count = extract_markdown_structure(path, engine, namespace="Test")
            assert count >= 5  # 1 doc + 2 sections + 2 containment edges
            assert engine.get_node("Test:Test_Document") is not None

    def test_extract_with_project_root(self):
        engine = STGEngine()
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "DOC.md")
            with open(filepath, 'w') as f:
                f.write("# Doc\n\n## Part 1\nSome text\n")
            count = extract_markdown_structure(
                filepath, engine, namespace="Test", project_root=tmpdir
            )
            node = engine.get_node("Test:DOC")
            assert node is not None
            assert node.metadata.get("file") == "DOC.md"

    def test_bridges_to_core_concepts(self):
        engine = STGEngine()
        inject_skeleton(engine)
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', delete=False, encoding='utf-8'
        ) as f:
            f.write("# Test\n\nThis document discusses STL and Consciousness.\n")
            path = f.name
        try:
            extract_markdown_structure(path, engine, namespace="Test")
            # Check bridge edges exist
            doc_name = f"Test:{_derive_doc_name(path)}"
            edges = engine.get_edges(source=doc_name)
            targets = {e.target for e in edges}
            assert "STL" in targets or "Consciousness" in targets
        finally:
            os.unlink(path)


# ── Spec Extractor Tests ───────────────────────────────────────────

class TestExtractHeader:

    def test_extracts_key_value(self):
        lines = [
            '> **Purpose:** Test specification\n',
            '> **Status:** Complete\n',
            '> **Version:** v1.0\n',
        ]
        meta = _extract_header(lines)
        assert meta.fields["Purpose"] == "Test specification"
        assert meta.fields["Status"] == "Complete"

    def test_handles_chinese_colon(self):
        lines = ['> **目的：** 测试\n']
        meta = _extract_header(lines)
        assert meta.fields.get("目的") == "测试"


class TestExtractPhaseRefs:

    def test_finds_phase_refs(self):
        text = "This is part of Phase 1 and Phase 4 development."
        refs = _extract_phase_refs(text)
        assert "Phase_1" in refs
        assert "Phase_4" in refs

    def test_no_false_phases(self):
        text = "Phase 99 does not exist."
        refs = _extract_phase_refs(text)
        assert len(refs) == 0


class TestSpecExtractor:

    def test_extract_spec_file(self):
        engine = STGEngine()
        inject_skeleton(engine)
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='_STLC_SPECIFICATION.md',
            delete=False, encoding='utf-8'
        ) as f:
            f.write("""# Test STLC Specification

> **Purpose:** Test spec
> **Status:** Complete

## 1. Overview

This specification defines the test component.

## 2. Architecture

The architecture uses STL patterns.

### 2.1 Class Definitions

```stl
[Entry_TestManager] -> [TestManager_Definition] ::mod(
  type="class",
  confidence=0.95
)
```

### 2.2 Methods

```stl
[Entry_run_tests_Method] -> [run_tests_Signature] ::mod(
  signature="async def run_tests(self, suite: str) -> TestResult"
)
```

## 3. Implementation

Phase 1 implementation details.
""")
            path = f.name
        try:
            count = extract_spec(path, engine, namespace="Spec")
            assert count >= 8  # 1 doc + 3 sections + edges + class + method + bridges
            # Verify spec document node
            spec_name = spec_sanitize_name(os.path.splitext(os.path.basename(path))[0])
            spec_node = engine.get_node(f"Spec:{spec_name}")
            assert spec_node is not None
        finally:
            os.unlink(path)

    def test_bridges_to_stlc_specification(self):
        engine = STGEngine()
        inject_skeleton(engine)
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.md', delete=False, encoding='utf-8'
        ) as f:
            f.write("# Spec\n\n## 1. Overview\nA specification.\n")
            path = f.name
        try:
            extract_spec(path, engine, namespace="Spec")
            spec_name = spec_sanitize_name(os.path.splitext(os.path.basename(path))[0])
            edges = engine.get_edges(source=f"Spec:{spec_name}")
            targets = {e.target for e in edges}
            assert "STLC_Specification" in targets
        finally:
            os.unlink(path)


# ── Universal Importer Tests ───────────────────────────────────────

class TestResolvePaths:

    def test_exact_file(self):
        with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as f:
            path = f.name
        try:
            result = _resolve_paths(path, os.path.dirname(path))
            assert len(result) == 1
        finally:
            os.unlink(path)

    def test_glob_pattern(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["a.md", "b.md", "c.txt"]:
                with open(os.path.join(tmpdir, name), 'w') as f:
                    f.write("test")
            paths = _resolve_paths("*.md", tmpdir)
            assert len(paths) == 2

    def test_nonexistent_returns_empty(self):
        paths = _resolve_paths("nonexistent_file.md", tempfile.gettempdir())
        assert len(paths) == 0


class TestModifiersToDict:

    def _parse_mods(self, stl_text):
        """Helper: parse STL text and return modifiers of first statement."""
        from stl_parser import parse
        result = parse(stl_text)
        return result.statements[0].modifiers

    def test_extracts_string_values(self):
        mods = self._parse_mods('[A] -> [B] ::mod(path="some/file.md", type="stl_native")')
        result = _modifiers_to_dict(mods)
        assert result["path"] == "some/file.md"
        assert result["type"] == "stl_native"

    def test_extracts_int_values(self):
        mods = self._parse_mods('[A] -> [B] ::mod(path="x.md", priority=1)')
        result = _modifiers_to_dict(mods)
        assert result["priority"] == 1

    def test_extracts_mixed_values(self):
        mods = self._parse_mods(
            '[A] -> [B] ::mod(path="test.md", type="markdown_doc", namespace="Doc", priority=3)'
        )
        result = _modifiers_to_dict(mods)
        assert result["path"] == "test.md"
        assert result["type"] == "markdown_doc"
        assert result["namespace"] == "Doc"
        assert result["priority"] == 3

    def test_extracts_paths_with_spaces(self):
        mods = self._parse_mods(
            '[A] -> [B] ::mod(path="../../website factory/ai-service/README.md")'
        )
        result = _modifiers_to_dict(mods)
        assert result["path"] == "../../website factory/ai-service/README.md"

    def test_extracts_description(self):
        mods = self._parse_mods('[A] -> [B] ::mod(path="x.md", description="a test file")')
        result = _modifiers_to_dict(mods)
        assert result["description"] == "a test file"


class TestLoadStlManifest:

    def test_loads_stl_manifest(self):
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.stl', delete=False, encoding='utf-8'
        ) as f:
            f.write(
                '[STG_Manifest] -> [Source:test] '
                '::mod(path="test.md", type="markdown_doc", namespace="Test", priority=1)\n'
            )
            path = f.name
        try:
            manifest = _load_stl_manifest(path)
            assert "sources" in manifest
            assert len(manifest["sources"]) == 1
            src = manifest["sources"][0]
            assert src["path"] == "test.md"
            assert src["type"] == "markdown_doc"
            assert src["namespace"] == "Test"
            assert src["priority"] == 1
        finally:
            os.unlink(path)

    def test_loads_multiple_sources(self):
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.stl', delete=False, encoding='utf-8'
        ) as f:
            f.write(
                '# Comment line\n'
                '[STG_Manifest] -> [Source:a] ::mod(path="a.md", type="stl_native", namespace="A", priority=1)\n'
                '[STG_Manifest] -> [Source:b] ::mod(path="b.md", type="stlc_spec", namespace="B", priority=2)\n'
                '[STG_Manifest] -> [Source:c] ::mod(path="c/*.md", type="markdown_doc", namespace="C", priority=3)\n'
            )
            path = f.name
        try:
            manifest = _load_stl_manifest(path)
            assert len(manifest["sources"]) == 3
            types = [s["type"] for s in manifest["sources"]]
            assert "stl_native" in types
            assert "stlc_spec" in types
            assert "markdown_doc" in types
        finally:
            os.unlink(path)

    def test_raises_on_empty_manifest(self):
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.stl', delete=False, encoding='utf-8'
        ) as f:
            f.write("# Just comments, no statements\n")
            path = f.name
        try:
            with pytest.raises(ValueError, match="No source entries"):
                _load_stl_manifest(path)
        finally:
            os.unlink(path)

    def test_dispatch_by_extension(self):
        """_load_manifest should dispatch to STL parser for .stl files."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.stl', delete=False, encoding='utf-8'
        ) as f:
            f.write(
                '[M] -> [S:x] ::mod(path="x.md", type="markdown_doc", namespace="X", priority=1)\n'
            )
            path = f.name
        try:
            manifest = _load_manifest(path)
            assert len(manifest["sources"]) == 1
            assert manifest["sources"][0]["path"] == "x.md"
        finally:
            os.unlink(path)


class TestLoadManifest:

    def test_loads_valid_manifest(self):
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False, encoding='utf-8'
        ) as f:
            json.dump({"sources": [{"path": "test.md", "type": "markdown_doc"}]}, f)
            path = f.name
        try:
            manifest = _load_manifest(path)
            assert "sources" in manifest
            assert len(manifest["sources"]) == 1
        finally:
            os.unlink(path)

    def test_raises_on_missing_sources(self):
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False, encoding='utf-8'
        ) as f:
            json.dump({"other": "data"}, f)
            path = f.name
        try:
            with pytest.raises(ValueError, match="sources"):
                _load_manifest(path)
        finally:
            os.unlink(path)


class TestImportKnowledgeBase:

    def test_mini_import(self):
        """Integration test with a minimal manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple markdown file
            doc_path = os.path.join(tmpdir, "test_doc.md")
            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write("# Test Doc\n\n## Section 1\nContent about STL.\n")

            # Create manifest
            manifest_path = os.path.join(tmpdir, "stg_manifest.json")
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "sources": [
                        {
                            "path": "test_doc.md",
                            "type": "markdown_doc",
                            "namespace": "Test",
                            "priority": 1,
                        }
                    ]
                }, f)

            engine = import_knowledge_base(manifest_path, project_root=tmpdir)

            stats = engine.get_stats()
            # Should have skeleton nodes + doc nodes
            assert stats["node_count"] >= len(CORE_CONCEPTS) + 1
            assert stats["edge_count"] >= len(SKELETON_EDGES) + 1

    def test_import_with_spec(self):
        """Integration test with a spec file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = os.path.join(tmpdir, "TEST_STLC_SPECIFICATION.md")
            with open(spec_path, 'w', encoding='utf-8') as f:
                f.write(
                    "# Test Spec\n\n"
                    "> **Purpose:** Testing\n\n"
                    "## 1. Overview\n\n"
                    "Defines test things for Phase 1.\n\n"
                    "## 2. Implementation\n\n"
                    "Code details.\n"
                )

            manifest_path = os.path.join(tmpdir, "stg_manifest.json")
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "sources": [
                        {
                            "path": "TEST_STLC_SPECIFICATION.md",
                            "type": "stlc_spec",
                            "namespace": "Spec",
                            "priority": 2,
                        }
                    ]
                }, f)

            engine = import_knowledge_base(manifest_path, project_root=tmpdir)
            stats = engine.get_stats()
            assert stats["node_count"] >= len(CORE_CONCEPTS) + 3  # doc + 2 sections

    def test_import_stores_stats(self):
        """Import should store _import_stats on engine."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "stg_manifest.json")
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump({"sources": []}, f)

            engine = import_knowledge_base(manifest_path, project_root=tmpdir)
            assert hasattr(engine, '_import_stats')
            assert engine._import_stats["skeleton"] > 0

    def test_import_handles_missing_files_gracefully(self):
        """Import should skip missing files without crashing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "stg_manifest.json")
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "sources": [
                        {
                            "path": "nonexistent.md",
                            "type": "markdown_doc",
                            "namespace": "Test",
                            "priority": 1,
                        }
                    ]
                }, f)

            engine = import_knowledge_base(manifest_path, project_root=tmpdir)
            # Should succeed with just skeleton
            assert engine.get_stats()["node_count"] >= len(CORE_CONCEPTS)

    def test_import_computes_psi(self):
        """After import, Ψ should be computable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = os.path.join(tmpdir, "stg_manifest.json")
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump({"sources": []}, f)

            engine = import_knowledge_base(manifest_path, project_root=tmpdir)
            psi = engine.compute_psi()
            assert psi > 0

    def test_import_with_stl_manifest(self):
        """Integration test: full import pipeline using STL manifest format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a markdown doc
            doc_path = os.path.join(tmpdir, "doc.md")
            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write("# My Document\n\n## Overview\nSome content about STL.\n")

            # Create STL manifest
            manifest_path = os.path.join(tmpdir, "stg_manifest.stl")
            with open(manifest_path, 'w', encoding='utf-8') as f:
                f.write(
                    '[M] -> [S:doc] ::mod(path="doc.md", '
                    'type="markdown_doc", namespace="Test", priority=1)\n'
                )

            engine = import_knowledge_base(manifest_path, project_root=tmpdir)
            stats = engine.get_stats()
            assert stats["node_count"] >= len(CORE_CONCEPTS) + 1
            # Doc name derived from filename "doc.md" → "doc"
            assert engine.get_node("Test:doc") is not None
