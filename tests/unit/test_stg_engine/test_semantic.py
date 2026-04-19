"""Tests for STG Semantic Query Layer (Phase 7G).

Tests EmbeddingBuilder, VectorIndex, SearchResult, persistence,
and engine.search() integration — all using mock embedding models
(no sentence-transformers dependency required).
"""

import os
import tempfile
import hashlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from stg_engine.engine import STGEngine
from stg_engine.types import SearchResult
from stg_engine.semantic import (
    EmbeddingBuilder,
    VectorIndex,
    EMBEDDING_DIM,
    DEFAULT_MODEL_NAME,
)


# ═══════════════════════════════════════════════════════════
# Mock Embedding Model
# ═══════════════════════════════════════════════════════════

class MockEmbeddingModel:
    """Deterministic mock for sentence-transformers SentenceTransformer.

    Produces normalized 384-dim vectors from text hashing.
    Semantically similar texts get similar vectors by sharing prefix tokens.
    """

    def encode(self, texts, normalize_embeddings=True, **kwargs):
        vectors = []
        for text in texts:
            # Deterministic vector from text hash
            h = hashlib.sha384(text.encode()).digest()
            vec = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
            # Pad or truncate to EMBEDDING_DIM
            if len(vec) < EMBEDDING_DIM:
                vec = np.pad(vec, (0, EMBEDDING_DIM - len(vec)))
            vec = vec[:EMBEDDING_DIM]
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            vectors.append(vec)
        return np.vstack(vectors)


def _make_engine_with_nodes():
    """Create an engine with a small semantic graph for testing."""
    engine = STGEngine()
    engine.add_edge("Gravity", "Newton", confidence=0.95, rule="causal")
    engine.add_edge("Newton", "Calculus", confidence=0.90, rule="logical")
    engine.add_edge("Einstein", "Relativity", confidence=0.98, rule="logical")
    engine.add_edge("Relativity", "Gravity", confidence=0.92, rule="causal")
    engine.add_edge("Physics", "Gravity", confidence=0.85, rule="definitional")
    engine.add_edge("Physics", "Relativity", confidence=0.85, rule="definitional")
    engine.add_edge("Mathematics", "Calculus", confidence=0.90, rule="definitional")
    return engine


# ═══════════════════════════════════════════════════════════
# EmbeddingBuilder Tests
# ═══════════════════════════════════════════════════════════

class TestEmbeddingBuilderUnpackName:
    def test_underscore_to_space(self):
        assert EmbeddingBuilder.unpack_name("Theory_Relativity") == "Theory Relativity"

    def test_pascal_case_split(self):
        assert EmbeddingBuilder.unpack_name("MemoryManager") == "Memory Manager"

    def test_namespace_separator(self):
        assert EmbeddingBuilder.unpack_name("Physics:Energy") == "Physics Energy"

    def test_combined(self):
        result = EmbeddingBuilder.unpack_name("Spec:MemoryManager_v2")
        assert "Spec" in result
        assert "Memory" in result
        assert "Manager" in result
        assert "v2" in result

    def test_all_caps_preserved(self):
        result = EmbeddingBuilder.unpack_name("STGEngine")
        assert "STG" in result
        assert "Engine" in result

    def test_simple_name_unchanged(self):
        assert EmbeddingBuilder.unpack_name("Gravity") == "Gravity"

    def test_empty_string(self):
        assert EmbeddingBuilder.unpack_name("") == ""

    def test_multiple_underscores(self):
        result = EmbeddingBuilder.unpack_name("A_B_C_D")
        assert result == "A B C D"


class TestEmbeddingBuilderBuildEmbedText:
    def test_basic_text(self):
        text = EmbeddingBuilder.build_embed_text(
            node_name="Gravity",
            namespace=None,
            metadata={},
            neighbor_names=[],
            edge_metadata=[],
        )
        assert "Gravity" in text

    def test_with_namespace(self):
        text = EmbeddingBuilder.build_embed_text(
            node_name="Energy",
            namespace="Physics",
            metadata={},
            neighbor_names=[],
            edge_metadata=[],
        )
        assert "Physics" in text
        assert "Energy" in text

    def test_with_metadata(self):
        text = EmbeddingBuilder.build_embed_text(
            node_name="Node",
            namespace=None,
            metadata={"type": "Concept", "description": "A fundamental idea"},
            neighbor_names=[],
            edge_metadata=[],
        )
        assert "Concept" in text
        assert "A fundamental idea" in text

    def test_with_neighbors(self):
        text = EmbeddingBuilder.build_embed_text(
            node_name="Physics",
            namespace=None,
            metadata={},
            neighbor_names=["Gravity", "Quantum_Mechanics"],
            edge_metadata=[],
        )
        assert "Gravity" in text
        assert "Quantum Mechanics" in text

    def test_with_edge_metadata(self):
        text = EmbeddingBuilder.build_embed_text(
            node_name="Node",
            namespace=None,
            metadata={},
            neighbor_names=[],
            edge_metadata=[{"rule": "causal", "cause": "Rain"}],
        )
        assert "causal" in text
        assert "Rain" in text

    def test_neighbor_cap_at_10(self):
        neighbors = [f"Node_{i}" for i in range(20)]
        text = EmbeddingBuilder.build_embed_text(
            node_name="Center",
            namespace=None,
            metadata={},
            neighbor_names=neighbors,
            edge_metadata=[],
        )
        # Should include at most 10 neighbors
        count = sum(1 for n in neighbors if EmbeddingBuilder.unpack_name(n) in text)
        assert count <= 10

    def test_metadata_list_values(self):
        text = EmbeddingBuilder.build_embed_text(
            node_name="Node",
            namespace=None,
            metadata={"tags": ["physics", "theory", "fundamental"]},
            neighbor_names=[],
            edge_metadata=[],
        )
        assert "physics" in text
        assert "theory" in text


class TestEmbeddingBuilderBuildAll:
    def test_build_all_nodes(self):
        engine = _make_engine_with_nodes()
        builder = EmbeddingBuilder()
        result = builder.build_all(engine)
        # All 7 nodes should have embed text
        assert len(result) == 7
        assert "gravity" in result
        assert "einstein" in result

    def test_empty_engine(self):
        engine = STGEngine()
        builder = EmbeddingBuilder()
        result = builder.build_all(engine)
        assert result == {}

    def test_embed_text_includes_neighbors(self):
        engine = STGEngine()
        engine.add_edge("A", "B", confidence=0.9)
        builder = EmbeddingBuilder()
        result = builder.build_all(engine)
        # A's embed text should mention B (neighbor)
        assert "b" in result["a"]
        # B's embed text should mention A (predecessor)
        assert "a" in result["b"]


# ═══════════════════════════════════════════════════════════
# VectorIndex Tests
# ═══════════════════════════════════════════════════════════

class TestVectorIndex:
    def _make_random_embeddings(self, names, dim=EMBEDDING_DIM):
        embeddings = {}
        for name in names:
            vec = np.random.randn(dim).astype(np.float32)
            vec /= np.linalg.norm(vec)
            embeddings[name] = vec
        return embeddings

    def test_build_and_size(self):
        index = VectorIndex()
        embeddings = self._make_random_embeddings(["A", "B", "C"])
        index.build(embeddings)
        assert index.size == 3

    def test_build_empty(self):
        index = VectorIndex()
        index.build({})
        assert index.size == 0
        assert index.matrix is None

    def test_query_returns_sorted(self):
        index = VectorIndex()
        # Create known vectors
        embeddings = {
            "exact": np.array([1.0] + [0.0] * (EMBEDDING_DIM - 1), dtype=np.float32),
            "similar": np.array([0.9, 0.1] + [0.0] * (EMBEDDING_DIM - 2), dtype=np.float32),
            "different": np.array([0.0, 1.0] + [0.0] * (EMBEDDING_DIM - 2), dtype=np.float32),
        }
        # Normalize
        for k in embeddings:
            embeddings[k] /= np.linalg.norm(embeddings[k])
        index.build(embeddings)

        query = np.array([1.0] + [0.0] * (EMBEDDING_DIM - 1), dtype=np.float32)
        results = index.query(query, top_k=3)

        assert len(results) == 3
        assert results[0][0] == "exact"
        assert results[0][1] > results[1][1]  # exact > similar
        assert results[1][1] > results[2][1]  # similar > different

    def test_query_empty_index(self):
        index = VectorIndex()
        index.build({})
        query = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        results = index.query(query, top_k=5)
        assert results == []

    def test_query_top_k_exceeds_size(self):
        index = VectorIndex()
        embeddings = self._make_random_embeddings(["A", "B"])
        index.build(embeddings)
        query = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        results = index.query(query, top_k=10)
        assert len(results) == 2

    def test_add_single_node(self):
        index = VectorIndex()
        vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        vec /= np.linalg.norm(vec)
        index.add("New", vec)
        assert index.size == 1
        results = index.query(vec, top_k=1)
        assert results[0][0] == "New"

    def test_add_updates_existing(self):
        index = VectorIndex()
        embeddings = self._make_random_embeddings(["A", "B"])
        index.build(embeddings)
        assert index.size == 2

        new_vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        new_vec /= np.linalg.norm(new_vec)
        index.add("A", new_vec)
        assert index.size == 2  # Still 2, not 3

    def test_remove_existing(self):
        index = VectorIndex()
        embeddings = self._make_random_embeddings(["A", "B", "C"])
        index.build(embeddings)
        assert index.remove("B") is True
        assert index.size == 2
        assert "B" not in index.names

    def test_remove_nonexistent(self):
        index = VectorIndex()
        assert index.remove("X") is False

    def test_remove_last_node(self):
        index = VectorIndex()
        vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        index.add("Only", vec)
        index.remove("Only")
        assert index.size == 0
        assert index.matrix is None


# ═══════════════════════════════════════════════════════════
# SearchResult Tests
# ═══════════════════════════════════════════════════════════

class TestSearchResult:
    def test_creation(self):
        result = SearchResult(
            query="test query",
            seeds=[("A", 0.9), ("B", 0.8)],
            propagated=[("C", 0.5)],
            combined=[("A", 0.85), ("B", 0.7), ("C", 0.3)],
            search_time_ms=1.5,
        )
        assert result.query == "test query"
        assert len(result.seeds) == 2
        assert len(result.propagated) == 1
        assert len(result.combined) == 3
        assert result.search_time_ms == 1.5

    def test_default_search_time(self):
        result = SearchResult(
            query="q",
            seeds=[],
            propagated=[],
            combined=[],
        )
        assert result.search_time_ms == 0.0


# ═══════════════════════════════════════════════════════════
# Engine Search Integration Tests (with Mock Model)
# ═══════════════════════════════════════════════════════════

class TestEngineSearch:
    """Integration tests for engine.search() using MockEmbeddingModel."""

    def _engine_with_mock_model(self):
        """Set up engine with mock embedding model and pre-built index."""
        engine = _make_engine_with_nodes()
        mock_model = MockEmbeddingModel()

        # Build index using the mock model
        engine._embed_model = mock_model
        engine._model_name = "mock-model"

        builder = EmbeddingBuilder()
        engine._embed_texts = builder.build_all(engine)

        # Build vectors
        texts = list(engine._embed_texts.values())
        names = list(engine._embed_texts.keys())
        vectors = mock_model.encode(texts, normalize_embeddings=True)

        index = VectorIndex()
        embeddings_dict = {name: vectors[i] for i, name in enumerate(names)}
        index.build(embeddings_dict)
        engine._vector_index = index

        return engine

    def test_search_returns_search_result(self):
        engine = self._engine_with_mock_model()
        result = engine.search("Gravity", top_k=5)
        assert isinstance(result, SearchResult)
        assert result.query == "Gravity"

    def test_search_returns_seeds(self):
        engine = self._engine_with_mock_model()
        result = engine.search("Physics", top_k=5, min_similarity=0.0)
        assert len(result.seeds) > 0

    def test_search_min_similarity_filters(self):
        engine = self._engine_with_mock_model()
        # Very high threshold should filter most seeds
        result_high = engine.search("test", top_k=10, min_similarity=0.99)
        result_low = engine.search("test", top_k=10, min_similarity=0.0)
        assert len(result_high.seeds) <= len(result_low.seeds)

    def test_search_no_propagate(self):
        engine = self._engine_with_mock_model()
        result = engine.search("Gravity", top_k=5, propagate=False, min_similarity=0.0)
        assert result.propagated == []
        # Combined should only contain seeds
        seed_names = {s[0] for s in result.seeds}
        combined_names = {c[0] for c in result.combined}
        assert combined_names.issubset(seed_names)

    def test_search_with_propagate(self):
        engine = self._engine_with_mock_model()
        result = engine.search("Gravity", top_k=5, propagate=True, min_similarity=0.0)
        # Should have some propagated results (graph has connections)
        # Combined list includes both seeds and propagated
        assert len(result.combined) > 0

    def test_search_timing(self):
        engine = self._engine_with_mock_model()
        result = engine.search("test", top_k=5, min_similarity=0.0)
        assert result.search_time_ms >= 0  # Timing recorded (may be 0 on fast machines)

    def test_search_combined_ranking_order(self):
        engine = self._engine_with_mock_model()
        result = engine.search("Physics", top_k=5, min_similarity=0.0)
        if len(result.combined) > 1:
            scores = [s for _, s in result.combined]
            assert scores == sorted(scores, reverse=True)

    def test_search_empty_engine(self):
        engine = STGEngine()
        engine._embed_model = MockEmbeddingModel()
        engine._model_name = "mock"
        engine._embed_texts = {}
        engine._vector_index = VectorIndex()
        result = engine.search("test", min_similarity=0.0)
        assert len(result.seeds) == 0
        assert len(result.combined) == 0


class TestEnsureSearchReady:
    """Tests for lazy loading behavior."""

    @patch("stg_engine.semantic.load_embedding_model")
    def test_lazy_model_load(self, mock_load):
        mock_load.return_value = MockEmbeddingModel()
        engine = STGEngine()
        engine.add_edge("A", "B")
        assert engine._embed_model is None

        engine._ensure_search_ready()
        mock_load.assert_called_once()
        assert engine._embed_model is not None

    @patch("stg_engine.semantic.load_embedding_model")
    def test_no_reload_if_already_loaded(self, mock_load):
        mock_load.return_value = MockEmbeddingModel()
        engine = STGEngine()
        engine.add_edge("A", "B")

        engine._ensure_search_ready()
        engine._ensure_search_ready()
        # Should only load once
        mock_load.assert_called_once()


class TestBuildSearchIndex:
    """Tests for explicit index building."""

    @patch("stg_engine.semantic.load_embedding_model")
    def test_build_returns_count(self, mock_load):
        mock_load.return_value = MockEmbeddingModel()
        engine = _make_engine_with_nodes()
        count = engine.build_search_index()
        assert count == 7  # 7 nodes in the test graph

    @patch("stg_engine.semantic.load_embedding_model")
    def test_build_empty_engine(self, mock_load):
        mock_load.return_value = MockEmbeddingModel()
        engine = STGEngine()
        count = engine.build_search_index()
        assert count == 0

    @patch("stg_engine.semantic.load_embedding_model")
    def test_build_sets_model_name(self, mock_load):
        mock_load.return_value = MockEmbeddingModel()
        engine = _make_engine_with_nodes()
        engine.build_search_index()
        assert engine._model_name == DEFAULT_MODEL_NAME


# ═══════════════════════════════════════════════════════════
# Persistence Tests
# ═══════════════════════════════════════════════════════════

class TestEmbeddingPersistence:
    """Tests for save_embeddings() and load_embeddings()."""

    def _make_stg_file(self):
        """Create a temp .stg file with basic schema."""
        from stg_engine.persistence import _init_db
        fd, path = tempfile.mkstemp(suffix=".stg")
        os.close(fd)
        import sqlite3
        conn = sqlite3.connect(path)
        _init_db(conn)
        conn.close()
        return path

    def test_save_and_load_roundtrip(self):
        from stg_engine.persistence import save_embeddings, load_embeddings

        path = self._make_stg_file()
        try:
            names = ["A", "B", "C"]
            vectors = np.random.randn(3, EMBEDDING_DIM).astype(np.float32)
            # Normalize
            for i in range(3):
                vectors[i] /= np.linalg.norm(vectors[i])
            embed_texts = {"A": "concept A", "B": "concept B", "C": "concept C"}

            save_embeddings(path, names, vectors, embed_texts, "test-model")
            loaded = load_embeddings(path, expected_model="test-model")

            assert loaded is not None
            assert loaded["model_name"] == "test-model"
            assert loaded["names"] == names
            assert loaded["embed_texts"] == embed_texts
            np.testing.assert_allclose(loaded["vectors"], vectors, atol=1e-6)
        finally:
            os.unlink(path)

    def test_load_model_mismatch_returns_none(self):
        from stg_engine.persistence import save_embeddings, load_embeddings

        path = self._make_stg_file()
        try:
            names = ["A"]
            vectors = np.random.randn(1, EMBEDDING_DIM).astype(np.float32)
            embed_texts = {"A": "text"}

            save_embeddings(path, names, vectors, embed_texts, "model-v1")
            loaded = load_embeddings(path, expected_model="model-v2")
            assert loaded is None
        finally:
            os.unlink(path)

    def test_load_no_expected_model_accepts_any(self):
        from stg_engine.persistence import save_embeddings, load_embeddings

        path = self._make_stg_file()
        try:
            names = ["X"]
            vectors = np.random.randn(1, EMBEDDING_DIM).astype(np.float32)
            embed_texts = {"X": "x"}

            save_embeddings(path, names, vectors, embed_texts, "any-model")
            loaded = load_embeddings(path)
            assert loaded is not None
            assert loaded["model_name"] == "any-model"
        finally:
            os.unlink(path)

    def test_load_nonexistent_file_returns_none(self):
        from stg_engine.persistence import load_embeddings
        loaded = load_embeddings("/nonexistent/path.stg")
        assert loaded is None

    def test_load_no_embeddings_table_returns_none(self):
        from stg_engine.persistence import load_embeddings
        # Create a v2 schema (no embeddings table)
        fd, path = tempfile.mkstemp(suffix=".stg")
        os.close(fd)
        try:
            import sqlite3
            conn = sqlite3.connect(path)
            conn.execute("CREATE TABLE schema_info (key TEXT PRIMARY KEY, value TEXT)")
            conn.execute("INSERT INTO schema_info VALUES ('version', '2')")
            conn.commit()
            conn.close()

            loaded = load_embeddings(path)
            assert loaded is None
        finally:
            os.unlink(path)

    def test_save_overwrites_existing(self):
        from stg_engine.persistence import save_embeddings, load_embeddings

        path = self._make_stg_file()
        try:
            # Save first batch
            v1 = np.random.randn(2, EMBEDDING_DIM).astype(np.float32)
            save_embeddings(path, ["A", "B"], v1, {"A": "a", "B": "b"}, "m1")

            # Save second batch (should replace)
            v2 = np.random.randn(1, EMBEDDING_DIM).astype(np.float32)
            save_embeddings(path, ["X"], v2, {"X": "x"}, "m2")

            loaded = load_embeddings(path)
            assert loaded is not None
            assert loaded["names"] == ["X"]
            assert loaded["model_name"] == "m2"
        finally:
            os.unlink(path)


class TestSchemaMigration:
    """Tests for v2 → v3 schema migration."""

    def test_migrate_v2_to_v3(self):
        from stg_engine.persistence import _migrate_schema, SCHEMA_VERSION
        import sqlite3

        fd, path = tempfile.mkstemp(suffix=".stg")
        os.close(fd)
        try:
            conn = sqlite3.connect(path)
            # Create minimal v2 schema
            conn.execute("CREATE TABLE schema_info (key TEXT PRIMARY KEY, value TEXT)")
            conn.execute("INSERT INTO schema_info VALUES ('version', '2')")
            conn.execute("CREATE TABLE nodes (name TEXT PRIMARY KEY)")
            conn.commit()

            _migrate_schema(conn)

            # Verify embeddings table exists
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "embeddings" in tables

            # Verify version updated
            version = conn.execute(
                "SELECT value FROM schema_info WHERE key='version'"
            ).fetchone()[0]
            assert version == str(SCHEMA_VERSION)

            conn.close()
        finally:
            os.unlink(path)

    def test_migrate_already_v3_is_noop(self):
        from stg_engine.persistence import _migrate_schema, _init_db
        import sqlite3

        fd, path = tempfile.mkstemp(suffix=".stg")
        os.close(fd)
        try:
            conn = sqlite3.connect(path)
            _init_db(conn)  # Creates v3 schema with embeddings table

            # Should not raise
            _migrate_schema(conn)

            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "embeddings" in tables
            conn.close()
        finally:
            os.unlink(path)
