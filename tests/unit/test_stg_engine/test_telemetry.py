"""Tests for STG Telemetry system (Phase 10)."""

import os
import sqlite3
import tempfile
import time

import pytest

from stg_engine import STGEngine
from stg_engine.telemetry import (
    TelemetryCollector,
    telemetry_status,
    telemetry_frequency,
    telemetry_salience,
    telemetry_learning,
    telemetry_report,
    generate_calibrated_queries,
)
from stg_engine.types import LearningEvent, PropagationMetrics


# ─── Helpers ──────────────────────────────────────────────


def _make_engine_with_data():
    """Create a small engine with enough structure for propagation."""
    engine = STGEngine()
    engine.ingest_stl('[A] -> [B] ::mod(confidence=0.9)')
    engine.ingest_stl('[B] -> [C] ::mod(confidence=0.8)')
    engine.ingest_stl('[C] -> [D] ::mod(confidence=0.7)')
    engine.ingest_stl('[A] -> [D] ::mod(confidence=0.6)')
    engine.ingest_stl('[X] -> [Y] ::mod(confidence=0.5)')
    return engine


def _save_engine(engine):
    """Save engine to temp file, return path."""
    fd, path = tempfile.mkstemp(suffix=".stg")
    os.close(fd)
    engine.save(path)
    return path


# ─── TelemetryCollector Tests ─────────────────────────────


class TestTelemetryCollector:
    def test_init_defaults(self):
        tc = TelemetryCollector()
        assert tc.max_propagations == 500
        assert tc.max_sessions == 200
        assert tc.max_mutations == 2000
        assert tc.mutation_threshold == 0.01

    def test_record_propagation(self):
        tc = TelemetryCollector()
        metrics = PropagationMetrics(
            input_text="test query",
            token_count=2,
            seed_node_count=1,
            activated_node_count=3,
            total_activation=1.5,
            max_activation=0.8,
            iterations_used=2,
            query_efficiency=0.5,
            resonance_score=0.6,
            coverage=0.3,
            top_nodes=[("A", 0.8), ("B", 0.5), ("C", 0.2)],
        )
        activation_map = {"A": 0.8, "B": 0.5, "C": 0.2, "D": 0.001}

        tc.record_propagation(metrics, activation_map, 2, 1)

        assert len(tc._propagations) == 1
        assert tc._propagations[0]["seed_count"] == 1
        assert tc._propagations[0]["strengthen_count"] == 2
        assert tc._propagations[0]["weaken_count"] == 1

        # Node freq should track A, B, C but not D (below threshold)
        assert tc._node_freq["A"] == 1
        assert tc._node_freq["B"] == 1
        assert tc._node_freq["C"] == 1
        assert tc._node_freq.get("D", 0) == 0  # below 0.01 threshold

    def test_record_edge_mutations(self):
        tc = TelemetryCollector(mutation_threshold=0.01)
        now = time.time()

        events = [
            LearningEvent(
                event_type="strengthen", source="A", target="B",
                old_confidence=0.5, new_confidence=0.55,
                timestamp=now, trigger="propagation",
            ),
            LearningEvent(
                event_type="weaken", source="C", target="D",
                old_confidence=0.5, new_confidence=0.49,
                timestamp=now, trigger="propagation",
            ),
            # Below threshold — should be filtered
            LearningEvent(
                event_type="strengthen", source="X", target="Y",
                old_confidence=0.5, new_confidence=0.505,
                timestamp=now, trigger="propagation",
            ),
        ]

        tc.record_edge_mutations(events)

        assert len(tc._edge_mutations) == 2
        assert tc._edge_mutations[0]["source"] == "A"
        assert tc._edge_mutations[1]["event_type"] == "weaken"

    def test_record_session_summary(self):
        engine = _make_engine_with_data()
        tc = TelemetryCollector()
        tc.record_session_summary(engine)

        assert tc._session_summary is not None
        assert tc._session_summary["node_count"] > 0
        assert tc._session_summary["edge_count"] > 0
        assert "salience_p50" in tc._session_summary

    def test_flush_and_read(self):
        engine = _make_engine_with_data()
        path = _save_engine(engine)

        try:
            tc = TelemetryCollector()

            # Record some data
            metrics = PropagationMetrics(
                input_text="flush test",
                token_count=2, seed_node_count=1,
                activated_node_count=2, total_activation=1.0,
                max_activation=0.7, iterations_used=2,
                query_efficiency=0.4, resonance_score=0.5,
                coverage=0.2,
                top_nodes=[("A", 0.7), ("B", 0.3)],
            )
            tc.record_propagation(metrics, {"A": 0.7, "B": 0.3}, 1, 0)
            tc.record_session_summary(engine)

            written = tc.flush(path)
            assert written > 0

            # Verify data in SQLite
            conn = sqlite3.connect(path)
            count = conn.execute(
                "SELECT COUNT(*) FROM telemetry_propagations"
            ).fetchone()[0]
            assert count == 1

            count = conn.execute(
                "SELECT COUNT(*) FROM telemetry_node_freq"
            ).fetchone()[0]
            assert count == 2  # A and B

            count = conn.execute(
                "SELECT COUNT(*) FROM telemetry_sessions"
            ).fetchone()[0]
            assert count == 1

            conn.close()
        finally:
            os.unlink(path)

    def test_flush_rolling_window(self):
        engine = _make_engine_with_data()
        path = _save_engine(engine)

        try:
            tc = TelemetryCollector(max_propagations=3)

            # Record 5 propagations
            for i in range(5):
                metrics = PropagationMetrics(
                    input_text=f"query {i}",
                    token_count=1, seed_node_count=1,
                    activated_node_count=1, total_activation=0.5,
                    max_activation=0.5, iterations_used=1,
                    query_efficiency=0.3, resonance_score=0.4,
                    coverage=0.1,
                    top_nodes=[("A", 0.5)],
                )
                tc.record_propagation(metrics, {"A": 0.5}, 0, 0)

            tc.flush(path)

            # Should be capped at 3
            conn = sqlite3.connect(path)
            count = conn.execute(
                "SELECT COUNT(*) FROM telemetry_propagations"
            ).fetchone()[0]
            assert count == 3
            conn.close()
        finally:
            os.unlink(path)

    def test_node_freq_upsert(self):
        """Node frequencies should accumulate across flushes."""
        engine = _make_engine_with_data()
        path = _save_engine(engine)

        try:
            for _ in range(2):
                tc = TelemetryCollector()
                metrics = PropagationMetrics(
                    input_text="upsert test",
                    token_count=1, seed_node_count=1,
                    activated_node_count=1, total_activation=0.5,
                    max_activation=0.5, iterations_used=1,
                    query_efficiency=0.3, resonance_score=0.4,
                    coverage=0.1,
                    top_nodes=[("A", 0.5)],
                )
                tc.record_propagation(metrics, {"A": 0.5}, 0, 0)
                tc.flush(path)

            conn = sqlite3.connect(path)
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT activation_count FROM telemetry_node_freq "
                "WHERE node_name = 'A'"
            ).fetchone()
            assert row["activation_count"] == 2  # accumulated
            conn.close()
        finally:
            os.unlink(path)

    def test_reset(self):
        tc = TelemetryCollector()
        tc._propagations.append({"test": True})
        tc._node_freq["A"] = 5
        tc._session_summary = {"test": True}

        tc.reset()

        assert len(tc._propagations) == 0
        assert len(tc._node_freq) == 0
        assert tc._session_summary is None


# ─── Analysis Function Tests ─────────────────────────────


class TestAnalysisFunctions:
    def _setup_with_data(self):
        """Create engine, save, populate telemetry, return path."""
        engine = _make_engine_with_data()
        path = _save_engine(engine)

        tc = TelemetryCollector()
        for q in ["query A", "query B"]:
            metrics = PropagationMetrics(
                input_text=q,
                token_count=2, seed_node_count=1,
                activated_node_count=3, total_activation=1.5,
                max_activation=0.8, iterations_used=2,
                query_efficiency=0.5, resonance_score=0.6,
                coverage=0.3,
                top_nodes=[("A", 0.8), ("B", 0.5), ("C", 0.2)],
            )
            tc.record_propagation(metrics, {"A": 0.8, "B": 0.5, "C": 0.2}, 1, 1)

        now = time.time()
        tc.record_edge_mutations([
            LearningEvent(
                event_type="strengthen", source="A", target="B",
                old_confidence=0.5, new_confidence=0.55,
                timestamp=now, trigger="propagation",
            ),
        ])
        tc.record_session_summary(engine)
        tc.flush(path)
        return path

    def test_telemetry_status(self):
        path = self._setup_with_data()
        try:
            status = telemetry_status(path)
            assert status["available"] is True
            assert status["telemetry_propagations_count"] == 2
            assert status["telemetry_node_freq_count"] == 3
            assert status["telemetry_sessions_count"] == 1
            assert status["telemetry_edge_mutations_count"] == 1
        finally:
            os.unlink(path)

    def test_telemetry_status_no_data(self):
        status = telemetry_status("/nonexistent/path.stg")
        assert status["available"] is False

    def test_telemetry_frequency(self):
        path = self._setup_with_data()
        try:
            freq = telemetry_frequency(path, top_n=5)
            assert freq["total_propagations"] == 2
            assert len(freq["nodes"]) == 3
            # A should be most activated (0.8 * 2 queries)
            assert freq["nodes"][0]["name"] == "A"
        finally:
            os.unlink(path)

    def test_telemetry_salience(self):
        path = self._setup_with_data()
        try:
            sal = telemetry_salience(path)
            assert len(sal["sessions"]) == 1
            assert "salience_p50" in sal["sessions"][0]
        finally:
            os.unlink(path)

    def test_telemetry_learning(self):
        path = self._setup_with_data()
        try:
            learn = telemetry_learning(path)
            assert "strengthen" in learn["summary"]
            assert learn["summary"]["strengthen"]["count"] == 1
        finally:
            os.unlink(path)

    def test_telemetry_report(self):
        path = self._setup_with_data()
        try:
            report = telemetry_report(path)
            assert "STG Telemetry Report" in report
            assert "Propagations: 2" in report
        finally:
            os.unlink(path)

    def test_generate_calibrated_queries(self):
        path = self._setup_with_data()
        try:
            queries = generate_calibrated_queries(path)
            assert len(queries) > 0
            assert all("text" in q and "expected" in q and "frequency" in q for q in queries)
        finally:
            os.unlink(path)


# ─── Engine Integration Tests ─────────────────────────────


class TestEngineIntegration:
    def test_enable_disable_telemetry(self):
        engine = _make_engine_with_data()
        assert not engine.telemetry_enabled

        engine.enable_telemetry()
        assert engine.telemetry_enabled

        engine.disable_telemetry()
        assert not engine.telemetry_enabled

    def test_propagate_records_telemetry(self):
        engine = _make_engine_with_data()
        engine.enable_telemetry()
        engine.enable_learning()

        engine.propagate("A B C")

        tc = engine._telemetry
        assert len(tc._propagations) == 1
        assert tc._propagations[0]["input_text"] == "A B C"
        assert len(tc._node_freq) > 0

    def test_propagate_without_telemetry(self):
        """Propagate should work normally without telemetry."""
        engine = _make_engine_with_data()
        result = engine.propagate("A B")
        assert isinstance(result, list)

    def test_telemetry_preserves_on_save(self):
        """Telemetry tables should survive engine save/load cycle."""
        engine = _make_engine_with_data()
        path = _save_engine(engine)

        try:
            # Write telemetry data
            tc = TelemetryCollector()
            metrics = PropagationMetrics(
                input_text="preserve test",
                token_count=1, seed_node_count=1,
                activated_node_count=1, total_activation=0.5,
                max_activation=0.5, iterations_used=1,
                query_efficiency=0.3, resonance_score=0.4,
                coverage=0.1,
                top_nodes=[("A", 0.5)],
            )
            tc.record_propagation(metrics, {"A": 0.5}, 0, 0)
            tc.flush(path)

            # Re-save engine (should preserve telemetry)
            engine2 = STGEngine.load(path)
            engine2.save(path)

            # Verify telemetry survived
            status = telemetry_status(path)
            assert status["telemetry_propagations_count"] == 1
            assert status["telemetry_node_freq_count"] == 1
        finally:
            os.unlink(path)


# ─── Schema Migration Test ────────────────────────────────


class TestSchemaMigration:
    def test_v6_to_v7_migration(self):
        """Engine with v6 schema should auto-migrate to v7 with telemetry tables."""
        engine = _make_engine_with_data()
        fd, path = tempfile.mkstemp(suffix=".stg")
        os.close(fd)

        try:
            # Save with current schema
            engine.save(path)

            # Manually remove telemetry tables to simulate v6
            conn = sqlite3.connect(path)
            for tbl in ["telemetry_propagations", "telemetry_node_freq",
                         "telemetry_sessions", "telemetry_edge_mutations"]:
                conn.execute(f"DROP TABLE IF EXISTS {tbl}")
            conn.execute(
                "UPDATE schema_info SET value = '6' WHERE key = 'version'"
            )
            conn.commit()
            conn.close()

            # Load should trigger migration
            engine2 = STGEngine.load(path)

            # Verify tables were created
            conn = sqlite3.connect(path)
            tables = {
                row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "telemetry_propagations" in tables
            assert "telemetry_node_freq" in tables
            assert "telemetry_sessions" in tables
            assert "telemetry_edge_mutations" in tables
            conn.close()
        finally:
            os.unlink(path)
