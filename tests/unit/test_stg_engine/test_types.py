"""Tests for STG Engine types."""

import pytest
from stg_engine.types import (
    STGNode, STGEdge, STGSession, STGEvent,
    STGTension, STGBeliefEvolution, SystemSnapshot,
)


class TestSTGNode:
    def test_basic_creation(self):
        node = STGNode(name="TestConcept")
        assert node.name == "TestConcept"
        assert node.namespace is None
        assert node.anchor_type is None
        assert node.tension == 0.0
        assert node.activation == 0.0
        assert node.self_relevance == 0.0

    def test_full_creation(self):
        node = STGNode(
            name="Energy",
            namespace="Physics",
            anchor_type="Concept",
            metadata={"domain": "physics"},
            tension=0.5,
            activation=0.8,
        )
        assert node.name == "Energy"
        assert node.namespace == "Physics"
        assert node.qualified_name == "Physics:Energy"

    def test_qualified_name_no_namespace(self):
        node = STGNode(name="Simple")
        assert node.qualified_name == "Simple"

    def test_serialization_roundtrip(self):
        node = STGNode(
            name="Test",
            namespace="NS",
            anchor_type="Event",
            metadata={"key": "value"},
            tension=0.3,
        )
        d = node.to_dict()
        restored = STGNode.from_dict(d)
        assert restored.name == "Test"
        assert restored.namespace == "NS"
        assert restored.anchor_type == "Event"
        assert restored.metadata == {"key": "value"}
        assert restored.tension == 0.3


class TestSTGEdge:
    def test_basic_creation(self):
        edge = STGEdge(source="A", target="B")
        assert edge.source == "A"
        assert edge.target == "B"
        assert edge.confidence == 0.5
        assert edge.strength == 0.5

    def test_uncertainty(self):
        edge = STGEdge(source="A", target="B", confidence=0.9)
        assert abs(edge.uncertainty - 0.1) < 1e-6

    def test_full_creation(self):
        edge = STGEdge(
            source="Theory",
            target="Prediction",
            confidence=0.95,
            strength=0.8,
            rule="logical",
            time="2025-01-15",
            modifiers={"author": "Einstein"},
            session_id="SESSION_001",
            event_id="E001",
        )
        assert edge.rule == "logical"
        assert edge.modifiers["author"] == "Einstein"

    def test_serialization_roundtrip(self):
        edge = STGEdge(
            source="X", target="Y",
            confidence=0.85, rule="causal",
            modifiers={"cause": "Rain"},
        )
        d = edge.to_dict()
        restored = STGEdge.from_dict(d)
        assert restored.source == "X"
        assert restored.confidence == 0.85
        assert restored.rule == "causal"
        assert restored.modifiers["cause"] == "Rain"


class TestSTGSession:
    def test_creation(self):
        s = STGSession(session_id="SESSION_020", date="2026-02-08", title="Test")
        assert s.session_id == "SESSION_020"
        assert s.status == "complete"

    def test_roundtrip(self):
        s = STGSession(session_id="S1", avg_importance=0.87)
        d = s.to_dict()
        restored = STGSession.from_dict(d)
        assert restored.session_id == "S1"
        assert restored.avg_importance == 0.87


class TestSTGEvent:
    def test_creation(self):
        e = STGEvent(
            event_id="E042",
            session_id="SESSION_019",
            importance_score=0.85,
            tags=["bug_fix", "pydantic"],
        )
        assert e.event_id == "E042"
        assert e.tags == ["bug_fix", "pydantic"]

    def test_roundtrip(self):
        e = STGEvent(event_id="E001", importance_score=0.95)
        d = e.to_dict()
        restored = STGEvent.from_dict(d)
        assert restored.event_id == "E001"
        assert restored.importance_score == 0.95


class TestSTGTension:
    def test_creation(self):
        t = STGTension(name="OAuth_API_Broken", initial_value=1.0, current_value=0.0, status="resolved")
        assert t.name == "OAuth_API_Broken"
        assert t.status == "resolved"


class TestSTGBeliefEvolution:
    def test_creation(self):
        be = STGBeliefEvolution(
            old_anchor="Memory_Architecture_v1",
            new_anchor="Memory_Architecture_v2",
            level=3,
            description="Static Archive → Living Memory",
        )
        assert be.level == 3
        assert "Living Memory" in be.description


class TestSystemSnapshot:
    def test_creation(self):
        ss = SystemSnapshot(psi_value=0.85, node_count=200, edge_count=600)
        assert ss.psi_value == 0.85
        assert ss.node_count == 200
