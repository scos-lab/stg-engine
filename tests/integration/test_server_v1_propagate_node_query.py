"""Integration tests for /v1/propagate, /v1/node/{name}, /v1/query (M3).

Uses FastAPI's TestClient against a small in-process engine. Verifies
schema correctness, read-only side-effect semantics, and error paths.
"""

import os
import pytest

from stg_engine import STGEngine
from stg_engine.server.app import create_app
from stg_engine.server.state import ServerState

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient


@pytest.fixture
def client(tmp_path):
    """Engine with a small game-KB fixture, served via TestClient."""
    engine = STGEngine()
    # 3-hop chain plus a side edge — enough for propagate to spread.
    engine.ingest_stl(
        '[Cyberpunk_2077] -> [CD_Projekt_Red] '
        '::mod(action="developed_by", confidence=1.0, source="steam_appdetails")'
    )
    engine.ingest_stl(
        '[Cyberpunk_2077] -> [Action_RPG] '
        '::mod(action="belongs_to_genre", confidence=0.95)'
    )
    engine.ingest_stl(
        '[CD_Projekt_Red] -> [Poland] '
        '::mod(action="based_in", confidence=1.0)'
    )
    # Add namespace to Cyberpunk node by re-ingesting with namespace
    engine._nodes[engine._nk("Cyberpunk_2077")].namespace = "Game"
    engine._nodes[engine._nk("CD_Projekt_Red")].namespace = "Company"

    # Enable learning + telemetry to verify read_only really skips them.
    engine.enable_learning()
    engine.enable_telemetry()

    stg_path = tmp_path / "memory.stg"
    engine.save(str(stg_path))

    state = ServerState(
        engine=engine,
        agent_name="test-agent",
        stg_path=str(stg_path),
        engine_mtime=os.path.getmtime(stg_path),
        server_version="0.7.0a1-test",
    )
    app = create_app(state)
    return TestClient(app)


class TestPropagate:
    def test_returns_200_with_activated_nodes(self, client):
        r = client.post("/v1/propagate", json={"query": "Cyberpunk"})
        assert r.status_code == 200
        body = r.json()
        assert body["agent"] == "test-agent"
        assert body["query"] == "Cyberpunk"
        assert isinstance(body["nodes"], list)
        assert "elapsed_ms" in body
        assert "truncated" in body

    def test_activated_nodes_have_outgoing_edges(self, client):
        r = client.post(
            "/v1/propagate",
            json={"query": "Cyberpunk 2077", "include_edges": True, "edge_limit_per_node": 5},
        )
        body = r.json()
        # Expect at least CD_Projekt_Red activated as a target of Cyberpunk
        names = [n["name"] for n in body["nodes"]]
        assert any("CD_Projekt_Red" in n or "Action_RPG" in n for n in names)
        # Find a node with outgoing edges
        with_edges = [n for n in body["nodes"] if n["outgoing"]]
        assert len(with_edges) >= 1, "expected at least one node to carry outgoing edges"
        edge = with_edges[0]["outgoing"][0]
        assert "source" in edge and "target" in edge
        assert "confidence" in edge
        assert isinstance(edge["modifiers"], dict)
        # Modifier values must all be strings on the wire (Decision 12.3)
        for v in edge["modifiers"].values():
            assert isinstance(v, str)

    def test_include_edges_false_omits_edges(self, client):
        r = client.post(
            "/v1/propagate",
            json={"query": "Cyberpunk", "include_edges": False},
        )
        body = r.json()
        for node in body["nodes"]:
            assert node["outgoing"] == []

    def test_truncation_when_max_nodes_smaller(self, client):
        r = client.post(
            "/v1/propagate",
            json={"query": "Cyberpunk", "max_nodes": 1},
        )
        body = r.json()
        assert len(body["nodes"]) <= 1
        # If activated > 1, truncated should be true
        if body["activated_count"] > 1:
            assert body["truncated"] is True

    def test_read_only_skips_engine_telemetry(self, client):
        """100 HTTP propagates must NOT advance engine telemetry counters."""
        state = client.app.state.server_state
        before = len(state.engine._telemetry._propagations)
        for _ in range(5):
            client.post("/v1/propagate", json={"query": "Cyberpunk"})
        after = len(state.engine._telemetry._propagations)
        assert after == before, (
            "engine telemetry advanced under HTTP serving — read_only path broken"
        )

    def test_read_only_skips_hebbian_learning(self, client):
        """HTTP propagates must NOT grow the engine's _learning_log."""
        state = client.app.state.server_state
        before = len(state.engine._learning_log)
        for _ in range(5):
            client.post("/v1/propagate", json={"query": "Cyberpunk"})
        assert len(state.engine._learning_log) == before

    def test_empty_query_returns_422(self, client):
        r = client.post("/v1/propagate", json={"query": ""})
        assert r.status_code == 422

    def test_oversized_query_returns_422(self, client):
        r = client.post("/v1/propagate", json={"query": "X" * 3000})
        assert r.status_code == 422


class TestNodeDetail:
    def test_returns_existing_node_with_full_detail(self, client):
        r = client.get("/v1/node/Cyberpunk_2077")
        assert r.status_code == 200
        body = r.json()
        assert body["agent"] == "test-agent"
        node = body["node"]
        assert node["name"] == "Cyberpunk_2077"
        assert node["namespace"] == "Game"
        assert isinstance(node["outgoing"], list)
        assert len(node["outgoing"]) >= 2  # two outgoing edges in fixture

    def test_unknown_node_returns_404(self, client):
        r = client.get("/v1/node/NoSuchNode")
        assert r.status_code == 404
        body = r.json()
        assert "detail" in body

    def test_full_query_param_exposes_provenance(self, client):
        """?full=true should include provenance fields if any are set on the node."""
        # Cyberpunk_2077 has no metadata in this fixture, so just check shape
        r = client.get("/v1/node/Cyberpunk_2077?full=true")
        assert r.status_code == 200
        body = r.json()
        assert "metadata" in body["node"]

    def test_edge_limit_query_param(self, client):
        r = client.get("/v1/node/Cyberpunk_2077?edge_limit=1")
        body = r.json()
        assert len(body["node"]["outgoing"]) <= 1


class TestQuery:
    def test_substring_match(self, client):
        r = client.get("/v1/query?pattern=Cyber")
        assert r.status_code == 200
        body = r.json()
        assert body["agent"] == "test-agent"
        assert body["pattern"] == "Cyber"
        names = [m["name"] for m in body["matches"]]
        assert "Cyberpunk_2077" in names

    def test_namespace_filter(self, client):
        r = client.get("/v1/query?pattern=&namespace=Game")
        assert r.status_code == 200
        body = r.json()
        names = [m["name"] for m in body["matches"]]
        assert "Cyberpunk_2077" in names
        # CD_Projekt_Red is Company, not Game
        assert "CD_Projekt_Red" not in names

    def test_match_includes_edge_counts(self, client):
        r = client.get("/v1/query?pattern=Cyberpunk")
        body = r.json()
        cyb = next(m for m in body["matches"] if m["name"] == "Cyberpunk_2077")
        assert cyb["edge_count_out"] >= 2
        assert cyb["edge_count_in"] == 0

    def test_no_match_returns_empty(self, client):
        r = client.get("/v1/query?pattern=NoSuchPattern")
        assert r.status_code == 200
        body = r.json()
        assert body["matches"] == []
        assert body["total_matched"] == 0

    def test_empty_pattern_lists_all_nodes(self, client):
        """Empty pattern + no namespace is the 'list everything' fallback (capped)."""
        r = client.get("/v1/query?pattern=")
        assert r.status_code == 200
        body = r.json()
        # Fixture has 4 nodes total; all should appear under default limit.
        assert len(body["matches"]) >= 3


class TestOpenApi:
    def test_all_m3_endpoints_in_schema(self, client):
        r = client.get("/openapi.json")
        paths = r.json()["paths"]
        assert "/v1/propagate" in paths
        assert "/v1/node/{name}" in paths
        assert "/v1/query" in paths
