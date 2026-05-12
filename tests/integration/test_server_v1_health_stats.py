"""Integration tests for /v1/health and /v1/stats (M1 surface).

Uses FastAPI's TestClient (httpx under the hood) — no real uvicorn process
is spawned. The handlers run in-process against a real STGEngine populated
with a tiny fixture graph.
"""

import os
import pytest

from stg_engine import STGEngine
from stg_engine.server.app import create_app
from stg_engine.server.state import ServerState

# Skip the whole module if [server] extras aren't installed.
fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient


@pytest.fixture
def client(tmp_path):
    """A test client wired to a small in-memory engine."""
    engine = STGEngine()
    engine.ingest_stl('[Game] -> [Company] ::mod(action="developed_by", confidence=1.0)')
    engine.ingest_stl('[Game] -> [Genre] ::mod(action="belongs_to", confidence=0.9)')

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


class TestHealth:
    def test_returns_200_with_agent_identity(self, client):
        r = client.get("/v1/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["agent"] == "test-agent"

    def test_reports_node_and_edge_counts(self, client):
        r = client.get("/v1/health")
        body = r.json()
        assert body["node_count"] >= 3
        assert body["edge_count"] >= 2

    def test_exposes_engine_mtime_for_staleness_detection(self, client):
        r = client.get("/v1/health")
        body = r.json()
        assert "engine_mtime" in body
        assert isinstance(body["engine_mtime"], float)
        assert body["engine_mtime"] > 0

    def test_exposes_uptime(self, client):
        r = client.get("/v1/health")
        body = r.json()
        assert "uptime_seconds" in body
        assert body["uptime_seconds"] >= 0


class TestStats:
    def test_returns_200_with_agent_and_stats_dict(self, client):
        r = client.get("/v1/stats")
        assert r.status_code == 200
        body = r.json()
        assert body["agent"] == "test-agent"
        assert isinstance(body["stats"], dict)

    def test_stats_dict_includes_node_count(self, client):
        r = client.get("/v1/stats")
        stats = r.json()["stats"]
        assert "node_count" in stats or "nodes" in stats  # tolerate engine field name


class TestApi:
    def test_unknown_endpoint_returns_404(self, client):
        r = client.get("/v1/nonexistent_endpoint")
        assert r.status_code == 404

    def test_openapi_docs_endpoint_serves(self, client):
        r = client.get("/docs")
        assert r.status_code == 200

    def test_openapi_schema_includes_health_and_stats(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        paths = schema["paths"]
        assert "/v1/health" in paths
        assert "/v1/stats" in paths
