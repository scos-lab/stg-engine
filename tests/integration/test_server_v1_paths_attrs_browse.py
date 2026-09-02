"""Integration tests for /v1/paths, /v1/attrs, /v1/browse (M4).

Uses FastAPI's TestClient against a small in-process engine modelling the
stg-steam shape: Game nodes carry intrinsic-property metadata and link to
Tag nodes via weighted `tagged_as` edges. Verifies path-finding, metadata
key discovery, and reverse-hub intersection/union with weighted ranking.
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
    """Engine with three games tagged into FPS / Co_op, plus metadata."""
    engine = STGEngine()

    # Two games tagged FPS + Co_op (intersection winners), one FPS-only.
    # Weights mimic SteamSpy vote counts so browse ranking is testable.
    engine.ingest_stl('[Counter_Strike] -> [FPS] ::mod(action="tagged_as", confidence=1.0, weight=5000)')
    engine.ingest_stl('[Counter_Strike] -> [Co_op] ::mod(action="tagged_as", confidence=1.0, weight=800)')
    engine.ingest_stl('[Left_4_Dead_2] -> [FPS] ::mod(action="tagged_as", confidence=1.0, weight=3000)')
    engine.ingest_stl('[Left_4_Dead_2] -> [Co_op] ::mod(action="tagged_as", confidence=1.0, weight=4000)')
    engine.ingest_stl('[Doom] -> [FPS] ::mod(action="tagged_as", confidence=1.0, weight=2000)')

    # Intrinsic-property self-loops → metadata on Game nodes.
    engine.ingest_stl(
        '[Counter_Strike] -> [Counter_Strike] '
        '::mod(action="intrinsic_properties", appid="10", price_usd="9.99", release_date_iso="2000-11-01")'
    )
    engine.ingest_stl(
        '[Left_4_Dead_2] -> [Left_4_Dead_2] '
        '::mod(action="intrinsic_properties", appid="550", price_usd="9.99")'
    )

    for g in ("Counter_Strike", "Left_4_Dead_2", "Doom"):
        engine._nodes[engine._nk(g)].namespace = "Game"
    for t in ("FPS", "Co_op"):
        engine._nodes[engine._nk(t)].namespace = "Tag"

    stg_path = tmp_path / "memory.stg"
    engine.save(str(stg_path))

    state = ServerState(
        engine=engine,
        agent_name="test-agent",
        stg_path=str(stg_path),
        engine_mtime=os.path.getmtime(stg_path),
        server_version="0.7.0a1-test",
    )
    return TestClient(create_app(state))


class TestPaths:
    def test_direct_path_found(self, client):
        r = client.get("/v1/paths", params={"source": "Counter_Strike", "target": "FPS"})
        assert r.status_code == 200
        body = r.json()
        assert body["agent"] == "test-agent"
        assert ["Counter_Strike", "FPS"] in body["paths"]
        assert body["path_count"] >= 1

    def test_paths_sorted_shortest_first(self, client):
        r = client.get("/v1/paths", params={"source": "Counter_Strike", "target": "Co_op", "max_depth": 4})
        body = r.json()
        lengths = [len(p) for p in body["paths"]]
        assert lengths == sorted(lengths)

    def test_unknown_node_returns_empty(self, client):
        r = client.get("/v1/paths", params={"source": "Doom", "target": "Nonexistent_Tag"})
        body = r.json()
        assert body["paths"] == []
        assert body["path_count"] == 0

    def test_limit_truncates(self, client):
        r = client.get("/v1/paths", params={"source": "Counter_Strike", "target": "FPS", "limit": 1})
        body = r.json()
        assert len(body["paths"]) <= 1


class TestAttrs:
    def test_namespace_scope_lists_keys(self, client):
        r = client.get("/v1/attrs", params={"namespace": "Game"})
        assert r.status_code == 200
        body = r.json()
        assert body["scope"] == "namespace:Game"
        keys = {k["key"]: k for k in body["keys"]}
        assert "appid" in keys
        # appid present on 2 of 2 metadata-carrying Game nodes (Doom has none)
        assert keys["appid"]["count"] == 2
        # release_date_iso only on Counter_Strike
        assert keys["release_date_iso"]["count"] == 1

    def test_node_scope(self, client):
        r = client.get("/v1/attrs", params={"node": "Counter_Strike"})
        body = r.json()
        assert body["scope"] == "node:Counter_Strike"
        keys = {k["key"] for k in body["keys"]}
        assert {"appid", "price_usd", "release_date_iso"} <= keys

    def test_node_precedence_over_namespace(self, client):
        r = client.get("/v1/attrs", params={"node": "Counter_Strike", "namespace": "Game"})
        assert r.json()["scope"] == "node:Counter_Strike"


class TestBrowse:
    def test_intersection_ranks_by_weight(self, client):
        r = client.get("/v1/browse", params={
            "targets": "FPS,Co_op", "mode": "intersection", "namespace": "Game",
        })
        assert r.status_code == 200
        body = r.json()
        names = [it["name"] for it in body["items"]]
        # Only games tagged BOTH FPS and Co_op — Doom is FPS-only, excluded.
        assert set(names) == {"Counter_Strike", "Left_4_Dead_2"}
        # Left_4_Dead_2 score 7000 > Counter_Strike 5800 → ranked first.
        assert names[0] == "Left_4_Dead_2"
        assert all(it["matched"] == 2 for it in body["items"])

    def test_union_includes_any_match(self, client):
        r = client.get("/v1/browse", params={
            "targets": "FPS,Co_op", "mode": "union", "namespace": "Game",
        })
        names = {it["name"] for it in r.json()["items"]}
        assert names == {"Counter_Strike", "Left_4_Dead_2", "Doom"}

    def test_namespace_filter_excludes_tags(self, client):
        r = client.get("/v1/browse", params={"targets": "FPS", "namespace": "Game"})
        for it in r.json()["items"]:
            assert it["namespace"] == "Game"

    def test_empty_targets_returns_400(self, client):
        r = client.get("/v1/browse", params={"targets": " , "})
        assert r.status_code == 400

    def test_invalid_mode_returns_422(self, client):
        r = client.get("/v1/browse", params={"targets": "FPS", "mode": "bogus"})
        assert r.status_code == 422


class TestOpenApi:
    def test_all_m4_endpoints_in_schema(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        assert "/v1/paths" in paths
        assert "/v1/attrs" in paths
        assert "/v1/browse" in paths
