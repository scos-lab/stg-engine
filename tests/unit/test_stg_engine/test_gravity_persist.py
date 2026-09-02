"""Unit tests for gravity map persistence (F6).

The built GravityMap is stored in the .stg and restored on the next process
instead of rebuilt (~700ms) when the graph is unchanged. These tests pin the
store/restore round-trip, the version (node/edge count) check, mutation
invalidation, and backward compatibility with .stg files that lack the table.
"""
import sqlite3

from stg_engine.engine import STGEngine


def _small_graph(tmp_path, name="g.stg"):
    e = STGEngine()
    for i in range(6):
        e.add_edge(f"Node_{i}", f"Node_{(i + 1) % 6}", is_a="x")
    p = str(tmp_path / name)
    e.save(p)  # no gravity built yet -> not persisted
    return e, p


def test_gravity_persisted_and_restored(tmp_path):
    e, p = _small_graph(tmp_path)
    gm = e.get_gravity_map()   # build
    e.save(p)                  # persist
    e2 = STGEngine.load(p)
    assert e2._gravity_cache_pending is not None
    gm2 = e2.get_gravity_map()  # restore, not rebuild
    assert gm2.node_elevation == gm.node_elevation
    assert gm2.community_names == gm.community_names
    assert gm2.elevation_by_resolution == gm.elevation_by_resolution
    assert gm2.node_community == gm.node_community


def test_gravity_cache_invalidated_on_graph_change(tmp_path):
    e, p = _small_graph(tmp_path)
    e.get_gravity_map()
    e.save(p)
    e2 = STGEngine.load(p)
    e2.add_edge("Extra_A", "Extra_B", is_a="x")     # mutation -> invalidate
    assert e2._gravity_cache_pending is None
    assert e2._gravity_map is None
    gm2 = e2.get_gravity_map()                       # fresh build over new graph
    assert gm2.node_count == e2._graph.number_of_nodes()


def test_version_mismatch_rebuilds(tmp_path):
    e, p = _small_graph(tmp_path)
    e.get_gravity_map()
    e.save(p)
    e2 = STGEngine.load(p)
    blob, nc, ec, built = e2._gravity_cache_pending
    e2._gravity_cache_pending = (blob, nc + 999, ec, built)  # counts won't match
    gm = e2.get_gravity_map()                                # -> rebuild
    assert e2._gravity_cache_pending is None
    assert gm is not None


def test_stg_without_gravity_table_loads(tmp_path):
    e, p = _small_graph(tmp_path)
    e.get_gravity_map()
    e.save(p)
    conn = sqlite3.connect(p)
    conn.execute("DROP TABLE IF EXISTS gravity_cache")
    conn.commit()
    conn.close()
    e2 = STGEngine.load(p)                    # must not raise
    assert e2._gravity_cache_pending is None
    assert e2.get_gravity_map() is not None   # rebuilds fine


def test_restored_propagate_matches_rebuilt(tmp_path):
    e, p = _small_graph(tmp_path)
    e.get_gravity_map()
    e.save(p)
    r_restored = sorted(STGEngine.load(p).propagate("node", read_only=True))
    fresh = STGEngine()
    for i in range(6):
        fresh.add_edge(f"Node_{i}", f"Node_{(i + 1) % 6}", is_a="x")
    fresh.get_gravity_map()
    assert r_restored == sorted(fresh.propagate("node", read_only=True))


def test_save_without_map_preserves_matching_cache(tmp_path):
    # A save that doesn't rebuild gravity (counts unchanged) keeps the cache.
    e, p = _small_graph(tmp_path)
    e.get_gravity_map()
    e.save(p)                       # persist
    e2 = STGEngine.load(p)          # _gravity_map None, pending present
    e2.save(p)                      # no gravity built, counts unchanged -> preserve
    e3 = STGEngine.load(p)
    assert e3._gravity_cache_pending is not None
