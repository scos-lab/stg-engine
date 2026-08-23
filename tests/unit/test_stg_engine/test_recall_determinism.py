"""Multi-seed recall: deterministic across processes (no PYTHONHASHSEED dependence) and bounded on dense graphs."""
import json
import os
import subprocess
import sys
import time

from stg_engine import STGEngine
from stg_engine.recall import multi_seed_propagate

_SCRIPT = r'''
import json, sys
from stg_engine import STGEngine
from stg_engine.recall import multi_seed_propagate
e = STGEngine()
# two overlapping clusters so the intersection has many equal-appearance ties
for i in range(12):
    e.add_edge(f"alpha_{i}", f"hub_{i%3}", role="r")
    e.add_edge(f"hub_{i%3}", f"beta_{i}", role="r")
    e.add_edge(f"beta_{i}", f"gamma_{i%4}", role="r")
names, _ = multi_seed_propagate(e, "hub beta", ["hub", "beta"], use_gravity=False)
print(json.dumps(names))
'''


def test_multi_seed_order_is_stable_across_hash_seeds():
    outs = []
    for seed in ("1", "7", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        r = subprocess.run([sys.executable, "-c", _SCRIPT], capture_output=True, text=True, env=env, timeout=120)
        assert r.returncode == 0, r.stderr[-500:]
        outs.append(json.loads(r.stdout.strip().splitlines()[-1]))
    assert outs[0] == outs[1] == outs[2]
    assert outs[0], "expected a non-empty intersection/union"


def test_multi_seed_is_bounded_on_dense_subgraph():
    e = STGEngine()
    n = 26
    # complete DAG on 26 nodes + a shared token so both seeds light up the same dense set
    for i in range(n):
        for j in range(i + 1, n):
            e.add_edge(f"dense_{i}", f"dense_{j}", role="r")
    t0 = time.perf_counter()
    names, _ = multi_seed_propagate(e, "dense dense", ["dense", "dense"], use_gravity=False)
    assert time.perf_counter() - t0 < 10.0   # unbounded all_simple_paths on K26 would run for a very long time
    assert names


def test_multi_seed_learns_once_not_per_token():
    e = STGEngine()
    for i in range(6):
        e.add_edge(f"x_{i}", "mid", role="r")
        e.add_edge("mid", f"y_{i}", role="r")
    e.enable_learning()
    calls = {"n": 0}
    real = e._learner.learn_from_propagation

    def counting(engine, amap):
        calls["n"] += 1
        return real(engine, amap)

    e._learner.learn_from_propagation = counting
    multi_seed_propagate(e, "mid y_1 y_2", ["mid", "y_1", "y_2"], use_gravity=False)
    assert calls["n"] == 1, f"learner ran {calls['n']}× for a 3-token query (expected once, on the merged result)"
