# STG Engine

> **Semantic Tension Graph** — a cognitive memory system for AI agents,
> grounded in Density Monism and Hebbian-inspired learning.

[![License: BUSL-1.1](https://img.shields.io/badge/license-BUSL--1.1-blue.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()
[![Tests](https://img.shields.io/badge/tests-1026%20passing-brightgreen.svg)]()

STG Engine gives AI agents a memory that **learns**, **forgets**, and
**generalizes** the way cognitive science says memories should — built
on a graph where nodes carry activation, edges carry both confidence
and salience, and propagation is gravity-aware (community-structure
amplifies the right concepts).

This is **not another vector store**. It is an **executable cognitive
architecture**.

---

## Why STG?

| | Vector DB | LLM context | **STG Engine** |
|---|---|---|---|
| **Storage** | Embeddings | Tokens | Graph (nodes + edges) |
| **Retrieval** | Cosine similarity | None (re-prompt) | Spreading activation |
| **Learning** | None | None | Hebbian, salience decay |
| **Forgetting** | Manual delete | Context window | Synaptic pruning |
| **Structure-aware** | No | No | **Yes** (gravity propagation) |
| **Provenance** | Limited | None | Per-edge confidence + source |
| **Determinism** | Yes | No | Yes |

STG is for AI agents that need to **remember across sessions**,
**update their world model**, and **resolve conflicting information**
without retraining.

---

## Quickstart (5 minutes)

### Install

```bash
pip install stg-engine
```

Pure-Python distribution — one wheel, all platforms (Linux, macOS, Windows) and all supported Python versions (3.10, 3.11, 3.12). No compilation needed.

An optional Rust acceleration core for hot-path algorithms lives in the
[source repository](https://github.com/scos-lab/stg-engine) under `rust/` and
is picked up automatically if compiled locally; the alpha PyPI release
ships the pure-Python path only.

### First propagation

```python
from stg_engine import STGEngine

engine = STGEngine()

# Ingest knowledge as STL (Semantic Tension Language)
engine.ingest_stl('[Newton] -> [Calculus] ::mod(rule="historical", confidence=0.95)')
engine.ingest_stl('[Leibniz] -> [Calculus] ::mod(rule="historical", confidence=0.95)')
engine.ingest_stl('[Calculus] -> [Physics] ::mod(rule="enables", confidence=0.9)')
engine.ingest_stl('[Physics] -> [Engineering] ::mod(rule="enables", confidence=0.85)')

# Activate from a query and let it spread through the graph
results = engine.propagate("Newton")
print(results)
# ['Newton', 'Calculus', 'Physics', 'Engineering', 'Leibniz']

# Persist
engine.save("my_memory.stg")

# Reload later
engine2 = STGEngine.load("my_memory.stg")
```

### Hebbian learning

After each `propagate()` call, edges between co-activated nodes are
strengthened — exactly like neurons that fire together.

```python
from stg_engine.learning import HebbianLearner

learner = HebbianLearner()

for _ in range(10):
    activations = engine.propagate("Newton")
    activation_map = {n: engine._nodes[n].activation for n in activations}
    learner.learn_from_propagation(engine, activation_map)

# The Newton → Calculus path is now stronger than before.
```

See the `examples/` directory for more.

---

## Architecture

```
stg_engine/                Pure Python — no compilation required
├── engine.py              The main STGEngine class
├── types.py               Node, Edge, Tension data structures
├── formulas.py            Ψ (system stability), tension, activation
├── gravity.py             Gravitational propagation + community detection
├── learning.py            Hebbian learner + synaptic pruner
├── persistence.py         .stg file format (SQLite-backed)
└── cli.py                 The `stg` command-line tool
```

The source repository additionally contains an optional Rust acceleration
core (`rust/`) implementing three hot-path algorithms — `propagate_inner_loop`,
`hebbian_update`, `compute_elevations`. If the extension is compiled locally
(`maturin develop`), stg_engine auto-detects it and uses it; otherwise the
pure-Python path handles everything. PyPI ships the pure-Python path only
during the alpha phase.

## Three things STG does that vector DBs cannot

### 1. Structure-aware retrieval (gravity propagation)

Nodes that bridge multiple communities are amplified. Nodes buried inside
small fragment clusters are suppressed. The graph's topology *is* the
prior — no manual labeling needed.

### 2. Hebbian learning over time

Edges between co-activated nodes get stronger. Edges that never co-activate
weaken and eventually get pruned. The graph adapts to actual usage.

### 3. Confidence vs salience separation

- **Confidence** = "how true is this?" — never auto-decays
- **Salience** = "how easily can I recall it?" — modified by use

Reading a fact 100 times makes it more *retrievable* but not more *true*.
This separation is critical and is missing from every vector store.

---

## CLI

```bash
# Stats
stg stats

# Add knowledge
stg ingest '[A] -> [B] ::mod(confidence=0.9, salience=0.7)'

# Spreading activation
stg propagate "your query here"

# Find paths between concepts
stg paths Newton Engineering

# Inspect a node
stg node Newton

# Save/load
stg save my_memory.stg
```

Full CLI reference: `stg --help`

---

## Precision recall (default in 0.4+)

`stg propagate` does more than spread activation — it now **converges**.
Four cooperating mechanisms sharpen retrieval without ever filtering an
edge:

| Mechanism | What it does | When it fires |
|-----------|--------------|---------------|
| **R1** Recency × supersede soft weight | Older / superseded edges get softly down-weighted (default 30-day half-life, supersede factor 0.3) | Every propagate |
| **R2** Multi-seed chain intersection | Each query token runs through propagate separately; output is the node intersection across each token's reconstructed chains | When ≥2 query tokens hit node names |
| **R5** active_context anchor | Nodes from your last `stg select` get a temporary +5.0 elevation boost, pulling propagate energy toward your current focus | When `active_context` is non-empty (TTL 30 min) |
| **R6** Edge-as-fallback-seed | Tokens that match no node name fall through to scan edge `description`/`lesson`/`action`/`role`/`status`/`is_a`. Returns event edges as a separate "Event-edge" view | When ≥1 query token misses all node names |
| **R7** Community dominance ratio | Communities scoring below `dominant/3.0` are folded out — except those containing precise query hits, which are always preserved | Every propagate (community mode) |

The **memory-never-vanishes** principle is hard-coded: superseded edges
are softly down-weighted, never deleted. Recall completeness is preserved
across all four mechanisms — they shape ranking, not membership.

Disable any mechanism via flag (rarely needed):

```bash
stg propagate "your query"                          # default — all five ON
stg propagate "your query" --no-recency-weight      # disable R1
stg propagate "your query" --no-multi-seed          # disable R2
stg propagate "your query" --no-context-anchor      # disable R5
stg propagate "your query" --no-edge-fallback       # disable R6
stg propagate "your query" --no-community-filter    # disable R7
stg propagate "your query" --legacy                 # all five OFF (pre-0.4)
```

Full design: see `STG_PRECISION_RECALL_DESIGN.md`.

---

## Skills: turning Skill nodes into runnable commands

Skill-namespaced nodes (`Skill:SomeName`) can be registered as **executable**
and invoked via `stg use`. This makes STG double as a lightweight capability
registry — ask what you can do, then do it, all inside one tool.

### One-time setup (opt-in)

```bash
# Master switch — off by default for safety
stg config set skill.enabled true

# Whitelist the directories where your scripts live
stg config set skill.roots "/home/you/my-tools,/home/you/workshop"

# (optional) Name interpreters you want to re-use across skills
stg config set skill.interpreters.myvenv "/home/you/proj/.venv/bin/python3"
```

### Register a skill

Add a `Skill:`-namespaced node whose outgoing edge carries `path`,
`executable`, and `interpreter`:

```bash
stg ingest '[Skill:Greet] -> [Greeting_Target] ::mod(
  rule="empirical", confidence=0.95,
  path="/home/you/my-tools/greet.py",
  description="say hello, optionally to someone",
  executable="true",
  interpreter="python3",
  args_template="[name]",
  timeout_s="10"
)'
```

Or retrofit invocation fields onto an existing Skill edge:

```bash
stg skill configure Greet --executable --interpreter python3 \
  --args-template "[name]" --timeout 10
```

### Browse and run

```bash
# See every registered skill, executable ones on top
stg skill list

# Detail on one skill + recent invocations
stg skill show Greet

# Run it — positional args pass through to the script
stg use Greet World

# Dry-run to see the resolved command line without executing
stg use Greet World --dry-run

# Audit log
stg skill history --limit 10
```

### Security model

Execution requires **double opt-in**: `skill.enabled=true` in user config AND
`executable="true"` on the edge. The resolved script path must lie under at
least one `skill.roots` entry (symlinks resolved). Subprocess invocation uses
list args (no shell), a hard timeout, and a 10 MB output cap. Every call
writes one audit row in the `skill_invocations` SQLite table.

Interpreters resolve in this order: absolute path → named entry in
`skill.interpreters.<name>` → builtin name (`python3`/`python`/`bash`/`sh`/
`node`) via `shutil.which`. No magic names are hardcoded into the engine;
portability across machines is the user's responsibility via config.

---

## HTTP server (optional)

For AI agent knowledge bases that need to serve one STG over a JSON API,
install the optional `server` extra and run `stg-server`:

```bash
pip install stg-engine[server]
stg-server --agent stg-steam            # default port 8765, bind 127.0.0.1
```

The server is resident — one engine load at startup, served to many
clients — eliminating the ~50 ms python-startup tax of subprocess
`stg propagate` calls. Default bind is localhost; non-localhost binds
require `--allow-external-bind` (v1 has no auth).

v1 endpoints (read-only — mutation stays on `stg` CLI):

| Endpoint | Purpose |
|---|---|
| `GET /v1/health` | Liveness + agent identity + `.stg` file mtime for staleness detection |
| `GET /v1/stats` | Full `engine.get_stats()` |
| `POST /v1/propagate` | Activation propagation, `read_only=True` (no Hebbian/telemetry side-effects) |
| `GET /v1/node/{name}` | Single-node detail with incoming/outgoing edges |
| `GET /v1/query` | Fuzzy substring search + namespace filter |

Interactive API docs at `http://127.0.0.1:8765/docs` (FastAPI auto-OpenAPI).

Example:

```bash
curl -X POST http://127.0.0.1:8765/v1/propagate \
     -H "Content-Type: application/json" \
     -d '{"query": "Elden Ring", "max_nodes": 5}'
```

**Design philosophy:** the HTTP server is JSON substrate. Anonymous
external traffic flows through `propagate(read_only=True)` so it
doesn't shape the agent's autonomous learning signal — agents learn
from their own CLI propagations, not from third-party HTTP reads.

Two more endpoints (`/v1/attrs/{name}`, `/v1/paths`) are pending in M4.
Full design and decisions:
`Semantic-Kernel-of-Consciousness/development/design/STG_HTTP_SERVER_DESIGN.md`.

---

## License

**STG Engine is dual-licensed:**

- **Free** under the [Business Source License 1.1](LICENSE) for personal,
  academic, educational, non-profit, government, freelancer, and open
  source use.
- **Commercial use by for-profit companies requires a separate license.**
  Contact **contact@stl-lang.org** for details.

After **2030-04-07**, this version automatically converts to Apache License 2.0.

### TL;DR

| You are | License you need |
|---|---|
| Individual developer / hobbyist | Free |
| Student / academic researcher | Free |
| Open source project | Free |
| Non-profit / government | Free |
| For-profit company (any size) | Commercial license required |

This is the **JetBrains / Unity model**, not the "100% open source" model.
We do this because STG represents years of original research and we want
it to remain free for the people who advance the field, while companies
that profit from it contribute back. Contact **contact@stl-lang.org**
for details.

## Theoretical foundation

STG is grounded in **Density Monism** — a research framework that derives
quantum field theory, particle masses (Koide formula), and consciousness
from a single scalar density field ρ(x,t).

- Project: [scos-lab/koide-lcft](https://github.com/scos-lab/koide-lcft)
- Architecture paper: forthcoming

## Status

**0.3.0a2** — Alpha. APIs may change before 1.0.

- 1026 unit tests passing
- Dogfooded daily as the long-term memory of an AI agent running inside Claude Code

## Contributing

Bug reports, issues, and pull requests are welcome. Note that contributors
agree their contributions are licensed under the same BSL 1.1 terms.

For substantial contributions or research collaboration, contact
**contact@stl-lang.org**.

## Naming

"STG" and "Semantic Tension Graph" are names used by wuko / scos-lab.

## Citation

If STG helps your research, please cite:

```
@software{stg_engine_2026,
  author = {wuko},
  title = {STG Engine: A Cognitive Memory System for AI Agents},
  year = {2026},
  url = {https://github.com/scos-lab/stg-engine},
  version = {0.3.0a2}
}
```

---

**Copyright (C) 2026 wuko / scos-lab. All rights reserved.**
