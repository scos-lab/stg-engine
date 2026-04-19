# STG Engine

> **Semantic Tension Graph** — a cognitive memory system for AI agents,
> grounded in Density Monism and Hebbian-inspired learning.

[![License: BUSL-1.1](https://img.shields.io/badge/license-BUSL--1.1-blue.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()
[![Tests](https://img.shields.io/badge/tests-1013%20passing-brightgreen.svg)]()
[![Powered by Rust](https://img.shields.io/badge/powered%20by-Rust-red.svg)]()

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

(Pre-built wheels available for CPython 3.10–3.12 on Linux, macOS, and Windows.
The Rust acceleration core is included automatically.)

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

STG is a **Python package with a Rust acceleration core**:

```
stg_engine/                Python — orchestration, persistence, CLI
├── engine.py              The main STGEngine class
├── types.py               Node, Edge, Tension data structures
├── formulas.py            Ψ (system stability), tension, activation
├── gravity.py             Gravitational propagation + community detection
├── learning.py            Hebbian learner + synaptic pruner
├── persistence.py         .stg file format (SQLite-backed)
├── cli.py                 The `stg` command-line tool
└── _rust_core.so          Compiled Rust algorithms (hot path)
```

The Rust core implements three hot-path algorithms:

1. **`propagate_inner_loop`** — spreading activation iteration
2. **`hebbian_update`** — co-activation-driven salience update
3. **`compute_elevations`** — gravity-based structural importance

All three have a pure-Python fallback if the Rust extension is unavailable.

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

**0.3.0a1** — Alpha. APIs may change before 1.0.

- 1013 unit tests passing
- 22 Rust core unit tests passing
- 5 Python ↔ Rust parity tests passing
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
  version = {0.3.0a1}
}
```

---

**Copyright (C) 2026 wuko / scos-lab. All rights reserved.**
