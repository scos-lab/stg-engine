# Changelog

All notable changes to STG Engine are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.0a1] — 2026-04-19

### Added — Community-Centric Propagation (Phase 7I)

- **Community-grouped `stg propagate` output** — default mode shifts from
  flat node list to community-grouped results. Each community shows its top
  representative nodes with full inline detail (edges, modifiers, descriptions).
  No re-query needed to understand a result.
- **Heat, recency, and structural baseline** — every community signal is
  derived on the fly from existing edge state (`salience`, `last_used`) and
  graph topology (`node_elevation`). Recently touched topics surface above
  cold ones; structurally central communities retain a baseline floor.
- **Sigmoid-normalized heat** — bounds temperature contribution to `[0, 1)`
  so a hot topic cannot drown out precise query matches.
- **Query-seed surfacing** — when a query-matching node lives inside a
  community but is not a top representative, it is still shown with a 🎯
  marker and full detail so the explicit hit is never lost to granularity.
- **Query tokenization** — `_ - .` are treated as equivalent to space in
  community-name and seed-node matching (`"website factory"` matches
  `website_factory`).
- **Two-tier sort** — name-matched active communities always outrank
  structural hubs regardless of raw score magnitude.
- **Virtual-edge filtering by default** — auto-generated sibling/xref
  bridge edges no longer clutter node detail; use `--virtual` to include.
- **New propagate flags** — `--nodes` (legacy flat output), `--brief`
  (terse community summary, no inline detail), `--top N` (cap community
  count, default 5), `--virtual` (include virtual edges in detail).
- **`stg gravity heat`** — new inspection command; computes community
  heat / recency / baseline on-the-fly. Accepts `[coarse|medium|fine]`
  resolution and `--community=<name>` filter.

### Added — User-Level Config (`~/.stg/config.json`)

- **`stg config` subcommand** — `get` / `set` / `unset` / `list`.
  Hides the config file path from users.
- **Agent-selection priority chain** — `--agent` flag > `STG_AGENT` env
  var > `~/.stg/config.json` `default_agent` > hardcoded fallback `"default"`.
  Allows per-machine default agent without per-command flags.

### Changed

- **Default agent name** — hardcoded fallback changed from `"syn-claude"`
  to `"default"`. Fresh installs now auto-create `~/.stg/default/memory.stg`
  on first use. Users who depended on the implicit `syn-claude` default
  can set it explicitly:

  ```
  stg config set default_agent syn-claude
  ```

  No data migration required — existing agent directories work unchanged.

### Fixed

- **Heat ranking bug** — unbounded raw heat (unnormalized
  `Σ salience · exp(-λΔt)`) could reach ~40 for hot topics and 20× a cold
  community's rep_activation contribution, causing unrelated hot topics to
  outrank precise query matches. Sigmoid normalization + two-tier sort
  resolve this.

### Internal

- Two new dataclasses: `CommunityPropagateResult`, `RepresentativeEntry`,
  `EventEntry`
- New pure functions in `gravity.py`: `aggregate_to_communities()`,
  `compute_community_signals()`, `_normalize_for_match()`
- 36 new unit tests (26 community propagation + 10 CLI config)
- Full regression at 1013 / 1015 tests passing (2 pre-existing failures
  unrelated to Phase 7I: Kanerva ordering nondeterminism, universal
  importer Windows path parsing)

## [0.2.0a1] — 2026-04-07

### Added — Initial public alpha release

- **STG Core Engine** — full Python implementation of the Semantic Tension Graph
  - 33 modules, ~14,000 LOC
  - Density-monism-grounded cognitive architecture
  - 974 unit tests passing
- **Rust acceleration core** (`stg_engine._rust_core`)
  - `propagate_inner_loop` — spreading-activation hot path
  - `hebbian_update` — Hebbian learning with confidence/salience split
  - `compute_elevations` — gravity-based structural importance
  - 22 Rust unit tests, byte-level parity with Python reference
  - Compiled binary protects calibrated parameters and algorithm details
- **CLI** — `stg` command (`stg propagate`, `stg ingest`, `stg query`, etc.)
- **Persistence** — `.stg` file format (SQLite-based)
- **Importers** — Markdown, Obsidian, Universal STL importers
- **Algorithms** — Gravitational propagation, community naming, Hebbian learning,
  synaptic pruning, Kanerva SDM convergence, edge merging, cross-reference
  resolution, temporal episode reconstruction, perception (CNN), world modeling
- **Documentation** — README, examples, contributing guide

### License

This is the first release under the Business Source License 1.1 with a custom
Additional Use Grant. Free for personal, academic, educational, non-profit,
government, freelancer, and open source use. Commercial use by for-profit
companies requires a separate commercial license.

Change Date: 2030-04-07 (this version converts to Apache License 2.0).

Contact for commercial licensing: licensing@scos-lab.org

### Notes

- This is an **alpha release**. APIs may change before 1.0.
- The Rust core requires a compiled wheel matching your Python version.
  Pre-built wheels are provided for CPython 3.10, 3.11, 3.12 on Linux,
  macOS, and Windows.
- If the Rust extension is unavailable, STG falls back gracefully to the
  pure Python implementation (with reduced performance).
