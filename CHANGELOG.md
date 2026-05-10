# Changelog

All notable changes to STG Engine are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.5.0a7] — 2026-05-10

### Added — Intrinsic-property self-loop edges (STL Protocol §9.4)

Self-loops marked with `action="intrinsic_properties"` are now recognized
as **storage-only attribute carriers** for node-identity attributes — id
values, registration codes, fixed metadata that belong to the node itself.

Use case: a node with many outgoing edges that would otherwise duplicate
the same identity attributes (e.g. a game node with `appid` / `release_year`
referenced by 35+ tag/genre/feature edges). Putting the attributes on a
single self-loop carrier keeps business edges clean while keeping the data
canonically attached to the node.

Runtime contract (per STL Operational Protocol §9.4):

- **Preserve** edge data in `_edges`, `_edges_lookup`, and `_graph`
- **Exclude from propagation** — the edge is not a path; activation must
  not flow `Node → Node` through it
- **Exclude from community detection** — the edge does not contribute
  to graph topology for Louvain / gravity
- **Render distinctly** — UIs surface the modifiers as a `Properties:`
  section in node detail views

Implementation:

- `types.py`: `STGEdge.is_intrinsic_property()` helper +
  `INTRINSIC_PROPERTIES_ACTION` reserved-value constant
- `engine.py`: filter in the `_rust_edges` build step inside
  `_propagate_from_seeds`
- `gravity.py`: drop intrinsic self-loops from the Louvain input in
  `build_gravity_map`; skip them in the heat-compute loop in
  `compute_community_signals`

12 new tests cover helper detection, storage preservation, propagation
exclusion, and community-detection exclusion. Backward compatible — no
existing edge changes behavior; only the reserved `action` value is
recognized.

Reference STL Protocol commit:
[scos-lab/semantic-tension-language@1d9bd5c](https://github.com/scos-lab/semantic-tension-language/commit/1d9bd5c).

### Added — `stg node` Properties: section rendering

`_render_node_detail` (CLI `stg node <name>` and inlined under
community-mode `stg propagate`) now extracts intrinsic-property self-loops
from outgoing/incoming edge lists and renders them as a dedicated
`Properties:` section between metadata and the edge listings:

```
Node: Elden_Ring
  Tension: 0.0000
  Activation: 0.0000

  Properties:
    appid: 1245620
    release_year: 2022
    price_usd: 59.99

  Outgoing (2):
    → [Souls_Like] (c=0.95, s=0.5)
    → [Action_RPG] (c=0.95, s=0.5)
```

Carrier-internal modifier keys (`action`, `rule`, `edge_class`,
`_epistemic_warnings`) are suppressed from the Properties view since they
describe the carrier convention itself, not the node identity.

4 additional tests cover the rendering, exclusion from outgoing/incoming
counts, carrier metadata suppression, and the no-Properties-when-no-
intrinsic-edges baseline.

### Documentation — STG_AGENT_GUIDE.md major revision

The agent-facing guide (`stg guide`) was several versions behind the engine. Audited against 0.5.0a6 behavior and updated:

- **Multi-edge supersede semantics rewritten** — the old text described the
  pre-0.5.0a6 Path-1 behavior ("old edge marked superseded, new becomes
  active"). Replaced with the current Path-2-only model: same (src,tgt)
  multi-edges coexist as complementary facets; supersede flags only fire
  on same source + same `(meta_field, value)` + DIFFERENT target.
- **New "Meta semantic fields — required, one per edge" subsection** —
  documents the nine SEMANTIC_FIELDS (`is_a`, `action`, `role`, `status`,
  `phase`, `relation`, plus the legacy `type`/`kind`/`predicate`), how to
  pick the most specific one, and that they are what drives supersede
  detection. Bumps the "always include" count from 3 to 4.
- **Timestamps section rewritten** — `timestamp=` → `occurred_time=` (the
  old name was ambiguous and a known LongMemEval ingest-fallback
  source). Documents the modifier-level `created_at` override for
  backfilled external dataset ingests. Drops the "recorded_at not
  implemented" line — that field is gone, merged into `created_at`.
- **New "Dual-anchor retrieval" subsection** — documents the 0.5.0a1
  retrieval mode: query containing ≥2 chunks that exact-match canonical
  node names triggers the focused 🎯 Anchor-pair view (3-6 lines vs the
  ranked-list 50-80). Includes the n-gram matching note (0.5.0a2: "Food
  For Thought" → `Food_For_Thought_Charity_Gala`) and the all-token
  edge-content scan (0.5.0a5).
- **New "STL Reference" top-level section** — three-tier modifier table
  (required / recommended / situational), namespace syntax, confidence
  calibration table, the four usage patterns (Pointer / Event / Skill /
  Property-Carrier), pointer to the external STL spec, and a `stl validate`
  mention.

Net change: 846 → 1013 lines, +196/-29 (plus the §9.4 Property-Carrier
addition above).

---

## [0.5.0a6] — 2026-04-30

### Fixed — `superseded_at` no longer fires on complementary multi-edges

Two paths set `superseded_at` in `add_edge`. The duplicate-edge handler
("Path 1", `engine.py:520-537`) marked any older edge with the same
`(source, target)` as superseded whenever the new edge differed in
`confidence / strength / rule / description` — **without checking semantic
fields**. The dedicated `_flag_suspected_supersede` ("Path 2",
`engine.py:574-614`) used the correct rule: same source + same
`(semantic_field, value)` + **different** target.

> User-reported failure case:
>
> ```
> ← [User] action="took"             description="User took a 5-day trip to SF..."   superseded_at=...
> ← [User] status="had_amazing_time" description="User had amazing time during trip"
> ```
>
> Both edges describe the same trip (action vs status — complementary
> facets), not a correction. Path 1 wrongly marked the first as
> superseded.

The footprint mattered because `superseded_at` is consumed in two places:
- `recall.py:114` — superseded edges weighted ×0.3 (soft attenuation)
- `learning.py:273` — superseded edges frozen from Hebbian learning

So a wrong flag both hides legitimate context from recall and prevents the
edge from ever being reinforced.

**Fix:** Path 1 no longer sets `superseded_at`. Same `(source, target)`
edges with different content are kept as multi-edges — both alive, lookup
points to newest, supersede left for `_flag_suspected_supersede` to
decide based on actual semantic conflict.

### Impact on existing data

A scan across all `~/.stg/` agents shows ~86K Path-1 wrong flags vs ~255K
Path-2 legitimate flags — about 25% of all `superseded_at` markers are
spurious. Cleanup script: `scripts/cleanup_path1_supersede.py`. Discriminator:
edge has `superseded_at` set but no `suspected_supersede` → Path-1
footprint, safe to clear.

### Tests

- `test_engine.py::test_different_confidence_creates_multi_edge` updated:
  asserts no `superseded_at` / `suspected_supersede` on the older edge.
- All Path 2 tests in `test_supersede.py` continue to pass unchanged.
- Full suite: 1126 passed (one pre-existing flaky kanerva test, one
  pre-existing windows-path test on Linux — both unrelated).

### Backward compat

`superseded_at` field semantics are unchanged for consumers (recall, learning,
CLI rendering). The only change is which path is allowed to set it.

---

## [0.5.0a5] — 2026-04-26

### Changed — Edge content scan now runs over ALL tokens

In 0.5.0a1–0.5.0a4, R6 used a strict node-first / edge-fallback split:
a token that matched any node name went to multi-seed propagate **only**,
never to edge content scan. This missed an important case:

> User-reported failure case: query
> `"User volunteered participated attended ran charity events before Run for the Cure"`
> failed to surface `[User] -[volunteer @2023-09-25]-> [Food_For_Thought_Charity_Gala]`.
>
> Root cause: `volunteered` matched the unrelated concept node
> `[Volunteering_Best_Practices]` so it was classified as `node_token`
> and never scanned against edge `description` / `action` / `role`.
> The actual fact lived in the edge's `description="User volunteered..."`
> and `role="volunteer"` — both invisible to node propagation.

`scan_edges_by_content` is now invoked over the **full** `_split_tokens`
result, not just the node-unmatched subset. A token's node match no
longer suppresses its edge-content scan. Both signals are real, both
should surface.

IDF weighting handles the noise: very common tokens (e.g. `user`)
appear in many edges so their per-edge contribution is small; rare
tokens (e.g. `volunteered`, `participated`) dominate the score.
The target edge above scores 5.10 (matched: `volunteered`) and shows
in the top 5 of the Event-edge view, with `🔗 double-hit` marking.

### Demo

```
$ stg propagate "User volunteered participated attended ran charity events before Run for the Cure"
  📍 exact anchors: [User, Charity_Events]
  🪢 token routing: node=[volunteered, run, cure] node_unmatched=[participated, attended, ran, before]
                    edge scan over all 7 tokens → 25 edge hits
...
🪢 Event-edge matches (25):
  [User] -[participated_in @2023-10-15]-> [Run_For_The_Cure_Event]   score=17.04  🔗 双重命中
  [User] -[raised_funds_for @2023-10-15]-> [Breast_Cancer_Research]  score=8.12   🔗 双重命中
  [User] -[is_motivated_to_continue @...]-> [Charity_Events]         score=8.12   🔗 双重命中
  [User] -[attended @2023-07-17]-> [Charity_Golf_Tournament]         score=5.50   🔗 双重命中
  [User] -[volunteer @2023-09-25]-> [Food_For_Thought_Charity_Gala]  score=5.10   🔗 双重命中
  ...
```

The Food_For_Thought edge — invisible in 0.5.0a4 — now ranks fifth.

### Backward compat

`scan_edges_by_content` itself is unchanged; only its caller (cli.py
`cmd_propagate`) passes a wider input. `--no-edge-fallback` still
disables the entire R6 path. Test suite: 1068 passed (one pre-existing
flaky kanerva test unrelated to recall code).

## [0.5.0a4] — 2026-04-26

### Changed — Anchor-aware community filter

When exact anchors are present, the community-mode renderer now performs
two additional passes after `community_dominance_filter`:

1. **Anchor-bearing community filter.** Only communities that actually
   contain an anchor node (or already passed via `query_seeds` / community
   name match) are kept. Communities that surfaced purely because
   propagate spread into their territory — and whose top representatives
   are unrelated to the user's intent — are dropped entirely.

2. **Representative replacement.** In an anchor-bearing community, the
   `representatives` list is **replaced** with the anchor entries. The
   community's original top-by-elevation reps (which are topic-context,
   not the answer) are no longer rendered. The 🎯 Anchor-pair edge view
   above already shows the precise connection; the community block here
   confirms the anchor's location and its incoming/outgoing edges.

### Why

User-reported case: query `"user Food for thought charity gala"` would
correctly surface the anchor-pair edge but then dump representatives
from the same Louvain cluster (`Food_Recovery`, `Community_Spirit`,
`Hunger_In_USA`) plus their full incoming-edge lists — twenty lines of
food-rescue-organization noise per query. Anchor-aware filtering and
representative replacement remove that noise without affecting any
non-anchor query.

### Behavior unchanged when no anchor

Queries without exact anchors (the common case) follow the existing
R7 dominance filter only. There's no new flag — anchor presence is the
trigger.

### Demo

```
$ stg propagate "user Food for thought charity gala"
  📍 exact anchors: [User, Food_For_Thought_Charity_Gala]
propagate(...) → 1 community(s) from 15 activated nodes:

🎯 Anchor-pair edges (1) — primary answer:
  [User] -[volunteer @2023-09-25]-> [Food_For_Thought_Charity_Gala]
     User volunteered at the 'Food for Thought' charity gala on September 25th...

[1] food_recovery  score=0.121  rep_act=0.076
  [1.1] Food_For_Thought_Charity_Gala  act=0.067  elev=2.582
      Node: Food_For_Thought_Charity_Gala
        Incoming (1):
          ← [User] (...) role: volunteer, occurred_time: 2023-09-25
```

Compare to 0.5.0a3 which would also dump `Food_Recovery`, `Community_Spirit`,
`Hunger_In_USA` representatives plus 6 incoming edges from
`Feeding_America` / `RescueFood` / `FoodFinder` / etc. — all noise for
this query.

## [0.5.0a3] — 2026-04-26

### Fixed — N-gram exact anchor matching

`match_exact_anchors` now greedily matches the **longest contiguous
whitespace N-gram** against canonical node names (joined with `_`),
not just single tokens. This makes natural-language queries work:

```
"user Food for thought charity gala"
  → 5-gram 'food_for_thought_charity_gala' matches → [Food_For_Thought_Charity_Gala]
  → 1-gram 'user' matches → [User]
```

Before 0.5.0a3, only the underscored form
`food_for_thought_charity_gala` would match. Now both forms route to
the same anchor pair → same anchor-pair edge view.

Greedy semantics: at each position, the longest N-gram that matches
consumes those N tokens before the scan moves on. This means
`User_Bike` is preferred over `User` alone when both nodes exist and
the query is `"User Bike"`.

### Changed — Anchor-pair edge view promoted to TOP of output

When ≥2 exact anchors are detected, the
`🎯 Anchor-pair edges — primary answer` block now renders **before**
the community list (was: at the bottom, below event-edge view). This
matches user intent: when you type two specific node names, the
edge between them is the answer, not background topic structure.

### Demo

```
$ stg propagate "user Food for thought charity gala"
  📍 exact anchors: [User, Food_For_Thought_Charity_Gala] (forced into result)
propagate(...) → 1 community(s) from 15 activated nodes:

🎯 Anchor-pair edges (1) — primary answer:
  [User] -[volunteer @2023-09-25]-> [Food_For_Thought_Charity_Gala]
     User volunteered at the 'Food for Thought' charity gala on September 25th...

[1] food_recovery  ...                          ← topic context, secondary
```

### Tests

`tests/unit/test_recall.py` — 4 new N-gram tests (66 total): space-
separated multi-word match; longest-prefix-wins; no overlap between
consumed N-grams; fallback to unigram when longer N-grams miss.

End-to-end on `lme-a3838d2b`: both
`stg propagate "User food_for_thought_charity_gala"` and
`stg propagate "user Food for thought charity gala"` produce identical
anchor-pair edge views with the target edge as primary answer.

## [0.5.0a2] — 2026-04-26

### Added — Exact-anchor matching (A) + Anchor-pair edge view (C / R8)

Closes a gap exposed during R6 testing: when a user types
`User food_for_thought_charity_gala`, they explicitly **want that exact
node**, but pre-0.5.0a2 the tokenizer split the underscored chunk into
five tokens (`food`, `for`, `thought`, `charity`, `gala`), every token
hit some node name (so R6 didn't trigger), and multi-seed chain
intersection over-collapsed to an unrelated community.

**A — Exact-anchor matching.** Before sub-token splitting, every
whitespace-separated chunk is tested against the canonical node-name
table. Chunks that exactly match a node (`User`, `Food_For_Thought_Charity_Gala`)
become **exact anchors**: forced into the activated set, displayed with
a `📍 exact anchors:` line, and excluded from sub-token splitting.

**C / R8 — Anchor-pair edge view.** When ≥2 exact anchors are present,
every edge whose `source` AND `target` are both in the anchor set
(bidirectional, self-loops excluded) is rendered as a dedicated
`🎯 Anchor-pair edges` block — surfacing the user's explicit "how do
these specific nodes connect?" intent at top priority.

### Improved — Edge label rendering

`_format_edge_label` now falls back through the meta semantic field
chain `action → role → status → phase → is_a → relation` when `action`
is missing. Real-world graphs often use `role="volunteer"` or
`status="active"` instead of `action`; the new label honors that.

### Demo

```
$ stg --agent lme-a3838d2b propagate "User food_for_thought_charity_gala"
  📍 exact anchors: [User, Food_For_Thought_Charity_Gala] (forced into result)
  ...
  🎯 Anchor-pair edges (1):
    [User] -[volunteer @2023-09-25]-> [Food_For_Thought_Charity_Gala]
       User volunteered at the 'Food for Thought' charity gala on September 25th...
```

### Public API

- `stg_engine.recall.match_exact_anchors(engine, query)` →
  `(anchor_names, remaining_tokens)` — anchor matching by full node name.
- `stg_engine.recall.find_edges_between(engine, anchor_names)` →
  `[STGEdge, ...]` — bidirectional edge lookup over an anchor set.

### Tests

11 new tests in `tests/unit/test_recall.py` (62 total): exact-name match,
case-insensitivity, display-case preservation, pass-through of
unmatched chunks, bidirectional symmetry, self-loop exclusion,
non-anchor endpoint exclusion, three-anchor cliques.

End-to-end: the original failing query
`stg --agent lme-a3838d2b propagate "User food_for_thought_charity_gala"`
now correctly surfaces the target edge with full description and
`occurred_time=2023-09-25`.

Full stg-engine suite: 1064 passed (one pre-existing flaky kanerva test
unrelated to recall code).

## [0.5.0a1] — 2026-04-26

### Added — R6 Edge-as-Fallback-Seed

Closes the **edge-content blind spot** in propagate. When a query token does
not match any node name (the existing seed-selection criterion), it now
falls through to scan **edge meta semantic fields** instead of being lost.

**Motivation.** STG's semantics are split between node names (topics) and
edge `modifiers` (events / predicates / occurred_time). Pre-0.5 propagate
only matched node names — query words appearing only in edge `description`
/ `action` / `lesson` (e.g. "volunteered", "participated") could not surface
the relevant event edges. LongMemEval Q16-style "what did the user
participate in?" queries failed precisely on this gap.

**Mechanism (node-first routing, no node priority lost).**

```
tokens = tokenize(query)
node_tokens, edge_tokens = classify_tokens(engine, tokens)
  # node_tokens: hit any node name (existing morphological prefix logic)
  # edge_tokens: did NOT hit any node name → fallback to edge scan

# node path runs unchanged R1+R2+R5+R7
activated = propagate(node_tokens)

# edge path scans 6 meta semantic fields per edge:
#   description / lesson / action / role / status / is_a
edge_hits = scan_edges_by_content(edge_tokens)  # IDF-ranked, capped at top 50
```

Output adds an **Event-edge view** below the existing community/node view.
Edges whose endpoint also appears in the propagate node result are tagged
`🔗 双重命中` (highest priority — both topic-matched and fact-matched).

**Why node-first and not search-everywhere.** Node-name matches dominate
because nodes are deliberately curated, high-information labels; edge text
is narrative prose with more noise. Edge fallback only activates for
tokens that node matching cannot reach — preserving the precision earned
by R1-R7 for the bulk of queries.

### Added — Public API

- **`stg_engine.recall.classify_tokens(engine, tokens)`** — splits a token
  list into `(node_tokens, edge_tokens)` using the same morphological
  prefix logic as `engine.propagate()`.
- **`stg_engine.recall.scan_edges_by_content(engine, edge_tokens, ...)`** —
  scans `EDGE_SCAN_FIELDS` for substring matches; returns
  `[(edge, matched_tokens, idf_score), ...]` sorted by score descending.
- **New constants:** `EDGE_SCAN_FIELDS = ("description", "lesson", "action",
  "role", "status", "is_a")`, `DEFAULT_MAX_EDGE_HITS = 50`,
  `DEFAULT_MIN_EDGE_TOKEN_LENGTH = 3`.

### Added — CLI flag

```
stg propagate <query> --no-edge-fallback   # disable R6 (legacy node-only routing)
stg propagate <query> --legacy             # disable R1+R2+R5+R6+R7 together
```

CLI output gains two lines when R6 fires:

```
🪢 token routing: node=[user, volunteered] edge=[participated, ran] → 9 edge hits
...
🪢 Event-edge matches (9):
  [User] -[participated_in @2023-10-15]-> [Run_For_The_Cure_Event]  score=8.93
     User ran 5 kilometers in the 'Run for the Cure' event on October 15th.
     matched: participated, ran
  ...
```

### Memory-never-vanishes guarantee

R6 only **adds** candidate edges to recall output. No edge or node is
removed, hidden, or filtered by edge-fallback. Disabling R6 via
`--no-edge-fallback` simply reverts to v0.4 behavior (pure node routing).

### Tests

`tests/unit/test_recall.py` — 13 new R6 tests (51 total): description /
lesson / action / role / status / is_a field coverage; substring match
without morphological prefix; IDF ranking validation; max-hits cap; short
token rejection; field whitelist enforcement.

End-to-end: query `"user volunteered participated ran"` on
`lme-a3838d2b` STG (LongMemEval Q16 source). Token routing splits as
`node=[user, volunteered] edge=[participated, ran]`. Edge scan returns 9
event edges; the top three are exactly Q16's three target events:
`Run_For_The_Cure_Event` / `Bike_a_Thon_Charity_Event` /
`Dance_For_A_Cause_Event` — each carrying the `participated_in` action and
`occurred_time` modifier the answer needs.

Full stg-engine suite: 1054 passed.

### Reference

- Design: `Semantic-Kernel-of-Consciousness/development/design/STG_R6_EDGE_FALLBACK_SEED_DESIGN.md` v0.1
- Builds on: v0.4.0a1 R1+R2+R5+R7 (Precision Recall)

## [0.4.0a1] — 2026-04-26

### Added — Precision Recall

Default `propagate` behavior upgraded with four cooperating mechanisms that
sharpen retrieval precision without filtering any edges or losing recall.
All four are pure postprocessing / wrappers around the existing pipeline —
**`engine.propagate`, the Rust core, `gravity.py`, and
`aggregate_to_communities` are unchanged.**

- **R1: Recency × supersede soft weight** — every activated node's stored
  activation is multiplied by the maximum in-subgraph edge weight, where
  edge weight = `salience × exp(-age × ln2 / halflife) × supersede_decay`.
  `superseded_at` edges are softly down-weighted (default factor `0.3`),
  **never filtered** — the memory-never-vanishes principle from
  `STG_PRECISION_RECALL_DESIGN.md` §2.4.
- **R2: Multi-seed chain intersection** — when a query tokenizes to ≥2
  effective tokens, each token runs through the existing propagate
  separately. Each token's chains are reconstructed via the existing
  `STLGraph.extract_chains()`. The output is the node intersection across
  every token's chain set (with union fallback if intersection is empty).
  Cross-topic word collisions ("dog food" → Max + Food_Recovery) are
  cleanly separated.
- **R5: active_context anchor** — nodes selected via `stg select` are
  loaded from the existing `active_context` SQLite table (with TTL filter,
  default 30 minutes) and their elevation in the live `GravityMap` is
  temporarily boosted (default `+5.0`) for the duration of the propagate
  call. Restored via context manager — even on exception. Nodes outside
  the GravityMap are silently skipped.
- **R7: Community dominance ratio** — communities whose `score` falls
  below `dominant.score / ratio` (default `3.0`) are folded out of the
  output. **Communities containing `query_seeds` are always preserved**
  regardless of score, so precise query hits are never hidden when
  representative activations happen to be zero.

### Added — Public API

- **`stg_engine.recall`** — new module with `apply_recency_weight`,
  `community_dominance_filter`, `multi_seed_propagate`,
  `context_anchor_boost` (contextmanager), and `_split_tokens`.
  Default tuning constants `DEFAULT_RECENCY_HALFLIFE_DAYS=30`,
  `DEFAULT_SUPERSEDE_DECAY_FACTOR=0.3`, `DEFAULT_DOMINANCE_RATIO=3.0`,
  `DEFAULT_ACTIVE_CONTEXT_BOOST=5.0`,
  `DEFAULT_ACTIVE_CONTEXT_TTL_SECONDS=1800`.
- **`feedback_select.load_active_context(ttl_seconds=...)`** — new
  optional argument filters out entries older than the cutoff.
  Backward compatible: omitting the argument keeps prior behavior.

### Added — CLI flags (escape hatches)

All new behavior is on by default. Flags below disable individual
mechanisms:

```
stg propagate <query> --no-recency-weight    # disable R1
stg propagate <query> --no-community-filter  # disable R7
stg propagate <query> --no-context-anchor    # disable R5
stg propagate <query> --no-multi-seed        # disable R2
stg propagate <query> --legacy               # all four disabled (≡ pre-0.4 behavior)
```

CLI also surfaces two informational lines when relevant:
- `🔗 multi-seed chain intersection: [token1, token2] → N nodes`
- `⚓ active_context anchored: [node1, node2, ...]`

### Memory-never-vanishes guarantee

There is **no** `--include-superseded` flag because superseded edges are
never excluded. Recall completeness is preserved — the four mechanisms
adjust ranking and grouping, not membership. To the user, the only
visible change in default mode is sharper output, never missing data.

### Tests

`tests/unit/test_recall.py` — 38 unit tests covering edge weight math,
half-life sanity (30d → ×0.5, 60d → ×0.25), supersede decay, dominance
ratio + query_seeds protection, anchor boost lifecycle (lifts / restores
on exit / restores on exception / case-insensitive / multi-anchor),
token splitting (stop-words / underscores / hyphens / CJK / punctuation).

End-to-end LongMemEval regression on `longmemeval-2026-04-25.bak`
(1539 nodes / 3903 edges):
- `dog food`: 5 communities, 19 nodes (legacy) → 1 community, 5 nodes
  (new). Cross-topic noise (`food_recovery`, `hawaii`, `senior_services`)
  removed.
- `bike repair`: precise hits (`User_Bike_Repair`, `Bike_Repair`) preserved.
- `Hawaii` (single token): recall completeness 9/9 nodes maintained
  while R1 softly attenuates ranking.

Full stg-engine test suite: 1040 passed (plus 1 pre-existing platform
test — Windows path on Linux — that is independent of these changes).

### Reference

- Design: `Semantic-Kernel-of-Consciousness/development/design/STG_PRECISION_RECALL_DESIGN.md` (v0.3)
- Diagnostic background: `Semantic-Kernel-of-Consciousness/research/STG_RETRIEVAL_NOISE_AND_BRAIN_ANALOGY.md`

## [0.3.0a3] — 2026-04-22

### Added — Skill Executor

- **`stg use <skill_name> [args...]`** — invoke a Skill node's script directly.
  Skill nodes live in the `Skill:` namespace and expose an executable script
  via the `path` modifier. `stg use` looks up the skill, validates it, and
  runs the script via subprocess with timeout, output cap, and audit logging.
- **`stg skill {list,show,use,configure,history}`** — new subcommand family
  for managing and running skills.
- **New reserved modifiers on Skill edges:** `executable`, `interpreter`,
  `args_template`, `stl_io`, `timeout_s`, `allow_root_override`. See
  `SKILL_INVOCATION_FIELDS` in `stg_engine.engine`.
- **`propagate skill` / `propagate --namespace=Skill`** — renders a catalog
  of skills instead of the usual community-grouped output. Case-insensitive
  `^skills?$` match.
- **`stg config set skill.*`** — new dotted-key support in `stg config`.
  Keys: `skill.enabled` (master switch, default `false`), `skill.roots`
  (whitelisted script directories), `skill.interpreters.<name>` (named
  binary paths), `skill.default_timeout_s`, `skill.output_cap_bytes`.
- **`skill_invocations` SQLite table** — every `stg use` call writes one
  audit row (rolling cap 10000). Queryable via `stg skill history`.

### Security

Execution requires **double opt-in**:
  1. `stg config set skill.enabled true` in user config, **AND**
  2. `executable="true"` modifier on the Skill edge.

Additionally the script path must resolve (after symlink resolution) under
at least one `skill.roots` entry. Defaults are empty — fresh installs cannot
`stg use` anything until the user configures roots explicitly. Subprocess
invocation uses `list` args (never `shell=True`) and a hard timeout.

### Multi-user design

No hardcoded paths or magic interpreter names. Builtin interpreters
(`python3`, `python`, `bash`, `sh`, `node`) resolve via `shutil.which` for
cross-platform portability. User-defined names live in
`skill.interpreters.<name>`. Users retrofit existing Skill edges via
`stg skill configure <name> --executable --interpreter ... --args-template ...`.

### Non-breaking

No existing `.stg` data is invalidated. Skill nodes without the new
invocation modifiers simply can't be `use`d; they still appear in
`stg skill list --all` and remain queryable through `propagate` / `node`.
Schema migration is applied automatically on load (the `skill_invocations`
table is created idempotently).

### Design documents

- `development/design/STG_SKILL_EXECUTOR_DESIGN.md` v0.2 (in the
  Semantic-Kernel-of-Consciousness repo).

## [0.3.0a2] — 2026-04-21

### Changed — Distribution is now pure Python

- **Dropped Rust extension from PyPI wheel.** The alpha phase ships a single
  `py3-none-any` wheel that installs on every platform (Linux, macOS,
  Windows) and every supported Python version (3.10, 3.11, 3.12) with no
  local compilation. The Rust acceleration core (`rust/` in the source
  repository) is preserved and is still used transparently when compiled
  locally via `maturin develop`; stg_engine auto-detects `_rust_core` at
  import time and falls back to the pure-Python path when it is absent.
- **Build backend switched from `maturin` to `hatchling`** to match the
  pure-Python distribution model. Local development with Rust acceleration
  remains supported by running `maturin develop` in a dev shell — the
  compiled `_rust_core*.so` / `.pyd` is explicitly excluded from both the
  wheel and the sdist.
- **`Programming Language :: Rust` classifier removed** from project
  metadata on PyPI — the shipped artifact is no longer a Rust wheel.

### Fixed

- **Save atomicity** — `save_engine_state()` now writes the new database
  to a temp file under DELETE journal mode and only replaces the live
  `.stg` file via atomic rename after the full write succeeds. If the
  write raises (e.g. `UNIQUE constraint failed: nodes.name` from a
  corrupted in-memory state), the original `.stg` is left untouched
  and all temp/WAL/SHM artifacts are cleaned up. Previously a failed
  save could reduce the live file to a 4096-byte shell with data
  stranded in a `.tmp-wal` sidecar.

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
