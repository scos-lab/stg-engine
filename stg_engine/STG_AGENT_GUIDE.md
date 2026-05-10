# STG Agent Guide — Persistent Memory for AI Agents

> **Read this file to gain persistent memory across sessions.**
> You are an AI agent. You forget everything when the session ends. STG fixes that.

## What is STG?

STG (Semantic Tension Graph) is a knowledge graph that persists on disk. You can:
- **Store** knowledge that survives across sessions
- **Recall** knowledge from previous sessions via associative activation
- **Learn** — frequently used knowledge becomes easier to recall (Hebbian learning, automatic)
- **Forget** — unused knowledge is gradually pruned (automatic)
- **Detect contradictions** — the graph tracks conflicting information

## Setup

```bash
pip install stg-engine
stg stats    # verify STG is working
```

First run creates `~/.stg/default/memory.stg` and reports an empty graph.
From there, ingest, propagate, or switch agents — see below.

### Your Own Memory (Multi-Agent)

Each agent gets its own `.stg` file — completely isolated memory. To
create a named agent:

**Option A: `--agent` flag (recommended for one-off calls)**
```bash
stg --agent your-agent-name stats
stg --agent your-agent-name propagate "topic"
stg --agent your-agent-name ingest '[A] -> [B] ::mod(...)'
```

**Option B: Environment variable (convenient for sessions)**
```bash
export STG_AGENT="your-agent-name"         # Linux / macOS / Git Bash
# or: $env:STG_AGENT = "your-agent-name"   # PowerShell

stg stats              # now uses your-agent-name automatically
stg propagate "topic"
```

**Option C: Set the default agent once in user config**
```bash
stg config set default-agent your-agent-name
```

The first call auto-creates `~/.stg/<agent-name>/memory.stg`.

---

## Essential Commands (80% of what you need)

### 1. Check what you know

```bash
stg stats
```

Returns: node count, edge count, Ψ (stability metric), active tensions.

### 2. Recall knowledge about a topic

```bash
stg propagate "your topic here"
```

Returns: a ranked list of related nodes with activation scores. Higher score = more relevant. **Gravity is ON by default** — hub nodes (high structural importance) are boosted, fragment nodes are suppressed.

**Use this BEFORE reasoning about project-specific topics.** Don't guess — check.

**Flags:**
- `--coarse` — project-level granularity (fewer, broader communities → only major hubs surface)
- `--fine` — detail-level granularity (more, smaller communities → more detail nodes surface)
- Default is `medium` (phase-level granularity)
- `--no-gravity` — disable gravitational reweighting entirely (faster, raw activation scores)

**After propagate, mark useful results:**
```bash
stg select 1,3,5    # reward selected nodes, penalize unselected
```
This adjusts edge salience so future propagates return better results.

### 3. Store new knowledge

```bash
stg ingest '[SourceConcept] -> [TargetConcept] ::mod(confidence=0.85, rule="empirical", description="what this means")'
```

After ingesting, STG automatically runs an internal propagate using the new node names and shows candidate communities for context binding:

```
Candidate communities for context binding:
    1. OAuth_Module (A=0.48)
    2. Auth_Architecture (A=0.37)
    3. API_Gateway (A=0.31)
    ...
Use `bind 1,3,5` to link new node(s) to selected candidates.
```

**Select relevant candidates to embed the new node into the graph:**
```bash
stg bind 1,2    # creates virtual edges from candidates to the new node
```

This prevents new nodes from being isolated endpoints. Use `--no-link` to skip candidate suggestion (for batch/script use).

**Syntax rules:**
- Anchors in `[BracketsLikeThis]` — PascalCase, Underscore_Case, or native language (CJK supported), no spaces
- Arrow `->` means "relates to" (directional)
- `::mod(...)` attaches metadata

**Always include four things:**
1. `confidence` (0-1, how true)
2. `rule` (what kind of claim)
3. `description` (what it means in plain language)
4. **A meta semantic field** (what kind of relation — see below)

**Rule types:**
- `"definitional"` — X is defined as Y (confidence usually 0.95+)
- `"causal"` — X causes Y (add `strength=0.8`)
- `"empirical"` — learned from experience (add `lesson="what we learned"`)
- `"logical"` — X implies Y through reasoning

### Meta semantic fields — required, one per edge

Every edge **must** carry at least one of these nine fields. They tell STG what *kind* of relation it is — not just metadata, but the edge's semantic spine. Empty-shell edges (description + confidence only, no semantic field) are bad ingest hygiene; choose the most specific field that fits.

| Field | Use when the edge expresses... | Example value |
|---|---|---|
| `is_a` | classification, type membership | `is_a="scheduling_algorithm"` |
| `action` | something happens / something causes / something does | `action="triggers"`, `action="took"` |
| `role` | functional role one node plays for another | `role="mentor"`, `role="entry_point"` |
| `status` | a state the source is in toward the target | `status="had_amazing_time"`, `status="deceased"` |
| `phase` | temporal phase of a process | `phase="initialization"`, `phase="cleanup"` |
| `relation` | generic relation (use only when nothing more specific fits) | `relation="depends_on"` |
| `type`, `kind`, `predicate` | legacy synonyms, accepted but prefer `is_a` / `action` / `role` | — |

**Rule of thumb when picking:**
- "X *is* / *is a kind of* Y" → `is_a`
- "X *does* / *causes* / *took* / *triggers* Y" → `action`
- "X *plays role* in Y / *acts as* Y" → `role`
- "X *is in state* Y / *enjoyed* Y / *failed at* Y" → `status`

These fields are also what STG uses for **supersede detection** (see "Knowledge evolution" below). Two edges with the same source and same `(meta_semantic_field, value)` pointing at *different* targets is the only configuration that signals an actual correction.

**Before ingesting:** Check if similar nodes already exist:
```bash
stg query "ConceptName"
```
Reuse existing node names rather than creating new ones for the same concept.

**Knowledge evolution (multi-edge):** You can ingest the same `[Source] -> [Target]` multiple times with different content. STG handles this automatically:
- **Different content** (different description, confidence, rule, or strength) → both edges are kept and both stay active. They are treated as **complementary facets** of the same relationship, not as corrections. Lookup points to the newest, but propagation considers all of them.
- **Identical content** → true duplicate, silently skipped.

```bash
# Two complementary facets of the same trip — both alive after ingest
stg ingest '[User] -> [San_Francisco] ::mod(confidence=0.95, rule="empirical", action="took", description="User took a 5-day trip to SF on 2023-03-27")'
stg ingest '[User] -> [San_Francisco] ::mod(confidence=0.95, rule="empirical", status="had_amazing_time", description="User reported having an amazing time during the trip")'
```

**When does an edge get marked superseded?** Only when STG sees an actual **correction**: same source + same `(meta_semantic_field, value)` pair, but a **different target**. That's the only configuration that signals "the previous answer was wrong, here's the new one." Same-target multi-edges (above) are never auto-flagged — they coexist as facets.

```bash
# Real supersede — User had teacher A, then switched to teacher B (same field+value, different target)
stg ingest '[User] -> [Teacher_A] ::mod(confidence=0.9, role="mentor", description="initial mentor")'
stg ingest '[User] -> [Teacher_B] ::mod(confidence=0.9, role="mentor", description="new mentor as of 2026-04")'
# → [User]->[Teacher_A] now carries suspected_supersede=True, superseded_by="Teacher_B"
```

Supersede is a **soft signal**, not ground truth: superseded edges aren't deleted, they're down-weighted by 0.3× in recall and frozen from Hebbian reinforcement. Your authoritative signal for "what's current" is still `occurred_time` plus the description content.

**Memory consolidation (merge/consolidate) — restricted access:**

`merge` and `consolidate` are memory tidying tools. They are **not for daily use** — only during dedicated memory consolidation sessions. Access control will be added in the future.

```bash
# ⛔ These commands are restricted to memory consolidation sessions only
stg merge '[Auth_Module] -> [Login_Bug] ::mod(path="src/auth/login.py")'
stg consolidate --all --dry-run
stg consolidate --all
```
Note: edges with different `occurred_time` values are never merged (they represent different events).

**Bulk ingest from file:**
```bash
stg ingest-file path/to/knowledge.md    # .md, .txt, .stl — any text file works
```

**Granularity — what should be a node?**

Nodes are anchor points for search and propagation. Edges carry the actual content in their descriptions. The decision criterion: **would you ever search for this, or propagate from it?** If yes → make it a node. If no → put it in an edge's description.

Three-layer guide:

| Layer | What qualifies | Make a node? | Examples |
|-------|---------------|-------------|---------|
| **Entity** | Has a name, has relationships, appears repeatedly | Yes, always | A person, a place, a product, a company |
| **Event** | Has causes/effects, drives a narrative or process | Yes | A bug report, a decision, a meeting outcome |
| **Detail** | One-time description, context, supporting info | No — put in edge description | A specific error message, a minor object, a passing observation |

The edge's `description` field can hold arbitrarily long text — use it generously. Nodes should be specific and reusable. `[Login_Bug]` is good. `[Thing]` or `[Issue]` is too vague. Don't create a node for something you'll never search for.

For non-English content, use native-language node names: `[贾宝玉]` not `[Jia_Baoyu]`. Propagate supports CJK substring matching — searching "宝玉" will find `[贾宝玉]` and `[通灵宝玉]`.

### 4. Run a registered capability (`stg use`)

```bash
stg use <skill_name> [args...]
```

A **Skill** is a node in the `Skill:` namespace whose edge points at an
executable script. Once configured (see below), `stg use` resolves the
skill, applies security gates, runs the script under subprocess with
audit + timeout + output cap, and streams stdout back to you.

See the dedicated **Skills** section below for the full walkthrough
(using, making, security model, multi-user portability).

### 5. End-of-session cleanup

```bash
stg feedback session-end
```

Run this before the session ends. It prunes unused edges, saves telemetry, and creates co-activation links.

---

## When to Use Each Command

| Situation | Command | Why |
|-----------|---------|-----|
| Starting a new session | `propagate "main topic"` | Load relevant context from past sessions |
| About to claim something about project history | `propagate "the claim"` | Verify before confabulating |
| Propagate returned good results | `select 1,3,5` | Reward useful nodes, improve future recall |
| Made a decision worth remembering | `ingest '[Decision] -> [Reason] ::mod(...)'` + `bind` | Persist and embed in context |
| Learned why something failed | `ingest '[Failure] -> [Cause] ::mod(rule="empirical", lesson="...")'` + `bind` | Don't repeat mistakes |
| Something seems contradictory | `tensions active` | Check known contradictions |
| Want to add info to an existing edge | ⛔ `merge` (restricted) | Only during memory consolidation sessions |
| Too many multi-edges accumulated | ⛔ `consolidate` (restricted) | Only during memory consolidation sessions |
| Want to understand graph structure | `gravity info` / `gravity node <name>` | See community structure, elevation |
| Session is ending | `feedback session-end` | Cleanup, persist, clear active context |
| Need to find a specific fact or detail | `grep "keyword"` | **Fast text search across all edge descriptions — use this first** |
| Need to find a specific concept by name | `query "name_fragment"` | Pattern-match node names |
| Need to understand how two concepts connect | `paths SourceNode TargetNode` | Find relationship chains |

---

## Communities — How Knowledge Is Organized

STG automatically groups nodes into **communities** — clusters of densely connected nodes that form a topic area. Think of communities like chapters in a book or sections in a library: they tell you *which domain* a piece of knowledge belongs to.

Communities are detected automatically (Louvain algorithm) at three resolutions:
- **coarse** — broad project-level areas (e.g., `density_monism`, `website_factory`)
- **medium** — phase-level topics (default, used in search results)
- **fine** — detail-level subclusters

Each community is named after its **top representative node** — the most structurally important node in that cluster. You don't need to name or manage communities manually.

### Community labels in search results

All search commands (`propagate`, `query`, `grep`, `search`) show a community label in brackets after each result:

```
propagate('Koide') → 4 nodes:
    1. Prediction_Tau_Mass (A=0.750)  [apply_deltas]
    2. Koide_Formula (A=0.440)  [apply_deltas]
    3. Density_Monism (A=0.320)  [density_monism]
    4. Heartbeat_System (A=0.090)  [specification]
```

The label `[apply_deltas]` means that node belongs to the `apply_deltas` community. Use this to:

- **Quickly scan which domains are represented** in your results — if all results say `[density_monism]`, your query hit one topic area. If you see `[density_monism]` + `[specification]` + `[website_factory]`, the query spans multiple domains.
- **Spot noise** — a result from `[specification]` in a query about physics is probably a structural hit, not a semantic one. You can mentally filter it out.
- **Understand graph structure** — nodes in the same community are densely interconnected. Nodes in different communities are connected through bridge nodes.

### Community types

Different kinds of knowledge form different kinds of communities. Knowing the type helps you choose the right ingest strategy and interpret search results.

| Type | What it contains | Ingest characteristics | Example community names |
|------|-----------------|----------------------|------------------------|
| **Book / Literature** | Knowledge decomposed from books or papers | High confidence, has `source`/DOI, stable over time | `sdm_hopfield_equivalence`, `kanerva_chapter_3` |
| **News / Events** | Time-bound events, announcements | Has `occurred_time`, may become stale, moderate confidence | `gpt_5_2_release`, `bushfire_season_2026` |
| **Daily life / Experience** | Personal episodes, routines | Episodic, time is primary axis, rich in `lesson` | Named by date or activity |
| **Project** | Ongoing work with phases and milestones | Has status, evolves across sessions, crosses many topics | `website_factory`, `stl_bridge` |
| **Theory / Concept** | Abstract frameworks, definitions | `rule="definitional"`, high confidence, referenced by other communities | `density_monism`, `epistemological_humility` |
| **Skill / Procedure** | Executable knowledge, how-to | Has `cause`/`effect`, `path` pointing to scripts | `website_factory_deploy`, `update_model_catalog` |
| **People / Organizations** | Entity relationships | Person/org as hub node, social/professional edges | `openai`, `anthropic` |
| **Code / Implementation** | Software structure | Class/function/module names, maps to spec communities | `stgengine_definition`, `class_definition` |
| **Identity / Cognition** | Agent's own cognitive state | Meta-cognitive, affects behavior not external knowledge | `separate_thinking_from_speaking` |

**Why this matters for ingest:**
- **Books/Theory**: decompose carefully, reuse concept nodes across chapters → forms one large community
- **News**: always include `occurred_time`, expect the community to grow as follow-up articles share nodes
- **Daily life**: use dates in node names or `occurred_time` modifier for temporal queries later
- **Projects**: reuse project-name nodes across sessions → all project work clusters automatically
- **Skills**: use `Skill:` namespace prefix, include `path`/`cause`/`effect` so the knowledge is actionable

**Why this matters for search:**
- A result from a **theory** community is likely a stable definition — trust it
- A result from a **news** community may be outdated — check the `occurred_time`
- A result from a **code** community in a conceptual query is probably noise — skip it
- A result from a **project** community tells you this knowledge is work-related, not theoretical

### Inspecting communities

```bash
stg gravity info                   # List all communities with names and sizes
stg gravity node <name>            # Which community does this node belong to?
```

---

## Retrieval: Five Ways to Search STG

STG has five search commands. Each sees different things. Choosing the right one matters.

### The Five Commands

| Command | Matches | Searches | Speed | Use when |
|---------|---------|----------|-------|----------|
| **`query <pattern>`** | Substring on **node names** | Node names only | Instant | You know (part of) the node name |
| **`grep <pattern>`** | Regex on **all text** | Descriptions, lessons, node names | Fast | You know a keyword from the content |
| **`propagate <text>`** | Token match + **graph traversal** | Node names → follows edges 2-3 hops | Fast | You want associated knowledge, not just matches |
| **`search <query>`** | **Embedding similarity** | All nodes by semantic meaning | Slow | You want conceptually related nodes, even with different names |
| **`converge <query>`** | Iterative propagation | Repeated activation spreading | Medium | Vague query, need to explore |

### What each command finds (example: searching "apple")

Assume the graph has nodes: `[Apple]`, `[Apple_Inc]`, `[apple_red]`, `[iPhone]`, `[Fruit]`, and an edge description mentioning "apple harvest season".

```
query "apple"     → [Apple], [Apple_Inc], [apple_red]
                    (node names containing "apple")

grep "apple"      → above + the edge about "apple harvest season"
                    (all text containing "apple")

propagate "apple" → [Apple], [Apple_Inc], [apple_red]
                    + their graph neighbors (whatever they connect to)
                    (name match → then follows edges)

search "apple"    → all of the above
                    + [iPhone], [Fruit] (semantically related, no name match)
                    (embedding knows apple ↔ iPhone ↔ fruit)
```

### Key insight: propagate and search see different dimensions

- **Propagate** is faithful to **graph structure**. It follows edges regardless of names. If `[Amplitude]` connects to `[Music]` but not `[Physics]`, propagate sends activation to Music — even though "amplitude" sounds like physics. Propagate shows you where a node **actually sits** in the knowledge graph.

- **Search** is faithful to **semantic meaning**. It uses embeddings to find conceptually similar nodes across the entire graph, even if they have no edges between them. It can find `[Wavelength]` when you search "wave", even if they're completely disconnected in the graph.

Neither is "better" — they answer different questions:
- "What does the graph know about X?" → **propagate**
- "What concepts are related to X?" → **search**

### Recommended Workflow: Grep → Propagate → Node

For most retrieval tasks, this three-step process works best:

**Step 1: Grep — Find the entry point**
```bash
stg grep "GPS malfunction"
# → [User_Car_GPS_Issue] -> [Replaced_At_Dealership]
#   "User's car GPS malfunctioned on 3/22, requiring dealership replacement"
```
Fast regex match across all text. Gives you the exact edge and its nodes.

**Step 2: Propagate — Expand via graph**
```bash
stg propagate "User_Car_GPS_Issue"
# → [Car_Serviced_March_15], [Honda_Civic], [Dealership_Visit], ...
```
Starting from the node found in Step 1, discover associated knowledge through edges.

**Step 3: Node — Inspect details**
```bash
stg node Car_Serviced_March_15
# → All edges with timestamps, descriptions, confidence, history
```

**Why not skip to search?** Search is slower (loads embedding model) and ranks by semantic similarity, which may miss exact matches. Grep → Propagate is faster and more precise for known keywords. Use `search` when grep returns nothing — it can find conceptual matches that text search misses.

### Tips for effective retrieval
- **grep**: Try multiple keywords and synonyms. Use regex: `grep "5K.*time|personal best"`. You (the LLM) are better at choosing search terms than any embedding model.
- **query**: Use when you remember the node name but not exactly. `query "OAuth"` finds all OAuth-related nodes.
- **propagate**: Best for "what do I know about this topic?" — gives you the neighborhood, not just matches. Also triggers Hebbian learning (strengthens recalled paths).
- **search**: Best for "is there anything related to this concept?" — casts the widest net.
- **converge**: Best for vague queries where a single propagation isn't enough.

### Dual-anchor retrieval — propagate's "primary answer" mode

When you know the **two specific nodes** the answer connects, drop them both into one `propagate` call. STG detects this and switches to a focused view that returns the answer in 3-6 lines instead of the usual 50-80.

```bash
# Single-bag propagate — returns ranked list, you scan it
stg propagate "User volunteered Food For Thought charity"
#   📍 exact anchors: [User, Charity_Events]
#   ... 50-80 lines of node activations + edge previews ...

# Dual-anchor propagate — when ≥2 query chunks exact-match canonical node names
stg propagate "User Food_For_Thought_Charity_Gala"
#   📍 exact anchors: [User, Food_For_Thought_Charity_Gala] (forced into result)
#   🎯 Anchor-pair edges (1) — primary answer:
#       [User] -[volunteer @2023-09-25]-> [Food_For_Thought_Charity_Gala]
```

**When dual-anchor triggers:** at least two query chunks (after token splitting) match canonical node names exactly. Underscore_or_NGram matching counts — a multi-word phrase like "Food For Thought" resolves to `Food_For_Thought_Charity_Gala` automatically (n-gram lookup, since 0.5.0a2). If only one chunk matches a node, the call falls back to single-bag propagate.

**When to use which:**

| Question shape | Use | Why |
|---|---|---|
| "When did User do X with EntityY?" / "How are User and EntityY connected?" | **Dual-anchor** (`propagate "User EntityY"`) | Direct edge between two known nodes, no ranking noise |
| "What do I know about EntityY?" | Single-bag (`propagate "EntityY"`) | You want the neighborhood, not one specific edge |
| "Aggregate / count / top-N" | Neither — use `node EntityY` and walk outgoing edges, or `temporal range` | Aggregation isn't a propagate task |
| "Vague topic, no canonical node names yet" | Single-bag, then `select` good results | Let propagate explore |

**Prerequisite: don't guess node names.** If you're not sure of the exact canonical name (`Samsung_Galaxy_S22` vs `User_Adrian_Smartphone_Samsung_Galaxy_S22`), run `query <pattern>` first to find the actual name, then dual-anchor with it. Wrong anchors → falls back to single-bag silently, you waste time.

**Empirical signal:** on the 2026-04-28 LongMemEval 20-question benchmark, single-bag and dual-anchor both achieved 20/20 recall — but dual-anchor returned 3-6 lines per question vs single-bag's 50-80 (47-line average savings, 1-5s → 0.3-0.4s).

### Edge-content scan covers all tokens (since 0.5.0a5)

When you call `propagate "User volunteered participated charity events"`, STG now scans **every token** against edge `description` / `action` / `role`, not just the tokens that don't match a node name. This recovers the case where a token (e.g. `volunteered`) happens to match an unrelated concept node like `Volunteering_Best_Practices` and would otherwise hide the actual fact buried in some edge's `description="User volunteered at..."`. IDF weighting handles the noise — common tokens contribute less, rare tokens dominate. The edge containing the real fact will show up under `🪢 Event-edge matches` with `🔗 double-hit` marking.

You don't need to do anything special — just write natural-language queries. The token-routing is automatic.

---

## Example Session

```bash
# Start: what do I know about OAuth?
stg propagate "OAuth authentication"
# → 1. OAuth_Module  2. Auth_Architecture  3. API_Gateway  ...

# Mark useful results (adjusts salience for future recall)
stg select 1,2

# Found relevant nodes. Now work on the task...
# ... (conversation happens) ...

# Learned something new: refresh tokens expire in testing mode
stg query "OAuth"  # check existing nodes first
stg ingest '[OAuth_Testing_Mode] -> [Token_Expiry_7Days] ::mod(confidence=0.95, rule="empirical", lesson="Refresh tokens in Google OAuth testing mode expire after 7 days. Must publish app to Production to get permanent tokens.")'
# → Candidate communities:
#     1. OAuth_Module (A=0.48)
#     2. Token_Management (A=0.31)
#     ...

# Bind new node to relevant communities
stg bind 1,2

# End of session
stg feedback session-end
```

---

## Ingesting Articles & Documents

When you need to ingest external content (articles, blog posts, news, papers) into STG, use a two-step process: **decompose** the content into structured STL statements, then **batch ingest**.

### Why not just `import-doc`?

`import-doc` dumps the full text into a single edge's description. This preserves the original text, but:
- **propagate can't find it** — the node name is a filename (`Blog_some_article`), not a concept
- **no graph structure** — one edge, two nodes, nothing to traverse
- **no cross-document connections** — each article is an isolated island

Decomposing into STL creates searchable nodes that participate in graph propagation and automatically bridge to existing knowledge via shared node names.

### The workflow

```
Article → Read & decompose → Write STL statements to file → Validate → ingest-file
```

**Step 1: Read the article and identify key knowledge units**

For each article, extract:
- **Claims** — what does it assert? (factual, theoretical, opinion)
- **Relationships** — what causes/implies/contradicts what?
- **Entities** — people, systems, concepts that should be reusable nodes

**Step 2: Check existing nodes**

Before writing STL, query the graph for existing related nodes:
```bash
stg query "relevant_concept"
```
**Reuse existing node names** — this is how cross-document bridging happens automatically (via sibling/co_source virtual edges).

**Step 3: Write STL to a file**

Write all statements into a single `.md` or `.txt` file. Multiple statements in one file:

```markdown
[RAG] -> [No_Learning] ::mod(confidence=0.90, rule="empirical", source="Chen 2026 blog", description="RAG retrieves same chunks regardless of usage frequency. No Hebbian-style reinforcement")
[RAG] -> [Chunk_Boundary_Problem] ::mod(confidence=0.88, rule="empirical", source="Chen 2026 blog", description="Retrieval boundary cuts through semantic units, losing cross-chunk context")
[Knowledge_Graph] -> [RAG_Alternative] ::mod(confidence=0.85, rule="causal", strength=0.8, source="Chen 2026 blog", description="KGs solve structural reasoning but need weighted edges, temporal awareness, and contradiction preservation")
```

**Step 4: Ingest the file**

```bash
stg ingest-file path/to/extracted.md
```

This ingests all statements at once and automatically creates sibling/co_source virtual edges between nodes that share parents or targets.

### Confidence calibration by claim type

Not all content in an article is equally reliable. Calibrate confidence based on the nature of the claim:

| Claim type | confidence | When to add certainty | Example |
|------------|-----------|----------------------|---------|
| **Verified fact** (data, citation, consensus) | 0.90-0.98 | Not needed | "GPT-5.2 has a 2M context window" |
| **Author's analysis/argument** | 0.75-0.85 | Not needed | "The future lies in weighted graphs + Hebbian learning" |
| **Theoretical/speculative** | 0.60-0.75 | Optional | "This equivalence may extend to transformer attention" |
| **Subjective claim / user-reported** | 0.85-0.95 | **Yes** — add `certainty=0.3-0.5` | "I experienced healing through meditation" |

**When to use `certainty`:** Only when confidence and your judgment diverge — the source reliably reports something (`confidence` high), but you judge the content unlikely to be objectively true (`certainty` low). This happens with UserClaimed and CosmicTrace content. For everyday factual content, `confidence` alone is sufficient.

### Decomposition by content type

| Content type | Focus on extracting | Typical edges per article |
|-------------|--------------------|----|
| **News** | Events, actors, numbers, dates. Use `occurred_time=` for event dates | 3-5 |
| **Blog/opinion** | Claims, arguments, cited evidence. Track `source=` to the author | 4-8 |
| **Paper/abstract** | Theorems, results, methods, metrics. Use `source=` with DOI | 4-7 |
| **Tutorial/how-to** | Procedures, dependencies, gotchas. Use `lesson=` for key takeaways | 3-6 |

### Cross-document bridging (automatic)

You don't need to manually create bridges between articles. STG does this automatically:

- **Same source node** → targets become **siblings** (virtual edge)
- **Same target node** → sources become **co_source** (virtual edge)

Example: Blog says `[Persistent_AI_Memory] -> [Hebbian_Learning]`, paper says `[Persistent_AI_Memory] -> [SDM_Hopfield]`. Because they share the source `[Persistent_AI_Memory]`, STG auto-creates a virtual edge between `[Hebbian_Learning]` and `[SDM_Hopfield]`.

**The key: reuse node names across articles.** Query before you ingest.

---

## Skills — running scripts from STG

*Introduced in v0.3.0a3.*

STG can act as a **capability registry**: register a script as a
`Skill:`-namespaced node, then invoke it by name. This is useful when an
agent builds up a palette of tools over time and wants them accessible
through the same interface it uses to recall knowledge.

### One-time opt-in

Skill execution is **disabled on fresh installs** — nothing can run until
you configure two keys:

```bash
# Master switch — off by default
stg config set skill.enabled true

# Whitelist the directories where your scripts live. Only scripts whose
# resolved path (after symlinks) falls under one of these directories can
# be executed. Multiple paths: comma-separated.
stg config set skill.roots "/home/you/my-tools,/home/you/workshop"
```

If either is missing, `stg use` prints a concrete error with the exact
command to run next.

Optional but convenient: name an interpreter you reuse across several
skills, so individual skill edges don't need an absolute interpreter path:

```bash
stg config set skill.interpreters.myvenv "/home/you/proj/.venv/bin/python3"
```

Named interpreters resolve first via your config; unknown names fall back
to the builtins `python3`, `python`, `bash`, `sh`, `node` (via
`shutil.which`, cross-platform). Any other string must be an absolute
path to a runnable binary.

### Running a skill

Once at least one Skill edge has `executable=true` and its script lives
under `skill.roots`:

```bash
# Positional args pass verbatim to the script
stg use My_Skill foo --option bar

# Resolve and print the command line without executing
stg use My_Skill foo --dry-run

# Override the declared timeout
stg use My_Skill foo --timeout 120

# Capture structured result as JSON (useful when embedding `stg use`)
stg use My_Skill foo --json

# For skills that declared stl_io=true: pipe an STL block on stdin
stg use My_Skill --args-stl '[Arg:input] → [File] ::mod(path="/tmp/x")'
stg use My_Skill --args-stl-file /tmp/params.stl
```

Every invocation writes one row to the `skill_invocations` audit table.

```bash
stg skill history --limit 10
stg skill history --skill My_Skill --limit 20
```

### Browsing available skills

```bash
# Executable skills first, rendered as a catalog
stg skill list
stg skill list --filter reddit   # substring over name/path/desc
stg skill list --all             # include not-yet-configured skills

# propagate skill / propagate Skill — triggers the same catalog view
stg propagate skill

# Full detail + recent invocations on one skill
stg skill show My_Skill
```

### Making a skill (how to register one)

A Skill is just an STG edge whose source node is in the `Skill` namespace
and whose modifiers include the invocation fields.

**Template:**

```bash
stg ingest '[Skill:My_Skill] -> [Purpose_Target] ::mod(
  rule="empirical",
  confidence=0.95,
  description="one-line description of what the skill does",
  path="/abs/path/to/script.py",

  executable="true",
  interpreter="python3",
  args_template="<input> [--out PATH]",
  timeout_s="60",
  stl_io="false"
)'
```

**Required invocation fields:**

| Field | Purpose |
|---|---|
| `path` | Absolute path to the script. Must resolve under `skill.roots`. |
| `executable` | Must equal `"true"` for `stg use` to run it. |
| `interpreter` | Named (`python3`, `bash`, user-defined in `skill.interpreters.*`) or absolute path to a binary. |

**Recommended invocation fields:**

| Field | Purpose |
|---|---|
| `args_template` | Human-readable signature shown by `stg skill show`. |
| `timeout_s` | Per-skill timeout. Defaults to `skill.default_timeout_s` (60). Max 600. |
| `stl_io` | `"true"` if the script reads STL on stdin and emits STL on stdout. Default `"false"`. |
| `description` | Purpose surfaced by `skill list` and `propagate`. |

**Retrofit an existing Skill edge** (already ingested before the
invocation fields were added):

```bash
stg skill configure My_Skill \
    --executable \
    --interpreter python3 \
    --args-template "<input> [--out PATH]" \
    --timeout 60
```

This calls `stg merge` under the hood. If the Skill has multiple outgoing
edges, `configure` auto-selects the one with a `path=` modifier. Use
`stg merge` directly to target a specific non-primary edge.

### STL-first I/O (`stl_io=true`)

For skills that want structured round-tripping with an LLM caller:

**Script reads on stdin:**

```
[Invocation] → [Skill:Extract_Citations] ::mod(invocation_id="inv_abc", caller="agent")
[Arg:pdf_path] → [File] ::mod(path="/tmp/paper.pdf")
[Arg:max_refs] → [Int] ::mod(value="100")
```

**Script writes on stdout:**

```
[Result] → [Success] ::mod(invocation_id="inv_abc", elapsed_ms="3200", citations="47")
[Citation:1] → [DOI] ::mod(doi="10.1038/...")
[Citation:2] → [arXiv] ::mod(arxiv_id="2403.12345")
```

This keeps the LLM ↔ script boundary consistent with the rest of STG —
no JSON round-trips. Parse output via `stl_parser.validate_llm_output`
for 21 automatic repair rules against drifted formatting.

Free-form progress goes to **stderr**, not stdout — stdout is reserved
for the STL payload.

### Security model

Every `stg use` call passes through five gates, in order:

1. `skill.enabled = true` in user config — else exit 6.
2. Skill edge has `executable = true` — else exit 6.
3. Script path exists and is a regular file — else exit 6.
4. Resolved path (symlinks followed) is under at least one `skill.roots` entry — else exit 6.
5. Interpreter resolves to an executable binary — else exit 6.

Subprocess is invoked with a **list** of args (never `shell=True`), a
hard timeout (default 60s, max 600s), and a 10 MB stdout cap. Every call
writes one audit row to `skill_invocations` even if it fails.

**Exit-code taxonomy:**

| Code | Meaning |
|---|---|
| 0 | Skill ran and returned 0 |
| 3 | Skill not found |
| 4 | Skill ran but returned non-zero (forwarded) |
| 5 | Ambiguous — multiple equally-ranked edges |
| 6 | Security gate failed (see list above) |
| 7 | Timeout |
| 8 | STL parse error on skill stdout (when `stl_io=true`) |

### Multi-user portability

`skill.roots`, `skill.interpreters`, etc. are **per-user** config in
`~/.stg/config.json`. No hardcoded paths in the engine — a script at
`/home/alice/tools/foo.py` on Alice's machine won't work on Bob's until
Bob ingests his own Skill edge with his local path. This is intentional:
`.stg` files should not silently run scripts from other users' environments.

When distributing a tool (e.g., on GitHub), ship a registration snippet
in the README:

```bash
# In your tool's README
cd /path/to/cloned/tool
stg ingest '[Skill:Tool_Name] -> [Purpose] ::mod(
  rule="empirical", confidence=0.95,
  path="'"$(pwd)"'/script.py",
  executable="true", interpreter="python3",
  args_template="<input>",
  description="what it does"
)'
```

Users edit paths to match their setup. No templating machinery required.

### Typical workflow for an agent

```bash
# Session start: orient to what's runnable
stg propagate skill                            # catalog of registered skills

# Recall + run: ask STG, pick a skill, run it
stg propagate reddit                           # relevant Skill + Lessons
stg use Reddit_Pipe_Dispatcher \
       https://www.reddit.com/r/linux/comments/xyz
# → markdown of parsed thread, direct to stdout

# Record a new capability you just built
stg ingest '[Skill:New_Tool] -> [Target] ::mod(
  path="...", executable="true", interpreter="python3", ...
)'
```

An agent never has to remember absolute script paths — they live in
STG. `stg use <name>` becomes the single gesture for "call a registered
capability by name."

---

## Additional Commands (use as needed)

| Command | Purpose |
|---------|---------|
| `grep <pattern>` | **Fast text search across all edge descriptions, node names, and lessons. Supports regex. Use this first — it's faster and more precise than `search`.** |
| `node <name>` | Details about a specific node |
| `select 1,3,5` | Mark useful propagate results (reward/penalize salience) |
| `bind 1,3,5` | Link new ingest nodes to selected candidate communities |
| `gravity info` | Community structure overview (elevation stats, top/bottom nodes) |
| `gravity node <name>` | Node's elevation, community assignments, representative roles |
| `search <query>` | Semantic search (embedding-based, slower but finds conceptual matches) |
| `psi` | Graph stability metric |
| `tensions active` | Current contradictions |
| `importance --top 10` | Most important nodes |
| `learn path <n1> <n2> <n3>` | Explicitly strengthen a known-good path |
| `prune --dry-run` | Preview what would be pruned |
| `cognitive self-model` | Assessment of knowledge gaps and strengths |
| `converge <query>` | Iterative propagation for vague queries |
| `telemetry report` | Usage statistics |

---

## STL Reference

STL (Semantic Tension Language) is the wire format for ingesting knowledge into STG. The full specification lives at <https://github.com/scos-lab/semantic-tension-language>; this section is a working reference.

### Statement shape

```
[Source] -> [Target] ::mod(key=value, key=value, ...)
```

- **Anchor**: `[Name]` — PascalCase, Underscore_Case, or native-language (CJK supported); valid chars `A-Za-z0-9_-:`; no spaces, no nesting.
- **Namespace**: `[Project:Thing]` — colon separates namespace and name. Agents commonly use `Skill:`, `Memory:`, `Spec:`, `Lesson:`, `User:`, `Concept:`. One colon only — `[A:B:C]` is **not supported** by stl_parser; use a single namespace prefix.
- **Arrow**: `->` (or `→`) — directional. `[A] -> [B]` ≠ `[B] -> [A]`.
- **Modifier block**: `::mod(...)` — optional but in practice always required (you need at least confidence + a meta semantic field).

Multiple `::mod()` blocks on one statement are allowed and merge. Multiple statements per file are allowed; one statement per line is the convention.

### Modifier reference

Modifiers fall into three tiers. Pick by purpose, not by alphabetical order.

**Tier 1 — required on every edge**

| Modifier | Type | Purpose |
|---|---|---|
| `confidence` | float 0-1 | How true the claim is. See calibration table below. |
| One meta semantic field | string | What kind of relation: `is_a` / `action` / `role` / `status` / `phase` / `relation` / `type` / `kind` / `predicate`. See "Meta semantic fields" in §3. |

**Tier 2 — strongly recommended**

| Modifier | Type | Purpose |
|---|---|---|
| `description` | string | What this edge means in plain language. Without it, the edge is unreadable in 3 months. |
| `rule` | enum | `"causal"` / `"empirical"` / `"definitional"` / `"logical"`. What kind of claim this is. |
| `source` | string | Where the claim came from (URL, DOI, "wuko 2026-04-30", "session 042"). Required for high-confidence factual edges. |

**Tier 3 — situational**

| Modifier | Type | When to use |
|---|---|---|
| `strength` | float 0-1 | For `rule="causal"`: how strongly cause produces effect (independent of confidence). |
| `lesson` | string | For `rule="empirical"`: the takeaway. Distinct from `description` (which is what happened). |
| `path` | string | Pointer to a file: `path="/abs/path/to/spec.md"` or `path="src/auth.py:42"`. Used for STG-as-a-Pointer pattern. |
| `cause` / `effect` | string | For causal edges, name the named cause/effect (auxiliary to source/target nodes). |
| `domain` | string | Topic tag for filter/aggregate use. |
| `author` | string | Who recorded this. |
| `occurred_time` | datetime | When the event itself happened. See Timestamps section. |
| `created_at` | float (epoch) | Override engine's auto-set ingest timestamp (for backfilled datasets). |
| `certainty` | float 0-1 | Independent of confidence: agent's judgment of objective truth when the source is reliable but the content is doubtful (UserClaimed / CosmicTrace pattern). |

**Skill-only modifiers** (see §Skills): `executable`, `interpreter`, `args_template`, `timeout_s`, `stl_io`. Don't put these on non-Skill edges.

### Confidence calibration

| Range | When to use |
|---|---|
| 0.95-1.0 | Definitional truth, mathematical fact, direct citation |
| 0.85-0.94 | Strong-evidence factual claim, broadly accepted theory |
| 0.70-0.84 | General knowledge, moderate evidence |
| 0.50-0.69 | Possible but unconfirmed, limited evidence |
| 0.30-0.49 | Speculative, weak evidence |
| 0.00-0.29 | Highly uncertain, hypothetical |

Subjective claims by users (e.g. "I had an amazing time", "meditation healed me") get **high confidence** (the source reliably reported it) plus **low certainty** (you doubt it's objectively true). For everyday factual content, just use confidence; certainty is for the corner case.

### Four usage patterns for edges

| Pattern | What lives where | Key modifiers |
|---|---|---|
| **STG-as-a-Pointer** | Detail is in a `.md` file; STG stores a one-line summary + path | `path=`, `description=` (one sentence) |
| **STG-as-an-Event** | The full content fits on the edge | `description=` or `lesson=`, `occurred_time=`, `rule="empirical"` |
| **STG-as-a-Skill** | Edge is an executable script registration | `path=`, `executable="true"`, `interpreter=`, `args_template=`, `timeout_s=` (see §Skills) |
| **STG-as-a-Property-Carrier** | Self-loop edge holds node-identity attributes; not a relationship | `action="intrinsic_properties"`, free-form attribute keys, `rule="definitional"` |

Threshold: if your content is more than 2-3 sentences → write a `.md`, store the path. If a single `lesson=` clause says it → keep it on the edge. If it's a runnable operation → make it a Skill. If a node has many attributes that would otherwise repeat across every outgoing edge → use a Property-Carrier self-loop.

**Property-Carrier example:**
```
[Elden_Ring] → [Elden_Ring] ::mod(
  action="intrinsic_properties",
  appid="1245620", release_year="2022", price_usd="59.99",
  confidence=0.99, rule="definitional"
)
```

Runtime: such self-loops are excluded from `propagate` and gravity community detection — the engine treats them as storage-only attribute bags. They appear as a `Properties:` section in `stg node <name>`. See STL Operational Protocol §9.4 for the full contract.

### Validation

Before ingesting a batch, validate the syntax. The `stl` CLI (sibling tool, `pip install stl-parser`) provides:

```bash
stl validate path/to/extracted.stl     # syntax check
stl parse path/to/extracted.stl        # parse + show structure
```

Avoid pre-commit shortcuts that skip validation — invented `action` values and missing required modifiers slip through and pollute the graph.

---

## Temporal Queries (What happened when?)

Every edge in STG has a `created_at` timestamp. You can use this to look back in time — what was learned on a specific day, what happened around a specific concept, and reconstruct the sequence of thought.

### Browse by date

```bash
# What was ingested on a specific date?
stg temporal range 2026-03-22

# What was ingested in a date range?
stg temporal range 2026-03-18 2026-03-22

# What was happening around a specific concept (within N hours)?
stg temporal around OAuth_Testing_Mode 24

# Overall temporal statistics
stg temporal stats
```

`temporal range` is read-only — it just lists edges created in that timeframe, ordered by timestamp. Use it to answer "what did I learn yesterday?" without modifying the graph.

### Reconstruct thought sequences

If you need to understand the *order* things were learned (not just what), build a temporal chain:

```bash
# Build a temporal sequence for a specific day
stg temporal build 2026-03-22

# Build for a date range
stg temporal build 2026-03-15 2026-03-22

# Walk through the sequence
stg temporal replay <start_node>
```

`temporal build` creates pointer-chain edges in the graph (permanent). `temporal replay` walks those chains. Only build when you actually need the sequence — don't build routinely.

### When to use temporal queries

| Situation | Command | Why |
|-----------|---------|-----|
| "What did I learn yesterday?" | `temporal range <date>` | Browse by date, no side effects |
| "What was I working on around concept X?" | `temporal around <node> [hours]` | Context around a node's creation |
| "Walk me through how this understanding developed" | `temporal build <date>` + `temporal replay <node>` | Full sequence reconstruction |
| "How much temporal data is there?" | `temporal stats` | Overview of timestamps coverage |

---

## What NOT to Do

1. **Don't ingest trivial information.** "User said hello" is not worth storing. Store decisions, lessons, discoveries, connections.
2. **Don't create vague anchors.** `[Thing]`, `[Data]`, `[System]` — useless. Be specific.
3. **Don't ingest without `description`.** In 3 months, `[A] -> [B] ::mod(confidence=0.9)` is meaningless. The description is what makes it useful.
4. **Reuse existing node names.** Nodes auto-merge if the name matches, so `[OAuth]` always refers to the same node. Use `query` if you're unsure what name an existing concept uses.
5. **Don't use `import-doc` as your primary ingestion method.** It dumps full text into one edge — invisible to `propagate`, no graph structure, no cross-document bridging. Use the decompose workflow instead (see "Ingesting Articles & Documents" above). Reserve `import-doc` only for archival purposes where you need the original text searchable via `grep`.
   ```bash
   # Archival only — decompose into STL for real knowledge ingestion
   stg import-doc meeting_notes.md --source meeting
   ```
5. **Don't skip `feedback session-end`.** Without it, Hebbian learning data and co-activation edges are lost.

---

## How STG Works (background, not required for use)

- **Confidence** = how true. Set at ingest time. Never changes automatically. A fact stays true regardless of time.
- **Salience** = how easy to recall. Modified by Hebbian learning: edges between frequently co-activated nodes become more salient. Unused edges eventually get pruned.
- **Propagate** = associative recall. Input text activates matching nodes, then activation spreads through the graph via edges, weighted by confidence × salience. **Gravity** (enabled by default) further reweights results by structural importance — hub nodes with high elevation are boosted, isolated fragments are suppressed.
- **Communities** = automatic topic clusters detected by Louvain algorithm at three resolutions (coarse/medium/fine). Each community is named after its top representative node. Community labels appear in search results as `[community_name]` — use them to understand which domain a result belongs to.
- **Gravity** = structural importance derived from community topology. Each node has an "elevation" based on community-local PageRank × cross-community bridging. High-elevation nodes are community representatives; low-elevation nodes are internal fragments. Gravity makes propagate results cleaner without manually cleaning the graph.
- **Tensions** = contradictions detected in the graph. STG preserves them rather than auto-resolving. Contradictions are cognitive signals.
- **Ψ (Psi)** = knowledge quality metric. Measures confidence distribution and structural coherence.
- **Pruning** = automatic removal of low-salience, long-unused, structurally unimportant edges. Triple-safety prevents removing critical connections.

### Timestamps — Two Layers, Clear Separation

STG separates "when the edge was written" from "when the event happened". Two layers, two responsibilities:

| Layer | Field | What it records | Set by |
|---|---|---|---|
| **Engine** | `created_at` | When this edge was written to the graph (epoch float) | Engine, automatic at ingest |
| **Engine** | `last_used` | When the edge was last activated during propagation | Engine, automatic during recall |
| **Semantic** | `occurred_time` | When the event itself happened (recommended field) | You, via modifier |

```bash
stg ingest '[Server_Outage] -> [Root_Cause_OOM] ::mod(action="caused_by", confidence=0.95, rule="empirical", occurred_time="2026-03-15T14:30:00", description="OOM killed the API server during traffic spike")'
```

**Use `occurred_time`, not `timestamp`.** The old field name `timestamp` was ambiguous — it could mean "event time" *or* "record time" — and is the documented source of LongMemEval ingest fallback bugs (LLMs sometimes filled it with the session timestamp instead of the real event time). `timestamp` is still accepted as a deprecated alias for backward compatibility, but new ingests should use `occurred_time`. The `_time` noun suffix is unambiguous in zero-context LLM use; the `_at` preposition was easy to mis-fill.

**`recorded_at` is gone — merged into `created_at`.** Earlier docs talked about a `recorded_at` field; it was conceptual only and has been folded into `created_at`. Don't write `recorded_at` in new edges.

**Override `created_at` for backfilled ingests.** When ingesting an external dataset where each piece of knowledge has its own original recording time (e.g., conversation logs with dates, archived emails), pass a `created_at=<float>` modifier to override the default ingest time:

```bash
# This edge will show as if it were recorded on 2023-09-25 13:45 UTC,
# not at the moment of ingest. Useful for replaying datasets while
# preserving their authentic temporal signature.
stg ingest '[User] -> [Charity_Gala] ::mod(action="volunteered", confidence=0.95, occurred_time="2023-09-25", created_at=1695645900.0, description="...")'
```

These drive two recall mechanisms:
- **Time decay** — salience decays based on `now - last_used`. Half-life = 30 days. Unused edges fade.
- **Temporal queries** — `temporal range <date>` reads `created_at` (ingest/replay time), not `occurred_time` (event time). Use `occurred_time` filters in `propagate` / `node` output to find by event date.

Don't confuse: `created_at` is **always present, always engine-set** (or modifier-overridden during backfill). `occurred_time` is **optional, always you-set** for events with real dates. `last_used` is engine bookkeeping for Hebbian decay.
