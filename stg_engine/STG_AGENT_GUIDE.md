# STG Agent Guide

A persistent knowledge graph with associative recall, Hebbian learning, and —
as of v0.3.0a3 — an executable skill layer. This guide covers the parts most
relevant to LLM-backed agents using STG as external memory **and** as a
lightweight capability registry.

Run `stg <command> --help` (or consult `stg` with no args) for terse CLI
reference. This document is the longer-form orientation.

---

## Contents

1.  Core concepts
2.  Storing knowledge — `ingest`
3.  Retrieval — `propagate`, `query`, `search`, `paths`
4.  Learning — Hebbian feedback
5.  **Skills — `stg use` (how to run, how to make)**
6.  Temporal queries
7.  Administration

---

## 1. Core concepts

STG stores **directed weighted edges** between symbolic nodes. An edge is
written in STL (Semantic Tension Language):

```
[Source_Node] -> [Target_Node] ::mod(key=value, key=value, ...)
```

Nodes have optional namespaces: `[Skill:Foo]` puts the node in the `Skill`
namespace; `[Memory:Session_042]` puts it in `Memory`. The reserved namespace
`Skill:` is recognized by the executor (see §5).

Modifiers on edges carry everything else: `confidence`, `rule`, `description`,
`timestamp`, `path`, domain-specific tags, and — for Skills — invocation
metadata (`executable`, `interpreter`, `args_template`, ...).

Data is persisted to a single `.stg` SQLite file. Default location is
`~/.stg/<agent>/memory.stg`; switch agents via `--agent <name>` or
`STG_AGENT` env var.

---

## 2. Storing knowledge

```bash
stg ingest '[Newton] -> [Calculus] ::mod(
  rule="empirical", confidence=0.98,
  description="Newton co-invented calculus with Leibniz",
  year="1665"
)'
```

Required fields: `confidence`, `description`. Recommended: `rule`
(`empirical` / `causal` / `definitional` / `logical`). Multi-edge:
re-ingesting the same (source, target) with different content keeps both
(old is marked superseded, history preserved).

For bulk import:

```bash
stg ingest-file knowledge.stl
```

Each line of the file is a single STL statement.

---

## 3. Retrieval

- `propagate <text>` — associative recall via spreading activation. Returns
  communities of related nodes with per-node activation scores. Default
  auto-expands the top 3 for full edge detail.
- `query <pattern>` — substring search over node names.
- `grep <pattern>` — regex search over all descriptions + lessons.
- `search <query>` — embedding-based semantic search (slower, off by
  default — requires `stg embed` first).
- `node <name>` — full detail dump of one node and all its incident edges.
- `paths <A> <B>` — shortest chain of edges connecting two nodes.

---

## 4. Hebbian learning

`propagate` applies Hebbian strengthening automatically along the activated
path. To explicitly reinforce a known-good chain:

```bash
stg learn path Newton Calculus Integration
```

Weak edges decay via `stg prune` (run periodically via `stg feedback
session-end` at the close of a working session).

---

## 5. Skills — running scripts from STG

*Introduced in v0.3.0a3.*

STG can act as a **capability registry**: register a script as a
`Skill:`-namespaced node, then invoke it by name. This is useful when an
LLM agent builds up a palette of tools over time and wants them accessible
through the same interface it uses to recall knowledge.

### 5.1 One-time opt-in

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

Optional but convenient: name an interpreter you use across several skills,
so individual skill edges don't need to carry an absolute path:

```bash
stg config set skill.interpreters.myvenv "/home/you/proj/.venv/bin/python3"
```

Named interpreters resolve first via your config; unknown names fall back
to the builtins `python3`, `python`, `bash`, `sh`, `node` (resolved via
`shutil.which`, so they work on Linux / macOS / Windows uniformly). Any
other string must be an absolute path to a runnable binary.

### 5.2 Running a skill

Once at least one Skill edge has `executable=true` and its script lives
under `skill.roots`:

```bash
# Positional args pass verbatim to the script
stg use My_Skill foo --option bar

# Resolve and print the command line without executing
stg use My_Skill foo --dry-run

# Override the declared timeout
stg use My_Skill foo --timeout 120

# Capture structured result {stdout, stderr, exit_code, ...} as JSON
# (useful for scripts embedding `stg use`)
stg use My_Skill foo --json

# For skills that declared stl_io=true: pipe an STL block on stdin
stg use My_Skill --args-stl '[Arg:input] → [File] ::mod(path="/tmp/x")'
stg use My_Skill --args-stl-file /tmp/params.stl
```

Every invocation writes one row to the `skill_invocations` SQLite table.
Review recent calls with:

```bash
stg skill history --limit 10
stg skill history --skill My_Skill --limit 20
```

### 5.3 Browsing available skills

```bash
# Executable skills first, rendered as a catalog
stg skill list
stg skill list --filter reddit   # substring filter across name/path/desc
stg skill list --all             # include not-yet-configured skills too

# propagate skill triggers the same catalog view
stg propagate skill

# Full detail + recent invocations on one skill
stg skill show My_Skill
```

### 5.4 Making a skill (`制作方法`)

A Skill is just an STG edge whose source node lives in the `Skill`
namespace and whose modifiers include five invocation fields.

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
| `args_template` | Human-readable signature displayed by `stg skill show`. Not currently validated. |
| `timeout_s` | Per-skill timeout in seconds. Defaults to `skill.default_timeout_s` (60). Max 600. |
| `stl_io` | `"true"` if the script reads STL on stdin and writes STL on stdout (see §5.5). Default `"false"`. |
| `description` | A clear purpose string surfaced by `skill list` and `propagate`. |

**Already-ingested Skill but missing invocation fields?** Retrofit without
re-ingesting:

```bash
stg skill configure My_Skill \
    --executable \
    --interpreter python3 \
    --args-template "<input> [--out PATH]" \
    --timeout 60
```

This calls `stg merge` under the hood, patching the existing edge. If the
Skill has multiple outgoing edges, `configure` auto-selects the one with a
`path=` modifier (the "primary" edge); use `stg merge` directly to target
a specific non-primary edge.

### 5.5 STL-first I/O (`stl_io=true`)

For skills that want structured round-tripping with an LLM caller:

**Script receives on stdin:**

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

This keeps the LLM ↔ script boundary consistent with the rest of STG (no
JSON round-trips). Parse output with `stl_parser.validate_llm_output` for
21 automatic repair rules against drifted formatting.

Free-form progress goes to **stderr**, not stdout — stdout is reserved
for the STL payload.

### 5.6 Security model

Every `stg use` call passes through five gates, in order:

1. `skill.enabled = true` in user config — else exit 6.
2. Skill edge has `executable = true` — else exit 6.
3. Script path exists and is a regular file — else exit 6.
4. Resolved path (symlinks followed) is under at least one `skill.roots`
   entry — else exit 6.
5. Interpreter resolves to an executable binary — else exit 6.

Subprocess is invoked with a **list** of args (never `shell=True`), a hard
timeout (default 60s, max 600s), and a 10 MB stdout cap. Every call writes
one audit row to `skill_invocations` even if it fails.

Exit-code taxonomy:

| Code | Meaning |
|---|---|
| 0 | Skill ran and returned 0 |
| 3 | Skill not found (name mismatch in STG) |
| 4 | Skill ran but returned non-zero (forwarded) |
| 5 | Ambiguous — multiple equally-ranked edges |
| 6 | Security gate failed (see §5.6 list) |
| 7 | Timeout |
| 8 | STL parse error on skill's stdout (when `stl_io=true`) |

### 5.7 Multi-user portability

`skill.roots`, `skill.interpreters`, etc. are **per-user** config in
`~/.stg/config.json`. No hardcoded paths in the engine — a script that
lives at `/home/alice/tools/foo.py` on Alice's machine won't work on
Bob's until Bob ingests his own Skill edge with his local path. This is
intentional: `.stg` files don't silently run unknown scripts from other
users' environments.

When distributing a tool (e.g., on GitHub), ship a registration snippet
in the README. Example:

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

Users edit paths to match their setup. No templating machinery needed.

### 5.8 Typical workflow for an LLM agent

```
# Session start: orient to what's available
stg propagate skill
   → catalog of registered skills

# Recall + run: ask STG what to do for a problem, pick a skill, run it
stg propagate reddit             → relevant Skill + Lessons
stg use Reddit_Pipe_Dispatcher https://www.reddit.com/r/linux/comments/xyz
   → markdown of parsed thread, direct to stdout

# Record a new capability you just built
stg ingest '[Skill:New_Tool] -> [Target] ::mod(path="...", executable="true", ...)'
```

The agent never needs to remember absolute script paths — they live in
STG. `stg use <name>` becomes the single gesture for "call a registered
capability by name."

---

## 6. Temporal queries

STG tracks per-edge timestamps (`created_at`, `last_used`). Query by time:

```bash
stg temporal range 2026-04-20 2026-04-22    # what was ingested in this range
stg temporal around NodeName 12h             # 12-hour window around a node's creation
stg temporal build 2026-04-20                # reify a thought sequence with linked edges
stg temporal replay StartNode                # walk a built sequence
```

---

## 7. Administration

```bash
stg backup [--keep 7]        # snapshot .stg with rotation
stg reload                   # reload from disk (drop in-memory state)
stg metrics                  # propagation stats from last query
stg psi                      # knowledge-quality metric breakdown
stg virtual list             # virtual edges (sibling, co_source, etc.)
stg feedback session-end     # end-of-session cleanup (prune + save)
stg telemetry report         # usage stats over time
```

---

## Further reading

- `CHANGELOG.md` — recent version history
- `README.md` — project overview, licensing, quickstart
- The STL language spec: <https://github.com/scos-lab/semantic-tension-language>
- STL-TOOLS (parser/builder/LLM helpers): <https://github.com/scos-lab/STL-TOOLS>
