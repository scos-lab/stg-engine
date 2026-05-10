#!/usr/bin/env python3
"""STG Quick CLI — Persistent memory for AI agents.

STG is a knowledge graph that persists on disk. Store knowledge that survives
across sessions, recall it via associative activation, and learn automatically.

Multi-agent: use --agent <name> or STG_AGENT env var for isolated memory.
  python stg_cli.py --agent my-agent stats

Direct file path: use --path <file.stg> to load any .stg file (overrides --agent).
  python stg_cli.py --path /tmp/external.stg stats

=== ESSENTIAL (start here) ================================================

  stats                                  Graph overview (nodes, edges, Ψ)
  propagate <text> [--full] [--expand N] Associative recall + auto Hebbian learning
                                         Default: auto-expands top 3 nodes (full edge detail)
                                         --expand N: expand top-N (use 0 to disable)
                                         --full: dump ALL edge modifiers in chain display
  ingest '<STL statement>'               Store knowledge (see syntax below)
  ingest-file <path> [--created-at T]    Bulk ingest from file (.md, .txt, .stl, ...)
  feedback session-end                   End-of-session cleanup (ALWAYS run this)

=== RETRIEVAL (find knowledge) ============================================

  query <pattern> [--limit N]            Search node names (substring match)
  grep <pattern> [--limit N]             Search all descriptions + lessons (regex)
  search <query> [--top N]               Semantic search (embedding similarity, slow)
  node <name>                            Full detail of one node + all its edges
  paths <source> <target>                Find relationship chains between two nodes
  converge <query>                       Iterative propagation for vague queries
  dump [--page N] [--start N]            Paginated dump of all nodes + edges (interactive)

=== KNOWLEDGE MANAGEMENT ==================================================

  ingest '<STL>'                         Single edge: '[A] -> [B] ::mod(...)'
  ingest '<STL>' --cognitive             Mark as agent's own reasoning (not external fact)
  tensions [active|resolved]             View contradictions in the graph
  epistemic [summary|query|validate]     Inspect knowledge quality metadata

  STL syntax:  [Source] -> [Target] ::mod(key=value, ...)
  Required:    confidence (0-1), description ("what this means")
  Recommended: rule ("causal"/"empirical"/"definitional"/"logical")
  Optional:    strength (0-1), lesson ("..."), timestamp ("2026-03-29"), source ("...")

  Multi-edge: same (Source, Target) with different content → both kept,
    both alive. Treated as complementary facets (e.g. action="took" and
    status="had_amazing_time" describe the same trip). Lookup points to
    newest. Supersede is flagged only on actual corrections: same source
    + same (semantic_field, value) + DIFFERENT target.

  Timestamps: each edge has created_at (auto, when ingested) and
    last_used (auto, when Hebbian-activated). Use timestamp= in mod
    for when the event itself happened.

=== LEARNING & DYNAMICS ===================================================

  learn status                           Hebbian learning statistics
  learn path <n1> <n2> [<n3>...]         Explicitly strengthen a known-good path
  learn propagate <text>                 Propagate + Hebbian (same as propagate)
  prune [--dry-run]                      Remove low-salience unused edges
  importance [--top N]                   Most structurally important nodes

=== TEMPORAL (what happened when?) ========================================

  temporal range <date> [end_date]       What was ingested on a date/range
  temporal around <node> [hours]         Context around a node's creation time
  temporal build <date>                  Build thought sequence (permanent edges)
  temporal replay <start_node>           Walk a built sequence
  temporal stats                         Timestamp coverage overview

=== ANALYSIS & ADMIN ======================================================

  psi                                    Knowledge quality metric breakdown
  metrics                                Propagation metrics from last query
  topology [communities|bridges|analyze] Graph structure analysis
  cognitive [self-model|hypotheses]      Knowledge gap assessment
  telemetry [status|frequency|report]    Usage statistics
  simulate run --calibrated              Parameter simulation with real data
  backup [--keep N]                      Backup .stg file (auto-rotated)
  alias <add|list|remove|resolve>         Entity resolution aliases (G7)
  virtual [stats|list|rebuild]           Virtual edge management
  embed [--model NAME]                   Build/rebuild embedding index
  preference [top|reward|decay]          Edge preference management
  reload                                 Reload .stg from disk
  import [manifest_path]                 Import from manifest file
  benchmark [full|propagation|...]       Performance benchmarks

=== SKILLS (executable capabilities — v0.3.0a3+) ==========================

  use <skill_name> [args...]             Run a registered Skill's script
                                         with audit + timeout + STL I/O
  skill list [--filter KW] [--all]       Catalog (executable skills first)
  skill show <name>                      Detail + recent invocations
  skill configure <name> --executable    Backfill invocation fields onto an
      --interpreter <NAME|/abs>          existing Skill edge (uses `merge`)
      --args-template '<sig>' [--timeout N] [--stl-io]
  skill history [--skill N] [--limit N]  Recent stg use calls
  propagate skill                        Render Skill catalog (instead of
                                         community-grouped default)

  One-time opt-in (disabled by default — fresh installs cannot run anything):
      stg config set skill.enabled true
      stg config set skill.roots "/abs/path/to/tools[,/abs/path/to/other]"
      stg config set skill.interpreters.<name> "/abs/path/to/binary"

  Full walkthrough + "how to make a skill" guide: run `stg guide`.

=== HEARTBEAT (contract execution) ========================================

  heartbeat run <dir> [--interval 10]    Execute contracts in directory
  heartbeat status <dir>                 Check contract states
  heartbeat verify <file>                Verify a single contract
  heartbeat reset <dir> [--contract ID]  Reset contract state
  heartbeat single <file>                Execute one contract

Load time: ~3ms from .stg file. All operations: microseconds.
Run 'python stg_cli.py guide' to read the full guide.
"""

import sys
import os
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Tuple

from stg_engine.engine import PROVENANCE_FIELDS

# --- Settings ---
_CLI_DIR = os.path.dirname(os.path.abspath(__file__))
_STG_ROOT = os.path.join(os.path.expanduser("~"), ".stg")
_USER_CONFIG_PATH = os.path.join(_STG_ROOT, "config.json")
# Neutral fallback for fresh installs. Personal agent name (if any) belongs in
# ~/.stg/config.json's "default_agent" key, set via `stg config set default-agent <name>`.
_DEFAULT_AGENT = "default"


def _read_user_config():
    """Read ~/.stg/config.json (user-level, cross-agent config).

    Missing file → empty dict. Malformed JSON → empty dict with warning.
    Never raises — config is always optional.
    """
    if not os.path.exists(_USER_CONFIG_PATH):
        return {}
    try:
        with open(_USER_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except (json.JSONDecodeError, OSError) as exc:
        print(f"warning: failed to read {_USER_CONFIG_PATH}: {exc}", file=sys.stderr)
    return {}


def _write_user_config(config: dict):
    """Write ~/.stg/config.json. Creates parent dir if needed."""
    os.makedirs(_STG_ROOT, exist_ok=True)
    with open(_USER_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def _load_settings():
    """Load agent settings from ~/.stg/<agent>/stg_settings.json.

    Agent selection priority (highest to lowest):
      1. --agent <name> flag
      2. STG_AGENT environment variable
      3. ~/.stg/config.json "default_agent" key
      4. Hardcoded fallback: "default"
    """
    agent = _DEFAULT_AGENT
    path_override = None

    # Check --agent flag
    if "--agent" in sys.argv:
        idx = sys.argv.index("--agent")
        if idx + 1 < len(sys.argv):
            agent = sys.argv[idx + 1]
            sys.argv = sys.argv[:idx] + sys.argv[idx + 2:]

    # Check environment variable
    elif os.environ.get("STG_AGENT"):
        agent = os.environ["STG_AGENT"]

    # Check user config file (~/.stg/config.json)
    else:
        user_cfg = _read_user_config()
        cfg_agent = user_cfg.get("default_agent")
        if isinstance(cfg_agent, str) and cfg_agent.strip():
            agent = cfg_agent.strip()

    # Check --path flag (overrides agent-based resolution)
    if "--path" in sys.argv:
        idx = sys.argv.index("--path")
        if idx + 1 < len(sys.argv):
            path_override = os.path.abspath(os.path.expanduser(sys.argv[idx + 1]))
            sys.argv = sys.argv[:idx] + sys.argv[idx + 2:]

    if path_override:
        stg_path = path_override
        agent_dir = os.path.dirname(path_override)
        settings_path = os.path.join(agent_dir, "stg_settings.json")
    else:
        agent_dir = os.path.join(_STG_ROOT, agent)
        settings_path = os.path.join(agent_dir, "stg_settings.json")
        stg_path = os.path.join(agent_dir, "memory.stg")

    settings = {"agent": agent}
    if os.path.exists(settings_path):
        with open(settings_path, "r", encoding="utf-8") as f:
            settings = json.load(f)
            settings["agent"] = agent

    settings["_agent_dir"] = agent_dir
    settings["_stg_path"] = stg_path
    return settings

SETTINGS = _load_settings()
STG_PATH = SETTINGS["_stg_path"]
# Matrix path: legacy, kept for reload command
MATRIX_PATH = os.path.join(_CLI_DIR, "memory", "Syn-claude", "memoryMatrix.md")

# --- Audit Logger ---
_AUDIT_LOG_PATH = os.path.join(SETTINGS["_agent_dir"], "audit.log")

def _init_audit_logger():
    logger = logging.getLogger("stg.audit")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        os.makedirs(os.path.dirname(_AUDIT_LOG_PATH), exist_ok=True)
        handler = logging.FileHandler(_AUDIT_LOG_PATH, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
    return logger

_audit = _init_audit_logger()

def _snap(engine):
    """Snapshot node/edge counts for audit comparison."""
    return {"nodes": len(engine._nodes), "edges": len(engine._edges)}

def _audit_log(cmd, args, before, after=None, detail=""):
    """Write one audit log entry."""
    if after and (before["nodes"] != after["nodes"] or before["edges"] != after["edges"]):
        delta_n = after["nodes"] - before["nodes"]
        delta_e = after["edges"] - before["edges"]
        msg = f"CMD={cmd} | args={args} | nodes: {before['nodes']}→{after['nodes']} ({delta_n:+d}) | edges: {before['edges']}→{after['edges']} ({delta_e:+d})"
    else:
        msg = f"CMD={cmd} | args={args} | nodes: {before['nodes']} | edges: {before['edges']}"
    if detail:
        msg += f" | {detail}"
    _audit.info(msg)


def load_engine():
    from stg_engine import STGEngine
    if os.path.exists(STG_PATH):
        return STGEngine.load(STG_PATH)
    # First use: create empty graph and parent directories
    os.makedirs(os.path.dirname(STG_PATH), exist_ok=True)
    engine = STGEngine()
    engine.save(STG_PATH)
    print(f"Created new STG: {STG_PATH}")
    return engine


def cmd_stats(engine):
    s = engine.get_stats()
    print(f"Nodes: {s['node_count']}")
    v = s.get('virtual_edge_count', 0)
    r = s.get('real_edge_count', s['edge_count'])
    if v > 0:
        print(f"Edges: {s['edge_count']} ({r} real + {v} virtual)")
    else:
        print(f"Edges: {s['edge_count']}")
    print(f"Sessions: {s['session_count']}")
    print(f"Events: {s['event_count']}")
    print(f"Tensions: {s['total_tensions']} ({s['active_tensions']} active)")
    print(f"Belief Evolutions: {s['belief_evolutions']}")
    print(f"Psi: {s['psi']:.4f}")
    print(f"Density: {s['graph_density']:.6f}")
    # Small graph: list all nodes for quick overview
    if 0 < s['node_count'] <= 30:
        names = sorted(engine._nodes.keys())
        print(f"All nodes: {', '.join(names)}")


def cmd_psi(engine):
    psi = engine.compute_psi()
    print(f"Ψ = {psi:.4f}")

    # Breakdown
    sc = engine._compute_structural_coherence()
    ec = engine._compute_epistemic_confidence()
    max_t = max((n.tension for n in engine._nodes.values()), default=0.0)
    print(f"  Structural Coherence: {sc:.4f}")
    print(f"  Epistemic Confidence: {ec:.4f}")
    print(f"  Max Tension: {max_t:.4f}")


def cmd_import_doc(engine, filepath, source_type="doc", max_desc=10000):
    """Import a text document into STG as a low-precision edge.

    Full content stored in description — zero information loss.
    Supports: .txt, .md, .eml, any plain text.
    """
    import re as _re
    path = os.path.join(os.getcwd(), filepath) if not os.path.isabs(filepath) else filepath
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    if not raw.strip():
        print(f"Empty file: {path}")
        return

    # Sanitize
    text = raw.replace("[", "(").replace("]", ")").replace('"', "'")
    text = _re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    if len(text) > max_desc:
        cut = text[:max_desc]
        last_period = cut.rfind(". ")
        if last_period > max_desc * 0.7:
            text = cut[:last_period + 1]
        else:
            text = cut + "..."

    # Node names
    basename = os.path.splitext(os.path.basename(path))[0]
    clean_name = _re.sub(r"[^A-Za-z0-9_-]", "_", basename)
    clean_name = _re.sub(r"_+", "_", clean_name).strip("_")
    prefix = {"chat": "Session", "blog": "Blog", "email": "Email", "meeting": "Meeting"}.get(source_type, "Doc")
    node_name = f"{prefix}_{clean_name}"
    content_node = f"{node_name}_Content"

    # Get file mtime as timestamp
    from datetime import datetime as _dt
    mtime = _dt.fromtimestamp(os.path.getmtime(path))
    ts = mtime.strftime("%Y-%m-%d %H:%M:%S")

    engine.add_edge(
        source=node_name,
        target=content_node,
        confidence=0.99,
        description=text,
        source_type=source_type,
        timestamp=ts,
    )
    engine.save(STG_PATH)
    print(f"Imported: [{node_name}] -> [{content_node}] ({len(text)} chars)")
    s = engine.get_stats()
    print(f"Graph: {s['node_count']} nodes, {s['edge_count']} edges")


def cmd_grep(engine, pattern, limit=20, full=False):
    """Grep edges by description text, node names, and modifier values.

    Unlike query (node names only) and search (semantic embeddings),
    this does fast substring matching across all edge data.

    Args:
        full: if True, show full description (no 150-char truncation).
              Default False preserves compact output for terminal use.
    """
    import re
    pat = re.compile(pattern, re.IGNORECASE)
    results = []
    for edge in engine._edges:
        desc = edge.modifiers.get("description", "")
        lesson = edge.modifiers.get("lesson", "")
        searchable = f"{edge.source} {edge.target} {desc} {lesson}"
        if pat.search(searchable):
            results.append(edge)

    if not results:
        print(f"No edges matching '{pattern}'")
        # Suggest shorter roots or alternatives
        words = [w for w in re.split(r'[^a-zA-Z]+', pattern) if len(w) > 3]
        if words:
            suggestions = []
            for w in words:
                # Suggest word root (first 5-6 chars)
                root = w[:min(len(w) - 2, 6)] if len(w) > 5 else w
                # Check if root finds anything
                root_pat = re.compile(root, re.IGNORECASE)
                root_count = sum(1 for e in engine._edges
                                 if root_pat.search(f"{e.source} {e.target} {e.modifiers.get('description', '')}"))
                if root_count > 0:
                    suggestions.append(f'  "{root}" ({root_count} matches)')
            if suggestions:
                print(f"\nTip: Try shorter word roots:")
                for s in suggestions[:5]:
                    print(s)
                print(f"  Or split into separate searches.")
        return

    print(f"Edges matching '{pattern}': {len(results)} (showing {min(limit, len(results))})")
    for e in results[:limit]:
        ts = e.modifiers.get("timestamp", "")
        raw_desc = e.modifiers.get("description", "")
        desc = raw_desc if full else raw_desc[:150]
        ts_str = f"[{ts}] " if ts else ""
        comm = _community_label(engine, e.source)
        comm_suffix = f"  [{comm}]" if comm else ""
        print(f"  {ts_str}[{e.source}] -> [{e.target}]{comm_suffix}")
        if desc:
            print(f"    {desc}")


def cmd_query(engine, pattern, limit=20):
    """Fuzzy node search with optional namespace scoping.

    Pattern grammar:
        <text>           — substring match across all node names (any namespace)
        <ns>:            — list every node in `<ns>` namespace (no name filter)
        <ns>:<text>      — substring match on name, scoped to `<ns>` namespace
    """
    # Parse `Namespace:NamePattern` form when a colon is present
    if ":" in pattern:
        ns_part, _, name_part = pattern.partition(":")
        namespace = ns_part if ns_part else None
        name_pattern = name_part if name_part else None
    else:
        namespace = None
        name_pattern = pattern

    nodes = engine.query_nodes(
        name_pattern=name_pattern, namespace=namespace, limit=limit,
    )
    if not nodes:
        print(f"No nodes matching '{pattern}'")
        return
    total = len(engine.query_nodes(
        name_pattern=name_pattern, namespace=namespace, limit=10000,
    ))
    shown = len(nodes)
    suffix = f" (showing {shown}/{total})" if total > shown else ""
    print(f"Nodes matching '{pattern}'{suffix}:")
    for n in nodes:
        display = f"{n.namespace}:{n.name}" if n.namespace else n.name
        parts = [f"  {display}"]
        if n.tension > 0:
            parts.append(f"T={n.tension:.3f}")
        if n.activation > 0:
            parts.append(f"A={n.activation:.3f}")
        comm = _community_label(engine, n.name)
        if comm:
            parts.append(f"[{comm}]")
        print(" | ".join(parts))

    # Helper: namespace-prefixed display name for edge endpoints
    def _ns_label(name: str) -> str:
        node = engine._nodes.get(name.lower().replace("-", "_"))
        if node and node.namespace:
            return f"{node.namespace}:{name}"
        return name

    # Related edges: filter by name part (skip when listing a whole namespace)
    if name_pattern:
        edges = engine.query_edges(limit=200)
        np = name_pattern.lower()
        related = [
            e for e in edges
            if np in e.source.lower() or np in e.target.lower()
        ]
        # If a namespace was given, also restrict to edges where at least one
        # endpoint is in that namespace (otherwise `Game:Elden` would surface
        # cross-namespace mentions of "Elden" too).
        if namespace is not None:
            def _in_ns(name: str) -> bool:
                node = engine._nodes.get(name.lower().replace("-", "_"))
                return bool(node and node.namespace == namespace)
            related = [e for e in related if _in_ns(e.source) or _in_ns(e.target)]
        if related:
            print(f"\nRelated edges ({len(related)}):")
            for e in related[:10]:
                is_virtual = e.modifiers.get("edge_class") == "virtual"
                arrow = " ~ " if is_virtual else " -> "
                mod = f"confidence={e.confidence}"
                if e.rule:
                    mod += f', rule="{e.rule}"'
                if is_virtual:
                    reason = e.modifiers.get("virtual_reason", "")
                    mod += f', virtual={reason}'
                print(f"  [{_ns_label(e.source)}]{arrow}[{_ns_label(e.target)}] ::mod({mod})")
                mod_lines, _ = _format_edge_modifiers(e, indent="    ")
                for line in mod_lines:
                    print(line)


def cmd_dump(engine, page_size=100, start=0, namespace=None):
    """Paginated dump of all nodes and their related edges.

    Sorted by node name. Each page shows `page_size` nodes; after each page the
    user is prompted to continue (ENTER), jump to a node (number), or quit (q).

    Args:
        namespace: If set, only nodes whose `namespace` field matches this
            string are dumped. Edges referencing such nodes still display
            their endpoints with full namespace prefixes.
    """
    all_nodes = sorted(engine._nodes.values(), key=lambda n: n.name.lower())
    if namespace is not None:
        all_nodes = [n for n in all_nodes if n.namespace == namespace]
    total_nodes = len(all_nodes)
    total_edges = len(engine._edges)

    if total_nodes == 0:
        if namespace is not None:
            print(f"No nodes in namespace '{namespace}'.")
        else:
            print("Graph is empty.")
        return

    # Build node -> edges index (both directions) for O(1) lookup per node.
    # Edge source/target are stored lowercased; node names preserve case, so
    # key the index by lowercase name.
    edge_index = {}
    for e in engine._edges:
        edge_index.setdefault(e.source.lower(), []).append(("out", e))
        if e.target.lower() != e.source.lower():
            edge_index.setdefault(e.target.lower(), []).append(("in", e))

    # Helper: format a name with its namespace prefix when available.
    # Node name lookup mirrors engine._nk normalization (lower + hyphen→_).
    def _ns_label(name: str) -> str:
        node = engine._nodes.get(name.lower().replace("-", "_"))
        if node and node.namespace:
            return f"{node.namespace}:{name}"
        return name

    scope_msg = f" in namespace '{namespace}'" if namespace else ""
    print(f"Dumping {total_nodes} nodes{scope_msg} / {total_edges} edges (page size {page_size})")
    print()

    idx = max(0, start)
    while idx < total_nodes:
        end = min(idx + page_size, total_nodes)
        print(f"=== Nodes {idx + 1}-{end} / {total_nodes} ===")
        for i in range(idx, end):
            n = all_nodes[i]
            display = f"{n.namespace}:{n.name}" if n.namespace else n.name
            parts = [f"[{i + 1}] {display}"]
            if n.tension > 0:
                parts.append(f"T={n.tension:.3f}")
            if n.activation > 0:
                parts.append(f"A={n.activation:.3f}")
            comm = _community_label(engine, n.name)
            if comm:
                parts.append(f"[{comm}]")
            print(" | ".join(parts))

            related = edge_index.get(n.name.lower(), [])
            if not related:
                print("    (no edges)")
                continue
            for direction, e in related:
                is_virtual = e.modifiers.get("edge_class") == "virtual"
                arrow = " ~ " if is_virtual else " -> "
                mod = f"confidence={e.confidence}"
                if e.rule:
                    mod += f', rule="{e.rule}"'
                if is_virtual:
                    reason = e.modifiers.get("virtual_reason", "")
                    mod += f", virtual={reason}"
                tag = "  out" if direction == "out" else "  in "
                src_label = _ns_label(e.source)
                tgt_label = _ns_label(e.target)
                print(f"  {tag} [{src_label}]{arrow}[{tgt_label}] ::mod({mod})")
                desc = e.modifiers.get("description") or e.modifiers.get("lesson")
                if desc:
                    if len(desc) > 120:
                        desc = desc[:117] + "..."
                    print(f"       desc: {desc}")
                if e.modifiers.get("superseded_at"):
                    print(f"       (superseded)")

        idx = end
        if idx >= total_nodes:
            print()
            print(f"=== End of dump ({total_nodes} nodes) ===")
            break

        remaining = total_nodes - idx
        print()
        try:
            resp = input(
                f"--- {idx}/{total_nodes} shown, {remaining} remaining. "
                f"[ENTER] next page, [number] jump to node #, [q] quit: "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if resp.lower() in ("q", "quit", "exit"):
            break
        if resp.isdigit():
            jump = int(resp) - 1
            if 0 <= jump < total_nodes:
                idx = jump
        print()


def cmd_tensions(engine, status=None):
    tensions = engine.query_tensions(status=status)
    if not tensions:
        print(f"No tensions" + (f" with status '{status}'" if status else ""))
        return
    label = f" ({status})" if status else ""
    print(f"Tensions{label} ({len(tensions)}):")
    for t in tensions:
        marker = {"active": "●", "resolved": "○", "persisting": "◐"}.get(t.status, "?")
        print(f"  {marker} {t.name}: {t.current_value:.2f} [{t.status}]")
        if t.created_session:
            detail = f"    created: {t.created_session}"
            if t.resolved_session:
                detail += f" → resolved: {t.resolved_session}"
            print(detail)


def _log_propagate(engine, text, elapsed):
    """Append propagate stats to propagate_log.stl.md for Hebbian learning data."""
    pm = engine.last_propagation_metrics
    if not pm:
        return
    log_path = os.path.join(os.path.dirname(__file__), "memory", "Syn-claude", "propagate_log.stl.md")
    now = datetime.now()
    date_header = f"## {now.strftime('%Y-%m-%d')}"
    time_header = f"### {now.strftime('%H:%M:%S')}"

    top_nodes_str = ", ".join(f"{name} ({act:.2f})" for name, act in pm.top_nodes[:5])

    entry = (
        f"\n{time_header}\n"
        f"- **Query**: \"{text}\"\n"
        f"- **Seeds**: {pm.seed_node_count} | **Activated**: {pm.activated_node_count} "
        f"| **QE**: {pm.query_efficiency:.3f} | **RS**: {pm.resonance_score:.3f} "
        f"| **Time**: {elapsed*1000:.1f}ms\n"
        f"- **Top nodes**: {top_nodes_str}\n"
    )

    try:
        existing = ""
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                existing = f.read()

        with open(log_path, "a", encoding="utf-8") as f:
            if not existing:
                f.write("# Propagate Log\n\n")
            if date_header not in existing:
                f.write(f"\n{date_header}\n")
            f.write(entry)
    except Exception:
        pass  # logging should never break propagate


def _format_edge_label(edge):
    """Compact edge label using meta semantic fields with fallback chain.

    Picks the first non-empty value from action / role / status / phase /
    is_a / relation, and appends @occurred_time / @timestamp if present.
    Used by anchor-pair and event-edge views to surface relationship type
    even when `action` is missing (common in human-curated graphs).
    """
    mods = edge.modifiers
    semantic = (
        mods.get("action") or mods.get("role") or mods.get("status")
        or mods.get("phase") or mods.get("is_a") or mods.get("relation")
        or ""
    )
    when = mods.get("occurred_time") or mods.get("timestamp") or ""
    parts = []
    if semantic:
        parts.append(str(semantic))
    if when:
        parts.append(f"@{when}")
    return " ".join(parts) if parts else "→"


def _show_propagation_chains(engine, activated, max_chains=5, min_length=2, all_chains=False, all_modifiers=False):
    """Extract and display chains from the activated subgraph.

    By default, chains only follow real edges (knowledge edges with descriptions).
    Virtual edges (sibling/co_source) are excluded to keep chains focused on
    actual reasoning paths. Use all_chains=True to include virtual edge paths.

    Args:
        all_chains: If True, include chains through virtual edges.
        all_modifiers: If True, display all edge modifiers (text/lesson/path/section/source/...)
                       instead of only description (truncated). Used by --full flag for
                       lossless retrieval where text= field carries original prose.
    """
    if len(activated) < min_length + 1:
        return

    from stl_parser.graph import STLGraph

    # Normalize to graph keys (lowercase) — propagate returns display names
    activated_set = set(n.lower() for n in activated)

    # Build full subgraph (all edges)
    full_subgraph = engine._graph.subgraph(activated_set).copy()
    if full_subgraph.number_of_edges() == 0:
        return

    # Build set of virtual edge pairs from engine._edges
    virtual_pairs = set()
    for edge in engine._edges:
        if edge.modifiers.get("edge_class") == "virtual":
            virtual_pairs.add((edge.source, edge.target))

    # Build real-only subgraph (exclude virtual edges)
    real_subgraph = full_subgraph.copy()
    virtual_edges_to_remove = [
        (u, v) for u, v in real_subgraph.edges()
        if (u, v) in virtual_pairs
    ]
    for u, v in virtual_edges_to_remove:
        real_subgraph.remove_edge(u, v)

    # Extract real chains
    real_chains = []
    if real_subgraph.number_of_edges() > 0:
        stl_graph = STLGraph.from_networkx(real_subgraph)
        real_chains = stl_graph.extract_chains(min_length=min_length)

    # Extract all chains (for counting virtual-only chains)
    all_raw_chains = []
    if full_subgraph.number_of_edges() > 0:
        stl_graph_full = STLGraph.from_networkx(full_subgraph)
        all_raw_chains = stl_graph_full.extract_chains(min_length=min_length)

    # Choose which chains to display
    display_chains = all_raw_chains if all_chains else real_chains

    if not display_chains and not all_raw_chains:
        return

    # Score and deduplicate
    def _score_and_dedup(chains):
        scored = []
        for path in chains:
            total_act = sum(
                engine.get_node(n).activation
                for n in path
                if engine.get_node(n)
            )
            scored.append((path, total_act))
        scored.sort(key=lambda x: -x[1])
        unique = []
        for path, total_act in scored:
            path_set = set(path)
            is_dup = False
            for existing_path, _ in unique:
                existing_set = set(existing_path)
                overlap = len(path_set & existing_set) / max(len(path_set), len(existing_set))
                if overlap > 0.7:
                    is_dup = True
                    break
            if not is_dup:
                unique.append((path, total_act))
        return unique

    unique_display = _score_and_dedup(display_chains)
    unique_all = _score_and_dedup(all_raw_chains)
    virtual_only_count = len(unique_all) - len(unique_display)

    if not unique_display and not all_chains:
        # No real chains, but virtual chains exist
        if virtual_only_count > 0:
            print(f"\n  Chains: 0 real chains (+{virtual_only_count} via virtual edges, use --all-chains to see)")
        return

    label = "Chains" if all_chains else "Chains (real edges)"
    print(f"\n  {label} ({len(unique_display)}):")
    for i, (path, total_act) in enumerate(unique_display[:max_chains]):
        # Chain header: node(A) → node(A) → ...
        parts = []
        for n in path:
            node = engine.get_node(n)
            a = f"{node.activation:.2f}" if node else "?"
            parts.append(f"{n}({a})")
        print(f"    {i+1}. {' → '.join(parts)}")

        # Edge details: show STL statement for each hop
        for j in range(len(path) - 1):
            src, tgt = path[j], path[j + 1]
            edges = engine.get_edges(source=src, target=tgt)
            edge = edges[0] if edges else None
            if edge:
                is_virtual = edge.modifiers.get("edge_class") == "virtual"
                if is_virtual:
                    reason = edge.modifiers.get("virtual_reason", "virtual")
                    parent = edge.modifiers.get("virtual_parent", "")
                    label_str = f"{reason} of {parent}" if parent else reason
                    print(f"       [{src}] ~[{label_str}]~ [{tgt}]")
                else:
                    mod_parts = [f"confidence={edge.confidence}"]
                    if edge.rule:
                        mod_parts.append(f'rule="{edge.rule}"')
                    if edge.modifiers:
                        if all_modifiers:
                            # --full: dump every modifier key=value (truncate huge values)
                            for k, v in edge.modifiers.items():
                                if k in ("edge_class", "virtual_reason", "virtual_parent"):
                                    continue
                                v_str = str(v).replace("\n", " ")
                                if len(v_str) > 500:
                                    v_str = v_str[:497] + "..."
                                mod_parts.append(f'{k}="{v_str}"')
                        else:
                            desc = edge.modifiers.get("description", "")
                            if desc:
                                if len(desc) > 80:
                                    desc = desc[:77] + "..."
                                mod_parts.append(f'description="{desc}"')
                    print(f"       [{src}] → [{tgt}] ::mod({', '.join(mod_parts)})")

    if len(unique_display) > max_chains:
        print(f"    ... +{len(unique_display) - max_chains} more")
    if not all_chains and virtual_only_count > 0:
        print(f"    (+{virtual_only_count} chains via virtual edges, use --all-chains to see all)")


def _community_label(engine, node_name, resolution="medium"):
    """Return community name for a node, or empty string if unavailable."""
    try:
        from stg_engine.gravity import community_name_for
        gravity = engine.get_gravity_map()
        name = community_name_for(gravity, node_name, resolution)
        if name and name.lower() != node_name.lower():
            return name
        # If node IS the representative, try coarse for a broader label
        if name and name.lower() == node_name.lower():
            coarse = community_name_for(gravity, node_name, "coarse")
            if coarse and coarse.lower() != node_name.lower():
                return coarse
        return name or ""
    except Exception:
        return ""


def cmd_propagate(engine, text, use_gravity=False, resolution="medium", all_chains=False, all_modifiers=False, expand_top=3, community_mode=True, top_m=5, brief=False, show_virtual=False,
                  no_recency_weight=False, no_community_filter=False,
                  no_context_anchor=False, no_multi_seed=False,
                  no_edge_fallback=False):
    # Auto-enable Hebbian learning on every propagate
    engine.enable_learning()
    engine.enable_telemetry()

    # ─── Phase 2: R5 active_context anchor (gravity elevation boost) ─
    # Load active_context with TTL filter; temporarily boost their
    # elevation in the GravityMap during propagate. Restored in finally.
    # See development/design/STG_PRECISION_RECALL_DESIGN.md §4.5
    from contextlib import nullcontext
    anchor_ctx = nullcontext()
    anchor_names: list = []
    if use_gravity and not no_context_anchor:
        from stg_engine.feedback_select import load_active_context
        from stg_engine.recall import (
            context_anchor_boost,
            DEFAULT_ACTIVE_CONTEXT_TTL_SECONDS,
        )
        ctx = load_active_context(STG_PATH, ttl_seconds=DEFAULT_ACTIVE_CONTEXT_TTL_SECONDS)
        anchor_names = [name for name, _ in ctx]
        if anchor_names:
            gravity_for_anchor = engine.get_gravity_map()
            anchor_ctx = context_anchor_boost(gravity_for_anchor, anchor_names)

    # ─── Phase 3: R2 multi-seed chain intersection dispatch ──────────
    # Split query into tokens. ≥2 tokens → run wrapped multi-seed path
    # (one propagate per token + chain-intersection rerank). Otherwise
    # fall through to single-seed path.
    # See STG_PRECISION_RECALL_DESIGN.md §4.3
    multi_seed_data = None
    from stg_engine.recall import _split_tokens
    split_tokens = _split_tokens(text)

    # ─── Phase 6: A — exact anchor match (whole-token node names) ────
    # Whitespace-split chunks that exactly match a node name are pulled
    # out before sub-token splitting. They become "exact anchors" — forced
    # into the activated set, and (if ≥2) trigger a direct edge-pair view.
    # See STG_R6_EDGE_FALLBACK_SEED_DESIGN.md §A
    from stg_engine.recall import match_exact_anchors, find_edges_between
    exact_anchors, remaining_chunks = match_exact_anchors(engine, text)
    # Tokenize only what wasn't already an exact anchor
    if exact_anchors and not remaining_chunks:
        # All input chunks are exact anchors — no further tokenization
        split_tokens = []
    elif exact_anchors:
        split_tokens = _split_tokens(" ".join(remaining_chunks))

    # ─── Phase 5: R6 token routing — node-first, edge always-scanned ─
    # `classify_tokens` decides which tokens drive multi-seed propagate
    # (the ones that match a node name). But edge content scan now runs
    # over ALL tokens — a token matching a node does NOT exclude it from
    # edge scan. Rationale: "volunteered" might match concept node
    # [Volunteering_Best_Practices] (irrelevant) AND appear in the
    # description of [User] -[role=volunteer]-> [Charity_Gala] (the
    # actual fact). Both are real signals, both should surface. IDF
    # ranks rare tokens (high signal) above common ones (e.g. "user").
    # See STG_R6_EDGE_FALLBACK_SEED_DESIGN.md
    edge_hits_data = []
    node_tokens: list = []
    edge_tokens: list = []
    if not no_edge_fallback and split_tokens:
        from stg_engine.recall import classify_tokens, scan_edges_by_content
        node_tokens, edge_tokens = classify_tokens(engine, split_tokens)
        # Scan ALL tokens against edge meta fields — not only the
        # node-unmatched ones. IDF naturally suppresses common-token noise.
        edge_hits_data = scan_edges_by_content(engine, split_tokens)
        # node-only tokens drive multi-seed dispatch
        propagate_tokens = node_tokens if node_tokens else split_tokens
    else:
        propagate_tokens = split_tokens

    use_multi_seed = (not no_multi_seed) and len(propagate_tokens) >= 2

    t0 = time.perf_counter()
    gravity = None
    with anchor_ctx:
        if use_multi_seed:
            from stg_engine.recall import multi_seed_propagate
            activated, multi_seed_data = multi_seed_propagate(
                engine, " ".join(propagate_tokens), propagate_tokens,
                use_gravity=use_gravity, resolution=resolution,
            )
        elif use_gravity:
            from stg_engine.gravity import gravitational_propagate
            gravity = engine.get_gravity_map()
            propagate_query = " ".join(propagate_tokens) if propagate_tokens else text
            activated = gravitational_propagate(engine, propagate_query, gravity, resolution=resolution)
        else:
            propagate_query = " ".join(propagate_tokens) if propagate_tokens else text
            activated = engine.propagate(propagate_query)
    elapsed = time.perf_counter() - t0
    # Ensure gravity is loaded for community aggregation regardless of which
    # propagate branch was taken (multi-seed wrapper does not return gravity).
    if use_gravity and gravity is None:
        gravity = engine.get_gravity_map()
    # ─── Phase 6 cont. — force-include exact anchors ──────────────────
    # Anchor nodes may not have been picked up by propagate (e.g. if all
    # query chunks were anchors and no propagate ran, or if multi-seed
    # chain intersection over-collapsed and dropped them). Ensure every
    # exact anchor is in the activated list with non-zero activation so
    # downstream community aggregate / rendering can surface its edges.
    if exact_anchors:
        activated_lower_set = {n.lower() for n in (activated or [])}
        prepend: list = []
        for anchor in exact_anchors:
            key = anchor.lower()
            if key not in activated_lower_set:
                node = engine._nodes.get(key)
                if node is not None:
                    if node.activation <= 0.0:
                        node.activation = 1.0
                    prepend.append(anchor)
                    activated_lower_set.add(key)
        if prepend:
            activated = prepend + (activated or [])
        if activated and exact_anchors:
            print(f"  📍 exact anchors: [{', '.join(exact_anchors)}] (forced into result)")

    # Compute anchor-pair edges once; rendered later in Phase 6 view (C / R8)
    anchor_pair_edges = find_edges_between(engine, exact_anchors) if exact_anchors else []

    if not activated and not edge_hits_data and not anchor_pair_edges:
        print(f"No nodes activated for '{text}'")
        return
    if not activated:
        activated = []

    if use_multi_seed and multi_seed_data:
        token_preview = ', '.join(t for t, _, _ in multi_seed_data)
        print(f"  🔗 multi-seed chain intersection: [{token_preview}] → {len(activated)} nodes")

    if edge_hits_data:
        # R6: surface routing — node_tokens drive multi-seed propagate,
        # but edge scan runs over ALL tokens (IDF re-ranks).
        node_preview = ', '.join(node_tokens) if node_tokens else '(none)'
        unmatched = [t for t in split_tokens if t not in node_tokens]
        unmatched_preview = ', '.join(unmatched) if unmatched else '(none)'
        print(f"  🪢 token routing: node=[{node_preview}] node_unmatched=[{unmatched_preview}]"
              f"  edge scan over all {len(split_tokens)} tokens → {len(edge_hits_data)} edge hits")

    if anchor_names:
        # Surface anchor info to user (debugging + transparency)
        anchor_preview = ', '.join(anchor_names[:3])
        more = '' if len(anchor_names) <= 3 else f' +{len(anchor_names) - 3} more'
        print(f"  ⚓ active_context anchored: [{anchor_preview}{more}]")

    # ─── Phase 1 postprocess: R1 recency soft weight ─────────────────
    # Pure postprocessing — does not touch propagate / Rust core / gravity.
    # superseded edges are softly down-weighted, NOT filtered.
    # See development/design/STG_PRECISION_RECALL_DESIGN.md §4.4
    if not no_recency_weight:
        from stg_engine.recall import apply_recency_weight
        activated = apply_recency_weight(engine, activated)
    res_suffix = f":{resolution}" if use_gravity and resolution != "medium" else ""
    gravity_label = f" [gravity{res_suffix}]" if use_gravity else ""

    # Community mode requires gravity (we need the GravityMap to know communities).
    # Fall back to node mode if gravity is disabled.
    if community_mode and not use_gravity:
        community_mode = False

    if community_mode:
        from stg_engine.gravity import aggregate_to_communities
        communities = aggregate_to_communities(
            engine, activated, gravity, resolution=resolution,
            k=3, query=text, top_m=top_m,
        )
        # ─── Phase 1 postprocess: R7 community dominance filter ───────
        # Fold weak communities below dominance/ratio threshold.
        # See STG_PRECISION_RECALL_DESIGN.md §4.6
        if not no_community_filter:
            from stg_engine.recall import community_dominance_filter
            communities = community_dominance_filter(communities)

        # ─── A/C anchor-aware community filter ────────────────────────
        # When exact anchors are present, only keep communities that
        # actually contain an anchor (or already have a precise hit via
        # query_seeds / name match). Pure propagate-spread topic noise
        # without query intent is suppressed entirely.
        # See STG_R6_EDGE_FALLBACK_SEED_DESIGN.md §C (community filter)
        if exact_anchors and gravity is not None:
            anchor_community_ids = set()
            for anchor in exact_anchors:
                comms_map = gravity.node_community.get(anchor.lower(), {})
                cid = comms_map.get(resolution)
                if cid is not None:
                    anchor_community_ids.add(cid)

            def _community_id(c):
                try:
                    return int(c.community_key.split("_")[-1])
                except (ValueError, IndexError, AttributeError):
                    return None

            kept = []
            for c in communities:
                cid = _community_id(c)
                if cid in anchor_community_ids or c.query_seeds or c.name_matched:
                    kept.append(c)
            communities = kept

            # Reorder representatives in anchor-bearing communities so the
            # anchor node sits at the top instead of (potentially unrelated)
            # high-elevation reps. The original aggregate didn't know about
            # the user's explicit anchor intent.
            anchor_lower = {a.lower(): a for a in exact_anchors}
            for c in communities:
                cid = _community_id(c)
                if cid not in anchor_community_ids:
                    continue
                # Build anchor RepresentativeEntry list (use existing if any)
                from stg_engine.types import RepresentativeEntry
                existing_by_key = {r.node_name.lower(): r for r in c.representatives}
                elevations = gravity.elevation_by_resolution.get(
                    resolution, gravity.node_elevation
                )
                anchor_entries: list = []
                for anchor in exact_anchors:
                    key = anchor.lower()
                    if key not in anchor_community_ids and key not in existing_by_key:
                        # Verify this anchor belongs to THIS community
                        anchor_cid = gravity.node_community.get(key, {}).get(resolution)
                        if anchor_cid != cid:
                            continue
                    if gravity.node_community.get(key, {}).get(resolution) != cid:
                        continue
                    if key in existing_by_key:
                        anchor_entries.append(existing_by_key[key])
                        continue
                    node = engine._nodes.get(key)
                    act = node.activation if node else 0.0
                    elev = elevations.get(key, 0.0)
                    anchor_entries.append(
                        RepresentativeEntry(node_name=anchor, activation=act, elevation=elev)
                    )
                # Replace representatives with anchor entries only.
                # When the user pointed at specific nodes, the other top-by-
                # elevation reps in the same community are topic context, not
                # the answer — surfacing their incoming/outgoing edges adds
                # noise. The 🎯 Anchor-pair view already shows the precise
                # connection; the community block here just confirms the
                # anchor's location.
                if anchor_entries:
                    c.representatives = anchor_entries

        print(f"propagate('{text}') → {len(communities)} community(s) "
              f"from {len(activated)} activated nodes ({elapsed*1000:.1f}ms){gravity_label}:")

        # ─── C / R8: Anchor-pair edge view (TOP PRIORITY) ────────────
        # Render BEFORE community list so the user's most explicit intent
        # ("how do these specific nodes connect?") is the first thing
        # they see, not buried under propagate's topic-level community
        # output. See STG_R6_EDGE_FALLBACK_SEED_DESIGN.md §C.
        if anchor_pair_edges:
            print(f"\n🎯 Anchor-pair edges ({len(anchor_pair_edges)}) — primary answer:")
            for edge in anchor_pair_edges:
                label = _format_edge_label(edge)
                print(f"  [{edge.source}] -[{label}]-> [{edge.target}]")
                desc = edge.modifiers.get("description")
                if desc:
                    desc_short = desc if len(desc) <= 200 else desc[:197] + "..."
                    print(f"     {desc_short}")
            print()  # blank line before community list

        for c_idx, comm in enumerate(communities, 1):
            tags = []
            if comm.name_matched:
                tags.append("✓name")
            tag_str = f"  {' '.join(tags)}" if tags else ""
            print(f"\n[{c_idx}] {comm.community_name}  "
                  f"score={comm.score:.3f}  rep_act={comm.rep_activation:.3f}{tag_str}")
            for r_idx, rep in enumerate(comm.representatives, 1):
                mark = "" if rep.activation > 0 else "  (not reached)"
                print(f"  [{c_idx}.{r_idx}] {rep.node_name}  "
                      f"act={rep.activation:.3f}  elev={rep.elevation:.3f}{mark}")
                # Inline full node detail — edges, modifiers, descriptions.
                # Skip for reps that weren't reached (no activation) to save tokens.
                if not brief and rep.activation > 0:
                    _render_node_detail(engine, rep.node_name, indent="      ", show_virtual=show_virtual)
            # Query-matching nodes inside community that aren't top reps.
            # These are the precise hits that would vanish at the community level.
            if comm.query_seeds:
                print(f"  🎯 Query matches inside community (not top reps):")
                for s_idx, seed in enumerate(comm.query_seeds, 1):
                    print(f"    [{c_idx}.s{s_idx}] {seed.node_name}  "
                          f"act={seed.activation:.3f}  elev={seed.elevation:.3f}")
                    if not brief and seed.activation > 0:
                        _render_node_detail(engine, seed.node_name, indent="      ", show_virtual=show_virtual)
    else:
        print(f"propagate('{text}') → {len(activated)} nodes ({elapsed*1000:.1f}ms){gravity_label}:")
        for idx, name in enumerate(activated, 1):
            node = engine.get_node(name)
            comm = _community_label(engine, name, resolution)
            comm_suffix = f"  [{comm}]" if comm else ""
            if node:
                print(f"  {idx:3d}. {name} (A={node.activation:.3f}){comm_suffix}")
            else:
                print(f"  {idx:3d}. {name}{comm_suffix}")

    # Show chains through activated subgraph (skip in community mode — reps already show structure)
    if not community_mode:
        _show_propagation_chains(engine, activated, all_chains=all_chains, all_modifiers=all_modifiers)

    # --expand N: auto-dump full node detail for top-N activated nodes (lossless retrieval)
    # Skip in community mode (representatives list already shows top nodes)
    if expand_top > 0 and not community_mode:
        print(f"\n  === Expanding top {min(expand_top, len(activated))} node(s) ===")
        for name in activated[:expand_top]:
            print(f"\n  --- {name} ---")
            cmd_node(engine, name)

    # Anchor-pair view rendered above (top of community block) when
    # community_mode is on. For node-mode (no gravity), render here as a
    # fallback so it still surfaces.
    if anchor_pair_edges and not community_mode:
        print(f"\n🎯 Anchor-pair edges ({len(anchor_pair_edges)}) — primary answer:")
        for edge in anchor_pair_edges:
            label = _format_edge_label(edge)
            print(f"  [{edge.source}] -[{label}]-> [{edge.target}]")
            desc = edge.modifiers.get("description")
            if desc:
                desc_short = desc if len(desc) <= 200 else desc[:197] + "..."
                print(f"     {desc_short}")

    # ─── R6: Event-Edge view ──────────────────────────────────────────
    # Render edges whose meta semantic fields matched edge-fallback tokens.
    # Mark "double hit" when an edge endpoint also appears in the propagate
    # node result (= both topic-matched and fact-matched, highest priority).
    if edge_hits_data:
        activated_lower = {n.lower() for n in activated} if activated else set()
        print(f"\n🪢 Event-edge matches ({len(edge_hits_data)}):")
        for edge, matched, score in edge_hits_data:
            label = _format_edge_label(edge)
            double_hit = (edge.source.lower() in activated_lower) or (edge.target.lower() in activated_lower)
            mark = "  🔗 双重命中" if double_hit else ""
            print(f"  [{edge.source}] -[{label}]-> [{edge.target}]  score={score:.2f}{mark}")
            desc = edge.modifiers.get("description")
            if desc:
                desc_short = desc if len(desc) <= 120 else desc[:117] + "..."
                print(f"     {desc_short}")
            print(f"     matched: {', '.join(matched)}")

    # Show learning summary
    log = engine.learning_log
    strengthened = sum(1 for e in log if e.event_type == "strengthen")
    weakened = sum(1 for e in log if e.event_type == "weaken")
    if strengthened or weakened:
        print(f"  Hebbian: +{strengthened} strengthen, -{weakened} weaken")

    # Show propagation metrics
    pm = engine.last_propagation_metrics
    if pm:
        print(f"\n  QE={pm.query_efficiency:.3f}  RS={pm.resonance_score:.3f}  "
              f"coverage={pm.coverage:.4f}  seeds={pm.seed_node_count}")

    # Save last propagate results for `select` command
    _save_last_propagate(activated)

    # Flush telemetry + save
    if engine._telemetry:
        engine._telemetry.flush(STG_PATH)
    engine.save(STG_PATH)

    # Log to propagate_log.stl.md
    _log_propagate(engine, text, elapsed)


def _save_last_propagate(activated):
    """Save last propagate results to a temp file for the `select` command."""
    import json
    cache_path = os.path.join(os.path.expanduser("~/.stg"), "last_propagate.json")
    try:
        with open(cache_path, "w") as f:
            json.dump(activated, f)
    except Exception:
        pass


def _load_last_propagate():
    """Load last propagate results."""
    import json
    cache_path = os.path.join(os.path.expanduser("~/.stg"), "last_propagate.json")
    try:
        with open(cache_path) as f:
            return json.load(f)
    except Exception:
        return None


def cmd_select(engine, args):
    """Apply user selection feedback to last propagate results.

    Usage: select 1,3,5  — select results 1, 3, and 5 as useful
    """
    from stg_engine.feedback_select import apply_selection, save_active_context

    last_results = _load_last_propagate()
    if not last_results:
        print("No previous propagate results. Run 'propagate' first.")
        return

    if not args:
        print("Usage: select 1,3,5")
        print(f"Last propagate had {len(last_results)} results.")
        return

    # Parse selection: "1,3,5" or "1 3 5"
    raw = " ".join(args).replace(",", " ")
    try:
        indices = [int(x) for x in raw.split() if x.strip()]
    except ValueError:
        print("Invalid selection. Use numbers: select 1,3,5")
        return

    # Validate range
    invalid = [i for i in indices if i < 1 or i > len(last_results)]
    if invalid:
        print(f"Invalid indices: {invalid}. Range is 1-{len(last_results)}.")
        return

    result = apply_selection(engine, last_results, indices)

    # Save engine first (salience changes), then active context
    # Order matters: engine.save() may recreate tables
    engine.save(STG_PATH)
    save_active_context(engine, result.selected_nodes, STG_PATH)

    print(f"Selected {len(result.selected_nodes)} node(s):")
    for name in result.selected_nodes:
        print(f"  ✓ {name}")
    print(f"Edges rewarded: {result.edges_rewarded}, penalized: {result.edges_penalized}")
    print(f"Active context set ({len(result.selected_nodes)} nodes)")


def _config_get_path(config: dict, dotted_key: str):
    """Look up a dotted key in a nested config dict. Returns (value, found)."""
    parts = dotted_key.split(".")
    cur = config
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return None, False
        cur = cur[p]
    return cur, True


def _config_set_path(config: dict, dotted_key: str, value) -> None:
    """Set a dotted key in a nested config dict, creating parents as needed."""
    parts = dotted_key.split(".")
    cur = config
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _config_unset_path(config: dict, dotted_key: str) -> bool:
    """Remove a dotted key. Returns True if deletion happened."""
    parts = dotted_key.split(".")
    cur = config
    for p in parts[:-1]:
        if not isinstance(cur, dict) or p not in cur:
            return False
        cur = cur[p]
    if isinstance(cur, dict) and parts[-1] in cur:
        del cur[parts[-1]]
        return True
    return False


def _config_coerce_value(raw: str, hint: str = None):
    """Coerce a raw CLI arg into a typed value based on the key hint.

    - keys ending in '.roots' / 'roots'   → split on comma
    - raw 'true' / 'false'                → bool
    - raw that's all-digits (signed OK)   → int
    - else                                → str
    """
    if hint and hint.endswith(".roots") or hint == "roots":
        return [p.strip() for p in raw.split(",") if p.strip()]
    s = raw.strip()
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    if s.lstrip("-").isdigit():
        try:
            return int(s)
        except ValueError:
            pass
    return raw


def _format_config_value(v) -> str:
    if isinstance(v, list):
        return ",".join(str(x) for x in v)
    return str(v)


def cmd_config(args):
    """User-level config (~/.stg/config.json) manipulation.

    Usage:
      stg config list                           — show all config keys
      stg config get <key>                      — read one key (supports dotted paths)
      stg config set <key> <value>              — set one key (supports dotted paths)
      stg config unset <key>                    — remove one key (supports dotted paths)

    Dotted keys nest into sub-objects. Examples:
      stg config set skill.enabled true
      stg config set skill.roots "/abs/path1,/abs/path2"
      stg config set skill.interpreters.myvenv "/abs/path/to/python3"
      stg config get skill.roots

    Value coercion:
      - "true" / "false" → boolean
      - all-digits       → integer
      - for keys ending in ".roots" or "roots" → comma-separated list
      - else             → string

    Known keys:
      default_agent                     — agent name used when no --agent/STG_AGENT given
      skill.enabled                     — bool, master switch for `stg use` (default: false)
      skill.roots                       — list[str], whitelisted script path roots (default: [])
      skill.interpreters.<name>         — str, absolute path to a named interpreter binary
      skill.default_timeout_s           — int, fallback timeout when Skill edge doesn't specify (default: 60)
      skill.max_timeout_s               — int, hard cap applied to any resolved timeout (default: 600)
      skill.output_cap_bytes            — int, max captured stdout (default: 10485760)
      feedback.session_end_hook         — str, shell-quoted command run after `feedback session-end` succeeds (default: unset). No shell invoked. Use to plug in backups, sync, etc.
      feedback.session_end_hook_timeout_s — int, max seconds the post-hook may run (default: 300)
    """
    if not args:
        print(cmd_config.__doc__.strip())
        return

    sub = args[0]
    config = _read_user_config()

    if sub == "list":
        if not config:
            print(f"(empty — {_USER_CONFIG_PATH} does not exist or has no keys)")
            return
        print(f"# {_USER_CONFIG_PATH}")
        # Render flat: walk nested dicts into dotted keys
        def _walk(prefix, node):
            if isinstance(node, dict):
                for k in sorted(node.keys()):
                    _walk(prefix + [k], node[k])
            else:
                key = ".".join(prefix)
                print(f"  {key} = {_format_config_value(node)!r}")
        _walk([], config)
        return

    if sub == "get" and len(args) >= 2:
        key = args[1]
        value, found = _config_get_path(config, key)
        if found:
            print(_format_config_value(value))
        else:
            print(f"(unset) — would fall back to built-in default", file=sys.stderr)
            sys.exit(1)
        return

    if sub == "set" and len(args) >= 3:
        key = args[1]
        raw = args[2]
        value = _config_coerce_value(raw, hint=key)
        _config_set_path(config, key, value)
        _write_user_config(config)
        print(f"{key} = {_format_config_value(value)!r}  (saved to {_USER_CONFIG_PATH})")
        return

    if sub == "unset" and len(args) >= 2:
        key = args[1]
        if _config_unset_path(config, key):
            _write_user_config(config)
            print(f"unset {key}  (saved to {_USER_CONFIG_PATH})")
        else:
            print(f"{key} was not set")
        return

    print(cmd_config.__doc__.strip())


def cmd_gravity(engine, subcmd, args):
    """Gravity map inspection commands."""
    from stg_engine.gravity import gravity_info, gravity_node_info

    t0 = time.perf_counter()
    gravity = engine.get_gravity_map()
    build_ms = (time.perf_counter() - t0) * 1000

    if subcmd == "info":
        info = gravity_info(gravity)
        if info.get("empty"):
            print("Graph is empty, no gravity map.")
            return
        print(f"=== Gravity Map ({build_ms:.0f}ms) ===")
        for res, count in info["communities"].items():
            print(f"  {res}: {count} communities")
        print(f"  Elevation: {info['elevation_min']:.3f} — {info['elevation_max']:.3f} (mean {info['elevation_mean']:.3f})")

        # Show communities with names
        listing = info.get("community_listing", {})
        for res in ("coarse", "medium", "fine"):
            entries = listing.get(res, [])
            if entries:
                print(f"\n  {res} communities:")
                for key, name, size in entries:
                    print(f"    {name:40s} ({size} nodes)")

        print(f"\n  Top 10 (highest elevation):")
        for name, elev in info["top_10"]:
            print(f"    {name:50s} {elev}")
        print(f"\n  Bottom 10 (lowest elevation):")
        for name, elev in info["bottom_10"]:
            print(f"    {name:50s} {elev}")

    elif subcmd == "node" and args:
        node_name = args[0]
        # Resolve node name
        resolved, node = _resolve_node_name(engine, node_name)
        if not resolved:
            return
        info = gravity_node_info(gravity, resolved)
        if not info:
            print(f"Node '{resolved}' not in gravity map")
            return
        print(f"Node: {info['node']}")
        print(f"  Elevation: {info['elevation']:.4f} (percentile: {info['percentile']:.1f}%)")
        if info['communities']:
            names = info.get('community_names', {})
            for res, comm_id in info['communities'].items():
                label = names.get(res, f"#{comm_id}")
                print(f"  {res} community: {label} (#{comm_id})")
        if info['representative_of']:
            print(f"  Representative of: {', '.join(info['representative_of'])}")

    elif subcmd == "heat":
        # Show community heat / recency / baseline — computed on-the-fly from
        # edge state. No stored heat to inspect; this reads current signals.
        from stg_engine.gravity import compute_community_signals
        resolution = "medium"
        target_comm = None
        for a in args:
            if a in ("coarse", "medium", "fine"):
                resolution = a
            elif a.startswith("--community="):
                target_comm = a.split("=", 1)[1]

        counts = gravity.community_counts.get(resolution, 0)
        if counts == 0:
            print(f"No communities at resolution '{resolution}'.")
            return
        all_ids = list(range(counts))
        signals = compute_community_signals(engine, gravity, all_ids, resolution=resolution)

        rows = []
        for cid, sig in signals.items():
            key = f"{resolution}_{cid}"
            name = gravity.community_names.get(key, key)
            rows.append((name, cid, sig["heat"], sig["recency"], sig["baseline"], sig["effective_heat"]))

        if target_comm:
            rows = [r for r in rows if target_comm in r[0] or str(r[1]) == target_comm]
            if not rows:
                print(f"No community matching '{target_comm}' at resolution '{resolution}'.")
                return

        # Sort by effective_heat descending
        rows.sort(key=lambda r: r[5], reverse=True)

        print(f"=== Community Heat ({resolution}, {len(rows)} communities) ===")
        print(f"  {'name':50s} {'heat':>10s} {'recency':>10s} {'baseline':>10s} {'effective':>10s}")
        for name, cid, heat, recency, baseline, eff in rows[:30]:
            print(f"  {name[:50]:50s} {heat:>10.4f} {recency:>10.4f} {baseline:>10.4f} {eff:>10.4f}")
        if not target_comm and len(rows) > 30:
            print(f"  ... ({len(rows) - 30} more; use --community=<name> to filter)")

    else:
        print("Usage: gravity info                          — show gravity map summary")
        print("       gravity node <name>                   — show node elevation and community")
        print("       gravity heat [coarse|medium|fine]     — show community heat/recency/baseline")
        print("       gravity heat --community=<name>       — filter to matching community")


def cmd_paths(engine, source, target):
    src_name, src_node = _resolve_node_name(engine, source)
    if not src_node:
        return
    tgt_name, tgt_node = _resolve_node_name(engine, target)
    if not tgt_node:
        return
    paths = engine.find_paths(src_name, tgt_name)
    if not paths:
        print(f"No paths from '{src_name}' to '{tgt_name}'")
        return
    print(f"Paths from [{src_name}] to [{tgt_name}] ({len(paths)}):")
    for i, path in enumerate(paths):
        print(f"  {i+1}. {' → '.join(path)}")
    tension = engine.compute_path_tension(src_name, tgt_name)
    print(f"Path tension: {tension:.4f}")


def _format_edge_modifiers(e, indent="      ", show_provenance=False):
    """Format edge modifiers for display.

    Returns (lines, hidden_count). Provenance/audit fields (PROVENANCE_FIELDS in
    stg_engine.engine — source, created_at, superseded_at, ...) are folded by
    default; pass show_provenance=True to expand them. The hidden_count lets
    callers print a summary footer.
    """
    lines: List[str] = []
    hidden = 0
    if not e.modifiers:
        return lines, hidden
    for k, v in e.modifiers.items():
        if k == "edge_class":
            continue  # already shown elsewhere
        if not show_provenance and k in PROVENANCE_FIELDS:
            hidden += 1
            continue
        lines.append(f"{indent}{k}: {v}")
    return lines, hidden


def _resolve_node_name(engine, name):
    """Resolve node name with case-insensitive fallback.

    Returns (resolved_name, node) or (None, None) if not found.
    If multiple case-insensitive matches, prints candidates and returns None.
    """
    # 1. Exact match
    node = engine.get_node(name)
    if node:
        return node.name, node

    # 2. Case-insensitive match
    name_lower = name.lower()
    candidates = [n for n in engine._nodes if n.lower() == name_lower]
    if len(candidates) == 1:
        resolved = candidates[0]
        return resolved, engine.get_node(resolved)
    if len(candidates) > 1:
        print(f"Ambiguous name '{name}'. Did you mean:")
        for c in sorted(candidates):
            print(f"  - {c}")
        return None, None

    # 3. Substring match (case-insensitive) — show suggestions
    matches = [n for n in engine._nodes if name_lower in n.lower()]
    if matches:
        print(f"Node '{name}' not found. Similar nodes:")
        for m in sorted(matches)[:10]:
            print(f"  - {m}")
    else:
        print(f"Node '{name}' not found")
    return None, None


def _is_virtual_edge(e):
    """Check whether an edge is an auto-generated virtual edge.

    Virtual edges are structural bridges (sibling/xref/co_source) created by
    the ingest pipeline without real descriptive content. They pollute the
    per-node view with repetitive 'virtual_reason: sibling' noise.
    """
    if getattr(e, "edge_class", "") == "virtual":
        return True
    mods = getattr(e, "modifiers", {}) or {}
    if mods.get("edge_class") == "virtual":
        return True
    return "virtual_reason" in mods


# Display thresholds for edge attributes — only render when the value carries
# discriminative signal. Defaults are silent.
_CONFIDENCE_DISPLAY_THRESHOLD = 0.5    # show c= only when "probable but uncertain" or worse
_STRENGTH_DEFAULT = 0.5                # show s= only when deviating from default
_SALIENCE_DEVIATION_TOLERANCE = 0.15   # show sal= only when Hebbian has moved it
                                       # significantly (3+ strengthen steps at default rate)


def _format_edge_attrs(edge) -> str:
    """Render the inline attribute parenthetical for an edge.

    Hides default/expected values; shows only outliers that carry signal:
    - confidence: only when < 0.5  (true low-confidence outlier — flag visually)
    - strength:   only when != 0.5 (non-default; matches STL export logic)
    - salience:   only when |sal - conf| > 0.15 (significantly Hebbian-modified;
                  ignores background micro-adjustments from a few activations)
    - rule:       only when present

    Returns empty string when no attributes deserve display, leading to a
    clean `→ [Target]` line for the common case.
    """
    parts = []
    if edge.confidence < _CONFIDENCE_DISPLAY_THRESHOLD:
        parts.append(f"c={edge.confidence}")
    if abs(edge.strength - _STRENGTH_DEFAULT) > 0.001:
        parts.append(f"s={edge.strength}")
    if abs(edge.salience - edge.confidence) > _SALIENCE_DEVIATION_TOLERANCE:
        parts.append(f"sal={edge.salience:.2f}")
    if edge.rule:
        parts.append(f'rule="{edge.rule}"')
    return f" ({', '.join(parts)})" if parts else ""


def _render_node_detail(engine, name, indent="", show_virtual=False, limit=None, show_provenance=False):
    """Print full node detail with configurable indent.

    Extracted from cmd_node so community-mode propagate can inline
    full detail under each representative without re-querying.

    Virtual edges are filtered by default (show_virtual=False) — they are
    auto-generated structural bridges with no real description, and clutter
    the output. Count of hidden virtual edges is still reported.

    Provenance fields (source/created_at/...) are folded by default to keep the
    semantic core readable; pass show_provenance=True (cli: --full) to expand.

    If `limit` is set, only the first N edges in each direction are rendered
    (useful for high-degree nodes like `Jesus` with 470+ real edges).
    """
    name, node = _resolve_node_name(engine, name)
    if not node:
        return
    pfx = indent
    print(f"{pfx}Node: {node.name}")
    if node.namespace:
        print(f"{pfx}  Namespace: {node.namespace}")
    if node.anchor_type:
        print(f"{pfx}  Type: {node.anchor_type}")
    print(f"{pfx}  Tension: {node.tension:.4f}")
    print(f"{pfx}  Activation: {node.activation:.4f}")
    print(f"{pfx}  Self-Relevance: {node.self_relevance:.4f}")

    # STL Protocol §9.4: node attributes (materialized from intrinsic-property
    # self-loops, or written by other ingest paths like markdown_extractor)
    # are summarized here with a count; `stg attrs <name>` lists them in full.
    # Default minimalism — node detail focuses on graph topology, attributes
    # are queried explicitly when needed.
    if node.metadata:
        n_keys = len(node.metadata)
        plural = "key" if n_keys == 1 else "keys"
        agent_flag = (
            f"--agent {SETTINGS['agent']} "
            if SETTINGS.get("agent", _DEFAULT_AGENT) != _DEFAULT_AGENT
            else ""
        )
        print(
            f"{pfx}  Properties: {n_keys} {plural} "
            f"(use 'stg {agent_flag}attrs \"{node.name}\"' to view)"
        )

    out_edges_all = engine.get_edges(source=name)
    in_edges_all = engine.get_edges(target=name)

    if show_virtual:
        out_edges, in_edges = out_edges_all, in_edges_all
        out_virtual = in_virtual = 0
    else:
        out_edges = [e for e in out_edges_all if not _is_virtual_edge(e)]
        in_edges = [e for e in in_edges_all if not _is_virtual_edge(e)]
        out_virtual = len(out_edges_all) - len(out_edges)
        in_virtual = len(in_edges_all) - len(in_edges)
    mod_indent = pfx + "      "
    out_shown = out_edges if limit is None else out_edges[:limit]
    in_shown = in_edges if limit is None else in_edges[:limit]
    out_truncated = len(out_edges) - len(out_shown)
    in_truncated = len(in_edges) - len(in_shown)
    total_provenance_hidden = 0
    if out_edges:
        header = f"\n{pfx}  Outgoing ({len(out_edges)})"
        if limit is not None and len(out_edges) > limit:
            header += f" [showing {limit}]"
        print(f"{header}:")
        for e in out_shown:
            print(f"{pfx}    → [{e.target}]{_format_edge_attrs(e)}")
            mod_lines, hidden = _format_edge_modifiers(
                e, indent=mod_indent, show_provenance=show_provenance
            )
            total_provenance_hidden += hidden
            for line in mod_lines:
                print(line)
        if out_truncated:
            print(f"{pfx}    (+ {out_truncated} more outgoing edge(s) truncated, raise --limit to show)")
    if out_virtual:
        print(f"{pfx}    (+ {out_virtual} virtual edge(s) hidden, use --virtual to show)")
    if in_edges:
        header = f"\n{pfx}  Incoming ({len(in_edges)})"
        if limit is not None and len(in_edges) > limit:
            header += f" [showing {limit}]"
        print(f"{header}:")
        for e in in_shown:
            print(f"{pfx}    ← [{e.source}]{_format_edge_attrs(e)}")
            mod_lines, hidden = _format_edge_modifiers(
                e, indent=mod_indent, show_provenance=show_provenance
            )
            total_provenance_hidden += hidden
            for line in mod_lines:
                print(line)
        if in_truncated:
            print(f"{pfx}    (+ {in_truncated} more incoming edge(s) truncated, raise --limit to show)")
    if in_virtual:
        print(f"{pfx}    (+ {in_virtual} virtual edge(s) hidden, use --virtual to show)")
    if total_provenance_hidden and not show_provenance:
        plural = "field" if total_provenance_hidden == 1 else "fields"
        print(
            f"{pfx}    (+ {total_provenance_hidden} provenance {plural} hidden "
            f"[source/created_at/...], use --full to show)"
        )


def cmd_node(engine, name, show_virtual=False, limit=None, show_provenance=False):
    _render_node_detail(
        engine, name, indent="",
        show_virtual=show_virtual, limit=limit, show_provenance=show_provenance,
    )


def cmd_attrs(engine, args):
    r"""Query node attributes (materialized from intrinsic-property self-loops
    or other ingest paths that write to node.metadata).

    Usage:
        stg attrs <node>                       # single node detail
        stg attrs --namespace <ns>             # all nodes in namespace
        stg attrs --field key=value [...]      # field filter (multiple, AND)
        stg attrs --where "<sql>"              # SQL where on metadata_json
        stg attrs --keys                       # discover available metadata keys
        stg attrs --limit N                    # truncate output

    Combinable:
        --namespace stacks with --field, --where, or --keys.
        --keys with a node argument lists that node's keys.

    Examples:
        stg attrs Elden_Ring
        stg attrs --namespace Game
        stg attrs --namespace Game --field release_year=2022
        stg attrs --where "JSON_EXTRACT(metadata_json,'$.release_year')>'2020'"
        stg attrs --keys                          # all keys in graph
        stg attrs --namespace Game --keys         # keys + coverage in namespace
        stg attrs Elden_Ring --keys               # just the keys, no values
    """
    namespace = None
    field_filters: Dict[str, str] = {}
    where_clause = None
    limit = None
    target = None
    keys_only = False

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--namespace" and i + 1 < len(args):
            namespace = args[i + 1]
            i += 2
        elif a == "--field" and i + 1 < len(args):
            f = args[i + 1]
            if "=" not in f:
                print(f"Invalid --field: '{f}' (expected key=value)")
                return
            k, v = f.split("=", 1)
            field_filters[k] = v
            i += 2
        elif a == "--where" and i + 1 < len(args):
            where_clause = args[i + 1]
            i += 2
        elif a == "--keys":
            keys_only = True
            i += 1
        elif a == "--limit" and i + 1 < len(args):
            try:
                limit = int(args[i + 1])
            except ValueError:
                print(f"Invalid --limit: {args[i + 1]}")
                return
            i += 2
        elif a.startswith("--"):
            print(f"Unknown flag: {a}")
            return
        else:
            target = a
            i += 1

    # ─── --keys discovery mode ─────────────────────────────────────────
    if keys_only:
        if target:
            # Node-level: list this node's keys (no values, no coverage)
            items = engine.query_metadata_keys(node_name=target)
            if not items:
                print(f"Node '{target}' has no metadata keys.")
                return
            print(f"Node: {target}")
            for k, _, _ in items:
                print(f"  {k}")
            return
        # Scope-level: namespace or whole graph, with coverage table
        items = engine.query_metadata_keys(namespace=namespace)
        if not items:
            scope_desc = f"namespace '{namespace}'" if namespace else "graph"
            print(f"No metadata keys found in {scope_desc}.")
            return
        total = items[0][2]
        FIELD_W = max((len(k) for k, _, _ in items), default=5)
        FIELD_W = max(FIELD_W, len("Field"))
        print(f"{'Field':<{FIELD_W}} | Coverage")
        print("─" * (FIELD_W + 14))
        for k, count, _ in items:
            pct = round(100 * count / total)
            print(f"{k:<{FIELD_W}} | {count}/{total} ({pct}%)")
        scope_desc = f" in namespace '{namespace}'" if namespace else ""
        print(f"\n({len(items)} unique keys across {total} nodes{scope_desc})")
        return

    # ─── Single-node mode ──────────────────────────────────────────────
    if target and not (namespace or field_filters or where_clause):
        _, node = _resolve_node_name(engine, target)
        if not node:
            print(f"Node not found: {target}")
            return
        print(f"Node: {node.name}")
        if node.namespace:
            print(f"  Namespace: {node.namespace}")
        if not node.metadata:
            print("  (no attributes)")
            return
        for k, v in node.metadata.items():
            print(f"  {k}: {v}")
        return

    # ─── List mode ─────────────────────────────────────────────────────
    if where_clause:
        try:
            results = engine.query_node_attrs_sql(
                where_clause, db_path=STG_PATH, namespace=namespace,
            )
        except ValueError as e:
            print(f"Error: {e}")
            return
        # Apply --field filters in-memory after the SQL where (AND composition)
        if field_filters:
            results = [
                n for n in results
                if all(str(n.metadata.get(k)) == str(v)
                       for k, v in field_filters.items())
            ]
    else:
        results = engine.query_node_attrs(
            namespace=namespace,
            field_filters=field_filters or None,
        )

    if not results:
        print("No matching nodes.")
        return

    # Collect union of attribute keys for table columns
    all_keys: set = set()
    for n in results:
        all_keys.update(n.metadata.keys())
    keys = sorted(all_keys)

    truncated = limit is not None and len(results) > limit
    if limit:
        results = results[:limit]

    # Render table
    NAME_W = 30
    KEY_W = 16
    header = f"{'Node':<{NAME_W}} | " + " | ".join(f"{k:<{KEY_W}}" for k in keys)
    print(header)
    print("─" * len(header))
    for n in results:
        row = " | ".join(
            f"{str(n.metadata.get(k, ''))[:KEY_W]:<{KEY_W}}"
            for k in keys
        )
        print(f"{n.name[:NAME_W]:<{NAME_W}} | {row}")

    suffix = f" (truncated to {limit})" if truncated else ""
    print(f"\n({len(results)} node(s){suffix})")


def _save_last_ingest(new_nodes, candidates):
    """Save ingest context for the `bind` command."""
    import json
    cache_path = os.path.join(os.path.expanduser("~/.stg"), "last_ingest.json")
    try:
        with open(cache_path, "w") as f:
            json.dump({"new_nodes": list(new_nodes), "candidates": candidates}, f)
    except Exception:
        pass


def _load_last_ingest():
    """Load last ingest context."""
    import json
    cache_path = os.path.join(os.path.expanduser("~/.stg"), "last_ingest.json")
    try:
        with open(cache_path) as f:
            return json.load(f)
    except Exception:
        return None


def _detect_community(engine, new_nodes, nodes_before, gravity_snapshot):
    """Detect community for new nodes based on their connections.

    Uses a pre-ingest gravity snapshot to look up community membership
    of existing nodes (nodes_before). Checks both source and target of
    edges involving new nodes.

    Args:
        engine: STG engine (post-ingest state)
        new_nodes: Set of newly created node names
        nodes_before: Set of node names that existed before ingest
        gravity_snapshot: (node_community, node_elevation) from pre-ingest gravity map.
                         node_community: Dict[str, Dict[str, int]]
                         node_elevation: Dict[str, float]

    Returns:
        (community_id, anchor_node) if determinable, (None, None) otherwise.
    """
    nc, elev = gravity_snapshot
    if not nc:
        return None, None

    max_elev = max(elev.values()) if elev else 1

    # Look at edges involving new nodes → find existing endpoints with community
    for e in engine._edges:
        if (e.modifiers or {}).get("edge_class") == "virtual":
            continue
        src_is_new = e.source in new_nodes
        tgt_is_new = e.target in new_nodes
        if not src_is_new and not tgt_is_new:
            continue

        # Check the existing endpoint (must be in nodes_before)
        if src_is_new and e.target in nodes_before:
            anchor = e.target
        elif tgt_is_new and e.source in nodes_before:
            anchor = e.source
        else:
            # Both new or neither connects to pre-existing node
            continue

        if anchor in nc:
            levels = nc[anchor]
            if isinstance(levels, dict) and "coarse" in levels:
                # Skip super-hubs: only in large graphs (1000+ nodes)
                # AND only if the anchor has extremely high elevation
                # (top 1%). In smaller graphs, every hub is a valid anchor.
                anchor_elev = elev.get(anchor, 0)
                if len(nc) > 1000 and anchor_elev > max_elev * 0.99:
                    continue
                return levels["coarse"], anchor

    return None, None


def cmd_ingest(engine, stl_text, edge_class="knowledge", no_link=False):
    """Ingest STL text into the graph.

    Community-aware auto-bind:
    - If community is determinable from existing endpoints → auto-bind
      to same-community candidates (no manual bind needed).
    - If not determinable → show candidate list for agent to bind.

    Args:
        edge_class: 'knowledge' (default) or 'cognitive' (agent's own reasoning)
        no_link: If True, skip candidate suggestion (for batch/script use)
    """
    # Track existing nodes to detect newly created ones
    nodes_before = set(engine._nodes.keys())

    # Snapshot gravity BEFORE ingest (for community detection)
    gravity_snapshot = ({}, {})
    try:
        gmap = engine.get_gravity_map()
        if gmap and hasattr(gmap, "node_community") and hasattr(gmap, "node_elevation"):
            gravity_snapshot = (dict(gmap.node_community), dict(gmap.node_elevation))
    except Exception:
        pass

    # For --cognitive flag, we need to temporarily patch ingest to add edge_class
    if edge_class != "knowledge":
        # Wrap add_edge to inject edge_class
        original_add_edge = engine.add_edge
        def patched_add_edge(*args, **kwargs):
            kwargs.setdefault("edge_class", edge_class)
            return original_add_edge(*args, **kwargs)
        engine.add_edge = patched_add_edge
        try:
            count = engine.ingest_stl(stl_text)
        finally:
            engine.add_edge = original_add_edge
    else:
        count = engine.ingest_stl(stl_text)

    new_nodes = set(engine._nodes.keys()) - nodes_before

    # G7: Show entity resolution warnings (similar existing nodes detected)
    er_candidates = getattr(engine, "_last_entity_candidates", None)
    if er_candidates:
        for display_name, score, reason in er_candidates:
            print(f"\n⚠ Similar node exists: \"{display_name}\" ({reason}, score={score:.2f})")
            print(f"  → To use existing node, re-ingest with that name")
            print(f"  → To register alias: stg alias add <new_name> {display_name}")
        engine._last_entity_candidates = None

    engine.save(STG_PATH)
    print(f"Ingested {count} edge(s) (edge_class={edge_class}). Saved to memory.stg.")
    s = engine.get_stats()
    print(f"Graph: {s['node_count']} nodes, {s['edge_count']} edges")

    if not new_nodes or no_link:
        return

    # Detect community from existing endpoints (using pre-ingest gravity)
    comm_id, anchor = _detect_community(engine, new_nodes, nodes_before, gravity_snapshot)

    # Build propagate query from new node names + descriptions
    query_parts = list(new_nodes)
    for nn in new_nodes:
        for e in engine._edges:
            if (e.source == nn or e.target == nn) and e.modifiers:
                desc = e.modifiers.get("description", "")
                if desc:
                    query_parts.append(desc)
    query_text = " ".join(query_parts)

    # Propagate to find candidates
    activated = engine.propagate(query_text, threshold=0.001)
    all_candidates = [n for n in activated if n not in new_nodes]

    if comm_id is not None:
        # --- AUTO-BIND: community known ---
        # Filter to same-community candidates only (using pre-ingest snapshot)
        nc_snap = gravity_snapshot[0]

        same_comm = [
            n for n in all_candidates
            if isinstance(nc_snap.get(n, {}), dict)
            and nc_snap.get(n, {}).get("coarse") == comm_id
        ][:5]  # top 5 same-community

        if same_comm:
            bound = 0
            for new_node in new_nodes:
                for ctx_node in same_comm:
                    if ctx_node != new_node and ctx_node in engine._nodes:
                        engine.add_edge(
                            ctx_node, new_node,
                            confidence=0.15,
                            edge_class="virtual",
                            virtual_reason="auto_bind",
                        )
                        bound += 1
            engine.save(STG_PATH)
            print(f"\nAuto-bind: community={comm_id} (anchor: {anchor})")
            print(f"  Bound {len(new_nodes)} new node(s) to {len(same_comm)} candidate(s) ({bound} virtual edge(s)):")
            for name in same_comm:
                print(f"    → {name}")
        else:
            print(f"\nCommunity detected ({comm_id}) but no propagate candidates in same community.")
    else:
        # --- MANUAL BIND: community unknown ---
        candidates = all_candidates[:15]
        if candidates:
            print(f"\nCommunity not determinable. Candidates for manual binding:")
            for idx, name in enumerate(candidates, 1):
                node = engine.get_node(name)
                act = f" (A={node.activation:.3f})" if node else ""
                print(f"  {idx:3d}. {name}{act}")
            print(f"\nUse `bind 1,3,5` to link new node(s) to selected candidates.")
            _save_last_ingest(new_nodes, candidates)
        else:
            print("No candidates found for context binding.")


def cmd_bind(engine, args):
    """Bind new nodes from last ingest to selected candidate communities.

    Usage: bind 1,3,5  — link to candidates 1, 3, and 5
    """
    last = _load_last_ingest()
    if not last:
        print("No previous ingest with candidates. Run 'ingest' first.")
        return

    new_nodes = last["new_nodes"]
    candidates = last["candidates"]

    if not args:
        print("Usage: bind 1,3,5")
        print(f"Last ingest created {len(new_nodes)} node(s), {len(candidates)} candidates.")
        return

    # Parse selection: "1,3,5" or "1 3 5"
    raw = " ".join(args).replace(",", " ")
    try:
        indices = [int(x) for x in raw.split() if x.strip()]
    except ValueError:
        print("Invalid selection. Use numbers: bind 1,3,5")
        return

    # Validate range
    invalid = [i for i in indices if i < 1 or i > len(candidates)]
    if invalid:
        print(f"Invalid indices: {invalid}. Range is 1-{len(candidates)}.")
        return

    selected = [candidates[i - 1] for i in indices]
    bound = 0
    for new_node in new_nodes:
        for ctx_node in selected:
            if ctx_node != new_node and ctx_node in engine._nodes:
                engine.add_edge(
                    ctx_node, new_node,
                    confidence=0.15,
                    edge_class="virtual",
                    virtual_reason="context_binding",
                )
                bound += 1

    engine.save(STG_PATH)
    print(f"Bound {len(new_nodes)} new node(s) to {len(selected)} candidate(s) ({bound} virtual edge(s)):")
    for name in selected:
        print(f"  → {name}")


def cmd_alias(engine, subcmd, args):
    """Manage entity aliases (G7 Entity Resolution).

    Subcommands:
        add <alias> <canonical>   Register alias → canonical mapping
        list                      Show all registered aliases
        remove <alias>            Remove an alias
        resolve <name>            Show what a name resolves to
    """
    if subcmd == "add" and len(args) >= 2:
        alias_name, canonical_name = args[0], args[1]
        ok = engine.register_alias(alias_name, canonical_name)
        if ok:
            engine.save(STG_PATH)
            resolved = engine.resolve_name(alias_name)
            print(f"Alias registered: \"{alias_name}\" → \"{resolved}\"")
        else:
            print(f"Failed: canonical node \"{canonical_name}\" not found in graph.")
    elif subcmd == "list":
        aliases = engine.list_aliases()
        if aliases:
            print(f"Aliases ({len(aliases)}):")
            for alias_key, canonical_display in aliases:
                print(f"  {alias_key} → {canonical_display}")
        else:
            print("No aliases registered.")
    elif subcmd == "remove" and len(args) >= 1:
        ok = engine.remove_alias(args[0])
        if ok:
            engine.save(STG_PATH)
            print(f"Alias removed: \"{args[0]}\"")
        else:
            print(f"Alias \"{args[0]}\" not found.")
    elif subcmd == "resolve" and len(args) >= 1:
        resolved = engine.resolve_name(args[0])
        if resolved != args[0]:
            print(f"\"{args[0]}\" → \"{resolved}\" (via alias)")
        else:
            node = engine.get_node(args[0])
            if node:
                print(f"\"{args[0]}\" → \"{node.name}\" (direct node)")
            else:
                print(f"\"{args[0]}\" — not found (no alias, no node)")
    else:
        print("Usage: stg alias <add|list|remove|resolve> [args...]")
        print("  add <alias> <canonical>   Register alias → canonical")
        print("  list                      Show all aliases")
        print("  remove <alias>            Remove an alias")
        print("  resolve <name>            Check resolution")


def cmd_ingest_file(engine, file_path, created_at=None):
    """Ingest STL statements from a text file into the graph.

    Reads any text file (.md, .txt, .stl, etc.) containing STL statements
    and ingests all parsed edges. The file extension does not matter —
    only the content is parsed.

    Args:
        file_path: Path to file containing STL statements
        created_at: Optional epoch timestamp for all edges
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    count = engine.ingest_stl_file(file_path, created_at=created_at)
    engine.save(STG_PATH)
    print(f"Ingested {count} edge(s) from {os.path.basename(file_path)}. Saved to memory.stg.")
    s = engine.get_stats()
    print(f"Graph: {s['node_count']} nodes, {s['edge_count']} edges")


def cmd_merge(engine, stl_text):
    """Merge (patch) modifiers into an existing edge.

    Accepts STL syntax: [Source] -> [Target] ::mod(key=value, ...)
    The (source, target) edge must already exist.
    """
    from stg_engine.merge import EdgeMerger

    # Parse STL without ingesting — extract source, target, and modifiers
    try:
        from stl_parser import parse
        result = parse(stl_text)
        if not result.statements:
            print(f"Could not parse STL: {stl_text}")
            return
        stmt = result.statements[0]
        source = f"{stmt.source.namespace}:{stmt.source.name}" if stmt.source.namespace else stmt.source.name
        target = f"{stmt.target.namespace}:{stmt.target.name}" if stmt.target.namespace else stmt.target.name

        # Collect modifiers — Modifier is a Pydantic v2 model, iterate as (key, value)
        patch = {}
        mod_obj = stmt.modifiers
        for key, value in mod_obj:
            if value is None or key == "custom":
                continue
            if isinstance(value, dict) and not value:
                continue
            patch[key] = value
        # Also check custom dict
        custom = getattr(mod_obj, "custom", {})
        if custom:
            patch.update(custom)

    except Exception as e:
        print(f"Parse error: {e}")
        return

    # Apply merge
    try:
        conf = patch.pop("confidence", None)
        if conf is not None:
            conf = float(conf)
        strength = patch.pop("strength", None)
        if strength is not None:
            strength = float(strength)
        rule = patch.pop("rule", None)

        EdgeMerger.merge_edge(
            engine, source, target,
            confidence=conf, strength=strength, rule=rule,
            **patch,
        )
        engine.save(STG_PATH)
        fields_added = len(patch) + (1 if conf is not None else 0) + (1 if rule is not None else 0)
        print(f"Merged [{source}] -> [{target}]: {fields_added} field(s) patched")
    except KeyError:
        print(f"No existing edge [{source}] -> [{target}] to merge into. Use 'ingest' first.")


def cmd_consolidate(engine, args):
    """Consolidate multi-edges on a pair, or scan the whole graph.

    Usage:
        consolidate <source> <target>   — consolidate one pair
        consolidate --all               — consolidate all mergeable pairs
        consolidate --all --dry-run     — show candidates without changing
    """
    from stg_engine.merge import EdgeMerger

    dry_run = "--dry-run" in args
    all_mode = "--all" in args
    clean_args = [a for a in args if not a.startswith("--")]

    if all_mode:
        candidates = EdgeMerger.find_mergeable_pairs(engine)
        if not candidates:
            print("No multi-edge pairs found.")
            return

        if dry_run:
            print(f"Found {len(candidates)} mergeable pair(s):")
            for src, tgt, count in candidates:
                print(f"  [{src}] -> [{tgt}]: {count} edges")
            return

        total_merged = 0
        total_errors = 0
        for src, tgt, count in candidates:
            try:
                result = EdgeMerger.consolidate_edges(engine, src, tgt)
                if result:
                    total_merged += 1
                    print(f"  [{src}] -> [{tgt}]: {result.edges_merged} edges → 1")
            except ValueError as e:
                total_errors += 1
                print(f"  [{src}] -> [{tgt}]: SKIP ({e})")

        engine.save(STG_PATH)
        s = engine.get_stats()
        print(f"\nConsolidated {total_merged} pair(s) ({total_errors} skipped)")
        print(f"Graph: {s['node_count']} nodes, {s['edge_count']} edges")

    elif len(clean_args) >= 2:
        source, target = clean_args[0], clean_args[1]
        try:
            result = EdgeMerger.consolidate_edges(engine, source, target)
            if result is None:
                print(f"[{source}] -> [{target}]: only 0-1 edges, nothing to consolidate.")
            else:
                engine.save(STG_PATH)
                print(f"Consolidated [{source}] -> [{target}]: {result.edges_merged} edges → 1")
                s = engine.get_stats()
                print(f"Graph: {s['node_count']} nodes, {s['edge_count']} edges")
        except ValueError as e:
            print(f"Cannot consolidate: {e}")
    else:
        print("Usage: consolidate <source> <target>")
        print("       consolidate --all [--dry-run]")


def cmd_xref(engine, args):
    """Cross-reference resolution: surface latent connections from descriptions.

    Usage:
        xref                    — scan all edges, create virtual XRef edges
        xref --dry-run          — show candidates without creating edges
        xref --node <name>      — only scan edges of one node
    """
    from stg_engine.xref import XRefResolver

    dry_run = "--dry-run" in args
    node_filter = None
    clean_args = [a for a in args if not a.startswith("--")]

    if "--node" in args:
        idx = args.index("--node")
        if idx + 1 < len(args):
            node_filter = args[idx + 1]
        else:
            print("Usage: xref --node <name>")
            return

    if node_filter:
        report = XRefResolver.resolve_node(engine, node_filter, dry_run=dry_run)
    else:
        report = XRefResolver.resolve(engine, dry_run=dry_run)

    label = " (dry run)" if dry_run else ""
    print(f"XRef Cross-Reference Resolution{label}")
    print(f"  Edges scanned: {report.edges_scanned}")
    print(f"  Descriptions found: {report.descriptions_found}")
    print(f"  Candidates: {report.candidates_found}")
    print(f"  {'Would create' if dry_run else 'Created'}: {report.edges_created} virtual edges")
    print(f"  Skipped (existing neighbor): {report.edges_skipped_existing}")
    print(f"  Skipped (low IDF): {report.edges_skipped_idf}")

    if report.results:
        # Show cross-community bridges (most interesting)
        created = [r for r in report.results if r.edge_created or dry_run]
        if created:
            created.sort(key=lambda r: r.idf_score, reverse=True)
            show = created[:20]
            print(f"\n  Top cross-references ({len(show)} of {len(created)}):")
            for r in show:
                marker = "+" if r.edge_created else "~"
                print(f"    {marker} [{r.source}] → [{r.target}] (token: \"{r.matched_token}\", IDF: {r.idf_score:.2f})")

    if not dry_run and report.edges_created > 0:
        engine.save(STG_PATH)
        s = engine.get_stats()
        print(f"\nGraph: {s['node_count']} nodes, {s['edge_count']} edges")


def cmd_epistemic(engine, subcmd, args):
    """Epistemic metadata operations."""
    if subcmd == "summary":
        _epistemic_summary(engine)
    elif subcmd == "query":
        _epistemic_query(engine, args)
    elif subcmd == "validate":
        _epistemic_validate(engine)
    else:
        print("Usage: epistemic [summary|query|validate]")
        print("  summary   — Show epistemic composition of the graph")
        print("  query     — Query edges with epistemic filters")
        print("  validate  — Validate epistemic metadata on knowledge edges")


def _epistemic_summary(engine):
    """Print epistemic composition of the graph."""
    summary = engine.epistemic_summary()
    total = summary["total_edge_count"]
    print(f"Epistemic Summary ({total} edges)")
    print("─" * 40)

    print("\nEdge Classes:")
    for cls, count in sorted(summary["edge_class_distribution"].items()):
        pct = (count / total * 100) if total else 0
        print(f"  {cls:12s}  {count:>5d}  ({pct:.1f}%)")

    kn = summary["knowledge_edge_count"]
    if kn > 0:
        print(f"\nKnowledge Edges by Trace Type:")
        for tt, count in sorted(summary["trace_type_distribution"].items()):
            if count > 0:
                pct = (count / kn * 100) if kn else 0
                print(f"  {tt:28s}  {count:>5d}  ({pct:.1f}%)")

    if summary["verification_status_distribution"]:
        print(f"\nVerification Status:")
        for vs, count in sorted(summary["verification_status_distribution"].items()):
            print(f"  {vs:28s}  {count:>5d}")

    if summary["scope_distribution"]:
        print(f"\nScope:")
        for sc, count in sorted(summary["scope_distribution"].items()):
            print(f"  {sc:14s}  {count:>5d}")


def _epistemic_query(engine, args):
    """Query edges with epistemic filters."""
    # Parse flags
    edge_class = None
    trace_type = None
    verification_status = None
    epistemic_status = None
    scope = None
    min_conf = None
    min_sc = None
    limit = 20

    i = 0
    while i < len(args):
        if args[i] == "--edge-class" and i + 1 < len(args):
            edge_class = args[i + 1]; i += 2
        elif args[i] == "--trace-type" and i + 1 < len(args):
            trace_type = args[i + 1]; i += 2
        elif args[i] == "--verification" and i + 1 < len(args):
            verification_status = args[i + 1]; i += 2
        elif args[i] == "--epistemic-status" and i + 1 < len(args):
            epistemic_status = args[i + 1]; i += 2
        elif args[i] == "--scope" and i + 1 < len(args):
            scope = args[i + 1]; i += 2
        elif args[i] == "--min-confidence" and i + 1 < len(args):
            try: min_conf = float(args[i + 1])
            except ValueError: pass
            i += 2
        elif args[i] == "--min-coherence" and i + 1 < len(args):
            try: min_sc = float(args[i + 1])
            except ValueError: pass
            i += 2
        elif args[i] == "--limit" and i + 1 < len(args):
            try: limit = int(args[i + 1])
            except ValueError: pass
            i += 2
        else:
            i += 1

    results = engine.query_edges(
        edge_class=edge_class,
        trace_type=trace_type,
        verification_status=verification_status,
        epistemic_status=epistemic_status,
        scope=scope,
        min_confidence=min_conf,
        min_structural_coherence=min_sc,
        limit=limit,
    )

    # Build filter description
    filters = []
    if edge_class: filters.append(f"edge_class={edge_class}")
    if trace_type: filters.append(f"trace_type={trace_type}")
    if verification_status: filters.append(f"verification={verification_status}")
    if epistemic_status: filters.append(f"epistemic_status={epistemic_status}")
    if scope: filters.append(f"scope={scope}")
    if min_conf is not None: filters.append(f"confidence>={min_conf}")
    if min_sc is not None: filters.append(f"coherence>={min_sc}")
    filter_str = ", ".join(filters) if filters else "none"

    print(f"Edges matching: {filter_str}  ({len(results)} results)")
    print("─" * 60)
    for e in results:
        ec = e.modifiers.get("edge_class", "structural")
        tt = e.modifiers.get("trace_type", "")
        vs = e.modifiers.get("verification_status", "")
        line = f"  [{e.source}] -> [{e.target}]  conf={e.confidence:.2f}"
        if ec != "structural":
            line += f"  class={ec}"
        if tt:
            line += f"  trace={tt}"
        if vs:
            line += f"  verified={vs}"
        print(line)


def _epistemic_validate(engine):
    """Validate epistemic metadata on knowledge edges."""
    from stg_engine.epistemic import validate_epistemic_metadata

    knowledge_edges = engine.query_edges(edge_class="knowledge", limit=10000)
    if not knowledge_edges:
        print("No knowledge edges found.")
        return

    print(f"Validating {len(knowledge_edges)} knowledge edges...")
    warn_count = 0
    pass_count = 0

    for e in knowledge_edges:
        warnings = validate_epistemic_metadata(e.confidence, e.modifiers)
        # Filter out the "epistemic on knowledge" warning — that's expected
        warnings = [w for w in warnings if "designed for knowledge" not in w]
        if warnings:
            warn_count += 1
            for w in warnings:
                print(f"  ⚠ [{e.source}] -> [{e.target}]: {w}")
        else:
            pass_count += 1

    total = len(knowledge_edges)
    print(f"  ✓ {pass_count}/{total} edges pass validation")
    if warn_count:
        print(f"  ⚠ {warn_count} warnings (see above)")


def cmd_virtual(engine, subcmd, args):
    """Virtual edge operations."""
    if subcmd == "stats":
        _virtual_stats(engine)
    elif subcmd == "list":
        _virtual_list(engine)
    elif subcmd == "clear":
        _virtual_clear(engine)
    elif subcmd == "rebuild":
        _virtual_rebuild(engine)
    else:
        print("Usage: virtual [stats|list|clear|rebuild]")
        print("  stats   — Virtual edge count and reason distribution")
        print("  list    — List all virtual edges")
        print("  clear   — Remove all virtual edges")
        print("  rebuild — Clear and rebuild virtual edges from graph topology")


def _virtual_stats(engine):
    vs = engine.get_virtual_edge_stats()
    print(f"Total edges: {vs['total_edges']}")
    print(f"  Real:    {vs['real_edges']}")
    print(f"  Virtual: {vs['virtual_edges']}")
    if vs['reason_distribution']:
        print(f"\nVirtual edge reasons:")
        for reason, count in sorted(vs['reason_distribution'].items(), key=lambda x: -x[1]):
            print(f"  {reason:12s}  {count}")


def _virtual_list(engine):
    virtual = [e for e in engine._edges if e.modifiers.get("edge_class") == "virtual"]
    if not virtual:
        print("No virtual edges.")
        return
    print(f"Virtual edges ({len(virtual)}):")
    for e in virtual:
        reason = e.modifiers.get("virtual_reason", "?")
        parent = e.modifiers.get("virtual_parent", "?")
        print(f"  [{e.source}] ~ [{e.target}]  reason={reason}  parent={parent}  conf={e.confidence:.2f}")


def _virtual_clear(engine):
    count = engine.clear_virtual_edges()
    engine.save(STG_PATH)
    print(f"Cleared {count} virtual edge(s). Saved.")


def _virtual_rebuild(engine):
    count = engine.rebuild_virtual_edges()
    engine.save(STG_PATH)
    print(f"Rebuilt {count} virtual edge(s). Saved.")
    vs = engine.get_virtual_edge_stats()
    if vs['reason_distribution']:
        for reason, c in sorted(vs['reason_distribution'].items(), key=lambda x: -x[1]):
            print(f"  {reason}: {c}")


def cmd_metrics(engine):
    t0 = time.perf_counter()
    m = engine.get_metrics()
    elapsed = time.perf_counter() - t0

    print("=== Graph Metrics ===")
    print(f"\nTopology:")
    print(f"  Nodes: {m.node_count}")
    print(f"  Edges: {m.edge_count}")
    print(f"  Density: {m.density:.6f}")
    print(f"  Avg Degree: {m.avg_degree:.2f}")
    print(f"  Max Degree: {m.max_degree} ({m.max_degree_node})")

    print(f"\nInformation Theory:")
    print(f"  Entropy: {m.entropy:.4f} bits")
    print(f"  Criticality: {m.criticality:.4f}", end="")
    if 0.5 <= m.criticality <= 0.7:
        print("  [edge of chaos — optimal]")
    elif m.criticality < 0.5:
        print("  [rigid]")
    else:
        print("  [chaotic]")

    print(f"\nConfidence Distribution:")
    print(f"  Mean: {m.confidence_mean:.3f}  Median: {m.confidence_median:.3f}  StDev: {m.confidence_stdev:.3f}")
    print(f"  High (>=0.8): {m.high_confidence_ratio:.1%}  Low (<0.3): {m.low_confidence_ratio:.1%}")

    print(f"\nConnectivity:")
    print(f"  Weakly Connected Components: {m.weakly_connected_components}")
    print(f"  Largest Component: {m.largest_component_ratio:.1%} of graph")

    print(f"\nNamespaces ({m.namespace_count}):")
    for ns, count in sorted(m.namespaces.items(), key=lambda x: -x[1]):
        print(f"  {ns}: {count}")

    print(f"\n({elapsed*1000:.1f}ms)")


def cmd_importance(engine, top_n=20):
    t0 = time.perf_counter()
    importance = engine.get_importance_field()
    elapsed = time.perf_counter() - t0

    if not importance:
        print("No importance data (empty graph)")
        return

    ranked = sorted(importance.items(), key=lambda x: -x[1])
    print(f"=== Importance Field (top {top_n}/{len(ranked)}) ===")
    for i, (name, score) in enumerate(ranked[:top_n]):
        bar = "█" * int(score * 1000)  # visual bar
        print(f"  {i+1:3d}. {name:40s} {score:.6f} {bar}")

    print(f"\n({elapsed*1000:.1f}ms)")


def cmd_learn(engine, subcmd, args):
    if subcmd == "status":
        print(f"Learning: {'enabled' if engine.learning_enabled else 'disabled'}")
        print(f"Importance weight: {engine.importance_weight}")
        print(f"Learning log entries: {len(engine.learning_log)}")
        if engine._learner:
            s = engine._learner.stats
            print(f"  Strengthened: {s['strengthened']}")
            print(f"  Weakened: {s['weakened']}")

    elif subcmd == "path" and args:
        path = args
        t0 = time.perf_counter()
        events = engine.learn_from_path(path)
        elapsed = time.perf_counter() - t0
        engine.save(STG_PATH)
        if events:
            print(f"Strengthened {len(events)} edge(s) ({elapsed*1000:.1f}ms):")
            for ev in events:
                print(f"  [{ev.source}] -> [{ev.target}]: "
                      f"{ev.old_confidence:.4f} → {ev.new_confidence:.4f}")
            print("Saved.")
        else:
            print("No edges found along path.")

    elif subcmd == "propagate" and args:
        text = " ".join(args)
        engine.enable_learning()
        t0 = time.perf_counter()
        activated = engine.propagate(text)
        elapsed = time.perf_counter() - t0

        # Show propagation results
        print(f"propagate('{text}') with learning → {len(activated)} nodes ({elapsed*1000:.1f}ms)")
        for name in activated[:10]:
            node = engine.get_node(name)
            if node:
                print(f"  {name} (A={node.activation:.3f})")

        # Show learning results
        log = engine.learning_log
        strengthened = [e for e in log if e.event_type == "strengthen"]
        weakened = [e for e in log if e.event_type == "weaken"]
        print(f"\nLearning: {len(strengthened)} strengthened, {len(weakened)} weakened")
        if strengthened:
            print("Top strengthened:")
            for ev in sorted(strengthened, key=lambda e: e.new_confidence - e.old_confidence, reverse=True)[:5]:
                print(f"  [{ev.source}] -> [{ev.target}]: "
                      f"{ev.old_confidence:.4f} → {ev.new_confidence:.4f} "
                      f"(+{ev.new_confidence - ev.old_confidence:.4f})")

        pm = engine.last_propagation_metrics
        if pm:
            print(f"\nQE={pm.query_efficiency:.3f}  RS={pm.resonance_score:.3f}  "
                  f"coverage={pm.coverage:.4f}")

        engine.save(STG_PATH)
        print("Saved.")
    else:
        print("Usage: learn status | learn path <n1> <n2> ... | learn propagate <text>")


def cmd_prune(engine, dry_run=False, conf=0.1, days=30.0):
    from stg_engine.learning import SynapticPruner
    pruner = SynapticPruner(
        confidence_threshold=conf,
        unused_days=days,
    )

    if dry_run:
        t0 = time.perf_counter()
        candidates = pruner.dry_run(engine)
        elapsed = time.perf_counter() - t0
        if candidates:
            print(f"Prune candidates ({len(candidates)}) ({elapsed*1000:.1f}ms):")
            for src, tgt, c, eid in candidates:
                print(f"  [{src}] -> [{tgt}] (conf={c:.4f}, eid={eid:.6f})")
        else:
            print(f"No prune candidates ({elapsed*1000:.1f}ms)")
    else:
        t0 = time.perf_counter()
        events = pruner.prune(engine, stg_path=STG_PATH)
        elapsed = time.perf_counter() - t0
        pruned_edges = [e for e in events if e.event_type == "prune"]
        pruned_nodes = [e for e in events if e.event_type == "prune_orphan"]
        print(f"Pruned {len(pruned_edges)} edge(s), {len(pruned_nodes)} orphan node(s) ({elapsed*1000:.1f}ms)")
        if pruned_edges:
            for ev in pruned_edges[:10]:
                print(f"  [{ev.source}] -> [{ev.target}] (was conf={ev.old_confidence:.4f})")
            if len(pruned_edges) > 10:
                print(f"  ... and {len(pruned_edges) - 10} more")
        if events:
            engine.save(STG_PATH)
            s = engine.get_stats()
            print(f"Graph: {s['node_count']} nodes, {s['edge_count']} edges. Saved.")
        else:
            print("Graph unchanged.")


def cmd_telemetry(engine, subcmd, args):
    """Telemetry commands for real usage data analysis."""
    from stg_engine.telemetry import (
        telemetry_status, telemetry_frequency, telemetry_salience,
        telemetry_learning, telemetry_report, generate_calibrated_queries,
    )
    import datetime

    if subcmd == "status":
        status = telemetry_status(STG_PATH)
        if not status.get("available"):
            print("No telemetry data found.")
            print("Enable telemetry: the engine auto-enables it when feedback is active.")
            return
        print("=== Telemetry Status ===")
        print(f"  Propagations: {status.get('telemetry_propagations_count', 0)}")
        print(f"  Tracked nodes: {status.get('telemetry_node_freq_count', 0)}")
        print(f"  Sessions: {status.get('telemetry_sessions_count', 0)}")
        print(f"  Edge mutations: {status.get('telemetry_edge_mutations_count', 0)}")
        if status.get("first_propagation"):
            first = datetime.datetime.fromtimestamp(status["first_propagation"])
            last = datetime.datetime.fromtimestamp(status["last_propagation"])
            print(f"  Date range: {first.strftime('%Y-%m-%d %H:%M')} → {last.strftime('%Y-%m-%d %H:%M')}")

    elif subcmd == "frequency":
        top_n = 20
        if args:
            try:
                top_n = int(args[0])
            except ValueError:
                if "--top" in args:
                    idx = args.index("--top")
                    if idx + 1 < len(args):
                        try:
                            top_n = int(args[idx + 1])
                        except ValueError:
                            pass
        freq = telemetry_frequency(STG_PATH, top_n=top_n)
        if not freq["nodes"]:
            print("No frequency data. Run some propagations with telemetry enabled first.")
            return
        print(f"Node Activation Frequency (top {top_n} of {freq['total_propagations']} propagations):")
        print(f"  {'Node':<40} {'Count':>6} {'Seeds':>6} {'AvgAct':>8}")
        print(f"  {'─'*40} {'─'*6} {'─'*6} {'─'*8}")
        for n in freq["nodes"]:
            print(
                f"  {n['name']:<40} {n['activation_count']:>6} "
                f"{n['seed_count']:>6} {n['avg_activation']:>8.4f}"
            )

    elif subcmd == "salience":
        sal = telemetry_salience(STG_PATH)
        if not sal["sessions"]:
            print("No session data. Run 'feedback session-end' to record a session snapshot.")
            return
        print("Salience Distribution Trend:")
        print(f"  {'Session':>20} {'P25':>7} {'P50':>7} {'P75':>7} {'Mean':>7} {'Ψ':>7}")
        print(f"  {'─'*20} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}")
        for s in sal["sessions"]:
            ts = datetime.datetime.fromtimestamp(s["timestamp"]).strftime("%Y-%m-%d %H:%M")
            print(
                f"  {ts:>20} {s['salience_p25']:>7.4f} {s['salience_p50']:>7.4f} "
                f"{s['salience_p75']:>7.4f} {s['salience_mean']:>7.4f} {s['psi']:>7.4f}"
            )

    elif subcmd == "learning":
        learn = telemetry_learning(STG_PATH)
        if not learn["summary"]:
            print("No learning event data.")
            return
        print("Learning Events Summary:")
        for etype, info in learn["summary"].items():
            print(
                f"  {etype}: {info['count']} events, "
                f"avg Δ={info['avg_delta']:.4f}, max Δ={info['max_delta']:.4f}"
            )
        if learn["mutations"]:
            print(f"\nMost Frequently Mutated Edges:")
            print(f"  {'Source':<25} {'Target':<25} {'Type':>10} {'Count':>5} {'AvgΔ':>8}")
            print(f"  {'─'*25} {'─'*25} {'─'*10} {'─'*5} {'─'*8}")
            for m in learn["mutations"][:15]:
                print(
                    f"  {m['source']:<25} {m['target']:<25} "
                    f"{m['event_type']:>10} {m['cnt']:>5} {m['avg_delta']:>8.4f}"
                )

    elif subcmd == "calibrate":
        queries = generate_calibrated_queries(STG_PATH)
        if not queries:
            print("No telemetry data to calibrate from.")
            return
        high = [q for q in queries if q["frequency"] == "high"]
        med = [q for q in queries if q["frequency"] == "medium"]
        low = [q for q in queries if q["frequency"] == "low"]
        print(f"Generated {len(queries)} calibrated queries from real usage data:")
        print(f"  High frequency: {len(high)}")
        print(f"  Medium frequency: {len(med)}")
        print(f"  Low frequency: {len(low)}")
        print()
        for q in queries:
            marker = {"high": "▲", "medium": "●", "low": "▽"}[q["frequency"]]
            print(f"  {marker} \"{q['text']}\" → {q['expected']}")

    elif subcmd == "report":
        report = telemetry_report(STG_PATH)
        print(report)

    else:
        print("Usage: telemetry [status|frequency|salience|learning|calibrate|report]")
        print("  status                    Data volume and date range")
        print("  frequency [--top N]       Node activation frequency ranking")
        print("  salience                  Salience distribution trends")
        print("  learning                  Strengthen/weaken ratio analysis")
        print("  calibrate                 Generate calibrated query set")
        print("  report                    Comprehensive telemetry report")


def cmd_pruned(limit=50, item_type=None):
    """Show pruning audit log."""
    from stg_engine.persistence import read_pruned_log
    import datetime

    records = read_pruned_log(STG_PATH, limit=limit, item_type=item_type)
    if not records:
        print("No pruning records found.")
        return

    print(f"Pruning audit log ({len(records)} records, most recent first):")
    print()
    for r in records:
        ts = datetime.datetime.fromtimestamp(r["pruned_at"]).strftime("%Y-%m-%d %H:%M")
        itype = r["item_type"]
        if itype == "orphan_node":
            print(f"  [{ts}] NODE  [{r['source']}]")
        else:
            print(f"  [{ts}] EDGE  [{r['source']}] -> [{r['target']}]")
            print(f"         conf={r['confidence']:.3f} sal={r['salience']:.3f} reason={r['reason']}")
        mods = r.get("modifiers_json", "{}")
        if mods and mods != "{}":
            print(f"         mods={mods}")
    print()
    total_edges = sum(1 for r in records if r["item_type"] in ("edge", "virtual_edge"))
    total_nodes = sum(1 for r in records if r["item_type"] == "orphan_node")
    print(f"Summary: {total_edges} edge(s), {total_nodes} node(s) pruned")


def cmd_simulate(engine, subcmd, args):
    """Session simulator for parameter tuning."""
    from stg_engine.simulator import (
        SessionSimulator, print_report, print_comparison,
    )

    # Parse common flags
    num_sessions = 50
    if "--sessions" in args:
        idx = args.index("--sessions")
        if idx + 1 < len(args):
            try:
                num_sessions = int(args[idx + 1])
            except ValueError:
                pass

    simulator = SessionSimulator(engine)

    calibrated = "--calibrated" in args

    if subcmd == "run":
        if calibrated:
            print(f"Simulating {num_sessions} sessions with calibrated queries...")
            report = simulator.run_calibrated(STG_PATH, num_sessions=num_sessions)
        else:
            print(f"Simulating {num_sessions} sessions with default parameters...")
            report = simulator.run(num_sessions=num_sessions)
        print_report(report)

    elif subcmd == "sweep":
        # Sweep a single parameter
        # Usage: simulate sweep strengthen_rate 0.02 0.05 0.1 0.2
        if not args:
            print("Usage: simulate sweep <param_name> <val1> <val2> ... [--sessions N]")
            print("Parameters: strengthen_rate, weaken_rate, activation_threshold,")
            print("            prune_salience_threshold, prune_unused_days, eid_safety_threshold")
            return

        # Filter out --sessions flag
        clean_args = []
        skip_next = False
        for a in args:
            if skip_next:
                skip_next = False
                continue
            if a == "--sessions":
                skip_next = True
                continue
            clean_args.append(a)

        if len(clean_args) < 2:
            print("Need at least: <param_name> <val1> <val2>")
            return

        param_name = clean_args[0]
        try:
            values = [float(v) for v in clean_args[1:]]
        except ValueError:
            print(f"Cannot parse values as numbers: {clean_args[1:]}")
            return

        print(f"Sweeping {param_name} = {values} over {num_sessions} sessions each...")
        reports = simulator.compare(
            param_name=param_name,
            values=values,
            num_sessions=num_sessions,
        )
        print_comparison(param_name, values, reports)

    elif subcmd == "compare":
        # Compare current defaults vs aggressive vs conservative
        print(f"Comparing 3 presets over {num_sessions} sessions...")
        presets = {
            "conservative": {
                "strengthen_rate": 0.02,
                "weaken_rate": 0.01,
                "prune_salience_threshold": 0.05,
                "prune_unused_days": 60.0,
            },
            "default": {},
            "aggressive": {
                "strengthen_rate": 0.10,
                "weaken_rate": 0.05,
                "prune_salience_threshold": 0.15,
                "prune_unused_days": 14.0,
            },
        }
        for name, params in presets.items():
            print(f"\n--- {name.upper()} ---")
            report = simulator.run(num_sessions=num_sessions, params=params)
            print_report(report)

    else:
        print("Usage: simulate <run|sweep|compare> [args...]")
        print()
        print("  run                          Run with default params")
        print("  sweep <param> <v1> <v2> ...  Sweep one parameter")
        print("  compare                      Compare conservative/default/aggressive presets")
        print()
        print("Flags:")
        print("  --sessions N                 Number of sessions to simulate (default: 50)")


def cmd_topology(engine, subcmd, args):
    from stg_engine.topology import (
        CommunityDetector, BridgeDiscoverer,
        RedundancyEliminator, TopologyOptimizer,
    )

    # Parse optional flags
    resolution = 1.0
    eid_threshold = 0.001
    if "--resolution" in args:
        idx = args.index("--resolution")
        if idx + 1 < len(args):
            try:
                resolution = float(args[idx + 1])
            except ValueError:
                pass
    if "--eid" in args:
        idx = args.index("--eid")
        if idx + 1 < len(args):
            try:
                eid_threshold = float(args[idx + 1])
            except ValueError:
                pass

    if subcmd in ("communities", "comm"):
        detector = CommunityDetector(resolution=resolution)
        t0 = time.perf_counter()
        communities = detector.detect(engine)
        mod = detector.compute_modularity(engine, communities)
        elapsed = time.perf_counter() - t0

        ns_align = (
            sum(c.namespace_purity for c in communities) / len(communities)
            if communities else 0.0
        )

        print(f"Communities: {len(communities)} (modularity={mod:.3f}, "
              f"ns_alignment={ns_align:.3f}) ({elapsed*1000:.1f}ms)")
        for c in communities:
            ns_label = c.dominant_namespace or "?"
            print(f"  #{c.community_id}: {c.size} nodes "
                  f"({ns_label}, purity={c.namespace_purity:.2f}, "
                  f"density={c.internal_density:.4f})")

    elif subcmd == "bridges":
        detector = CommunityDetector(resolution=resolution)
        discoverer = BridgeDiscoverer()
        t0 = time.perf_counter()
        communities = detector.detect(engine)
        suggestions = discoverer.discover(engine, communities)
        elapsed = time.perf_counter() - t0

        if suggestions:
            print(f"Bridge suggestions: {len(suggestions)} ({elapsed*1000:.1f}ms)")
            for s in suggestions:
                print(f"  [{s.source}] -> [{s.target}] "
                      f"(conf={s.confidence}, {s.rationale})")
        else:
            print(f"No bridge suggestions needed ({elapsed*1000:.1f}ms)")

    elif subcmd == "redundancy":
        eliminator = RedundancyEliminator(eid_threshold=eid_threshold)
        t0 = time.perf_counter()
        redundant = eliminator.find_redundant(engine)
        elapsed = time.perf_counter() - t0

        if redundant:
            print(f"Redundant edges: {len(redundant)} (eid < {eid_threshold}) ({elapsed*1000:.1f}ms)")
            for src, tgt, eid in redundant[:20]:
                print(f"  [{src}] -> [{tgt}] (eid={eid:.6f})")
            if len(redundant) > 20:
                print(f"  ... and {len(redundant) - 20} more")
        else:
            print(f"No redundant edges (eid < {eid_threshold}) ({elapsed*1000:.1f}ms)")

    elif subcmd == "optimize":
        no_bridges = "--no-bridges" in args
        no_redundancy = "--no-redundancy" in args

        optimizer = TopologyOptimizer(
            resolution=resolution,
            eid_threshold=eid_threshold,
        )
        t0 = time.perf_counter()
        report = optimizer.optimize(
            engine,
            apply_bridges=not no_bridges,
            apply_redundancy=not no_redundancy,
        )
        elapsed = time.perf_counter() - t0

        print(f"=== Topology Optimize ({elapsed*1000:.1f}ms) ===")
        print(f"Communities: {report.community_count} "
              f"(modularity={report.modularity:.3f}, "
              f"ns_alignment={report.namespace_alignment:.3f})")
        if report.bridge_suggestions:
            print(f"Bridges applied: {len(report.bridge_suggestions)}")
        if report.redundant_edges:
            print(f"Redundant removed: {report.redundant_count}")
        print(f"Before: {report.node_count} nodes, {report.edge_count} edges")

        engine.save(STG_PATH)
        s = engine.get_stats()
        print(f"After:  {s['node_count']} nodes, {s['edge_count']} edges. Saved.")

    else:  # "analyze" or default
        optimizer = TopologyOptimizer(
            resolution=resolution,
            eid_threshold=eid_threshold,
        )
        t0 = time.perf_counter()
        report = optimizer.analyze(engine)
        elapsed = time.perf_counter() - t0

        print(f"=== Topology Analysis ({elapsed*1000:.1f}ms) ===")
        print(f"Communities: {report.community_count} "
              f"(modularity={report.modularity:.3f}, "
              f"ns_alignment={report.namespace_alignment:.3f})")
        for c in report.communities[:8]:
            ns_label = c.dominant_namespace or "?"
            print(f"  #{c.community_id}: {c.size} nodes "
                  f"({ns_label}, purity={c.namespace_purity:.2f})")
        if len(report.communities) > 8:
            print(f"  ... and {len(report.communities) - 8} more")

        print(f"\nBridge suggestions: {len(report.bridge_suggestions)}")
        for s in report.bridge_suggestions[:5]:
            print(f"  [{s.source}] -> [{s.target}] "
                  f"(conf={s.confidence}, {s.rationale})")
        if len(report.bridge_suggestions) > 5:
            print(f"  ... and {len(report.bridge_suggestions) - 5} more")

        print(f"\nRedundant edges: {report.redundant_count} "
              f"(eid < {eid_threshold})")

        print(f"\nGraph: {report.node_count} nodes, {report.edge_count} edges")


def cmd_cognitive(engine, subcmd, args):
    """Cognitive architecture commands (Phase 7D)."""
    from stg_engine.cognitive import (
        GoalRegister, PredictiveWarmer, HypothesisGenerator,
        SelfModel, MultiStrategyRouter, TemporalDynamics,
        CognitiveArchitecture,
    )

    t0 = time.time()

    if subcmd == "goals":
        if not engine._cognitive:
            engine.enable_cognitive()

        if args and args[0] == "add" and len(args) >= 3:
            name = args[1]
            keywords = args[2].split(",")
            priority = float(args[3]) if len(args) >= 4 else 1.0
            entry = engine.add_goal(name, keywords, priority)
            print(f"Goal added: {entry.name} (keywords={entry.keywords}, priority={entry.priority:.1f})")
        elif args and args[0] == "remove" and len(args) >= 2:
            removed = engine._cognitive.goals.remove_goal(args[1])
            print(f"Goal {'removed' if removed else 'not found'}: {args[1]}")
        else:
            goals = engine._cognitive.goals.current_goals
            if not goals:
                print("No active goals.")
            else:
                print(f"Active goals ({len(goals)}):")
                for g in goals:
                    print(f"  {g.name}: keywords={g.keywords}, priority={g.priority:.1f}")

    elif subcmd == "hypotheses":
        apply_flag = "--apply" in args
        hypotheses = engine.generate_hypotheses()
        elapsed = (time.time() - t0) * 1000
        print(f"=== Hypotheses ({len(hypotheses)} generated) ({elapsed:.1f}ms) ===")
        for h in hypotheses:
            print(f"  [{h.source}] -> [{h.target}] "
                  f"(conf={h.confidence:.2f}, {h.evidence_count} common neighbors)")
        if apply_flag and hypotheses:
            from stg_engine.cognitive import HypothesisGenerator
            gen = HypothesisGenerator()
            created = gen.apply_hypotheses(engine, hypotheses)
            print(f"\nApplied: {created} edges created")
            engine.save(STG_PATH)
            print("Saved to .stg")
        elif hypotheses and not apply_flag:
            print("\nUse --apply to commit as edges.")

    elif subcmd == "self-model":
        report = engine.get_self_model()
        elapsed = (time.time() - t0) * 1000
        print(f"=== Self-Model Report ({elapsed:.1f}ms) ===")
        print(f"Connectivity: {report.connectivity_health:.1%}")
        print(f"Cross-namespace integration: {report.cross_namespace_score:.1%}")
        print(f"Fragile nodes: {report.isolation_count} "
              f"({report.isolation_count / max(1, len(engine._nodes)):.1%})")
        print(f"\nTop hubs:")
        for name, imp in report.top_hubs[:10]:
            print(f"  {name} ({imp:.4f})")
        if report.gap_namespaces:
            print(f"\nKnowledge gaps: {', '.join(report.gap_namespaces)}")
        print(f"\nAssessment: {report.assessment}")

    elif subcmd == "route":
        query_text = " ".join(args) if args else ""
        if not query_text:
            print("Usage: cognitive route <query>")
            return
        if not engine._cognitive:
            engine.enable_cognitive()
        result = engine._cognitive.route_query(engine, query_text)
        elapsed = (time.time() - t0) * 1000
        print(f"=== Route: {result.strategy} ({elapsed:.1f}ms) ===")
        print(f"Query: {result.query}")
        print(f"Results ({len(result.results)}):")
        for r in result.results:
            print(f"  {r}")

    elif subcmd == "temporal":
        updated = engine.apply_temporal()
        elapsed = (time.time() - t0) * 1000
        # Compute average activation
        activations = [n.activation for n in engine._nodes.values()]
        avg_act = sum(activations) / len(activations) if activations else 0.0
        print(f"=== Temporal Dynamics ({elapsed:.1f}ms) ===")
        print(f"Nodes updated: {updated}")
        print(f"Average activation: {avg_act:.4f}")

    else:
        print("Usage: cognitive [goals|hypotheses|self-model|route|temporal]")
        print("  goals [list|add <name> <kw1,kw2,...> [priority]|remove <name>]")
        print("  hypotheses [--apply]")
        print("  self-model")
        print("  route <query>")
        print("  temporal")


def cmd_feedback(engine, subcmd, args):
    """Feedback loop commands (Phase 7E)."""
    from stg_engine.feedback import FeedbackLoopManager
    from stg_engine.types import FeedbackLoopConfig

    t0 = time.time()

    if subcmd == "status":
        if not engine._feedback:
            print("Feedback not active (no turns processed).")
            print("Use 'feedback simulate <N>' to run simulated turns.")
            return
        stats = engine._feedback.get_stats()
        print("=== Feedback Loop Status ===")
        print(f"  Total turns:       {stats.total_turns:>6}")
        print(f"  Periodic runs:     {stats.total_periodic_runs:>6}")
        print()
        print("Self-Improvement Loop:")
        print(f"  Self-models built:    {stats.self_models_built:>4}")
        print(f"  Goals auto-generated: {stats.goals_auto_generated:>4}")
        print()
        print("Predictive Loop:")
        print(f"  Total warmups:       {stats.total_warmups:>5}")
        print(f"  Hebbian events:      {stats.total_hebbian_events:>5}")
        print()
        print("Creative Loop:")
        print(f"  Hypotheses found:    {stats.hypotheses_generated:>5}")
        print(f"  Hypotheses applied:  {stats.hypotheses_applied:>5}")
        print(f"  Hypotheses rejected: {stats.hypotheses_rejected:>5}")
        print(f"  Edges pruned:        {stats.edges_pruned:>5}")

    elif subcmd == "simulate":
        n_turns = int(args[0]) if args else 10
        context = " ".join(args[1:]) if len(args) > 1 else "knowledge consciousness memory"

        if not engine._feedback:
            engine.enable_feedback()

        psi_before = engine.compute_psi()
        print(f"Simulating {n_turns} turns with context: \"{context}\"")
        print(f"Psi before: {psi_before:.4f}")
        print()

        for i in range(1, n_turns + 1):
            pre = engine._feedback.pre_turn(engine, context)
            # Simulate a query via propagate
            results = engine.propagate(context)
            post = engine._feedback.post_turn(
                engine, f"tell me about {context}", results, bool(results),
            )

            line = f"Turn {i:>3}: pre(warmup={pre['warmup_count']}) | post(hebbian={post['hebbian_events']})"
            if post.get("periodic"):
                p = post["periodic"]
                si = p.get("self_improvement", {})
                cr = p.get("creative", {})
                line += f" | PERIODIC(hyp={cr.get('hypotheses_applied', 0)}, goals={si.get('goals_added', 0)})"
            print(line)

        psi_after = engine.compute_psi()
        elapsed = (time.time() - t0) * 1000
        stats = engine._feedback.get_stats()
        print(f"\nSummary after {n_turns} turns ({elapsed:.0f}ms):")
        print(f"  Hebbian events: {stats.total_hebbian_events}")
        print(f"  Hypotheses applied: {stats.hypotheses_applied}")
        print(f"  Goals auto-generated: {stats.goals_auto_generated}")
        print(f"  Psi: {psi_before:.4f} -> {psi_after:.4f}")

    elif subcmd == "periodic":
        if not engine._feedback:
            engine.enable_feedback()
        result = engine._feedback.run_periodic(engine)
        elapsed = (time.time() - t0) * 1000

        si = result.get("self_improvement", {})
        cr = result.get("creative", {})
        print(f"=== Periodic Tasks ({elapsed:.1f}ms) ===")
        print()
        print("Self-Improvement:")
        print(f"  Connectivity: {si.get('self_model_connectivity', 0):.1%}")
        gaps = si.get("gap_namespaces", [])
        if gaps:
            print(f"  Gaps: {', '.join(gaps)}")
        print(f"  Goals added: {si.get('goals_added', 0)}")
        print()
        print("Creative:")
        print(f"  Hypotheses found: {cr.get('hypotheses_found', 0)}")
        print(f"  Applied: {cr.get('hypotheses_applied', 0)}")
        print(f"  Rejected: {cr.get('hypotheses_rejected', 0)}")

    elif subcmd == "session-end":
        if not engine._feedback:
            engine.enable_feedback(stg_path=STG_PATH)
        elif not engine._feedback._stg_path:
            engine._feedback._stg_path = STG_PATH
        # Auto-enable telemetry for flush
        if not engine.telemetry_enabled:
            engine.enable_telemetry()
        result = engine._feedback.session_end(engine)
        # Clear active_context (GP Phase 4 — §8.6)
        from stg_engine.feedback_select import clear_active_context
        clear_active_context(STG_PATH)
        elapsed = (time.time() - t0) * 1000
        print(f"=== Session End ({elapsed:.1f}ms) ===")
        print(f"  Edges pruned: {result.get('edges_pruned', 0)}")
        print(f"  Orphans removed: {result.get('orphans_removed', 0)}")
        print(f"  Active context cleared")
        telemetry_written = result.get("telemetry_written", 0)
        if telemetry_written:
            print(f"  Telemetry records written: {telemetry_written}")
        fs = result.get("final_stats", {})
        print(f"  Total turns: {fs.get('total_turns', 0)}")
        print(f"  Hebbian events: {fs.get('total_hebbian_events', 0)}")

        # Post-hook (opt-in, multi-user safe — defaults empty, configured per-box).
        # Set via: stg config set feedback.session_end_hook '<command line>'
        # Shell-style quoting honored via shlex; no shell is invoked. Hook failure
        # is logged but does NOT fail session-end (cleanup already succeeded).
        user_cfg = _read_user_config()
        feedback_cfg = user_cfg.get("feedback") or {}
        hook_cmd = feedback_cfg.get("session_end_hook")
        if hook_cmd:
            import shlex
            import subprocess
            try:
                timeout_s = int(feedback_cfg.get("session_end_hook_timeout_s") or 300)
            except (TypeError, ValueError):
                timeout_s = 300
            argv = shlex.split(hook_cmd) if isinstance(hook_cmd, str) else list(hook_cmd)
            try:
                r = subprocess.run(
                    argv, timeout=timeout_s, capture_output=True, text=True
                )
                if r.returncode == 0:
                    last_line = (r.stdout.strip().splitlines() or [""])[-1]
                    print(f"  Post-hook ok: {argv[0]}{(' — ' + last_line) if last_line else ''}")
                else:
                    err_snip = (r.stderr or r.stdout or "").strip()[:200]
                    print(
                        f"  Post-hook FAILED (exit {r.returncode}): {err_snip}",
                        file=sys.stderr,
                    )
            except subprocess.TimeoutExpired:
                print(f"  Post-hook TIMEOUT after {timeout_s}s", file=sys.stderr)
            except FileNotFoundError as e:
                print(f"  Post-hook ERROR: command not found ({e})", file=sys.stderr)
            except Exception as e:
                print(f"  Post-hook ERROR: {type(e).__name__}: {e}", file=sys.stderr)

    else:
        print("Usage: feedback [status|simulate|periodic|session-end]")
        print("  status                     Show feedback loop statistics")
        print("  simulate <N> [context]     Run N simulated turns")
        print("  periodic                   Manually run periodic tasks")
        print("  session-end                Run session-end cleanup")


def cmd_benchmark(engine, subcmd, args):
    from stg_engine.benchmark import STGBenchmark

    bench = STGBenchmark(engine)

    if subcmd == "full" or subcmd == "":
        print("Running full benchmark suite...")
        t0 = time.perf_counter()
        report = bench.run_all()
        print(bench.format_report(report))

    elif subcmd == "propagation":
        accuracy, results = bench.run_propagation_accuracy()
        print(f"Propagation Accuracy: {sum(1 for r in results if r.success)}/{len(results)} ({accuracy:.0%})")
        for r in results:
            mark = "+" if r.success else "-"
            nodes_str = ", ".join(r.nodes_found[:3]) if r.nodes_found else "(none)"
            print(f"  {mark} {r.question}")
            print(f"      hit={r.hit_count}, QE={r.qe:.3f}, RS={r.rs:.3f} -> {nodes_str}")
        avg_qe = sum(r.qe for r in results) / len(results) if results else 0
        avg_rs = sum(r.rs for r in results) / len(results) if results else 0
        print(f"\nAvg QE: {avg_qe:.3f}  |  Avg RS: {avg_rs:.3f}")

    elif subcmd == "routing":
        rates = bench.run_strategy_routing()
        print("Strategy Routing Success Rates:")
        for strat, rate in sorted(rates.items()):
            mark = "+" if rate > 0.7 else "-"
            print(f"  {mark} {strat}: {rate:.0%}")

    elif subcmd == "temporal":
        ratio = bench.run_temporal_dynamics()
        mark = "+" if ratio >= 2.0 else "-"
        print(f"Temporal Dynamics Ratio: {ratio:.2f}x (target: >= 2.0x) [{mark}]")

    elif subcmd == "hypotheses":
        gen, val, qual = bench.run_hypothesis_quality()
        mark = "+" if qual > 0.30 else "-"
        print(f"Hypothesis Quality: {val}/{gen} validated ({qual:.0%}) [{mark}]")

    elif subcmd == "emergence":
        n = int(args[0]) if args else 50
        print(f"Running {n}-turn emergence test...")
        result = bench.run_emergence(n_turns=n)
        print(f"  Psi: {result['psi_before']:.4f} -> {result['psi_after']:.4f} (delta={result['psi_delta']:+.4f})")
        print(f"  Gaps: {result['gaps_before']} -> {result['gaps_after']}")
        print(f"  Hebbian events: {result['edges_learned']}")
        print(f"  Hypotheses applied: {result['hypotheses_applied']}")
        print(f"  Goals auto-generated: {result['goals_auto_generated']}")
        mark = "+" if result['auto_improvement_detected'] else "-"
        print(f"  Auto-improvement: {'YES' if result['auto_improvement_detected'] else 'NO'} [{mark}]")

    elif subcmd == "perf":
        n = int(args[0]) if args else 20
        print(f"Performance benchmark ({n} turns)...")
        perf = bench.run_performance(n_turns=n)
        for k, v in sorted(perf.items()):
            print(f"  {k}: {v:.2f}ms")

    else:
        print("Usage: stg_cli.py benchmark [subcommand]")
        print("  full                       Run all benchmarks")
        print("  propagation                Propagation accuracy test")
        print("  routing                    Strategy routing test")
        print("  temporal                   Temporal dynamics test")
        print("  hypotheses                 Hypothesis quality test")
        print("  emergence [N]              Emergence test (default 50 turns)")
        print("  perf [N]                   Performance benchmark (default 20 turns)")


def cmd_search(engine, query, top_k=10, propagate=True, min_similarity=0.3):
    """Semantic search: Flash (vector similarity) + Unfold (graph propagation)."""
    t0 = time.perf_counter()
    result = engine.search(
        query,
        top_k=top_k,
        propagate=propagate,
        min_similarity=min_similarity,
    )
    elapsed = time.perf_counter() - t0

    print(f"search('{query}') → {len(result.combined)} results ({elapsed*1000:.1f}ms)")

    if result.seeds:
        print(f"\nSeeds ({len(result.seeds)}):")
        for name, sim in result.seeds:
            print(f"  {name:40s}  sim={sim:.4f}")

    if result.propagated:
        print(f"\nPropagated ({len(result.propagated)}):")
        for name, act in result.propagated[:10]:
            print(f"  {name:40s}  act={act:.4f}")
        if len(result.propagated) > 10:
            print(f"  ... and {len(result.propagated) - 10} more")

    if result.combined:
        print(f"\nCombined ranking ({len(result.combined)}):")
        for i, (name, score) in enumerate(result.combined):
            comm = _community_label(engine, name)
            comm_suffix = f"  [{comm}]" if comm else ""
            print(f"  {i+1:3d}. {name:40s}  score={score:.4f}{comm_suffix}")


def cmd_embed(engine, model_name=None):
    """Build or rebuild the embedding index for all nodes."""
    t0 = time.perf_counter()
    count = engine.build_search_index(model_name=model_name)
    elapsed = time.perf_counter() - t0
    print(f"Embedded {count} nodes ({elapsed:.1f}s)")
    print(f"Model: {engine._model_name}")

    # Save embeddings to .stg file
    if engine._vector_index and engine._vector_index.size > 0:
        from stg_engine.persistence import save_embeddings
        save_embeddings(
            STG_PATH,
            engine._vector_index.names,
            engine._vector_index.matrix,
            engine._embed_texts or {},
            engine._model_name,
        )
        print(f"Embeddings saved to memory.stg")


def cmd_reload():
    from stg_engine import import_memory_matrix
    engine = import_memory_matrix(MATRIX_PATH)
    engine.compute_all_tensions()
    engine.compute_activations()
    engine.save(STG_PATH)
    s = engine.get_stats()
    print(f"Reloaded from memoryMatrix.md → memory.stg")
    print(f"Graph: {s['node_count']} nodes, {s['edge_count']} edges")


def cmd_import(manifest_path=None):
    from stg_engine import import_knowledge_base
    project_root = os.path.dirname(os.path.abspath(__file__))
    if manifest_path:
        manifest = manifest_path
    else:
        stl_path = os.path.join(project_root, "stg_manifest.stl")
        json_path = os.path.join(project_root, "stg_manifest.json")
        manifest = stl_path if os.path.isfile(stl_path) else json_path

    t0 = time.perf_counter()
    engine = import_knowledge_base(manifest, project_root=project_root)
    elapsed = time.perf_counter() - t0

    engine.save(STG_PATH)
    s = engine.get_stats()

    print(f"Knowledge base imported → memory.stg ({elapsed:.1f}s)")
    print(f"Graph: {s['node_count']} nodes, {s['edge_count']} edges")
    print(f"Sessions: {s['session_count']}, Events: {s['event_count']}")
    print(f"Tensions: {s['total_tensions']} ({s['active_tensions']} active)")
    print(f"Psi: {s['psi']:.4f}")

    if hasattr(engine, '_import_stats'):
        st = engine._import_stats
        print(f"\nImport breakdown:")
        print(f"  Skeleton: {st['skeleton']} elements")
        print(f"  STL native: {st['stl_native']} elements")
        print(f"  STLC specs: {st['stlc_spec']} elements")
        print(f"  Markdown docs: {st['markdown_doc']} elements")
        print(f"  Files processed: {st['files_processed']}")
        print(f"  Files skipped: {st['files_skipped']}")


def cmd_converge(engine, query_text, top_k=5, max_iter=5, threshold=0.8):
    """Iterative propagation until convergence (Kanerva F5)."""
    result = engine.convergent_propagate(
        query_text, top_k=top_k, max_iterations=max_iter,
        convergence_threshold=threshold,
    )
    status = "CONVERGED" if result.converged else "NOT CONVERGED"
    print(f"Status: {status} in {result.iterations_used} iteration(s)")
    print(f"Stability history: {[f'{s:.2f}' for s in result.stability_history]}")
    print(f"\nTop nodes:")
    for i, name in enumerate(result.top_nodes, 1):
        print(f"  {i}. {name}")


def cmd_preference(engine, subcmd, args):
    """Preference function commands (Kanerva F7)."""
    from stg_engine.kanerva import PreferenceFunction
    pf = PreferenceFunction()

    if subcmd == "top":
        top_n = 20
        if args:
            try:
                top_n = int(args[0])
            except ValueError:
                pass
        top = pf.get_top_preferred(engine, top_n=top_n)
        if not top:
            print("No edges with non-zero preference.")
            return
        print(f"Top {len(top)} preferred edges:")
        for src, tgt, pref in top:
            sign = "+" if pref > 0 else ""
            print(f"  [{src}] -> [{tgt}]  preference={sign}{pref:.4f}")

    elif subcmd == "reward" and len(args) >= 2:
        reward = 1.0
        path = []
        i = 0
        while i < len(args):
            if args[i] == "--reward" and i + 1 < len(args):
                try:
                    reward = float(args[i + 1])
                except ValueError:
                    pass
                i += 2
            else:
                path.append(args[i])
                i += 1
        updated = pf.reward_path(engine, path, reward=reward)
        engine.save(STG_PATH)
        print(f"Rewarded {updated} edge(s) along path: {' -> '.join(path)}")

    elif subcmd == "decay":
        affected = pf.decay_preferences(engine)
        engine.save(STG_PATH)
        print(f"Decayed {affected} edge(s) with non-zero preference.")

    else:
        print("Usage: preference top [N] | preference reward <n1> <n2> ... [--reward R] | preference decay")


def cmd_coactivation(engine, subcmd, args):
    """Co-activation edge management."""
    from stg_engine.coactivation import (
        find_coactivation_candidates,
        create_coactivation_edges,
        coactivation_report,
        record_coactivation_event,
    )

    if subcmd == "report":
        print(coactivation_report(STG_PATH))

    elif subcmd == "candidates":
        min_count = 5
        for i, a in enumerate(args):
            if a == "--min" and i + 1 < len(args):
                min_count = int(args[i + 1])
        candidates = find_coactivation_candidates(
            engine, STG_PATH, min_cooccurrence=min_count,
        )
        if not candidates:
            print("No candidates found.")
            return
        print(f"Found {len(candidates)} candidate(s):")
        for a, b, count in candidates:
            conf = min(0.5, count * 0.05)
            print(f"  {a} ↔ {b}  (count={count}, would_conf={conf:.2f})")

    elif subcmd == "apply":
        candidates = find_coactivation_candidates(engine, STG_PATH)
        if not candidates:
            print("No candidates to apply.")
            return
        events = create_coactivation_edges(engine, candidates)
        # Record telemetry for parameter tuning
        record_coactivation_event(
            STG_PATH,
            candidates_found=len(candidates),
            edges_created=len(events),
            candidates_detail=candidates,
        )
        engine.save(STG_PATH)
        print(f"Created {len(events)} co-activation edge(s). Saved.")
        for ev in events:
            print(f"  {ev.source} → {ev.target} (conf={ev.new_confidence:.2f})")

    else:
        print("Usage: coactivation report | candidates [--min N] | apply")


def cmd_visualize(engine, args):
    """Generate and open a 3D star map visualization."""
    from stg_engine.visualize import generate_starmap

    mode = "full"
    output = None
    auto_open = True
    use_embeddings = True
    kwargs = {}

    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("full", "ego", "community", "propagate"):
            mode = arg
            if mode == "ego" and i + 1 < len(args) and not args[i + 1].startswith("--"):
                i += 1
                kwargs["ego_node"] = args[i]
            elif mode == "community" and i + 1 < len(args) and not args[i + 1].startswith("--"):
                i += 1
                try:
                    kwargs["community_id"] = int(args[i])
                except ValueError:
                    pass
            elif mode == "propagate" and i + 1 < len(args):
                i += 1
                kwargs["query"] = args[i]
        elif arg == "--output" and i + 1 < len(args):
            i += 1
            output = args[i]
        elif arg == "--no-open":
            auto_open = False
        elif arg == "--no-embeddings":
            use_embeddings = False
        elif arg == "--filter-dead-ends":
            kwargs["filter_dead_ends"] = True
        elif arg == "--depth" and i + 1 < len(args):
            i += 1
            try:
                kwargs["ego_depth"] = int(args[i])
            except ValueError:
                pass
        i += 1

    print(f"Generating 3D star map (mode={mode})...")
    path = generate_starmap(
        engine, mode=mode, output=output,
        auto_open=auto_open, use_embeddings=use_embeddings,
        **kwargs,
    )
    print(f"Star map saved to: {path}")


def cmd_temporal(engine, subcmd, args):
    """Temporal structure commands: range, around, replay, build."""
    from stg_engine.temporal import (
        query_time_range, query_temporal_neighborhood,
        build_episode_sequence, replay_episode,
        epoch_to_str, parse_date_str,
    )

    if subcmd == "range":
        if len(args) < 2:
            print("Usage: temporal range <start_date> <end_date>")
            print("  Dates: YYYY-MM-DD or 'YYYY-MM-DD HH:MM:SS'")
            return
        try:
            start = parse_date_str(args[0])
            end = parse_date_str(args[1])
            # If end is date-only, extend to end of day
            if " " not in args[1]:
                end += 86400 - 1  # 23:59:59
        except ValueError as e:
            print(f"Error: {e}")
            return

        edge_class = None
        if "--class" in args:
            idx = args.index("--class")
            if idx + 1 < len(args):
                edge_class = args[idx + 1]

        edges = query_time_range(engine, start, end, edge_class=edge_class)
        print(f"Edges created between {args[0]} and {args[1]}: {len(edges)}")
        for e in edges[:50]:
            cls_tag = f" [{e.edge_class}]" if e.edge_class != "knowledge" else ""
            print(f"  {epoch_to_str(e.created_at)}  [{e.source}] → [{e.target}]{cls_tag}")

    elif subcmd == "around":
        if not args:
            print("Usage: temporal around <node_name> [window_hours]")
            return
        node = args[0]
        window_hours = 1.0
        if len(args) >= 2:
            try:
                window_hours = float(args[1])
            except ValueError:
                pass
        edges = query_temporal_neighborhood(engine, node, window_seconds=window_hours * 3600)
        print(f"Edges within ±{window_hours}h of '{node}': {len(edges)}")
        for e in edges[:50]:
            marker = " ◄" if e.source == node or e.target == node else ""
            print(f"  {epoch_to_str(e.created_at)}  [{e.source}] → [{e.target}]{marker}")

    elif subcmd == "replay":
        if not args:
            print("Usage: temporal replay <entry_node> [session_id]")
            return
        entry_node = args[0]
        session_id = args[1] if len(args) >= 2 else None
        sequence = replay_episode(engine, entry_node, session_id=session_id)
        print(f"Episode replay from '{entry_node}': {len(sequence)} nodes")
        for i, node in enumerate(sequence):
            print(f"  {i + 1}. {node}")

    elif subcmd == "build":
        if not args:
            print("Usage: temporal build <session_id|start_date> [end_date] [--k N]")
            print("  By session:    temporal build SESSION_042")
            print("  By date range: temporal build 2026-03-18 2026-03-18")
            return
        k_fold = 2
        if "--k" in args:
            idx = args.index("--k")
            if idx + 1 < len(args):
                try:
                    k_fold = int(args[idx + 1])
                except ValueError:
                    pass
            args = [a for i, a in enumerate(args) if i != idx and i != idx + 1]

        # Detect: date range (YYYY-MM-DD) or session_id
        try:
            start = parse_date_str(args[0])
            # It's a date — use time range mode
            if len(args) >= 2:
                end = parse_date_str(args[1])
                if " " not in args[1]:
                    end += 86400 - 1
            else:
                end = start + 86400 - 1  # same day
            edges = build_episode_sequence(engine, time_start=start, time_end=end, k_fold=k_fold)
            label = f"{args[0]} to {args[1] if len(args) >= 2 else args[0]}"
        except ValueError:
            # It's a session_id
            session_id = args[0]
            edges = build_episode_sequence(engine, session_id=session_id, k_fold=k_fold)
            label = f"session '{session_id}'"

        print(f"Built episode for {label}: {len(edges)} temporal edges (k={k_fold})")
        for e in edges:
            print(f"  [{e.source}] →(k={e.delay_k}) [{e.target}]")
        if edges:
            engine.save(STG_PATH)
            print("Saved.")

    elif subcmd == "stats":
        # Count edges by class and temporal coverage
        total = len(engine._edges)
        knowledge = sum(1 for e in engine._edges if e.edge_class == "knowledge")
        temporal = sum(1 for e in engine._edges if e.edge_class == "temporal")
        virtual = sum(1 for e in engine._edges if e.edge_class == "virtual" or e.modifiers.get("edge_class") == "virtual")
        timestamped = sum(1 for e in engine._edges if e.created_at > 0.0)
        legacy = total - timestamped

        print(f"Temporal Stats:")
        print(f"  Total edges:     {total}")
        print(f"  Knowledge:       {knowledge}")
        print(f"  Temporal:        {temporal}")
        print(f"  Virtual:         {virtual}")
        print(f"  Timestamped:     {timestamped} ({100*timestamped/total:.1f}%)" if total else "  Timestamped:     0")
        print(f"  Legacy (no ts):  {legacy}")

        if timestamped:
            times = [e.created_at for e in engine._edges if e.created_at > 0.0]
            print(f"  Earliest:        {epoch_to_str(min(times))}")
            print(f"  Latest:          {epoch_to_str(max(times))}")

    else:
        print("temporal subcommands:")
        print("  range <start> <end>     Query edges by creation time range")
        print("  around <node> [hours]   Temporal neighborhood of a node")
        print("  replay <node> [session] Replay episode sequence from entry node")
        print("  build <session> [k]     Build temporal episode chain from session")
        print("  stats                   Show temporal coverage statistics")


def cmd_heartbeat(args):
    """Heartbeat orchestrator — moved out of stg-engine.

    Heartbeat depends on the SKC heartbeat module which is not part
    of stg-engine. To use heartbeat, install the full SKC package.
    """
    print("ERROR: heartbeat subcommand is not available in stg-engine.")
    print("Heartbeat orchestrator lives in the SKC package, not stg-engine.")
    print("If you need heartbeat, use the SKC repository instead.")


# ═══════════════════════════════════════════════════════════
# Phase 12: Perception Commands
# ═══════════════════════════════════════════════════════════


def cmd_perceive(engine, args):
    """Perceive a grid frame and show results."""
    import json
    from pathlib import Path

    if not args:
        print("Usage: stg perceive <grid_json_or_file> [--game <game_id>] [--step <n>] [--level <n>]")
        return

    # Parse flags
    game_id = None
    step_number = 0
    level = 0
    grid_arg = args[0]

    i = 1
    while i < len(args):
        if args[i] == "--game" and i + 1 < len(args):
            game_id = args[i + 1]
            i += 2
        elif args[i] == "--step" and i + 1 < len(args):
            step_number = int(args[i + 1])
            i += 2
        elif args[i] == "--level" and i + 1 < len(args):
            level = int(args[i + 1])
            i += 2
        else:
            i += 1

    # Load grid from file or JSON string
    grid_path = Path(grid_arg)
    if grid_path.exists():
        grid = json.loads(grid_path.read_text())
    else:
        grid = json.loads(grid_arg)

    fhash, similar = engine.perceive(grid, game_id, step_number, level)
    print(f"Frame hash: {fhash}")
    print(f"Node: Visual:frame_{fhash}")
    h, w = len(grid), len(grid[0]) if grid else 0
    print(f"Grid: {w}x{h}")
    if similar:
        print(f"Similar past frames ({len(similar)}):")
        for name in similar:
            print(f"  {name}")
    else:
        print("No similar past frames found.")


def cmd_perception(engine, args):
    """Perception subsystem commands."""
    subcmd = args[0] if args else "stats"

    if subcmd == "stats":
        index = getattr(engine, "_perception_index", None)
        fixed = getattr(engine, "_fixed_filters", None)
        learnable = getattr(engine, "_perception_filters", None)

        print("=== Perception Stats ===")
        print(f"Index size: {index.size if index else 0} frames")
        print(f"Fixed filters: {'loaded' if fixed is not None else 'not loaded'}"
              f" ({fixed.shape[0] if fixed is not None else 0} filters)")
        print(f"Learnable filters: {'loaded' if learnable is not None else 'not initialized'}"
              f" ({learnable.shape[0] if learnable is not None else 0} filters)")

        # Count perception nodes in graph
        perc_nodes = [n for n in engine._nodes if n.startswith("Visual:")]
        perc_edges = [
            e for e in engine._edges
            if e.source.startswith("Visual:") or e.target.startswith("Visual:")
        ]
        print(f"Perception nodes: {len(perc_nodes)}")
        print(f"Perception edges: {len(perc_edges)}")

    elif subcmd == "similar" and len(args) >= 2:
        import json
        from pathlib import Path

        grid_arg = args[1]
        top_k = 5
        if "--top" in args:
            idx = args.index("--top")
            if idx + 1 < len(args):
                top_k = int(args[idx + 1])

        grid_path = Path(grid_arg)
        if grid_path.exists():
            grid = json.loads(grid_path.read_text())
        else:
            grid = json.loads(grid_arg)

        from stg_engine.perception import find_similar_states
        results = find_similar_states(engine, grid, top_k=top_k)
        if results:
            print(f"Top {len(results)} similar frames:")
            for name, sim in results:
                print(f"  {name}: similarity={sim:.4f}")
        else:
            print("No similar frames found.")

    elif subcmd == "filters":
        learnable = getattr(engine, "_perception_filters", None)
        if learnable is None:
            print("Learnable filters not initialized.")
            return
        import numpy as np
        print(f"Learnable filters: {learnable.shape}")
        for i in range(learnable.shape[0]):
            norm = float(np.linalg.norm(learnable[i]))
            mean = float(learnable[i].mean())
            print(f"  Filter {i}: norm={norm:.4f}, mean={mean:.6f}")

    elif subcmd == "reset-filters":
        from stg_engine.perception import init_learnable_filters
        engine._perception_filters = init_learnable_filters()
        print("Learnable filters reset to random initialization.")

    else:
        print("Usage: stg perception <subcommand>")
        print("  stats              Show perception system status")
        print("  similar <grid>     Find similar past frames")
        print("  filters            Show learnable filter stats")
        print("  reset-filters      Re-initialize learnable filters")


# ===========================================================================
# Skill executor commands (v0.3.1+)
# See development/design/STG_SKILL_EXECUTOR_DESIGN.md
# ===========================================================================

def cmd_use(engine, args):
    """stg use <skill_name> [script_args...]

    Execute a Skill node via its registered script. Requires:
      - skill.enabled=true in user config (`stg config set skill.enabled true`)
      - skill.roots whitelisting the script's directory
      - executable=true modifier on the Skill edge

    Flags (consumed by this command, not passed to the script):
      --timeout N        override timeout seconds
      --args-stl STL     pass an STL block via the script's stdin
      --stdin STL        alias for --args-stl
      --json             emit {stdout, stderr, exit_code, elapsed_s} as JSON
      --dry-run          resolve + validate but don't execute
    """
    from stg_engine import skill_runner

    if not args:
        print(cmd_use.__doc__.strip())
        sys.exit(skill_runner.EXIT_ENGINE_ERROR)

    skill_name = args[0]
    rest = list(args[1:])

    timeout_override: Optional[int] = None
    stdin_stl: Optional[str] = None
    emit_json = False
    dry_run = False

    filtered: list = []
    i = 0
    while i < len(rest):
        tok = rest[i]
        if tok == "--timeout" and i + 1 < len(rest):
            try:
                timeout_override = int(rest[i + 1])
            except ValueError:
                print(f"invalid --timeout value: {rest[i+1]!r}", file=sys.stderr)
                sys.exit(skill_runner.EXIT_ENGINE_ERROR)
            i += 2
            continue
        if tok in ("--args-stl", "--stdin") and i + 1 < len(rest):
            stdin_stl = rest[i + 1]
            i += 2
            continue
        if tok == "--args-stl-file" and i + 1 < len(rest):
            try:
                stdin_stl = Path(rest[i + 1]).read_text()
            except OSError as e:
                print(f"cannot read --args-stl-file: {e}", file=sys.stderr)
                sys.exit(skill_runner.EXIT_ENGINE_ERROR)
            i += 2
            continue
        if tok == "--json":
            emit_json = True
            i += 1
            continue
        if tok == "--dry-run":
            dry_run = True
            i += 1
            continue
        filtered.append(tok)
        i += 1

    user_cfg = _read_user_config()

    result = skill_runner.run_skill(
        engine=engine,
        skill_name=skill_name,
        args=filtered,
        user_config=user_cfg,
        stdin_stl=stdin_stl,
        timeout_override=timeout_override,
        dry_run=dry_run,
    )

    # Write audit row (only for real attempts, not config-disabled)
    if result.exit_code != skill_runner.EXIT_NOT_EXECUTABLE or result.path:
        skill_runner.write_audit_row(STG_PATH, result)

    if emit_json:
        import json as _json
        payload = {
            "skill_name": result.skill_name,
            "target": result.target,
            "path": result.path,
            "interpreter": result.interpreter,
            "args": result.args,
            "exit_code": result.exit_code,
            "elapsed_s": result.elapsed_s,
            "bytes_out": result.bytes_out,
            "bytes_err": result.bytes_err,
            "truncated": result.truncated_stdout,
            "timed_out": result.timed_out,
            "invocation_id": result.invocation_id,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error": result.error,
        }
        print(_json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        if result.stdout:
            sys.stdout.write(result.stdout)
            if not result.stdout.endswith("\n"):
                sys.stdout.write("\n")
        if result.stderr:
            sys.stderr.write(result.stderr)
            if not result.stderr.endswith("\n"):
                sys.stderr.write("\n")
        if result.error:
            print(f"error: {result.error}", file=sys.stderr)
        if result.exit_code != 0:
            tag = (
                "[timeout]" if result.timed_out
                else "[child-exit]" if result.exit_code == skill_runner.EXIT_CHILD_NONZERO
                else "[error]"
            )
            print(f"\n{tag} skill={result.skill_name} "
                  f"exit={result.exit_code} elapsed={result.elapsed_s:.2f}s",
                  file=sys.stderr)

    sys.exit(result.exit_code)


def cmd_skill(engine, args):
    """stg skill <subcommand> [...]

    Subcommands:
      list [--filter KEYWORD] [--all]   Catalog of skills
      show <name>                       Full detail on one skill
      use <name> [args...]              Alias for `stg use <name> [args...]`
      configure <name> [--executable] [--interpreter NAME] [--args-template T] [--stl-io] [--timeout N]
                                         Backfill invocation fields on an existing Skill edge
      history [--skill N] [--limit N]   Recent invocations
    """
    from stg_engine import skill_runner

    if not args:
        print(cmd_skill.__doc__.strip())
        return

    sub = args[0]
    rest = list(args[1:])

    if sub == "list":
        filter_kw: Optional[str] = None
        show_all = False
        i = 0
        while i < len(rest):
            tok = rest[i]
            if tok == "--filter" and i + 1 < len(rest):
                filter_kw = rest[i + 1]
                i += 2
                continue
            if tok == "--all":
                show_all = True
                i += 1
                continue
            i += 1
        skills = skill_runner.list_skills(
            engine,
            filter_keyword=filter_kw,
            executable_only=not show_all,
        )
        print(skill_runner.render_catalog(skills))
        return

    if sub == "show" and rest:
        name = rest[0]
        edges = skill_runner.find_skill_edges_by_name(engine, name)
        if not edges:
            print(f"no Skill node named '{name}' found.")
            sys.exit(skill_runner.EXIT_NOT_FOUND)
        for u, v, data in edges:
            from stg_engine.engine import _get_skill_invocation as _inv
            inv = _inv(data)
            print(f"Skill: {u}  →  {v}")
            print(f"  confidence: {data.get('confidence', '?')}")
            if data.get("rule"):
                print(f"  rule:       {data.get('rule')}")
            if data.get("description"):
                print(f"  description: {data.get('description')}")
            if data.get("path"):
                print(f"  path:       {data.get('path')}")
            for k in ("executable", "interpreter", "args_template",
                      "stl_io", "timeout_s"):
                if k in inv:
                    print(f"  {k}: {inv[k]}")
            print()
        # Also show recent invocations
        history = skill_runner.read_audit_history(STG_PATH, skill_name=name, limit=5)
        if history:
            print(f"Recent invocations (last {len(history)}):")
            for row in history:
                status = "ok" if row["exit_code"] == 0 else f"exit {row['exit_code']}"
                import datetime as _dt
                ts = _dt.datetime.fromtimestamp(row["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                print(f"  {ts}  [{status}]  {row['elapsed_s']:.2f}s  "
                      f"args: {row['args_preview']}")
        return

    if sub == "use" and rest:
        cmd_use(engine, rest)
        return  # cmd_use calls sys.exit()

    if sub == "configure" and rest:
        _skill_configure(engine, rest)
        return

    if sub == "history":
        filter_name: Optional[str] = None
        limit = 20
        i = 0
        while i < len(rest):
            tok = rest[i]
            if tok == "--skill" and i + 1 < len(rest):
                filter_name = rest[i + 1]
                i += 2
                continue
            if tok == "--limit" and i + 1 < len(rest):
                try:
                    limit = int(rest[i + 1])
                except ValueError:
                    pass
                i += 2
                continue
            i += 1
        history = skill_runner.read_audit_history(
            STG_PATH, skill_name=filter_name, limit=limit
        )
        if not history:
            print("(no invocations recorded)")
            return
        import datetime as _dt
        for row in history:
            status = "ok" if row["exit_code"] == 0 else f"exit {row['exit_code']}"
            ts = _dt.datetime.fromtimestamp(row["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            flags = []
            if row["truncated"]: flags.append("truncated")
            if row["timed_out"]: flags.append("timeout")
            flag_s = f" [{','.join(flags)}]" if flags else ""
            print(f"{ts}  {row['skill_name']:<35}  [{status}]  "
                  f"{row['elapsed_s']:.2f}s{flag_s}  {row['args_preview']}")
        return

    print(cmd_skill.__doc__.strip())


def _skill_configure(engine, args):
    """Backfill invocation fields on an existing Skill edge via stg merge.

    Usage:
      stg skill configure <name> [--executable] [--no-executable]
                                 [--interpreter NAME]
                                 [--args-template STRING]
                                 [--stl-io] [--no-stl-io]
                                 [--timeout N]
    """
    from stg_engine import skill_runner

    if not args:
        print(_skill_configure.__doc__.strip())
        return
    name = args[0]
    rest = args[1:]

    fields: dict = {}
    i = 0
    while i < len(rest):
        tok = rest[i]
        if tok == "--executable":
            fields["executable"] = "true"; i += 1
        elif tok == "--no-executable":
            fields["executable"] = "false"; i += 1
        elif tok == "--interpreter" and i + 1 < len(rest):
            fields["interpreter"] = rest[i + 1]; i += 2
        elif tok == "--args-template" and i + 1 < len(rest):
            fields["args_template"] = rest[i + 1]; i += 2
        elif tok == "--stl-io":
            fields["stl_io"] = "true"; i += 1
        elif tok == "--no-stl-io":
            fields["stl_io"] = "false"; i += 1
        elif tok == "--timeout" and i + 1 < len(rest):
            fields["timeout_s"] = rest[i + 1]; i += 2
        else:
            print(f"unknown flag: {tok}", file=sys.stderr)
            sys.exit(skill_runner.EXIT_ENGINE_ERROR)

    if not fields:
        print("no fields given — see `stg skill configure --help`")
        sys.exit(skill_runner.EXIT_ENGINE_ERROR)

    edges = skill_runner.find_skill_edges_by_name(engine, name)
    if not edges:
        print(f"no Skill node named '{name}' found. Run `stg skill list` to see available skills.",
              file=sys.stderr)
        sys.exit(skill_runner.EXIT_NOT_FOUND)

    if len(edges) > 1:
        # Auto-pick the edge that has a `path` modifier — that's the primary
        # "what this skill does" edge. Other edges (e.g. to a Lesson) are
        # secondary and don't need invocation fields.
        with_path = [e for e in edges if e[2].get("path")]
        if len(with_path) == 1:
            edges = with_path
        else:
            targets = [v for _, v, _ in edges]
            with_paths = [v for _, v, d in edges if d.get("path")]
            print(
                f"'{name}' has {len(edges)} edges and no single one is the "
                f"obvious primary (by path= modifier).\n"
                f"  all targets:       {targets}\n"
                f"  targets with path: {with_paths}\n"
                f"Disambiguate with the full `stg merge` form specifying the target:\n"
                f"  stg merge '[{name}] -> [TARGET] ::mod(executable=\"true\", ...)'",
                file=sys.stderr,
            )
            sys.exit(skill_runner.EXIT_AMBIGUOUS)

    # Build STL for merge
    source, target, _ = edges[0]
    mod_kv = ", ".join(f'{k}="{v}"' for k, v in fields.items())
    stl_text = f'[{source}] -> [{target}] ::mod({mod_kv})'
    cmd_merge(engine, stl_text)


def cmd_skill_propagate_catalog(engine, query: str):
    """When propagate query matches /^skills?$/i, render a catalog instead of
    the usual community-grouped output."""
    from stg_engine import skill_runner
    skills = skill_runner.list_skills(engine, executable_only=False)
    print(skill_runner.render_catalog(skills))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "reload":
        _audit.info(f"CMD=reload | args={sys.argv[2:]} | FULL RELOAD (pre-engine)")
        cmd_reload()
        return

    # config subcommand runs before engine load (it only touches user config)
    if cmd == "config":
        cmd_config(sys.argv[2:])
        return

    if cmd == "import":
        manifest = sys.argv[2] if len(sys.argv) >= 3 else None
        _audit.info(f"CMD=import | args={sys.argv[2:]} | manifest={manifest}")
        cmd_import(manifest)
        return

    t0 = time.perf_counter()
    engine = load_engine()
    load_ms = (time.perf_counter() - t0) * 1000

    # --- Audit: snapshot before ---
    _before = _snap(engine)

    if cmd == "stats":
        cmd_stats(engine)
    elif cmd == "psi":
        cmd_psi(engine)
    elif cmd == "import-doc" and len(sys.argv) >= 3:
        args = sys.argv[2:]
        source_type = "doc"
        if "--source" in args:
            idx = args.index("--source")
            if idx + 1 < len(args):
                source_type = args[idx + 1]
                args = args[:idx] + args[idx + 2:]
        cmd_import_doc(engine, args[0], source_type=source_type)
    elif cmd == "grep" and len(sys.argv) >= 3:
        args = sys.argv[2:]
        limit = 20
        full = False
        if "--full" in args:
            full = True
            args = [a for a in args if a != "--full"]
        if "--limit" in args:
            idx = args.index("--limit")
            if idx + 1 < len(args):
                try:
                    limit = int(args[idx + 1])
                except ValueError:
                    pass
                args = args[:idx] + args[idx + 2:]
        cmd_grep(engine, " ".join(args), limit=limit, full=full)
    elif cmd == "dump":
        args = sys.argv[2:]
        page_size = 100
        start = 0
        namespace = None
        if "--page" in args:
            idx = args.index("--page")
            if idx + 1 < len(args):
                try:
                    page_size = int(args[idx + 1])
                except ValueError:
                    pass
        if "--start" in args:
            idx = args.index("--start")
            if idx + 1 < len(args):
                try:
                    start = int(args[idx + 1]) - 1
                except ValueError:
                    pass
        if "--namespace" in args:
            idx = args.index("--namespace")
            if idx + 1 < len(args):
                namespace = args[idx + 1]
        cmd_dump(engine, page_size=page_size, start=start, namespace=namespace)
    elif cmd == "query" and len(sys.argv) >= 3:
        # Parse --limit N flag
        args = sys.argv[2:]
        limit = 20
        if "--limit" in args:
            idx = args.index("--limit")
            if idx + 1 < len(args):
                try:
                    limit = int(args[idx + 1])
                except ValueError:
                    pass
                args = args[:idx] + args[idx + 2:]
        cmd_query(engine, " ".join(args), limit=limit)
    elif cmd == "tensions":
        status = sys.argv[2] if len(sys.argv) >= 3 else None
        cmd_tensions(engine, status)
    elif cmd == "propagate" and len(sys.argv) >= 3:
        args = sys.argv[2:]
        # Short-circuit: `propagate skill` (or skills / SKILL / --namespace=Skill)
        # renders the Skill catalog instead of the usual community-grouped view.
        _catalog_trigger = False
        if len(args) == 1 and re.match(r"^skills?$", args[0], re.IGNORECASE):
            _catalog_trigger = True
        if "--namespace=Skill" in args or "--namespace=skill" in args:
            _catalog_trigger = True
        if _catalog_trigger:
            cmd_skill_propagate_catalog(engine, args[0] if args else "skill")
            return
        # Gravity is ON by default (GP Phase 4). Use --no-gravity to disable.
        if "--no-gravity" in args:
            use_gravity = False
            args = [a for a in args if a != "--no-gravity"]
        elif "--gravity" in args:
            use_gravity = True
            args = [a for a in args if a != "--gravity"]
        else:
            use_gravity = True  # default ON
        # Resolution: --coarse / --fine (default medium)
        resolution = "medium"
        if "--coarse" in args:
            resolution = "coarse"
            args = [a for a in args if a != "--coarse"]
        elif "--fine" in args:
            resolution = "fine"
            args = [a for a in args if a != "--fine"]
        # --all-chains: include virtual edge chains
        all_chains = "--all-chains" in args
        if all_chains:
            args = [a for a in args if a != "--all-chains"]
        # --full: show all edge modifiers in chain display (not just description)
        all_modifiers = "--full" in args
        if all_modifiers:
            args = [a for a in args if a != "--full"]
        # --expand N: after propagate, dump full node detail for top-N activated
        # Default: expand top 3. Use --expand 0 to disable, or --expand N to override.
        expand_top = 3
        if "--expand" in args:
            i = args.index("--expand")
            if i + 1 < len(args) and args[i + 1].isdigit():
                expand_top = int(args[i + 1])
                args = args[:i] + args[i + 2:]
            else:
                args = args[:i] + args[i + 1:]
        # Phase 7I: --nodes preserves legacy node-level output
        community_mode = True
        if "--nodes" in args:
            community_mode = False
            args = [a for a in args if a != "--nodes"]
        # Phase 7I: --top N controls how many communities to show (default 5)
        top_m = 5
        if "--top" in args:
            i = args.index("--top")
            if i + 1 < len(args) and args[i + 1].isdigit():
                top_m = int(args[i + 1])
                args = args[:i] + args[i + 2:]
            else:
                args = args[:i] + args[i + 1:]
        # Phase 7I: --brief suppresses inlined node detail (terse community summary)
        brief = "--brief" in args
        if brief:
            args = [a for a in args if a != "--brief"]
        # Phase 7I: --virtual shows auto-generated virtual edges (hidden by default)
        show_virtual = "--virtual" in args
        if show_virtual:
            args = [a for a in args if a != "--virtual"]
        # Precision Recall escape hatches (default: postprocess ON):
        #   --no-recency-weight   disable R1 (recency × supersede soft decay)
        #   --no-community-filter disable R7 (community dominance ratio)
        #   --no-context-anchor   disable R5 (active_context elevation boost)
        #   --no-multi-seed       disable R2 (multi-token chain intersection)
        #   --no-edge-fallback    disable R6 (edge-content fallback for unmatched tokens)
        #   --legacy              equivalent to all --no-* flags above
        legacy_mode = "--legacy" in args
        if legacy_mode:
            args = [a for a in args if a != "--legacy"]
        no_recency_weight = legacy_mode or "--no-recency-weight" in args
        if "--no-recency-weight" in args:
            args = [a for a in args if a != "--no-recency-weight"]
        no_community_filter = legacy_mode or "--no-community-filter" in args
        if "--no-community-filter" in args:
            args = [a for a in args if a != "--no-community-filter"]
        no_context_anchor = legacy_mode or "--no-context-anchor" in args
        if "--no-context-anchor" in args:
            args = [a for a in args if a != "--no-context-anchor"]
        no_multi_seed = legacy_mode or "--no-multi-seed" in args
        if "--no-multi-seed" in args:
            args = [a for a in args if a != "--no-multi-seed"]
        no_edge_fallback = legacy_mode or "--no-edge-fallback" in args
        if "--no-edge-fallback" in args:
            args = [a for a in args if a != "--no-edge-fallback"]
        cmd_propagate(engine, " ".join(args), use_gravity=use_gravity, resolution=resolution,
                      all_chains=all_chains, all_modifiers=all_modifiers, expand_top=expand_top,
                      community_mode=community_mode, top_m=top_m, brief=brief,
                      show_virtual=show_virtual,
                      no_recency_weight=no_recency_weight,
                      no_community_filter=no_community_filter,
                      no_context_anchor=no_context_anchor,
                      no_multi_seed=no_multi_seed,
                      no_edge_fallback=no_edge_fallback)
    elif cmd == "gravity":
        subcmd = sys.argv[2] if len(sys.argv) >= 3 else "info"
        cmd_gravity(engine, subcmd, sys.argv[3:] if len(sys.argv) >= 4 else [])
    elif cmd == "select":
        cmd_select(engine, sys.argv[2:])
    elif cmd == "bind":
        cmd_bind(engine, sys.argv[2:])
    elif cmd == "paths" and len(sys.argv) >= 4:
        cmd_paths(engine, sys.argv[2], sys.argv[3])
    elif cmd == "node" and len(sys.argv) >= 3:
        node_args = sys.argv[2:]
        show_virtual = "--virtual" in node_args
        show_provenance = "--full" in node_args
        node_args = [a for a in node_args if a not in ("--virtual", "--full")]
        limit = None
        if "--limit" in node_args:
            idx = node_args.index("--limit")
            if idx + 1 < len(node_args):
                try:
                    limit = int(node_args[idx + 1])
                except ValueError:
                    pass
                node_args = node_args[:idx] + node_args[idx + 2:]
        if node_args:
            cmd_node(
                engine, node_args[0],
                show_virtual=show_virtual, limit=limit,
                show_provenance=show_provenance,
            )
    elif cmd == "ingest" and len(sys.argv) >= 3:
        args = sys.argv[2:]
        cognitive = "--cognitive" in args
        no_link = "--no-link" in args
        if cognitive:
            args = [a for a in args if a != "--cognitive"]
        if no_link:
            args = [a for a in args if a != "--no-link"]
        edge_class = "cognitive" if cognitive else "knowledge"
        cmd_ingest(engine, " ".join(args), edge_class=edge_class, no_link=no_link)
    elif cmd == "ingest-file" and len(sys.argv) >= 3:
        args = sys.argv[2:]
        ca = None
        if "--created-at" in args:
            idx = args.index("--created-at")
            if idx + 1 < len(args):
                ca = float(args[idx + 1])
                args = args[:idx] + args[idx + 2:]
        cmd_ingest_file(engine, args[0], created_at=ca)
    elif cmd == "merge" and len(sys.argv) >= 3:
        cmd_merge(engine, " ".join(sys.argv[2:]))
    elif cmd == "consolidate":
        cmd_consolidate(engine, sys.argv[2:])
    elif cmd == "xref":
        cmd_xref(engine, sys.argv[2:])
    elif cmd == "epistemic":
        subcmd = sys.argv[2] if len(sys.argv) >= 3 else ""
        args = sys.argv[3:] if len(sys.argv) >= 4 else []
        cmd_epistemic(engine, subcmd, args)
    elif cmd == "virtual":
        subcmd = sys.argv[2] if len(sys.argv) >= 3 else ""
        args = sys.argv[3:] if len(sys.argv) >= 4 else []
        cmd_virtual(engine, subcmd, args)
    elif cmd == "alias":
        subcmd = sys.argv[2] if len(sys.argv) >= 3 else ""
        args = sys.argv[3:] if len(sys.argv) >= 4 else []
        cmd_alias(engine, subcmd, args)
    elif cmd == "attrs":
        cmd_attrs(engine, sys.argv[2:])
    elif cmd == "metrics":
        cmd_metrics(engine)
    elif cmd == "importance":
        # Parse --top N flag
        top_n = 20
        args = sys.argv[2:]
        if "--top" in args:
            idx = args.index("--top")
            if idx + 1 < len(args):
                try:
                    top_n = int(args[idx + 1])
                except ValueError:
                    pass
        elif args:
            try:
                top_n = int(args[0])
            except ValueError:
                pass
        cmd_importance(engine, top_n)
    elif cmd == "learn" and len(sys.argv) >= 3:
        subcmd = sys.argv[2]
        args = sys.argv[3:]
        cmd_learn(engine, subcmd, args)
    elif cmd == "prune":
        # Parse --dry-run, --conf, --days flags
        args = sys.argv[2:]
        dry_run = "--dry-run" in args
        conf = 0.1
        days = 30.0
        if "--conf" in args:
            idx = args.index("--conf")
            if idx + 1 < len(args):
                try:
                    conf = float(args[idx + 1])
                except ValueError:
                    pass
        if "--days" in args:
            idx = args.index("--days")
            if idx + 1 < len(args):
                try:
                    days = float(args[idx + 1])
                except ValueError:
                    pass
        cmd_prune(engine, dry_run=dry_run, conf=conf, days=days)
    elif cmd == "pruned":
        args = sys.argv[2:]
        limit = 50
        item_type = None
        if "--limit" in args:
            idx = args.index("--limit")
            if idx + 1 < len(args):
                try:
                    limit = int(args[idx + 1])
                except ValueError:
                    pass
        if "--type" in args:
            idx = args.index("--type")
            if idx + 1 < len(args):
                item_type = args[idx + 1]
        cmd_pruned(limit=limit, item_type=item_type)
    elif cmd == "telemetry":
        subcmd = sys.argv[2] if len(sys.argv) >= 3 else ""
        args = sys.argv[3:] if len(sys.argv) >= 4 else []
        cmd_telemetry(engine, subcmd, args)
    elif cmd == "simulate":
        subcmd = sys.argv[2] if len(sys.argv) >= 3 else ""
        args = sys.argv[3:] if len(sys.argv) >= 4 else []
        cmd_simulate(engine, subcmd, args)
    elif cmd == "topology":
        subcmd = sys.argv[2] if len(sys.argv) >= 3 else "analyze"
        args = sys.argv[3:] if len(sys.argv) >= 4 else []
        cmd_topology(engine, subcmd, args)
    elif cmd == "cognitive":
        subcmd = sys.argv[2] if len(sys.argv) >= 3 else ""
        args = sys.argv[3:] if len(sys.argv) >= 4 else []
        cmd_cognitive(engine, subcmd, args)
    elif cmd == "feedback":
        subcmd = sys.argv[2] if len(sys.argv) >= 3 else ""
        args = sys.argv[3:] if len(sys.argv) >= 4 else []
        cmd_feedback(engine, subcmd, args)
    elif cmd == "benchmark":
        subcmd = sys.argv[2] if len(sys.argv) >= 3 else ""
        args = sys.argv[3:] if len(sys.argv) >= 4 else []
        cmd_benchmark(engine, subcmd, args)
    elif cmd == "search" and len(sys.argv) >= 3:
        args = sys.argv[2:]
        top_k = 10
        propagate = True
        min_sim = 0.3
        query_parts = []
        i = 0
        while i < len(args):
            if args[i] == "--top" and i + 1 < len(args):
                try:
                    top_k = int(args[i + 1])
                except ValueError:
                    pass
                i += 2
            elif args[i] == "--no-propagate":
                propagate = False
                i += 1
            elif args[i] == "--min-sim" and i + 1 < len(args):
                try:
                    min_sim = float(args[i + 1])
                except ValueError:
                    pass
                i += 2
            else:
                query_parts.append(args[i])
                i += 1
        cmd_search(engine, " ".join(query_parts), top_k=top_k,
                   propagate=propagate, min_similarity=min_sim)
    elif cmd == "embed":
        model_name = None
        args = sys.argv[2:]
        if "--model" in args:
            idx = args.index("--model")
            if idx + 1 < len(args):
                model_name = args[idx + 1]
        cmd_embed(engine, model_name=model_name)
    elif cmd == "converge" and len(sys.argv) >= 3:
        args = sys.argv[2:]
        top_k, max_iter, threshold = 5, 5, 0.8
        query_parts = []
        i = 0
        while i < len(args):
            if args[i] == "--top-k" and i + 1 < len(args):
                try: top_k = int(args[i + 1])
                except ValueError: pass
                i += 2
            elif args[i] == "--max-iter" and i + 1 < len(args):
                try: max_iter = int(args[i + 1])
                except ValueError: pass
                i += 2
            elif args[i] == "--threshold" and i + 1 < len(args):
                try: threshold = float(args[i + 1])
                except ValueError: pass
                i += 2
            else:
                query_parts.append(args[i])
                i += 1
        cmd_converge(engine, " ".join(query_parts), top_k=top_k,
                     max_iter=max_iter, threshold=threshold)
    elif cmd == "preference":
        subcmd = sys.argv[2] if len(sys.argv) >= 3 else ""
        args = sys.argv[3:] if len(sys.argv) >= 4 else []
        cmd_preference(engine, subcmd, args)
    elif cmd == "visualize":
        cmd_visualize(engine, sys.argv[2:])
    elif cmd == "coactivation":
        subcmd = sys.argv[2] if len(sys.argv) >= 3 else ""
        cmd_coactivation(engine, subcmd, sys.argv[3:])
    elif cmd == "temporal":
        subcmd = sys.argv[2] if len(sys.argv) >= 3 else ""
        cmd_temporal(engine, subcmd, sys.argv[3:])
    elif cmd == "guide":
        guide_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "STG_AGENT_GUIDE.md")
        if os.path.exists(guide_path):
            with open(guide_path, encoding="utf-8") as f:
                print(f.read())
        else:
            print(f"Guide not found at {guide_path}")
    elif cmd == "backup":
        from stg_engine.persistence import backup_database
        keep = 7
        backup_dir = None
        args = sys.argv[2:]
        i = 0
        while i < len(args):
            if args[i] == "--keep" and i + 1 < len(args):
                keep = int(args[i + 1])
                i += 2
            elif args[i] == "--dir" and i + 1 < len(args):
                backup_dir = args[i + 1]
                i += 2
            else:
                i += 1
        backup_path = backup_database(STG_PATH, backup_dir=backup_dir, keep=keep)
        print(f"Backup created: {backup_path}")
        print(f"Nodes: {len(engine._nodes)}, Edges: {len(engine._edges)}")
        # List existing backups
        from pathlib import Path
        bak_dir = Path(backup_dir) if backup_dir else Path(STG_PATH).parent / "backups"
        backups = sorted(bak_dir.glob("memory-*.stg"))
        print(f"Retained backups: {len(backups)} (keep={keep})")
        for b in backups:
            size_kb = b.stat().st_size / 1024
            print(f"  {b.name}  ({size_kb:.0f} KB)")
    elif cmd == "heartbeat":
        cmd_heartbeat(sys.argv[2:])
        return  # heartbeat manages its own lifecycle, skip audit
    elif cmd == "perceive" and len(sys.argv) >= 3:
        cmd_perceive(engine, sys.argv[2:])
    elif cmd == "perception":
        cmd_perception(engine, sys.argv[2:])
    elif cmd == "use" and len(sys.argv) >= 3:
        cmd_use(engine, sys.argv[2:])
    elif cmd == "skill":
        cmd_skill(engine, sys.argv[2:])
    else:
        print(__doc__)
        sys.exit(1)

    # --- Audit: snapshot after ---
    # Only log write operations (skip read-only commands to keep log clean)
    _WRITE_CMDS = {
        "ingest", "ingest-file", "prune", "feedback", "learn",
        "virtual", "preference", "embed", "backup", "converge",
        "merge", "consolidate", "xref", "select", "bind", "alias",
    }
    if cmd in _WRITE_CMDS:
        _after = _snap(engine)
        _audit_log(cmd, sys.argv[2:], _before, _after)

    if os.environ.get("STG_DEBUG"):
        print(f"\n[load: {load_ms:.1f}ms]")


if __name__ == "__main__":
    main()
