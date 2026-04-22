"""Skill Executor — resolve, validate, and invoke Skill nodes stored in the STG.

This module turns `Skill:`-namespaced nodes into callable capabilities via the
`stg use` / `stg skill use` CLI surface. See
`development/design/STG_SKILL_EXECUTOR_DESIGN.md` (v0.2) for the full spec.

Design invariants (enforced here):

  1. `skill.enabled` must be `true` in the user's config. No execution on
     fresh installs (gate 1 of double opt-in).
  2. The edge must have `executable=true` modifier. (gate 2).
  3. The resolved Skill script path must live under at least one entry in
     `skill.roots`. Symlinks are resolved; no escape by indirection.
  4. Interpreter resolution order: absolute path → named in `skill.interpreters`
     → builtin (`python3`/`python`/`bash`/`sh`/`node`) via `shutil.which`.
     No hardcoded user-specific names like `python_venv`.
  5. subprocess is invoked with a list of args (never `shell=True`) and a
     hard timeout (default 60s, capped by user config, overridable per-call
     via CLI flag).
  6. Output captured up to `skill.output_cap_bytes` (default 10 MB); beyond
     that a `<<<TRUNCATED>>>` marker is appended.
  7. Every invocation writes one row to the `skill_invocations` SQLite table
     for audit.

No personal paths, no hardcoded home-dir assumptions. All path/interpreter
defaults live in `~/.stg/config.json` per user.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple, List

from stg_engine.engine import (
    SKILL_INVOCATION_FIELDS,
    SKILL_NAMESPACE,
    _get_skill_invocation,
)

# --- Exit codes (matches spec §5.3) -----------------------------------------
EXIT_OK = 0
EXIT_ENGINE_ERROR = 1
EXIT_NOT_FOUND = 3
EXIT_CHILD_NONZERO = 4
EXIT_AMBIGUOUS = 5
EXIT_NOT_EXECUTABLE = 6
EXIT_TIMEOUT = 7
EXIT_STL_PARSE_ERROR = 8

# --- Builtin interpreter names that resolve via shutil.which ---------------
BUILTIN_INTERPRETERS: Tuple[str, ...] = (
    "python3", "python", "bash", "sh", "node",
)

# --- Defaults (used when user config hasn't set a value) -------------------
DEFAULT_TIMEOUT_S = 60
DEFAULT_OUTPUT_CAP = 10 * 1024 * 1024  # 10 MB
MAX_TIMEOUT_S = 600  # hard cap even if user config says more


# =============================================================================
# Skill resolution
# =============================================================================

class SkillResolutionError(Exception):
    """Raised by resolve_skill when a Skill cannot be uniquely identified."""
    def __init__(self, message: str, exit_code: int = EXIT_NOT_FOUND):
        super().__init__(message)
        self.exit_code = exit_code


def _node_namespace(engine, display_name: str) -> Optional[str]:
    """Look up a node's namespace given its display-name casing.

    The engine stores nodes keyed by normalized (lowercase) name but STGEdge
    objects carry the original-casing source/target strings. We normalize
    ourselves to look up the namespace.
    """
    key = display_name.lower().replace("-", "_")
    node = engine._nodes.get(key)
    return node.namespace if node else None


def _node_display_name(engine, key_or_name: str) -> str:
    """Look up a node's original-casing display name."""
    key = key_or_name.lower().replace("-", "_")
    node = engine._nodes.get(key)
    return node.name if node else key_or_name


def iter_skill_edges(engine):
    """Yield every STGEdge whose source node is in the Skill namespace.

    Yields triples of (source_display, target_display, edge_data_dict).
    The edge_data_dict is constructed from the STGEdge dataclass fields —
    modifiers are flattened to the top level for easy lookup, and the
    confidence/path/description etc. are included directly.
    """
    for edge in engine._edges:
        if _node_namespace(engine, edge.source) != SKILL_NAMESPACE:
            continue
        data: dict = {}
        # Flatten modifiers first (so explicit edge fields can override)
        if edge.modifiers:
            data.update(edge.modifiers)
        # Canonical fields from STGEdge
        data["confidence"] = edge.confidence
        data["strength"] = edge.strength
        data["rule"] = edge.rule
        data["salience"] = edge.salience
        data["created_at"] = edge.created_at
        yield edge.source, edge.target, data


def find_skill_edges_by_name(engine, skill_name: str) -> List[Tuple[str, str, dict]]:
    """Return all edges whose source node name matches `skill_name` in the Skill namespace.

    Skill names match case-insensitively against the base name (the part after
    the `Skill:` prefix). E.g. `Clean_Reddit_Thread_Save` matches the node whose
    full name is `Skill:Clean_Reddit_Thread_Save`.
    """
    target = skill_name.strip()
    if target.startswith(f"{SKILL_NAMESPACE}:"):
        target = target[len(SKILL_NAMESPACE) + 1:]
    target_lower = target.lower()
    prefix_lower = f"{SKILL_NAMESPACE}:".lower()
    matches = []
    for u, v, data in iter_skill_edges(engine):
        u_base = u.lower()
        if u_base.startswith(prefix_lower):
            u_base = u_base[len(prefix_lower):]
        if u_base == target_lower:
            matches.append((u, v, data))
    return matches


def resolve_skill(engine, skill_name: str) -> Tuple[str, str, dict]:
    """Find exactly one executable Skill edge for the given name.

    Priority when multiple edges exist:
      1. Prefer edges with executable=true
      2. Highest confidence
      3. Most recent created_at (if tracked)
      4. Highest salience

    Raises SkillResolutionError on not-found or ambiguous.
    """
    edges = find_skill_edges_by_name(engine, skill_name)
    if not edges:
        raise SkillResolutionError(
            f"no Skill node named '{skill_name}' found.\n"
            f"Run `stg skill list` to see available skills.",
            exit_code=EXIT_NOT_FOUND,
        )

    def _key(edge):
        _, _, d = edge
        inv = _get_skill_invocation(d)
        return (
            bool(inv.get("executable")),
            float(d.get("confidence") or 0.0),
            float(d.get("created_at") or 0.0),
            float(d.get("salience") or 0.0),
        )

    edges_sorted = sorted(edges, key=_key, reverse=True)
    top = edges_sorted[0]

    # Ambiguity check: if the top two are very close in (executable, confidence)
    if len(edges_sorted) >= 2:
        k1 = _key(edges_sorted[0])
        k2 = _key(edges_sorted[1])
        if (k1[0] == k2[0]
                and abs(k1[1] - k2[1]) < 0.02
                and k1[2] == k2[2]):
            t1 = edges_sorted[0][1]
            t2 = edges_sorted[1][1]
            raise SkillResolutionError(
                f"ambiguous: Skill '{skill_name}' has multiple equally-ranked edges:\n"
                f"  → {t1}\n"
                f"  → {t2}\n"
                f"Add a more specific skill name, or prune duplicate edges.",
                exit_code=EXIT_AMBIGUOUS,
            )
    return top


# =============================================================================
# Config-driven validation
# =============================================================================

def validate_config_enabled(user_config: dict) -> Optional[str]:
    """Return an error message if skill execution isn't enabled, else None."""
    skill_cfg = user_config.get("skill") if isinstance(user_config, dict) else None
    if not isinstance(skill_cfg, dict):
        return (
            "skill execution is disabled (no skill.* config).\n"
            "Run: stg config set skill.enabled true\n"
            "Then: stg config set skill.roots \"/abs/path/to/your/tools\""
        )
    if not skill_cfg.get("enabled"):
        return (
            "skill execution is disabled.\n"
            "Run: stg config set skill.enabled true"
        )
    roots = skill_cfg.get("roots") or []
    if not isinstance(roots, list) or not roots:
        return (
            "no skill.roots configured. Run:\n"
            "  stg config set skill.roots \"/abs/path1,/abs/path2\"\n"
            "Only scripts under one of these directories can be executed."
        )
    return None


def path_under_roots(script_path: Path, roots: List[str]) -> bool:
    """True if `script_path` (after resolving symlinks) lies under at least one root."""
    try:
        resolved = script_path.resolve(strict=True)
    except (FileNotFoundError, OSError):
        return False
    for root in roots:
        try:
            root_resolved = Path(root).expanduser().resolve(strict=True)
        except (FileNotFoundError, OSError):
            continue
        try:
            resolved.relative_to(root_resolved)
            return True
        except ValueError:
            continue
    return False


def resolve_interpreter(raw: str, user_config: dict) -> Optional[str]:
    """Turn a Skill edge's `interpreter` value into an absolute binary path.

    Resolution order:
      1. Absolute path (exists & is a file)           → return as-is
      2. Name in user_config.skill.interpreters       → return that absolute path
      3. Builtin name (python3/python/bash/sh/node)   → shutil.which()
      4. None (not resolvable)
    """
    if not raw:
        return None
    raw = raw.strip()

    # 1. Absolute path
    if os.path.isabs(raw) and os.path.isfile(raw):
        return raw

    # 2. User-configured name
    skill_cfg = user_config.get("skill") if isinstance(user_config, dict) else None
    if isinstance(skill_cfg, dict):
        user_interps = skill_cfg.get("interpreters") or {}
        if isinstance(user_interps, dict) and raw in user_interps:
            path = user_interps[raw]
            if path and os.path.isabs(path) and os.path.isfile(path):
                return path

    # 3. Builtin name
    if raw in BUILTIN_INTERPRETERS:
        found = shutil.which(raw)
        if found:
            return found

    return None


# =============================================================================
# Invocation
# =============================================================================

class SkillInvocationResult:
    __slots__ = (
        "skill_name", "target", "path", "interpreter",
        "args", "stdin_stl",
        "exit_code", "stdout", "stderr",
        "elapsed_s", "bytes_out", "bytes_err",
        "truncated_stdout", "timed_out",
        "error", "invocation_id",
    )

    def __init__(self, **kw):
        for s in self.__slots__:
            setattr(self, s, kw.get(s))


def _hash_args(args: List[str]) -> str:
    """Short hash of args for audit rows (avoid storing huge arg lists verbatim)."""
    h = hashlib.sha1("\0".join(args).encode("utf-8", "replace")).hexdigest()[:12]
    return h


def _generate_invocation_id() -> str:
    return "inv_" + hashlib.sha1(
        f"{time.time_ns()}{os.getpid()}".encode()
    ).hexdigest()[:12]


def invoke_skill(
    edge: Tuple[str, str, dict],
    skill_name: str,
    args: List[str],
    user_config: dict,
    stdin_stl: Optional[str] = None,
    timeout_override: Optional[int] = None,
    output_cap: Optional[int] = None,
) -> SkillInvocationResult:
    """Run a resolved Skill edge. Does NOT do policy checks — caller must have
    already called validate_config_enabled() and confirmed the edge is ok.
    """
    source, target, data = edge
    inv = _get_skill_invocation(data)
    path = data.get("path")
    script_path = Path(str(path)).expanduser()

    # Resolve timeout
    skill_cfg = user_config.get("skill", {}) if isinstance(user_config, dict) else {}
    declared_timeout = inv.get("timeout_s")
    default_timeout = int(skill_cfg.get("default_timeout_s") or DEFAULT_TIMEOUT_S)
    timeout_s = timeout_override or declared_timeout or default_timeout
    timeout_s = max(1, min(int(timeout_s), MAX_TIMEOUT_S))

    # Resolve output cap
    cap = output_cap or int(skill_cfg.get("output_cap_bytes") or DEFAULT_OUTPUT_CAP)

    # Resolve interpreter
    interpreter_raw = inv.get("interpreter", "")
    interpreter = resolve_interpreter(interpreter_raw, user_config)
    if not interpreter:
        return SkillInvocationResult(
            skill_name=skill_name,
            target=target,
            path=str(script_path),
            interpreter=None,
            args=args,
            stdin_stl=stdin_stl,
            exit_code=EXIT_NOT_EXECUTABLE,
            stdout="",
            stderr="",
            elapsed_s=0.0,
            bytes_out=0,
            bytes_err=0,
            truncated_stdout=False,
            timed_out=False,
            error=(
                f"cannot resolve interpreter {interpreter_raw!r}.\n"
                f"Either set an absolute path on the Skill's interpreter modifier, "
                f"configure it with `stg config set skill.interpreters.{interpreter_raw} /abs/path`, "
                f"or use a builtin: {', '.join(BUILTIN_INTERPRETERS)}."
            ),
            invocation_id=_generate_invocation_id(),
        )

    cmd = [interpreter, str(script_path), *args]
    invocation_id = _generate_invocation_id()
    t0 = time.time()
    stdout_s = ""
    stderr_s = ""
    truncated = False
    timed_out = False
    exit_code = 0

    try:
        proc = subprocess.run(
            cmd,
            input=stdin_stl,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            shell=False,
        )
        stdout_s = proc.stdout or ""
        stderr_s = proc.stderr or ""
        exit_code = proc.returncode
    except subprocess.TimeoutExpired as e:
        stdout_s = (e.stdout or b"")
        stderr_s = (e.stderr or b"")
        if isinstance(stdout_s, bytes):
            stdout_s = stdout_s.decode("utf-8", "replace")
        if isinstance(stderr_s, bytes):
            stderr_s = stderr_s.decode("utf-8", "replace")
        exit_code = EXIT_TIMEOUT
        timed_out = True
    except FileNotFoundError as e:
        return SkillInvocationResult(
            skill_name=skill_name, target=target, path=str(script_path),
            interpreter=interpreter, args=args, stdin_stl=stdin_stl,
            exit_code=EXIT_NOT_EXECUTABLE, stdout="", stderr=str(e),
            elapsed_s=time.time() - t0, bytes_out=0, bytes_err=0,
            truncated_stdout=False, timed_out=False,
            error=f"executable not found: {e}",
            invocation_id=invocation_id,
        )

    # Cap output size
    if len(stdout_s) > cap:
        stdout_s = stdout_s[:cap] + "\n<<<TRUNCATED at output_cap_bytes>>>\n"
        truncated = True

    elapsed = time.time() - t0

    # Translate child non-zero exit (unless it was our timeout marker) into our
    # EXIT_CHILD_NONZERO taxonomy for the outer `stg use` caller.
    final_code = (
        EXIT_TIMEOUT if timed_out
        else (EXIT_CHILD_NONZERO if exit_code != 0 else EXIT_OK)
    )

    return SkillInvocationResult(
        skill_name=skill_name,
        target=target,
        path=str(script_path),
        interpreter=interpreter,
        args=args,
        stdin_stl=stdin_stl,
        exit_code=final_code if (timed_out or exit_code != 0) else 0,
        stdout=stdout_s,
        stderr=stderr_s,
        elapsed_s=elapsed,
        bytes_out=len(stdout_s),
        bytes_err=len(stderr_s),
        truncated_stdout=truncated,
        timed_out=timed_out,
        error=None,
        invocation_id=invocation_id,
    )


# =============================================================================
# High-level entry point used by cli.py
# =============================================================================

def run_skill(
    engine,
    skill_name: str,
    args: List[str],
    user_config: dict,
    stdin_stl: Optional[str] = None,
    timeout_override: Optional[int] = None,
    dry_run: bool = False,
) -> SkillInvocationResult:
    """Full pipeline: validate config → resolve → validate edge → invoke → audit.

    Returns a SkillInvocationResult. The caller (cli.py) is responsible for
    pretty-printing the result and choosing the process exit code.
    """
    # Gate 1: user config enabled
    err = validate_config_enabled(user_config)
    if err:
        return SkillInvocationResult(
            skill_name=skill_name, target=None, path=None, interpreter=None,
            args=args, stdin_stl=stdin_stl,
            exit_code=EXIT_NOT_EXECUTABLE,
            stdout="", stderr="",
            elapsed_s=0.0, bytes_out=0, bytes_err=0,
            truncated_stdout=False, timed_out=False,
            error=err,
            invocation_id=_generate_invocation_id(),
        )

    # Resolve
    try:
        edge = resolve_skill(engine, skill_name)
    except SkillResolutionError as e:
        return SkillInvocationResult(
            skill_name=skill_name, target=None, path=None, interpreter=None,
            args=args, stdin_stl=stdin_stl,
            exit_code=e.exit_code, stdout="", stderr="",
            elapsed_s=0.0, bytes_out=0, bytes_err=0,
            truncated_stdout=False, timed_out=False,
            error=str(e),
            invocation_id=_generate_invocation_id(),
        )
    source, target, data = edge
    inv = _get_skill_invocation(data)

    # Gate 2: executable=true on edge
    if not inv.get("executable"):
        return SkillInvocationResult(
            skill_name=skill_name, target=target,
            path=str(data.get("path")), interpreter=None,
            args=args, stdin_stl=stdin_stl,
            exit_code=EXIT_NOT_EXECUTABLE, stdout="", stderr="",
            elapsed_s=0.0, bytes_out=0, bytes_err=0,
            truncated_stdout=False, timed_out=False,
            error=(
                f"Skill '{skill_name}' edge does not have executable=true.\n"
                f"Run: stg skill configure {skill_name} --executable "
                f"--interpreter <name> --args-template '...'"
            ),
            invocation_id=_generate_invocation_id(),
        )

    # Gate 3: path exists + under skill.roots
    path_str = data.get("path")
    if not path_str:
        return SkillInvocationResult(
            skill_name=skill_name, target=target, path=None,
            interpreter=None, args=args, stdin_stl=stdin_stl,
            exit_code=EXIT_NOT_EXECUTABLE, stdout="", stderr="",
            elapsed_s=0.0, bytes_out=0, bytes_err=0,
            truncated_stdout=False, timed_out=False,
            error=f"Skill '{skill_name}' edge has no `path` modifier.",
            invocation_id=_generate_invocation_id(),
        )
    script_path = Path(str(path_str)).expanduser()
    if not script_path.is_file():
        return SkillInvocationResult(
            skill_name=skill_name, target=target, path=str(script_path),
            interpreter=None, args=args, stdin_stl=stdin_stl,
            exit_code=EXIT_NOT_EXECUTABLE, stdout="", stderr="",
            elapsed_s=0.0, bytes_out=0, bytes_err=0,
            truncated_stdout=False, timed_out=False,
            error=f"Skill path does not exist or is not a file: {script_path}",
            invocation_id=_generate_invocation_id(),
        )
    roots = (user_config.get("skill") or {}).get("roots") or []
    if not path_under_roots(script_path, roots):
        return SkillInvocationResult(
            skill_name=skill_name, target=target, path=str(script_path),
            interpreter=None, args=args, stdin_stl=stdin_stl,
            exit_code=EXIT_NOT_EXECUTABLE, stdout="", stderr="",
            elapsed_s=0.0, bytes_out=0, bytes_err=0,
            truncated_stdout=False, timed_out=False,
            error=(
                f"Skill path {script_path} is not under any configured "
                f"skill.roots entry.\n"
                f"Current roots: {roots}\n"
                f"Add with: stg config set skill.roots \"{','.join(roots + [str(script_path.parent)])}\""
            ),
            invocation_id=_generate_invocation_id(),
        )

    # Dry-run short-circuit
    if dry_run:
        interp = resolve_interpreter(inv.get("interpreter", ""), user_config) or "<unresolved>"
        return SkillInvocationResult(
            skill_name=skill_name, target=target, path=str(script_path),
            interpreter=interp, args=args, stdin_stl=stdin_stl,
            exit_code=EXIT_OK,
            stdout=f"[dry-run] would run: {interp} {script_path} {' '.join(args)}\n",
            stderr="",
            elapsed_s=0.0, bytes_out=0, bytes_err=0,
            truncated_stdout=False, timed_out=False,
            error=None,
            invocation_id=_generate_invocation_id(),
        )

    # Invoke
    return invoke_skill(
        edge,
        skill_name,
        args,
        user_config,
        stdin_stl=stdin_stl,
        timeout_override=timeout_override,
    )


# =============================================================================
# Audit log (skill_invocations table in .stg file)
# =============================================================================

HISTORY_CAP = 10000  # rolling window; oldest rows evicted beyond this


def write_audit_row(stg_path: str, result: SkillInvocationResult) -> None:
    """Append one row to skill_invocations and enforce the rolling cap.

    Fails silently (logged to stderr) if the database is unavailable — audit
    must never break a user's `stg use` call.
    """
    try:
        conn = sqlite3.connect(str(stg_path))
        try:
            args_preview = " ".join(result.args or [])
            if len(args_preview) > 400:
                args_preview = args_preview[:397] + "..."
            conn.execute(
                "INSERT INTO skill_invocations "
                "(invocation_id, timestamp, skill_name, target, path, interpreter,"
                " args_hash, args_preview, exit_code, elapsed_s, bytes_out, bytes_err,"
                " truncated, timed_out, error_msg) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    result.invocation_id,
                    time.time(),
                    result.skill_name,
                    result.target,
                    result.path,
                    result.interpreter,
                    _hash_args(result.args or []),
                    args_preview,
                    int(result.exit_code or 0),
                    float(result.elapsed_s or 0.0),
                    int(result.bytes_out or 0),
                    int(result.bytes_err or 0),
                    1 if result.truncated_stdout else 0,
                    1 if result.timed_out else 0,
                    result.error,
                ),
            )
            # Rolling cap
            count = conn.execute(
                "SELECT COUNT(*) FROM skill_invocations"
            ).fetchone()[0]
            if count > HISTORY_CAP:
                conn.execute(
                    "DELETE FROM skill_invocations WHERE id IN "
                    "(SELECT id FROM skill_invocations "
                    "ORDER BY timestamp ASC LIMIT ?)",
                    (count - HISTORY_CAP,),
                )
            conn.commit()
        finally:
            conn.close()
    except sqlite3.Error as e:
        import sys as _sys
        print(f"warning: skill audit write failed: {e}", file=_sys.stderr)


def read_audit_history(
    stg_path: str,
    skill_name: Optional[str] = None,
    since_epoch: Optional[float] = None,
    limit: int = 50,
) -> List[dict]:
    """Read recent skill_invocations rows. Returns newest first."""
    try:
        conn = sqlite3.connect(str(stg_path))
    except sqlite3.Error:
        return []
    try:
        sql = (
            "SELECT invocation_id, timestamp, skill_name, target, path, interpreter, "
            "args_preview, exit_code, elapsed_s, bytes_out, bytes_err, truncated, "
            "timed_out, error_msg "
            "FROM skill_invocations WHERE 1=1"
        )
        params: list = []
        if skill_name:
            sql += " AND skill_name = ?"
            params.append(skill_name)
        if since_epoch:
            sql += " AND timestamp >= ?"
            params.append(since_epoch)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(int(limit))
        rows = conn.execute(sql, params).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    keys = ["invocation_id", "timestamp", "skill_name", "target", "path",
            "interpreter", "args_preview", "exit_code", "elapsed_s",
            "bytes_out", "bytes_err", "truncated", "timed_out", "error_msg"]
    return [dict(zip(keys, r)) for r in rows]


# =============================================================================
# Catalog (for `stg skill list` and `stg propagate skill`)
# =============================================================================

def list_skills(engine, filter_keyword: Optional[str] = None,
                executable_only: bool = False) -> List[dict]:
    """Return a list of dicts summarising every Skill edge.

    Each dict: {name, target, path, interpreter, args_template, executable,
               stl_io, timeout_s, description}.
    """
    # Collect all edges, then dedupe by name: keep the "best" edge per skill
    # (executable=true > has_path > highest confidence).
    prefix = f"{SKILL_NAMESPACE}:"
    by_name: dict = {}
    for u, v, data in iter_skill_edges(engine):
        # u may be stored as "Skill:Echo" (full) or just "Echo"; normalize to
        # the bare name for display.
        base_name = u[len(prefix):] if u.startswith(prefix) else u
        inv = _get_skill_invocation(data)
        rec = {
            "name": base_name,
            "target": v,
            "path": data.get("path") or "",
            "interpreter": inv.get("interpreter", ""),
            "args_template": inv.get("args_template", ""),
            "executable": bool(inv.get("executable")),
            "stl_io": bool(inv.get("stl_io")),
            "timeout_s": inv.get("timeout_s"),
            "description": data.get("description") or "",
            "confidence": float(data.get("confidence") or 0.0),
        }

        existing = by_name.get(base_name)
        if existing is None:
            by_name[base_name] = rec
        else:
            # Replace if this edge is "better"
            better = (
                (rec["executable"] and not existing["executable"])
                or (rec["path"] and not existing["path"])
                or (rec["confidence"] > existing["confidence"])
            )
            if better:
                by_name[base_name] = rec

    out = []
    for rec in by_name.values():
        if executable_only and not rec["executable"]:
            continue
        if filter_keyword:
            kw = filter_keyword.lower()
            hay = " ".join([rec["name"], rec["path"], rec["description"]]).lower()
            if kw not in hay:
                continue
        out.append(rec)
    # Sort: executable first, then by name
    out.sort(key=lambda r: (not r["executable"], r["name"].lower()))
    return out


def render_catalog(skills: List[dict]) -> str:
    """Human-readable catalog format for `stg skill list` / `propagate skill`."""
    if not skills:
        return "(no skills registered)"
    lines = []
    n_exec = sum(1 for s in skills if s["executable"])
    lines.append(f"Skills ({n_exec} executable, {len(skills) - n_exec} knowledge-only):")
    lines.append("")
    for s in skills:
        mark = "✓" if s["executable"] else "·"
        lines.append(f"  {mark} {s['name']}")
        if s["path"]:
            lines.append(f"        path:        {s['path']}")
        if s["interpreter"]:
            lines.append(f"        interpreter: {s['interpreter']}")
        if s["stl_io"]:
            lines.append(f"        stl_io:      true")
        if s["timeout_s"]:
            lines.append(f"        timeout:     {s['timeout_s']}s")
        if s["args_template"]:
            lines.append(f"        args:        {s['args_template']}")
        if s["description"]:
            desc = s["description"]
            if len(desc) > 100:
                desc = desc[:100] + "…"
            lines.append(f"        purpose:     {desc}")
        lines.append("")
    return "\n".join(lines)
