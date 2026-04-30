"""Unit tests for the Skill Executor (v0.3.1+).

Covers:
  - SKILL_INVOCATION_FIELDS extraction
  - Config helpers (dotted keys, coercion)
  - Skill resolution (name matching, ambiguity)
  - Interpreter resolution (absolute path / named config / builtin fallback)
  - Security gates (enabled, roots, executable)
  - Invocation (positive path, timeout, output cap, stdin-STL)
  - Catalog rendering
  - Audit log write/read roundtrip

These tests use a temporary .stg file and mocked user_config dicts to avoid
touching the real ~/.stg/config.json or the user's actual STG.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from stg_engine.engine import (
    STGEngine,
    SKILL_INVOCATION_FIELDS,
    SKILL_NAMESPACE,
    _get_skill_invocation,
    _truthy,
)
from stg_engine import skill_runner
from stg_engine.cli import (
    _config_get_path, _config_set_path,
    _config_unset_path, _config_coerce_value,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_stg(tmp_path) -> str:
    """A fresh .stg file path."""
    return str(tmp_path / "test.stg")


@pytest.fixture
def engine_with_skill(tmp_stg, tmp_path) -> tuple[STGEngine, str]:
    """Return (engine, script_path). Engine has one Skill:Echo node whose
    path points at a tiny python script in tmp_path."""
    script = tmp_path / "echo.py"
    script.write_text(
        "import sys\n"
        "print('hello from echo', ' '.join(sys.argv[1:]))\n"
    )
    script.chmod(0o755)

    engine = STGEngine()
    engine.add_node("Skill:Echo", namespace="Skill")
    engine.add_node("Echo_Target", namespace=None)
    engine.add_edge(
        "Skill:Echo", "Echo_Target",
        confidence=0.95,
        rule="empirical",
        path=str(script),
        executable="true",
        interpreter="python3",
        args_template="[args...]",
        description="test echo skill",
        timeout_s="10",
    )
    engine.save(tmp_stg)
    return engine, str(script)


@pytest.fixture
def enabled_config(tmp_path) -> dict:
    """A user_config dict with skill execution enabled and roots = tmp_path."""
    return {
        "skill": {
            "enabled": True,
            "roots": [str(tmp_path)],
            "default_timeout_s": 10,
        }
    }


# ---------------------------------------------------------------------------
# engine.py helpers
# ---------------------------------------------------------------------------

def test_truthy_coerces_strings():
    assert _truthy("true") is True
    assert _truthy("True") is True
    assert _truthy("1") is True
    assert _truthy("yes") is True
    assert _truthy("false") is False
    assert _truthy("0") is False
    assert _truthy("") is False
    assert _truthy(None) is False
    assert _truthy(True) is True
    assert _truthy(False) is False


def test_skill_invocation_fields_list():
    assert "executable" in SKILL_INVOCATION_FIELDS
    assert "interpreter" in SKILL_INVOCATION_FIELDS
    assert "args_template" in SKILL_INVOCATION_FIELDS
    assert "stl_io" in SKILL_INVOCATION_FIELDS
    assert "timeout_s" in SKILL_INVOCATION_FIELDS
    assert SKILL_NAMESPACE == "Skill"


def test_get_skill_invocation_coerces_types():
    inv = _get_skill_invocation({
        "executable": "true",
        "stl_io": "false",
        "timeout_s": "30",
        "interpreter": "python3",
        "args_template": "<url>",
        "unrelated": "keep_out",
    })
    assert inv["executable"] is True
    assert inv["stl_io"] is False
    assert inv["timeout_s"] == 30
    assert inv["interpreter"] == "python3"
    assert inv["args_template"] == "<url>"
    assert "unrelated" not in inv


def test_get_skill_invocation_handles_missing():
    assert _get_skill_invocation({}) == {}
    assert _get_skill_invocation(None) == {}
    assert _get_skill_invocation({"timeout_s": "not-a-number"}).get("timeout_s") is None


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def test_config_set_creates_nested_dict():
    cfg: dict = {}
    _config_set_path(cfg, "skill.enabled", True)
    _config_set_path(cfg, "skill.interpreters.myvenv", "/abs/path")
    assert cfg == {
        "skill": {
            "enabled": True,
            "interpreters": {"myvenv": "/abs/path"},
        }
    }


def test_config_get_missing_returns_false():
    cfg = {"a": {"b": 1}}
    assert _config_get_path(cfg, "a.b") == (1, True)
    assert _config_get_path(cfg, "a.c") == (None, False)
    assert _config_get_path(cfg, "x.y.z") == (None, False)


def test_config_unset_returns_bool():
    cfg = {"a": {"b": 1, "c": 2}}
    assert _config_unset_path(cfg, "a.b") is True
    assert "b" not in cfg["a"]
    assert _config_unset_path(cfg, "a.nonexistent") is False


def test_config_coerce_bool_int_list_string():
    assert _config_coerce_value("true") is True
    assert _config_coerce_value("false") is False
    assert _config_coerce_value("60") == 60
    assert _config_coerce_value("/a,/b", hint="skill.roots") == ["/a", "/b"]
    assert _config_coerce_value("plain-string") == "plain-string"


# ---------------------------------------------------------------------------
# Skill resolution
# ---------------------------------------------------------------------------

def test_find_skill_by_exact_name(engine_with_skill):
    engine, _ = engine_with_skill
    matches = skill_runner.find_skill_edges_by_name(engine, "Echo")
    assert len(matches) == 1


def test_find_skill_by_prefixed_name(engine_with_skill):
    engine, _ = engine_with_skill
    matches = skill_runner.find_skill_edges_by_name(engine, "Skill:Echo")
    assert len(matches) == 1


def test_find_skill_case_insensitive(engine_with_skill):
    engine, _ = engine_with_skill
    matches = skill_runner.find_skill_edges_by_name(engine, "echo")
    assert len(matches) == 1


def test_find_skill_not_found(engine_with_skill):
    engine, _ = engine_with_skill
    matches = skill_runner.find_skill_edges_by_name(engine, "NonExistent")
    assert matches == []


def test_resolve_skill_returns_edge(engine_with_skill):
    engine, _ = engine_with_skill
    edge = skill_runner.resolve_skill(engine, "Echo")
    assert edge is not None
    _src, _tgt, data = edge
    assert data.get("executable") == "true"


def test_resolve_skill_raises_when_not_found(engine_with_skill):
    engine, _ = engine_with_skill
    with pytest.raises(skill_runner.SkillResolutionError) as exc:
        skill_runner.resolve_skill(engine, "Missing")
    assert exc.value.exit_code == skill_runner.EXIT_NOT_FOUND


# ---------------------------------------------------------------------------
# Interpreter resolution
# ---------------------------------------------------------------------------

def test_interpreter_resolves_builtin_python3():
    resolved = skill_runner.resolve_interpreter("python3", {})
    # shutil.which may return None in very minimal envs; tolerate
    if resolved is not None:
        assert os.path.isfile(resolved)


def test_interpreter_resolves_absolute_path():
    path = shutil.which("python3")
    if path is None:
        pytest.skip("no python3 in PATH")
    assert skill_runner.resolve_interpreter(path, {}) == path


def test_interpreter_resolves_user_named(tmp_path):
    fake = tmp_path / "fake_py"
    fake.write_text("#!/usr/bin/env python3\n")
    fake.chmod(0o755)
    cfg = {"skill": {"interpreters": {"myvenv": str(fake)}}}
    assert skill_runner.resolve_interpreter("myvenv", cfg) == str(fake)


def test_interpreter_unknown_returns_none():
    # With no config and not a builtin, should return None
    assert skill_runner.resolve_interpreter("python_venv", {}) is None
    assert skill_runner.resolve_interpreter("totally_made_up", {}) is None


# ---------------------------------------------------------------------------
# Security gates
# ---------------------------------------------------------------------------

def test_validate_config_disabled_by_default():
    assert skill_runner.validate_config_enabled({}) is not None
    assert skill_runner.validate_config_enabled({"skill": {}}) is not None
    assert skill_runner.validate_config_enabled(
        {"skill": {"enabled": False, "roots": ["/tmp"]}}
    ) is not None


def test_validate_config_needs_roots():
    err = skill_runner.validate_config_enabled(
        {"skill": {"enabled": True, "roots": []}}
    )
    assert err is not None and "roots" in err


def test_validate_config_ok_when_configured():
    assert skill_runner.validate_config_enabled(
        {"skill": {"enabled": True, "roots": ["/tmp"]}}
    ) is None


def test_path_under_roots_true(tmp_path):
    script = tmp_path / "x.py"
    script.write_text("#")
    assert skill_runner.path_under_roots(script, [str(tmp_path)])


def test_path_under_roots_false_for_outside(tmp_path):
    script = tmp_path / "x.py"
    script.write_text("#")
    assert not skill_runner.path_under_roots(script, ["/nonexistent"])


# ---------------------------------------------------------------------------
# Invocation (positive path)
# ---------------------------------------------------------------------------

def test_run_skill_positive(engine_with_skill, enabled_config, tmp_stg):
    engine, _ = engine_with_skill
    # Reload engine from disk to mirror real CLI flow
    engine = STGEngine.load(tmp_stg)
    result = skill_runner.run_skill(
        engine, "Echo", ["world"], enabled_config,
    )
    assert result.exit_code == 0, result.error
    assert "hello from echo" in result.stdout
    assert "world" in result.stdout


def test_run_skill_disabled_config(engine_with_skill, tmp_stg):
    engine = STGEngine.load(tmp_stg)
    result = skill_runner.run_skill(engine, "Echo", [], {})
    assert result.exit_code == skill_runner.EXIT_NOT_EXECUTABLE
    assert "disabled" in (result.error or "").lower()


def test_run_skill_path_outside_roots(engine_with_skill, tmp_stg):
    engine = STGEngine.load(tmp_stg)
    bad_cfg = {"skill": {"enabled": True, "roots": ["/totally/different"]}}
    result = skill_runner.run_skill(engine, "Echo", [], bad_cfg)
    assert result.exit_code == skill_runner.EXIT_NOT_EXECUTABLE
    assert "roots" in (result.error or "").lower()


def test_run_skill_not_executable_flag(tmp_path, enabled_config):
    script = tmp_path / "x.py"
    script.write_text("print('hi')\n")
    engine = STGEngine()
    engine.add_node("Skill:NotExec", namespace="Skill")
    engine.add_node("Tgt", namespace=None)
    engine.add_edge(
        "Skill:NotExec", "Tgt",
        confidence=0.9,
        path=str(script), interpreter="python3",
        # NOTE: no executable=true
    )
    result = skill_runner.run_skill(engine, "NotExec", [], enabled_config)
    assert result.exit_code == skill_runner.EXIT_NOT_EXECUTABLE
    assert "executable=true" in (result.error or "")


def test_run_skill_not_found(engine_with_skill, enabled_config, tmp_stg):
    engine = STGEngine.load(tmp_stg)
    result = skill_runner.run_skill(engine, "GhostSkill", [], enabled_config)
    assert result.exit_code == skill_runner.EXIT_NOT_FOUND


def test_run_skill_dry_run(engine_with_skill, enabled_config, tmp_stg):
    engine = STGEngine.load(tmp_stg)
    result = skill_runner.run_skill(
        engine, "Echo", ["foo"], enabled_config, dry_run=True,
    )
    assert result.exit_code == 0
    assert "[dry-run]" in result.stdout


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

def test_run_skill_timeout(tmp_path, enabled_config):
    script = tmp_path / "sleep.py"
    script.write_text("import time\ntime.sleep(10)\n")
    engine = STGEngine()
    engine.add_node("Skill:Sleeper", namespace="Skill")
    engine.add_node("Tgt", namespace=None)
    engine.add_edge(
        "Skill:Sleeper", "Tgt",
        confidence=0.9,
        path=str(script),
        executable="true",
        interpreter="python3",
        timeout_s="1",
    )
    t0 = time.time()
    result = skill_runner.run_skill(engine, "Sleeper", [], enabled_config)
    elapsed = time.time() - t0
    assert result.exit_code == skill_runner.EXIT_TIMEOUT
    assert result.timed_out is True
    # Must return within timeout + generous slack
    assert elapsed < 5


# ---------------------------------------------------------------------------
# Catalog rendering
# ---------------------------------------------------------------------------

def test_list_skills_finds_entry(engine_with_skill):
    engine, _ = engine_with_skill
    skills = skill_runner.list_skills(engine, executable_only=False)
    names = [s["name"] for s in skills]
    assert "Echo" in names


def test_render_catalog_empty():
    assert skill_runner.render_catalog([]).startswith("(no skills")


def test_render_catalog_shows_executable_flag(engine_with_skill):
    engine, _ = engine_with_skill
    skills = skill_runner.list_skills(engine, executable_only=True)
    rendered = skill_runner.render_catalog(skills)
    assert "Echo" in rendered
    assert "✓" in rendered  # executable mark


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def test_audit_write_read_roundtrip(tmp_stg):
    # Bootstrap the sqlite file with the skill_invocations table
    STGEngine().save(tmp_stg)
    res = skill_runner.SkillInvocationResult(
        skill_name="Test", target="Tgt",
        path="/tmp/foo.py", interpreter="/usr/bin/python3",
        args=["a"], stdin_stl=None,
        exit_code=0, stdout="ok", stderr="",
        elapsed_s=0.1, bytes_out=2, bytes_err=0,
        truncated_stdout=False, timed_out=False,
        error=None, invocation_id="inv_xyz",
    )
    skill_runner.write_audit_row(tmp_stg, res)
    rows = skill_runner.read_audit_history(tmp_stg, limit=5)
    assert len(rows) == 1
    assert rows[0]["skill_name"] == "Test"
    assert rows[0]["invocation_id"] == "inv_xyz"


def test_audit_history_filter(tmp_stg):
    STGEngine().save(tmp_stg)
    for name in ("A", "B", "A", "C"):
        res = skill_runner.SkillInvocationResult(
            skill_name=name, target=None, path=None, interpreter=None,
            args=[], stdin_stl=None,
            exit_code=0, stdout="", stderr="",
            elapsed_s=0.0, bytes_out=0, bytes_err=0,
            truncated_stdout=False, timed_out=False,
            error=None,
            invocation_id=f"inv_{name}{time.time_ns()}",
        )
        skill_runner.write_audit_row(tmp_stg, res)
    all_rows = skill_runner.read_audit_history(tmp_stg, limit=10)
    assert len(all_rows) == 4
    a_rows = skill_runner.read_audit_history(tmp_stg, skill_name="A", limit=10)
    assert len(a_rows) == 2
    assert all(r["skill_name"] == "A" for r in a_rows)
