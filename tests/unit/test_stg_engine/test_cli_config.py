"""Tests for ~/.stg/config.json user-level config + agent priority chain.

Priority (highest to lowest):
  1. --agent flag
  2. STG_AGENT env var
  3. ~/.stg/config.json default_agent
  4. hardcoded "default"
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest


def _run_load_settings(monkeypatch, tmp_path, env=None, argv_extra=None, config=None):
    """Run cli._load_settings() with isolated ~/.stg (tmp_path as HOME)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    # Rebuild module paths to point at tmp HOME
    stg_root = tmp_path / ".stg"
    stg_root.mkdir(exist_ok=True)
    if config is not None:
        (stg_root / "config.json").write_text(json.dumps(config))
    # Clear STG_AGENT unless env supplies it
    monkeypatch.delenv("STG_AGENT", raising=False)
    if env:
        for k, v in env.items():
            monkeypatch.setenv(k, v)

    # Manipulate sys.argv
    saved_argv = sys.argv
    sys.argv = ["stg"] + (argv_extra or [])

    # Reload cli module with fresh _STG_ROOT / _USER_CONFIG_PATH paths
    # Patch module-level constants instead of reimporting (importing has side effects).
    from stg_engine import cli as cli_mod
    monkeypatch.setattr(cli_mod, "_STG_ROOT", str(stg_root))
    monkeypatch.setattr(cli_mod, "_USER_CONFIG_PATH", str(stg_root / "config.json"))

    try:
        return cli_mod._load_settings()
    finally:
        sys.argv = saved_argv


class TestAgentPriorityChain:

    def test_fallback_to_default(self, tmp_path, monkeypatch):
        """No flag, no env, no config → 'default' agent."""
        s = _run_load_settings(monkeypatch, tmp_path)
        assert s["agent"] == "default"

    def test_config_file_sets_default(self, tmp_path, monkeypatch):
        """config.json default_agent overrides hardcoded fallback."""
        s = _run_load_settings(
            monkeypatch, tmp_path, config={"default_agent": "alice"},
        )
        assert s["agent"] == "alice"

    def test_env_var_overrides_config(self, tmp_path, monkeypatch):
        """STG_AGENT env beats config file."""
        s = _run_load_settings(
            monkeypatch, tmp_path,
            env={"STG_AGENT": "bob"},
            config={"default_agent": "alice"},
        )
        assert s["agent"] == "bob"

    def test_flag_overrides_env(self, tmp_path, monkeypatch):
        """--agent flag beats STG_AGENT env."""
        s = _run_load_settings(
            monkeypatch, tmp_path,
            env={"STG_AGENT": "bob"},
            argv_extra=["--agent", "carol"],
        )
        assert s["agent"] == "carol"

    def test_flag_overrides_config(self, tmp_path, monkeypatch):
        """--agent flag beats config.json too."""
        s = _run_load_settings(
            monkeypatch, tmp_path,
            config={"default_agent": "alice"},
            argv_extra=["--agent", "carol"],
        )
        assert s["agent"] == "carol"

    def test_malformed_config_gracefully_falls_back(self, tmp_path, monkeypatch):
        """Bad JSON in config file must not crash — warn + fallback."""
        stg_root = tmp_path / ".stg"
        stg_root.mkdir(exist_ok=True)
        (stg_root / "config.json").write_text("not json {")

        from stg_engine import cli as cli_mod
        monkeypatch.setattr(cli_mod, "_STG_ROOT", str(stg_root))
        monkeypatch.setattr(cli_mod, "_USER_CONFIG_PATH", str(stg_root / "config.json"))
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("STG_AGENT", raising=False)
        saved_argv = sys.argv
        sys.argv = ["stg"]
        try:
            s = cli_mod._load_settings()
        finally:
            sys.argv = saved_argv
        assert s["agent"] == "default"  # fell back cleanly

    def test_empty_string_config_agent_falls_back(self, tmp_path, monkeypatch):
        """default_agent = '' should fall back, not resolve to empty."""
        s = _run_load_settings(
            monkeypatch, tmp_path, config={"default_agent": "  "},
        )
        assert s["agent"] == "default"


class TestConfigReadWrite:

    def test_read_missing_file_returns_empty_dict(self, tmp_path, monkeypatch):
        from stg_engine import cli as cli_mod
        cfg_path = tmp_path / "does_not_exist.json"
        monkeypatch.setattr(cli_mod, "_USER_CONFIG_PATH", str(cfg_path))
        assert cli_mod._read_user_config() == {}

    def test_write_creates_parent_dir(self, tmp_path, monkeypatch):
        from stg_engine import cli as cli_mod
        nested = tmp_path / ".stg"
        cfg_path = nested / "config.json"
        monkeypatch.setattr(cli_mod, "_STG_ROOT", str(nested))
        monkeypatch.setattr(cli_mod, "_USER_CONFIG_PATH", str(cfg_path))
        cli_mod._write_user_config({"default_agent": "xyz"})
        assert cfg_path.exists()
        loaded = json.loads(cfg_path.read_text())
        assert loaded == {"default_agent": "xyz"}

    def test_roundtrip_preserves_keys(self, tmp_path, monkeypatch):
        from stg_engine import cli as cli_mod
        cfg_path = tmp_path / "config.json"
        monkeypatch.setattr(cli_mod, "_STG_ROOT", str(tmp_path))
        monkeypatch.setattr(cli_mod, "_USER_CONFIG_PATH", str(cfg_path))
        cli_mod._write_user_config({"a": "1", "b": "2"})
        assert cli_mod._read_user_config() == {"a": "1", "b": "2"}
