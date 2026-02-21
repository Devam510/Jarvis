"""Tests for jarvis.plugins.plugin_manager — Dynamic plugin system."""

import json
import pytest

from jarvis.plugins.plugin_manager import PluginManager, PluginManifest, LoadedPlugin


class TestDiscovery:
    def test_discover_empty_dir(self, tmp_path):
        pm = PluginManager(plugins_dir=str(tmp_path))
        assert pm.discover() == []

    def test_discover_no_dir(self):
        pm = PluginManager(plugins_dir="/fake/nonexistent")
        assert pm.discover() == []

    def test_discover_valid_plugin(self, tmp_path):
        plugin_dir = tmp_path / "my_plugin"
        plugin_dir.mkdir()
        manifest = {
            "name": "my_plugin",
            "version": "1.0.0",
            "description": "Test plugin",
            "entry_point": "main.py",
            "risk_tier": 1,
            "tools": ["my_tool"],
        }
        (plugin_dir / "plugin.json").write_text(json.dumps(manifest))

        pm = PluginManager(plugins_dir=str(tmp_path))
        results = pm.discover()
        assert len(results) == 1
        assert results[0].name == "my_plugin"
        assert results[0].risk_tier == 1
        assert "my_tool" in results[0].tools

    def test_discover_bad_json(self, tmp_path):
        plugin_dir = tmp_path / "bad_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.json").write_text("not valid json")

        pm = PluginManager(plugins_dir=str(tmp_path))
        results = pm.discover()
        assert len(results) == 0
        assert pm.total_errors == 1


class TestLoading:
    def test_load_valid_plugin(self, tmp_path):
        plugin_dir = tmp_path / "hello_plugin"
        plugin_dir.mkdir()
        manifest = {
            "name": "hello_plugin",
            "version": "1.0",
            "entry_point": "main.py",
            "tools": ["greet"],
        }
        (plugin_dir / "plugin.json").write_text(json.dumps(manifest))
        (plugin_dir / "main.py").write_text("def greet(): return 'hello'")

        pm = PluginManager(plugins_dir=str(tmp_path))
        result = pm.load("hello_plugin")
        assert result is not None
        assert result.active
        assert hasattr(result.module, "greet")
        assert pm.total_loaded == 1

    def test_load_nonexistent_plugin(self, tmp_path):
        pm = PluginManager(plugins_dir=str(tmp_path))
        result = pm.load("no_such_plugin")
        assert result is None

    def test_load_duplicate(self, tmp_path):
        plugin_dir = tmp_path / "dup_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.json").write_text(
            json.dumps({"name": "dup_plugin", "entry_point": "main.py"})
        )
        (plugin_dir / "main.py").write_text("x = 1")

        pm = PluginManager(plugins_dir=str(tmp_path))
        pm.load("dup_plugin")
        result = pm.load("dup_plugin")  # second load
        assert result is not None
        assert pm.total_loaded == 1  # only loaded once

    def test_load_missing_entry_point(self, tmp_path):
        plugin_dir = tmp_path / "no_entry"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.json").write_text(
            json.dumps({"name": "no_entry", "entry_point": "missing.py"})
        )

        pm = PluginManager(plugins_dir=str(tmp_path))
        result = pm.load("no_entry")
        assert result is None


class TestUnloading:
    def test_unload_loaded_plugin(self, tmp_path):
        plugin_dir = tmp_path / "unload_test"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.json").write_text(
            json.dumps({"name": "unload_test", "entry_point": "main.py"})
        )
        (plugin_dir / "main.py").write_text("def cleanup(): pass")

        pm = PluginManager(plugins_dir=str(tmp_path))
        pm.load("unload_test")
        assert pm.unload("unload_test")
        assert pm.total_unloaded == 1
        assert "unload_test" not in pm.list_loaded()

    def test_unload_not_loaded(self):
        pm = PluginManager()
        assert pm.unload("nonexistent") is False


class TestTools:
    def test_get_tools(self, tmp_path):
        plugin_dir = tmp_path / "tool_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.json").write_text(
            json.dumps(
                {"name": "tool_plugin", "entry_point": "main.py", "tools": ["a", "b"]}
            )
        )
        (plugin_dir / "main.py").write_text("x = 1")

        pm = PluginManager(plugins_dir=str(tmp_path))
        pm.load("tool_plugin")
        tools = pm.get_tools()
        assert "a" in tools
        assert "b" in tools
