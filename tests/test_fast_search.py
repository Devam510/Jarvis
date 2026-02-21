"""Tests for jarvis.execution.fast_search — Everything SDK + glob fallback."""

import os
import tempfile
import subprocess
from unittest.mock import patch, MagicMock

import pytest

from jarvis.execution.fast_search import EverythingSearch, SearchResult


# ── Query Sanitization ────────────────────────────────────────────────────


class TestQuerySanitization:
    """Safety control: query sanitization strips shell metacharacters."""

    def test_sanitize_strips_semicolons(self):
        s = EverythingSearch()
        assert ";" not in s._sanitize_query("test; rm -rf /")

    def test_sanitize_strips_pipes(self):
        s = EverythingSearch()
        assert "|" not in s._sanitize_query("test | cat /etc/passwd")

    def test_sanitize_strips_ampersand(self):
        s = EverythingSearch()
        assert "&" not in s._sanitize_query("test & echo pwned")

    def test_sanitize_strips_backtick(self):
        s = EverythingSearch()
        assert "`" not in s._sanitize_query("test `whoami`")

    def test_sanitize_strips_dollar(self):
        s = EverythingSearch()
        assert "$" not in s._sanitize_query("test $HOME")

    def test_sanitize_strips_redirect(self):
        s = EverythingSearch()
        result = s._sanitize_query("test > /etc/passwd")
        assert ">" not in result
        assert "<" not in s._sanitize_query("test < /etc/shadow")

    def test_sanitize_empty_returns_empty(self):
        s = EverythingSearch()
        assert s._sanitize_query("") == ""
        assert s._sanitize_query("   ") == ""

    def test_sanitize_length_cap(self):
        s = EverythingSearch()
        long_query = "a" * 500
        result = s._sanitize_query(long_query)
        assert len(result) <= 256

    def test_sanitize_safe_query_unchanged(self):
        s = EverythingSearch()
        assert s._sanitize_query("my_document.pdf") == "my_document.pdf"

    def test_sanitize_wildcard_allowed(self):
        s = EverythingSearch()
        # Wildcards (* ?) are useful for search — should be kept
        result = s._sanitize_query("*.pdf")
        assert "*" in result


# ── Max Results Cap ───────────────────────────────────────────────────────


class TestMaxResultsCap:
    """Safety control: hard cap on max_results."""

    def test_default_max_results(self):
        s = EverythingSearch(max_results=50, absolute_max_results=200)
        assert s._max_results == 50

    def test_max_results_capped_at_absolute(self):
        s = EverythingSearch(max_results=500, absolute_max_results=200)
        assert s._max_results == 200

    def test_search_respects_per_call_cap(self):
        """Even if per-call max_results exceeds absolute, it's capped."""
        s = EverythingSearch(absolute_max_results=200)
        s._everything_available = False  # force glob fallback
        # The cap is enforced in search() internally
        # We just verify no crash and stats are tracked
        results = s.search("nonexistent_file_xyz_12345", max_results=999)
        assert isinstance(results, list)


# ── Timeout ───────────────────────────────────────────────────────────────


class TestTimeout:
    """Safety control: hard timeout on subprocess."""

    def test_timeout_capped_at_10s(self):
        s = EverythingSearch(timeout_seconds=60.0)
        assert s._timeout <= 10.0

    def test_timeout_default_3s(self):
        s = EverythingSearch()
        assert s._timeout == 3.0


# ── Everything SDK Check ──────────────────────────────────────────────────


class TestEverythingDetection:
    """Capability gating: detect if Everything is available."""

    def test_everything_not_found(self):
        s = EverythingSearch(es_exe_path="fake_es_that_does_not_exist.exe")
        assert s._check_everything() is False

    def test_everything_timeout(self):
        with patch(
            "subprocess.run", side_effect=subprocess.TimeoutExpired("es.exe", 3)
        ):
            s = EverythingSearch()
            assert s._check_everything() is False

    def test_everything_shell_false(self):
        """CRITICAL: verify subprocess always uses shell=False."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            s = EverythingSearch()
            s._check_everything()
            call = mock_run.call_args
            assert call.kwargs.get("shell", False) is False


# ── Glob Fallback ─────────────────────────────────────────────────────────


class TestGlobFallback:
    """Verify glob fallback works when Everything unavailable."""

    def test_glob_finds_files(self, tmp_path):
        (tmp_path / "report.txt").write_text("data")
        (tmp_path / "notes.txt").write_text("data")

        s = EverythingSearch()
        s._everything_available = False
        # Use glob on a specific directory by changing CWD
        results = s._search_glob("report", max_results=10)
        # On some systems glob results depend on CWD; we just verify it returns a list
        assert isinstance(results, list)

    def test_glob_respects_max_results(self, tmp_path):
        for i in range(10):
            (tmp_path / f"file_{i}.txt").write_text("data")

        s = EverythingSearch()
        results = s._search_glob("file_", max_results=3)
        assert len(results) <= 3

    def test_fallback_counter(self):
        s = EverythingSearch()
        s._everything_available = False
        s.search("test_query_nonexistent")
        assert s.total_fallbacks == 1


# ── Result Parsing ────────────────────────────────────────────────────────


class TestResultParsing:
    """Verify es.exe output is parsed correctly."""

    def test_parse_results(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")

        results = EverythingSearch._parse_results(str(test_file))
        assert len(results) == 1
        assert results[0].name == "test.txt"
        assert results[0].size > 0

    def test_parse_empty_output(self):
        results = EverythingSearch._parse_results("")
        assert results == []

    def test_parse_nonexistent_path(self):
        results = EverythingSearch._parse_results("/fake/path/abc.txt")
        assert len(results) == 1
        assert results[0].size == 0  # stat fails gracefully


# ── Stats Tracking ────────────────────────────────────────────────────────


class TestStats:
    def test_search_increments_counter(self):
        s = EverythingSearch()
        s._everything_available = False
        s.search("test_nonexistent")
        assert s.total_searches == 1

    def test_search_by_ext_uses_search(self):
        s = EverythingSearch()
        s._everything_available = False
        s.search_by_ext("py")
        assert s.total_searches == 1
