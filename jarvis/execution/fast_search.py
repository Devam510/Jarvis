"""
jarvis.execution.fast_search — Everything SDK fast file search with safety.

Safety controls:
  - subprocess.run(shell=False) only — no shell injection
  - Hard timeout: ≤3s (configurable, capped)
  - Query sanitized: strip shell metacharacters
  - max_results hard cap (absolute_max_results = 200)
  - Falls back to glob/os.walk if Everything not installed
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Characters that could be used for shell injection
_SHELL_METACHARACTERS = re.compile(r'[;|&$`><\'"\\!{}()\[\]]')
_MAX_QUERY_LENGTH = 256


@dataclass
class SearchResult:
    """A single file search result."""

    path: str
    name: str
    size: int = 0
    modified: float = 0.0


class EverythingSearch:
    """Fast file search using Everything SDK (es.exe) with glob fallback.

    Safety:
      - All queries sanitized before execution
      - subprocess.run(shell=False) only
      - Hard timeout cap (default 3s)
      - max_results capped at absolute_max_results
    """

    def __init__(
        self,
        es_exe_path: str = "es.exe",
        timeout_seconds: float = 3.0,
        max_results: int = 50,
        absolute_max_results: int = 200,
    ):
        self._es_exe = es_exe_path
        self._timeout = min(timeout_seconds, 10.0)  # cap at 10s
        self._max_results = min(max_results, absolute_max_results)
        self._absolute_max = absolute_max_results
        self._everything_available: Optional[bool] = None

        # Stats
        self.total_searches = 0
        self.total_results = 0
        self.total_fallbacks = 0

    # ── Public API ────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
    ) -> list[SearchResult]:
        """Search for files matching query.

        Uses Everything SDK if available, else falls back to glob.
        """
        query = self._sanitize_query(query)
        if not query:
            return []

        cap = min(max_results or self._max_results, self._absolute_max)
        self.total_searches += 1

        if self._everything_available is None:
            self._everything_available = self._check_everything()

        if self._everything_available:
            results = self._search_everything(query, cap)
            if results is not None:
                self.total_results += len(results)
                return results

        # Fallback to glob
        self.total_fallbacks += 1
        results = self._search_glob(query, cap)
        self.total_results += len(results)
        return results

    def search_by_ext(
        self, ext: str, directory: str = "", max_results: Optional[int] = None
    ) -> list[SearchResult]:
        """Search for files by extension."""
        ext = ext.lstrip(".")
        if directory:
            query = f"{directory} *.{ext}"
        else:
            query = f"*.{ext}"
        return self.search(query, max_results)

    def search_regex(
        self, pattern: str, max_results: Optional[int] = None
    ) -> list[SearchResult]:
        """Search using regex pattern (Everything regex mode)."""
        pattern = self._sanitize_query(pattern)
        if not pattern:
            return []

        cap = min(max_results or self._max_results, self._absolute_max)
        self.total_searches += 1

        if self._everything_available is None:
            self._everything_available = self._check_everything()

        if self._everything_available:
            results = self._search_everything_regex(pattern, cap)
            if results is not None:
                self.total_results += len(results)
                return results

        return []  # No regex fallback for glob

    # ── Everything SDK ────────────────────────────────────────────────────

    def _check_everything(self) -> bool:
        """Check if es.exe is available."""
        try:
            result = subprocess.run(
                [self._es_exe, "-get-result-count", "jarvis_probe_query"],
                capture_output=True,
                timeout=3,
                shell=False,  # SAFETY: never use shell=True
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            logger.info("Everything SDK not available — using glob fallback")
            return False

    def _search_everything(
        self, query: str, max_results: int
    ) -> Optional[list[SearchResult]]:
        """Execute search via es.exe."""
        try:
            cmd = [
                self._es_exe,
                "-n",
                str(max_results),
                "-s",  # sort by path
                query,
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                shell=False,  # SAFETY: never shell=True
            )
            if result.returncode != 0:
                return None

            return self._parse_results(result.stdout)
        except subprocess.TimeoutExpired:
            logger.warning("Everything search timed out for: %s", query[:50])
            return None
        except Exception as e:
            logger.error("Everything search failed: %s", e)
            return None

    def _search_everything_regex(
        self, pattern: str, max_results: int
    ) -> Optional[list[SearchResult]]:
        """Execute regex search via es.exe."""
        try:
            cmd = [
                self._es_exe,
                "-n",
                str(max_results),
                "-regex",
                pattern,
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                shell=False,
            )
            if result.returncode != 0:
                return None
            return self._parse_results(result.stdout)
        except (subprocess.TimeoutExpired, Exception) as e:
            logger.warning("Everything regex search failed: %s", e)
            return None

    @staticmethod
    def _parse_results(stdout: str) -> list[SearchResult]:
        """Parse es.exe output into SearchResult objects."""
        results = []
        for line in stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            try:
                stat = p.stat() if p.exists() else None
                results.append(
                    SearchResult(
                        path=str(p),
                        name=p.name,
                        size=stat.st_size if stat else 0,
                        modified=stat.st_mtime if stat else 0,
                    )
                )
            except OSError:
                results.append(SearchResult(path=str(p), name=p.name))
        return results

    # ── Glob Fallback ─────────────────────────────────────────────────────

    def _search_glob(self, query: str, max_results: int) -> list[SearchResult]:
        """Fallback search using glob (slower, local only)."""
        import glob as glob_mod

        results = []

        # Try glob patterns
        patterns = [
            f"**/*{query}*",
            f"**/{query}",
        ]

        searched = set()
        for pattern in patterns:
            try:
                for match in glob_mod.iglob(pattern, recursive=True):
                    if match in searched:
                        continue
                    searched.add(match)
                    p = Path(match)
                    try:
                        stat = p.stat()
                        results.append(
                            SearchResult(
                                path=str(p),
                                name=p.name,
                                size=stat.st_size,
                                modified=stat.st_mtime,
                            )
                        )
                    except OSError:
                        results.append(SearchResult(path=str(p), name=p.name))

                    if len(results) >= max_results:
                        return results
            except Exception:
                continue

        return results

    # ── Safety ────────────────────────────────────────────────────────────

    @staticmethod
    def _sanitize_query(query: str) -> str:
        """Sanitize search query to prevent injection.

        Strips shell metacharacters and enforces length limit.
        """
        if not query or not query.strip():
            return ""

        # Truncate to max length
        query = query[:_MAX_QUERY_LENGTH]

        # Strip dangerous characters
        query = _SHELL_METACHARACTERS.sub("", query)

        return query.strip()
