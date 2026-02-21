"""
jarvis.execution.executor — Execution dispatcher and individual tool executors.

Handles: file operations, content search, system info, code sandbox, browser automation.
All executors validate paths and operate within allowed roots.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from jarvis.core.event_bus import AsyncEventBus
from jarvis.utils.config import ExecutionConfig
from jarvis.utils.types import PlannedAction, StepResult

logger = logging.getLogger(__name__)


class PathViolationError(Exception):
    """Raised when a path is outside allowed roots."""

    pass


class PathValidator:
    """Validates file paths using blacklist + optional whitelist.

    Security layers (checked in order):
      1. UNC / \\\\?\\ prefix normalization
      2. Path.resolve(strict=False) for canonical form
      3. Symlink escape detection (resolved path must not land in denied)
      4. Path traversal check (literal '..' in input)
      5. Drive-level deny
      6. System directory deny (hardcoded)
      7. User-configured deny list
      8. Whitelist check (only if unrestricted_mode=False)
    """

    # Hardcoded system directories — always denied
    SYSTEM_DIRS = [
        "C:/Windows",
        "C:/Windows/System32",
        "C:/Windows/System32/config",
        "C:/Program Files",
        "C:/Program Files (x86)",
        "C:/ProgramData/Microsoft",
    ]

    def __init__(
        self,
        allowed_roots: list[str],
        denied_paths: list[str] | None = None,
        denied_drives: list[str] | None = None,
        unrestricted_mode: bool = False,
    ):
        self.allowed_roots = [Path(r).resolve() for r in allowed_roots]
        self.unrestricted_mode = unrestricted_mode

        # Build deny list: system dirs + user-configured
        self.denied_paths = [Path(d).resolve() for d in self.SYSTEM_DIRS]
        if denied_paths:
            self.denied_paths.extend(Path(d).resolve() for d in denied_paths)

        # Denied drives normalized to uppercase with colon
        self.denied_drives: set[str] = set()
        if denied_drives:
            for drv in denied_drives:
                self.denied_drives.add(drv.upper().rstrip(":\\") + ":")

        # Always allow user home in whitelist mode
        if not unrestricted_mode:
            home = Path.home().resolve()
            if home not in self.allowed_roots:
                self.allowed_roots.append(home)

    @staticmethod
    def _normalize_path(path_str: str) -> str:
        """Strip UNC / \\\\?\\ prefixes and normalize separators."""
        s = path_str.replace("\\", "/")
        # Strip \\\\?\\ prefix (long path support on Windows)
        if s.startswith("//?/"):
            s = s[4:]
        # Strip UNC prefix \\\\server\\share → deny (we don't allow network)
        if s.startswith("//"):
            raise PathViolationError(f"UNC/network paths denied: {path_str}")
        return s

    def validate(self, path_str: str) -> Path:
        """Resolve and validate a path. Raises PathViolationError if invalid."""
        # 1. Normalize UNC / special prefixes
        try:
            normalized = self._normalize_path(path_str)
        except PathViolationError:
            raise
        except Exception as e:
            raise PathViolationError(f"Invalid path: {path_str}") from e

        # 2. Resolve to canonical absolute path
        try:
            p = Path(normalized).resolve(strict=False)
        except (ValueError, OSError) as e:
            raise PathViolationError(f"Invalid path: {path_str}") from e

        # 3. Path traversal check (literal '..' in raw input)
        if ".." in Path(normalized).parts:
            raise PathViolationError(f"Path traversal detected: {path_str}")

        # 4. Drive-level deny
        if self.denied_drives:
            drive = p.drive.upper()
            if drive and drive in self.denied_drives:
                raise PathViolationError(
                    f"Access to drive {drive} is denied: {path_str}"
                )

        # 5. System / user deny list (catches symlink escapes too,
        #    because we check the RESOLVED path)
        #    Uses os.path.normcase for case-insensitive comparison on Windows
        p_norm = os.path.normcase(str(p))
        for denied in self.denied_paths:
            denied_norm = os.path.normcase(str(denied))
            if p_norm == denied_norm or p_norm.startswith(denied_norm + os.sep):
                raise PathViolationError(
                    f"System/denied path access blocked: {path_str}"
                )

        # 6. Symlink escape guard: if path is a symlink, resolve target
        #    and re-check against deny list
        try:
            if p.is_symlink():
                real_target = p.resolve(strict=True)
                real_norm = os.path.normcase(str(real_target))
                for denied in self.denied_paths:
                    denied_norm = os.path.normcase(str(denied))
                    if real_norm == denied_norm or real_norm.startswith(
                        denied_norm + os.sep
                    ):
                        raise PathViolationError(
                            f"Symlink escape to denied path: {path_str} → {real_target}"
                        )
        except OSError:
            pass  # broken symlink is fine — will fail at I/O

        # 7. Whitelist check (only in restricted mode)
        if not self.unrestricted_mode:
            for root in self.allowed_roots:
                try:
                    p.relative_to(root)
                    return p
                except ValueError:
                    continue
            raise PathViolationError(
                f"Path {path_str} is not under any allowed root: "
                f"{[str(r) for r in self.allowed_roots]}"
            )

        return p


class ExecutionDispatcher:
    """
    Routes validated tool calls to the appropriate executor.
    Emits step_result events after each execution.

    Safety features:
      - Per-tool timeout: prevents any tool from blocking the voice loop
      - Circuit breaker: disables tools after N consecutive failures
    """

    def __init__(self, config: ExecutionConfig, event_bus: AsyncEventBus):
        self.config = config
        self.event_bus = event_bus
        self.path_validator = PathValidator(
            allowed_roots=config.allowed_roots,
            denied_paths=config.denied_paths,
            denied_drives=config.denied_drives,
            unrestricted_mode=config.unrestricted_mode,
        )
        self._default_timeout = config.tool_timeout_seconds
        self._cb_threshold = config.circuit_breaker_threshold
        self._cb_failures: dict[str, int] = {}  # tool → consecutive failure count
        self._cb_disabled: dict[str, float] = {}  # tool → disabled-until timestamp
        self._CB_COOLDOWN = 60.0  # seconds a tripped tool stays disabled

        # Phase 6: Guardrail engine (injected after construction)
        self._guardrail = None

        # Optional components (injected after construction)
        self._messaging = None
        self._safe_executor = None
        self._system_monitor = None
        self._fast_search = None  # Future: EverythingSearch
        self._copilot = None  # Future: CodeCoPilot

        # BUG-07 FIX: Persistent browser context (lazy-init, reuse, shutdown-close)
        self._browser_context = None  # Playwright BrowserContext
        self._playwright = None  # Playwright instance
        self._reminder_tasks: set[asyncio.Task] = set()  # BUG-08 FIX: track reminders

        self._executors = {
            "file_read": self._exec_file_read,
            "file_write": self._exec_file_write,
            "file_delete": self._exec_file_delete,
            "file_move": self._exec_file_move,
            "file_copy": self._exec_file_copy,
            "search_files": self._exec_search_files,
            "search_content": self._exec_search_content,
            "system_info": self._exec_system_info,
            "execute_code": self._exec_code,
            "browser_navigate": self._exec_browser_navigate,
            "browser_interact": self._exec_browser_interact,
            "set_reminder": self._exec_reminder,
            "store_memory": self._exec_store_memory,
            "app_launch": self._exec_app_launch,
            "get_time": self._exec_get_time,
            # Phase 2 tools
            "send_message": self._exec_send_message,
            "read_messages": self._exec_read_messages,
            "run_script": self._exec_run_script,
            "install_package": self._exec_install_package,
            "system_query": self._exec_system_query,
            # Future modules
            "fast_file_search": self._exec_fast_search,
            "code_copilot": self._exec_code_copilot,
        }

        # Per-tool timeout overrides (seconds)
        self._tool_timeouts: dict[str, int] = {
            "browser_navigate": 45,
            "browser_interact": 20,
            "execute_code": config.sandbox_timeout_seconds,
            "send_message": 60,
            "read_messages": 45,
            "run_script": config.sandbox_timeout_seconds,
            "install_package": 120,
            "fast_file_search": 5,
            "code_copilot": 60,
        }

    async def execute_tool_call(self, event: dict):
        """Execute a single authorized tool call with timeout + circuit breaker."""
        action: PlannedAction = event.get("action")
        correlation_id = event.get("correlation_id", "")
        start = time.time()

        if not action:
            return

        tool_name = action.tool_name

        # ── Phase 6: Guardrail pre-check ──────────────────────────────
        if self._guardrail:
            try:
                self._guardrail.check(tool_name, action.arguments)
            except Exception as gv:
                result = StepResult(
                    status="failed",
                    error=f"Guardrail blocked: {gv}",
                )
                result.duration_ms = (time.time() - start) * 1000
                logger.warning("Guardrail blocked %s: %s", tool_name, gv)
                await self.event_bus.emit(
                    "execution.step_result",
                    {
                        "action": tool_name,
                        "result": result,
                        "correlation_id": correlation_id,
                    },
                )
                return

        # ── Circuit breaker check ─────────────────────────────────────
        disabled_until = self._cb_disabled.get(tool_name, 0)
        if disabled_until > time.time():
            result = StepResult(
                status="failed",
                error=f"Tool '{tool_name}' is temporarily disabled (circuit breaker). "
                f"Retry in {int(disabled_until - time.time())}s.",
            )
        else:
            executor = self._executors.get(tool_name)
            if not executor:
                result = StepResult(
                    status="failed",
                    error=f"Unknown tool: {tool_name}",
                )
            else:
                timeout = self._tool_timeouts.get(tool_name, self._default_timeout)
                try:
                    result = await asyncio.wait_for(
                        executor(action.arguments), timeout=timeout
                    )
                except asyncio.TimeoutError:
                    result = StepResult(
                        status="timeout",
                        error=f"Tool '{tool_name}' exceeded {timeout}s timeout",
                    )
                except PathViolationError as e:
                    result = StepResult(status="failed", error=f"Path violation: {e}")
                except Exception as e:
                    logger.exception("Executor error for %s", tool_name)
                    result = StepResult(status="failed", error=str(e))

            # ── Circuit breaker bookkeeping ───────────────────────────
            if result.status in ("failed", "timeout"):
                self._cb_failures[tool_name] = self._cb_failures.get(tool_name, 0) + 1
                if self._cb_failures[tool_name] >= self._cb_threshold:
                    self._cb_disabled[tool_name] = time.time() + self._CB_COOLDOWN
                    logger.warning(
                        "Circuit breaker TRIPPED for '%s' — disabled for %ds",
                        tool_name,
                        self._CB_COOLDOWN,
                    )
            else:
                self._cb_failures[tool_name] = 0  # reset on success

        result.duration_ms = (time.time() - start) * 1000
        logger.info(
            "Execution %s: %s (%.0fms)",
            tool_name,
            result.status,
            result.duration_ms,
        )

        await self.event_bus.emit(
            "execution.step_result",
            {
                "action": tool_name,
                "result": result,
                "correlation_id": correlation_id,
            },
        )

    # ── File Operations ──────────────────────────────────────────────────

    async def _exec_file_read(self, args: dict) -> StepResult:
        path = self.path_validator.validate(args["path"])
        if not path.exists():
            return StepResult(status="failed", error=f"File not found: {path}")
        if path.stat().st_size > self.config.max_file_size_mb * 1024 * 1024:
            return StepResult(status="failed", error="File too large")

        content = await asyncio.get_event_loop().run_in_executor(
            None, lambda: path.read_text(encoding="utf-8", errors="replace")
        )
        return StepResult(status="success", output=content)

    async def _exec_file_write(self, args: dict) -> StepResult:
        path = self.path_validator.validate(args["path"])
        content = args.get("content", "")

        path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: path.write_text(content, encoding="utf-8")
        )
        return StepResult(
            status="success", output=f"Written {len(content)} bytes to {path}"
        )

    async def _exec_file_delete(self, args: dict) -> StepResult:
        path = self.path_validator.validate(args["path"])
        if not path.exists():
            return StepResult(status="failed", error=f"Path not found: {path}")

        try:
            from send2trash import send2trash

            await asyncio.get_event_loop().run_in_executor(None, send2trash, str(path))
            return StepResult(status="success", output=f"Moved to recycle bin: {path}")
        except ImportError:
            return StepResult(
                status="failed",
                error="send2trash not installed — refusing to permanently delete",
            )

    async def _exec_file_move(self, args: dict) -> StepResult:
        src = self.path_validator.validate(args["source"])
        dest = self.path_validator.validate(args["destination"])
        if not src.exists():
            return StepResult(status="failed", error=f"Source not found: {src}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: shutil.move(str(src), str(dest))
        )
        return StepResult(status="success", output=f"Moved {src} → {dest}")

    async def _exec_file_copy(self, args: dict) -> StepResult:
        src = self.path_validator.validate(args["source"])
        dest = self.path_validator.validate(args["destination"])
        if not src.exists():
            return StepResult(status="failed", error=f"Source not found: {src}")
        dest.parent.mkdir(parents=True, exist_ok=True)

        if src.is_dir():
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: shutil.copytree(str(src), str(dest))
            )
        else:
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: shutil.copy2(str(src), str(dest))
            )
        return StepResult(status="success", output=f"Copied {src} → {dest}")

    # ── Search Operations ────────────────────────────────────────────────

    async def _exec_search_files(self, args: dict) -> StepResult:
        # ── Normalize LLM argument variants ──────────────────────────
        # LLM may send: {path: [dir, pattern]}, {path: dir}, {folder: dir}, etc.
        raw_dir = (
            args.get("directory")
            or args.get("path")
            or args.get("dir")
            or args.get("folder")
            or args.get("location")
        )
        pattern = args.get("pattern", "*")

        # Handle list-type path: ['D:', '*video*'] → directory='D:', pattern='*video*'
        if isinstance(raw_dir, list):
            if len(raw_dir) >= 2:
                pattern = raw_dir[-1]  # last element is likely the pattern
                raw_dir = raw_dir[0]
            elif len(raw_dir) == 1:
                raw_dir = raw_dir[0]
            else:
                return StepResult(status="failed", error="Empty path list")

        if not raw_dir:
            return StepResult(status="failed", error="No directory specified")

        # Ensure drive paths like 'D:' become 'D:\' for valid directory resolution
        if isinstance(raw_dir, str) and len(raw_dir) == 2 and raw_dir[1] == ":":
            raw_dir = raw_dir + "\\"

        directory = self.path_validator.validate(raw_dir)
        max_results = args.get("max_results", self.config.max_file_count)

        if not directory.is_dir():
            return StepResult(status="failed", error=f"Not a directory: {directory}")

        def _search():
            matches = []
            for path in directory.rglob(pattern):
                matches.append(str(path))
                if len(matches) >= max_results:
                    break
            return matches

        matches = await asyncio.get_event_loop().run_in_executor(None, _search)

        return StepResult(
            status="success", output={"count": len(matches), "files": matches[:50]}
        )

    async def _exec_search_content(self, args: dict) -> StepResult:
        directory = self.path_validator.validate(args["directory"])
        query = args["query"]
        file_type = args.get("file_type")

        cmd = ["rg", "--json", "--max-count", "20", query, str(directory)]
        if file_type:
            cmd.extend(["-t", file_type])

        try:
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=30)
            lines = stdout.decode("utf-8", errors="replace").strip().split("\n")
            matches = []
            for line in lines[:50]:
                try:
                    data = json.loads(line)
                    if data.get("type") == "match":
                        match_data = data["data"]
                        matches.append(
                            {
                                "file": match_data["path"]["text"],
                                "line": match_data["line_number"],
                                "text": match_data["lines"]["text"].strip(),
                            }
                        )
                except (json.JSONDecodeError, KeyError):
                    continue
            return StepResult(status="success", output={"matches": matches})
        except FileNotFoundError:
            return StepResult(
                status="failed", error="ripgrep (rg) not found — install it first"
            )
        except asyncio.TimeoutError:
            return StepResult(status="failed", error="Search timed out")

    # ── System Info ──────────────────────────────────────────────────────

    async def _exec_system_info(self, args: dict) -> StepResult:
        """System info — runs psutil in thread pool to avoid blocking event loop."""
        category = args.get("category", "all")

        def _collect():
            import datetime
            import psutil

            info: dict[str, Any] = {}
            now = datetime.datetime.now()
            info["current_time"] = now.strftime("%Y-%m-%d %H:%M:%S")
            info["timezone"] = str(now.astimezone().tzinfo)
            if category in ("cpu", "all"):
                info["cpu"] = {
                    "cores_physical": psutil.cpu_count(logical=False),
                    "cores_logical": psutil.cpu_count(logical=True),
                    "usage_percent": psutil.cpu_percent(interval=0.5),
                    "freq_mhz": (
                        psutil.cpu_freq().current if psutil.cpu_freq() else None
                    ),
                }
            if category in ("memory", "all"):
                mem = psutil.virtual_memory()
                info["memory"] = {
                    "total_gb": round(mem.total / 1e9, 1),
                    "used_gb": round(mem.used / 1e9, 1),
                    "percent": mem.percent,
                }
            if category in ("disk", "all"):
                disk = psutil.disk_usage("/")
                info["disk"] = {
                    "total_gb": round(disk.total / 1e9, 1),
                    "used_gb": round(disk.used / 1e9, 1),
                    "percent": disk.percent,
                }
            if category in ("processes", "all"):
                procs = []
                for p in psutil.process_iter(
                    ["pid", "name", "cpu_percent", "memory_percent"]
                ):
                    try:
                        pi = p.info
                        if pi["cpu_percent"] and pi["cpu_percent"] > 0:
                            procs.append(pi)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                procs.sort(key=lambda x: x.get("cpu_percent", 0), reverse=True)
                info["top_processes"] = procs[:10]
            return info

        info = await asyncio.get_event_loop().run_in_executor(None, _collect)
        return StepResult(status="success", output=info)

    # ── Code Execution (Docker Sandbox) ──────────────────────────────────

    async def _exec_code(self, args: dict) -> StepResult:
        language = args.get("language", "python")
        code = args.get("code", "")

        # Try Docker sandbox first
        if await self._docker_available():
            return await self._exec_code_docker(language, code)
        else:
            # Fallback: direct execution with strong warnings
            if language == "python":
                return await self._exec_code_direct_python(code)
            return StepResult(
                status="failed",
                error=f"Docker not available and no fallback for {language}",
            )

    async def _docker_available(self) -> bool:
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "info",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            code = await asyncio.wait_for(proc.wait(), timeout=5)
            return code == 0
        except (FileNotFoundError, asyncio.TimeoutError):
            return False

    async def _exec_code_docker(self, language: str, code: str) -> StepResult:
        images = {
            "python": "python:3.11-slim",
            "bash": "bash:latest",
            "javascript": "node:20-slim",
        }
        image = images.get(language, "python:3.11-slim")

        cmd = [
            "docker",
            "run",
            "--rm",
            "--network=none",
            f"--memory={self.config.sandbox_memory_limit}",
            f"--cpus={self.config.sandbox_cpu_limit}",
            "--pids-limit=50",
            image,
        ]

        if language == "python":
            cmd.extend(["python", "-c", code])
        elif language == "bash":
            cmd.extend(["bash", "-c", code])
        elif language == "javascript":
            cmd.extend(["node", "-e", code])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.config.sandbox_timeout_seconds,
            )
            return StepResult(
                status="success" if proc.returncode == 0 else "failed",
                output=stdout.decode("utf-8", errors="replace")[:5000],
                error=(
                    stderr.decode("utf-8", errors="replace")[:2000]
                    if proc.returncode != 0
                    else None
                ),
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            return StepResult(status="timeout", error="Code execution timed out")

    async def _exec_code_direct_python(self, code: str) -> StepResult:
        """Fallback: run Python code via AuditedPythonRunner (no Docker).

        Routes through the safe executor so audit hooks + import whitelist
        are enforced even without Docker.
        """
        if self._safe_executor:
            result = await self._safe_executor.run_code(code)
            return StepResult(
                status=result["status"],
                output=result.get("stdout", ""),
                error=result.get("stderr"),
            )

        # Last resort: bare subprocess (should never reach here in production)
        try:
            proc = await asyncio.create_subprocess_exec(
                "python",
                "-c",
                code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.config.sandbox_timeout_seconds,
            )
            return StepResult(
                status="success" if proc.returncode == 0 else "failed",
                output=stdout.decode("utf-8", errors="replace")[:5000],
                error=(
                    stderr.decode("utf-8", errors="replace")[:2000]
                    if proc.returncode != 0
                    else None
                ),
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            return StepResult(status="timeout", error="Code execution timed out")

    # ── Browser Automation ───────────────────────────────────────────────

    async def _get_browser_context(self):
        """BUG-07 FIX: Lazy-init persistent browser context, reuse across calls."""
        if self._browser_context is not None:
            return self._browser_context

        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        self._browser_context = (
            await self._playwright.chromium.launch_persistent_context(
                user_data_dir=self.config.browser_user_data_dir,
                headless=False,
            )
        )
        logger.info("Browser context initialized (persistent, reusable)")
        return self._browser_context

    async def close_browser(self):
        """BUG-07 FIX: Close the persistent browser context on shutdown."""
        if self._browser_context:
            try:
                await self._browser_context.close()
            except Exception as e:
                logger.debug("Browser context close error: %s", e)
            self._browser_context = None
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception as e:
                logger.debug("Playwright stop error: %s", e)
            self._playwright = None

    async def _exec_browser_navigate(self, args: dict) -> StepResult:
        url = args.get("url", "")
        action = args.get("action", "get_text")

        try:
            browser = await self._get_browser_context()
            page = browser.pages[0] if browser.pages else await browser.new_page()
            await page.goto(url, timeout=self.config.browser_timeout_ms)

            result_data = {"url": url}
            if action == "screenshot":
                path = f"data/screenshots/screenshot_{int(time.time())}.png"
                os.makedirs(os.path.dirname(path), exist_ok=True)
                await page.screenshot(path=path)
                result_data["screenshot"] = path
            elif action == "get_text":
                result_data["text"] = await page.inner_text("body")
            elif action == "wait":
                await asyncio.sleep(3)

            return StepResult(status="success", output=result_data)

        except ImportError:
            return StepResult(status="failed", error="Playwright not installed")
        except Exception as e:
            return StepResult(status="failed", error=str(e))

    async def _exec_browser_interact(self, args: dict) -> StepResult:
        selector = args.get("selector", "")
        action = args.get("action", "click")
        text = args.get("text", "")

        try:
            browser = await self._get_browser_context()
            page = browser.pages[0] if browser.pages else await browser.new_page()

            if action == "click":
                await page.click(selector, timeout=10000)
            elif action == "type":
                await page.fill(selector, text, timeout=10000)
            elif action == "scroll":
                await page.evaluate("window.scrollBy(0, 500)")

            return StepResult(
                status="success", output=f"Browser {action} on {selector}"
            )

        except ImportError:
            return StepResult(status="failed", error="Playwright not installed")
        except Exception as e:
            return StepResult(status="failed", error=str(e))

    # ── Miscellaneous ────────────────────────────────────────────────────

    async def _exec_reminder(self, args: dict) -> StepResult:
        message = args.get("message", "Reminder!")
        delay = args.get("delay_seconds", 60)

        async def _remind():
            try:
                await asyncio.sleep(delay)
                await self.event_bus.emit("system.reminder", {"message": message})
                logger.info("⏰ Reminder: %s", message)
            except asyncio.CancelledError:
                logger.debug("Reminder cancelled: %s", message)
            finally:
                # BUG-08 FIX: remove self from tracking set
                self._reminder_tasks.discard(task)

        task = asyncio.create_task(_remind())
        self._reminder_tasks.add(task)  # BUG-08 FIX: track for shutdown
        return StepResult(
            status="success", output=f"Reminder set for {delay}s: {message}"
        )

    async def _exec_store_memory(self, args: dict) -> StepResult:
        await self.event_bus.emit(
            "memory.store",
            {
                "content": args.get("content", ""),
                "type": args.get("type", "fact"),
            },
        )
        return StepResult(status="success", output="Memory stored")

    # ── App Launch ───────────────────────────────────────────────────────

    async def _exec_app_launch(self, args: dict) -> StepResult:
        """Launch an application or open a file/URL using the OS default handler."""
        target = args.get("target", args.get("app", args.get("name", "")))
        if not target:
            return StepResult(status="failed", error="No target specified")

        # Common app name mappings for Windows
        app_aliases = {
            "chrome": "chrome",
            "google chrome": "chrome",
            "firefox": "firefox",
            "notepad": "notepad",
            "calculator": "calc",
            "calc": "calc",
            "explorer": "explorer",
            "file explorer": "explorer",
            "cmd": "cmd",
            "terminal": "cmd",
            "powershell": "powershell",
            "paint": "mspaint",
            "word": "winword",
            "excel": "excel",
            "settings": "ms-settings:",
            "task manager": "taskmgr",
            "vs code": "code",
            "vscode": "code",
            "visual studio code": "code",
            "spotify": "spotify",
            "discord": "discord",
            "slack": "slack",
            "teams": "msteams",
            "microsoft teams": "msteams",
            "edge": "msedge",
            "microsoft edge": "msedge",
            "snipping tool": "snippingtool",
            "photos": "ms-photos:",
            "whatsapp": "https://web.whatsapp.com",
            "telegram": "https://web.telegram.org",
            "youtube": "https://www.youtube.com",
            "gmail": "https://mail.google.com",
            "maps": "https://maps.google.com",
        }

        resolved = app_aliases.get(target.lower(), target)

        try:
            if platform.system() == "Windows":
                # os.startfile works for apps, files, and URLs on Windows
                await asyncio.get_event_loop().run_in_executor(
                    None, os.startfile, resolved
                )
            else:
                # macOS / Linux
                opener = "open" if platform.system() == "Darwin" else "xdg-open"
                proc = await asyncio.create_subprocess_exec(
                    opener,
                    resolved,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()

            return StepResult(
                status="success",
                output=target,
            )
        except Exception as e:
            return StepResult(status="failed", error=f"Failed to launch {target}: {e}")

    # ── Get Time ─────────────────────────────────────────────────────────

    async def _exec_get_time(self, args: dict) -> StepResult:
        """Return the current date and time."""
        import datetime

        now = datetime.datetime.now()
        return StepResult(
            status="success",
            output={
                "time": now.strftime("%I:%M %p"),
                "date": now.strftime("%A, %B %d, %Y"),
                "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
                "timezone": str(now.astimezone().tzinfo),
            },
        )

    # ── Phase 2: Messaging ────────────────────────────────────────────────

    async def _exec_send_message(self, args: dict) -> StepResult:
        """Send a message via MessagingTool."""
        if not self._messaging:
            return StepResult(status="failed", error="MessagingTool not initialized")
        try:
            result = await self._messaging.send_message(
                platform=args.get("platform", "whatsapp"),
                contact=args.get("contact", ""),
                message=args.get("message", ""),
            )
            return StepResult(status="success", output=result)
        except Exception as e:
            return StepResult(status="failed", error=str(e))

    async def _exec_read_messages(self, args: dict) -> StepResult:
        """Read messages via MessagingTool."""
        if not self._messaging:
            return StepResult(status="failed", error="MessagingTool not initialized")
        try:
            result = await self._messaging.read_messages(
                platform=args.get("platform", "whatsapp"),
                contact=args.get("contact", ""),
                count=args.get("count", 5),
            )
            return StepResult(status="success", output=result)
        except Exception as e:
            return StepResult(status="failed", error=str(e))

    # ── Phase 2: Safe Code Execution ──────────────────────────────────────

    async def _exec_run_script(self, args: dict) -> StepResult:
        """Run a Python script via AuditedPythonRunner."""
        if not self._safe_executor:
            return StepResult(
                status="failed", error="AuditedPythonRunner not initialized"
            )
        try:
            code = args.get("code", "")
            script_path = args.get("path", "")
            if script_path:
                # Validate path through PathValidator before execution
                self.path_validator.validate(script_path)
                result = await self._safe_executor.run_script(script_path)
            elif code:
                result = await self._safe_executor.run_code(code)
            else:
                return StepResult(status="failed", error="No code or path provided")
            return StepResult(
                status=result["status"],
                output=result.get("stdout", ""),
                error=result.get("stderr"),
            )
        except PathViolationError as e:
            return StepResult(status="failed", error=f"Path violation: {e}")
        except Exception as e:
            return StepResult(status="failed", error=str(e))

    async def _exec_install_package(self, args: dict) -> StepResult:
        """Install a Python package via pip."""
        if not self._safe_executor:
            return StepResult(
                status="failed", error="AuditedPythonRunner not initialized"
            )
        try:
            result = await self._safe_executor.install_package(
                args.get("package", args.get("name", ""))
            )
            return StepResult(
                status=result["status"],
                output=result.get("stdout", ""),
                error=result.get("stderr"),
            )
        except Exception as e:
            return StepResult(status="failed", error=str(e))

    # ── Phase 2: System Monitor ───────────────────────────────────────────

    async def _exec_system_query(self, args: dict) -> StepResult:
        """Query system health via SystemMonitor."""
        if not self._system_monitor:
            return StepResult(status="failed", error="SystemMonitor not initialized")
        try:
            query = args.get("query", "summary")
            if query in ("summary", "health", "status"):
                output = self._system_monitor.get_health_summary()
            elif query in ("processes", "top"):
                count = args.get("count", 5)
                output = self._system_monitor.get_top_processes(count)
            else:
                output = self._system_monitor.get_snapshot()
            return StepResult(status="success", output=output)
        except Exception as e:
            return StepResult(status="failed", error=str(e))

    # ── Future: Fast File Search ───────────────────────────────────────────

    async def _exec_fast_search(self, args: dict) -> StepResult:
        """Search files via Everything SDK / glob fallback."""
        if not self._fast_search:
            return StepResult(status="failed", error="FastSearch not initialized")
        try:
            query = args.get("query", "")
            max_results = args.get("max_results")
            ext = args.get("extension")

            if ext:
                results = self._fast_search.search_by_ext(
                    ext, args.get("directory", ""), max_results
                )
            else:
                results = self._fast_search.search(query, max_results)

            output = [{"path": r.path, "name": r.name, "size": r.size} for r in results]
            return StepResult(status="success", output=output)
        except Exception as e:
            return StepResult(status="failed", error=str(e))

    # ── Future: Code Co-Pilot ──────────────────────────────────────────────

    async def _exec_code_copilot(self, args: dict) -> StepResult:
        """Generate and optionally run+fix code via CoPilot."""
        if not self._copilot:
            return StepResult(status="failed", error="CodeCoPilot not initialized")
        try:
            prompt = args.get("prompt", "")
            project_path = args.get("project_path", ".")
            run = args.get("run", False)
            save_path = args.get("save_path", "")

            context = self._copilot.scan_project(project_path)
            result = await self._copilot.generate_code(prompt, context)

            if result.status != "success":
                return StepResult(status="failed", error=result.error)

            if run and save_path:
                result = await self._copilot.run_and_fix(
                    result.code, save_path, context
                )

            return StepResult(
                status=result.status,
                output=result.output or result.code,
                error=result.error,
            )
        except Exception as e:
            return StepResult(status="failed", error=str(e))
