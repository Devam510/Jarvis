"""
jarvis.perception.screen_watcher — Continuous screen capture and analysis.

Safety guarantees:
  [S1] Disabled by default — requires explicit opt-in
  [S2] Hard privacy blackout — app name + title regex, no OCR, no screenshot saved
  [S4] Resource budget — OCR rate limit, disk cap, CPU auto-disable
  [S6] Context immutability — frozen ScreenContext
  [S7] Crash isolation — auto-restart once, disable on second failure
"""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

from jarvis.core.event_bus import AsyncEventBus
from jarvis.perception.ocr_analyzer import OCRAnalyzer
from jarvis.utils.config import PerceptionConfig
from jarvis.utils.types import ScreenContext

logger = logging.getLogger(__name__)


class ScreenWatcher:
    """Background 1 FPS screen watcher with change detection and privacy guard."""

    def __init__(self, config: PerceptionConfig, event_bus: AsyncEventBus):
        self._config = config
        self._event_bus = event_bus
        self._ocr = OCRAnalyzer(language=config.ocr_language)
        self._running = False
        self._disabled_permanently = False  # [S7] set after second crash
        self._crash_count = 0  # [S7] crash counter

        # Previous frame for change detection
        self._prev_frame_hash: Optional[int] = None

        # [S4] OCR rate limiting
        self._ocr_calls_this_minute: int = 0
        self._ocr_minute_start: float = 0.0

        # [S4] CPU monitoring
        self._high_cpu_since: Optional[float] = None

        # [S2] Pre-compile title regex patterns
        self._title_patterns = [re.compile(p) for p in config.privacy_title_patterns]

        # Screenshot rolling buffer
        self._screenshot_index = 0

    async def run(self):
        """Main perception loop. Implements S7 crash isolation."""
        if not self._config.enabled:
            logger.info("ScreenWatcher disabled (perception.enabled=False)")
            return

        # [S1] Log activation at WARNING level for audit trail
        logger.warning(
            "ScreenWatcher ACTIVATED — continuous screen capture enabled "
            "(fps=%.1f, change_threshold=%.2f)",
            self._config.capture_fps,
            self._config.change_threshold,
        )

        while not self._disabled_permanently:
            try:
                self._running = True
                await self._capture_loop()
                # Clean exit (stop() called) → don't restart
                break
            except asyncio.CancelledError:
                logger.info("ScreenWatcher cancelled")
                break
            except Exception as e:
                self._crash_count += 1
                if self._crash_count >= 2:
                    # [S7] Second crash → disable permanently
                    self._disabled_permanently = True
                    logger.error(
                        "ScreenWatcher crashed %d times — PERMANENTLY DISABLED: %s",
                        self._crash_count,
                        e,
                    )
                    await self._event_bus.emit(
                        "perception.disabled",
                        {
                            "reason": f"repeated_crash: {e}",
                            "crash_count": self._crash_count,
                        },
                    )
                    break
                else:
                    # [S7] First crash → log and restart
                    logger.warning(
                        "ScreenWatcher crashed (attempt %d/2), restarting: %s",
                        self._crash_count,
                        e,
                    )
                    await asyncio.sleep(2.0)  # Brief cooldown before restart
            finally:
                self._running = False

    async def _capture_loop(self):
        """Core 1 FPS capture loop."""
        interval = 1.0 / self._config.capture_fps

        while self._running:
            loop_start = time.monotonic()

            # [S4] Check CPU threshold
            if await self._should_disable_for_cpu():
                logger.warning(
                    "ScreenWatcher auto-disabled: CPU >%.0f%% for >%.0fs",
                    self._config.cpu_disable_threshold,
                    self._config.cpu_check_window_seconds,
                )
                self._disabled_permanently = True
                await self._event_bus.emit(
                    "perception.disabled", {"reason": "cpu_threshold_exceeded"}
                )
                return

            try:
                # Get active window info
                app_name, window_title = await self._get_active_window()

                # [S2] Check privacy blackout
                if self._is_blacked_out(app_name, window_title):
                    context = ScreenContext(
                        active_app=app_name,
                        window_title="[REDACTED]",
                        visible_text="",
                        detected_errors=(),
                        timestamp=time.time(),
                        screenshot_path="",
                        change_magnitude=0.0,
                        is_blacked_out=True,
                    )
                    await self._event_bus.emit("perception.context_updated", context)
                    await self._sleep_remaining(loop_start, interval)
                    continue

                # Capture screenshot
                frame_data, screenshot_path = await self._capture_screenshot()

                # Compute change magnitude
                change = self._compute_change(frame_data)

                # Only run OCR if change exceeds threshold and rate limit allows
                visible_text = ""
                detected_errors: tuple[str, ...] = ()
                if (
                    change >= self._config.change_threshold
                    and screenshot_path
                    and self._ocr_rate_ok()
                ):
                    visible_text = await self._ocr.extract_text(screenshot_path)
                    error_list = self._ocr.detect_errors(visible_text)
                    detected_errors = tuple(error_list)
                    self._ocr_calls_this_minute += 1

                # Build immutable context [S6]
                context = ScreenContext(
                    active_app=app_name,
                    window_title=window_title,
                    visible_text=visible_text,
                    detected_errors=detected_errors,
                    timestamp=time.time(),
                    screenshot_path=screenshot_path,
                    change_magnitude=change,
                    is_blacked_out=False,
                )

                await self._event_bus.emit("perception.context_updated", context)

            except Exception as e:
                logger.debug("Capture cycle error (non-fatal): %s", e)

            await self._sleep_remaining(loop_start, interval)

    def _is_blacked_out(self, app_name: str, window_title: str) -> bool:
        """[S2] Check if current window should be blacked out."""
        # Check app name blacklist
        for blocked in self._config.privacy_apps:
            if blocked.lower() in app_name.lower():
                return True

        # Check window title regex patterns
        for pattern in self._title_patterns:
            if pattern.search(window_title):
                return True

        return False

    def _ocr_rate_ok(self) -> bool:
        """[S4] Check if OCR rate limit allows another call."""
        now = time.time()
        # Reset counter every minute
        if now - self._ocr_minute_start >= 60.0:
            self._ocr_calls_this_minute = 0
            self._ocr_minute_start = now
        return self._ocr_calls_this_minute < self._config.max_ocr_per_minute

    async def _should_disable_for_cpu(self) -> bool:
        """[S4] Check if CPU is sustained above threshold."""
        try:
            import psutil

            cpu_pct = psutil.cpu_percent(interval=0)
            if cpu_pct > self._config.cpu_disable_threshold:
                if self._high_cpu_since is None:
                    self._high_cpu_since = time.monotonic()
                elif (
                    time.monotonic() - self._high_cpu_since
                    >= self._config.cpu_check_window_seconds
                ):
                    return True
            else:
                self._high_cpu_since = None
        except ImportError:
            pass  # psutil not available — skip CPU check
        return False

    async def _get_active_window(self) -> tuple[str, str]:
        """Get active window app name and title. Returns ("", "") on failure."""
        try:
            import pygetwindow as gw

            win = await asyncio.get_event_loop().run_in_executor(
                None, gw.getActiveWindow
            )
            if win:
                title = win.title or ""
                # Extract app name from process (simplified)
                app_name = title.split(" - ")[-1] if " - " in title else title
                return app_name, title
        except ImportError:
            logger.debug("pygetwindow not installed — window detection disabled")
        except Exception as e:
            logger.debug("Window detection failed: %s", e)
        return "", ""

    async def _capture_screenshot(self) -> tuple[bytes, str]:
        """Capture primary monitor screenshot. Returns (raw_bytes, saved_path)."""
        try:
            import mss

            screenshot_path = ""
            with mss.mss() as sct:
                monitor = sct.monitors[1]  # Primary monitor
                img = sct.grab(monitor)
                raw = bytes(img.raw)

                # [S4] Check disk usage before saving (BUG-18: non-blocking)
                if await self._disk_usage_ok_async():
                    # Rolling buffer
                    idx = self._screenshot_index % self._config.max_screenshots
                    self._screenshot_index += 1
                    screenshot_dir = Path(self._config.screenshot_dir)
                    screenshot_dir.mkdir(parents=True, exist_ok=True)
                    screenshot_path = str(screenshot_dir / f"frame_{idx}.png")

                    # Save in executor
                    from mss.tools import to_png

                    png_data = to_png(img.rgb, img.size)
                    await asyncio.get_event_loop().run_in_executor(
                        None, self._write_file, screenshot_path, png_data
                    )

                return raw, screenshot_path
        except ImportError:
            logger.debug("mss not installed — screen capture disabled")
            return b"", ""
        except Exception as e:
            logger.debug("Screenshot failed: %s", e)
            return b"", ""

    def _write_file(self, path: str, data: bytes):
        """Write bytes to file (runs in executor)."""
        with open(path, "wb") as f:
            f.write(data)

    def _disk_usage_ok(self) -> bool:
        """[S4] Check if screenshot disk usage is within budget (sync I/O)."""
        try:
            screenshot_dir = Path(self._config.screenshot_dir)
            if not screenshot_dir.exists():
                return True
            total = sum(
                f.stat().st_size for f in screenshot_dir.iterdir() if f.is_file()
            )
            max_bytes = self._config.max_screenshot_disk_mb * 1024 * 1024
            return total < max_bytes
        except Exception:
            return True

    async def _disk_usage_ok_async(self) -> bool:
        """BUG-18 FIX: Non-blocking wrapper — runs sync disk I/O in thread pool."""
        return await asyncio.to_thread(self._disk_usage_ok)

    def _compute_change(self, frame_data: bytes) -> float:
        """BUG-17 FIX: Actual byte-level sampling comparison. Returns 0-1 magnitude."""
        if not frame_data:
            return 0.0

        # Sample every Nth byte for speed (target ~10k samples)
        sample_size = min(len(frame_data), 10000)
        step = max(1, len(frame_data) // sample_size)
        current_samples = frame_data[::step]

        if self._prev_frame_hash is None:
            # Store sampled bytes for comparison (reuse _prev_frame_hash field name)
            self._prev_frame_hash = current_samples
            return 1.0  # First frame = maximum change

        prev_samples = self._prev_frame_hash

        # Fast identity check
        if current_samples == prev_samples:
            self._prev_frame_hash = current_samples
            return 0.0

        # Compute mean absolute difference between sampled bytes
        min_len = min(len(current_samples), len(prev_samples))
        if min_len == 0:
            self._prev_frame_hash = current_samples
            return 1.0

        total_diff = sum(
            abs(current_samples[i] - prev_samples[i]) for i in range(min_len)
        )
        # Normalize: max possible diff per byte is 255
        change = total_diff / (min_len * 255.0)
        change = min(1.0, max(0.0, change))

        self._prev_frame_hash = current_samples
        return change

    async def _sleep_remaining(self, loop_start: float, interval: float):
        """Sleep for the remainder of the frame interval."""
        elapsed = time.monotonic() - loop_start
        remaining = interval - elapsed
        if remaining > 0:
            await asyncio.sleep(remaining)

    def stop(self):
        """Stop the watcher."""
        self._running = False
        logger.info("ScreenWatcher stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_permanently_disabled(self) -> bool:
        return self._disabled_permanently
