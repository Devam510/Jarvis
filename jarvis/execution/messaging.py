"""
jarvis.execution.messaging — Persistent browser-based messaging automation.

Supports WhatsApp Web and Telegram Web with:
  - Persistent Playwright context (stays logged in)
  - Stale context detection + auto-recreate
  - Retry with exponential backoff
  - Per-step interaction timeout
  - Graceful browser shutdown on agent stop
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# ── Platform Definitions ─────────────────────────────────────────────────────

PLATFORM_CONFIG = {
    "whatsapp": {
        "url": "https://web.whatsapp.com",
        "search_selector": 'div[contenteditable="true"][data-tab="3"]',
        "message_input_placeholder": "Type a message",
        "contact_list_selector": "span[title='{contact}']",
        "send_button_aria": "Send",
        "message_bubble_selector": "div.message-in, div.message-out",
    },
    "telegram": {
        "url": "https://web.telegram.org/a/",
        "search_selector": "#telegram-search-input",
        "message_input_placeholder": "Message",
        "contact_list_selector": "a[href] >> text='{contact}'",
        "send_button_aria": "Send Message",
        "message_bubble_selector": "div.Message",
    },
}

# ── Retry / Timeout Constants ────────────────────────────────────────────────

MAX_RETRIES = 3
BASE_BACKOFF_S = 1.0  # 1s, 2s, 4s
STEP_TIMEOUT_MS = 15000  # 15s per individual interaction step
SYNC_WAIT_S = 4.0  # Wait for WhatsApp/Telegram to sync


class MessagingTool:
    """High-level messaging automation via persistent Playwright browser."""

    def __init__(self, browser_user_data_dir: str = "data/browser_profile"):
        self._user_data_dir = browser_user_data_dir
        self._context = None  # Persistent browser context
        self._playwright = None
        self._lock = asyncio.Lock()  # protects _context lifecycle
        self._op_lock = asyncio.Lock()  # serializes page operations

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def _ensure_browser(self) -> Any:
        """Lazy-init or recreate the persistent browser context.

        Detects stale/crashed contexts and rebuilds automatically.
        """
        async with self._lock:
            if self._context is not None:
                # Test liveness — if pages are gone, context is stale
                try:
                    _ = self._context.pages
                except Exception:
                    logger.warning("Stale browser context detected — recreating")
                    await self._teardown_unlocked()

            if self._context is None:
                from playwright.async_api import async_playwright

                self._playwright = await async_playwright().start()
                self._context = (
                    await self._playwright.chromium.launch_persistent_context(
                        user_data_dir=self._user_data_dir,
                        headless=False,
                        args=["--disable-blink-features=AutomationControlled"],
                    )
                )
                logger.info("Persistent browser context created")

            return self._context

    async def _teardown_unlocked(self):
        """Close context + playwright without holding lock (caller must hold)."""
        if self._context:
            try:
                await self._context.close()
            except Exception:
                pass
            self._context = None
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

    async def close(self):
        """Graceful shutdown — called from orchestrator.stop()."""
        async with self._lock:
            await self._teardown_unlocked()
        logger.info("MessagingTool shut down")

    # ── Retry Wrapper ─────────────────────────────────────────────────────

    async def _with_retry(
        self, coro_factory, description: str, total_timeout: float = 90.0
    ) -> Any:
        """Execute an async operation with exponential backoff retries.

        Args:
            coro_factory: Callable that returns a new coroutine on each call.
            description: Human-readable description for logging.
            total_timeout: Hard cap on total retry duration (seconds).
        """
        deadline = time.monotonic() + total_timeout
        last_error = None
        for attempt in range(MAX_RETRIES):
            if time.monotonic() >= deadline:
                break
            remaining = deadline - time.monotonic()
            try:
                return await asyncio.wait_for(
                    coro_factory(), timeout=min(remaining, 60.0)
                )
            except asyncio.TimeoutError as e:
                last_error = e
                logger.warning(
                    "%s timed out (attempt %d/%d)",
                    description,
                    attempt + 1,
                    MAX_RETRIES,
                )
            except Exception as e:
                last_error = e
                wait = BASE_BACKOFF_S * (2**attempt)
                logger.warning(
                    "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                    description,
                    attempt + 1,
                    MAX_RETRIES,
                    e,
                    wait,
                )
                # On connection/stale errors, recreate browser
                error_name = type(e).__name__
                if "TargetClosedError" in error_name or "Connection" in error_name:
                    async with self._lock:
                        await self._teardown_unlocked()
                await asyncio.sleep(wait)

        raise RuntimeError(
            f"{description} failed after {MAX_RETRIES} retries: {last_error}"
        )

    # ── Core Actions ──────────────────────────────────────────────────────

    async def send_message(
        self, platform: str, contact: str, message: str
    ) -> dict[str, Any]:
        """Send a message to a contact on the specified platform.

        Returns: {"status": "sent", "platform": ..., "contact": ..., "message": ...}
        """
        platform = platform.lower()
        if platform not in PLATFORM_CONFIG:
            raise ValueError(
                f"Unsupported platform: {platform}. "
                f"Supported: {list(PLATFORM_CONFIG.keys())}"
            )

        config = PLATFORM_CONFIG[platform]

        async def _do_send():
            async with self._op_lock:  # serialize page operations
                ctx = await self._ensure_browser()
                page = ctx.pages[0] if ctx.pages else await ctx.new_page()

                # Navigate to platform
                if platform not in (page.url or ""):
                    await page.goto(config["url"], timeout=30000)
                    await asyncio.sleep(SYNC_WAIT_S)

                # Search for contact
                search = page.locator(config["search_selector"]).first
                await search.click(timeout=STEP_TIMEOUT_MS)
                await search.fill("", timeout=STEP_TIMEOUT_MS)
                await search.type(contact, delay=80, timeout=STEP_TIMEOUT_MS)
                await asyncio.sleep(1.5)

                # Click on contact result
                contact_sel = config["contact_list_selector"].format(contact=contact)
                await page.locator(contact_sel).first.click(timeout=STEP_TIMEOUT_MS)
                await asyncio.sleep(1.0)

                # Type message
                msg_input = page.get_by_placeholder(
                    config["message_input_placeholder"]
                ).first
                await msg_input.click(timeout=STEP_TIMEOUT_MS)
                await msg_input.type(message, delay=30, timeout=STEP_TIMEOUT_MS)

                # Send
                send_btn = page.get_by_role(
                    "button", name=config["send_button_aria"]
                ).first
                await send_btn.click(timeout=STEP_TIMEOUT_MS)
                await asyncio.sleep(0.5)

                return {
                    "status": "sent",
                    "platform": platform,
                    "contact": contact,
                    "message": message,
                }

        return await self._with_retry(_do_send, f"send_message({platform}, {contact})")

    async def read_messages(
        self, platform: str, contact: str, count: int = 5
    ) -> dict[str, Any]:
        """Read the last N messages from a contact.

        Returns: {"platform": ..., "contact": ..., "messages": [...]}
        """
        platform = platform.lower()
        if platform not in PLATFORM_CONFIG:
            raise ValueError(f"Unsupported platform: {platform}")

        config = PLATFORM_CONFIG[platform]

        async def _do_read():
            async with self._op_lock:  # serialize page operations
                ctx = await self._ensure_browser()
                page = ctx.pages[0] if ctx.pages else await ctx.new_page()

                if platform not in (page.url or ""):
                    await page.goto(config["url"], timeout=30000)
                    await asyncio.sleep(SYNC_WAIT_S)

                # Search for contact
                search = page.locator(config["search_selector"]).first
                await search.click(timeout=STEP_TIMEOUT_MS)
                await search.fill("", timeout=STEP_TIMEOUT_MS)
                await search.type(contact, delay=80, timeout=STEP_TIMEOUT_MS)
                await asyncio.sleep(1.5)

                contact_sel = config["contact_list_selector"].format(contact=contact)
                await page.locator(contact_sel).first.click(timeout=STEP_TIMEOUT_MS)
                await asyncio.sleep(1.5)

                # Scrape message bubbles
                bubbles = page.locator(config["message_bubble_selector"])
                all_texts = await bubbles.all_text_contents()
                messages = [t.strip() for t in all_texts if t.strip()][-count:]

                return {
                    "platform": platform,
                    "contact": contact,
                    "messages": messages,
                }

        return await self._with_retry(_do_read, f"read_messages({platform}, {contact})")
