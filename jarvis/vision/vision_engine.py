"""
jarvis.vision.vision_engine — Screenshot capture and LLaVA-based screen understanding.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from jarvis.utils.types import VisionResult

logger = logging.getLogger(__name__)


class ScreenCapture:
    """Capture screenshots using mss (cross-platform)."""

    def __init__(self, output_dir: str = "data/screenshots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def capture_primary(self) -> str:
        """Capture the entire primary monitor. Returns path to PNG."""
        try:
            import mss

            with mss.mss() as sct:
                monitor = sct.monitors[1]  # Primary monitor
                screenshot = sct.grab(monitor)
                path = str(self.output_dir / f"screen_{int(time.time())}.png")
                mss.tools.to_png(screenshot.rgb, screenshot.size, output=path)
                logger.debug("Screenshot captured: %s", path)
                return path
        except ImportError:
            logger.error("mss not installed — cannot capture screenshots")
            return ""
        except Exception as e:
            logger.error("Screenshot capture failed: %s", e)
            return ""

    def capture_region(self, x: int, y: int, width: int, height: int) -> str:
        """Capture a specific screen region."""
        try:
            import mss

            with mss.mss() as sct:
                region = {"left": x, "top": y, "width": width, "height": height}
                screenshot = sct.grab(region)
                path = str(self.output_dir / f"region_{int(time.time())}.png")
                mss.tools.to_png(screenshot.rgb, screenshot.size, output=path)
                return path
        except Exception as e:
            logger.error("Region capture failed: %s", e)
            return ""


class VisionEngine:
    """
    Analyze screenshots using LLaVA (via Ollama multimodal API).
    """

    def __init__(
        self, ollama_base_url: str = "http://localhost:11434", model: str = "llava:7b"
    ):
        self.base_url = ollama_base_url
        self.model = model
        self.capture = ScreenCapture()

    async def analyze_screen(
        self, prompt: str = "Describe what you see on the screen."
    ) -> VisionResult:
        """Capture a screenshot and analyze it with LLaVA."""
        screenshot_path = self.capture.capture_primary()
        if not screenshot_path:
            return VisionResult(
                description="Failed to capture screenshot.", screenshot_path=""
            )

        description = await self._analyze_image(screenshot_path, prompt)
        return VisionResult(
            description=description,
            screenshot_path=screenshot_path,
        )

    async def analyze_image(self, image_path: str, prompt: str) -> VisionResult:
        """Analyze a specific image with LLaVA."""
        description = await self._analyze_image(image_path, prompt)
        return VisionResult(
            description=description,
            screenshot_path=image_path,
        )

    async def _analyze_image(self, image_path: str, prompt: str) -> str:
        """Send image to LLaVA via Ollama for analysis."""
        try:
            # Read and base64 encode the image
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            try:
                import ollama

                client = ollama.AsyncClient(host=self.base_url)
                response = await asyncio.wait_for(
                    client.chat(
                        model=self.model,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                                "images": [image_data],
                            }
                        ],
                    ),
                    timeout=60,
                )
                return response["message"]["content"]
            except ImportError:
                # Fallback: use aiohttp directly
                import aiohttp

                payload = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [image_data],
                        }
                    ],
                    "stream": False,
                }
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/api/chat",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=60),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return data.get("message", {}).get("content", "")
                        else:
                            error = await resp.text()
                            logger.error("LLaVA API error: %s", error)
                            return f"Vision analysis failed: {error}"

        except asyncio.TimeoutError:
            return "Vision analysis timed out."
        except Exception as e:
            logger.error("Vision analysis error: %s", e)
            return f"Vision analysis error: {e}"

    async def get_ui_state(self) -> dict:
        """Analyze the screen and extract UI state information."""
        result = await self.analyze_screen(
            prompt=(
                "List all visible UI elements on the screen. For each element, provide: "
                "type (button, text field, menu, etc.), label/text, and approximate position. "
                'Respond with JSON: {"elements": [{"type": "...", "label": "...", "position": "..."}]}'
            )
        )
        try:
            return json.loads(result.description)
        except json.JSONDecodeError:
            return {"raw_description": result.description}
