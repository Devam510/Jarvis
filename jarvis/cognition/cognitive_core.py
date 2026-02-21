"""
jarvis.cognition.cognitive_core — LLM interface, planning, and self-critique.

This is the "brain" of Jarvis. It takes transcripts, retrieves memory context,
constructs the system prompt with tool schemas, calls Ollama, parses the
structured JSON response, and optionally runs self-critique.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import re
from typing import Any, Optional

from jarvis.core.event_bus import AsyncEventBus
from jarvis.utils.config import LLMConfig
from jarvis.utils.types import TranscriptEvent, Plan, PlannedAction, CritiqueResult
from jarvis.cognition.tool_schemas import ToolRegistry

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_TEMPLATE = """You are Jarvis, a helpful personal desktop assistant running locally.

Respond ONLY with a JSON object. No other text.

Format for questions: {{"thought": "reasoning", "confidence": 0.9, "response": "answer"}}
Format for actions: {{"thought": "reasoning", "confidence": 0.9, "response": "what to say", "actions": [{{"tool": "name", "args": {{}}, "requires_confirmation": false, "depends_on": null}}]}}

For MULTIPLE actions in one request, include ALL actions in one response:
{{"thought": "reasoning", "confidence": 0.9, "response": "Opening both.", "actions": [{{"tool": "app_launch", "args": {{"target": "chrome"}}, "requires_confirmation": false, "depends_on": null}}, {{"tool": "app_launch", "args": {{"target": "notepad"}}, "requires_confirmation": false, "depends_on": null}}]}}

## Tools
{tool_schemas}

## Memory
{memory_context}

## Rules
1. TIME/DATE questions → use "get_time" tool. NEVER guess.
2. OPEN app → use "app_launch" tool with {{"target": "app_name"}}.
3. Multiple apps in one request → multiple actions in one response.
4. COMPUTER info → use "system_info" tool.
5. Greetings/knowledge → respond directly, NO actions.
6. ALWAYS valid JSON. No markdown.
7. Destructive actions → "requires_confirmation": true.
8. FIND/SEARCH/LOCATE files or folders → use "search_files" tool. NEVER guess paths.
"""


class CognitiveCore:
    """
    LLM interface — turns transcripts into structured plans.
    """

    def __init__(self, config: LLMConfig, event_bus: AsyncEventBus, memory=None):
        self.config = config
        self.event_bus = event_bus
        self.memory = memory
        self.tool_registry = ToolRegistry()
        self._client = None
        self._init_client()

    def _init_client(self):
        try:
            import ollama

            self._client = ollama.AsyncClient(host=self.config.base_url)
            logger.info("Ollama client initialized: %s", self.config.base_url)
        except Exception as e:
            logger.error("Ollama client init failed: %s", e)

    async def process_transcript(self, transcript: TranscriptEvent):
        """Process a transcript event: retrieve memory, call LLM, emit plan or response."""
        start = time.time()
        text = transcript.text
        correlation_id = transcript.correlation_id

        logger.info("Processing transcript: '%s'", text)

        # Store user turn in session memory
        if self.memory:
            self.memory.add_user_turn(text)

        # Build context
        memory_context = ""
        if self.memory:
            memory_context = self.memory.format_memory_context(text)

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            tool_schemas=self.tool_registry.to_prompt_block(),
            memory_context=memory_context or "No relevant memories found.",
        )

        # Call LLM
        messages = []
        messages.append({"role": "system", "content": system_prompt})

        # Include conversation history
        if self.memory:
            for turn in self.memory.get_conversation_messages()[-10:]:
                messages.append(turn)

        messages.append({"role": "user", "content": text})

        response_json = await self.call_llm(messages)
        elapsed = (time.time() - start) * 1000

        if response_json is None:
            await self.event_bus.emit(
                "cognition.direct_response",
                {
                    "text": "I'm sorry, I wasn't able to process that. Could you try again?",
                    "correlation_id": correlation_id,
                },
            )
            return

        logger.info("LLM response parsed in %.0fms", elapsed)
        logger.info("LLM raw response: %s", response_json)

        # Store assistant turn
        if self.memory:
            resp_text = response_json.get("response", "")
            if resp_text:
                self.memory.add_assistant_turn(resp_text)

        # ── Normalize LLM response into an actions list ──────────────────
        # The 3B model returns many variant formats. We handle them ALL:
        #  1. {"actions": [{"tool": "app_launch", "args": {...}}]}  (correct)
        #  2. {"type": "action", "name": "app_launch", "parameters": {...}}
        #  3. {"type": "app_launch", "parameters": {...}}  (type=tool name)
        #  4. {"action": "app_launch", "parameters": {...}}  (action=tool name)
        #  5. {"tool": "app_launch", "args": {...}}  (flat single-action)
        #  6. {"name": "open_app", "parameters": {...}}  (invented name)
        # Strategy: scan ALL common keys, resolve via alias map.
        valid_tools = set(self.tool_registry.tool_names())

        # Alias map: LLM-invented tool names → actual registered tools
        TOOL_ALIASES = {
            "open_app": "app_launch",
            "launch_app": "app_launch",
            "run_app": "app_launch",
            "start_app": "app_launch",
            "open_application": "app_launch",
            "launch": "app_launch",
            "open": "app_launch",
            "run": "app_launch",
            "open_browser": "browser_navigate",
            "browse": "browser_navigate",
            "open_url": "browser_navigate",
            "web_search": "browser_navigate",
            "search_web": "browser_navigate",
            "google": "browser_navigate",
            "run_code": "execute_code",
            "exec": "execute_code",
            "read_file": "file_read",
            "write_file": "file_write",
            "delete_file": "file_delete",
            "move_file": "file_move",
            "copy_file": "file_copy",
            "get_system_info": "system_info",
            "sysinfo": "system_info",
            "remember": "store_memory",
            "save_memory": "store_memory",
            "remind": "set_reminder",
            "timer": "set_reminder",
            "time": "get_time",
            "clock": "get_time",
            "what_time": "get_time",
            "find_files": "search_files",
            "find_file": "search_files",
            "find_folder": "search_files",
            "find": "search_files",
            "search": "search_files",
            "locate": "search_files",
            "grep": "search_content",
        }

        def _resolve_tool(name: str) -> str:
            if name in valid_tools:
                return name
            alias = TOOL_ALIASES.get(name.lower())
            return alias if alias and alias in valid_tools else ""

        actions = response_json.get("actions", [])
        if isinstance(actions, dict):
            actions = [actions]

        if not actions:
            TOOL_NAME_KEYS = (
                "type",
                "action",
                "tool",
                "name",
                "function",
                "action_type",
            )
            ARGS_KEYS = ("parameters", "args", "arguments", "params")

            found_tool = ""
            for key in TOOL_NAME_KEYS:
                val = response_json.get(key, "")
                if isinstance(val, str) and val.strip():
                    candidate = val.strip()
                    resolved = _resolve_tool(candidate)
                    if resolved:
                        found_tool = resolved
                        break
                    if candidate in ("action", "actions"):
                        for k2 in TOOL_NAME_KEYS:
                            if k2 == key:
                                continue
                            v2 = response_json.get(k2, "")
                            if isinstance(v2, str) and v2.strip():
                                r2 = _resolve_tool(v2.strip())
                                if r2:
                                    found_tool = r2
                                    break
                        if found_tool:
                            break

            if found_tool:
                args = None
                for ak in ARGS_KEYS:
                    v = response_json.get(ak)
                    if isinstance(v, dict):
                        args = v
                        break
                if args is None:
                    SKIP = (
                        set(TOOL_NAME_KEYS)
                        | set(ARGS_KEYS)
                        | {
                            "thought",
                            "confidence",
                            "response",
                            "requires_confirmation",
                            "depends_on",
                        }
                    )
                    args = {k: v for k, v in response_json.items() if k not in SKIP}
                actions = [{"tool": found_tool, "args": args or {}}]
                logger.info(
                    "Normalized variant LLM format → tool=%s, args=%s", found_tool, args
                )

        if not actions:
            # Direct conversational response
            await self.event_bus.emit(
                "cognition.direct_response",
                {
                    "text": response_json.get("response", "I processed your request."),
                    "correlation_id": correlation_id,
                },
            )
        else:
            # Build plan
            planned_actions = []
            valid_tools = set(self.tool_registry.tool_names())
            for i, action in enumerate(actions):
                if not isinstance(action, dict):
                    logger.warning(
                        "Skipping action %d: not a dict (%s)", i, type(action)
                    )
                    continue
                # Try multiple key names that LLMs commonly use for tool names
                TOOL_NAME_KEYS = ("tool", "name", "function", "action_type", "action")
                tool_name = ""
                tool_name_key = None
                for key in TOOL_NAME_KEYS:
                    val = action.get(key, "")
                    if val and isinstance(val, str):
                        tool_name = val.strip()
                        tool_name_key = key
                        break

                if not tool_name:
                    logger.warning(
                        "Skipping action %d: empty tool name (keys: %s)",
                        i,
                        list(action.keys()),
                    )
                    continue
                if tool_name not in valid_tools:
                    logger.warning(
                        "Skipping action %d: unknown tool '%s'", i, tool_name
                    )
                    continue

                # Extract arguments — try nested dict first, then collect flat keys
                args = (
                    action.get("args")
                    or action.get("arguments")
                    or action.get("parameters")
                )
                if not args or not isinstance(args, dict):
                    # Arguments are at the same level as tool name (flat format)
                    # e.g. {"action_type": "app_launch", "target": "chrome"}
                    SKIP_KEYS = set(TOOL_NAME_KEYS) | {
                        "type",
                        "requires_confirmation",
                        "depends_on",
                    }
                    args = {k: v for k, v in action.items() if k not in SKIP_KEYS}

                planned_actions.append(
                    PlannedAction(
                        tool_name=tool_name,
                        arguments=args or {},
                        requires_confirmation=action.get(
                            "requires_confirmation", False
                        ),
                        depends_on=action.get("depends_on"),
                    )
                )

            if not planned_actions:
                # All actions were invalid — fall back to direct response
                logger.warning(
                    "All actions had invalid tool names, responding directly"
                )
                await self.event_bus.emit(
                    "cognition.direct_response",
                    {
                        "text": response_json.get(
                            "response", "I processed your request."
                        ),
                        "correlation_id": correlation_id,
                    },
                )
                return

            plan = Plan(
                thought=response_json.get("thought", ""),
                confidence=response_json.get("confidence", 0.5),
                actions=planned_actions,
                correlation_id=correlation_id,
            )

            # Self-critique
            critique = await self._self_critique(plan, text)
            if not critique.approved:
                logger.warning("Self-critique rejected plan: %s", critique.issues)
                # Re-plan or inform user
                await self.event_bus.emit(
                    "cognition.direct_response",
                    {
                        "text": f"I reviewed my plan and found issues: {', '.join(critique.issues)}. Let me reconsider.",
                        "correlation_id": correlation_id,
                    },
                )
                return

            # Emit plan with response text
            await self.event_bus.emit(
                "cognition.plan_ready",
                {
                    "plan": plan,
                    "response_text": response_json.get("response", ""),
                    "correlation_id": correlation_id,
                },
            )

    async def call_llm(
        self, messages: list[dict], max_tokens: Optional[int] = None
    ) -> Optional[dict]:
        """Call Ollama and parse JSON response.

        Args:
            messages: Chat messages to send.
            max_tokens: Override num_predict for this call (used by S9 reflection bounds).
        """
        if self._client is None:
            return None

        num_predict = max_tokens if max_tokens is not None else self.config.num_predict

        try:
            response = await asyncio.wait_for(
                self._client.chat(
                    model=self.config.model,
                    messages=messages,
                    options={
                        "temperature": self.config.temperature,
                        "num_predict": num_predict,
                        "num_ctx": self.config.num_ctx,
                        "num_gpu": self.config.num_gpu,
                    },
                    format="json",
                ),
                timeout=self.config.timeout_seconds,
            )

            content = response["message"]["content"]
            # Parse JSON from response
            return self._extract_json(content)

        except asyncio.TimeoutError:
            logger.error("LLM call timed out after %ds", self.config.timeout_seconds)
            return None
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return None

    def _extract_json(self, text: str) -> Optional[dict]:
        """Extract JSON from LLM response, handling markdown fences."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code fence
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding first { to last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        logger.error("Failed to parse LLM response as JSON: %s", text[:200])
        return None

    async def _self_critique(self, plan: Plan, original_query: str) -> CritiqueResult:
        """
        Run self-critique on a plan by asking the LLM to review it.
        Fast-path: skip for simple/safe plans.
        """
        # Safe tools that never need critique
        SAFE_TOOLS = {
            "get_time",
            "app_launch",
            "system_info",
            "memory_store",
            "memory_recall",
        }

        # Skip critique for small plans with well-known tools
        all_safe = all(a.tool_name in SAFE_TOOLS for a in plan.actions)
        if all_safe and len(plan.actions) <= 3:
            return CritiqueResult(approved=True)

        # Skip for any confident plan
        if plan.confidence > 0.3 and len(plan.actions) <= 2:
            return CritiqueResult(approved=True)

        critique_prompt = f"""Review this action plan for the user query: "{original_query}"

Plan:
{json.dumps([a.to_dict() for a in plan.actions], indent=2)}

Respond with JSON:
{{"approved": true/false, "issues": ["list of issues if any"]}}

Check for:
1. Path traversal or unsafe paths
2. Unnecessary destructive operations
3. Missing dependencies between steps
4. Actions that don't match the user's intent"""

        try:
            result = await self.call_llm(
                [
                    {
                        "role": "system",
                        "content": "You are a safety reviewer. Respond with JSON only.",
                    },
                    {"role": "user", "content": critique_prompt},
                ]
            )
            if result:
                return CritiqueResult(
                    approved=result.get("approved", True),
                    issues=result.get("issues", []),
                )
        except Exception as e:
            logger.warning("Self-critique failed: %s", e)

        # BUG-14 FIX: fail-CLOSED for plans that contain non-safe tools
        if not all_safe:
            logger.warning(
                "Critique unavailable for non-trivial plan — defaulting to rejected"
            )
            return CritiqueResult(
                approved=False,
                issues=["Self-critique unavailable — manual review required"],
            )
        return CritiqueResult(approved=True)  # fail-open only for safe tools

    async def generate_response_text(self, context: str) -> str:
        """Generate a natural language response given context."""
        messages = [
            {
                "role": "system",
                "content": 'Generate a concise, natural spoken response. Respond with JSON: {"response": "..."}',
            },
            {"role": "user", "content": context},
        ]
        result = await self.call_llm(messages)
        if result:
            return result.get("response", "Done.")
        return "Done."
