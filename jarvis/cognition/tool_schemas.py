"""
jarvis.cognition.tool_schemas — Tool definitions available to the LLM.

Each tool has a name, description, parameter schema, and risk hints.
The registry generates the JSON schema block injected into the system prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolParameter:
    name: str
    type: str  # "string", "integer", "boolean", "array"
    description: str
    required: bool = True
    enum: list[str] | None = None


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: list[ToolParameter]
    risk_hints: list[str] = field(
        default_factory=list
    )  # e.g. ["destructive", "network"]

    def to_schema(self) -> dict:
        props = {}
        required = []
        for p in self.parameters:
            prop: dict[str, Any] = {"type": p.type, "description": p.description}
            if p.enum:
                prop["enum"] = p.enum
            props[p.name] = prop
            if p.required:
                required.append(p.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        }


# ── Built-in tool definitions ──────────────────────────────────────────────

BUILTIN_TOOLS: list[ToolDefinition] = [
    ToolDefinition(
        name="file_read",
        description="Read the contents of a file at the given path.",
        parameters=[
            ToolParameter("path", "string", "Absolute file path to read"),
        ],
        risk_hints=["read_only"],
    ),
    ToolDefinition(
        name="file_write",
        description="Write content to a file. Creates the file if it does not exist.",
        parameters=[
            ToolParameter("path", "string", "Absolute file path to write"),
            ToolParameter("content", "string", "Text content to write"),
        ],
        risk_hints=["write"],
    ),
    ToolDefinition(
        name="file_delete",
        description="Move a file or directory to the recycle bin (safe delete).",
        parameters=[
            ToolParameter("path", "string", "Absolute path to delete"),
            ToolParameter(
                "recursive",
                "boolean",
                "If true, delete directory recursively",
                required=False,
            ),
        ],
        risk_hints=["destructive"],
    ),
    ToolDefinition(
        name="file_move",
        description="Move or rename a file or directory.",
        parameters=[
            ToolParameter("source", "string", "Source path"),
            ToolParameter("destination", "string", "Destination path"),
        ],
        risk_hints=["write"],
    ),
    ToolDefinition(
        name="file_copy",
        description="Copy a file or directory.",
        parameters=[
            ToolParameter("source", "string", "Source path"),
            ToolParameter("destination", "string", "Destination path"),
        ],
        risk_hints=["write"],
    ),
    ToolDefinition(
        name="search_files",
        description="Search for files matching a pattern in a directory.",
        parameters=[
            ToolParameter(
                "pattern", "string", "Search pattern (glob or ripgrep pattern)"
            ),
            ToolParameter("directory", "string", "Directory to search in"),
            ToolParameter(
                "max_results", "integer", "Maximum results to return", required=False
            ),
        ],
        risk_hints=["read_only"],
    ),
    ToolDefinition(
        name="search_content",
        description="Search file contents using ripgrep.",
        parameters=[
            ToolParameter("query", "string", "Text or regex to search for"),
            ToolParameter("directory", "string", "Directory to search in"),
            ToolParameter(
                "file_type",
                "string",
                "File extension filter (e.g. 'py', 'txt')",
                required=False,
            ),
        ],
        risk_hints=["read_only"],
    ),
    ToolDefinition(
        name="execute_code",
        description="Execute code in a sandboxed Docker container.",
        parameters=[
            ToolParameter(
                "language",
                "string",
                "Programming language",
                enum=["python", "bash", "javascript"],
            ),
            ToolParameter("code", "string", "Source code to execute"),
        ],
        risk_hints=["execution", "sandbox"],
    ),
    ToolDefinition(
        name="browser_navigate",
        description="Navigate to a URL in a browser.",
        parameters=[
            ToolParameter("url", "string", "URL to navigate to"),
            ToolParameter(
                "action",
                "string",
                "Action to perform after navigation",
                required=False,
                enum=["screenshot", "get_text", "wait"],
            ),
        ],
        risk_hints=["network", "external"],
    ),
    ToolDefinition(
        name="browser_interact",
        description="Interact with a web page element.",
        parameters=[
            ToolParameter("selector", "string", "CSS selector of the element"),
            ToolParameter(
                "action", "string", "Interaction type", enum=["click", "type", "scroll"]
            ),
            ToolParameter(
                "text", "string", "Text to type (for 'type' action)", required=False
            ),
        ],
        risk_hints=["network", "write"],
    ),
    ToolDefinition(
        name="system_info",
        description="Get system information (CPU, memory, disk, processes).",
        parameters=[
            ToolParameter(
                "category",
                "string",
                "Info category",
                enum=["cpu", "memory", "disk", "processes", "all"],
            ),
        ],
        risk_hints=["read_only"],
    ),
    ToolDefinition(
        name="set_reminder",
        description="Set a timed reminder.",
        parameters=[
            ToolParameter("message", "string", "Reminder message"),
            ToolParameter("delay_seconds", "integer", "Seconds from now"),
        ],
        risk_hints=["benign"],
    ),
    ToolDefinition(
        name="store_memory",
        description="Store a fact, preference, or note in long-term memory.",
        parameters=[
            ToolParameter("content", "string", "The information to remember"),
            ToolParameter(
                "type", "string", "Memory type", enum=["fact", "preference", "skill"]
            ),
        ],
        risk_hints=["benign"],
    ),
    ToolDefinition(
        name="app_launch",
        description="Launch a desktop application by name (e.g. 'chrome', 'notepad', 'calculator', 'explorer').",
        parameters=[
            ToolParameter(
                "target",
                "string",
                "Application name to launch (e.g. 'chrome', 'notepad', 'calc')",
            ),
        ],
        risk_hints=["benign"],
    ),
    ToolDefinition(
        name="get_time",
        description="Get the current date and time. Use this whenever the user asks about the time or date.",
        parameters=[],
        risk_hints=["read_only"],
    ),
]


class ToolRegistry:
    """Dynamic tool registry with schema generation for LLM prompt injection."""

    def __init__(self):
        self.tools: dict[str, ToolDefinition] = {}
        for tool in BUILTIN_TOOLS:
            self.register(tool)

    def register(self, tool: ToolDefinition):
        self.tools[tool.name] = tool

    def unregister(self, name: str):
        self.tools.pop(name, None)

    def get(self, name: str) -> ToolDefinition | None:
        return self.tools.get(name)

    def all_schemas(self) -> list[dict]:
        return [t.to_schema() for t in self.tools.values()]

    def tool_names(self) -> list[str]:
        return list(self.tools.keys())

    def to_prompt_block(self) -> str:
        """Generate the tool definitions block for the system prompt."""
        import json

        schemas = self.all_schemas()
        return json.dumps(schemas, indent=2)
