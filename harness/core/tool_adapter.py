"""
Tool adapter for tool-using LLMs.

Provides a standardized tool interface that can be exposed to LLMs
supporting function/tool calls (OpenAI, Anthropic, etc.).
"""

from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
import json

from .types import ToolResult, ParsedCommand


@dataclass
class ToolSchema:
    """Schema for a tool that can be exposed to LLMs."""
    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str] = field(default_factory=list)
    handler: Optional[Callable] = None

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                }
            }
        }

    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic tool use format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required,
            }
        }


# Core harness tools - these map to the command system
HARNESS_TOOLS: Dict[str, ToolSchema] = {
    # Shell & Files
    "harness_shell": ToolSchema(
        name="harness_shell",
        description="Execute a shell command in the sandbox",
        parameters={
            "command": {
                "type": "string",
                "description": "The shell command to execute"
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 30)"
            }
        },
        required=["command"]
    ),

    "harness_read": ToolSchema(
        name="harness_read",
        description="Read a file or directory listing",
        parameters={
            "path": {
                "type": "string",
                "description": "Path to file or directory"
            },
            "lines": {
                "type": "integer",
                "description": "Maximum lines to read"
            },
            "offset": {
                "type": "integer",
                "description": "Line offset to start from"
            }
        },
        required=["path"]
    ),

    "harness_write": ToolSchema(
        name="harness_write",
        description="Write content to a file",
        parameters={
            "path": {
                "type": "string",
                "description": "Path to file"
            },
            "content": {
                "type": "string",
                "description": "Content to write"
            },
            "append": {
                "type": "boolean",
                "description": "Append instead of overwrite"
            }
        },
        required=["path", "content"]
    ),

    "harness_edit": ToolSchema(
        name="harness_edit",
        description="Edit a file by replacing text",
        parameters={
            "path": {
                "type": "string",
                "description": "Path to file"
            },
            "old_text": {
                "type": "string",
                "description": "Text to find and replace"
            },
            "new_text": {
                "type": "string",
                "description": "Replacement text"
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences"
            }
        },
        required=["path", "old_text", "new_text"]
    ),

    "harness_search": ToolSchema(
        name="harness_search",
        description="Search for files or content",
        parameters={
            "pattern": {
                "type": "string",
                "description": "Search pattern (regex or glob)"
            },
            "path": {
                "type": "string",
                "description": "Path to search in"
            },
            "file_type": {
                "type": "string",
                "description": "Filter by file type"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results"
            }
        },
        required=["pattern"]
    ),

    # Sandbox management
    "harness_sandbox": ToolSchema(
        name="harness_sandbox",
        description="Manage sandbox environments",
        parameters={
            "action": {
                "type": "string",
                "enum": ["list", "create", "switch", "share", "delete", "env", "state"],
                "description": "Action to perform"
            },
            "name": {
                "type": "string",
                "description": "Sandbox name"
            },
            "agents": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Agent IDs for sharing"
            },
            "sandbox_type": {
                "type": "string",
                "enum": ["personal", "shared", "ephemeral"],
                "description": "Type of sandbox to create"
            },
            "key": {
                "type": "string",
                "description": "Environment or state key"
            },
            "value": {
                "type": "string",
                "description": "Environment or state value"
            }
        },
        required=["action"]
    ),

    # Skills
    "harness_skill": ToolSchema(
        name="harness_skill",
        description="Work with skills",
        parameters={
            "action": {
                "type": "string",
                "enum": ["list", "info", "invoke", "install", "create", "remove"],
                "description": "Action to perform"
            },
            "name": {
                "type": "string",
                "description": "Skill name"
            },
            "args": {
                "type": "object",
                "description": "Arguments for skill invocation"
            },
            "category": {
                "type": "string",
                "description": "Filter by category"
            }
        },
        required=["action"]
    ),

    # Agent communication
    "harness_agent": ToolSchema(
        name="harness_agent",
        description="Communicate with other agents",
        parameters={
            "action": {
                "type": "string",
                "enum": ["list", "message", "broadcast", "query", "subscribe", "unsubscribe"],
                "description": "Action to perform"
            },
            "target": {
                "type": "string",
                "description": "Target agent ID"
            },
            "content": {
                "type": "string",
                "description": "Message content"
            },
            "pattern": {
                "type": "string",
                "description": "Subscription pattern"
            },
            "timeout": {
                "type": "integer",
                "description": "Query timeout in seconds"
            }
        },
        required=["action"]
    ),

    # Evaluation
    "harness_eval": ToolSchema(
        name="harness_eval",
        description="Self-evaluation and improvement",
        parameters={
            "action": {
                "type": "string",
                "enum": ["start", "record", "assess", "gaps", "improve", "report", "status", "history"],
                "description": "Action to perform"
            },
            "task": {
                "type": "string",
                "description": "Task description to record"
            },
            "eval_type": {
                "type": "string",
                "enum": ["task", "session", "periodic"],
                "description": "Type of evaluation cycle"
            },
            "format": {
                "type": "string",
                "enum": ["text", "json", "markdown"],
                "description": "Report format"
            }
        },
        required=["action"]
    ),

    # Q&A
    "harness_qa": ToolSchema(
        name="harness_qa",
        description="Question and answer system for learning",
        parameters={
            "action": {
                "type": "string",
                "enum": ["generate", "pending", "answer", "review", "learn", "export"],
                "description": "Action to perform"
            },
            "topic": {
                "type": "string",
                "description": "Topic for question generation"
            },
            "question_id": {
                "type": "string",
                "description": "Question ID for answer/review"
            },
            "answer": {
                "type": "string",
                "description": "Answer to submit"
            },
            "count": {
                "type": "integer",
                "description": "Number of questions to generate"
            }
        },
        required=["action"]
    ),

    # Marketplace
    "harness_market": ToolSchema(
        name="harness_market",
        description="Marketplace for skills, agents, and resources",
        parameters={
            "action": {
                "type": "string",
                "enum": ["search", "info", "install", "publish", "rate", "my"],
                "description": "Action to perform"
            },
            "query": {
                "type": "string",
                "description": "Search query"
            },
            "entry_id": {
                "type": "string",
                "description": "Marketplace entry ID"
            },
            "path": {
                "type": "string",
                "description": "Path for publishing"
            },
            "rating": {
                "type": "integer",
                "description": "Rating (1-5)"
            },
            "review": {
                "type": "string",
                "description": "Review text"
            },
            "entry_type": {
                "type": "string",
                "enum": ["skill", "agent", "template", "dataset", "model"],
                "description": "Filter by type"
            }
        },
        required=["action"]
    ),

    # Meta
    "harness_status": ToolSchema(
        name="harness_status",
        description="Get harness and agent status",
        parameters={
            "include": {
                "type": "array",
                "items": {"type": "string"},
                "description": "What to include: sandbox, skills, agents, eval, qa"
            }
        },
        required=[]
    ),

    "harness_config": ToolSchema(
        name="harness_config",
        description="Get or set configuration",
        parameters={
            "action": {
                "type": "string",
                "enum": ["get", "set", "list"],
                "description": "Action to perform"
            },
            "key": {
                "type": "string",
                "description": "Config key"
            },
            "value": {
                "type": "string",
                "description": "Config value"
            }
        },
        required=["action"]
    ),
}


class ToolAdapter:
    """
    Adapter that handles tool calls from LLMs and routes them
    to the appropriate harness components.
    """

    def __init__(self, harness: Optional[Any] = None):
        """
        Initialize the adapter.

        Args:
            harness: Reference to the UniversalHarness instance
        """
        self.harness = harness
        self.tools = dict(HARNESS_TOOLS)
        self.handlers: Dict[str, Callable] = {}

    def register_handler(self, tool_name: str, handler: Callable):
        """Register a handler for a tool."""
        if tool_name in self.tools:
            self.tools[tool_name].handler = handler
        self.handlers[tool_name] = handler

    def get_all_tools(self, format: str = "openai") -> List[Dict[str, Any]]:
        """
        Get all tool schemas in the specified format.

        Args:
            format: "openai" or "anthropic"

        Returns:
            List of tool schemas
        """
        result = []
        for tool in self.tools.values():
            if format == "openai":
                result.append(tool.to_openai_format())
            elif format == "anthropic":
                result.append(tool.to_anthropic_format())
            else:
                raise ValueError(f"Unknown format: {format}")
        return result

    def parse_tool_call(
        self,
        tool_call: Dict[str, Any],
        format: str = "openai"
    ) -> tuple[str, Dict[str, Any]]:
        """
        Parse a tool call from an LLM response.

        Args:
            tool_call: The tool call from the LLM
            format: "openai" or "anthropic"

        Returns:
            Tuple of (tool_name, arguments)
        """
        if format == "openai":
            name = tool_call.get("function", {}).get("name", "")
            args_str = tool_call.get("function", {}).get("arguments", "{}")
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        elif format == "anthropic":
            name = tool_call.get("name", "")
            args = tool_call.get("input", {})
        else:
            raise ValueError(f"Unknown format: {format}")

        return name, args

    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> ToolResult:
        """
        Execute a tool call.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            ToolResult with execution results
        """
        import time

        start = time.time()

        if tool_name not in self.tools:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"Unknown tool: {tool_name}"
            )

        handler = self.handlers.get(tool_name) or self.tools[tool_name].handler

        if not handler:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=f"No handler for tool: {tool_name}"
            )

        try:
            # Check if handler is async
            import asyncio
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**arguments)
            else:
                result = handler(**arguments)

            elapsed = (time.time() - start) * 1000

            return ToolResult(
                tool_name=tool_name,
                success=True,
                result=result,
                execution_time_ms=elapsed
            )

        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e),
                execution_time_ms=elapsed
            )

    def tool_to_command(self, tool_name: str, arguments: Dict[str, Any]) -> ParsedCommand:
        """
        Convert a tool call to equivalent command format.

        This allows unified handling of both text commands and tool calls.
        """
        # Map tool names to commands
        tool_cmd_map = {
            "harness_shell": "cmd",
            "harness_read": "read",
            "harness_write": "write",
            "harness_edit": "edit",
            "harness_search": "search",
            "harness_sandbox": "sandbox",
            "harness_skill": "skill",
            "harness_agent": "agent",
            "harness_eval": "eval",
            "harness_qa": "qa",
            "harness_market": "market",
            "harness_status": "status",
            "harness_config": "config",
        }

        cmd = tool_cmd_map.get(tool_name, tool_name.replace("harness_", ""))

        # Extract subcommand from action if present
        subcommand = arguments.pop("action", None)

        # Build args list from remaining arguments
        args = []
        flags = {}

        for key, value in arguments.items():
            if isinstance(value, bool):
                if value:
                    flags[key] = True
            elif value is not None:
                flags[key] = value

        return ParsedCommand(
            raw=f"/{cmd} {subcommand or ''} {arguments}",
            command=cmd,
            subcommand=subcommand,
            args=args,
            flags=flags,
            valid=True
        )


def get_tool_descriptions() -> str:
    """
    Get human-readable descriptions of all tools.

    Useful for including in system prompts for LLMs.
    """
    lines = ["# Available Harness Tools\n"]

    for name, tool in HARNESS_TOOLS.items():
        lines.append(f"## {name}")
        lines.append(f"{tool.description}\n")
        lines.append("Parameters:")
        for param, details in tool.parameters.items():
            required = " (required)" if param in tool.required else ""
            desc = details.get("description", "")
            lines.append(f"  - {param}: {desc}{required}")
        lines.append("")

    return '\n'.join(lines)
