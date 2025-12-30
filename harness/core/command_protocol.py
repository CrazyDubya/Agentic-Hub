"""
Universal Command Protocol
Unified command interface that works for both tool-using and non-tool-using LLMs.

Key insight: Commands can be expressed as:
1. Structured JSON (for tool-using LLMs with function calling)
2. Text blocks (for non-tool-using LLMs via parsing)
3. Natural language (parsed via intent recognition)

The protocol normalizes all three into a unified Command object.
"""

import re
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Callable
from enum import Enum
from abc import ABC, abstractmethod
import hashlib
from datetime import datetime


class CommandType(Enum):
    """Categories of commands available in the harness"""
    # File operations
    FILE_READ = "file.read"
    FILE_WRITE = "file.write"
    FILE_EDIT = "file.edit"
    FILE_LIST = "file.list"
    FILE_SEARCH = "file.search"

    # Shell/execution
    SHELL_EXEC = "shell.exec"
    SHELL_BACKGROUND = "shell.background"
    SHELL_KILL = "shell.kill"

    # Communication
    MSG_SEND = "msg.send"
    MSG_BROADCAST = "msg.broadcast"
    MSG_SUBSCRIBE = "msg.subscribe"

    # Sandbox
    SANDBOX_CREATE = "sandbox.create"
    SANDBOX_DESTROY = "sandbox.destroy"
    SANDBOX_SHARE = "sandbox.share"
    SANDBOX_SNAPSHOT = "sandbox.snapshot"
    SANDBOX_RESTORE = "sandbox.restore"

    # Skills
    SKILL_INVOKE = "skill.invoke"
    SKILL_INSTALL = "skill.install"
    SKILL_LIST = "skill.list"

    # Agent management
    AGENT_SPAWN = "agent.spawn"
    AGENT_QUERY = "agent.query"
    AGENT_TERMINATE = "agent.terminate"

    # Marketplace
    MARKET_SEARCH = "market.search"
    MARKET_INSTALL = "market.install"
    MARKET_PUBLISH = "market.publish"

    # State management
    STATE_GET = "state.get"
    STATE_SET = "state.set"
    STATE_WATCH = "state.watch"

    # Meta operations
    META_HELP = "meta.help"
    META_STATUS = "meta.status"
    META_COMPACT = "meta.compact"


@dataclass
class Command:
    """
    Universal command object.
    Can be constructed from tool calls, text blocks, or parsed natural language.
    """
    type: CommandType
    params: Dict[str, Any] = field(default_factory=dict)
    command_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    source_format: str = "unknown"  # "tool", "text", "natural"
    raw_input: str = ""

    def __post_init__(self):
        if not self.command_id:
            self.command_id = self._generate_id()

    def _generate_id(self) -> str:
        content = f"{self.type.value}-{self.timestamp.isoformat()}-{json.dumps(self.params, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def to_json(self) -> str:
        return json.dumps({
            "type": self.type.value,
            "params": self.params,
            "command_id": self.command_id,
            "timestamp": self.timestamp.isoformat()
        })

    def to_text_block(self) -> str:
        """Convert to text block format for non-tool LLMs"""
        lines = [f"```command:{self.type.value}"]
        for key, value in self.params.items():
            if isinstance(value, str) and '\n' in value:
                lines.append(f"{key}:")
                lines.append('"""')
                lines.append(value)
                lines.append('"""')
            else:
                lines.append(f"{key}: {json.dumps(value) if not isinstance(value, str) else value}")
        lines.append("```")
        return '\n'.join(lines)


@dataclass
class CommandResult:
    """Result of command execution"""
    command_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "command_id": self.command_id,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata
        })

    def to_text_block(self) -> str:
        """Convert to text block format for readability"""
        status = "SUCCESS" if self.success else "ERROR"
        lines = [f"```result:{status}"]
        lines.append(f"command_id: {self.command_id}")
        if self.error:
            lines.append(f"error: {self.error}")
        if self.output is not None:
            if isinstance(self.output, str):
                lines.append(f"output:\n{self.output}")
            else:
                lines.append(f"output: {json.dumps(self.output, indent=2)}")
        lines.append("```")
        return '\n'.join(lines)


class CommandParser(ABC):
    """Abstract base class for command parsers"""

    @abstractmethod
    def can_parse(self, input_data: Any) -> bool:
        """Check if this parser can handle the input"""
        pass

    @abstractmethod
    def parse(self, input_data: Any) -> Optional[Command]:
        """Parse input into Command"""
        pass


class ToolCallParser(CommandParser):
    """
    Parse structured tool calls (for tool-using LLMs).
    Handles OpenAI function calling, Anthropic tool use, etc.
    """

    # Mapping from tool names to command types
    TOOL_TO_COMMAND = {
        # File operations
        "read_file": CommandType.FILE_READ,
        "Read": CommandType.FILE_READ,
        "write_file": CommandType.FILE_WRITE,
        "Write": CommandType.FILE_WRITE,
        "edit_file": CommandType.FILE_EDIT,
        "Edit": CommandType.FILE_EDIT,
        "list_files": CommandType.FILE_LIST,
        "Glob": CommandType.FILE_LIST,
        "search_files": CommandType.FILE_SEARCH,
        "Grep": CommandType.FILE_SEARCH,

        # Shell
        "run_command": CommandType.SHELL_EXEC,
        "Bash": CommandType.SHELL_EXEC,
        "execute": CommandType.SHELL_EXEC,

        # etc. - extensible
    }

    def can_parse(self, input_data: Any) -> bool:
        if isinstance(input_data, dict):
            return "tool" in input_data or "function" in input_data or "name" in input_data
        return False

    def parse(self, input_data: Dict[str, Any]) -> Optional[Command]:
        # Handle different tool call formats
        tool_name = input_data.get("tool") or input_data.get("function") or input_data.get("name")
        params = input_data.get("arguments") or input_data.get("parameters") or input_data.get("input", {})

        if isinstance(params, str):
            try:
                params = json.loads(params)
            except:
                params = {"raw": params}

        # Map to command type
        cmd_type = self.TOOL_TO_COMMAND.get(tool_name)
        if not cmd_type:
            # Try to infer from name
            for key, value in self.TOOL_TO_COMMAND.items():
                if key.lower() in tool_name.lower():
                    cmd_type = value
                    break

        if not cmd_type:
            return None

        return Command(
            type=cmd_type,
            params=params,
            source_format="tool",
            raw_input=json.dumps(input_data)
        )


class TextBlockParser(CommandParser):
    """
    Parse text block commands (for non-tool-using LLMs).

    Format:
    ```command:file.read
    path: /path/to/file
    encoding: utf-8
    ```

    Or multiline values:
    ```command:file.write
    path: /path/to/file
    content:
    \"\"\"
    File content here
    Multi-line supported
    \"\"\"
    ```
    """

    # Pattern to match command blocks
    BLOCK_PATTERN = re.compile(
        r'```command:(\S+)\s*\n(.*?)```',
        re.DOTALL
    )

    def can_parse(self, input_data: Any) -> bool:
        if isinstance(input_data, str):
            return bool(self.BLOCK_PATTERN.search(input_data))
        return False

    def parse(self, input_data: str) -> Optional[List[Command]]:
        """Parse all command blocks from text"""
        commands = []

        for match in self.BLOCK_PATTERN.finditer(input_data):
            cmd_type_str = match.group(1)
            body = match.group(2).strip()

            try:
                cmd_type = CommandType(cmd_type_str)
            except ValueError:
                continue

            params = self._parse_params(body)

            commands.append(Command(
                type=cmd_type,
                params=params,
                source_format="text",
                raw_input=match.group(0)
            ))

        return commands if commands else None

    def _parse_params(self, body: str) -> Dict[str, Any]:
        """Parse parameter block"""
        params = {}
        lines = body.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            if not line or line.startswith('#'):
                i += 1
                continue

            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()

                # Check for multiline value
                if not value and i + 1 < len(lines) and lines[i + 1].strip() == '"""':
                    # Multiline string
                    i += 2
                    multiline_parts = []
                    while i < len(lines) and lines[i].strip() != '"""':
                        multiline_parts.append(lines[i])
                        i += 1
                    value = '\n'.join(multiline_parts)
                else:
                    # Try to parse as JSON for complex types
                    if value.startswith('{') or value.startswith('['):
                        try:
                            value = json.loads(value)
                        except:
                            pass
                    elif value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    elif value.isdigit():
                        value = int(value)

                params[key] = value

            i += 1

        return params


class NaturalLanguageParser(CommandParser):
    """
    Parse natural language into commands.
    Uses pattern matching and keyword extraction.

    Examples:
    - "Read the file at /path/to/file" -> FILE_READ
    - "Run 'npm install' in the terminal" -> SHELL_EXEC
    - "Send a message to agent-123" -> MSG_SEND
    """

    PATTERNS = [
        # File operations
        (r'(?:read|show|display|cat|view)\s+(?:the\s+)?(?:file\s+)?(?:at\s+)?["\']?([^\'"]+)["\']?',
         CommandType.FILE_READ, lambda m: {"path": m.group(1).strip()}),

        (r'(?:write|save|create)\s+(?:to\s+)?(?:file\s+)?["\']?([^\'"]+)["\']?\s*(?:with\s+content)?[:\s]+(.+)',
         CommandType.FILE_WRITE, lambda m: {"path": m.group(1).strip(), "content": m.group(2).strip()}),

        (r'(?:list|ls|show)\s+(?:files\s+)?(?:in\s+)?["\']?([^\'"]+)["\']?',
         CommandType.FILE_LIST, lambda m: {"path": m.group(1).strip()}),

        (r'(?:search|find|grep)\s+(?:for\s+)?["\']?([^\'"]+)["\']?\s+(?:in\s+)?["\']?([^\'"]+)?["\']?',
         CommandType.FILE_SEARCH, lambda m: {"pattern": m.group(1), "path": m.group(2) or "."}),

        # Shell operations
        (r'(?:run|execute|exec)\s+["\']?([^\'"]+)["\']?(?:\s+in\s+(?:the\s+)?terminal)?',
         CommandType.SHELL_EXEC, lambda m: {"command": m.group(1).strip()}),

        # Messages
        (r'(?:send|message)\s+(?:to\s+)?(?:agent\s+)?["\']?([^\'"]+)["\']?[:\s]+(.+)',
         CommandType.MSG_SEND, lambda m: {"to": m.group(1), "message": m.group(2)}),

        (r'broadcast[:\s]+(.+)',
         CommandType.MSG_BROADCAST, lambda m: {"message": m.group(1)}),

        # Sandbox
        (r'(?:create|new)\s+sandbox(?:\s+named?\s+)?["\']?([^\'"]*)["\']?',
         CommandType.SANDBOX_CREATE, lambda m: {"name": m.group(1) or None}),

        (r'(?:share|invite)\s+sandbox\s+(?:with\s+)?["\']?([^\'"]+)["\']?',
         CommandType.SANDBOX_SHARE, lambda m: {"with_agent": m.group(1)}),

        # Skills
        (r'(?:invoke|use|call)\s+(?:skill\s+)?["\']?([^\'"]+)["\']?(?:\s+with\s+(.+))?',
         CommandType.SKILL_INVOKE, lambda m: {"skill": m.group(1), "params": m.group(2) or "{}"}),

        (r'(?:install|add)\s+skill\s+["\']?([^\'"]+)["\']?',
         CommandType.SKILL_INSTALL, lambda m: {"skill": m.group(1)}),

        # Agent
        (r'(?:spawn|create|start)\s+(?:new\s+)?agent\s+["\']?([^\'"]+)["\']?',
         CommandType.AGENT_SPAWN, lambda m: {"agent_type": m.group(1)}),

        # Marketplace
        (r'(?:search|find)\s+(?:in\s+)?market(?:place)?(?:\s+for)?\s+["\']?([^\'"]+)["\']?',
         CommandType.MARKET_SEARCH, lambda m: {"query": m.group(1)}),

        # Status
        (r'(?:show\s+)?status',
         CommandType.META_STATUS, lambda m: {}),

        (r'help(?:\s+(?:with\s+)?(.+))?',
         CommandType.META_HELP, lambda m: {"topic": m.group(1) if m.group(1) else None}),
    ]

    def can_parse(self, input_data: Any) -> bool:
        if isinstance(input_data, str):
            # Check if it's not already a command block
            if '```command:' in input_data:
                return False
            return True
        return False

    def parse(self, input_data: str) -> Optional[Command]:
        text = input_data.strip().lower()

        for pattern, cmd_type, param_extractor in self.PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                params = param_extractor(match)
                return Command(
                    type=cmd_type,
                    params=params,
                    source_format="natural",
                    raw_input=input_data
                )

        return None


class UniversalCommandParser:
    """
    Master parser that tries all available parsers in order.
    This is the main entry point for parsing commands from any LLM.
    """

    def __init__(self):
        self.parsers: List[CommandParser] = [
            ToolCallParser(),
            TextBlockParser(),
            NaturalLanguageParser(),
        ]

    def parse(self, input_data: Any) -> List[Command]:
        """
        Parse input and return list of commands.
        Tries each parser in order until one succeeds.
        """
        for parser in self.parsers:
            if parser.can_parse(input_data):
                result = parser.parse(input_data)
                if result:
                    if isinstance(result, list):
                        return result
                    return [result]

        return []

    def add_parser(self, parser: CommandParser, priority: int = None):
        """Add a custom parser, optionally at specific priority"""
        if priority is not None:
            self.parsers.insert(priority, parser)
        else:
            self.parsers.append(parser)


# Example command templates for documentation
COMMAND_TEMPLATES = {
    CommandType.FILE_READ: '''```command:file.read
path: /path/to/file
encoding: utf-8
```''',

    CommandType.FILE_WRITE: '''```command:file.write
path: /path/to/file
content:
"""
Your file content here
Supports multiple lines
"""
```''',

    CommandType.SHELL_EXEC: '''```command:shell.exec
command: npm install
timeout: 30000
working_dir: /project
```''',

    CommandType.MSG_SEND: '''```command:msg.send
to: agent-id-here
message: Your message content
priority: normal
```''',

    CommandType.SKILL_INVOKE: '''```command:skill.invoke
skill: code-review
params: {"file": "/path/to/file.py"}
```''',

    CommandType.SANDBOX_CREATE: '''```command:sandbox.create
name: my-sandbox
template: python
resources: {"cpu": 1, "memory": "512MB"}
```''',
}


def generate_help_text() -> str:
    """Generate help text showing all available commands"""
    help_lines = [
        "# Universal Command Reference",
        "",
        "Commands can be issued in three formats:",
        "1. **Tool calls** - For LLMs with function calling",
        "2. **Text blocks** - For any LLM (```command:type ... ```)",
        "3. **Natural language** - Parsed automatically",
        "",
        "## Available Commands",
        ""
    ]

    categories = {}
    for cmd_type in CommandType:
        category = cmd_type.value.split('.')[0].upper()
        if category not in categories:
            categories[category] = []
        categories[category].append(cmd_type)

    for category, commands in categories.items():
        help_lines.append(f"### {category}")
        for cmd in commands:
            help_lines.append(f"- `{cmd.value}` - {cmd.name.replace('_', ' ').title()}")
            if cmd in COMMAND_TEMPLATES:
                help_lines.append(f"  ```\n{COMMAND_TEMPLATES[cmd]}  ```")
        help_lines.append("")

    return '\n'.join(help_lines)
