"""
Command parser for text-based LLM interaction.

Parses commands in the format:
    /command [subcommand] [--flag value] [args...]

This enables non-tool-using LLMs to interact with the harness
through natural text output containing embedded commands.
"""

import re
import shlex
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field

from .types import ParsedCommand


# Command definitions with their expected structure
COMMAND_DEFINITIONS = {
    # Shell & Files
    "cmd": {"args": ["command"], "flags": {"timeout": int}},
    "read": {"args": ["path"], "flags": {"lines": int, "offset": int}},
    "write": {"args": ["path"], "flags": {"content": str, "append": bool}},
    "edit": {"args": ["path"], "flags": {"old": str, "new": str, "all": bool}},
    "search": {"args": ["pattern"], "optional_args": ["path"], "flags": {"type": str, "limit": int}},
    "tree": {"args": [], "optional_args": ["path"], "flags": {"depth": int}},

    # Sandbox
    "sandbox": {
        "subcommands": {
            "list": {"args": []},
            "create": {"args": ["name"], "flags": {"type": str}},
            "switch": {"args": ["name"]},
            "share": {"args": ["name", "agents"]},
            "delete": {"args": ["name"], "flags": {"force": bool}},
            "env": {"args": ["action"], "optional_args": ["key", "value"]},
            "state": {"args": ["action"], "optional_args": ["key", "value"]},
        }
    },

    # Skills
    "skill": {
        "subcommands": {
            "list": {"flags": {"category": str}},
            "info": {"args": ["name"]},
            "invoke": {"args": ["name"], "varargs": True},
            "install": {"args": ["name"], "flags": {"version": str}},
            "create": {"args": ["name"]},
            "remove": {"args": ["name"]},
        }
    },

    # Agent Communication
    "agent": {
        "subcommands": {
            "list": {"flags": {"status": str}},
            "msg": {"args": ["agent", "message"]},
            "broadcast": {"args": ["message"]},
            "query": {"args": ["agent", "question"], "flags": {"timeout": int}},
            "subscribe": {"args": ["pattern"]},
            "unsubscribe": {"args": ["pattern"]},
        }
    },

    # Self-Evaluation
    "eval": {
        "subcommands": {
            "start": {"flags": {"type": str}},
            "record": {"args": ["task"]},
            "assess": {},
            "gaps": {},
            "improve": {},
            "report": {"flags": {"format": str}},
            "status": {},
            "history": {"flags": {"limit": int}},
        }
    },

    # Q&A System
    "qa": {
        "subcommands": {
            "generate": {"optional_args": ["topic"], "flags": {"count": int}},
            "pending": {"flags": {"limit": int}},
            "answer": {"args": ["id", "answer"]},
            "review": {"args": ["id"]},
            "learn": {},
            "export": {"flags": {"format": str}},
        }
    },

    # Marketplace
    "market": {
        "subcommands": {
            "search": {"args": ["query"], "flags": {"type": str, "limit": int}},
            "info": {"args": ["id"]},
            "install": {"args": ["id"], "flags": {"sandbox": str}},
            "publish": {"args": ["path"]},
            "rate": {"args": ["id", "rating"], "optional_args": ["review"]},
            "my": {},
        }
    },

    # Meta
    "help": {"optional_args": ["topic"]},
    "status": {},
    "config": {"args": ["action"], "optional_args": ["key", "value"]},
    "history": {"flags": {"limit": int}},
    "clear": {},
    "version": {},
}


class CommandParser:
    """
    Parser for text-based commands.

    Extracts commands from LLM text output and parses them into
    structured form for execution.
    """

    # Pattern to find commands in text
    COMMAND_PATTERN = re.compile(
        r'(?:^|\s)/([a-z_]+)(?:\s+(.+?))?(?=\s*/[a-z_]|$)',
        re.MULTILINE | re.DOTALL
    )

    # Simpler pattern for single command parsing
    SINGLE_COMMAND_PATTERN = re.compile(r'^/([a-z_]+)(?:\s+(.*))?$', re.DOTALL)

    def __init__(self):
        self.definitions = COMMAND_DEFINITIONS

    def extract_commands(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract all commands from text.

        Returns list of (command_name, rest_of_line) tuples.
        """
        commands = []

        # Split by lines and look for commands
        lines = text.split('\n')
        current_command = None
        current_args = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith('/'):
                # Save previous command if exists
                if current_command:
                    commands.append((current_command, ' '.join(current_args)))

                # Parse new command
                match = self.SINGLE_COMMAND_PATTERN.match(stripped)
                if match:
                    current_command = match.group(1)
                    current_args = [match.group(2)] if match.group(2) else []
                else:
                    current_command = None
                    current_args = []
            elif current_command and stripped:
                # Continuation of multi-line argument
                current_args.append(stripped)

        # Don't forget the last command
        if current_command:
            commands.append((current_command, ' '.join(current_args)))

        return commands

    def parse(self, text: str) -> ParsedCommand:
        """
        Parse a single command string.

        Args:
            text: Command text like "/read file.txt --lines 10"

        Returns:
            ParsedCommand with parsed components
        """
        text = text.strip()

        # Must start with /
        if not text.startswith('/'):
            return ParsedCommand(
                raw=text,
                command="",
                valid=False,
                error="Command must start with /"
            )

        # Remove leading /
        text = text[1:]

        # Split into tokens
        try:
            tokens = shlex.split(text)
        except ValueError as e:
            return ParsedCommand(
                raw=f"/{text}",
                command="",
                valid=False,
                error=f"Parse error: {e}"
            )

        if not tokens:
            return ParsedCommand(
                raw=f"/{text}",
                command="",
                valid=False,
                error="Empty command"
            )

        command = tokens[0].lower()
        rest = tokens[1:]

        # Check if command exists
        if command not in self.definitions:
            return ParsedCommand(
                raw=f"/{text}",
                command=command,
                valid=False,
                error=f"Unknown command: {command}"
            )

        definition = self.definitions[command]

        # Parse subcommand if applicable
        subcommand = None
        if "subcommands" in definition and rest:
            potential_sub = rest[0].lower()
            if potential_sub in definition["subcommands"]:
                subcommand = potential_sub
                rest = rest[1:]
                definition = definition["subcommands"][subcommand]

        # Parse flags and args
        args = []
        flags = {}
        i = 0

        while i < len(rest):
            token = rest[i]

            if token.startswith('--'):
                # Long flag
                flag_name = token[2:]
                if '=' in flag_name:
                    # --flag=value
                    flag_name, flag_value = flag_name.split('=', 1)
                    flags[flag_name] = self._parse_flag_value(flag_value)
                elif i + 1 < len(rest) and not rest[i + 1].startswith('-'):
                    # --flag value
                    flags[flag_name] = self._parse_flag_value(rest[i + 1])
                    i += 1
                else:
                    # --flag (boolean)
                    flags[flag_name] = True
            elif token.startswith('-') and len(token) == 2:
                # Short flag
                flag_name = token[1]
                if i + 1 < len(rest) and not rest[i + 1].startswith('-'):
                    flags[flag_name] = self._parse_flag_value(rest[i + 1])
                    i += 1
                else:
                    flags[flag_name] = True
            else:
                # Positional argument
                args.append(token)

            i += 1

        return ParsedCommand(
            raw=f"/{text}",
            command=command,
            subcommand=subcommand,
            args=args,
            flags=flags,
            valid=True
        )

    def _parse_flag_value(self, value: str) -> Any:
        """Parse a flag value to appropriate type."""
        # Boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False

        # Integer
        try:
            return int(value)
        except ValueError:
            pass

        # Float
        try:
            return float(value)
        except ValueError:
            pass

        # String
        return value

    def parse_all(self, text: str) -> List[ParsedCommand]:
        """
        Extract and parse all commands from text.

        Args:
            text: Text potentially containing multiple commands

        Returns:
            List of ParsedCommand objects
        """
        extracted = self.extract_commands(text)
        results = []

        for cmd_name, cmd_args in extracted:
            full_cmd = f"/{cmd_name}"
            if cmd_args:
                full_cmd += f" {cmd_args}"
            results.append(self.parse(full_cmd))

        return results

    def get_help(self, command: Optional[str] = None) -> str:
        """Get help text for commands."""
        if command is None:
            # General help
            lines = ["Available commands:\n"]
            for cmd in sorted(self.definitions.keys()):
                lines.append(f"  /{cmd}")
            lines.append("\nUse /help <command> for details.")
            return '\n'.join(lines)

        if command not in self.definitions:
            return f"Unknown command: {command}"

        definition = self.definitions[command]
        lines = [f"/{command}"]

        if "subcommands" in definition:
            lines.append("\nSubcommands:")
            for sub, sub_def in definition["subcommands"].items():
                args_str = ' '.join(f"<{a}>" for a in sub_def.get("args", []))
                opt_args_str = ' '.join(f"[{a}]" for a in sub_def.get("optional_args", []))
                lines.append(f"  {sub} {args_str} {opt_args_str}".strip())
        else:
            args_str = ' '.join(f"<{a}>" for a in definition.get("args", []))
            opt_args_str = ' '.join(f"[{a}]" for a in definition.get("optional_args", []))
            if args_str or opt_args_str:
                lines[0] += f" {args_str} {opt_args_str}".strip()

            if "flags" in definition:
                lines.append("\nFlags:")
                for flag, ftype in definition["flags"].items():
                    lines.append(f"  --{flag} <{ftype.__name__}>")

        return '\n'.join(lines)


class MultiLineCommandBuilder:
    """
    Helper for building commands with multi-line content.

    Some commands (like /write) may have content that spans multiple lines.
    This helper accumulates lines until the command is complete.
    """

    def __init__(self):
        self.buffer: List[str] = []
        self.in_command = False
        self.current_command = ""
        self.delimiter = "EOF"  # Heredoc-style delimiter

    def add_line(self, line: str) -> Optional[str]:
        """
        Add a line to the builder.

        Returns the complete command if finished, None otherwise.
        """
        if not self.in_command:
            if line.strip().startswith('/'):
                # Check for heredoc start
                if '<<' in line:
                    # Extract delimiter
                    parts = line.split('<<')
                    self.current_command = parts[0]
                    self.delimiter = parts[1].strip().strip("'\"")
                    self.in_command = True
                    self.buffer = []
                    return None
                else:
                    # Single line command
                    return line.strip()
            return None
        else:
            # In multi-line mode
            if line.strip() == self.delimiter:
                # End of heredoc
                content = '\n'.join(self.buffer)
                result = f"{self.current_command} --content '{content}'"
                self.in_command = False
                self.buffer = []
                return result
            else:
                self.buffer.append(line)
                return None

    def reset(self):
        """Reset the builder state."""
        self.buffer = []
        self.in_command = False
        self.current_command = ""
