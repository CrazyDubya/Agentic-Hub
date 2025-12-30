"""
Unified Command Bus

Provides a single execution point for both text commands and tool calls.
This is the central router that dispatches to appropriate handlers.
"""

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import logging

from .types import (
    CommandResult,
    ToolResult,
    ParsedCommand,
    ExecutionMode,
    AgentConfig,
)
from .command_parser import CommandParser
from .tool_adapter import ToolAdapter

logger = logging.getLogger(__name__)


class RequestType(Enum):
    """Type of incoming request."""
    TEXT_COMMAND = auto()
    TOOL_CALL = auto()


@dataclass
class ExecutionRequest:
    """A unified execution request."""
    request_id: str
    request_type: RequestType
    agent_id: str

    # For text commands
    raw_text: Optional[str] = None
    parsed_command: Optional[ParsedCommand] = None

    # For tool calls
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None

    # Metadata
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResponse:
    """Response from execution."""
    request_id: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_command_result(self) -> CommandResult:
        """Convert to CommandResult for text LLMs."""
        return CommandResult(
            command=self.metadata.get("command", ""),
            success=self.success,
            output=self.result,
            error=self.error,
            execution_time_ms=self.execution_time_ms,
            metadata=self.metadata,
        )

    def to_tool_result(self) -> ToolResult:
        """Convert to ToolResult for tool-using LLMs."""
        return ToolResult(
            tool_name=self.metadata.get("tool_name", ""),
            success=self.success,
            result=self.result,
            error=self.error,
            execution_time_ms=self.execution_time_ms,
            metadata=self.metadata,
        )


class CommandHandler:
    """Base class for command handlers."""

    def __init__(self, name: str):
        self.name = name
        self.subhandlers: Dict[str, Callable] = {}

    def register_subhandler(self, subcommand: str, handler: Callable):
        """Register a handler for a subcommand."""
        self.subhandlers[subcommand] = handler

    async def execute(
        self,
        parsed: ParsedCommand,
        context: Dict[str, Any]
    ) -> Any:
        """Execute the command."""
        if parsed.subcommand and parsed.subcommand in self.subhandlers:
            handler = self.subhandlers[parsed.subcommand]
            if asyncio.iscoroutinefunction(handler):
                return await handler(parsed, context)
            return handler(parsed, context)
        return await self.handle(parsed, context)

    async def handle(self, parsed: ParsedCommand, context: Dict[str, Any]) -> Any:
        """Override in subclasses to handle the command."""
        raise NotImplementedError(f"Handler for {self.name} not implemented")


class UnifiedCommandBus:
    """
    Central bus for all command/tool execution.

    Routes requests to appropriate handlers regardless of whether
    they came from text commands or tool calls.
    """

    def __init__(self):
        self.command_parser = CommandParser()
        self.tool_adapter = ToolAdapter()
        self.handlers: Dict[str, CommandHandler] = {}
        self.middlewares: List[Callable] = []
        self.history: List[ExecutionRequest] = []
        self.max_history = 1000

    def register_handler(self, command: str, handler: CommandHandler):
        """Register a handler for a command."""
        self.handlers[command] = handler
        # Also register as tool handler
        tool_name = f"harness_{command}"
        self.tool_adapter.register_handler(tool_name, self._make_tool_handler(handler))

    def _make_tool_handler(self, handler: CommandHandler) -> Callable:
        """Create a tool handler from a command handler."""
        async def tool_handler(**kwargs):
            # Convert tool args to parsed command
            parsed = self.tool_adapter.tool_to_command(
                f"harness_{handler.name}",
                dict(kwargs)
            )
            parsed.command = handler.name
            return await handler.execute(parsed, {"source": "tool"})
        return tool_handler

    def add_middleware(self, middleware: Callable):
        """
        Add middleware that runs before each command.

        Middleware signature: async (request, next) -> response
        """
        self.middlewares.append(middleware)

    async def execute_text(
        self,
        text: str,
        agent_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ExecutionResponse]:
        """
        Execute text containing one or more commands.

        Args:
            text: Text potentially containing /commands
            agent_id: ID of the agent making the request
            context: Additional context

        Returns:
            List of execution responses
        """
        context = context or {}
        context["agent_id"] = agent_id
        context["source"] = "text"

        # Parse all commands from text
        parsed_commands = self.command_parser.parse_all(text)

        responses = []
        for parsed in parsed_commands:
            request = ExecutionRequest(
                request_id=f"req-{time.time_ns()}",
                request_type=RequestType.TEXT_COMMAND,
                agent_id=agent_id,
                raw_text=parsed.raw,
                parsed_command=parsed,
            )

            response = await self._execute(request, context)
            responses.append(response)

        return responses

    async def execute_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        agent_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResponse:
        """
        Execute a tool call.

        Args:
            tool_name: Name of the tool
            tool_args: Tool arguments
            agent_id: ID of the agent making the request
            context: Additional context

        Returns:
            Execution response
        """
        context = context or {}
        context["agent_id"] = agent_id
        context["source"] = "tool"

        request = ExecutionRequest(
            request_id=f"req-{time.time_ns()}",
            request_type=RequestType.TOOL_CALL,
            agent_id=agent_id,
            tool_name=tool_name,
            tool_args=tool_args,
        )

        return await self._execute(request, context)

    async def _execute(
        self,
        request: ExecutionRequest,
        context: Dict[str, Any]
    ) -> ExecutionResponse:
        """Execute a request through the middleware chain."""
        start = time.time()

        # Add to history
        self.history.append(request)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        try:
            # Run through middlewares
            async def final_handler(req: ExecutionRequest) -> ExecutionResponse:
                return await self._execute_core(req, context)

            handler = final_handler
            for middleware in reversed(self.middlewares):
                handler = self._wrap_middleware(middleware, handler)

            response = await handler(request)

        except Exception as e:
            logger.exception(f"Error executing request: {request.request_id}")
            response = ExecutionResponse(
                request_id=request.request_id,
                success=False,
                result=None,
                error=str(e),
            )

        response.execution_time_ms = (time.time() - start) * 1000
        return response

    def _wrap_middleware(
        self,
        middleware: Callable,
        next_handler: Callable
    ) -> Callable:
        """Wrap a middleware with the next handler."""
        async def wrapped(request: ExecutionRequest) -> ExecutionResponse:
            return await middleware(request, next_handler)
        return wrapped

    async def _execute_core(
        self,
        request: ExecutionRequest,
        context: Dict[str, Any]
    ) -> ExecutionResponse:
        """Core execution logic."""
        if request.request_type == RequestType.TEXT_COMMAND:
            return await self._execute_text_command(request, context)
        else:
            return await self._execute_tool_call(request, context)

    async def _execute_text_command(
        self,
        request: ExecutionRequest,
        context: Dict[str, Any]
    ) -> ExecutionResponse:
        """Execute a text command."""
        parsed = request.parsed_command

        if not parsed or not parsed.valid:
            return ExecutionResponse(
                request_id=request.request_id,
                success=False,
                result=None,
                error=parsed.error if parsed else "Failed to parse command",
                metadata={"command": parsed.raw if parsed else request.raw_text},
            )

        command = parsed.command

        if command not in self.handlers:
            # Check for built-in handlers
            if command == "help":
                return ExecutionResponse(
                    request_id=request.request_id,
                    success=True,
                    result=self.command_parser.get_help(
                        parsed.args[0] if parsed.args else None
                    ),
                    metadata={"command": parsed.raw},
                )

            return ExecutionResponse(
                request_id=request.request_id,
                success=False,
                result=None,
                error=f"Unknown command: {command}",
                metadata={"command": parsed.raw},
            )

        handler = self.handlers[command]

        try:
            result = await handler.execute(parsed, context)
            return ExecutionResponse(
                request_id=request.request_id,
                success=True,
                result=result,
                metadata={"command": parsed.raw, "handler": handler.name},
            )
        except Exception as e:
            return ExecutionResponse(
                request_id=request.request_id,
                success=False,
                result=None,
                error=str(e),
                metadata={"command": parsed.raw, "handler": handler.name},
            )

    async def _execute_tool_call(
        self,
        request: ExecutionRequest,
        context: Dict[str, Any]
    ) -> ExecutionResponse:
        """Execute a tool call."""
        tool_name = request.tool_name
        tool_args = request.tool_args or {}

        # Convert tool name to command
        cmd_name = tool_name.replace("harness_", "")

        if cmd_name in self.handlers:
            # Route through command handler
            parsed = self.tool_adapter.tool_to_command(tool_name, dict(tool_args))
            parsed.command = cmd_name

            handler = self.handlers[cmd_name]

            try:
                result = await handler.execute(parsed, context)
                return ExecutionResponse(
                    request_id=request.request_id,
                    success=True,
                    result=result,
                    metadata={"tool_name": tool_name, "handler": handler.name},
                )
            except Exception as e:
                return ExecutionResponse(
                    request_id=request.request_id,
                    success=False,
                    result=None,
                    error=str(e),
                    metadata={"tool_name": tool_name, "handler": handler.name},
                )

        # Try direct tool execution
        tool_result = await self.tool_adapter.execute(tool_name, tool_args)
        return ExecutionResponse(
            request_id=request.request_id,
            success=tool_result.success,
            result=tool_result.result,
            error=tool_result.error,
            metadata={"tool_name": tool_name},
        )

    def get_history(
        self,
        agent_id: Optional[str] = None,
        limit: int = 50
    ) -> List[ExecutionRequest]:
        """Get execution history."""
        history = self.history
        if agent_id:
            history = [r for r in history if r.agent_id == agent_id]
        return history[-limit:]

    def format_for_prompt(self) -> str:
        """
        Format command reference for inclusion in LLM prompts.

        This helps non-tool LLMs understand available commands.
        """
        return self.command_parser.get_help()
