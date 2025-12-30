"""
Universal LLM Agent Harness
The core orchestration layer that enables ANY LLM to function as a capable agent.

Key design principles:
1. Works with both tool-using and non-tool-using LLMs
2. Provides unified interface regardless of LLM backend
3. Maintains state across interactions (via todo-list attention trick from Manus)
4. Supports compaction for infinite context (inspired by Codex)
5. Layered prompt architecture (inspired by Copilot)

This harness is designed to be at least as capable as:
- Claude Code
- Manus
- OpenAI Codex
- GitHub Copilot Agent Mode
"""

import json
import time
import uuid
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from datetime import datetime
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod

# Import harness components
from .command_protocol import (
    Command, CommandResult, CommandType,
    UniversalCommandParser, generate_help_text
)
from ..sandbox.sandbox_manager import (
    SandboxManager, Sandbox, SandboxType, ResourceLimits
)
from ..skills.skill_system import (
    SkillRegistry, Skill, SkillResult, SkillMetadata
)
from ..marketplace.registry import (
    AgentDirectory, Marketplace, AgentProfile, AgentCapabilityLevel
)
from ..communication.message_bus import (
    MessageBus, Message, MessageType as MsgType, AgentMailbox,
    format_inbox_for_llm
)


class LLMType(Enum):
    """Classification of LLM capabilities"""
    TOOL_NATIVE = "tool_native"      # Native function/tool calling (Claude, GPT-4)
    TEXT_COMMAND = "text_command"     # Text-based command blocks
    NATURAL_ONLY = "natural_only"     # Natural language only (parsed)
    HYBRID = "hybrid"                 # Supports multiple modes


class HarnessMode(Enum):
    """Harness operational modes"""
    INTERACTIVE = "interactive"   # Single-turn interactions
    AUTONOMOUS = "autonomous"     # Multi-step autonomous execution
    COLLABORATIVE = "collaborative"  # Multi-agent collaboration
    SUPERVISED = "supervised"     # Human-in-the-loop


@dataclass
class AgentState:
    """
    Current state of an agent in the harness.
    This gets persisted and can be restored.
    """
    agent_id: str
    session_id: str
    created_at: datetime
    last_active: datetime

    # Execution state
    current_task: Optional[str] = None
    task_stack: List[str] = field(default_factory=list)
    completed_tasks: List[str] = field(default_factory=list)

    # Todo list (Manus-style attention management)
    todos: List[Dict[str, Any]] = field(default_factory=list)

    # Working memory
    working_memory: Dict[str, Any] = field(default_factory=dict)
    short_term_memory: List[Dict[str, Any]] = field(default_factory=list)

    # Context management
    context_tokens_used: int = 0
    compaction_count: int = 0

    # Metrics
    commands_executed: int = 0
    skills_invoked: int = 0
    messages_sent: int = 0
    errors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "current_task": self.current_task,
            "todos": self.todos,
            "working_memory": self.working_memory,
            "metrics": {
                "commands_executed": self.commands_executed,
                "skills_invoked": self.skills_invoked,
                "messages_sent": self.messages_sent,
                "errors": self.errors
            }
        }


@dataclass
class ExecutionContext:
    """
    Context provided to the LLM for each execution turn.
    This is the "harness" that wraps around the LLM.
    """
    # Identity
    agent_id: str
    session_id: str

    # Current state
    state: AgentState

    # Available resources
    sandbox: Optional[Sandbox]
    available_skills: List[SkillMetadata]
    available_agents: List[AgentProfile]

    # Messages and events
    pending_messages: List[Message]
    recent_events: List[Dict[str, Any]]

    # Task context
    original_task: str
    current_objective: str
    constraints: List[str]

    # System info
    timestamp: datetime
    harness_mode: HarnessMode


class LLMBackend(ABC):
    """
    Abstract backend for LLM interactions.
    Implement this for different LLM providers.
    """

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """Generate a response from the LLM"""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        pass

    @property
    @abstractmethod
    def supports_tools(self) -> bool:
        """Whether this backend supports native tool calling"""
        pass

    @property
    @abstractmethod
    def max_context_tokens(self) -> int:
        """Maximum context window size"""
        pass


class MockLLMBackend(LLMBackend):
    """
    Mock backend for testing ONLY.

    DEPRECATED: For actual LLM connections, use the real backends:

        from harness.adapters import create_backend

        # Anthropic (Claude 4.x)
        backend = create_backend("anthropic", model="claude-sonnet-4-20250514")

        # OpenAI (GPT-5, GPT-4o)
        backend = create_backend("openai", model="gpt-4o")

        # OpenRouter (400+ models)
        backend = create_backend("openrouter", model="deepseek/deepseek-v3")

        # Ollama (local models)
        backend = create_backend("ollama", model="llama3.3:70b")

    See harness/adapters/ for real implementations.
    """

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        import warnings
        warnings.warn(
            "MockLLMBackend is for testing only. "
            "Use harness.adapters.create_backend() for real LLM connections.",
            DeprecationWarning,
            stacklevel=2
        )
        return {
            "content": "[MOCK] I'll help you with that task.",
            "tool_calls": []
        }

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    @property
    def supports_tools(self) -> bool:
        return True

    @property
    def max_context_tokens(self) -> int:
        return 128000


class CommandExecutor:
    """
    Executes commands from any source (tool calls, text blocks, natural language).
    This is the bridge between LLM output and system actions.
    """

    def __init__(
        self,
        sandbox_manager: SandboxManager,
        skill_registry: SkillRegistry,
        message_bus: MessageBus,
        agent_directory: AgentDirectory,
        marketplace: Marketplace
    ):
        self.sandbox_manager = sandbox_manager
        self.skill_registry = skill_registry
        self.message_bus = message_bus
        self.agent_directory = agent_directory
        self.marketplace = marketplace
        self.parser = UniversalCommandParser()

    def execute(
        self,
        input_data: Any,
        agent_id: str,
        sandbox_id: str = None
    ) -> List[CommandResult]:
        """Execute commands from parsed input"""
        commands = self.parser.parse(input_data)
        results = []

        for command in commands:
            result = self._execute_single(command, agent_id, sandbox_id)
            results.append(result)

        return results

    def _execute_single(
        self,
        command: Command,
        agent_id: str,
        sandbox_id: str = None
    ) -> CommandResult:
        """Execute a single command"""
        start_time = time.time()

        try:
            # Route to appropriate handler
            handlers = {
                CommandType.FILE_READ: self._handle_file_read,
                CommandType.FILE_WRITE: self._handle_file_write,
                CommandType.FILE_EDIT: self._handle_file_edit,
                CommandType.FILE_LIST: self._handle_file_list,
                CommandType.FILE_SEARCH: self._handle_file_search,
                CommandType.SHELL_EXEC: self._handle_shell_exec,
                CommandType.MSG_SEND: self._handle_msg_send,
                CommandType.MSG_BROADCAST: self._handle_msg_broadcast,
                CommandType.MSG_SUBSCRIBE: self._handle_msg_subscribe,
                CommandType.SANDBOX_CREATE: self._handle_sandbox_create,
                CommandType.SANDBOX_SHARE: self._handle_sandbox_share,
                CommandType.SANDBOX_SNAPSHOT: self._handle_sandbox_snapshot,
                CommandType.SKILL_INVOKE: self._handle_skill_invoke,
                CommandType.SKILL_LIST: self._handle_skill_list,
                CommandType.AGENT_SPAWN: self._handle_agent_spawn,
                CommandType.AGENT_QUERY: self._handle_agent_query,
                CommandType.MARKET_SEARCH: self._handle_market_search,
                CommandType.MARKET_INSTALL: self._handle_market_install,
                CommandType.STATE_GET: self._handle_state_get,
                CommandType.STATE_SET: self._handle_state_set,
                CommandType.META_HELP: self._handle_help,
                CommandType.META_STATUS: self._handle_status,
            }

            handler = handlers.get(command.type)
            if not handler:
                return CommandResult(
                    command_id=command.command_id,
                    success=False,
                    error=f"No handler for command type: {command.type.value}",
                    execution_time_ms=(time.time() - start_time) * 1000
                )

            output = handler(command.params, agent_id, sandbox_id)

            return CommandResult(
                command_id=command.command_id,
                success=True,
                output=output,
                execution_time_ms=(time.time() - start_time) * 1000
            )

        except Exception as e:
            return CommandResult(
                command_id=command.command_id,
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000
            )

    # Command handlers
    def _handle_file_read(self, params: Dict, agent_id: str, sandbox_id: str) -> Any:
        path = params.get("path")
        if sandbox_id:
            sandbox = self.sandbox_manager.get_sandbox(sandbox_id, agent_id)
            if sandbox:
                full_path = sandbox.root_path / "workspace" / path.lstrip("/")
                return full_path.read_text()
        return Path(path).read_text()

    def _handle_file_write(self, params: Dict, agent_id: str, sandbox_id: str) -> Any:
        path = params.get("path")
        content = params.get("content")
        if sandbox_id:
            sandbox = self.sandbox_manager.get_sandbox(sandbox_id, agent_id)
            if sandbox:
                full_path = sandbox.root_path / "workspace" / path.lstrip("/")
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content)
                return {"written": str(full_path), "size": len(content)}
        Path(path).write_text(content)
        return {"written": path, "size": len(content)}

    def _handle_file_edit(self, params: Dict, agent_id: str, sandbox_id: str) -> Any:
        path = params.get("path")
        old_string = params.get("old_string")
        new_string = params.get("new_string")

        if sandbox_id:
            sandbox = self.sandbox_manager.get_sandbox(sandbox_id, agent_id)
            if sandbox:
                full_path = sandbox.root_path / "workspace" / path.lstrip("/")
                content = full_path.read_text()
                content = content.replace(old_string, new_string)
                full_path.write_text(content)
                return {"edited": str(full_path)}
        content = Path(path).read_text()
        content = content.replace(old_string, new_string)
        Path(path).write_text(content)
        return {"edited": path}

    def _handle_file_list(self, params: Dict, agent_id: str, sandbox_id: str) -> Any:
        pattern = params.get("pattern", "*")
        path = params.get("path", ".")

        if sandbox_id:
            sandbox = self.sandbox_manager.get_sandbox(sandbox_id, agent_id)
            if sandbox:
                base = sandbox.root_path / "workspace" / path.lstrip("/")
                return [str(p.relative_to(base)) for p in base.glob(pattern)]

        return [str(p) for p in Path(path).glob(pattern)]

    def _handle_file_search(self, params: Dict, agent_id: str, sandbox_id: str) -> Any:
        pattern = params.get("pattern")
        path = params.get("path", ".")

        import subprocess
        cmd = f"grep -r '{pattern}' {path}"

        if sandbox_id:
            result = self.sandbox_manager.execute_in_sandbox(sandbox_id, agent_id, cmd)
        else:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            result = {"stdout": result.stdout, "stderr": result.stderr}

        return result

    def _handle_shell_exec(self, params: Dict, agent_id: str, sandbox_id: str) -> Any:
        command = params.get("command")
        timeout = params.get("timeout", 30)

        if sandbox_id:
            return self.sandbox_manager.execute_in_sandbox(sandbox_id, agent_id, command, timeout)

        import subprocess
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

    def _handle_msg_send(self, params: Dict, agent_id: str, sandbox_id: str) -> Any:
        to = params.get("to")
        content = params.get("message")

        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MsgType.DIRECT,
            sender_id=agent_id,
            recipient_id=to,
            content=content
        )
        self.message_bus.send(message)
        return {"sent": True, "message_id": message.message_id}

    def _handle_msg_broadcast(self, params: Dict, agent_id: str, sandbox_id: str) -> Any:
        content = params.get("message")

        message = Message(
            message_id=str(uuid.uuid4()),
            message_type=MsgType.BROADCAST,
            sender_id=agent_id,
            recipient_id=None,
            content=content
        )
        self.message_bus.send(message)
        return {"broadcast": True, "message_id": message.message_id}

    def _handle_msg_subscribe(self, params: Dict, agent_id: str, sandbox_id: str) -> Any:
        topic = params.get("topic")
        mailbox = self.message_bus.get_mailbox(agent_id)
        if mailbox:
            mailbox.subscribe(topic)
            return {"subscribed": topic}
        return {"error": "Mailbox not found"}

    def _handle_sandbox_create(self, params: Dict, agent_id: str, sandbox_id: str) -> Any:
        name = params.get("name")
        template = params.get("template")

        sandbox = self.sandbox_manager.create_sandbox(
            owner_id=agent_id,
            name=name,
            template=template
        )
        if sandbox:
            return {"sandbox_id": sandbox.sandbox_id, "path": str(sandbox.root_path)}
        return {"error": "Failed to create sandbox"}

    def _handle_sandbox_share(self, params: Dict, agent_id: str, sandbox_id: str) -> Any:
        target_sandbox = params.get("sandbox_id", sandbox_id)
        with_agent = params.get("with_agent")
        readonly = params.get("readonly", False)

        success = self.sandbox_manager.share_sandbox(
            target_sandbox, agent_id, with_agent, readonly
        )
        return {"shared": success}

    def _handle_sandbox_snapshot(self, params: Dict, agent_id: str, sandbox_id: str) -> Any:
        target_sandbox = params.get("sandbox_id", sandbox_id)
        description = params.get("description", "")

        snapshot = self.sandbox_manager.create_snapshot(
            target_sandbox, agent_id, description
        )
        if snapshot:
            return {"snapshot_id": snapshot.snapshot_id}
        return {"error": "Failed to create snapshot"}

    def _handle_skill_invoke(self, params: Dict, agent_id: str, sandbox_id: str) -> Any:
        skill_id = params.get("skill")
        skill_params = params.get("params", {})
        if isinstance(skill_params, str):
            skill_params = json.loads(skill_params)

        result = self.skill_registry.execute(skill_id, skill_params, {"agent_id": agent_id})
        return {"success": result.success, "output": result.output, "error": result.error}

    def _handle_skill_list(self, params: Dict, agent_id: str, sandbox_id: str) -> Any:
        category = params.get("category")
        skills = self.skill_registry.list_skills()
        return [s.to_dict() for s in skills]

    def _handle_agent_spawn(self, params: Dict, agent_id: str, sandbox_id: str) -> Any:
        agent_type = params.get("agent_type")
        # This would spawn a new agent - implementation depends on harness setup
        return {"spawned": False, "error": "Agent spawning not implemented in this context"}

    def _handle_agent_query(self, params: Dict, agent_id: str, sandbox_id: str) -> Any:
        capability = params.get("capability")
        agents = self.agent_directory.find_by_capability(capability)
        return [a.to_dict() for a in agents]

    def _handle_market_search(self, params: Dict, agent_id: str, sandbox_id: str) -> Any:
        query = params.get("query")
        results = self.marketplace.search(query)
        return [r.to_dict() for r in results]

    def _handle_market_install(self, params: Dict, agent_id: str, sandbox_id: str) -> Any:
        asset_id = params.get("asset_id")
        success, message = self.marketplace.install(asset_id)
        return {"success": success, "message": message}

    def _handle_state_get(self, params: Dict, agent_id: str, sandbox_id: str) -> Any:
        key = params.get("key")
        # Would retrieve from agent state - placeholder
        return {"key": key, "value": None}

    def _handle_state_set(self, params: Dict, agent_id: str, sandbox_id: str) -> Any:
        key = params.get("key")
        value = params.get("value")
        # Would store in agent state - placeholder
        return {"set": True, "key": key}

    def _handle_help(self, params: Dict, agent_id: str, sandbox_id: str) -> Any:
        topic = params.get("topic")
        if topic:
            skill = self.skill_registry.get_skill(topic)
            if skill:
                return skill.get_help()
        return generate_help_text()

    def _handle_status(self, params: Dict, agent_id: str, sandbox_id: str) -> Any:
        sandboxes = self.sandbox_manager.get_agent_sandboxes(agent_id)
        mailbox = self.message_bus.get_mailbox(agent_id)

        return {
            "agent_id": agent_id,
            "sandboxes": [s.sandbox_id for s in sandboxes],
            "pending_messages": mailbox.inbox.qsize() if mailbox else 0,
            "skills_available": len(self.skill_registry.skills),
            "message_bus_stats": self.message_bus.get_stats()
        }


class UniversalLLMHarness:
    """
    The main harness that wraps any LLM and provides agent capabilities.

    This is the core "game engine" for AI agents, providing:
    1. Unified command interface (tool calls, text blocks, natural language)
    2. Sandbox management (individual and shared)
    3. Skill/capability system
    4. Agent directory and marketplace
    5. Inter-agent communication
    6. State management and persistence
    7. Context compaction for infinite tasks
    """

    def __init__(
        self,
        llm_backend: LLMBackend = None,
        base_path: str = "/tmp/llm-harness",
        harness_mode: HarnessMode = HarnessMode.INTERACTIVE
    ):
        self.llm = llm_backend or MockLLMBackend()
        self.base_path = Path(base_path)
        self.harness_mode = harness_mode

        # Initialize subsystems
        self.sandbox_manager = SandboxManager(str(self.base_path / "sandboxes"))
        self.skill_registry = SkillRegistry(str(self.base_path / "skills"))
        self.message_bus = MessageBus()
        self.agent_directory = AgentDirectory()
        self.marketplace = Marketplace(str(self.base_path / "marketplace"))

        # Command executor
        self.executor = CommandExecutor(
            self.sandbox_manager,
            self.skill_registry,
            self.message_bus,
            self.agent_directory,
            self.marketplace
        )

        # Agent states
        self.agents: Dict[str, AgentState] = {}
        self.sessions: Dict[str, str] = {}  # session_id -> agent_id

        self._lock = threading.Lock()

    def create_agent(
        self,
        agent_id: str = None,
        name: str = None,
        capabilities: List[str] = None,
        specializations: List[str] = None
    ) -> Tuple[str, AgentState]:
        """Create a new agent in the harness"""
        agent_id = agent_id or str(uuid.uuid4())[:8]
        session_id = str(uuid.uuid4())[:12]

        # Create agent state
        state = AgentState(
            agent_id=agent_id,
            session_id=session_id,
            created_at=datetime.now(),
            last_active=datetime.now()
        )

        self.agents[agent_id] = state
        self.sessions[session_id] = agent_id

        # Register in directory
        profile = AgentProfile(
            agent_id=agent_id,
            name=name or f"Agent-{agent_id}",
            model_type="unknown",
            capability_level=AgentCapabilityLevel.INTERMEDIATE,
            capabilities=capabilities or ["general"],
            supports_tools=self.llm.supports_tools,
            max_context_tokens=self.llm.max_context_tokens,
            specializations=specializations or []
        )
        self.agent_directory.register(profile)

        # Create mailbox
        self.message_bus.register_agent(agent_id)

        # Create default sandbox
        self.sandbox_manager.create_sandbox(
            owner_id=agent_id,
            name=f"workspace-{agent_id}",
            template="empty"
        )

        return session_id, state

    def get_agent(self, agent_id: str) -> Optional[AgentState]:
        """Get agent state by ID"""
        return self.agents.get(agent_id)

    def get_agent_by_session(self, session_id: str) -> Optional[AgentState]:
        """Get agent state by session ID"""
        agent_id = self.sessions.get(session_id)
        if agent_id:
            return self.agents.get(agent_id)
        return None

    def build_context(
        self,
        agent_id: str,
        task: str,
        additional_context: Dict[str, Any] = None
    ) -> ExecutionContext:
        """
        Build the execution context for an agent turn.
        This is the "harness" that wraps around the LLM.
        """
        state = self.agents.get(agent_id)
        if not state:
            raise ValueError(f"Agent not found: {agent_id}")

        # Get agent resources
        sandboxes = self.sandbox_manager.get_agent_sandboxes(agent_id)
        sandbox = sandboxes[0] if sandboxes else None

        mailbox = self.message_bus.get_mailbox(agent_id)
        pending_messages = mailbox.peek_messages(5) if mailbox else []

        available_skills = self.skill_registry.list_skills()
        available_agents = self.agent_directory.list_all(status="active")

        return ExecutionContext(
            agent_id=agent_id,
            session_id=state.session_id,
            state=state,
            sandbox=sandbox,
            available_skills=available_skills,
            available_agents=[a for a in available_agents if a.agent_id != agent_id],
            pending_messages=pending_messages,
            recent_events=[],
            original_task=task,
            current_objective=state.current_task or task,
            constraints=[],
            timestamp=datetime.now(),
            harness_mode=self.harness_mode
        )

    def build_system_prompt(self, context: ExecutionContext) -> str:
        """
        Build the system prompt that defines agent behavior.
        This implements the layered prompt architecture.
        """
        layers = []

        # Layer 1: Core identity and rules
        layers.append("""# Agent Identity
You are an AI agent operating within the Universal LLM Harness.
You have access to a set of commands and skills to accomplish tasks.

# Core Rules
1. Always break complex tasks into smaller steps
2. Use the todo list to track progress and maintain focus
3. Execute commands to interact with the environment
4. Communicate with other agents when collaboration is needed
5. Create snapshots before making significant changes
6. Report errors and blockers clearly
""")

        # Layer 2: Available capabilities
        skills_summary = "\n".join([
            f"- {s.skill_id}: {s.description}"
            for s in context.available_skills[:20]
        ])

        layers.append(f"""# Available Skills
{skills_summary}

# Command Format
You can issue commands in these formats:

1. Text blocks (for any LLM):
```command:command.type
param1: value1
param2: value2
```

2. Natural language (will be parsed):
"Read the file at /path/to/file"
"Run 'npm test' in the terminal"
"Send a message to agent-123"

# Available Command Types
- file.read, file.write, file.edit, file.list, file.search
- shell.exec, shell.background
- msg.send, msg.broadcast, msg.subscribe
- sandbox.create, sandbox.share, sandbox.snapshot
- skill.invoke, skill.list
- agent.spawn, agent.query
- market.search, market.install
- state.get, state.set
- meta.help, meta.status
""")

        # Layer 3: Current state and context
        todos_str = json.dumps(context.state.todos, indent=2) if context.state.todos else "[]"

        layers.append(f"""# Current State
Agent ID: {context.agent_id}
Session: {context.session_id}
Mode: {context.harness_mode.value}
Sandbox: {context.sandbox.sandbox_id if context.sandbox else 'None'}

# Todo List (IMPORTANT - maintain focus)
{todos_str}

# Current Objective
{context.current_objective}
""")

        # Layer 4: Messages and events
        if context.pending_messages:
            msg_str = "\n".join([m.to_text_format() for m in context.pending_messages])
            layers.append(f"""# Pending Messages
{msg_str}
""")

        # Layer 5: Other agents
        if context.available_agents:
            agents_str = "\n".join([
                f"- {a.agent_id} ({a.name}): {', '.join(a.specializations)}"
                for a in context.available_agents[:10]
            ])
            layers.append(f"""# Available Agents for Collaboration
{agents_str}
""")

        return "\n\n---\n\n".join(layers)

    def execute_turn(
        self,
        agent_id: str,
        user_input: str,
        max_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Execute a single turn of agent interaction.

        For autonomous mode, this loops until task completion or max iterations.
        For interactive mode, this processes a single exchange.
        """
        state = self.agents.get(agent_id)
        if not state:
            return {"error": "Agent not found"}

        # Update state
        state.last_active = datetime.now()

        # Build context
        context = self.build_context(agent_id, user_input)
        system_prompt = self.build_system_prompt(context)

        # Get sandbox for command execution
        sandboxes = self.sandbox_manager.get_agent_sandboxes(agent_id)
        sandbox_id = sandboxes[0].sandbox_id if sandboxes else None

        results = []
        iteration = 0
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        while iteration < max_iterations:
            iteration += 1

            # Generate LLM response
            if self.llm.supports_tools:
                tools = self._build_tool_definitions()
                response = self.llm.generate(messages, tools=tools)
            else:
                response = self.llm.generate(messages)

            content = response.get("content", "")
            tool_calls = response.get("tool_calls", [])

            # Execute any commands (from tools or text parsing)
            if tool_calls:
                for tool_call in tool_calls:
                    cmd_results = self.executor.execute(tool_call, agent_id, sandbox_id)
                    results.extend(cmd_results)
                    state.commands_executed += len(cmd_results)
            elif content:
                # Try to parse text commands
                cmd_results = self.executor.execute(content, agent_id, sandbox_id)
                if cmd_results:
                    results.extend(cmd_results)
                    state.commands_executed += len(cmd_results)

            # Add to conversation
            messages.append({"role": "assistant", "content": content})

            # Check if done (for autonomous mode)
            if self.harness_mode == HarnessMode.INTERACTIVE:
                break

            if self._check_task_complete(state, content):
                break

            # Add command results to context for next iteration
            if results:
                result_summary = "\n".join([r.to_text_block() for r in results[-5:]])
                messages.append({
                    "role": "user",
                    "content": f"[COMMAND RESULTS]\n{result_summary}\n[/COMMAND RESULTS]\n\nContinue with your task."
                })
                results = []  # Clear processed results

        return {
            "agent_id": agent_id,
            "iterations": iteration,
            "results": [r.to_json() for r in results],
            "state": state.to_dict(),
            "final_response": messages[-1]["content"] if messages else ""
        }

    def compact_context(self, agent_id: str) -> Dict[str, Any]:
        """
        Compact agent context when approaching limits.
        Inspired by OpenAI Codex's compaction feature.
        """
        state = self.agents.get(agent_id)
        if not state:
            return {"error": "Agent not found"}

        # Summarize completed work
        summary = {
            "completed_tasks": state.completed_tasks[-10:],
            "current_task": state.current_task,
            "todos": state.todos,
            "key_findings": state.working_memory.get("key_findings", []),
            "errors_encountered": state.working_memory.get("errors", [])
        }

        # Clear old short-term memory
        state.short_term_memory = state.short_term_memory[-5:]

        # Increment compaction counter
        state.compaction_count += 1

        # Update working memory with summary
        state.working_memory["last_compaction"] = datetime.now().isoformat()
        state.working_memory["compaction_summary"] = summary

        return {
            "compacted": True,
            "compaction_count": state.compaction_count,
            "summary": summary
        }

    def update_todos(
        self,
        agent_id: str,
        todos: List[Dict[str, Any]]
    ) -> bool:
        """
        Update agent's todo list (Manus-style attention management).
        The todo list is constantly written to the end of context
        to keep objectives in the model's attention.
        """
        state = self.agents.get(agent_id)
        if not state:
            return False

        state.todos = todos
        return True

    def _build_tool_definitions(self) -> List[Dict[str, Any]]:
        """Build tool definitions for tool-using LLMs"""
        tools = [
            {
                "name": "file_read",
                "description": "Read contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to file"}
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "file_write",
                "description": "Write contents to a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["path", "content"]
                }
            },
            {
                "name": "shell_exec",
                "description": "Execute a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"}
                    },
                    "required": ["command"]
                }
            },
            {
                "name": "skill_invoke",
                "description": "Invoke a skill by ID",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill": {"type": "string"},
                        "params": {"type": "object"}
                    },
                    "required": ["skill"]
                }
            },
            {
                "name": "msg_send",
                "description": "Send a message to another agent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string"},
                        "message": {"type": "string"}
                    },
                    "required": ["to", "message"]
                }
            }
        ]
        return tools

    def _check_task_complete(self, state: AgentState, content: str) -> bool:
        """Check if agent indicates task is complete"""
        complete_indicators = [
            "task complete",
            "all done",
            "finished",
            "[DONE]",
            "completed successfully"
        ]
        content_lower = content.lower()
        return any(ind in content_lower for ind in complete_indicators)

    def get_status(self) -> Dict[str, Any]:
        """Get overall harness status"""
        return {
            "agents": len(self.agents),
            "active_sandboxes": len(self.sandbox_manager.sandboxes),
            "registered_skills": len(self.skill_registry.skills),
            "message_bus": self.message_bus.get_stats(),
            "marketplace_assets": len(self.marketplace.assets)
        }


# Convenience function for quick harness setup
def create_harness(
    mode: HarnessMode = HarnessMode.INTERACTIVE,
    base_path: str = "/tmp/llm-harness"
) -> UniversalLLMHarness:
    """Create and configure a harness instance"""
    return UniversalLLMHarness(
        harness_mode=mode,
        base_path=base_path
    )
