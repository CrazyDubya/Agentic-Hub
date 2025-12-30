"""
Core type definitions for the Universal LLM Agent Harness.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Literal, Optional, Union
from pathlib import Path
import uuid


class ExecutionMode(Enum):
    """How the LLM interacts with the harness."""
    TEXT_COMMAND = auto()  # Non-tool LLM using /commands
    TOOL_CALL = auto()     # Tool-using LLM with function calls
    HYBRID = auto()        # Can use both modes


class SandboxType(Enum):
    """Type of sandbox environment."""
    PERSONAL = auto()   # Private to one agent
    SHARED = auto()     # Shared between agents
    EPHEMERAL = auto()  # Temporary, destroyed after use


class MessageType(Enum):
    """Types of inter-agent messages."""
    REQUEST = auto()
    RESPONSE = auto()
    EVENT = auto()
    QUERY = auto()
    BROADCAST = auto()


class QuestionType(Enum):
    """Types of Q&A questions."""
    CLARIFICATION = auto()  # Need more info
    VERIFICATION = auto()   # Confirm understanding
    GUIDANCE = auto()       # Need direction
    LEARNING = auto()       # Learning opportunity


class SelectionMode(Enum):
    """How answers can be selected."""
    SINGLE = auto()    # Pick one
    MULTIPLE = auto()  # Pick some
    ALL = auto()       # Pick all that apply
    NONE = auto()      # Blank/skip allowed


class EvalCycleType(Enum):
    """Types of evaluation cycles."""
    TASK = auto()      # After each task
    SESSION = auto()   # After session ends
    PERIODIC = auto()  # On schedule


@dataclass
class ResourceLimits:
    """Resource limits for sandboxes."""
    max_memory_mb: int = 512
    max_disk_mb: int = 1024
    max_processes: int = 10
    max_open_files: int = 100
    max_execution_time_s: int = 300
    network_enabled: bool = False
    allowed_hosts: List[str] = field(default_factory=list)


@dataclass
class AgentConfig:
    """Configuration for an agent in the harness."""
    agent_id: str
    name: str
    execution_mode: ExecutionMode = ExecutionMode.HYBRID
    model_id: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=lambda: [
        "sandbox.read", "sandbox.write", "sandbox.execute"
    ])
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.agent_id:
            self.agent_id = f"agent-{uuid.uuid4().hex[:8]}"


@dataclass
class HarnessConfig:
    """Configuration for the harness itself."""
    harness_id: str = field(default_factory=lambda: f"harness-{uuid.uuid4().hex[:8]}")
    data_dir: Path = field(default_factory=lambda: Path.home() / ".harness")
    sandbox_root: Path = field(default_factory=lambda: Path.home() / ".harness" / "sandboxes")
    skills_dir: Path = field(default_factory=lambda: Path.home() / ".harness" / "skills")
    marketplace_url: Optional[str] = None
    enable_networking: bool = False
    enable_evaluation: bool = True
    enable_qa: bool = True
    eval_interval_tasks: int = 5  # Evaluate every N tasks
    log_level: str = "INFO"
    max_agents: int = 100
    default_resource_limits: ResourceLimits = field(default_factory=ResourceLimits)


@dataclass
class CommandResult:
    """Result of executing a text command."""
    command: str
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def as_text(self) -> str:
        """Format result as text for non-tool LLMs."""
        if self.success:
            if isinstance(self.output, str):
                return self.output
            elif isinstance(self.output, (list, dict)):
                import json
                return json.dumps(self.output, indent=2)
            else:
                return str(self.output)
        else:
            return f"ERROR: {self.error}"


@dataclass
class ToolResult:
    """Result of executing a tool call."""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_tool_response(self) -> Dict[str, Any]:
        """Format for tool-using LLM."""
        return {
            "tool": self.tool_name,
            "success": self.success,
            "result": self.result,
            "error": self.error,
        }


@dataclass
class ParsedCommand:
    """A parsed text command."""
    raw: str
    command: str
    subcommand: Optional[str] = None
    args: List[str] = field(default_factory=list)
    flags: Dict[str, Any] = field(default_factory=dict)
    valid: bool = True
    error: Optional[str] = None


@dataclass
class Sandbox:
    """A sandbox environment."""
    id: str
    name: str
    owner: str
    sandbox_type: SandboxType
    root_path: Path
    members: List[str] = field(default_factory=list)
    permissions: Dict[str, List[str]] = field(default_factory=dict)
    env_vars: Dict[str, str] = field(default_factory=dict)
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    created_at: datetime = field(default_factory=datetime.now)
    state: Dict[str, Any] = field(default_factory=dict)
    active: bool = True


@dataclass
class Skill:
    """A skill/capability that can be invoked."""
    id: str
    name: str
    description: str
    version: str = "1.0.0"
    author: str = "system"

    # Invocation
    invoke_type: Literal["command", "tool", "hybrid"] = "hybrid"
    command_template: Optional[str] = None
    tool_schema: Optional[Dict[str, Any]] = None

    # Execution
    handler: Optional[Callable] = None
    handler_path: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    required_permissions: List[str] = field(default_factory=list)

    # Metadata
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    documentation: str = ""


@dataclass
class Message:
    """Inter-agent message."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    from_agent: str = ""
    to_agent: Union[str, List[str], Literal["broadcast"]] = ""
    message_type: MessageType = MessageType.REQUEST
    content: Any = None
    priority: int = 5
    reply_to: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    delivered: bool = False
    read: bool = False


@dataclass
class Question:
    """A Q&A question for learning/clarification."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    question_text: str = ""
    context: str = ""
    question_type: QuestionType = QuestionType.CLARIFICATION

    # Answer options
    believed_answer: Optional[str] = None
    confidence: float = 0.0  # 0-1
    answer_options: List[str] = field(default_factory=list)
    selection_mode: SelectionMode = SelectionMode.SINGLE

    # Status
    status: Literal["pending", "answered", "confirmed", "rejected"] = "pending"
    user_answer: Optional[str] = None
    user_correction: Optional[str] = None

    # Learning
    learned_from_answer: bool = False
    incorporated_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class TaskRecord:
    """Record of a task attempt for evaluation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    task_description: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    success: bool = False
    error: Optional[str] = None
    skills_used: List[str] = field(default_factory=list)
    commands_executed: List[str] = field(default_factory=list)
    self_rating: Optional[int] = None  # 1-10
    user_rating: Optional[int] = None
    notes: str = ""


@dataclass
class EvaluationCycle:
    """A self-evaluation cycle."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    cycle_type: EvalCycleType = EvalCycleType.TASK
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    # What was attempted
    tasks_attempted: List[TaskRecord] = field(default_factory=list)
    skills_used: List[str] = field(default_factory=list)
    errors_encountered: List[Dict[str, Any]] = field(default_factory=list)

    # Self-assessment
    self_rating: Optional[int] = None  # 1-10
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    identified_gaps: List[str] = field(default_factory=list)
    improvement_ideas: List[str] = field(default_factory=list)

    # External feedback
    user_rating: Optional[int] = None
    user_feedback: Optional[str] = None

    # Actions
    next_steps: List[str] = field(default_factory=list)
    skills_to_learn: List[str] = field(default_factory=list)
    questions_for_user: List[Question] = field(default_factory=list)


@dataclass
class MarketplaceEntry:
    """An entry in the marketplace."""
    id: str
    entry_type: Literal["skill", "agent", "template", "dataset", "model"]
    name: str
    description: str
    version: str
    author: str

    # Content
    content_hash: str = ""
    download_url: str = ""
    source_url: Optional[str] = None

    # Metadata
    tags: List[str] = field(default_factory=list)
    category: str = "general"
    license: str = "MIT"

    # Stats
    downloads: int = 0
    ratings: List[int] = field(default_factory=list)

    # Compatibility
    harness_version: str = "0.1.0"
    dependencies: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)

    @property
    def avg_rating(self) -> float:
        if not self.ratings:
            return 0.0
        return sum(self.ratings) / len(self.ratings)
