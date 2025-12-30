"""
Universal LLM Agent Harness

A comprehensive framework for enabling ANY LLM to function as a capable agent.
Works with both tool-using and non-tool-using models.

Key Components:
- Command Protocol: Unified command interface
- Sandbox Manager: Isolated execution environments
- Skill System: Extensible capabilities
- Marketplace: Agent and skill discovery
- Message Bus: Inter-agent communication
- LLM Harness: Core orchestration layer
- Self-Evaluation: Flywheel for continuous improvement
- Q&A Generation: Training data creation

Example usage:

    from harness import UniversalLLMHarness, HarnessMode

    # Create harness
    harness = UniversalLLMHarness(harness_mode=HarnessMode.INTERACTIVE)

    # Create an agent
    session_id, state = harness.create_agent(
        name="MyAgent",
        capabilities=["code", "research"]
    )

    # Execute a turn
    result = harness.execute_turn(
        state.agent_id,
        "Help me analyze this codebase"
    )

    print(result["final_response"])

CLI usage:

    # Interactive mode
    python -m harness interactive

    # Run a task
    python -m harness run "Analyze the codebase"

    # Start API server
    python -m harness server --port 8080
"""

__version__ = "0.1.0"

from .core.llm_harness import (
    UniversalLLMHarness,
    HarnessMode,
    LLMType,
    LLMBackend,
    MockLLMBackend,
    AgentState,
    ExecutionContext,
    CommandExecutor,
    create_harness
)

from .core.command_protocol import (
    Command,
    CommandResult,
    CommandType,
    UniversalCommandParser,
    generate_help_text
)

from .sandbox.sandbox_manager import (
    SandboxManager,
    Sandbox,
    SandboxType,
    SandboxSnapshot,
    ResourceLimits,
    IsolationLevel
)

from .skills.skill_system import (
    SkillRegistry,
    Skill,
    SkillResult,
    SkillMetadata,
    SkillParameter,
    SkillOutput,
    SkillType,
    SkillCategory,
    PythonSkill,
    PromptSkill,
    CompositeSkill
)

from .marketplace.registry import (
    AgentDirectory,
    Marketplace,
    AgentProfile,
    AgentCapabilityLevel,
    MarketplaceAsset,
    AssetType
)

from .communication.message_bus import (
    MessageBus,
    Message,
    MessageType,
    MessagePriority,
    AgentMailbox,
    Conversation,
    TextMessageParser,
    format_inbox_for_llm
)

__all__ = [
    # Core harness
    "UniversalLLMHarness",
    "HarnessMode",
    "LLMType",
    "LLMBackend",
    "MockLLMBackend",
    "AgentState",
    "ExecutionContext",
    "CommandExecutor",
    "create_harness",

    # Commands
    "Command",
    "CommandResult",
    "CommandType",
    "UniversalCommandParser",
    "generate_help_text",

    # Sandbox
    "SandboxManager",
    "Sandbox",
    "SandboxType",
    "SandboxSnapshot",
    "ResourceLimits",
    "IsolationLevel",

    # Skills
    "SkillRegistry",
    "Skill",
    "SkillResult",
    "SkillMetadata",
    "SkillParameter",
    "SkillOutput",
    "SkillType",
    "SkillCategory",
    "PythonSkill",
    "PromptSkill",
    "CompositeSkill",

    # Marketplace
    "AgentDirectory",
    "Marketplace",
    "AgentProfile",
    "AgentCapabilityLevel",
    "MarketplaceAsset",
    "AssetType",

    # Communication
    "MessageBus",
    "Message",
    "MessageType",
    "MessagePriority",
    "AgentMailbox",
    "Conversation",
    "TextMessageParser",
    "format_inbox_for_llm",
]
