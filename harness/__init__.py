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
"""

__version__ = "0.1.0"

from .core.llm_harness import (
    UniversalLLMHarness,
    HarnessMode,
    LLMBackend,
    AgentState,
    ExecutionContext,
    create_harness
)

from .core.command_protocol import (
    Command,
    CommandResult,
    CommandType,
    UniversalCommandParser
)

from .sandbox.sandbox_manager import (
    SandboxManager,
    Sandbox,
    SandboxType,
    ResourceLimits
)

from .skills.skill_system import (
    SkillRegistry,
    Skill,
    SkillResult,
    SkillMetadata,
    PythonSkill,
    PromptSkill,
    CompositeSkill
)

from .marketplace.registry import (
    AgentDirectory,
    Marketplace,
    AgentProfile,
    MarketplaceAsset
)

from .communication.message_bus import (
    MessageBus,
    Message,
    MessageType,
    AgentMailbox
)

__all__ = [
    # Core harness
    "UniversalLLMHarness",
    "HarnessMode",
    "LLMBackend",
    "AgentState",
    "ExecutionContext",
    "create_harness",

    # Commands
    "Command",
    "CommandResult",
    "CommandType",
    "UniversalCommandParser",

    # Sandbox
    "SandboxManager",
    "Sandbox",
    "SandboxType",
    "ResourceLimits",

    # Skills
    "SkillRegistry",
    "Skill",
    "SkillResult",
    "SkillMetadata",
    "PythonSkill",
    "PromptSkill",
    "CompositeSkill",

    # Marketplace
    "AgentDirectory",
    "Marketplace",
    "AgentProfile",
    "MarketplaceAsset",

    # Communication
    "MessageBus",
    "Message",
    "MessageType",
    "AgentMailbox",
]
