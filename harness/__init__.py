"""
Universal LLM Agent Harness

A minimal yet complete harness enabling any reasonably intelligent LLM
to operate with capabilities matching or exceeding current systems.
"""

__version__ = "0.1.0"

from .core.harness import UniversalHarness
from .core.types import (
    AgentConfig,
    HarnessConfig,
    CommandResult,
    ToolResult,
)
from .sandbox.manager import SandboxManager
from .skills.registry import SkillRegistry
from .communication.bus import AgentCommBus
from .evaluation.loop import EvaluationLoop
from .qa.generator import QAGenerator

__all__ = [
    "UniversalHarness",
    "AgentConfig",
    "HarnessConfig",
    "CommandResult",
    "ToolResult",
    "SandboxManager",
    "SkillRegistry",
    "AgentCommBus",
    "EvaluationLoop",
    "QAGenerator",
]
