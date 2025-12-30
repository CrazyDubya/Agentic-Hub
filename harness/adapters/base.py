"""
Base Backend Infrastructure

Provides flexible base classes and model registry for LLM backends.
Supports multiple providers, model families, and capabilities.
"""

import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Iterator, Callable

logger = logging.getLogger(__name__)


class ModelCapability(Enum):
    """Capabilities that models may support."""
    CHAT = "chat"
    COMPLETION = "completion"
    TOOL_USE = "tool_use"
    VISION = "vision"
    AUDIO = "audio"
    CODE = "code"
    REASONING = "reasoning"
    STREAMING = "streaming"
    LONG_CONTEXT = "long_context"  # 100k+ tokens


@dataclass
class ModelInfo:
    """
    Metadata about a specific model.

    This allows the system to know model capabilities without hardcoding.
    """
    id: str  # Model identifier (e.g., "claude-sonnet-4-20250514")
    name: str  # Human-readable name
    provider: str  # Provider name (anthropic, openai, etc.)
    context_window: int  # Max context tokens
    max_output_tokens: int  # Max output tokens
    capabilities: List[ModelCapability] = field(default_factory=list)

    # Pricing per million tokens (for cost estimation)
    input_price_per_m: float = 0.0
    output_price_per_m: float = 0.0

    # Model family for grouping
    family: str = ""

    # Whether this is the latest/recommended in its tier
    recommended: bool = False

    # Release date (YYYY-MM-DD format)
    release_date: str = ""

    def supports(self, capability: ModelCapability) -> bool:
        """Check if model supports a capability."""
        return capability in self.capabilities


# ============================================================================
# Current SOTA Model Registry (December 2025)
# ============================================================================

ANTHROPIC_MODELS = {
    # Claude 4.x Family
    "claude-opus-4-20250514": ModelInfo(
        id="claude-opus-4-20250514",
        name="Claude Opus 4",
        provider="anthropic",
        context_window=200000,
        max_output_tokens=32000,
        capabilities=[
            ModelCapability.CHAT, ModelCapability.TOOL_USE,
            ModelCapability.VISION, ModelCapability.CODE,
            ModelCapability.REASONING, ModelCapability.STREAMING,
            ModelCapability.LONG_CONTEXT
        ],
        input_price_per_m=15.0,
        output_price_per_m=75.0,
        family="claude-4",
        release_date="2025-05-22"
    ),
    "claude-sonnet-4-20250514": ModelInfo(
        id="claude-sonnet-4-20250514",
        name="Claude Sonnet 4",
        provider="anthropic",
        context_window=200000,
        max_output_tokens=64000,
        capabilities=[
            ModelCapability.CHAT, ModelCapability.TOOL_USE,
            ModelCapability.VISION, ModelCapability.CODE,
            ModelCapability.REASONING, ModelCapability.STREAMING,
            ModelCapability.LONG_CONTEXT
        ],
        input_price_per_m=3.0,
        output_price_per_m=15.0,
        family="claude-4",
        recommended=True,
        release_date="2025-05-22"
    ),
    # Claude Opus 4.5 (latest)
    "claude-opus-4-5-20251101": ModelInfo(
        id="claude-opus-4-5-20251101",
        name="Claude Opus 4.5",
        provider="anthropic",
        context_window=200000,
        max_output_tokens=32000,
        capabilities=[
            ModelCapability.CHAT, ModelCapability.TOOL_USE,
            ModelCapability.VISION, ModelCapability.CODE,
            ModelCapability.REASONING, ModelCapability.STREAMING,
            ModelCapability.LONG_CONTEXT
        ],
        input_price_per_m=15.0,
        output_price_per_m=75.0,
        family="claude-4.5",
        recommended=True,
        release_date="2025-11-01"
    ),
    # Claude Sonnet 4.5
    "claude-sonnet-4-5-20250929": ModelInfo(
        id="claude-sonnet-4-5-20250929",
        name="Claude Sonnet 4.5",
        provider="anthropic",
        context_window=1000000,  # 1M context!
        max_output_tokens=64000,
        capabilities=[
            ModelCapability.CHAT, ModelCapability.TOOL_USE,
            ModelCapability.VISION, ModelCapability.CODE,
            ModelCapability.REASONING, ModelCapability.STREAMING,
            ModelCapability.LONG_CONTEXT
        ],
        input_price_per_m=3.0,
        output_price_per_m=15.0,
        family="claude-4.5",
        release_date="2025-09-29"
    ),
    # Claude Haiku 4.5 (fast/cheap)
    "claude-haiku-4-5-20251015": ModelInfo(
        id="claude-haiku-4-5-20251015",
        name="Claude Haiku 4.5",
        provider="anthropic",
        context_window=200000,
        max_output_tokens=8192,
        capabilities=[
            ModelCapability.CHAT, ModelCapability.TOOL_USE,
            ModelCapability.VISION, ModelCapability.CODE,
            ModelCapability.STREAMING
        ],
        input_price_per_m=1.0,
        output_price_per_m=5.0,
        family="claude-4.5",
        recommended=True,  # Recommended for fast/cheap
        release_date="2025-10-15"
    ),
}

OPENAI_MODELS = {
    # GPT-5 Family
    "gpt-5": ModelInfo(
        id="gpt-5",
        name="GPT-5",
        provider="openai",
        context_window=256000,
        max_output_tokens=32000,
        capabilities=[
            ModelCapability.CHAT, ModelCapability.TOOL_USE,
            ModelCapability.VISION, ModelCapability.CODE,
            ModelCapability.REASONING, ModelCapability.STREAMING,
            ModelCapability.LONG_CONTEXT
        ],
        input_price_per_m=10.0,
        output_price_per_m=30.0,
        family="gpt-5",
        recommended=True,
        release_date="2025-03-01"
    ),
    # GPT-4o (still widely used)
    "gpt-4o": ModelInfo(
        id="gpt-4o",
        name="GPT-4o",
        provider="openai",
        context_window=128000,
        max_output_tokens=16384,
        capabilities=[
            ModelCapability.CHAT, ModelCapability.TOOL_USE,
            ModelCapability.VISION, ModelCapability.AUDIO,
            ModelCapability.CODE, ModelCapability.STREAMING,
            ModelCapability.LONG_CONTEXT
        ],
        input_price_per_m=2.5,
        output_price_per_m=10.0,
        family="gpt-4o",
        release_date="2024-05-13"
    ),
    "gpt-4o-mini": ModelInfo(
        id="gpt-4o-mini",
        name="GPT-4o Mini",
        provider="openai",
        context_window=128000,
        max_output_tokens=16384,
        capabilities=[
            ModelCapability.CHAT, ModelCapability.TOOL_USE,
            ModelCapability.VISION, ModelCapability.CODE,
            ModelCapability.STREAMING, ModelCapability.LONG_CONTEXT
        ],
        input_price_per_m=0.15,
        output_price_per_m=0.60,
        family="gpt-4o",
        recommended=True,  # Recommended for fast/cheap
        release_date="2024-07-18"
    ),
    # O1 reasoning models
    "o1": ModelInfo(
        id="o1",
        name="O1",
        provider="openai",
        context_window=200000,
        max_output_tokens=100000,
        capabilities=[
            ModelCapability.CHAT, ModelCapability.REASONING,
            ModelCapability.CODE, ModelCapability.LONG_CONTEXT
        ],
        input_price_per_m=15.0,
        output_price_per_m=60.0,
        family="o1",
        release_date="2024-12-05"
    ),
    "o1-mini": ModelInfo(
        id="o1-mini",
        name="O1 Mini",
        provider="openai",
        context_window=128000,
        max_output_tokens=65536,
        capabilities=[
            ModelCapability.CHAT, ModelCapability.REASONING,
            ModelCapability.CODE, ModelCapability.LONG_CONTEXT
        ],
        input_price_per_m=3.0,
        output_price_per_m=12.0,
        family="o1",
        release_date="2024-09-12"
    ),
}

GOOGLE_MODELS = {
    "gemini-2.5-pro": ModelInfo(
        id="gemini-2.5-pro",
        name="Gemini 2.5 Pro",
        provider="google",
        context_window=1000000,
        max_output_tokens=65536,
        capabilities=[
            ModelCapability.CHAT, ModelCapability.TOOL_USE,
            ModelCapability.VISION, ModelCapability.AUDIO,
            ModelCapability.CODE, ModelCapability.REASONING,
            ModelCapability.STREAMING, ModelCapability.LONG_CONTEXT
        ],
        input_price_per_m=1.25,
        output_price_per_m=5.0,
        family="gemini-2.5",
        recommended=True,
        release_date="2025-03-25"
    ),
    "gemini-2.0-flash": ModelInfo(
        id="gemini-2.0-flash",
        name="Gemini 2.0 Flash",
        provider="google",
        context_window=1000000,
        max_output_tokens=8192,
        capabilities=[
            ModelCapability.CHAT, ModelCapability.TOOL_USE,
            ModelCapability.VISION, ModelCapability.CODE,
            ModelCapability.STREAMING, ModelCapability.LONG_CONTEXT
        ],
        input_price_per_m=0.075,
        output_price_per_m=0.30,
        family="gemini-2.0",
        recommended=True,  # Best value
        release_date="2024-12-11"
    ),
}

DEEPSEEK_MODELS = {
    "deepseek-v3": ModelInfo(
        id="deepseek-v3",
        name="DeepSeek V3",
        provider="deepseek",
        context_window=128000,
        max_output_tokens=8192,
        capabilities=[
            ModelCapability.CHAT, ModelCapability.TOOL_USE,
            ModelCapability.CODE, ModelCapability.REASONING,
            ModelCapability.STREAMING, ModelCapability.LONG_CONTEXT
        ],
        input_price_per_m=0.27,
        output_price_per_m=1.10,
        family="deepseek-v3",
        recommended=True,  # Best budget option
        release_date="2024-12-26"
    ),
    "deepseek-r1": ModelInfo(
        id="deepseek-r1",
        name="DeepSeek R1",
        provider="deepseek",
        context_window=128000,
        max_output_tokens=8192,
        capabilities=[
            ModelCapability.CHAT, ModelCapability.REASONING,
            ModelCapability.CODE, ModelCapability.STREAMING,
            ModelCapability.LONG_CONTEXT
        ],
        input_price_per_m=0.55,
        output_price_per_m=2.19,
        family="deepseek-r1",
        release_date="2025-01-20"
    ),
}

# Local models available via Ollama
OLLAMA_MODELS = {
    "llama3.3:70b": ModelInfo(
        id="llama3.3:70b",
        name="Llama 3.3 70B",
        provider="ollama",
        context_window=128000,
        max_output_tokens=8192,
        capabilities=[
            ModelCapability.CHAT, ModelCapability.TOOL_USE,
            ModelCapability.CODE, ModelCapability.STREAMING,
            ModelCapability.LONG_CONTEXT
        ],
        family="llama-3.3",
        recommended=True,
        release_date="2024-12-06"
    ),
    "llama3.2:3b": ModelInfo(
        id="llama3.2:3b",
        name="Llama 3.2 3B",
        provider="ollama",
        context_window=128000,
        max_output_tokens=8192,
        capabilities=[
            ModelCapability.CHAT, ModelCapability.CODE,
            ModelCapability.STREAMING
        ],
        family="llama-3.2",
        release_date="2024-09-25"
    ),
    "mistral:7b": ModelInfo(
        id="mistral:7b",
        name="Mistral 7B",
        provider="ollama",
        context_window=32000,
        max_output_tokens=8192,
        capabilities=[
            ModelCapability.CHAT, ModelCapability.TOOL_USE,
            ModelCapability.CODE, ModelCapability.STREAMING
        ],
        family="mistral",
        release_date="2024-03-01"
    ),
    "qwen2.5:72b": ModelInfo(
        id="qwen2.5:72b",
        name="Qwen 2.5 72B",
        provider="ollama",
        context_window=128000,
        max_output_tokens=8192,
        capabilities=[
            ModelCapability.CHAT, ModelCapability.TOOL_USE,
            ModelCapability.CODE, ModelCapability.STREAMING,
            ModelCapability.LONG_CONTEXT
        ],
        family="qwen-2.5",
        recommended=True,
        release_date="2024-09-19"
    ),
    "codellama:34b": ModelInfo(
        id="codellama:34b",
        name="Code Llama 34B",
        provider="ollama",
        context_window=16000,
        max_output_tokens=8192,
        capabilities=[
            ModelCapability.CHAT, ModelCapability.CODE,
            ModelCapability.STREAMING
        ],
        family="codellama",
        release_date="2024-01-29"
    ),
    "deepseek-r1:70b": ModelInfo(
        id="deepseek-r1:70b",
        name="DeepSeek R1 70B (Local)",
        provider="ollama",
        context_window=128000,
        max_output_tokens=8192,
        capabilities=[
            ModelCapability.CHAT, ModelCapability.REASONING,
            ModelCapability.CODE, ModelCapability.STREAMING
        ],
        family="deepseek-r1",
        release_date="2025-01-20"
    ),
}

# Combined registry
MODEL_REGISTRY: Dict[str, ModelInfo] = {
    **ANTHROPIC_MODELS,
    **OPENAI_MODELS,
    **GOOGLE_MODELS,
    **DEEPSEEK_MODELS,
    **OLLAMA_MODELS,
}


def get_model_info(model_id: str) -> Optional[ModelInfo]:
    """Get model info by ID, with fuzzy matching."""
    # Exact match
    if model_id in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_id]

    # Fuzzy match (prefix matching for versioned models)
    for key, info in MODEL_REGISTRY.items():
        if model_id.startswith(key.split("-")[0]) or key.startswith(model_id):
            return info

    return None


def list_models_by_provider(provider: str) -> List[ModelInfo]:
    """List all models for a provider."""
    return [m for m in MODEL_REGISTRY.values() if m.provider == provider]


def list_recommended_models() -> List[ModelInfo]:
    """List recommended models across all providers."""
    return [m for m in MODEL_REGISTRY.values() if m.recommended]


# ============================================================================
# Base Backend Classes
# ============================================================================

@dataclass
class GenerateResponse:
    """
    Unified response from any LLM backend.
    """
    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    # Optional metadata
    model: str = ""
    usage: Optional[Dict[str, int]] = None  # tokens used
    finish_reason: str = ""
    raw_response: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for harness compatibility."""
        return {
            "content": self.content,
            "tool_calls": self.tool_calls,
        }


class BaseLLMBackend(ABC):
    """
    Enhanced base class for LLM backends.

    Provides common functionality and a flexible interface that
    supports various API styles (completions, chat, responses).
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_key_env_var: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        temperature: float = 0.7,
        **kwargs
    ):
        """
        Initialize backend with common parameters.

        Args:
            model: Model identifier
            api_key: API key (or loaded from env)
            api_key_env_var: Environment variable name for API key
            base_url: Optional base URL override
            timeout: Request timeout
            temperature: Sampling temperature
            **kwargs: Provider-specific options
        """
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.temperature = temperature
        self.extra_options = kwargs

        # Load API key
        self._api_key = api_key
        if not self._api_key and api_key_env_var:
            self._api_key = os.environ.get(api_key_env_var)

        # Get model info if available
        self._model_info = get_model_info(model)

        logger.info(f"Initialized {self.__class__.__name__} with model: {model}")

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response.

        Args:
            messages: Conversation messages
            tools: Tool definitions for function calling
            max_tokens: Maximum output tokens
            **kwargs: Additional options

        Returns:
            Dict with 'content' and 'tool_calls'
        """
        pass

    def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        **kwargs
    ) -> Iterator[str]:
        """
        Generate a streaming response.

        Default implementation falls back to non-streaming.
        Override in subclasses that support streaming.
        """
        response = self.generate(messages, tools, max_tokens, **kwargs)
        yield response["content"]

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass

    @property
    def supports_tools(self) -> bool:
        """Whether this backend supports tool/function calling."""
        if self._model_info:
            return self._model_info.supports(ModelCapability.TOOL_USE)
        return False

    @property
    def supports_streaming(self) -> bool:
        """Whether this backend supports streaming."""
        if self._model_info:
            return self._model_info.supports(ModelCapability.STREAMING)
        return False

    @property
    def supports_vision(self) -> bool:
        """Whether this backend supports image inputs."""
        if self._model_info:
            return self._model_info.supports(ModelCapability.VISION)
        return False

    @property
    def max_context_tokens(self) -> int:
        """Maximum context window size."""
        if self._model_info:
            return self._model_info.context_window
        return 8192  # Conservative default

    @property
    def model_info(self) -> Optional[ModelInfo]:
        """Get full model metadata."""
        return self._model_info

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD for a request."""
        if self._model_info:
            input_cost = (input_tokens / 1_000_000) * self._model_info.input_price_per_m
            output_cost = (output_tokens / 1_000_000) * self._model_info.output_price_per_m
            return input_cost + output_cost
        return 0.0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"


# ============================================================================
# Tool Format Converters
# ============================================================================

def convert_tools_to_openai_format(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert generic tool definitions to OpenAI format."""
    openai_tools = []
    for tool in tools:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {"type": "object", "properties": {}})
            }
        }
        openai_tools.append(openai_tool)
    return openai_tools


def convert_tools_to_anthropic_format(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert generic tool definitions to Anthropic format."""
    anthropic_tools = []
    for tool in tools:
        anthropic_tool = {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "input_schema": tool.get("parameters", {"type": "object", "properties": {}})
        }
        anthropic_tools.append(anthropic_tool)
    return anthropic_tools


def parse_openai_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
    """Parse OpenAI tool calls to generic format."""
    import json
    result = []
    if tool_calls:
        for tc in tool_calls:
            result.append({
                "id": getattr(tc, "id", ""),
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments) if tc.function.arguments else {}
            })
    return result


def parse_anthropic_tool_calls(content_blocks: Any) -> List[Dict[str, Any]]:
    """Parse Anthropic tool use blocks to generic format."""
    result = []
    if content_blocks:
        for block in content_blocks:
            if hasattr(block, "type") and block.type == "tool_use":
                result.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input if hasattr(block, "input") else {}
                })
    return result
