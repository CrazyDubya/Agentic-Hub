"""
OpenRouter Backend Implementation

Real, working backend that provides access to 400+ models through a unified API.
OpenRouter acts as a gateway to models from OpenAI, Anthropic, Google, Meta, Mistral, and more.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Iterator

from .base import (
    BaseLLMBackend,
    convert_tools_to_openai_format,
    ModelInfo,
    ModelCapability,
)

logger = logging.getLogger(__name__)


# Popular models available on OpenRouter
OPENROUTER_MODELS = {
    # Anthropic
    "anthropic/claude-opus-4": ModelInfo(
        id="anthropic/claude-opus-4",
        name="Claude Opus 4 (via OpenRouter)",
        provider="openrouter",
        context_window=200000,
        max_output_tokens=32000,
        capabilities=[
            ModelCapability.CHAT, ModelCapability.TOOL_USE,
            ModelCapability.VISION, ModelCapability.CODE,
            ModelCapability.STREAMING, ModelCapability.LONG_CONTEXT
        ],
        input_price_per_m=15.0,
        output_price_per_m=75.0,
        family="claude-4",
    ),
    "anthropic/claude-sonnet-4": ModelInfo(
        id="anthropic/claude-sonnet-4",
        name="Claude Sonnet 4 (via OpenRouter)",
        provider="openrouter",
        context_window=200000,
        max_output_tokens=64000,
        capabilities=[
            ModelCapability.CHAT, ModelCapability.TOOL_USE,
            ModelCapability.VISION, ModelCapability.CODE,
            ModelCapability.STREAMING, ModelCapability.LONG_CONTEXT
        ],
        input_price_per_m=3.0,
        output_price_per_m=15.0,
        family="claude-4",
    ),
    # OpenAI
    "openai/gpt-5": ModelInfo(
        id="openai/gpt-5",
        name="GPT-5 (via OpenRouter)",
        provider="openrouter",
        context_window=256000,
        max_output_tokens=32000,
        capabilities=[
            ModelCapability.CHAT, ModelCapability.TOOL_USE,
            ModelCapability.VISION, ModelCapability.CODE,
            ModelCapability.STREAMING, ModelCapability.LONG_CONTEXT
        ],
        input_price_per_m=10.0,
        output_price_per_m=30.0,
        family="gpt-5",
    ),
    "openai/gpt-4o": ModelInfo(
        id="openai/gpt-4o",
        name="GPT-4o (via OpenRouter)",
        provider="openrouter",
        context_window=128000,
        max_output_tokens=16384,
        capabilities=[
            ModelCapability.CHAT, ModelCapability.TOOL_USE,
            ModelCapability.VISION, ModelCapability.CODE,
            ModelCapability.STREAMING, ModelCapability.LONG_CONTEXT
        ],
        input_price_per_m=2.5,
        output_price_per_m=10.0,
        family="gpt-4o",
    ),
    # Google
    "google/gemini-2.5-pro": ModelInfo(
        id="google/gemini-2.5-pro",
        name="Gemini 2.5 Pro (via OpenRouter)",
        provider="openrouter",
        context_window=1000000,
        max_output_tokens=65536,
        capabilities=[
            ModelCapability.CHAT, ModelCapability.TOOL_USE,
            ModelCapability.VISION, ModelCapability.CODE,
            ModelCapability.STREAMING, ModelCapability.LONG_CONTEXT
        ],
        input_price_per_m=1.25,
        output_price_per_m=5.0,
        family="gemini-2.5",
    ),
    "google/gemini-2.0-flash": ModelInfo(
        id="google/gemini-2.0-flash",
        name="Gemini 2.0 Flash (via OpenRouter)",
        provider="openrouter",
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
    ),
    # Meta
    "meta-llama/llama-4-maverick": ModelInfo(
        id="meta-llama/llama-4-maverick",
        name="Llama 4 Maverick (via OpenRouter)",
        provider="openrouter",
        context_window=256000,
        max_output_tokens=16384,
        capabilities=[
            ModelCapability.CHAT, ModelCapability.TOOL_USE,
            ModelCapability.CODE, ModelCapability.STREAMING,
            ModelCapability.LONG_CONTEXT
        ],
        input_price_per_m=0.0,  # Free tier
        output_price_per_m=0.0,
        family="llama-4",
    ),
    "meta-llama/llama-3.3-70b-instruct": ModelInfo(
        id="meta-llama/llama-3.3-70b-instruct",
        name="Llama 3.3 70B (via OpenRouter)",
        provider="openrouter",
        context_window=128000,
        max_output_tokens=8192,
        capabilities=[
            ModelCapability.CHAT, ModelCapability.TOOL_USE,
            ModelCapability.CODE, ModelCapability.STREAMING,
            ModelCapability.LONG_CONTEXT
        ],
        input_price_per_m=0.10,
        output_price_per_m=0.40,
        family="llama-3.3",
    ),
    # DeepSeek
    "deepseek/deepseek-v3": ModelInfo(
        id="deepseek/deepseek-v3",
        name="DeepSeek V3 (via OpenRouter)",
        provider="openrouter",
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
    ),
    "deepseek/deepseek-r1": ModelInfo(
        id="deepseek/deepseek-r1",
        name="DeepSeek R1 (via OpenRouter)",
        provider="openrouter",
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
    ),
    # Mistral
    "mistralai/mistral-large": ModelInfo(
        id="mistralai/mistral-large",
        name="Mistral Large (via OpenRouter)",
        provider="openrouter",
        context_window=128000,
        max_output_tokens=8192,
        capabilities=[
            ModelCapability.CHAT, ModelCapability.TOOL_USE,
            ModelCapability.CODE, ModelCapability.STREAMING,
            ModelCapability.LONG_CONTEXT
        ],
        input_price_per_m=2.0,
        output_price_per_m=6.0,
        family="mistral-large",
    ),
    "mistralai/mistral-small-3.1": ModelInfo(
        id="mistralai/mistral-small-3.1",
        name="Mistral Small 3.1 (via OpenRouter)",
        provider="openrouter",
        context_window=96000,
        max_output_tokens=8192,
        capabilities=[
            ModelCapability.CHAT, ModelCapability.TOOL_USE,
            ModelCapability.VISION, ModelCapability.CODE,
            ModelCapability.STREAMING
        ],
        input_price_per_m=0.0,  # Free
        output_price_per_m=0.0,
        family="mistral-small",
    ),
    # Qwen
    "qwen/qwen-2.5-72b-instruct": ModelInfo(
        id="qwen/qwen-2.5-72b-instruct",
        name="Qwen 2.5 72B (via OpenRouter)",
        provider="openrouter",
        context_window=128000,
        max_output_tokens=8192,
        capabilities=[
            ModelCapability.CHAT, ModelCapability.TOOL_USE,
            ModelCapability.CODE, ModelCapability.STREAMING,
            ModelCapability.LONG_CONTEXT
        ],
        input_price_per_m=0.15,
        output_price_per_m=0.40,
        family="qwen-2.5",
    ),
}


class OpenRouterBackend(BaseLLMBackend):
    """
    OpenRouter backend - access 400+ models through one API.

    OpenRouter provides a unified, OpenAI-compatible API that routes to
    models from Anthropic, OpenAI, Google, Meta, Mistral, DeepSeek, and more.

    Benefits:
    - Single API key for all providers
    - Automatic fallbacks
    - Consistent interface across providers
    - Access to free models (Llama, Mistral Small, etc.)
    - Usage consolidation and billing

    Usage:
        backend = OpenRouterBackend(
            model="anthropic/claude-sonnet-4",
            api_key="sk-or-..."  # or set OPENROUTER_API_KEY env var
        )
        response = backend.generate(messages=[{"role": "user", "content": "Hello"}])

    Model format: "provider/model-name" (e.g., "openai/gpt-5", "anthropic/claude-opus-4")
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    # Model aliases for convenience
    MODEL_ALIASES = {
        # Anthropic shortcuts
        "claude-opus": "anthropic/claude-opus-4",
        "claude-sonnet": "anthropic/claude-sonnet-4",
        "claude-haiku": "anthropic/claude-haiku-4",
        # OpenAI shortcuts
        "gpt-5": "openai/gpt-5",
        "gpt-4o": "openai/gpt-4o",
        "gpt-4o-mini": "openai/gpt-4o-mini",
        # Google shortcuts
        "gemini-pro": "google/gemini-2.5-pro",
        "gemini-flash": "google/gemini-2.0-flash",
        # DeepSeek shortcuts
        "deepseek": "deepseek/deepseek-v3",
        "deepseek-r1": "deepseek/deepseek-r1",
        # Meta shortcuts
        "llama": "meta-llama/llama-3.3-70b-instruct",
        "llama-4": "meta-llama/llama-4-maverick",
        # Free models
        "free": "meta-llama/llama-4-maverick",
        "free-mistral": "mistralai/mistral-small-3.1",
    }

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4",
        api_key: Optional[str] = None,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
        timeout: float = 120.0,
        temperature: float = 0.7,
        **kwargs
    ):
        """
        Initialize OpenRouter backend.

        Args:
            model: Model ID in "provider/model" format or alias
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            site_url: Optional URL of your app (for rankings/analytics)
            site_name: Optional name of your app
            timeout: Request timeout in seconds
            temperature: Sampling temperature
            **kwargs: Additional options
        """
        # Resolve model alias
        resolved_model = self.MODEL_ALIASES.get(model, model)

        super().__init__(
            model=resolved_model,
            api_key=api_key,
            api_key_env_var="OPENROUTER_API_KEY",
            base_url=self.OPENROUTER_BASE_URL,
            timeout=timeout,
            temperature=temperature,
            **kwargs
        )

        # Import openai (used for OpenAI-compatible API)
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI package not installed (required for OpenRouter). "
                "Install with: pip install openai"
            )

        if not self._api_key:
            raise ValueError(
                "OpenRouter API key required. Provide api_key parameter or set "
                "OPENROUTER_API_KEY environment variable. "
                "Get your key at https://openrouter.ai/keys"
            )

        # Extra headers for OpenRouter
        self.site_url = site_url
        self.site_name = site_name

        # Initialize OpenAI client pointing to OpenRouter
        self.client = openai.OpenAI(
            api_key=self._api_key,
            base_url=self.OPENROUTER_BASE_URL,
            timeout=timeout,
            default_headers=self._get_headers(),
        )
        self._openai = openai

        # Get model info
        self._model_info = OPENROUTER_MODELS.get(resolved_model)

        logger.info(f"OpenRouter backend initialized with model: {resolved_model}")

    def _get_headers(self) -> Dict[str, str]:
        """Get extra headers for OpenRouter."""
        headers = {}
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name
        return headers

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response via OpenRouter.

        Uses OpenAI-compatible API format.
        """
        try:
            request_kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": self.temperature,
            }

            # Add tools if provided
            if tools:
                request_kwargs["tools"] = convert_tools_to_openai_format(tools)
                request_kwargs["tool_choice"] = "auto"

            # OpenRouter-specific options
            extra_body = {}
            if "provider" in kwargs:
                extra_body["provider"] = kwargs.pop("provider")
            if "transforms" in kwargs:
                extra_body["transforms"] = kwargs.pop("transforms")

            if extra_body:
                request_kwargs["extra_body"] = extra_body

            request_kwargs.update(kwargs)

            # Make API call
            response = self.client.chat.completions.create(**request_kwargs)

            # Parse response
            message = response.choices[0].message
            content = message.content or ""

            tool_calls = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments) if tc.function.arguments else {}
                    })

            logger.debug(
                f"OpenRouter response: {len(content)} chars, "
                f"{len(tool_calls)} tool calls"
            )

            usage = {}
            if response.usage:
                usage = {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                }

            return {
                "content": content,
                "tool_calls": tool_calls,
                "usage": usage,
                "finish_reason": response.choices[0].finish_reason,
            }

        except self._openai.APIError as e:
            logger.error(f"OpenRouter API error: {e}")
            raise RuntimeError(f"OpenRouter API call failed: {e}") from e

    def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        **kwargs
    ) -> Iterator[str]:
        """Generate a streaming response via OpenRouter."""
        try:
            request_kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": self.temperature,
                "stream": True,
            }

            if tools:
                request_kwargs["tools"] = convert_tools_to_openai_format(tools)
                request_kwargs["tool_choice"] = "auto"

            request_kwargs.update(kwargs)

            stream = self.client.chat.completions.create(**request_kwargs)

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except self._openai.APIError as e:
            logger.error(f"OpenRouter streaming error: {e}")
            raise RuntimeError(f"OpenRouter streaming failed: {e}") from e

    def count_tokens(self, text: str) -> int:
        """
        Approximate token count.

        OpenRouter doesn't provide token counting, so we estimate.
        """
        # Use tiktoken if available
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            # Rough approximation
            return len(text) // 4

    @property
    def supports_tools(self) -> bool:
        """Check if the current model supports tools."""
        if self._model_info:
            return self._model_info.supports(ModelCapability.TOOL_USE)
        # Assume yes for most models
        return True

    @property
    def supports_streaming(self) -> bool:
        """OpenRouter supports streaming for most models."""
        return True

    @classmethod
    def list_available_models(cls) -> List[str]:
        """List popular models available on OpenRouter."""
        return list(OPENROUTER_MODELS.keys())

    @classmethod
    def get_free_models(cls) -> List[str]:
        """Get list of free models available on OpenRouter."""
        return [
            model_id for model_id, info in OPENROUTER_MODELS.items()
            if info.input_price_per_m == 0.0
        ]

    def __repr__(self) -> str:
        return f"OpenRouterBackend(model={self.model})"
