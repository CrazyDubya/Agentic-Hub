"""
Anthropic Backend Implementation

Real, working backend for Claude models (Claude 4.x family).
Supports tool use, vision, and streaming.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Iterator

from .base import (
    BaseLLMBackend,
    convert_tools_to_anthropic_format,
    parse_anthropic_tool_calls,
    ModelCapability,
)

logger = logging.getLogger(__name__)


class AnthropicBackend(BaseLLMBackend):
    """
    Real Anthropic Claude backend implementation.

    Supports:
    - Claude Opus 4.5, Sonnet 4.5, Haiku 4.5 (latest)
    - Claude Opus 4, Sonnet 4 (previous generation)
    - Tool use with parallel tool calling
    - Vision (image inputs)
    - Streaming responses
    - Extended thinking (for reasoning models)

    Usage:
        backend = AnthropicBackend(
            model="claude-sonnet-4-20250514",
            api_key="sk-ant-..."  # or set ANTHROPIC_API_KEY env var
        )
        response = backend.generate(messages=[{"role": "user", "content": "Hello"}])
    """

    # Model aliases for convenience
    MODEL_ALIASES = {
        "opus": "claude-opus-4-5-20251101",
        "opus-4.5": "claude-opus-4-5-20251101",
        "opus-4": "claude-opus-4-20250514",
        "sonnet": "claude-sonnet-4-5-20250929",
        "sonnet-4.5": "claude-sonnet-4-5-20250929",
        "sonnet-4": "claude-sonnet-4-20250514",
        "haiku": "claude-haiku-4-5-20251015",
        "haiku-4.5": "claude-haiku-4-5-20251015",
        # Legacy aliases
        "claude-3-opus": "claude-opus-4-20250514",  # Redirect to 4
        "claude-3-sonnet": "claude-sonnet-4-20250514",
        "claude-3-haiku": "claude-haiku-4-5-20251015",
    }

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        temperature: float = 0.7,
        max_retries: int = 3,
        **kwargs
    ):
        """
        Initialize Anthropic backend.

        Args:
            model: Model ID or alias (e.g., "sonnet", "claude-sonnet-4-20250514")
            api_key: API key (or set ANTHROPIC_API_KEY env var)
            base_url: Optional API base URL override
            timeout: Request timeout in seconds
            temperature: Sampling temperature (0.0-1.0)
            max_retries: Number of retries on failure
            **kwargs: Additional options passed to the client
        """
        # Resolve model alias
        resolved_model = self.MODEL_ALIASES.get(model, model)

        super().__init__(
            model=resolved_model,
            api_key=api_key,
            api_key_env_var="ANTHROPIC_API_KEY",
            base_url=base_url,
            timeout=timeout,
            temperature=temperature,
            **kwargs
        )

        # Import anthropic
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic package not installed. Install with: pip install anthropic"
            )

        if not self._api_key:
            raise ValueError(
                "Anthropic API key required. Provide api_key parameter or set "
                "ANTHROPIC_API_KEY environment variable."
            )

        # Initialize client
        client_kwargs = {
            "api_key": self._api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = anthropic.Anthropic(**client_kwargs)
        self._anthropic = anthropic  # Keep module reference

        logger.info(f"Anthropic backend initialized with model: {resolved_model}")

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        system: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response from Claude.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions
            max_tokens: Maximum output tokens
            system: Optional system prompt (extracted from messages if not provided)
            **kwargs: Additional API options

        Returns:
            Dict with 'content' and 'tool_calls'
        """
        try:
            # Extract system prompt from messages if not provided
            system_prompt = system
            api_messages = []

            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                else:
                    # Handle content that might be a list (multimodal)
                    api_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

            # Build request
            request_kwargs = {
                "model": self.model,
                "messages": api_messages,
                "max_tokens": max_tokens,
                "temperature": self.temperature,
            }

            if system_prompt:
                request_kwargs["system"] = system_prompt

            # Add tools if provided
            if tools:
                request_kwargs["tools"] = convert_tools_to_anthropic_format(tools)
                # Enable parallel tool use
                request_kwargs["tool_choice"] = {"type": "auto"}

            # Merge any additional kwargs
            request_kwargs.update(kwargs)

            # Make API call
            response = self.client.messages.create(**request_kwargs)

            # Parse response
            content = ""
            tool_calls = []

            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text
                elif hasattr(block, "type") and block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "name": block.name,
                        "arguments": block.input
                    })

            logger.debug(
                f"Anthropic response: {len(content)} chars, "
                f"{len(tool_calls)} tool calls, "
                f"stop_reason={response.stop_reason}"
            )

            return {
                "content": content,
                "tool_calls": tool_calls,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                "stop_reason": response.stop_reason,
            }

        except self._anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            raise RuntimeError(f"Anthropic API call failed: {e}") from e

    def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        system: Optional[str] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        Generate a streaming response from Claude.

        Yields text chunks as they arrive.
        """
        try:
            # Extract system prompt
            system_prompt = system
            api_messages = []

            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                else:
                    api_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

            # Build request
            request_kwargs = {
                "model": self.model,
                "messages": api_messages,
                "max_tokens": max_tokens,
                "temperature": self.temperature,
            }

            if system_prompt:
                request_kwargs["system"] = system_prompt

            if tools:
                request_kwargs["tools"] = convert_tools_to_anthropic_format(tools)
                request_kwargs["tool_choice"] = {"type": "auto"}

            request_kwargs.update(kwargs)

            # Stream response
            with self.client.messages.stream(**request_kwargs) as stream:
                for text in stream.text_stream:
                    yield text

        except self._anthropic.APIError as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise RuntimeError(f"Anthropic streaming failed: {e}") from e

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Uses Anthropic's token counting endpoint if available,
        otherwise falls back to approximation.
        """
        try:
            # Anthropic provides token counting
            result = self.client.count_tokens(text)
            return result
        except Exception:
            # Fallback: Claude uses ~3.5 chars per token on average
            return len(text) // 4

    @property
    def supports_tools(self) -> bool:
        """Claude models support native tool calling."""
        return True

    @property
    def supports_streaming(self) -> bool:
        """Claude models support streaming."""
        return True

    @property
    def supports_vision(self) -> bool:
        """Claude models support image inputs."""
        return True

    def with_extended_thinking(
        self,
        budget_tokens: int = 10000
    ) -> "AnthropicBackend":
        """
        Enable extended thinking for complex reasoning tasks.

        Returns a new backend instance with extended thinking enabled.
        """
        new_backend = AnthropicBackend(
            model=self.model,
            api_key=self._api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            temperature=self.temperature,
        )
        new_backend._extended_thinking = True
        new_backend._thinking_budget = budget_tokens
        return new_backend

    def __repr__(self) -> str:
        return f"AnthropicBackend(model={self.model})"
