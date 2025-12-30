"""
OpenAI Backend Implementation

Real, working backend for OpenAI models (GPT-5, GPT-4o, O1 family).
Supports both the new Responses API and legacy Chat Completions API.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Iterator
from enum import Enum

from .base import (
    BaseLLMBackend,
    convert_tools_to_openai_format,
    parse_openai_tool_calls,
)

logger = logging.getLogger(__name__)


class OpenAIAPIMode(Enum):
    """Which OpenAI API to use."""
    RESPONSES = "responses"  # New Responses API (recommended for GPT-5+)
    CHAT_COMPLETIONS = "chat_completions"  # Legacy Chat Completions
    AUTO = "auto"  # Auto-select based on model


class OpenAIBackend(BaseLLMBackend):
    """
    Real OpenAI backend implementation.

    Supports:
    - GPT-5 (latest flagship)
    - GPT-4o, GPT-4o-mini (multimodal)
    - O1, O1-mini (reasoning models)
    - Tool/function calling
    - Vision (image inputs)
    - Streaming responses
    - Both Responses API (new) and Chat Completions (legacy)

    Usage:
        backend = OpenAIBackend(
            model="gpt-5",
            api_key="sk-..."  # or set OPENAI_API_KEY env var
        )
        response = backend.generate(messages=[{"role": "user", "content": "Hello"}])
    """

    # Model aliases for convenience
    MODEL_ALIASES = {
        "gpt5": "gpt-5",
        "gpt-5": "gpt-5",
        "gpt4o": "gpt-4o",
        "gpt-4": "gpt-4o",  # Redirect legacy to 4o
        "gpt4o-mini": "gpt-4o-mini",
        "gpt-4-mini": "gpt-4o-mini",
        "mini": "gpt-4o-mini",
        "o1": "o1",
        "o1-mini": "o1-mini",
    }

    # Models that should use Responses API
    RESPONSES_API_MODELS = {"gpt-5", "gpt-4.1", "o1", "o1-mini", "o1-preview"}

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        temperature: float = 0.7,
        api_mode: OpenAIAPIMode = OpenAIAPIMode.AUTO,
        **kwargs
    ):
        """
        Initialize OpenAI backend.

        Args:
            model: Model ID or alias
            api_key: API key (or set OPENAI_API_KEY env var)
            organization: Optional organization ID
            base_url: Optional API base URL override
            timeout: Request timeout in seconds
            temperature: Sampling temperature
            api_mode: Which API to use (AUTO, RESPONSES, CHAT_COMPLETIONS)
            **kwargs: Additional client options
        """
        # Resolve model alias
        resolved_model = self.MODEL_ALIASES.get(model, model)

        super().__init__(
            model=resolved_model,
            api_key=api_key,
            api_key_env_var="OPENAI_API_KEY",
            base_url=base_url,
            timeout=timeout,
            temperature=temperature,
            **kwargs
        )

        # Import openai
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )

        if not self._api_key:
            raise ValueError(
                "OpenAI API key required. Provide api_key parameter or set "
                "OPENAI_API_KEY environment variable."
            )

        # Initialize client
        client_kwargs = {
            "api_key": self._api_key,
            "timeout": timeout,
        }
        if organization:
            client_kwargs["organization"] = organization
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = openai.OpenAI(**client_kwargs)
        self._openai = openai

        # Determine API mode
        self.api_mode = api_mode
        if api_mode == OpenAIAPIMode.AUTO:
            # Use Responses API for newer models
            if resolved_model in self.RESPONSES_API_MODELS:
                self._use_responses_api = True
            else:
                self._use_responses_api = False
        else:
            self._use_responses_api = (api_mode == OpenAIAPIMode.RESPONSES)

        # Initialize tokenizer
        self._tokenizer = None
        self._init_tokenizer()

        logger.info(
            f"OpenAI backend initialized with model: {resolved_model}, "
            f"api={'responses' if self._use_responses_api else 'chat_completions'}"
        )

    def _init_tokenizer(self):
        """Initialize tiktoken tokenizer for token counting."""
        try:
            import tiktoken
            try:
                self._tokenizer = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # Fallback for newer models
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            logger.warning(
                "tiktoken not installed. Token counting will be approximate. "
                "Install with: pip install tiktoken"
            )

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response from OpenAI.

        Automatically selects Responses API or Chat Completions based on model.
        """
        if self._use_responses_api:
            return self._generate_responses_api(messages, tools, max_tokens, **kwargs)
        else:
            return self._generate_chat_completions(messages, tools, max_tokens, **kwargs)

    def _generate_chat_completions(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate using Chat Completions API."""
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

            request_kwargs.update(kwargs)

            # Make API call
            response = self.client.chat.completions.create(**request_kwargs)

            # Parse response
            message = response.choices[0].message
            content = message.content or ""
            tool_calls = parse_openai_tool_calls(message.tool_calls)

            logger.debug(
                f"OpenAI response: {len(content)} chars, "
                f"{len(tool_calls)} tool calls"
            )

            return {
                "content": content,
                "tool_calls": tool_calls,
                "usage": {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                },
                "finish_reason": response.choices[0].finish_reason,
            }

        except self._openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"OpenAI API call failed: {e}") from e

    def _generate_responses_api(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate using the new Responses API.

        The Responses API provides better performance with GPT-5 and reasoning models.
        """
        try:
            # Convert messages to Responses API format
            # The Responses API uses 'input' instead of 'messages'
            request_kwargs = {
                "model": self.model,
                "input": messages,  # Responses API accepts messages as input
                "max_output_tokens": max_tokens,
                "temperature": self.temperature,
            }

            # Add tools if provided
            if tools:
                openai_tools = []
                for tool in tools:
                    openai_tools.append({
                        "type": "function",
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {"type": "object", "properties": {}})
                    })
                request_kwargs["tools"] = openai_tools

            request_kwargs.update(kwargs)

            # Make API call to Responses endpoint
            response = self.client.responses.create(**request_kwargs)

            # Parse response - Responses API has different structure
            content = ""
            tool_calls = []

            # The Responses API returns output differently
            if hasattr(response, 'output'):
                for item in response.output:
                    if hasattr(item, 'type'):
                        if item.type == 'message':
                            for content_block in item.content:
                                if hasattr(content_block, 'text'):
                                    content += content_block.text
                        elif item.type == 'function_call':
                            tool_calls.append({
                                "id": getattr(item, 'call_id', ''),
                                "name": item.name,
                                "arguments": json.loads(item.arguments) if isinstance(item.arguments, str) else item.arguments
                            })
            elif hasattr(response, 'output_text'):
                content = response.output_text

            logger.debug(
                f"OpenAI Responses API: {len(content)} chars, "
                f"{len(tool_calls)} tool calls"
            )

            usage = {}
            if hasattr(response, 'usage'):
                usage = {
                    "input_tokens": getattr(response.usage, 'input_tokens', 0),
                    "output_tokens": getattr(response.usage, 'output_tokens', 0),
                }

            return {
                "content": content,
                "tool_calls": tool_calls,
                "usage": usage,
            }

        except AttributeError:
            # Responses API not available, fall back to Chat Completions
            logger.warning("Responses API not available, falling back to Chat Completions")
            self._use_responses_api = False
            return self._generate_chat_completions(messages, tools, max_tokens, **kwargs)
        except self._openai.APIError as e:
            logger.error(f"OpenAI Responses API error: {e}")
            raise RuntimeError(f"OpenAI Responses API call failed: {e}") from e

    def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        **kwargs
    ) -> Iterator[str]:
        """
        Generate a streaming response.

        Respects the _use_responses_api setting:
        - For Responses API models: uses responses.create with stream=True
        - For Chat Completions models: uses chat.completions.create with stream=True
        """
        if self._use_responses_api:
            yield from self._stream_responses_api(messages, tools, max_tokens, **kwargs)
        else:
            yield from self._stream_chat_completions(messages, tools, max_tokens, **kwargs)

    def _stream_chat_completions(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        **kwargs
    ) -> Iterator[str]:
        """Stream using Chat Completions API."""
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
            logger.error(f"OpenAI Chat Completions streaming error: {e}")
            raise RuntimeError(f"OpenAI streaming failed: {e}") from e

    def _stream_responses_api(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        **kwargs
    ) -> Iterator[str]:
        """
        Stream using the Responses API.

        The Responses API provides semantic streaming events for GPT-5 and O1 models.
        """
        try:
            request_kwargs = {
                "model": self.model,
                "input": messages,
                "max_output_tokens": max_tokens,
                "temperature": self.temperature,
                "stream": True,
            }

            # Add tools if provided
            if tools:
                openai_tools = []
                for tool in tools:
                    openai_tools.append({
                        "type": "function",
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {"type": "object", "properties": {}})
                    })
                request_kwargs["tools"] = openai_tools

            request_kwargs.update(kwargs)

            # Stream using Responses API
            stream = self.client.responses.create(**request_kwargs)

            for event in stream:
                # Handle different event types from Responses API
                if hasattr(event, 'type'):
                    if event.type == 'content.delta':
                        if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
                            yield event.delta.text
                    elif event.type == 'content.text.delta':
                        if hasattr(event, 'text'):
                            yield event.text
                # Fallback for simpler response structure
                elif hasattr(event, 'choices') and event.choices:
                    delta = event.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        yield delta.content

        except AttributeError:
            # Responses API streaming not available, fall back to Chat Completions
            logger.warning(
                f"Responses API streaming not available for {self.model}, "
                "falling back to Chat Completions"
            )
            self._use_responses_api = False
            yield from self._stream_chat_completions(messages, tools, max_tokens, **kwargs)
        except self._openai.APIError as e:
            logger.error(f"OpenAI Responses API streaming error: {e}")
            raise RuntimeError(f"OpenAI Responses API streaming failed: {e}") from e

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        else:
            # Fallback approximation
            return len(text) // 4

    @property
    def supports_tools(self) -> bool:
        """OpenAI models support function/tool calling."""
        # O1 models have limited tool support
        if "o1" in self.model:
            return False
        return True

    @property
    def supports_streaming(self) -> bool:
        """Most OpenAI models support streaming."""
        return True

    @property
    def supports_vision(self) -> bool:
        """GPT-4o and GPT-5 support vision."""
        return "4o" in self.model or "5" in self.model or "gpt-4-vision" in self.model

    def __repr__(self) -> str:
        api = "responses" if self._use_responses_api else "chat_completions"
        return f"OpenAIBackend(model={self.model}, api={api})"
