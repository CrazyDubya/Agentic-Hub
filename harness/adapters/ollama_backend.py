"""
Ollama Backend Implementation

Real, working backend for running local LLM models via Ollama.
Supports Llama, Mistral, Qwen, DeepSeek, CodeLlama, and more.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Iterator

from .base import (
    BaseLLMBackend,
    ModelInfo,
    ModelCapability,
    OLLAMA_MODELS,
)

logger = logging.getLogger(__name__)


class OllamaBackend(BaseLLMBackend):
    """
    Ollama backend for running local LLM models.

    Ollama runs models locally on your machine. No API key needed,
    just install Ollama and pull models.

    Supports:
    - Llama 3.3, 3.2 (Meta)
    - Mistral, Mixtral (Mistral AI)
    - Qwen 2.5 (Alibaba)
    - DeepSeek R1, V3
    - CodeLlama (code-focused)
    - Gemma 2 (Google)
    - Phi-3 (Microsoft)
    - Tool/function calling (for supported models)
    - Streaming responses

    Prerequisites:
        1. Install Ollama: https://ollama.ai
        2. Pull a model: ollama pull llama3.3
        3. Ollama runs on http://localhost:11434 by default

    Usage:
        backend = OllamaBackend(model="llama3.3:70b")
        response = backend.generate(messages=[{"role": "user", "content": "Hello"}])
    """

    DEFAULT_BASE_URL = "http://localhost:11434"

    # Model aliases for convenience
    MODEL_ALIASES = {
        # Llama family
        "llama": "llama3.3:70b",
        "llama3": "llama3.3:70b",
        "llama3.3": "llama3.3:70b",
        "llama3.2": "llama3.2:3b",
        "llama-small": "llama3.2:3b",
        # Mistral family
        "mistral": "mistral:7b",
        "mixtral": "mixtral:8x7b",
        # Qwen family
        "qwen": "qwen2.5:72b",
        "qwen-small": "qwen2.5:7b",
        # DeepSeek
        "deepseek": "deepseek-r1:70b",
        "deepseek-coder": "deepseek-coder:33b",
        # Code models
        "codellama": "codellama:34b",
        "code": "codellama:34b",
        # Other
        "gemma": "gemma2:27b",
        "phi": "phi3:14b",
    }

    # Models that support tool calling
    TOOL_CAPABLE_MODELS = {
        "llama3.3", "llama3.2", "llama3.1",
        "mistral", "mixtral",
        "qwen2.5", "qwen2",
        "command-r",
    }

    def __init__(
        self,
        model: str = "llama3.3:70b",
        base_url: Optional[str] = None,
        timeout: float = 300.0,  # Longer timeout for local inference
        temperature: float = 0.7,
        num_ctx: int = 8192,  # Context window (adjust based on RAM)
        num_gpu: Optional[int] = None,  # GPU layers to offload
        **kwargs
    ):
        """
        Initialize Ollama backend.

        Args:
            model: Model name (e.g., "llama3.3:70b", "mistral:7b")
            base_url: Ollama API URL (default: http://localhost:11434)
            timeout: Request timeout (longer for local inference)
            temperature: Sampling temperature
            num_ctx: Context window size (memory-dependent)
            num_gpu: Number of GPU layers to offload (None = auto)
            **kwargs: Additional Ollama options
        """
        # Resolve model alias
        resolved_model = self.MODEL_ALIASES.get(model, model)

        super().__init__(
            model=resolved_model,
            api_key=None,  # No API key needed for local
            base_url=base_url or self.DEFAULT_BASE_URL,
            timeout=timeout,
            temperature=temperature,
            **kwargs
        )

        self.num_ctx = num_ctx
        self.num_gpu = num_gpu

        # Check if Ollama is running
        self._check_ollama_available()

        # Get model info
        self._model_info = OLLAMA_MODELS.get(resolved_model)

        logger.info(f"Ollama backend initialized with model: {resolved_model}")

    def _check_ollama_available(self):
        """Check if Ollama server is running."""
        import urllib.request
        import urllib.error

        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionRefusedError):
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: https://ollama.ai"
            )
        except Exception as e:
            logger.warning(f"Error checking Ollama availability: {e}")

    def _get_ollama_client(self):
        """Get or create Ollama client."""
        try:
            import ollama
            return ollama.Client(host=self.base_url)
        except ImportError:
            # Fallback to direct HTTP if ollama package not installed
            return None

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response from Ollama.

        Uses the ollama Python package if available, otherwise falls back
        to direct HTTP API calls.
        """
        client = self._get_ollama_client()

        if client:
            return self._generate_with_client(client, messages, tools, max_tokens, **kwargs)
        else:
            return self._generate_with_http(messages, tools, max_tokens, **kwargs)

    def _generate_with_client(
        self,
        client,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate using the ollama Python package."""
        try:
            # Build options
            options = {
                "temperature": self.temperature,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
            }
            if self.num_gpu is not None:
                options["num_gpu"] = self.num_gpu

            # Build request
            request_kwargs = {
                "model": self.model,
                "messages": messages,
                "options": options,
            }

            # Add tools if provided and model supports them
            if tools and self._supports_tools_for_model():
                # Convert to Ollama tool format (OpenAI-compatible)
                ollama_tools = []
                for tool in tools:
                    ollama_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {"type": "object", "properties": {}})
                        }
                    })
                request_kwargs["tools"] = ollama_tools

            # Make API call
            response = client.chat(**request_kwargs)

            # Parse response
            content = response.get("message", {}).get("content", "")

            tool_calls = []
            if "message" in response and "tool_calls" in response["message"]:
                for tc in response["message"]["tool_calls"]:
                    tool_calls.append({
                        "id": tc.get("id", ""),
                        "name": tc["function"]["name"],
                        "arguments": tc["function"].get("arguments", {})
                    })

            logger.debug(
                f"Ollama response: {len(content)} chars, "
                f"{len(tool_calls)} tool calls"
            )

            # Get token counts if available
            usage = {}
            if "prompt_eval_count" in response:
                usage["input_tokens"] = response["prompt_eval_count"]
            if "eval_count" in response:
                usage["output_tokens"] = response["eval_count"]

            return {
                "content": content,
                "tool_calls": tool_calls,
                "usage": usage,
            }

        except Exception as e:
            logger.error(f"Ollama error: {e}")
            raise RuntimeError(f"Ollama API call failed: {e}") from e

    def _generate_with_http(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate using direct HTTP API calls (fallback)."""
        import urllib.request
        import urllib.error

        try:
            # Build request
            options = {
                "temperature": self.temperature,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
            }
            if self.num_gpu is not None:
                options["num_gpu"] = self.num_gpu

            payload = {
                "model": self.model,
                "messages": messages,
                "options": options,
                "stream": False,
            }

            # Add tools if supported
            if tools and self._supports_tools_for_model():
                ollama_tools = []
                for tool in tools:
                    ollama_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {"type": "object", "properties": {}})
                        }
                    })
                payload["tools"] = ollama_tools

            # Make HTTP request
            req = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))

            # Parse response
            content = result.get("message", {}).get("content", "")

            tool_calls = []
            if "message" in result and "tool_calls" in result["message"]:
                for tc in result["message"]["tool_calls"]:
                    tool_calls.append({
                        "id": tc.get("id", ""),
                        "name": tc["function"]["name"],
                        "arguments": tc["function"].get("arguments", {})
                    })

            usage = {}
            if "prompt_eval_count" in result:
                usage["input_tokens"] = result["prompt_eval_count"]
            if "eval_count" in result:
                usage["output_tokens"] = result["eval_count"]

            return {
                "content": content,
                "tool_calls": tool_calls,
                "usage": usage,
            }

        except urllib.error.URLError as e:
            logger.error(f"Ollama HTTP error: {e}")
            raise RuntimeError(f"Ollama HTTP call failed: {e}") from e

    def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        **kwargs
    ) -> Iterator[str]:
        """Generate a streaming response from Ollama."""
        client = self._get_ollama_client()

        if client:
            yield from self._stream_with_client(client, messages, tools, max_tokens, **kwargs)
        else:
            yield from self._stream_with_http(messages, tools, max_tokens, **kwargs)

    def _stream_with_client(
        self,
        client,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        **kwargs
    ) -> Iterator[str]:
        """Stream using ollama Python package."""
        try:
            options = {
                "temperature": self.temperature,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
            }

            request_kwargs = {
                "model": self.model,
                "messages": messages,
                "options": options,
                "stream": True,
            }

            # Add tools if provided and model supports them
            if tools and self._supports_tools_for_model():
                ollama_tools = []
                for tool in tools:
                    ollama_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {"type": "object", "properties": {}})
                        }
                    })
                request_kwargs["tools"] = ollama_tools
                # Note: Ollama streaming with tools may not return tool calls inline
                # Tool calls typically come at the end of the stream
                logger.debug("Streaming with tools enabled (tool calls may appear at stream end)")

            for chunk in client.chat(**request_kwargs):
                if "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]

        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            raise RuntimeError(f"Ollama streaming failed: {e}") from e

    def _stream_with_http(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        **kwargs
    ) -> Iterator[str]:
        """Stream using direct HTTP (fallback)."""
        import urllib.request

        try:
            options = {
                "temperature": self.temperature,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
            }

            payload = {
                "model": self.model,
                "messages": messages,
                "options": options,
                "stream": True,
            }

            # Add tools if provided and model supports them
            if tools and self._supports_tools_for_model():
                ollama_tools = []
                for tool in tools:
                    ollama_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {"type": "object", "properties": {}})
                        }
                    })
                payload["tools"] = ollama_tools
                logger.debug("HTTP streaming with tools enabled")

            req = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                for line in response:
                    if line:
                        chunk = json.loads(line.decode("utf-8"))
                        if "message" in chunk and "content" in chunk["message"]:
                            yield chunk["message"]["content"]

        except Exception as e:
            logger.error(f"Ollama HTTP streaming error: {e}")
            raise RuntimeError(f"Ollama HTTP streaming failed: {e}") from e

    def count_tokens(self, text: str) -> int:
        """
        Approximate token count for Ollama models.

        Most Ollama models use similar tokenizers to Llama.
        """
        # Try tiktoken first (Llama uses similar tokenization)
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            # Fallback: ~3.5 chars per token for Llama-like models
            return int(len(text) / 3.5)

    def _supports_tools_for_model(self) -> bool:
        """Check if current model supports tool calling."""
        model_base = self.model.split(":")[0]
        for capable in self.TOOL_CAPABLE_MODELS:
            if model_base.startswith(capable):
                return True
        return False

    @property
    def supports_tools(self) -> bool:
        """Check if current model supports tools."""
        return self._supports_tools_for_model()

    @property
    def supports_streaming(self) -> bool:
        """Ollama supports streaming for all models."""
        return True

    def list_local_models(self) -> List[str]:
        """List models available locally in Ollama."""
        import urllib.request

        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode("utf-8"))
                return [m["name"] for m in result.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list local models: {e}")
            return []

    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry."""
        client = self._get_ollama_client()
        if client:
            try:
                client.pull(model_name)
                logger.info(f"Successfully pulled model: {model_name}")
                return True
            except Exception as e:
                logger.error(f"Failed to pull model: {e}")
                return False
        else:
            logger.warning("ollama package not installed, cannot pull models")
            return False

    def __repr__(self) -> str:
        return f"OllamaBackend(model={self.model}, url={self.base_url})"
