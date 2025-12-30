"""
LLM Backend Adapters

Real, working implementations for various LLM providers.
All backends connect to actual APIs and support tool calling.

Available Backends:
    - AnthropicBackend: Claude models (Opus 4.5, Sonnet 4.5, Haiku 4.5)
    - OpenAIBackend: OpenAI models (GPT-5, GPT-4o, O1)
    - OpenRouterBackend: 400+ models via unified API
    - OllamaBackend: Local models (Llama, Mistral, Qwen, etc.)

Quick Start:
    from harness.adapters import create_backend

    # Use environment variables for API keys
    backend = create_backend("anthropic", model="claude-sonnet-4-20250514")
    backend = create_backend("openai", model="gpt-4o")
    backend = create_backend("openrouter", model="deepseek/deepseek-v3")
    backend = create_backend("ollama", model="llama3.3:70b")

    # Generate response
    response = backend.generate(messages=[{"role": "user", "content": "Hello"}])
"""

import os
import logging
from typing import Optional, Dict, Any

# Base classes and utilities
from .base import (
    BaseLLMBackend,
    ModelInfo,
    ModelCapability,
    GenerateResponse,
    MODEL_REGISTRY,
    ANTHROPIC_MODELS,
    OPENAI_MODELS,
    GOOGLE_MODELS,
    DEEPSEEK_MODELS,
    OLLAMA_MODELS,
    get_model_info,
    list_models_by_provider,
    list_recommended_models,
    convert_tools_to_openai_format,
    convert_tools_to_anthropic_format,
)

# Re-export LLMBackend from core for compatibility
from ..core.llm_harness import LLMBackend

logger = logging.getLogger(__name__)


# ============================================================================
# Lazy imports for backends (avoid import errors if dependencies missing)
# ============================================================================

def _import_anthropic_backend():
    """Lazily import AnthropicBackend."""
    from .anthropic_backend import AnthropicBackend
    return AnthropicBackend


def _import_openai_backend():
    """Lazily import OpenAIBackend."""
    from .openai_backend import OpenAIBackend, OpenAIAPIMode
    return OpenAIBackend


def _import_openrouter_backend():
    """Lazily import OpenRouterBackend."""
    from .openrouter_backend import OpenRouterBackend
    return OpenRouterBackend


def _import_ollama_backend():
    """Lazily import OllamaBackend."""
    from .ollama_backend import OllamaBackend
    return OllamaBackend


# ============================================================================
# Backend Factory
# ============================================================================

def create_backend(
    provider: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> LLMBackend:
    """
    Create a backend for the specified provider.

    This is the recommended way to create backends as it handles
    import errors gracefully and provides sensible defaults.

    Args:
        provider: Provider name ("anthropic", "openai", "openrouter", "ollama")
        model: Model name/ID (provider-specific defaults if not specified)
        api_key: API key (or use environment variables)
        **kwargs: Additional provider-specific options

    Returns:
        Configured LLM backend

    Examples:
        # Anthropic (uses ANTHROPIC_API_KEY env var)
        backend = create_backend("anthropic", model="claude-sonnet-4-20250514")

        # OpenAI (uses OPENAI_API_KEY env var)
        backend = create_backend("openai", model="gpt-4o")

        # OpenRouter (uses OPENROUTER_API_KEY env var)
        backend = create_backend("openrouter", model="deepseek/deepseek-v3")

        # Ollama (no API key needed, runs locally)
        backend = create_backend("ollama", model="llama3.3:70b")

    Environment Variables:
        ANTHROPIC_API_KEY - For Anthropic backend
        OPENAI_API_KEY - For OpenAI backend
        OPENROUTER_API_KEY - For OpenRouter backend
    """
    provider = provider.lower().strip()

    # Provider aliases
    provider_aliases = {
        "claude": "anthropic",
        "gpt": "openai",
        "chatgpt": "openai",
        "router": "openrouter",
        "local": "ollama",
    }
    provider = provider_aliases.get(provider, provider)

    try:
        if provider == "anthropic":
            Backend = _import_anthropic_backend()
            return Backend(
                model=model or "claude-sonnet-4-20250514",
                api_key=api_key,
                **kwargs
            )

        elif provider == "openai":
            Backend = _import_openai_backend()
            return Backend(
                model=model or "gpt-4o",
                api_key=api_key,
                **kwargs
            )

        elif provider == "openrouter":
            Backend = _import_openrouter_backend()
            return Backend(
                model=model or "anthropic/claude-sonnet-4",
                api_key=api_key,
                **kwargs
            )

        elif provider == "ollama":
            Backend = _import_ollama_backend()
            return Backend(
                model=model or "llama3.3:70b",
                **kwargs  # No API key for Ollama
            )

        else:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Supported: anthropic, openai, openrouter, ollama"
            )

    except ImportError as e:
        raise ImportError(
            f"Failed to import {provider} backend. "
            f"Make sure required packages are installed:\n"
            f"  - anthropic: pip install anthropic\n"
            f"  - openai: pip install openai\n"
            f"  - openrouter: pip install openai\n"
            f"  - ollama: pip install ollama (optional)\n"
            f"Original error: {e}"
        )


def create_backend_from_env() -> LLMBackend:
    """
    Create a backend based on available environment variables.

    Checks for API keys in order of preference:
    1. ANTHROPIC_API_KEY -> Anthropic backend
    2. OPENAI_API_KEY -> OpenAI backend
    3. OPENROUTER_API_KEY -> OpenRouter backend
    4. Falls back to Ollama (local) if available

    Returns:
        Configured LLM backend

    Raises:
        ValueError: If no API keys found and Ollama not available
    """
    # Check for API keys
    if os.environ.get("ANTHROPIC_API_KEY"):
        logger.info("Using Anthropic backend (found ANTHROPIC_API_KEY)")
        return create_backend("anthropic")

    if os.environ.get("OPENAI_API_KEY"):
        logger.info("Using OpenAI backend (found OPENAI_API_KEY)")
        return create_backend("openai")

    if os.environ.get("OPENROUTER_API_KEY"):
        logger.info("Using OpenRouter backend (found OPENROUTER_API_KEY)")
        return create_backend("openrouter")

    # Try Ollama as fallback
    try:
        backend = create_backend("ollama")
        logger.info("Using Ollama backend (local models)")
        return backend
    except (ConnectionError, ImportError) as e:
        raise ValueError(
            "No LLM backend available. Please set one of:\n"
            "  - ANTHROPIC_API_KEY\n"
            "  - OPENAI_API_KEY\n"
            "  - OPENROUTER_API_KEY\n"
            "Or install Ollama for local models: https://ollama.ai"
        ) from e


def list_available_providers() -> Dict[str, bool]:
    """
    Check which providers are available (have required packages).

    Returns:
        Dict mapping provider name to availability status
    """
    availability = {}

    # Check Anthropic
    try:
        import anthropic
        availability["anthropic"] = True
    except ImportError:
        availability["anthropic"] = False

    # Check OpenAI
    try:
        import openai
        availability["openai"] = True
        availability["openrouter"] = True  # Uses openai package
    except ImportError:
        availability["openai"] = False
        availability["openrouter"] = False

    # Check Ollama
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:11434/api/tags")
        urllib.request.urlopen(req, timeout=2)
        availability["ollama"] = True
    except Exception:
        availability["ollama"] = False

    return availability


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Base classes
    "LLMBackend",
    "BaseLLMBackend",
    "ModelInfo",
    "ModelCapability",
    "GenerateResponse",

    # Backend classes (lazy loaded)
    "AnthropicBackend",
    "OpenAIBackend",
    "OpenRouterBackend",
    "OllamaBackend",

    # Model registries
    "MODEL_REGISTRY",
    "ANTHROPIC_MODELS",
    "OPENAI_MODELS",
    "GOOGLE_MODELS",
    "DEEPSEEK_MODELS",
    "OLLAMA_MODELS",

    # Utilities
    "get_model_info",
    "list_models_by_provider",
    "list_recommended_models",
    "convert_tools_to_openai_format",
    "convert_tools_to_anthropic_format",

    # Factory functions
    "create_backend",
    "create_backend_from_env",
    "list_available_providers",
]


# Lazy class references for convenient imports
def __getattr__(name):
    """Lazy load backend classes."""
    if name == "AnthropicBackend":
        return _import_anthropic_backend()
    elif name == "OpenAIBackend":
        return _import_openai_backend()
    elif name == "OpenRouterBackend":
        return _import_openrouter_backend()
    elif name == "OllamaBackend":
        return _import_ollama_backend()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
