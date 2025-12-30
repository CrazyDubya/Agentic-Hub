#!/usr/bin/env python3
"""
Universal LLM Agent Harness - Setup Script

This allows installation via:
    pip install .
    pip install -e .  (for development)

After installation, the 'harness' command will be available.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text()

setup(
    name="llm-agent-harness",
    version="0.1.0",
    description="Universal LLM Agent Harness - A game engine for AI agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Agentic-Hub",
    author_email="",
    url="https://github.com/CrazyDubya/Agentic-Hub",
    license="MIT",

    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),

    python_requires=">=3.8",

    install_requires=[
        # Core dependencies (minimal - stdlib only for base functionality)
    ],

    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.20",
            "pytest-cov>=4.0",
            "black>=23.0",
            "isort>=5.12",
            "mypy>=1.0",
        ],
        "openai": [
            "openai>=1.0",
            "tiktoken>=0.5",  # For token counting
        ],
        "anthropic": [
            "anthropic>=0.40",  # Updated for Claude 4.x support
        ],
        "ollama": [
            "ollama>=0.3",
        ],
        "all": [
            "openai>=1.0",
            "anthropic>=0.40",
            "ollama>=0.3",
            "tiktoken>=0.5",
            "httpx>=0.25",
            "pyyaml>=6.0",
            "rich>=13.0",  # For better terminal output
        ]
    },

    entry_points={
        "console_scripts": [
            "harness=harness.cli:main",
            "llm-harness=harness.cli:main",
        ],
    },

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
    ],

    keywords=[
        "llm",
        "agent",
        "ai",
        "harness",
        "autonomous",
        "multi-agent",
        "tool-use",
        "framework"
    ],

    project_urls={
        "Bug Reports": "https://github.com/CrazyDubya/Agentic-Hub/issues",
        "Source": "https://github.com/CrazyDubya/Agentic-Hub",
    },
)
