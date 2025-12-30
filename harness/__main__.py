#!/usr/bin/env python3
"""
Universal LLM Agent Harness - Module Entry Point

This allows running the harness as a module:
    python -m harness
    python -m harness interactive
    python -m harness run "task"
"""

from .cli import main

if __name__ == "__main__":
    main()
