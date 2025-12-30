# Agentic-Hub Architecture

This document describes the two-layer harness architecture in Agentic-Hub.

## Overview

Agentic-Hub has two distinct orchestration systems that operate at different abstraction layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Orchestrator                        │
│              (workflows/pipeline_orchestrator.py)               │
│                                                                 │
│   Chains execution engines: n8n → LangChain → Python → ...     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ can invoke
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Universal LLM Harness                        │
│                        (harness/)                               │
│                                                                 │
│   Makes ANY LLM function as an agent with tools & memory        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ connects to
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       LLM Backends                              │
│                    (harness/adapters/)                          │
│                                                                 │
│   Anthropic │ OpenAI │ OpenRouter │ Ollama                      │
└─────────────────────────────────────────────────────────────────┘
```

## Layer 1: Universal LLM Harness

**Location:** `harness/`

**Purpose:** Transform any LLM into a capable agent with unified command interface, sandboxed execution, skills, and multi-agent communication.

### Core Components

| Component | Location | Responsibility |
|-----------|----------|----------------|
| **LLM Harness** | `core/llm_harness.py` | Main orchestrator - manages agent lifecycle, context, turns |
| **Command Protocol** | `core/command_protocol.py` | Parses tool calls, text blocks, and natural language into commands |
| **Sandbox Manager** | `sandbox/sandbox_manager.py` | Isolated/shared execution environments with snapshots |
| **Skill System** | `skills/skill_system.py` | Extensible Python/Prompt/Composite skills |
| **Message Bus** | `communication/message_bus.py` | Inter-agent pub/sub and direct messaging |
| **Marketplace** | `marketplace/registry.py` | Agent directory and skill marketplace |
| **Self-Evaluation** | `evaluation/self_eval.py` | Flywheel for continuous improvement |
| **Q&A Generation** | `evaluation/qa_generation.py` | Training data generation from interactions |
| **CLI** | `cli.py` | Interactive, run, and server modes |

### LLM Backends

| Backend | Location | Models |
|---------|----------|--------|
| **Anthropic** | `adapters/anthropic_backend.py` | Claude 4.x (Opus, Sonnet, Haiku) |
| **OpenAI** | `adapters/openai_backend.py` | GPT-5, GPT-4o, O1 |
| **OpenRouter** | `adapters/openrouter_backend.py` | 400+ models (DeepSeek, Gemini, Llama, etc.) |
| **Ollama** | `adapters/ollama_backend.py` | Local models (Llama, Mistral, Qwen) |

### Key Design Principles

1. **Universal Command Protocol** - Works with both tool-using and non-tool-using LLMs
2. **Todo-List Attention** - Manus-inspired technique to maintain focus across turns
3. **Context Compaction** - Codex-inspired summarization for infinite context
4. **Layered Prompts** - Copilot-inspired prompt architecture
5. **Shared Sandboxes** - Multi-agent collaboration in same workspace

### Harness Modes

| Mode | Description |
|------|-------------|
| `INTERACTIVE` | Single-turn interactions with human |
| `AUTONOMOUS` | Multi-step execution until task completion |
| `COLLABORATIVE` | Multi-agent collaboration |
| `SUPERVISED` | Human-in-the-loop approval |

## Layer 2: Pipeline Orchestrator

**Location:** `workflows/pipeline_orchestrator.py`

**Purpose:** Chain multiple execution engines together for complex workflows.

### Components

| Component | Class | Purpose |
|-----------|-------|---------|
| **N8N Connector** | `N8NConnector` | Execute n8n automation workflows |
| **LangChain Orchestrator** | `LangChainOrchestrator` | Run LangChain chains |
| **Python Engine** | `PythonScriptEngine` | Execute Python scripts safely |
| **Text Config Parser** | `TextConfigParser` | Parse simple text configurations |
| **Workflow Harness** | `UniversalWorkflowHarness` | Main orchestrator for hybrid workflows |

### Workflow Types

```python
class WorkflowType(Enum):
    N8N = "n8n"           # n8n automation platform
    LANGCHAIN = "langchain"  # LangChain workflows
    PYTHON = "python"      # Custom Python scripts
    TEXT_CONFIG = "text"   # Simple text configurations
    HYBRID = "hybrid"      # Multi-step mixed workflows
```

### Example Hybrid Workflow

```yaml
workflow_type: hybrid
steps:
  - type: python
    script_file: setup.py
    required: true
  - type: langchain
    chain_id: analysis
  - type: n8n
    workflow_id: "abc123"
```

## How They Relate

The Pipeline Orchestrator can use the LLM Harness as one of its execution engines:

```
Pipeline Orchestrator
    │
    ├── Step 1: Python script (data prep)
    │
    ├── Step 2: LLM Harness (agent analysis)  ◄── Integration point
    │
    └── Step 3: n8n workflow (notifications)
```

However, they can also operate independently:
- **LLM Harness alone**: Interactive agent sessions, autonomous task completion
- **Pipeline Orchestrator alone**: Data processing pipelines, automation workflows

## Test Architecture

| Test File | Purpose | Dependencies |
|-----------|---------|--------------|
| `test_harness_integration.py` | Full harness + real LLM backends | OpenRouter API |
| `test_integration.py` | Pytest-style unit/integration tests | pytest |
| `run_tests.py` | Fallback non-pytest runner | None |
| `test_stress_smol_models.py` | Performance with small/cheap models | OpenRouter API |
| `test_advanced_llm_capabilities.py` | Advanced LLM feature tests | OpenRouter API |

## Directory Structure

```
Agentic-Hub/
├── harness/                    # Layer 1: LLM Agent Harness
│   ├── core/
│   │   ├── llm_harness.py     # Main orchestrator
│   │   └── command_protocol.py # Command parsing
│   ├── adapters/              # LLM backends
│   ├── sandbox/               # Execution isolation
│   ├── skills/                # Capability system
│   ├── communication/         # Message bus
│   ├── marketplace/           # Agent/skill registry
│   ├── evaluation/            # Self-eval + Q&A gen
│   ├── examples/              # Usage examples
│   └── cli.py                 # CLI interface
│
├── workflows/                  # Layer 2: Pipeline Orchestrator
│   └── pipeline_orchestrator.py
│
├── agents/                     # Framework adapters (stubs)
│   └── adapters/              # GPTSwarm, Swarms, etc.
│
├── tests/                      # Test suites
│
└── bin/
    └── harness                # CLI entry point
```

## Design Inspirations

| Feature | Inspired By |
|---------|-------------|
| Todo-list attention | Manus |
| Context compaction | OpenAI Codex |
| Layered prompts | GitHub Copilot |
| Skills & dynamic loading | Claude Code |
| Shared sandboxes | Original design |
| Universal command protocol | Original design |

## Future Integration

The architecture supports future enhancements:

1. **Pipeline → Harness Integration**: Add `LLMHarnessEngine` to Pipeline Orchestrator
2. **Agent Framework Adapters**: Complete the stub implementations in `agents/adapters/`
3. **Distributed Execution**: Message bus already supports multi-agent; extend to multi-node
4. **Training Loop**: Connect Q&A generation → fine-tuning → improved agents
