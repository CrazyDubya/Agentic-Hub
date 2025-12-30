# Universal LLM Agent Harness

A comprehensive framework that enables **ANY LLM** to function as a capable agent, regardless of whether it supports native tool/function calling.

## Design Philosophy

This harness is designed to match or exceed the capabilities of:
- **Claude Code** - Anthropic's CLI agent
- **Manus** - Butterfly Effect's autonomous agent
- **OpenAI Codex** - OpenAI's coding agent
- **GitHub Copilot Agent Mode** - GitHub's AI pair programmer

### Key Innovation: Universal Command Protocol

The harness works with both **tool-using** and **non-tool-using** LLMs through a unified command protocol:

1. **Tool Calls** (for LLMs with function calling): Native JSON-based tool invocations
2. **Text Blocks** (for any LLM): Structured text blocks parsed by the harness
3. **Natural Language** (fallback): Intent recognition and parameter extraction

```
┌─────────────────────────────────────────────────────────────┐
│                        LLM Output                           │
├─────────────────────────────────────────────────────────────┤
│  Tool Call JSON  │  Text Block     │  Natural Language      │
│  (GPT-4, Claude) │  (Any LLM)      │  (Fallback)            │
└────────┬─────────┴───────┬─────────┴──────────┬─────────────┘
         │                 │                    │
         └─────────────────┼────────────────────┘
                           ▼
              ┌────────────────────────┐
              │ Universal Command Parser│
              └───────────┬────────────┘
                          ▼
              ┌────────────────────────┐
              │   Command Executor     │
              └────────────────────────┘
```

## Architecture Overview

```
harness/
├── core/
│   ├── command_protocol.py  # Universal command interface
│   └── llm_harness.py       # Main orchestration layer
├── sandbox/
│   └── sandbox_manager.py   # Isolated execution environments
├── skills/
│   └── skill_system.py      # Extensible capability system
├── marketplace/
│   └── registry.py          # Agent directory & asset marketplace
├── communication/
│   └── message_bus.py       # Inter-agent messaging
└── __init__.py
```

## Quick Start

```python
from harness import UniversalLLMHarness, HarnessMode

# Create harness instance
harness = UniversalLLMHarness(
    harness_mode=HarnessMode.INTERACTIVE,
    base_path="/tmp/my-harness"
)

# Create an agent
session_id, state = harness.create_agent(
    name="ResearchAgent",
    capabilities=["research", "code-analysis"],
    specializations=["python", "data-science"]
)

# Execute a turn
result = harness.execute_turn(
    state.agent_id,
    "Analyze the codebase structure and identify key patterns"
)

print(result["final_response"])
```

## Components

### 1. Command Protocol (`core/command_protocol.py`)

Unified command interface that works across all LLM types.

**For Tool-Using LLMs:**
```json
{
  "tool": "file_read",
  "arguments": {"path": "/path/to/file.py"}
}
```

**For Non-Tool-Using LLMs (Text Blocks):**
```
```command:file.read
path: /path/to/file.py
encoding: utf-8
```
```

**Natural Language (Auto-parsed):**
```
Read the file at /path/to/file.py
```

### 2. Sandbox Manager (`sandbox/sandbox_manager.py`)

Isolated execution environments for agents.

```python
# Create private sandbox
sandbox = sandbox_manager.create_sandbox(
    owner_id="agent-1",
    name="my-workspace",
    template="python"
)

# Share with another agent
sandbox_manager.share_sandbox(
    sandbox.sandbox_id,
    owner_id="agent-1",
    target_agent_id="agent-2",
    readonly=False
)

# Create snapshot before risky changes
snapshot = sandbox_manager.create_snapshot(
    sandbox.sandbox_id,
    agent_id="agent-1",
    description="Before major refactor"
)

# Restore if needed
sandbox_manager.restore_snapshot(
    sandbox.sandbox_id,
    agent_id="agent-1",
    snapshot_id=snapshot.snapshot_id
)
```

**Sandbox Types:**
- `PRIVATE` - Single agent, isolated
- `SHARED` - Multiple agents collaborate
- `READONLY` - Read-only access
- `EPHEMERAL` - Temporary, auto-cleanup

### 3. Skill System (`skills/skill_system.py`)

Extensible capability system with multiple skill types.

**Python Skills:**
```python
from harness.skills import PythonSkill, SkillMetadata

def my_skill_func(params):
    return {"result": params["input"].upper()}

skill = PythonSkill(
    SkillMetadata(
        skill_id="text.uppercase",
        name="Uppercase Text",
        description="Convert text to uppercase",
        parameters=[
            SkillParameter("input", "string", "Text to convert")
        ],
        output=SkillOutput("string", "Uppercased text")
    ),
    my_skill_func
)

skill_registry.register(skill)
```

**Prompt Skills (YAML):**
```yaml
skill_id: code.review
name: Code Review
description: Review code for issues and improvements
category: code_analysis
parameters:
  - name: code
    type: string
    description: Code to review
  - name: focus
    type: string
    enum: [bugs, security, performance, style]
prompt: |
  Review the following code with focus on {{focus}}:

  ```
  {{code}}
  ```

  Provide detailed feedback on issues and improvements.
```

**Composite Skills:**
```yaml
skill_id: workflow.full-review
name: Full Code Review Workflow
steps:
  - skill_id: file.read
    params: {path: "$file_path"}
    output_key: code
  - skill_id: code.review
    params: {code: "$code", focus: "all"}
    output_key: review
  - skill_id: file.write
    params: {path: "$output_path", content: "$review"}
```

### 4. Agent Directory & Marketplace (`marketplace/registry.py`)

Discovery and distribution of agents, skills, and assets.

**Agent Discovery:**
```python
# Register agent capabilities
directory.register(AgentProfile(
    agent_id="code-expert",
    name="Code Expert",
    capabilities=["code-review", "refactoring", "testing"],
    specializations=["python", "typescript"]
))

# Find agents for a task
matches = directory.match_task(
    required_capabilities=["code-review"],
    preferred_specializations=["python"],
    require_tools=True
)
```

**Marketplace:**
```python
# Search for skills
results = marketplace.search("code analysis", asset_type=AssetType.SKILL)

# Install a skill
success, msg = marketplace.install("skill.code-review")

# Publish your skill
marketplace.publish(my_asset, content_path="/path/to/skill")
```

### 5. Message Bus (`communication/message_bus.py`)

Inter-agent communication with multiple patterns.

**Direct Messaging:**
```python
message_bus.send(Message(
    message_type=MessageType.DIRECT,
    sender_id="agent-1",
    recipient_id="agent-2",
    content="Please review the changes in /src/main.py"
))
```

**Topic-Based Pub/Sub:**
```python
# Subscribe to a topic
mailbox.subscribe("code-changes")

# Publish to topic
message_bus.send(Message(
    message_type=MessageType.PUBLISH,
    sender_id="agent-1",
    topic="code-changes",
    content={"file": "/src/main.py", "action": "modified"}
))
```

**Request/Response:**
```python
response = message_bus.request(
    sender_id="agent-1",
    recipient_id="agent-2",
    content={"query": "What is the status of task X?"},
    timeout=30
)
```

**Text Format for Non-Tool LLMs:**
```
@agent-2 Please review the changes
#code-changes Modified /src/main.py
@broadcast Starting deployment process
```

## Command Reference

### File Operations
| Command | Description |
|---------|-------------|
| `file.read` | Read file contents |
| `file.write` | Write to file |
| `file.edit` | Edit file (find/replace) |
| `file.list` | List files (glob pattern) |
| `file.search` | Search file contents (grep) |

### Shell Operations
| Command | Description |
|---------|-------------|
| `shell.exec` | Execute shell command |
| `shell.background` | Run in background |
| `shell.kill` | Kill background process |

### Messaging
| Command | Description |
|---------|-------------|
| `msg.send` | Send direct message |
| `msg.broadcast` | Broadcast to all agents |
| `msg.subscribe` | Subscribe to topic |

### Sandbox
| Command | Description |
|---------|-------------|
| `sandbox.create` | Create new sandbox |
| `sandbox.destroy` | Destroy sandbox |
| `sandbox.share` | Share with agent |
| `sandbox.snapshot` | Create snapshot |
| `sandbox.restore` | Restore from snapshot |

### Skills
| Command | Description |
|---------|-------------|
| `skill.invoke` | Execute a skill |
| `skill.install` | Install from marketplace |
| `skill.list` | List available skills |

### Agents
| Command | Description |
|---------|-------------|
| `agent.spawn` | Create new agent |
| `agent.query` | Find agents by capability |
| `agent.terminate` | Stop an agent |

### Marketplace
| Command | Description |
|---------|-------------|
| `market.search` | Search assets |
| `market.install` | Install asset |
| `market.publish` | Publish asset |

## Design Patterns (Inspired by Industry Leaders)

### Manus: Context Engineering
- **Todo List Attention**: Constantly rewriting the todo list to the end of context keeps objectives in the model's attention window
- **Event Stream**: All actions and observations form an event stream that the agent processes

### OpenAI Codex: Compaction
- **Context Compaction**: When approaching context limits, summarize and prepare state for fresh context window
- **Model-Harness Co-training**: The harness is designed to work optimally with the model

### GitHub Copilot: Layered Prompts
- **Layer 1**: Core identity and rules
- **Layer 2**: Available capabilities and tools
- **Layer 3**: Current state and context
- **Layer 4**: Pending messages and events
- **Layer 5**: Other available agents

### Claude Code: Skills & MCP
- **Declarative Skills**: Skills defined in YAML/Markdown
- **Dynamic Loading**: Skills loaded on-demand
- **Extensibility**: Easy to add new capabilities

## Harness Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `INTERACTIVE` | Single-turn exchanges | Chat interface |
| `AUTONOMOUS` | Multi-step execution | Background tasks |
| `COLLABORATIVE` | Multi-agent coordination | Team projects |
| `SUPERVISED` | Human-in-the-loop | Sensitive operations |

## Extending the Harness

### Custom LLM Backend

```python
from harness import LLMBackend

class MyLLMBackend(LLMBackend):
    def generate(self, messages, tools=None, max_tokens=4096):
        # Call your LLM API
        response = my_api.complete(messages)
        return {
            "content": response.text,
            "tool_calls": response.tool_calls or []
        }

    def count_tokens(self, text):
        return len(text) // 4  # Rough estimate

    @property
    def supports_tools(self):
        return True

    @property
    def max_context_tokens(self):
        return 128000

harness = UniversalLLMHarness(llm_backend=MyLLMBackend())
```

### Custom Command Handler

```python
from harness.core.command_protocol import CommandType

# Add to CommandExecutor
def _handle_custom_command(self, params, agent_id, sandbox_id):
    # Your custom logic
    return {"result": "success"}

executor.handlers[CommandType.CUSTOM] = _handle_custom_command
```

### Custom Skill

```python
from harness.skills import PythonSkill, SkillMetadata, SkillParameter

skill = PythonSkill(
    SkillMetadata(
        skill_id="custom.my-skill",
        name="My Custom Skill",
        description="Does something useful",
        parameters=[
            SkillParameter("input", "string", "Input data")
        ],
        output={"type": "object", "description": "Result"}
    ),
    lambda params: {"processed": params["input"]}
)

harness.skill_registry.register(skill)
```

## Comparison with Other Harnesses

| Feature | This Harness | Claude Code | Manus | Codex | Copilot |
|---------|--------------|-------------|-------|-------|---------|
| Tool-using LLMs | ✅ | ✅ | ✅ | ✅ | ✅ |
| Non-tool LLMs | ✅ | ❌ | ❌ | ❌ | ❌ |
| Sandboxed execution | ✅ | ✅ | ✅ | ✅ | ✅ |
| Multi-agent | ✅ | ✅ | ✅ | ❌ | ✅ |
| Shared sandboxes | ✅ | ❌ | ❌ | ❌ | ❌ |
| Skill marketplace | ✅ | ❌ | ❌ | ❌ | ✅ |
| Context compaction | ✅ | ❌ | ✅ | ✅ | ❌ |
| Todo-list attention | ✅ | ✅ | ✅ | ❌ | ❌ |
| Open source | ✅ | ✅ | ❌ | ✅ | ❌ |

## License

MIT License - See LICENSE file for details.
