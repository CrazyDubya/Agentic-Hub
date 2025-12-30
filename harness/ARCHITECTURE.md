# Universal LLM Agent Harness Architecture

## Vision

A minimal yet complete harness enabling **any reasonably intelligent LLM** to operate with capabilities matching or exceeding current systems (Claude Code, Manus, Codex, Copilot, etc.), regardless of whether they support native tool use.

## Core Design Principles

1. **Dual-Mode Operation**: Support both tool-using and text-command LLMs
2. **Isolation First**: Personal sandboxes with explicit sharing
3. **Self-Improvement**: Built-in evaluation loops and flywheel mechanics
4. **Discoverability**: Skills, agents, and resources through marketplace
5. **Composability**: Everything is a skill that can be combined

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     UNIVERSAL LLM AGENT HARNESS                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    COMMAND LAYER                             │    │
│  │  ┌──────────────────┐  ┌──────────────────────────────────┐ │    │
│  │  │  Text Commands   │  │      Tool/Function Calls         │ │    │
│  │  │  /cmd arg1 arg2  │  │  {"tool": "x", "params": {...}}  │ │    │
│  │  └────────┬─────────┘  └───────────────┬──────────────────┘ │    │
│  │           └──────────────┬─────────────┘                    │    │
│  │                          ▼                                   │    │
│  │           ┌──────────────────────────┐                      │    │
│  │           │   UNIFIED COMMAND BUS    │                      │    │
│  │           └──────────────────────────┘                      │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│  ┌───────────────────────────┼───────────────────────────────────┐  │
│  │                           ▼                                    │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │  │
│  │  │ SANDBOX │  │  SKILL  │  │  AGENT  │  │  EVAL   │          │  │
│  │  │ MANAGER │  │ REGISTRY│  │  COMM   │  │  LOOPS  │          │  │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘          │  │
│  │       │            │            │            │                │  │
│  │       ▼            ▼            ▼            ▼                │  │
│  │  ┌─────────────────────────────────────────────────────────┐ │  │
│  │  │              SHARED EXECUTION CONTEXT                    │ │  │
│  │  └─────────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                       STORAGE LAYER                            │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐  │  │
│  │  │  Personal  │  │   Shared   │  │      Marketplace       │  │  │
│  │  │  Sandbox   │  │  Sandboxes │  │     (Skills/Agents)    │  │  │
│  │  └────────────┘  └────────────┘  └────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### 1. Command Layer

#### Text Commands (for non-tool LLMs)
```
COMMAND FORMAT: /command [subcommand] [--flag value] [args...]

CORE COMMANDS:
  /help [topic]                    - Show help
  /cmd <shell_command>             - Execute shell command
  /read <path>                     - Read file/directory
  /write <path> <content>          - Write to file
  /edit <path> <old> <new>         - Edit file
  /search <pattern> [path]         - Search files
  /skill <name> [args]             - Invoke skill
  /agent <name> <message>          - Message agent
  /sandbox create|list|switch|share - Manage sandboxes
  /eval start|status|report        - Self-evaluation
  /qa generate|answer|review       - Q&A system
  /market search|install|publish   - Marketplace
```

#### Tool Calls (for tool-using LLMs)
```json
{
  "tool": "harness_execute",
  "parameters": {
    "action": "shell|read|write|skill|agent|sandbox|eval|qa|market",
    "target": "...",
    "options": {}
  }
}
```

### 2. Sandbox System

#### Personal Sandbox
- Isolated filesystem (chroot/container)
- Private environment variables
- Personal history and state
- Configurable resource limits

#### Shared Sandboxes
- Explicit creation and membership
- Access control (owner, members, public)
- Shared filesystem regions
- Shared state and artifacts

```python
@dataclass
class Sandbox:
    id: str
    owner: str
    sandbox_type: Literal["personal", "shared"]
    members: List[str]
    permissions: Dict[str, List[str]]  # agent_id -> [read, write, execute]
    root_path: Path
    env_vars: Dict[str, str]
    resource_limits: ResourceLimits
    created_at: datetime
    state: Dict[str, Any]
```

### 3. Skills Registry

Skills are the fundamental unit of capability:

```python
@dataclass
class Skill:
    id: str
    name: str
    description: str
    version: str
    author: str

    # Invocation
    invoke_type: Literal["command", "tool", "hybrid"]
    command_template: Optional[str]      # For text invocation
    tool_schema: Optional[Dict]          # For tool invocation

    # Execution
    handler: Callable | str              # Function or script path
    dependencies: List[str]              # Other skills needed
    required_permissions: List[str]      # sandbox, network, etc.

    # Metadata
    category: str
    tags: List[str]
    examples: List[Dict]
    documentation: str
```

### 4. Agent Communication Bus

```python
@dataclass
class Message:
    id: str
    from_agent: str
    to_agent: str | List[str] | Literal["broadcast"]
    message_type: Literal["request", "response", "event", "query"]
    content: Any
    priority: int
    reply_to: Optional[str]
    timestamp: datetime
    expires_at: Optional[datetime]

class AgentCommBus:
    def send(self, message: Message) -> str
    def receive(self, agent_id: str, filter: Optional[Dict]) -> List[Message]
    def subscribe(self, agent_id: str, pattern: str, callback: Callable)
    def broadcast(self, from_agent: str, content: Any)
    def request_response(self, from_agent: str, to_agent: str, content: Any, timeout: float) -> Message
```

### 5. Self-Evaluation Loops (Flywheel)

```python
@dataclass
class EvaluationCycle:
    id: str
    agent_id: str
    cycle_type: Literal["task", "session", "periodic"]
    started_at: datetime
    completed_at: Optional[datetime]

    # What was attempted
    tasks_attempted: List[TaskRecord]
    skills_used: List[str]
    errors_encountered: List[ErrorRecord]

    # Self-assessment
    self_rating: Optional[int]  # 1-10
    confidence_scores: Dict[str, float]
    identified_gaps: List[str]
    improvement_ideas: List[str]

    # External feedback
    user_rating: Optional[int]
    user_feedback: Optional[str]

    # Actions
    next_steps: List[str]
    skills_to_learn: List[str]
    questions_for_user: List[Question]

class EvaluationLoop:
    def start_cycle(self, trigger: str) -> EvaluationCycle
    def record_task(self, task: TaskRecord)
    def self_evaluate(self) -> Dict
    def generate_improvements(self) -> List[str]
    def complete_cycle(self) -> EvaluationCycle
```

### 6. Q&A Generation System

```python
@dataclass
class Question:
    id: str
    question_text: str
    context: str
    question_type: Literal["clarification", "verification", "guidance", "learning"]

    # Answers
    believed_answer: Optional[str]
    confidence: float  # 0-1
    answer_options: List[str]  # If multiple choice
    selection_mode: Literal["single", "multiple", "all", "none"]

    # Status
    status: Literal["pending", "answered", "confirmed", "rejected"]
    user_answer: Optional[str]
    user_correction: Optional[str]

    # Learning
    learned_from_answer: bool
    incorporated_at: Optional[datetime]

class QASystem:
    def generate_question(self, context: str, q_type: str) -> Question
    def generate_batch(self, topics: List[str]) -> List[Question]
    def submit_answer(self, question_id: str, answer: str)
    def get_pending(self, agent_id: str) -> List[Question]
    def learn_from_feedback(self, question: Question) -> Dict
```

### 7. Marketplace

```python
@dataclass
class MarketplaceEntry:
    id: str
    entry_type: Literal["skill", "agent", "template", "dataset", "model"]
    name: str
    description: str
    version: str
    author: str

    # Content
    content_hash: str
    download_url: str
    source_url: Optional[str]

    # Metadata
    tags: List[str]
    category: str
    license: str

    # Stats
    downloads: int
    ratings: List[int]
    avg_rating: float

    # Compatibility
    harness_version: str
    dependencies: List[str]
    conflicts: List[str]

class Marketplace:
    def search(self, query: str, filters: Dict) -> List[MarketplaceEntry]
    def install(self, entry_id: str, sandbox_id: str) -> bool
    def publish(self, entry: MarketplaceEntry, content: bytes) -> str
    def update(self, entry_id: str, content: bytes) -> bool
    def rate(self, entry_id: str, rating: int, review: str)
```

---

## Command Reference (Complete)

### Shell & Files
```
/cmd <command>                Execute shell command
/read <path> [--lines N]      Read file or directory
/write <path> --content       Write content to file
/edit <path> --old --new      Replace text in file
/search <pattern> [path]      Search with regex/glob
/tree [path] [--depth N]      Show directory tree
```

### Sandbox Management
```
/sandbox list                 List all sandboxes
/sandbox create <name>        Create new sandbox
/sandbox switch <name>        Switch active sandbox
/sandbox share <name> <agents> Share with other agents
/sandbox env get|set|list     Manage environment
/sandbox state get|set        Manage sandbox state
```

### Skills
```
/skill list [--category]      List available skills
/skill info <name>            Show skill details
/skill invoke <name> [args]   Execute a skill
/skill install <name>         Install from marketplace
/skill create <name>          Create new skill
```

### Agent Communication
```
/agent list                   List known agents
/agent msg <agent> <text>     Send message to agent
/agent broadcast <text>       Broadcast to all agents
/agent query <agent> <q>      Query and wait for response
/agent subscribe <pattern>    Subscribe to message pattern
```

### Self-Evaluation
```
/eval start [type]            Start evaluation cycle
/eval record <task>           Record task completion
/eval assess                  Run self-assessment
/eval gaps                    Identify capability gaps
/eval improve                 Generate improvements
/eval report                  Generate full report
/eval status                  Current eval cycle status
```

### Q&A System
```
/qa generate [topic]          Generate questions
/qa pending                   Show pending questions
/qa answer <id> <answer>      Submit answer
/qa review <id>               Review question/answer
/qa learn                     Learn from feedback
/qa export                    Export Q&A history
```

### Marketplace
```
/market search <query>        Search marketplace
/market info <id>             Get entry details
/market install <id>          Install entry
/market publish <path>        Publish new entry
/market rate <id> <1-5>       Rate an entry
/market my                    List my published
```

### Meta
```
/help [command]               Show help
/status                       Show harness status
/config get|set|list          Manage configuration
/history [--limit N]          Show command history
/clear                        Clear screen/context
/version                      Show version info
```

---

## Execution Flow

### For Non-Tool LLM
```
1. LLM receives prompt + command reference
2. LLM outputs text with embedded commands: "Let me check that. /read src/main.py"
3. Harness parses output, extracts commands
4. Commands execute in sandbox
5. Results returned to LLM as next prompt input
6. Loop continues
```

### For Tool-Using LLM
```
1. LLM receives prompt + tool schemas
2. LLM makes tool calls: {"tool": "harness_read", "path": "src/main.py"}
3. Tools execute in sandbox
4. Results returned as tool results
5. LLM continues with results
6. Loop continues
```

### Evaluation Flywheel
```
┌─────────────────────────────────────────────────────────┐
│                    FLYWHEEL CYCLE                        │
│                                                          │
│    ┌──────────┐     ┌──────────┐     ┌──────────┐      │
│    │  TASK    │────▶│  SELF    │────▶│ GENERATE │      │
│    │ COMPLETE │     │  EVAL    │     │    Q&A   │      │
│    └──────────┘     └──────────┘     └────┬─────┘      │
│         ▲                                  │            │
│         │                                  ▼            │
│    ┌────┴─────┐     ┌──────────┐     ┌──────────┐      │
│    │  APPLY   │◀────│  LEARN   │◀────│   USER   │      │
│    │ LEARNING │     │ FROM FB  │     │ FEEDBACK │      │
│    └──────────┘     └──────────┘     └──────────┘      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Security Model

### Permission Levels
- `sandbox.read` - Read files in sandbox
- `sandbox.write` - Write files in sandbox
- `sandbox.execute` - Execute commands
- `network.outbound` - Make network requests
- `agent.communicate` - Message other agents
- `market.install` - Install from marketplace
- `market.publish` - Publish to marketplace
- `eval.self` - Run self-evaluation
- `eval.modify` - Modify evaluation criteria

### Isolation
- Each agent has personal sandbox (default)
- Shared sandboxes require explicit grant
- Resource limits enforced per sandbox
- Network access is opt-in
- Cross-agent communication is logged

---

## File Structure
```
harness/
├── ARCHITECTURE.md          # This document
├── __init__.py              # Package init
├── core/
│   ├── __init__.py
│   ├── harness.py           # Main harness class
│   ├── command_parser.py    # Text command parsing
│   ├── tool_adapter.py      # Tool call handling
│   └── unified_bus.py       # Unified command bus
├── sandbox/
│   ├── __init__.py
│   ├── manager.py           # Sandbox management
│   ├── personal.py          # Personal sandbox
│   ├── shared.py            # Shared sandbox
│   └── permissions.py       # Permission system
├── skills/
│   ├── __init__.py
│   ├── registry.py          # Skill registry
│   ├── loader.py            # Skill loader
│   └── builtin/             # Built-in skills
│       ├── file_ops.py
│       ├── shell.py
│       ├── search.py
│       └── ...
├── communication/
│   ├── __init__.py
│   ├── bus.py               # Message bus
│   ├── protocols.py         # Message types
│   └── router.py            # Message routing
├── evaluation/
│   ├── __init__.py
│   ├── loop.py              # Evaluation loop
│   ├── metrics.py           # Metrics collection
│   ├── flywheel.py          # Flywheel mechanics
│   └── reporter.py          # Report generation
├── qa/
│   ├── __init__.py
│   ├── generator.py         # Question generation
│   ├── validator.py         # Answer validation
│   └── learner.py           # Learning from feedback
├── marketplace/
│   ├── __init__.py
│   ├── client.py            # Marketplace client
│   ├── publisher.py         # Publishing
│   └── installer.py         # Installation
└── cli/
    ├── __init__.py
    ├── main.py              # CLI entry point
    └── repl.py              # Interactive REPL
```
