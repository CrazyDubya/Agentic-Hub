#!/usr/bin/env python3
"""
Basic Usage Examples for Universal LLM Agent Harness

This file demonstrates how to use the harness with different types of LLMs.
"""

import sys
sys.path.insert(0, '/home/user/Agentic-Hub')

from harness import (
    UniversalLLMHarness,
    HarnessMode,
    LLMBackend,
    SkillRegistry,
    PythonSkill,
    SkillMetadata,
    SkillParameter,
    SkillOutput,
)


# Example 1: Simple Interactive Usage
def example_interactive():
    """Basic interactive agent usage"""
    print("=" * 60)
    print("Example 1: Interactive Mode")
    print("=" * 60)

    harness = UniversalLLMHarness(
        harness_mode=HarnessMode.INTERACTIVE,
        base_path="/tmp/harness-example"
    )

    # Create an agent
    session_id, state = harness.create_agent(
        name="TestAgent",
        capabilities=["general", "code"]
    )

    print(f"Created agent: {state.agent_id}")
    print(f"Session: {session_id}")

    # The agent is now ready to receive tasks
    # In a real scenario, you'd integrate an LLM backend

    print(f"Status: {harness.get_status()}")
    print()


# Example 2: Custom LLM Backend
class EchoLLMBackend(LLMBackend):
    """Simple echo backend for testing"""

    def generate(self, messages, tools=None, max_tokens=4096):
        # Extract last user message
        user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
                break

        # Echo with analysis
        return {
            "content": f"I received your message: '{user_msg[:100]}...'\n\nI will now analyze this and provide assistance.",
            "tool_calls": []
        }

    def count_tokens(self, text):
        return len(text) // 4

    @property
    def supports_tools(self):
        return False

    @property
    def max_context_tokens(self):
        return 4096


def example_custom_backend():
    """Using a custom LLM backend"""
    print("=" * 60)
    print("Example 2: Custom LLM Backend")
    print("=" * 60)

    harness = UniversalLLMHarness(
        llm_backend=EchoLLMBackend(),
        harness_mode=HarnessMode.INTERACTIVE,
        base_path="/tmp/harness-custom"
    )

    session_id, state = harness.create_agent(name="EchoAgent")

    result = harness.execute_turn(
        state.agent_id,
        "Help me analyze the codebase structure"
    )

    print(f"Agent response: {result['final_response']}")
    print()


# Example 3: Skill Registration and Usage
def example_skills():
    """Working with the skill system"""
    print("=" * 60)
    print("Example 3: Skill System")
    print("=" * 60)

    harness = UniversalLLMHarness(base_path="/tmp/harness-skills")

    # Register a custom skill
    def word_count_skill(params):
        text = params.get("text", "")
        words = len(text.split())
        chars = len(text)
        lines = len(text.splitlines())
        return {
            "words": words,
            "characters": chars,
            "lines": lines
        }

    skill = PythonSkill(
        SkillMetadata(
            skill_id="text.wordcount",
            name="Word Count",
            version="1.0.0",
            description="Count words, characters, and lines in text",
            author="example",
            skill_type="local",
            category="utility",
            parameters=[
                SkillParameter("text", "string", "Text to analyze")
            ],
            output=SkillOutput("object", "Word count statistics")
        ),
        word_count_skill
    )

    harness.skill_registry.register(skill)

    # Execute the skill
    result = harness.skill_registry.execute(
        "text.wordcount",
        {"text": "Hello world! This is a test.\nSecond line here."}
    )

    print(f"Skill result: {result.output}")

    # List all skills
    skills = harness.skill_registry.list_skills()
    print(f"\nAvailable skills ({len(skills)}):")
    for s in skills[:5]:
        print(f"  - {s.skill_id}: {s.description}")
    print()


# Example 4: Sandbox Management
def example_sandbox():
    """Working with sandboxes"""
    print("=" * 60)
    print("Example 4: Sandbox Management")
    print("=" * 60)

    harness = UniversalLLMHarness(base_path="/tmp/harness-sandbox")

    # Create agent (also creates default sandbox)
    _, state = harness.create_agent(name="SandboxAgent")
    agent_id = state.agent_id

    # Get agent's sandboxes
    sandboxes = harness.sandbox_manager.get_agent_sandboxes(agent_id)
    print(f"Agent sandboxes: {[s.sandbox_id for s in sandboxes]}")

    # Create additional sandbox with template
    new_sandbox = harness.sandbox_manager.create_sandbox(
        owner_id=agent_id,
        name="python-project",
        template="python"
    )

    print(f"Created sandbox: {new_sandbox.sandbox_id}")
    print(f"Sandbox path: {new_sandbox.root_path}")

    # Execute command in sandbox
    result = harness.sandbox_manager.execute_in_sandbox(
        new_sandbox.sandbox_id,
        agent_id,
        "ls -la workspace/"
    )

    print(f"Sandbox contents:\n{result['stdout']}")

    # Create snapshot
    snapshot = harness.sandbox_manager.create_snapshot(
        new_sandbox.sandbox_id,
        agent_id,
        "Initial state"
    )

    print(f"Created snapshot: {snapshot.snapshot_id}")
    print()


# Example 5: Multi-Agent Communication
def example_communication():
    """Agent-to-agent communication"""
    print("=" * 60)
    print("Example 5: Multi-Agent Communication")
    print("=" * 60)

    harness = UniversalLLMHarness(base_path="/tmp/harness-comm")

    # Create two agents
    _, agent1_state = harness.create_agent(name="Agent1")
    _, agent2_state = harness.create_agent(name="Agent2")

    agent1_id = agent1_state.agent_id
    agent2_id = agent2_state.agent_id

    print(f"Created agents: {agent1_id}, {agent2_id}")

    # Subscribe to a topic
    mailbox1 = harness.message_bus.get_mailbox(agent1_id)
    mailbox2 = harness.message_bus.get_mailbox(agent2_id)

    mailbox1.subscribe("project-updates")
    mailbox2.subscribe("project-updates")

    # Agent1 publishes to topic
    from harness.communication import Message, MessageType

    message = Message(
        message_id="msg-001",
        message_type=MessageType.PUBLISH,
        sender_id=agent1_id,
        recipient_id=None,
        content={"event": "task-completed", "task": "code-review"},
        topic="project-updates"
    )

    harness.message_bus.send(message)

    # Check Agent2's mailbox
    pending = mailbox2.peek_messages(5)
    print(f"Agent2 received {len(pending)} message(s)")
    for msg in pending:
        print(f"  From: {msg.sender_id}, Topic: {msg.topic}")
        print(f"  Content: {msg.content}")

    print()


# Example 6: Command Protocol Demonstration
def example_commands():
    """Demonstrating different command formats"""
    print("=" * 60)
    print("Example 6: Universal Command Protocol")
    print("=" * 60)

    from harness.core.command_protocol import UniversalCommandParser

    parser = UniversalCommandParser()

    # Format 1: Tool call (JSON)
    tool_call = {
        "tool": "file_read",
        "arguments": {"path": "/etc/hostname"}
    }

    commands = parser.parse(tool_call)
    print(f"Tool call parsed: {commands[0].type.value}")

    # Format 2: Text block
    text_block = '''```command:shell.exec
command: echo "Hello World"
timeout: 10
```'''

    commands = parser.parse(text_block)
    print(f"Text block parsed: {commands[0].type.value}, params: {commands[0].params}")

    # Format 3: Natural language
    natural = "Read the file at /tmp/test.txt"

    commands = parser.parse(natural)
    if commands:
        print(f"Natural language parsed: {commands[0].type.value}, params: {commands[0].params}")

    # Show text block format for output
    print("\nCommand as text block (for non-tool LLMs):")
    print(commands[0].to_text_block() if commands else "No command parsed")
    print()


# Example 7: Complete Workflow
def example_complete_workflow():
    """A complete agent workflow demonstration"""
    print("=" * 60)
    print("Example 7: Complete Workflow")
    print("=" * 60)

    harness = UniversalLLMHarness(
        harness_mode=HarnessMode.AUTONOMOUS,
        base_path="/tmp/harness-workflow"
    )

    # Create agent with specific profile
    session_id, state = harness.create_agent(
        name="WorkflowAgent",
        capabilities=["code-analysis", "testing", "documentation"],
        specializations=["python", "testing"]
    )

    agent_id = state.agent_id
    print(f"Agent: {agent_id}")

    # Update todo list (Manus-style attention management)
    todos = [
        {"content": "Analyze codebase structure", "status": "pending"},
        {"content": "Identify code patterns", "status": "pending"},
        {"content": "Generate documentation", "status": "pending"}
    ]
    harness.update_todos(agent_id, todos)
    print(f"Set {len(todos)} todos")

    # Build context (what the LLM would see)
    context = harness.build_context(
        agent_id,
        "Analyze this Python project and generate documentation"
    )

    print(f"\nExecution Context:")
    print(f"  Mode: {context.harness_mode.value}")
    print(f"  Sandbox: {context.sandbox.sandbox_id if context.sandbox else 'None'}")
    print(f"  Available skills: {len(context.available_skills)}")
    print(f"  Other agents: {len(context.available_agents)}")

    # Show system prompt structure
    system_prompt = harness.build_system_prompt(context)
    print(f"\nSystem prompt length: {len(system_prompt)} chars")
    print("System prompt preview (first 500 chars):")
    print(system_prompt[:500])
    print("...")

    # Show harness status
    print(f"\nHarness status: {harness.get_status()}")
    print()


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("UNIVERSAL LLM AGENT HARNESS - EXAMPLES")
    print("=" * 60 + "\n")

    example_interactive()
    example_custom_backend()
    example_skills()
    example_sandbox()
    example_communication()
    example_commands()
    example_complete_workflow()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
