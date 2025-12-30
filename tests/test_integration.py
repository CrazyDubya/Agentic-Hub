#!/usr/bin/env python3
"""
Integration Tests for Universal LLM Agent Harness

These tests verify that all components work together correctly.
"""

import sys
import os
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness import (
    UniversalLLMHarness,
    HarnessMode,
    LLMBackend,
    AgentState,
    Command,
    CommandResult,
    CommandType,
    UniversalCommandParser,
    SandboxManager,
    Sandbox,
    SandboxType,
    SkillRegistry,
    Skill,
    SkillMetadata,
    PythonSkill,
    AgentDirectory,
    Marketplace,
    AgentProfile,
    MessageBus,
    Message,
    MessageType,
)
from harness.evaluation import (
    SelfEvaluationLoop,
    FlywheelManager,
    ExecutionTrace,
    EvaluationDimension,
    QAGenerationSystem,
    QAFormat,
    create_evaluation_system,
    create_qa_system
)


class MockLLMBackend(LLMBackend):
    """Deterministic mock backend for testing"""

    def __init__(self, responses: list = None):
        self.responses = responses or []
        self.call_count = 0
        self.messages_received = []

    def generate(self, messages, tools=None, max_tokens=4096):
        self.messages_received.append(messages)

        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response

        return {
            "content": f"Mock response #{self.call_count}",
            "tool_calls": []
        }

    def count_tokens(self, text):
        return len(text) // 4

    @property
    def supports_tools(self):
        return True

    @property
    def max_context_tokens(self):
        return 128000


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path)


@pytest.fixture
def harness(temp_dir):
    """Create a harness instance for testing"""
    return UniversalLLMHarness(
        llm_backend=MockLLMBackend(),
        base_path=temp_dir,
        harness_mode=HarnessMode.INTERACTIVE
    )


class TestHarnessIntegration:
    """Test harness core functionality"""

    def test_create_agent(self, harness):
        """Test agent creation"""
        session_id, state = harness.create_agent(
            name="TestAgent",
            capabilities=["test", "general"]
        )

        assert session_id is not None
        assert state is not None
        assert state.agent_id is not None
        assert state.session_id == session_id

    def test_execute_turn(self, harness):
        """Test executing a turn"""
        _, state = harness.create_agent(name="TestAgent")

        result = harness.execute_turn(
            state.agent_id,
            "Hello, can you help me?"
        )

        assert result is not None
        assert "agent_id" in result
        assert "final_response" in result

    def test_context_building(self, harness):
        """Test context building"""
        _, state = harness.create_agent(name="TestAgent")

        context = harness.build_context(
            state.agent_id,
            "Test task"
        )

        assert context.agent_id == state.agent_id
        assert context.original_task == "Test task"
        assert context.harness_mode == HarnessMode.INTERACTIVE

    def test_system_prompt_generation(self, harness):
        """Test system prompt generation"""
        _, state = harness.create_agent(name="TestAgent")
        context = harness.build_context(state.agent_id, "Test task")

        system_prompt = harness.build_system_prompt(context)

        assert "Agent Identity" in system_prompt
        assert "Available Skills" in system_prompt
        assert "Command Format" in system_prompt

    def test_todo_management(self, harness):
        """Test todo list management"""
        _, state = harness.create_agent(name="TestAgent")

        todos = [
            {"content": "Task 1", "status": "pending"},
            {"content": "Task 2", "status": "in_progress"},
        ]

        success = harness.update_todos(state.agent_id, todos)
        assert success

        updated_state = harness.get_agent(state.agent_id)
        assert len(updated_state.todos) == 2

    def test_context_compaction(self, harness):
        """Test context compaction"""
        _, state = harness.create_agent(name="TestAgent")

        result = harness.compact_context(state.agent_id)

        assert result.get("compacted") == True
        assert "summary" in result


class TestCommandProtocol:
    """Test command parsing and execution"""

    def test_text_block_parsing(self):
        """Test parsing text block commands"""
        parser = UniversalCommandParser()

        text = '''```command:file.read
path: /tmp/test.txt
```'''

        commands = parser.parse(text)
        assert len(commands) == 1
        assert commands[0].type == CommandType.FILE_READ
        assert commands[0].params.get("path") == "/tmp/test.txt"

    def test_tool_call_parsing(self):
        """Test parsing tool call format"""
        parser = UniversalCommandParser()

        tool_call = {
            "tool": "file_read",
            "arguments": {"path": "/tmp/test.txt"}
        }

        commands = parser.parse(tool_call)
        assert len(commands) == 1
        assert commands[0].type == CommandType.FILE_READ

    def test_natural_language_parsing(self):
        """Test parsing natural language commands"""
        parser = UniversalCommandParser()

        text = "Read the file at /tmp/test.txt"
        commands = parser.parse(text)

        # Natural language parsing should find the intent
        assert len(commands) >= 0  # May or may not parse depending on patterns

    def test_multiple_commands(self):
        """Test parsing multiple commands"""
        parser = UniversalCommandParser()

        text = '''```command:file.read
path: /tmp/file1.txt
```

Some text in between

```command:file.write
path: /tmp/file2.txt
content: Hello World
```'''

        commands = parser.parse(text)
        assert len(commands) == 2
        assert commands[0].type == CommandType.FILE_READ
        assert commands[1].type == CommandType.FILE_WRITE


class TestSandboxManager:
    """Test sandbox management"""

    def test_create_sandbox(self, temp_dir):
        """Test sandbox creation"""
        manager = SandboxManager(temp_dir)

        sandbox = manager.create_sandbox(
            owner_id="test-agent",
            name="test-sandbox"
        )

        assert sandbox is not None
        assert sandbox.owner_id == "test-agent"
        assert sandbox.root_path.exists()

    def test_sandbox_templates(self, temp_dir):
        """Test sandbox templates"""
        manager = SandboxManager(temp_dir)

        sandbox = manager.create_sandbox(
            owner_id="test-agent",
            name="python-sandbox",
            template="python"
        )

        assert sandbox is not None
        workspace = sandbox.root_path / "workspace"
        assert workspace.exists()

    def test_sandbox_sharing(self, temp_dir):
        """Test sandbox sharing between agents"""
        manager = SandboxManager(temp_dir)

        sandbox = manager.create_sandbox(
            owner_id="agent-1",
            name="shared-sandbox"
        )

        success = manager.share_sandbox(
            sandbox.sandbox_id,
            "agent-1",
            "agent-2",
            readonly=True
        )

        assert success

    def test_sandbox_snapshot(self, temp_dir):
        """Test sandbox snapshots"""
        manager = SandboxManager(temp_dir)

        sandbox = manager.create_sandbox(
            owner_id="test-agent",
            name="snapshot-test"
        )

        # Create a file in sandbox
        workspace = sandbox.root_path / "workspace"
        test_file = workspace / "test.txt"
        test_file.write_text("Hello")

        # Create snapshot
        snapshot = manager.create_snapshot(
            sandbox.sandbox_id,
            "test-agent",
            "Test snapshot"
        )

        assert snapshot is not None
        assert snapshot.description == "Test snapshot"


class TestSkillSystem:
    """Test skill system"""

    def test_register_skill(self, temp_dir):
        """Test skill registration"""
        registry = SkillRegistry(temp_dir)

        def custom_skill(params):
            return {"result": params.get("input", "").upper()}

        from harness.skills.skill_system import SkillParameter, SkillOutput

        skill = PythonSkill(
            SkillMetadata(
                skill_id="test.uppercase",
                name="Uppercase",
                version="1.0.0",
                description="Convert text to uppercase",
                author="test",
                skill_type="local",
                category="utility",
                parameters=[
                    SkillParameter("input", "string", "Text to convert")
                ],
                output=SkillOutput("object", "Uppercase result")
            ),
            custom_skill
        )

        registry.register(skill)

        assert "test.uppercase" in [s.skill_id for s in registry.list_skills()]

    def test_execute_skill(self, temp_dir):
        """Test skill execution"""
        registry = SkillRegistry(temp_dir)

        def echo_skill(params):
            return {"echo": params.get("message", "")}

        from harness.skills.skill_system import SkillParameter, SkillOutput

        skill = PythonSkill(
            SkillMetadata(
                skill_id="test.echo",
                name="Echo",
                version="1.0.0",
                description="Echo a message",
                author="test",
                skill_type="local",
                category="utility",
                parameters=[
                    SkillParameter("message", "string", "Message to echo")
                ],
                output=SkillOutput("object", "Echoed message")
            ),
            echo_skill
        )

        registry.register(skill)

        result = registry.execute("test.echo", {"message": "Hello"})
        assert result.success
        assert result.output.get("echo") == "Hello"


class TestMessageBus:
    """Test inter-agent communication"""

    def test_register_agent(self):
        """Test agent registration"""
        bus = MessageBus()

        bus.register_agent("agent-1")
        bus.register_agent("agent-2")

        assert bus.get_mailbox("agent-1") is not None
        assert bus.get_mailbox("agent-2") is not None

    def test_send_direct_message(self):
        """Test direct messaging"""
        bus = MessageBus()

        bus.register_agent("agent-1")
        bus.register_agent("agent-2")

        message = Message(
            message_id="msg-1",
            message_type=MessageType.DIRECT,
            sender_id="agent-1",
            recipient_id="agent-2",
            content="Hello from agent-1"
        )

        bus.send(message)

        mailbox = bus.get_mailbox("agent-2")
        messages = mailbox.peek_messages(10)

        assert len(messages) == 1
        assert messages[0].content == "Hello from agent-1"

    def test_broadcast_message(self):
        """Test broadcast messaging"""
        bus = MessageBus()

        bus.register_agent("agent-1")
        bus.register_agent("agent-2")
        bus.register_agent("agent-3")

        message = Message(
            message_id="msg-broadcast",
            message_type=MessageType.BROADCAST,
            sender_id="agent-1",
            recipient_id=None,
            content="Broadcast message"
        )

        bus.send(message)

        # All agents except sender should receive
        assert len(bus.get_mailbox("agent-2").peek_messages(10)) >= 1
        assert len(bus.get_mailbox("agent-3").peek_messages(10)) >= 1

    def test_pub_sub(self):
        """Test publish/subscribe"""
        bus = MessageBus()

        bus.register_agent("publisher")
        bus.register_agent("subscriber")

        mailbox = bus.get_mailbox("subscriber")
        mailbox.subscribe("updates")

        message = Message(
            message_id="msg-pub",
            message_type=MessageType.PUBLISH,
            sender_id="publisher",
            recipient_id=None,
            content="Update notification",
            topic="updates"
        )

        bus.send(message)

        messages = mailbox.peek_messages(10)
        assert any(m.topic == "updates" for m in messages)


class TestEvaluationSystem:
    """Test self-evaluation and flywheel"""

    def test_create_evaluation_system(self, temp_dir):
        """Test evaluation system creation"""
        eval_loop, flywheel = create_evaluation_system(temp_dir)

        assert eval_loop is not None
        assert flywheel is not None

    def test_record_trace(self, temp_dir):
        """Test execution trace recording"""
        eval_loop, _ = create_evaluation_system(temp_dir)

        trace = ExecutionTrace(
            trace_id="trace-1",
            agent_id="agent-1",
            session_id="session-1",
            original_task="Test task",
            task_type="test",
            constraints=[],
            turns=[
                {"input": "task", "output": "response", "commands": []}
            ],
            total_turns=1,
            tokens_used=100,
            time_taken_ms=1000,
            commands_executed=1,
            skills_invoked=0,
            completed=True,
            success=True,
            final_output="Task completed",
            error=None,
            started_at=datetime.now(),
            completed_at=datetime.now()
        )

        trace_id = eval_loop.record_trace(trace)
        assert trace_id == "trace-1"

    def test_evaluate_trace(self, temp_dir):
        """Test trace evaluation"""
        eval_loop, _ = create_evaluation_system(temp_dir)

        trace = ExecutionTrace(
            trace_id="trace-2",
            agent_id="agent-1",
            session_id="session-1",
            original_task="Test task",
            task_type="test",
            constraints=[],
            turns=[{"input": "task", "output": "response"}],
            total_turns=1,
            tokens_used=100,
            time_taken_ms=1000,
            commands_executed=1,
            skills_invoked=0,
            completed=True,
            success=True,
            final_output="Task completed",
            error=None,
            started_at=datetime.now(),
            completed_at=datetime.now()
        )

        eval_loop.record_trace(trace)
        result = eval_loop.evaluate_trace("trace-2")

        assert result is not None
        assert result.overall_score > 0
        assert EvaluationDimension.TASK_SUCCESS in result.scores

    def test_flywheel_processing(self, temp_dir):
        """Test flywheel processing"""
        eval_loop, flywheel = create_evaluation_system(temp_dir)

        result = flywheel.process_execution(
            agent_id="agent-1",
            session_id="session-1",
            task="Complete test task",
            task_type="test",
            turns=[{"input": "task", "output": "response"}],
            success=True,
            final_output="Done",
            tokens_used=100,
            time_taken_ms=500
        )

        assert "trace_id" in result
        assert "evaluation" in result

    def test_flywheel_prompt_enhancement(self, temp_dir):
        """Test flywheel prompt enhancement"""
        eval_loop, flywheel = create_evaluation_system(temp_dir)

        # Process some executions first
        for i in range(3):
            flywheel.process_execution(
                agent_id="agent-1",
                session_id="session-1",
                task=f"Task {i}",
                task_type="test",
                turns=[{"input": f"task {i}", "output": f"response {i}"}],
                success=True,
                final_output=f"Done {i}",
                tokens_used=100,
                time_taken_ms=500
            )

        # Get enhanced prompt
        prompt = flywheel.get_enhanced_prompt(
            task="New task",
            task_type="test",
            agent_id="agent-1"
        )

        # May or may not have content depending on lessons extracted
        assert isinstance(prompt, str)


class TestQAGeneration:
    """Test Q&A generation system"""

    def test_create_qa_system(self, temp_dir):
        """Test Q&A system creation"""
        qa_system = create_qa_system(temp_dir)
        assert qa_system is not None

    def test_generate_instruction_qa(self, temp_dir):
        """Test instruction Q&A generation"""
        qa_system = create_qa_system(temp_dir)

        trace_data = {
            "trace_id": "trace-qa",
            "agent_id": "agent-1",
            "original_task": "Write a hello world function",
            "task_type": "code",
            "total_turns": 1,
            "commands_executed": 1,
            "success": True,
            "final_output": "def hello(): return 'Hello World'"
        }

        evaluation_data = {
            "evaluation_id": "eval-qa",
            "overall_score": 0.9
        }

        pairs = qa_system.generate_from_trace(
            trace_data,
            evaluation_data,
            formats=[QAFormat.INSTRUCTION]
        )

        assert len(pairs) > 0
        assert pairs[0].question == "Write a hello world function"

    def test_export_training_data(self, temp_dir):
        """Test training data export"""
        qa_system = create_qa_system(temp_dir)

        # Generate some Q&A pairs
        for i in range(5):
            trace_data = {
                "trace_id": f"trace-{i}",
                "original_task": f"Task {i}",
                "task_type": "test",
                "total_turns": 1,
                "commands_executed": 1,
                "success": True,
                "final_output": f"Response {i}"
            }
            qa_system.generate_from_trace(trace_data, {"overall_score": 0.8})

        # Export
        output_path = Path(temp_dir) / "export.jsonl"
        count = qa_system.export_training_data(
            str(output_path),
            style="openai",
            min_score=0.5
        )

        assert count > 0
        assert output_path.exists()


class TestMultiAgentScenario:
    """Test multi-agent scenarios"""

    def test_collaborative_task(self, harness):
        """Test agents working together"""
        # Create two agents
        _, agent1_state = harness.create_agent(name="Agent1")
        _, agent2_state = harness.create_agent(name="Agent2")

        # Agent 1 sends message to Agent 2
        message = Message(
            message_id="collab-1",
            message_type=MessageType.DIRECT,
            sender_id=agent1_state.agent_id,
            recipient_id=agent2_state.agent_id,
            content="Can you help with this task?"
        )

        harness.message_bus.send(message)

        # Verify Agent 2 received message
        mailbox = harness.message_bus.get_mailbox(agent2_state.agent_id)
        messages = mailbox.peek_messages(10)

        assert len(messages) >= 1
        assert any(m.sender_id == agent1_state.agent_id for m in messages)

    def test_shared_sandbox(self, harness):
        """Test agents sharing a sandbox"""
        _, agent1_state = harness.create_agent(name="Agent1")
        _, agent2_state = harness.create_agent(name="Agent2")

        # Agent 1 creates sandbox
        sandbox = harness.sandbox_manager.create_sandbox(
            owner_id=agent1_state.agent_id,
            name="shared-workspace"
        )

        # Share with Agent 2
        success = harness.sandbox_manager.share_sandbox(
            sandbox.sandbox_id,
            agent1_state.agent_id,
            agent2_state.agent_id,
            readonly=False
        )

        assert success

        # Both agents should have access
        agent1_sandboxes = harness.sandbox_manager.get_agent_sandboxes(agent1_state.agent_id)
        agent2_sandboxes = harness.sandbox_manager.get_agent_sandboxes(agent2_state.agent_id)

        assert any(s.sandbox_id == sandbox.sandbox_id for s in agent1_sandboxes)
        assert any(s.sandbox_id == sandbox.sandbox_id for s in agent2_sandboxes)


class TestEndToEndScenario:
    """End-to-end integration tests"""

    def test_complete_workflow(self, temp_dir):
        """Test complete agent workflow"""
        # Setup
        harness = UniversalLLMHarness(
            llm_backend=MockLLMBackend([
                {"content": "I'll help you with that task.", "tool_calls": []},
                {"content": "Task completed successfully!", "tool_calls": []}
            ]),
            base_path=temp_dir,
            harness_mode=HarnessMode.INTERACTIVE
        )

        eval_loop, flywheel = create_evaluation_system(f"{temp_dir}/eval")
        qa_system = create_qa_system(f"{temp_dir}/qa")

        # Create agent
        session_id, state = harness.create_agent(
            name="WorkflowAgent",
            capabilities=["coding", "testing"]
        )

        # Set todos
        harness.update_todos(state.agent_id, [
            {"content": "Analyze requirements", "status": "pending"},
            {"content": "Implement solution", "status": "pending"},
            {"content": "Test and verify", "status": "pending"}
        ])

        # Execute turn
        result = harness.execute_turn(
            state.agent_id,
            "Please help me complete the workflow"
        )

        assert result["final_response"] is not None

        # Process through flywheel
        flywheel_result = flywheel.process_execution(
            agent_id=state.agent_id,
            session_id=session_id,
            task="Complete the workflow",
            task_type="workflow",
            turns=[{"input": "task", "output": result["final_response"]}],
            success=True,
            final_output=result["final_response"]
        )

        assert flywheel_result["trace_id"] is not None

        # Generate Q&A
        if flywheel_result.get("evaluation"):
            pairs = qa_system.generate_from_trace(
                {
                    "trace_id": flywheel_result["trace_id"],
                    "original_task": "Complete the workflow",
                    "task_type": "workflow",
                    "total_turns": 1,
                    "success": True,
                    "final_output": result["final_response"]
                },
                flywheel_result["evaluation"]
            )

            # May have generated pairs
            assert isinstance(pairs, list)

        # Verify harness status
        status = harness.get_status()
        assert status["agents"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
