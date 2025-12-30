#!/usr/bin/env python3
"""
Test Runner for Universal LLM Agent Harness

Runs tests without requiring pytest.
"""

import sys
import os
import json
import tempfile
import shutil
import traceback
from pathlib import Path
from datetime import datetime

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


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def colorize(text, color):
    if sys.stdout.isatty():
        return f"{color}{text}{Colors.ENDC}"
    return text


class MockLLMBackend(LLMBackend):
    """Deterministic mock backend for testing"""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0
        self.messages_received = []

    def generate(self, messages, tools=None, max_tokens=4096):
        self.messages_received.append(messages)
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return {"content": f"Mock response #{self.call_count}", "tool_calls": []}

    def count_tokens(self, text):
        return len(text) // 4

    @property
    def supports_tools(self):
        return True

    @property
    def max_context_tokens(self):
        return 128000


class TestRunner:
    """Simple test runner"""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.temp_dir = None

    def setup(self):
        self.temp_dir = tempfile.mkdtemp()

    def teardown(self):
        if self.temp_dir:
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None

    def run_test(self, name, func):
        """Run a single test"""
        try:
            self.setup()
            func()
            self.passed += 1
            print(f"  {colorize('PASS', Colors.GREEN)} {name}")
        except AssertionError as e:
            self.failed += 1
            self.errors.append((name, e))
            print(f"  {colorize('FAIL', Colors.RED)} {name}: {e}")
        except Exception as e:
            self.failed += 1
            self.errors.append((name, e))
            print(f"  {colorize('ERROR', Colors.RED)} {name}: {e}")
            traceback.print_exc()
        finally:
            self.teardown()

    # ============ HARNESS TESTS ============

    def test_create_agent(self):
        harness = UniversalLLMHarness(
            llm_backend=MockLLMBackend(),
            base_path=self.temp_dir
        )
        session_id, state = harness.create_agent(name="TestAgent", capabilities=["test"])
        assert session_id is not None, "Session ID should not be None"
        assert state.agent_id is not None, "Agent ID should not be None"

    def test_execute_turn(self):
        harness = UniversalLLMHarness(
            llm_backend=MockLLMBackend(),
            base_path=self.temp_dir
        )
        _, state = harness.create_agent(name="TestAgent")
        result = harness.execute_turn(state.agent_id, "Hello")
        assert "final_response" in result, "Result should contain final_response"

    def test_context_building(self):
        harness = UniversalLLMHarness(
            llm_backend=MockLLMBackend(),
            base_path=self.temp_dir
        )
        _, state = harness.create_agent(name="TestAgent")
        context = harness.build_context(state.agent_id, "Test task")
        assert context.agent_id == state.agent_id, "Context agent_id should match"
        assert context.original_task == "Test task", "Task should match"

    def test_todo_management(self):
        harness = UniversalLLMHarness(
            llm_backend=MockLLMBackend(),
            base_path=self.temp_dir
        )
        _, state = harness.create_agent(name="TestAgent")
        todos = [{"content": "Task 1", "status": "pending"}]
        success = harness.update_todos(state.agent_id, todos)
        assert success, "Update todos should succeed"
        updated = harness.get_agent(state.agent_id)
        assert len(updated.todos) == 1, "Should have 1 todo"

    def test_context_compaction(self):
        harness = UniversalLLMHarness(
            llm_backend=MockLLMBackend(),
            base_path=self.temp_dir
        )
        _, state = harness.create_agent(name="TestAgent")
        result = harness.compact_context(state.agent_id)
        assert result.get("compacted") == True, "Should be compacted"

    # ============ COMMAND PROTOCOL TESTS ============

    def test_text_block_parsing(self):
        parser = UniversalCommandParser()
        text = '''```command:file.read
path: /tmp/test.txt
```'''
        commands = parser.parse(text)
        assert len(commands) == 1, "Should parse 1 command"
        assert commands[0].type == CommandType.FILE_READ, "Should be file read"

    def test_tool_call_parsing(self):
        parser = UniversalCommandParser()
        tool_call = {"tool": "file_read", "arguments": {"path": "/tmp/test.txt"}}
        commands = parser.parse(tool_call)
        assert len(commands) == 1, "Should parse 1 command"

    # ============ SANDBOX TESTS ============

    def test_create_sandbox(self):
        manager = SandboxManager(self.temp_dir)
        sandbox = manager.create_sandbox(owner_id="test", name="test-sb")
        assert sandbox is not None, "Sandbox should be created"
        assert sandbox.root_path.exists(), "Sandbox path should exist"

    def test_sandbox_sharing(self):
        manager = SandboxManager(self.temp_dir)
        sandbox = manager.create_sandbox(owner_id="agent-1", name="shared")
        success = manager.share_sandbox(sandbox.sandbox_id, "agent-1", "agent-2")
        assert success, "Sharing should succeed"

    def test_sandbox_snapshot(self):
        manager = SandboxManager(self.temp_dir)
        sandbox = manager.create_sandbox(owner_id="test", name="snapshot-test")
        workspace = sandbox.root_path / "workspace"
        (workspace / "test.txt").write_text("Hello")
        snapshot = manager.create_snapshot(sandbox.sandbox_id, "test", "Test")
        assert snapshot is not None, "Snapshot should be created"

    # ============ SKILL TESTS ============

    def test_register_skill(self):
        from harness.skills.skill_system import SkillParameter, SkillOutput
        registry = SkillRegistry(self.temp_dir)

        def test_skill(params):
            return {"result": "OK"}

        skill = PythonSkill(
            SkillMetadata(
                skill_id="test.skill",
                name="Test",
                version="1.0.0",
                description="Test skill",
                author="test",
                skill_type="local",
                category="utility",
                parameters=[],
                output=SkillOutput("object", "Result")
            ),
            test_skill
        )
        registry.register(skill)
        assert "test.skill" in [s.skill_id for s in registry.list_skills()]

    def test_execute_skill(self):
        from harness.skills.skill_system import SkillParameter, SkillOutput
        registry = SkillRegistry(self.temp_dir)

        def echo_skill(params):
            return {"echo": params.get("msg", "")}

        skill = PythonSkill(
            SkillMetadata(
                skill_id="test.echo",
                name="Echo",
                version="1.0.0",
                description="Echo",
                author="test",
                skill_type="local",
                category="utility",
                parameters=[SkillParameter("msg", "string", "Message")],
                output=SkillOutput("object", "Echo")
            ),
            echo_skill
        )
        registry.register(skill)
        result = registry.execute("test.echo", {"msg": "Hello"})
        assert result.success, "Skill should succeed"
        assert result.output.get("echo") == "Hello", "Should echo message"

    # ============ MESSAGE BUS TESTS ============

    def test_register_agent_bus(self):
        bus = MessageBus()
        bus.register_agent("agent-1")
        assert bus.get_mailbox("agent-1") is not None

    def test_send_message(self):
        bus = MessageBus()
        bus.register_agent("agent-1")
        bus.register_agent("agent-2")
        msg = Message(
            message_id="msg-1",
            message_type=MessageType.DIRECT,
            sender_id="agent-1",
            recipient_id="agent-2",
            content="Hello"
        )
        bus.send(msg)
        mailbox = bus.get_mailbox("agent-2")
        messages = mailbox.peek_messages(10)
        assert len(messages) == 1, "Should have 1 message"

    def test_broadcast_message(self):
        bus = MessageBus()
        bus.register_agent("agent-1")
        bus.register_agent("agent-2")
        bus.register_agent("agent-3")
        msg = Message(
            message_id="broadcast",
            message_type=MessageType.BROADCAST,
            sender_id="agent-1",
            recipient_id=None,
            content="Broadcast"
        )
        bus.send(msg)
        assert len(bus.get_mailbox("agent-2").peek_messages(10)) >= 1
        assert len(bus.get_mailbox("agent-3").peek_messages(10)) >= 1

    # ============ EVALUATION TESTS ============

    def test_evaluation_system(self):
        eval_loop, flywheel = create_evaluation_system(self.temp_dir)
        assert eval_loop is not None
        assert flywheel is not None

    def test_record_trace(self):
        eval_loop, _ = create_evaluation_system(self.temp_dir)
        trace = ExecutionTrace(
            trace_id="trace-1",
            agent_id="agent-1",
            session_id="session-1",
            original_task="Test",
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
            final_output="Done",
            error=None,
            started_at=datetime.now(),
            completed_at=datetime.now()
        )
        trace_id = eval_loop.record_trace(trace)
        assert trace_id == "trace-1"

    def test_evaluate_trace(self):
        eval_loop, _ = create_evaluation_system(self.temp_dir)
        trace = ExecutionTrace(
            trace_id="trace-2",
            agent_id="agent-1",
            session_id="session-1",
            original_task="Test",
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
            final_output="Done",
            error=None,
            started_at=datetime.now(),
            completed_at=datetime.now()
        )
        eval_loop.record_trace(trace)
        result = eval_loop.evaluate_trace("trace-2")
        assert result is not None
        assert result.overall_score > 0

    def test_flywheel_processing(self):
        _, flywheel = create_evaluation_system(self.temp_dir)
        result = flywheel.process_execution(
            agent_id="agent-1",
            session_id="session-1",
            task="Test task",
            task_type="test",
            turns=[{"input": "task", "output": "response"}],
            success=True,
            final_output="Done",
            tokens_used=100,
            time_taken_ms=500
        )
        assert "trace_id" in result
        assert "evaluation" in result

    # ============ Q&A TESTS ============

    def test_qa_system(self):
        qa_system = create_qa_system(self.temp_dir)
        assert qa_system is not None

    def test_generate_qa(self):
        qa_system = create_qa_system(self.temp_dir)
        trace_data = {
            "trace_id": "qa-1",
            "original_task": "Write hello world",
            "task_type": "code",
            "total_turns": 1,
            "commands_executed": 1,
            "success": True,
            "final_output": "def hello(): print('Hello')"
        }
        pairs = qa_system.generate_from_trace(
            trace_data,
            {"overall_score": 0.9},
            formats=[QAFormat.INSTRUCTION]
        )
        assert len(pairs) > 0

    def test_export_qa(self):
        qa_system = create_qa_system(self.temp_dir)
        for i in range(3):
            qa_system.generate_from_trace(
                {"trace_id": f"t-{i}", "original_task": f"Task {i}", "success": True, "final_output": f"Output {i}"},
                {"overall_score": 0.8}
            )
        output = f"{self.temp_dir}/export.jsonl"
        count = qa_system.export_training_data(output)
        assert count > 0
        assert Path(output).exists()

    # ============ INTEGRATION TESTS ============

    def test_multi_agent_scenario(self):
        harness = UniversalLLMHarness(
            llm_backend=MockLLMBackend(),
            base_path=self.temp_dir
        )
        _, agent1 = harness.create_agent(name="Agent1")
        _, agent2 = harness.create_agent(name="Agent2")

        msg = Message(
            message_id="collab",
            message_type=MessageType.DIRECT,
            sender_id=agent1.agent_id,
            recipient_id=agent2.agent_id,
            content="Hello"
        )
        harness.message_bus.send(msg)

        mailbox = harness.message_bus.get_mailbox(agent2.agent_id)
        assert len(mailbox.peek_messages(10)) >= 1

    def test_shared_sandbox_scenario(self):
        harness = UniversalLLMHarness(
            llm_backend=MockLLMBackend(),
            base_path=self.temp_dir
        )
        _, agent1 = harness.create_agent(name="Agent1")
        _, agent2 = harness.create_agent(name="Agent2")

        sandbox = harness.sandbox_manager.create_sandbox(
            owner_id=agent1.agent_id,
            name="shared"
        )
        success = harness.sandbox_manager.share_sandbox(
            sandbox.sandbox_id,
            agent1.agent_id,
            agent2.agent_id
        )
        assert success

    def test_end_to_end(self):
        harness = UniversalLLMHarness(
            llm_backend=MockLLMBackend([
                {"content": "I'll help with that.", "tool_calls": []},
            ]),
            base_path=f"{self.temp_dir}/harness"
        )
        eval_loop, flywheel = create_evaluation_system(f"{self.temp_dir}/eval")
        qa_system = create_qa_system(f"{self.temp_dir}/qa")

        session_id, state = harness.create_agent(name="E2EAgent")
        result = harness.execute_turn(state.agent_id, "Test task")
        assert result["final_response"] is not None

        flywheel_result = flywheel.process_execution(
            agent_id=state.agent_id,
            session_id=session_id,
            task="Test",
            task_type="test",
            turns=[{"input": "task", "output": result["final_response"]}],
            success=True,
            final_output=result["final_response"]
        )
        assert flywheel_result["trace_id"] is not None


def main():
    """Run all tests"""
    print(colorize("\n" + "=" * 60, Colors.BOLD))
    print(colorize("UNIVERSAL LLM AGENT HARNESS - TEST SUITE", Colors.BOLD))
    print(colorize("=" * 60 + "\n", Colors.BOLD))

    runner = TestRunner()

    test_groups = [
        ("Harness Core", [
            ("Create Agent", runner.test_create_agent),
            ("Execute Turn", runner.test_execute_turn),
            ("Context Building", runner.test_context_building),
            ("Todo Management", runner.test_todo_management),
            ("Context Compaction", runner.test_context_compaction),
        ]),
        ("Command Protocol", [
            ("Text Block Parsing", runner.test_text_block_parsing),
            ("Tool Call Parsing", runner.test_tool_call_parsing),
        ]),
        ("Sandbox Manager", [
            ("Create Sandbox", runner.test_create_sandbox),
            ("Sandbox Sharing", runner.test_sandbox_sharing),
            ("Sandbox Snapshot", runner.test_sandbox_snapshot),
        ]),
        ("Skill System", [
            ("Register Skill", runner.test_register_skill),
            ("Execute Skill", runner.test_execute_skill),
        ]),
        ("Message Bus", [
            ("Register Agent", runner.test_register_agent_bus),
            ("Send Message", runner.test_send_message),
            ("Broadcast Message", runner.test_broadcast_message),
        ]),
        ("Evaluation System", [
            ("Create Evaluation System", runner.test_evaluation_system),
            ("Record Trace", runner.test_record_trace),
            ("Evaluate Trace", runner.test_evaluate_trace),
            ("Flywheel Processing", runner.test_flywheel_processing),
        ]),
        ("Q&A Generation", [
            ("Create Q&A System", runner.test_qa_system),
            ("Generate Q&A", runner.test_generate_qa),
            ("Export Q&A", runner.test_export_qa),
        ]),
        ("Integration", [
            ("Multi-Agent Scenario", runner.test_multi_agent_scenario),
            ("Shared Sandbox Scenario", runner.test_shared_sandbox_scenario),
            ("End-to-End", runner.test_end_to_end),
        ]),
    ]

    for group_name, tests in test_groups:
        print(colorize(f"\n{group_name}:", Colors.BLUE))
        for test_name, test_func in tests:
            runner.run_test(test_name, test_func)

    # Summary
    print(colorize("\n" + "=" * 60, Colors.BOLD))
    total = runner.passed + runner.failed
    print(f"Results: {colorize(str(runner.passed), Colors.GREEN)} passed, "
          f"{colorize(str(runner.failed), Colors.RED)} failed, "
          f"{total} total")

    if runner.errors:
        print(colorize("\nFailed Tests:", Colors.RED))
        for name, error in runner.errors:
            print(f"  - {name}: {error}")

    print(colorize("=" * 60 + "\n", Colors.BOLD))

    return 0 if runner.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
