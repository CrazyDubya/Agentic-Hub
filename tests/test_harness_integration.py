#!/usr/bin/env python3
"""
Harness Integration Test Suite

Tests the Universal LLM Harness with real backends under various conditions:
1. Command Protocol - Text block and tool-call command parsing/execution
2. Agent Lifecycle - Creation, state management, context building
3. Todo Management - Manus-style attention tracking
4. Sandbox Operations - File operations within sandboxed environments
5. Skill System - Skill registration and execution
6. Multi-Agent Communication - Message bus and agent collaboration
7. Context Compaction - Long-running task memory management
8. Backend Switching - Hot-swapping LLM backends

Uses OpenRouter to test with multiple real LLM backends.
"""

import os
import sys
import json
import time
import tempfile
import shutil
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness import (
    UniversalLLMHarness,
    HarnessMode,
    LLMBackend,
    AgentState,
)
from harness.core.command_protocol import (
    UniversalCommandParser, CommandType
)
from harness.skills.skill_system import (
    SkillRegistry, PythonSkill, SkillMetadata, SkillParameter, SkillOutput
)
from harness.communication.message_bus import (
    MessageBus, Message, MessageType
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class TestCategory(Enum):
    COMMAND_PROTOCOL = "command_protocol"
    AGENT_LIFECYCLE = "agent_lifecycle"
    TODO_MANAGEMENT = "todo_management"
    SANDBOX_OPS = "sandbox_ops"
    SKILL_SYSTEM = "skill_system"
    MULTI_AGENT = "multi_agent"
    CONTEXT_COMPACTION = "context_compaction"
    LLM_INTEGRATION = "llm_integration"


@dataclass
class HarnessTestResult:
    category: TestCategory
    test_name: str
    passed: bool
    duration_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class OpenRouterBackend(LLMBackend):
    """Real OpenRouter backend for harness integration testing."""

    def __init__(self, api_key: str, model: str, supports_tool_use: bool = True):
        self.api_key = api_key
        self.model = model
        self._supports_tools = supports_tool_use
        self._call_count = 0
        self._total_tokens = 0

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/CrazyDubya/Agentic-Hub"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3
        }

        if tools and self._supports_tools:
            formatted_tools = []
            for tool in tools:
                formatted_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name"),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {})
                    }
                })
            payload["tools"] = formatted_tools

        try:
            response = requests.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=90
            )

            if response.status_code != 200:
                return {"content": "", "tool_calls": [], "error": response.text[:200]}

            data = response.json()
            self._call_count += 1
            self._total_tokens += data.get("usage", {}).get("total_tokens", 0)

            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})

            return {
                "content": message.get("content", ""),
                "tool_calls": message.get("tool_calls", [])
            }
        except Exception as e:
            return {"content": "", "tool_calls": [], "error": str(e)}

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    @property
    def supports_tools(self) -> bool:
        return self._supports_tools

    @property
    def max_context_tokens(self) -> int:
        return 128000

    def get_stats(self) -> Dict[str, int]:
        return {"calls": self._call_count, "tokens": self._total_tokens}


class HarnessIntegrationTester:
    """Runs harness integration tests."""

    def __init__(self, api_key: str, temp_dir: str):
        self.api_key = api_key
        self.temp_dir = temp_dir
        self.results: List[HarnessTestResult] = []

    def _time_test(self, fn) -> Tuple[Any, float]:
        start = time.time()
        result = fn()
        elapsed = (time.time() - start) * 1000
        return result, elapsed

    # =========================================================================
    # COMMAND PROTOCOL TESTS
    # =========================================================================

    def test_text_block_parsing(self) -> HarnessTestResult:
        """Test parsing text block commands."""
        parser = UniversalCommandParser()

        text = '''I'll read that file for you.

```command:file.read
path: /tmp/test.txt
```

Let me also check the directory.

```command:file.list
path: /tmp
pattern: *.txt
```'''

        def run():
            commands = parser.parse(text)
            return {
                "count": len(commands),
                "types": [c.type.value for c in commands],
                "has_file_read": any(c.type == CommandType.FILE_READ for c in commands),
                "has_file_list": any(c.type == CommandType.FILE_LIST for c in commands)
            }

        result, duration = self._time_test(run)
        passed = result["count"] == 2 and result["has_file_read"] and result["has_file_list"]

        return HarnessTestResult(
            TestCategory.COMMAND_PROTOCOL, "text_block_parsing",
            passed, duration, result
        )

    def test_tool_call_parsing(self) -> HarnessTestResult:
        """Test parsing tool call format commands."""
        parser = UniversalCommandParser()

        tool_call = {
            "tool": "Bash",
            "arguments": {"command": "ls -la"}
        }

        def run():
            commands = parser.parse(tool_call)
            return {
                "count": len(commands),
                "type": commands[0].type.value if commands else None,
                "has_command": commands[0].params.get("command") == "ls -la" if commands else False
            }

        result, duration = self._time_test(run)
        passed = result["count"] == 1 and result["type"] == "shell.exec"

        return HarnessTestResult(
            TestCategory.COMMAND_PROTOCOL, "tool_call_parsing",
            passed, duration, result
        )

    # =========================================================================
    # AGENT LIFECYCLE TESTS
    # =========================================================================

    def test_agent_creation(self) -> HarnessTestResult:
        """Test agent creation and state initialization."""
        harness = UniversalLLMHarness(
            base_path=f"{self.temp_dir}/harness1",
            harness_mode=HarnessMode.INTERACTIVE
        )

        def run():
            session_id, state = harness.create_agent(
                name="TestAgent",
                capabilities=["coding", "testing"],
                specializations=["python"]
            )
            return {
                "has_session": session_id is not None,
                "has_state": state is not None,
                "has_agent_id": state.agent_id is not None if state else False,
                "agent_in_harness": state.agent_id in harness.agents if state else False
            }

        result, duration = self._time_test(run)
        passed = all(result.values())

        return HarnessTestResult(
            TestCategory.AGENT_LIFECYCLE, "agent_creation",
            passed, duration, result
        )

    def test_context_building(self) -> HarnessTestResult:
        """Test execution context building."""
        harness = UniversalLLMHarness(
            base_path=f"{self.temp_dir}/harness2",
            harness_mode=HarnessMode.AUTONOMOUS
        )

        def run():
            _, state = harness.create_agent(name="ContextTestAgent")
            context = harness.build_context(state.agent_id, "Test the system")

            return {
                "has_agent_id": context.agent_id == state.agent_id,
                "has_task": context.original_task == "Test the system",
                "has_mode": context.harness_mode == HarnessMode.AUTONOMOUS,
                "has_sandbox": context.sandbox is not None
            }

        result, duration = self._time_test(run)
        passed = result["has_agent_id"] and result["has_task"] and result["has_mode"]

        return HarnessTestResult(
            TestCategory.AGENT_LIFECYCLE, "context_building",
            passed, duration, result
        )

    def test_system_prompt_generation(self) -> HarnessTestResult:
        """Test system prompt contains required sections."""
        harness = UniversalLLMHarness(
            base_path=f"{self.temp_dir}/harness3",
            harness_mode=HarnessMode.INTERACTIVE
        )

        def run():
            _, state = harness.create_agent(name="PromptTestAgent")
            context = harness.build_context(state.agent_id, "Test task")
            prompt = harness.build_system_prompt(context)

            return {
                "has_identity": "Agent Identity" in prompt,
                "has_rules": "Core Rules" in prompt,
                "has_commands": "Command Format" in prompt or "command" in prompt.lower(),
                "has_state": state.agent_id in prompt or "Agent ID" in prompt,
                "length": len(prompt)
            }

        result, duration = self._time_test(run)
        passed = result["has_identity"] and result["has_rules"]

        return HarnessTestResult(
            TestCategory.AGENT_LIFECYCLE, "system_prompt_generation",
            passed, duration, result
        )

    # =========================================================================
    # TODO MANAGEMENT TESTS
    # =========================================================================

    def test_todo_update(self) -> HarnessTestResult:
        """Test todo list update functionality."""
        harness = UniversalLLMHarness(
            base_path=f"{self.temp_dir}/harness4",
            harness_mode=HarnessMode.AUTONOMOUS
        )

        def run():
            _, state = harness.create_agent(name="TodoTestAgent")

            todos = [
                {"content": "Analyze requirements", "status": "completed"},
                {"content": "Write implementation", "status": "in_progress"},
                {"content": "Run tests", "status": "pending"},
            ]

            success = harness.update_todos(state.agent_id, todos)
            updated_state = harness.get_agent(state.agent_id)

            return {
                "update_success": success,
                "todo_count": len(updated_state.todos) if updated_state else 0,
                "has_completed": any(t["status"] == "completed" for t in updated_state.todos) if updated_state else False,
                "has_pending": any(t["status"] == "pending" for t in updated_state.todos) if updated_state else False
            }

        result, duration = self._time_test(run)
        passed = result["update_success"] and result["todo_count"] == 3

        return HarnessTestResult(
            TestCategory.TODO_MANAGEMENT, "todo_update",
            passed, duration, result
        )

    def test_todo_in_context(self) -> HarnessTestResult:
        """Test that todos appear in execution context."""
        harness = UniversalLLMHarness(
            base_path=f"{self.temp_dir}/harness5",
            harness_mode=HarnessMode.AUTONOMOUS
        )

        def run():
            _, state = harness.create_agent(name="TodoContextAgent")
            harness.update_todos(state.agent_id, [
                {"content": "Important task", "status": "pending"}
            ])

            context = harness.build_context(state.agent_id, "Continue work")
            prompt = harness.build_system_prompt(context)

            return {
                "todos_in_state": len(context.state.todos) > 0,
                "todos_in_prompt": "Important task" in prompt or "Todo" in prompt
            }

        result, duration = self._time_test(run)
        passed = result["todos_in_state"]

        return HarnessTestResult(
            TestCategory.TODO_MANAGEMENT, "todo_in_context",
            passed, duration, result
        )

    # =========================================================================
    # SANDBOX OPERATIONS TESTS
    # =========================================================================

    def test_sandbox_creation(self) -> HarnessTestResult:
        """Test sandbox creation for agent."""
        harness = UniversalLLMHarness(
            base_path=f"{self.temp_dir}/harness6",
            harness_mode=HarnessMode.INTERACTIVE
        )

        def run():
            _, state = harness.create_agent(name="SandboxTestAgent")
            sandboxes = harness.sandbox_manager.get_agent_sandboxes(state.agent_id)

            return {
                "has_sandbox": len(sandboxes) > 0,
                "sandbox_exists": sandboxes[0].root_path.exists() if sandboxes else False,
                "has_workspace": (sandboxes[0].root_path / "workspace").exists() if sandboxes else False
            }

        result, duration = self._time_test(run)
        passed = result["has_sandbox"] and result["sandbox_exists"]

        return HarnessTestResult(
            TestCategory.SANDBOX_OPS, "sandbox_creation",
            passed, duration, result
        )

    def test_sandbox_file_operations(self) -> HarnessTestResult:
        """Test file operations within sandbox."""
        harness = UniversalLLMHarness(
            base_path=f"{self.temp_dir}/harness7",
            harness_mode=HarnessMode.INTERACTIVE
        )

        def run():
            _, state = harness.create_agent(name="FileOpAgent")
            sandboxes = harness.sandbox_manager.get_agent_sandboxes(state.agent_id)

            if not sandboxes:
                return {"error": "No sandbox created"}

            workspace = sandboxes[0].root_path / "workspace"
            test_file = workspace / "test.txt"
            test_file.write_text("Hello from harness test")

            content = test_file.read_text()

            return {
                "file_created": test_file.exists(),
                "content_correct": content == "Hello from harness test",
                "workspace_path": str(workspace)
            }

        result, duration = self._time_test(run)
        passed = result.get("file_created", False) and result.get("content_correct", False)

        return HarnessTestResult(
            TestCategory.SANDBOX_OPS, "sandbox_file_operations",
            passed, duration, result
        )

    # =========================================================================
    # SKILL SYSTEM TESTS
    # =========================================================================

    def test_skill_registration(self) -> HarnessTestResult:
        """Test custom skill registration."""
        registry = SkillRegistry(f"{self.temp_dir}/skills1")

        def run():
            def echo_skill(params):
                return {"echo": params.get("message", "")}

            skill = PythonSkill(
                SkillMetadata(
                    skill_id="test.echo",
                    name="Echo",
                    version="1.0.0",
                    description="Echo a message",
                    author="test",
                    skill_type="local",
                    category="utility",
                    parameters=[SkillParameter("message", "string", "Message to echo")],
                    output=SkillOutput("object", "Echoed message")
                ),
                echo_skill
            )

            registry.register(skill)
            skills = registry.list_skills()

            return {
                "registered": "test.echo" in [s.skill_id for s in skills],
                "skill_count": len(skills)
            }

        result, duration = self._time_test(run)
        passed = result["registered"]

        return HarnessTestResult(
            TestCategory.SKILL_SYSTEM, "skill_registration",
            passed, duration, result
        )

    def test_skill_execution(self) -> HarnessTestResult:
        """Test skill execution."""
        registry = SkillRegistry(f"{self.temp_dir}/skills2")

        def run():
            def upper_skill(params):
                return {"result": params.get("text", "").upper()}

            skill = PythonSkill(
                SkillMetadata(
                    skill_id="test.upper",
                    name="Uppercase",
                    version="1.0.0",
                    description="Convert to uppercase",
                    author="test",
                    skill_type="local",
                    category="utility",
                    parameters=[SkillParameter("text", "string", "Text to convert")],
                    output=SkillOutput("object", "Uppercase text")
                ),
                upper_skill
            )

            registry.register(skill)
            result = registry.execute("test.upper", {"text": "hello world"})

            return {
                "success": result.success,
                "output_correct": result.output.get("result") == "HELLO WORLD" if result.output else False
            }

        result, duration = self._time_test(run)
        passed = result["success"] and result["output_correct"]

        return HarnessTestResult(
            TestCategory.SKILL_SYSTEM, "skill_execution",
            passed, duration, result
        )

    # =========================================================================
    # MULTI-AGENT COMMUNICATION TESTS
    # =========================================================================

    def test_message_bus_setup(self) -> HarnessTestResult:
        """Test message bus agent registration."""
        bus = MessageBus()

        def run():
            bus.register_agent("agent-1")
            bus.register_agent("agent-2")

            return {
                "agent1_mailbox": bus.get_mailbox("agent-1") is not None,
                "agent2_mailbox": bus.get_mailbox("agent-2") is not None
            }

        result, duration = self._time_test(run)
        passed = result["agent1_mailbox"] and result["agent2_mailbox"]

        return HarnessTestResult(
            TestCategory.MULTI_AGENT, "message_bus_setup",
            passed, duration, result
        )

    def test_direct_messaging(self) -> HarnessTestResult:
        """Test direct message delivery between agents."""
        bus = MessageBus()

        def run():
            bus.register_agent("sender")
            bus.register_agent("receiver")

            msg = Message(
                message_id="test-msg-1",
                message_type=MessageType.DIRECT,
                sender_id="sender",
                recipient_id="receiver",
                content="Hello from sender"
            )

            bus.send(msg)

            mailbox = bus.get_mailbox("receiver")
            messages = mailbox.peek_messages(10)

            return {
                "message_delivered": len(messages) > 0,
                "correct_content": messages[0].content == "Hello from sender" if messages else False,
                "correct_sender": messages[0].sender_id == "sender" if messages else False
            }

        result, duration = self._time_test(run)
        passed = result["message_delivered"] and result["correct_content"]

        return HarnessTestResult(
            TestCategory.MULTI_AGENT, "direct_messaging",
            passed, duration, result
        )

    # =========================================================================
    # CONTEXT COMPACTION TESTS
    # =========================================================================

    def test_context_compaction(self) -> HarnessTestResult:
        """Test context compaction functionality."""
        harness = UniversalLLMHarness(
            base_path=f"{self.temp_dir}/harness8",
            harness_mode=HarnessMode.AUTONOMOUS
        )

        def run():
            _, state = harness.create_agent(name="CompactionAgent")

            # Simulate work
            state.completed_tasks = [f"Task {i}" for i in range(20)]
            state.working_memory["key_findings"] = ["Finding 1", "Finding 2"]

            result = harness.compact_context(state.agent_id)

            return {
                "compacted": result.get("compacted", False),
                "has_summary": "summary" in result,
                "compaction_count": result.get("compaction_count", 0)
            }

        result, duration = self._time_test(run)
        passed = result["compacted"] and result["has_summary"]

        return HarnessTestResult(
            TestCategory.CONTEXT_COMPACTION, "context_compaction",
            passed, duration, result
        )

    # =========================================================================
    # LLM INTEGRATION TESTS
    # =========================================================================

    def test_llm_backend_integration(self, model_id: str, model_name: str) -> HarnessTestResult:
        """Test harness with real LLM backend."""
        backend = OpenRouterBackend(self.api_key, model_id)
        harness = UniversalLLMHarness(
            llm_backend=backend,
            base_path=f"{self.temp_dir}/harness_llm_{model_name.replace(' ', '_')}",
            harness_mode=HarnessMode.INTERACTIVE
        )

        def run():
            _, state = harness.create_agent(name=f"LLMTest-{model_name}")

            # Execute a simple turn
            result = harness.execute_turn(
                state.agent_id,
                "Say hello and confirm you are working."
            )

            stats = backend.get_stats()

            return {
                "has_response": len(result.get("final_response", "")) > 0,
                "no_error": "error" not in result.get("final_response", "").lower(),
                "llm_called": stats["calls"] > 0,
                "tokens_used": stats["tokens"]
            }

        result, duration = self._time_test(run)
        passed = result.get("has_response", False) and result.get("llm_called", False)

        return HarnessTestResult(
            TestCategory.LLM_INTEGRATION, f"backend_{model_name}",
            passed, duration, result
        )

    # =========================================================================
    # RUNNER
    # =========================================================================

    def run_all_tests(self, models: List[Tuple[str, str]] = None) -> List[HarnessTestResult]:
        """Run all harness integration tests."""
        print("\n" + "=" * 80)
        print("HARNESS INTEGRATION TEST SUITE")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Temp directory: {self.temp_dir}")
        print("=" * 80)

        # Core harness tests (no LLM needed)
        core_tests = [
            ("Command Protocol", [
                self.test_text_block_parsing,
                self.test_tool_call_parsing,
            ]),
            ("Agent Lifecycle", [
                self.test_agent_creation,
                self.test_context_building,
                self.test_system_prompt_generation,
            ]),
            ("Todo Management", [
                self.test_todo_update,
                self.test_todo_in_context,
            ]),
            ("Sandbox Operations", [
                self.test_sandbox_creation,
                self.test_sandbox_file_operations,
            ]),
            ("Skill System", [
                self.test_skill_registration,
                self.test_skill_execution,
            ]),
            ("Multi-Agent", [
                self.test_message_bus_setup,
                self.test_direct_messaging,
            ]),
            ("Context Compaction", [
                self.test_context_compaction,
            ]),
        ]

        for category_name, tests in core_tests:
            print(f"\n{'─' * 80}")
            print(f"{category_name}")
            print("─" * 80)

            for test_fn in tests:
                print(f"  {test_fn.__name__:40} ", end="", flush=True)
                try:
                    result = test_fn()
                    self.results.append(result)
                    status = "PASS" if result.passed else "FAIL"
                    print(f"{status:6} | {result.duration_ms:6.0f}ms")
                except Exception as e:
                    print(f"ERROR  | {str(e)[:40]}")
                    self.results.append(HarnessTestResult(
                        TestCategory.COMMAND_PROTOCOL, test_fn.__name__,
                        False, 0, error=str(e)
                    ))

        # LLM backend integration tests
        if models:
            print(f"\n{'─' * 80}")
            print("LLM Backend Integration")
            print("─" * 80)

            for model_id, model_name in models:
                print(f"  {model_name:40} ", end="", flush=True)
                try:
                    result = self.test_llm_backend_integration(model_id, model_name)
                    self.results.append(result)
                    status = "PASS" if result.passed else "FAIL"
                    tokens = result.details.get("tokens_used", 0)
                    print(f"{status:6} | {result.duration_ms:6.0f}ms | {tokens} tokens")
                except Exception as e:
                    print(f"ERROR  | {str(e)[:40]}")
                    self.results.append(HarnessTestResult(
                        TestCategory.LLM_INTEGRATION, f"backend_{model_name}",
                        False, 0, error=str(e)
                    ))
                time.sleep(1)

        return self.results

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 80)
        print("HARNESS TEST SUMMARY")
        print("=" * 80)

        # By category
        categories = {}
        for r in self.results:
            cat = r.category.value
            if cat not in categories:
                categories[cat] = {"passed": 0, "failed": 0}
            if r.passed:
                categories[cat]["passed"] += 1
            else:
                categories[cat]["failed"] += 1

        print(f"\n{'Category':<25} {'Passed':>8} {'Failed':>8} {'Rate':>10}")
        print("-" * 55)

        for cat, counts in sorted(categories.items()):
            total = counts["passed"] + counts["failed"]
            rate = counts["passed"] / total if total > 0 else 0
            print(f"{cat:<25} {counts['passed']:>8} {counts['failed']:>8} {rate:>9.0%}")

        # Overall
        total_passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        print("\n" + "-" * 55)
        print(f"{'OVERALL':<25} {total_passed:>8} {total - total_passed:>8} {total_passed/total:>9.0%}")
        print("=" * 80)


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        api_key = "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"

    temp_dir = tempfile.mkdtemp(prefix="harness_test_")

    try:
        tester = HarnessIntegrationTester(api_key, temp_dir)

        # Models to test with harness
        models = [
            ("anthropic/claude-haiku-4.5", "Claude 4.5 Haiku"),
            ("google/gemini-3-flash-preview", "Gemini 3 Flash"),
            ("amazon/nova-micro-v1", "Nova Micro"),
        ]

        tester.run_all_tests(models)
        tester.print_summary()

        failed = sum(1 for r in tester.results if not r.passed)
        sys.exit(0 if failed == 0 else 1)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
