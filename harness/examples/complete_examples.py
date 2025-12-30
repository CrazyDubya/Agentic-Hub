#!/usr/bin/env python3
"""
Complete Examples for Universal LLM Agent Harness

This file demonstrates all features of the harness including:
1. Basic agent creation and task execution
2. Multi-agent collaboration
3. Self-evaluation and flywheel
4. Q&A generation for training
5. CLI usage patterns
6. Advanced scenarios
"""

import sys
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from harness import (
    UniversalLLMHarness,
    HarnessMode,
    LLMBackend,
    PythonSkill,
    SkillMetadata,
    Message,
    MessageType,
)
from harness.skills.skill_system import SkillParameter, SkillOutput
from harness.evaluation import (
    SelfEvaluationLoop,
    FlywheelManager,
    ExecutionTrace,
    QAGenerationSystem,
    QAFormat,
    QAGenerationConfig,
    create_evaluation_system,
    create_qa_system
)


class DemoLLMBackend(LLMBackend):
    """Demo backend that simulates intelligent responses"""

    def __init__(self):
        self.turn = 0

    def generate(self, messages, tools=None, max_tokens=4096):
        self.turn += 1

        # Extract user message
        user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")[:100]
                break

        # Simulate contextual responses
        if "analyze" in user_msg.lower():
            response = """I'll analyze that for you.

```command:file.read
path: ./src/main.py
```

Based on my analysis, the code structure looks well-organized.
The main components are properly separated and follow good practices."""

        elif "write" in user_msg.lower() or "create" in user_msg.lower():
            response = """I'll create that for you.

```command:file.write
path: ./output/result.txt
content: This is the generated content.
```

I've created the file with the requested content."""

        elif "test" in user_msg.lower():
            response = """I'll run the tests.

```command:shell.exec
command: pytest tests/ -v
```

All tests passed! The implementation is working correctly."""

        else:
            response = f"""I understand you want me to: {user_msg}

Let me work on that step by step:
1. First, I'll analyze the requirements
2. Then, I'll implement the solution
3. Finally, I'll verify the results

[DONE] Task completed successfully."""

        return {
            "content": response,
            "tool_calls": []
        }

    def count_tokens(self, text):
        return len(text) // 4

    @property
    def supports_tools(self):
        return False

    @property
    def max_context_tokens(self):
        return 128000


def example_1_basic_usage():
    """Example 1: Basic agent creation and task execution"""
    print("\n" + "=" * 60)
    print("Example 1: Basic Agent Usage")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create harness with demo backend
        harness = UniversalLLMHarness(
            llm_backend=DemoLLMBackend(),
            base_path=temp_dir,
            harness_mode=HarnessMode.INTERACTIVE
        )

        # Create an agent
        session_id, state = harness.create_agent(
            name="BasicAgent",
            capabilities=["general", "analysis"]
        )

        print(f"Created agent: {state.agent_id}")
        print(f"Session: {session_id}")

        # Execute a simple task
        result = harness.execute_turn(
            state.agent_id,
            "Analyze the current codebase structure"
        )

        print(f"\nAgent response:\n{result['final_response'][:500]}...")
        print(f"\nCommands executed: {result.get('state', {}).get('metrics', {}).get('commands_executed', 0)}")


def example_2_todo_management():
    """Example 2: Todo list and task management"""
    print("\n" + "=" * 60)
    print("Example 2: Todo List Management")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        harness = UniversalLLMHarness(
            llm_backend=DemoLLMBackend(),
            base_path=temp_dir,
            harness_mode=HarnessMode.AUTONOMOUS
        )

        _, state = harness.create_agent(name="TodoAgent")

        # Set up todo list
        todos = [
            {"content": "Analyze requirements", "status": "completed"},
            {"content": "Design solution", "status": "in_progress"},
            {"content": "Implement core features", "status": "pending"},
            {"content": "Write tests", "status": "pending"},
            {"content": "Documentation", "status": "pending"}
        ]

        harness.update_todos(state.agent_id, todos)

        print("Todo list updated:")
        for todo in todos:
            status_icon = "✓" if todo["status"] == "completed" else "○" if todo["status"] == "pending" else "◔"
            print(f"  {status_icon} {todo['content']}")

        # The todo list will be included in the context
        context = harness.build_context(state.agent_id, "Continue working")
        print(f"\nTodos visible to agent: {len(context.state.todos)}")


def example_3_custom_skills():
    """Example 3: Creating and using custom skills"""
    print("\n" + "=" * 60)
    print("Example 3: Custom Skills")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        harness = UniversalLLMHarness(
            llm_backend=DemoLLMBackend(),
            base_path=temp_dir
        )

        # Create custom analysis skill
        def code_metrics_skill(params):
            code = params.get("code", "")
            return {
                "lines": len(code.splitlines()),
                "characters": len(code),
                "words": len(code.split()),
                "functions": code.count("def "),
                "classes": code.count("class ")
            }

        skill = PythonSkill(
            SkillMetadata(
                skill_id="analysis.code_metrics",
                name="Code Metrics",
                version="1.0.0",
                description="Analyze code and return metrics",
                author="example",
                skill_type="local",
                category="code_analysis",
                parameters=[
                    SkillParameter("code", "string", "Source code to analyze")
                ],
                output=SkillOutput("object", "Code metrics")
            ),
            code_metrics_skill
        )

        harness.skill_registry.register(skill)
        print(f"Registered skill: {skill.metadata.skill_id}")

        # Execute the skill
        test_code = """
def hello():
    print("Hello World")

class Greeter:
    def greet(self, name):
        return f"Hello, {name}!"
"""

        result = harness.skill_registry.execute(
            "analysis.code_metrics",
            {"code": test_code}
        )

        print(f"\nSkill execution result:")
        print(json.dumps(result.output, indent=2))


def example_4_multi_agent_collaboration():
    """Example 4: Multi-agent collaboration"""
    print("\n" + "=" * 60)
    print("Example 4: Multi-Agent Collaboration")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        harness = UniversalLLMHarness(
            llm_backend=DemoLLMBackend(),
            base_path=temp_dir,
            harness_mode=HarnessMode.COLLABORATIVE
        )

        # Create specialized agents
        _, coder = harness.create_agent(
            name="CodeWriter",
            capabilities=["coding", "implementation"],
            specializations=["python", "javascript"]
        )

        _, reviewer = harness.create_agent(
            name="CodeReviewer",
            capabilities=["review", "testing"],
            specializations=["code-quality", "security"]
        )

        _, documenter = harness.create_agent(
            name="Documenter",
            capabilities=["documentation"],
            specializations=["technical-writing", "api-docs"]
        )

        print(f"Created agents:")
        print(f"  - {coder.agent_id} (CodeWriter)")
        print(f"  - {reviewer.agent_id} (CodeReviewer)")
        print(f"  - {documenter.agent_id} (Documenter)")

        # Create shared sandbox
        sandbox = harness.sandbox_manager.create_sandbox(
            owner_id=coder.agent_id,
            name="team-workspace"
        )

        # Share with team
        harness.sandbox_manager.share_sandbox(sandbox.sandbox_id, coder.agent_id, reviewer.agent_id)
        harness.sandbox_manager.share_sandbox(sandbox.sandbox_id, coder.agent_id, documenter.agent_id)

        print(f"\nShared sandbox: {sandbox.sandbox_id}")

        # Simulate collaboration via messages
        # Coder sends work to reviewer
        msg1 = Message(
            message_id="collab-1",
            message_type=MessageType.DIRECT,
            sender_id=coder.agent_id,
            recipient_id=reviewer.agent_id,
            content={"type": "review_request", "file": "main.py"}
        )
        harness.message_bus.send(msg1)

        # Reviewer approves and notifies documenter
        msg2 = Message(
            message_id="collab-2",
            message_type=MessageType.DIRECT,
            sender_id=reviewer.agent_id,
            recipient_id=documenter.agent_id,
            content={"type": "document_request", "approved": True}
        )
        harness.message_bus.send(msg2)

        print("\nMessages sent for collaboration flow")

        # Check mailboxes
        reviewer_mail = harness.message_bus.get_mailbox(reviewer.agent_id)
        doc_mail = harness.message_bus.get_mailbox(documenter.agent_id)

        print(f"Reviewer has {reviewer_mail.inbox.qsize()} messages")
        print(f"Documenter has {doc_mail.inbox.qsize()} messages")


def example_5_self_evaluation_flywheel():
    """Example 5: Self-evaluation and flywheel effect"""
    print("\n" + "=" * 60)
    print("Example 5: Self-Evaluation and Flywheel")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create evaluation system
        eval_loop, flywheel = create_evaluation_system(f"{temp_dir}/eval")

        print("Simulating execution traces and evaluations...")

        # Simulate multiple executions
        executions = [
            {"task": "Write a sorting function", "success": True, "turns": 2},
            {"task": "Debug the login issue", "success": True, "turns": 5},
            {"task": "Implement caching", "success": False, "turns": 8},
            {"task": "Add unit tests", "success": True, "turns": 3},
            {"task": "Optimize database queries", "success": True, "turns": 4},
        ]

        for i, exec_data in enumerate(executions):
            result = flywheel.process_execution(
                agent_id="test-agent",
                session_id=f"session-{i}",
                task=exec_data["task"],
                task_type="coding",
                turns=[{"input": exec_data["task"], "output": f"Response for {exec_data['task']}"}],
                success=exec_data["success"],
                final_output=f"Completed: {exec_data['task']}",
                tokens_used=500,
                time_taken_ms=2000
            )

            status = "✓" if exec_data["success"] else "✗"
            score = result.get("evaluation", {}).get("overall_score", 0)
            print(f"  {status} {exec_data['task'][:40]:<40} Score: {score:.2f}")

        # Get flywheel status
        status = flywheel.get_flywheel_status()
        print(f"\nFlywheel Status:")
        print(f"  Total traces: {status['metrics']['traces_recorded']}")
        print(f"  Evaluations: {status['metrics']['evaluations_performed']}")
        print(f"  Lessons learned: {status['metrics']['lessons_extracted']}")
        print(f"  Training data: {status['metrics']['training_data_generated']}")
        print(f"  Average score: {status['average_score']:.2f}")
        print(f"  Trend: {status['trend_direction']}")

        # Get enhanced prompt with learned lessons
        prompt = flywheel.get_enhanced_prompt(
            task="New coding task",
            task_type="coding",
            agent_id="test-agent"
        )

        if prompt:
            print(f"\nEnhanced prompt with lessons:\n{prompt[:300]}...")


def example_6_qa_generation():
    """Example 6: Q&A Generation for Training"""
    print("\n" + "=" * 60)
    print("Example 6: Q&A Generation System")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        qa_system = create_qa_system(f"{temp_dir}/qa")

        # Generate Q&A pairs from traces
        traces = [
            {
                "trace_id": "qa-1",
                "original_task": "Write a Python function to reverse a string",
                "task_type": "coding",
                "total_turns": 1,
                "commands_executed": 1,
                "success": True,
                "final_output": "def reverse_string(s): return s[::-1]"
            },
            {
                "trace_id": "qa-2",
                "original_task": "Explain how list comprehensions work in Python",
                "task_type": "explanation",
                "total_turns": 1,
                "commands_executed": 0,
                "success": True,
                "final_output": "List comprehensions provide a concise way to create lists. The syntax is [expr for item in iterable if condition]."
            },
            {
                "trace_id": "qa-3",
                "original_task": "Debug this function that's returning None",
                "task_type": "debugging",
                "total_turns": 3,
                "commands_executed": 2,
                "success": True,
                "final_output": "The issue was a missing return statement. I added 'return result' at the end of the function.",
                "error": "Function returns None"  # Had error but was resolved
            }
        ]

        print("Generating Q&A pairs from traces...")

        for trace in traces:
            pairs = qa_system.generate_from_trace(
                trace,
                {"overall_score": 0.85},
                formats=[QAFormat.INSTRUCTION, QAFormat.REASONING]
            )
            print(f"  Generated {len(pairs)} pairs from: {trace['original_task'][:40]}...")

        # Show stats
        stats = qa_system.get_stats()
        print(f"\nQ&A Generation Stats:")
        print(f"  Total pairs: {stats['total_generated']}")
        print(f"  By format: {stats['by_format']}")
        print(f"  By quality: {stats['by_quality']}")

        # Export training data
        output_path = f"{temp_dir}/training_data.jsonl"
        count = qa_system.export_training_data(
            output_path,
            style="openai",
            min_score=0.5
        )

        print(f"\nExported {count} Q&A pairs to {output_path}")

        # Show sample
        samples = qa_system.get_sample_pairs(2)
        print("\nSample Q&A pairs:")
        for sample in samples:
            print(f"\n  Q: {sample['question'][:60]}...")
            print(f"  A: {sample['answer'][:60]}...")


def example_7_context_compaction():
    """Example 7: Context compaction for long tasks"""
    print("\n" + "=" * 60)
    print("Example 7: Context Compaction")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        harness = UniversalLLMHarness(
            llm_backend=DemoLLMBackend(),
            base_path=temp_dir,
            harness_mode=HarnessMode.AUTONOMOUS
        )

        _, state = harness.create_agent(name="LongTaskAgent")

        # Simulate work
        state.completed_tasks = [f"Task {i}" for i in range(20)]
        state.working_memory["key_findings"] = ["Finding 1", "Finding 2"]
        state.working_memory["errors"] = ["Error 1"]

        print(f"Before compaction:")
        print(f"  Completed tasks: {len(state.completed_tasks)}")
        print(f"  Compaction count: {state.compaction_count}")

        # Compact context
        result = harness.compact_context(state.agent_id)

        print(f"\nAfter compaction:")
        print(f"  Compacted: {result.get('compacted')}")
        print(f"  Compaction count: {result.get('compaction_count')}")
        print(f"  Summary keys: {list(result.get('summary', {}).keys())}")


def example_8_full_integration():
    """Example 8: Full integration of all systems"""
    print("\n" + "=" * 60)
    print("Example 8: Full System Integration")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize all systems
        harness = UniversalLLMHarness(
            llm_backend=DemoLLMBackend(),
            base_path=f"{temp_dir}/harness",
            harness_mode=HarnessMode.AUTONOMOUS
        )

        eval_loop, flywheel = create_evaluation_system(f"{temp_dir}/eval")
        qa_system = create_qa_system(f"{temp_dir}/qa")

        print("Systems initialized:")
        print(f"  - Harness: {harness.harness_mode.value} mode")
        print(f"  - Evaluation system: ready")
        print(f"  - Q&A system: ready")

        # Create agent with custom skills
        _, state = harness.create_agent(
            name="IntegrationAgent",
            capabilities=["coding", "testing", "documentation"]
        )

        # Register custom skill
        def analyze_skill(params):
            return {"analysis": f"Analyzed: {params.get('target', 'unknown')}"}

        harness.skill_registry.register(PythonSkill(
            SkillMetadata(
                skill_id="custom.analyze",
                name="Analyzer",
                version="1.0.0",
                description="Custom analysis skill",
                author="integration",
                skill_type="local",
                category="analysis",
                parameters=[SkillParameter("target", "string", "Target to analyze")],
                output=SkillOutput("object", "Analysis result")
            ),
            analyze_skill
        ))

        print(f"\nAgent created: {state.agent_id}")
        print(f"Skills available: {len(harness.skill_registry.list_skills())}")

        # Set up task
        harness.update_todos(state.agent_id, [
            {"content": "Analyze requirements", "status": "pending"},
            {"content": "Implement solution", "status": "pending"},
            {"content": "Test implementation", "status": "pending"}
        ])

        # Execute task
        result = harness.execute_turn(
            state.agent_id,
            "Complete all the tasks in my todo list"
        )

        print(f"\nTask execution:")
        print(f"  Iterations: {result.get('iterations', 1)}")
        print(f"  Response length: {len(result.get('final_response', ''))}")

        # Process through flywheel
        flywheel_result = flywheel.process_execution(
            agent_id=state.agent_id,
            session_id=state.session_id,
            task="Complete all tasks",
            task_type="workflow",
            turns=[{"input": "Complete tasks", "output": result.get('final_response', '')}],
            success=True,
            final_output=result.get('final_response', ''),
            tokens_used=500,
            time_taken_ms=1000
        )

        print(f"\nFlywheel processing:")
        print(f"  Trace ID: {flywheel_result['trace_id']}")
        eval_score = flywheel_result.get('evaluation', {}).get('overall_score', 0)
        print(f"  Evaluation score: {eval_score:.2f}")

        # Generate Q&A
        if flywheel_result.get('evaluation'):
            pairs = qa_system.generate_from_trace(
                {
                    "trace_id": flywheel_result['trace_id'],
                    "original_task": "Complete all tasks",
                    "task_type": "workflow",
                    "success": True,
                    "final_output": result.get('final_response', '')
                },
                flywheel_result['evaluation']
            )
            print(f"  Q&A pairs generated: {len(pairs)}")

        # Final status
        print(f"\nFinal System Status:")
        harness_status = harness.get_status()
        print(f"  Agents: {harness_status['agents']}")
        print(f"  Sandboxes: {harness_status['active_sandboxes']}")
        print(f"  Skills: {harness_status['registered_skills']}")

        flywheel_status = flywheel.get_flywheel_status()
        print(f"  Flywheel health: {flywheel_status['health']}")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("UNIVERSAL LLM AGENT HARNESS - COMPLETE EXAMPLES")
    print("=" * 60)

    examples = [
        ("Basic Usage", example_1_basic_usage),
        ("Todo Management", example_2_todo_management),
        ("Custom Skills", example_3_custom_skills),
        ("Multi-Agent Collaboration", example_4_multi_agent_collaboration),
        ("Self-Evaluation Flywheel", example_5_self_evaluation_flywheel),
        ("Q&A Generation", example_6_qa_generation),
        ("Context Compaction", example_7_context_compaction),
        ("Full Integration", example_8_full_integration),
    ]

    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n[ERROR in {name}]: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
