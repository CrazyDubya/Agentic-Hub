#!/usr/bin/env python3
"""
Advanced LLM Capabilities Test Suite

Tests multi-step reasoning, todo management, and multi-agent interaction
patterns across a broad spectrum of LLM providers through OpenRouter.

Test Categories:
1. Multi-Step Reasoning - Complex tasks requiring sequential thought
2. Todo Management - Task planning and status tracking
3. Multi-Agent Interaction - Simulated agent collaboration
4. Tool Orchestration - Chaining multiple tool calls

Model Spectrum:
- Premium: Mistral Large, DeepSeek, Gemini 3 Flash
- Mid-tier: Qwen 2.5, Phi-4, Llama 3.3
- Budget: Gemma 3, Nova Micro, Mistral Small
- Free tier: Various community/free models
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class TestCategory(Enum):
    MULTI_STEP = "multi_step"
    TODO_MANAGEMENT = "todo_management"
    MULTI_AGENT = "multi_agent"
    TOOL_ORCHESTRATION = "tool_orchestration"


class ModelTier(Enum):
    PREMIUM = "premium"
    MID = "mid"
    BUDGET = "budget"
    FREE = "free"


@dataclass
class ModelConfig:
    id: str
    name: str
    tier: ModelTier
    supports_tools: bool
    description: str


@dataclass
class TestResult:
    model_id: str
    model_name: str
    model_tier: ModelTier
    test_category: TestCategory
    test_name: str
    success: bool
    score: float  # 0.0 to 1.0
    response_time_ms: float
    tokens_used: int = 0
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


# Model configurations across price spectrum
MODELS = [
    # Premium tier
    ModelConfig("mistralai/mistral-large-2411", "Mistral Large 2411",
                ModelTier.PREMIUM, True, "Premium tool-calling model"),
    ModelConfig("deepseek/deepseek-chat", "DeepSeek Chat",
                ModelTier.PREMIUM, True, "DeepSeek flagship"),
    ModelConfig("google/gemini-3-flash-preview", "Gemini 3 Flash",
                ModelTier.PREMIUM, True, "Google's latest Flash model"),

    # Mid tier
    ModelConfig("qwen/qwen-2.5-72b-instruct", "Qwen 2.5 72B",
                ModelTier.MID, False, "Alibaba's large model"),
    ModelConfig("meta-llama/llama-3.3-70b-instruct:free", "Llama 3.3 70B",
                ModelTier.MID, True, "Meta's latest Llama"),
    ModelConfig("microsoft/phi-4", "Phi-4",
                ModelTier.MID, False, "Microsoft's efficient model"),

    # Budget tier
    ModelConfig("amazon/nova-micro-v1", "Nova Micro",
                ModelTier.BUDGET, False, "Amazon's micro model"),
    ModelConfig("mistralai/mistral-nemo", "Mistral Nemo",
                ModelTier.BUDGET, True, "Mistral's efficient model"),
    ModelConfig("cohere/command-r7b-12-2024", "Command R 7B",
                ModelTier.BUDGET, True, "Cohere's small model"),

    # Free tier
    ModelConfig("google/gemma-3-12b-it:free", "Gemma 3 12B",
                ModelTier.FREE, False, "Google's free Gemma"),
    ModelConfig("mistralai/mistral-small-3.1-24b-instruct:free", "Mistral Small 3.1",
                ModelTier.FREE, True, "Free Mistral Small"),
    ModelConfig("qwen/qwen3-4b:free", "Qwen 3 4B",
                ModelTier.FREE, False, "Qwen's tiny model"),
]


class OpenRouterClient:
    """Client for OpenRouter API calls."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = OPENROUTER_BASE_URL

    def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> Tuple[Dict[str, Any], float]:
        """Make a chat completion request. Returns (response, time_ms)."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/CrazyDubya/Agentic-Hub",
            "X-Title": "Agentic-Hub Advanced Tests"
        }

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        if tools:
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

        start = time.time()
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=90
        )
        elapsed_ms = (time.time() - start) * 1000

        if response.status_code != 200:
            error_msg = response.text[:200]
            try:
                error_msg = response.json().get("error", {}).get("message", error_msg)
            except:
                pass
            return {"error": f"API error ({response.status_code}): {error_msg}"}, elapsed_ms

        return response.json(), elapsed_ms


class AdvancedTestRunner:
    """Runs advanced capability tests across models."""

    def __init__(self, api_key: str):
        self.client = OpenRouterClient(api_key)
        self.results: List[TestResult] = []

    # =========================================================================
    # MULTI-STEP REASONING TESTS
    # =========================================================================

    def test_multi_step_math(self, model: ModelConfig) -> TestResult:
        """Test multi-step mathematical reasoning."""
        messages = [
            {"role": "system", "content": "You are a precise math tutor. Show your work step by step."},
            {"role": "user", "content": """Solve this step by step:
A store has 3 shelves. The first shelf has 24 books. The second shelf has twice as many books as the first.
The third shelf has half as many books as the second shelf minus 5.
How many books are there in total?

Show each step clearly, then give the final answer."""}
        ]

        response, time_ms = self.client.chat(model.id, messages, max_tokens=500)

        if "error" in response:
            return TestResult(
                model_id=model.id, model_name=model.name, model_tier=model.tier,
                test_category=TestCategory.MULTI_STEP, test_name="multi_step_math",
                success=False, score=0.0, response_time_ms=time_ms,
                error=response["error"]
            )

        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        tokens = response.get("usage", {}).get("total_tokens", 0)

        # Check for correct answer (24 + 48 + 19 = 91)
        has_correct_answer = "91" in content
        has_steps = any(step in content.lower() for step in ["step", "first", "second", "third", "therefore", "total"])
        shows_work = "24" in content and "48" in content

        score = (0.5 if has_correct_answer else 0.0) + (0.3 if has_steps else 0.0) + (0.2 if shows_work else 0.0)

        return TestResult(
            model_id=model.id, model_name=model.name, model_tier=model.tier,
            test_category=TestCategory.MULTI_STEP, test_name="multi_step_math",
            success=has_correct_answer, score=score, response_time_ms=time_ms,
            tokens_used=tokens,
            details={"correct_answer": has_correct_answer, "shows_steps": has_steps, "response_preview": content[:200]}
        )

    def test_multi_step_logic(self, model: ModelConfig) -> TestResult:
        """Test multi-step logical reasoning."""
        messages = [
            {"role": "system", "content": "You are a logical reasoning expert. Analyze carefully."},
            {"role": "user", "content": """There are 4 people: Alice, Bob, Carol, and Dave.
- Alice is taller than Bob
- Carol is shorter than Dave
- Bob is taller than Carol
- Dave is not the tallest

Who is the tallest? Explain your reasoning step by step."""}
        ]

        response, time_ms = self.client.chat(model.id, messages, max_tokens=400)

        if "error" in response:
            return TestResult(
                model_id=model.id, model_name=model.name, model_tier=model.tier,
                test_category=TestCategory.MULTI_STEP, test_name="multi_step_logic",
                success=False, score=0.0, response_time_ms=time_ms,
                error=response["error"]
            )

        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        tokens = response.get("usage", {}).get("total_tokens", 0)

        # Correct answer is Alice (Alice > Bob > Carol, Dave > Carol but Dave not tallest)
        has_correct = "alice" in content.lower() and "tallest" in content.lower()
        has_reasoning = len(content) > 100

        score = (0.7 if has_correct else 0.0) + (0.3 if has_reasoning else 0.0)

        return TestResult(
            model_id=model.id, model_name=model.name, model_tier=model.tier,
            test_category=TestCategory.MULTI_STEP, test_name="multi_step_logic",
            success=has_correct, score=score, response_time_ms=time_ms,
            tokens_used=tokens,
            details={"correct_answer": has_correct, "response_preview": content[:200]}
        )

    # =========================================================================
    # TODO MANAGEMENT TESTS
    # =========================================================================

    def test_todo_parsing(self, model: ModelConfig) -> TestResult:
        """Test ability to parse and structure todo items."""
        messages = [
            {"role": "system", "content": "You are a task management assistant. Output structured JSON only."},
            {"role": "user", "content": """Parse these tasks into a JSON todo list with status:
1. Review the pull request (completed yesterday)
2. Write unit tests for the new feature
3. Deploy to staging (in progress, 50% done)
4. Update documentation
5. Send weekly report (due tomorrow, not started)

Output as JSON array with fields: task, status (pending/in_progress/completed), priority (high/medium/low)"""}
        ]

        response, time_ms = self.client.chat(model.id, messages, max_tokens=600)

        if "error" in response:
            return TestResult(
                model_id=model.id, model_name=model.name, model_tier=model.tier,
                test_category=TestCategory.TODO_MANAGEMENT, test_name="todo_parsing",
                success=False, score=0.0, response_time_ms=time_ms,
                error=response["error"]
            )

        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        tokens = response.get("usage", {}).get("total_tokens", 0)

        # Try to parse JSON
        is_valid_json = False
        has_all_tasks = False
        has_correct_structure = False

        try:
            # Extract JSON from response
            json_start = content.find("[")
            json_end = content.rfind("]") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                todos = json.loads(json_str)
                is_valid_json = True
                has_all_tasks = len(todos) >= 4
                has_correct_structure = all(
                    "task" in t and "status" in t
                    for t in todos
                )
        except:
            pass

        score = (0.4 if is_valid_json else 0.0) + (0.3 if has_all_tasks else 0.0) + (0.3 if has_correct_structure else 0.0)

        return TestResult(
            model_id=model.id, model_name=model.name, model_tier=model.tier,
            test_category=TestCategory.TODO_MANAGEMENT, test_name="todo_parsing",
            success=is_valid_json and has_correct_structure, score=score, response_time_ms=time_ms,
            tokens_used=tokens,
            details={"valid_json": is_valid_json, "has_all_tasks": has_all_tasks, "correct_structure": has_correct_structure}
        )

    def test_todo_prioritization(self, model: ModelConfig) -> TestResult:
        """Test ability to prioritize and reorder tasks."""
        messages = [
            {"role": "system", "content": "You are a productivity expert. Be decisive and clear."},
            {"role": "user", "content": """Given these tasks and constraints, output them in priority order (1 = highest):

Tasks:
- Fix critical security bug (blocks release)
- Add dark mode feature (customer request)
- Update dependencies (routine maintenance)
- Write blog post about new features (marketing wants it)
- Respond to support tickets (3 urgent ones waiting)

Constraints:
- Release deadline is tomorrow
- Only 4 hours available today

Output a numbered list 1-5 with brief justification for each position."""}
        ]

        response, time_ms = self.client.chat(model.id, messages, max_tokens=500)

        if "error" in response:
            return TestResult(
                model_id=model.id, model_name=model.name, model_tier=model.tier,
                test_category=TestCategory.TODO_MANAGEMENT, test_name="todo_prioritization",
                success=False, score=0.0, response_time_ms=time_ms,
                error=response["error"]
            )

        content = response.get("choices", [{}])[0].get("message", {}).get("content", "").lower()
        tokens = response.get("usage", {}).get("total_tokens", 0)

        # Check reasonable prioritization (security bug should be #1)
        security_first = "security" in content[:200] or ("1" in content and "security" in content.split("2")[0] if "2" in content else False)
        has_numbered_list = "1." in content or "1)" in content or "#1" in content
        has_justifications = len(content) > 200

        score = (0.5 if security_first else 0.0) + (0.25 if has_numbered_list else 0.0) + (0.25 if has_justifications else 0.0)

        return TestResult(
            model_id=model.id, model_name=model.name, model_tier=model.tier,
            test_category=TestCategory.TODO_MANAGEMENT, test_name="todo_prioritization",
            success=security_first, score=score, response_time_ms=time_ms,
            tokens_used=tokens,
            details={"security_prioritized": security_first, "has_list": has_numbered_list}
        )

    # =========================================================================
    # MULTI-AGENT INTERACTION TESTS
    # =========================================================================

    def test_agent_role_play(self, model: ModelConfig) -> TestResult:
        """Test ability to simulate different agent perspectives."""
        messages = [
            {"role": "system", "content": "You will simulate a conversation between two AI agents."},
            {"role": "user", "content": """Simulate a brief exchange between:
- Agent A (Code Reviewer): Reviews code for quality
- Agent B (Developer): Defends their implementation

Topic: Agent B submitted code that works but has no error handling.

Format each message as:
[Agent X]: message

Show 2-3 exchanges that reach a constructive conclusion."""}
        ]

        response, time_ms = self.client.chat(model.id, messages, max_tokens=600)

        if "error" in response:
            return TestResult(
                model_id=model.id, model_name=model.name, model_tier=model.tier,
                test_category=TestCategory.MULTI_AGENT, test_name="agent_role_play",
                success=False, score=0.0, response_time_ms=time_ms,
                error=response["error"]
            )

        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        tokens = response.get("usage", {}).get("total_tokens", 0)

        # Check for proper agent simulation
        has_agent_a = "[agent a]" in content.lower() or "agent a:" in content.lower()
        has_agent_b = "[agent b]" in content.lower() or "agent b:" in content.lower()
        has_multiple_turns = content.lower().count("agent") >= 4
        mentions_error_handling = "error" in content.lower()

        score = (0.3 if has_agent_a else 0.0) + (0.3 if has_agent_b else 0.0) + \
                (0.2 if has_multiple_turns else 0.0) + (0.2 if mentions_error_handling else 0.0)

        return TestResult(
            model_id=model.id, model_name=model.name, model_tier=model.tier,
            test_category=TestCategory.MULTI_AGENT, test_name="agent_role_play",
            success=has_agent_a and has_agent_b, score=score, response_time_ms=time_ms,
            tokens_used=tokens,
            details={"has_both_agents": has_agent_a and has_agent_b, "multi_turn": has_multiple_turns}
        )

    def test_agent_delegation(self, model: ModelConfig) -> TestResult:
        """Test ability to understand agent delegation patterns."""
        messages = [
            {"role": "system", "content": "You are a task routing AI that delegates to specialized agents."},
            {"role": "user", "content": """You have access to these agents:
- CodeAgent: Writes and reviews code
- TestAgent: Creates and runs tests
- DocAgent: Writes documentation
- DeployAgent: Handles deployments

For this request, output which agents should be involved and in what order:
"Create a new REST API endpoint for user profiles, make sure it's tested and documented, then deploy to staging."

Format as:
1. [AgentName]: task description
2. [AgentName]: task description
..."""}
        ]

        response, time_ms = self.client.chat(model.id, messages, max_tokens=400)

        if "error" in response:
            return TestResult(
                model_id=model.id, model_name=model.name, model_tier=model.tier,
                test_category=TestCategory.MULTI_AGENT, test_name="agent_delegation",
                success=False, score=0.0, response_time_ms=time_ms,
                error=response["error"]
            )

        content = response.get("choices", [{}])[0].get("message", {}).get("content", "").lower()
        tokens = response.get("usage", {}).get("total_tokens", 0)

        # Check for proper delegation
        has_code = "codeagent" in content or "code agent" in content
        has_test = "testagent" in content or "test agent" in content
        has_doc = "docagent" in content or "doc agent" in content
        has_deploy = "deployagent" in content or "deploy agent" in content
        has_order = any(f"{i}." in content or f"{i})" in content for i in range(1, 5))

        agents_mentioned = sum([has_code, has_test, has_doc, has_deploy])
        score = (agents_mentioned * 0.2) + (0.2 if has_order else 0.0)

        return TestResult(
            model_id=model.id, model_name=model.name, model_tier=model.tier,
            test_category=TestCategory.MULTI_AGENT, test_name="agent_delegation",
            success=agents_mentioned >= 3, score=score, response_time_ms=time_ms,
            tokens_used=tokens,
            details={"agents_mentioned": agents_mentioned, "has_order": has_order}
        )

    # =========================================================================
    # TOOL ORCHESTRATION TESTS
    # =========================================================================

    def test_tool_chain(self, model: ModelConfig) -> TestResult:
        """Test ability to chain multiple tool calls."""
        if not model.supports_tools:
            return TestResult(
                model_id=model.id, model_name=model.name, model_tier=model.tier,
                test_category=TestCategory.TOOL_ORCHESTRATION, test_name="tool_chain",
                success=True, score=0.0, response_time_ms=0,
                details={"skipped": "Model does not support tools"}
            )

        messages = [
            {"role": "system", "content": "You are a helpful assistant with access to tools. Use them when needed."},
            {"role": "user", "content": "I need to know the weather in Tokyo and then convert the temperature from Celsius to Fahrenheit."}
        ]

        tools = [
            {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"]
                }
            },
            {
                "name": "convert_temperature",
                "description": "Convert temperature between units",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number", "description": "Temperature value"},
                        "from_unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        "to_unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["value", "from_unit", "to_unit"]
                }
            }
        ]

        response, time_ms = self.client.chat(model.id, messages, tools=tools, max_tokens=300)

        if "error" in response:
            return TestResult(
                model_id=model.id, model_name=model.name, model_tier=model.tier,
                test_category=TestCategory.TOOL_ORCHESTRATION, test_name="tool_chain",
                success=False, score=0.0, response_time_ms=time_ms,
                error=response["error"]
            )

        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        tool_calls = message.get("tool_calls", [])
        tokens = response.get("usage", {}).get("total_tokens", 0)

        # Check for appropriate tool usage
        made_tool_call = len(tool_calls) > 0
        called_weather = any("weather" in (tc.get("function", {}).get("name", "") or "").lower() for tc in tool_calls)

        score = (0.5 if made_tool_call else 0.0) + (0.5 if called_weather else 0.0)

        return TestResult(
            model_id=model.id, model_name=model.name, model_tier=model.tier,
            test_category=TestCategory.TOOL_ORCHESTRATION, test_name="tool_chain",
            success=made_tool_call, score=score, response_time_ms=time_ms,
            tokens_used=tokens,
            details={"made_tool_call": made_tool_call, "called_weather": called_weather, "num_calls": len(tool_calls)}
        )

    def test_tool_selection(self, model: ModelConfig) -> TestResult:
        """Test ability to select the right tool for a task."""
        if not model.supports_tools:
            return TestResult(
                model_id=model.id, model_name=model.name, model_tier=model.tier,
                test_category=TestCategory.TOOL_ORCHESTRATION, test_name="tool_selection",
                success=True, score=0.0, response_time_ms=0,
                details={"skipped": "Model does not support tools"}
            )

        messages = [
            {"role": "system", "content": "You have access to specialized tools. Select the most appropriate one."},
            {"role": "user", "content": "Search for Python files in the src directory that contain the word 'async'."}
        ]

        tools = [
            {
                "name": "file_read",
                "description": "Read contents of a specific file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }
            },
            {
                "name": "file_search",
                "description": "Search for files matching a pattern and containing text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory": {"type": "string"},
                        "pattern": {"type": "string", "description": "File glob pattern like *.py"},
                        "contains": {"type": "string", "description": "Text to search for in files"}
                    },
                    "required": ["directory"]
                }
            },
            {
                "name": "file_write",
                "description": "Write content to a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"]
                }
            }
        ]

        response, time_ms = self.client.chat(model.id, messages, tools=tools, max_tokens=200)

        if "error" in response:
            return TestResult(
                model_id=model.id, model_name=model.name, model_tier=model.tier,
                test_category=TestCategory.TOOL_ORCHESTRATION, test_name="tool_selection",
                success=False, score=0.0, response_time_ms=time_ms,
                error=response["error"]
            )

        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        tool_calls = message.get("tool_calls", [])
        tokens = response.get("usage", {}).get("total_tokens", 0)

        # Check for correct tool selection
        selected_search = any("search" in (tc.get("function", {}).get("name", "") or "").lower() for tc in tool_calls)

        score = 1.0 if selected_search else 0.0

        return TestResult(
            model_id=model.id, model_name=model.name, model_tier=model.tier,
            test_category=TestCategory.TOOL_ORCHESTRATION, test_name="tool_selection",
            success=selected_search, score=score, response_time_ms=time_ms,
            tokens_used=tokens,
            details={"selected_correct_tool": selected_search}
        )

    # =========================================================================
    # TEST RUNNER
    # =========================================================================

    def run_all_tests(self, models: List[ModelConfig] = None) -> List[TestResult]:
        """Run all tests across all models."""
        if models is None:
            models = MODELS

        tests = [
            # Multi-step reasoning
            ("multi_step_math", self.test_multi_step_math),
            ("multi_step_logic", self.test_multi_step_logic),
            # Todo management
            ("todo_parsing", self.test_todo_parsing),
            ("todo_prioritization", self.test_todo_prioritization),
            # Multi-agent
            ("agent_role_play", self.test_agent_role_play),
            ("agent_delegation", self.test_agent_delegation),
            # Tool orchestration
            ("tool_chain", self.test_tool_chain),
            ("tool_selection", self.test_tool_selection),
        ]

        print("\n" + "=" * 80)
        print("ADVANCED LLM CAPABILITIES TEST SUITE")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Models: {len(models)}")
        print(f"Tests per model: {len(tests)}")
        print(f"Total tests: {len(models) * len(tests)}")
        print("=" * 80)

        all_results = []

        for model in models:
            print(f"\n{'─' * 80}")
            print(f"Model: {model.name} ({model.tier.value})")
            print(f"ID: {model.id}")
            print(f"Tools: {'Yes' if model.supports_tools else 'No'}")
            print("─" * 80)

            for test_name, test_func in tests:
                print(f"  Running {test_name}...", end=" ", flush=True)

                try:
                    result = test_func(model)
                    all_results.append(result)

                    if result.error:
                        print(f"ERROR ({result.response_time_ms:.0f}ms)")
                        print(f"    Error: {result.error[:60]}...")
                    elif result.details.get("skipped"):
                        print("SKIPPED")
                    else:
                        status = "PASS" if result.success else "FAIL"
                        print(f"{status} (score: {result.score:.2f}, {result.response_time_ms:.0f}ms, {result.tokens_used} tokens)")

                except Exception as e:
                    print(f"EXCEPTION: {e}")
                    all_results.append(TestResult(
                        model_id=model.id, model_name=model.name, model_tier=model.tier,
                        test_category=TestCategory.MULTI_STEP, test_name=test_name,
                        success=False, score=0.0, response_time_ms=0,
                        error=str(e)
                    ))

                time.sleep(1.5)  # Rate limiting

        self.results = all_results
        return all_results

    def print_summary(self):
        """Print comprehensive test summary."""
        print("\n" + "=" * 80)
        print("TEST SUMMARY REPORT")
        print("=" * 80)

        # By model
        print("\n## Results by Model\n")
        model_stats = {}
        for r in self.results:
            if r.model_name not in model_stats:
                model_stats[r.model_name] = {
                    "tier": r.model_tier.value,
                    "passed": 0, "failed": 0, "skipped": 0,
                    "total_score": 0.0, "total_time": 0.0, "count": 0
                }
            stats = model_stats[r.model_name]

            if r.details.get("skipped"):
                stats["skipped"] += 1
            elif r.success:
                stats["passed"] += 1
            else:
                stats["failed"] += 1

            if not r.details.get("skipped"):
                stats["total_score"] += r.score
                stats["total_time"] += r.response_time_ms
                stats["count"] += 1

        print(f"{'Model':<30} {'Tier':<10} {'Pass':<6} {'Fail':<6} {'Skip':<6} {'Avg Score':<10} {'Avg Time':<10}")
        print("-" * 80)

        for model, stats in model_stats.items():
            avg_score = stats["total_score"] / stats["count"] if stats["count"] > 0 else 0
            avg_time = stats["total_time"] / stats["count"] if stats["count"] > 0 else 0
            print(f"{model:<30} {stats['tier']:<10} {stats['passed']:<6} {stats['failed']:<6} {stats['skipped']:<6} {avg_score:<10.2f} {avg_time:<10.0f}ms")

        # By tier
        print("\n## Results by Tier\n")
        tier_stats = {}
        for r in self.results:
            tier = r.model_tier.value
            if tier not in tier_stats:
                tier_stats[tier] = {"passed": 0, "failed": 0, "total_score": 0, "count": 0}
            if not r.details.get("skipped"):
                if r.success:
                    tier_stats[tier]["passed"] += 1
                else:
                    tier_stats[tier]["failed"] += 1
                tier_stats[tier]["total_score"] += r.score
                tier_stats[tier]["count"] += 1

        for tier, stats in sorted(tier_stats.items()):
            total = stats["passed"] + stats["failed"]
            pct = (stats["passed"] / total * 100) if total > 0 else 0
            avg_score = stats["total_score"] / stats["count"] if stats["count"] > 0 else 0
            print(f"  {tier.upper()}: {stats['passed']}/{total} passed ({pct:.0f}%), avg score: {avg_score:.2f}")

        # By test category
        print("\n## Results by Test Category\n")
        cat_stats = {}
        for r in self.results:
            cat = r.test_category.value
            if cat not in cat_stats:
                cat_stats[cat] = {"passed": 0, "failed": 0, "total_score": 0, "count": 0}
            if not r.details.get("skipped"):
                if r.success:
                    cat_stats[cat]["passed"] += 1
                else:
                    cat_stats[cat]["failed"] += 1
                cat_stats[cat]["total_score"] += r.score
                cat_stats[cat]["count"] += 1

        for cat, stats in sorted(cat_stats.items()):
            total = stats["passed"] + stats["failed"]
            pct = (stats["passed"] / total * 100) if total > 0 else 0
            avg_score = stats["total_score"] / stats["count"] if stats["count"] > 0 else 0
            print(f"  {cat}: {stats['passed']}/{total} passed ({pct:.0f}%), avg score: {avg_score:.2f}")

        # Overall
        total_tests = len([r for r in self.results if not r.details.get("skipped")])
        total_passed = len([r for r in self.results if r.success and not r.details.get("skipped")])
        total_score = sum(r.score for r in self.results if not r.details.get("skipped"))

        print("\n" + "=" * 80)
        print(f"OVERALL: {total_passed}/{total_tests} tests passed ({total_passed/total_tests*100:.0f}%)")
        print(f"AVERAGE SCORE: {total_score/total_tests:.2f}")
        print("=" * 80)


def main():
    """Main entry point."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        api_key = "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"

    runner = AdvancedTestRunner(api_key)

    try:
        runner.run_all_tests()
        runner.print_summary()

        # Return exit code
        failed = len([r for r in runner.results if not r.success and not r.details.get("skipped")])
        sys.exit(0 if failed == 0 else 1)

    except KeyboardInterrupt:
        print("\nTest interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
