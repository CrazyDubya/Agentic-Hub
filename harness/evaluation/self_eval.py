"""
Self-Evaluation Loops and Flywheel System

This module implements a comprehensive self-evaluation system that enables agents to:
1. Evaluate their own performance against multiple criteria
2. Learn from successes and failures
3. Generate training data from execution traces
4. Create a flywheel effect where improvements compound over time

Inspired by:
- Constitutional AI's self-critique approach
- RLHF reward modeling
- Process supervision and outcome supervision
- Meta-learning and few-shot learning
"""

import json
import time
import uuid
import hashlib
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod
import sqlite3


class EvaluationDimension(Enum):
    """Dimensions along which agent performance is evaluated"""
    # Task completion
    TASK_SUCCESS = "task_success"           # Did the agent complete the task?
    TASK_COMPLETENESS = "task_completeness" # How completely was task addressed?

    # Quality metrics
    CODE_QUALITY = "code_quality"           # Quality of generated code
    REASONING_QUALITY = "reasoning_quality" # Quality of reasoning steps
    OUTPUT_QUALITY = "output_quality"       # Quality of final output

    # Efficiency metrics
    EFFICIENCY = "efficiency"               # Steps taken vs optimal
    TOKEN_EFFICIENCY = "token_efficiency"   # Tokens used vs needed
    TIME_EFFICIENCY = "time_efficiency"     # Time taken vs expected

    # Safety and alignment
    SAFETY = "safety"                       # Avoided harmful outputs
    ALIGNMENT = "alignment"                 # Followed instructions properly

    # Communication
    CLARITY = "clarity"                     # Clear communication
    HELPFULNESS = "helpfulness"             # Actually helpful response

    # Tool use
    TOOL_SELECTION = "tool_selection"       # Chose appropriate tools
    TOOL_USAGE = "tool_usage"               # Used tools correctly

    # Collaboration
    COLLABORATION = "collaboration"         # Worked well with others
    DELEGATION = "delegation"               # Delegated appropriately


class EvaluationLevel(Enum):
    """Granularity levels for evaluation"""
    TURN = "turn"               # Single interaction turn
    TASK = "task"               # Complete task (multiple turns)
    SESSION = "session"         # Entire session
    AGENT = "agent"             # Agent lifetime performance


@dataclass
class EvaluationCriteria:
    """Defines criteria for a specific evaluation"""
    dimension: EvaluationDimension
    weight: float = 1.0
    threshold: float = 0.5      # Minimum acceptable score
    description: str = ""

    # Optional custom evaluator
    custom_evaluator: Optional[Callable] = None


@dataclass
class EvaluationResult:
    """Result of a single evaluation"""
    evaluation_id: str
    timestamp: datetime
    agent_id: str
    level: EvaluationLevel
    target_id: str              # Turn ID, task ID, session ID

    # Scores (0.0 to 1.0)
    scores: Dict[EvaluationDimension, float]
    overall_score: float

    # Qualitative feedback
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]

    # Training data potential
    training_worthy: bool       # Good enough to be training data
    training_category: Optional[str]  # Category for training

    # Metadata
    evaluator: str              # Who/what evaluated (self, peer, model)
    confidence: float           # Confidence in evaluation
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "level": self.level.value,
            "target_id": self.target_id,
            "scores": {k.value: v for k, v in self.scores.items()},
            "overall_score": self.overall_score,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "suggestions": self.suggestions,
            "training_worthy": self.training_worthy,
            "training_category": self.training_category,
            "evaluator": self.evaluator,
            "confidence": self.confidence
        }


@dataclass
class ExecutionTrace:
    """Complete trace of an agent's execution for evaluation"""
    trace_id: str
    agent_id: str
    session_id: str

    # Task information
    original_task: str
    task_type: str
    constraints: List[str]

    # Execution steps
    turns: List[Dict[str, Any]]     # List of {input, output, commands, results}
    total_turns: int

    # Resource usage
    tokens_used: int
    time_taken_ms: float
    commands_executed: int
    skills_invoked: int

    # Outcome
    completed: bool
    success: bool
    final_output: str
    error: Optional[str]

    # Timestamps
    started_at: datetime
    completed_at: Optional[datetime]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "original_task": self.original_task,
            "task_type": self.task_type,
            "turns": self.turns,
            "total_turns": self.total_turns,
            "tokens_used": self.tokens_used,
            "time_taken_ms": self.time_taken_ms,
            "commands_executed": self.commands_executed,
            "completed": self.completed,
            "success": self.success,
            "final_output": self.final_output[:500] if self.final_output else "",
            "error": self.error
        }


@dataclass
class Lesson:
    """A learned lesson from evaluation"""
    lesson_id: str
    created_at: datetime

    # Source
    source_trace_id: str
    source_evaluation_id: str

    # Content
    lesson_type: str            # success_pattern, failure_pattern, optimization
    category: str               # code, reasoning, tool_use, etc.
    description: str

    # Conditions
    applies_when: List[str]     # Conditions where this lesson applies
    task_types: List[str]       # Relevant task types

    # Guidance
    do_this: List[str]          # Positive examples
    avoid_this: List[str]       # Anti-patterns

    # Confidence and usage
    confidence: float
    times_applied: int = 0
    success_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lesson_id": self.lesson_id,
            "created_at": self.created_at.isoformat(),
            "lesson_type": self.lesson_type,
            "category": self.category,
            "description": self.description,
            "applies_when": self.applies_when,
            "do_this": self.do_this,
            "avoid_this": self.avoid_this,
            "confidence": self.confidence,
            "success_rate": self.success_rate
        }


class Evaluator(ABC):
    """Abstract base class for evaluators"""

    @abstractmethod
    def evaluate(
        self,
        trace: ExecutionTrace,
        criteria: List[EvaluationCriteria]
    ) -> EvaluationResult:
        """Evaluate an execution trace"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get evaluator name"""
        pass


class RuleBasedEvaluator(Evaluator):
    """Rule-based evaluation using heuristics"""

    def __init__(self):
        self.rules: Dict[EvaluationDimension, List[Callable]] = {}
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Setup default evaluation rules"""
        # Task success rules
        self.rules[EvaluationDimension.TASK_SUCCESS] = [
            lambda t: 1.0 if t.success else 0.0,
            lambda t: 1.0 if t.completed else 0.3,
            lambda t: 0.0 if t.error else 1.0
        ]

        # Efficiency rules
        self.rules[EvaluationDimension.EFFICIENCY] = [
            lambda t: min(1.0, 5.0 / max(1, t.total_turns)),  # Penalize many turns
            lambda t: min(1.0, 10000 / max(1, t.tokens_used)),  # Token efficiency
        ]

        # Tool usage rules
        self.rules[EvaluationDimension.TOOL_USAGE] = [
            lambda t: min(1.0, t.commands_executed / max(1, t.total_turns)),
            lambda t: 1.0 if t.skills_invoked > 0 or t.commands_executed > 0 else 0.5
        ]

    def evaluate(
        self,
        trace: ExecutionTrace,
        criteria: List[EvaluationCriteria]
    ) -> EvaluationResult:
        scores = {}

        for criterion in criteria:
            dim = criterion.dimension
            if dim in self.rules:
                rule_scores = [rule(trace) for rule in self.rules[dim]]
                scores[dim] = sum(rule_scores) / len(rule_scores)
            else:
                # Custom evaluator or default
                if criterion.custom_evaluator:
                    scores[dim] = criterion.custom_evaluator(trace)
                else:
                    scores[dim] = 0.5  # Neutral score

        # Calculate weighted overall score
        total_weight = sum(c.weight for c in criteria)
        overall_score = sum(
            scores.get(c.dimension, 0.5) * c.weight / total_weight
            for c in criteria
        )

        # Generate feedback
        strengths = []
        weaknesses = []
        suggestions = []

        for dim, score in scores.items():
            if score >= 0.8:
                strengths.append(f"Strong {dim.value}: {score:.2f}")
            elif score <= 0.4:
                weaknesses.append(f"Weak {dim.value}: {score:.2f}")
                suggestions.append(f"Improve {dim.value}")

        # Determine if training-worthy
        training_worthy = overall_score >= 0.7 and trace.success

        return EvaluationResult(
            evaluation_id=str(uuid.uuid4())[:12],
            timestamp=datetime.now(),
            agent_id=trace.agent_id,
            level=EvaluationLevel.TASK,
            target_id=trace.trace_id,
            scores=scores,
            overall_score=overall_score,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            training_worthy=training_worthy,
            training_category="success" if trace.success else "failure",
            evaluator=self.get_name(),
            confidence=0.7  # Rule-based has moderate confidence
        )

    def get_name(self) -> str:
        return "rule_based"


class LLMEvaluator(Evaluator):
    """LLM-based evaluation using model self-critique"""

    def __init__(self, llm_backend=None):
        self.llm = llm_backend

    def evaluate(
        self,
        trace: ExecutionTrace,
        criteria: List[EvaluationCriteria]
    ) -> EvaluationResult:
        """Use LLM to evaluate the trace"""
        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(trace, criteria)

        if self.llm:
            # Get LLM evaluation
            response = self.llm.generate([
                {"role": "system", "content": "You are an expert evaluator of AI agent performance."},
                {"role": "user", "content": prompt}
            ])
            return self._parse_llm_evaluation(response, trace, criteria)
        else:
            # Fallback to simulated evaluation
            return self._simulated_evaluation(trace, criteria)

    def _build_evaluation_prompt(
        self,
        trace: ExecutionTrace,
        criteria: List[EvaluationCriteria]
    ) -> str:
        criteria_list = "\n".join([
            f"- {c.dimension.value}: {c.description or c.dimension.value}"
            for c in criteria
        ])

        return f"""Evaluate this agent's execution:

TASK: {trace.original_task}

EXECUTION SUMMARY:
- Turns: {trace.total_turns}
- Commands: {trace.commands_executed}
- Completed: {trace.completed}
- Success: {trace.success}
- Error: {trace.error or 'None'}

FINAL OUTPUT (truncated):
{trace.final_output[:1000] if trace.final_output else 'None'}

EVALUATION CRITERIA:
{criteria_list}

Please evaluate each criterion on a scale of 0.0 to 1.0 and provide:
1. Scores for each criterion
2. Overall score (weighted average)
3. Top 3 strengths
4. Top 3 weaknesses
5. Top 3 suggestions for improvement
6. Is this execution worthy of being training data? (yes/no)

Format your response as JSON:
{{
  "scores": {{"criterion": score, ...}},
  "overall_score": float,
  "strengths": ["...", "...", "..."],
  "weaknesses": ["...", "...", "..."],
  "suggestions": ["...", "...", "..."],
  "training_worthy": true/false,
  "confidence": float
}}"""

    def _parse_llm_evaluation(
        self,
        response: Dict[str, Any],
        trace: ExecutionTrace,
        criteria: List[EvaluationCriteria]
    ) -> EvaluationResult:
        """Parse LLM response into evaluation result"""
        content = response.get("content", "")

        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found")

            # Parse scores
            scores = {}
            for c in criteria:
                dim_name = c.dimension.value
                if dim_name in data.get("scores", {}):
                    scores[c.dimension] = float(data["scores"][dim_name])
                else:
                    scores[c.dimension] = 0.5

            return EvaluationResult(
                evaluation_id=str(uuid.uuid4())[:12],
                timestamp=datetime.now(),
                agent_id=trace.agent_id,
                level=EvaluationLevel.TASK,
                target_id=trace.trace_id,
                scores=scores,
                overall_score=float(data.get("overall_score", 0.5)),
                strengths=data.get("strengths", []),
                weaknesses=data.get("weaknesses", []),
                suggestions=data.get("suggestions", []),
                training_worthy=data.get("training_worthy", False),
                training_category="llm_evaluated",
                evaluator=self.get_name(),
                confidence=float(data.get("confidence", 0.8))
            )
        except Exception as e:
            # Fallback if parsing fails
            return self._simulated_evaluation(trace, criteria)

    def _simulated_evaluation(
        self,
        trace: ExecutionTrace,
        criteria: List[EvaluationCriteria]
    ) -> EvaluationResult:
        """Simulated evaluation when LLM unavailable"""
        scores = {}
        for c in criteria:
            # Simple heuristic scoring
            if c.dimension == EvaluationDimension.TASK_SUCCESS:
                scores[c.dimension] = 1.0 if trace.success else 0.0
            elif c.dimension == EvaluationDimension.EFFICIENCY:
                scores[c.dimension] = min(1.0, 5.0 / max(1, trace.total_turns))
            else:
                scores[c.dimension] = 0.6 if trace.completed else 0.4

        overall = sum(scores.values()) / max(1, len(scores))

        return EvaluationResult(
            evaluation_id=str(uuid.uuid4())[:12],
            timestamp=datetime.now(),
            agent_id=trace.agent_id,
            level=EvaluationLevel.TASK,
            target_id=trace.trace_id,
            scores=scores,
            overall_score=overall,
            strengths=["Task attempted"] if trace.completed else [],
            weaknesses=["Evaluation simulated"],
            suggestions=["Enable LLM evaluation for better insights"],
            training_worthy=overall >= 0.7 and trace.success,
            training_category="simulated",
            evaluator=self.get_name(),
            confidence=0.5
        )

    def get_name(self) -> str:
        return "llm_evaluator"


class PeerEvaluator(Evaluator):
    """Evaluation by peer agents"""

    def __init__(self, message_bus, agent_directory):
        self.message_bus = message_bus
        self.agent_directory = agent_directory

    def evaluate(
        self,
        trace: ExecutionTrace,
        criteria: List[EvaluationCriteria]
    ) -> EvaluationResult:
        """Request evaluation from peer agents"""
        # Find evaluator agents
        evaluators = self.agent_directory.find_by_capability("evaluation")

        if not evaluators:
            # Fallback to rule-based
            return RuleBasedEvaluator().evaluate(trace, criteria)

        # TODO: Implement peer evaluation protocol
        # For now, use rule-based
        return RuleBasedEvaluator().evaluate(trace, criteria)

    def get_name(self) -> str:
        return "peer_evaluator"


class SelfEvaluationLoop:
    """
    The main self-evaluation loop that drives continuous improvement.

    This implements a flywheel effect:
    1. Execute tasks
    2. Evaluate performance
    3. Extract lessons
    4. Apply lessons to future tasks
    5. Generate training data
    6. Repeat
    """

    def __init__(
        self,
        base_path: str = "/tmp/self-eval",
        llm_backend=None
    ):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Initialize evaluators
        self.evaluators: Dict[str, Evaluator] = {
            "rule_based": RuleBasedEvaluator(),
            "llm": LLMEvaluator(llm_backend)
        }

        # Default criteria
        self.default_criteria = [
            EvaluationCriteria(EvaluationDimension.TASK_SUCCESS, weight=2.0),
            EvaluationCriteria(EvaluationDimension.TASK_COMPLETENESS, weight=1.5),
            EvaluationCriteria(EvaluationDimension.EFFICIENCY, weight=1.0),
            EvaluationCriteria(EvaluationDimension.TOOL_USAGE, weight=1.0),
            EvaluationCriteria(EvaluationDimension.REASONING_QUALITY, weight=1.5),
            EvaluationCriteria(EvaluationDimension.SAFETY, weight=2.0),
        ]

        # Storage
        self.traces: Dict[str, ExecutionTrace] = {}
        self.evaluations: Dict[str, EvaluationResult] = {}
        self.lessons: Dict[str, Lesson] = {}

        # Lesson index by category
        self.lessons_by_category: Dict[str, List[str]] = {}
        self.lessons_by_task_type: Dict[str, List[str]] = {}

        # Performance tracking
        self.agent_performance: Dict[str, Dict[str, Any]] = {}

        # Persistence
        self.db_path = self.base_path / "evaluations.db"
        self._init_db()

        self._lock = threading.Lock()

    def _init_db(self):
        """Initialize SQLite database"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    agent_id TEXT,
                    session_id TEXT,
                    task TEXT,
                    task_type TEXT,
                    success INTEGER,
                    overall_score REAL,
                    created_at TEXT,
                    data TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    evaluation_id TEXT PRIMARY KEY,
                    trace_id TEXT,
                    agent_id TEXT,
                    overall_score REAL,
                    training_worthy INTEGER,
                    evaluator TEXT,
                    created_at TEXT,
                    data TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS lessons (
                    lesson_id TEXT PRIMARY KEY,
                    lesson_type TEXT,
                    category TEXT,
                    description TEXT,
                    confidence REAL,
                    times_applied INTEGER,
                    success_rate REAL,
                    created_at TEXT,
                    data TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_traces_agent ON traces(agent_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_evaluations_agent ON evaluations(agent_id)
            """)
            conn.commit()

    def record_trace(self, trace: ExecutionTrace) -> str:
        """Record an execution trace for evaluation"""
        with self._lock:
            self.traces[trace.trace_id] = trace

            # Persist to database
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO traces
                    (trace_id, agent_id, session_id, task, task_type, success, overall_score, created_at, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trace.trace_id,
                    trace.agent_id,
                    trace.session_id,
                    trace.original_task,
                    trace.task_type,
                    1 if trace.success else 0,
                    0.0,  # Updated after evaluation
                    trace.started_at.isoformat(),
                    json.dumps(trace.to_dict())
                ))
                conn.commit()

            return trace.trace_id

    def evaluate_trace(
        self,
        trace_id: str,
        evaluator_name: str = "rule_based",
        criteria: List[EvaluationCriteria] = None
    ) -> EvaluationResult:
        """Evaluate a recorded trace"""
        trace = self.traces.get(trace_id)
        if not trace:
            raise ValueError(f"Trace not found: {trace_id}")

        evaluator = self.evaluators.get(evaluator_name)
        if not evaluator:
            evaluator = self.evaluators["rule_based"]

        criteria = criteria or self.default_criteria
        result = evaluator.evaluate(trace, criteria)

        with self._lock:
            self.evaluations[result.evaluation_id] = result

            # Update trace score
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(
                    "UPDATE traces SET overall_score = ? WHERE trace_id = ?",
                    (result.overall_score, trace_id)
                )
                conn.execute("""
                    INSERT OR REPLACE INTO evaluations
                    (evaluation_id, trace_id, agent_id, overall_score, training_worthy, evaluator, created_at, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.evaluation_id,
                    trace_id,
                    result.agent_id,
                    result.overall_score,
                    1 if result.training_worthy else 0,
                    result.evaluator,
                    result.timestamp.isoformat(),
                    json.dumps(result.to_dict())
                ))
                conn.commit()

            # Update agent performance tracking
            self._update_performance(result)

        return result

    def _update_performance(self, result: EvaluationResult):
        """Update agent performance tracking"""
        agent_id = result.agent_id
        if agent_id not in self.agent_performance:
            self.agent_performance[agent_id] = {
                "total_evaluations": 0,
                "average_score": 0.0,
                "score_history": [],
                "dimension_averages": {},
                "improvement_trend": 0.0,
                "training_worthy_count": 0
            }

        perf = self.agent_performance[agent_id]
        perf["total_evaluations"] += 1
        perf["score_history"].append(result.overall_score)

        if result.training_worthy:
            perf["training_worthy_count"] += 1

        # Update running average
        n = perf["total_evaluations"]
        perf["average_score"] = (
            (perf["average_score"] * (n - 1) + result.overall_score) / n
        )

        # Update dimension averages
        for dim, score in result.scores.items():
            dim_key = dim.value
            if dim_key not in perf["dimension_averages"]:
                perf["dimension_averages"][dim_key] = {"sum": 0, "count": 0}
            perf["dimension_averages"][dim_key]["sum"] += score
            perf["dimension_averages"][dim_key]["count"] += 1

        # Calculate improvement trend (last 10 vs previous 10)
        history = perf["score_history"]
        if len(history) >= 20:
            recent = sum(history[-10:]) / 10
            previous = sum(history[-20:-10]) / 10
            perf["improvement_trend"] = recent - previous

    def extract_lesson(
        self,
        evaluation_id: str,
        lesson_type: str = "auto"
    ) -> Optional[Lesson]:
        """Extract a lesson from an evaluation"""
        result = self.evaluations.get(evaluation_id)
        if not result:
            return None

        trace = self.traces.get(result.target_id)
        if not trace:
            return None

        # Determine lesson type
        if lesson_type == "auto":
            if result.overall_score >= 0.8:
                lesson_type = "success_pattern"
            elif result.overall_score <= 0.4:
                lesson_type = "failure_pattern"
            else:
                lesson_type = "optimization"

        # Build lesson from evaluation
        lesson = Lesson(
            lesson_id=str(uuid.uuid4())[:12],
            created_at=datetime.now(),
            source_trace_id=trace.trace_id,
            source_evaluation_id=evaluation_id,
            lesson_type=lesson_type,
            category=trace.task_type or "general",
            description=self._generate_lesson_description(result, trace),
            applies_when=[trace.task_type] if trace.task_type else ["general"],
            task_types=[trace.task_type] if trace.task_type else [],
            do_this=result.strengths if lesson_type == "success_pattern" else result.suggestions,
            avoid_this=result.weaknesses if lesson_type == "failure_pattern" else [],
            confidence=result.confidence * result.overall_score
        )

        with self._lock:
            self.lessons[lesson.lesson_id] = lesson

            # Index by category
            if lesson.category not in self.lessons_by_category:
                self.lessons_by_category[lesson.category] = []
            self.lessons_by_category[lesson.category].append(lesson.lesson_id)

            # Index by task type
            for task_type in lesson.task_types:
                if task_type not in self.lessons_by_task_type:
                    self.lessons_by_task_type[task_type] = []
                self.lessons_by_task_type[task_type].append(lesson.lesson_id)

            # Persist
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO lessons
                    (lesson_id, lesson_type, category, description, confidence, times_applied, success_rate, created_at, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    lesson.lesson_id,
                    lesson.lesson_type,
                    lesson.category,
                    lesson.description,
                    lesson.confidence,
                    lesson.times_applied,
                    lesson.success_rate,
                    lesson.created_at.isoformat(),
                    json.dumps(lesson.to_dict())
                ))
                conn.commit()

        return lesson

    def _generate_lesson_description(
        self,
        result: EvaluationResult,
        trace: ExecutionTrace
    ) -> str:
        """Generate a lesson description from evaluation"""
        if result.overall_score >= 0.8:
            return f"Successful approach to {trace.task_type or 'task'}: {trace.original_task[:100]}"
        elif result.overall_score <= 0.4:
            return f"Failed approach to {trace.task_type or 'task'}: Avoid {'; '.join(result.weaknesses[:2])}"
        else:
            return f"Partial success: {'; '.join(result.suggestions[:2])}"

    def get_relevant_lessons(
        self,
        task_type: str = None,
        category: str = None,
        limit: int = 10
    ) -> List[Lesson]:
        """Get lessons relevant to a task"""
        lesson_ids = set()

        if task_type and task_type in self.lessons_by_task_type:
            lesson_ids.update(self.lessons_by_task_type[task_type])

        if category and category in self.lessons_by_category:
            lesson_ids.update(self.lessons_by_category[category])

        if not lesson_ids:
            # Return top lessons by confidence
            lessons = sorted(
                self.lessons.values(),
                key=lambda l: l.confidence * (l.success_rate + 0.1),
                reverse=True
            )
            return lessons[:limit]

        # Return matched lessons sorted by relevance
        lessons = [self.lessons[lid] for lid in lesson_ids if lid in self.lessons]
        lessons.sort(key=lambda l: l.confidence * (l.success_rate + 0.1), reverse=True)
        return lessons[:limit]

    def apply_lesson(self, lesson_id: str, success: bool):
        """Record that a lesson was applied with outcome"""
        if lesson_id in self.lessons:
            lesson = self.lessons[lesson_id]
            lesson.times_applied += 1

            # Update success rate with exponential moving average
            alpha = 0.3
            lesson.success_rate = (
                alpha * (1.0 if success else 0.0) +
                (1 - alpha) * lesson.success_rate
            )

    def get_flywheel_prompt(
        self,
        task: str,
        task_type: str = None,
        agent_id: str = None
    ) -> str:
        """
        Generate a prompt addendum with learned lessons.
        This is the flywheel effect - past learnings improve future performance.
        """
        sections = []

        # Get relevant lessons
        lessons = self.get_relevant_lessons(task_type=task_type, limit=5)

        if lessons:
            success_lessons = [l for l in lessons if l.lesson_type == "success_pattern"]
            failure_lessons = [l for l in lessons if l.lesson_type == "failure_pattern"]

            if success_lessons:
                sections.append("# Learned Success Patterns")
                for l in success_lessons[:3]:
                    sections.append(f"- {l.description}")
                    if l.do_this:
                        sections.append(f"  DO: {'; '.join(l.do_this[:2])}")

            if failure_lessons:
                sections.append("\n# Known Failure Patterns to Avoid")
                for l in failure_lessons[:3]:
                    sections.append(f"- {l.description}")
                    if l.avoid_this:
                        sections.append(f"  AVOID: {'; '.join(l.avoid_this[:2])}")

        # Add agent-specific performance context
        if agent_id and agent_id in self.agent_performance:
            perf = self.agent_performance[agent_id]
            sections.append(f"\n# Your Performance Context")
            sections.append(f"- Average score: {perf['average_score']:.2f}")
            sections.append(f"- Improvement trend: {perf['improvement_trend']:+.2f}")

            # Find weakest dimensions
            weak_dims = sorted(
                perf["dimension_averages"].items(),
                key=lambda x: x[1]["sum"] / max(1, x[1]["count"])
            )[:2]
            if weak_dims:
                sections.append(f"- Focus areas: {', '.join(d[0] for d in weak_dims)}")

        return "\n".join(sections) if sections else ""

    def get_agent_report(self, agent_id: str) -> Dict[str, Any]:
        """Get performance report for an agent"""
        if agent_id not in self.agent_performance:
            return {"error": "No performance data"}

        perf = self.agent_performance[agent_id]

        # Calculate dimension averages
        dim_avgs = {
            k: v["sum"] / max(1, v["count"])
            for k, v in perf["dimension_averages"].items()
        }

        return {
            "agent_id": agent_id,
            "total_evaluations": perf["total_evaluations"],
            "average_score": perf["average_score"],
            "improvement_trend": perf["improvement_trend"],
            "training_worthy_ratio": perf["training_worthy_count"] / max(1, perf["total_evaluations"]),
            "dimension_averages": dim_avgs,
            "strongest_dimensions": sorted(dim_avgs.items(), key=lambda x: x[1], reverse=True)[:3],
            "weakest_dimensions": sorted(dim_avgs.items(), key=lambda x: x[1])[:3],
            "recent_scores": perf["score_history"][-10:]
        }

    def get_training_data(
        self,
        min_score: float = 0.7,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get training-worthy traces for model improvement"""
        training_data = []

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("""
                SELECT t.data, e.data
                FROM traces t
                JOIN evaluations e ON t.trace_id = e.trace_id
                WHERE e.training_worthy = 1 AND e.overall_score >= ?
                ORDER BY e.overall_score DESC
                LIMIT ?
            """, (min_score, limit))

            for trace_json, eval_json in cursor.fetchall():
                try:
                    trace_data = json.loads(trace_json)
                    eval_data = json.loads(eval_json)
                    training_data.append({
                        "trace": trace_data,
                        "evaluation": eval_data,
                        "type": "successful_execution"
                    })
                except json.JSONDecodeError:
                    continue

        return training_data


class FlywheelManager:
    """
    Manages the flywheel effect across multiple agents.

    The flywheel:
    1. Agents execute tasks
    2. Self-evaluation scores performance
    3. High-scoring traces become training data
    4. Lessons are extracted from all evaluations
    5. Future tasks benefit from lessons
    6. Better performance leads to better training data
    7. Cycle continues with compounding improvements
    """

    def __init__(
        self,
        eval_loop: SelfEvaluationLoop,
        auto_evaluate: bool = True,
        auto_extract_lessons: bool = True,
        min_score_for_lesson: float = 0.3  # Extract lessons from all (learn from failures too)
    ):
        self.eval_loop = eval_loop
        self.auto_evaluate = auto_evaluate
        self.auto_extract_lessons = auto_extract_lessons
        self.min_score_for_lesson = min_score_for_lesson

        # Flywheel metrics
        self.metrics = {
            "traces_recorded": 0,
            "evaluations_performed": 0,
            "lessons_extracted": 0,
            "lessons_applied": 0,
            "training_data_generated": 0,
            "average_score_trend": []
        }

    def process_execution(
        self,
        agent_id: str,
        session_id: str,
        task: str,
        task_type: str,
        turns: List[Dict[str, Any]],
        success: bool,
        final_output: str,
        error: Optional[str] = None,
        tokens_used: int = 0,
        time_taken_ms: float = 0
    ) -> Dict[str, Any]:
        """
        Process a complete execution through the flywheel.

        Returns evaluation result and any extracted lessons.
        """
        # Create execution trace
        trace = ExecutionTrace(
            trace_id=str(uuid.uuid4())[:12],
            agent_id=agent_id,
            session_id=session_id,
            original_task=task,
            task_type=task_type,
            constraints=[],
            turns=turns,
            total_turns=len(turns),
            tokens_used=tokens_used,
            time_taken_ms=time_taken_ms,
            commands_executed=sum(1 for t in turns if t.get("commands")),
            skills_invoked=sum(1 for t in turns if t.get("skills")),
            completed=True,
            success=success,
            final_output=final_output,
            error=error,
            started_at=datetime.now() - timedelta(milliseconds=time_taken_ms),
            completed_at=datetime.now()
        )

        # Record trace
        self.eval_loop.record_trace(trace)
        self.metrics["traces_recorded"] += 1

        result = {
            "trace_id": trace.trace_id,
            "evaluation": None,
            "lessons": []
        }

        # Auto-evaluate if enabled
        if self.auto_evaluate:
            evaluation = self.eval_loop.evaluate_trace(trace.trace_id)
            self.metrics["evaluations_performed"] += 1
            result["evaluation"] = evaluation.to_dict()

            # Track score trend
            self.metrics["average_score_trend"].append(evaluation.overall_score)
            if len(self.metrics["average_score_trend"]) > 100:
                self.metrics["average_score_trend"] = self.metrics["average_score_trend"][-100:]

            # Track training data
            if evaluation.training_worthy:
                self.metrics["training_data_generated"] += 1

            # Auto-extract lessons if enabled
            if self.auto_extract_lessons:
                if (evaluation.overall_score >= 0.7 or  # Success pattern
                    evaluation.overall_score <= self.min_score_for_lesson):  # Failure pattern
                    lesson = self.eval_loop.extract_lesson(evaluation.evaluation_id)
                    if lesson:
                        self.metrics["lessons_extracted"] += 1
                        result["lessons"].append(lesson.to_dict())

        return result

    def get_enhanced_prompt(
        self,
        task: str,
        task_type: str = None,
        agent_id: str = None
    ) -> str:
        """
        Get an enhanced prompt that includes flywheel learnings.
        This is how the flywheel feeds back into performance.
        """
        return self.eval_loop.get_flywheel_prompt(task, task_type, agent_id)

    def record_lesson_application(self, lesson_id: str, success: bool):
        """Record that a lesson was applied"""
        self.eval_loop.apply_lesson(lesson_id, success)
        self.metrics["lessons_applied"] += 1

    def get_flywheel_status(self) -> Dict[str, Any]:
        """Get current flywheel status and health"""
        avg_score = 0.0
        trend = self.metrics["average_score_trend"]
        if trend:
            avg_score = sum(trend) / len(trend)

            # Calculate trend direction
            if len(trend) >= 20:
                recent = sum(trend[-10:]) / 10
                previous = sum(trend[-20:-10]) / 10
                trend_direction = "improving" if recent > previous else "declining" if recent < previous else "stable"
            else:
                trend_direction = "insufficient_data"
        else:
            trend_direction = "no_data"

        return {
            "metrics": self.metrics,
            "average_score": avg_score,
            "trend_direction": trend_direction,
            "total_lessons": len(self.eval_loop.lessons),
            "training_data_ratio": (
                self.metrics["training_data_generated"] /
                max(1, self.metrics["evaluations_performed"])
            ),
            "health": "healthy" if avg_score >= 0.6 else "needs_attention"
        }


# Convenience function for creating evaluation system
def create_evaluation_system(
    base_path: str = "/tmp/eval-system",
    llm_backend=None
) -> Tuple[SelfEvaluationLoop, FlywheelManager]:
    """Create and configure evaluation system"""
    eval_loop = SelfEvaluationLoop(base_path, llm_backend)
    flywheel = FlywheelManager(eval_loop)
    return eval_loop, flywheel
