"""
Q&A Generation System

This module generates question-answer pairs from agent execution traces,
creating training data for model improvement and the flywheel effect.

Supported Q&A formats:
1. Instruction-following (task -> response)
2. Multi-turn conversation
3. Reasoning chains (with thought process)
4. Tool use examples
5. Error recovery patterns

The generated Q&A pairs can be used for:
- Model fine-tuning
- Few-shot prompting
- Evaluation benchmarks
- Documentation generation
"""

import json
import uuid
import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Tuple, Union, Generator
from datetime import datetime
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod
import sqlite3
import threading


class QAFormat(Enum):
    """Format types for generated Q&A pairs"""
    INSTRUCTION = "instruction"         # Simple instruction -> response
    CONVERSATION = "conversation"       # Multi-turn dialogue
    REASONING = "reasoning"             # Shows reasoning steps
    TOOL_USE = "tool_use"               # Tool calling examples
    ERROR_RECOVERY = "error_recovery"   # Error handling patterns
    CODE_GENERATION = "code_generation" # Code writing examples
    CODE_REVIEW = "code_review"         # Code analysis examples
    DEBUGGING = "debugging"             # Bug fixing examples
    PLANNING = "planning"               # Task planning examples
    COLLABORATION = "collaboration"     # Multi-agent interaction


class QAQuality(Enum):
    """Quality levels for Q&A pairs"""
    GOLD = "gold"           # High quality, manually verified
    SILVER = "silver"       # Good quality, auto-verified
    BRONZE = "bronze"       # Acceptable quality
    UNVERIFIED = "unverified"


@dataclass
class QAPair:
    """A single question-answer pair"""
    qa_id: str
    created_at: datetime

    # Source information
    source_trace_id: Optional[str]
    source_evaluation_id: Optional[str]

    # Format
    format: QAFormat
    quality: QAQuality

    # Content
    system_prompt: Optional[str]      # System context if needed
    question: str                      # The input/question
    answer: str                        # The response/answer

    # For multi-turn
    conversation: List[Dict[str, str]] = field(default_factory=list)

    # Metadata
    tags: List[str] = field(default_factory=list)
    task_type: Optional[str] = None
    model_used: Optional[str] = None
    score: float = 0.0                 # Quality score (0-1)

    # For tool use format
    tools_used: List[Dict[str, Any]] = field(default_factory=list)

    # Training metadata
    token_count: int = 0
    difficulty: str = "medium"         # easy, medium, hard
    domain: str = "general"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qa_id": self.qa_id,
            "created_at": self.created_at.isoformat(),
            "source_trace_id": self.source_trace_id,
            "format": self.format.value,
            "quality": self.quality.value,
            "system_prompt": self.system_prompt,
            "question": self.question,
            "answer": self.answer,
            "conversation": self.conversation,
            "tags": self.tags,
            "task_type": self.task_type,
            "score": self.score,
            "tools_used": self.tools_used,
            "token_count": self.token_count,
            "difficulty": self.difficulty,
            "domain": self.domain
        }

    def to_training_format(self, style: str = "alpaca") -> Dict[str, Any]:
        """Convert to standard training format"""
        if style == "alpaca":
            return {
                "instruction": self.question,
                "input": "",
                "output": self.answer
            }
        elif style == "sharegpt":
            if self.conversation:
                return {
                    "conversations": self.conversation
                }
            return {
                "conversations": [
                    {"from": "human", "value": self.question},
                    {"from": "gpt", "value": self.answer}
                ]
            }
        elif style == "openai":
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            if self.conversation:
                messages.extend(self.conversation)
            else:
                messages.append({"role": "user", "content": self.question})
                messages.append({"role": "assistant", "content": self.answer})
            return {"messages": messages}
        elif style == "chatml":
            parts = []
            if self.system_prompt:
                parts.append(f"<|im_start|>system\n{self.system_prompt}<|im_end|>")
            parts.append(f"<|im_start|>user\n{self.question}<|im_end|>")
            parts.append(f"<|im_start|>assistant\n{self.answer}<|im_end|>")
            return {"text": "\n".join(parts)}
        else:
            return self.to_dict()


@dataclass
class QAGenerationConfig:
    """Configuration for Q&A generation"""
    formats: List[QAFormat] = field(default_factory=lambda: [QAFormat.INSTRUCTION])
    min_score: float = 0.6              # Minimum evaluation score
    include_failures: bool = True       # Include failure examples
    include_reasoning: bool = True      # Include reasoning traces
    max_turns_per_conversation: int = 10
    max_token_length: int = 4096
    dedup_similarity_threshold: float = 0.9
    augment_questions: bool = True      # Generate question variations


class QAGenerator(ABC):
    """Abstract base class for Q&A generators"""

    @abstractmethod
    def generate(
        self,
        trace_data: Dict[str, Any],
        evaluation_data: Optional[Dict[str, Any]] = None,
        config: QAGenerationConfig = None
    ) -> List[QAPair]:
        """Generate Q&A pairs from execution data"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get generator name"""
        pass


class InstructionQAGenerator(QAGenerator):
    """Generates instruction-following Q&A pairs"""

    def generate(
        self,
        trace_data: Dict[str, Any],
        evaluation_data: Optional[Dict[str, Any]] = None,
        config: QAGenerationConfig = None
    ) -> List[QAPair]:
        config = config or QAGenerationConfig()
        pairs = []

        task = trace_data.get("original_task", "")
        final_output = trace_data.get("final_output", "")
        success = trace_data.get("success", False)

        if not task or not final_output:
            return pairs

        # Only include successful executions for instruction format
        # (or failures if configured)
        score = evaluation_data.get("overall_score", 0.5) if evaluation_data else 0.5
        if not success and not config.include_failures:
            return pairs

        if score < config.min_score:
            return pairs

        # Create basic instruction pair
        pair = QAPair(
            qa_id=str(uuid.uuid4())[:12],
            created_at=datetime.now(),
            source_trace_id=trace_data.get("trace_id"),
            source_evaluation_id=evaluation_data.get("evaluation_id") if evaluation_data else None,
            format=QAFormat.INSTRUCTION,
            quality=self._determine_quality(score, success),
            system_prompt=None,
            question=task,
            answer=final_output,
            tags=["instruction", trace_data.get("task_type", "general")],
            task_type=trace_data.get("task_type"),
            score=score,
            difficulty=self._determine_difficulty(trace_data)
        )
        pairs.append(pair)

        # Generate question variations if configured
        if config.augment_questions:
            variations = self._generate_variations(task)
            for var in variations[:2]:  # Limit variations
                var_pair = QAPair(
                    qa_id=str(uuid.uuid4())[:12],
                    created_at=datetime.now(),
                    source_trace_id=trace_data.get("trace_id"),
                    source_evaluation_id=evaluation_data.get("evaluation_id") if evaluation_data else None,
                    format=QAFormat.INSTRUCTION,
                    quality=QAQuality.BRONZE,  # Variations are lower quality
                    system_prompt=None,
                    question=var,
                    answer=final_output,
                    tags=["instruction", "variation"],
                    task_type=trace_data.get("task_type"),
                    score=score * 0.9
                )
                pairs.append(var_pair)

        return pairs

    def _determine_quality(self, score: float, success: bool) -> QAQuality:
        if score >= 0.9 and success:
            return QAQuality.GOLD
        elif score >= 0.7 and success:
            return QAQuality.SILVER
        elif score >= 0.5:
            return QAQuality.BRONZE
        return QAQuality.UNVERIFIED

    def _determine_difficulty(self, trace_data: Dict[str, Any]) -> str:
        turns = trace_data.get("total_turns", 1)
        commands = trace_data.get("commands_executed", 0)

        if turns <= 2 and commands <= 3:
            return "easy"
        elif turns <= 5 and commands <= 10:
            return "medium"
        return "hard"

    def _generate_variations(self, question: str) -> List[str]:
        """Generate variations of a question"""
        variations = []

        # Add politeness variations
        if not question.lower().startswith("please"):
            variations.append(f"Please {question.lower()}")

        # Add imperative variations
        if question.lower().startswith("can you"):
            variations.append(question.replace("Can you ", "").replace("can you ", ""))

        # Add context request
        variations.append(f"I need help with this: {question}")

        return variations

    def get_name(self) -> str:
        return "instruction_generator"


class ConversationQAGenerator(QAGenerator):
    """Generates multi-turn conversation Q&A pairs"""

    def generate(
        self,
        trace_data: Dict[str, Any],
        evaluation_data: Optional[Dict[str, Any]] = None,
        config: QAGenerationConfig = None
    ) -> List[QAPair]:
        config = config or QAGenerationConfig()
        pairs = []

        turns = trace_data.get("turns", [])
        if len(turns) < 2:
            return pairs

        score = evaluation_data.get("overall_score", 0.5) if evaluation_data else 0.5
        if score < config.min_score:
            return pairs

        # Build conversation from turns
        conversation = []
        for i, turn in enumerate(turns[:config.max_turns_per_conversation]):
            user_input = turn.get("input", "")
            assistant_output = turn.get("output", "")

            if user_input:
                conversation.append({"role": "user", "content": user_input})
            if assistant_output:
                conversation.append({"role": "assistant", "content": assistant_output})

        if len(conversation) < 2:
            return pairs

        # Create conversation pair
        pair = QAPair(
            qa_id=str(uuid.uuid4())[:12],
            created_at=datetime.now(),
            source_trace_id=trace_data.get("trace_id"),
            source_evaluation_id=evaluation_data.get("evaluation_id") if evaluation_data else None,
            format=QAFormat.CONVERSATION,
            quality=self._determine_quality(score),
            system_prompt=self._extract_system_prompt(trace_data),
            question=conversation[0].get("content", "") if conversation else "",
            answer=conversation[-1].get("content", "") if conversation else "",
            conversation=conversation,
            tags=["conversation", "multi-turn"],
            task_type=trace_data.get("task_type"),
            score=score
        )
        pairs.append(pair)

        return pairs

    def _determine_quality(self, score: float) -> QAQuality:
        if score >= 0.85:
            return QAQuality.GOLD
        elif score >= 0.65:
            return QAQuality.SILVER
        return QAQuality.BRONZE

    def _extract_system_prompt(self, trace_data: Dict[str, Any]) -> Optional[str]:
        """Extract system prompt if available"""
        turns = trace_data.get("turns", [])
        if turns and turns[0].get("system"):
            return turns[0]["system"]
        return None

    def get_name(self) -> str:
        return "conversation_generator"


class ReasoningQAGenerator(QAGenerator):
    """Generates reasoning chain Q&A pairs with thought process"""

    def generate(
        self,
        trace_data: Dict[str, Any],
        evaluation_data: Optional[Dict[str, Any]] = None,
        config: QAGenerationConfig = None
    ) -> List[QAPair]:
        config = config or QAGenerationConfig()
        pairs = []

        task = trace_data.get("original_task", "")
        turns = trace_data.get("turns", [])
        final_output = trace_data.get("final_output", "")

        if not task or not turns:
            return pairs

        score = evaluation_data.get("overall_score", 0.5) if evaluation_data else 0.5
        if score < config.min_score:
            return pairs

        # Build reasoning chain
        reasoning_steps = []
        for i, turn in enumerate(turns):
            output = turn.get("output", "")
            commands = turn.get("commands", [])

            step = f"Step {i + 1}:"
            if output:
                step += f"\nThought: {output[:500]}"
            if commands:
                cmd_summary = ", ".join([c.get("type", "unknown") for c in commands[:3]])
                step += f"\nActions: {cmd_summary}"

            reasoning_steps.append(step)

        # Create reasoning answer
        reasoning_answer = "Let me think through this step by step:\n\n"
        reasoning_answer += "\n\n".join(reasoning_steps[:10])
        reasoning_answer += f"\n\nFinal Answer:\n{final_output}"

        pair = QAPair(
            qa_id=str(uuid.uuid4())[:12],
            created_at=datetime.now(),
            source_trace_id=trace_data.get("trace_id"),
            source_evaluation_id=evaluation_data.get("evaluation_id") if evaluation_data else None,
            format=QAFormat.REASONING,
            quality=self._determine_quality(score, len(turns)),
            system_prompt="You are a helpful assistant that thinks step by step.",
            question=task,
            answer=reasoning_answer,
            tags=["reasoning", "chain-of-thought"],
            task_type=trace_data.get("task_type"),
            score=score,
            difficulty="hard"  # Reasoning is always complex
        )
        pairs.append(pair)

        return pairs

    def _determine_quality(self, score: float, num_steps: int) -> QAQuality:
        if score >= 0.8 and num_steps >= 3:
            return QAQuality.GOLD
        elif score >= 0.6 and num_steps >= 2:
            return QAQuality.SILVER
        return QAQuality.BRONZE

    def get_name(self) -> str:
        return "reasoning_generator"


class ToolUseQAGenerator(QAGenerator):
    """Generates tool use Q&A pairs"""

    def generate(
        self,
        trace_data: Dict[str, Any],
        evaluation_data: Optional[Dict[str, Any]] = None,
        config: QAGenerationConfig = None
    ) -> List[QAPair]:
        config = config or QAGenerationConfig()
        pairs = []

        task = trace_data.get("original_task", "")
        turns = trace_data.get("turns", [])
        commands_executed = trace_data.get("commands_executed", 0)

        if not task or commands_executed == 0:
            return pairs

        score = evaluation_data.get("overall_score", 0.5) if evaluation_data else 0.5
        if score < config.min_score:
            return pairs

        # Extract tool calls
        tools_used = []
        for turn in turns:
            for cmd in turn.get("commands", []):
                tools_used.append({
                    "name": cmd.get("type", "unknown"),
                    "arguments": cmd.get("params", {}),
                    "result": cmd.get("result", {})
                })

        if not tools_used:
            return pairs

        # Build tool use response
        tool_response = "I'll help you with that using the following tools:\n\n"
        for i, tool in enumerate(tools_used[:5]):
            tool_response += f"{i + 1}. **{tool['name']}**\n"
            tool_response += f"   Arguments: {json.dumps(tool['arguments'], indent=2)[:200]}\n"
            if tool.get("result"):
                tool_response += f"   Result: {str(tool['result'])[:200]}\n"
            tool_response += "\n"

        tool_response += f"\nBased on the tool results: {trace_data.get('final_output', '')[:500]}"

        pair = QAPair(
            qa_id=str(uuid.uuid4())[:12],
            created_at=datetime.now(),
            source_trace_id=trace_data.get("trace_id"),
            source_evaluation_id=evaluation_data.get("evaluation_id") if evaluation_data else None,
            format=QAFormat.TOOL_USE,
            quality=self._determine_quality(score, len(tools_used)),
            system_prompt="You are an assistant with access to tools. Use them to complete tasks.",
            question=task,
            answer=tool_response,
            tags=["tool-use", "function-calling"],
            task_type=trace_data.get("task_type"),
            score=score,
            tools_used=tools_used
        )
        pairs.append(pair)

        return pairs

    def _determine_quality(self, score: float, num_tools: int) -> QAQuality:
        if score >= 0.85 and num_tools >= 2:
            return QAQuality.GOLD
        elif score >= 0.7:
            return QAQuality.SILVER
        return QAQuality.BRONZE

    def get_name(self) -> str:
        return "tool_use_generator"


class ErrorRecoveryQAGenerator(QAGenerator):
    """Generates error recovery pattern Q&A pairs"""

    def generate(
        self,
        trace_data: Dict[str, Any],
        evaluation_data: Optional[Dict[str, Any]] = None,
        config: QAGenerationConfig = None
    ) -> List[QAPair]:
        config = config or QAGenerationConfig()
        pairs = []

        error = trace_data.get("error")
        success = trace_data.get("success", False)
        turns = trace_data.get("turns", [])

        # We want cases where there was an error but eventual success
        # or where we can learn from the error handling
        if not error and success:
            return pairs  # No errors to learn from

        task = trace_data.get("original_task", "")
        final_output = trace_data.get("final_output", "")

        # Find error and recovery pattern
        error_turn = None
        recovery_turns = []

        for i, turn in enumerate(turns):
            if turn.get("error"):
                error_turn = turn
                recovery_turns = turns[i + 1:]
                break

        if not error_turn:
            # Use the overall error
            error_msg = error
            recovery_output = final_output if success else "Unable to recover from this error."
        else:
            error_msg = error_turn.get("error", error)
            recovery_output = "\n".join([
                t.get("output", "")[:200] for t in recovery_turns[:3]
            ]) if recovery_turns else final_output

        # Build error recovery response
        response = f"I encountered an error: {error_msg}\n\n"
        if success:
            response += "Here's how I recovered:\n"
            response += recovery_output
        else:
            response += "This error could not be automatically recovered. "
            response += "Here's what I recommend:\n"
            response += self._suggest_recovery(error_msg)

        pair = QAPair(
            qa_id=str(uuid.uuid4())[:12],
            created_at=datetime.now(),
            source_trace_id=trace_data.get("trace_id"),
            source_evaluation_id=evaluation_data.get("evaluation_id") if evaluation_data else None,
            format=QAFormat.ERROR_RECOVERY,
            quality=QAQuality.SILVER if success else QAQuality.BRONZE,
            system_prompt="You are an assistant that gracefully handles errors and helps users recover.",
            question=f"{task}\n\n[Error occurred: {error_msg}]",
            answer=response,
            tags=["error-recovery", "debugging"],
            task_type=trace_data.get("task_type"),
            score=0.6 if success else 0.4
        )
        pairs.append(pair)

        return pairs

    def _suggest_recovery(self, error_msg: str) -> str:
        """Suggest recovery steps based on error type"""
        error_lower = error_msg.lower()

        if "not found" in error_lower or "no such file" in error_lower:
            return "1. Check if the file/path exists\n2. Verify the path is correct\n3. Create the file if needed"
        elif "permission" in error_lower:
            return "1. Check file permissions\n2. Ensure you have the necessary access rights"
        elif "timeout" in error_lower:
            return "1. Retry the operation\n2. Increase the timeout value\n3. Check network connectivity"
        elif "syntax" in error_lower or "parse" in error_lower:
            return "1. Check the syntax of your input\n2. Validate the format\n3. Review the error location"
        else:
            return "1. Review the error message\n2. Check the inputs\n3. Try an alternative approach"

    def get_name(self) -> str:
        return "error_recovery_generator"


class QAGenerationSystem:
    """
    Complete Q&A generation system that processes execution traces
    and produces training data in multiple formats.
    """

    def __init__(
        self,
        base_path: str = "/tmp/qa-generation",
        config: QAGenerationConfig = None
    ):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.config = config or QAGenerationConfig()

        # Initialize generators
        self.generators: Dict[QAFormat, QAGenerator] = {
            QAFormat.INSTRUCTION: InstructionQAGenerator(),
            QAFormat.CONVERSATION: ConversationQAGenerator(),
            QAFormat.REASONING: ReasoningQAGenerator(),
            QAFormat.TOOL_USE: ToolUseQAGenerator(),
            QAFormat.ERROR_RECOVERY: ErrorRecoveryQAGenerator(),
        }

        # Storage
        self.qa_pairs: Dict[str, QAPair] = {}
        self.pairs_by_format: Dict[QAFormat, List[str]] = {f: [] for f in QAFormat}
        self.pairs_by_quality: Dict[QAQuality, List[str]] = {q: [] for q in QAQuality}

        # Deduplication
        self.question_hashes: set = set()

        # Persistence
        self.db_path = self.base_path / "qa_database.db"
        self._init_db()

        self._lock = threading.Lock()

        # Stats
        self.stats = {
            "total_generated": 0,
            "by_format": {f.value: 0 for f in QAFormat},
            "by_quality": {q.value: 0 for q in QAQuality},
            "duplicates_skipped": 0
        }

    def _init_db(self):
        """Initialize database"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS qa_pairs (
                    qa_id TEXT PRIMARY KEY,
                    format TEXT,
                    quality TEXT,
                    question TEXT,
                    answer TEXT,
                    score REAL,
                    source_trace_id TEXT,
                    task_type TEXT,
                    created_at TEXT,
                    data TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_qa_format ON qa_pairs(format)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_qa_quality ON qa_pairs(quality)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_qa_score ON qa_pairs(score)
            """)
            conn.commit()

    def generate_from_trace(
        self,
        trace_data: Dict[str, Any],
        evaluation_data: Optional[Dict[str, Any]] = None,
        formats: List[QAFormat] = None
    ) -> List[QAPair]:
        """Generate Q&A pairs from a single trace"""
        formats = formats or self.config.formats
        all_pairs = []

        for format_type in formats:
            generator = self.generators.get(format_type)
            if generator:
                pairs = generator.generate(trace_data, evaluation_data, self.config)
                all_pairs.extend(pairs)

        # Deduplicate and store
        stored_pairs = []
        for pair in all_pairs:
            if self._store_pair(pair):
                stored_pairs.append(pair)

        return stored_pairs

    def _store_pair(self, pair: QAPair) -> bool:
        """Store a pair if it's not a duplicate"""
        # Check for duplicates
        question_hash = hashlib.md5(pair.question.encode()).hexdigest()

        with self._lock:
            if question_hash in self.question_hashes:
                self.stats["duplicates_skipped"] += 1
                return False

            self.question_hashes.add(question_hash)
            self.qa_pairs[pair.qa_id] = pair
            self.pairs_by_format[pair.format].append(pair.qa_id)
            self.pairs_by_quality[pair.quality].append(pair.qa_id)

            # Update stats
            self.stats["total_generated"] += 1
            self.stats["by_format"][pair.format.value] += 1
            self.stats["by_quality"][pair.quality.value] += 1

            # Persist
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO qa_pairs
                    (qa_id, format, quality, question, answer, score, source_trace_id, task_type, created_at, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pair.qa_id,
                    pair.format.value,
                    pair.quality.value,
                    pair.question,
                    pair.answer,
                    pair.score,
                    pair.source_trace_id,
                    pair.task_type,
                    pair.created_at.isoformat(),
                    json.dumps(pair.to_dict())
                ))
                conn.commit()

            return True

    def get_pairs(
        self,
        format: QAFormat = None,
        quality: QAQuality = None,
        min_score: float = 0.0,
        limit: int = None,
        task_type: str = None
    ) -> List[QAPair]:
        """Get Q&A pairs with filters"""
        query = "SELECT data FROM qa_pairs WHERE score >= ?"
        params = [min_score]

        if format:
            query += " AND format = ?"
            params.append(format.value)

        if quality:
            query += " AND quality = ?"
            params.append(quality.value)

        if task_type:
            query += " AND task_type = ?"
            params.append(task_type)

        query += " ORDER BY score DESC"

        if limit:
            query += f" LIMIT {limit}"

        pairs = []
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute(query, params)
            for (data_json,) in cursor.fetchall():
                try:
                    data = json.loads(data_json)
                    pair = QAPair(
                        qa_id=data["qa_id"],
                        created_at=datetime.fromisoformat(data["created_at"]),
                        source_trace_id=data.get("source_trace_id"),
                        source_evaluation_id=data.get("source_evaluation_id"),
                        format=QAFormat(data["format"]),
                        quality=QAQuality(data["quality"]),
                        system_prompt=data.get("system_prompt"),
                        question=data["question"],
                        answer=data["answer"],
                        conversation=data.get("conversation", []),
                        tags=data.get("tags", []),
                        task_type=data.get("task_type"),
                        score=data.get("score", 0.0),
                        tools_used=data.get("tools_used", [])
                    )
                    pairs.append(pair)
                except (json.JSONDecodeError, KeyError):
                    continue

        return pairs

    def export_training_data(
        self,
        output_path: str,
        style: str = "alpaca",
        format: QAFormat = None,
        min_quality: QAQuality = QAQuality.BRONZE,
        min_score: float = 0.5,
        limit: int = None
    ) -> int:
        """Export Q&A pairs as training data"""
        # Get quality order
        quality_order = [QAQuality.GOLD, QAQuality.SILVER, QAQuality.BRONZE, QAQuality.UNVERIFIED]
        min_quality_idx = quality_order.index(min_quality)
        acceptable_qualities = quality_order[:min_quality_idx + 1]

        pairs = self.get_pairs(format=format, min_score=min_score, limit=limit)

        # Filter by quality
        pairs = [p for p in pairs if p.quality in acceptable_qualities]

        # Convert to training format
        training_data = [p.to_training_format(style) for p in pairs]

        # Write output
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".jsonl":
            with open(output_path, "w") as f:
                for item in training_data:
                    f.write(json.dumps(item) + "\n")
        else:
            with open(output_path, "w") as f:
                json.dump(training_data, f, indent=2)

        return len(training_data)

    def export_for_finetuning(
        self,
        output_dir: str,
        train_ratio: float = 0.9,
        style: str = "openai"
    ) -> Dict[str, int]:
        """Export data split into train/validation sets"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get all good quality pairs
        pairs = self.get_pairs(min_score=0.6)

        # Split
        split_idx = int(len(pairs) * train_ratio)
        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]

        # Export
        train_data = [p.to_training_format(style) for p in train_pairs]
        val_data = [p.to_training_format(style) for p in val_pairs]

        with open(output_path / "train.jsonl", "w") as f:
            for item in train_data:
                f.write(json.dumps(item) + "\n")

        with open(output_path / "validation.jsonl", "w") as f:
            for item in val_data:
                f.write(json.dumps(item) + "\n")

        return {
            "train": len(train_data),
            "validation": len(val_data)
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return {
            **self.stats,
            "stored_pairs": len(self.qa_pairs),
            "unique_questions": len(self.question_hashes)
        }

    def get_sample_pairs(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get sample pairs for review"""
        pairs = self.get_pairs(limit=n)
        return [p.to_dict() for p in pairs]


class IntegratedQASystem:
    """
    Integrated Q&A system that works with the evaluation flywheel.

    This connects:
    1. Execution traces
    2. Evaluations
    3. Q&A generation
    4. Training data export

    Creating a complete data flywheel for model improvement.
    """

    def __init__(
        self,
        eval_loop,  # SelfEvaluationLoop
        base_path: str = "/tmp/integrated-qa"
    ):
        self.eval_loop = eval_loop
        self.qa_system = QAGenerationSystem(base_path)

        # Processing config
        self.auto_generate = True
        self.min_score_for_qa = 0.5

    def process_evaluation(
        self,
        evaluation_id: str
    ) -> List[QAPair]:
        """Process an evaluation and generate Q&A pairs"""
        evaluation = self.eval_loop.evaluations.get(evaluation_id)
        if not evaluation:
            return []

        trace = self.eval_loop.traces.get(evaluation.target_id)
        if not trace:
            return []

        if evaluation.overall_score < self.min_score_for_qa:
            return []

        # Generate Q&A pairs
        return self.qa_system.generate_from_trace(
            trace.to_dict(),
            evaluation.to_dict()
        )

    def process_all_pending(self) -> Dict[str, int]:
        """Process all evaluations that haven't been converted to Q&A"""
        processed = 0
        pairs_generated = 0

        for eval_id, evaluation in self.eval_loop.evaluations.items():
            # Check if already processed (could add tracking)
            pairs = self.process_evaluation(eval_id)
            if pairs:
                processed += 1
                pairs_generated += len(pairs)

        return {
            "evaluations_processed": processed,
            "pairs_generated": pairs_generated
        }

    def export_flywheel_data(
        self,
        output_dir: str,
        include_lessons: bool = True
    ) -> Dict[str, Any]:
        """Export complete flywheel data for training"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export Q&A pairs
        qa_counts = self.qa_system.export_for_finetuning(
            str(output_path / "qa"),
            style="openai"
        )

        # Export lessons if requested
        lessons_count = 0
        if include_lessons:
            lessons = [l.to_dict() for l in self.eval_loop.lessons.values()]
            with open(output_path / "lessons.json", "w") as f:
                json.dump(lessons, f, indent=2)
            lessons_count = len(lessons)

        # Export flywheel stats
        stats = {
            "qa_system": self.qa_system.get_stats(),
            "evaluation_system": {
                "total_traces": len(self.eval_loop.traces),
                "total_evaluations": len(self.eval_loop.evaluations),
                "total_lessons": len(self.eval_loop.lessons)
            }
        }
        with open(output_path / "flywheel_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        return {
            **qa_counts,
            "lessons": lessons_count,
            "output_dir": str(output_path)
        }


# Convenience function
def create_qa_system(
    base_path: str = "/tmp/qa-system",
    config: QAGenerationConfig = None
) -> QAGenerationSystem:
    """Create Q&A generation system"""
    return QAGenerationSystem(base_path, config)
