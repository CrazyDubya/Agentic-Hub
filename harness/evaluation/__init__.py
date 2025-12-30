"""
Evaluation Module

Self-evaluation loops and flywheel system for continuous agent improvement.
Includes Q&A generation for training data creation.
"""

from .self_eval import (
    # Enums
    EvaluationDimension,
    EvaluationLevel,

    # Data classes
    EvaluationCriteria,
    EvaluationResult,
    ExecutionTrace,
    Lesson,

    # Evaluators
    Evaluator,
    RuleBasedEvaluator,
    LLMEvaluator,
    PeerEvaluator,

    # Core systems
    SelfEvaluationLoop,
    FlywheelManager,

    # Convenience
    create_evaluation_system
)

from .qa_generation import (
    # Enums
    QAFormat,
    QAQuality,

    # Data classes
    QAPair,
    QAGenerationConfig,

    # Generators
    QAGenerator,
    InstructionQAGenerator,
    ConversationQAGenerator,
    ReasoningQAGenerator,
    ToolUseQAGenerator,
    ErrorRecoveryQAGenerator,

    # Core systems
    QAGenerationSystem,
    IntegratedQASystem,

    # Convenience
    create_qa_system
)

__all__ = [
    # Self-evaluation
    "EvaluationDimension",
    "EvaluationLevel",
    "EvaluationCriteria",
    "EvaluationResult",
    "ExecutionTrace",
    "Lesson",
    "Evaluator",
    "RuleBasedEvaluator",
    "LLMEvaluator",
    "PeerEvaluator",
    "SelfEvaluationLoop",
    "FlywheelManager",
    "create_evaluation_system",

    # Q&A Generation
    "QAFormat",
    "QAQuality",
    "QAPair",
    "QAGenerationConfig",
    "QAGenerator",
    "InstructionQAGenerator",
    "ConversationQAGenerator",
    "ReasoningQAGenerator",
    "ToolUseQAGenerator",
    "ErrorRecoveryQAGenerator",
    "QAGenerationSystem",
    "IntegratedQASystem",
    "create_qa_system"
]
