"""
Universal Skill System
Provides a plugin-like architecture for extending agent capabilities.

Key features:
- Declarative skill definitions (YAML/JSON/Python)
- Dynamic loading and unloading
- Skill composition and chaining
- Version management
- Dependency resolution

Inspired by:
- GitHub Copilot's Agent Skills
- Claude Code's Skills system
- Plugin architectures from editors (VS Code, etc.)
"""

import os
import json
import yaml
import hashlib
import importlib.util
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod
import threading
import inspect


class SkillType(Enum):
    """Types of skills"""
    BUILTIN = "builtin"       # Core skills that come with harness
    INSTALLED = "installed"   # Downloaded from marketplace
    LOCAL = "local"          # User-defined local skills
    REMOTE = "remote"        # Skills executed on remote servers


class SkillCategory(Enum):
    """Skill categories for organization"""
    FILE_OPS = "file_operations"
    CODE_ANALYSIS = "code_analysis"
    CODE_GENERATION = "code_generation"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    DEPLOYMENT = "deployment"
    DATA_PROCESSING = "data_processing"
    WEB_INTERACTION = "web_interaction"
    COMMUNICATION = "communication"
    UTILITY = "utility"
    CUSTOM = "custom"


@dataclass
class SkillParameter:
    """Definition of a skill parameter"""
    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None
    pattern: Optional[str] = None  # Regex for validation


@dataclass
class SkillOutput:
    """Definition of skill output"""
    type: str
    description: str
    schema: Optional[Dict[str, Any]] = None


@dataclass
class SkillDependency:
    """Skill dependency specification"""
    skill_id: str
    version: str = "*"
    optional: bool = False


@dataclass
class SkillMetadata:
    """Full metadata for a skill"""
    skill_id: str
    name: str
    version: str
    description: str
    author: str
    skill_type: SkillType
    category: SkillCategory
    parameters: List[SkillParameter]
    output: SkillOutput
    dependencies: List[SkillDependency] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    source_url: Optional[str] = None
    license: str = "MIT"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "type": self.skill_type.value,
            "category": self.category.value,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default
                }
                for p in self.parameters
            ],
            "output": {
                "type": self.output.type,
                "description": self.output.description
            },
            "dependencies": [
                {"skill_id": d.skill_id, "version": d.version}
                for d in self.dependencies
            ],
            "tags": self.tags,
            "examples": self.examples
        }


@dataclass
class SkillResult:
    """Result of skill execution"""
    success: bool
    output: Any
    execution_time_ms: float
    skill_id: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Skill(ABC):
    """Abstract base class for skills"""

    def __init__(self, metadata: SkillMetadata):
        self.metadata = metadata
        self._initialized = False

    @abstractmethod
    def execute(self, params: Dict[str, Any], context: Dict[str, Any]) -> SkillResult:
        """Execute the skill with given parameters and context"""
        pass

    def validate_params(self, params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate parameters against skill definition"""
        for param_def in self.metadata.parameters:
            if param_def.required and param_def.name not in params:
                return False, f"Missing required parameter: {param_def.name}"

            if param_def.name in params:
                value = params[param_def.name]
                # Type validation
                if param_def.type == "string" and not isinstance(value, str):
                    return False, f"Parameter {param_def.name} must be string"
                elif param_def.type == "number" and not isinstance(value, (int, float)):
                    return False, f"Parameter {param_def.name} must be number"
                elif param_def.type == "boolean" and not isinstance(value, bool):
                    return False, f"Parameter {param_def.name} must be boolean"
                elif param_def.type == "array" and not isinstance(value, list):
                    return False, f"Parameter {param_def.name} must be array"

                # Enum validation
                if param_def.enum and value not in param_def.enum:
                    return False, f"Parameter {param_def.name} must be one of {param_def.enum}"

        return True, None

    def get_help(self) -> str:
        """Generate help text for this skill"""
        lines = [
            f"# {self.metadata.name}",
            "",
            self.metadata.description,
            "",
            "## Parameters",
        ]

        for param in self.metadata.parameters:
            req = "(required)" if param.required else "(optional)"
            lines.append(f"- **{param.name}** `{param.type}` {req}")
            lines.append(f"  {param.description}")
            if param.default is not None:
                lines.append(f"  Default: `{param.default}`")

        if self.metadata.examples:
            lines.append("")
            lines.append("## Examples")
            for example in self.metadata.examples:
                lines.append(f"```")
                lines.append(json.dumps(example, indent=2))
                lines.append(f"```")

        return '\n'.join(lines)


class PythonSkill(Skill):
    """
    Skill implemented as a Python function.
    Can be loaded from a Python file or defined inline.
    """

    def __init__(self, metadata: SkillMetadata, func: Callable):
        super().__init__(metadata)
        self.func = func

    def execute(self, params: Dict[str, Any], context: Dict[str, Any]) -> SkillResult:
        import time
        start = time.time()

        valid, error = self.validate_params(params)
        if not valid:
            return SkillResult(
                success=False,
                output=None,
                execution_time_ms=0,
                skill_id=self.metadata.skill_id,
                error=error
            )

        try:
            # Check function signature to see if it expects context
            sig = inspect.signature(self.func)
            if 'context' in sig.parameters:
                result = self.func(params, context)
            else:
                result = self.func(params)

            execution_time = (time.time() - start) * 1000

            return SkillResult(
                success=True,
                output=result,
                execution_time_ms=execution_time,
                skill_id=self.metadata.skill_id
            )

        except Exception as e:
            return SkillResult(
                success=False,
                output=None,
                execution_time_ms=(time.time() - start) * 1000,
                skill_id=self.metadata.skill_id,
                error=str(e)
            )


class PromptSkill(Skill):
    """
    Skill defined as a prompt template.
    Useful for LLM-based skills that don't require code execution.
    """

    def __init__(self, metadata: SkillMetadata, prompt_template: str):
        super().__init__(metadata)
        self.prompt_template = prompt_template

    def execute(self, params: Dict[str, Any], context: Dict[str, Any]) -> SkillResult:
        import time
        start = time.time()

        valid, error = self.validate_params(params)
        if not valid:
            return SkillResult(
                success=False,
                output=None,
                execution_time_ms=0,
                skill_id=self.metadata.skill_id,
                error=error
            )

        try:
            # Render the prompt template with parameters
            rendered = self.prompt_template
            for key, value in params.items():
                rendered = rendered.replace(f"{{{{{key}}}}}", str(value))

            execution_time = (time.time() - start) * 1000

            return SkillResult(
                success=True,
                output={"prompt": rendered, "params": params},
                execution_time_ms=execution_time,
                skill_id=self.metadata.skill_id
            )

        except Exception as e:
            return SkillResult(
                success=False,
                output=None,
                execution_time_ms=(time.time() - start) * 1000,
                skill_id=self.metadata.skill_id,
                error=str(e)
            )


class CompositeSkill(Skill):
    """
    Skill composed of multiple other skills executed in sequence.
    Supports data passing between skills.
    """

    def __init__(self, metadata: SkillMetadata, steps: List[Dict[str, Any]]):
        super().__init__(metadata)
        self.steps = steps  # [{"skill_id": "...", "params": {...}, "output_key": "..."}]

    def execute(self, params: Dict[str, Any], context: Dict[str, Any]) -> SkillResult:
        import time
        start = time.time()

        skill_registry = context.get('skill_registry')
        if not skill_registry:
            return SkillResult(
                success=False,
                output=None,
                execution_time_ms=0,
                skill_id=self.metadata.skill_id,
                error="No skill registry in context"
            )

        results = {}
        current_data = dict(params)

        for step in self.steps:
            skill_id = step['skill_id']
            step_params = step.get('params', {})

            # Resolve parameter references
            resolved_params = {}
            for key, value in step_params.items():
                if isinstance(value, str) and value.startswith('$'):
                    ref = value[1:]
                    if ref in current_data:
                        resolved_params[key] = current_data[ref]
                    elif ref in results:
                        resolved_params[key] = results[ref]
                    else:
                        resolved_params[key] = value
                else:
                    resolved_params[key] = value

            # Execute step skill
            skill = skill_registry.get_skill(skill_id)
            if not skill:
                return SkillResult(
                    success=False,
                    output=results,
                    execution_time_ms=(time.time() - start) * 1000,
                    skill_id=self.metadata.skill_id,
                    error=f"Skill not found: {skill_id}"
                )

            result = skill.execute(resolved_params, context)

            output_key = step.get('output_key', f"step_{len(results)}")
            results[output_key] = result.output

            if not result.success:
                return SkillResult(
                    success=False,
                    output=results,
                    execution_time_ms=(time.time() - start) * 1000,
                    skill_id=self.metadata.skill_id,
                    error=f"Step {skill_id} failed: {result.error}"
                )

        return SkillResult(
            success=True,
            output=results,
            execution_time_ms=(time.time() - start) * 1000,
            skill_id=self.metadata.skill_id
        )


class SkillRegistry:
    """
    Central registry for all available skills.
    Handles loading, caching, and lookup.
    """

    def __init__(self, skills_dir: str = None):
        self.skills: Dict[str, Skill] = {}
        self.skills_dir = Path(skills_dir) if skills_dir else None
        self._lock = threading.Lock()

        # Register builtin skills
        self._register_builtins()

    def _register_builtins(self):
        """Register built-in core skills"""

        # File read skill
        self.register(PythonSkill(
            SkillMetadata(
                skill_id="file.read",
                name="Read File",
                version="1.0.0",
                description="Read contents of a file",
                author="system",
                skill_type=SkillType.BUILTIN,
                category=SkillCategory.FILE_OPS,
                parameters=[
                    SkillParameter("path", "string", "Path to file"),
                    SkillParameter("encoding", "string", "File encoding", required=False, default="utf-8")
                ],
                output=SkillOutput("string", "File contents")
            ),
            lambda params: open(params['path'], encoding=params.get('encoding', 'utf-8')).read()
        ))

        # File write skill
        self.register(PythonSkill(
            SkillMetadata(
                skill_id="file.write",
                name="Write File",
                version="1.0.0",
                description="Write contents to a file",
                author="system",
                skill_type=SkillType.BUILTIN,
                category=SkillCategory.FILE_OPS,
                parameters=[
                    SkillParameter("path", "string", "Path to file"),
                    SkillParameter("content", "string", "Content to write")
                ],
                output=SkillOutput("boolean", "Success status")
            ),
            lambda params: (Path(params['path']).write_text(params['content']), True)[1]
        ))

        # Shell execute skill
        def shell_execute(params):
            import subprocess
            result = subprocess.run(
                params['command'],
                shell=True,
                capture_output=True,
                text=True,
                timeout=params.get('timeout', 30),
                cwd=params.get('cwd')
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }

        self.register(PythonSkill(
            SkillMetadata(
                skill_id="shell.exec",
                name="Shell Execute",
                version="1.0.0",
                description="Execute a shell command",
                author="system",
                skill_type=SkillType.BUILTIN,
                category=SkillCategory.UTILITY,
                parameters=[
                    SkillParameter("command", "string", "Command to execute"),
                    SkillParameter("timeout", "number", "Timeout in seconds", required=False, default=30),
                    SkillParameter("cwd", "string", "Working directory", required=False)
                ],
                output=SkillOutput("object", "Command output with stdout, stderr, returncode")
            ),
            shell_execute
        ))

        # Code analysis skill (prompt-based)
        self.register(PromptSkill(
            SkillMetadata(
                skill_id="code.analyze",
                name="Analyze Code",
                version="1.0.0",
                description="Analyze code for patterns, issues, and improvements",
                author="system",
                skill_type=SkillType.BUILTIN,
                category=SkillCategory.CODE_ANALYSIS,
                parameters=[
                    SkillParameter("code", "string", "Code to analyze"),
                    SkillParameter("language", "string", "Programming language"),
                    SkillParameter("focus", "string", "Analysis focus", required=False,
                                 enum=["bugs", "performance", "security", "style", "all"])
                ],
                output=SkillOutput("object", "Analysis results")
            ),
            """Analyze the following {{language}} code:

```{{language}}
{{code}}
```

Focus on: {{focus}}

Provide a detailed analysis including:
1. Identified issues
2. Improvement suggestions
3. Best practice recommendations"""
        ))

    def register(self, skill: Skill) -> bool:
        """Register a skill"""
        with self._lock:
            self.skills[skill.metadata.skill_id] = skill
            return True

    def unregister(self, skill_id: str) -> bool:
        """Unregister a skill"""
        with self._lock:
            if skill_id in self.skills:
                del self.skills[skill_id]
                return True
            return False

    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """Get a skill by ID"""
        return self.skills.get(skill_id)

    def list_skills(
        self,
        category: Optional[SkillCategory] = None,
        skill_type: Optional[SkillType] = None,
        tags: Optional[List[str]] = None
    ) -> List[SkillMetadata]:
        """List skills with optional filtering"""
        results = []
        for skill in self.skills.values():
            if category and skill.metadata.category != category:
                continue
            if skill_type and skill.metadata.skill_type != skill_type:
                continue
            if tags and not any(t in skill.metadata.tags for t in tags):
                continue
            results.append(skill.metadata)
        return results

    def search(self, query: str) -> List[SkillMetadata]:
        """Search skills by name, description, or tags"""
        query_lower = query.lower()
        results = []

        for skill in self.skills.values():
            score = 0
            if query_lower in skill.metadata.name.lower():
                score += 10
            if query_lower in skill.metadata.description.lower():
                score += 5
            if any(query_lower in tag.lower() for tag in skill.metadata.tags):
                score += 3
            if score > 0:
                results.append((score, skill.metadata))

        results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in results]

    def load_from_file(self, file_path: Union[str, Path]) -> Optional[Skill]:
        """Load a skill from a file (Python, YAML, or JSON)"""
        path = Path(file_path)

        if path.suffix == '.py':
            return self._load_python_skill(path)
        elif path.suffix in ['.yaml', '.yml']:
            return self._load_yaml_skill(path)
        elif path.suffix == '.json':
            return self._load_json_skill(path)

        return None

    def _load_python_skill(self, path: Path) -> Optional[Skill]:
        """Load a Python skill module"""
        try:
            spec = importlib.util.spec_from_file_location(path.stem, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, 'SKILL_METADATA') and hasattr(module, 'execute'):
                metadata = module.SKILL_METADATA
                if isinstance(metadata, dict):
                    metadata = SkillMetadata(**metadata)

                skill = PythonSkill(metadata, module.execute)
                self.register(skill)
                return skill

        except Exception as e:
            print(f"Failed to load Python skill: {e}")

        return None

    def _load_yaml_skill(self, path: Path) -> Optional[Skill]:
        """Load a YAML skill definition"""
        try:
            with open(path) as f:
                data = yaml.safe_load(f)

            metadata = SkillMetadata(
                skill_id=data['skill_id'],
                name=data['name'],
                version=data.get('version', '1.0.0'),
                description=data['description'],
                author=data.get('author', 'unknown'),
                skill_type=SkillType(data.get('type', 'local')),
                category=SkillCategory(data.get('category', 'custom')),
                parameters=[
                    SkillParameter(**p) for p in data.get('parameters', [])
                ],
                output=SkillOutput(**data.get('output', {'type': 'any', 'description': 'Output'})),
                tags=data.get('tags', [])
            )

            if 'prompt' in data:
                skill = PromptSkill(metadata, data['prompt'])
            elif 'steps' in data:
                skill = CompositeSkill(metadata, data['steps'])
            else:
                return None

            self.register(skill)
            return skill

        except Exception as e:
            print(f"Failed to load YAML skill: {e}")

        return None

    def _load_json_skill(self, path: Path) -> Optional[Skill]:
        """Load a JSON skill definition"""
        try:
            with open(path) as f:
                data = json.load(f)

            # Similar to YAML loading
            return self._load_yaml_skill(path)  # JSON is subset of YAML

        except Exception as e:
            print(f"Failed to load JSON skill: {e}")

        return None

    def load_directory(self, dir_path: Union[str, Path]) -> int:
        """Load all skills from a directory"""
        path = Path(dir_path)
        count = 0

        for file_path in path.rglob("*"):
            if file_path.suffix in ['.py', '.yaml', '.yml', '.json']:
                if self.load_from_file(file_path):
                    count += 1

        return count

    def execute(
        self,
        skill_id: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> SkillResult:
        """Execute a skill by ID"""
        skill = self.get_skill(skill_id)
        if not skill:
            return SkillResult(
                success=False,
                output=None,
                execution_time_ms=0,
                skill_id=skill_id,
                error=f"Skill not found: {skill_id}"
            )

        context = context or {}
        context['skill_registry'] = self

        return skill.execute(params, context)

    def get_skill_prompt(self, skill_id: str) -> str:
        """Get a formatted prompt describing how to use a skill"""
        skill = self.get_skill(skill_id)
        if not skill:
            return f"Skill '{skill_id}' not found"

        return skill.get_help()

    def get_all_skills_prompt(self) -> str:
        """Generate a prompt listing all available skills"""
        lines = [
            "# Available Skills",
            "",
            "The following skills are available for use:",
            ""
        ]

        by_category = {}
        for skill in self.skills.values():
            cat = skill.metadata.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(skill.metadata)

        for category, skills in sorted(by_category.items()):
            lines.append(f"## {category.replace('_', ' ').title()}")
            for skill in skills:
                lines.append(f"- **{skill.skill_id}**: {skill.description}")
                params_str = ", ".join(p.name for p in skill.parameters if p.required)
                if params_str:
                    lines.append(f"  Required: `{params_str}`")
            lines.append("")

        return '\n'.join(lines)


# Skill definition file examples
EXAMPLE_YAML_SKILL = '''
skill_id: custom.greet
name: Greeting Generator
version: 1.0.0
description: Generate a personalized greeting
author: example
type: local
category: utility
parameters:
  - name: name
    type: string
    description: Name of the person to greet
    required: true
  - name: style
    type: string
    description: Greeting style
    required: false
    default: friendly
    enum: [friendly, formal, casual]
output:
  type: string
  description: The generated greeting
prompt: |
  Generate a {{style}} greeting for someone named {{name}}.
  Be warm and personable in the greeting.
tags:
  - greeting
  - text-generation
'''

EXAMPLE_PYTHON_SKILL = '''
"""
Example Python skill definition.
Save as: my_skill.py
"""

from datetime import datetime

SKILL_METADATA = {
    "skill_id": "custom.timestamp",
    "name": "Timestamp Generator",
    "version": "1.0.0",
    "description": "Generate timestamps in various formats",
    "author": "example",
    "skill_type": "local",
    "category": "utility",
    "parameters": [
        {"name": "format", "type": "string", "description": "Timestamp format", "required": False, "default": "iso"}
    ],
    "output": {"type": "string", "description": "Formatted timestamp"}
}

def execute(params):
    """Execute the skill"""
    fmt = params.get('format', 'iso')

    now = datetime.now()

    if fmt == 'iso':
        return now.isoformat()
    elif fmt == 'unix':
        return str(int(now.timestamp()))
    elif fmt == 'human':
        return now.strftime('%B %d, %Y at %I:%M %p')
    else:
        return now.strftime(fmt)
'''
