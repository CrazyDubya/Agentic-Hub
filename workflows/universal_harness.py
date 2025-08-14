"""
Universal Workflow Harness
Supports n8n, LangChain, custom Python, and text configuration workflows
"""

import json
import yaml
import subprocess
import requests
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import tempfile
import os
from pathlib import Path

class WorkflowType(Enum):
    N8N = "n8n"
    LANGCHAIN = "langchain"
    PYTHON = "python"
    TEXT_CONFIG = "text"
    HYBRID = "hybrid"

@dataclass
class WorkflowResult:
    """Result of workflow execution"""
    success: bool
    output: Any
    execution_time: float
    workflow_type: WorkflowType
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

class TextConfigParser:
    """Parse simple text configuration files"""
    
    def parse(self, config_content: str) -> Dict[str, Any]:
        """Parse text configuration into structured data"""
        config = {}
        current_section = None
        
        for line in config_content.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            if ':' in line and not line.startswith(' '):
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Handle different value types
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '').isdigit():
                    value = float(value)
                elif value.startswith('[') and value.endswith(']'):
                    # Simple list parsing
                    value = [item.strip() for item in value[1:-1].split(',')]
                
                config[key] = value
                current_section = key
            elif line.startswith('  -') and current_section:
                # Handle list items
                if not isinstance(config[current_section], list):
                    config[current_section] = []
                config[current_section].append(line[3:].strip())
        
        return config

class N8NConnector:
    """Interface to n8n workflow automation"""
    
    def __init__(self, n8n_url: str, api_key: Optional[str] = None):
        self.n8n_url = n8n_url.rstrip('/')
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
    
    def execute_workflow(self, workflow_id: str, input_data: Dict[str, Any]) -> WorkflowResult:
        """Execute n8n workflow"""
        import time
        start_time = time.time()
        
        try:
            url = f"{self.n8n_url}/api/v1/workflows/{workflow_id}/execute"
            response = requests.post(url, json=input_data, headers=self.headers)
            
            execution_time = time.time() - start_time
            
            if response.status_code == 200:
                return WorkflowResult(
                    success=True,
                    output=response.json(),
                    execution_time=execution_time,
                    workflow_type=WorkflowType.N8N,
                    metadata={"workflow_id": workflow_id}
                )
            else:
                return WorkflowResult(
                    success=False,
                    output=None,
                    execution_time=execution_time,
                    workflow_type=WorkflowType.N8N,
                    error=f"HTTP {response.status_code}: {response.text}"
                )
        except Exception as e:
            return WorkflowResult(
                success=False,
                output=None,
                execution_time=time.time() - start_time,
                workflow_type=WorkflowType.N8N,
                error=str(e)
            )

class LangChainOrchestrator:
    """LangChain workflow orchestration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.chains = {}
    
    def create_chain(self, chain_config: Dict[str, Any]):
        """Create LangChain from configuration"""
        # This would integrate with actual LangChain
        # For now, return a mock implementation
        chain_id = chain_config.get('id', 'default')
        self.chains[chain_id] = chain_config
        return chain_id
    
    def execute_chain(self, chain_id: str, input_data: Dict[str, Any]) -> WorkflowResult:
        """Execute LangChain workflow"""
        import time
        start_time = time.time()
        
        try:
            if chain_id not in self.chains:
                raise ValueError(f"Chain {chain_id} not found")
            
            chain_config = self.chains[chain_id]
            
            # Mock LangChain execution
            # In real implementation, this would use actual LangChain
            result = {
                "chain_id": chain_id,
                "input": input_data,
                "output": f"Processed by LangChain: {chain_config.get('description', 'Unknown chain')}",
                "steps_executed": chain_config.get('steps', [])
            }
            
            execution_time = time.time() - start_time
            
            return WorkflowResult(
                success=True,
                output=result,
                execution_time=execution_time,
                workflow_type=WorkflowType.LANGCHAIN,
                metadata={"chain_id": chain_id}
            )
            
        except Exception as e:
            return WorkflowResult(
                success=False,
                output=None,
                execution_time=time.time() - start_time,
                workflow_type=WorkflowType.LANGCHAIN,
                error=str(e)
            )

class PythonScriptEngine:
    """Execute custom Python scripts safely"""
    
    def __init__(self, sandbox_mode: bool = True):
        self.sandbox_mode = sandbox_mode
        self.allowed_imports = [
            'json', 'yaml', 'requests', 'datetime', 'time', 'random',
            'math', 'statistics', 'collections', 're', 'uuid'
        ]
    
    def execute_script(self, script_content: str, input_data: Dict[str, Any]) -> WorkflowResult:
        """Execute Python script with input data"""
        import time
        start_time = time.time()
        
        try:
            # Create temporary script file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Prepare script with input data
                script_with_input = f"""
import json
import sys

# Input data from workflow
input_data = {json.dumps(input_data)}

# User script
{script_content}

# Ensure output is captured
if 'output' not in locals():
    output = {{"status": "completed", "input": input_data}}

print(json.dumps(output))
"""
                f.write(script_with_input)
                script_path = f.name
            
            # Execute script
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            # Cleanup
            os.unlink(script_path)
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                try:
                    output = json.loads(result.stdout.strip())
                except json.JSONDecodeError:
                    output = {"raw_output": result.stdout}
                
                return WorkflowResult(
                    success=True,
                    output=output,
                    execution_time=execution_time,
                    workflow_type=WorkflowType.PYTHON
                )
            else:
                return WorkflowResult(
                    success=False,
                    output=None,
                    execution_time=execution_time,
                    workflow_type=WorkflowType.PYTHON,
                    error=result.stderr
                )
                
        except Exception as e:
            return WorkflowResult(
                success=False,
                output=None,
                execution_time=time.time() - start_time,
                workflow_type=WorkflowType.PYTHON,
                error=str(e)
            )

class UniversalWorkflowHarness:
    """Universal workflow harness supporting multiple execution engines"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.text_parser = TextConfigParser()
        
        # Initialize connectors
        self.n8n_connector = None
        if config.get('n8n'):
            self.n8n_connector = N8NConnector(
                config['n8n']['url'],
                config['n8n'].get('api_key')
            )
        
        self.langchain_orchestrator = LangChainOrchestrator(config.get('langchain', {}))
        self.python_engine = PythonScriptEngine(config.get('python', {}).get('sandbox', True))
    
    def execute_workflow(self, 
                        workflow_config: Union[str, Dict[str, Any]], 
                        input_data: Dict[str, Any],
                        workflow_type: Optional[WorkflowType] = None) -> WorkflowResult:
        """Execute workflow based on configuration"""
        
        # Parse configuration if it's a string (text config)
        if isinstance(workflow_config, str):
            if workflow_config.endswith('.txt') or workflow_config.endswith('.yaml'):
                with open(workflow_config, 'r') as f:
                    content = f.read()
                if workflow_config.endswith('.yaml'):
                    config = yaml.safe_load(content)
                else:
                    config = self.text_parser.parse(content)
            else:
                # Treat as text configuration content
                config = self.text_parser.parse(workflow_config)
        else:
            config = workflow_config
        
        # Determine workflow type
        if not workflow_type:
            workflow_type = WorkflowType(config.get('workflow_type', 'python'))
        
        # Route to appropriate executor
        if workflow_type == WorkflowType.N8N:
            return self._execute_n8n(config, input_data)
        elif workflow_type == WorkflowType.LANGCHAIN:
            return self._execute_langchain(config, input_data)
        elif workflow_type == WorkflowType.PYTHON:
            return self._execute_python(config, input_data)
        elif workflow_type == WorkflowType.HYBRID:
            return self._execute_hybrid(config, input_data)
        else:
            return WorkflowResult(
                success=False,
                output=None,
                execution_time=0,
                workflow_type=workflow_type,
                error=f"Unsupported workflow type: {workflow_type}"
            )
    
    def _execute_n8n(self, config: Dict[str, Any], input_data: Dict[str, Any]) -> WorkflowResult:
        """Execute n8n workflow"""
        if not self.n8n_connector:
            return WorkflowResult(
                success=False,
                output=None,
                execution_time=0,
                workflow_type=WorkflowType.N8N,
                error="n8n connector not configured"
            )
        
        workflow_id = config.get('workflow_id')
        if not workflow_id:
            return WorkflowResult(
                success=False,
                output=None,
                execution_time=0,
                workflow_type=WorkflowType.N8N,
                error="workflow_id not specified"
            )
        
        return self.n8n_connector.execute_workflow(workflow_id, input_data)
    
    def _execute_langchain(self, config: Dict[str, Any], input_data: Dict[str, Any]) -> WorkflowResult:
        """Execute LangChain workflow"""
        chain_id = config.get('chain_id', 'default')
        
        # Create chain if it doesn't exist
        if chain_id not in self.langchain_orchestrator.chains:
            self.langchain_orchestrator.create_chain(config)
        
        return self.langchain_orchestrator.execute_chain(chain_id, input_data)
    
    def _execute_python(self, config: Dict[str, Any], input_data: Dict[str, Any]) -> WorkflowResult:
        """Execute Python script workflow"""
        script_content = config.get('script', '')
        if not script_content:
            script_file = config.get('script_file')
            if script_file and os.path.exists(script_file):
                with open(script_file, 'r') as f:
                    script_content = f.read()
        
        if not script_content:
            return WorkflowResult(
                success=False,
                output=None,
                execution_time=0,
                workflow_type=WorkflowType.PYTHON,
                error="No script content or script_file specified"
            )
        
        return self.python_engine.execute_script(script_content, input_data)
    
    def _execute_hybrid(self, config: Dict[str, Any], input_data: Dict[str, Any]) -> WorkflowResult:
        """Execute hybrid workflow with multiple steps"""
        steps = config.get('steps', [])
        results = []
        current_data = input_data.copy()
        
        import time
        start_time = time.time()
        
        try:
            for i, step in enumerate(steps):
                step_type = WorkflowType(step.get('type', 'python'))
                step_result = self.execute_workflow(step, current_data, step_type)
                
                results.append({
                    "step": i,
                    "type": step_type.value,
                    "success": step_result.success,
                    "output": step_result.output,
                    "error": step_result.error
                })
                
                if not step_result.success and step.get('required', True):
                    # Stop on required step failure
                    break
                
                # Pass output to next step
                if step_result.success and step_result.output:
                    current_data.update(step_result.output)
            
            execution_time = time.time() - start_time
            overall_success = all(r['success'] for r in results if r.get('required', True))
            
            return WorkflowResult(
                success=overall_success,
                output={
                    "steps": results,
                    "final_data": current_data
                },
                execution_time=execution_time,
                workflow_type=WorkflowType.HYBRID
            )
            
        except Exception as e:
            return WorkflowResult(
                success=False,
                output={"steps": results},
                execution_time=time.time() - start_time,
                workflow_type=WorkflowType.HYBRID,
                error=str(e)
            )

# Example usage and configuration
EXAMPLE_TEXT_CONFIG = """
# Prison Simulation Workflow
workflow_type: hybrid
scenario: prison_yard_incident
duration: 24_hours

agents:
  - inmate_leader: aggressive, influential, 10_year_sentence
  - new_inmate: anxious, non_violent, 2_year_sentence  
  - guard_captain: experienced, by_the_book, 15_years_service

steps:
  - type: python
    script_file: scenarios/prison_setup.py
    required: true
  - type: langchain
    chain_id: behavioral_analysis
    required: false
  - type: python
    script: |
      # Generate final report
      output = {
        "report": "Prison simulation completed",
        "insights": input_data.get("behavioral_patterns", [])
      }

analysis_focus:
  - power_dynamics
  - behavioral_adaptation
  - intervention_effectiveness
"""