"""
Universal Agent Interface
Standardized interface for all agent frameworks in Agentic-Hub
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

class AgentStatus(Enum):
    """Agent status enumeration"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class AgentCapability:
    """Represents an agent capability"""
    name: str
    description: str
    parameters: Dict[str, Any]
    required_tools: List[str] = None

@dataclass
class TaskResult:
    """Result of task execution"""
    task_id: str
    status: TaskStatus
    result: Any
    error: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = None

@dataclass
class AgentContext:
    """Context information for agent execution"""
    world_state: Dict[str, Any]
    available_tools: List[str]
    other_agents: List[str]
    scenario_rules: Dict[str, Any]
    time_constraints: Optional[Dict[str, Any]] = None

class UniversalAgentInterface(ABC):
    """
    Universal interface that all agent adapters must implement
    Provides standardized way to interact with any agent framework
    """
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        self.status = AgentStatus.UNINITIALIZED
        self.capabilities: List[AgentCapability] = []
        self.active_tasks: Dict[str, TaskResult] = {}
        self.created_at = datetime.now()
        self.last_activity = None
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the agent with given configuration
        Returns True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def execute_task(self, task: str, context: AgentContext, task_id: str = None) -> TaskResult:
        """
        Execute a task with given context
        Returns TaskResult with execution details
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[AgentCapability]:
        """
        Return list of agent capabilities
        """
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Get current agent status and metrics
        """
        pass
    
    @abstractmethod
    def shutdown(self):
        """
        Gracefully shutdown the agent
        """
        pass
    
    # Optional methods with default implementations
    
    def pause(self) -> bool:
        """Pause agent execution"""
        if self.status == AgentStatus.BUSY:
            self.status = AgentStatus.READY
            return True
        return False
    
    def resume(self) -> bool:
        """Resume agent execution"""
        if self.status == AgentStatus.READY:
            return True
        return False
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a specific task"""
        if task_id in self.active_tasks:
            self.active_tasks[task_id].status = TaskStatus.CANCELLED
            return True
        return False
    
    def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """Get status of specific task"""
        return self.active_tasks.get(task_id)
    
    def get_active_tasks(self) -> List[TaskResult]:
        """Get all active tasks"""
        return [task for task in self.active_tasks.values() 
                if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]]
    
    def update_context(self, context: AgentContext):
        """Update agent context (world state, available tools, etc.)"""
        # Default implementation - can be overridden
        pass
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get agent memory usage statistics"""
        return {
            "active_tasks": len(self.active_tasks),
            "status": self.status.value,
            "uptime": (datetime.now() - self.created_at).total_seconds()
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        completed_tasks = [t for t in self.active_tasks.values() 
                          if t.status == TaskStatus.COMPLETED]
        failed_tasks = [t for t in self.active_tasks.values() 
                       if t.status == TaskStatus.FAILED]
        
        avg_execution_time = 0
        if completed_tasks:
            avg_execution_time = sum(t.execution_time or 0 for t in completed_tasks) / len(completed_tasks)
        
        return {
            "total_tasks": len(self.active_tasks),
            "completed_tasks": len(completed_tasks),
            "failed_tasks": len(failed_tasks),
            "success_rate": len(completed_tasks) / len(self.active_tasks) if self.active_tasks else 0,
            "average_execution_time": avg_execution_time,
            "last_activity": self.last_activity
        }

class AgentManager:
    """
    Manages multiple agents and provides orchestration capabilities
    """
    
    def __init__(self):
        self.agents: Dict[str, UniversalAgentInterface] = {}
        self.agent_registry: Dict[str, Dict[str, Any]] = {}
    
    def register_agent(self, agent: UniversalAgentInterface, metadata: Dict[str, Any] = None):
        """Register an agent with the manager"""
        self.agents[agent.agent_id] = agent
        self.agent_registry[agent.agent_id] = metadata or {}
    
    def get_agent(self, agent_id: str) -> Optional[UniversalAgentInterface]:
        """Get agent by ID"""
        return self.agents.get(agent_id)
    
    def get_agents_by_capability(self, capability: str) -> List[UniversalAgentInterface]:
        """Get all agents with specific capability"""
        matching_agents = []
        for agent in self.agents.values():
            agent_capabilities = [cap.name for cap in agent.get_capabilities()]
            if capability in agent_capabilities:
                matching_agents.append(agent)
        return matching_agents
    
    def get_available_agents(self) -> List[UniversalAgentInterface]:
        """Get all agents that are ready for tasks"""
        return [agent for agent in self.agents.values() 
                if agent.status == AgentStatus.READY]
    
    def shutdown_all(self):
        """Shutdown all registered agents"""
        for agent in self.agents.values():
            agent.shutdown()
        self.agents.clear()
        self.agent_registry.clear()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        total_agents = len(self.agents)
        ready_agents = len([a for a in self.agents.values() if a.status == AgentStatus.READY])
        busy_agents = len([a for a in self.agents.values() if a.status == AgentStatus.BUSY])
        error_agents = len([a for a in self.agents.values() if a.status == AgentStatus.ERROR])
        
        return {
            "total_agents": total_agents,
            "ready_agents": ready_agents,
            "busy_agents": busy_agents,
            "error_agents": error_agents,
            "system_health": "healthy" if error_agents == 0 else "degraded"
        }