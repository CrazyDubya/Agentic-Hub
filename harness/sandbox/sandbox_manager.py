"""
Sandbox Manager
Provides isolated execution environments for agents with support for:
- Individual sandboxes (one per agent)
- Shared sandboxes (multiple agents collaborate)
- Snapshots and restoration
- Resource limits and isolation

Design inspired by:
- OpenAI Codex's sandbox architecture
- Manus's isolated execution environments
- Container-based isolation patterns
"""

import os
import json
import shutil
import tempfile
import subprocess
import uuid
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from enum import Enum
from pathlib import Path
import threading
import queue
from abc import ABC, abstractmethod


class SandboxType(Enum):
    """Types of sandbox environments"""
    PRIVATE = "private"      # Single agent, isolated
    SHARED = "shared"        # Multiple agents, collaborative
    READONLY = "readonly"    # Read-only access (for reference)
    EPHEMERAL = "ephemeral"  # Temporary, auto-cleanup


class SandboxStatus(Enum):
    """Sandbox lifecycle states"""
    CREATING = "creating"
    READY = "ready"
    BUSY = "busy"
    SUSPENDED = "suspended"
    DESTROYING = "destroying"
    DESTROYED = "destroyed"
    ERROR = "error"


class IsolationLevel(Enum):
    """Level of isolation for sandboxes"""
    NONE = "none"           # No isolation (development mode)
    PROCESS = "process"     # Subprocess isolation
    CONTAINER = "container" # Docker/container isolation
    VM = "vm"              # Full VM isolation (most secure)


@dataclass
class ResourceLimits:
    """Resource constraints for sandbox"""
    cpu_cores: float = 1.0
    memory_mb: int = 512
    disk_mb: int = 1024
    network_enabled: bool = True
    max_processes: int = 10
    timeout_seconds: int = 300
    allowed_ports: List[int] = field(default_factory=list)


@dataclass
class SandboxSnapshot:
    """Point-in-time snapshot of sandbox state"""
    snapshot_id: str
    sandbox_id: str
    created_at: datetime
    description: str
    file_manifest: Dict[str, str]  # path -> hash
    metadata: Dict[str, Any]
    size_bytes: int


@dataclass
class FileChange:
    """Represents a file change in sandbox"""
    path: str
    change_type: str  # "create", "modify", "delete"
    timestamp: datetime
    agent_id: str
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None


@dataclass
class Sandbox:
    """
    Represents an isolated execution environment.
    Can be private (single agent) or shared (multiple agents).
    """
    sandbox_id: str
    name: str
    sandbox_type: SandboxType
    owner_id: str
    created_at: datetime
    status: SandboxStatus
    root_path: Path
    isolation_level: IsolationLevel
    resource_limits: ResourceLimits
    allowed_agents: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    snapshots: List[str] = field(default_factory=list)
    change_history: List[FileChange] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.root_path, str):
            self.root_path = Path(self.root_path)
        self.allowed_agents.add(self.owner_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sandbox_id": self.sandbox_id,
            "name": self.name,
            "type": self.sandbox_type.value,
            "owner_id": self.owner_id,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "root_path": str(self.root_path),
            "isolation_level": self.isolation_level.value,
            "resource_limits": {
                "cpu_cores": self.resource_limits.cpu_cores,
                "memory_mb": self.resource_limits.memory_mb,
                "disk_mb": self.resource_limits.disk_mb,
                "network_enabled": self.resource_limits.network_enabled
            },
            "allowed_agents": list(self.allowed_agents),
            "metadata": self.metadata,
            "snapshots": self.snapshots
        }


class SandboxBackend(ABC):
    """Abstract backend for sandbox execution"""

    @abstractmethod
    def create_environment(self, sandbox: Sandbox) -> bool:
        pass

    @abstractmethod
    def destroy_environment(self, sandbox: Sandbox) -> bool:
        pass

    @abstractmethod
    def execute(self, sandbox: Sandbox, command: str, timeout: int = 30) -> Dict[str, Any]:
        pass

    @abstractmethod
    def copy_file(self, sandbox: Sandbox, src: str, dst: str) -> bool:
        pass


class LocalSandboxBackend(SandboxBackend):
    """
    Local filesystem-based sandbox backend.
    Uses subprocess isolation with resource monitoring.
    """

    def create_environment(self, sandbox: Sandbox) -> bool:
        try:
            sandbox.root_path.mkdir(parents=True, exist_ok=True)

            # Create standard directories
            (sandbox.root_path / "workspace").mkdir(exist_ok=True)
            (sandbox.root_path / "shared").mkdir(exist_ok=True)
            (sandbox.root_path / "tmp").mkdir(exist_ok=True)
            (sandbox.root_path / ".snapshots").mkdir(exist_ok=True)
            (sandbox.root_path / ".meta").mkdir(exist_ok=True)

            # Write sandbox metadata
            meta_file = sandbox.root_path / ".meta" / "sandbox.json"
            with open(meta_file, 'w') as f:
                json.dump(sandbox.to_dict(), f, indent=2)

            return True
        except Exception as e:
            print(f"Failed to create sandbox: {e}")
            return False

    def destroy_environment(self, sandbox: Sandbox) -> bool:
        try:
            if sandbox.root_path.exists():
                shutil.rmtree(sandbox.root_path)
            return True
        except Exception as e:
            print(f"Failed to destroy sandbox: {e}")
            return False

    def execute(self, sandbox: Sandbox, command: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute command within sandbox context"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(sandbox.root_path / "workspace"),
                capture_output=True,
                text=True,
                timeout=min(timeout, sandbox.resource_limits.timeout_seconds),
                env={
                    **os.environ,
                    "SANDBOX_ID": sandbox.sandbox_id,
                    "SANDBOX_ROOT": str(sandbox.root_path),
                    "SANDBOX_WORKSPACE": str(sandbox.root_path / "workspace"),
                    "SANDBOX_SHARED": str(sandbox.root_path / "shared"),
                    "HOME": str(sandbox.root_path / "workspace"),
                }
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Command timed out",
                "return_code": -1
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "return_code": -1
            }

    def copy_file(self, sandbox: Sandbox, src: str, dst: str) -> bool:
        try:
            src_path = Path(src)
            dst_path = sandbox.root_path / "workspace" / dst
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            if src_path.is_file():
                shutil.copy2(src_path, dst_path)
            elif src_path.is_dir():
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

            return True
        except Exception as e:
            print(f"Copy failed: {e}")
            return False


class DockerSandboxBackend(SandboxBackend):
    """
    Docker container-based sandbox backend.
    Provides stronger isolation than local backend.
    """

    def __init__(self, base_image: str = "python:3.11-slim"):
        self.base_image = base_image
        self.containers: Dict[str, str] = {}  # sandbox_id -> container_id

    def create_environment(self, sandbox: Sandbox) -> bool:
        try:
            # Create host directory for volume mount
            sandbox.root_path.mkdir(parents=True, exist_ok=True)

            # Build container command
            container_name = f"sandbox-{sandbox.sandbox_id}"
            cmd = [
                "docker", "run", "-d",
                "--name", container_name,
                "--memory", f"{sandbox.resource_limits.memory_mb}m",
                "--cpus", str(sandbox.resource_limits.cpu_cores),
                "-v", f"{sandbox.root_path}:/workspace",
                "-w", "/workspace",
            ]

            if not sandbox.resource_limits.network_enabled:
                cmd.extend(["--network", "none"])

            cmd.extend([self.base_image, "tail", "-f", "/dev/null"])

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                self.containers[sandbox.sandbox_id] = container_name
                return True
            return False
        except Exception as e:
            print(f"Docker sandbox creation failed: {e}")
            return False

    def destroy_environment(self, sandbox: Sandbox) -> bool:
        try:
            container_name = self.containers.get(sandbox.sandbox_id)
            if container_name:
                subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
                del self.containers[sandbox.sandbox_id]

            if sandbox.root_path.exists():
                shutil.rmtree(sandbox.root_path)
            return True
        except Exception as e:
            print(f"Docker sandbox destruction failed: {e}")
            return False

    def execute(self, sandbox: Sandbox, command: str, timeout: int = 30) -> Dict[str, Any]:
        container_name = self.containers.get(sandbox.sandbox_id)
        if not container_name:
            return {"success": False, "stderr": "Container not found", "stdout": "", "return_code": -1}

        try:
            result = subprocess.run(
                ["docker", "exec", container_name, "sh", "-c", command],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "stderr": "Timeout", "stdout": "", "return_code": -1}

    def copy_file(self, sandbox: Sandbox, src: str, dst: str) -> bool:
        container_name = self.containers.get(sandbox.sandbox_id)
        if not container_name:
            return False

        try:
            result = subprocess.run(
                ["docker", "cp", src, f"{container_name}:/workspace/{dst}"],
                capture_output=True
            )
            return result.returncode == 0
        except:
            return False


class SandboxManager:
    """
    Manages all sandboxes in the system.
    Handles creation, destruction, sharing, and snapshotting.
    """

    def __init__(
        self,
        base_path: str = "/tmp/agentic-sandboxes",
        isolation_level: IsolationLevel = IsolationLevel.PROCESS
    ):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.isolation_level = isolation_level
        self.sandboxes: Dict[str, Sandbox] = {}
        self.agent_sandboxes: Dict[str, Set[str]] = {}  # agent_id -> sandbox_ids
        self.snapshots: Dict[str, SandboxSnapshot] = {}

        # Select backend based on isolation level
        if isolation_level == IsolationLevel.CONTAINER:
            self.backend = DockerSandboxBackend()
        else:
            self.backend = LocalSandboxBackend()

        self._lock = threading.Lock()
        self._event_queue = queue.Queue()

    def create_sandbox(
        self,
        owner_id: str,
        name: Optional[str] = None,
        sandbox_type: SandboxType = SandboxType.PRIVATE,
        resource_limits: Optional[ResourceLimits] = None,
        template: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Sandbox]:
        """Create a new sandbox"""
        with self._lock:
            sandbox_id = str(uuid.uuid4())[:8]
            name = name or f"sandbox-{sandbox_id}"

            sandbox = Sandbox(
                sandbox_id=sandbox_id,
                name=name,
                sandbox_type=sandbox_type,
                owner_id=owner_id,
                created_at=datetime.now(),
                status=SandboxStatus.CREATING,
                root_path=self.base_path / sandbox_id,
                isolation_level=self.isolation_level,
                resource_limits=resource_limits or ResourceLimits(),
                metadata=metadata or {}
            )

            if self.backend.create_environment(sandbox):
                sandbox.status = SandboxStatus.READY
                self.sandboxes[sandbox_id] = sandbox

                # Track agent's sandboxes
                if owner_id not in self.agent_sandboxes:
                    self.agent_sandboxes[owner_id] = set()
                self.agent_sandboxes[owner_id].add(sandbox_id)

                # Apply template if specified
                if template:
                    self._apply_template(sandbox, template)

                return sandbox

            return None

    def destroy_sandbox(self, sandbox_id: str, agent_id: str) -> bool:
        """Destroy a sandbox (only owner can destroy)"""
        with self._lock:
            sandbox = self.sandboxes.get(sandbox_id)
            if not sandbox:
                return False

            if sandbox.owner_id != agent_id:
                return False  # Only owner can destroy

            sandbox.status = SandboxStatus.DESTROYING

            if self.backend.destroy_environment(sandbox):
                sandbox.status = SandboxStatus.DESTROYED

                # Clean up references
                del self.sandboxes[sandbox_id]
                for agent_sboxes in self.agent_sandboxes.values():
                    agent_sboxes.discard(sandbox_id)

                return True

            sandbox.status = SandboxStatus.ERROR
            return False

    def share_sandbox(
        self,
        sandbox_id: str,
        owner_id: str,
        target_agent_id: str,
        readonly: bool = False
    ) -> bool:
        """Share sandbox access with another agent"""
        with self._lock:
            sandbox = self.sandboxes.get(sandbox_id)
            if not sandbox:
                return False

            if sandbox.owner_id != owner_id:
                return False

            sandbox.allowed_agents.add(target_agent_id)

            if target_agent_id not in self.agent_sandboxes:
                self.agent_sandboxes[target_agent_id] = set()
            self.agent_sandboxes[target_agent_id].add(sandbox_id)

            # Convert to shared type if not already
            if sandbox.sandbox_type == SandboxType.PRIVATE:
                sandbox.sandbox_type = SandboxType.SHARED

            sandbox.metadata.setdefault("access_control", {})[target_agent_id] = {
                "readonly": readonly,
                "granted_at": datetime.now().isoformat()
            }

            return True

    def revoke_access(
        self,
        sandbox_id: str,
        owner_id: str,
        target_agent_id: str
    ) -> bool:
        """Revoke sandbox access from an agent"""
        with self._lock:
            sandbox = self.sandboxes.get(sandbox_id)
            if not sandbox:
                return False

            if sandbox.owner_id != owner_id:
                return False

            if target_agent_id == owner_id:
                return False  # Can't revoke own access

            sandbox.allowed_agents.discard(target_agent_id)

            if target_agent_id in self.agent_sandboxes:
                self.agent_sandboxes[target_agent_id].discard(sandbox_id)

            if "access_control" in sandbox.metadata:
                sandbox.metadata["access_control"].pop(target_agent_id, None)

            return True

    def get_sandbox(self, sandbox_id: str, agent_id: str) -> Optional[Sandbox]:
        """Get sandbox if agent has access"""
        sandbox = self.sandboxes.get(sandbox_id)
        if sandbox and agent_id in sandbox.allowed_agents:
            return sandbox
        return None

    def get_agent_sandboxes(self, agent_id: str) -> List[Sandbox]:
        """Get all sandboxes an agent has access to"""
        sandbox_ids = self.agent_sandboxes.get(agent_id, set())
        return [
            self.sandboxes[sid]
            for sid in sandbox_ids
            if sid in self.sandboxes
        ]

    def execute_in_sandbox(
        self,
        sandbox_id: str,
        agent_id: str,
        command: str,
        timeout: int = 30
    ) -> Dict[str, Any]:
        """Execute command in sandbox"""
        sandbox = self.get_sandbox(sandbox_id, agent_id)
        if not sandbox:
            return {"success": False, "error": "Access denied or sandbox not found"}

        if sandbox.status != SandboxStatus.READY:
            return {"success": False, "error": f"Sandbox not ready: {sandbox.status.value}"}

        # Check readonly access
        if agent_id != sandbox.owner_id:
            access_control = sandbox.metadata.get("access_control", {})
            agent_access = access_control.get(agent_id, {})
            if agent_access.get("readonly"):
                # Only allow read operations
                readonly_commands = ["cat", "ls", "head", "tail", "grep", "find", "tree"]
                cmd_start = command.strip().split()[0] if command.strip() else ""
                if cmd_start not in readonly_commands:
                    return {"success": False, "error": "Readonly access - write operations not permitted"}

        return self.backend.execute(sandbox, command, timeout)

    def create_snapshot(
        self,
        sandbox_id: str,
        agent_id: str,
        description: str = ""
    ) -> Optional[SandboxSnapshot]:
        """Create a point-in-time snapshot of sandbox"""
        sandbox = self.get_sandbox(sandbox_id, agent_id)
        if not sandbox or sandbox.owner_id != agent_id:
            return None

        snapshot_id = str(uuid.uuid4())[:8]
        snapshot_path = sandbox.root_path / ".snapshots" / snapshot_id

        try:
            # Create snapshot directory
            snapshot_path.mkdir(parents=True)

            # Copy workspace
            workspace = sandbox.root_path / "workspace"
            shutil.copytree(workspace, snapshot_path / "workspace")

            # Build file manifest
            manifest = {}
            total_size = 0
            for file_path in (snapshot_path / "workspace").rglob("*"):
                if file_path.is_file():
                    rel_path = str(file_path.relative_to(snapshot_path / "workspace"))
                    file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
                    manifest[rel_path] = file_hash
                    total_size += file_path.stat().st_size

            snapshot = SandboxSnapshot(
                snapshot_id=snapshot_id,
                sandbox_id=sandbox_id,
                created_at=datetime.now(),
                description=description,
                file_manifest=manifest,
                metadata={"agent_id": agent_id},
                size_bytes=total_size
            )

            # Save snapshot metadata
            with open(snapshot_path / "snapshot.json", 'w') as f:
                json.dump({
                    "snapshot_id": snapshot.snapshot_id,
                    "sandbox_id": snapshot.sandbox_id,
                    "created_at": snapshot.created_at.isoformat(),
                    "description": snapshot.description,
                    "file_manifest": snapshot.file_manifest,
                    "metadata": snapshot.metadata,
                    "size_bytes": snapshot.size_bytes
                }, f, indent=2)

            sandbox.snapshots.append(snapshot_id)
            self.snapshots[snapshot_id] = snapshot

            return snapshot

        except Exception as e:
            print(f"Snapshot failed: {e}")
            if snapshot_path.exists():
                shutil.rmtree(snapshot_path)
            return None

    def restore_snapshot(
        self,
        sandbox_id: str,
        agent_id: str,
        snapshot_id: str
    ) -> bool:
        """Restore sandbox to a previous snapshot"""
        sandbox = self.get_sandbox(sandbox_id, agent_id)
        if not sandbox or sandbox.owner_id != agent_id:
            return False

        if snapshot_id not in sandbox.snapshots:
            return False

        snapshot_path = sandbox.root_path / ".snapshots" / snapshot_id
        if not snapshot_path.exists():
            return False

        try:
            # Clear current workspace
            workspace = sandbox.root_path / "workspace"
            shutil.rmtree(workspace)

            # Restore from snapshot
            shutil.copytree(snapshot_path / "workspace", workspace)

            sandbox.change_history.append(FileChange(
                path="*",
                change_type="restore",
                timestamp=datetime.now(),
                agent_id=agent_id,
                new_hash=snapshot_id
            ))

            return True

        except Exception as e:
            print(f"Restore failed: {e}")
            return False

    def _apply_template(self, sandbox: Sandbox, template: str) -> bool:
        """Apply a template to sandbox"""
        templates = {
            "python": [
                ("requirements.txt", "# Python dependencies\n"),
                ("main.py", "#!/usr/bin/env python3\n\ndef main():\n    pass\n\nif __name__ == '__main__':\n    main()\n"),
                (".gitignore", "__pycache__/\n*.pyc\n.env\nvenv/\n"),
            ],
            "node": [
                ("package.json", '{\n  "name": "sandbox",\n  "version": "1.0.0",\n  "main": "index.js"\n}\n'),
                ("index.js", "// Entry point\nconsole.log('Hello from sandbox');\n"),
                (".gitignore", "node_modules/\n.env\n"),
            ],
            "web": [
                ("index.html", "<!DOCTYPE html>\n<html>\n<head>\n  <title>Sandbox</title>\n</head>\n<body>\n  <h1>Hello from Sandbox</h1>\n</body>\n</html>\n"),
                ("style.css", "/* Styles */\nbody { font-family: sans-serif; }\n"),
                ("script.js", "// JavaScript\nconsole.log('Loaded');\n"),
            ],
            "empty": [],
        }

        if template not in templates:
            return False

        try:
            workspace = sandbox.root_path / "workspace"
            for filename, content in templates[template]:
                file_path = workspace / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content)
            return True
        except:
            return False

    def list_shared_with_agent(self, agent_id: str) -> List[Dict[str, Any]]:
        """List all sandboxes shared with an agent (that they don't own)"""
        result = []
        for sid in self.agent_sandboxes.get(agent_id, set()):
            sandbox = self.sandboxes.get(sid)
            if sandbox and sandbox.owner_id != agent_id:
                result.append({
                    "sandbox_id": sandbox.sandbox_id,
                    "name": sandbox.name,
                    "owner_id": sandbox.owner_id,
                    "type": sandbox.sandbox_type.value,
                    "readonly": sandbox.metadata.get("access_control", {}).get(agent_id, {}).get("readonly", False)
                })
        return result

    def get_global_shared_sandboxes(self) -> List[Dict[str, Any]]:
        """Get sandboxes that are marked as globally shared"""
        result = []
        for sandbox in self.sandboxes.values():
            if sandbox.metadata.get("global_shared"):
                result.append(sandbox.to_dict())
        return result

    def cleanup_ephemeral(self, max_age_hours: int = 24):
        """Clean up old ephemeral sandboxes"""
        now = datetime.now()
        to_delete = []

        for sandbox_id, sandbox in self.sandboxes.items():
            if sandbox.sandbox_type == SandboxType.EPHEMERAL:
                age = (now - sandbox.created_at).total_seconds() / 3600
                if age > max_age_hours:
                    to_delete.append((sandbox_id, sandbox.owner_id))

        for sandbox_id, owner_id in to_delete:
            self.destroy_sandbox(sandbox_id, owner_id)
