"""
Agent Directory and Marketplace
Provides discovery, registration, and distribution of:
- Agent capabilities
- Skills and tools
- Templates and workflows
- Shared resources

Key features:
- Agent discovery and capability matching
- Skill/tool marketplace
- Version management
- Ratings and reviews
- Dependency resolution
"""

import json
import hashlib
import requests
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime
from enum import Enum
from pathlib import Path
import threading
import sqlite3
from abc import ABC, abstractmethod


class AssetType(Enum):
    """Types of marketplace assets"""
    SKILL = "skill"
    WORKFLOW = "workflow"
    TEMPLATE = "template"
    AGENT = "agent"
    TOOL = "tool"
    MODEL_CONFIG = "model_config"
    PROMPT = "prompt"


class AgentCapabilityLevel(Enum):
    """Capability levels for agents"""
    BASIC = "basic"         # Simple tasks, limited autonomy
    INTERMEDIATE = "intermediate"  # Multi-step tasks
    ADVANCED = "advanced"   # Complex reasoning, tool use
    EXPERT = "expert"       # Full autonomy, multi-agent coordination


@dataclass
class AgentProfile:
    """Profile for a registered agent"""
    agent_id: str
    name: str
    model_type: str  # e.g., "claude-3-opus", "gpt-4", "llama-70b"
    capability_level: AgentCapabilityLevel
    capabilities: List[str]
    supports_tools: bool
    max_context_tokens: int
    specializations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    status: str = "active"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "model_type": self.model_type,
            "capability_level": self.capability_level.value,
            "capabilities": self.capabilities,
            "supports_tools": self.supports_tools,
            "max_context_tokens": self.max_context_tokens,
            "specializations": self.specializations,
            "metadata": self.metadata,
            "status": self.status
        }


@dataclass
class MarketplaceAsset:
    """An asset available in the marketplace"""
    asset_id: str
    asset_type: AssetType
    name: str
    version: str
    description: str
    author: str
    license: str
    tags: List[str]
    downloads: int
    rating: float
    rating_count: int
    dependencies: List[str]
    created_at: datetime
    updated_at: datetime
    source_url: Optional[str] = None
    documentation_url: Optional[str] = None
    install_command: Optional[str] = None
    size_bytes: int = 0
    verified: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "type": self.asset_type.value,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "license": self.license,
            "tags": self.tags,
            "downloads": self.downloads,
            "rating": self.rating,
            "rating_count": self.rating_count,
            "verified": self.verified
        }


@dataclass
class Review:
    """User review of a marketplace asset"""
    review_id: str
    asset_id: str
    agent_id: str
    rating: int  # 1-5
    title: str
    content: str
    created_at: datetime
    helpful_count: int = 0


class AgentDirectory:
    """
    Directory of all registered agents in the system.
    Enables discovery and capability matching.
    """

    def __init__(self, db_path: str = None):
        self.agents: Dict[str, AgentProfile] = {}
        self.capability_index: Dict[str, Set[str]] = {}  # capability -> agent_ids
        self.specialization_index: Dict[str, Set[str]] = {}
        self._lock = threading.Lock()

        if db_path:
            self._init_db(db_path)

    def _init_db(self, db_path: str):
        """Initialize SQLite database for persistence"""
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute('''
            CREATE TABLE IF NOT EXISTS agents (
                agent_id TEXT PRIMARY KEY,
                profile_json TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        ''')
        self.db.commit()

    def register(self, profile: AgentProfile) -> bool:
        """Register an agent in the directory"""
        with self._lock:
            self.agents[profile.agent_id] = profile

            # Update capability index
            for cap in profile.capabilities:
                if cap not in self.capability_index:
                    self.capability_index[cap] = set()
                self.capability_index[cap].add(profile.agent_id)

            # Update specialization index
            for spec in profile.specializations:
                if spec not in self.specialization_index:
                    self.specialization_index[spec] = set()
                self.specialization_index[spec].add(profile.agent_id)

            return True

    def unregister(self, agent_id: str) -> bool:
        """Remove an agent from the directory"""
        with self._lock:
            if agent_id not in self.agents:
                return False

            profile = self.agents[agent_id]

            # Clean up indexes
            for cap in profile.capabilities:
                if cap in self.capability_index:
                    self.capability_index[cap].discard(agent_id)

            for spec in profile.specializations:
                if spec in self.specialization_index:
                    self.specialization_index[spec].discard(agent_id)

            del self.agents[agent_id]
            return True

    def get(self, agent_id: str) -> Optional[AgentProfile]:
        """Get an agent profile by ID"""
        return self.agents.get(agent_id)

    def find_by_capability(
        self,
        capability: str,
        min_level: AgentCapabilityLevel = None
    ) -> List[AgentProfile]:
        """Find agents with a specific capability"""
        agent_ids = self.capability_index.get(capability, set())
        results = []

        for aid in agent_ids:
            profile = self.agents.get(aid)
            if profile:
                if min_level and profile.capability_level.value < min_level.value:
                    continue
                results.append(profile)

        return results

    def find_by_specialization(self, specialization: str) -> List[AgentProfile]:
        """Find agents with a specific specialization"""
        agent_ids = self.specialization_index.get(specialization, set())
        return [self.agents[aid] for aid in agent_ids if aid in self.agents]

    def match_task(
        self,
        required_capabilities: List[str],
        preferred_specializations: List[str] = None,
        require_tools: bool = False,
        min_context: int = None
    ) -> List[Tuple[AgentProfile, float]]:
        """
        Find agents that match task requirements.
        Returns list of (profile, match_score) tuples, sorted by score.
        """
        candidates = []

        for profile in self.agents.values():
            if profile.status != "active":
                continue

            # Check tool requirement
            if require_tools and not profile.supports_tools:
                continue

            # Check context requirement
            if min_context and profile.max_context_tokens < min_context:
                continue

            # Calculate match score
            score = 0

            # Capability matching (required)
            matched_caps = set(profile.capabilities) & set(required_capabilities)
            if len(matched_caps) < len(required_capabilities):
                continue  # Missing required capabilities

            score += len(matched_caps) * 10

            # Specialization matching (preferred)
            if preferred_specializations:
                matched_specs = set(profile.specializations) & set(preferred_specializations)
                score += len(matched_specs) * 5

            # Capability level bonus
            level_scores = {
                AgentCapabilityLevel.BASIC: 1,
                AgentCapabilityLevel.INTERMEDIATE: 2,
                AgentCapabilityLevel.ADVANCED: 3,
                AgentCapabilityLevel.EXPERT: 4
            }
            score += level_scores.get(profile.capability_level, 0) * 2

            candidates.append((profile, score))

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def list_all(
        self,
        status: str = None,
        capability_level: AgentCapabilityLevel = None
    ) -> List[AgentProfile]:
        """List all agents with optional filtering"""
        results = []
        for profile in self.agents.values():
            if status and profile.status != status:
                continue
            if capability_level and profile.capability_level != capability_level:
                continue
            results.append(profile)
        return results

    def update_status(self, agent_id: str, status: str) -> bool:
        """Update agent status"""
        if agent_id in self.agents:
            self.agents[agent_id].status = status
            self.agents[agent_id].last_seen = datetime.now()
            return True
        return False

    def get_capabilities(self) -> List[str]:
        """Get list of all known capabilities"""
        return list(self.capability_index.keys())

    def get_specializations(self) -> List[str]:
        """Get list of all known specializations"""
        return list(self.specialization_index.keys())


class Marketplace:
    """
    Marketplace for skills, tools, workflows, and other assets.
    Supports both local and remote registries.
    """

    def __init__(self, local_path: str = None, remote_url: str = None):
        self.local_path = Path(local_path) if local_path else None
        self.remote_url = remote_url
        self.assets: Dict[str, MarketplaceAsset] = {}
        self.reviews: Dict[str, List[Review]] = {}
        self.installed: Dict[str, str] = {}  # asset_id -> version
        self._lock = threading.Lock()

        if self.local_path:
            self._init_local_cache()

    def _init_local_cache(self):
        """Initialize local cache directory"""
        self.local_path.mkdir(parents=True, exist_ok=True)
        (self.local_path / "assets").mkdir(exist_ok=True)
        (self.local_path / "installed").mkdir(exist_ok=True)

        # Load cached assets
        cache_file = self.local_path / "cache.json"
        if cache_file.exists():
            with open(cache_file) as f:
                data = json.load(f)
                for asset_data in data.get("assets", []):
                    self._load_asset(asset_data)

    def _load_asset(self, data: Dict[str, Any]) -> MarketplaceAsset:
        """Load asset from dict"""
        return MarketplaceAsset(
            asset_id=data["asset_id"],
            asset_type=AssetType(data["type"]),
            name=data["name"],
            version=data["version"],
            description=data["description"],
            author=data["author"],
            license=data.get("license", "MIT"),
            tags=data.get("tags", []),
            downloads=data.get("downloads", 0),
            rating=data.get("rating", 0.0),
            rating_count=data.get("rating_count", 0),
            dependencies=data.get("dependencies", []),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(),
            source_url=data.get("source_url"),
            verified=data.get("verified", False),
            metadata=data.get("metadata", {})
        )

    def search(
        self,
        query: str = None,
        asset_type: AssetType = None,
        tags: List[str] = None,
        min_rating: float = None,
        verified_only: bool = False,
        sort_by: str = "relevance"  # relevance, downloads, rating, recent
    ) -> List[MarketplaceAsset]:
        """Search marketplace assets"""

        # If remote, fetch from server
        if self.remote_url and not self.assets:
            self._refresh_from_remote()

        results = []
        query_lower = query.lower() if query else None

        for asset in self.assets.values():
            # Apply filters
            if asset_type and asset.asset_type != asset_type:
                continue

            if verified_only and not asset.verified:
                continue

            if min_rating and asset.rating < min_rating:
                continue

            if tags and not any(t in asset.tags for t in tags):
                continue

            # Calculate relevance score
            score = 0
            if query_lower:
                if query_lower in asset.name.lower():
                    score += 10
                if query_lower in asset.description.lower():
                    score += 5
                if any(query_lower in tag.lower() for tag in asset.tags):
                    score += 3

                if score == 0:
                    continue  # No match

            asset_with_score = (asset, score)
            results.append(asset_with_score)

        # Sort results
        if sort_by == "downloads":
            results.sort(key=lambda x: x[0].downloads, reverse=True)
        elif sort_by == "rating":
            results.sort(key=lambda x: x[0].rating, reverse=True)
        elif sort_by == "recent":
            results.sort(key=lambda x: x[0].updated_at, reverse=True)
        else:  # relevance
            results.sort(key=lambda x: x[1], reverse=True)

        return [r[0] for r in results]

    def get(self, asset_id: str) -> Optional[MarketplaceAsset]:
        """Get asset by ID"""
        return self.assets.get(asset_id)

    def install(
        self,
        asset_id: str,
        version: str = None,
        target_dir: str = None
    ) -> Tuple[bool, str]:
        """Install an asset"""
        asset = self.get(asset_id)
        if not asset:
            return False, f"Asset not found: {asset_id}"

        version = version or asset.version
        target = Path(target_dir) if target_dir else self.local_path / "installed" / asset_id

        try:
            # Resolve dependencies first
            for dep in asset.dependencies:
                if dep not in self.installed:
                    success, msg = self.install(dep)
                    if not success:
                        return False, f"Failed to install dependency {dep}: {msg}"

            # Download/copy asset
            if asset.source_url:
                self._download_asset(asset, target)
            elif self.local_path:
                source = self.local_path / "assets" / asset_id
                if source.exists():
                    import shutil
                    shutil.copytree(source, target, dirs_exist_ok=True)

            self.installed[asset_id] = version
            asset.downloads += 1

            return True, f"Installed {asset.name} v{version}"

        except Exception as e:
            return False, str(e)

    def uninstall(self, asset_id: str) -> Tuple[bool, str]:
        """Uninstall an asset"""
        if asset_id not in self.installed:
            return False, f"Asset not installed: {asset_id}"

        try:
            target = self.local_path / "installed" / asset_id
            if target.exists():
                import shutil
                shutil.rmtree(target)

            del self.installed[asset_id]
            return True, f"Uninstalled {asset_id}"

        except Exception as e:
            return False, str(e)

    def publish(self, asset: MarketplaceAsset, content_path: str) -> Tuple[bool, str]:
        """Publish an asset to the marketplace"""
        with self._lock:
            if asset.asset_id in self.assets:
                # Update existing
                existing = self.assets[asset.asset_id]
                if asset.version <= existing.version:
                    return False, "Version must be higher than existing"

            self.assets[asset.asset_id] = asset

            # Copy content to local cache
            if self.local_path:
                import shutil
                target = self.local_path / "assets" / asset.asset_id
                shutil.copytree(content_path, target, dirs_exist_ok=True)

            # Push to remote if available
            if self.remote_url:
                self._push_to_remote(asset, content_path)

            return True, f"Published {asset.name} v{asset.version}"

    def add_review(
        self,
        asset_id: str,
        agent_id: str,
        rating: int,
        title: str,
        content: str
    ) -> bool:
        """Add a review for an asset"""
        if asset_id not in self.assets:
            return False

        review = Review(
            review_id=hashlib.sha256(f"{asset_id}-{agent_id}-{datetime.now()}".encode()).hexdigest()[:12],
            asset_id=asset_id,
            agent_id=agent_id,
            rating=max(1, min(5, rating)),
            title=title,
            content=content,
            created_at=datetime.now()
        )

        if asset_id not in self.reviews:
            self.reviews[asset_id] = []
        self.reviews[asset_id].append(review)

        # Update asset rating
        asset = self.assets[asset_id]
        total_rating = sum(r.rating for r in self.reviews[asset_id])
        asset.rating = total_rating / len(self.reviews[asset_id])
        asset.rating_count = len(self.reviews[asset_id])

        return True

    def get_reviews(self, asset_id: str) -> List[Review]:
        """Get reviews for an asset"""
        return self.reviews.get(asset_id, [])

    def list_installed(self) -> Dict[str, str]:
        """List installed assets and their versions"""
        return dict(self.installed)

    def check_updates(self) -> List[Tuple[str, str, str]]:
        """Check for updates to installed assets"""
        updates = []
        for asset_id, installed_version in self.installed.items():
            asset = self.get(asset_id)
            if asset and asset.version > installed_version:
                updates.append((asset_id, installed_version, asset.version))
        return updates

    def _refresh_from_remote(self):
        """Refresh asset cache from remote"""
        if not self.remote_url:
            return

        try:
            response = requests.get(f"{self.remote_url}/assets")
            if response.status_code == 200:
                data = response.json()
                for asset_data in data.get("assets", []):
                    asset = self._load_asset(asset_data)
                    self.assets[asset.asset_id] = asset
        except Exception as e:
            print(f"Failed to refresh from remote: {e}")

    def _download_asset(self, asset: MarketplaceAsset, target: Path):
        """Download asset from source URL"""
        if not asset.source_url:
            raise ValueError("No source URL")

        response = requests.get(asset.source_url)
        response.raise_for_status()

        target.mkdir(parents=True, exist_ok=True)
        # Handle different content types...

    def _push_to_remote(self, asset: MarketplaceAsset, content_path: str):
        """Push asset to remote marketplace"""
        if not self.remote_url:
            return

        # Implementation would POST to remote API


# Built-in assets for bootstrapping
BUILTIN_ASSETS = [
    {
        "asset_id": "skill.file-ops",
        "type": "skill",
        "name": "File Operations",
        "version": "1.0.0",
        "description": "Core file reading, writing, and manipulation skills",
        "author": "system",
        "license": "MIT",
        "tags": ["core", "files", "io"],
        "downloads": 0,
        "rating": 5.0,
        "rating_count": 0,
        "dependencies": [],
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
        "verified": True
    },
    {
        "asset_id": "skill.code-analysis",
        "type": "skill",
        "name": "Code Analysis",
        "version": "1.0.0",
        "description": "Analyze code for bugs, style issues, and improvements",
        "author": "system",
        "license": "MIT",
        "tags": ["core", "code", "analysis"],
        "downloads": 0,
        "rating": 5.0,
        "rating_count": 0,
        "dependencies": ["skill.file-ops"],
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
        "verified": True
    },
    {
        "asset_id": "workflow.code-review",
        "type": "workflow",
        "name": "Code Review Workflow",
        "version": "1.0.0",
        "description": "Automated code review with multiple analysis passes",
        "author": "system",
        "license": "MIT",
        "tags": ["workflow", "code-review", "automation"],
        "downloads": 0,
        "rating": 4.8,
        "rating_count": 0,
        "dependencies": ["skill.code-analysis", "skill.file-ops"],
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
        "verified": True
    },
    {
        "asset_id": "template.python-project",
        "type": "template",
        "name": "Python Project Template",
        "version": "1.0.0",
        "description": "Standard Python project structure with testing and CI",
        "author": "system",
        "license": "MIT",
        "tags": ["template", "python", "project"],
        "downloads": 0,
        "rating": 4.5,
        "rating_count": 0,
        "dependencies": [],
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
        "verified": True
    }
]
