#!/usr/bin/env python3
"""
Fork and Track Agent Repositories
Automatically forks agent repositories and sets up tracking
"""

import json
import requests
import subprocess
import os
from datetime import datetime
from typing import Dict, List

class AgentForker:
    def __init__(self, github_token: str):
        self.token = github_token
        self.headers = {
            'Authorization': f'token {github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.base_url = 'https://api.github.com'
    
    def load_registry(self) -> Dict:
        """Load agent registry from JSON file"""
        with open('agents/registry.json', 'r') as f:
            return json.load(f)
    
    def save_registry(self, registry: Dict):
        """Save updated registry to JSON file"""
        registry['metadata']['last_updated'] = datetime.now().isoformat()
        with open('agents/registry.json', 'w') as f:
            json.dump(registry, f, indent=2)
    
    def fork_repository(self, upstream_repo: str) -> bool:
        """Fork a repository to our account"""
        url = f"{self.base_url}/repos/{upstream_repo}/forks"
        response = requests.post(url, headers=self.headers)
        
        if response.status_code == 202:
            print(f"✅ Successfully forked {upstream_repo}")
            return True
        elif response.status_code == 422:
            print(f"ℹ️  Fork already exists for {upstream_repo}")
            return True
        else:
            print(f"❌ Failed to fork {upstream_repo}: {response.status_code}")
            return False
    
    def clone_to_original(self, repo_name: str, upstream_repo: str):
        """Clone forked repo to agents/original/"""
        clone_url = f"https://github.com/CrazyDubya/{repo_name}.git"
        target_dir = f"agents/original/{repo_name}"
        
        if os.path.exists(target_dir):
            print(f"ℹ️  {target_dir} already exists, pulling latest...")
            subprocess.run(['git', 'pull'], cwd=target_dir)
        else:
            print(f"📥 Cloning {repo_name} to {target_dir}")
            subprocess.run(['git', 'clone', clone_url, target_dir])
    
    def setup_enhanced_copy(self, repo_name: str):
        """Create enhanced copy for our modifications"""
        original_dir = f"agents/original/{repo_name}"
        enhanced_dir = f"agents/enhanced/{repo_name}"
        
        if not os.path.exists(enhanced_dir):
            print(f"📋 Creating enhanced copy of {repo_name}")
            subprocess.run(['cp', '-r', original_dir, enhanced_dir])
            
            # Initialize as separate git repo for our modifications
            subprocess.run(['git', 'init'], cwd=enhanced_dir)
            subprocess.run(['git', 'add', '.'], cwd=enhanced_dir)
            subprocess.run(['git', 'commit', '-m', 'Initial enhanced copy'], cwd=enhanced_dir)
    
    def create_adapter_template(self, repo_name: str, agent_info: Dict):
        """Create adapter template for universal interface"""
        adapter_dir = f"agents/adapters"
        os.makedirs(adapter_dir, exist_ok=True)
        
        adapter_content = f'''"""
Universal Adapter for {repo_name}
Provides standardized interface to {agent_info['description']}
"""

from core.interfaces.agent_interface import UniversalAgentInterface
from typing import Dict, Any, List

class {repo_name.title()}Adapter(UniversalAgentInterface):
    """Adapter for {repo_name} agent framework"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent = None
        self.capabilities = {agent_info['capabilities']}
    
    def initialize(self) -> bool:
        """Initialize the {repo_name} agent"""
        try:
            # TODO: Import and initialize {repo_name} agent
            # from agents.enhanced.{repo_name} import Agent
            # self.agent = Agent(**self.config)
            return True
        except Exception as e:
            print(f"Failed to initialize {repo_name}: {{e}}")
            return False
    
    def execute_task(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task using {repo_name}"""
        # TODO: Implement task execution
        return {{
            "status": "pending",
            "result": None,
            "agent": "{repo_name}",
            "task": task
        }}
    
    def get_capabilities(self) -> List[str]:
        """Return agent capabilities"""
        return self.capabilities
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {{
            "agent": "{repo_name}",
            "initialized": self.agent is not None,
            "capabilities": self.capabilities,
            "config": self.config
        }}
    
    def shutdown(self):
        """Cleanup and shutdown agent"""
        if self.agent:
            # TODO: Implement proper shutdown
            pass
'''
        
        with open(f"{adapter_dir}/{repo_name}_adapter.py", 'w') as f:
            f.write(adapter_content)
        
        print(f"📝 Created adapter template for {repo_name}")
    
    def process_all_agents(self):
        """Process all agents in registry"""
        registry = self.load_registry()
        
        for agent_name, agent_info in registry['agents'].items():
            print(f"\n🔄 Processing {agent_name}...")
            
            # Fork repository
            if self.fork_repository(agent_info['upstream']):
                # Clone to original
                self.clone_to_original(agent_name, agent_info['upstream'])
                
                # Create enhanced copy
                self.setup_enhanced_copy(agent_name)
                
                # Create adapter template
                self.create_adapter_template(agent_name, agent_info)
                
                # Update registry
                registry['agents'][agent_name]['fork_status'] = 'completed'
                registry['agents'][agent_name]['last_sync'] = datetime.now().isoformat()
        
        self.save_registry(registry)
        print("\n✅ All agents processed!")

if __name__ == "__main__":
    # Read GitHub token from environment or config
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        print("❌ GITHUB_TOKEN environment variable not set")
        exit(1)
    
    forker = AgentForker(token)
    forker.process_all_agents()