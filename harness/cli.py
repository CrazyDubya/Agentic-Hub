#!/usr/bin/env python3
"""
Universal LLM Agent Harness - Command Line Interface

A comprehensive CLI for interacting with the harness, managing agents,
and running tasks in various modes.

Usage:
    harness interactive [--agent NAME] [--mode MODE]
    harness run TASK [--agent NAME] [--autonomous]
    harness server [--port PORT]
    harness agent create [--name NAME] [--capabilities CAPS]
    harness agent list
    harness skill list [--category CAT]
    harness skill install SKILL_ID
    harness sandbox list [--agent AGENT]
    harness sandbox create [--name NAME] [--template TEMPLATE]
    harness eval export [--format FORMAT] [--output PATH]
    harness status
"""

import argparse
import sys
import json
import os
import signal
import readline
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from harness import (
    UniversalLLMHarness,
    HarnessMode,
    LLMBackend,
    AgentState,
    SandboxType,
)
from harness.evaluation import (
    SelfEvaluationLoop,
    FlywheelManager,
    QAGenerationSystem,
    create_evaluation_system,
    create_qa_system
)


# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


def colorize(text: str, color: str) -> str:
    """Add color to text if terminal supports it"""
    if sys.stdout.isatty():
        return f"{color}{text}{Colors.ENDC}"
    return text


def print_banner():
    """Print the harness banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║     Universal LLM Agent Harness                               ║
║     A game engine for AI agents                               ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(colorize(banner, Colors.CYAN))


def print_status(harness: UniversalLLMHarness):
    """Print harness status"""
    status = harness.get_status()
    print(colorize("\n📊 Harness Status:", Colors.BOLD))
    print(f"  Agents:      {status['agents']}")
    print(f"  Sandboxes:   {status['active_sandboxes']}")
    print(f"  Skills:      {status['registered_skills']}")
    print(f"  Marketplace: {status['marketplace_assets']} assets")
    print(f"  Messages:    {status['message_bus'].get('pending_messages', 0)} pending")


def print_agent_status(state: AgentState):
    """Print agent status"""
    print(colorize(f"\n🤖 Agent: {state.agent_id}", Colors.GREEN))
    print(f"  Session:     {state.session_id}")
    print(f"  Created:     {state.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Commands:    {state.commands_executed}")
    print(f"  Skills used: {state.skills_invoked}")
    print(f"  Messages:    {state.messages_sent}")
    if state.todos:
        print(f"  Todos:       {len(state.todos)}")


class CLILLMBackend(LLMBackend):
    """
    CLI-based LLM backend that prompts user for responses.
    Useful for testing and human-in-the-loop scenarios.
    """

    def __init__(self, mock_mode: bool = False):
        self.mock_mode = mock_mode

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        if self.mock_mode:
            return {
                "content": "[Mock response - enable real LLM for actual responses]",
                "tool_calls": []
            }

        # Show context to user
        print(colorize("\n--- LLM Context ---", Colors.DIM))
        for msg in messages[-3:]:  # Show last 3 messages
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:200]
            print(f"{role}: {content}...")
        print(colorize("--- End Context ---\n", Colors.DIM))

        # Prompt for response
        print(colorize("Enter agent response (end with blank line):", Colors.YELLOW))
        lines = []
        while True:
            try:
                line = input()
                if line == "":
                    break
                lines.append(line)
            except EOFError:
                break

        return {
            "content": "\n".join(lines),
            "tool_calls": []
        }

    def count_tokens(self, text: str) -> int:
        return len(text) // 4

    @property
    def supports_tools(self) -> bool:
        return False

    @property
    def max_context_tokens(self) -> int:
        return 128000


class HarnessCLI:
    """Main CLI handler"""

    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path or os.path.expanduser("~/.llm-harness"))
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.config_path = self.base_path / "config.json"
        self.config = self._load_config()

        self.harness: Optional[UniversalLLMHarness] = None
        self.current_agent_id: Optional[str] = None

        # Evaluation systems
        self.eval_loop: Optional[SelfEvaluationLoop] = None
        self.flywheel: Optional[FlywheelManager] = None
        self.qa_system: Optional[QAGenerationSystem] = None

    def _load_config(self) -> Dict[str, Any]:
        """Load CLI configuration"""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return json.load(f)
        return {
            "default_mode": "interactive",
            "llm_backend": "cli",  # or "openai", "anthropic", etc.
            "base_path": str(self.base_path),
            "auto_evaluate": True,
            "history_file": str(self.base_path / "history")
        }

    def _save_config(self):
        """Save CLI configuration"""
        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=2)

    def initialize(self, mode: HarnessMode = HarnessMode.INTERACTIVE, mock: bool = False):
        """Initialize the harness"""
        print(colorize("Initializing harness...", Colors.DIM))

        # Create LLM backend based on config
        llm_backend = CLILLMBackend(mock_mode=mock)

        # Initialize harness
        self.harness = UniversalLLMHarness(
            llm_backend=llm_backend,
            base_path=str(self.base_path / "harness"),
            harness_mode=mode
        )

        # Initialize evaluation systems
        eval_path = str(self.base_path / "evaluation")
        self.eval_loop, self.flywheel = create_evaluation_system(eval_path)
        self.qa_system = create_qa_system(str(self.base_path / "qa"))

        print(colorize("✓ Harness initialized", Colors.GREEN))

    def create_agent(
        self,
        name: str = None,
        capabilities: List[str] = None
    ) -> str:
        """Create a new agent"""
        if not self.harness:
            self.initialize()

        session_id, state = self.harness.create_agent(
            name=name or f"agent-{datetime.now().strftime('%H%M%S')}",
            capabilities=capabilities or ["general"]
        )

        self.current_agent_id = state.agent_id
        print(colorize(f"✓ Created agent: {state.agent_id}", Colors.GREEN))
        print(f"  Session: {session_id}")

        return state.agent_id

    def interactive_mode(self, agent_id: str = None):
        """Run interactive REPL"""
        if not self.harness:
            self.initialize(HarnessMode.INTERACTIVE)

        if not agent_id:
            agent_id = self.create_agent()

        self.current_agent_id = agent_id
        state = self.harness.get_agent(agent_id)

        print_banner()
        print_agent_status(state)
        print(colorize("\nCommands: /help, /status, /todos, /skills, /agents, /quit", Colors.DIM))
        print(colorize("Enter your task or message:\n", Colors.CYAN))

        # Setup history
        history_file = self.config.get("history_file")
        if history_file:
            try:
                readline.read_history_file(history_file)
            except FileNotFoundError:
                pass

        while True:
            try:
                # Prompt
                prompt = colorize(f"[{agent_id}] > ", Colors.GREEN)
                user_input = input(prompt).strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    if self._handle_command(user_input):
                        continue
                    else:
                        break

                # Execute turn
                print(colorize("\nProcessing...", Colors.DIM))
                result = self.harness.execute_turn(agent_id, user_input)

                # Display response
                print(colorize("\n📝 Response:", Colors.CYAN))
                print(result.get("final_response", "[No response]"))

                # Show command results if any
                if result.get("results"):
                    print(colorize("\n⚙️  Commands executed:", Colors.DIM))
                    for r in result["results"][:5]:
                        status = "✓" if r.get("success") else "✗"
                        print(f"  {status} {r.get('command_id', 'unknown')}")

                print()  # Blank line

            except KeyboardInterrupt:
                print(colorize("\n\nInterrupted. Use /quit to exit.", Colors.YELLOW))
            except EOFError:
                break

        # Save history
        if history_file:
            readline.write_history_file(history_file)

        print(colorize("\nGoodbye! 👋", Colors.CYAN))

    def _handle_command(self, cmd: str) -> bool:
        """Handle CLI commands. Returns True to continue, False to exit."""
        parts = cmd.split()
        command = parts[0].lower()

        if command == "/quit" or command == "/exit":
            return False

        elif command == "/help":
            self._show_help()

        elif command == "/status":
            print_status(self.harness)
            if self.current_agent_id:
                state = self.harness.get_agent(self.current_agent_id)
                if state:
                    print_agent_status(state)

        elif command == "/todos":
            self._show_todos()

        elif command == "/skills":
            self._list_skills(parts[1] if len(parts) > 1 else None)

        elif command == "/agents":
            self._list_agents()

        elif command == "/sandbox":
            if len(parts) > 1 and parts[1] == "create":
                name = parts[2] if len(parts) > 2 else None
                self._create_sandbox(name)
            else:
                self._list_sandboxes()

        elif command == "/eval":
            self._show_evaluation()

        elif command == "/export":
            output = parts[1] if len(parts) > 1 else "training_data.jsonl"
            self._export_data(output)

        elif command == "/clear":
            os.system('clear' if os.name == 'posix' else 'cls')

        elif command == "/mode":
            if len(parts) > 1:
                self._set_mode(parts[1])
            else:
                print(f"Current mode: {self.harness.harness_mode.value}")

        elif command == "/compact":
            if self.current_agent_id:
                result = self.harness.compact_context(self.current_agent_id)
                print(colorize(f"✓ Context compacted (#{result.get('compaction_count', 0)})", Colors.GREEN))

        elif command == "/msg":
            if len(parts) >= 3:
                to_agent = parts[1]
                message = " ".join(parts[2:])
                self._send_message(to_agent, message)
            else:
                print("Usage: /msg <agent_id> <message>")

        else:
            print(colorize(f"Unknown command: {command}", Colors.YELLOW))
            print("Use /help for available commands")

        return True

    def _show_help(self):
        """Show help text"""
        help_text = """
Available Commands:
  /help      - Show this help
  /status    - Show harness and agent status
  /todos     - Show and manage todo list
  /skills    - List available skills
  /agents    - List all agents
  /sandbox   - List sandboxes
  /sandbox create [name] - Create a sandbox
  /eval      - Show evaluation metrics
  /export [file] - Export training data
  /mode [mode] - Get/set harness mode
  /compact   - Compact agent context
  /msg <agent> <msg> - Send message to agent
  /clear     - Clear screen
  /quit      - Exit

Text Commands (for non-tool LLMs):
  You can also use text-based commands:
  ```command:file.read
  path: /path/to/file
  ```

  Or natural language:
  "Read the file at /path/to/file"
  "Run 'npm test' in the terminal"
"""
        print(colorize(help_text, Colors.CYAN))

    def _show_todos(self):
        """Show agent todos"""
        if not self.current_agent_id:
            print("No active agent")
            return

        state = self.harness.get_agent(self.current_agent_id)
        if not state or not state.todos:
            print("No todos")
            return

        print(colorize("\n📋 Todo List:", Colors.BOLD))
        for i, todo in enumerate(state.todos, 1):
            status = todo.get("status", "pending")
            icon = "✓" if status == "completed" else "○" if status == "pending" else "◔"
            content = todo.get("content", "Unknown")
            print(f"  {icon} {i}. {content}")

    def _list_skills(self, category: str = None):
        """List available skills"""
        skills = self.harness.skill_registry.list_skills()

        if category:
            skills = [s for s in skills if s.category == category]

        print(colorize(f"\n🔧 Skills ({len(skills)}):", Colors.BOLD))
        for skill in skills[:20]:
            print(f"  • {skill.skill_id}: {skill.description[:50]}...")

        if len(skills) > 20:
            print(f"  ... and {len(skills) - 20} more")

    def _list_agents(self):
        """List all agents"""
        agents = self.harness.agent_directory.list_all()

        print(colorize(f"\n🤖 Agents ({len(agents)}):", Colors.BOLD))
        for agent in agents:
            active = "●" if agent.status == "active" else "○"
            current = " (current)" if agent.agent_id == self.current_agent_id else ""
            print(f"  {active} {agent.agent_id}: {agent.name}{current}")

    def _list_sandboxes(self):
        """List sandboxes"""
        if not self.current_agent_id:
            print("No active agent")
            return

        sandboxes = self.harness.sandbox_manager.get_agent_sandboxes(self.current_agent_id)

        print(colorize(f"\n📦 Sandboxes ({len(sandboxes)}):", Colors.BOLD))
        for sb in sandboxes:
            print(f"  • {sb.sandbox_id}: {sb.name} ({sb.sandbox_type.value})")
            print(f"    Path: {sb.root_path}")

    def _create_sandbox(self, name: str = None):
        """Create a sandbox"""
        if not self.current_agent_id:
            print("No active agent")
            return

        sandbox = self.harness.sandbox_manager.create_sandbox(
            owner_id=self.current_agent_id,
            name=name or f"sandbox-{datetime.now().strftime('%H%M%S')}"
        )

        if sandbox:
            print(colorize(f"✓ Created sandbox: {sandbox.sandbox_id}", Colors.GREEN))
            print(f"  Path: {sandbox.root_path}")
        else:
            print(colorize("✗ Failed to create sandbox", Colors.RED))

    def _show_evaluation(self):
        """Show evaluation metrics"""
        if not self.flywheel:
            print("Evaluation system not initialized")
            return

        status = self.flywheel.get_flywheel_status()
        metrics = status["metrics"]

        print(colorize("\n📊 Flywheel Status:", Colors.BOLD))
        print(f"  Traces recorded:      {metrics['traces_recorded']}")
        print(f"  Evaluations:          {metrics['evaluations_performed']}")
        print(f"  Lessons extracted:    {metrics['lessons_extracted']}")
        print(f"  Training data:        {metrics['training_data_generated']}")
        print(f"  Average score:        {status['average_score']:.2f}")
        print(f"  Trend:                {status['trend_direction']}")
        print(f"  Health:               {status['health']}")

    def _export_data(self, output: str):
        """Export training data"""
        if not self.qa_system:
            print("Q&A system not initialized")
            return

        count = self.qa_system.export_training_data(
            output,
            style="openai",
            min_score=0.5
        )

        print(colorize(f"✓ Exported {count} Q&A pairs to {output}", Colors.GREEN))

    def _set_mode(self, mode_str: str):
        """Set harness mode"""
        mode_map = {
            "interactive": HarnessMode.INTERACTIVE,
            "autonomous": HarnessMode.AUTONOMOUS,
            "collaborative": HarnessMode.COLLABORATIVE,
            "supervised": HarnessMode.SUPERVISED
        }

        mode = mode_map.get(mode_str.lower())
        if mode:
            self.harness.harness_mode = mode
            print(colorize(f"✓ Mode set to: {mode.value}", Colors.GREEN))
        else:
            print(f"Unknown mode: {mode_str}")
            print(f"Available: {', '.join(mode_map.keys())}")

    def _send_message(self, to_agent: str, message: str):
        """Send message to another agent"""
        from harness.communication import Message, MessageType
        import uuid

        msg = Message(
            message_id=str(uuid.uuid4())[:12],
            message_type=MessageType.DIRECT,
            sender_id=self.current_agent_id,
            recipient_id=to_agent,
            content=message
        )

        self.harness.message_bus.send(msg)
        print(colorize(f"✓ Message sent to {to_agent}", Colors.GREEN))

    def run_task(self, task: str, agent_id: str = None, autonomous: bool = False):
        """Run a single task"""
        if not self.harness:
            mode = HarnessMode.AUTONOMOUS if autonomous else HarnessMode.INTERACTIVE
            self.initialize(mode)

        if not agent_id:
            agent_id = self.create_agent()

        print(colorize(f"\n🚀 Running task: {task[:50]}...", Colors.CYAN))

        result = self.harness.execute_turn(
            agent_id,
            task,
            max_iterations=10 if autonomous else 1
        )

        print(colorize("\n📝 Result:", Colors.GREEN))
        print(result.get("final_response", "[No response]"))

        if result.get("iterations", 0) > 1:
            print(colorize(f"\n⚙️  Completed in {result['iterations']} iterations", Colors.DIM))

        return result

    def server_mode(self, host: str = "127.0.0.1", port: int = 8080):
        """Run in server mode (HTTP API)"""
        try:
            from http.server import HTTPServer, BaseHTTPRequestHandler
        except ImportError:
            print("HTTP server not available")
            return

        if not self.harness:
            self.initialize(HarnessMode.COLLABORATIVE)

        cli = self

        class HarnessHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)

                try:
                    data = json.loads(post_data)
                    path = self.path

                    if path == "/agent/create":
                        agent_id = cli.create_agent(
                            name=data.get("name"),
                            capabilities=data.get("capabilities")
                        )
                        response = {"agent_id": agent_id}

                    elif path == "/task/run":
                        result = cli.run_task(
                            data.get("task", ""),
                            data.get("agent_id"),
                            data.get("autonomous", False)
                        )
                        response = result

                    elif path == "/message/send":
                        cli._send_message(data.get("to"), data.get("message"))
                        response = {"sent": True}

                    else:
                        response = {"error": "Unknown endpoint"}

                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(response).encode())

                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode())

            def do_GET(self):
                path = self.path

                if path == "/status":
                    response = cli.harness.get_status()
                elif path == "/agents":
                    agents = cli.harness.agent_directory.list_all()
                    response = [a.to_dict() for a in agents]
                elif path == "/skills":
                    skills = cli.harness.skill_registry.list_skills()
                    response = [s.to_dict() for s in skills]
                else:
                    response = {"message": "Universal LLM Harness API"}

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())

            def log_message(self, format, *args):
                print(colorize(f"[API] {args[0]}", Colors.DIM))

        server = HTTPServer((host, port), HarnessHandler)
        print(colorize(f"\n🌐 Server running at http://{host}:{port}", Colors.GREEN))
        print("Endpoints:")
        print("  GET  /status         - Harness status")
        print("  GET  /agents         - List agents")
        print("  GET  /skills         - List skills")
        print("  POST /agent/create   - Create agent")
        print("  POST /task/run       - Run task")
        print("  POST /message/send   - Send message")
        print(colorize("\nPress Ctrl+C to stop", Colors.DIM))

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print(colorize("\n\nServer stopped", Colors.YELLOW))
            server.shutdown()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Universal LLM Agent Harness CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s interactive                  # Start interactive mode
  %(prog)s run "Analyze this code"      # Run a single task
  %(prog)s server --port 8080           # Start API server
  %(prog)s agent create --name MyAgent  # Create a named agent
  %(prog)s skill list                   # List available skills
  %(prog)s eval export --output data.jsonl  # Export training data
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Interactive mode
    interactive_parser = subparsers.add_parser("interactive", help="Start interactive mode")
    interactive_parser.add_argument("--agent", "-a", help="Agent ID to use")
    interactive_parser.add_argument("--mode", "-m", choices=["interactive", "autonomous", "supervised"],
                                    default="interactive", help="Harness mode")
    interactive_parser.add_argument("--mock", action="store_true", help="Use mock LLM backend")

    # Run task
    run_parser = subparsers.add_parser("run", help="Run a single task")
    run_parser.add_argument("task", help="Task to run")
    run_parser.add_argument("--agent", "-a", help="Agent ID to use")
    run_parser.add_argument("--autonomous", action="store_true", help="Run in autonomous mode")
    run_parser.add_argument("--mock", action="store_true", help="Use mock LLM backend")

    # Server mode
    server_parser = subparsers.add_parser("server", help="Start API server")
    server_parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    server_parser.add_argument("--port", "-p", type=int, default=8080, help="Port to bind")

    # Agent commands
    agent_parser = subparsers.add_parser("agent", help="Agent management")
    agent_sub = agent_parser.add_subparsers(dest="agent_command")

    create_parser = agent_sub.add_parser("create", help="Create agent")
    create_parser.add_argument("--name", "-n", help="Agent name")
    create_parser.add_argument("--capabilities", "-c", nargs="+", help="Agent capabilities")

    agent_sub.add_parser("list", help="List agents")

    # Skill commands
    skill_parser = subparsers.add_parser("skill", help="Skill management")
    skill_sub = skill_parser.add_subparsers(dest="skill_command")

    list_skill_parser = skill_sub.add_parser("list", help="List skills")
    list_skill_parser.add_argument("--category", "-c", help="Filter by category")

    install_parser = skill_sub.add_parser("install", help="Install skill")
    install_parser.add_argument("skill_id", help="Skill ID to install")

    # Sandbox commands
    sandbox_parser = subparsers.add_parser("sandbox", help="Sandbox management")
    sandbox_sub = sandbox_parser.add_subparsers(dest="sandbox_command")

    sandbox_sub.add_parser("list", help="List sandboxes")

    create_sb_parser = sandbox_sub.add_parser("create", help="Create sandbox")
    create_sb_parser.add_argument("--name", "-n", help="Sandbox name")
    create_sb_parser.add_argument("--template", "-t", help="Template to use")

    # Evaluation commands
    eval_parser = subparsers.add_parser("eval", help="Evaluation and training")
    eval_sub = eval_parser.add_subparsers(dest="eval_command")

    eval_sub.add_parser("status", help="Show evaluation status")

    export_parser = eval_sub.add_parser("export", help="Export training data")
    export_parser.add_argument("--output", "-o", default="training_data.jsonl", help="Output file")
    export_parser.add_argument("--format", "-f", choices=["alpaca", "sharegpt", "openai"],
                               default="openai", help="Output format")
    export_parser.add_argument("--min-score", type=float, default=0.5, help="Minimum score threshold")

    # Status
    subparsers.add_parser("status", help="Show harness status")

    # Parse arguments
    args = parser.parse_args()

    # Initialize CLI
    cli = HarnessCLI()

    # Handle commands
    if args.command == "interactive":
        mode = HarnessMode.AUTONOMOUS if args.mode == "autonomous" else HarnessMode.INTERACTIVE
        cli.initialize(mode, mock=args.mock)
        cli.interactive_mode(args.agent)

    elif args.command == "run":
        cli.initialize(mock=args.mock)
        cli.run_task(args.task, args.agent, args.autonomous)

    elif args.command == "server":
        cli.server_mode(args.host, args.port)

    elif args.command == "agent":
        cli.initialize(mock=True)
        if args.agent_command == "create":
            cli.create_agent(args.name, args.capabilities)
        elif args.agent_command == "list":
            cli._list_agents()
        else:
            agent_parser.print_help()

    elif args.command == "skill":
        cli.initialize(mock=True)
        if args.skill_command == "list":
            cli._list_skills(args.category)
        elif args.skill_command == "install":
            success, msg = cli.harness.marketplace.install(args.skill_id)
            print(colorize(f"{'✓' if success else '✗'} {msg}", Colors.GREEN if success else Colors.RED))
        else:
            skill_parser.print_help()

    elif args.command == "sandbox":
        cli.initialize(mock=True)
        cli.create_agent()  # Need an agent for sandbox ops
        if args.sandbox_command == "list":
            cli._list_sandboxes()
        elif args.sandbox_command == "create":
            cli._create_sandbox(args.name)
        else:
            sandbox_parser.print_help()

    elif args.command == "eval":
        cli.initialize(mock=True)
        if args.eval_command == "status":
            cli._show_evaluation()
        elif args.eval_command == "export":
            cli.qa_system.export_training_data(
                args.output,
                style=args.format,
                min_score=args.min_score
            )
            print(colorize(f"✓ Exported to {args.output}", Colors.GREEN))
        else:
            eval_parser.print_help()

    elif args.command == "status":
        cli.initialize(mock=True)
        print_banner()
        print_status(cli.harness)

    else:
        # Default to interactive mode
        parser.print_help()
        print(colorize("\nTip: Run 'harness interactive' to start interactive mode", Colors.CYAN))


if __name__ == "__main__":
    main()
