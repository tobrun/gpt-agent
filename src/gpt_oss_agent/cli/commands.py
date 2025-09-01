"""Interactive CLI commands for GPT-OSS Agent."""

import logging
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.spinner import Spinner

from ..config import get_settings
from ..core import create_agent
from ..tools import get_available_tools, check_all_tools_status
from ..exceptions import GPTOSSAgentError, EmptyResponseError


console = Console()
logger = logging.getLogger(__name__)


def show_help() -> None:
    """Show help information."""
    help_text = """
# GPT-OSS Agent Commands

- `/help` - Show this help message
- `/info` - Show agent and system information  
- `/tools` - Show available tools and their status
- `/debug` - Show debug session information
- `/quit` or `/exit` - Exit the application

Just type your message to chat with the agent!
    """
    console.print(Markdown(help_text))


def show_agent_info(agent) -> None:
    """Show agent information."""
    info = agent.get_info()
    
    panel_content = f"""
**Agent Name:** {info['agent_name']}
**Model:** {info['model']}
**vLLM Server:** {info['vllm_base_url']}
**Tools:** {', '.join(info['available_tools']) if info['available_tools'] else 'None'}
**Debug Logging:** {'✅ Enabled' if info['settings']['debug_enabled'] else '❌ Disabled'}
**Web Search:** {'✅ Available' if info['settings']['web_search_enabled'] else '❌ Not configured'}
    """
    
    console.print(Panel(panel_content, title="Agent Information", style="blue"))


def show_tools_status() -> None:
    """Show tools status information."""
    status = check_all_tools_status()
    
    panel_content = f"""
**Available Tools:** {status['available_tools']}/{status['total_tools']}

**Tool Categories:**
"""
    
    for category, info in status['categories'].items():
        panel_content += f"- **{category.title()}:** {info['available']}/{info['total']} available\n"
    
    if status['available_tool_names']:
        panel_content += f"\n**Active Tools:** {', '.join(status['available_tool_names'])}"
    
    console.print(Panel(panel_content, title="Tools Status", style="green"))


def show_debug_info() -> None:
    """Show debug session information."""
    from ..utils import get_debug_logger
    
    debug_logger = get_debug_logger()
    summary = debug_logger.get_session_summary()
    
    if not summary.get("enabled"):
        console.print("Debug logging is disabled", style="dim")
        return
    
    panel_content = f"""
**Session ID:** {summary['session_id']}
**Messages Logged:** {summary['message_count']}
**Log Files:** {summary['total_log_files']}
**Log Directory:** {summary['log_dir']}

Use `python scripts/view_debug_logs.py --list` to view available sessions.
Use `python scripts/view_debug_logs.py --session {summary['session_id']}` for details.
    """
    
    console.print(Panel(panel_content, title="Debug Session", style="yellow"))


def interactive_chat(model: Optional[str] = None) -> None:
    """Start interactive chat session.
    
    Args:
        model: Model to use (optional)
    """
    console.print(Panel(
        "[bold green]🤖 GPT-OSS Agent[/bold green]\n"
        "Type '/help' for commands, '/quit' to exit",
        style="green"
    ))
    
    try:
        # Get settings and tools
        settings = get_settings()
        
        # Override model if specified
        if model:
            settings.vllm.model = model
        
        
        # Get available tools
        tools = get_available_tools(settings)
        
        if not tools:
            console.print("⚠️  No tools available - web search requires Exa API key", style="yellow")
        
        # Create agent
        console.print("Initializing agent...", style="dim")
        agent = create_agent(settings=settings, tools=tools)
        
        console.print(f"✅ Agent ready! Using model: {settings.vllm.model}", style="green")
        if tools:
            console.print(f"🔧 {len(tools)} tools available", style="dim")
        
        
        while True:
            try:
                # Get user input
                message = Prompt.ask("\n[bold cyan]You[/bold cyan]")
                
                if not message.strip():
                    continue
                
                # Handle commands
                if message.startswith('/'):
                    command = message.lower()
                    
                    if command in ['/quit', '/exit']:
                        break
                    elif command == '/help':
                        show_help()
                    elif command == '/info':
                        show_agent_info(agent)
                    elif command == '/tools':
                        show_tools_status()
                    elif command == '/debug':
                        show_debug_info()
                    else:
                        console.print(f"Unknown command: {message}", style="red")
                    continue
                
                # Send message to agent
                console.print("\n[bold green]Assistant[/bold green]")
                
                # Non-streaming response only
                with Live(Spinner("dots", text="Thinking..."), refresh_per_second=10):
                    response = agent.chat(message)
                console.print(f"\n{response}")
            
            except EmptyResponseError as e:
                console.print(f"\n⚠️  Empty response from agent: {e.message}", style="yellow")
                if e.details:
                    console.print(f"Details: {e.details}", style="dim yellow")
                console.print("This may be a vLLM compatibility issue. Try rephrasing your question.", style="dim")
            
            except GPTOSSAgentError as e:
                console.print(f"\n❌ Agent error: {e.message}", style="red")
                if e.details:
                    console.print(f"Details: {e.details}", style="dim red")
            
            except KeyboardInterrupt:
                console.print("\n⏸️  Interrupted", style="yellow")
                continue
            
            except Exception as e:
                console.print(f"\n❌ Error: {e}", style="red")
                logger.error(f"Chat error: {e}", exc_info=True)
    
    except Exception as e:
        console.print(f"❌ Failed to initialize agent: {e}", style="red")
        logger.error(f"Agent initialization failed: {e}", exc_info=True)
        return
    
    console.print("\n👋 Goodbye!", style="green")