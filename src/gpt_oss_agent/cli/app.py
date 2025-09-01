"""Main CLI application for GPT-OSS Agent."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..config import get_settings, setup_openai_env
from ..clients import setup_vllm_client, get_model_info
from ..core import create_agent
from ..tools import get_available_tools, check_all_tools_status
from ..utils import setup_logging, get_debug_logger
from ..exceptions import GPTOSSAgentError


console = Console()
logger = logging.getLogger(__name__)


def setup_application() -> None:
    """Set up the application with logging and configuration."""
    settings = get_settings()
    
    # Set up logging
    setup_logging(settings)
    
    # Set up OpenAI environment for vLLM compatibility
    setup_openai_env(settings)
    
    logger.info("GPT-OSS Agent application initialized")


def quick_chat(message: str, model: Optional[str] = None) -> str:
    """Quick one-off chat without creating persistent agent.
    
    Args:
        message: Message to send to agent
        model: Model to use (optional)
        
    Returns:
        Agent response
    """
    try:
        settings = get_settings()
        
        # Override model if specified
        if model:
            settings.vllm.model = model
        
        # Get available tools
        tools = get_available_tools(settings)
        
        # Create agent
        agent = create_agent(settings=settings, tools=tools)
        
        # Send message
        response = agent.chat(message)
        return response
        
    except Exception as e:
        logger.error(f"Quick chat failed: {e}")
        return f"Error: {str(e)}"


def test_agent() -> bool:
    """Test agent functionality.
    
    Returns:
        True if test passed, False otherwise
    """
    try:
        console.print("🧪 Testing agent functionality...", style="yellow")
        
        # Test vLLM connection
        client = setup_vllm_client()
        test_result = client.test_connection()
        
        if not test_result["health_check_passed"]:
            console.print("❌ vLLM server connection failed", style="red")
            return False
        
        console.print("✅ vLLM server connection successful", style="green")
        
        # Test agent creation and simple chat
        tools = get_available_tools()
        agent = create_agent(tools=tools)
        response = agent.chat("Hello, please introduce yourself briefly.")
        
        if response and len(response) > 10:
            console.print("✅ Agent test passed", style="green")
            console.print(f"Response preview: {response[:100]}...", style="dim")
            return True
        else:
            console.print("❌ Agent test failed - empty or invalid response", style="red")
            return False
            
    except Exception as e:
        console.print(f"❌ Agent test failed: {e}", style="red")
        logger.error(f"Agent test failed: {e}", exc_info=True)
        return False


def show_info() -> None:
    """Show system information."""
    settings = get_settings()
    
    # Create info table
    table = Table(title="GPT-OSS Agent Information")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="white")
    
    # vLLM connection
    try:
        client = setup_vllm_client(wait_for_server=False)
        model_info = get_model_info(settings)
        table.add_row(
            "vLLM Server",
            "✅ Connected" if client.health_check() else "❌ Disconnected",
            f"{settings.vllm.base_url} ({settings.vllm.model})"
        )
        table.add_row(
            "Models Available", 
            str(model_info.get("model_count", 0)),
            ", ".join(model_info.get("available_models", [])[:3])
        )
    except Exception as e:
        table.add_row("vLLM Server", "❌ Error", str(e))
    
    # Tools status
    tools_status = check_all_tools_status(settings)
    table.add_row(
        "Tools Available",
        f"{tools_status['available_tools']}/{tools_status['total_tools']}",
        ", ".join(tools_status["available_tool_names"])
    )
    
    # Configuration
    table.add_row("Log Level", settings.logging.level, "")
    table.add_row(
        "Debug Logging", 
        "✅ Enabled" if settings.debug.enabled else "❌ Disabled",
        settings.debug.log_dir if settings.debug.enabled else ""
    )
    
    console.print(table)
    
    # Debug session info
    if settings.debug.enabled:
        debug_logger = get_debug_logger()
        summary = debug_logger.get_session_summary()
        debug_panel = Panel(
            f"Session ID: {summary['session_id']}\n"
            f"Messages: {summary['message_count']}\n"
            f"Log Files: {summary['total_log_files']}\n"
            f"Log Directory: {summary['log_dir']}",
            title="Debug Session",
            style="dim"
        )
        console.print(debug_panel)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GPT-OSS Agent - AI agent for local GPT-OSS models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Interactive chat
  %(prog)s --chat "What is the weather?" # Single question
  %(prog)s --test                       # Test connection
  %(prog)s --info                       # Show system info
  %(prog)s --model gpt-oss-20b          # Use specific model
        """
    )
    
    parser.add_argument(
        "--chat", "-c",
        type=str,
        help="Send a single message and exit"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Model to use (overrides config)"
    )
    
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Test agent functionality"
    )
    
    parser.add_argument(
        "--info", "-i",
        action="store_true",
        help="Show system information"
    )
    
    parser.add_argument(
        "--log-level", "-l",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set log level"
    )
    
    
    args = parser.parse_args()
    
    try:
        # Set up application
        setup_application()
        
        # Override log level if specified
        if args.log_level:
            from ..utils import set_log_level
            set_log_level(args.log_level)
        
        # Handle different modes
        if args.test:
            success = test_agent()
            sys.exit(0 if success else 1)
        
        elif args.info:
            show_info()
            return
        
        elif args.chat:
            response = quick_chat(args.chat, model=args.model)
            console.print(response)
            return
        
        else:
            # Interactive mode - import here to avoid circular imports
            from .commands import interactive_chat
            interactive_chat(model=args.model)
    
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye!", style="yellow")
        sys.exit(0)
    
    except GPTOSSAgentError as e:
        console.print(f"❌ Agent Error: {e.message}", style="red")
        if e.details:
            console.print(f"Details: {e.details}", style="dim red")
        sys.exit(1)
    
    except Exception as e:
        console.print(f"❌ Unexpected error: {e}", style="red")
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()