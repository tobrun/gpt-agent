"""
Command-line interface for the GPT-OSS AI agent.
"""

import os
import sys
import asyncio
import logging
import argparse
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.table import Table

from agent import create_agent, GPTOSSAgent
from streaming import process_streaming_response
from tools import check_tools_status

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce log noise in CLI
logger = logging.getLogger(__name__)

# Initialize rich console
console = Console()

class GPTOSSCli:
    """Interactive CLI for the GPT-OSS agent."""
    
    def __init__(
        self, 
        model: Optional[str] = None,
        streaming: bool = True,
        show_reasoning: bool = False,
        wait_for_server: bool = True
    ):
        """
        Initialize the CLI.
        
        Args:
            model: Model name to use
            streaming: Whether to use streaming responses
            show_reasoning: Whether to show reasoning process
            wait_for_server: Whether to wait for vLLM server
        """
        self.model = model
        self.streaming = streaming
        self.show_reasoning = show_reasoning
        self.agent: Optional[GPTOSSAgent] = None
        self.history: List[str] = []
        self.session_active = False
        
        # Initialize agent
        try:
            self.agent = create_agent(
                model=model,
                wait_for_server=wait_for_server
            )
            console.print("[green]✅ Agent initialized successfully![/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to initialize agent: {e}[/red]")
            console.print("[yellow]Make sure vLLM is running with GPT-OSS model at localhost:8000[/yellow]")
            sys.exit(1)
    
    def print_welcome(self):
        """Print welcome message and instructions."""
        welcome_text = """
# GPT-OSS AI Agent

Welcome to your local AI assistant powered by GPT-OSS!

**Available Commands:**
- `/help` - Show this help message
- `/info` - Show agent and model information  
- `/tools` - Show available tools status
- `/toggle-stream` - Toggle streaming mode
- `/toggle-reasoning` - Toggle reasoning display
- `/history` - Show conversation history
- `/clear` - Clear conversation history
- `/quit` or `/exit` - Exit the application

**Tips:**
- Type your questions or requests naturally
- Use web search by asking about current events or specific information
- Press Ctrl+C to interrupt long responses
- The agent has access to web search when EXA_API_KEY is configured

Ready to chat!
"""
        console.print(Panel(Markdown(welcome_text), title="Welcome", border_style="blue"))
    
    def print_agent_info(self):
        """Print agent configuration information."""
        if not self.agent:
            console.print("[red]Agent not initialized[/red]")
            return
            
        info = self.agent.get_info()
        
        # Create info table
        table = Table(title="Agent Information")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Agent Name", info["agent_name"])
        table.add_row("Model", info["model"])
        table.add_row("Server URL", info["vllm_server_url"])
        table.add_row("Streaming Mode", "✅ Enabled" if self.streaming else "❌ Disabled")
        table.add_row("Show Reasoning", "✅ Enabled" if self.show_reasoning else "❌ Disabled")
        
        # Tools status
        for tool_name in info["available_tools"]:
            table.add_row(f"Tool: {tool_name}", "✅ Available")
        
        tools_status = info["tools_status"]
        if not tools_status["exa_api_configured"]:
            table.add_row("Web Search", "❌ EXA_API_KEY not configured")
        
        console.print(table)
    
    def print_tools_status(self):
        """Print tools status information."""
        tools_status = check_tools_status()
        
        table = Table(title="Tools Status")
        table.add_column("Tool", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Notes")
        
        table.add_row(
            "Web Search", 
            "✅ Available" if tools_status["web_search"] else "❌ Unavailable",
            "Requires EXA_API_KEY" if not tools_status["exa_api_configured"] else "Ready"
        )
        
        table.add_row(
            "Page Content", 
            "✅ Available" if tools_status["get_page_content"] else "❌ Unavailable",
            "Requires EXA_API_KEY" if not tools_status["exa_api_configured"] else "Ready"
        )
        
        console.print(table)
        
        if not tools_status["exa_api_configured"]:
            console.print("\n[yellow]💡 To enable web search tools:[/yellow]")
            console.print("1. Get an API key from https://exa.ai")
            console.print("2. Add it to your .env file: EXA_API_KEY=your_key_here")
            console.print("3. Restart the agent")
    
    def print_history(self):
        """Print conversation history."""
        if not self.history:
            console.print("[yellow]No conversation history yet.[/yellow]")
            return
        
        console.print(Panel("Conversation History", border_style="blue"))
        for i, message in enumerate(self.history, 1):
            # Truncate long messages for history display
            display_message = message[:100] + "..." if len(message) > 100 else message
            console.print(f"[dim]{i}.[/dim] {display_message}")
    
    def handle_command(self, user_input: str) -> bool:
        """
        Handle special commands.
        
        Args:
            user_input: User input string
            
        Returns:
            True if command was handled, False if regular message
        """
        if not user_input.startswith('/'):
            return False
        
        command = user_input.lower().strip()
        
        if command in ['/quit', '/exit']:
            console.print("[yellow]👋 Goodbye![/yellow]")
            return True
        
        elif command == '/help':
            self.print_welcome()
        
        elif command == '/info':
            self.print_agent_info()
        
        elif command == '/tools':
            self.print_tools_status()
        
        elif command == '/toggle-stream':
            self.streaming = not self.streaming
            status = "enabled" if self.streaming else "disabled"
            console.print(f"[green]Streaming mode {status}[/green]")
        
        elif command == '/toggle-reasoning':
            self.show_reasoning = not self.show_reasoning
            status = "enabled" if self.show_reasoning else "disabled"
            console.print(f"[green]Reasoning display {status}[/green]")
        
        elif command == '/history':
            self.print_history()
        
        elif command == '/clear':
            self.history.clear()
            console.print("[green]Conversation history cleared[/green]")
        
        else:
            console.print(f"[red]Unknown command: {command}[/red]")
            console.print("Type /help for available commands")
        
        return True
    
    async def handle_user_message(self, message: str):
        """Handle a user message with the agent."""
        try:
            # Add to history
            self.history.append(f"User: {message}")
            
            if self.streaming:
                # Streaming response
                stream_result = self.agent.chat_stream(message)
                response = await process_streaming_response(
                    stream_result,
                    use_rich=True,
                    show_reasoning=self.show_reasoning
                )
            else:
                # Non-streaming response
                console.print("[dim]Thinking...[/dim]")
                response = self.agent.chat(message)
                console.print(Panel(Markdown(response), title="Response", border_style="green"))
            
            # Add response to history
            if response:
                self.history.append(f"Assistant: {response[:200]}..." if len(response) > 200 else f"Assistant: {response}")
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Response interrupted by user[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            logger.error(f"Error handling message: {e}")
    
    async def run_interactive(self):
        """Run the interactive CLI loop."""
        self.session_active = True
        self.print_welcome()
        
        while self.session_active:
            try:
                # Get user input
                user_input = Prompt.ask("\n[blue]You[/blue]", default="").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if self.handle_command(user_input):
                    if user_input.lower() in ['/quit', '/exit']:
                        break
                    continue
                
                # Handle regular message
                await self.handle_user_message(user_input)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Use /quit to exit[/yellow]")
            except EOFError:
                break
            except Exception as e:
                console.print(f"[red]Unexpected error: {e}[/red]")
                logger.error(f"Unexpected error in CLI loop: {e}")
        
        self.session_active = False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="GPT-OSS AI Agent CLI")
    parser.add_argument("--model", type=str, help="Model name to use")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming responses")
    parser.add_argument("--show-reasoning", action="store_true", help="Show reasoning process")
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for server to be ready")
    
    args = parser.parse_args()
    
    # Create and run CLI
    cli = GPTOSSCli(
        model=args.model,
        streaming=not args.no_stream,
        show_reasoning=args.show_reasoning,
        wait_for_server=not args.no_wait
    )
    
    try:
        asyncio.run(cli.run_interactive())
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Goodbye![/yellow]")
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()