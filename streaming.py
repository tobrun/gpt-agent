"""
Streaming response handler for the AI agent.
"""

import asyncio
import logging
import sys
from typing import AsyncGenerator, Dict, Any, Optional, Callable

from rich.console import Console
from rich.live import Live
from rich.text import Text
from rich.panel import Panel
from rich.spinner import Spinner
from rich.markdown import Markdown

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize rich console
console = Console()

class StreamHandler:
    """Handles streaming responses from the agent."""
    
    def __init__(self, show_reasoning: bool = True, show_tools: bool = True):
        """
        Initialize stream handler.
        
        Args:
            show_reasoning: Whether to display reasoning/thinking process
            show_tools: Whether to display tool invocations
        """
        self.show_reasoning = show_reasoning
        self.show_tools = show_tools
        self.buffer = ""
        self.reasoning_buffer = ""
        self.final_response = ""
        self.tool_calls = []
        self.is_complete = False
    
    async def process_stream(self, stream_result, display_callback: Optional[Callable] = None):
        """
        Process streaming results from agent.
        
        Args:
            stream_result: RunResultStreaming object from agent
            display_callback: Optional callback for custom display handling
            
        Returns:
            Final complete response
        """
        self.buffer = ""
        self.reasoning_buffer = ""
        self.final_response = ""
        self.tool_calls = []
        self.is_complete = False
        
        try:
            async for event in stream_result.stream_events():
                await self._handle_event(event, display_callback)
            
            # Ensure we have the final response
            if hasattr(stream_result, 'final_output'):
                self.final_response = stream_result.final_output
            
            self.is_complete = True
            return self.final_response
            
        except Exception as e:
            logger.error(f"Error processing stream: {e}")
            if display_callback:
                display_callback(f"\nError processing stream: {e}", "error")
            return f"Error: {str(e)}"
    
    async def _handle_event(self, event, display_callback: Optional[Callable] = None):
        """Handle individual stream events."""
        event_type = getattr(event, 'type', None)
        event_data = getattr(event, 'data', None)
        
        if not event_type:
            return
        
        # Handle different event types
        if event_type == "response.output_text.delta":
            await self._handle_text_delta(event_data, display_callback)
        
        elif event_type == "response.reasoning.delta":
            await self._handle_reasoning_delta(event_data, display_callback)
        
        elif event_type == "tool.invocation":
            await self._handle_tool_invocation(event_data, display_callback)
        
        elif event_type == "tool.result":
            await self._handle_tool_result(event_data, display_callback)
        
        elif event_type == "response.created":
            await self._handle_response_created(event_data, display_callback)
        
        elif event_type == "response.done":
            await self._handle_response_done(event_data, display_callback)
        
        else:
            # Handle unknown event types
            if display_callback:
                display_callback(f"[Unknown event: {event_type}]", "debug")
    
    async def _handle_text_delta(self, data, display_callback):
        """Handle text delta events (actual response content)."""
        if data and hasattr(data, 'delta'):
            delta = data.delta
            self.buffer += delta
            
            if display_callback:
                display_callback(delta, "response")
    
    async def _handle_reasoning_delta(self, data, display_callback):
        """Handle reasoning delta events (thinking process)."""
        if self.show_reasoning and data and hasattr(data, 'delta'):
            delta = data.delta
            self.reasoning_buffer += delta
            
            if display_callback:
                display_callback(delta, "reasoning")
    
    async def _handle_tool_invocation(self, data, display_callback):
        """Handle tool invocation events."""
        if self.show_tools and data:
            tool_name = getattr(data, 'name', 'unknown_tool')
            tool_args = getattr(data, 'arguments', {})
            
            self.tool_calls.append({
                'name': tool_name,
                'arguments': tool_args,
                'status': 'invoked'
            })
            
            if display_callback:
                display_callback(f"\n🔧 Using tool: {tool_name}", "tool")
                if tool_args:
                    display_callback(f"   Arguments: {tool_args}", "tool_detail")
    
    async def _handle_tool_result(self, data, display_callback):
        """Handle tool result events."""
        if self.show_tools and data:
            result = getattr(data, 'result', 'No result')
            
            # Update the last tool call with result
            if self.tool_calls:
                self.tool_calls[-1]['result'] = result
                self.tool_calls[-1]['status'] = 'completed'
            
            if display_callback:
                # Truncate very long results for display
                display_result = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                display_callback(f"   Result: {display_result}", "tool_result")
    
    async def _handle_response_created(self, data, display_callback):
        """Handle response creation events."""
        if display_callback:
            display_callback("", "response_start")
    
    async def _handle_response_done(self, data, display_callback):
        """Handle response completion events."""
        self.is_complete = True
        if display_callback:
            display_callback("", "response_end")


class RichStreamDisplay:
    """Rich console display for streaming responses."""
    
    def __init__(self, show_reasoning: bool = True):
        self.show_reasoning = show_reasoning
        self.response_text = Text()
        self.reasoning_text = Text()
        self.tool_text = Text()
        self.live = None
    
    async def display_stream(self, stream_result):
        """Display streaming response with rich formatting."""
        handler = StreamHandler(show_reasoning=self.show_reasoning)
        
        # Set up live display
        with Live(self._create_display_panel(), refresh_per_second=10, console=console) as live:
            self.live = live
            
            async for event in stream_result.stream_events():
                await self._handle_display_event(event)
                live.update(self._create_display_panel())
            
            # Final update
            live.update(self._create_display_panel())
        
        return handler.final_response
    
    def _create_display_panel(self) -> Panel:
        """Create the display panel for streaming."""
        content = Text()
        
        if self.reasoning_text and self.show_reasoning:
            content.append("🤔 Reasoning:\n", style="blue bold")
            content.append(self.reasoning_text)
            content.append("\n\n")
        
        if self.tool_text:
            content.append(self.tool_text)
            content.append("\n\n")
        
        if self.response_text:
            content.append("💬 Response:\n", style="green bold")
            content.append(self.response_text)
        
        return Panel(content, title="GPT-OSS Assistant", border_style="blue")
    
    async def _handle_display_event(self, event):
        """Handle events for rich display."""
        event_type = getattr(event, 'type', None)
        event_data = getattr(event, 'data', None)
        
        if event_type == "response.output_text.delta" and event_data:
            self.response_text.append(event_data.delta)
        
        elif event_type == "response.reasoning.delta" and event_data and self.show_reasoning:
            self.reasoning_text.append(event_data.delta, style="dim")
        
        elif event_type == "tool.invocation" and event_data:
            tool_name = getattr(event_data, 'name', 'unknown')
            self.tool_text.append(f"🔧 Using tool: {tool_name}\n", style="yellow")


def simple_display_callback(content: str, event_type: str):
    """Simple callback for displaying stream content."""
    if event_type == "response":
        print(content, end="", flush=True)
    elif event_type == "reasoning":
        print(f"\033[90m{content}\033[0m", end="", flush=True)  # Gray text
    elif event_type == "tool":
        print(f"\n\033[93m{content}\033[0m", flush=True)  # Yellow text
    elif event_type == "tool_detail":
        print(f"\033[93m{content}\033[0m", flush=True)  # Yellow text
    elif event_type == "tool_result":
        print(f"\033[92m{content}\033[0m", flush=True)  # Green text
    elif event_type == "response_end":
        print()  # New line at end


async def process_streaming_response(stream_result, use_rich: bool = True, show_reasoning: bool = True):
    """
    Process a streaming response from the agent.
    
    Args:
        stream_result: RunResultStreaming object
        use_rich: Whether to use rich console formatting
        show_reasoning: Whether to show reasoning process
        
    Returns:
        Final response string
    """
    if use_rich:
        display = RichStreamDisplay(show_reasoning=show_reasoning)
        return await display.display_stream(stream_result)
    else:
        handler = StreamHandler(show_reasoning=show_reasoning)
        return await handler.process_stream(stream_result, simple_display_callback)


# Synchronous wrapper for backward compatibility
def process_streaming_response_sync(stream_result, use_rich: bool = True, show_reasoning: bool = True):
    """Synchronous wrapper for processing streaming responses."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(
        process_streaming_response(stream_result, use_rich, show_reasoning)
    )