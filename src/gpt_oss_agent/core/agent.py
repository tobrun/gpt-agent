"""Core agent implementation using OpenAI Agents SDK."""

import logging
from typing import Any, Dict, List, Optional

from agents import Agent, Runner, set_tracing_disabled

from ..config import Settings, get_settings
from ..exceptions import AgentError, EmptyResponseError
from .instructions import get_default_instructions


logger = logging.getLogger(__name__)


class GPTOSSAgent:
    """Main AI agent powered by local GPT-OSS model via vLLM.
    
    This agent connects to a locally-hosted vLLM server and provides
    conversational AI capabilities with optional tool integration.
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        tools: Optional[List[Any]] = None,
        instructions: Optional[str] = None,
    ):
        """Initialize the GPT-OSS agent.
        
        Args:
            settings: Configuration settings (uses global if None)
            tools: List of tools to make available to the agent
            instructions: Custom instructions (uses default if None)
        """
        self.settings = settings or get_settings()
        self.tools = tools or []
        
        # Disable OpenAI tracing completely for local operation
        set_tracing_disabled(True)
        
        # Get instructions
        if instructions is None:
            instructions = self._get_default_instructions()
        
        # Create the agent
        self.agent = Agent(
            name=self.settings.agent.name,
            instructions=instructions,
            tools=self.tools,
            model=self.settings.vllm.model
        )
        
        logger.info(f"Agent initialized with model: {self.settings.vllm.model}")
        logger.info(f"Available tools: {self._get_tool_names()}")
    
    def _get_default_instructions(self) -> str:
        """Get default instructions for the agent."""
        has_web_search = any(
            hasattr(tool, '__name__') and tool.__name__ in ['web_search', 'get_page_content']
            for tool in self.tools
        )
        return get_default_instructions(has_web_search=has_web_search)
    
    def _get_tool_names(self) -> List[str]:
        """Get list of tool names."""
        names = []
        for tool in self.tools:
            name = getattr(tool, 'name', getattr(tool, '__name__', str(tool)))
            names.append(name)
        return names
    
    def chat(self, message: str, **kwargs) -> str:
        """Send a synchronous message to the agent.
        
        Args:
            message: User message
            **kwargs: Additional arguments for Runner.run_sync
            
        Returns:
            Agent response as string
            
        Raises:
            AgentError: If the agent encounters an error
            EmptyResponseError: If the agent returns no output
        """
        try:
            logger.info(f"Starting chat with message: {message[:50]}...")
            
            # Run the agent
            result = Runner.run_sync(
                self.agent,
                message,
                **kwargs
            )
            
            logger.info(f"Runner completed. Result type: {type(result)}")
            
            # Extract response
            response = self._extract_response(result)
            
            if not response:
                # Create detailed debug information
                debug_info = {
                    "result_type": str(type(result)),
                    "has_final_output": hasattr(result, 'final_output'),
                    "final_output_value": getattr(result, 'final_output', None),
                    "has_new_items": hasattr(result, 'new_items'),
                    "new_items_count": len(result.new_items) if hasattr(result, 'new_items') else 0,
                    "result_attributes": list(result.__dict__.keys()) if hasattr(result, '__dict__') else [],
                    "result_str": str(result)[:500]
                }
                
                if hasattr(result, 'new_items') and result.new_items:
                    debug_info["new_items_details"] = []
                    for i, item in enumerate(result.new_items[:3]):  # First 3 items
                        item_debug = {
                            "index": i,
                            "type": getattr(item, 'type', 'unknown'),
                            "attributes": list(item.__dict__.keys()) if hasattr(item, '__dict__') else [],
                            "str_repr": str(item)[:200]
                        }
                        debug_info["new_items_details"].append(item_debug)
                
                logger.error(f"Empty response debug info: {debug_info}")
                
                raise EmptyResponseError(
                    details=f"Result had {debug_info['new_items_count']} new_items. Debug info logged."
                )
            
            return response
            
        except Exception as e:
            if isinstance(e, (AgentError, EmptyResponseError)):
                raise
            logger.error(f"Error in chat: {e}", exc_info=True)
            raise AgentError(f"Chat failed: {str(e)}")
    
    def _extract_response(self, result: Any) -> str:
        """Extract response from Runner result with detailed debugging.
        
        Args:
            result: RunResult object from OpenAI Agents SDK
            
        Returns:
            Response string or empty string if none found
        """
        logger.debug(f"Extracting response from result type: {type(result)}")
        
        # Detailed debugging of the result object
        if hasattr(result, '__dict__'):
            logger.debug(f"Result attributes: {list(result.__dict__.keys())}")
        
        # Check if there are any raw_responses early
        if hasattr(result, 'raw_responses') and result.raw_responses:
            logger.debug(f"Result has {len(result.raw_responses)} raw responses")
        
        # Try final_output first
        if hasattr(result, 'final_output'):
            logger.debug(f"final_output exists: {result.final_output}")
            if result.final_output:
                return str(result.final_output)
        
        # Try alternative output attributes
        for attr in ['output', 'outputs', 'response', 'content']:
            if hasattr(result, attr):
                output = getattr(result, attr)
                logger.debug(f"{attr} exists: {type(output)} - {repr(output)[:200]}")
                if output:
                    if isinstance(output, list) and output:
                        output = output[-1]
                    if output:
                        return str(output)
        
        # Try extracting from new_items with detailed logging
        if hasattr(result, 'new_items') and result.new_items:
            logger.debug(f"new_items count: {len(result.new_items)}")
            
            # Look for message output in reverse order (most recent first)
            for i, item in enumerate(reversed(result.new_items)):
                actual_index = len(result.new_items) - 1 - i
                logger.debug(f"Item {actual_index}: type={getattr(item, 'type', 'no_type')}, attrs={list(item.__dict__.keys()) if hasattr(item, '__dict__') else 'no_dict'}")
                
                # Look for message output items
                if hasattr(item, 'type') and item.type == 'message_output_item':
                    logger.debug(f"Found message_output_item at index {actual_index}")
                    if hasattr(item, 'raw_item') and hasattr(item.raw_item, 'content'):
                        logger.debug(f"Content items: {len(item.raw_item.content)}")
                        for j, content in enumerate(item.raw_item.content):
                            logger.debug(f"Content {j}: type={type(content)}, has_text={hasattr(content, 'text')}")
                            if hasattr(content, 'text') and content.text:
                                logger.debug(f"Found text content: {repr(content.text)[:100]}")
                                return content.text
                
                # Look for assistant message items (alternative format)
                elif hasattr(item, 'type') and 'message' in item.type:
                    logger.debug(f"Found message item: {item.type}")
                    if hasattr(item, 'content'):
                        logger.debug(f"Direct content: {repr(item.content)[:100]}")
                        if item.content:
                            return str(item.content)
                
                # Look for reasoning items that might contain the actual response
                elif hasattr(item, 'type') and item.type == 'reasoning_item':
                    logger.debug(f"Found reasoning_item at index {actual_index}")
                    if hasattr(item, 'raw_item'):
                        raw_item = item.raw_item
                        logger.debug(f"Raw item type: {type(raw_item)}")
                        
                        # Check if raw_item has content
                        if hasattr(raw_item, 'content'):
                            logger.debug(f"Reasoning content items: {len(raw_item.content) if hasattr(raw_item.content, '__len__') else 'not_list'}")
                            
                            # Handle content as list
                            if hasattr(raw_item.content, '__iter__') and not isinstance(raw_item.content, str):
                                for j, content in enumerate(raw_item.content):
                                    logger.debug(f"Reasoning content {j}: type={type(content)}")
                                    if hasattr(content, 'text') and content.text and content.text.strip():
                                        # Only return reasoning if it looks like a response, not just reasoning
                                        text = content.text.strip()
                                        logger.debug(f"Reasoning text content: {repr(text)[:200]}")
                                        if not text.startswith("I need to") and not text.startswith("Let me") and len(text) > 50 and not text.startswith("{"):
                                            logger.debug(f"Found response in reasoning: {repr(text)[:100]}")
                                            return text
                            
                            # Handle content as string
                            elif isinstance(raw_item.content, str) and raw_item.content.strip():
                                text = raw_item.content.strip()
                                logger.debug(f"Reasoning string content: {repr(text)[:200]}")
                                if not text.startswith("I need to") and not text.startswith("Let me") and len(text) > 50 and not text.startswith("{"):
                                    logger.debug(f"Found response in reasoning content: {repr(text)[:100]}")
                                    return text
                
                # Fallback: look at any item with content
                if hasattr(item, 'content') and item.content:
                    logger.debug(f"Item {actual_index} has direct content: {repr(item.content)[:100]}")
                    if isinstance(item.content, str) and item.content.strip():
                        return item.content
        
        # Final fallback: try to extract from raw_responses
        if hasattr(result, 'raw_responses') and result.raw_responses:
            logger.debug(f"Checking raw_responses: {len(result.raw_responses)}")
            for i, response in enumerate(result.raw_responses):
                logger.debug(f"Raw response {i}: type={type(response)}")
                logger.debug(f"Raw response attributes: {list(response.__dict__.keys()) if hasattr(response, '__dict__') else 'no_dict'}")
                
                if hasattr(response, 'choices') and response.choices:
                    logger.debug(f"Response has {len(response.choices)} choices")
                    choice = response.choices[0]
                    logger.debug(f"Choice type: {type(choice)}, attrs: {list(choice.__dict__.keys()) if hasattr(choice, '__dict__') else 'no_dict'}")
                    
                    if hasattr(choice, 'message') and choice.message:
                        logger.debug(f"Message type: {type(choice.message)}, attrs: {list(choice.message.__dict__.keys()) if hasattr(choice.message, '__dict__') else 'no_dict'}")
                        if hasattr(choice.message, 'content') and choice.message.content:
                            content = choice.message.content.strip()
                            logger.debug(f"Found content in raw response: {repr(content)[:200]}")
                            if content:
                                return content
                
                # Also check if response itself has content
                if hasattr(response, 'content') and response.content:
                    logger.debug(f"Response has direct content: {repr(response.content)[:200]}")
                    if response.content.strip():
                        return response.content
        
        # Final fallback: try to extract any text from the result
        result_str = str(result)
        logger.debug(f"Result string representation: {repr(result_str)[:200]}")
        
        return ""
    
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the agent configuration."""
        return {
            "agent_name": self.agent.name,
            "model": self.settings.vllm.model,
            "vllm_base_url": self.settings.vllm.base_url,
            "available_tools": self._get_tool_names(),
            "instructions_length": len(self.agent.instructions),
            "settings": {
                "debug_enabled": self.settings.debug.enabled,
                "web_search_enabled": bool(self.settings.exa.api_key and self.settings.exa.enabled),
            }
        }
    
    def update_instructions(self, new_instructions: str) -> None:
        """Update agent instructions.
        
        Args:
            new_instructions: New instruction string
        """
        self.agent.instructions = new_instructions
        logger.info("Agent instructions updated")
    
    def add_tool(self, tool_func: Any) -> None:
        """Add a new tool to the agent.
        
        Args:
            tool_func: Tool function to add
        """
        if tool_func not in self.tools:
            self.tools.append(tool_func)
            self.agent.tools = self.tools
            tool_name = getattr(tool_func, 'name', getattr(tool_func, '__name__', str(tool_func)))
            logger.info(f"Added tool: {tool_name}")
        else:
            tool_name = getattr(tool_func, 'name', getattr(tool_func, '__name__', str(tool_func)))
            logger.warning(f"Tool {tool_name} already exists")
    
    def remove_tool(self, tool_func: Any) -> None:
        """Remove a tool from the agent.
        
        Args:
            tool_func: Tool function to remove
        """
        if tool_func in self.tools:
            self.tools.remove(tool_func)
            self.agent.tools = self.tools
            tool_name = getattr(tool_func, 'name', getattr(tool_func, '__name__', str(tool_func)))
            logger.info(f"Removed tool: {tool_name}")
        else:
            tool_name = getattr(tool_func, 'name', getattr(tool_func, '__name__', str(tool_func)))
            logger.warning(f"Tool {tool_name} not found")


def create_agent(
    settings: Optional[Settings] = None,
    tools: Optional[List[Any]] = None,
    instructions: Optional[str] = None
) -> GPTOSSAgent:
    """Factory function to create a GPT-OSS agent.
    
    Args:
        settings: Configuration settings
        tools: List of tools to enable
        instructions: Custom instructions
        
    Returns:
        GPTOSSAgent instance
    """
    return GPTOSSAgent(
        settings=settings,
        tools=tools,
        instructions=instructions
    )