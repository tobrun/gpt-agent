"""
Main AI agent implementation using OpenAI Agents SDK.
"""

import os
import logging
from typing import Optional, List, Dict, Any

from agents import Agent, Runner, set_tracing_disabled
from dotenv import load_dotenv

from model_client import setup_vllm_client, get_model_info
from tools import web_search, get_page_content, check_tools_status
from debug_logger import get_debug_logger

# Load environment variables
load_dotenv()

# Disable OpenAI tracing/telemetry completely for local operation
set_tracing_disabled(True)
os.environ['OPENAI_AGENTS_DISABLE_TRACING'] = "1"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPTOSSAgent:
    """Main AI agent powered by local GPT-OSS model via vLLM."""
    
    def __init__(
        self,
        model: Optional[str] = None,
        instructions: Optional[str] = None,
        enable_tools: bool = True,
        wait_for_server: bool = True
    ):
        """
        Initialize the GPT-OSS agent.
        
        Args:
            model: Model name to use (defaults to environment variable)
            instructions: Custom instructions for the agent
            enable_tools: Whether to enable web search tools
            wait_for_server: Whether to wait for vLLM server to be available
        """
        self.model = model or os.getenv("DEFAULT_MODEL", "openai/gpt-oss-20b")
        
        # Set up vLLM client
        try:
            self.vllm_client = setup_vllm_client(wait_for_server=wait_for_server)
            logger.info("vLLM client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vLLM client: {e}")
            raise
        
        # Determine available tools first
        self.available_tools = []
        if enable_tools:
            tools_status = check_tools_status()
            if tools_status["web_search"]:
                self.available_tools.extend([web_search, get_page_content])
                logger.info("Web search tools enabled")
            else:
                logger.warning("Web search tools disabled - EXA_API_KEY not configured")
        
        # Configure agent instructions (after tools are determined)
        if not instructions:
            instructions = self._get_default_instructions()
        
        # Create the agent
        self.agent = Agent(
            name="GPT-OSS Assistant",
            instructions=instructions,
            tools=self.available_tools,
            model=self.model
        )
        
        logger.info(f"Agent initialized with model: {self.model}")
        logger.info(f"Available tools: {[getattr(tool, 'name', getattr(tool, '__name__', str(tool))) for tool in self.available_tools]}")
    
    def _get_default_instructions(self) -> str:
        """Get default instructions for the agent."""
        tools_info = ""
        if self.available_tools or check_tools_status()["web_search"]:
            tools_info = """
You have access to web search capabilities through the following tools:
- web_search: Search the internet for current information
- get_page_content: Retrieve the full content of specific web pages

Use these tools when you need to:
- Find current news, events, or information
- Research topics that may have recent developments
- Get specific details from web pages
- Verify or supplement your knowledge with up-to-date information

Always be clear about when you're using web search vs. your training knowledge.
"""
        
        return f"""You are a helpful and knowledgeable AI assistant powered by the GPT-OSS model running locally. 

You should:
- Be concise but thorough in your responses
- Use web search when you need current information or when explicitly asked
- Explain your reasoning process when solving complex problems
- Ask clarifying questions when the user's request is ambiguous
- Be honest about the limitations of your knowledge

{tools_info}

Remember that you're running on a local model, so you have the benefit of privacy and control, but you should use web search to supplement your knowledge when needed for current events or specific factual queries."""
    
    def chat(self, message: str, **kwargs) -> str:
        """
        Send a synchronous message to the agent.
        
        Args:
            message: User message
            **kwargs: Additional arguments for Runner.run_sync
            
        Returns:
            Agent response as string
        """
        debug_logger = get_debug_logger()
        
        try:
            # Log user input
            debug_logger.log_user_input(message)
            
            logger.info(f"Starting sync chat with message: {message[:50]}...")
            result = Runner.run_sync(
                self.agent,
                message,
                **kwargs
            )
            logger.info(f"Runner completed. Result type: {type(result)}")
            
            # Log complete runner result
            debug_logger.log_runner_result(result)
            
            # Debug the result object
            logger.info(f"Result attributes: {dir(result)}")
            
            if hasattr(result, 'final_output'):
                output = result.final_output
                logger.info(f"Final output type: {type(output)}")
                logger.info(f"Final output length: {len(str(output)) if output else 0}")
                logger.info(f"Final output preview: {str(output)[:100] if output else 'None'}")
                
                # Check if final_output is empty but there are other outputs
                if not output:
                    logger.info("Final output is empty, checking alternative sources...")
                    
                    if hasattr(result, 'output'):
                        logger.info(f"Checking result.output: {result.output}")
                        output = result.output
                    elif hasattr(result, 'outputs'):
                        logger.info(f"Checking result.outputs: {result.outputs}")
                        if result.outputs:
                            output = result.outputs[-1] if isinstance(result.outputs, list) else result.outputs
                    elif hasattr(result, 'new_items') and result.new_items:
                        logger.info("Checking new_items for message output...")
                        # Look for MessageOutputItem in new_items
                        for item in result.new_items:
                            if hasattr(item, 'type') and item.type == 'message_output_item':
                                if hasattr(item, 'raw_item') and hasattr(item.raw_item, 'content'):
                                    for content in item.raw_item.content:
                                        if hasattr(content, 'text'):
                                            logger.info(f"Found text in message output: {content.text[:100]}...")
                                            output = content.text
                                            break
                                    if output:
                                        break
                
                final_response = output if output else "Agent completed but returned no output"
                
                # Log agent response
                debug_logger.log_agent_response(
                    final_response, 
                    metadata={
                        "had_final_output": bool(output),
                        "result_type": str(type(result)),
                        "new_items_count": len(result.new_items) if hasattr(result, 'new_items') else 0
                    }
                )
                
                return final_response
            else:
                error_msg = "Error: No final output available from agent"
                logger.error(f"Result has no final_output attribute. Available attributes: {dir(result)}")
                debug_logger.log_agent_response(error_msg, metadata={"error": "no_final_output_attribute"})
                return error_msg
            
        except Exception as e:
            logger.error(f"Error in sync chat: {e}", exc_info=True)
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            debug_logger.log_error(e, "sync_chat")
            return error_msg
    
    def chat_stream(self, message: str, **kwargs):
        """
        Send a streaming message to the agent.
        
        Args:
            message: User message
            **kwargs: Additional arguments for Runner.run_streamed
            
        Returns:
            RunResultStreaming object for streaming responses
        """
        try:
            return Runner.run_streamed(
                self.agent,
                message,
                **kwargs
            )
            
        except Exception as e:
            logger.error(f"Error in streaming chat: {e}")
            raise
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the agent configuration."""
        model_info = get_model_info()
        tools_status = check_tools_status()
        
        return {
            "agent_name": self.agent.name,
            "model": self.model,
            "model_info": model_info,
            "available_tools": [getattr(tool, 'name', getattr(tool, '__name__', str(tool))) for tool in self.available_tools],
            "tools_status": tools_status,
            "instructions_length": len(self.agent.instructions),
            "vllm_server_url": model_info["base_url"]
        }
    
    def update_instructions(self, new_instructions: str):
        """Update agent instructions."""
        self.agent.instructions = new_instructions
        logger.info("Agent instructions updated")
    
    def add_tool(self, tool_func):
        """Add a new tool to the agent."""
        if tool_func not in self.available_tools:
            self.available_tools.append(tool_func)
            self.agent.tools = self.available_tools
            tool_name = getattr(tool_func, 'name', getattr(tool_func, '__name__', str(tool_func)))
            logger.info(f"Added tool: {tool_name}")
        else:
            tool_name = getattr(tool_func, 'name', getattr(tool_func, '__name__', str(tool_func)))
            logger.warning(f"Tool {tool_name} already exists")
    
    def remove_tool(self, tool_func):
        """Remove a tool from the agent."""
        if tool_func in self.available_tools:
            self.available_tools.remove(tool_func)
            self.agent.tools = self.available_tools
            tool_name = getattr(tool_func, 'name', getattr(tool_func, '__name__', str(tool_func)))
            logger.info(f"Removed tool: {tool_name}")
        else:
            tool_name = getattr(tool_func, 'name', getattr(tool_func, '__name__', str(tool_func)))
            logger.warning(f"Tool {tool_name} not found")


def create_agent(
    model: Optional[str] = None,
    custom_instructions: Optional[str] = None,
    enable_tools: bool = True,
    wait_for_server: bool = True
) -> GPTOSSAgent:
    """
    Factory function to create a GPT-OSS agent.
    
    Args:
        model: Model name to use
        custom_instructions: Custom instructions for the agent
        enable_tools: Whether to enable web search tools
        wait_for_server: Whether to wait for vLLM server
        
    Returns:
        GPTOSSAgent instance
    """
    return GPTOSSAgent(
        model=model,
        instructions=custom_instructions,
        enable_tools=enable_tools,
        wait_for_server=wait_for_server
    )


# Example usage functions
def quick_chat(message: str, model: Optional[str] = None) -> str:
    """Quick one-off chat without creating persistent agent."""
    agent = create_agent(model=model, wait_for_server=False)
    return agent.chat(message)


def test_agent() -> bool:
    """Test agent functionality."""
    try:
        agent = create_agent(wait_for_server=True)
        response = agent.chat("Hello, please introduce yourself briefly.")
        
        if response and len(response) > 10:
            logger.info("Agent test passed")
            return True
        else:
            logger.error("Agent test failed - empty or invalid response")
            return False
            
    except Exception as e:
        logger.error(f"Agent test failed: {e}")
        return False