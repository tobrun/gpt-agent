"""Base tool interface and utilities."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol

from agents import function_tool


logger = logging.getLogger(__name__)


class ToolProtocol(Protocol):
    """Protocol for tools used by the agent."""
    
    name: str
    description: str
    
    def __call__(self, *args, **kwargs) -> str:
        """Execute the tool and return results."""
        ...


class BaseTool(ABC):
    """Base class for agent tools.
    
    This provides a common interface for all tools and handles
    logging and error management.
    """
    
    def __init__(self, name: str, description: str):
        """Initialize the base tool.
        
        Args:
            name: Tool name
            description: Tool description
        """
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> str:
        """Execute the tool logic.
        
        This method should be implemented by subclasses to provide
        the actual tool functionality.
        
        Returns:
            Tool execution result as string
        """
        pass
    
    def __call__(self, *args, **kwargs) -> str:
        """Execute the tool with logging and error handling.
        
        Returns:
            Tool execution result
        """
        self.logger.info(f"Executing tool: {self.name}")
        
        try:
            result = self.execute(*args, **kwargs)
            self.logger.info(f"Tool execution completed successfully")
            return result
        except Exception as e:
            error_msg = f"Tool '{self.name}' failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return f"Error: {error_msg}"
    
    def get_info(self) -> Dict[str, Any]:
        """Get tool information.
        
        Returns:
            Dictionary with tool metadata
        """
        return {
            "name": self.name,
            "description": self.description,
            "type": self.__class__.__name__,
        }


def create_function_tool(tool_instance: BaseTool):
    """Create a function tool from a BaseTool instance.
    
    Args:
        tool_instance: Instance of a BaseTool subclass
        
    Returns:
        Function tool for use with OpenAI Agents SDK
    """
    # Get the original execute method
    execute_method = tool_instance.execute
    
    # Create the function tool using the execute method
    func_tool = function_tool(execute_method)
    
    # Set the name and description
    func_tool.name = tool_instance.name
    func_tool.description = tool_instance.description
    
    return func_tool


def validate_tool_result(result: str, max_length: int = 10000) -> str:
    """Validate and sanitize tool result.
    
    Args:
        result: Tool result string
        max_length: Maximum allowed length
        
    Returns:
        Validated and potentially truncated result
    """
    if not isinstance(result, str):
        result = str(result)
    
    if len(result) > max_length:
        truncated = result[:max_length]
        result = f"{truncated}... [Result truncated from {len(result)} to {max_length} characters]"
    
    return result


def log_tool_execution(
    tool_name: str,
    args: Dict[str, Any],
    result: str,
    success: bool = True,
    debug_logger: Optional[Any] = None
) -> None:
    """Log tool execution for debugging.
    
    Args:
        tool_name: Name of the executed tool
        args: Tool arguments
        result: Tool result
        success: Whether execution was successful
        debug_logger: Optional debug logger instance
    """
    logger.info(f"Tool {tool_name} executed: {success}")
    
    if debug_logger and hasattr(debug_logger, 'log_tool_execution'):
        try:
            debug_logger.log_tool_execution(tool_name, args, result)
        except Exception as e:
            logger.warning(f"Failed to log tool execution to debug logger: {e}")