"""Tool registry for managing available tools."""

import logging
from typing import Any, Dict, List, Optional, Type

from ..config import Settings, get_settings
from .base import BaseTool
from .web_search import get_available_tools as get_web_search_tools, check_tools_status


logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing agent tools.
    
    This registry handles tool discovery, availability checking,
    and provides a unified interface for tool management.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize tool registry.
        
        Args:
            settings: Configuration settings (uses global if None)
        """
        self.settings = settings or get_settings()
        self._tools: Dict[str, Any] = {}
        self._tool_status: Dict[str, Dict[str, Any]] = {}
        
        # Auto-discover tools
        self._discover_tools()
    
    def _discover_tools(self) -> None:
        """Discover and register available tools."""
        logger.info("Discovering available tools...")
        
        # Web search tools
        web_tools = get_web_search_tools()
        for tool in web_tools:
            tool_name = getattr(tool, 'name', getattr(tool, '__name__', str(tool)))
            self._tools[tool_name] = tool
            logger.debug(f"Registered tool: {tool_name}")
        
        # Update status
        self._update_tool_status()
        
        available_count = len([t for t, s in self._tool_status.items() if s.get('available', False)])
        logger.info(f"Tool discovery complete. {available_count}/{len(self._tools)} tools available")
    
    def _update_tool_status(self) -> None:
        """Update tool availability status."""
        # Web search tools status
        web_status = check_tools_status()
        
        for tool_name, tool in self._tools.items():
            status = {
                "available": False,
                "name": tool_name,
                "description": getattr(tool, 'description', 'No description available'),
                "type": "function_tool",
            }
            
            # Check if tool is available
            if tool_name in ["web_search", "get_page_content"]:
                status["available"] = web_status.get(tool_name, False)
                status["category"] = "web_search"
                status["requires"] = "exa_api_key"
            
            self._tool_status[tool_name] = status
    
    def get_available_tools(self) -> List[Any]:
        """Get list of available tools.
        
        Returns:
            List of available tool functions
        """
        available = []
        
        for tool_name, tool in self._tools.items():
            if self._tool_status.get(tool_name, {}).get('available', False):
                available.append(tool)
        
        return available
    
    def get_tool(self, name: str) -> Optional[Any]:
        """Get a specific tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool function or None if not found
        """
        return self._tools.get(name)
    
    def is_tool_available(self, name: str) -> bool:
        """Check if a tool is available.
        
        Args:
            name: Tool name
            
        Returns:
            True if tool exists and is available
        """
        return self._tool_status.get(name, {}).get('available', False)
    
    def get_tool_status(self, name: str = None) -> Dict[str, Any]:
        """Get tool status information.
        
        Args:
            name: Specific tool name (returns all if None)
            
        Returns:
            Tool status information
        """
        if name:
            return self._tool_status.get(name, {})
        
        return self._tool_status.copy()
    
    def get_tools_by_category(self, category: str) -> List[Any]:
        """Get tools by category.
        
        Args:
            category: Tool category (e.g., 'web_search', 'file_system')
            
        Returns:
            List of tools in the specified category
        """
        tools = []
        
        for tool_name, tool in self._tools.items():
            status = self._tool_status.get(tool_name, {})
            if status.get('category') == category and status.get('available', False):
                tools.append(tool)
        
        return tools
    
    def register_tool(self, tool: Any, name: str = None) -> None:
        """Register a custom tool.
        
        Args:
            tool: Tool function or instance
            name: Tool name (uses tool name attribute if None)
        """
        tool_name = name or getattr(tool, 'name', getattr(tool, '__name__', 'unknown'))
        
        self._tools[tool_name] = tool
        
        # Add basic status entry
        self._tool_status[tool_name] = {
            "available": True,  # Assume custom tools are available
            "name": tool_name,
            "description": getattr(tool, 'description', 'Custom tool'),
            "type": "custom",
            "category": "custom",
        }
        
        logger.info(f"Registered custom tool: {tool_name}")
    
    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool.
        
        Args:
            name: Tool name
            
        Returns:
            True if tool was removed, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            del self._tool_status[name]
            logger.info(f"Unregistered tool: {name}")
            return True
        
        return False
    
    def refresh(self) -> None:
        """Refresh tool discovery and status."""
        logger.info("Refreshing tool registry...")
        self._tools.clear()
        self._tool_status.clear()
        self._discover_tools()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get registry summary.
        
        Returns:
            Summary of registry state
        """
        total_tools = len(self._tools)
        available_tools = len([t for t, s in self._tool_status.items() if s.get('available', False)])
        
        categories = {}
        for status in self._tool_status.values():
            category = status.get('category', 'unknown')
            if category not in categories:
                categories[category] = {'total': 0, 'available': 0}
            categories[category]['total'] += 1
            if status.get('available', False):
                categories[category]['available'] += 1
        
        return {
            "total_tools": total_tools,
            "available_tools": available_tools,
            "categories": categories,
            "tool_names": list(self._tools.keys()),
            "available_tool_names": [
                name for name, status in self._tool_status.items() 
                if status.get('available', False)
            ],
        }


# Global registry instance
_registry: Optional[ToolRegistry] = None


def get_tool_registry(settings: Optional[Settings] = None) -> ToolRegistry:
    """Get global tool registry instance.
    
    Args:
        settings: Configuration settings
        
    Returns:
        ToolRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = ToolRegistry(settings)
    return _registry


def get_available_tools(settings: Optional[Settings] = None) -> List[Any]:
    """Get all available tools.
    
    Args:
        settings: Configuration settings
        
    Returns:
        List of available tools
    """
    registry = get_tool_registry(settings)
    return registry.get_available_tools()


def check_all_tools_status(settings: Optional[Settings] = None) -> Dict[str, Any]:
    """Check status of all tools.
    
    Args:
        settings: Configuration settings
        
    Returns:
        Comprehensive tool status information
    """
    registry = get_tool_registry(settings)
    return registry.get_summary()