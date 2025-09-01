"""Web search tools using Exa API."""

import logging
from typing import Any, Dict, Optional

from agents import function_tool

from ..clients.exa import ExaSearchClient
from ..config import get_settings
from ..exceptions import WebSearchError
from .base import log_tool_execution


logger = logging.getLogger(__name__)


class WebSearchTool:
    """Web search tool using Exa API.
    
    This tool provides web search capabilities and integrates
    with the debug logging system.
    """
    
    def __init__(self, client: Optional[ExaSearchClient] = None):
        """Initialize web search tool.
        
        Args:
            client: Exa search client (creates new if None)
        """
        self.client = client or ExaSearchClient()
        self.name = "web_search"
        self.description = "Search the web for information using Exa search engine"
    
    def is_available(self) -> bool:
        """Check if web search is available."""
        return self.client.is_available()
    
    def execute(self, query: str, num_results: int = 5) -> str:
        """Execute web search.
        
        Args:
            query: Search query string
            num_results: Number of results to return (1-10)
            
        Returns:
            Formatted search results or error message
        """
        if not self.is_available():
            return "Error: Web search not available - Exa API key not configured"
        
        if not query or not query.strip():
            return "Error: Search query cannot be empty"
        
        # Validate num_results
        if num_results < 1 or num_results > 10:
            num_results = 5
        
        try:
            logger.info(f"Performing web search for: {query}")
            
            # Perform search
            results = self.client.search(query, num_results=num_results)
            
            # Format results
            formatted_results = self.client.format_search_results(results, query)
            
            logger.info(f"Search completed, result length: {len(formatted_results)}")
            return formatted_results
            
        except WebSearchError as e:
            error_msg = f"Web search error: {e.message}"
            if e.details:
                error_msg += f" ({e.details})"
            return error_msg
        except Exception as e:
            logger.error(f"Unexpected error in web search: {e}", exc_info=True)
            return f"Error: Unexpected web search error: {str(e)}"


class PageContentTool:
    """Tool for retrieving specific webpage content."""
    
    def __init__(self, client: Optional[ExaSearchClient] = None):
        """Initialize page content tool.
        
        Args:
            client: Exa search client (creates new if None)
        """
        self.client = client or ExaSearchClient()
        self.name = "get_page_content"
        self.description = "Get the content of a specific webpage"
    
    def is_available(self) -> bool:
        """Check if page content retrieval is available."""
        return self.client.is_available()
    
    def execute(self, url: str) -> str:
        """Execute page content retrieval.
        
        Args:
            url: URL of the webpage to retrieve
            
        Returns:
            Formatted page content or error message
        """
        if not self.is_available():
            return "Error: Page content retrieval not available - Exa API key not configured"
        
        if not url:
            return "Error: URL cannot be empty"
        
        try:
            logger.info(f"Retrieving content for: {url}")
            
            # Get content
            content_data = self.client.get_content(url)
            
            # Format content
            formatted_content = self.client.format_page_content(content_data, url)
            
            logger.info(f"Content retrieval completed, length: {len(formatted_content)}")
            return formatted_content
            
        except WebSearchError as e:
            error_msg = f"Page content error: {e.message}"
            if e.details:
                error_msg += f" ({e.details})"
            return error_msg
        except Exception as e:
            logger.error(f"Unexpected error in page content retrieval: {e}", exc_info=True)
            return f"Error: Unexpected page content error: {str(e)}"


# Global tool instances
_web_search_tool = None
_page_content_tool = None


def get_web_search_tool() -> WebSearchTool:
    """Get global web search tool instance."""
    global _web_search_tool
    if _web_search_tool is None:
        _web_search_tool = WebSearchTool()
    return _web_search_tool


def get_page_content_tool() -> PageContentTool:
    """Get global page content tool instance."""
    global _page_content_tool
    if _page_content_tool is None:
        _page_content_tool = PageContentTool()
    return _page_content_tool


# Function tools for OpenAI Agents SDK
@function_tool
def web_search(query: str, num_results: int = 5) -> str:
    """Search the web for information using Exa search engine.
    
    Args:
        query: The search query string
        num_results: Number of search results to return (1-10, default 5)
        
    Returns:
        Formatted search results or error message
    """
    tool = get_web_search_tool()
    result = tool.execute(query, num_results)
    
    # Log for debugging
    try:
        from ..utils.debug_logger import get_debug_logger
        debug_logger = get_debug_logger()
        log_tool_execution(
            "web_search",
            {"query": query, "num_results": num_results},
            result,
            success=not result.startswith("Error"),
            debug_logger=debug_logger
        )
    except ImportError:
        # Debug logger not available
        pass
    
    return result


@function_tool
def get_page_content(url: str) -> str:
    """Get the content of a specific webpage.
    
    Args:
        url: The URL of the webpage to retrieve
        
    Returns:
        The text content of the webpage or error message
    """
    tool = get_page_content_tool()
    result = tool.execute(url)
    
    # Log for debugging
    try:
        from ..utils.debug_logger import get_debug_logger
        debug_logger = get_debug_logger()
        log_tool_execution(
            "get_page_content",
            {"url": url},
            result,
            success=not result.startswith("Error"),
            debug_logger=debug_logger
        )
    except ImportError:
        # Debug logger not available
        pass
    
    return result


def check_tools_status() -> Dict[str, Any]:
    """Check status of web search tools.
    
    Returns:
        Status information for all web search tools
    """
    settings = get_settings()
    client = ExaSearchClient(settings)
    
    return {
        "web_search": client.is_available(),
        "get_page_content": client.is_available(),
        "exa_api_configured": client.get_status()["api_key_configured"],
        "exa_status": client.get_status(),
    }


def get_available_tools() -> list:
    """Get list of available web search tools.
    
    Returns:
        List of available function tools
    """
    tools = []
    
    if get_web_search_tool().is_available():
        tools.extend([web_search, get_page_content])
    
    return tools