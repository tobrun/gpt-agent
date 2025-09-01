"""
Tool implementations for the AI agent.
"""

import os
import logging
from typing import List, Dict, Any, Optional

import requests
from agents import function_tool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EXASearchClient:
    """Client for Exa search API."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("EXA_API_KEY")
        self.base_url = "https://api.exa.ai"
        
        if not self.api_key or self.api_key == "your_exa_api_key_here":
            logger.warning("EXA_API_KEY not configured. Web search will not work.")
            self.api_key = None
    
    def search(
        self, 
        query: str, 
        num_results: int = 5,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        use_autoprompt: bool = True
    ) -> Dict[str, Any]:
        """
        Search the web using Exa API.
        
        Args:
            query: Search query
            num_results: Number of results to return
            include_domains: Domains to include in search
            exclude_domains: Domains to exclude from search
            use_autoprompt: Whether to use Exa's autoprompt feature
            
        Returns:
            Dictionary containing search results
        """
        if not self.api_key:
            return {
                "error": "EXA_API_KEY not configured. Please add your API key to .env file."
            }
        
        try:
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "x-api-key": self.api_key
            }
            
            payload = {
                "query": query,
                "numResults": num_results,
                "useAutoprompt": use_autoprompt,
                "contents": {
                    "text": True,
                    "highlights": True,
                    "summary": True
                }
            }
            
            if include_domains:
                payload["includeDomains"] = include_domains
            if exclude_domains:
                payload["excludeDomains"] = exclude_domains
            
            response = requests.post(
                f"{self.base_url}/search",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"Exa API request failed with status {response.status_code}: {response.text}"
                }
                
        except requests.exceptions.Timeout:
            return {"error": "Search request timed out"}
        except requests.exceptions.RequestException as e:
            return {"error": f"Search request failed: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            return {"error": f"Unexpected error: {str(e)}"}


# Initialize search client
search_client = EXASearchClient()


@function_tool
def web_search(query: str, num_results: int = 5) -> str:
    """
    Search the web for information using Exa search engine.
    
    Args:
        query: The search query string
        num_results: Number of search results to return (1-10, default 5)
        
    Returns:
        Formatted search results or error message
    """
    logger.info(f"Performing web search for: {query}")
    
    # Validate input
    if not query or not query.strip():
        return "Error: Search query cannot be empty"
    
    if num_results < 1 or num_results > 10:
        num_results = 5
    
    # Perform search
    results = search_client.search(query, num_results=num_results)
    
    # Handle errors
    if "error" in results:
        return f"Search error: {results['error']}"
    
    # Format results
    try:
        formatted_results = []
        formatted_results.append(f"Web search results for: '{query}'")
        formatted_results.append("=" * 50)
        
        if "results" in results and results["results"]:
            for i, result in enumerate(results["results"], 1):
                title = result.get("title", "No title")
                url = result.get("url", "No URL")
                
                # Get snippet from text, highlights, or summary
                snippet = ""
                if "text" in result and result["text"]:
                    snippet = result["text"][:300] + "..." if len(result["text"]) > 300 else result["text"]
                elif "highlights" in result and result["highlights"]:
                    snippet = " ".join(result["highlights"][:2])  # First 2 highlights
                elif "summary" in result and result["summary"]:
                    snippet = result["summary"]
                else:
                    snippet = "No description available"
                
                formatted_results.append(f"\n{i}. {title}")
                formatted_results.append(f"   URL: {url}")
                formatted_results.append(f"   {snippet}")
        else:
            formatted_results.append("No results found for this query.")
        
        return "\n".join(formatted_results)
        
    except Exception as e:
        logger.error(f"Error formatting search results: {e}")
        return f"Error processing search results: {str(e)}"


@function_tool 
def get_page_content(url: str) -> str:
    """
    Get the content of a specific webpage.
    
    Args:
        url: The URL of the webpage to retrieve
        
    Returns:
        The text content of the webpage or error message
    """
    if not search_client.api_key:
        return "Error: EXA_API_KEY not configured. Cannot retrieve page content."
    
    try:
        headers = {
            "accept": "application/json",
            "content-type": "application/json", 
            "x-api-key": search_client.api_key
        }
        
        payload = {
            "ids": [url],
            "contents": {
                "text": True,
                "summary": True
            }
        }
        
        response = requests.post(
            f"{search_client.base_url}/contents",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if "results" in data and data["results"]:
                result = data["results"][0]
                title = result.get("title", "No title")
                text = result.get("text", "")
                summary = result.get("summary", "")
                
                content = f"Title: {title}\nURL: {url}\n\n"
                if summary:
                    content += f"Summary: {summary}\n\n"
                if text:
                    # Truncate very long content
                    if len(text) > 2000:
                        content += f"Content (truncated): {text[:2000]}...\n"
                    else:
                        content += f"Content: {text}\n"
                
                return content
            else:
                return f"Error: No content found for URL: {url}"
        else:
            return f"Error: Failed to retrieve content (status {response.status_code})"
            
    except Exception as e:
        logger.error(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"


def get_available_tools() -> List[str]:
    """Get list of available tools."""
    tools = ["web_search"]
    
    if search_client.api_key:
        tools.append("get_page_content")
    
    return tools


def check_tools_status() -> Dict[str, bool]:
    """Check status of all tools."""
    return {
        "web_search": bool(search_client.api_key),
        "get_page_content": bool(search_client.api_key),
        "exa_api_configured": bool(search_client.api_key)
    }