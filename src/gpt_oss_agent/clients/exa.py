"""Exa search API client for web search functionality."""

import logging
from typing import Any, Dict, List, Optional

import requests

from ..config import Settings, get_settings
from ..exceptions import WebSearchError


logger = logging.getLogger(__name__)


class ExaSearchClient:
    """Client for Exa search API.
    
    This client provides web search capabilities using the Exa search engine,
    with support for content retrieval and customizable search parameters.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize Exa search client.
        
        Args:
            settings: Configuration settings (uses global if None)
        """
        self.settings = settings or get_settings()
        self.api_key = self.settings.exa.api_key
        self.base_url = "https://api.exa.ai"
        self.timeout = self.settings.exa.timeout
        self.enabled = self.settings.exa.enabled and bool(self.api_key)
        
        if not self.enabled:
            logger.warning("Exa search client disabled - no API key configured")
        else:
            logger.info("Exa search client initialized")
    
    def is_available(self) -> bool:
        """Check if Exa search is available.
        
        Returns:
            True if API key is configured and client is enabled
        """
        return self.enabled and bool(self.api_key) and self.api_key != "your_exa_api_key_here"
    
    def search(
        self,
        query: str,
        num_results: int = None,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        use_autoprompt: bool = True,
        include_text: bool = True,
        include_highlights: bool = True,
        include_summary: bool = True
    ) -> Dict[str, Any]:
        """Search the web using Exa API.
        
        Args:
            query: Search query string
            num_results: Number of results to return (uses config default if None)
            include_domains: Domains to include in search
            exclude_domains: Domains to exclude from search
            use_autoprompt: Whether to use Exa's autoprompt feature
            include_text: Include full text content
            include_highlights: Include highlights
            include_summary: Include summaries
            
        Returns:
            Dictionary containing search results
            
        Raises:
            WebSearchError: If search fails or client not available
        """
        if not self.is_available():
            raise WebSearchError(
                "Exa search not available",
                details="API key not configured or client disabled"
            )
        
        if not query or not query.strip():
            raise WebSearchError("Search query cannot be empty")
        
        num_results = num_results or self.settings.exa.max_results
        if num_results < 1 or num_results > 10:
            num_results = 5
        
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
                    "text": include_text,
                    "highlights": include_highlights,
                    "summary": include_summary
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
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise WebSearchError(
                    f"Search request failed with status {response.status_code}",
                    details=response.text
                )
                
        except requests.exceptions.Timeout:
            raise WebSearchError("Search request timed out")
        except requests.exceptions.RequestException as e:
            raise WebSearchError(f"Search request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            raise WebSearchError(f"Unexpected search error: {str(e)}")
    
    def get_content(
        self,
        url: str,
        include_text: bool = True,
        include_summary: bool = True
    ) -> Dict[str, Any]:
        """Get content from a specific webpage.
        
        Args:
            url: URL of the webpage to retrieve
            include_text: Include full text content
            include_summary: Include summary
            
        Returns:
            Dictionary containing page content
            
        Raises:
            WebSearchError: If content retrieval fails
        """
        if not self.is_available():
            raise WebSearchError(
                "Exa search not available",
                details="API key not configured or client disabled"
            )
        
        if not url:
            raise WebSearchError("URL cannot be empty")
        
        try:
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "x-api-key": self.api_key
            }
            
            payload = {
                "ids": [url],
                "contents": {
                    "text": include_text,
                    "summary": include_summary
                }
            }
            
            response = requests.post(
                f"{self.base_url}/contents",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise WebSearchError(
                    f"Content retrieval failed with status {response.status_code}",
                    details=response.text
                )
                
        except requests.exceptions.Timeout:
            raise WebSearchError("Content retrieval timed out")
        except requests.exceptions.RequestException as e:
            raise WebSearchError(f"Content retrieval failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during content retrieval: {e}")
            raise WebSearchError(f"Unexpected content retrieval error: {str(e)}")
    
    def format_search_results(self, results: Dict[str, Any], query: str) -> str:
        """Format search results into a readable string.
        
        Args:
            results: Raw search results from API
            query: Original search query
            
        Returns:
            Formatted results string
        """
        if "error" in results:
            return f"Search error: {results['error']}"
        
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
    
    def format_page_content(self, content_data: Dict[str, Any], url: str) -> str:
        """Format page content into a readable string.
        
        Args:
            content_data: Raw content data from API
            url: URL of the page
            
        Returns:
            Formatted content string
        """
        if "error" in content_data:
            return f"Error: {content_data['error']}"
        
        if "results" not in content_data or not content_data["results"]:
            return f"Error: No content found for URL: {url}"
        
        result = content_data["results"][0]
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
    
    def get_status(self) -> Dict[str, Any]:
        """Get client status information.
        
        Returns:
            Status information dictionary
        """
        return {
            "enabled": self.enabled,
            "api_key_configured": bool(self.api_key and self.api_key != "your_exa_api_key_here"),
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_results": self.settings.exa.max_results,
        }