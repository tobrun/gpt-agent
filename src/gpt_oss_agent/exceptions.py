"""Custom exceptions for GPT-OSS Agent."""


class GPTOSSAgentError(Exception):
    """Base exception for all GPT-OSS Agent errors."""
    
    def __init__(self, message: str, details: str = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class VLLMConnectionError(GPTOSSAgentError):
    """Raised when unable to connect to vLLM server."""
    pass


class VLLMServerError(GPTOSSAgentError):
    """Raised when vLLM server returns an error."""
    
    def __init__(self, message: str, status_code: int = None, details: str = None):
        super().__init__(message, details)
        self.status_code = status_code


class ToolError(GPTOSSAgentError):
    """Raised when a tool encounters an error."""
    
    def __init__(self, tool_name: str, message: str, details: str = None):
        super().__init__(f"Tool '{tool_name}' error: {message}", details)
        self.tool_name = tool_name


class WebSearchError(ToolError):
    """Raised when web search tools encounter errors."""
    
    def __init__(self, message: str, details: str = None):
        super().__init__("web_search", message, details)


class ConfigurationError(GPTOSSAgentError):
    """Raised when configuration is invalid."""
    pass


class AgentError(GPTOSSAgentError):
    """Raised when the agent encounters an error."""
    pass


class EmptyResponseError(AgentError):
    """Raised when the agent returns an empty response."""
    
    def __init__(self, details: str = None):
        super().__init__(
            "Agent completed but returned no output", 
            details or "This may be due to vLLM /v1/responses endpoint compatibility issues"
        )