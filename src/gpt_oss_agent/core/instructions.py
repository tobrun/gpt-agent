"""Default instructions for the GPT-OSS Agent."""

from typing import List


def get_default_instructions(has_web_search: bool = False) -> str:
    """Get default instructions for the agent.
    
    Args:
        has_web_search: Whether web search tools are available
        
    Returns:
        Default instruction string
    """
    base_instructions = """You are a helpful and knowledgeable AI assistant powered by the GPT-OSS model running locally.

You should:
- Be concise but thorough in your responses
- Use web search when you need current information or when explicitly asked
- Explain your reasoning process when solving complex problems
- Ask clarifying questions when the user's request is ambiguous
- Be honest about the limitations of your knowledge"""

    if has_web_search:
        web_search_instructions = """

You have access to web search capabilities through the following tools ONLY:
- web_search: Search the internet for current information
- get_page_content: Retrieve the full content of specific web pages

These are the ONLY tools available. Do NOT attempt to use any other tools like find_in_page, extract_text, or similar - they do not exist.

Use these tools when you need to:
- Find current news, events, or information
- Research topics that may have recent developments  
- Get specific details from web pages
- Verify or supplement your knowledge with up-to-date information

IMPORTANT: Only use the two tools listed above. If you need to find information on a page, use get_page_content and analyze the content yourself.

Always be clear about when you're using web search vs. your training knowledge."""

        base_instructions += web_search_instructions

    footer = """

Remember that you're running on a local model, so you have the benefit of privacy and control, but you should use web search to supplement your knowledge when needed for current events or specific factual queries."""

    return base_instructions + footer


def get_tool_descriptions() -> dict:
    """Get descriptions of available tools for instruction generation."""
    return {
        "web_search": "Search the internet for current information",
        "get_page_content": "Retrieve the full content of specific web pages",
    }


def build_custom_instructions(
    base_behavior: str,
    available_tools: List[str] = None,
    additional_context: str = None
) -> str:
    """Build custom instructions with specified behavior and tools.
    
    Args:
        base_behavior: Core behavior description
        available_tools: List of available tool names
        additional_context: Additional context to include
        
    Returns:
        Custom instruction string
    """
    instructions = base_behavior
    
    if available_tools:
        tool_descriptions = get_tool_descriptions()
        tools_section = "\n\nAvailable tools:\n"
        for tool in available_tools:
            if tool in tool_descriptions:
                tools_section += f"- {tool}: {tool_descriptions[tool]}\n"
        instructions += tools_section
    
    if additional_context:
        instructions += f"\n\nAdditional context:\n{additional_context}"
    
    return instructions