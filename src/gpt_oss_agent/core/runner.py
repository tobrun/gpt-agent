"""Runner utilities for handling agent execution results."""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def analyze_runner_result(result: Any) -> Dict[str, Any]:
    """Analyze a Runner result and extract detailed information.
    
    Args:
        result: RunResult object from OpenAI Agents SDK
        
    Returns:
        Dictionary with analysis of the result
    """
    analysis = {
        "result_type": str(type(result)),
        "has_final_output": hasattr(result, 'final_output'),
        "final_output_length": 0,
        "attributes": dir(result) if result else [],
        "new_items_count": 0,
        "new_items_types": [],
        "reasoning_items": [],
        "tool_calls": [],
        "message_outputs": [],
    }
    
    # Analyze final output
    if hasattr(result, 'final_output'):
        final_output = getattr(result, 'final_output')
        analysis["final_output_length"] = len(str(final_output)) if final_output else 0
        analysis["final_output_empty"] = not bool(final_output)
    
    # Analyze new_items
    if hasattr(result, 'new_items') and result.new_items:
        analysis["new_items_count"] = len(result.new_items)
        
        for i, item in enumerate(result.new_items):
            item_type = getattr(item, 'type', str(type(item)))
            analysis["new_items_types"].append(item_type)
            
            # Extract reasoning items
            if item_type == 'reasoning_item':
                reasoning_text = extract_reasoning_text(item)
                if reasoning_text:
                    analysis["reasoning_items"].append({
                        "index": i,
                        "text": reasoning_text[:200] + "..." if len(reasoning_text) > 200 else reasoning_text
                    })
            
            # Extract tool calls
            elif 'tool' in item_type.lower():
                tool_info = extract_tool_info(item)
                if tool_info:
                    analysis["tool_calls"].append({
                        "index": i,
                        "type": item_type,
                        **tool_info
                    })
            
            # Extract message outputs
            elif item_type == 'message_output_item':
                message_text = extract_message_text(item)
                if message_text:
                    analysis["message_outputs"].append({
                        "index": i,
                        "text": message_text[:200] + "..." if len(message_text) > 200 else message_text
                    })
    
    return analysis


def extract_reasoning_text(item: Any) -> Optional[str]:
    """Extract text from a reasoning item.
    
    Args:
        item: Reasoning item from runner result
        
    Returns:
        Reasoning text or None
    """
    if hasattr(item, 'raw_item') and hasattr(item.raw_item, 'content'):
        texts = []
        for content in item.raw_item.content:
            if hasattr(content, 'text') and content.text:
                texts.append(content.text)
        return " ".join(texts) if texts else None
    return None


def extract_message_text(item: Any) -> Optional[str]:
    """Extract text from a message output item.
    
    Args:
        item: Message output item from runner result
        
    Returns:
        Message text or None
    """
    if hasattr(item, 'raw_item') and hasattr(item.raw_item, 'content'):
        texts = []
        for content in item.raw_item.content:
            if hasattr(content, 'text') and content.text:
                texts.append(content.text)
        return " ".join(texts) if texts else None
    return None


def extract_tool_info(item: Any) -> Optional[Dict[str, Any]]:
    """Extract information from a tool-related item.
    
    Args:
        item: Tool item from runner result
        
    Returns:
        Tool information dictionary or None
    """
    info = {}
    
    if hasattr(item, 'raw_item'):
        raw_item = item.raw_item
        
        # Tool call information
        if hasattr(raw_item, 'name'):
            info["tool_name"] = raw_item.name
        if hasattr(raw_item, 'arguments'):
            info["arguments"] = raw_item.arguments
        if hasattr(raw_item, 'call_id'):
            info["call_id"] = raw_item.call_id
    
    # Tool output information
    if hasattr(item, 'output'):
        output = item.output
        info["output_length"] = len(str(output)) if output else 0
        info["has_output"] = bool(output)
        if output and len(str(output)) <= 500:
            info["output_preview"] = str(output)[:200] + "..." if len(str(output)) > 200 else str(output)
    
    return info if info else None


def find_alternative_responses(result: Any) -> List[str]:
    """Find alternative response sources when final_output is empty.
    
    Args:
        result: RunResult object
        
    Returns:
        List of potential response texts
    """
    responses = []
    
    # Check alternative output attributes
    for attr in ['output', 'outputs', 'response', 'content']:
        if hasattr(result, attr):
            value = getattr(result, attr)
            if value:
                if isinstance(value, list) and value:
                    responses.append(str(value[-1]))
                else:
                    responses.append(str(value))
    
    # Extract from new_items
    if hasattr(result, 'new_items') and result.new_items:
        # Look for message outputs first
        for item in result.new_items:
            if hasattr(item, 'type') and item.type == 'message_output_item':
                text = extract_message_text(item)
                if text:
                    responses.append(text)
        
        # If no message outputs, try reasoning as fallback
        if not responses:
            for item in result.new_items:
                if hasattr(item, 'type') and item.type == 'reasoning_item':
                    text = extract_reasoning_text(item)
                    if text:
                        responses.append(f"[Reasoning] {text}")
    
    return responses


def create_debug_summary(result: Any) -> str:
    """Create a debug summary of a runner result.
    
    Args:
        result: RunResult object
        
    Returns:
        Debug summary string
    """
    analysis = analyze_runner_result(result)
    
    summary = [
        f"Runner Result Analysis:",
        f"  Type: {analysis['result_type']}",
        f"  Has final output: {analysis['has_final_output']}",
        f"  Final output length: {analysis['final_output_length']}",
        f"  New items count: {analysis['new_items_count']}",
    ]
    
    if analysis["new_items_types"]:
        summary.append(f"  Item types: {', '.join(set(analysis['new_items_types']))}")
    
    if analysis["reasoning_items"]:
        summary.append(f"  Reasoning items: {len(analysis['reasoning_items'])}")
        for reasoning in analysis["reasoning_items"][:2]:  # Show first 2
            summary.append(f"    - {reasoning['text']}")
    
    if analysis["tool_calls"]:
        summary.append(f"  Tool calls: {len(analysis['tool_calls'])}")
        for tool in analysis["tool_calls"]:
            tool_name = tool.get("tool_name", "unknown")
            summary.append(f"    - {tool_name}: {tool.get('has_output', False)}")
    
    if analysis["message_outputs"]:
        summary.append(f"  Message outputs: {len(analysis['message_outputs'])}")
    
    return "\n".join(summary)