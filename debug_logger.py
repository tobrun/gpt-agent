"""
Debug logging system for GPT-OSS Agent.
Logs all messages, responses, and internal state for debugging.
"""

import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DebugLogger:
    """Debug logger that dumps all messages and responses to files."""
    
    def __init__(self, log_dir: str = "logs/debug"):
        """Initialize debug logger.
        
        Args:
            log_dir: Directory to store debug logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.message_count = 0
        
        logger.info(f"Debug logger initialized. Session ID: {self.session_id}")
        logger.info(f"Debug logs will be saved to: {self.log_dir}")
    
    def log_user_input(self, message: str) -> str:
        """Log user input message.
        
        Args:
            message: User input message
            
        Returns:
            Log file path
        """
        self.message_count += 1
        timestamp = datetime.now().isoformat()
        
        log_data = {
            "session_id": self.session_id,
            "message_count": self.message_count,
            "timestamp": timestamp,
            "type": "user_input",
            "message": message,
            "message_length": len(message)
        }
        
        filename = f"{self.session_id}_msg{self.message_count:03d}_input.json"
        filepath = self.log_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Logged user input to: {filename}")
        return str(filepath)
    
    def log_agent_response(self, response: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Log agent response.
        
        Args:
            response: Agent response
            metadata: Additional metadata about the response
            
        Returns:
            Log file path
        """
        timestamp = datetime.now().isoformat()
        
        log_data = {
            "session_id": self.session_id,
            "message_count": self.message_count,
            "timestamp": timestamp,
            "type": "agent_response",
            "response": response,
            "response_length": len(response) if response else 0,
            "is_empty": not bool(response),
            "metadata": metadata or {}
        }
        
        filename = f"{self.session_id}_msg{self.message_count:03d}_response.json"
        filepath = self.log_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Logged agent response to: {filename} (length: {len(response) if response else 0})")
        return str(filepath)
    
    def log_runner_result(self, result: Any) -> str:
        """Log complete Runner result object.
        
        Args:
            result: RunResult object from OpenAI Agents SDK
            
        Returns:
            Log file path
        """
        timestamp = datetime.now().isoformat()
        
        # Extract key information from result
        result_data = {
            "session_id": self.session_id,
            "message_count": self.message_count,
            "timestamp": timestamp,
            "type": "runner_result",
            "result_type": str(type(result)),
            "has_final_output": hasattr(result, 'final_output'),
            "final_output": getattr(result, 'final_output', None),
            "final_output_length": len(str(getattr(result, 'final_output', ''))) if getattr(result, 'final_output', None) else 0,
            "attributes": dir(result) if result else [],
        }
        
        # Add more detailed information if available
        if hasattr(result, '__dict__'):
            try:
                # Convert complex objects to strings for JSON serialization
                result_dict = {}
                for key, value in result.__dict__.items():
                    try:
                        # Try to serialize directly
                        json.dumps(value)
                        result_dict[key] = value
                    except (TypeError, ValueError):
                        # If not serializable, convert to string
                        result_dict[key] = str(value)
                
                result_data["result_dict"] = result_dict
            except Exception as e:
                result_data["result_dict_error"] = str(e)
        
        # Log new_items if available
        if hasattr(result, 'new_items') and result.new_items:
            items_info = []
            for i, item in enumerate(result.new_items):
                item_info = {
                    "index": i,
                    "type": getattr(item, 'type', str(type(item))),
                    "attributes": dir(item),
                }
                
                # Extract text from MessageOutputItem if present
                if hasattr(item, 'type') and item.type == 'message_output_item':
                    if hasattr(item, 'raw_item') and hasattr(item.raw_item, 'content'):
                        texts = []
                        for content in item.raw_item.content:
                            if hasattr(content, 'text'):
                                texts.append(content.text)
                        item_info["message_texts"] = texts
                
                # Extract reasoning if present
                if hasattr(item, 'type') and item.type == 'reasoning_item':
                    if hasattr(item, 'raw_item') and hasattr(item.raw_item, 'content'):
                        reasoning_texts = []
                        for content in item.raw_item.content:
                            if hasattr(content, 'text'):
                                reasoning_texts.append(content.text)
                        item_info["reasoning_texts"] = reasoning_texts
                
                # Extract tool call information
                if hasattr(item, 'type') and 'tool' in item.type.lower():
                    if hasattr(item, 'raw_item'):
                        tool_info = str(item.raw_item)[:500]  # Truncate long tool outputs
                        item_info["tool_info"] = tool_info
                
                items_info.append(item_info)
            
            result_data["new_items_info"] = items_info
        
        filename = f"{self.session_id}_msg{self.message_count:03d}_runner_result.json"
        filepath = self.log_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Logged runner result to: {filename}")
        return str(filepath)
    
    def log_tool_execution(self, tool_name: str, args: Dict[str, Any], result: str) -> str:
        """Log tool execution details.
        
        Args:
            tool_name: Name of the tool
            args: Tool arguments
            result: Tool result
            
        Returns:
            Log file path
        """
        timestamp = datetime.now().isoformat()
        
        log_data = {
            "session_id": self.session_id,
            "message_count": self.message_count,
            "timestamp": timestamp,
            "type": "tool_execution",
            "tool_name": tool_name,
            "arguments": args,
            "result": result,
            "result_length": len(result) if result else 0,
            "success": bool(result and not result.startswith("Error"))
        }
        
        filename = f"{self.session_id}_msg{self.message_count:03d}_tool_{tool_name}.json"
        filepath = self.log_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Logged tool execution to: {filename}")
        return str(filepath)
    
    def log_error(self, error: Exception, context: str = "") -> str:
        """Log error details.
        
        Args:
            error: Exception that occurred
            context: Context where the error occurred
            
        Returns:
            Log file path
        """
        timestamp = datetime.now().isoformat()
        
        log_data = {
            "session_id": self.session_id,
            "message_count": self.message_count,
            "timestamp": timestamp,
            "type": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context
        }
        
        filename = f"{self.session_id}_msg{self.message_count:03d}_error.json"
        filepath = self.log_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        logger.error(f"Logged error to: {filename}")
        return str(filepath)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current debug session.
        
        Returns:
            Session summary
        """
        log_files = list(self.log_dir.glob(f"{self.session_id}_*.json"))
        
        return {
            "session_id": self.session_id,
            "message_count": self.message_count,
            "log_dir": str(self.log_dir),
            "total_log_files": len(log_files),
            "log_files": [f.name for f in sorted(log_files)]
        }


# Global debug logger instance
_debug_logger = None

def get_debug_logger() -> DebugLogger:
    """Get global debug logger instance."""
    global _debug_logger
    if _debug_logger is None:
        _debug_logger = DebugLogger()
    return _debug_logger