"""Debug logging system for GPT-OSS Agent.

Logs all messages, responses, and internal state for debugging.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..config import get_settings


logger = logging.getLogger(__name__)


class DebugLogger:
    """Debug logger that dumps all messages and responses to files.
    
    This logger provides detailed debugging capabilities by saving
    all agent interactions, tool executions, and internal state
    to structured JSON files.
    """
    
    def __init__(self, log_dir: Optional[str] = None, session_id: Optional[str] = None):
        """Initialize debug logger.
        
        Args:
            log_dir: Directory to store debug logs (uses config if None)
            session_id: Session ID (auto-generated if None)
        """
        settings = get_settings()
        
        self.log_dir = Path(log_dir or settings.debug.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.message_count = 0
        self.enabled = settings.debug.enabled
        
        if self.enabled:
            logger.info(f"Debug logger initialized. Session ID: {self.session_id}")
            logger.info(f"Debug logs will be saved to: {self.log_dir}")
        else:
            logger.debug("Debug logging is disabled")
    
    def _write_log_file(self, filename: str, data: Dict[str, Any]) -> Optional[str]:
        """Write data to log file.
        
        Args:
            filename: Log file name
            data: Data to write
            
        Returns:
            File path if written, None if disabled
        """
        if not self.enabled:
            return None
        
        try:
            filepath = self.log_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to write debug log {filename}: {e}")
            return None
    
    def log_user_input(self, message: str) -> Optional[str]:
        """Log user input message.
        
        Args:
            message: User input message
            
        Returns:
            Log file path or None if disabled
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
        filepath = self._write_log_file(filename, log_data)
        
        if filepath:
            logger.info(f"Logged user input to: {filename}")
        
        return filepath
    
    def log_agent_response(
        self, 
        response: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Log agent response.
        
        Args:
            response: Agent response
            metadata: Additional metadata about the response
            
        Returns:
            Log file path or None if disabled
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
        filepath = self._write_log_file(filename, log_data)
        
        if filepath:
            logger.info(f"Logged agent response to: {filename} (length: {len(response) if response else 0})")
        
        return filepath
    
    def log_runner_result(self, result: Any) -> Optional[str]:
        """Log complete Runner result object.
        
        Args:
            result: RunResult object from OpenAI Agents SDK
            
        Returns:
            Log file path or None if disabled
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
                        json.dumps(value, default=str)
                        result_dict[key] = value
                    except (TypeError, ValueError):
                        # If not serializable, convert to string
                        result_dict[key] = str(value)[:1000]  # Limit length
                
                result_data["result_dict"] = result_dict
            except Exception as e:
                result_data["result_dict_error"] = str(e)
        
        # Log new_items if available with enhanced extraction
        if hasattr(result, 'new_items') and result.new_items:
            items_info = self._extract_new_items_info(result.new_items)
            result_data["new_items_info"] = items_info
        
        filename = f"{self.session_id}_msg{self.message_count:03d}_runner_result.json"
        filepath = self._write_log_file(filename, result_data)
        
        if filepath:
            logger.info(f"Logged runner result to: {filename}")
        
        return filepath
    
    def _extract_new_items_info(self, new_items: list) -> list:
        """Extract information from new_items with enhanced detail.
        
        Args:
            new_items: List of new items from runner result
            
        Returns:
            List of item information dictionaries
        """
        items_info = []
        
        for i, item in enumerate(new_items):
            item_info = {
                "index": i,
                "type": getattr(item, 'type', str(type(item))),
                "attributes": dir(item),
            }
            
            # Extract text from MessageOutputItem
            if hasattr(item, 'type') and item.type == 'message_output_item':
                item_info["message_texts"] = self._extract_message_texts(item)
            
            # Extract reasoning if present
            elif hasattr(item, 'type') and item.type == 'reasoning_item':
                item_info["reasoning_texts"] = self._extract_reasoning_texts(item)
            
            # Extract tool call information
            elif hasattr(item, 'type') and 'tool' in item.type.lower():
                item_info["tool_info"] = self._extract_tool_info(item)
            
            items_info.append(item_info)
        
        return items_info
    
    def _extract_message_texts(self, item: Any) -> list:
        """Extract texts from message item."""
        texts = []
        if hasattr(item, 'raw_item') and hasattr(item.raw_item, 'content'):
            for content in item.raw_item.content:
                if hasattr(content, 'text') and content.text:
                    texts.append(content.text)
        return texts
    
    def _extract_reasoning_texts(self, item: Any) -> list:
        """Extract texts from reasoning item."""
        texts = []
        if hasattr(item, 'raw_item') and hasattr(item.raw_item, 'content'):
            for content in item.raw_item.content:
                if hasattr(content, 'text') and content.text:
                    texts.append(content.text)
        return texts
    
    def _extract_tool_info(self, item: Any) -> dict:
        """Extract tool information."""
        tool_info = {}
        if hasattr(item, 'raw_item'):
            raw_item = item.raw_item
            if hasattr(raw_item, 'name'):
                tool_info["name"] = raw_item.name
            if hasattr(raw_item, 'arguments'):
                tool_info["arguments"] = raw_item.arguments
        
        # Add truncated string representation for debugging
        tool_info["raw_item_str"] = str(item)[:500] + "..." if len(str(item)) > 500 else str(item)
        return tool_info
    
    def log_tool_execution(
        self, 
        tool_name: str, 
        args: Dict[str, Any], 
        result: str
    ) -> Optional[str]:
        """Log tool execution details.
        
        Args:
            tool_name: Name of the tool
            args: Tool arguments
            result: Tool result
            
        Returns:
            Log file path or None if disabled
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
        filepath = self._write_log_file(filename, log_data)
        
        if filepath:
            logger.info(f"Logged tool execution to: {filename}")
        
        return filepath
    
    def log_error(self, error: Exception, context: str = "") -> Optional[str]:
        """Log error details.
        
        Args:
            error: Exception that occurred
            context: Context where the error occurred
            
        Returns:
            Log file path or None if disabled
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
        filepath = self._write_log_file(filename, log_data)
        
        if filepath:
            logger.error(f"Logged error to: {filename}")
        
        return filepath
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current debug session.
        
        Returns:
            Session summary
        """
        if not self.enabled:
            return {"enabled": False}
        
        log_files = list(self.log_dir.glob(f"{self.session_id}_*.json"))
        
        return {
            "enabled": True,
            "session_id": self.session_id,
            "message_count": self.message_count,
            "log_dir": str(self.log_dir),
            "total_log_files": len(log_files),
            "log_files": [f.name for f in sorted(log_files)]
        }


# Global debug logger instance
_debug_logger: Optional[DebugLogger] = None


def get_debug_logger() -> DebugLogger:
    """Get global debug logger instance.
    
    Returns:
        DebugLogger instance
    """
    global _debug_logger
    if _debug_logger is None:
        _debug_logger = DebugLogger()
    return _debug_logger


def set_debug_logger(logger_instance: DebugLogger) -> None:
    """Set global debug logger instance.
    
    Args:
        logger_instance: DebugLogger instance to set
    """
    global _debug_logger
    _debug_logger = logger_instance