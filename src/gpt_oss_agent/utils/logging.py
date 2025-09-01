"""Logging configuration utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional

from ..config import Settings, get_settings


def setup_logging(settings: Optional[Settings] = None) -> None:
    """Set up application logging.
    
    Args:
        settings: Configuration settings (uses global if None)
    """
    settings = settings or get_settings()
    
    # Configure root logger
    root_logger = logging.getLogger()
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set log level
    log_level = getattr(logging, settings.logging.level.upper(), logging.INFO)
    root_logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(settings.logging.format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if settings.logging.file:
        log_file = Path(settings.logging.file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Reduce noise from some libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.INFO)
    
    logging.info(f"Logging configured at {settings.logging.level} level")


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def set_log_level(level: str) -> None:
    """Set the log level for all loggers.
    
    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    # Update root logger and all handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    for handler in root_logger.handlers:
        handler.setLevel(numeric_level)
    
    logging.info(f"Log level set to {level.upper()}")


class StructuredLogger:
    """Structured logger with context support."""
    
    def __init__(self, name: str, context: Optional[dict] = None):
        """Initialize structured logger.
        
        Args:
            name: Logger name
            context: Default context to include in all log messages
        """
        self.logger = logging.getLogger(name)
        self.context = context or {}
    
    def _format_message(self, message: str, extra_context: Optional[dict] = None) -> str:
        """Format message with context.
        
        Args:
            message: Log message
            extra_context: Additional context for this message
            
        Returns:
            Formatted message
        """
        all_context = {**self.context}
        if extra_context:
            all_context.update(extra_context)
        
        if all_context:
            context_str = " | ".join(f"{k}={v}" for k, v in all_context.items())
            return f"{message} [{context_str}]"
        
        return message
    
    def debug(self, message: str, context: Optional[dict] = None) -> None:
        """Log debug message."""
        self.logger.debug(self._format_message(message, context))
    
    def info(self, message: str, context: Optional[dict] = None) -> None:
        """Log info message."""
        self.logger.info(self._format_message(message, context))
    
    def warning(self, message: str, context: Optional[dict] = None) -> None:
        """Log warning message."""
        self.logger.warning(self._format_message(message, context))
    
    def error(self, message: str, context: Optional[dict] = None, exc_info: bool = False) -> None:
        """Log error message."""
        self.logger.error(self._format_message(message, context), exc_info=exc_info)
    
    def critical(self, message: str, context: Optional[dict] = None) -> None:
        """Log critical message."""
        self.logger.critical(self._format_message(message, context))