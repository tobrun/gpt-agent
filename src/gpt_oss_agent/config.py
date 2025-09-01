"""Configuration management for GPT-OSS Agent.

This module provides type-safe configuration management using Pydantic,
supporting environment variables, config files, and programmatic configuration.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class VLLMConfig(BaseModel):
    """Configuration for vLLM client."""
    
    base_url: str = Field(
        default="http://localhost:8000/v1",
        description="Base URL for vLLM server"
    )
    model: str = Field(
        default="gpt-oss-120b",
        description="Model name to use"
    )
    timeout: int = Field(
        default=30,
        gt=0,
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retries for failed requests"
    )
    wait_for_server: bool = Field(
        default=True,
        description="Whether to wait for server to be available on startup"
    )


class ExaConfig(BaseModel):
    """Configuration for Exa search API."""
    
    api_key: Optional[str] = Field(
        default=None,
        description="Exa API key for web search functionality"
    )
    enabled: bool = Field(
        default=True,
        description="Whether web search tools are enabled"
    )
    timeout: int = Field(
        default=30,
        gt=0,
        description="Request timeout in seconds"
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Default maximum number of search results"
    )
    
    @validator('enabled')
    def validate_enabled(cls, v, values):
        """Disable if no API key provided."""
        if v and not values.get('api_key'):
            return False
        return v


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    
    level: str = Field(
        default="INFO",
        description="Logging level"
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    file: Optional[str] = Field(
        default=None,
        description="Log file path (optional)"
    )
    
    @validator('level')
    def validate_level(cls, v):
        """Validate logging level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v_upper


class DebugConfig(BaseModel):
    """Configuration for debug functionality."""
    
    enabled: bool = Field(
        default=False,
        description="Enable debug logging to files"
    )
    log_dir: str = Field(
        default="logs/debug",
        description="Directory for debug log files"
    )
    keep_sessions: int = Field(
        default=10,
        ge=1,
        description="Number of debug sessions to keep"
    )


class AgentConfig(BaseModel):
    """Configuration for the agent."""
    
    name: str = Field(
        default="GPT-OSS Assistant",
        description="Agent name"
    )
    instructions: Optional[str] = Field(
        default=None,
        description="Custom agent instructions (overrides default)"
    )
    enable_tools: bool = Field(
        default=True,
        description="Whether to enable tools"
    )
    tool_use_behavior: str = Field(
        default="run_llm_again",
        description="Tool use behavior for OpenAI Agents SDK"
    )


class Settings(BaseSettings):
    """Main application settings.
    
    This class loads configuration from environment variables, .env files,
    and provides defaults for all configuration options.
    """
    
    # Core components
    vllm: VLLMConfig = Field(default_factory=VLLMConfig)
    exa: ExaConfig = Field(default_factory=ExaConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)
    
    # Legacy environment variable support
    vllm_base_url: Optional[str] = Field(default=None, alias="VLLM_BASE_URL")
    default_model: Optional[str] = Field(default=None, alias="DEFAULT_MODEL")
    exa_api_key: Optional[str] = Field(default=None, alias="EXA_API_KEY")
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        case_sensitive = False
        
    def __init__(self, **kwargs):
        """Initialize settings with legacy environment variable support."""
        super().__init__(**kwargs)
        
        # Handle legacy environment variables
        if self.vllm_base_url:
            self.vllm.base_url = self.vllm_base_url
        if self.default_model:
            self.vllm.model = self.default_model
        if self.exa_api_key:
            self.exa.api_key = self.exa_api_key
    
    @validator('debug')
    def create_debug_dir(cls, v):
        """Ensure debug directory exists if debug is enabled."""
        if v.enabled:
            Path(v.log_dir).mkdir(parents=True, exist_ok=True)
        return v


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment/files."""
    global _settings
    _settings = Settings()
    return _settings


def configure_from_dict(config_dict: dict) -> Settings:
    """Configure settings from dictionary."""
    global _settings
    _settings = Settings(**config_dict)
    return _settings


# Environment setup for OpenAI SDK compatibility
def setup_openai_env(settings: Optional[Settings] = None) -> None:
    """Set up environment variables for OpenAI SDK compatibility."""
    if settings is None:
        settings = get_settings()
    
    # Set OpenAI environment variables for SDK compatibility
    os.environ['OPENAI_BASE_URL'] = settings.vllm.base_url
    os.environ['OPENAI_API_KEY'] = "dummy"  # Required by SDK but not used
    os.environ['OPENAI_AGENTS_DISABLE_TRACING'] = "1"  # Disable telemetry