"""Test configuration management."""

import pytest
from gpt_oss_agent.config import Settings, get_settings


def test_default_settings():
    """Test default settings creation."""
    settings = Settings()
    
    assert settings.vllm.base_url == "http://localhost:8000/v1"
    assert settings.vllm.model == "gpt-oss-120b"
    assert settings.logging.level == "INFO"


def test_custom_settings(test_settings):
    """Test custom settings."""
    assert test_settings.vllm.base_url == "http://localhost:8000/v1"
    assert test_settings.vllm.model == "test-model"
    assert test_settings.exa.api_key == "test-key"
    assert test_settings.logging.level == "DEBUG"


def test_get_settings():
    """Test global settings getter."""
    settings = get_settings()
    assert isinstance(settings, Settings)