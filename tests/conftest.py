"""Pytest configuration and fixtures for gpt-oss-agent tests."""

import pytest
from pathlib import Path
import sys

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from gpt_oss_agent.config import Settings


@pytest.fixture
def test_settings():
    """Provide test settings."""
    return Settings(
        vllm__base_url="http://localhost:8000/v1",
        vllm__model="test-model",
        exa__api_key="test-key",
        debug__enabled=False,
        logging__level="DEBUG"
    )


@pytest.fixture
def mock_vllm_client():
    """Mock vLLM client for testing."""
    # TODO: Implement mock client
    pass


@pytest.fixture 
def mock_exa_client():
    """Mock Exa client for testing."""
    # TODO: Implement mock client
    pass