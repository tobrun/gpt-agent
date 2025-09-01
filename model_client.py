"""
Model client configuration for connecting to local vLLM server.
"""

import os
import time
import logging
from typing import Optional

import httpx
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VLLMClient:
    """Custom OpenAI-compatible client for vLLM server."""
    
    def __init__(
        self, 
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3
    ):
        self.base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        self.model = model or os.getenv("DEFAULT_MODEL", "openai/gpt-oss-20b")
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Create OpenAI client pointing to vLLM
        self.client = OpenAI(
            base_url=self.base_url,
            api_key="dummy",  # vLLM doesn't require real API key
            timeout=timeout,
            max_retries=max_retries
        )
        
        logger.info(f"Initialized vLLM client with base_url: {self.base_url}")
        logger.info(f"Using model: {self.model}")
    
    def health_check(self) -> bool:
        """Check if vLLM server is running and responsive."""
        try:
            # Remove /v1 suffix for health check
            health_url = self.base_url.replace("/v1", "") + "/health"
            
            with httpx.Client(timeout=5.0) as client:
                response = client.get(health_url)
                if response.status_code == 200:
                    logger.info("vLLM server health check passed")
                    return True
                else:
                    logger.warning(f"vLLM server health check failed: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to connect to vLLM server: {e}")
            return False
    
    def wait_for_server(self, timeout: int = 30) -> bool:
        """Wait for vLLM server to become available."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.health_check():
                return True
            
            logger.info("Waiting for vLLM server to become available...")
            time.sleep(2)
        
        logger.error(f"vLLM server did not become available within {timeout} seconds")
        return False
    
    def test_connection(self) -> bool:
        """Test connection with a simple completion request."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
                stream=False
            )
            
            if response.choices and len(response.choices) > 0:
                logger.info("vLLM connection test successful")
                return True
            else:
                logger.warning("vLLM connection test failed: no response content")
                return False
                
        except Exception as e:
            logger.error(f"vLLM connection test failed: {e}")
            return False
    
    def get_available_models(self) -> list:
        """Get list of available models from vLLM server."""
        try:
            models = self.client.models.list()
            model_names = [model.id for model in models.data]
            logger.info(f"Available models: {model_names}")
            return model_names
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []


def setup_vllm_client(
    base_url: Optional[str] = None, 
    model: Optional[str] = None,
    wait_for_server: bool = True
) -> VLLMClient:
    """
    Set up vLLM client and configure it as the default OpenAI client for agents.
    
    Args:
        base_url: vLLM server URL (defaults to localhost:8000)
        model: Model name to use (defaults to gpt-oss-20b)
        wait_for_server: Whether to wait for server to be available
        
    Returns:
        VLLMClient instance
        
    Raises:
        ConnectionError: If server is not available and wait_for_server is True
    """
    client = VLLMClient(base_url=base_url, model=model)
    
    # Wait for server if requested
    if wait_for_server:
        if not client.wait_for_server():
            raise ConnectionError(
                f"Could not connect to vLLM server at {client.base_url}. "
                "Please ensure vLLM is running with the GPT-OSS model."
            )
    
    # Test connection
    if not client.test_connection():
        logger.warning("Connection test failed, but proceeding anyway")
    
    # Configure environment variables for OpenAI Agents SDK
    os.environ['OPENAI_BASE_URL'] = client.base_url
    os.environ['OPENAI_API_KEY'] = "dummy"
    
    # Disable tracing to prevent calls to OpenAI servers
    os.environ['OPENAI_AGENTS_DISABLE_TRACING'] = "1"
    
    logger.info(f"OpenAI environment configured for vLLM at {client.base_url}")
    return client


def get_model_info() -> dict:
    """Get information about the current model configuration."""
    return {
        "base_url": os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        "model": os.getenv("DEFAULT_MODEL", "openai/gpt-oss-20b"),
        "supports_streaming": True,
        "supports_tools": True,
        "supports_responses_api": True,
        "supports_chat_completions": True
    }