"""vLLM client for connecting to local GPT-OSS model server."""

import logging
import time
from typing import Any, Dict, Optional

import httpx
from openai import OpenAI

from ..config import Settings, get_settings, setup_openai_env
from ..exceptions import VLLMConnectionError, VLLMServerError


logger = logging.getLogger(__name__)


class VLLMClient:
    """Client for connecting to vLLM server.
    
    This client manages the connection to a locally-hosted vLLM server
    and provides health checking and model information retrieval.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize vLLM client.
        
        Args:
            settings: Configuration settings (uses global if None)
        """
        self.settings = settings or get_settings()
        self.base_url = self.settings.vllm.base_url
        self.timeout = self.settings.vllm.timeout
        self.max_retries = self.settings.vllm.max_retries
        
        # Set up OpenAI environment variables for SDK compatibility
        setup_openai_env(self.settings)
        
        # Create OpenAI client configured for vLLM
        self.openai_client = OpenAI(
            base_url=self.base_url,
            api_key="dummy",  # Required by OpenAI SDK but not used by vLLM
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        
        # Force update environment for OpenAI Agents SDK compatibility
        import os
        os.environ['OPENAI_BASE_URL'] = self.base_url
        os.environ['OPENAI_API_KEY'] = "dummy"
        
        logger.info(f"vLLM client initialized for {self.base_url}")
    
    def health_check(self, timeout: Optional[int] = None) -> bool:
        """Check if vLLM server is healthy.
        
        Args:
            timeout: Request timeout in seconds
            
        Returns:
            True if server is healthy, False otherwise
        """
        timeout = timeout or self.timeout
        
        try:
            # Try to get models endpoint
            with httpx.Client(timeout=timeout) as client:
                response = client.get(f"{self.base_url}/models")
                
                if response.status_code == 200:
                    logger.debug("vLLM server health check passed")
                    return True
                else:
                    logger.warning(f"vLLM health check failed: HTTP {response.status_code}")
                    return False
                    
        except httpx.TimeoutException:
            logger.warning(f"vLLM health check timed out after {timeout}s")
            return False
        except httpx.ConnectError:
            logger.warning(f"Cannot connect to vLLM server at {self.base_url}")
            return False
        except Exception as e:
            logger.warning(f"vLLM health check failed: {e}")
            return False
    
    def wait_for_server(self, max_wait: int = 60, check_interval: int = 2) -> bool:
        """Wait for vLLM server to become available.
        
        Args:
            max_wait: Maximum time to wait in seconds
            check_interval: Time between health checks in seconds
            
        Returns:
            True if server becomes available, False if timeout
            
        Raises:
            VLLMConnectionError: If server doesn't become available within max_wait
        """
        logger.info(f"Waiting for vLLM server at {self.base_url}")
        
        start_time = time.time()
        while time.time() - start_time < max_wait:
            if self.health_check(timeout=5):
                logger.info("vLLM server is ready")
                return True
            
            logger.debug(f"vLLM server not ready, waiting {check_interval}s...")
            time.sleep(check_interval)
        
        raise VLLMConnectionError(
            f"vLLM server at {self.base_url} did not become available within {max_wait}s"
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models.
        
        Returns:
            Dictionary with model information
            
        Raises:
            VLLMServerError: If unable to get model info
        """
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.base_url}/models")
                
                if response.status_code != 200:
                    raise VLLMServerError(
                        f"Failed to get model info: HTTP {response.status_code}",
                        status_code=response.status_code,
                        details=response.text
                    )
                
                data = response.json()
                models = data.get('data', [])
                
                return {
                    "base_url": self.base_url,
                    "available_models": [model.get('id', 'unknown') for model in models],
                    "configured_model": self.settings.vllm.model,
                    "model_count": len(models),
                    "server_healthy": True
                }
                
        except httpx.HTTPStatusError as e:
            raise VLLMServerError(
                f"HTTP error getting model info: {e.response.status_code}",
                status_code=e.response.status_code
            )
        except httpx.RequestError as e:
            raise VLLMConnectionError(f"Connection error getting model info: {e}")
        except Exception as e:
            raise VLLMServerError(f"Unexpected error getting model info: {e}")
    
    def test_connection(self) -> Dict[str, Any]:
        """Test the connection to vLLM server.
        
        Returns:
            Dictionary with connection test results
        """
        result = {
            "server_url": self.base_url,
            "configured_model": self.settings.vllm.model,
            "health_check_passed": False,
            "model_info_available": False,
            "error": None
        }
        
        try:
            # Test health check
            result["health_check_passed"] = self.health_check()
            
            if result["health_check_passed"]:
                # Test model info
                model_info = self.get_model_info()
                result["model_info_available"] = True
                result["available_models"] = model_info["available_models"]
                result["model_count"] = model_info["model_count"]
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Connection test failed: {e}")
        
        return result
    
    def get_openai_client(self) -> OpenAI:
        """Get configured OpenAI client for vLLM.
        
        Returns:
            OpenAI client configured for vLLM
        """
        return self.openai_client


def setup_vllm_client(
    settings: Optional[Settings] = None,
    wait_for_server: bool = True
) -> VLLMClient:
    """Set up and return a vLLM client.
    
    Args:
        settings: Configuration settings
        wait_for_server: Whether to wait for server to be available
        
    Returns:
        Configured VLLMClient instance
        
    Raises:
        VLLMConnectionError: If server is not available and wait_for_server is True
    """
    settings = settings or get_settings()
    client = VLLMClient(settings)
    
    if wait_for_server and settings.vllm.wait_for_server:
        client.wait_for_server()
    
    return client


def get_model_info(settings: Optional[Settings] = None) -> Dict[str, Any]:
    """Get model information from vLLM server.
    
    Args:
        settings: Configuration settings
        
    Returns:
        Model information dictionary
    """
    client = VLLMClient(settings)
    return client.get_model_info()