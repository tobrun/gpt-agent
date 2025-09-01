#!/usr/bin/env python3
"""
Main entry point for the GPT-OSS AI Agent.

This script provides the main entry point for running the GPT-OSS agent
with various modes and configurations.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from cli import main as cli_main
from agent import test_agent, quick_chat
from model_client import get_model_info
from tools import check_tools_status

def setup_logging(level: str = "INFO"):
    """Set up logging configuration."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def print_system_info():
    """Print system and configuration information."""
    print("GPT-OSS AI Agent - System Information")
    print("=" * 40)
    
    # Model info
    model_info = get_model_info()
    print(f"Model: {model_info['model']}")
    print(f"Server URL: {model_info['base_url']}")
    print(f"Supports Streaming: {model_info['supports_streaming']}")
    print(f"Supports Tools: {model_info['supports_tools']}")
    
    # Tools status
    print("\nTools Status:")
    tools_status = check_tools_status()
    for tool, status in tools_status.items():
        status_symbol = "✅" if status else "❌"
        print(f"  {tool}: {status_symbol}")
    
    print()

def test_connection():
    """Test connection to vLLM server and agent functionality."""
    print("Testing agent connection...")
    
    try:
        if test_agent():
            print("✅ Agent test passed!")
            return True
        else:
            print("❌ Agent test failed!")
            return False
    except Exception as e:
        print(f"❌ Agent test failed with error: {e}")
        return False

def quick_chat_mode(message: str, model: str = None):
    """Run a single chat interaction."""
    try:
        response = quick_chat(message, model=model)
        print(f"\nAssistant: {response}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="GPT-OSS AI Agent - Local AI Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Start interactive CLI
  python main.py --test                   # Test agent connection
  python main.py --info                   # Show system information
  python main.py --chat "Hello, world!"  # Single chat interaction
  python main.py --model openai/gpt-oss-120b  # Use specific model
        """
    )
    
    # Mode selection
    parser.add_argument(
        "--test", 
        action="store_true", 
        help="Test agent connection and exit"
    )
    parser.add_argument(
        "--info", 
        action="store_true", 
        help="Show system information and exit"
    )
    parser.add_argument(
        "--chat", 
        type=str, 
        help="Send a single message and exit"
    )
    
    # Configuration options
    parser.add_argument(
        "--model", 
        type=str, 
        help="Model name to use (e.g., openai/gpt-oss-20b, openai/gpt-oss-120b)"
    )
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
        default="WARNING",
        help="Set logging level"
    )
    
    # CLI-specific options (passed through to cli.main)
    parser.add_argument(
        "--no-stream", 
        action="store_true", 
        help="Disable streaming responses"
    )
    parser.add_argument(
        "--show-reasoning", 
        action="store_true", 
        help="Show reasoning process"
    )
    parser.add_argument(
        "--no-wait", 
        action="store_true", 
        help="Don't wait for server to be ready"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Handle different modes
    if args.info:
        print_system_info()
        return
    
    if args.test:
        print_system_info()
        success = test_connection()
        sys.exit(0 if success else 1)
    
    if args.chat:
        quick_chat_mode(args.chat, model=args.model)
        return
    
    # Default: run interactive CLI
    # Pass relevant arguments to CLI
    sys.argv = ['cli.py']  # Reset argv for cli.main
    
    if args.model:
        sys.argv.extend(['--model', args.model])
    if args.no_stream:
        sys.argv.append('--no-stream')
    if args.show_reasoning:
        sys.argv.append('--show-reasoning')
    if args.no_wait:
        sys.argv.append('--no-wait')
    
    try:
        cli_main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()