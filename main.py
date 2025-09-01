#!/usr/bin/env python3
"""
Main entry point for the GPT-OSS AI Agent.

This is the primary entry point that uses the modern restructured codebase.
"""

import sys
from pathlib import Path

# Add src directory to path for development
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

try:
    from gpt_oss_agent.cli import main
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"Error importing restructured modules: {e}")
    print("Make sure to install the package with: pip install -e .")
    sys.exit(1)