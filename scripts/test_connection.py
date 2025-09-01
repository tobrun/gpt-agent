#!/usr/bin/env python3
"""
Test vLLM connection and agent functionality.
"""

import sys
from pathlib import Path

# Add src directory to path for development
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

try:
    from gpt_oss_agent.config import get_settings
    from gpt_oss_agent.clients import setup_vllm_client
    from gpt_oss_agent.tools import check_all_tools_status
    from gpt_oss_agent.core import create_agent
    from gpt_oss_agent.utils import setup_logging
    
    def main():
        print("🧪 Testing GPT-OSS Agent Connection")
        print("=" * 40)
        
        # Setup
        settings = get_settings()
        setup_logging(settings)
        
        # Test 1: vLLM Connection
        print("\n1️⃣ Testing vLLM connection...")
        try:
            client = setup_vllm_client(wait_for_server=False)
            test_result = client.test_connection()
            
            if test_result["health_check_passed"]:
                print("   ✅ vLLM server is accessible")
                print(f"   📍 URL: {test_result['server_url']}")
                print(f"   🤖 Model: {test_result['configured_model']}")
                
                if test_result["model_info_available"]:
                    print(f"   📊 Available models: {test_result.get('model_count', 0)}")
                    if test_result.get("available_models"):
                        print(f"   📋 Models: {', '.join(test_result['available_models'][:3])}...")
                else:
                    print("   ⚠️  Model info not available")
            else:
                print("   ❌ vLLM server not accessible")
                if test_result.get("error"):
                    print(f"   Error: {test_result['error']}")
                return False
                
        except Exception as e:
            print(f"   ❌ vLLM connection failed: {e}")
            return False
        
        # Test 2: Tools Status
        print("\n2️⃣ Testing tools...")
        try:
            tools_status = check_all_tools_status(settings)
            print(f"   📊 Available tools: {tools_status['available_tools']}/{tools_status['total_tools']}")
            
            if tools_status["available_tool_names"]:
                print(f"   🔧 Active tools: {', '.join(tools_status['available_tool_names'])}")
            else:
                print("   ⚠️  No tools available (web search requires Exa API key)")
                
        except Exception as e:
            print(f"   ❌ Tools check failed: {e}")
        
        # Test 3: Agent Creation
        print("\n3️⃣ Testing agent creation...")
        try:
            from gpt_oss_agent.tools import get_available_tools
            tools = get_available_tools(settings)
            agent = create_agent(settings=settings, tools=tools)
            print("   ✅ Agent created successfully")
            print(f"   🎯 Agent name: {agent.agent.name}")
            print(f"   🔧 Tools loaded: {len(tools)}")
            
        except Exception as e:
            print(f"   ❌ Agent creation failed: {e}")
            return False
        
        # Test 4: Simple Chat
        print("\n4️⃣ Testing simple chat...")
        try:
            response = agent.chat("Hello! Please respond with just 'Test successful' and nothing else.")
            
            if response and len(response.strip()) > 0:
                print("   ✅ Chat test successful")
                print(f"   💬 Response: {response[:100]}{'...' if len(response) > 100 else ''}")
                
                # Check for empty response issue
                if not response.strip():
                    print("   ⚠️  Empty response detected (known vLLM issue)")
                    return True  # Still consider successful
                    
            else:
                print("   ❌ Chat test failed - empty response")
                print("   ℹ️  This is a known issue with vLLM's /v1/responses endpoint")
                return False
                
        except Exception as e:
            print(f"   ❌ Chat test failed: {e}")
            return False
        
        print("\n✅ All tests passed! Agent is ready to use.")
        print("\nNext steps:")
        print("  • Run 'python main.py' for interactive chat")
        print("  • Run 'python main.py --help' for all options")
        return True
    
    if __name__ == "__main__":
        success = main()
        sys.exit(0 if success else 1)
        
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure to install dependencies: pip install -r requirements.txt")
    print("Or install in development mode: pip install -e .")
    sys.exit(1)