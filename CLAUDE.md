# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a GPT-OSS AI Agent that connects to a locally-hosted vLLM server running GPT-OSS models. The architecture consists of several key layers:

### Core Components

1. **OpenAI Agents SDK Integration**: Uses OpenAI's Agents SDK as the orchestration layer, configured to work with local vLLM instead of OpenAI's servers
2. **vLLM Client Layer** (`model_client.py`): Manages connection to local vLLM server with health checks, retries, and environment configuration
3. **Agent Wrapper** (`agent.py`): `GPTOSSAgent` class that wraps the OpenAI Agent with local model configuration and tool integration
4. **Tool System** (`tools.py`): Function-based tools using `@function_tool` decorator, primarily for web search via Exa API
5. **CLI Interface** (`cli.py`): Rich-based interactive CLI with streaming support and command handling

### Critical Architecture Details

- **Privacy-First Design**: All OpenAI tracing/telemetry is explicitly disabled via `set_tracing_disabled(True)` and environment variables
- **Environment Variable Configuration**: Uses environment variables rather than `set_default_openai_client()` to configure the OpenAI SDK for local operation
- **Tool Registration**: Tools are dynamically registered based on API key availability (EXA_API_KEY)
- **Streaming Architecture**: Supports both sync and async response modes, though sync is more reliable with vLLM

## Development Commands

### Basic Operations
```bash
# Install dependencies
pip install -r requirements.txt

# Run interactive agent
python main.py

# Test connection and configuration
python main.py --test

# Quick single interaction
python main.py --chat "your question"

# Show system information
python main.py --info
```

### Configuration Management
```bash
# Configure environment (required for web search)
# Edit .env file and set:
# EXA_API_KEY=your_key_here
# VLLM_BASE_URL=http://localhost:8000/v1  
# DEFAULT_MODEL=gpt-oss-120b
```

### vLLM Server Requirements
- vLLM server must be running at `localhost:8000` with GPT-OSS model loaded
- Server should support both `/v1/chat/completions` and `/v1/responses` endpoints
- Health endpoint available at `/health` (without /v1 prefix)

## Key Implementation Patterns

### Tool Creation
Tools must use the `@function_tool` decorator from the agents library:
```python
from agents import function_tool

@function_tool
def your_tool(param: str) -> str:
    """Tool description for the agent."""
    # Implementation
    return result
```

### Agent Initialization Order
1. Setup vLLM client first (handles environment configuration)
2. Initialize tools array based on API key availability
3. Generate instructions (references available tools)
4. Create OpenAI Agent with tools and instructions

### Privacy Configuration
The codebase explicitly disables all OpenAI telemetry:
- `set_tracing_disabled(True)` - SDK-level tracing disable
- `os.environ['OPENAI_AGENTS_DISABLE_TRACING'] = "1"` - Environment-level disable
- Uses dummy API key for local operation

### Error Handling Patterns
- Connection testing with health checks and retries
- Graceful degradation when tools are unavailable (missing API keys)
- Proper exception handling in both sync and async contexts

## Model and Tool Integration

### Supported Models
- `gpt-oss-20b` - Faster, smaller model
- `gpt-oss-120b` - Larger, more capable model (default)

### Tool System
- **Web Search**: `web_search(query, num_results)` - Searches web via Exa API
- **Page Content**: `get_page_content(url)` - Retrieves specific webpage content
- Tools are conditionally enabled based on EXA_API_KEY configuration

### CLI Commands
Interactive CLI supports these commands:
- `/help`, `/info`, `/tools` - Information commands
- `/toggle-stream`, `/toggle-reasoning` - Mode toggles
- `/history`, `/clear` - Session management
- `/quit`, `/exit` - Exit commands

## Environment Configuration

Required environment variables in `.env`:
- `EXA_API_KEY` - Web search functionality (get from exa.ai)
- `VLLM_BASE_URL` - vLLM server endpoint (default: localhost:8000/v1)
- `DEFAULT_MODEL` - Model name (default: gpt-oss-20b)

The agent automatically configures `OPENAI_BASE_URL` and `OPENAI_API_KEY` environment variables to point the OpenAI SDK to the local vLLM server.