# GPT-OSS AI Agent

A Python-based AI agent that connects to your locally-hosted GPT-OSS model via vLLM, with web search capabilities, using OpenAI Agents SDK.

## Quick Start

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Configure (optional):**
   - Get an API key from [exa.ai](https://exa.ai) for web search
   - `mv .env.example .env`
   - Edit `.env`: set `EXA_API_KEY=your_key_here`

3. **Run the agent:**

```bash
python main.py
```

## Usage Examples

```bash
# Interactive chat (default)
python main.py

# Single question
python main.py --chat "What's the weather like today?"

# Test connection
python main.py --test

# Use specific model
python main.py --model gpt-oss-20b

# Show system info
python main.py --info

# Non-streaming mode
python main.py --no-stream
```

## Interactive Commands

Once in the chat interface:
- `/help` - Show help
- `/info` - Agent information
- `/tools` - Tools status
- `/toggle-stream` - Toggle streaming
- `/quit` - Exit

## Features

- ✅ **Local GPT-OSS Integration** - Connects to vLLM server at localhost:8000
- ✅ **Web Search** - Real-time web search via Exa API
- ✅ **Streaming Responses** - Real-time response streaming
- ✅ **Rich CLI** - Beautiful command-line interface
- ✅ **Tool Support** - Extensible tool system
- ✅ **Error Handling** - Robust error management

## Prerequisites

- vLLM server running GPT-OSS model at localhost:8000
- Python 3.8+
- Optional: Exa API key for web search

## File Structure

```
gpt-agent/
├── main.py          # Entry point
├── cli.py           # Interactive CLI
├── agent.py         # Main agent logic
├── model_client.py  # vLLM connection
├── tools.py         # Web search tools
├── streaming.py     # Response streaming
├── requirements.txt # Dependencies
├── .env            # Configuration
└── README.md       # This file
```