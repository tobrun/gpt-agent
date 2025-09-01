# GPT-OSS Agent v0.2.0

A modern Python-based AI agent that connects to your locally-hosted GPT-OSS model via vLLM, with web search capabilities, using OpenAI Agents SDK.

## ✨ What's New in v0.2.0

- **🏗️ Modern Python Structure**: Proper package structure with `src/` layout
- **⚙️ Type-Safe Configuration**: Pydantic-based configuration management  
- **🔧 Improved Tooling**: Structured tool system with registry
- **📊 Better Logging**: Enhanced logging and debug capabilities
- **🧪 Testing Ready**: Foundation for comprehensive testing
- **📦 Pip Installable**: Proper Python packaging with `pyproject.toml`

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd gpt-agent

# Install in development mode (recommended)
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

### Configuration

1. **Optional - Web Search**: Get an API key from [exa.ai](https://exa.ai)
2. **Copy environment template**: `cp .env.example .env`
3. **Edit configuration**: Set `EXA_API_KEY=your_key_here` in `.env`

### Usage

```bash
# Test the connection
python scripts/test_connection.py

# Interactive chat
python main.py

# Single question
python main.py --chat "What's the weather like today?"

# Show system info
python main.py --info

# Test functionality
python main.py --test

# Use specific model
python main.py --model gpt-oss-20b

# Non-streaming mode
python main.py --no-stream
```

### Interactive Commands

Once in chat mode:
- `/help` - Show help
- `/info` - Agent information  
- `/tools` - Tools status
- `/debug` - Debug session info
- `/toggle-stream` - Toggle streaming
- `/quit` - Exit

## 📁 Project Structure

```
gpt-oss-agent/
├── src/gpt_oss_agent/          # Main package
│   ├── core/                   # Core agent functionality
│   ├── clients/                # External service clients
│   ├── tools/                  # Tool implementations
│   ├── cli/                    # CLI interface
│   ├── utils/                  # Utilities
│   └── api/                    # Future API endpoints
├── tests/                      # Test suite
├── scripts/                    # Utility scripts
├── docs/                       # Documentation
├── examples/                   # Usage examples
├── pyproject.toml             # Modern Python config
└── README.md                  # This file
```

## 🛠️ Development

### Using the New Structure

```python
# Import the agent
from gpt_oss_agent import create_agent, get_settings

# Create and use agent
settings = get_settings()
agent = create_agent(settings=settings)
response = agent.chat("Hello!")
```

### Configuration Management

```python
from gpt_oss_agent.config import get_settings, Settings

# Get current settings
settings = get_settings()

# Create custom settings
custom_settings = Settings(
    vllm__model="gpt-oss-20b",
    exa__api_key="your-key-here"
)
```

### Debug Logging

```bash
# View debug sessions
python scripts/view_debug_logs.py --list

# View session details
python scripts/view_debug_logs.py --session 20250901_123456

# View detailed message logs
python scripts/view_debug_logs.py --session 20250901_123456 --detailed 1
```

## 🔧 Tools & Features

- ✅ **Local GPT-OSS Integration** - Connects to vLLM server
- ✅ **Web Search** - Real-time web search via Exa API  
- ✅ **Streaming Responses** - Real-time response streaming
- ✅ **Rich CLI** - Beautiful command-line interface
- ✅ **Tool Registry** - Extensible tool system
- ✅ **Debug Logging** - Comprehensive debugging
- ✅ **Type Safety** - Full type hints and validation
- ✅ **Modern Config** - Pydantic-based configuration

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [Configuration Guide](docs/configuration.md)  
- [Architecture Overview](docs/architecture.md)
- [Troubleshooting](docs/troubleshooting.md)
- [vLLM Setup](docs/VLLM_GPT_OSS.md)

## 🧪 Testing

```bash
# Run tests (when implemented)
pytest

# Test with coverage
pytest --cov=gpt_oss_agent

# Type checking
mypy src/

# Code formatting
black src/ tests/
ruff check src/ tests/
```

## 🆘 Troubleshooting

### Common Issues

1. **Empty Responses**: Known vLLM `/v1/responses` endpoint issue
   - Check [troubleshooting guide](docs/troubleshooting.md)
   - Use debug logging: Set `DEBUG__ENABLED=true` in `.env`

2. **Connection Errors**: vLLM server not accessible
   - Verify server is running: `curl http://localhost:8000/v1/models`
   - Check configuration: `python main.py --info`

3. **No Web Search**: Exa API key not configured
   - Get key from [exa.ai](https://exa.ai)
   - Set in `.env`: `EXA_API_KEY=your_key`

### Debug Logging

Debug logs are saved to `logs/debug/` when enabled:

```bash
# Enable debug logging
echo "DEBUG__ENABLED=true" >> .env

# View logs
python scripts/view_debug_logs.py --list
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Install dev dependencies**: `pip install -e .[dev]`
4. **Make your changes**
5. **Run tests**: `pytest`
6. **Format code**: `black . && ruff check .`
7. **Commit**: `git commit -m 'Add amazing feature'`
8. **Push**: `git push origin feature/amazing-feature`
9. **Open Pull Request**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
- [vLLM](https://github.com/vllm-project/vllm) 
- [Exa Search](https://exa.ai)
- [Rich](https://rich.readthedocs.io/) for beautiful CLI