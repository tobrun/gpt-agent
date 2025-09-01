# GPT-OSS Agent Troubleshooting

## Issue: Empty Responses with Tool Use

### Problem Description
When using tools (web search), the agent sometimes returns empty responses despite successful tool execution.

### Root Cause
vLLM's `/v1/responses` API endpoint has compatibility issues with complex tool workflows. The OpenAI Agents SDK calls the responses endpoint, but vLLM doesn't always generate a final message after tool calls complete.

### Symptoms
- Tool calls execute successfully (web search works)
- Search results are retrieved (4000+ characters)
- Agent reasoning shows in logs
- Final output is empty or None
- No final `MessageOutputItem` in `result.new_items`

### Current Investigation
The issue occurs specifically with:
1. Complex tool workflows (multiple tool calls)
2. vLLM's `/v1/responses` endpoint implementation
3. OpenAI Agents SDK expecting specific response format

### Debug Evidence

From debug logs session `20250901_202534`:

**Tools Execute Successfully:**
- `web_search` tool: ✅ returns 4655 characters
- `get_page_content` tool: ✅ executed successfully
- Agent shows reasoning in `reasoning_item` entries

**Runner Result Analysis:**
```json
{
  "has_final_output": true,
  "final_output": "",  # ← EMPTY STRING
  "final_output_length": 0,
  "new_items_count": 26
}
```

**Key Finding:** The runner has 26 `new_items` including:
- `ReasoningItem` - Agent thinking process works
- `ToolCallItem` - Tool calls are generated correctly  
- `ToolCallOutputItem` - Tool outputs are captured
- **Missing:** No `MessageOutputItem` with final response text

**Root Cause Confirmed:** vLLM's `/v1/responses` endpoint processes tool calls and reasoning but fails to generate the final user-facing message after tool workflow completion.

### Temporary Workaround
Enhanced debugging shows tools work but final response generation fails in vLLM responses API.

### Potential Solutions

**Option 1: Extract from Reasoning Items**
Since reasoning shows the agent understands the query and processes tool results, we could extract the final reasoning as a fallback response.

**Option 2: Force Message Generation**
Modify the agent to explicitly request a final user message after tool completion.

**Option 3: Switch to Chat Completions API**
Test if using only `/v1/chat/completions` endpoint (not `/v1/responses`) resolves the issue.

**Option 4: Custom Response Handler**
Implement a custom response extraction that processes the reasoning items into a user-facing answer.

### Next Steps
1. ✅ **Debug logging implemented** - Issue root cause identified
2. Test Option 3: Chat completions API only
3. Implement Option 4: Custom response extraction from reasoning items
4. Add timeout handling for stuck responses