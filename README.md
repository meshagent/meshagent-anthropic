# [Meshagent](https://www.meshagent.com)

## MeshAgent Anthropic
The `meshagent.anthropic` package provides adapters to integrate Anthropic models (Messages API) with MeshAgent tools and agents.

### Messages Adapter
- `AnthropicMessagesAdapter`: wraps the Anthropic Messages API. It turns `Toolkit` objects into Anthropic tool definitions, executes tool calls, and returns the final assistant response.

```Python Python
from meshagent.anthropic import AnthropicMessagesAdapter

adapter = AnthropicMessagesAdapter(model="claude-3-5-sonnet-latest")
```

Compaction can be enabled through Anthropic context management:

```Python Python
adapter = AnthropicMessagesAdapter(
    model="claude-sonnet-4-5",
    context_management="auto",
    compaction_threshold=150000,
)
```

### Tool Response Adapter
`AnthropicMessagesToolResponseAdapter` converts a tool's structured response into Anthropic `tool_result` blocks that can be inserted back into the conversation.

---
### Learn more about MeshAgent on our website or check out the docs for additional examples!

**Website**: [www.meshagent.com](https://www.meshagent.com/)

**Documentation**: [docs.meshagent.com](https://docs.meshagent.com/)

---
