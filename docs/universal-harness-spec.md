# Universal Harness Context Proxy Architecture
**Target: `laya-router` as a Harness-Agnostic, Zero-Inference Context Optimization Gateway**

## 1. Executive Summary & Philosophy
`laya-router` acts as a transparent, high-speed reverse proxy between coding agents (Hermes, Claude Code, OpenCode, Aider, Neovim Avante) and OpenAI-compatible local model backends (`llama-swap`, `llama-server`, `vLLM`). 

Unlike monolithic in-harness plugins (which fight agent state machines) or black-box daemons like Sleev (which run hidden, expensive auxiliary LLM passes without feedback), `laya-router` operates deterministically at the HTTP wire protocol level (`/v1/chat/completions`) with sub-millisecond overhead, radical transparency, and zero local/cloud inference cost.

---

## 2. Universal Protocol Architecture

```
[Agent Harness] (Hermes / Claude Code / OpenCode / Aider / Neovim)
       │
       │ HTTP /v1/chat/completions (OPENAI_BASE_URL=http://127.0.0.1:8090/v1)
       ▼
┌────────────────────────────────────────────────────────┐
│ 1. Harness Auto-Detection & Payload Normalization      │
│ • Detect client dialect (OpenAI, Anthropic adapters)   │
│ • Normalize tool envelopes & system prompt structures  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 2. Semantic Intent Classification (DCP Layer)          │
│ • Heuristic categorization: read / search / exec       │
│ • Deduplicate redundant idempotent tool invocations    │
│ • Tombstone resolved error tracebacks (>3 turns ago)   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 3. Pinned-Head + Sliding-Tail Windowing                │
│ • Option B: Pin conversational spec & prompt setup     │
│ • Safe-Tail: Strict tool_call_id boundary protection   │
│ • Authoritative execution marker for omitted middle    │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ 4. Multi-Tier Model & MoE Tool Router                  │
│ • Classify task complexity via Laya systemone          │
│ • Route basic tool iterations to fast MoE lanes        │
│ • Route complex synthesis to primary coding models     │
└──────────────────────────┬─────────────────────────────┘
       │
       ▼
[Local Model Backends] (llama-swap :8080 -> V320 / 7700XT)
```

---

## 3. Core Functional Pillars

### A. Semantic Tool Intent Classification
Rather than hardcoding tool names (`read_file`, `terminal`), the proxy generalizes using intent heuristics:
* **Read Intent:** `{"read", "view", "cat", "get", "fetch", "inspect", "show"}`
* **Search Intent:** `{"search", "grep", "find", "glob", "locate", "list", "ls"}`
* **Exec Intent:** `{"bash", "terminal", "exec", "cmd", "run", "sh"}`

**Behavioral Rules:**
1. **Idempotent Deduplication:** Repeated calls matching `read` or `search` with identical normalized arguments (sorted JSON keys / trimmed paths) supersede earlier occurrences. The earlier payload is stubbed:
   `"[Duplicate {tool_name} pruned: {N} chars. Superseded by newer call at turn {T}.]"`
2. **Resolved Error Tombstoning:** Any tool resulting in an exception, stack trace, or non-zero exit older than the active tail is tombstoned to its core error signature:
   `"[Historical error output pruned: {N} chars. Summary: {err_summary}. Resolved in subsequent turns.]"`

### B. Dialect & Payload Normalization
* **Standard Tool Calls:** Standard `tool_calls: [...]` and `role: "tool"` pairs.
* **Block-Content Formats:** Anthropic-style content blocks (`type: "tool_use"`, `type: "tool_result"`) parsed and pruned in-place without altering block identifiers.
* **Legacy Function Calling:** `function_call` and `role: "function"` mapped to standard envelope rules.

### C. Client Environment Configurations
Clients integrate with zero codebase modifications:
* **Generic Shell:**
  ```bash
  export OPENAI_BASE_URL="http://127.0.0.1:8090/v1"
  export OPENAI_API_KEY="local-key"
  ```
* **Claude Code:**
  ```bash
  export ANTHROPIC_BASE_URL="http://127.0.0.1:8090"
  ```
* **OpenCode (`~/.config/opencode/config.json`):**
  ```json
  {
    "providers": {
      "local": {
        "baseURL": "http://127.0.0.1:8090/v1"
      }
    }
  }
  ```
* **Aider:**
  ```bash
  aider --openai-api-base http://127.0.0.1:8090/v1
  ```
* **Neovim (Avante.nvim / CodeCompanion):**
  ```lua
  provider = "openai",
  endpoint = "http://127.0.0.1:8090/v1"
  ```

---

## 4. Radical Transparency & Telemetry
Every proxied response carries explicit metrics headers, eliminating the opaque black-box problem:
* `X-DCP-Dedupes`: Count of redundant idempotent tool calls pruned.
* `X-DCP-Errors-Purged`: Count of historical stack traces tombstoned.
* `X-DCP-Reclaimed-Chars`: Exact count of characters eliminated from the payload.
* `X-Router-Tier`: Active backend tier dispatched (`tier1-v320`, `tier2-7700xt`, etc.).
* `X-Router-Model`: Active model handling the request.
