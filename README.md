# Laya Decision Router

A lightweight, local-first tiered LLM decision router. Uses [Laya](https://github.com/jamestb/laya) (ModernBERT classifier) running on CPU to classify incoming prompt intent in ~8–12 ms and dispatch requests to the optimal local or cloud tier without GPU VRAM overhead.

## Architecture & Ports

```text
                  [Client: Hermes / CLI / Scripts]
                                 │
                                 ▼
                     [Thin Router (:8090)]
                                 │
        (1. Classify Prompt)     ▼
      ┌───────────────── [Laya Server (:8081)]
      │                  (ModernBERT CPU, ~10ms)
      │
      │ 2. Dispatched Route:
      ├─── Tier 1 (Simple / Aux):
      │    └──► [llama-swap :8080] ──► V320 (Tiel-Coder-35B / Dirk-9B)
      │
      ├─── Tier 2 (Deep Local / Coding):
      │    └──► [llama-swap :8080] ──► 7700 XT (Ternary-Bonsai-2-27B)
      │
      └─── Tier 3 (Frontier / Massive Context / Cloud):
           └──► [llm-keypool :8005] ──► Free Cloud Pool (Gemini, Groq, Mistral, Cerebras...)
```

### Port Allocation
* `:8080` — `llama-swap` (`llama-swap.service`)
* `:8081` — `laya-cli` HTTP server (`laya.service`)
* `:8005` — `llm-keypool proxy` (`llm-keypool.service`)
* `:8090` — Laya Decision Router (`laya-router.service`)

## Routing Logic (`model: "auto"`)
* **Large Context (> 24,000 chars):** Dispatched to `llm-keypool:8005` with `X-Keypool-Capabilities: large_context` (Google Gemini 1M window).
* **Complex Coding / Systems:** Dispatched to `llama-swap:8080` rewrote to `Ternary-Bonsai-2-27B-7700XT` on RX 7700 XT.
* **Frontier / Live Web:** Dispatched to `llm-keypool:8005` with `X-Keypool-Capabilities: general_purpose` / `agentic`.
* **Routine / Simple Queries:** Dispatched to `llama-swap:8080` rewrote to `Tiel-Coder-35B-A3B-MTP-IQ3_XXS` on Radeon PRO V320.

Explicit models (`local-v320`, `local-7700xt`, `cloud-keypool`, or provider model names) bypass classification and route directly.

## Systemd Units
Service unit templates are in `systemd/`:
* `systemd/laya.service`
* `systemd/llm-keypool.service`
* `systemd/laya-router.service`
