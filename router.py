import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from vram_calculator import calculate_fit

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("laya-router")

CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

HOST = config.get("server", {}).get("host", "127.0.0.1")
PORT = config.get("server", {}).get("port", 8090)

UPSTREAM_LAYA = config.get("upstreams", {}).get("laya", "http://127.0.0.1:8081")
UPSTREAM_LLAMA_SWAP = config.get("upstreams", {}).get("llama_swap", "http://127.0.0.1:8080")
UPSTREAM_KEYPOOL = config.get("upstreams", {}).get("keypool", "http://127.0.0.1:8005")
TIER1_SEMAPHORE = asyncio.Semaphore(2)  # was misspelled TEIR1_SEMAPHORE: every tier-1 request raised NameError
TIER1_V320 = config.get("defaults", {}).get("tier1_v320_model", "Ternary-Bonsai-2-27B-PQ2_0")  # Tiel tier removed 2026-09-24
TIER2_7700XT = config.get("defaults", {}).get("tier2_7700xt_model", "Swift-Bonsai-2-7700XT")
LARGE_CONTEXT_CHARS = config.get("defaults", {}).get("large_context_chars", 24000)

CODING_THRESHOLD = config.get("thresholds", {}).get("complex_coding_noul", 0.50)
CLOUD_THRESHOLD = config.get("thresholds", {}).get("frontier_cloud_noul", 0.65)


def _prune_max_chars() -> int:
    """The pruner's message budget, derived rather than hardcoded.

    The pruner only sees `messages`; the static floor (system prompt + tool schemas + skills +
    memory) sits outside it and is charged against the same lane. A round 80000 meant messages of
    80000 chars plus a ~74k-char floor could exceed a 32k-token lane -- the 2026-09-25 overflow,
    where the request died with `exceeds the available context size` instead of compacting.

    Assumes ~3 chars/token, NOT 4. The first derivation used 4 and a worker still overflowed at
    11:21 on 2026-09-25, ~15 min after the declaration fix: code-heavy context runs closer to 3
    chars/token, which put the same 51,072 chars at ~17k tokens and the composed total over the lane.
    The conservative ratio costs a little message budget and removes the failure mode.
    Floored at 8192 so a misconfigured pair can never drive the budget to zero. dfl/audit-declarations.py
    CI-4 re-derives this from the MEASURED floor and the lane's real --ctx-size.
    """
    d = config.get("defaults", {})
    lane = int(d.get("lane_ctx_tokens", 32768))
    reserve = int(d.get("reserve_floor_tokens", 20000))
    return max(8192, (lane - reserve) * 3)

app = FastAPI(title="Laya Smart LLM Router", version="1.0.0")
client: Optional[httpx.AsyncClient] = None


@app.on_event("startup")
async def startup():
    global client
    client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=60.0))


@app.on_event("shutdown")
async def shutdown():
    global client
    if client:
        await client.aclose()


@app.get("/health")
async def health():
    statuses = {}
    assert client is not None

    # Check Laya
    try:
        r = await client.get(f"{UPSTREAM_LAYA}/health", timeout=2.0)
        statuses["laya"] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
    except Exception as e:
        statuses["laya"] = f"down ({e.__class__.__name__})"

    # Check Llama-Swap
    try:
        r = await client.get(f"{UPSTREAM_LLAMA_SWAP}/v1/models", timeout=2.0)
        statuses["llama_swap"] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
    except Exception as e:
        statuses["llama_swap"] = f"down ({e.__class__.__name__})"

    # Check Keypool
    try:
        r = await client.get(f"{UPSTREAM_KEYPOOL}/health", timeout=2.0)
        statuses["keypool"] = "ok" if r.status_code == 200 else f"http_{r.status_code}"
    except Exception as e:
        statuses["keypool"] = f"down ({e.__class__.__name__})"

    return {
        "status": "ok",
        "router": "listening",
        "upstreams": statuses,
    }


@app.get("/v1/models")
async def list_models():
    assert client is not None
    virtual_models = [
        {"id": "auto", "object": "model", "owned_by": "laya-router", "permission": []},
        {"id": "local-v320", "object": "model", "owned_by": "laya-router", "permission": []},
        {"id": "local-7700xt", "object": "model", "owned_by": "laya-router", "permission": []},
        {"id": "cloud-keypool", "object": "model", "owned_by": "laya-router", "permission": []},
    ]

    upstream_models = []
    try:
        r = await client.get(f"{UPSTREAM_LLAMA_SWAP}/v1/models", timeout=3.0)
        if r.status_code == 200:
            data = r.json()
            upstream_models = data.get("data", [])
    except Exception:
        pass

    return {
        "object": "list",
        "data": virtual_models + upstream_models,
    }


async def classify_prompt(prompt: str) -> tuple[float, float]:
    """Query Laya systemone to get coding and cloud noul scores."""
    assert client is not None
    payload = {
        "state": prompt,
        "questions": {
          "coding": {
            "type": "noul",
            "instructions": "Is this a complex coding, deep math, or low-level systems programming task?"
          },
          "cloud": {
            "type": "noul",
            "instructions": "Does this request need current web info, broad world knowledge, or massive frontier reasoning?"
          }
        }
    }
    try:
        r = await client.post(f"{UPSTREAM_LAYA}/v1/systemone", json=payload, timeout=6.0)
        if r.status_code == 200:
            data = r.json()
            answers = data.get("answers", {})
            coding_score = float(answers.get("coding", {}).get("noul", 0.0))
            cloud_score = float(answers.get("cloud", {}).get("noul", 0.0))
            return coding_score, cloud_score
    except Exception as e:
        logger.warning(f"Laya classification failed: {type(e).__name__}: {repr(e)}")
    return 0.0, 0.0


IDEMPOTENT_TOOLS = {"read_file", "search_files", "lcm_grep", "lcm_describe"}


def normalize_tool_args(raw_args: str) -> str:
    try:
        data = json.loads(raw_args)
        if isinstance(data, dict):
            return json.dumps(data, sort_keys=True)
    except Exception:
        pass
    return raw_args.strip()


def extract_error_summary(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "Unknown error"
    for line in reversed(lines):
        for err_prefix in ("Error:", "Exception:", "SyntaxError:", "ValueError:", "TypeError:", "FileNotFoundError:"):
            if err_prefix in line:
                return line[:100]
    return lines[0][:100]


def is_error_content(text: str) -> bool:
    if "Traceback (most recent call last):" in text:
        return True
    if any(k in text for k in ("SyntaxError:", "FileNotFoundError:", "ImportError:", "ModuleNotFoundError:")):
        return True
    if '"exit_code": 1' in text or '"exit_code": 2' in text or '"exit_code": 127' in text:
        return True
    if text.strip().startswith("Error:") or text.strip().startswith("error:"):
        return True
    return False


def _truncate_middle(text: str, limit: int) -> str:
    """Keep the head and tail of `text`, drop the middle, with an explicit marker.

    Truncating a message's content in place cannot break the chat envelope: no message is
    added or removed and no field other than `content` is touched, so role sequencing and
    tool_call/tool_result pairing both survive. That is the property Pass 4 lacks, which is
    why Pass 4 needs a large enough message count to work with at all.
    """
    if len(text) <= limit:
        return text
    marker = f"\n\n[... {len(text)} chars; middle omitted by the laya pruner to fit the "
    marker += "target lane. Head and tail retained verbatim. ...]\n\n"
    room = max(0, limit - len(marker))
    if room <= 0:
        return marker[:limit]
    head = max(1, int(room * 0.65))
    tail = room - head
    return text[:head] + marker + (text[len(text) - tail:] if tail else "")


def _truncate_oversize(
    msgs: list[dict[str, Any]], budget: int, stats: dict[str, Any]
) -> tuple[list[dict[str, Any]], int]:
    """Shrink the largest message contents until the total fits `budget`.

    Pass 4 can only DROP whole messages, so it gives up when there are few of them
    (`len(msgs) <= min_tail_messages + 2`). A prompt that is one enormous message -- a pasted
    log, a large retrieved document, a tool result that Pass 3 could not touch because it sat
    inside the protected recent window -- then passes through every pass untouched and hits the
    lane wall. Measured 2026-09-26: "Pruned: 103508 -> 103508 chars (1 -> 1 msgs)" on three
    consecutive requests, all of which died with `proxy error: EOF` after minutes of retrying.

    `system` is truncated only after every other role has given what it can: its middle holds
    the agent's tool-use rules, which is the last thing worth losing.
    """
    total = sum(len(str(m.get("content", ""))) for m in msgs)
    floor = 200  # below this a message is a stub already; truncating it buys nothing

    while total > budget:
        candidates = [i for i in range(len(msgs)) if len(str(msgs[i].get("content", ""))) > floor]
        if not candidates:
            break
        # largest first, but never start on a system message while a normal one can give
        candidates.sort(
            key=lambda i: (
                msgs[i].get("role") == "system",
                -len(str(msgs[i].get("content", ""))),
            )
        )
        idx = candidates[0]
        cur = str(msgs[idx].get("content", ""))
        new_text = _truncate_middle(cur, max(floor, len(cur) - (total - budget)))
        if len(new_text) >= len(cur):
            break  # no progress possible; stop rather than spin
        msgs[idx] = dict(msgs[idx])
        msgs[idx]["content"] = new_text
        stats["oversize_truncated"] += 1
        stats["reclaimed_chars"] += len(cur) - len(new_text)
        total = sum(len(str(m.get("content", ""))) for m in msgs)

    return msgs, total


def prune_messages(
    messages: list[dict[str, Any]],
    max_chars: int = 80000,
    keep_recent_tool_turns: int = 3,
    min_tail_messages: int = 10,
) -> tuple[list[dict[str, Any]], bool, int, int, dict[str, Any]]:
    """Deterministically prune messages using DCP strategies + Donut windowing.

    Pass 1: Deduplicate redundant idempotent tool reads (read_file, search_files).
    Pass 2: Tombstone resolved error tracebacks older than keep_recent_tool_turns.
    Pass 3: Truncate ephemeral tool outputs (>3 turns ago) while preserving OpenAI envelope.
    Pass 4: Pinned-Head (Option B) + Sliding-Tail if context still exceeds max_chars.
    Pass 5: Truncate the middle of the largest over-budget messages if Pass 4 could not free
            enough (few messages, or one very large one). Content-only, so envelope-safe.
    """
    stats = {
        "dedupes": 0,
        "errors_purged": 0,
        "evicted_tools": 0,
        "donut_omitted": 0,
        "oversize_truncated": 0,
        "reclaimed_chars": 0,
    }
    if not messages:
        return messages, False, 0, 0, stats

    msgs = [dict(m) for m in messages if isinstance(m, dict)]
    original_chars = sum(len(str(m.get("content", ""))) for m in msgs)

    # Index assistant tool calls
    tool_call_turn_indices = []
    tool_calls_by_id = {}
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            tool_call_turn_indices.append(i)
            for tc in m.get("tool_calls", []):
                tc_id = tc.get("id")
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = normalize_tool_args(fn.get("arguments", ""))
                tool_calls_by_id[tc_id] = (i, name, args)

    # Protected recent cutoff (last keep_recent_tool_turns)
    protected_turn_cutoff = -1
    if len(tool_call_turn_indices) > keep_recent_tool_turns:
        protected_turn_cutoff = tool_call_turn_indices[-keep_recent_tool_turns]

    # PASS 1: DCP Redundant Tool Deduplication
    latest_idempotent_call = {}
    for tc_id, (turn_idx, name, args) in tool_calls_by_id.items():
        if name in IDEMPOTENT_TOOLS:
            key = (name, args)
            if key not in latest_idempotent_call or turn_idx > latest_idempotent_call[key][0]:
                latest_idempotent_call[key] = (turn_idx, tc_id)

    pruned_tool_call_ids = set()
    for i, m in enumerate(msgs):
        if m.get("role") == "tool":
            tc_id = m.get("tool_call_id")
            if tc_id in tool_calls_by_id:
                turn_idx, name, args = tool_calls_by_id[tc_id]
                key = (name, args)
                if name in IDEMPOTENT_TOOLS and latest_idempotent_call.get(key)[1] != tc_id:
                    content = str(m.get("content", ""))
                    if len(content) > 200:
                        latest_turn = latest_idempotent_call[key][0]
                        stub = f"[Duplicate {name} pruned: {len(content)} chars. Superseded by newer call at turn {latest_turn}.]"
                        msgs[i] = dict(m)
                        msgs[i]["content"] = stub
                        pruned_tool_call_ids.add(tc_id)
                        stats["dedupes"] += 1
                        stats["reclaimed_chars"] += (len(content) - len(stub))

    # PASS 2 & 3: Resolved Error Tombstoning and Ephemeral Tool Truncation
    for i, m in enumerate(msgs):
        if m.get("role") == "tool":
            tc_id = m.get("tool_call_id")
            if tc_id in pruned_tool_call_ids:
                continue

            if protected_turn_cutoff != -1 and i < protected_turn_cutoff:
                content = str(m.get("content", ""))
                if len(content) > 300:
                    msgs[i] = dict(m)
                    if is_error_content(content):
                        err_summary = extract_error_summary(content)
                        stub = (
                            f"[Historical error output pruned: {len(content)} chars. "
                            f"Summary: {err_summary}. Resolved in subsequent turns.]"
                        )
                        msgs[i]["content"] = stub
                        stats["errors_purged"] += 1
                        stats["reclaimed_chars"] += (len(content) - len(stub))
                    else:
                        stub = (
                            f"[Output evicted: {len(content)} chars truncated to preserve working window. "
                            f"Ground truth preserved in LCM history / on disk.]"
                        )
                        msgs[i]["content"] = stub
                        stats["evicted_tools"] += 1
                        stats["reclaimed_chars"] += (len(content) - len(stub))

    chars_after_tool_pruning = sum(len(str(m.get("content", ""))) for m in msgs)
    if chars_after_tool_pruning <= max_chars:
        was_pruned = chars_after_tool_pruning < original_chars
        return msgs, was_pruned, original_chars, chars_after_tool_pruning, stats

    # PASS 4: Pinned Head (Option B) + Sliding Tail ("Donut" Window)
    if len(msgs) <= min_tail_messages + 2:
        # Too few messages for windowing to drop anything, but the budget is still blown --
        # almost always one very large message. Truncate content in place instead.
        msgs, final_chars = _truncate_oversize(msgs, max_chars, stats)
        return msgs, True, original_chars, final_chars, stats

    head_end_idx = 0
    for idx, m in enumerate(msgs):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            head_end_idx = idx
            break

    if head_end_idx == 0:
        for idx, m in enumerate(msgs):
            if m.get("role") == "user":
                head_end_idx = idx + 1
                break

    head_end_idx = max(1, min(head_end_idx, 8))

    # HEAD must not split a unit either. The tail loop below already walks backwards off a `tool`
    # or a tool_calls assistant; the head had no equivalent, so a cut that landed just after an
    # assistant message carrying tool_calls kept the call and omitted its results -- a dangling
    # tool_call with no answers, which makes the chat template fail and llama-server answer 500.
    # (Measured 2026-09-25: 298 such 500s in 6 h, appearing only once the derived cap made this pass
    # fire. The marker inserted at the end made it worse by putting a system message right after the
    # dangling call.) Advance past the whole unit instead, and never start the omission on a result.
    while head_end_idx < len(msgs):
        prev_msg = msgs[head_end_idx - 1] if head_end_idx > 0 else None
        this_msg = msgs[head_end_idx]
        if prev_msg and prev_msg.get("role") == "assistant" and prev_msg.get("tool_calls"):
            head_end_idx += 1
            continue
        if this_msg.get("role") == "tool":
            head_end_idx += 1
            continue
        break

    tail_start_idx = len(msgs) - min_tail_messages

    while tail_start_idx > head_end_idx:
        current_msg = msgs[tail_start_idx]
        prev_msg = msgs[tail_start_idx - 1] if tail_start_idx > 0 else None

        if current_msg.get("role") == "tool":
            tail_start_idx -= 1
            continue

        if prev_msg and prev_msg.get("role") == "assistant" and prev_msg.get("tool_calls"):
            tail_start_idx -= 1
            continue

        break

    if tail_start_idx <= head_end_idx:
        # Windowing collapsed to nothing: the head and tail already overlap, so there is no
        # whole message left to drop. Fall through to content truncation, otherwise a prompt
        # with only a few large messages still reaches the lane over budget.
        msgs, final_chars = _truncate_oversize(msgs, max_chars, stats)
        return msgs, True, original_chars, final_chars, stats

    omitted_count = tail_start_idx - head_end_idx
    note = (
        f"[{omitted_count} intermediate tool turns omitted to preserve working context. "
        f"All earlier file operations, tests, and edits completed and reside on disk. "
        f"Complete transcript is preserved losslessly in LCM history. "
        f"Proceed with current task execution based on the latest working state above — "
        f"do not repeat completed searches or re-read unchanged files unless required.]"
    )

    head = msgs[:head_end_idx]
    tail = msgs[tail_start_idx:]

    # Append the note to the LAST KEPT message instead of inserting a synthetic `system` turn.
    # A system message appearing mid-conversation, right after tool results, is precisely what the
    # agent guidance forbids ("never a synthetic message injected mid-loop") and what a strict chat
    # template rejects: measured 13:32, "Pruned: 129085 -> 33452 chars (68 -> 14 msgs)" followed by a
    # 500 in the same second, 33 of them in 10 minutes. Appending to existing content adds no
    # message, so role sequencing cannot be broken by this pass at all. The note is advisory; the
    # omission is the actual optimisation, so it is dropped rather than forced if there is no
    # message to attach it to.
    if head:
        last = dict(head[-1])
        prev = str(last.get("content", "") or "")
        last["content"] = (prev + "\n\n" + note) if prev else note
        head = head[:-1] + [last]

    pruned_msgs = head + tail
    final_chars = sum(len(str(m.get("content", ""))) for m in pruned_msgs)
    stats["donut_omitted"] = omitted_count
    stats["reclaimed_chars"] = original_chars - final_chars

    return pruned_msgs, True, original_chars, final_chars, stats


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    assert client is not None
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    model_requested = str(body.get("model", "auto")).strip()
    messages = body.get("messages", [])
    stream = bool(body.get("stream", False))
    has_tools = bool(body.get("tools"))

    # Deterministic context window pruning: zero LLM inference cost, prevents 32k slot overflow
    pruned_messages, was_pruned, orig_chars, final_chars, stats = prune_messages(
        messages,
        max_chars=_prune_max_chars(),
        keep_recent_tool_turns=3,
        min_tail_messages=10,
    )
    if was_pruned:
        logger.info(
            f"[DCP] Pruned: {orig_chars} -> {final_chars} chars "
            f"({len(messages)} -> {len(pruned_messages)} msgs) | "
            f"Dedupes: {stats['dedupes']}, Errors: {stats['errors_purged']}, "
            f"Evicted: {stats['evicted_tools']}, Donut: {stats['donut_omitted']}"
        )
        messages = pruned_messages
        body["messages"] = pruned_messages

    # Compute context characters
    total_chars = sum(len(str(m.get("content", ""))) for m in messages if isinstance(m, dict))

    target_upstream = UPSTREAM_LLAMA_SWAP
    target_model = model_requested
    tier_tag = "passthrough"
    extra_headers: dict[str, str] = {}

    # Explicit routing
    if model_requested == "local-v320":
        target_upstream = UPSTREAM_LLAMA_SWAP
        target_model = TIER1_V320
        tier_tag = "tier1-v320"
    elif model_requested == "local-7700xt":
        target_upstream = UPSTREAM_LLAMA_SWAP
        target_model = TIER2_7700XT
        tier_tag = "tier2-7700xt"
    elif model_requested == "cloud-keypool" or model_requested.startswith("cloud:"):
        target_upstream = UPSTREAM_KEYPOOL
        target_model = model_requested.removeprefix("cloud:") if model_requested.startswith("cloud:") else ""
        tier_tag = "cloud-keypool"
        if has_tools:
            extra_headers["X-Keypool-Capabilities"] = "agentic"
    elif model_requested in ["auto", "smart", ""]:
        # Extract prompt for classification
        last_prompt = ""
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                last_prompt = str(m.get("content", ""))
                break

        coding_score, cloud_score = await classify_prompt(last_prompt)

        # Lane selection is by CONTEXT SIZE, not task type. Both local lanes serve the SAME
        # ternary model, so the only differences are the card and its ctx: the 7700XT entry is
        # 65536 (and would 400 above that), the V320 entry is 262144. Measured on this box for
        # identical weights and KV type: 7700XT = 515 pp / 35.6 tg, V320 = 77 pp / 15.6 tg.
        # So anything that fits 65k belongs on the 7700XT, even though it was the "coding lane".
        # The coding score is still computed and logged, but no longer picks the lane.
        _chars = sum(len(str(m.get("content") or "")) for m in messages if isinstance(m, dict))
        if body.get("tools"):
            _chars += len(json.dumps(body.get("tools")))
        est_tokens = _chars // 4
        logger.info(
            f"Laya scores: coding={coding_score:.3f}, cloud={cloud_score:.3f} | "
            f"est_tokens={est_tokens} (large-context threshold {LARGE_CONTEXT_CHARS // 4})"
        )

        if cloud_score >= CLOUD_THRESHOLD:
            target_upstream = UPSTREAM_KEYPOOL
            target_model = ""
            tier_tag = "cloud-frontier"
            caps = "agentic" if has_tools else "general_purpose"
            extra_headers["X-Keypool-Capabilities"] = caps
        elif est_tokens > LARGE_CONTEXT_CHARS // 4:
            # LARGE_CONTEXT_CHARS is now actually wired: only the 262k lane has the room, and
            # the 65536 entry would reject this prompt outright.
            target_upstream = UPSTREAM_LLAMA_SWAP
            target_model = TIER1_V320
            tier_tag = "tier1-v320"
        else:
            target_upstream = UPSTREAM_LLAMA_SWAP
            target_model = TIER2_7700XT
            tier_tag = "tier2-7700xt"
    else:
        cloud_prefixes = ["google/", "groq/", "mistral/", "cerebras/", "openrouter/", "cloudflare/"]
        if any(model_requested.lower().startswith(p) for p in cloud_prefixes):
            target_upstream = UPSTREAM_KEYPOOL
            target_model = model_requested
            tier_tag = "cloud-explicit"
            if has_tools:
                extra_headers["X-Keypool-Capabilities"] = "agentic"
        else:
            target_upstream = UPSTREAM_LLAMA_SWAP
            target_model = model_requested
            tier_tag = "llama-swap-explicit"

# Deterministic VRAM Fit & Contention Guard
    # Available VRAM is read live from sysfs inside calculate_fit, so this reflects what the
    # card has FREE right now -- including models a concurrent worker already made resident.
    can_fit, card_or_reason, metrics = calculate_fit(
        target_model, 65536, 2 if "35B" in target_model else 1)
    if not can_fit:
        # Overflow must cascade to the lane with the LARGER, headless budget -- the V320.
        # It must never fall back to the 7700 XT: that card drives both monitors, and a load
        # that leaves it under ~1 GiB free breaks the compositor's atomic page-flips
        # (`Cannot allocate memory`) and crash-loops Hyprland into --safe-mode. The old
        # fallback sent overflow HERE, i.e. a guard that failed open onto the fragile lane.
        logger.warning(f"[VRAM Calculator] REJECTED: {card_or_reason} "
                       f"(measured={metrics.get('measured')} "
                       f"live_free={metrics.get('live_free_mb')}MB) "
                       f"Cascading to V320 headless lane...")
        target_upstream = UPSTREAM_LLAMA_SWAP
        target_model = TIER1_V320
        tier_tag = "tier1-vram-overflow-fallback-v320"

    # --- NEW CONCURRENCY GUARD ---
    # Keyed on the tag substring "tier1", so the overflow tag MUST contain it -- otherwise
    # overflow traffic (which now also targets the V320) would bypass the V320 semaphore.
    if "tier1" in tier_tag and TIER1_SEMAPHORE.locked():
        logger.warning("Tier 1 (V320) parallel slots full. Proactively cascading to Tier 2 (7700XT)...")
        target_upstream = UPSTREAM_LLAMA_SWAP
        target_model = TIER2_7700XT
        tier_tag = "tier2-7700xt-fallback"
    # -----------------------------

# Cascading dispatch helper
    async def try_dispatch(upstream: str, model: str, tag: str):
        body_to_send = dict(body)
        h = {"Content-Type": "application/json"}
        h.update(extra_headers)
        if model:
            body_to_send["model"] = model
        elif "model" in body_to_send and upstream == UPSTREAM_KEYPOOL:
            del body_to_send["model"]

        url = f"{upstream}/v1/chat/completions"
        logger.info(f"Dispatching to {tag} -> {url} (model={model or 'default'})")

        # Local execution function
        async def _execute():
            if stream:
                req = client.build_request("POST", url, json=body_to_send, headers=h)
                return await client.send(req, stream=True)
            else:
                return await client.post(url, json=body_to_send, headers=h)

        # Acquire semaphore only if hitting Tier 1
        if "tier1" in tag:
            async with TIER1_SEMAPHORE:
                res = await _execute()
                return res, tag, model
        else:
            res = await _execute()
            return res, tag, model

    res, active_tag, active_model = await try_dispatch(target_upstream, target_model, tier_tag)

    # Automatic cascade if Tier 1 is busy (concurrency limit reached on V320)
    if res.status_code == 429 and active_tag == "tier1-v320":
        logger.warning("Tier 1 (V320) busy (429 concurrency limit). Cascading to Tier 2 (7700XT)...")
        if stream:
            await res.aclose()
        res, active_tag, active_model = await try_dispatch(UPSTREAM_LLAMA_SWAP, TIER2_7700XT, "tier2-7700xt-fallback")

    # Automatic cascade to Cloud Frontier if local GPU tiers are busy
    if res.status_code == 429 and ("tier2" in active_tag):
        logger.warning(f"{active_tag} busy (429). Cascading to Cloud Frontier...")
        if stream:
            await res.aclose()
        extra_headers["X-Keypool-Capabilities"] = "agentic" if has_tools else "general_purpose"
        res, active_tag, active_model = await try_dispatch(UPSTREAM_KEYPOOL, "", "cloud-frontier-fallback")

    # Automatic cascade if Cloud is rate-limited (429), fall back to local GPU
    if res.status_code == 429 and active_tag.startswith("cloud-"):
        logger.warning(f"{active_tag} rate limited (429). Falling back to local GPU Tier 1/2...")
        if stream:
            await res.aclose()
        res, active_tag, active_model = await try_dispatch(UPSTREAM_LLAMA_SWAP, TIER1_V320, "local-fallback-v320")
        if res.status_code == 429:
            if stream:
                await res.aclose()
            res, active_tag, active_model = await try_dispatch(UPSTREAM_LLAMA_SWAP, TIER2_7700XT, "local-fallback-7700xt")

    resp_headers = {
        "X-Router-Tier": active_tag,
        "X-Router-Model": active_model or "default",
        "X-DCP-Dedupes": str(stats.get("dedupes", 0)),
        "X-DCP-Errors-Purged": str(stats.get("errors_purged", 0)),
        "X-DCP-Reclaimed-Chars": str(stats.get("reclaimed_chars", 0)),
    }

    if stream:
        if res.status_code != 200:
            content = await res.aread()
            return Response(content=content, status_code=res.status_code, media_type="application/json", headers=resp_headers)

        async def stream_generator():
            async for chunk in res.aiter_bytes():
                yield chunk

        resp_headers.update({
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
        })
        return StreamingResponse(stream_generator(), headers=resp_headers)
    else:
        return Response(content=res.content, status_code=res.status_code, media_type="application/json", headers=resp_headers)


if __name__ == "__main__":
    uvicorn.run("router:app", host=HOST, port=PORT, log_level="info")
