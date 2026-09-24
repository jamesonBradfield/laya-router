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

TIER1_V320 = config.get("defaults", {}).get("tier1_v320_model", "Tiel-Coder-35B-A3B-MTP-IQ3_XXS")
TIER2_7700XT = config.get("defaults", {}).get("tier2_7700xt_model", "Ternary-Bonsai-2-27B-7700XT")
LARGE_CONTEXT_CHARS = config.get("defaults", {}).get("large_context_chars", 24000)

CODING_THRESHOLD = config.get("thresholds", {}).get("complex_coding_noul", 0.50)
CLOUD_THRESHOLD = config.get("thresholds", {}).get("frontier_cloud_noul", 0.65)

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
        r = await client.post(f"{UPSTREAM_LAYA}/v1/systemone", json=payload, timeout=2.0)
        if r.status_code == 200:
            data = r.json()
            answers = data.get("answers", {})
            coding_score = float(answers.get("coding", {}).get("noul", 0.0))
            cloud_score = float(answers.get("cloud", {}).get("noul", 0.0))
            return coding_score, cloud_score
    except Exception as e:
        logger.warning(f"Laya classification failed: {e}")
    return 0.0, 0.0


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

        # Check large context first
        if total_chars > LARGE_CONTEXT_CHARS:
            logger.info(f"Routing to Keypool (large context: {total_chars} chars)")
            target_upstream = UPSTREAM_KEYPOOL
            target_model = ""  # Let keypool select default large_context provider (Google)
            tier_tag = "cloud-large-context"
            extra_headers["X-Keypool-Capabilities"] = "large_context"
        else:
            coding_score, cloud_score = await classify_prompt(last_prompt)
            logger.info(f"Laya scores: coding={coding_score:.3f}, cloud={cloud_score:.3f}")

            if cloud_score >= CLOUD_THRESHOLD:
                target_upstream = UPSTREAM_KEYPOOL
                target_model = ""
                tier_tag = "cloud-frontier"
                caps = "agentic" if has_tools else "general_purpose"
                extra_headers["X-Keypool-Capabilities"] = caps
            elif coding_score >= CODING_THRESHOLD:
                target_upstream = UPSTREAM_LLAMA_SWAP
                target_model = TIER2_7700XT
                tier_tag = "tier2-7700xt"
            else:
                target_upstream = UPSTREAM_LLAMA_SWAP
                target_model = TIER1_V320
                tier_tag = "tier1-v320"
    else:
        # Check if requested model belongs to cloud providers
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

    # Mutate model in body if resolved
    forward_body = dict(body)
    if target_model:
        forward_body["model"] = target_model
    elif "model" in forward_body and target_upstream == UPSTREAM_KEYPOOL:
        # Keypool can pick default based on capabilities if model is omitted
        del forward_body["model"]

    forward_url = f"{target_upstream}/v1/chat/completions"
    logger.info(f"Dispatching to {tier_tag} -> {forward_url} (model={target_model or 'default'})")

    headers = {"Content-Type": "application/json"}
    headers.update(extra_headers)

    if stream:
        req = client.build_request("POST", forward_url, json=forward_body, headers=headers)
        r = await client.send(req, stream=True)

        if r.status_code != 200:
            content = await r.aread()
            return Response(content=content, status_code=r.status_code, media_type="application/json")

        async def stream_generator():
            async for chunk in r.aiter_bytes():
                yield chunk

        resp_headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Router-Tier": tier_tag,
            "X-Router-Model": target_model or "default",
        }
        return StreamingResponse(stream_generator(), headers=resp_headers)
    else:
        r = await client.post(forward_url, json=forward_body, headers=headers)
        resp_headers = {
            "X-Router-Tier": tier_tag,
            "X-Router-Model": target_model or "default",
        }
        return Response(content=r.content, status_code=r.status_code, media_type="application/json", headers=resp_headers)


if __name__ == "__main__":
    uvicorn.run("router:app", host=HOST, port=PORT, log_level="info")
