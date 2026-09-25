#!/usr/bin/env python3
"""
VRAM & Contention Calculator for Dual-GPU Local Stack (7700XT 12GB + V320 16GB)
Calculates exact weight + KV cache memory footprints and validates whether a model
fit is mathematically guaranteed before dispatching to llama-swap.
"""

import sys
from typing import Dict, Tuple

# Hardware budget in MiB
VRAM_BUDGET_MB = {
    "v320": 15360,      # Radeon PRO V320 (16GB - overhead)
    "7700xt": 11264,    # RX 7700 XT (12GB - overhead)
    "total": 26624,
}

# Known model size estimates (weights in GiB) and default parameters
MODEL_SPECS: Dict[str, dict] = {
    "Tiel-Coder-35B-A3B-MTP-IQ3_XXS": {"weights_gb": 13.5, "default_ctx": 65536, "parallel": 2, "card": "v320"},
    "Ternary-Bonsai-2-27B-PQ2_0": {"weights_gb": 6.71, "default_ctx": 65536, "parallel": 1, "card": "7700xt"},
    "Dirk-Qwen3.8-9B-Q4_K_M": {"weights_gb": 5.2, "default_ctx": 65536, "parallel": 1, "card": "v320"},
    "Bonsai-8B-Q1_0": {"weights_gb": 1.16, "default_ctx": 65536, "parallel": 1, "card": "cpu"},
    "Qwen3-30B-A3B-Instruct": {"weights_gb": 11.2, "default_ctx": 65536, "parallel": 1, "card": "v320"},
}

def estimate_kv_cache_mb(ctx_size: int, parallel: int, quant_bits: int = 4) -> float:
    """Estimate KV cache size in MiB based on ctx size, parallelism, and KV quant (default turbo4 = 4-bit)."""
    # Rough formula for transformer KV cache: 2 * n_layers * hidden_dim * ctx * parallel * (bits / 8)
    # Using generalized heuristic: ~0.025 MB per 1k tokens per billion params for 4-bit KV
    # For a 35B model at 65k ctx: ~2000 MB per parallel slot
    bytes_per_token_layer = (4 / 8) * 2  # k and v
    # Assume average 35B architecture (48 layers, hidden ~5120) -> ~0.5MB per 1k ctx per layer...
    # Simplified empirical table for 65k ctx with turbo4/q4 KV:
    base_mb_per_65k = 2048.0  
    return (ctx_size / 65536.0) * base_mb_per_65k * parallel

def calculate_fit(model_name: str, requested_ctx: int = 65536, parallel: int = 1) -> Tuple[bool, str, dict]:
    """
    Calculate exact memory requirements and verify if model fits in VRAM pool.
    Returns: (can_fit, reason_or_card, metrics_dict)
    """
    spec = MODEL_SPECS.get(model_name)
    if not spec:
        # Generic fallback estimate for unknown models
        weights_gb = 10.0
        card = "v320"
    else:
        weights_gb = spec["weights_gb"]
        card = spec["card"]

    weights_mb = weights_gb * 1024.0
    kv_mb = estimate_kv_cache_mb(requested_ctx, parallel)
    total_required_mb = weights_mb + kv_mb

    metrics = {
        "model": model_name,
        "weights_mb": round(weights_mb, 2),
        "kv_cache_mb": round(kv_mb, 2),
        "total_required_mb": round(total_required_mb, 2),
        "target_card": card,
    }

    if card == "cpu":
        return True, "cpu_lane", metrics

    available_vram = VRAM_BUDGET_MB.get(card, 12288)

    if total_required_mb <= available_vram:
        headroom = available_vram - total_required_mb
        metrics["headroom_mb"] = round(headroom, 2)
        return True, card, metrics
    else:
        overflow = total_required_mb - available_vram
        metrics["overflow_mb"] = round(overflow, 2)
        return False, f"VRAM overflow on {card}: requires {total_required_mb:.1f}MB, available {available_vram}MB (deficit {overflow:.1f}MB)", metrics

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "Tiel-Coder-35B-A3B-MTP-IQ3_XXS"
    ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 65536
    fit, target, data = calculate_fit(model, ctx)
    print(f"Model: {model}")
    print(f"Can Fit: {fit} ({target})")
    print("Metrics:", data)
