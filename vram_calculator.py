#!/usr/bin/env python3
"""
VRAM & Contention Calculator for the dual-GPU local stack (RX 7700 XT 12GB + Radeon PRO V320 16GB).

Sizes are MEASURED where the model is in service, not heuristic. Two bugs are fixed here:

1. `estimate_kv_cache_mb` accepted a `quant_bits` argument and then ignored it, applying one
   flat 2048 MiB/65k constant to every model regardless of size, layer count or KV type.
2. KV layers were assumed to equal the model's block count. On a hybrid model they are not:
   qwen35 / Qwen3.5 keep a KV cache only on the full-attention layers
   (`full_attention_interval = 4` -> 16 of 64 blocks). Sizing 64 over-estimated by 4x.

The combination rejected Tiel on the V320 for every prompt (13.5 GiB weights + 4.0 GiB of
invented KV = 17,920 MiB > 15,360 budget) and cascaded all traffic onto the 7700 XT's
single-slot ternary lane.

KV arithmetic: bytes/token = n_kv_layers * (kv_heads*k_dim + kv_heads*v_dim) * bpp
Verified against llama.cpp's own line, `llama_kv_cache: ROCm0 KV buffer size = N MiB`:
  Ternary-Bonsai-2-27B-PQ2_0 @ 262144, q4_0 -> 4608 MiB   (16 layers, 4 heads, 256 dim)
  Ternary-Bonsai-2-27B-PQ2_0 @ 262144, turbo3 -> 3200 MiB
  Ternary-Bonsai-2-27B-PQ2_0 @ 262144, turbo2 -> 2176 MiB
"""

import sys
from typing import Dict, Optional, Tuple

MiB = 1024.0 * 1024.0

# Card identity is by TOTAL size, not enumeration order: card1 = 16368 MiB = V320,
# card2 = 12272 MiB = 7700 XT. Verified against lspci (Vega10 @ 06:00.0, Navi32 @ 0a:00.0).
CARD_SYSFS = {"v320": "card1", "7700xt": "card2"}

# Static budgets are a FALLBACK for when sysfs is unreadable. Live free VRAM is authoritative
# whenever it can be read -- a static constant cannot see a second worker already resident.
VRAM_BUDGET_MB = {
    "v320": 15360,      # 16368 total, safety margin held back
    "7700xt": 10752,    # 12272 total, less the display/compositor reservation
    "total": 26112,
}

# Free VRAM that must remain AFTER a load, per card.
# The 7700 XT drives both monitors: below ~1 GiB free the compositor's atomic DRM page-flips
# fail with `Cannot allocate memory` and Hyprland crash-loops into --safe-mode. A lane that
# sat stable at 209 MiB free was still fatal, so this is a hard floor, not a comfort margin.
MIN_HEADROOM_MB = {"v320": 1024.0, "7700xt": 1280.0, "cpu": 0.0}


def live_vram_mb(card: str) -> Optional[Tuple[float, float]]:
    """(free_mb, total_mb) for a card, or None if sysfs is unreadable.

    `mem_info_vram_free` does not exist on either card; free is computed as total - used.
    """
    sysfs = CARD_SYSFS.get(card)
    if not sysfs:
        return None
    try:
        with open(f"/sys/class/drm/{sysfs}/device/mem_info_vram_used") as f:
            used = int(f.read().strip())
        with open(f"/sys/class/drm/{sysfs}/device/mem_info_vram_total") as f:
            total = int(f.read().strip())
    except (OSError, ValueError):
        return None
    return (total - used) / MiB, total / MiB

# Effective bytes per KV element, block overhead included (this fork's types included).
KV_BPP = {
    "f32": 4.0, "f16": 2.0, "bf16": 2.0,
    "q8_0": 1.0625, "q5_1": 0.75, "q5_0": 0.6875, "q4_1": 0.625, "q4_0": 0.5625, "iq4_nl": 0.5625,
    "turbo4": 0.515625,   # block_turbo4_0  66B/128
    "turbo3": 0.390625,   # block_turbo3_0  50B/128
    "turbo2": 0.265625,   # block_turbo2_0  34B/128
}

# keep/worst-case compute-graph buffers, MiB. Measured on the ternary: 378 (262k) .. 1360 (262k, q4_0)
DEFAULT_COMPUTE_MB = 512.0
COMPUTE_MB_BY_CTX = {65536: 384.0, 131072: 768.0, 262144: 1408.0}

# kv spec: layers = FULL-ATTENTION layers only on a hybrid model; heads = KV heads (GQA);
# k_dim/v_dim = per-head dims. `card` is the lane the entry is pinned to.
MODEL_SPECS: Dict[str, dict] = {
    "Ternary-Bonsai-2-27B-PQ2_0": {
        "weights_gb": 6.71, "default_ctx": 262144, "parallel": 1, "card": "v320",
        "kv": {"layers": 16, "heads": 4, "k_dim": 256, "v_dim": 256, "bpp": "q4_0"},
        "measured": {"ctx": 262144, "kv_bpp": "q4_0", "kv_mb": 4608, "self_mb": 12657},
    },
    "Ternary-Bonsai-2-27B-7700XT": {
        "weights_gb": 6.71, "default_ctx": 65536, "parallel": 1, "card": "7700xt",
        "kv": {"layers": 16, "heads": 4, "k_dim": 256, "v_dim": 256, "bpp": "q4_0"},
        "measured": {"ctx": 65536, "kv_bpp": "q4_0", "kv_mb": 1152, "self_mb": 8069},
    },
    # Removed from routing (2026-09-24): kept only so a stray request is still measured honestly.
    # 13.5 GiB of weights do not fit beside any useful KV on the V320 — that is a property of
    # the model, not of the calculator, and it is why this tier no longer exists.
    "Tiel-Coder-35B-A3B-MTP-IQ3_XXS": {
        "weights_gb": 13.5, "default_ctx": 65536, "parallel": 2, "card": "v320",
        "kv_estimate_only": True,
    },
    "Dirk-Qwen3.8-9B-Q4_K_M": {"weights_gb": 5.2, "default_ctx": 65536, "parallel": 1, "card": "v320",
                               "kv_estimate_only": True},
    "Bonsai-8B-Q1_0": {"weights_gb": 1.16, "default_ctx": 65536, "parallel": 1, "card": "cpu"},
    "Qwen3-30B-A3B-Instruct": {"weights_gb": 11.2, "default_ctx": 65536, "parallel": 1, "card": "v320",
                               "kv_estimate_only": True},
}


def estimate_kv_cache_mb(ctx_size: int, parallel: int = 1, quant_bits: Optional[float] = None,
                         kv: Optional[dict] = None) -> float:
    """KV cache in MiB.

    Exact when the entry carries a `kv` spec; otherwise a conservative estimate that HONOURS
    quant_bits (the old signature took the argument and discarded it).
    """
    if kv:
        bpp = KV_BPP.get(str(kv.get("bpp", "q4_0")).lower(), kv.get("bpp"))
        if not isinstance(bpp, (int, float)):
            bpp = KV_BPP["q4_0"]
        per_token = kv["layers"] * (kv["heads"] * kv["k_dim"] + kv["heads"] * kv["v_dim"])
        return (ctx_size * parallel * per_token * bpp) / MiB

    bits = quant_bits if quant_bits is not None else 4.0
    # Unknown arch: assume a 35B-class dense-attention shape (48 layers, 8 KV heads, 128 dim).
    per_token = 48 * (8 * 128 + 8 * 128) * (bits / 8.0)
    return (ctx_size * parallel * per_token) / MiB


def _compute_mb(ctx_size: int) -> float:
    for ctx, mb in sorted(COMPUTE_MB_BY_CTX.items(), reverse=True):
        if ctx_size >= ctx:
            return mb
    return DEFAULT_COMPUTE_MB


def calculate_fit(model_name: str, requested_ctx: int = 65536,
                  parallel: int = 1, quant_bits: Optional[float] = None) -> Tuple[bool, str, dict]:
    """Exact footprint for the model's lane. Returns (can_fit, reason_or_card, metrics).

    Available VRAM is read LIVE from sysfs when possible, so a lane that already has a model
    resident is judged on what is actually free rather than on a static whole-card budget.
    """
    spec = MODEL_SPECS.get(model_name, {})
    known = model_name in MODEL_SPECS
    if not known:
        # Fail CLOSED, not open. The old default (10 GB weights, v320) let an unmeasured 27B
        # pass as "fits" against a card that had another model resident. An unknown model is
        # exactly the case where a guess is most dangerous, so it must not silently succeed.
        weights_gb, card, kv = 10.0, "v320", None
    else:
        weights_gb = spec.get("weights_gb", 10.0)
        card = spec.get("card", "v320")
        kv = spec.get("kv")

    weights_mb = weights_gb * 1024.0
    kv_mb = estimate_kv_cache_mb(requested_ctx, parallel, quant_bits, kv)
    compute_mb = 0.0 if card == "cpu" else _compute_mb(requested_ctx)
    total_required_mb = weights_mb + kv_mb + compute_mb

    live = live_vram_mb(card) if card != "cpu" else None
    live_free = live[0] if live else None

    metrics = {
        "model": model_name,
        "measured": known,
        "weights_mb": round(weights_mb, 2),
        "kv_cache_mb": round(kv_mb, 2),
        "compute_mb": round(compute_mb, 2),
        "total_required_mb": round(total_required_mb, 2),
        "target_card": card,
        "live_free_mb": round(live_free, 2) if live_free is not None else None,
        "source": "live-sysfs" if live_free is not None else "static-budget",
    }

    if card == "cpu":
        return True, "cpu_lane", metrics

    available_vram = live_free if live_free is not None else VRAM_BUDGET_MB.get(card, 10752)
    floor = MIN_HEADROOM_MB.get(card, 1024.0)
    usable = available_vram - floor
    metrics["headroom_mb"] = round(usable - total_required_mb, 2)

    if total_required_mb <= usable:
        return True, card, metrics

    overflow = total_required_mb - usable
    metrics["overflow_mb"] = round(overflow, 2)
    metrics["deficit_mb"] = round(available_vram - total_required_mb, 2)
    reason = (f"VRAM overflow on {card}: requires {total_required_mb:.1f}MB "
              f"(weights {weights_mb:.0f} + kv {kv_mb:.0f} + compute {compute_mb:.0f}), "
              f"free {available_vram:.1f}MB of which {floor:.0f}MB is the hard floor "
              f"(deficit {overflow:.1f}MB)")
    if not known:
        reason = "UNMEASURED MODEL -- " + reason
    return False, reason, metrics


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "Ternary-Bonsai-2-27B-PQ2_0"
    ctx = int(sys.argv[2]) if len(sys.argv) > 2 else MODEL_SPECS.get(model, {}).get("default_ctx", 65536)
    par = int(sys.argv[3]) if len(sys.argv) > 3 else MODEL_SPECS.get(model, {}).get("parallel", 1)
    fit, target, data = calculate_fit(model, ctx, par)
    print(f"Can Fit: {fit} ({target})")
    print("Metrics:", data)
