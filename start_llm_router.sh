#!/bin/bash
# SYCL router-mode launcher — single llama-server instance, multiple model
# presets defined in models.ini. Switch models at request time via the
# OpenAI-compatible "model" field; the router auto-loads/unloads.
#
# Usage:
#   ./start_llm_router.sh
# Then POST to http://<host>:11434/v1/chat/completions with
#   {"model": "barozp/Qwen3.6-28B-REAP20-A3B-GGUF:Q4_K_M", "messages": [...]}
# (or whichever section in models.ini you want).
#
# Vulkan equivalent: not provided — Vulkan/SYCL can't share a router (different
# bin); SYCL is the production backend on this host (~1.8x faster gen).

# --- Intel oneAPI runtime (required, else: libsvml.so load error) ---
ONEAPI_SETVARS=/opt/intel/oneapi/setvars.sh
if [ ! -f "$ONEAPI_SETVARS" ]; then
  echo "ERROR: $ONEAPI_SETVARS not found. Install Intel oneAPI." >&2
  exit 1
fi
# setvars.sh inspects $1: clear positional args before sourcing.
__SAVED=("$@"); set --
source "$ONEAPI_SETVARS" >/dev/null 2>&1
set -- "${__SAVED[@]}"

# --- SYCL runtime env for the Arc B580 ---
export ZES_ENABLE_SYSMAN=1
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export SYCL_CACHE_PERSISTENT=1
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1  # allow >4 GiB single allocs
# Optional VRAM overcommit (B580=Xe2 + kernel CONFIG_DRM_XE_GPUSVM):
# export GGML_SYCL_USM_SYSTEM=1
# If output ever becomes garbled (Battlemage bmg_g21+F16 bug #21893), uncomment:
# export GGML_SYCL_DISABLE_OPT=1

# --- Router mode: NO -hf / -m on the command line. Model selection happens
# per-request; tuning per model comes from models.ini.
# --models-max 1 → strict single-model resident. Switching between models
# triggers unload+load. REAP20 reload ~30-60s; embedder/reranker ~5-10s.
# Trade-off: RAG (chat → embed → rerank → chat) pays ~3 unload/load cycles
# per query. Acceptable for occasional RAG; not for high-throughput RAG.
./build-sycl/bin/llama-server \
  --host 0.0.0.0 --port 11434 \
  --models-preset ./models.ini \
  --models-max 1
