#!/bin/bash
# SYCL (Intel oneAPI) launcher — uses build-sycl/ (Intel Arc B580, bmg_g21 AOT).
# Generation is ~2.3x faster than the Vulkan build for this MoE model.
# Vulkan equivalent: ./start_llm.sh  (uses ./build/bin/)

set -e

# --- Intel oneAPI runtime (required, else: libsvml.so load error) ---
ONEAPI_SETVARS=/opt/intel/oneapi/setvars.sh
if [ ! -f "$ONEAPI_SETVARS" ]; then
  echo "ERROR: $ONEAPI_SETVARS not found. Install Intel oneAPI (intel-oneapi-compiler-dpcpp-cpp)." >&2
  exit 1
fi
source "$ONEAPI_SETVARS" >/dev/null 2>&1

# --- SYCL runtime env for the Arc B580 ---
export ZES_ENABLE_SYSMAN=1            # let SYCL query free GPU memory
export ONEAPI_DEVICE_SELECTOR=level_zero:0   # pin to the discrete Arc via Level-Zero
export SYCL_CACHE_PERSISTENT=1        # persist JIT kernel cache across runs
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1  # allow >4 GiB single allocs (large f16 KV)
# Optional: enable USM system allocations (B580=Xe2 supports it; requires
# kernel CONFIG_DRM_XE_GPUSVM). Allows VRAM overcommit by migrating buffers
# to/from system RAM as needed. Test cautiously — could slow tg if migration
# thrashes. Verify SVM available via server log "USM_SYSTEM: 1" line.
# export GGML_SYCL_USM_SYSTEM=1
# If output ever becomes garbled (Battlemage bmg_g21+F16 bug #21893), uncomment:
# export GGML_SYCL_DISABLE_OPT=1

#  --mlock \
#  -hf unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL \
#  --alias "gemma-4-26b-a4b" \
#
#  --- MTP tuning slot (defaults: n-max 16 / n-min 0) ---
#  --spec-draft-n-max 16 \
#  --spec-draft-n-min 0 \

# Active: Nex-N2-mini (text-only main coder, reasoning OFF).
# Bench (SYCL b9716, ctx 64K verified, clean VRAM state):
#   d=0:   pp 231 / tg 31.95
#   d=64K: pp 138 / tg 24.78  (64K guaranteed)
# Config: ncmoe 29 + f16 KV + mmap (Q5_K_S 24.2GB > 24GB usable RAM).
# qwen35moe architecture. Reasoning + multimodal both off.

./build-sycl/bin/llama-server \
  -hf bartowski/nex-agi_Nex-N2-mini-GGUF:Q5_K_S \
  --alias "nex-n2-mini" \
  --host 0.0.0.0 --port 11434 \
  -ngl 999 \
  -fit off \
  --n-cpu-moe 29 \
  --no-mmproj \
  -np 1 \
  -fa on \
  --cache-type-k f16 --cache-type-v f16 \
  -t 9 -b 2048 -ub 2048 \
  --ctx-size 65536 \
  --cache-ram 4096 \
  --jinja \
  --cache-reuse 256 \
  --reasoning off --reasoning-budget 0 \
  --temp 0.2 --top-k 40 --top-p 0.95 --min-p 0.00 \
  --repeat-penalty 1.0 --presence-penalty 0.0
  
