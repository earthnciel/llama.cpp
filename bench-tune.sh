#!/bin/bash
# bench-tune.sh — thin wrapper around llama-bench for parameter sweeps on B580.
#
# Usage:
#   ./bench-tune.sh <vulkan|sycl> [llama-bench args...]
#
# Examples:
#   # Baseline single point (current default config, current model)
#   ./bench-tune.sh sycl -hf unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_XL \
#       -ngl 999 -ncmoe 22 -fa on -ctk q4_0 -ctv q4_0 \
#       -ub 512 -t 10 -p 2048 -n 256 -r 2
#
#   # -ub sweep (matrix expanded automatically)
#   ./bench-tune.sh sycl -hf unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_XL \
#       -ngl 999 -ncmoe 22 -fa on -ctk q4_0 -ctv q4_0 \
#       -ub 512,1024,2048 -t 9,10 -p 2048 -n 256 -r 2
#
#   # Agent workload (gen at depth 8K/32K = long-context speed)
#   ./bench-tune.sh sycl -hf barozp/Qwen3.6-28B-REAP20-A3B-GGUF:Q4_K_M \
#       -ngl 999 -ncmoe 18 -fa on -ctk q4_0 -ctv q4_0 \
#       -ub 1024 -t 9 -n 128 -d 0,8192,32768 -r 2
#
# Output:
#   - stdout: live markdown table (you see results as they finish)
#   - bench-results/<timestamp>-<backend>.md : same table, archived
#   - bench-results/<timestamp>-<backend>.csv : machine-readable copy
#   - bench-results/<timestamp>-<backend>.cmd : exact command for reproducibility

BACKEND="${1:-}"
shift || true

case "$BACKEND" in
  vulkan)
    BIN=./build/bin/llama-bench
    ;;
  sycl)
    BIN=./build-sycl/bin/llama-bench
    # Intel oneAPI runtime — required (else libsvml.so load error).
    # setvars.sh inspects $1: must clear positional args before sourcing or it
    # interprets our forwarded args (e.g. --help) as its own and exits early.
    __SAVED_ARGS=("$@"); set --
    source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
    set -- "${__SAVED_ARGS[@]}"
    export ZES_ENABLE_SYSMAN=1
    export ONEAPI_DEVICE_SELECTOR=level_zero:0
    export SYCL_CACHE_PERSISTENT=1
    # Uncomment if output corruption appears on Battlemage (#21893):
    # export GGML_SYCL_DISABLE_OPT=1
    ;;
  *)
    echo "Usage: $0 <vulkan|sycl> [llama-bench args...]" >&2
    echo "Run '$0 sycl --help' for full llama-bench option list." >&2
    exit 1
    ;;
esac

if [ ! -x "$BIN" ]; then
  echo "ERROR: $BIN not found or not executable. Build it first." >&2
  exit 1
fi

mkdir -p bench-results
TS=$(date +%Y%m%d-%H%M%S)
BASE="bench-results/${TS}-${BACKEND}"
GIT_REV=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Record exact invocation for reproducibility
{
  echo "# bench-tune $BACKEND  ($TS, git $GIT_REV)"
  echo "$BIN --progress -o md $*"
} > "${BASE}.cmd"

# Run twice: one md (stdout+archive), one csv (machine-readable archive).
# Use --progress so the user sees live status.
"$BIN" --progress -o md  "$@" | tee "${BASE}.md"
echo
echo "[bench-tune] re-running silently for CSV archive..." >&2
"$BIN" -o csv "$@" > "${BASE}.csv" 2>/dev/null

echo
echo "[bench-tune] saved: ${BASE}.{md,csv,cmd}"
