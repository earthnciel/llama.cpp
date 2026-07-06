#!/bin/bash

#  --mlock \
#  -hf unsloth/Qwen3.6-35B-A3B-MTP-GGUF:Q4_K_XL \
#  --alias "qwen3.6-35b-a3b" \
#  -hf unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_XL \
#  --alias "qwen3.6-35b-a3b" \
#  -hf unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL \
#  --alias "gemma-4-26b-a4b" \
#  --n-cpu-moe 30 \
#  --ctx-size 65536 \
#  --cache-type-k q8_0 --cache-type-v q8_0 \
#
#  --- MTP tuning slot (move into the llama-server args above to use; defaults: n-max 16 / n-min 0) ---
#  --spec-draft-n-max 16 \
#  --spec-draft-n-min 0 \

./build/bin/llama-server \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_XL \
  --alias "qwen3.6-35b-a3b" \
  --host 0.0.0.0 --port 11434 \
  -ngl 999 \
  -fit off \
  --n-cpu-moe 22 \
  --no-mmproj \
  -np 1 \
  --no-mmap \
  -fa on \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  -t 10 -b 2048 -ub 512 \
  --ctx-size 65536 \
  --cache-ram 4096 \
  --jinja \
  --reasoning off \
  --temp 0.2 --top-k 40 --top-p 0.95 --min-p 0.00 \
  --presence-penalty 0.0 \
  --repeat-penalty 1.05

