# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

IMPORTANT: Ensure you've thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work. This project does **not** accept pull requests that are fully or predominantly AI-generated; AI may be used only assistively, AI use must be disclosed, and AI-written PR descriptions / commit messages / reviewer responses are prohibited. Treat AGENTS.md and [CONTRIBUTING.md](CONTRIBUTING.md) as binding.

## Fork governance — READ FIRST (binding for all agents & contributors)

This repository is a **private-purpose personal fork** of upstream
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), maintained for local
operational use only. The rules below are **BINDING** and override any default
willingness to contribute to, communicate with, or publish through the upstream
project. When any other instruction conflicts with these rules, these rules win.

1. **No upstream-affecting actions.** Never perform any action that reaches or
   affects upstream (`ggml-org/llama.cpp`) or its community — no pull / merge
   requests, no opening / editing / commenting on issues or discussions, no
   reviews or reactions on upstream PRs / issues / commits, no pushing branches
   or tags to any upstream-owned remote, and no other outbound contribution or
   contact. Upstream interaction is limited to **read-only sync**:
   `git fetch upstream` and merging its changes into this fork.

2. **No license violations.** llama.cpp is **MIT-licensed** (see [LICENSE](LICENSE)).
   Never remove, alter, or obscure the LICENSE file or any copyright / license
   headers; always preserve attribution; keep this fork MIT-compatible.

3. **No sensitive data — this is a PUBLIC repo.** Never commit or push anything
   that could aid an attacker or expose private information: direct-contact
   details (phone / mobile numbers, physical addresses), IP addresses /
   hostnames of private machines or networks, API keys, authentication keys,
   access tokens, passwords, private keys, certificates, or any other secret.
   (Email is treated as acceptable — it is only an indirect contact and the
   owner has chosen to expose theirs.) Scan every added file before each commit
   / push. When in doubt, do not commit it. This rule is enforced by a
   pre-commit secret scanner in [`.githooks/pre-commit`](.githooks/pre-commit);
   on a fresh clone, activate it once with
   `git config core.hooksPath .githooks`.

4. **Nothing otherwise unlawful.** Do not perform any action in or through this
   fork that is illegal or legally problematic (e.g. redistributing others'
   proprietary code without a license, circumventing access controls, or
   publishing others' personal data).

## Build

**Backend scope (this fork's default):** Unless a request explicitly says
otherwise, **SYCL is the only managed, built, and tested backend** — build in
`build-sycl/` (Intel Arc B580, Intel oneAPI; ~2.3× faster generation than Vulkan
for the MoE workloads used here) and run tests against it. Vulkan (`build/`) is
kept only as an occasional fallback: do **not** build, test, or update it unless
the user explicitly asks for Vulkan. Other backends (CUDA / Metal / HIP / …) are
out of scope in this fork.

CMake is the primary build system (the top-level `Makefile` is deprecated).

```bash
cmake -B build                              # configure (CPU)
cmake --build build --config Release -j 8   # build all targets in parallel
```

- Debug build (single-config generators): `cmake -B build -DCMAKE_BUILD_TYPE=Debug`
- This checkout's `build/` is already configured with the CPU and Vulkan backends; binaries land in `build/bin/` (e.g. `llama-cli`, `llama-server`, `llama-quantize`).
- Backend selection is via CMake flags, e.g. `-DGGML_CUDA=ON`, `-DGGML_METAL=ON`, `-DGGML_VULKAN=ON`. See [docs/build.md](docs/build.md) for every backend.
- Static lib: add `-DBUILD_SHARED_LIBS=OFF`.

## Test

Tests are CTest targets, registered only when configured with `-DLLAMA_BUILD_TESTS=ON`.

```bash
ctest --test-dir build --output-on-failure          # full suite
ctest --test-dir build -R test-tokenizer            # run tests matching a regex (single test)
./build/bin/test-backend-ops                        # run a test binary directly
```

- `test-backend-ops` is mandatory when touching `ggml`: it cross-checks operator results across backends. Run it if you modified any `ggml` operator, and add a test case there if you added/changed one.
- For correctness/perf regressions use `tools/perplexity` (`llama-perplexity`) and `tools/llama-bench` (`llama-bench`) — required evidence for quantization/perf changes.

## Full local CI

Heavy workflows run on self-hosted runners; reproduce locally before publishing:

```bash
mkdir -p tmp
bash ./ci/run.sh ./tmp/results ./tmp/mnt          # CPU-only
GG_BUILD_CUDA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt   # with a backend
```

## Architecture

The codebase is a layered stack; understanding the layer boundaries is the key to navigating it:

- **`ggml/`** — the tensor library and compute engine, developed in-tree (mirrored from the separate `ggml-org/ggml` repo). `ggml/src/ggml.c` is the core; each backend is an isolated subdir implementing the `ggml-backend` interface: `ggml-cpu`, `ggml-cuda`, `ggml-metal`, `ggml-vulkan`, `ggml-sycl`, `ggml-hip`, `ggml-blas`, etc. Backends are pluggable and discovered via `ggml-backend-reg.cpp`. Everything above this layer is backend-agnostic.

- **`src/` (libllama)** — the model runtime built on ggml. Split into focused units: `llama-model.cpp` / `llama-arch.cpp` (architecture definitions and tensor wiring), `llama-graph.cpp` (compute graph construction), `llama-context.cpp` (inference state), `llama-kv-cache*.cpp` / `llama-memory*.cpp` (KV cache and recurrent/hybrid memory), `llama-vocab.cpp` + `unicode.cpp` (tokenization), `llama-sampler.cpp`, `llama-grammar.cpp`, `llama-quant.cpp` (quantization), `llama-model-loader.cpp` (GGUF loading). Per-architecture graph code lives in `src/models/*.cpp` (one file per model family). The public C API is **`include/llama.h`** — the project's main product.

- **`common/`** — shared helpers for the tools/examples (not part of libllama): argument parsing (`arg.cpp`), chat templating and output parsing (`chat.cpp`, the PEG parser `peg-parser.cpp` / `chat-peg-parser.cpp`, the auto parser, and the Jinja engine in `common/jinja/`), sampling glue, speculative decoding, GGUF/HF download.

- **`tools/`** — production binaries: `server` (OpenAI-compatible HTTP server, the largest sub-project), `cli`, `quantize`, `perplexity`, `llama-bench`, `imatrix`, `mtmd` (multimodal), `rpc`. **`examples/`** — minimal, didactic programs.

- **`gguf-py/`** + `convert_hf_to_gguf.py` — Python side: converts HuggingFace/other checkpoints into the `.gguf` format that libllama loads. Adding a model usually means matching changes here and in `src/llama-arch.cpp` + `src/models/`. See [docs/development/HOWTO-add-model.md](docs/development/HOWTO-add-model.md).

Data flow: a checkpoint is converted to GGUF (Python) → loaded by `llama-model-loader` → an architecture-specific compute graph is built in `src/models/` → executed by a `ggml` backend → driven by a tool in `tools/` through `include/llama.h` + `common/`.

## Conventions (from CONTRIBUTING.md)

- No third-party dependencies / extra headers. Keep C++ simple: basic `for` loops, avoid templates and fancy STL, cross-platform.
- `snake_case` for functions/variables/types; names optimize for longest common prefix (`number_small`, not `small_number`). Enum values UPPER_CASE prefixed with the enum name.
- 4-space indent, brackets on the same line, `void * ptr`, `int & a`, vertical alignment. Use `clang-format` (clang-tools v15+) when in doubt. Sized integer types (`int32_t`) in the public API.
- ggml tensors are row-major; dim 0 = columns, 1 = rows, 2 = matrices. Matmul is transposed: `C = ggml_mul_mat(ctx, A, B)` computes `C = B Aᵀ`.
- Scope PRs to one feature/fix. New model/feature PRs should be **CPU-only first**; other backends follow up. New `ggml_type` quantization types carry extra evidence requirements (see CONTRIBUTING.md).

## Server feature scope

Before implementing a new server feature, confirm it falls within scope as defined in [tools/server/README-dev.md](tools/server/README-dev.md). Server usage is documented in [tools/server/README.md](tools/server/README.md).
