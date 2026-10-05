<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma reference fusion-family controls

This diagnosis isolates pinned llama.cpp's norm/RoPE and norm/residual
fusion families on the same 1,024-token War and Peace prefixes used by the
[representative comparison](../gemma-quality/PROTOCOL.md). It changes no
native operation, model policy or numerical acceptance bound.

Use llama.cpp `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4` from image
`ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7`.
The approved Gemma 26B-A4B GGUF has SHA-256
`f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f`.
Inputs are the first 1,024 explicit I32 token IDs including BOS 2, SHA-256
`b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610`.
Context is 4,096, chunks are 128 rows, KV is F16, FlashAttention and CUDA
graphs are allowed, and every row publishes its full 262,144-entry F32 head.
No prompt template, speculative decoder or implicit tokens are introduced.
Row j scores token j+1; BOS and the final retained head are unscored.

Export the pinned `ggml` source to external scratch. `patch_controller.py`
checks the exact original `ggml-cuda.cu` hash before adding bounded
selection logging and explicit diagnostic policies:

- `all`: the original fusion-selection body and allocator hints.
- `norm_rope`: only the original RMS_NORM → MUL → ROPE predicates and
  launchers, including the original optional VIEW → SET_ROWS continuation.
- `norm_add`: only the original RMS_NORM → MUL → ADD predicate and launcher.
- `both`: admit both norm families above, and no other operation fusion.
- `none`: no operation fusion, matching the separately retained ordinary
  reference control.

For the individual and combined norm allowlists and `none`, omit the
allocator's weighted-reduction fusion dependencies because that fusion is unavailable. Preserve graph
tensors, streams, model arithmetic, ordinary kernels and capture support.
These restrictions are diagnostic; `all` remains the production reference.

Build only the controller translation unit with the declared SDK's NVCC
13.4.92, the pinned image's GCC 13.3 and the upstream fast-math flags. Link
and preload it against the image's original CUDA math library and runtime.
The translation unit's compiled batched-pointer kernel performs integer
pointer arithmetic; floating-point launchers remain in the image. This
controller build is a toolchain change, so it is usable for attribution only
if `all` reproduces all original production heads byte for byte and `none`
reproduces all original unfused heads byte for byte. Also require each
process's policy banner, so a preload that never dispatches cannot pass.
No callback requests or retained intermediates may suppress fusion.

Run each individual norm allowlist and the combined norm allowlist twice.
Verify all repeated rows byte for byte;
compare complete distributions, teacher-forced NLL/PPL and strict argmax IDs
with both the original production and ordinary outputs. Authenticate the
unchanged native calibration
`2c97b2a4d65277f00b242180d7f54128a395284490d84db0db24e634adcfc901`
before analysis. Its p99 margin movement is zero. Do not recalibrate or widen
that allowance to accommodate an allowlist result. The old calibration
continues to describe its original native paths only.

Report host `try_fuse` decisions separately from CUDA launches and graph
replays. Logging makes these runs unsuitable for competitive timing. This
control can identify a fusion-family numerical effect; it cannot alone
establish which kernel instruction caused it, the correctness of every
family combination, optimized batching, long-context quality or support.
