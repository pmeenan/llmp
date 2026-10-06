<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma 3 legacy foundation

`src/model/gemma3.h` supplies a separate fixed profile and strict tensor
binding for the approved Gemma 3 4B QAT Q4_0 text checkpoint. It supplies no
execution graph, state layout, importer, runner, serving route or media
adapter. This model remains outside the supported execution matrix; legacy
execution, linear RoPE, batching, restore, reference quality and performance
qualification remain open. Existing native SentencePiece and Gemma 3 chat
rendering are reusable prerequisites, not proof of model execution.

## Actual checkpoint contract

The approved source is
[`ggml-org/gemma-3-4b-it-qat-GGUF@bbcac0d065076c47042838c0675c602411b0dd4c`](https://huggingface.co/ggml-org/gemma-3-4b-it-qat-GGUF/tree/bbcac0d065076c47042838c0675c602411b0dd4c),
`gemma-3-4b-it-qat-Q4_0.gguf`, 2,526,080,992 bytes. The exact-revision
primary HF API LFS object and HEAD `X-Linked-ETag` report whole-file SHA-256
`ee91c3e7a4ab95d8c95672f9fcb58bf236b257e9f217966bcf53a5a6df4ab49a`.
No whole payload was downloaded or locally hashed for this slice.

| Metadata fact | Value |
| --- | ---: |
| Architecture / GGUF version | `gemma3` / 3 |
| Layers / hidden width / FFN width | 34 / 2,560 / 10,240 |
| Query / KV heads | 8 / 4 |
| Key / value dimension | 256 / 256 |
| Vocabulary / configured context | 262,208 / 131,072 |
| Sliding window | 1,024 |
| Stored RoPE base / linear scaling factor | 1,000,000 / 8 |
| RMS epsilon | F32 1e-6 |
| Actual tensor count | 444 |
| Tensor types | F32 ×205; Q4_0 ×238; Q8_0 ×1 |

These are stored metadata facts. They do not implement layer scheduling,
rotary behavior, or a context execution limit. All 34 layers have separate
Q, K and V matrices, per-head Q/K norms, attention output, gate/up/down FFN
matrices, and four hidden-width norms. Layer matrices are Q4_0; norms are
F32. The embedding is Q8_0 `[2560,262208]`, 713,205,760 bytes. There is no
`output.weight`; the binding aliases the embedding and accepts an optional
prepared output role only on that same resource.

The binder rejects edited profiles, wrong architecture, missing, duplicate
or unused roles, independent output heads, wrong type/shape, short readable
storage, and expert arrays. It uses the current native GGML representation
helpers. The used F32/Q4_0/Q8_0 IDs and block geometry agree with locked
llama.cpp v0.6.0 commit `d81235049384534c167caea52b85a694f6103d14`;
their existing helper-table origin remains b29c606e. Every actual quantized
row is a multiple of the native 512-element padding boundary.

## Metadata fixture provenance

`tests/unit/data/gemma3/gemma3_4b_qat.json` preserves selected scalar metadata
and all tensor descriptors, without token vocabulary or weight payloads.
Its source is the existing authenticated 6,514,895-byte metadata prefix
SHA-256 `3073f37db9c1977ed7d68cd64707ce32043eecd250000a929a2ceb6723b1ad4c`
plus one bounded 262,144-byte HTTP range. Retrieval required HTTP 206 and
exact `Content-Range: bytes 6514895-6777038/2526080992` before reading the
body; a whole-response fallback was refused. Captured range SHA-256 is
`62abe6abbea861399b424e362fb77839ccc6ab67a9a3cb09b12bf978419d1c33`.
Raw response bytes, headers and primary API receipts remain external.

The complete table is 26,361 bytes, SHA-256
`e47e271674ae26f87da138b369f4e8518f99f1be5af0413bb9f2f98b68065b4a`.
The header ends at 6,541,256 and tensor data starts at 6,541,280. Alignment
32 is the GGUF v3 default: `general.alignment` is absent. All 444 names
are unique; descriptor spans are aligned, disjoint and within the primary
reported file size, with the last tensor ending at that file boundary.

Reproduce the factual fixture without network or model execution:

```sh
python3 tests/unit/data/gemma3/generate.py PREFIX RANGE OUTPUT
```

The generator authenticates both inputs, table identity, bounds and native
type-helper sizes. Checkpoint metadata declares `general.license="gemma"`;
that weight notice is informational under D-087 and does not relicense
the Apache-2.0 binding, factual fixture or generator. No upstream graph or
kernel source is copied in this foundation.

The focused Spark B check completed on 2026-10-06: one `gemma3_test` target,
three CPU controls passed with no skips, and all six supervised steps finished
with status DONE0. The controls cover all factual profile fields and 444 tensor
storage descriptors, separate V/tied-head identities and malformed binding
refusals. Frozen source manifests and the published test binary bind this
check; the unchanged native build receipt records dependency/toolchain inputs.
Workstation REUSE, headers, portability boundaries, changed C++ formatting,
whitespace and byte-identical fixture reproduction also passed. Raw receipts
and logs remain external. This makes no execution, numerical or batching claim.
