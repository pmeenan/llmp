<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma 4 native RMS scale convention

The approved `nvidia/Gemma-4-26B-A4B-NVFP4` revision
`a19cfe00be84568a6867111c9a68c9c44fdcffe6` exposed an incorrect native
mapping: Gemma 4 stores RMS gamma directly, but the mapping declared
zero-centered scales and requested `norm.plus_one` at import. The repair
removes both declarations for Gemma 4 native sources. GGUF is already direct
gamma; Gemma 2/3 keep their distinct zero-centered convention.

## Source contract and scope

The checkpoint declares Transformers 5.6.2, resolved to commit
`aa935fb53dc38b56f10cc86b3a74354c2e99412f`.
[`Gemma4RMSNorm.forward`](https://github.com/huggingface/transformers/blob/aa935fb53dc38b56f10cc86b3a74354c2e99412f/src/transformers/models/gemma4/modeling_gemma4.py)
multiplies normalized activations by the stored weight directly. The same
class supplies input/post-attention, Q/K, pre/post-feedforward, both branch
post norms, routed-input norm and final norm. The inspected source file's
SHA-256 is `c99a0717221ff1b5ffe91ace1d8be59b8ad8ba78271acdbc5990bb1e016c3df7`.

The locked llama.cpp commit `d81235049384534c167caea52b85a694f6103d14`
[`Gemma4Model.norm_shift`](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/conversion/gemma.py)
returns zero. Its inherited converter therefore leaves norms unchanged;
Gemma 2/3 explicitly add one. Converter SHA-256:
`b127a524d92660c86c67a371996ef29076c94cb8bcb9a8c4facf060a901d1671`.
This is an architecture convention, not a compensating change to a
previously converted GGUF or a checkpoint-name exception.

TensorFold `cb2ebf0540f42604e2759b2ddef497861e928248` reads loaded MLX
Gemma 4 norm weights directly in `kernels/gemma/v1/decode.py`.
Its declared mlx-lm dependency floor, 0.31.3, uses ordinary `nn.RMSNorm`
and its [`sanitize`](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/gemma4_text.py)
preserves norm tensors. This is supporting source-contract evidence,
not a new reference-engine pin or an MLX quality result.

## Bounded checkpoint audit

Reading the two safetensors headers and selected small vectors consumed
6,220,862 bytes, with no model execution or whole-shard scan. Headers describe
271 text norm vectors (nine per block plus final), totaling 1,224,192 BF16
bytes. A subsequent bounded audit read all 1,224,192 source bytes and
2,448,384 published artifact bytes using recorded tensor extents. All 612,096
elements in all 271 vectors exactly equal source BF16 gamma plus one,
rounded to F32; none of the vectors retains direct gamma. Measured pairs:

| Tensor | Source BF16 value | Published artifact F32 value |
| --- | ---: | ---: |
| Layer 0 K norm, every element | 0.1220703125 | 1.1220703125 |
| Layer 5 K norm, every element | 0.062255859375 | 1.062255859375 |
| Layer 0 Q norm, every element | 1.0234375 | 2.0234375 |

The artifact manifest records `norm.plus_one` for every affected vector;
the spec also records that fold, and the index describes each vector as
dense F32. This confirms both incorrect arithmetic and unnecessary widening.
The affected per-block roles are `attn_norm`, `attn_q_norm`, `attn_k_norm`,
`post_attention_norm`, `ffn_norm`, `post_ffw_norm`, `pre_ffw_norm_2`,
`post_ffw_norm_1` and `post_ffw_norm_2` (30 each), plus `output_norm`.

The audited original artifact's metadata SHA-256 values are:

| Metadata | SHA-256 |
| --- | --- |
| Manifest | `2228547054f4244bbfff5f6c7e38c25065a6072a2272968eed3dfb9eacc86731` |
| Index | `7574466a278a0087a9248e8ebb6f3531579249631c379d107d795befddb58a47` |
| Spec | `ecd096c13c3fb89a53f052b20d3206f119b17e6fb14c9f35137bc69a95432e3e` |

These are metadata and selective-payload checks, not a fresh whole-artifact
integrity scan. The manifest's dirty source-Git field is not an authenticated
build-source receipt; the payload comparison independently demonstrates the
published fold.

Metadata, tensor shapes and source graphs agree on width 2816, 30 blocks,
16 query heads, local 8 KV heads at width 256, global 2 KV heads at width
512 with K/V tying, 128 experts/top-8, dense width 2112 and expert width 704.
Embedding scale is sqrt(width); layer scalars remain separate learned
multipliers. Router scale and per-expert contribution scale are separate
from RMS gamma. Native expert relayout separates NVFP4 tensor scales from
codes and applies each projection scale once; the down contribution also
receives the independent router expert scale. These static checks found
no second mismatch; they do not establish execution parity.

## Storage and validation

Removing the fold preserves the norm vectors' original dense BF16/F16/F32
storage. Existing component `F32Parameters`/`NormScale` converts only small
parameter vectors to transient F32 graph operands. GGML `CheckConvert`
already admits BF16/F16 to F32; fused norms receive the converted F32 operand.
No bulk weight widening or execution-family branch is introduced.

The regression fixture covers dense and MoE native sources through HF and
MLX readers, all norm roles plus final norm, and F32/BF16/F16 byte routes.
It checks unchanged signed/zero/small gamma values, dtype/layout, absence
of a spurious fold and zero-centered flag. Existing Gemma 2/3 plus-one and
Gemma 4 GGUF tests remain controls. The integrated wave-K native build
passed on Spark B, followed by all 3,472 unique shared checks with no failures
or skips. This includes the new importer regression. The source base was
`dfba7a78cd6f6a821148d3cc434ab182153ece7c`, with integrated source patch
SHA-256 `18d5cc309d70799895d8e7ee96eb0e57b4ddc39ac4d0db1f8e4d8d22d8d00789`.
That integrated patch also contains separately reviewed work; it is not the
Gemma-only delta. No source-quality qualification exists yet.

## Preserved negative evidence

Artifact `0ce111cc287b80c9fb1ff248c03115f5e5e3ec87bc0400f257260f0c7569ccba`
with runtime SHA-256
`dc0360616643d221034ede4e4fb74ee712ec0ae81fd919189f5dc28917703550`
passed startup after the separate tokenizer list repair. The unchanged
five-token literal input `[818,5279,529,7001,563]` and ordinary Paris chat
both produced garbled 64-token outputs, ending by length; the literal
returned 64 finite log probabilities. Readiness took 2.006368 seconds and
TERM retirement returned zero. The runtime hash remained unchanged.
This is a negative serving screen, not a source-quality comparison. Preserve
that artifact; re-import the approved source under a new identity before
repeating the exact screen. No sole-cause or repaired-quality claim follows
from the mapping diagnosis.


## Corrected import and unchanged screen

The bounded exclusive `m35-u3-gemma26-direct-gamma-import-r1` job published
new artifact `1d20ce879b2fe72c947699357da5e20d3f6388e9cc7858b9eb6fcc2516cd3875`.
Publication passed the writer's deep verification, StoredSpec and ValidateSpec.
The original artifact remains preserved. Import CLI SHA-256:
`cc49910f28d551f61b6dc0cd049046c36a998b5da8a8773975eea647a4eb4471`.

The corrected artifact's metadata SHA-256 values are:

| Metadata | SHA-256 |
| --- | --- |
| Manifest | `aeedabdf082a32883e9c197a7a6347135ac0cdfb190aacca833003bc26fababa` |
| Index | `470041519492eae2e8c084b8b1ed6182a8bd72d60da42d326c4c102f542ffc2a` |
| Spec | `7e163fda1b1e528b38f1c1f65d905de8a08acc44dea5ca452304e80760525d91` |

A separate bounded payload audit compared all 271 norm vectors / 612,096
elements to the same approved source. Every vector is byte-identical source
BF16 gamma in dense layout: 1,224,192 bytes read from each side. The spec has
no `norm.plus_one` fold or `norm_zero_centered` parameter, and the manifest
records no plus-one transformation. The audit did not rescan whole shards.

The unchanged screen used frozen runtime SHA-256
`894ce8998791951fd9ae71afbb38cb263a384b726699369ce3afa9f770ca6e9b`
in exclusive job `m35-u3-gemma26-direct-gamma-smoke-r1`. Settings remained
context 4096, one slot, F16 KV, prefill 512, temperature zero and maximum
64 output tokens. The raw five IDs remained exactly `[818,5279,529,7001,563]`,
with no added special tokens. The chat messages remained system
“Answer only the requested capital city, without explanation.” and user
“What is the capital of France?”.

| Control | Result | Completion | Verdict |
| --- | --- | --- | --- |
| Raw five IDs | ` is` repeated 64 times; 64 finite log probabilities | 64 tokens, length | Negative; unresolved |
| Ordinary chat, 35 input IDs | `Paris` | 2 tokens, normal stop | Positive bounded screen |

Both responses returned HTTP 200. Readiness took 1.809482 seconds; TERM
retirement returned zero, and the runtime's before/after hashes matched.
The process succeeded, but the raw-prompt behavior remains a negative cell.
The norm defect and its byte-level repair are established independently;
this small screen does not establish general quality, context coverage,
batching or speed. The cause of raw-token repetition remains open pending
an exact-source reference diagnostic. Adding BOS or replacing the prompt
would constitute a separate prospective control, not repair this cell.


## Reference diagnostic preparation

A CPU-only, network-disabled inspection of the existing immutable vLLM image
`sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`
confirmed built-in Gemma 4 causal and conditional registrations, direct RMS
gamma, checkpoint prefix/expert remapping and ModelOpt NVFP4 W4A4 support in
source. The inspected Gemma implementation SHA-256 is
`6074e4a2f0c40a5bb28d101060fb4fe64f11ac6ee0cb6f524a9bb3deaceab797`;
ModelOpt source SHA-256 is
`3f3ca743fd3c66d72be92b7544591a8632f1aa422b73bb50c83e8a3281196e7d`.
That inspection loaded no model and ran no reference request. The
[prospective exact-ID protocol and helpers](gemma26/PROTOCOL.md) now pass the
pinned image's real CPU-only `TokensPrompt`, parser/renderer, `EngineArgs`
and sampling API checks: the five IDs survive without tokenizer access or
added BOS. The first registered source launch subsequently completed BF16/BF16-KV
startup, then failed while serializing the owned inspection callback, before
any `generate` request. Its logs selected FlashInfer CUTLASS MoE and Triton
attention and warned about differing W1/W3 global scales; no scale repair or
quality conclusion follows. The [prospective transport amendment](gemma26/PROTOCOL.md#prospective-callback-transport-amendment-2026-10-22)
adds the supported container-local callback serialization setting and a real
CPU codec round-trip check. The first failure remains preserved, and review
and an amendment commit precede a fresh launch. Native F16 KV still differs
from source BF16 KV; no source raw behavior has been collected.
