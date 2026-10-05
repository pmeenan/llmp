<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma 4 assistant binding foundation

`model/gemma4_assistant.{h,cc}` defines closed profiles and checked semantic
bindings for the approved Q8_0 assistants paired with Gemma 4 26B-A4B and 31B.
This foundation does not execute assistants, enable speculation, or qualify
target or assistant model support. The target's existing numerical and
performance failures remain unchanged.

The contracts follow the [pinned llama.cpp assistant loader and graph](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/src/models/gemma4-assistant.cpp),
its [shared-cache mapping](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/src/llama-model.cpp),
and the exact Google [26B config](https://huggingface.co/google/gemma-4-26B-A4B-it-assistant/blob/6e5aaaf4c42b98394530b8fda2e95cadd65c151c/config.json)
and [31B config](https://huggingface.co/google/gemma-4-31B-it-assistant/blob/627c5ec1458b9086b841a91e0512fd31fd2fbbf1/config.json).
Lean fixtures retain the measured header identities and all 49 tensor contracts;
weights, vocabulary arrays and raw inspection logs remain external.

## Closed tensor and semantic contracts

Both assistants have width 1,024, FFN width 8,192 and four Q-only blocks:
local/local/local/global. Each has 23 Q8_0 matrices and 26 F32 norm, scalar or
frequency-factor tensors. Binding requires exactly these 49 ordinary resources,
with exact role, type, shape and semantic readable byte length. All matrix input
widths are multiples of 512, so the approved payloads need no extra GGML quant
row tail. Resource ordering may vary; identities must cover the complete domain
without duplicates. The canonical full-vocabulary head aliases the assistant
embedding `[1024,262144]`. K/V matrices, private KV, expert arrays, extra roles,
centroid/order tensors and other representations are refused.

| Contract | 26B-A4B companion | 31B companion |
| --- | --- | --- |
| Target feature width | 2,816 | 5,376 |
| Pre-projection, GGML dimensions | `[5632,1024]` | `[10752,1024]` |
| Post-projection | `[1024,2816]` | `[1024,5376]` |
| Q heads | 16 | 32 |
| Borrowed local KV heads/layer | 8 / 28 | 16 / 58 |
| Borrowed global KV heads/layer | 2 / 29 | 4 / 59 |

Local head/rotary width is 256; global head/rotary width is 512. Global frequency
factors are F32 `[256]`; global rotary width is not reduced to 128. Context and
vocabulary are 262,144, local window 1,024, RMS epsilon `1e-6`, and rotary bases
10,000 local / 1,000,000 global.

The artifact overload authenticates its one bounded kept GGUF file and checks
these dimensions, four shared layers, attention pattern, head/KV counts, no
per-layer input, rotary/norm settings and target output width before binding.
Legacy `rope.scale_linear` and the common loader's rotary attention-factor,
alpha, original-context and finetuned overrides must remain absent; these
same-shaped metadata additions could otherwise change reference rotary math.
Enabled final softcap, ordered embeddings, unsupported rotary scaling or
noncausal attention are refused even when all tensor shapes remain unchanged.
The resource-span overload binds shapes only; its caller must separately check
metadata semantics. Mutable public bindings are revalidated before producing
a target-sharing map, including rank/type bounds before tied-head comparison.

`CheckGemma4AssistantTarget` revalidates the complete approved target profile
and binding, then returns four semantic cache references. It creates no process
addresses, residency leases, state generations or writable cache ownership.
Input uses the **target** token embedding scaled by `sqrt(target_width)` and
the target feature **after final output normalization**, not the pre-norm
`graph.hidden`. Assistant final normalization feeds its complete tied head and
the post-projection recurrent feature. Every draft in one chain uses the same
position and reads the frozen completed target prefix. There is no assistant
cache catchup or cache write.

## Canonical vocabulary is an independent admission check

`CheckGemma4AssistantVocabulary` compares borrowed decoded values without
allocating payload copies: all 262,144 token strings and raw finite scores,
514,906 merge pairs in rank order, and raw token kinds. Score equality is bit
identity after F32-to-double conversion, including signed zero. The checker
bounds individual strings and their aggregate across both vocabularies.
Actual native checks of **both paired targets** confirm equal strings, merges
and scores. Only these two serialized kind differences are permitted:

| Canonical ID/string | Target kind | Assistant kind |
| --- | --- | --- |
| `1`, `<eos>` | 1 | 3 |
| `258884`, `<\|video\|>` | 3 | 1 |

Equal kinds also pass; every other mismatch is refused. The target keeps all
tokenizer and stop authority. A matching head width or shape binding does not
establish vocabulary compatibility, and this foundation does not add a runtime
assistant admission route.

The existing `ReadGgufTokenizer` supplies token strings and merge pairs.
`ReadGgufMetadata` supplies raw score/type arrays because Gemma's tokenizer spec
does not retain scores and applies token-kind overrides. Future admission must
fund the kept bytes and decoded containers **before** calling either parser,
then run the ordinary tokenizer validation and this paired alignment check.
Likewise artifact binding allocates a bounded metadata buffer and descriptor
containers; it supplies semantic validation, not a host capacity reservation.
No parser or tokenizer API was changed.

## Pins and evidence limits

| Carrier | Approved revision | Full GGUF SHA-256 | Prepared artifact |
| --- | --- | --- | --- |
| `mtp-gemma-4-26B-A4B-it.gguf` | `c099eb48e663fd284577b04978a94ffccb261841` | `6326fb9f5e487aa8dcdd313a091e3c67724cb2a666ec3b7d2895b5b26d93ed1b` | `1040a0299a459e00ad0a77efd77bd319ac593986ba2c9ef29eb03d07ce97db42` |
| `mtp-gemma-4-31B-it.gguf` | `c1ac76e99d5513b141e8adde7288b85c3f9c32ec` | `5ae8b0117bed601e8924c6305bd5b0585de361d51f0e77091bcb4252cf1f27de` | `447a5c20a0a25632bf35e118d5dde1867a182cd209b9ccd3afe93866c3696120` |

Both complete assets were already size/hash verified and imported through the
generic converter `m3-1` / `layout-a0d1980a9eddd1ad`. This task read their existing
indexes and kept metadata; it did not download weights again. The actual kept
assistant metadata SHA-256 values are `f9139d76248f46c9e0a039876dfc4be2f11de7f54b81b0d615f97b93dc17608c`
(26B) and `81e4444a11b9608cf80ef4ab0e2bbbafa29e5b5c737c017ce46023a1b8caca93`
(31B).

At this task's entry, 2026-10-05 07:15 UTC, the latest observed TensorFold HEAD
was `609ca419abecebdc5a059498a613680bd3aa847f`, version 0.6.5. Its pinned
[Gemma recipe](https://github.com/ashhart/TensorFold/blob/609ca419abecebdc5a059498a613680bd3aa847f/docs/recipes/gemma-4.md)
remained MLX-only for 26B, without a CUDA Q8_0 assistant comparison. A future
execution task must refresh that observation and qualify its matched reference;
this foundation ran no assistant inference or quality/speed comparison.

All six CPU fixture controls and four actual paired metadata/vocabulary/artifact
checks passed. The final integrated Spark-b locked build passed; all 1,678 tests
passed in 590.65 s with no skips. Local REUSE/headers on 1,391 files, portability
on 386 sources and changed-source format/diff passed. These controls cover both
closed profiles, malformed and mutated descriptors and same-shaped semantic
metadata changes. Raw logs stay external.

## Execution work still owed

The next graph/runner slice must keep or copy normalized target features for
every potentially accepted verify row, including narrowed final heads; borrow
only the same slot's initialized cache with residency/generation protections;
and fund assistant weights, scratch, features and proposals separately.
Target verification must restore rejected overwritten local-ring bytes and
visibility before continuation. Constant position, seed/feature alignment,
wrap/rejection/acceptance, cancellation, switch/restart and independent joined
chains require real controls. Greedy and sampled target acceptance remain the
authoritative policies; applicable Qwen/DeepSeek graph, staging, adaptive-draft
and batching optimizations need actual family/format qualification. Existing
norm/RoPE contracts can be assessed at D256/D512, but the norm/residual fused
contracts admit target widths 2,816/5,376; assistant width 1,024 needs its own
checked operand/shape eligibility. Q8_0 products, shared preparation and
row-invariant joining likewise need controls at the actual 1,024/8,192 and
projection/head dimensions, followed by paid whole-chain qualification. Full heads
are the baseline; no centroid optimization is enabled by these carriers.
