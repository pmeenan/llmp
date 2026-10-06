<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma 4 assistant contracts and native component

`model/gemma4_assistant.{h,cc}` defines closed profiles and checked semantic
bindings for the approved Q8_0 assistants paired with Gemma 4 26B-A4B and 31B.
The native graph/component now executes bounded Q-only chains over protected
target operands. It does not enable serving speculation or qualify target or
assistant model support. The target's existing numerical and performance
failures remain unchanged.

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
the binding task ran no assistant inference or quality/speed comparison.

All six CPU fixture controls and four actual paired metadata/vocabulary/artifact
checks passed. The final integrated Spark-b locked build passed; all 1,678 tests
passed in 590.65 s with no skips. Local REUSE/headers on 1,391 files, portability
on 386 sources and changed-source format/diff passed. These controls cover both
closed profiles, malformed and mutated descriptors and same-shaped semantic
metadata changes. Raw logs stay external.

The [original-image assistant oracle seam](experiments/gemma-assistant-reference/README.md)
now captures frozen target inputs and full assistant heads/recurrent features
at C1, serial C2 and genuine batch two. Its exact own repeats and unchanged
physical target-state witnesses establish a reference seam; native assistant
quality, performance and speculation qualification remain open.

## Native component and protected target operands

`kernels/ggml/gemma4_assistant_graph` and `engine/gemma4_assistant_plan` build
and bind one Q-only query per independent slot. All four blocks reuse the same
slot's local/global K/V, including raw target K-as-V semantics. Query position
P, initialized endpoint P and ring retention are separate checked values:
attention sees positions through P−1, while each recurrent assistant query
stays at P. Funded host masks exclude the unwritten current cell and all padded
query columns. This is a bounded diagnostic mask path; a checked device producer
is still needed for optimized execution.

The target-owned `Gemma4Assistant` shares the target stream, request cohort,
catalog and library handle, using its own PagedWeights, PlanCache, GraphRuns,
staging and recurrent feature storage. `SetupAssistant` requires authenticated,
caller-funded paired vocabulary views and runs compatibility admission itself.
Views need remain alive through synchronous admission; the caller authenticates
them to the exact loaded immutable artifact identities. Successful admission
retains the artifact/profile pairing, not borrowed parser containers. Failed
setup cannot register, bind or step and remains retained for fenced release.

`retain_features` is an explicit target setup policy, false by default. It
retains requested POST-finalnorm rows through a separate paid feature gather,
with independent feature IDs even when head outputs are narrower. It disables
final-layer narrowing for that immutable policy; its feature shape participates
in plan identity and maximum envelopes. Default target graphs remain unchanged.
This enabled-feature target policy requires its own arithmetic evidence rather
than inheriting the old narrowed-head target proof.

A move-only `FrozenBorrow` requires a completed nonzero prefix, initialized
latest feature P−1, usable nonrestoring state and an existing held request.
It authenticates logical epoch/address and catalog extent content/backing
generations. Feature backing is pinned for the runner lifetime. Same-slot
writes, clear, spill, replacement/restore and release refuse during the borrow;
other owners can progress. The trusted caller retains the request through all
component work. Guard release is not GPU retirement. Feature validity is
invalidated by state replacement/spill/restore and is not serialized as a
checkpoint; a new completed target row is required before borrowing again.

Initial features remain unchanged. Later recurrence writes component-owned
storage; complete finite heads/features publish only after successful retirement
and whole-batch validation. Uncertain completion retains execution owners and
faults the shared cohort. External feature copies require caller-funded,
node-owned pinned destinations; failed copies conservatively retain that destination to process exit. The component neither samples nor verifies proposals and never
writes target caches.

The [bounded execution report](experiments/gemma-assistant-execution/README.md)
records complete original-image C1 head/projection byte identity at one frozen
P64 input and native-target C1/C2 own repeats with unequal histories and direct
cache/feature witnesses. This is component evidence, not optimized C2 or
end-to-end target/assistant quality or competitive performance qualification.

The [Gemma31 frozen C1 extension](experiments/gemma31-assistant-reference/README.md)
now also matches all complete original-image heads and 5,376-value recurrent
features for one/three-step P64 chains in both exact own repeats. Original
complete target state/caches/cells and native 52 MiB borrowed tensors remain
unchanged. This is a frozen-input component prerequisite; full native-target
chain quality, C2, serving/speculation and performance qualification remain open.

The default-off [bounded engine C1 greedy unit](experiments/gemma-assistant-greedy-unit/README.md)
now releases its frozen borrow before verifying the authoritative anchor plus
up to three drafts, commits only the accepted prefix after retirement, and
carries its selected target head/feature plus an uncommitted next anchor.
Both approved real target/assistant pairs pass 11 focused controls, using an
independent same-four-query target wave, semantic accepted KV, rejected-tail
restoration, projected continuation and whole-discard/refusal evidence.
This is transaction semantics, not scalar-width quality or serving qualification.

The [matched Gemma31 greedy transaction](experiments/gemma-assistant-greedy-reference/README.md)
also matches the pinned original for one C1/P64/depth-three unit with query64
prefill and query4 verification, using the target's existing norm/RoPE and norm/add
chains. All complete target/assistant heads and retained features are byte-exact;
accepted count and pending carry agree. The plain-norm baseline failure remains
recorded. This adds one original transaction control, without 26B, scalar-width,
serving or competitive whole-chain performance qualification from that one unit.

The [Gemma31 repeated all-cost screen](experiments/gemma-assistant-throughput/README.md)
now commits the same 32 tokens and pending anchor across native/original plain
and actual assistant unit execution at C1/P64. Native assistant mean 1.315482 s
versus plain 2.939108 s reduces elapsed 55.24%, with 12 units/46 target rows/34 drafts;
it remains 1.25% slower than original assistant. Full own-repeat/timed witnesses
pass before cross read. EOG is ignored consistently; this adds no serving,
terminal, other-prefix, sustained or 26 qualification.

## Execution work still owed

The component supplies explicit retained-feature rows, same-slot initialized
cache borrows and separate funding. The default-off
[engine-only target verifier](experiments/gemma-target-verify/README.md) now
retains all heads/features for one to four C1 rows, restores rejected KV writes,
and publishes the accepted feature/cursor only after retirement. Both profiles
pass focused transaction controls; scalar-prefix byte parity and deterministic
verify budget-pressure refusal remain unestablished. Assistant serving must
release its frozen cache borrow before verification and separately fund an
explicit verify/head envelope; neither is integrated into serving.
Constant position, seed/feature alignment,
wrap/rejection/acceptance, cancellation, switch/restart and independent joined
chains need their wider serving controls beyond the bounded engine evidence above. Greedy and sampled target acceptance remain the
authoritative policies; applicable Qwen/DeepSeek graph, staging, adaptive-draft
and batching optimizations need actual family/format qualification. Existing
norm/RoPE contracts can be assessed at D256/D512, but the norm/residual fused
contracts admit target widths 2,816/5,376; assistant width 1,024 needs its own
checked operand/shape eligibility. Q8_0 products, shared preparation and
row-invariant joining likewise need controls at the actual 1,024/8,192 and
projection/head dimensions, followed by paid whole-chain qualification. Full heads
are the baseline; no centroid optimization is enabled by these carriers.
