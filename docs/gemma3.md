<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma 3 legacy foundation

`src/model/gemma3.h` supplies a separate fixed profile and strict tensor
binding for the approved Gemma 3 4B QAT Q4_0 text checkpoint. Its state/input
and descriptor graph/plan foundation passes 13 focused CPU/no-launch controls.
The generic importer supplies the authenticated
[prepared artifact](experiments/gemma3-execution/README.md), and
`engine/gemma3_runner.*` now executes it through the shared native skeleton.
A bounded C1 screen checks finite, byte-identical eager/captured/replayed heads,
Clear reuse, malformed-input refusal and initialized-state spill/restore.
There is no serving route or media adapter. This model remains outside the
supported execution matrix. The primitive stock screen retains one positive-margin
greedy difference; explicit checked norm chains then match all 33 stock heads
byte for byte. Explicit device greedy preserves exact choices/state. Allocation-free public
binding validation narrows the latest short C1 screen to 0.56% slower than
stock backend greedy. Broader quality, performance, longer contexts and
batching remain open. An explicit [C2 decode screen](experiments/gemma3-execution/README.md#independent-prefill-c2-decode-screen)
now checks two independently-prefilled slots: zero strict greedy differences,
64/66 byte-identical heads and +2.08% paid latency versus stock. Joint prefill,
wider/ragged cohorts, longer context and serving remain unqualified.

## Actual checkpoint contract

The approved source is
[`ggml-org/gemma-3-4b-it-qat-GGUF@bbcac0d065076c47042838c0675c602411b0dd4c`](https://huggingface.co/ggml-org/gemma-3-4b-it-qat-GGUF/tree/bbcac0d065076c47042838c0675c602411b0dd4c),
`gemma-3-4b-it-qat-Q4_0.gguf`, 2,526,080,992 bytes. The exact-revision
primary HF API LFS object and HEAD `X-Linked-ETag` report whole-file SHA-256
`ee91c3e7a4ab95d8c95672f9fcb58bf236b257e9f217966bcf53a5a6df4ab49a`.
The foundation slice used metadata only; the later
[prepared import](experiments/gemma3-execution/README.md) locally authenticates
the whole payload on Spark A.

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

These are stored metadata facts; execution limits and model numerical
qualification must be checked independently. All 34 layers have separate
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
the Apache-2.0 binding, factual fixture or generator. The later descriptor
graph port below has its own upstream attribution; no backend runtime is copied.

The focused Spark B check completed on 2026-10-06: one `gemma3_test` target,
three CPU controls passed with no skips, and all six supervised steps finished
with status DONE0. The controls cover all factual profile fields and 444 tensor
storage descriptors, separate V/tied-head identities and malformed binding
refusals. Frozen source manifests and the published test binary bind this
check; the unchanged native build receipt records dependency/toolchain inputs.
Workstation REUSE, headers, portability boundaries, changed C++ formatting,
whitespace and byte-identical fixture reproduction also passed. Raw receipts
and logs remain external. This makes no execution, numerical or batching claim.

## Checked state and execution-plan foundation

`model/gemma3.*` rechecks public binding identities and supplies separate K/V
F16 planes for each independent slot. Global layers are 5, 11, 17, 23 and 29;
the other 29 layers, including layer 33, use a local ring. Each plane begins on
a 2 MiB boundary. Local capacity is `pad(min(context,1024+max_rows),256)` so
writing the whole chunk preserves every query's sliding window. Initialized
read ranges and completed write ranges are separate; wraparound writes split.
Globals can append/truncate; local rings advertise append only and no rollback
snapshots. The runner funds, materializes and initializes every read,
and retains each write through completion before recording progress.

`kernels/ggml/gemma3_graph.*` ports the text graph and GeGLU build-FFN arithmetic
from llama.cpp release v0.6.0, exact commit
`d81235049384534c167caea52b85a694f6103d14`,
[`src/models/gemma3.cpp`](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/src/models/gemma3.cpp)
and [`src/llama-graph.cpp`](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/src/llama-graph.cpp).
The derivative retains GGML authors' copyright and MIT AND Apache-2.0.
Norm weights are direct multipliers. Q/K use learned per-head RMS norms,
full-256 NEOX RoPE (local base 10,000/scale 1; global base 1,000,000/scale
0.125), then Q alone scales by 1/16 before attention scale 1. V stays raw;
there are no Gemma4 V norms, tied K/V, RoPE factors, output scales or assistants.
This actual checkpoint has no logit softcap. Embeddings scale by sqrt(2560).

Products share chunk columns while attention and F16 cache writes remain
separate for ragged slots. State-only prefill writes every layer's K/V and
omits only the final Q/attention/FFN/head tail. Frontier narrowing retains
full final-layer attention before selecting final residual/FFN rows.
`engine/gemma3_plan.*` uses sized descriptor arenas and the existing two-pass
placement check. It preflights weight/state/activation spans and aliases
and complete consumer/leaf correspondence before pre-placement binding, and
budgets copied sources and move-only owned padded host masks. Relocating roots
requires new placement/view bindings and invalidates old launches/captures.
Fresh positions, indices and causal masks are revalidated for cache reuse;
caller-owned chunk/hidden data must survive staging, and graph backing must
survive GPU retirement. The adapter does not establish catalog residency.

At the foundation snapshot, the 13 passing controls were limited to state,
descriptors and no-launch planning. D256/GQA2 primitive attention admitted H8
structurally; norm-ADD then required width2816 or 5376. The later
[checked-chain screen](experiments/gemma3-execution/README.md#checked-norm-chains-exact-bounded-c1-quality)
adds width2560 operand/refusal controls and exact bounded C1 model quality with
explicit norm/RoPE, norm/ADD, generic norm and quantized-GLU policies. No Gemma4
policy is enabled automatically. The explicit C2 screen additionally admits
D256/H8/GQA2 owner attention only for two actual owners, logical cohort2 and
offset0, with operand, occupancy, replay and model controls. H8/D512 and wider
H8 cohorts remain excluded; existing H16/32 domains are unchanged. State bounds
are context<=131072, max_rows<=8192 and slots<=16;
these are host contract bounds, not qualified long-context CUDA execution or
batching support.

The execution-plan check completed on Spark B on 2026-10-06: all eight
supervised steps finished with exit 0, and the three targets passed 6 profile/state,
3 graph and 4 plan controls without skips. Controls include both placement passes,
state-only/narrowed tails, ragged masks, atomic alias/span/leaf refusal, move-only
host-input ownership and unique declared source correspondence. Two missing
include build failures and the subsequent source-bijection control failure are
retained externally with successful separate post-authentication; the final
recovery preserves all assertions. No model or CUDA kernel was executed.

The compiled snapshot is SOURCE6 (21 changed/new paths), based on `c1b47b1`
with earlier parent docs from `80f4cb0`; final status docs preserve `73b89f4`.
The reviewed source-lock change records the Gemma3 graph derivative only;
GGML archive, patches and prepared tree are unchanged. The native receipt
SHA-256 is `c79bf0b93b3337f2b0264617469d6669723e9541a908191c088aaa79d38e3b9f`.
External source manifests, official job records and the three binary identities
retain exact provenance; none establishes runner residency, real-model
numerical quality, restore, batching, media or performance support.

## Bounded native runner

`engine/gemma3_runner.*` composes `PagedWeights`, per-slot `LiveState`,
`RunnerResources`, `RequestCohort`, `PlanCache` and shared `GraphRuns`.
Weights, state, activations, staging and output copies use the existing
catalog/VMM and commitment accounting. Held requests use direct steps;
plans recheck stable places after closure changes. State-only chunks omit an
intermediate head, while a separate head-row capacity bounds publication.
Host causal masks and their padded source copies are both funded. Public
binding checks validate all 444 fixed typed descriptors and unique indices
directly, with a shared shape/byte helper; fresh mutable source checks remain
per wave, without reconstructing resource vectors and role maps.
Clear retains mapped backing for reuse; spill/restore preserves logical
positions and refreshes the closure. Failed retirement retains the whole
probe lifetime rather than treating destruction as completion.

The internal `jitllm_gemma3_probe` uses the actual native tokenizer and fixed
4096-context, C1, F16-cache, 128-row geometry. Its first screen consumes 256
prompt rows, three supplied scalar warm rows and 32 teacher-forced rows.
It emits 33 full vocabulary heads and 32 pre-step greedy choices; the final
head has no target. These are representative screen dimensions, not newly
qualified context or support limits. The later explicit C2 probe executes
two independently-prefilled slots with joined decode; joint prefill and broader
batching remain unqualified. The ordinary primitive
policy is explicit; generic norm, quantized FFN, D256 norm/RoPE and width2560
norm/residual switches are default-off diagnostics,
with no automatic Gemma4 shape-policy inheritance. Explicit `Work::token`
publication uses the existing kept device argmax with a separate graph/cache
key and funded output; full heads remain available. The bounded device-greedy
control checks 32 choices, exact initialized state and final heads, plus
two-slot alias/mixed-publication refusals before state mutation. At that C1
snapshot only one slot executes; the later C2 control executes both slots and
checks both initialized states. Device masks, lookahead and broader optimized batching remain later work.
The following bounded route integrates ordinary serving.

## Bounded ordinary serving

The [checked serving unit](experiments/gemma3-execution/README.md#bounded-serving-unequal-widths-and-wrapped-rings)
registers this approved artifact through the normal runtime route. Defaults
are context 4096, rows 128 and one slot; two slots are admitted explicitly,
with the checked four-fusion/owner/device-greedy recipe and two funded head
rows. Larger contexts/cohorts and speculation refuse. Host contract bounds
above remain separate from these product admission bounds.

Actual two-slot unequal padded widths, wrapped local rings, partial departure,
Clear/spill/checkpoint restore and kept-conversation adoption pass focused
state/reference/HTTP controls. Cached scalar and cached joined responses
repeat exactly across restart; cold/cached and scalar/joined token trajectories
can differ. The current short runner screens are +0.96% C1 and +2.36% C2
paid latency versus ordinary stock backend greedy, not parity qualifications.
Joint prefill, wider cohorts, broader memory/swap and sustained gates remain
open. The serving GPU fixture currently requires provisioning on Spark B
before its full models tier can run there.
