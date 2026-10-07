<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma 3 prepared import and bounded native execution

The [approved pin](pins.json) names the exact 4B QAT Q4_0 source from
[the Gemma3 foundation](../../gemma3.md). The existing generic
`artifact-layout/import_m3.py build` path prepares its F32/Q4_0/Q8_0
weights with the unchanged v0 writer. Gemma3 preflight is closed to this
one source identity: its typed profile metadata, vocabulary count and all
444 tensor descriptors must match the checked factual fixture. Its complete
header digest must also match before planning. The actual approved table has
separate V matrices and no independent output head; the native binding aliases
the embedding at execution time.

The importer still authenticates the whole source against the pin, reparses
its metadata and descriptors, rechecks source bytes during writing, deep
verifies the prepared artifact and publishes atomically. A header digest or
preflight success alone does not authenticate weight payloads. Other GGUF
architectures keep the existing generic path. No artifact schema or writer
identity changes. Import alone establishes no model execution support.

On a Spark, after downloading the exact-revision source into `SOURCE` and
checking its size and whole-file SHA-256 against `pins.json`:

```sh
python3 docs/experiments/artifact-layout/import_m3.py build \
  STORE docs/experiments/gemma3-execution/pins.json \
  gemma3-4b-qat-q4_0 SOURCE/gemma-3-4b-it-qat-Q4_0.gguf
python3 docs/experiments/artifact-layout/import_m3.py verify STORE/ARTIFACT_ID
```

Run downloads, import and verification through the installed `spark-job
start --gpu` supervisor, with bounded timeouts and its official `wait`.
The checkpoint repository is `ggml-org/gemma-3-4b-it-qat-GGUF`, revision
`bbcac0d065076c47042838c0675c602411b0dd4c`. Checkpoint terms are `gemma`
and remain informational under D-087. Checkpoints and prepared model artifacts
stay in the external model store; this change ships no weights in the core.

The seven `test_gemma3_import` controls require no model payload. They check
recipe identity, every typed metadata field, the actual parser's compact
vocabulary count, the complete
tensor table and tied-head contract, source length/pin/header/shard refusals,
refusal before output publication and the unchanged generic architecture path.
The existing importer controls retain whole-source hash/publication coverage.
On 2026-10-07, Spark A downloaded and locally hashed the full approved
2,526,080,992-byte source, prepared it and passed an explicit independent deep
verification. The artifact ID is
`8c7103418a6608022e5eda50a0dcc4b7688a0d59ef239813c9de0984161397fb`;
its converter remains `m3-1+layout-a0d1980a9eddd1ad`. It contains 444 resources,
36 groups, 1,226 chunks and one 2,519,662,592-byte safetensors shard. The
19 focused importer controls passed on both the workstation and Spark A.
Official supervised jobs `m35-gemma3-download` and `m35-gemma3-import`
completed with exit 0; their logs and verification receipt remain external.

The source lives under Spark A's
`~/.local/share/jitllm/models/ggml-org/gemma-3-4b-it-qat-GGUF@bbcac0d0/`.
The prepared store is
`~/.local/share/jitllm/gemma3-import-20261007/artifacts/`.

## Bounded native own control

On 2026-10-07, `jitllm_gemma3_probe` executed that artifact on Spark A
(GB10, driver 580.178.04, SDK `aarch64-c09daba6ac31edee`, core profile,
locked GGML tree `d50cb7f97867b4c5…`). This is an internal C1 probe at
context 4096, F16 K/V, 128-row chunks and one published head row. It consumes
256 prompt rows, three supplied scalar warm rows and 32 teacher-forced rows.
The actual native tokenizer produced the first 291 IDs from an authored prose
input; their 1,164 bytes have SHA-256
`3f94d579e6749a32aee252d01562102fcd835bdb231e1620229b77d0d7308753`.
The source text SHA-256 is
`3016ddb3960e77f9a9dc649bca88dad0a7578d8afb07dd8ecf00b98144dc3756`.
The text and IDs stay in external scratch; no user prompt or model payload is
checked in.

All 33 full 262208-vocabulary heads are finite and byte-identical between eager
execution and two graph runs. Their complete-file SHA-256 is
`ff2cf41c82b19f6996f1c646a8cb3fcd96039bebc87058c6363ae6c9a81b5324`.
Each graph teacher run captures once and replays 33 times, with no graph refusal
or coverage violation. Clear preserves 68 resident state extent identities;
full versus state-only prefill produces byte-identical heads and initialized
state hashes. Malformed token, past and head-capacity requests preserve the
completed prefix and initialized state. Spill/restore preserves the logical
positions and exact initialized-state hash. All 32 pre-step choices and final
heads also match the lifetime run; every probe retires successfully.

The measured envelope is 23,068,672 activation bytes, 10,485,760 scratch bytes,
4,194,304 host-input bytes and 1,341,520 plan-floor bytes. The probe separately
funds an 8 MiB caller floor and, for the lifetime control, an 8 MiB pinned state-copy
buffer. This screen runs ordinary primitives; generic norm and quantized FFN
switches remain explicit diagnostics.

The 13 focused Gemma3 foundation/graph/plan tests pass, including the regression
that actual unmasked host inputs refuse before launch. Workstation changed-file
formatting, REUSE/header 1795 and portability boundary 411 checks pass. The initial
probe construction error and an overly broad CTest selector are retained in
external official job records; their narrow corrections preserve all controls.
The final focused tests and own controls are repeated in
`m35-gemma3-runner-final`. The initial own-control binary SHA-256 is
`6e52c84ea3eef1af2065ebece5bd76a2dba77c6036844253ffe8178f2b0b96ec`.
The final binary adds selection counters and has SHA-256
`bb7902b05b4a1859d66993db1c1350b6ba9a8538017fb1fb595e424a498f3fd4`,
with native receipt SHA-256
`cf6bce22c79008fc49245a74206f778b036a4efb6b69ae4f7aa1f572fb6469f5`.
Full source identities, raw heads and logs remain external.

These dimensions establish a representative own control, not new supported
context limits, batching, sustained performance, a serving route or media
support. Reference quality/performance qualification remains open.

## Primitive stock first screen

The same Spark runs the original llama.cpp v0.6.0 public API from commit
`d81235049384534c167caea52b85a694f6103d14`, image
`ghcr.io/ggml-org/llama.cpp@sha256:c604ea4f1c2e8d5c8b27d89fef727384d59e23c5b07e369cde5393820e0607db`.
[`llama_probe.cc`](llama_probe.cc) keeps stock fusion and graphs, explicitly
matches context 4096/C1/F16 K/V/128-row batches, and independently reproduces
the native 291-ID prefix from the same text. Its first 128-row chunk is state-only,
the second publishes one full head; three supplied scalar rows precede 32
teacher-forced rows. Two stock runs have identical finite 33-row files, SHA-256
`01aaccb8c1b50538d9499489c385a7005cf2f68a5d3d941c768502fae0c527dd`.
All contexts, models and owned containers retire before the queue continues.
The native own freeze completes before the first stock evaluation.

[`analyze.py`](analyze.py) reuses the existing Gemma quality analysis arithmetic.
It scores targets 259..290 against the first 32 heads and leaves the final 33rd
head unscored; all 33 rows participate in finite, distribution and greedy checks.
The first-screen gate requires zero positive-margin greedy differences and
`expm1(mean_target_nll_delta)<=0.03`, with exact ties reported separately and
no inherited numerical allowance. This is a 32-target screen, not corpus PPL.

| Primitive native versus stock | Result |
| --- | ---: |
| Positive-margin greedy differences / full rows | 1 / 33 |
| Exact-tie differences | 0 |
| Miss row / native ID / stock ID | 23 / 15808 / 28251 |
| Stock margin at that miss | 0.2225341796875 |
| Native / stock mean target NLL | 3.8654058210 / 3.8594656475 |
| Mean target NLL delta | +0.0059401735 |
| `expm1` of mean target NLL delta | +0.595785% |
| Mean / maximum total variation | 0.03453917 / 0.08167455 |
| Maximum raw logit difference | 0.72173023 |

The strict greedy gate **fails** despite the likelihood screen remaining within
3%. The matched cycle timing queue therefore did not run; this unit makes no
performance or stock-parity claim. The separate norm-only and quantized-GLU-only
teacher repeats remain byte-exact to the primitive heads and preserve the same
quality miss. Runtime plan counters establish actual selections:

| Explicit native policy | Bound plans | Selected RMSNorm/Mul steps | Selected quantized GeGLU steps |
| --- | ---: | ---: | ---: |
| Primitive | 3 | 0 | 0 |
| Norm only | 3 | 610 | 0 |
| Quantized GLU only | 3 | 0 | 35 |

These cumulative counts describe successfully bound plans, excluding Setup's
envelope probes; they do not count kernel execution or graph replay. All three
counter-instrumented teacher runs preserve the frozen primitive head, choice
and final-head bytes. Both switches remain explicit diagnostics and do not fix
quality. The supported execution matrix remains unchanged.

## Checked norm chains: exact bounded C1 quality

The explicit `normrope` diagnostic transfers the existing D256 norm/Mul/RoPE
launcher to query H8 and key H4, with local base 10000/scale 1 and global base
1000000/scale 0.125. Eight operand cases cover one and 128 rows, independent
FP64 normalization, primitive phase checks, unchanged raw inputs and fresh
captured positions. All three focused GPU tests pass. Across three runtime
plans this policy selects 203 norm/RoPE steps. Two teacher repeats are exact,
but stock comparison still has the same row-23 positive-margin difference;
`expm1(mean_target_nll_delta)` is +0.440470%, mean total variation 0.02806966.
The official `m35-gemma3-normrope` queue stops at its quality gate and runs no
timing. This isolated transfer is insufficient.

The next addition deliberately extends the existing norm/Mul/ADD check from
widths 2816/5376 to include 2560, using the unchanged upstream launcher. FP64
GPU controls cover one and 128 rows, both operand orders, raw-input preservation
and fresh captured inputs. Host controls check both orders, paid residual
GET_ROWS, malformed strides/overlap and refusal of unapproved widths
2559/2561/4096. All 19 focused norm host/GPU tests pass on Spark A.

The `normropeadd` diagnostic selects 203 norm/RoPE and 202 norm/ADD steps across
three runtime plans. Its two finite 33-head files are byte-identical to frozen
stock: SHA-256 `01aaccb8c1b50538d9499489c385a7005cf2f68a5d3d941c768502fae0c527dd`.
The `optimized` diagnostic additionally selects 205 generic RMSNorm/Mul and
35 quantized GeGLU steps; its two teacher files preserve those same stock bytes.
Both policies have zero greedy differences, target-NLL deltas, total variation
and raw-logit deltas. Each teacher captures once and replays 33 times. These
remain cumulative bound-plan selections, excluding Setup probes, rather than
counts of executed or replayed kernels. All four fusion switches default off.

Only after both quality gates pass, one short stock/native/native/stock cycle
screen runs the optimized candidate. Each arm warms 256 rows, three supplied
scalar rows and eight greedy steps, then clears. Paid work is 256 prompt rows
and 32 greedy decode steps, with the three supplied scalar rows between them
excluded from timing. All arms publish full heads to the host. This is the
same 4096-context/C1/F16/128-row topology as the teacher screen.

| RNNR arm | Prefill ms | Decode ms | Paid total ms |
| --- | ---: | ---: | ---: |
| Stock 1 | 55.4588 | 403.974 | 459.4328 |
| Native 1 | 56.5638 | 414.036 | 470.5998 |
| Native 2 | 57.8917 | 413.652 | 471.5437 |
| Stock 2 | 55.3528 | 404.610 | 459.9628 |

Mean paid latency is 471.07175 ms native versus 459.6978 ms stock, or +2.474223%.
Generated tokens per paid second are 67.9302 versus 69.6109. All four 32-ID
greedy histories match (SHA-256
`c828ecf66c451fff2aface3ba592b32e77792f7e2848127f59e1132b22e20b7f`), as do
all four finite final heads (SHA-256
`a9c7ff3747cafa8224a23f014888bf5ff2229e50f1df72d13d4447682d0d4c34`).
Native cycles capture three times and replay 44 times, with no refusal or
coverage violation; native lifetimes and stock containers retire successfully.

The official `m35-gemma3-normropeadd` job completes all 19 steps with exit 0.
The actual native binary SHA-256 is
`d6ca09c4e2aed2744352d7f2d51fb601a22787003642ce2772881095b4fb9e06`;
the native receipt and locked environment remain as above. Exact source/binary
bindings, own-repeat records, quality results and raw outputs remain external
under Spark A's `~/.local/share/jitllm/gemma3-execution-20261007/`, in
`normrope-only`, `normropeadd-only` and `optimized`.

This clears the representative C1 quality screen, without establishing speed
parity, corpus quality, 8K or maximum context, sustained performance, optimized
batching or serving support. The later device-greedy screen below preserves
these bounded quality results; H8 batching remains open. No default recipe is
selected here.

## Device greedy: exact choices and backend-sampler comparison

The runner's explicit `Work::token` contract publishes a completed I32 device
argmax instead of a full vocabulary row. The graph/cache key distinguishes
this output, Setup funds its kept output and plan envelope, and every wave
requires one publication kind. Alias, mixed row/token, all-output and state-only
greedy requests refuse before state growth or staging. Completion is proven
before publishing any host choice; an out-of-range choice quarantines state.
The existing argmax implementation is unchanged. Full rows remain available for
scoring, teacher comparisons and final verification.

Two `greedy-own optimized` runs each compare 32 GPU choices against full-head
argmaxes, then compare their final full head and all initialized K/V bytes.
Both traversals retain 68 state extents through Clear and produce state SHA-256
`087796779a4dc74509809ddf9854a5439aec0370469f92fa749e8c028367c87b`.
The control allocates two slots to prove output-alias refusal preserves both
prefixes and slot0's state bytes, but executes only slot0. It establishes no
executed batching. Each run publishes 35 device tokens, captures three graphs
and replays 66 times, including distinct output-shape eager/capture/replay.
All 33 full heads, 32 choices and the final head match the frozen optimized
teacher bytes above; the unchanged strict stock quality screen is exact again.

The stock caller additionally selects the original d812 public backend greedy
sampler chain for sequence0 through `cp.samplers`, without modifying stock's
chain or detaching it for final verification. Full GPU offload, valid public
sampled-token results and the exact repeated sampled rows establish this API
selection, not observed kernel-launch counts. Two `greedy-teacher` arms each
report 36 backend samples and reproduce all 33 frozen stock heads, choices and
final head byte for byte. The public sampled-logits count is 262208 at every
observed row: stock retains and copies full sampled rows internally. This is
not a zero-copy stock comparison.

After both own/reference controls pass, one RNNR compares native device
publication against that stock backend sampler. The same 4096-context/C1/F16/
128-row topology pays 256 prompt rows and 32 generated steps; three supplied
scalar rows remain untimed. Every arm pays and checks the final full head.
Native warms its token shape, then the final three warm steps publish full
heads to exercise eager/capture/replay of the verification shape; stock keeps
its backend sampler throughout. Warm-up consumes eight generated steps before
Clear; paid histories match across all arms.
Across warm and paid work, native publishes 44 GPU choices; stock reports 48
backend samples, including unconsumed samples on full-head verification steps.
Each native cycle captures four graphs and replays 42 times, without refusal or
coverage violations. Counts describe actual publications/API samples, not
kernel-launch totals.

| Backend/device greedy RNNR arm | Prefill ms | Decode ms | Paid total ms |
| --- | ---: | ---: | ---: |
| Stock 1 | 55.4134 | 390.889 | 446.3024 |
| Native 1 | 56.4489 | 400.490 | 456.9389 |
| Native 2 | 56.7723 | 401.603 | 458.3753 |
| Stock 2 | 56.2137 | 390.846 | 447.0597 |

Mean paid latency is 457.6571 ms native versus 446.68105 ms stock, or
+2.4572455%. Generated tokens per paid second are 69.9213 versus 71.6395.
All four histories and finite final heads preserve the preceding full-head
cycle hashes. This short comparison retains a latency gap; it does not
establish parity or isolate the speed effect relative to the earlier run.

On Spark A, official `m35-gemma3-greedy` completes all 18 steps with exit0,
including nine focused graph/plan tests and explicit native/container
retirement. The actual native binary SHA-256 is
`aaa95401e41cb30db27c29e54356d2e23d58356927ecdd37317b2b1ea1d0ce09`;
receipt, artifact, source pin and locked environment remain as above.
Source/binary binding, own controls, quality and cycle aggregates remain
external under `gemma3-execution-20261007/device-greedy/`. All fusion switches
remain default off, and token publication is explicitly requested per work.
Optimized batching, wider context/quality, sustained performance and serving
qualification remain open.

TensorFold's per-task current source was checked on 2026-10-07 at
[`ed78d6fc204d89d90b045bf033d6551e7714f3a1`](https://github.com/ashhart/TensorFold/blob/ed78d6fc204d89d90b045bf033d6551e7714f3a1/README.md)
(release 0.6.6). Its documented model table lists Gemma 4 through MLX; there is no
matching documented Gemma3 CUDA/GGUF recipe for this screen. This makes that
comparator ineligible here, without claiming that all Gemma3 execution is
unsupported by TensorFold.
