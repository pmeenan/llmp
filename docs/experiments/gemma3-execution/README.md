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
these bounded quality results. At this C1 snapshot H8 batching remains open;
the later independent-prefill C2 screen closes only two-owner joined decode.
No default recipe is selected here.

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
not a zero-copy stock comparison. In exact d812, the greedy sampler leaves
`data.logits` intact; the graph exports it as `t_sampled_logits`, which the
context copies independently of the separate `needs_raw_logits` guard.
The recorded C1 helper uses `llama_memory_clear(...,true)`; its measured timings
retain that reset configuration, rather than the logical-only reset used by
the subsequent ordinary Gemma4 reference screen.

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

On Spark A, official `m35-gemma3-greedy` completes all 18 steps with exit 0,
including nine focused graph/plan tests and explicit native/container
retirement. The actual native binary SHA-256 is
`aaa95401e41cb30db27c29e54356d2e23d58356927ecdd37317b2b1ea1d0ce09`;
receipt, artifact, source pin and locked environment remain as above.
Source/binary binding, own controls, quality and cycle aggregates remain
external under `gemma3-execution-20261007/device-greedy/`. All fusion switches
remain default off, and token publication is explicitly requested per work.
Optimized batching, wider context/quality, sustained performance and serving
qualification remain open.

## Allocation-free binding validation: measured host cost

Source inspection found that each wave calls `Gemma3SourceBytes`, then
`Gemma3Sources` calls it again. Each validation formerly reconstructed 444
resource vectors/role strings and rebound the entire model. A host-only
helper opens the approved artifact metadata and constructs the same immutable
C1 scalar greedy descriptors at past259/read512, context 4096/maxrows128. It
performs no placement, device query/allocation or submission. Each span has
16 warm calls followed by 256 timed successful calls; the five spans are
nested and must not be added. The pair measures the actual duplicated call
sequence, including fresh source checks and padded-mask allocation.

| Host span | Before µs/call | After µs/call |
| --- | ---: | ---: |
| Public binding check | 102.121 | 3.25623 |
| Graph check | 105.565 | 7.12240 |
| Source-byte check | 105.976 | 7.65384 |
| Source construction | 107.528 | 8.94070 |
| Wave's source-byte + source pair | 213.776 | 16.5198 |

The pair falls 92.272%. The replacement checks each fixed role's type, rank,
shape and minimum readable bytes directly, plus the tied output and all 444
bounded distinct resource indices. The binder and checker share the same
shape-before-byte-arithmetic helper. Successful checks allocate no resources,
role strings or replacement binding. Public mutable graph and fresh host
position/cell/mask checks still run on every call; there is no trusted-handle
bypass. Focused controls mutate every descriptor's type, dimensions, rank,
readable size and identity, and also accept reordered resource indices and
excess readable storage. All 17 foundation/state/graph/plan tests pass.

Two native own controls preserve all frozen 33 heads, 32 choices, final head
and initialized state bytes above, including refusal and Clear controls. The
unchanged strict fixed-stock quality screen remains exact. Only then, one
short RNNR uses the unchanged original-image backend greedy caller and the
same paid geometry and final full-head boundary as the preceding screen.

| Validated-binding RNNR arm | Prefill ms | Decode ms | Paid total ms |
| --- | ---: | ---: | ---: |
| Stock 1 | 57.3012 | 390.060 | 447.3612 |
| Native 1 | 55.9064 | 392.319 | 448.2254 |
| Native 2 | 55.3291 | 395.327 | 450.6561 |
| Stock 2 | 55.7526 | 390.734 | 446.4866 |

Mean paid latency is 449.44075 ms native versus 446.9239 ms stock, or
+0.5631496%. Generated tokens per paid second are 71.1996 versus 71.6006.
All four paid histories and finite final heads preserve the prior hashes;
native cycles retain four captures/42 replays/44 device publications, and
stock reports 48 backend samples with full 262208-element sampled rows.
This narrows the measured short C1 gap from the preceding 2.46% screen.
These are sequential bookended screens, not an interleaved old/new A/B.
At this C1 snapshot broader quality, sustained performance, context, batching
and serving gates remain open. Host timing identifies a contributor, without attributing every
whole-cycle difference to that span.

Official `m35-gemma3-source-cost` completes its two host-diagnostic steps;
`m35-gemma3-validation` completes all 15 build/control/quality/timing steps,
both with exit 0. The final native binary SHA-256 is
`e72bbf9d7663a409f1ec899b6a12586ac95802891752b9234ea56beff627d127`.
The unchanged diagnostic source SHA-256 is
`714844b5dfe777465dafb51005b0637123b053d0f5ebcbac688c26c6052959e2`;
its before/after binaries are `cc30ddd3…ada8eb7`/`b2ccc021…4ab2ad1`.
Exact source/binary/recipe bindings, host spans and own/quality/cycle aggregates
remain external under `gemma3-execution-20261007/source-cost/` and
`source-validation/`. Artifact, environment, sampler and false-default fusion
policies are unchanged.

TensorFold's per-task current source was checked on 2026-10-07 at
[`ed78d6fc204d89d90b045bf033d6551e7714f3a1`](https://github.com/ashhart/TensorFold/blob/ed78d6fc204d89d90b045bf033d6551e7714f3a1/README.md)
(release 0.6.6). Its documented model table lists Gemma 4 through MLX; there is no
matching documented Gemma3 CUDA/GGUF recipe for this screen. This makes that
comparator ineligible here, without claiming that all Gemma3 execution is
unsupported by TensorFold.

## Independent-prefill C2 decode screen

The explicit `owner_decode` option reuses the existing independent-root
attention contract and unchanged upstream kernel for D256/H8/GQA2, only with
two actual owners, logical cohort2 and offset0. Existing H16/32 domains are
unchanged; H8/D512 and H8 cohorts3/4 refuse. The Gemma3 graph views its already
contiguous scaled/roped Q rows, joins the two padded host masks once per wave,
and preserves each slot's K/V root and writes. Only two one-row segments with
equal bounded read widths select this path; other chunks retain ordinary
attention. Setup measures both output forms through the same planner and
funds the joined masks, custom output and workspace. The option defaults off.

Twenty focused foundation/state/graph/plan controls pass. The existing
parameterized GPU owner test retains its H16/32 cases and adds H8/C2 at
cells256/512/1024: ordinary packed MMA and actual separate roots are
byte-identical, FP64 NMSE is 1.148e-7/1.258e-7/1.588e-7 against the existing
5e-4 bound, and fresh-Q/mask capture replay preserves source bytes. On the
48-SM GB10, both original and owner occupancy are one block/SM; grids are
32/48/48 and funded scratch is 266496/399616/399616 bytes. Unsupported shape,
overlap, incomplete root and bounded plan/source refusals remain checked.
The first prerequisite run caught a test-only F16 download being interpreted
as F32 in three preservation assertions; the failed receipt is retained.
Correct typed raw-byte checks pass without changing production arithmetic.

`jitllm_gemma3_c2_probe` executes two initialized slots. Each receives its own
actual native-tokenized 291-ID prose input, authenticated before either model
runs and independently checked by stock tokenization. Slot0 retains the C1
input; slot1 input/text SHA-256 are
`405cfff4be661ddd6949f0a3882d760e1edaefc262a8226b662804ef4a0408f7` /
`7d5ddc0e2abe262ea2eabeefad16c68c7d07e52633f11ca73b059e6e8ce73001`.
Each slot independently prefills 128 state-only rows then 128 head rows,
followed by three supplied scalar warm tokens. Joined decode then executes
32 one-row waves across both owners, with F16 KV and context 4096 per slot.
Joint prefill is outside this screen.

Two native own runs freeze 66 finite teacher heads and 64 pre-step choices,
then compare device publication against those choices and both final heads
and initialized states after Clear. Actual two-slot alias/mixed/wrong-prefix
refusals preserve both prefix metadata and initialized bytes. Both slots also
spill and restore their exact initialized states. Each own run publishes
70 GPU tokens and records eight captures/64 replays. Teacher/full-head SHA-256
is `f3b355b5038d0338deb331ecc3030521c9c883e0e5d1b82a7b0ff144118cb030`;
choice/final hashes are `78420433…73d62` / `c6a66e3c…de1e`.

Only after own controls pass, the tiny public-API caller runs twice in the
unchanged original v0.6.0 image, with an independent official greedy chain for
each sequence. Its teacher-only metadata callback observes ordinary FLASH
Q `[256,1,8,2]`, K `[256,512,4,2]` and logical mask `[512,1,1,2]` for all
1088 joined layer/step observations. Exact pinned source dispatches this
ordinary multistream geometry through MMA; the callback does not observe
kernel launch counts. Native global/local reads are both512, with mask rows
padded to32. The callback is absent from both timing arms.

The strict 64-target stock screen has zero greedy differences, zero tie
differences and relative conditional-loss delta −0.00532347% (mean target NLL
delta −5.3236115e-5). Of 66 full heads, 64 are byte-identical; rows 39 and 56
(step19/slot1 and step28/slot0, zero-based) differ. Mean total variation is
2.5559011e-5 and maximum raw logit delta is 0.0373087. The final two rows are
finite and checked, but unscored. Stock repeat head SHA-256 is
`1e9108a7681e4671e3db972af008718a7077ed984a94909caa7337ee4edb7743`.
This passes the zero-positive-margin/≤3% conditional-loss first screen,
without claiming exact whole-head parity.

One subsequent short RNNR pays both slots' 256-row prompt and 32 generated
steps, excludes the six supplied warm tokens, and pays both final full heads
in each arm. Warm-up uses eight joined generated steps before Clear. Native
uses GPU publication for the first five warm decode waves, then three full
head waves to exercise that distinct shape's eager/capture/replay lifecycle;
stock retains its official backend sampler throughout. The new C2 stock
caller leaves state-only prefill asynchronous, allowing the next public
decode and final head to establish normal ordering/completion. Historical C1
caller/results above remain unchanged; no timing magnitude is attributed to
that source-derived overlap difference.

| C2 RNNR arm | Prefill ms | Decode ms | Paid total ms |
| --- | ---: | ---: | ---: |
| Stock 1 | 103.303 | 423.598 | 526.901 |
| Native 1 | 110.181 | 425.192 | 535.373 |
| Native 2 | 112.689 | 425.633 | 538.322 |
| Stock 2 | 101.688 | 423.205 | 524.893 |

Mean paid latency is 536.8475 ms native versus 525.897 ms stock, or
+2.0822518%; generated tokens per paid second are 119.2145 versus 121.6968.
Mean independent prefill is 111.435 versus 102.4955 ms, and decode is
425.4125 versus 423.4015 ms (+0.475%). Of the 10.9505 ms paid gap,
8.9395 ms lies in the measured independent-prefill span and 2.011 ms in
decode. These are phase boundaries, not a causal kernel attribution.
All four 64-token histories and finite two-row final heads are byte-identical:
SHA-256 `92556d0c…752f` / `21192b80…3136`. Native cycles record eight
captures/44 replays and 88 GPU publications; stock records 96 backend samples
including unused final decisions and full 262208-element sampled rows. There
is no zero-copy or broad parity claim.

Bound-plan selection counts exclude Setup probes and replay. Each own run
selects owner attention 68, plain norm824, quantized GeGLU140, norm/RoPE814 and
norm/ADD812; each timing cycle selects 68/548/70/542/540 respectively. The
quantized fusion remains one-column only and applies to independent chunks,
not joined two-column decode. All fusion and owner switches remain explicit
and default off in the runner.

Official `m35-gemma3-c2-build2` completes four prerequisite steps and
`m35-gemma3-c2` completes all 17 build/own/reference/quality/timing steps,
both exit 0. Native binary SHA-256 is
`9b48244d3188dabc792fa8066abd81d32d52ffa675721ad88b3985e407007698`;
the original-image caller is
`301ccb418e723a37414c121a9da6f4bad6ec98df011019041be00f8aaa7cf197`.
Actual source/binary/receipt bindings, inputs and own/quality/cycle aggregates
remain external under `gemma3-execution-20261007/c2-build/` and `c2/`.
At this snapshot, joint prefill, wider/partial/ragged cohorts, context/ring-state
and serving qualification remained open. The following bounded serving unit
adds actual unequal-width/ring/departure and product lifecycle controls. At
that serving snapshot, joint prefill remained open; the later transfer below
qualifies bounded compatible chunks. Wider cohorts and sustained gates remain
open.

## Bounded serving, unequal widths and wrapped rings

The approved artifact now registers through the ordinary `gemma3` runtime
route, including Chat Completions and literal Completions. The default is
context 4096, rows 128 and one slot; an override admits two slots and funds two
head rows. Larger contexts, more owners and speculation are refused. The
adapter explicitly selects the checked generic norm, quantized GeGLU,
norm/RoPE and width 2560 norm/ADD recipe, plus eligible two-owner attention
and device greedy; the internal runner's diagnostic options still default off.
This does not inherit other Gemma4 policies.

```toml
[models.gemma3]
artifact = "8c7103418a6608022e5eda50a0dcc4b7688a0d59ef239813c9de0984161397fb"
context = 4096
prefill_chunk = 128
max_slots = 2
```

Each branch owns its cursor, history, state and checkpoints. Checkpoint
restore funds whole extents but requires every logical initialized range to
finish copying before publishing the cursor. Pending restore refuses execution,
growth, spill and the wrong copy direction; Clear can cancel it. Partial
capacity refusal preserves the old completed ledger and the peer. The shared
copy-retirement controls retain staging and quarantine uncertain work.
Adoption checks exact layout, footprint and empty/idle ownership before
publishing a kept conversation. The GPU fixture requires the approved artifact;
it was provisioned on Spark A, and its Spark B provisioning remains owed.

[The context probe](../../../benchmarks/gemma3_context_probe.cc) independently
prefills two different real tokenized inputs in 128-row chunks, supplies three
scalar warm tokens per owner, evaluates 32 joined teacher steps, then departs
each owner for four scalar steps. Each run produces 74 full heads and scores 72
actual targets; the last two heads have no target. Own repeats also prove
GPU/full-head choices, both initialized states, refusal before mutation, Clear,
spill/restore, and checkpoint advance 128/restore/repeat on both slots.

| Prefix positions | Actual native decode read cells, global/local | Exact stock heads | Strict greedy differences | Relative target-loss delta | Mean TV | Max raw logit delta |
| --- | --- | --- | --- | --- | --- | --- |
| 256 / 768 | 512/512 and 1024/1024 | 2/74 | 0 | −0.250833% | 0.0127580 | 2.04506 |
| 1280 / 1536 | 1536/1280 and 1792/1280 | 2/74 | 0 | +0.00017135% | 0.0000177987 | 2.35639 |
| 256 / 256 | 512/512 for both | 72/74 | 0 | −0.00473199% | 0.0000227959 | 0.0373087 |

The unequal/ring shapes use ordinary segmented attention where the joined
owner adapter requires equal padded widths. These are strict greedy and
≤3% relative conditional-loss screens, with no new numerical allowance and
no full-head equality claim. The wrapped case crosses the 1024-token local
window for both owners. Broader/ragged/wider batching remains unqualified.

The new original-image caller leaves state-only `llama_decode` calls naturally
ordered without an extra `llama_synchronize`; final head publication completes
paid prefill. Stock's unmodified backend greedy sampler still exports and
copies full 262208-element sampled-logit rows to the host, independently of the
raw-logit-copy guard. Native greedy publishes 4 bytes per choice. A short C2
runner RNNR retains exact 64 generated IDs and final heads: native 537.158 ms,
stock 524.775 ms, **+2.35968%** paid latency (n2 per engine). This is an internal
runner screen, not endpoint latency: its initial frontier is selected on the
GPU, whereas ordinary serving prefill publishes a full row. Joint prefill is
still missing, and the earlier C2 measurement attributes most of the total gap
to its measured prefill span.

## HTTP lifecycle and retained conversations

[The HTTP controller](http_control.py) uses ordinary product configuration,
with the actual stored template SHA-256
`7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4`,
BOS 2/EOS 1 and the existing native Gemma3 renderer/stop rules. The focused
renderer fixtures pass. Literal requests at 1280 and 1536 positions have exact
same-geometry response/likelihood and fresh-continuation repeats; context 4097
refuses and a subsequent valid request succeeds. Chat checkpoint replay,
SSE, suppressed user stop, client departure, a third request with two slots,
and subsequent healthy completion all pass.

Final drained runtime metadata counts only successful two-owner generation
waves: epoch 1 completes 162 groups/324 units and 447 GPU decisions; epoch 2
completes 26/52 and 90. Both bind 34 owner-attention plans (selection counts,
not launch counts). Concurrent client reads alone are not execution evidence,
and the third request's client observations do not identify its precise
scheduler wait interval.

Two 68-extent kept records validate inode, generation, size, digests, permissions
and layout, then are adopted by a clean restart. Records remain byte-identical
before the first replay. Both cached scalar responses and cached parallel
responses are exact before/after restart, with 21 cached tokens per request;
same-policy repeats are exact too. Cold and cached/cohort trajectories are
not invariant: one prompt emits a curly apostrophe in cold/scalar execution
and an ASCII apostrophe in cached joined execution. That difference occurs
before restart as well. The original failed cold-versus-cached assertion and
private response snapshots remain external; correcting the test to compare
matching replay geometry adds no stock quality allowance and does not claim
cross-geometry token equality.

Serving selects the cohort before every completed unit. Gemma3 now reuses an
unchanged selection within an already held request, as DeepSeek/Qwen do;
bound/released/fault and slot-count/range/duplicate checks still precede this
fast path. Changed selections and all state mutations still refresh closures.
The focused GPU control covers held repeats, outside-request selection,
changed-mask peer protection, malformed selections and pending restore.

[The native HTTP bookend](http_cycle.py) compares the preserved pre-guard
runtime with the otherwise identical guarded build, old/new/new/old. One cold
primer is descriptive; two cached warm repeats and the paid request must have
exact choices/usage and positive cached reuse, within and across all arms.
Each request generates 32 tokens, with C1 and C2 exercised (n2 per runtime).
C1 latency falls 437.305→421.496 ms (**−3.61525%**); C2 falls 518.224→501.756 ms
(**−3.17767%**). This is a native endpoint attribution screen, separate from
the runner-versus-stock comparison; it is not a stock HTTP or sustained rate
qualification.

## Current scalar reference refresh and verification

[The fair scalar caller](llama_fair_probe.cc) preserves the historical caller
and changes only its artificial state-only wait. Native own repeats retain
exact 32 GPU choices, all 33 full heads, final head and initialized state;
two new stock teachers match all 33 heads byte for byte. The short RNNR
is native 449.3537 ms versus stock 445.09905 ms, **+0.955888%** paid latency
(n2 per engine), with exact generated histories/final heads. The stock helper's
historical memory-clear(true) warm reset remains off-clock; full sampled-logit
host transfers remain unchanged. Earlier C1/C2 results retain their original
methods, and these small screens establish no universal parity.

Spark A official jobs complete the prerequisite six steps, focused HTTP
build/check five steps, context 33 steps, final HTTP controls, fair-C1 all 17
steps and the corrected HTTP bookend. Focused final checks pass 30 CTest cases
plus 3 Gemma3 serving GPU cases, with no skips. The failed input-sizing stage,
relocated-runtime cuBLAS bootstrap, and cold/cached harness assertions are
preserved externally. No full suite was run. Relocated diagnostic runtimes
use the exact authenticated pinned cuBLAS/Lt closure for both arms.

The runtime binary is
`e133403cabb507cb0c26e9e210f9a60c6d5f8191aafe6719bdf676ee3f0f55b2`;
actual source/binary/receipt bindings, official logs and own/quality/cycle
aggregates remain outside Git under `gemma3-execution-20261007/serving-build/`,
`serving-model2/`, `serving-http/` and `fair-c1/`. The frozen final source merges
concurrent Gemma2 changes without altering this unit's executed C++ bytes.
At that serving snapshot, compatible joint prefill remained open. The later
screen below qualifies the bounded transfer; wider cohorts, broader
context/memory/reclaim/swap and sustained quality/performance remain open.

## Compatible joint prefill and common attention reads

The bounded two-slot route now joins compatible plain prefill chunks through
one shared prepare/dispatch/retire/apply seam. Each owner retains its 128-row
chunk, independent history, cursor, cache roots and checkpoint layout; the
explicit total wave limit is 256 rows. Equal chunks of 2–128 rows with the same
head mode and padded local/global read widths may join. One-row boundaries,
different final row counts, scoring, checkpoint/reuse units and other families
keep their scalar paths. The scheduler retains its original prompt/decode
choice, then settles at most one eligible peer's readiness unit so two cold
prompts can actually join. Cancellation is checked before that owner's funding;
a cancellation arriving during peer preparation can leave a safely completed
prefix. Preparation time is included in the wave duration and apportioned by
admitted rows for per-session calibration.

Two attention geometry differences explained the first strict failure.
Combined dense products alone left prefill attention split into two one-stream
MMA calls, while stock uses one two-stream call. Common-operand controls found
both plans launch 96 blocks but use 16 versus 32 destination tiles and disable
versus enable the mask prepass. Their outputs differ despite passing FP64
controls. Packing real K/V activations and masks for the existing two-stream
MMA restores both initial model heads exactly.

The remaining differences began at joined decode with actual 512/1024-cell
cache reads. Stock exposes the maximum padded read width to both streams;
the previous native equality gate selected independent attention. The repair
keeps each cache descriptor, initialized footprint and cursor unchanged. Only
temporary activations pad the shorter K/V prefix with real F16 zero storage,
and all 32 mask rows receive an invisible tail. The original independent-root
owner kernel then sees a common read width. Source-free graph-owned fill nodes
retain the strict undeclared-root check. Mixed-endpoint Setup probes fund these
additional activations, scratch and source envelopes.

| Unequal-prefix candidate | Exact stock heads | Positive-margin differences / 72 targets | Relative target-loss delta |
| --- | --- | --- | --- |
| Combined products, split prefill attention | 0/74 | 3 | −2.6386611% |
| Packed prefill, split unequal-width decode | 2/74 | 2 | −0.5152126% |
| Packed prefill and funded common-width decode | 73/74 | 0 | +0.0000000888% |

The first two failed strict gates and their raw evidence remain preserved.
The final common-operand decode control uses actual 512/1024-cell roots and a
1024-cell temporary read: columns4, 48 blocks, mask prepass enabled and 399616
scratch bytes. Packed attention and independent padded owners match byte for
byte, including eager/captured replay, separate short/long input perturbations,
poisoned fill destinations and preservation of all original source bytes.

| Prefix positions | Exact stock heads / 74 | Strict greedy differences / 72 | Relative target-loss delta | Mean TV | Max raw logit delta |
| --- | --- | --- | --- | --- | --- |
| 256 / 768 | 73 | 0 | +0.0000000888% | 1.03532e−9 | 0.0211544 |
| 1280 / 1536 | 70 | 0 | −0.0000006274% | 7.66411e−9 | 0.0834565 |
| 256 / 256 | 74 | 0 | 0 | 0 | 0 |

Each shape retains exact own repeats, device/full-head choices, both initialized
states, refusal before mutation, Clear, spill/restore, checkpoint advance and
restore, and partial-departure controls. All 216 scored transitions pass the
unchanged strict choice/loss screen. The unmatched raw heads above remain
reported; this is not a claim of complete bitwise equality at every geometry.
Actual bound-plan selections include 33 packed-prefill and 68 owner-attention
steps in the unequal own control, excluding Setup and replay. Successful
prefill groups/rows are counted separately after runner completion.

One short same-geometry RNNR screen pays two 128-row chunks per owner, three
supplied off-clock scalar warm rows, 32 joined decode steps and final full
heads. Native mean paid time is 0.51563615 s versus stock 0.5080885 s
(+1.4854991%, n=2). Prefill means are 0.08887415/0.085028 s and decode means
0.426762/0.4230605 s. All 64 generated IDs and final heads match exactly across
arms. These are short runner costs, with stock's normal state-only overlap and
unmodified full sampled-logit host transfers; they establish no sustained or
endpoint parity.

The installed Spark A `m35-gemma3-common-owner2` job completes all 28 steps:
42 focused host cases, the new operand control, four serving GPU cases and
three shared retirement controls, followed by own/reference gates and the
single short timing comparison. The prior split/packed strict failures and
compile-only harness failures remain external. Model data, input records,
actual binary/source receipts and aggregates live under
`gemma3-execution-20261007/joint-prefill-model/`, `joint-prefill-packed/` and
`joint-prefill-common-owner2/`; the common synthetic proof is under
`joint-prefill-common2/`. The source/artifact, SDK and stock-image pins above are
unchanged. Final current-parent composition then rebuilds the actual runtime,
probe and operator binary, passes the new padding/common-prefill controls and
both legacy cap0/Gemma2 cap50 controls, and repeats the unequal initialized
own control byte for byte before any HTTP request.

All five HTTP cases pass: actual compatible ring prefill, literal likelihoods
and bounds, chat/cache/SSE/stops, a queued third request with partial departure,
and two-slot checkpoint adoption with exact matched cached restart replay.
The first cleanly drained epoch records 15 successful joined-prefill groups
and 2410 processed rows, plus 182 joined decode groups/364 units. It selects
265 packed-prefill attention steps; bound-plan counts remain distinct from
completed wave counts. Both restart epochs retire normally.

| Cold endpoint | Old seconds | New seconds | New/old latency delta |
| --- | --- | --- | --- |
| C1, 1280 prompt tokens + 32 generated | 0.689161111 | 0.687750228 | −0.2047248% |
| C2, 1280/1536 prompt tokens + 32 generated per owner | 1.233151416 | 1.060144664 | −14.0296439% |

This native old/new/new/old bookend has two paid observations per policy for
each workload, with exact generated IDs/usage within and across arms. Every warm/paid request
recomputes the cold literal prompt (zero cached tokens) and publishes full
score-bearing rows (zero GPU-greedy tokens), with exact likelihood lengths
and finite values. Both new arms complete 27 joined-prefill groups/6912 rows.
The boundary is complete cold endpoint latency, not isolated prefill gain or
stock HTTP parity; the separate runner RNNR above measures the matched
prefill/decode boundary. All four endpoint processes retire normally.

The final Spark A `m35-gemma3-joint-final` job completes all nine steps with
no skips. Its runtime SHA-256 is
`8683588f6c16046b947df565609d5f3f54942e19f7641ed52deec22ccb783daf`;
actual source/binary/receipt and HTTP aggregates remain external under
`joint-prefill-composed/` and `joint-prefill-http/`. No full suite was run.
Wider cohorts, broader context/memory/reclaim/swap and sustained qualification
remain open.

At this optimization's entry, TensorFold HEAD was rechecked as
[`041d14a94e951834470fd514ed33e65b8be1059a`](https://github.com/ashhart/TensorFold/blob/041d14a94e951834470fd514ed33e65b8be1059a/README.md).
Its native Zig recipes document no matching Gemma3 CUDA/GGUF target, so it is
ineligible for this comparator screen. Earlier pinned observations above
remain historical.

## Copy-free bounded owner reads — 2026-10-07

The bounded two-owner recipe now reads independent cache roots without
materializing the shorter full K/V prefix. One short same-binary causal screen
reduces native paid latency from 692.1795 to 652.215 ms (−5.77372%, n=2),
against 654.1425 ms for the current original stock comparator (−0.29466%).
All six arms retain identical 64 natural choices and both final full heads.
This is representative short C2 evidence at context 4096, not sustained,
endpoint, wider-cohort or maximum-context qualification.

The checked transfer admits no-softcap D256/H8/C2, with aligned independent
actual roots and the original common logical width, stream-K partition,
query precision and floating tile/reduction arithmetic. Nonempty partitions
stop at each actual cache bound; wholly absent partitions publish neutral
metadata and zero numerators without preloading absent tiles. Sinks, sparse
inputs and unsupported layouts refuse. The exact bounded specialization has
its own occupancy query while the original specialization determines logical
grid and scratch. Equal-width/legacy paths and internal false defaults remain;
the existing bounded serving factory explicitly selects the policy. Multirow
packed prefill, state descriptors and checkpoint layout are unchanged.

Four cap0 operand configurations cover 512/1024 and 256/1536 roots in both
owner orders, poisoned invisible tails and scratch, both empty-partition
metadata banks, FP64 error and fresh captured replays. They are byte exact to
the padded original MMA oracle; maximum NMSE is 6.11982e−7 against the existing
5e−4 bound. The representative grid retains columns4, 48 blocks, mask prepass
and 399616 scratch bytes, with bounded occupancy1. The focused Spark B check
passes all 22 cases, including cap50, legacy no-cap and D256/D512 controls.

Native padded/bounded own runs freeze identical full heads, choices, final
heads and both initialized states before stock is read. Device/full-head
choices, eager/capture, Clear, spill/restore, refusal before mutation and
partial departure remain exact. The first representative model screen uses
256/768 prefixes; the separate adoption screen wraps the 1280-cell local ring
with 1280/1536 prefixes. Its three supplied warm writes use local indices 0–2
and 256–258; global read widths differ while both local widths remain 1280.
The state cursors finish at 1319/1575 after joined and departure controls.

| Prefix positions | Exact stock heads / 74 | Strict differences / 72 targets | Relative target-loss delta | Mean TV |
| --- | --- | --- | --- | --- |
| 256 / 768 | 73 | 0 | +8.8776068e−10 | 1.0353239e−9 |
| 1280 / 1536, wrapped local rings | 70 | 0 | −6.2739432e−9 | 7.6641055e−9 |

Both stock teachers repeat exactly. These unmatched raw heads remain explicit;
this is not complete stock byte equality at every geometry. Padded and bounded
native policies remain byte exact in both screens. Bound-plan counts in the
wrapped own control include 10 bounded-owner and 165 packed-prefill nodes,
separate from 18 successful joined-prefill groups/4608 processed rows.

| Short paid C2 cycle, mean of two processes | Prefill ms | Decode ms / 32 steps | Total ms |
| --- | --- | --- | --- |
| Original stock backend greedy | 190.825 | 463.3175 | 654.1425 |
| Native padded roots | 200.4145 | 491.765 | 692.1795 |
| Native bounded roots | 200.5645 | 451.6505 | 652.2150 |

The order is stock/padded/bounded/bounded/padded/stock, with one frozen native
binary and only the explicit policy argument changed. Warm work is the same
prefix plus three supplied rows and eight greedy rows, followed by logical
Clear off clock. Paid work includes compatible prefill, 32 joined decode steps
and final full-head publication on both engines; three supplied warm rows are
off clock. Stock retains ordinary state-only overlap and its original full
sampled-logit host transfers. Its two bookends are 649.278/659.007 ms, so this
small screen carries clock/startup uncertainty. The causal policy saving
includes copy, launch and storage effects; it is not a subtraction of the
instrumented copy-kernel sum from an earlier Gemma2 trace. No new trace or
performance grid was run. The public stock caller now uses ordinary logical
resetfalse; historical true-reset comparisons above are preserved, and both
reset variants are off clock.

The adopted runtime passes four HTTP aggregate cases: literal ring/repeat/
likelihood/bounds plus simultaneous unequal 256/768×32 requests, chat/cache/
SSE/stops, queued peer completion after disconnect, and kept checkpoint/restart
replay. Both epochs exit 0; matched cached scalar and joined responses remain
exact. The first drained epoch records 34 bounded-owner plan selections,
196 successful joined decode groups/392 units and 7 joined-prefill groups/
362 processed rows. These are bound/completed counters, not inferred kernel
replay counts. Client reads may be buffered; no precise backend cancellation
or endpoint timing claim follows. The first HTTP attempt is retained as FAIL:
its unequal literal requests were serial and concurrent short requests stayed
in equal cache buckets, so the required bounded counter was 0. A separately
hashed helper-only retry added the actual simultaneous unequal pair; the
runtime and all compiled sources stayed unchanged. Four focused adoption jobs
pass after the earlier build/operator, model and timing jobs; the failed
HTTP attempt remains additional preserved evidence. No full suite was run.

The actual native probe SHA-256 is
`3078e9d8eadc8330139504dfc4f17932140eef337c8bf1c3b333434d45a1961a`;
the adopted runtime is
`5868d608f59bb6076027d1e3ce74a70e8d7fb949a3fba7d86fe7615e0d202067`.
The build receipt is
`874aaf7a5967cfbe91054e0d8fc1a0630f952e0e8eb54b831d09719f1e08ce89`;
a complete checksum source inventory binds the actual build independently of
its unknown Git version. Original source/artifact, SDK and d812 image pins
above are unchanged. The approved source and prepared payload are fully
hashed and deep verified before inference. TensorFold was rechecked for this
task at 041d14a94e951834470fd514ed33e65b8be1059a: its documented CUDA recipes
still offer no matching Gemma3 GGUF comparator.

Reproduce own/teacher/cycle with `jitllm_gemma3_joint_prefill_probe` and
`llama_joint_prefill_probe.cc` at the supplied input geometry; the native
optional `bounded-roots` argument selects the candidate. The durable
`analyze_bounded_roots.py` prepares authenticated ring inputs and gates native
own freezes and strict quality. Complete raw inputs, responses, state/head
payloads, source/binary receipts, official job outcomes and aggregates remain
external under Spark B `scratch/m35-gemma3-owner-bounded-roots/run1/` and
`run2/`, with local copies under `/tmp/jitllm-m35-coordination/`.


## Internal 8K scalar depth screen (2026-10-07)

The explicit `depth` mode of `jitllm_gemma3_probe` and
`llama_fair_probe.cc` executes one slot at context 8448, with 8192 actual
prompt tokens, 128-row chunks and 64 continuation tokens. The approved source,
artifact, tokenizer, ordinary optimized arithmetic and original d812 image
above are unchanged. Preparation repeats the authenticated 2276-byte source
text 32 times, tokenizes it independently in both engines and freezes 8256 IDs
including BOS. This is one repeated-text depth screen, not broad long-context
quality qualification. At that internal-screen snapshot, ordinary serving
still admitted context <=4096; the later public scalar gate is below.

Twelve focused host controls pass, including 8448 frontier/read bounds, local
ring writes, causal padded masks, initialized checkpoint ranges and refusal
beyond capacity. Two native own traversals repeat all 65 finite heads, 64
choices and final initialized state exactly. The full-head and device-greedy
paths agree. Malformed publication and over-context growth leave prefix
metadata and initialized bytes unchanged. A separate lifetime traversal
checks state-only versus full-head prefill, Clear reuse, exact 8192-position
state through spill/restore and exact 64-token continuation. Its local cache
has 1280 physical cells; the first decode writes cell 512 after multiple wraps.
It retains 63 captured graphs across spill/restore, finishes with 64 graphs and
records zero coverage violations and zero host overcharges. Two allocated
slots fund the own refusal control; only slot 0 executes throughout this screen.

Stock teachers repeat exactly. Native matches 63/65 complete stock heads and
all 65 greedy choices, with zero positive-margin or tie differences. The 64
scored targets have mean target-NLL delta −1.81517024e−11 and relative
conditional-loss delta −1.81517024e−11. Mean/max total variation are
2.07715415e−11/9.86876055e−10; maximum raw-logit difference is 0.049402714.
The final 65th row is unscored. Mean target NLL is about 1.47e−6 in both
engines: these repeated targets are highly predictable. This establishes
boundary, shape and state evidence; full-corpus and depth-retrieval quality
remain open. Nonexact heads remain explicit; this is no
full-perplexity or complete stock-byte-equality claim.

| Paid C1 depth cycle, mean of two processes | Prefill ms | Decode ms / 64 steps | Total ms |
| --- | --- | --- | --- |
| Original stock backend greedy | 1550.310 | 854.7165 | 2405.0265 |
| Native optimized | 1573.010 | 861.7805 | 2434.7905 |

The stock/native/native/stock bookend has two paid observations per engine.
Native paid latency is 1.2375747% higher: prefill +1.4642233%, decode +0.8264729%.
Both arms first warm the same full prefix and eight greedy rows, then clear
logical state off clock while retaining backing/plans. Paid work includes the
8192-token frontier, 64 greedy steps and the final full head. All four natural
histories and final heads are byte exact. Stock retains natural state-only
overlap and its ordinary full sampled-logit host transfers; native publishes
GPU decisions except the paid final verification head. This bounded cost
screen establishes no parity, sustained-load or endpoint-performance result.

The standalone probe has no serving plan reclaimer. Its depth-only diagnostic
cap explicitly funds fixed catalog occupancy, weights, one full registered
state capacity, host floor, one extent of charge rounding and a conservative
finite plan/graph allowance. There are at most 67 keys: 32 aligned global-read
buckets for each of full-head and state-only prefill, one GPU frontier and two
scalar publication modes. Setup's maximum plan charge bounds node count via
`kPlanNodeHostBytes`; `kGraphNodeHostBytes` then bounds each captured graph.
Actual key/graph counts, combined bytes and zero overcharges are checked at
completion. The C1 lifetime cap is 6,090,039,296 bytes, while its actual retained
plan/graph charge is 1,079,693,904 bytes and full state layout 432,013,312 bytes.
This allowance is a synthetic cap, not an allocation or measured physical peak.

Process-lifetime MemAvailable sampling at 20 ms observes native cycle deltas
4.051–4.159 GB and stock deltas 4.496–4.534 GB. It includes model load, warmup,
paid execution, output and retirement plus unified host/device/page-cache
effects. It is neither catalog occupancy nor isolated paid inference peak,
and excludes a swap-table whole-state snapshot. Cross-model exact-state swap,
public 8K admission, wider cohorts and maximum-context memory gates remain owed.

The first quality job (at lifetime) and first cycle job are retained as FAIL: their old
synthetic caps could not fund state restore/growth alongside retained plans.
An intermediate quality pass retired obsolete plans off clock; that separate
evidence does not prove graph retention. The final source replaces that
workaround with the explicit finite allowance, reruns both own traversals and
the lifetime/stock quality gate, and preserves graphs through restore. No
production reclaimer or arithmetic changed; historical default probe budgets
are unchanged. Official final jobs `m35-gemma3-depth-quality3` (13 steps) and
`m35-gemma3-depth-cycle2` (6 steps) finish DONE0. The unchanged 12 host cases
were executed in the preserved first quality job; no full suite was run.

The actual native probe SHA-256 is
`b90a26b3d9646e8c4f6cff22eeb1c477001cca03c043ee031b3563c01c27211e`;
its build receipt is
`0296e41b77f3db5f50dfc06b68ebd87906a683d45350ea2d5586f929c8333ecd`.
The executed fa6defd-based source inventory is bound to those actual bytes.
Final composition on 21c64d6 preserves its independent internal C4 additions;
all five owned source files are byte identical to the executed depth source.
CUDA 13.4 cuBLAS/Lt resolve to the authenticated SDK c09 closure explicitly.
TensorFold was refreshed at task entry to 041d14a94e951834470fd514ed33e65b8be1059a;
its documented CUDA recipes provide no matching approved Gemma3 GGUF comparator.
Raw inputs, heads, state hashes, source/binary receipts, failed and completed
supervisor jobs and aggregates remain outside Git under Spark A
`~/.local/share/jitllm/gemma-context-depth{1,2,3}` and local
`/tmp/jitllm-m35-coordination/gemma-context-depth-result{1,2,3}`.


## Public scalar 8K and one model-switch pair (2026-10-07)

Ordinary Gemma3 serving now accepts an explicit context up to 8448 with one
resolved request slot. The default stays 4096/128/1; two slots remain admitted
only at context<=4096. The scalar restriction is checked after calibration and
explicit overrides, including an override that replaces a calibrated slot
count. No model arithmetic, artifact format or public schema changes.

Four focused settings cases pass, including the new default/calibration/override
control (the filter also selects two existing Gemma31 cases). The actual HTTP
runtime executes the same authenticated repeated-text 8192-token prefix and 64
natural tokens as the prior depth cycle. All choices, finite target scores,
complete top-logprob rows and usage repeat exactly in three literal requests;
`cached_tokens` is zero, so each request pays full prefill. An over-context
request refuses and a healthy request then succeeds. Both HTTP cases and clean
runtime retirement pass at context 8448/slots1. These are boundary controls,
not corpus/retrieval quality or endpoint-performance measurements.

One existing `swap-table` pair switches Gemma3 to the approved Qwen3.8 NVFP4
artifact and back, with 8192 saved Gemma3 tokens and 64 continuation tokens,
two cycles and zero-context controls. Both saved-context returns compare the
initialized-state snapshot SHA-256 and every continuation token/full head to
the unswapped reference exactly. The prepared return keeps 32 graphs and
replays 63 times. The other rows' default state flags are not additional
saved-state proofs.

| Handoff to first output | First use, saved 8192 | Prepared, saved 8192 | Prepared, zero context |
| --- | ---: | ---: | ---: |
| Gemma3 → Qwen3.8 | 10.007763 s | 5.977315 s | 5.958623 s |
| Qwen3.8 → Gemma3 | 1.598893 s | 1.589650 s | 1.640332 s |

These six totals exclude diagnostic state hashing and end at the first output;
all 64 continuation rows are checked separately. They are one pair's completed
observations, not a sustained swap ladder or a new throughput result. The worst
prepared handoff is 5.977315 s against the existing 20 s bound. Qwen uses the
existing approved external tokenizer/template files explicitly; no runtime
metadata-discovery fallback was added.

The shared execution budget is 117,514,120,988 bytes, with 2,122,526,492 fixed;
the diagnostic whole-state snapshot is charged within that budget. The largest
row `peak_bytes` is the 20 ms sampled whole-node MemAvailable decrease from the
process-start baseline: 81,822,568,448 bytes, exactly the difference between
124,897,492,992 at start and 43,074,924,544 at the lowest sample. It is neither
catalog occupancy nor an isolated process/inference allocation peak, and does
not isolate the snapshot's cost. Only the approved 2,526,350,862-byte prepared Gemma3 directory is
copied into the existing M3 artifact store; the approximately 104 GB Qwen
artifact remains in place. The streamed copy authenticates all four payloads,
uses the runtime's private-group/no-ACL path policy, exclusive private files,
held directory FDs and atomic no-replace publication. Originals remain intact.

The actual runtime SHA-256 is
`72ede5c8ab239c7b8b3378fb95fd92ffe41fc517ed94902b3ce79e84e7fbdd46`;
its executed source/library binding is
`00cecae3912d55d7705e5caa01a58c9980ee4b595ea59333003b35db040c9f95`.
The source was built on the 20e0990-based public tree; final composition on
af71732 preserves the independent internal C3 changes and all six executed
public source files byte for byte. CUDA 13.4 cuBLAS/Lt resolve to the same
pinned SDK c09 closure. TensorFold task-entry HEAD remains
041d14a94e951834470fd514ed33e65b8be1059a, with no documented matching approved
Gemma3 GGUF CUDA recipe.

The first supervisor failure refused group-writable artifact ancestors before
copying. The second passed copy/build/settings/HTTP but stopped Table bootstrap
because Qwen metadata paths were omitted. Both are retained; neither produced
swap inference. The final Table-only retry adds approved explicit paths and
reuses the successful runtime/HTTP evidence. Official jobs
`m35-gemma3-public-8k-swap2` (steps 1–6 PASS, step7 bootstrap FAIL) and
`m35-gemma3-public-8k-swap3` (3 steps DONE0) retain the actual checks. Raw data,
receipts and failures stay outside Git under Spark A
`~/.local/share/jitllm/gemma3-public-8k-swap{1,2,3}` and local
`/tmp/jitllm-m35-coordination/gemma3-public-8k-result{1,2,3}`.
Maximum context, broader cohorts/model pairs, real-corpus retrieval and sustained
memory/swap qualification remain open; the repeated-text 8K screen closes none
of those gates.
