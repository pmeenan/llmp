<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Shared spatial programs and image generation

The first private Spark A screen (2026-10-12) executes existing F32 VAE weights through the shared typed graph, planner, GGML lowering, StepProgram and StepRuns completion/capture path. It uses no image-family dispatcher, fabricated token shape or mandatory BF16 intermediate rounding. Main integration also passes (see below).

The source mapper consumes two serial spans from the existing VAE plan: the actual 64×64 post-quantization projection and 64→1152-channel decoder input convolution; and the final 288-channel nearest2/resample span at explicitly reduced 4×4→8×8 geometry. Inputs are deterministic synthetic F32 values. Six stored F32 weight/bias parts total 5,662,592 bytes; only their five physical 2 MiB pages are materialized under ordinary catalog coverage. All virtual reservations and runtime allocations remain owned through completed retirement.

The private focused native build and 26 host controls pass (two spatial, eight lowering, sixteen scorer). The actual VAE target compiles and its one case passes in 2.700 seconds (supervised job four seconds). All 4,737,024 output values are finite and compared with the catalog's double-precision CPU reference. The reported bounds were derived from FP32 source arithmetic before execution; they are local arithmetic enclosures, not image-quality thresholds.

| Span | Complete output values | Maximum absolute error | RMS error | Predeclared enclosure |
| --- | ---: | ---: | ---: | ---: |
| Actual decoder prefix | 4,718,592 | 1.1809659108408255e-6 | 9.087178159658703e-8 | 0.0018622070022278321 |
| Reduced nearest/resample | 18,432 | 1.3665311648480838e-5 | 1.3866630128825131e-6 | 0.05607674926568447 |

Actual bound implementations are `conv2d.f32 → add → conv2d.f32 → add` and `upsample.nearest → conv2d.f32 → add`. Each span completes eager execution, capture alongside execution, and recorded replay. Both replay outputs match their own selected program's full output hash; there is no cross-implementation bit-equality gate. Totals are two eager, two capture and two replay paths with no refusal. Source/recipe/binary guards pass before and after; the child process group and GPU processes retire, and all confinement limit/OOM events remain zero. Observed process/cache peak is 293,462,016 bytes under the 4 GiB cap. The separate native domain is 1 GiB, including its prepaid 512 MiB metadata allowance.

Single elapsed observations from the private screen around completed `PagedNode::Job` are below. They include scheduled input/output copies and completion (and capture in that column), but exclude initial source loading and CPU reference construction. These are descriptive first-use measurements, not a matched speed comparison or migration gate.

| Span | Eager | Capture | Replay |
| --- | ---: | ---: | ---: |
| Decoder prefix | 14.149741 ms | 11.586266 ms | 10.957196 ms |
| Reduced nearest/resample | 0.756765 ms | 0.329215 ms | 0.320430 ms |

This evidence covers the two stated spatial spans, full outputs, source formats and common execution lifetime. Whole VAE/flow/timestep composition, original image quality, alternative-kernel comparisons, pipeline serving and migration gates remain open. Direct F32 reuse is now executable; this screen does not select it as the fastest whole-pipeline policy.

Raw logs/XML and failed compile/harness attempts remain in session scratch. The aggregate binds source05, reviewed fixture02, focused04 and VAE build02, then the independently reviewed actual wrapper/models01. No additional inference was performed to write this report.

## Main integration

The main candidate based on `b8d156d0` passes a full locked native build and
129 affected host cases across 12 binaries in `image-execution-main-native02`.
The complete build corrected all 27 old token-constructor calls across five
existing tests; their arguments, defaults and assertions are preserved.
The host selection excludes the two actual GPU cases in the row-decoder binary.

`image-execution-main-models01` passes the same single actual VAE registration
in 2.693 seconds (five-second supervised job). All 4,737,024 values are finite
and within the same predeclared CPU-reference bounds, with identical source-part
and output hashes. Both spans execute eager, capture and replay with no refusal.
The cgroup process/cache peak is 292,986,880 bytes; this is not total GPU
residency. All source/binary/recipe guards, memory events and complete retirement
checks pass. Final format, boundary, header and REUSE checks pass.

Main native recipe SHA-256:
`728f3d4394f76c6cfcd168bd93e9d40b3c8ce85e34be6f0d06b44d116515b242`.
Main actual recipe manifest SHA-256:
`4b3c6ebde23fd66644d8c94d810bf60d1a57a280a2e65bb507b95a77057eef15`.
Actual binary SHA-256:
`199dc557652f8d536b01fedf42db57f673f2382a75acab93a311776714e10bda`.

Production/test code is identical to this checked image-only source export;
final prose/aggregate additions record the result. The independently in-flight
DSpark factory is excluded from this commit. Whole-pipeline quality, speed and
serving remain open; the main repeat is not another timing qualification.

## Complete first-frame decoder

The next bounded source candidate consumes the existing `VaeDecoderPlan` and
all 128 actual F32 decoder parts through shared components and the native role
visitor. At a 2x2 latent, its 113 source steps become 109 components and produce
32x32x4 outputs. Four saved copies become backward SSA aliases. The graph reuses
convolution, nearest upsampling, clamped channel normalization, residual addition,
duplicate-up addition and explicit spatial attention through the existing owned
backend, program planner and completion/capture path. It adds no image-family
executor, token-shaped surrogate or forced BF16 intermediate-rounding policy.

The source parts contain 1,012,281,936 bytes. Their actual group-relative
intervals require 486 physical 2MiB pages (1,019,215,872 bytes); the six containing
groups reserve 647 pages of virtual space. The latter is not a claim of complete
residency. Whole-graph compiler metadata is prepaid at 9,248,768 bytes. Disabling
unused scalar-reference buffers retains that payment and all source checks,
while avoiding a full-model CPU payload oracle. The small source-reference
prefix retains its normal reference payment and checks the actual four source
parts independently.

The focused Spark B build compiles the four host targets, CUDA module and new
actual fixture; all 52 host controls pass without skips. The full main native
build then passes 227 host cases across 17 targets, with successful source/binary
guards and child retirement.

The single actual first-frame case passes on Spark B in 0.883 seconds including
setup and storage (two-second supervised job). All 4,096 whole-decoder outputs
are finite. The separate 4,608-value prefix has maximum absolute error
6.2162465908066622e-7 and RMS error 5.6679068707238546e-8, within its predeclared
0.0016099519558288599 F32 source-arithmetic enclosure. That enclosure applies
only to the local prefix, not the whole decoder. Each program completes eager
execution, capture and replay, with identical complete output hashes under its
own selected implementation sequence and no capture refusal.

The whole bound plan contains 46 F32 convolutions, 36 clamped-L2 operations,
36 scales, 36 multiplications, 35 unary operations, 68 additions, 92 contiguous
copies, two existing cuBLAS products, one softmax, four nearest upsamplings and
four repeats. Source ranks/roles remain intact; no legacy BF16 rounding gate is
introduced. All source/binary/recipe guards pass, memory limit/OOM events are
zero, the native process group retires completely and no GPU process remains.
The process/cache cgroup peak is 399,495,168 bytes; this is not device residency.
Final source checks pass on Spark A: formatting, portability boundaries,
SPDX headers and REUSE 6.2.0.

This is complete bounded first-frame execution and same-selected replay evidence,
not a full 8.28-billion-MAC scalar oracle, original-image quality result,
comparative-speed result or serving admission. The fixture wall time is not a
latency gate. Whole flow/timestep composition, broader VAE geometries and the
family migration gate remain open. The compact
[actual aggregate](whole-vae.json) records counts, scope and provenance; raw
logs/XML remain in session scratch.

Source01 manifest SHA-256:
`5b440ee8c5e7809b82af14531e966157fb0e7dcf0b10c0ad472475afe95db1b4`.
Source02 correction manifest SHA-256:
`94e0c6c9237bb53cb01e8968ff2f714c0a4b931d09515a389a33aeffe6257e7c`.
Actual fixture01 manifest SHA-256:
`afa90908ca4711412516e09febd101c2070d702fb064c513c7318e786b515543`.

Main native receipt SHA-256:
`6b6437171c8579537b199c2903d938c85a6ec1c823e4a353e59775d7bab68f6b`.
Actual fixture binary SHA-256:
`ab5006ce7f027408874f59e80ffd818d991f4708600fc6ff1457ff7af419188a`
(187,960,048 bytes).

Actual recipe manifest SHA-256:
`cd495f76fe1ef7477354640a868dba380e74acf418a15ec462051bdf9bcc5940`.
Actual result SHA-256:
`86aa3600cf38ca111474352b1a49e6c9f253d08e2946c48b57c5815f21ccd147`.

## Complete timestep conditioner

`ImageTimestepSpec` composes four shared source-typed projections, with SiLU after
both timestep projections and explicit exports for modulation and final scale.
The prior single-output default is unchanged. The source's four BF16 matrices
contain 203,423,744 bytes. Their actual intervals lease 99 physical pages
(207,618,048 bytes); the containing global group reserves 130 virtual pages
(272,629,760 bytes). Resident weights retain BF16 encoding. The existing source
sinusoid creates two unequal conditioning rows at `[t, 0]`; its BF16 values widen
exactly to F32 at the input boundary. No legacy intermediate rounding is imposed.

Independent reduced host controls cover both branches, multiple owners and
changed inputs against the scalar reference, plus source-shaped metadata,
export identity, malformed geometry and full reference-payment refusal. The
main Spark B native build passes 443 host cases across 25 targets, including all
55 cases in the four directly affected targets. The actual source fixture then
passes in 0.444 seconds (one-second supervised job). This elapsed observation
includes setup and storage and is not a performance comparison.

All 40,960 output values are finite at each of five completed steps, totaling
204,800 checked values. Both first-row branches change when timestep1000 becomes
500; both zero-time rows remain unchanged. Restoring the original input restores
the complete original output hashes. Execution uses one eager call, one capture
and three replays of that retained recording, with no refusal. The bound plan
contains four `ggml.mul_mat.mmf` products, two unary operations and eight contiguous
copies; the complete producer and both outputs are executed. These are bounded
finite-output, row-attribution and same-selected replay controls, not a full-size
reference quality comparison.

| Resource | Paid bound | Actual program observation |
| --- | ---: | ---: |
| Graph metadata | Within64MiB metadata allowance | 491,520 bytes |
| Activation | 2MiB | 294,912 bytes |
| Product pool | 2MiB | 0 bytes |
| Program host metadata | 2MiB | 51,536 bytes |

The separate compiler/owner prepayment is32MiB, pinned staging/output/baseline
storage1MiB, and managed node capacity512MiB. The invocation uses a2GiB zero-swap
process/cache ceiling,9GiB available-memory preflight and 6GiB physical margin.
Observed process/cache peak is227,135,488 bytes, not total device residency.
All pre/post source, recipe and binary guards pass; memory limit/OOM events are
zero, the whole child process group retires and no GPU process remains. Source
quality checks are recorded by the final integration separately.

Full flow composition, broader geometries, same-source reference image quality,
comparative speed, serving and the family migration gate remain open. A reference
implementation using the same source encoding is the relevant quality baseline;
unquantized results are only needed for closer/farther-from-original claims.
No such quality claim is made by this execution screen.

Provenance: `image-timestep-main01` result SHA-256
`88683f38cabcf4211fa3d680e55d97f86065365ca11ccf130421f85f1b4d75e6`;
`image-timestep-models01` recipe
`1848fcf2fbcbe697f670cd81ba3327bbaeb3a163c01b821dda668a1ad3c37c7a`
and result
`7c161f68e53225fcb204d974e56562af9b2a7a292a99e97c26c57e7e73b6bf66`.
Actual binary SHA-256
`c218d1392b7e71c69cc08fe2e51b7cd7e7c014834cd8687e0dcd1867c9bb40c2`
(187,663,232 bytes). Raw logs/XML and preserved failed compile/inventory attempts
remain in session scratch; the successful main receipt is used unchanged.

## Conditioned transformer block

The complete timestep conditioner and source-shaped transformer block 0 execute
in one shared typed graph. Modulation feeds both adaptive normalizations and
both gated residuals; the first carry is retained through the MLP. Three existing
NORMAL RoPE operations per Q/K use source axis widths 16/56/56, including negative
positions. Explicit F32 attention preserves the source's causal-text/all-image
mask. Existing packed-F32 LayerNorm and learned RMS scaling fusion use the shared
GGML lowerer and physical validators. Weights remain BF16; no mandatory legacy
intermediate casts or new kernels are added.

Reduced independent host arithmetic covers the full conditioner/block formulas,
multiple owners, both exports, changed inputs and negative positions. Additional
controls cover malformed ports/geometry, complete source-shaped planning and
reference-payment refusal before allocation. Scoped Spark B compilation and all
63 affected host cases pass. The actual existing conditioner regression passes
204,800 finite values, and the new block passes 229,376 finite values across seven
steps: one eager execution, one capture and five replays, with no capture refusal.
Each step checks all 32,768 output values. Changed time affects both output
branches while zero-time scale and masked text remain unchanged; changed image
input affects another image row, leaves text and independent final scale intact,
and restored input restores full output hashes. This uses a synthetic
preprojected six-row stream (two text/four image), not a complete image pipeline.

The actual block binds 13 BF16 parts containing 639,631,872 source bytes. Their
closure leases 308 pages (645,922,816 bytes); the containing groups reserve 339
virtual pages (710,934,528 bytes). The actual plan has 211 graph nodes/tensors and
13 products, two LayerNorms, two fused learned RMS scalings, six rotations,
one softmax and one SwiGLU. Both complete outputs are executed.

| Resource | Paid bound | Observation or derived bound |
| --- | ---: | ---: |
| Graph metadata | Within 96MiB host allowance | 2,912,256 bytes |
| Compiler construction | 64MiB within that allowance | 48,085,088-byte source-derived bound |
| Activation | 8MiB | 1,114,112 bytes |
| Product pool | 8MiB | 0 bytes |
| Program host metadata | 4MiB | 229,392 bytes |

The fixed managed node is 1GiB; pinned staging/output/reference-hash storage is 1MiB.
Compiler/reference payment is established before construction; unused full-size
scalar weight buffers are omitted. The process/cache ceiling is 2GiB with zero
swap, 9GiB preflight headroom and 6GiB physical margin. Observed process/cache peaks
are 399,900,672 bytes for the block and 227,995,648 bytes for the conditioner,
not total device occupancy. All source/binary/recipe guards pass, memory events
and remaining compute PIDs are zero, and each whole child process group retires.
The observed fixture durations 0.618s/0.435s include setup/storage and are not a
performance comparison. The full main Spark B native build and 452 host cases
across 25 targets pass in the 127-second supervised build/check job, with no
failures or skips and the same model binary used by both actual cases. Final
source quality checks pass: format25, boundaries21, headers2,245 and REUSE6.2.0.
Failed earlier compile/host attempts are preserved externally.

This qualifies one bounded complete block's finite execution and same-selected
capture/replay, with reduced independent reference evidence. Full 32-block flow,
broader geometry, a full-size same-source reference quality comparison, end-to-end
speed, `EngineImage` admission and the family migration gate remain open.

Provenance: main01 result
`b7597ce07a8d60949b82a3afc6e90d2d1cfc2cdf1d6cfb6f45be4cc321538f1a`;
focused03 result `a687ab1319ea59ce24bfd56f338f125d14cf63e6d6615faf113d7ad0203e69e0`;
block result `7de24f20a05a3f9b221033e5794c887961b174069739faefcb8bfe5651cc22fb`;
conditioner result `e9b65e31c67548ae1472dfcdb3a04347554d9bd3c717a96c91b0ccbc41d4e1f4`.
Actual model binary `11d63cadeb6fd1952146721c4b519cceda5396c6ca23b51ba124932dc1aa83de`
(188,396,320 bytes). Raw XML/logs and sealed recipes remain in session scratch.

## Complete denoiser forward and Euler step

One shared graph consumes two 4,096-channel text rows and four 64-channel image
latent rows through all 32 source blocks. Text uses the stored zero-centered RMS
gamma and two projections with GELU; image rows enter through their source
projection. The qualified conditioner and block formulas are reused. Final
normalization takes the first timestep scale row, the source output projection
produces noise, and a second port returns `latent + dt * noise`. The source-owned
mapping has 563 components and 297 BF16 roles; no family executor, new kernel or
mandatory intermediate rounding is introduced.

Independent reduced double equations cover the complete two-block forward and
Euler step for two owners, changed text/image/time, zero step size and restoration.
Full source-shaped planning preserves all 296 products, 192 NORMAL rotations,
32 softmax and SwiGLU operations, 65 LayerNorms and 65 learned-RMS fusions. The four
affected targets pass all 66 host controls. Full main Spark B native compilation
and all 455 host cases across 25 targets pass in the 61-second supervised job,
with no failures or skips.

All three actual cases pass on Spark B: the retained conditioner and single-block
regressions, and the complete flow. The flow completes seven steps (one eager,
one capture and five replays), checking all 512 output values each time: 3,584
finite values and 1,792 Euler contributions. Changed time and image/position
inputs affect both branches; restored inputs reproduce the same selected program's
outputs. Its graph has 4,651 nodes and tensors. The native case reports 4.812 seconds;
this is an execution observation, not comparative performance evidence.

All source weights remain BF16: 14,230,249,472 bytes in 6,818 selected pages
(14,298,382,336 bytes), also the full virtual group reservation. No decoded
full-model CPU oracle is allocated.

| Resource | Paid bound | Source-derived bound or observation |
| --- | ---: | ---: |
| Graph metadata | 66,072,576-byte ceiling within 1GiB allowance | 65,253,376 bytes charged |
| Source mapping | Separately charged within that allowance | 2,675,712 bytes |
| Compiler/owner construction | 768MiB within that allowance | 624,475,488-byte conservative bound including reader scratch |
| Activation | 32MiB | 1,410,816 bytes |
| Product pool | 32MiB | 0 bytes |
| Program host metadata | 16MiB | 5,161,648 bytes |

The fixed managed domain is 16GiB; measured programs fit before node admission.
The invocation uses a 2GiB zero-swap process/cache ceiling, 24GiB physical
preflight and 6GiB margin. Peak process/cache occupancy is 497,651,712 bytes;
this is not device peak occupancy. All pre/post source, binary, recipe and
artifact guards pass; memory-limit/OOM events and remaining compute PIDs are zero,
and every child process group retires. The complete source/program/capture owner
remains retained until successful completion and teardown.

The metadata host control derives 5,545 primitive slots (32 × 169 plus 137
boundary slots), verifies the actual charge and exercises underpayment refusal.
The fixture treats its metadata allowance as a ceiling. An earlier setup-only
attempt incorrectly required equality to an overestimated charge and refused
before flow execution; its raw failure remains in scratch. The previous wrapper's
stale host-count refusal is also retained. No budget or numerical policy was
increased to pass either check.

The existing reference admission formula conservatively charges 64 times
aggregate element count: source elements alone yield a 455,367,983,104-byte bound,
and the 128MiB host control refuses before decoded copies. This is not measured
decoded storage; one raw F32 source copy would be 28,460,498,944 bytes.

This bounded screen does not qualify realistic image geometry, full-source image
quality against a same-encoding reference, comparative speed, repeated diffusion
sampling, text encoder/VAE composition or serving. Final formatting, portability
boundaries, header and REUSE checks pass on Spark A; raw logs stay outside Git.

Provenance: main02 result
`6eacb329f05944397a5079121169234bbb6ffa2930ef6d2f8339300d925d78e8`;
models04 flow result
`318f830d32de932fa70723e020ae9c7d58c1c779f477f9a6a4775c710c5bf721`.
Actual model binary
`1eb54e128b9209a09b040e49009f0b22b5162c1dd2ff5b367087cc1a032604be`
(188,854,360 bytes). The literal main result and all three raw XML/log/guard
receipts remain in the sealed session packages.


## Complete shared image generation

The controlled 1024×1024 path now executes the actual 36-layer text encoder,
two complete 32-block flow/Euler steps and the whole first-frame VAE. The fixed
native prompt has 39 tokens, with 25 retained text rows; the flow has 4,096
latent rows and 4,121 joint rows. Source weights remain BF16 for text/flow and
F32 for the decoder. The source factory reuses shared components and the owned
GGML backend; it introduces no family dispatcher or mandatory intermediate casts.

`ProgramLifecycle` is consumed by both language and image adapters for measured
program bounds, binding/coverage, plan accounting and retained execution.
Token `SlotLifecycle` and cache keys are unchanged. Image stage ports transfer
on device inside the current leased job, after execution and before completed
handoff. Source, workspace, ports and plans remain owned through completion;
unknown work prevents continuation and reclamation. Five fake lifecycle controls
cover completed reuse, prelaunch refusal/retry, null source refusal and unknown
completion retention, including positioning on every noncapturable eager run.

Full main Spark B native compilation and all 197 host controls across eight
related targets pass in the 76-second supervised job, with no failures or skips.
The real source case passes in 125.073 seconds, including setup and both complete
generations; this is not a matched performance measurement. Eight stages complete,
and every one of the 4,194,304 decoded F32 values per generation is finite before
postprocessing. The second generation reuses retained typed plans/captures and
reproduces its own first-flow and final pixel digests. The final RGBA digest is
`6dd66c48649d75057a996f3dd7bed9a0151ffaa22beac81f942641b16b1fc4e1`;
it is a candidate-repeat witness, not a reference-quality target.

| Resource | Source-derived bound or actual observation |
| --- | ---: |
| Construction allowance, admitted before source parsing/building | 4,294,967,296 bytes |
| Derived construction categories, before the small owned backend declaration | 2,669,795,776 bytes |
| Selected source pages across all three stages | 30,530,338,816 bytes |
| Measured fixed occupancy, including construction allowance and stable ports/staging | 4,444,683,008 bytes |
| Measured maximum activation workspace | 6,027,214,848 bytes |
| Measured maximum product pool | 127,926,272 bytes |
| Retained program/capture floor | 89,800,704 bytes |
| Pinned staging, already included in fixed occupancy | 107,772,672 bytes |
| Complete admitted envelope, including two 2MiB spare extents | 41,224,157,952 bytes |

Only the current stage closure is leased; the full sum allows prior source pages
to remain cached. Virtual reservations are separate from physical residency.
The measured sum is checked before weight admission against the unchanged 48GiB
node cap and available physical memory plus a 6GiB margin. The invocation has a
52GiB zero-swap process/cache ceiling, 54GiB preflight and bounded logs/reports.
Observed process/cache peak is 899,457,024 bytes, not total device occupancy.
All source/recipe/binary/artifact guards pass, all memory-limit/OOM events and
remaining compute PIDs are zero, and the whole child process group retires.

The first actual attempt refused a valid odd-39 attention product during
measurement, before weight execution. The shared cuBLAS precheck had inherited
paired-kernel alignment and 32-bit byte-span restrictions. The corrected contract
keeps actual element alignment, int GEMM dimensions/leading dimensions, checked
64-bit object/offset bounds and exact scalar/vec4 conversion-grid limits; MMVF/MMF
pair restrictions and existing fusions are unchanged. Descriptor controls cover
every physical product in the complete text/flow/decoder plans, odd-stride GQA,
wide batches, conversions and refusal boundaries. The earlier failed receipt
remains external; no source padding or bit-forcing rounding was added.

This closes the bounded complete generation core, including actual text mapping
and repeated generation ownership. It does not qualify a full-size image against
the same-source reference, broader prompts/geometries/schedules, production
serving/swap behavior or the matched 1% migration gate. Legacy serving remains
in place. Final integration source-quality checks remain root's landing step.

Provenance: main02 result
`cc6863bfb7085e051b79160622a086d22b794836159f05e9f580d9dc378fcca3`;
models02 result `47f72105ea8be1670b1de3815f5b563f9597e7cf0c55b96464f2bbf8f7dec129`;
actual model `30d5681f89bea2fcbe6a0ce2d8b93f0e106f94d73f1002d5a69010878a20331a`
(192,179,136 bytes). Raw records and sealed invocation remain in session scratch.


## Two-step native migration quality screen

One prompt now has a direct completed-image comparison between the existing
`QwenImageRunner` and shared `EngineImage`, using the same BF16 text/flow and F32
VAE checkpoint, tokenizer, initial noise and two-step Euler schedule at 1024×1024.
This is the native migration baseline, not an independent upstream diffusers
reference. Each arm completes two generations and reproduces its own output pin.
No cross-arm bit equality is required.

The existing image-GGUF comparator scores the completed lossless RGBA exports.
RGB PSNR is **46.438 dB**, luma SSIM **0.99873**, mean absolute RGB difference
**0.837/255** and maximum difference **6/255**. This passes the predeclared local
32 dB / 0.98 agreement screen. Pixel identity is false and diagnostic only.
Both exported two-step images are low-detail blurred forms. This checks
implementation agreement on that sample, not usable 40-step perceptual output
quality. The result covers this prompt and short schedule; independent upstream
same-source quality, broader prompts/schedules, serving/swap qualification and
the frozen 40-step matched 1% family gate remain open.

| Descriptive quality-run interval (seconds) | Legacy | Shared |
| --- | ---: | ---: |
| Setup | 0.838428 | 0.884807 |
| First generation | 12.046084 | 68.598573 |
| Warm generation | 2.069708 | 55.972123 |
| Warm first-output boundary | 0.878826 | 4.890541 |
| Warm remainder through final output | 1.190882 | 51.081582 |

These are ordinary counters from quality runs, **not matched timing samples**:
legacy Finish includes its private file write/sync, while shared Finish includes
the observation copy and writes the file after the recorded interval. The
first-output boundary includes text encoding and the first flow step; the
remainder includes the second flow step and decoder, so it is not an isolated
VAE measurement. Legacy records no denoising graph at two steps (capture begins
at the third step); shared retained execution is deliberately allowed to differ.
The large shared remainder identifies useful follow-up work on existing shared
operation implementations; no ABBA, unmatched timing rerun, kernel selection or
performance adoption is part of this slice.

The single driver reuses both existing adapters and their complete ownership and
retirement paths. Explicit legacy `out` now writes its already-completed RGBA
buffer as `image.rgba8` in an existing private directory, atomically with mode
0600. Empty `out` writes nothing. The old serving and swap harness assignments
were removed so their default timing path remains free of image exports. Shared
quality observation uses 20 MiB of the already-held 4 GiB construction allowance;
actual local usage including the existing owned-backend declaration is
2,691,659,919 bytes. No second construction reservation or resident reference
weight copy was added.

Accounting review distinguishes legacy source reads (30,718,271,488 bytes) from
14,719 rounded pages (30,867,980,288 bytes). The original recorded post-Setup sum
38,102,882,048 was read-based; replacing reads with capacity gives
**38,252,590,848 bytes**. The original preflight 38,248,397,072 omitted the opened
4 MiB landing zone while adding 528 bytes of allowance overhead, understating
that reconciled bound by 4,193,776 bytes. The fixture correction uses actual
opening catalog occupancy and retained rounded page capacity in both admission
and the memory guard. Original output receipts remain from the earlier exact
binary; they are not relabelled as a run of the corrected accounting. The actual
node scheduler/48 GiB cap remained enforced, and the reconciled envelope remains
below that cap and the available physical memory with the 6 GiB margin. Shared
admission was 41,224,157,952 bytes. These are accounting envelopes, not measured
device peaks.

The scoped Spark B checks pass all 37 affected host cases with no skips; the
changed swap target and actual diagnostic compile successfully. The two native
cases complete with every source/binary/artifact guard passing, zero memory-limit
or OOM events, zero remaining compute PIDs and positive whole-process-group
retirement. Process/cache peak is 928,067,584 bytes, not device occupancy.
Postprocessing reuses the existing comparator in a networkless CPU-only 1 GiB
container with zero extra swap; it exits successfully and is removed. Earlier
recipe-only inventory initialization and container-entrypoint failures remain
preserved outside Git. Root integration passes the full Spark-native build and
all 37 affected host cases on the corrected accounting source, with both model
registrations inventoried and no additional model execution. Pinned format,
boundary, whole-source header and REUSE checks also pass; the numerical evidence
above remains attributed to its original executed binary.

Compact observations and own-arm pins are in [image-quality.json](image-quality.json).
Provenance: model result `4361f62501623c4a94dfd8bfecd77871731bece03b98e6edd5610f77d7425fa7`;
actual binary `0aa624d0845e58d18039c8d6a55c5b3b40dd925860ba543f2f978e6501b0cc37`
(198,195,200 bytes). Raw outputs, logs and invocation receipts remain in scratch.

## Shared pointwise convolution reuse

An output-disabled diagnostic using the same complete image graph moved the
first-output boundary after both flow steps. Its warm decoder job took
46.242 seconds, with 0.001605 seconds of host job time; this includes device
copies and output handoff, not an isolated convolution. The complete decoder
contains 46 convolutions: six 1×1 and forty 3×3. A derived full-column im2col
inventory would require up to 10.125GiB for one 3×3 operation, beyond the current
workspace envelope. No such allocation or new kernel was introduced.

Supported F32 1×1, zero-padding `conv.spatial` components now express their
existing algebra through shared reshape/transpose, two real contiguous copies,
MatMul and bias Add. Unequal width/height, batches and separate owners are
covered. BF16 source, padded and 3×3 cases keep the existing direct path. No
source weight conversion, hidden intermediate narrowing, backend-specific
selector or forced accumulation recipe was added. Existing product selection
and fusions remain available.

The simple static nine-node component ceiling adds 39,437,824 bytes to the
complete factory's derived compiler/graph allowance. Its 34,293,760-byte
conservative overhead versus a shape-dependent exact ceiling is below 1% of
the existing 4GiB construction credit; avoiding a new sizing callback was the
chosen simplicity tradeoff. Normal placement pays both contiguous activations.
The complete decoder's metadata bound is 12,263,424 bytes. The measured
whole-generation admission remains within the unchanged 48GiB cap.

A representative actual post-quantization source tensor (64 channels in/out,
64×64 spatial plane) compares the direct convolution with the emitted shared
product, including copies and completed-job retirement. Each arm runs once
eagerly, once while recording, then five retained replays; all 262,144 values
per execution are checked against a scalar F64 reference. Both arms satisfy the
predeclared 0.002646865 enclosure. Maximum errors are 2.7562e-7 for direct and
1.7918e-4 for the selected cuBLAS product; their outputs need not be identical.
The enclosure includes the selected implementation's product rounding, rather
than forcing historical intermediate bits.

| Completed representative job (µs) | Direct | Shared copies/product |
| --- | ---: | ---: |
| First eager | 3,976.067 | 46,508.030 |
| Capture | 264.304 | 183.104 |
| Five warm replays, median | 249.584 | 73.552 |
| Warm minimum–maximum | 246.608–253.472 | 65.776–73.808 |

The warm median falls **70.53%** on this shape, while the first eager execution
is slower. This sequential small screen is not a family timing qualification.
The observed implementation keys are direct convolution+Add versus
Cont+cuBLAS MatMul+Cont+Add; no speed claim excludes the layout copies.

The candidate then completes two full 1024×1024 two-step generations and
exports its own repeatable image. Against the retained native migration output,
the unchanged comparator reports **46.477 dB PSNR**, **0.99873 luma SSIM**,
**0.8331/255 mean absolute difference** and **6/255 maximum difference**, passing
the original 32dB / 0.98 thresholds. This is the same low-detail two-step sample;
it does not qualify usable 40-step output, independent upstream same-source
quality, broader prompts, serving/swap behavior or the 1% migration gate. No
legacy model rerun or old shared pixel pin was required.

Only after the quality screen passes, one output-disabled run uses the new
candidate's own qualified pixel pin. Setup takes0.933s, first generation64.877s
and warm generation53.164s. The prior output-disabled observation was66.279s
first /55.927s warm: the warm observation is4.94% lower. This is a descriptive
whole-generation comparison, not a matched1% result. Its first-output boundary
is after flow1 rather than after flow2 in the earlier diagnostic, so phase
intervals do not isolate the decoder's change. There is no exported file or
quality observation copy in this cost run. The remaining performance gap is
large; this bounded improvement does not qualify shared default image serving.

The quality run charges 2,731,097,743 bytes of construction, including its
20MiB observation credit, and admits 41,224,391,616 bytes of managed resources.
The source closure remains 30,530,338,816 bytes. The output-disabled run omits
the observation credit. Both are bounded by the existing 48GiB node, 4GiB
construction and 6GiB physical margin; the supervisor separately limits
process/cache memory to 52GiB with zero swap. These are accounting envelopes,
not measured total device occupancy.

Focused native builds and all 25 affected host controls pass. The representative
actual, complete-image quality and conditional output-disabled cases all retire
with source/binary/artifact guards passing, no memory-limit/OOM events and no
remaining compute PIDs. CPU scoring uses the existing comparator in a removed
networkless 1GiB container, retaining the original baseline and candidate
receipts separately. Full current-main Spark B native build and all62 affected
host controls pass, along with six-file formatting, boundaries,2256 headers
and REUSE6.2. Final generation and representative binaries are byte-identical
to the executed candidates, so their actual evidence carries without rerunning.
The main receipt is`2f16ce1cb2f06bd1c9d5e3023ae04ad53db5d7e244c08db6b8282ea6f072f685`.
Raw outputs and receipts stay outside Git;
[pointwise-reuse.json](pointwise-reuse.json) records compact observations.

Provenance: representative result `e12403cae71c26fd1592b972fe60f2b09180d432702663d03e38d5bf7daaa546`;
quality result `0b93331d4bebcb5ab4d745e01a679b031433c5707b1acbf7594da849898ac4f7`;
cost result `e33abef2c16d590084da207bed7ce2b68128ec794837f804b1a3bfbf5756590f`.
Actual generation binary `63e739113f86b28c56bad548e1ced598872f6f056711e1533ef70caf8637e8f1`
(198,334,592 bytes); all model records remain bound to their executed source.


## Shared tiled F32 convolution reuse

The shared Conv2d implementation now has a bounded tiled alternative for packed
F32 3×3, stride-one, pad-one operands. It reuses the pinned GGML F32 im2col body,
llmpalooza's existing cuBLAS product, transpose and pitched output copy. Source
weights and activations stay F32. No new GPU kernel, forced intermediate
rounding or family executor was added. The physical planner and executor use
the same checked stripe plan; each stripe preserves the original image/channel
stride and has its own exact tail pitch and signed halo offset. Ordinary cuBLAS
callers retain their existing validation and workspace ownership.

The executed qualification used a default-off candidate with an immutable,
paid tuning record supplying thirteen source-derived shape keys. After it passed,
ordinary shared selection adopts those thirteen exact W/H/input-channel/output-
channel geometries for packed F32, batch1 on sm_121. They are measured evidence
entries, not model capacity limits; a differently shaped image with the same
flattened row count does not qualify. Other shapes/devices retain direct
convolution unless explicitly tuned. No model/profile-name supplier is needed.
Forty decoder convolutions use the tiled implementation. The initial 64MiB qualification workspace bounds columns,
product and transposed output together, rather than allocating the derived
10.125GiB full-column worst case. Complete decoder planning reports64,880,640
bytes of tiled pool, unchanged3,623,878,656-byte activation placement and7,706
actual launch calls. Capture accounting includes every stripe call and boundary
copy. These are planned bounds, not measured device peaks.

The independent representative screen uses the real F32 decoder input weights
with64 input and1,152 output channels. It covers the full7×5 scalar case and
128×128 case with45/45/38-row stripes, including both halo boundaries and the
last row. All outputs are finite,108 guard values per geometry remain intact,
and independent F64 checks cover40,320 small outputs or32,256 representative
samples per execution. Changed-input retained replay changes the output and
restoring the input restores each implementation's own pin. Both satisfy their
predeclared arithmetic enclosure; cross-implementation bits need not agree.
The representative pool is66,355,200 bytes, including all stripe buffers.

| Complete representative job (ms) | Direct | Tiled copies/product |
| --- | ---: | ---: |
| First eager |44.653300|5.565182|
| Five warm replays, median |44.619044|4.175491|
| Warm minimum–maximum |44.581924–44.786851|4.156595–4.176019|

The warm median is90.64% lower, including copies and completion. The small case
also improves warm median, but its first tiled eager run is46.496ms versus
4.467ms direct; startup is not hidden in the warm statistic. This local screen
is not a family speed qualification.

Two complete1024×1024 generations then pass the unchanged retained native
migration comparison at **46.495dB PSNR and0.99873 luma SSIM** (floors32dB/0.98).
Mean absolute RGB difference is0.8315/255, maximum6/255. The candidate owns its
new repeatable pixel pin; no legacy rerun or previous shared-pixel requirement
was imposed. This remains the low-detail two-step sample, not independent
upstream diffusers quality or usable40-step perceptual qualification.

After that pass, an output-disabled run takes0.928s to set up,21.660s for its
first complete generation and **12.459s warm**, versus the prior pointwise
candidate's64.877s first and53.164s warm. The warm observation is **76.57% lower**.
Both ordinary runs place the first-output boundary after flow1. These are
single-run descriptive complete-generation observations, not an isolated
convolution attribution, matched1% gate or production serving claim. Quality
runs include observation/I/O and are kept separate from these cost counters.

The actual managed admission is41,344,519,104 bytes under the unchanged48GiB
node cap. Construction uses2,731,256,290 bytes with quality observation or
2,710,284,770 without, under the existing4GiB allowance. The immutable tuning
preparation/ownership credit is155,765 bytes; source closure remains
30,530,338,816 bytes. The6GiB physical margin and separate52GiB zero-swap
process/cache ceiling remain enforced. Process/cache peaks are993,865,728 bytes
for quality and964,177,920 for cost, not total device occupancy.

Full current-main Spark B native build and all54 affected host controls pass,
with exact3 spatial/2 generation/1 whole-VAE registrations inventoried. These
qualification binaries match those executed. The later selector-only adoption
changes capability identity; it does not relabel historical binaries or plan
digests. A focused control compares default and explicit selection's operation
sequence, placement, tensor geometry and complete stripe partitions, including
all40 convolutions, pool and launch counts. Final current-main integration passes the full native build and55 affected
host controls in14s, including the ordinary-selection comparison. No additional
model execution is attributed to the selector-only binary. Formatting19 files,
16 boundary checks,2,260 headers and REUSE6.2 pass; the final added documentation
and aggregate receive the separate landing source-quality check. The earlier qualifying main check passes formatting19 files,16 boundary checks,
2,260 headers and REUSE6.2 pass. Representative, quality and cost runs retire
completely with source/binary/artifact guards passing, zero memory/OOM events
and zero remaining compute PIDs; the existing CPU comparator's confined
container also retires. Raw outputs and receipts remain outside Git;
[tiled-convolution.json](tiled-convolution.json) contains compact observations.

Provenance: main receipt`9f896ddaf1275b21ce1be0a7aefa2d14042fae73366178350a99cac536ff712b`;
quality`2210d7fedcbfb6f4f09b446c516a8beeaa97d80c5391ca811430885fa58fd7af`;
cost`dd7c735f9bbc56c2d980746b0a81141bfa14a2535dfdcec34fad0481ca3f2556`.
Generation binary`b44dd5bc61ce80bbb20048321609bbc666b649efcc0f51c523d61f4edabb598d`
(198,819,672 bytes). Broader same-source quality, serving/swap and the matched
family migration gate remain open.

Final selector integration receipt`b177ea323170e2eb02cedd565a87f83edbe973b38ba6717fe037f9a7b3bf0b7f`;
final generation binary`7259cb25c96ddb315f94cf39991a9afdd2c936a175c036e99a4b061641271b77`
(198,829,472 bytes). The comparison proves the same selected operations,
placement and physical stripe plans, not equality of historical capability
identities; numerical evidence remains bound to the executed b44dd binary above.


## Larger convolution workspace and remaining stage cost

A completed experiment raises the shared tiled target to1GiB of temporary workspace,
allocating only its checked stripe requirement. No kernel, source encoding,
precision policy or shape-selection rule changes. The same forty decoder
convolutions need134 rather than1,843 stripes:870 total decoder launch calls
instead of7,706. The actual planner reports1,073,577,984 bytes of raw pool and
unchanged3,623,878,656-byte activation placement. Explicit64MiB, underpaid,
batched-tail and1GiB partition host controls passed in the experiment; these targets
are implementation choices, not image-dimension limits.

Two complete1024²/two-step generations pass the same retained native migration
screen at **46.495dB PSNR /0.99873 SSIM**, with their own repeatable output.
The source remains BF16 text/flow and F32 VAE. This low-detail sample neither
qualifies usable40-step perceptual quality nor replaces an independent upstream
reference. No historical cross-version pixel equality is required.

The output-disabled observation moves FirstOutput after both flow steps, then
uses existing completed-job counters for the decoder. Setup is0.982s, first
complete generation21.936s and warm generation **12.435s**, versus12.459s in
the earlier64MiB ordinary observation. The0.194% difference is descriptive,
not a matched speed result: the boundary differs and there is no repeated A/B.
The warm prefix through text and both flows is9.653s; decoder completion is
2.782s, with2.752s inside its one completed device job. Host job submission is
5.758ms and dispatch0.288µs. These include the existing copies/handoff; they do
not isolate individual convolutions. No64MiB decoder CPU-counter comparison
was obtained, so fewer launches do not establish a measured CPU or bandwidth
improvement. The larger remaining interval is text/flow, directing subsequent
reuse work toward existing attention/product implementations.

The production default remains64MiB. The1GiB experiment preserves current
quality and reduces prepaid capture metadata by106.8125MiB, but complete
admission increases by1,102,249,984 bytes without a meaningful observed latency
gain or measured CPU/bandwidth improvement. Warm graph replay bypasses most
capture construction. Fewer structural calls alone do not establish a useful
resource benefit for this workload. More memory remains available to future
candidates when measurements justify it; the unchanged48GiB screen is not a
product memory ceiling.

The exact admission is **42,446,769,088 bytes**, not measured device occupancy.
The preliminary additive estimate42,353,216,448 was not an upper bound: it
omitted the runtime's maximum-stage pool and25% aligned workspace margin.
The new raw pool becomes1,342,177,280 reserved bytes after margin/2MiB rounding,
versus the prior whole-stage127,926,272. The1,214,251,008-byte pool increase,
less112,001,024 bytes of prepaid capture savings, reconciles exactly with the
prior41,344,519,104-byte total. Quality construction uses2,731,256,290 bytes;
output-disabled ordinary selection uses2,710,126,223, both under4GiB. The
30,530,338,816-byte selected source closure,6GiB physical margin and52GiB
zero-swap process/cache ceiling are unchanged. Process/cache peaks are
927,784,960 bytes for quality and898,191,360 for cost, not device peaks.

Full current-main native build and55 affected host controls pass in14s, with
all model registrations inventoried; formatting4 files,1 boundary check,
2,261 headers and REUSE6.2 pass. The executed generation binary is exactly the
experimental main binary. Final landing restores the three workspace/test
paths to their previously qualified64MiB versions; only the stale diagnostic
pixel assertion removal remains. Restored64MiB final main passes full native
build and55 host controls in28s, with all model inventories and guards clean.
No new64MiB model run is attributed to this experiment. Both experimental
actual runs retire completely with all source, artifact,
binary and recipe guards passing, zero memory events and no remaining GPU
processes; the confined CPU scorer also retires. Final documentation QA remains
a landing check. The earlier residual run is retained as a failure: both
generations completed, but a stale pre-pointwise pixel assertion prevented
counter publication. It was removed without weakening same-arm repeat checks
or the controller's qualified output pin.

[convolution-workspace.json](convolution-workspace.json) records compact
observations and exact receipt identities. Main receipt
`8ba022d9680eefe8bcac939130de216ebfe7e4d1e82b096c5721fe0b87db437e`;
quality`ba16392bf150e07c94c28b9c9610fc6672ea2bb2e596ac218da6bc1df3281ceb`;
cost`afffa17ab6adc6cbbcfd63e76369aed9afe3caee5ee7ed08cc18ec7c665812ae`.
Generation binary`ede1367bc402c16c109705172589b5ab0061c975c25674374be6ed0981c188f5`
(198,842,448 bytes). Request-driven geometry/schedules and shared serving,
independent upstream quality and the family migration gate remain open.

Final restored64MiB integration receipt
`2066eeef9c255382db0bf48c2ab30ec4e31dd1152abdb833ab99fdd0f1027da9`;
final generation binary`fa5420de2613836b7e345e9a7074794b56e5058971662ecc0bd3b2255e46a7cb`
(198,842,440 bytes). This final binary contains only the diagnostic assertion
fix over the qualified64MiB implementation; the1GiB actuals retain the distinct
experimental binary identity above.

## Opt-in packed masked attention

The shared spatial component now emits the existing Attention primitive. Its
ordinary decomposition retains the dense F32 products, scale/mask order and
SoftMax. An explicitly selected D128 implementation reuses GGML's masked MMA
attention with paid transient packing. Resident source formats do not change;
padded K/V cells are zero and padded mask cells are negative infinity. This
is an approximate F16 operand policy, including finite mask bias, rather than
an unrestricted F32 implementation. Zero/negative-infinity mask values remain
exact. Existing selected attention families retain their original paths.

A complete-output representative screen at1,057 rows/four heads reduces the
five warm completed-job median from712.190 to419.215 microseconds (41.14%).
The seven-row/two-head control instead rises from27.200 to29.472 microseconds
(8.35%). These copy-inclusive observations do not establish a universal shape
policy. All9,773,568 qualified output values are finite; full-small and sampled
representative F64 checks pass the predeclared0.01 bound. Changed masks and
inputs, restored capture, tails, pitched V,228 canaries and two underpayment
refusals pass. The deliberately out-of-range70,000-valued V control produces
1,536 nonfinite candidate values, demonstrating why shape alone cannot make
this approximate path the generic F32 default.

Fully masked queries exposed two existing upstream zero-denominator seams.
The pinned SoftMax bodies now write zero for empty rows; MMA direct stores,
stream-k merges and parallel combine likewise guard final zero denominators.
Finite positive-denominator arithmetic is unchanged, and nonfinite numerators
still propagate. The raw failed attempts are retained. Representative execution
covers the selected paths, not every cooperative or merge variant. IR/reference
sinks and ALiBi decomposition does not add GGML runtime support: existing
controlled refusals for those cases remain.

The complete1024-square/two-step screen uses one paid immutable tuning tuple
and the existing shared generation lifecycle. The source graph contains32
matching Attention nodes; this is source/selected-plan evidence, not a new
runtime operation counter. Dense production selection stays unchanged.

Two complete generations pass the retained native migration comparison at
**48.229 dB PSNR / 0.99877 SSIM**, above the unchanged32dB/.98 floors.
The exported4MiB RGBA output matches the candidate's own completed digest and
repeat. This remains a low-detail two-step sample, not independent upstream
quality or usable40-step perceptual qualification. No legacy model rerun or
cross-implementation bit gate was used.

The output-disabled observation uses the existing completed boundary after both
flow steps. Setup is0.863s; first generation is
20.344s and warm generation is
**9.603s**. The warm prefix is
6.778s and decoder completion
2.826s; the decoder's one device job is
2.794s. These include existing stage copies and
handoff, and are descriptive observations rather than a matched1% result.
Quality clocks include the export observer and file I/O and are not timing
samples. The approximate operand policy remains opt-in; no automatic F32
shape-only adoption follows from these results.

Actual quality admission is39,862,800,816 bytes and
construction2,731,102,696 bytes; output-disabled admission
is39,862,800,816 bytes and construction
2,710,131,176 bytes. The fixed screen allows48GiB managed
resources including4GiB construction, plus6GiB physical margin. The complete
selected source closure remains30,530,338,816 bytes. Process/cache peaks are
992,067,584 and962,564,096 bytes,
respectively, not device occupancy. Actual measured stage placement, pool margins,
retained captures and paid packing determine admission before work.

The current prepared GGML tree passes a full native build and101 affected host
controls across five targets. The later one-path full-image fixture change passes
full native compilation and both model inventories; those101 unchanged host
binaries/source are retained explicitly rather than claimed rerun. Representative
and image actuals authenticate their own distinct binaries. All source, artifact,
recipe and binary guards pass before and after execution; all children retire,
all memory limit/OOM events remain zero and no GPU processes remain. Final
source checks pass on Spark B:17 formatted C++ paths,13 portability boundary
checks,2,269 SPDX headers and REUSE6.2.0, over the2,549-file landing export.

[masked-attention.json](masked-attention.json) binds compact observations and
literal receipt/model identities. Request geometry/prompt/schedule flexibility,
independent upstream quality and the family serving/swap/speed gates remain open.
