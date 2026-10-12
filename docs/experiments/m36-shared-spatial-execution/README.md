<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Shared spatial StepProgram: first actual VAE spans

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
