<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Normalized expert combine: rejected first screen

On 2026-10-18, Spark B (`spark`, hostname `spark-c4e2`, GB10) ran a
bounded kernel microdiagnostic at Gemma26's actual width 2,816 and eight selected
experts, with rows 1, 2, 3 and 4. This is not an end-to-end serving result.
The native locked-SDK build, one planner authority control and all four GPU
controls passed; no skips or hidden failures were recorded.

The candidate preserved `(expert * gathered_scale) * route_weight` in ascending
selected-slot order, followed by routed RMS normalization, its norm weight and
**shared-first** addition. Routing normalization, outer postnorm and residual
were unchanged. Each row's reduced values lived in registers until its RMS
scale was known. It used one CTA per row, 1,024 threads at this width and zero
scratch. No production registry, pattern, policy or numerical grant selected it.

The complete incumbent did three kernels: scaled expert reduction, routed
RMS/multiply, then dense RMS/multiply/add. The candidate did two: **separate
dense RMS/multiply**, then fused expert reduction/routed RMS/multiply/add.
Charging that dense normalization matters: it replaces an incumbent fusion.
Both complete paths were captured on the same llmpalooza stream and used the
same unchanged operands. After eight warm-up replays per arm, CUDA events
measured 256 replays in incumbent/candidate/incumbent order for each row count.
The criterion, fixed before execution, required at least 5% lower latency than
**both** incumbent bookends in at least one useful cell before any serving
integration. Synthetic input setup and graph construction were outside timing.

| Rows | Incumbent before (µs) | Candidate (µs) | Incumbent after (µs) | Candidate versus bookend mean |
|---:|---:|---:|---:|---:|
| 1 | 8.18738 | 8.19538 | 8.18750 | +0.097% |
| 2 | 8.19150 | 8.35987 | 8.18738 | +2.081% |
| 3 | 8.18750 | 8.37062 | 8.19137 | +2.213% |
| 4 | 8.19125 | 8.33275 | 8.19150 | +1.726% |

No useful cell met the criterion. The candidate was rejected and its unselected
standalone API/kernel/microtest were removed; there is no alternative tuning or
whole-family ladder for this factor. Both paths used zero scratch; possible
activation elision was not measured as a model memory benefit.

Before timing, every output passed an independent FP64 complete-chain bound
`abs(error) <= 2e-5 * max(1, abs(reference))` and incumbent NMSE `<= 1e-10`.
Selected-order cancellation, changed inputs/scales/weights under the same
capture, eager/captured equality, output guards, short backing/alias/geometry/
epsilon refusals and normal retirement were checked. Separate existing
VecQ/GeGLU slab and per-slot down controls gained an actual row-3 iteration and
passed without changing their original oracles. Their retained coverage does
not extend four-row evidence to seven rows or grant model quality permission.

The source was `r7-stage8k-gemma-candidates`, manifest
`cb6f9f908eccd62be4e260c2f385d7bcb3bed913089303db7b6c0ebe5bc5203e`,
built with the pinned Spark-native SDK `aarch64-2c4eb5b2f68f3fa5` and existing
GGML device math flags. The experiment patch, preregistered recipe, full logs
and machine-readable receipt were retained externally under
`/home/pmeenan/scratch/m36-review-followups/`; GPU job
`r7-stage8k-focused-gpu-B-20261018a`, PID 312634. The receipt is
`r7-stage8k-gemma-candidates-execution-receipt.json`, with per-cell raw values;
the GPU log SHA-256 is
`e8192576bcfa324437d28833da6c554dd6d9214bf34b510d88df52f3d65123fa`.
No separate toolkit/driver probe was captured by this micro recipe. Historical
oracles, production grants and serving defaults were not changed.
