<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma26 coarse prefill CPU diagnosis

The two graph-planning passes consume about **1.093 s of charged caller CPU**
within a 1.206 s plan-miss interval, with no recorded GPU overlap. This is the
strongest measured lead for reducing the native prefill gap. The experiment
changes no planning algorithm, arithmetic, cache policy or production selection.
It does not identify particular matcher instructions or prove an optimization.

The competitive result remains the untraced [ring-cache comparison](../gemma26-swa-ring-transfer/README.md):
native prefill takes 57.6433% more time. The preceding [GPU profile](../gemma26-prefill-profile/README.md)
found a large interval without recorded GPU activity, but could not attribute
it to CPU work. This follow-up records coarse intervals and charged CPU on the
actual executing threads. [Results](results.json) and [provenance](provenance.json)
contain aggregates and immutable identities; raw records stay external.

## Work and fidelity

One approved Gemma26 artifact, C1, context16,384, max_rows1,024,
local2,048/global16,384 and F16 KV. The unchanged workload warms six rows,
clears, pays eight 1,024-row prefill chunks, builds three untimed anchors, and
completes 32 forced decode units. Each decode argmaxes the incoming head.
Native still publishes eight prefill heads. The original recipe publishes one;
its final-block/head work differs too. No head suppression is introduced.

Selected-plan counts remain 60 norm/RoPE, 90 norm/add, 30 routing and 30
reduction; other optional counters are zero. These counts are not CUDA launch
counts or original-engine eligibility parity. Exactly one annotated untraced
control and one trace completed. Both reproduce Task40's complete prefill/final
heads, all 32 choices, all 592,445,440 initialized state bytes and the same
layout. Application, profiler and installed supervisor retirement passed.
Cross-engine full-head quality and competitive qualification remain open. No
all-32 full-vector publication, new PPL scan or noise calibration is claimed.

## Observed coarse phases

The trace validates 201 NVTX ranges: one outer range plus the exact 25-phase
roster for each of eight ordinals. Caller, scheduler and submission roles are
bound to actual thread counters and the model process. Parent containment,
source order and worker ownership pass. All kernel/copy/memset ownership is
known. No CPU samples or stack traces were captured.

| Phase | NVTX elapsed, ms | Charged thread CPU, ms | Recorded GPU overlap, ms |
| --- | ---: | ---: | ---: |
| Wave, caller | 3,808.885 | 3,733.778 | 2,469.403 |
| Plan miss, caller | 1,206.500 | 1,203.247 | 0 |
| First graph plan, caller | 546.782 | 546.746 | 0 |
| Activation placement, caller | 13.050 | 13.054 | 0 |
| Second graph plan, caller | 546.531 | 546.525 | 0 |
| Implementation/scratch binding, caller | 40.128 | 40.132 | 0 |
| State growth, caller | 67.847 | 3.052 | 11.344 |
| Completed job, caller | 2,527.061 | 2,526.803 | 2,458.059 |
| Job submission body, worker | 890.956 | 891.007 | 836.723 |

Parent phases include their children; concurrent thread scopes also overlap.
Do not add these rows. The two graph-plan passes are ordered, disjoint children,
so their charged CPU can be combined as 1.093271 s. Their first/second-pass
validation and SamePlan checks remain intact. Coarse scopes do not separate
norm reader scans, MoE private-output scans, graph construction or individual
matcher instructions.

NVTX and monotonic/thread-CPU clocks have different origins and observer
boundaries. Tiny elapsed/CPU differences are not a clock-origin subtraction.
Charged CPU means scheduled work on that thread; it is not a sampled stack or
classification of driver polling, blocked waits or useful application work.
In particular, the completed-job scope overlaps almost all GPU work and charges
about 2.527 s of caller CPU. Neither its elapsed time nor OSRT waits establish
why. Its value must not be added to GPU time or subtracted as an independent
host cost. The model scopes are instrumented diagnostics, not replacement
competitive timings.

## Controlled instrumentation and capability

A reversible [three-unit overlay](instrumentation.patch) applies only to an
isolated measurement tree based on d08eea75. It adds scopes to the existing
Gemma runner, live-state initialization and first/placement/second planning
boundaries. Existing scheduler/submission closures carry the bounded ordinal
by value. Conditional closure refresh and cohort lease replacement remain part
of paid state growth. Canonical production sources remain unchanged. Diagnostic engine archives and
objects stay in the isolated tree; no default executable was rebuilt or
transferred. Only dedicated manual targets and the atomic child were compiled.

The shared counter singleton has 2,726 known static bytes, bounded thread
ownership and checked monotonic/thread CPU clocks. The public NVTX headers are
external, under Apache-2.0 WITH LLVM-exception. A single no-model probe measured
150.000 ms elapsed/149.994 ms CPU for known busy work and 150.545 ms elapsed/
0.092 ms CPU for a known wait. No device provider or model is linked into that
probe. Earlier CPU-sampling capability was unavailable; no permission or
security setting changed.

The NVTX-only probe export omitted its child PROCESSES name row. Initial
metadata validation refused it. The separately reviewed [analysis derivative](analyze_v2.py)
validated all 464 actual process serialization pairs and one namespace, then
bound both positive, sequential ranges to the atomic child PID and actual
counter gettid. Missing names are reported explicitly. The model trace still
requires its actual child PROCESSES row. Wrong PID, TID, namespace, payload,
process pair, overlap and reversed order controls all refuse. The source roster
also has three positive and 20 omission/thread/parent negative controls.

## Reproduction and lineage

The [protocol](PROTOCOL.md) is the unchanged preregistration snapshot, including
its before-build status. The 16 acquisition files remain byte-exact, bundle
58e41ba4; their measured environment is source receipt f7c8846d. The later
analysis17 source is 7a0df7aa. Original analyze.py, failed records and the earlier
derivative receipt remain archived. No model/probe rerun repaired those records.

In a separate tree at the recorded base, first supply only the 16 acquisition
files listed in provenance.json. Run validate.py prepare before adding the
analysis derivative or report files; its closed dirty-path guard creates the
source-input frame. Then use apply_overlay.py to authenticate and apply the
three changes. Copy the frame into the measurement scratch directory. Configure the locked SDK with
JITLLM_BENCHMARK_COARSE_PREFILL=ON and JITLLM_BENCHMARK_NVTX_INCLUDE_DIR pointing
to the authenticated public headers. Build only jitllm_gemma26_prefill_coarse
and jitllm_gemma26_clock_probe; compile profile_child.cc with the exact probe
compiler/link arguments. Preserve the resulting command/dependency inventory.
Prepare the canonical IDs, Task40 receipts and public-interface record in the
external scratch directory. Supplied raw inputs and dependency pins are
identified in provenance.json; they are not vendored here.

Run probe.sh once under the installed Spark supervisor, then validate its
retained SQLite/log with analyze_v2.py probe, supplying that analyzer's SHA.
Use unchanged validate.py source to freeze the environment after proven build
and probe retirement. Run native.sh control under supervision, validate against
Task40 using the new source SHA, and admit native.sh trace only after fidelity
passes. Validate traced fidelity before analysis. The full analyzer requires
explicit source/control/trace/analyzer SHAs and checks actual model ownership,
all eight interval rosters, clipping and thread counters. Its fixed
native-control-validated.json path can be an exclusive byte-identical alias of
the admitted receipt; actual output/log identities must remain unchanged.

Every build/probe/model/analysis job used installed GPU admission, stop-on-fail
and a 600-second limit. The narrow child-link parser build failure, missing
probe-name validation failure and missing-ID source-freeze failure are retained.
Corrected metadata/setup retries passed without changing the acquisition
sources. Compile closure records 264 transitive commands, 232 objects and eight
used NVTX headers within the authenticated 33-header interface. Fresh primary
TensorFold checks at entry and immediately before the model control still
resolve 609ca419/0.6.5; Gemma26 remains an MLX recipe there.

Planning now has a measured CPU lead. A subsequent optimization must preserve
both graph-plan passes, SamePlan, primitive fallbacks, malformed-view guards,
kept/readable outputs and post-placement alias checks, then measure actual
model fidelity and paid performance. Dense31 and earlier-family applicability
need their own checked transfer evidence. This diagnostic selects no policy.

Final local REUSE/header checks cover 1,558 files; 392 boundary sources, changed
format/diff/AST/JSON/bash checks and the roster/ownership controls pass.
