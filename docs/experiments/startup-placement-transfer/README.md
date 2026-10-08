<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepSeek and Qwen startup placement transfer

The shared T96 placement shortcut is selected during DeepSeek/DSpark and
native/GGUF Qwen Setup. Actual configured Setup preserves every funding maximum
and outer planned operation count across five recipes. The representative
native+MTP O/A/A/O comparison takes 2.441% less Setup time (n=2 per policy), with
non-overlapping ranges and −0.692% ordinary bookend movement. Other recipe pairs
are single observations, without per-recipient speed claims. This measures
Setup alone; it establishes no inference throughput or model-quality result.

[Aggregate results and immutable identities](results.json) contain every arm,
full unchanged budget, descriptor counts, artifact pins and completion proof.
The [probe](../../../benchmarks/startup_measurement_probe.cc) remains reusable
without the raw milestone bundle.

## Exact maximum and runtime boundary

The previously qualified [checked disjoint bound](../gemma4-original-q8/README.md)
is unchanged. Each startup plan compares its checked rounded root-storage sum
against the current exact activation maximum. If the sum fits, greedy placement
cannot raise that maximum and is skipped. Otherwise exact placement runs on the
same selected graph. Scratch, sources, lanes, host ownership and node accounting
remain exact. Measurement-only plans cannot bind, including exact fallbacks.

Explicit optional overloads preserve all original DS/Qwen planning signatures.
Nonzero runtime activation storage with measurement refuses before any model
access or descriptor mutation. Runtime calls retain the original exact path.
DeepSeek uses its existing single maximum. Qwen keeps independent scalar and
wave maxima, including measurement-only scalar subplans used to authenticate
wave composition. Inner placement offsets have no composition consumer; the
outer plan remains nonbindable. Null measurement preserves the original inner
and outer path.

`StartupPlacementStats` counts outer planned nodes/selections and placement
branches, not executed GPU kernels. Qwen's inner wave subplans are not counted.
The internal `startup_activation_threshold` defaults true; explicit false
retains ordinary sizing for comparisons. There is no new public setting.
Remaining graph construction, sizing and selection cost stays open across the
families; this shortcut removes only eligible activation placement.

## Actual configured Setup checks

Spark B, 2026-10-08, NVIDIA GB10/driver580.178.04, official Spark-native SDK `aarch64-c09daba6ac31edee`,
receipt `874aaf7a…`, actual private cuBLAS/Lt payloads `ee7c1657…`/`ba3b942f…`.
Task-entry TensorFold main `f8fe17d2…` and python-0.6 `ed78d6fc…` are unchanged;
no reference inference is needed for a startup-only transfer.

One supervised batch runs twelve fresh processes: four ordinary/threshold pairs,
then native+MTP O/A/A/O, reusing its arms as the fifth funding gate. The timer
covers actual `Runner::Setup` after node Open, including Setup resource
allocation. Observations and explicit positive teardown precede result
publication outside the timer. No Register, Start, Load, initialized state or
model dispatch occurs. Failed Open/retirement retains the node+runner owner.

| Recipe | Configuration | Ordinary/on Setup seconds | On bounded/exact outer plans |
| --- | --- | --- | --- |
| DeepSeek target | context8704/chunk512/C2 | .167082931 / .138903579 | 6/2 |
| DeepSeek + DSpark | same target, draft3/verify4 | .204383273 / .186820779 | 6/7 |
| Qwen native target | context8192/chunk512/C4 | .473553742 / .447151826 | scalar4/3; wave0/2 |
| Qwen GGUF target | context8192/chunk512/C4 | .567510897 / .560711134 | scalar4/3; wave0/2 |
| Qwen native + MTP | same target, draft2/head capacity65536 | O1 .372779766; A1 .357234837; A2 .367607535; O2 .370200362 | scalar5/8; wave1/1 |

The first four pairs are n=1 diagnostics. Native+MTP ordinary mean .371490064s,
threshold mean .362421186s (−2.44121684%, n=2). Ranges
[.370200362,.372779766] and [.357234837,.367607535] do not overlap; ordinary
bookend movement is −.69193777%. These short observations are not a sustained
startup distribution. Mathematical maximum preservation plus the representative
factor supports shared selection, without individual latency claims from n=1.

DeepSeek selects the approved frontier-head and full/partial HCA prefill
policies, retaining device/weight gates. MTP retains its BF16 selected head.
Actual Setup enumerates all configured target/token/draft/injection/joined
probes. Every pair preserves the complete budget, plan-floor/report, graph-index
allowance, planned nodes/counts and unrounded activation/source/host maxima.
Scalar/wave summaries compare independently. Qwen plain/GGUF waves show zero
shortcuts and retain exact placement. All initialized-state counts are zero.

Five focused cases pass: all eight overloads reject runtime-address measurement,
and native/GGUF target plus BF16/Q4_1 odd-prefix MTP controls preserve selected
implementations, operations, shapes, lanes, sources and host ownership under
exact fallback and positive bound. The unchanged shared T96 controls and
3,098-composition oracle carry forward. After default selection, only the two
modified policy/refusal cases run again; unchanged graph controls/apps are reused.
The first compile fails before tests on an existing missing empty initializer;
that attempt remains failed and unpooled. Successful build/acquisition retire
positively: all12app returns0, stable boots, empty kernel deltas/GPU, final
MemAvailable125,274,136,576 bytes. Results include exact hashes and final-build
provenance. Final default compilation records two unrelated Ubuntu Pro AppArmor
perfmon denials, outside the fixed driver-error refusal expressions; both final
policy cases have empty kernel deltas. No inference/reference/quality gate is
inferred or repeated.

## Durable replay

Build `jitllm_startup_measurement_probe`, `jitllm-runtime`,
`dsv4_prefill_plan_test` and `qwen38_wave_plan_test` with the official Spark-native
SDK. The five focused case names are in `results.json`. The ordinary/threshold probe
is the same executable; each output directory must be new. Run heavy work with
the installed supervisor and GPU lock, for example:

```sh
spark-job start --name startup-replay --gpu --timeout 600 --grace 30 --stop-on-fail -- \
  /ABS/BUILD/benchmarks/jitllm_startup_measurement_probe \
  qmtp /ABS/NATIVE_TARGET /ABS/BF16_MTP /ABS/NEW_OUTPUT on
spark-job wait startup-replay
```

Use `ds`, `dspark`, `qn`, `qmtp` or `qg`; target-only recipes require `-` as the
drafter argument. Supply the artifacts from the standing approved stores named
in `results.json`, verifying their full manifest/index hashes before and after.
These are existing prepared artifacts, not raw checkpoints. The native target,
DS and DSpark/MTP companions live under `~/.local/share/jitllm/m3-artifacts/`;
the checked GGUF target lives under `~/.local/share/jitllm/qgguf-artifacts/`.
Their complete immutable directory identities and paths are in the aggregate.
No prompt inputs, checkpoint copy or model-weight load is required.

Before acquisition, record exact source, executable and official build receipt
identities plus actual resolved cuBLAS/Lt paths/hashes; preserve raw library
resolution for provenance but compare parsed identities, since raw output
contains ASLR addresses. Require no foreign GPU compute and adequate free memory.
Record a kernel journal cursor and boot ID for every child, retain the delta,
and reject new NVRM/Xid/UVM/OOM evidence. Bound each process group to120s with
cleanup and positive wait. Require the probe's explicit retirement marker,
`retired=true`, zero initialized-state bytes and finite positive Setup time.

For each pair, compare `budget` and the emitted plan-budget line exactly.
Compare scalar/wave summaries independently, excluding only `bounded` and
`exact`; require their sum equals `plans`. Ordinary must have zero bounded
plans. Native+MTP both on arms must show positive shortcuts. For a repeated
factor, run fresh O/A/A/O processes and report all times, means, ranges and
ordinary bookend movement. Failure stops the batch; do not pool incomplete
attempts or automatically retry. Capture installed supervisor retirement and
GPU absence separately. This recipe is independent of the deleted raw bundle.
