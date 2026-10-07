<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma31 serving gap: state reuse, capture beside execution, fused decode FFN

Three engine changes close most of the bounded Gemma31 serving gap to official
llama.cpp v0.6.0 (d812350, the [bridge](../gemma31-serving-bridge/README.md)'s
reference, image c604…). Every native cycle below emitted the same tokens and
history as the bridge's frozen native proof, byte for byte. Measured on
`spark-b` (GB10), 2026-10-06, with the bridge's inputs, configurations and
artifact. The `cycles` mode of the [native caller](../../../benchmarks/gemma31_production.cc)
runs warm, second and third cycles in one process. The reference helper is the
bridge's unchanged `llama_serving.cc` binary.

## Final bookends

The bookends ran reference/native/native/reference in fresh processes. Each
process had a warm cycle, then a paid cycle (the reference's) or a second
cycle (native's). Both engines measured about 2% slower in this window than
in the bridge's runs, so compare engines within this table only.

| Workload | Reference paid | Native second | Native third | Native − reference |
| --- | --- | --- | --- | --- |
| C1 | 23.475 / 23.576 s | 23.684 / 23.711 s | 23.533 / 23.559 s | +0.73% (bridge: +3.15%) |
| C4 | 59.865 / 59.732 s | 60.765 / 60.464 s | 59.938 / 59.688 s | +1.36% (bridge: +3.51%) |

C1 prefill is +0.44% (10.876/10.900 versus 10.828/10.852 s) and C1 decode is
+0.98% (12.809/12.810 versus 12.647/12.723 s). C4 prefill is +0.6%
(43.920/43.620 versus 43.553/43.388 s) and C4 decode is +3.2% (16.845/16.844
versus 16.313/16.344 s). Native's third cycles are within 0.1% of the
reference's paid cycles in both workloads. What remains is mostly C4's
four-owner decode (see Profiles).

## What changed

**State reuse across Clear.** Before, Clear evicted a conversation's 2 MiB
state extents. The next prefill then mapped fresh extents and loaded them
from a sparse zero file. For an 8K Gemma31 owner that is 820 extents: about
57 ms of Clear plus 162 ms of growth inside the paid cycle. Clear now runs
`LiveState::ZeroForReuse` within the slot's request: one device fill zeroes
the used extents in place. The discard then keeps their backing outside the
state:

- The kept extents are discarded in the catalog, so they can be neither
  leased nor registered, and the cleared slot holds no state.
- The next growth takes them back with no load (`Catalog::ReviveDiscarded`,
  at a new content generation).
- Every runtime reclaim releases kept backing before pricing anything else.
  So do materialization's victims, model eviction, swaps, teardown and plain
  discards.

Zeroing stays necessary. Qwen3.8's recurrent state reads its initial zeros,
masked padded reads multiply stale values, and checkpoint and spill bytes
must not carry a previous conversation. The same change applies to DeepSeek
V4's and Qwen3.8's Clear.

**Capture beside the eager run.** Gemma keys prefill plans by position. A
repeated prompt layout therefore captured a CUDA graph on its second
sighting: about 8 ms of host capture and instantiation per 256-row chunk
while the device sat idle, 265 ms for an 8K C1 prefill and four times that
for C4. `GraphRuns::Queue` now queues the eager run first. It then records
the same sequence, instantiates it and uploads it while the device executes,
and the graph replays from the next sighting. Plans with work queued between
inputs and plan (Qwen3.8's n-gram gather) keep the old capture-first path.

**Upstream's fused quantized FFN gate/up/GeGLU for decode.** Node-level nsys
of both engines' C1 decode gave identical attention and identical products
for every shape except the FFN gate/up. Stock runs one fused
`mul_mat_vec_q` per layer, with the gate and the GeGLU in the same kernel:
60 fewer products, quantizations and GLU launches per token. Native ran them
separately, +2.2 ms of GPU time per token. A new registry implementation
(`ggml.mul_mat_{glu,geglu}.mmvq_fused`) calls upstream's
`ggml_cuda_mul_mat_vec_q` with fusion arguments under upstream's conditions:
one output column, matching quantized weights, F32 activations, and never
in row-invariant (verify) plans. The fused output equals the separate
products plus GLU bit for bit
(`GgmlExtOpsTest.FusedQuantizedGluEqualsItsProductsAndGlu`). Only the
bounded dense31 production candidate enables it so far.

## Each change, isolated

Same binary, env-toggled arms (the toggles are not committed), bookended
off/on/on/off in fresh processes. C1 second and third cycles, seconds:

| Change | Off: second | On: second | Off: third | On: third |
| --- | --- | --- | --- | --- |
| State reuse | 23.798 / 23.786 | 23.599 / 23.599 | 23.499 / 23.468 | 23.286 / 23.291 |
| Capture beside (state reuse on) | 23.693 / 23.646 | 23.484 / 23.500 | 23.366 / 23.341 | 23.327 / 23.334 |
| Fused decode FFN (both above on) | 23.377 / 23.533 | 23.283 / 23.244 | 23.290 / 23.348 | 23.156 / 23.112 |

C1 decode with the fused FFN went from 12.616/12.664 s to 12.461/12.452 s.
For C4, state reuse took the second cycle from 61.539/61.343 s to
60.541/60.411 s. Capture beside then took it to 59.879/59.824 s.

The handoff's first experiment, turning prefill capture off, was a wash.
Capture-off arms measured 47.30 s for second plus third cycles, against
47.22 s with capture on. Replay saves about 190 ms on each later prefill of
the same layout, about what the one-time capture costs. Capture beside keeps
the replay and hides the acquisition.

## Profiles

These come from node-level `nsys` (2025.3.2) on both engines, decode only.
The reference's helper was profiled from a diagnostic copy built without its
LD_PRELOAD refusal; that copy is retained externally. The C1 decode step
spans 95.8 ms in the reference and 97.9 ms in native before the fused FFN.
In C4 (four owners, four-column products, where upstream does not fuse
either), native's remaining step excess is:

- `k_set_rows`: +1.2 ms, because native stores each owner's K/V rows
  separately (480 launches against 120);
- spread product time: about +1.0 ms;
- owner attention: +0.7 ms;
- `rms_norm_f32`: +0.5 ms.

The norm kernels have identical sm_121a SASS in both builds but run slower
in native in both decode and prefill. That points at the memory or context
they read, not code generation; it is not yet located.

## Checks

These passed on `spark-b` against the final source:

- GPU tests: `catalog_test` (22), `live_state_test` (24), the fused-GLU
  and quantized-product `ggml_ext_ops_test` cases, `gemma4_runner_gpu_test`
  (27), `gemma4_serving_gpu_test` (26), `gemma_joined_serving_gpu_test`
  (10), `gemma4_greedy_gpu_test` (6), `gemma4_verify_gpu_test` (7) and
  `cuda_graph_test` (7).
- The DeepSeek/Qwen3.8 8K swap table: four rows exact with exact states and
  no problems. Its second pair clears each model from the first pair's state.

The full spark-native suite, the workstation tiers and Gemma26 adoption
remain owed.
