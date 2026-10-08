<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Decode hot path: device greedy tokens, direct steps, no per-wave scheduler calls

A decode step's GPU work was already matched to llama.cpp's
([Gemma gap closing](../gemma-gap-closing/README.md)). What remained between
two steps was host work and thread handoffs, while the GPU sat idle. Three
changes remove almost all of it:

1. **Device greedy tokens** (Gemma). A greedy step chooses its token on the
   device and copies back 4 bytes instead of a 1 MiB row.
2. **Direct steps** (every model, D-106). The driver queues a request's step
   and waits for its fence itself. Before, the step crossed four threads
   there and back.
3. **No scheduler call per wave.** The Gemma runner no longer charges its host
   inputs inside the budget each wave, which double-counted memory the
   start already sets apart. Each LLM runner now skips its pinned-place check
   while nothing has been placed since the last clean one.

## Results

These are third-cycle decode times on `spark-b` (GB10) for the
[serving bridge's](../gemma31-serving-bridge/README.md) 4×8,192-ID inputs,
each from a single fresh process. Every column is cumulative. All 129 tokens
per owner match the frozen proofs in every cycle of every run.

| Workload | Before | Device greedy | + direct steps | + no per-wave calls |
| --- | --- | --- | --- | --- |
| Gemma26 C1 | 2.665 s | 2.646 s | 2.667 s | **2.590 s (−2.8%)** |
| Gemma31 C1 | 12.905 s | 12.867 s | 12.924 s | **12.833 s (−0.6%)** |
| Gemma31 C4 | 16.62 s | 16.485 s | 16.517 s | **16.434 s (−1.1%)** |

A step's round trip is its wall time less its device time (`StepTimes`). In
decode it fell as follows:

| Workload | Before | Now |
| --- | --- | --- |
| Gemma26 C1 | 85.4 µs | 11.5 µs |
| Gemma31 C1 | 127.0 µs | 6.3 µs |
| Gemma31 C4 | 188.0 µs | 10.3 µs |

The runner's host phases fell from about 570 µs to 12 µs per Gemma26 decode
step. In the final runs, decode wall time is within 0.1 ms per step of the
device's own time.

Direct steps on their own were slightly slower. With nothing else on a
step's path, the scheduler slept through decode. Each wave's two remaining
scheduler calls (the host-input charge and the place check) then paid a full
wake on the Spark, 200–480 µs ([RE-017](../../rough-edges.md)). Removing those
calls is what turned the shorter round trip into a gain. Whole-cycle times,
for comparison with the recorded llama.cpp v0.6.0 runs (not fresh bookends,
so only approximate):

- **Gemma31 C1:** 23.508 s third cycle, against the reference's
  23.475/23.576 s.
- **Gemma31 C4:** 59.470 s, against 59.865/59.732 s.
- **Gemma26 C1:** 4.934 s second cycle, against 5.013/5.023 s.

### DeepSeek V4 and Qwen3.8

These are `jitllm-runtime chat` runs of one prompt (37 tokens on DeepSeek V4,
85 on Qwen3.8), with 384 greedy tokens, stop ignored and speculation on (the
swap table's configuration). Each process ran fresh, in the order
before/after/after/before, with `ba67d9e` as the before build. Every run of a
model wrote the same text.

| Model | Before (tok/s) | After (tok/s) |
| --- | --- | --- |
| DeepSeek V4 | 36.68 / 36.62 | 36.85 / 36.82 |
| Qwen3.8 | 47.25 / 47.40 | 47.98 / 47.44 |

Neither model regresses. The apparent gains (about +0.5% and +0.8% in the
means) are within run-to-run noise at two runs each.

The second Qwen3.8 "after" run did not match the others. It started from the
draft-depth cost the first "after" run had just recorded (calibration is in
force from the next start of the same build), where the others used the
fallback. That different schedule gave its lower acceptance (0.660 against
0.720) with the same text. (The second DeepSeek "after" run also started from
a recorded decode floor, which only the chat route's watchdog reads, not
this command; its rate and acceptance match the first.) The one matched pair, the first "after" run
against both "before" runs, is +1.4%, from a single run.

DeepSeek V4's and Qwen3.8's speculative steps are fewer per token than
Gemma's plain steps, so a
smaller saving is expected; that was not measured separately.

## Device greedy tokens

A Gemma generation takes its token from the device when it is greedy and
nothing needs its logit rows:

- The chunk graph appends `jitllm.argmax` over the frontier logits
  (`Gemma4ChunkShape::greedy`). It returns the lowest ID among equal maxima,
  as the host's greedy choice does.
- Only the I32 tokens are copied back.
- Joined, scalar-cohort and single-session (`Generate`, `RunGreedyChunkFor`)
  steps are all covered. A wave is all rows or all tokens.

The rule (`Llm::DeviceGreedy`) is: a plain step (not speculative), no
sampling, and no `on_logits` or `keep_logits`. Chat requests at temperature 0
qualify. Scoring, sampling and speculation keep the rows. Each greedy shape is
its own plan and graph, and the startup scratch measurement covers it.

The original factor agreed on finite rows but used native device NaN behavior.
The later [plain-token transfer](../plain-gpu-tokens/README.md) closes that
limitation for plain Gemma, DeepSeek and native/GGUF Qwen outputs: the explicit
host-greedy flavor preserves lowest-index ties and selects zero when the
first row element is NaN, matching `max_element`. Native draft/verify behavior
is unchanged.

## Direct steps

Within a request that holds its lease, `PagedNode::Step` runs the job on the
driver, records the step's fence and waits for it there (`DirectStep`).

**The lease.** The request's task still takes the lease and ends it. The lease
counts the driver's steps in flight (`RequestChannel::external`, shared by
the channel and the lease, so either may go first) and is never released
while that count is not zero, whatever ends it. A step whose end cannot be
proven keeps the lease for good, and the stop reports it unproven. The
step's wait (and the hang ladder's watch) begins before its launch, so a
launch blocked on a full stream (RE-029) is still watched.

**Waiting.** Nothing wakes the driver at a step's end:

- A blocking wait or a host-function signal would put a 200–480 µs wake on
  every step. So the driver sleeps toward the step's likely end and spins on
  a non-blocking fence query around it.
- The likely ends are the last eight steps' ends. Each is measured as when
  the step began plus the device's own span, not when the driver noticed it.
- It never sleeps more than 1 ms at a time.

**Prediction errors.** A sleep that overshoots therefore never carries into
the next prediction. A step that ends earlier than any recent one is seen at
most about a millisecond late, once. A step that runs longer than all of them
is checked every 200 µs.

The first version predicted from walls, which included its own oversleep. It
locked into steps 36% slower than the device's. `cuda_paged_node_test` now
bounds each gated step's lag, for steps ending on time, early, late and at
once. Steady steps are seen 5 µs after their end; the two that end
unexpectedly early are seen at about 1 ms. Stepping took 0.15 cores against
0.41 through the task, and the scheduler and lanes sleep through a decode.

## No per-wave scheduler calls

**Host-input charge.** `Gemma4Runner::WaveWithMode` charged
`host_input_bytes` (128 MiB for Gemma26) inside the budget before every wave
and gave it back after. The host total was already past the node's floor, so
both moved the runtime extent through a scheduler `Call`. The runtime's start
already sets those bytes apart beside the budget (serving.cc: one chunk's
host inputs at a time, "not in the budget the catalog enforces"), so the
charge is removed.

**Place check.** `CheckPlaces` made a scheduler `Call` every wave in Gemma,
DeepSeek V4 and Qwen3.8, although each object inside it already skips when
the scheduler's placement stamp has not moved. The scheduler now publishes a
count of placement changes that any thread can read. A runner skips the
`Call` while the count is the one at its last clean check, and forgets that
count whenever its closures (and so the states it checks) change.

## Checks

These ran on `spark-b`:

- **Gemma suites:**
  - `gemma4_graph_test`: a greedy shape adds only the frontier argmax, and
    state-only and feature outputs refuse it.
  - `gemma4_plan_test`.
  - `gemma4_runner_gpu_test`: for one, two and four owners, the device's
    tokens are the host's greedy choices of the replayed rows, the KV bytes
    are unchanged and a mixed wave refuses; lookahead pressure now fills the
    whole budget.
  - `gemma4_serving_gpu_test`: `Generate` without rows matches the host's
    choices from rows, and the runner counts its device tokens.
  - `gemma_joined_serving_gpu_test`: a pass without rows matches the solo
    tokens and state for 1–12 owners, and only that pass counts device
    tokens.
  - `gemma4_greedy_gpu_test`, `gemma4_verify_gpu_test` and `runtime_test`.
- **Scheduler and node:**
  - `held_lease_test` (14): external steps keep an ending, cancelled or
    finished lease until they drain, the count outlives a freed channel,
    and an undrained one faults the stop as unproven.
  - `cuda_paged_node_test` (15): both step paths; the hang cases, including a
    hung direct step whose request is cancelled and whose lease is kept
    until its fence.
- **DeepSeek V4 ↔ Qwen3.8:** the swap table at 8,192 context tokens with
  64-token continuations gives exact states and continuations in all four
  rows.
- **Chat:** a natural two-turn Gemma26 `jitllm-runtime chat` stops normally
  with the expected answers.
- **Full spark-native suite:** 1,806 of 1,808 tests pass. The two failures
  exercise nothing this change touches, and both follow from the GGML
  v0.6.0 refresh (`9ff4e98`), which landed while the full suite was deferred:
  - `GgmlOpsPlanMatchTest.DecodeStepStartRecordsAsTheSample`: the recorded
    plan sample predates v0.6.0's `rms_norm_f32` signature.
  - `GgmlExtOpsTest.GemmaAndLegacyMmaPlansBoundTheLastPaddedKvTile`: a D256,
    8-row, 128-head shape is no longer refused.

The two stale expectations were reconciled on 2026-10-07 without changing
production code. The current RMSNorm launch signature now matches the recorded
sample; attention bounds use the refreshed 64-cell tiles while retaining all
overflow refusals. Focused Spark B checks pass with no skips: 12 GGML ops,
45 extended ops and 3 recording tests, plus 35 Python comparator controls.
The frozen P0 signature remains distinct and strictly compared. The full
1,808-test suite was not repeated; its result above records the original run.

## Provenance

Native ran on main `ba67d9e` plus this change (spark-b build tree, NVCC 13.4
SDK). Raw logs stay external on spark-b.
