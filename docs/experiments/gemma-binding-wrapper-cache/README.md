<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Reuse validated kernel wrappers within each bind

Repeated kernel declaration validation accounts for a material part of the
previously measured [binding interval](../gemma-current-phases/README.md).
Reusing validated wrappers within one `BoundGraph::Bind` reduces Gemma31 binding
from 174.483 ms to 45.828 / 44.989 ms across 32 plans, a 73.98% reduction against
the candidate mean. Gemma26 transfers the improvement: 39.755 ms to
12.320 / 12.097 ms across eight plans, a 69.29% reduction. This comparison
isolates the repeated wrapper-binding path as a contributor; it does not split
its declaration construction, identity hashing and implementation lookup costs.

| Profile / milliseconds | Before | Candidate first | Candidate repeat |
| --- | ---: | ---: | ---: |
| 31B prefill wall | 11293.4 | 11235.2 | 11110.3 |
| 31B nested binding | 174.483 | 45.828 | 44.989 |
| 31B publication and cleanup | 193.031 | 67.571 | 67.146 |
| 31B state growth | 166.547 | 208.478 | 165.711 |
| 31B completed execution | 10888.2 | 10919.6 | 10836.2 |
| 26B prefill wall | 2525.80 | 2510.71 | 2483.44 |
| 26B nested binding | 39.755 | 12.320 | 12.097 |
| 26B publication and cleanup | 41.679 | 15.923 | 15.574 |
| 26B state growth | 63.265 | 69.255 | 63.005 |
| 26B completed execution | 2401.81 | 2410.21 | 2389.95 |

Mean wall reductions are 120.65 ms / 1.07% at 31B and 28.725 ms / 1.14% at
26B. Candidate spreads are 124.9 ms and 27.27 ms; each has only one preceding
baseline. These runs establish the binding reduction and favor the candidate,
but do not establish sustained wall speed or current reference parity. Decode
is essentially unchanged. State growth and execution variability remain.
Binding is nested within the first required plan and later publication; never
add it to those outer intervals. CUDA event stream elapsed includes CPU
submission gaps and is not active-kernel time. Complete phase data is retained
in [results](results.json).

The cache lives only inside each `BoundGraph::Bind`. Its key is the immutable
resolved registry `Implementation` pointer, and its value is the first bound
step index. Later occurrences copy that step's already validated wrapper.
`execution::Plan::Build` and `Resolve` run unchanged. Every step still checks
its own current operands, arity and cuBLAS lane restrictions; no descriptors,
addresses, workspace, residency or launch context are cached. A failed first
occurrence is never inserted. RMS and general wrappers remain distinct variants.

The temporary vector reserves at most `min(step_count, 256)` two-word entries,
16 bytes each (at most 4 KiB), with smaller plans reserving their actual bound.
Wrappers remain owned by the existing step vector; there is no persistent pool.
This metadata fits the runner's existing 512-byte-per-node binding allowance.
The shared executor applies the change to all its model consumers, including
Qwen and DeepSeek, without selecting new arithmetic. Their model performance
has not been measured in this task.

Both profiles retain two complete 262,144-float heads per arm, all initialized
state and 32 continuation choices exactly against the preceding native baseline.
All 12 retained heads are finite. Initialized state is 1,740,636,160 bytes at
31B and 592,445,440 bytes at 26B. Native identity is the gate here; broader
corpus quality, the known reference prefill-logit divergence, competitive C4
batching and current reference performance remain separate open qualifications.
No fresh llama.cpp or TensorFold inference ran.

Seven focused controls pass: repeated mixed RMS/general binding with later
invalid descriptors and arity, separately stale RMS and general declarations,
cross-call descriptor repair, later cuBLAS lane/operand rejection, existing
launch-versus-capture and concurrent-lane equality, both selected norm-chain
capture controls, and the existing stale/foreign registry control. The first
31B checker driver used a JSON steps line and failed before executing the
checker; its corrected shell-text retry passes against unchanged outputs.
No full suite ran; the owner defers it while optimization work continues.

Reproduce with `jitllm_gemma_prefill ARTIFACT IDS NEW_OUT 31 both 256
normmul-on state-only lookahead-on phases-on`, and `26 all 1024` in place of
`31 both 256` for 26B. Each sequence runs before, candidate first and candidate
repeat. The inherited [lookahead recipe](../gemma-prefill-lookahead/README.md)
uses six discarded warm rows, Clear, 8,192 paid rows, three untimed anchors and
32 forced units, context 16,384 and F16 KV. Local/global cells are 1,280/16,384
at 31B and 2,048/16,384 at 26B. Snapshots, hashes and finite scans are outside
timers. Both recipes explicitly enable their previously checked research
chains; the wrapper cache itself is shared executor behavior.

Candidate source base is `e3e48a5`; the before helper is the retained Task54
split helper, with explicit normmul-on matching the later default. Builds and
focused controls run on Spark B (`spark-56f5`), 31B acquisition on Spark A
(`spark-c4e2`), and 26B on B. The inherited pinned ring reference launcher and
provenance are linked from the lookahead report. Native NVCC is 13.4.92,
toolkit 13.4.2 and driver 580.178.04. Candidate helper is `5467bcaf`, before
`333f7650`, SDK receipt `b2174122`; full identities and exact-output hashes are
in results. The task-entry TensorFold query at 2026-10-05 23:34:06 UTC remains
`609ca419` / 0.6.5, with a Gemma26 MLX recipe and no matching CUDA target.
Installed-supervisor builds, stages, screens, corrected checkers and focused
controls retire successfully. Raw logs, vectors and snapshots remain external
under `gemma-binding-wrapper-cache-raw`.
