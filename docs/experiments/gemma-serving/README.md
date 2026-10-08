<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Bounded Gemma26 serving route

The approved 26B-A4B artifact now plugs into the existing native runtime driver
for chat and literal completions, target likelihoods, continuous independent
cohorts, exact continuation, spill and kept restart. Cohorts admit up to twelve
owners and execute scalar completed units. This slice does not select joined
model arithmetic or establish model support, optimized batching, competitive
latency, long-context qualification or representative reference quality.

The route keeps device masks on and optional norm, shared-Q8, row-invariant and
RoPE/cache-store policies off. Prefill is capped at 128 rows and context at
262,144; the uncalibrated fallback uses one owner. The 31B serving profile,
assistants/speculation and generated thought/tool parsing are unavailable.
Plain chat disables thinking in its actual prompt. Extra chat channel stops
are request-local; literal completion stop authority and teacher forcing remain
separate. The generic HTTP parser's existing historical-tool-message refusal
is retained; the broader template controls below are normalized Conversation
controls, not proof of an HTTP tool-history route.

## Identity and comparison contract

Implementation base is `d5539c8aae40c82d21892a23d4bd467a60aed931`. The installed
artifact is
`4ddb360c9ce08f1e984ab304b6af918be44246d52346734066b06443f7c249d3`, produced by
the checked importer from the approved mixed Q4_K/Q5_1/Q8_0 GGUF. The retained
actual template has SHA-256
`845f1ee48e39fc942fe190da9df6a1c5db229e17a96ea08966ad1c9274e73d1b`,
18,924 bytes, and selects native `gemma-4-unsloth` by probe.

The task-entry TensorFold observation was
`609ca419abecebdc5a059498a613680bd3aa847f`, version 0.6.5; primary coverage
still placed Gemma26 on MLX. The existing pinned llama.cpp CUDA reference
`b29c606e28a01b1bc8c1351026a0fa6e616bf6c4` remains applicable to the later
whole-model qualification task, which must refresh its own comparisons under
[the reference policy](../../reference-comparisons.md). This slice ran no new
reference inference or whole-model quality/performance benchmark.

The independent template oracle uses Jinja2 3.1.6's immutable sandbox with
trim/lstrip, loop-control and JSON filters, executing the actual retained
845f template. Fixture inputs preserve outer OpenAI-style JSON argument strings;
the oracle and native test helper each deserialize them exactly once into the
same normalized Conversation values. JSON string, null and list arguments
are therefore distinct values supplied to the template. These inputs are not
a claim about another reference engine's HTTP normalization. All 65 resulting
byte hashes agree with both native and checked interpreter rendering; their
marked-control token IDs agree using the actual target tokenizer, including
one initial BOS. Gold hashes come from pinned Jinja, not native output.

## State and memory contract

Checkpoints validate owned logical positions separately from initialized
extent footprints. Save boundaries must equal the completed native ledger.
Gemma kept checkpoints additionally require cursor equal to boundary position;
a kept main conversation requires cursor equal to its complete token history,
with no speculative pending row.
a redigested record containing cursor5/boundary6 is omitted before file adoption,
even though both positions fit the same 256-cell padded cache footprint.
The ordinary kept conversation remains usable and resumes exactly.

Preparation protects selected peers and funds the destination footprint before
copy. Execution and state publication remain blocked while restoration is
pending. Successful retired copies must cover every required logical range
before completion can change the ledger. Missing or partial copies and a
mismatched completion position refuse. A clean capacity refusal before copies
preserves the previous completed prefix, usable state and peer leases. Unproven
copy retirement uses the existing cohort fault/quarantine contract.

Native pinned output backing is independently cataloged during setup:
128 rows times 262,144 F32 logits is 134,217,728 bytes. The serving path narrows
each native scalar unit to one frontier row; literal scoring similarly uses
one full-vocabulary row per completed token, not a 128-row CPU score matrix.
The startup host-workspace guard additionally counts each owner's retained
frontier, prepared result and maximum sampling-candidate capacity before those
heap vectors grow. TopK can reserve twice its requested count; Gemma reserves
exactly twice the vocabulary's candidate capacity before first sampling,
from empty storage. Later rows cannot reallocate that capacity, eliminating
old/new allocation overlap. Every completed, paused, cancelled or failed-setup
session Close frees it, so it cannot outlive the active-model workspace floor.
Twelve owners require 125,829,120 bytes with the pinned 16-byte SamplingCandidate,
separately from the measured native 2 MiB source staging and the driver's
request memory. Native all-row diagnostic callers
fund their own heap copies; the 128-row replay control explicitly funds both
128 MiB copies before allocation. No new uncounted memory margin is introduced.

## Bounded controls

- Seeded solo results equal scalar cohort results for 1, 2, 4, 8 and 12 owners;
  retained plans and captured graphs replay. This verifies independence, not
  optimized joining or model-reference batching quality.
- Snapshot rollback, occupied-state replacement, turn checkpoint reuse,
  spill/restore, kept restart and a forged cursor/boundary checkpoint retain
  exact own continuations. Selected peer state remains intact.
- A seeded streamed generation pauses at a completed unit, spills its KV,
  lets a peer progress, restores and resumes with the same complete output
  and each token callback exactly once.
- A 128-token target-likelihood session executes full-vocabulary rows,
  interleaves with a seeded peer and resumes midway without repeating a score.
  Initial BOS, expected target IDs and literal/chat stop separation are checked.
- Actual large top_k values 200,000 and 262,143 use the fixed funded candidate
  capacity; normal, cancelled and failed-first-sampling retirement free it.
  A peer's capacity remains independent.
- Shared driver scalar-unit publication retains an earlier owner's completed
  result when a later owner cleanly refuses after preparation. Retry uses the
  same absolute sampling positions and publishes no repeated callbacks.
- Public metadata controls refuse missing, duplicate, short, unaligned and
  wrong-region footprints without inferring positions from padded bytes.
  Actual device controls refuse uncompleted/partial restores and clean growth
  pressure before exposing output or changing the prior ledger.

The actual HTTP service also completed seeded literal requests at concurrent
1/2/4/8/12 widths with exact solo response content, chat SSE with exact
non-streamed content and `[DONE]`, and a 128-token echo/prompt-likelihood request
with 127 scored rows, initial null score and zero generated tokens. Runtime
SIGTERM retirement exited zero. Concurrent request counts are admission/control
coverage, not a claim that every request occupied a slot simultaneously.

The final Spark-b locked build is `gemma-serving-build16`. Source/test identity
is `d386521c519179992a3703ac22aa0d22751e1697e1d483b4c7a32f7c3f9713f0`,
computed from the 17 runner/serving source and test paths and their file
SHA-256s (path, NUL, SHA, newline), before the test-registration change below. The runtime binary used by the final HTTP control
has SHA-256 `59b031058d860f4ad58e777626a5966e4cf17011d8b9a12a07ad80a91c246d41`.

Focused `gemma-serving-final2` exercised 119 controls (20 GPU, including 12 model controls,
plus 99 unit controls): 118 passed and one new KV fixture failed because it compared a
full-extent snapshot with differently packed logical ranges. The corrected
fixture copies the identical original footprint to separately funded pinned
storage. `gemma-serving-final3` then passed that control, the large-top_k
lifetime control and the 128-row output replay control, 3/3 with no skips.
The final fixture SHA-256 is
`a5e8d6bfa5b877d90f26b714d3e1004b2a1f930efd085149e61e971c106df351`.
`gemma-serving-http2` passed the final production HTTP controls and exited
zero. Local REUSE, 1,276 header and 373 boundary checks, changed-source format
and diff checks passed. Raw jobs, HTTP samples, renderer script and logs
remain outside Git; these results use the GB10 native profile and installed
supervised jobs, not a discrete GPU run.

The integrated parallel suite exposed independent full-model test processes
competing for the Spark's physical memory: four tests were killed by global
OOM, and SSH became temporarily unresponsive. GPU tests marked `MODELS` now
share a CTest resource lock, as recorded in [RE-047](../../rough-edges.md#re-047-parallel-full-model-gpu-tests-exhaust-spark-memory--2026-10-04-status-worked-around).
This changes test scheduling only; it does not establish concurrent model
execution or alter the route's scalar scheduling.

The final integrated `gemma-serving-root-final3-full` locked Spark-b suite
passes 1,613/1,613 tests, including 287 GPU and 36 model tests, with no skips
in 184.32 seconds. The generated registrations give all 13 full-model GPU
test entries the shared lock, including the two parameterized policy cases
within their wildcard entry. Final local REUSE, header, boundary, changed-source
format and diff checks also pass.
