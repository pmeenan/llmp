<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Bounded Gemma prefill plan lookahead

Gemma can build the next prompt chunk's CPU graph and placement plan while the
current chunk executes. The same-binary off/on/on/off control recovered
275 ms (2.36%) from Gemma31's 8K prefill. This addresses part of the remaining
gap; it does not establish reference parity or a dominant explanation for all
remaining cost.

Structural hints carry only the next rows, slots and output mode. Both the
incremental and legacy runtime prompt paths supply them; unequal-slot wave
callers use the same runner API. Invalid hints, optional host-charge refusal,
cached plans and an inline node all fall back to the current path. Scoring and
final prompt chunks retain their full heads.

The optional host capacity charge is acquired before submission. The callback
constructs only descriptors and the unbound plan. CUDA workspace binding,
coverage checks and cache insertion happen after proven current completion;
MMA binding queries CUDA occupancy and therefore cannot run in the callback.
The temporary charge stays live through binding and coverage, then transfers
immediately before the ordinary cache charge. No future inputs or KV backing
are prepared while the current job runs. A failed prediction cannot invalidate
a successful current prefix. Dropped plans and Clear retain their existing
reclaim and cursor behavior.

| Gemma31 both256, state-only, normmul off | 8K prefill, s | 32 decode units, s |
| --- | ---: | ---: |
| Fresh reference first / repeat | 10.8302 / 10.8740 | 3.119015 mean |
| Task51 helper before | 11.6145 | 3.18458 |
| Task52 on first / repeat | 11.8712 / 11.4374 | 3.17947 / 3.20001 |
| Same new binary off first / repeat | 11.6488 / 11.6776 | 3.18199 / 3.19148 |
| Same new binary on first / repeat | 11.3660 / 11.4103 | 3.19363 / 3.19311 |

The first five-arm comparison was unresolved: its candidate mean was 0.34%
slower than the old helper and the two candidate arms differed by 434 ms. Both
samples remain recorded. The following same-binary control isolates lookahead
from the runtime refactor: mean 11.6632 s off versus 11.38815 s on, a 2.36%
reduction. Decode changes by +0.21%; no decode benefit is claimed. Comparing
the later on mean with the preceding fresh reference mean gives a qualified
4.94% remaining prefill gap, not a newly reference-bookended result.

Each on arm attempted, built and cached all 31 future plans without refusal.
The decisive on arms spent 214.200 / 213.492 ms in CPU graph/placement builds.
Those durations describe callback host work, not GPU kernel activity. Off
arms report zero attempts. Remaining serialized binding, state growth,
submission and GPU work are separate costs. These timing percentages describe
the optional both256 research policy, without an ordinary-serving percentage.

The same a5ea helper transferred privately to Spark B preserves this benefit
for Gemma26 all1024. Off 2.61482 / 2.62515 s versus on 2.58653 / 2.57208 s gives
means 2.619985 / 2.579305 s: 40.68 ms (1.55%) recovered. Decode changes by
−0.32%. Each on arm builds and caches all seven predicted plans, with no
refusal, spending 46.5643 / 44.2414 ms in the CPU build. All four arms retain
exact complete heads, 592,445,440 initialized state bytes and 32 choices.
This is a same-binary transfer control with no fresh reference comparison.

All 31B native arms retain exact prior full prefill/final vocabulary heads,
1,740,636,160 initialized state bytes and 32 greedy choices. Reference
bookends match their fixed heads and each other. All retained heads are finite;
32 native/reference choices agree. Final native/reference heads are byte-exact;
the known prefill frontier difference remains max raw delta 0.41361475. This
checks two complete vectors per arm, not every decode vector, corpus quality
or the open optimized C4 gate.

Six focused runtime controls cover tiled remainder, scoring, cancellation,
capacity retry and completed-prefix publication. Three node controls exercise
the host callback under held and standalone requests, clean rejection and the
inline fallback. Four real-model controls cover ordinary and optimized 26B
and 31B scalar/unequal waves, exact state and continuation, and captured replay.
A separate actual 31B control refuses the optional charge while completing the
current prefix, then abandons a valid future plan and verifies DropPlans/Clear
return host accounting and cursors to their expected values. Independent
source reviews pass. Local header/REUSE, boundary, format and diff checks pass.
The owner deferred full regression suites until selected performance work is
settled; no routine full suite was run for this incremental change.

Reproduce with the existing manual helper:
`llmp_gemma_prefill ARTIFACT IDS NEW_OUT 31 both 256 normmul-off state-only lookahead-off|lookahead-on`.
For the 26B transfer use `26 all 1024` with the same trailing modes. It exercises
the production prefill API. Both engines discard six warm rows,
Clear, pay 8,192 prompt rows, append three untimed anchors and complete 32 forced
units. Both publish one prompt head. Snapshot writes, finite scans and hashes
remain outside paid timers. Reference uses the original pinned ring launcher,
context 16,384, F16 KV, 31B local/global 1,280/16,384 and SWA/unified false.
The 26B transfer uses local/global 2,048/16,384 cells.

Gemma31 runs on Spark A (`spark-c4e2`); the 26B transfer runs on Spark B
(`spark-56f5`). Reference launcher, approved model/source pins and the
matched cache recipe are inherited from the
[31B ring comparison](../gemma-swa-ring-h1/README.md) and
[26B ring transfer](../gemma26-swa-ring-transfer/README.md). The pinned original
image is `837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7`.

Source base is 1ceb7d2; the candidate helper is a5ea8065, preserved old helper
db3e85a3, and receipt 995ee819. Native uses NVCC 13.4.92/toolkit 13.4.2; the
original llama.cpp b29c606e image uses CUDA 13.3.0. The task-entry TensorFold
check remained 609ca419/0.6.5, with Gemma26 on MLX and no matching Gemma31 CUDA
recipe. [Aggregate results](results.json) retain full measured identities and
checks. Raw job logs, vectors and snapshots stay external. The separate next
norm-policy diagnostic is not part of this lookahead comparison.
