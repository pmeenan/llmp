<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dense31 ordinary-product norm C4 first screen

Ordinary product dispatch with both checked norm policies still fails the
natural C4 stock comparison. Native C4 repeats are byte exact, but 15/128
argmax choices differ from the primary fresh ubatch128 reference. No candidate
is selected. Production defaults remain off; no context or concurrency ladder
follows this negative screen. The prior
[row-invariant/norm experiment](../gemma-joined-norm/README.md), failed ordinary
controls and their own freezes remain unchanged.

## Measured math and paid work

Four independent owners use the same supplied natural prefixes of 64..67 IDs.
After eight discarded warm units, reset/prefill and three untimed anchors, the
paid region processes 32 forced completed units per owner. All 128 full heads
and argmax publications are inside the timer. Native uses a real four-segment
wave; stock uses a genuine four-sequence C-API batch. Both engines have 256
effective context per owner. Stock physical ubatch128 can prefill each owner's
64..67 IDs in one chunk, reducing that recipe difference.

Candidate joined-first/repeat heads are byte exact, SHA-256
`4e92da753659b62e186c208bd0c89bda5cff5c084ee9314cd5770fd246d3e1d8`.
Own freeze `75f47c1f…` completed before every new stock acquisition/comparison.
The unchanged row-invariant/norm control equals the prior full-head identity
`7cd4511d…`. Disabling the row override changes 15/128 native argmax choices
from that control; this is a policy change, not own-repeat noise. No scalar
comparison or C1=C4 arithmetic assertion is part of this axis.

| Stock physical ubatch | Acquisition | Strict differences / 128 | Outside zero own margin movement | Max full-head raw delta | Selected TV max | Selected mean target-NLL delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 128 | Fresh primary bookend | 15 | 15 | 29.084824 | 0.673273 | +0.169778 |
| 4 | Retained compatible recipe | 32 | 32 | 26.363071 | 0.934942 | +0.634896 |
| 32 | Retained compatible recipe | 22 | 22 | 27.258722 | 0.946984 | +0.519191 |

All complete heads are finite. No complete row equals stock bytes; none of the
strict differences is an exact stock tie. Positive stock winner margins are
compared with the measured zero margin movement implied by native full-byte
repeat, not a bound adjusted using stock. Selected TV and target NLL summarize
only eight rows: steps0/7/15/30 and owners0/3. These are not corpus PPL.
[results.json](results.json) retains aggregate metrics without raw argmax or
per-step token/likelihood vectors. Raw heads and IDs stay external.

| Native own acquisition | Paid seconds |
| --- | ---: |
| Unchanged row-invariant/norm control | 3.38718 |
| Ordinary-product norm first | 3.40256 |
| Ordinary-product norm repeat | 3.39685 |

The fresh primary competitive bookend is stock ubatch128 **3.35290 s**,
candidate **3.38459 s**, stock ubatch128 **3.34156 s**. Candidate latency is
**1.1161% higher** than the mean stock bookend. Its full heads still equal the
native own freeze. Retained ubatch4/32 vectors supply additional quality axes,
not fresh timing claims. Each paid arm completes 32 GPU groups/replays and
128 full-head publications, with zero new captures. Retained/working head heap
is explicitly charged 142,606,336 bytes; native state/plan/scratch and pinned
output funding stay separate. This capacity charge is not a physical peak.

## Actual policy and source inference

The candidate selects existing `fuse_norm_rope` and `fuse_norm_add` with
`row_invariant=false`. Actual counters at every fresh 64..67-row prefill and
fresh four-row/four-segment decode plan show 120 norm/RoPE and 120 norm/add
chains, zero row-invariant products and zero legacy norm, RoPE/store, sharedQ8,
MoE or lane steps. The control has the same prefill choices and 411 decode
row-invariant products. Cached reset prefills are not labelled as fresh
selection; final summaries explicitly report last-built policy.

This restores the native ordinary product selector for eligible small-row
calls. Source inspection shows that its quant path follows pinned MMVQ/MMQ
predicates at actual type/device/column count, while row-invariant kernels
force one-column reduction parameters. Stock's GB10 selector admits MMVQ up
to six Q2_K columns and up to eight for other supported quant types. Ordinary
float selection and stock graph fusions also have their own predicates.
Those facts are **source inference**, not an actual per-operation CUDA trace
or proof that native and stock select every same kernel. No eval callback,
keep/alias change or production instrumentation was used to obtain them.
Pinned selector source hashes are in [provenance.json](provenance.json).

The candidate keeps both norm policies during prefill and decode. Disabling row
invariance changes its eligible small-row products; the 64..67-row native
prefills already exceed the row-override bound. Reference physical ubatches
4/32/128 also differ during prefill. This screen does not close the source of
the stock gap. The prior exact 128-row teacher-forced norm evidence does not
qualify natural C4, and earlier row-invariant solo/joined agreement does not
establish ordinary-products C1=C4 equivalence. Matched C1, state/continuation,
physical peak, broader quality and full optimized-batching/model support
remain separate work. No new kernel, graph, runtime option or default lands.

## Provenance and reproduction

[PROTOCOL.md](PROTOCOL.md) fixes source rationale, exact inputs and positions,
funding, timer and stop boundaries. Measured base is `fdf6d1a`; the only
executable change is a closed benchmark `norm` policy mode. The task-entry
primary TensorFold refresh at 2026-10-05 08:28:50 UTC is HEAD
`609ca419abecebdc5a059498a613680bd3aa847f`, version0.6.5; its pinned recipe still
covers 26B-A4B on MLX rather than dense31 CUDA. Same-format pinned stock is the
GB10 comparator.

The approved artifact is `32c92e07…`; raw dense31 GGUF revision
`c1ac76e99d5513b141e8adde7288b85c3f9c32ec` has SHA-256 `9e92cb62…`.
Externally supplied 1,024 IDs have SHA-256 `b2d7aaf6…`, BOS2 once. Complete
checkpoint, corpus and input hashes are in the protocol. Reference pin is
`b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`, original digest image `837fc732…`.
Only the thin C-API client is rebuilt, against unchanged original image math
libraries. Its source/binary, input/position/context recipe and all reference
head hashes authenticate against the original natural C4 acquisition. Fresh
ubatch128 heads equal the retained ubatch128 file.

Synchronize the isolated source with checksum comparison and fresh mtimes,
then run a locked native build under Spark-b's installed supervisor. Supply
`ids31.i32` and exact pinned llama.cpp headers in
`~/.local/share/jitllm/gemma-joined-norm-ordinary`, obtained with `git show` at
the pin without changing the owner checkout. Invoke `reference.sh build` under
the same supervisor. Wrapper scratch/source/model paths are owner-local
reproduction paths, not application configuration.

Create a fresh external `native/` directory and record source/binary/receipt
identities in `native/source-identities.json` before acquiring arms. Invoke
`jitllm_gemma_joined ARTIFACT OUTPUT 31 4 joined POLICY IDS` first with
`rows-norm-control/rows-norm`, then `joined-first/norm` and
`joined-repeat/norm`. Run `own_freeze.py NATIVE_DIRECTORY`, which refuses an
existing freeze path. Only afterward run `reference.sh 31 REF_FIRST 4 128
ids31.i32`, the joined candidate and `reference.sh 31 REF_REPEAT 4 128
ids31.i32`. Require the intervening candidate bytes to equal its own-frozen
file. Invoke committed `../gemma-joined-serving/analyze.py compare
NATIVE_DIRECTORY REF_FIRST REF_REPEAT FREEZE_SHA`. Retained4/32 comparison
requires the authenticated same recipe; source identity alone is insufficient
for a different prompt/position/physical-ubatch claim. Builds, inference and
full-vector scans use the installed GPU supervisor and official waits.

The Spark-b locked build, three native arms, own freeze, fresh primary
bookend and two retained comparisons completed successfully. Raw logs stay
external. Local REUSE/headers (1,403 files), boundaries (386 sources),
changed format/diff/script syntax and signed-zero/tie/non-finite controls passed.
A routine whole-unit suite was not run for this benchmark-only
first screen under the experimental-comparison workflow.
