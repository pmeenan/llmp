<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dense31 natural C4 norm/row first screen

The checked norm chains plus row-invariant products preserve same-policy native
solo/joined heads at this shape, but fail the stock comparison. No candidate is
selected. Production defaults remain off. This benchmark-only axis leaves the
[original joined experiment](../gemma-joined-serving/README.md), its failed
quality results and own freezes unchanged.

## Measured result

Four independent owners receive natural War and Peace prefixes of 64..67 IDs,
then identical supplied next IDs. The timer includes 32 completed units per
owner, all 128 full-vocabulary heads and argmax publication. Both engines use
256 effective context per owner. Native joined execution uses one real
four-segment wave; stock uses genuine four-sequence C-API batches.

All four candidate scalar-a/joined-first/joined-repeat/scalar-b head files are
byte exact, SHA-256 `7cd4511d44af8fff4aa88e2b94ba2b2eed792fdd3f3a196e0cbc41c3244951b4`.
Own freeze `94bf17dd…` precedes every new stock acquisition and comparison.
Its zero repeat movement is measured from complete bytes, not a tolerance
chosen from stock. The unchanged rows-only control remains byte exact to the
original natural ordinary/rows control, SHA-256 `769e1d4e…`. Adding norm policies
changes 36/128 native argmax choices from that control. This movement is an
arithmetic policy change and is not own-repeat noise.

| Stock physical ubatch | Acquisition | Strict argmax differences / 128 | Outside zero own margin movement | Max full-head raw delta | Selected TV max | Selected mean target-NLL delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4 | Fresh bookend | 33 | 33 | 26.429946 | 0.896528 | +0.662777 |
| 32 | Retained compatible recipe | 25 | 25 | 23.802520 | 0.926950 | +0.547072 |
| 128 | Retained compatible recipe | 15 | 15 | 28.164619 | 0.573557 | +0.197659 |

No complete row is byte exact to stock, and none of the strict differences is
an exact reference tie. All 128 complete rows are finite. TV and target NLL
summarize only eight labelled rows: steps 0/7/15/30 and owners 0/3. They are not
corpus PPL. The complete-head counts and selected summaries are retained in
[results.json](results.json); raw heads, IDs and per-step output remain external.

| Native own acquisition | Paid seconds |
| --- | ---: |
| Unchanged rows-only joined control | 3.40442 |
| Candidate scalar-a | 12.38300 |
| Candidate joined-first | 3.39027 |
| Candidate joined-repeat | 3.38695 |
| Candidate scalar-b | 12.40770 |

The fresh competitive bookend is stock ubatch4 **3.35283 s**, candidate
**3.41039 s**, stock ubatch4 **3.35049 s**: candidate latency is **1.7523% higher**
than the mean stock bookend. Its heads still equal the native own freeze.
The retained ubatch32/128 heads supply quality controls, not fresh timing claims.
Paid joined work is 32 GPU groups/replays and 128 completed units/full heads;
scalar work is 128 groups/replays. Each paid region captures zero new graphs.
The retained/working head heap is charged 142,606,336 bytes, separately from
native state, workspace and pinned-output funding. This is not a physical peak
measurement.

## Actual policy scope and limits

The candidate uses existing `row_invariant`, `fuse_norm_rope` and `fuse_norm_add`.
Each fresh 64..67-row prefill and fresh scalar-one-row/joined-four-row decode
plan selects 120 norm/RoPE and 120 norm/add chains. Prefill selects zero
row-invariant products because rows exceed eight; decode selects 411. Legacy
two-node norm fusion, RoPE/store, sharedQ8, MoE fusion and lane steps remain zero.
First-build rows/segments are checked before reporting counts. Final summaries
are labelled last-built policy; cached reset prefill is not reported as a fresh
selection.

Both prefill and decode norm arithmetic change. Reference physical ubatch also
changes prefill math. This screen does not isolate the source of the stock gap
or transfer the earlier exact 128-row teacher-forced proof to natural joining.
No context/concurrency ladder follows this negative candidate. State/spill,
broader quality, physical peak, C12 competitive batching and full model support
remain separate qualification work. No kernel, planner, runtime option, config
schema or production default changes in this packet.

## Identities and reproduction

[PROTOCOL.md](PROTOCOL.md) fixes exact inputs, positions, paid work, funding,
closed contracts and TensorFold refresh. [provenance.json](provenance.json)
records all measured source hashes, native/C-API binary and receipt identities,
reference compatibility and official completion/log hashes. Measured base is
`422c687`; the only executable change is the benchmark's closed `rows-norm`
option and first-build diagnostic reporting.

The prepared dense31 artifact is `32c92e07…`; same-format raw GGUF revision
`c1ac76e99d5513b141e8adde7288b85c3f9c32ec` has SHA-256 `9e92cb62…`.
The 1,024 little-endian IDs are supplied externally, SHA-256 `b2d7aaf6…`, BOS 2
once. Raw checkpoint, corpus and complete input identities are in the protocol.
The pinned original stock image is digest `837fc732…`, llama.cpp
`b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`. This packet rebuilds only its thin
C-API client against the original image libraries. The stock client source,
client binary, input/position/context recipe and retained head identities match
the original natural C4 acquisition; the fresh ubatch4 heads also equal that
retained ubatch4 file.

On Spark-b, synchronize the isolated source with checksum comparison and fresh
mtimes, then use the installed supervisor for a locked native build. Supply
`ids31.i32` and the exact pinned headers in
`~/.local/share/jitllm/gemma-joined-norm`; obtain headers from `git show` at the
pinned llama.cpp revision, without changing the owner checkout. Run
`reference.sh build` under the same supervisor. Reference paths in this wrapper
are the experiment's owner-local paths, not application configuration.

Acquire into a fresh external `native/` directory, recording source and binary
hashes in `native/source-identities.json` before the arms. Run the existing
`jitllm_gemma_joined ARTIFACT OUTPUT 31 4 MODE POLICY IDS` with these five arms:
`rows-control/joined/rows`, `scalar-a/scalar/rows-norm`,
`joined-first/joined/rows-norm`, `joined-repeat/joined/rows-norm`,
`scalar-b/scalar/rows-norm`. Invoke `own_freeze.py NATIVE_DIRECTORY`; existing
freeze paths refuse replacement. Only then run `reference.sh 31 REF_FIRST 4 4
ids31.i32`, the joined candidate, and `reference.sh 31 REF_REPEAT 4 4 ids31.i32`.
Compare candidate bookend bytes to the own-frozen file, then invoke the committed
`../gemma-joined-serving/analyze.py compare NATIVE_DIRECTORY REF_FIRST
REF_REPEAT FREEZE_SHA`. Every build, inference and full-vector scan uses the
installed GPU supervisor and official wait. Retained vectors may be compared
only after authenticating the same recipe; source hashes alone do not authorize
a different prompt, position or physical ubatch as a matched comparison.

Spark-b's locked build, five native arms, freeze, three-arm fresh bookend and
two retained comparisons completed successfully. Raw logs and vectors remain
external. Local REUSE/headers (1,389 files), boundaries (384 files), changed
format/diff/script syntax and signed-zero/tie/non-finite controls passed.
A routine whole-unit suite was not run for this benchmark-only first
screen under the experimental-comparison workflow.
