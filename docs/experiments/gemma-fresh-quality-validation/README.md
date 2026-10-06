<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Fresh Gemma26 schedule and heldout corpus control

The independent heldout history passes the predeclared operational-margin and
PPL gates. All 17 strict token disagreements fit the unchanged bound frozen from
native scheduling variation on a different history; relative PPL increases
0.1970%. The strict zero-difference gate still fails. Earlier strict failures
remain failures, and this bounded control selects no runtime default or admission.

The measured source is main `c3300ab` plus the separately **uncommitted** runtime
candidate 15 (`c40ce2bf…`) and this manual helper. The helper uses existing runner
APIs without changing runtime, kernel or selection policy. Its measured source
and binary identities in [results.json](results.json) do not describe an adopted
main runtime. The [prospective method](PROTOCOL.md) was fixed before the new
texts, tokens or likelihood outputs.

## Work and result

Untouched Wiki histories 13 and 14 use the same fixed paragraph rule after byte
offsets `65536 * history`. Their text manifest was frozen before tokenization;
native and public vocabulary-only tools agree on each complete 1,024-token
prefix with one BOS. Each history is independent: all-vocabulary FP64 NLL scores
exactly 1,023 within-history transitions. Row 1,023 has a predicted choice and
no next-token likelihood target.

Both native schedules configure maximum rows 1,024, context 4,096, F16 local
capacity 2,048, global capacity 4,096, one slot and full-head capacity. Each funds
the same 1 GiB publication vector. Only the host `Chunk` schedule changes between
128 and 1,024. Every actual chunk selects 121 plain norm, 60 norm/RoPE, 90
norm/Add, 30 routing and 30 reduction fusions; shared and row products remain
off. Owner attention and state-only pruning are unused. Lookahead is enabled
but not exercised because these direct calls supply no forecast.

History 13 freezes two fresh complete finite own repeats for each schedule,
then compares the first outputs. The bound is the existing nearest-rank p99 of
`abs((A[top1]-A[top2])-(B[top1]-B[top2]))`, with A = 1,024, B = 128 and lower-ID
tie resolution for A's top two. It is frozen before any reference or heldout
likelihood output. The maximum is diagnostic and does not change the bound.

| History 13 native scheduling comparison | Result |
| --- | ---: |
| Exact own-repeat rows, each schedule | 1,024 / 1,024 |
| Exact rows across schedules | 0 / 1,024 |
| Different native choices across schedules | 209 / 1,024 |
| Frozen p99 margin movement | 9.5025564432 |
| Maximum margin movement, diagnostic | 18.5776886940 |

This large bound measures supported scheduling variation rather than repeat
noise. It is transferred unchanged to history 14, with no recalibration or
outside-bound exception. History 14's two native processes freeze all finite
heads byte for byte before the reference pair. Stock uses the same raw GGUF,
IDs, physical C1, context 4,096, batch/ubatch 1,024, F16 local/global capacities
2,048/4,096, normal ring, independent KV, ordinary dispatch and complete heads.
Both stock repeats are exact; public teardown, owned-container absence and
official retirement are checked before comparison.

| History 14 complete comparison | Result |
| --- | ---: |
| Scored transitions / prediction rows | 1,023 / 1,024 |
| Native / reference exact own-repeat rows | 1,024 / 1,024 |
| Exact native-reference rows | 3 / 1,024 |
| Strict positive-margin disagreements | 17 |
| Maximum reference margin of a disagreement | 0.2220959663 |
| Disagreements outside frozen p99 | 0 |
| Native / reference mean NLL | 11.5046507708 / 11.5026826642 |
| Native / reference PPL | 99,175.9447 / 98,980.9478 |
| Relative PPL increase | +0.1970044551% |
| Maximum / mean full-vocabulary TV | 0.2309053762 / 0.0118788858 |

The unchanged operational bound and the separate PPL increase ≤3% requirement
both pass. The zero-difference requirement fails on all 17 disagreements.
Both engines' high absolute PPL remains an observation with no inferred cause.
This one calibration/heldout pair supplies no corpus-wide, batching, assistant,
performance or retrieval-at-depth qualification.

## Reproduction and provenance

Build the manual `jitllm_gemma_quality_schedule` target and run
`ARTIFACT IDS_I32 NEW_OUTPUT_DIR 128|1024` under installed Spark supervision.
The normal build path has its build RPATH; a copied private helper must use the
qualified `LD_LIBRARY_PATH=$build/spark-native/cublas` binding and pass the SDK
version/identity guard before execution.
Follow [PROTOCOL.md](PROTOCOL.md): freeze texts and checkpoint-specific IDs,
authenticate successful completion and full finite own repeats, freeze history
13's p99 receipt, then acquire and compare history 14. The supplied raw input
and receipt identities must match [results.json](results.json); raw texts, IDs,
full heads, row records and logs stay external.

Measurements ran on Spark A (`spark-c4e2`, GB10), using the pinned native SDK
receipt `e57a1624…` and qualified cuBLAS 13.8.0.4-1 binding. Driver 580.178.04 is
inherited from the qualified Spark environment, not a new driver probe. The
unchanged public client `0a7cfdbc…` runs in image `837fc732…`. The prepared
artifact, inherited raw GGUF identity, complete source frame, text/input
manifests, bound and output hashes are recorded in the aggregate.

The fixed-capacity causal-mask test and three synthetic metadata controls
pass. Every admitted acquisition and analysis job retires successfully; the
initial native attempt failed before model setup on an unqualified library
search path and is retained in the official aggregate. No full suite was run
under the owner's experimental override. Measurement completion and the
individual numerical gates are recorded separately.
