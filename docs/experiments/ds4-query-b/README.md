<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepSeek query-B consumer factor

2026-10-02, Spark A (GB10), SDK `aarch64-e0a0c85c42806fb1`. Restore
one native consumer in the complete literal ds4 pipeline to isolate its
effect; quality differences are recorded independently of performance.

Community artifact `cd39d504dc2dbfe911a4a521fa8efc8053dc3e80e99738a9b25fa6b70c97a1ac`,
pinned 8,192-token input, two 4,096-row chunks, 43 layers and final-only
full head. Query-B changes from original DenseD2r to the existing native
borrowed-D4 Q8 MMQ consumer, T4096/K1024/M32768. Original D4 production
remains paid; native also pays its guard/sanitizer and extra Run boundary.
Raw/aligned weight operands and canonical F32 output are unchanged, with
no conversion. All other stages, especially output-B, remain original.

| Paid arm | Wall plus complete-head copy (s) |
| --- | ---: |
| Original before | 7.555047535 |
| Native query-B | 7.814307481 |
| Original after | 7.587525148 |

Native throughput is **3.110% lower** than the mean original bookends;
bookend movement is **0.430%**. Both arms retain the paid 512 MiB pool,
99,123,342,848-byte execution budget and 93 scheduler jobs. Native query-B
needs zero extra scratch at this shape. Its numerical time rises to
7.705943068 s from original 7.446831395 / 7.479878940 s. This is a small
consumer effect, below D-085's coarse 10% regression threshold, and does
not explain a substantial whole-engine gap.

All four original warmup/sample/return heads match the pinned full-head
golden byte for byte. The two native heads repeat exactly, with all six
129,280-F32 outputs finite. Native changes 129,279 logits, max absolute
difference 1.003210068, RMS 0.135054002; final argmax remains 554. No task,
perplexity, longer-context quality or default acceptance is claimed.

One process prepares and loads weights once, then runs three warmups
and original/native/original samples. Actual successful child
time is 352.229 s: preparation 294.996 s, load 6.407 s and retirement
3.175 s. Three private host compiles/link take 10.130 s; no numerical
unit, suite, production implementation or default changes.

The first preparation attempt refused the unchanged allocation guard
before loading weights or running a head. Existing allocation-only
controls and advice on only the inactive Mia packed PLE restored guard
admission without removing any operand or study allocation. The paired
CUDA/free-memory evidence and inconclusive root-owned-file mincore
vectors are recorded in [RE-041](../../rough-edges.md#re-041-cuda-current-free-memory-can-exclude-reclaimable-inactive-gguf-file-cache--2026-10-01-status-worked-around)
and the [CUDA upstream log](../../upstream/cuda.md#cuda-current-free-memory-excludes-inactive-file-cache-re-041).

Recommendation: park this private consumer factor and continue with
remaining Q/KV producers, compression/indexer, shared FFN or HC producers.
The complete Q/KV chain is still open. Borrowed-D4 remains a reusable
checked consumer seam; this experiment establishes no cross-family gain.

Actual receipts, commands, six raw heads, failure and cache controls stay
outside Git at `spark:~/scratch/m3-ds4-query-b-factor-r1/`; local mirror
`/home/pmeenan/scratch/m3-ds4-query-b-factor-records/`. Successful model
rc0/reaped, normal retirement and final strong gate 117.204 GiB clear.
