<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dense31 packed C4 attention first screen

Packing four owner streams makes the candidate **byte-exact with fresh original
llama.cpp across all 128 complete heads** in this closed dense31 screen. The
unchanged norm/ordinary-product control retains 15 positive-margin greedy
mismatches. Packing is slower: 3.65318 s against reference bookends of
3.24939 and 3.25803 s, **12.2774% more latency** than their mean. This is a
benchmark-only correctness diagnosis; production selection stays unchanged.

[Protocol](PROTOCOL.md), [aggregate results](results.json),
[provenance](provenance.json). The earlier [real-operand attention control](../gemma-attention-c4/README.md)
established original/native four-stream MMA equality while segmented vector and
segmented MMA differed. This full-head experiment changes local dispatch and
local/global stream geometry together. It does not isolate either factor alone.

## Closed graph derivative

Measured production base is `7842403`. A separate manual helper derives from
[gemma_joined.cc](../../../benchmarks/gemma_joined.cc), preserving its paid
staging, completed full-vocabulary publication, argmax and graph lifecycle.
It adds funded initialized-state witnesses and complete finite checks **outside
the timer**. The original helper, engine, graph builder, planner, kernels and
source lock are unchanged. Both process modes use the same new binary.

The candidate's benchmark link wrapper accepts only the complete approved
31B graph: width 5,376, context 256, max_rows 128, four independent owners,
one query row each, equal local/global read widths 256, four head outputs and
no feature export. Other shapes call the original builder; the first eight
fallback samples are logged. Thus every 64..67-row C1 prefill is unchanged.
The environment flag `LLMP_GEMMA_PACKED_C4=0|1` is fixed for the whole process,
so a cached plan never changes mode.

Each of the 60 layers authenticates four original attention descriptors and
eight K/V `SET_ROWS` writers. Raw K/V views reference those write results and
the exact owned cache leaves, making all eight writes DAG dependencies before
packing. Balanced dimension-3 CONCATs pack raw Q/K/V; a subsequent permutation
yields Q `[D,1,heads,4]` and K/V `[D,256,kv_heads,4]`. Cache pitches are
`[2,kv_width*2,D*2,kv_width*256*2]` bytes. Separately padded F16 masks become
`[256,32,1,4]`. There is no assumed virtual-address pitch between owners.

One checked attention node produces `[D,heads,1,4]`; owner views preserve the
original flattening/output order and named readers. Every old FLASH node is
removed from executable graph order, every consumer/view root is rewired,
and write-before-attention order is checked. Extra descriptors enter the
existing TensorArena/SizedArena estimates, and normal placement/Setup funds
CONCAT outputs, activations, scratch and held execution closures. There is no
new provider, stream, paging policy or unmanaged scratch.

Local D256 has GQA 2 and `ne[3]=4`, which excludes the native one-stream vector
selector and admits existing group-2 MMA. Global D512 already uses MMA; packing
changes its stream-K work geometry. Both native modes retain exactly 120
norm/RoPE and 120 norm/residual selections. Row-invariant, sharedQ8, legacy
norm fusion, RoPE/store and MoE counts remain zero. Broad fusion stays off.

## Inputs, calibration and full heads

Both engines use approved same-format dense31 weights, F16 KV, four sequences,
context 256 per owner, and the exact supplied 1,024 IDs with SHA-256
`b2d7aaf6aa2ef06d82591a3794f36640e192f429539ec934fd74bf4d81df1610`.
The [preceding protocol](../gemma-joined-norm-ordinary/PROTOCOL.md) records the
checkpoint and War and Peace corpus identities. Owner i prefills IDs [0,64+i),
discards eight warm waves, resets/prefills identically, then completes three
untimed anchors. Paid wave s processes ID/position 67+i+s; its likelihood target
is ID 68+i+s. All 32 waves publish 128 complete 262,144-logit heads. Disk output,
finite scans and state witnesses remain outside both paid intervals.

Control and candidate first/repeat were acquired before the fresh oracle.
Own freeze SHA-256
`09e8cf2878db6f1a1194d22dabc5e6c5aca0ac70d8470459e0eabacb822f748b`
authenticates both policies' exact full-head repeats and four initialized-state
repeats. Each owner witness has 120 ranges and 230,686,720 bytes, including
initialized padded cells; completed endpoints are 99/100/101/102. A reusable
node-owned pinned witness is funded before admission and reused only after
completed copies. Failed/unretired work retains its owner lifetime bundle.
Neither unchanged state across decode nor control=candidate state is asserted.

The original comparator uses unchanged b29 C-API client/math libraries in
image digest `837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7`,
physical C4, batch/ubatch 128, fusion enabled and CUDA graphs allowed. Its
source, public headers, binary, image libraries and ID bytes were authenticated
before the bookend. Prepared manifest/index metadata and the approved raw-model
size/pin were checked; verified historical model payloads were not rescanned.

| Native mode versus fresh original | Full heads byte-exact | Strict greedy differences | Maximum raw-logit delta |
| --- | ---: | ---: | ---: |
| Unchanged norm/ordinary control | 0/128 | 15/128 | 29.0848241 |
| Packed candidate | 128/128 | 0/128 | 0 |

All 15 control misses have positive reference-winner-over-native-choice margins
against its frozen zero own-repeat movement; none is a reference tie. The first
wave has 1/4 misses, the first three waves 2/12, and the later 29 waves 13/116.
The candidate is exact in all three groups. Its complete 134,217,728-byte head
file and both fresh reference files share SHA-256
`9de4f80bf3fc9a8f37de296a10933c7ee91a15005a4a62dedbd22c353150cb7e`.

Sixteen explicitly selected likelihood rows cover steps 0/1/2/31 and all four
owners. Control maximum TV is 0.6396373 and maximum **absolute** target-NLL
delta is 1.4229461; signed mean delta is recorded separately in results.
Candidate TV/NLL deltas are zero. These fixed rows are not corpus PPL or a
whole-model quality allowance. No tolerance was widened.

## Paid result and remaining work

| Arm | Seconds for 32 waves / 128 published heads |
| --- | ---: |
| Fresh original A | 3.24939 |
| Packed native | 3.65318 |
| Fresh original B | 3.25803 |

Earlier same-job native control first/repeat measured 3.32051/3.30566 s and
candidate 3.65447/3.65473 s. The candidate bookend's full heads and all four
initialized states match its frozen own control exactly. Native records 32
actual graph replays; reference `reused_delta=32` is GGML graph reuse, not proof
of a particular CUDA replay count. Timers pay all packing/staging/publication
and argmax; supervisor wall time includes setup and is not the rate.

Analytically, the final Q/K/V/mask packed roots sum to 892.5 MiB across 60 layers
per wave. Balanced two-plus-two CONCATs read and write twice that final size,
so their calculated total traffic is 3.486328125 GiB per wave. This is layout
arithmetic, not measured traffic, workspace or physical peak. The normal
placement budget remains authoritative; this screen makes no memory gate claim.

The result establishes the joint packed-attention mechanism at this shape,
while its cost blocks selection. Gemma26 needs its own same-format full-head
control next; its smaller local/global head counts change stream geometry.
Assistant C2, other contexts/query shapes, representative PPL and optimized
serving need independent evidence. Existing primitive fallback and production
options remain intact; no C12, depth or ubatch ladder was run.

## Reproduction and checks

Build with `mise run build -- spark-native --locked`. The manual
`llmp_gemma_attention_packed_metadata` target uses `LLMP_TEST_DATA` and
runs descriptor controls under each fixed environment mode. The model helper
accepts only `ARTIFACT NEW_OUTPUT_DIR 31 4 joined norm IDS_I32`; select control
or candidate with `LLMP_GEMMA_PACKED_C4=0|1`. [reference.sh](reference.sh)
builds/runs the unchanged original client against the pinned image. Retain
raw heads/states/logs externally; [own_freeze.py](own_freeze.py) creates an
exclusive own receipt before fresh reference acquisition, and
[compare.py](compare.py) verifies complete repeat/bytes/finite/choice alignment
plus designated likelihood rows.

Locked Spark-A build and 24 descriptor cases, four native acquisitions,
exclusive own freeze, fresh bookend, both complete comparisons and paid-output
verification passed. Local REUSE/header checks passed 1,458 files, portability
boundaries 392, and changed formatting/syntax/analyzer controls passed; raw logs
remain external. The first metadata assertion
omitted SizedArena's intentional one-descriptor slack, and two reference-client
compile attempts lacked copied headers; those failed logs are retained.
