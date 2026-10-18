<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Decode ladder competition on GB10

## Protocol fixed before measurements (2026-10-17)

The R6 registry exposes the existing block decode templates in production. This
experiment decides which implementations to rank; reachability alone is not a
reason to change a served choice. Historical decode-template numbers are not
used as ranks. In particular the earlier GLU benchmark paid for two products
without the final SwiGLU and is not comparable to this experiment.

The bounded competition inventory is eleven shape bands: MXFP8 split dense
K2560/N10240 at 1/16/512 input rows; BF16 dense K2304/N9216 at 1/16/512 rows;
MXFP8 split gate/up GLU K2560/N640 at 1/16 rows; and row-interleaved NVFP4
160-value rows, 16 IDs per token at 1/16/512 tokens. The first screen runs four
representatives, each at `--rows 1`: `mxfp8`, `dense`, `glu`, and `rows`.
Grouped expert timings in the harness compare fused and primitive work and are
not interchangeable implementation ranks; they are outside this first screen.

Run the immutable captured candidate on `spark` in one supervised exclusive
job, with no other build or inference. Record source-manifest SHA256, source
lock, device name, SM count, architecture, CUDA driver/runtime and exact command.
The harness `llmp_decode_templates --case CASE --rows 1 --reps 64 --batches 9`
uses its existing CUDA-event timing around CUDA graphs, 64 launches per graph,
two warm graph replays per arm and nine timed replays per arm. Arms alternate
forward/reverse order (ABBA for pairs). MXFP8 dense rotates four complete weights;
BF16 dense rotates four weights; GLU rotates sixteen gate/up pairs; rows rotate
eight deterministic ID lists over a table larger than L2. Shapes and RNG seeds
are fixed by the checked-in harness. The GLU incumbent includes both products
and SwiGLU, matching the decode GLU's paid work.

Opening bookend order is `mxfp8 dense glu rows`; closing order is
`rows glu dense mxfp8`, with the same inputs/options and fresh processes. Record
all medians and ranges. A candidate qualifies for further evaluation only if
its median latency is at least 1% lower in both bookends and the incumbent's
opening/closing medians differ by no more than 5%. An unstable bookend is
inconclusive, never an inferred win. A losing first screen stops expansion for
that pair and leaves other bands explicitly unranked. A winning screen expands
the affected served bands before adoption. Served selection changes additionally
require matched family speed/quality and an explained frozen-digest change.
No threshold is adjusted after observing the measurements.

Before timings, replay each competing arm and compare the final rotating
weight/ID set's outputs. Require finite outputs and relative RMS error
`sqrt(sum((candidate-incumbent)^2)/max(sum(incumbent^2),1e-20)) <= 0.01` for
products/GLU, and exactly zero error for gathered rows. The 1% RMS screen permits
expected F16/BF16 activation rounding in matrix paths; it supplements the
independent CPU-reference kernel tests and does not replace family quality.
BF16 matrix, matrix-FMA and vector arms are compared separately with the same
incumbent. A failed correctness screen stops that process and cannot qualify.

Record `cudaMemGetInfo` used/total bytes and the harness's cumulative exact
payload allocations (including its 512 MiB launch workspace). Report the
resident encoded rotating weights, inputs and outputs separately from runtime,
CUDA graph and library overhead; the CUDA used-memory counter includes the
latter. Capture ordinary GPU memory/utilization counters immediately before and
after each bookend. No profiler is part of this decision.

The reference for any necessary matched family comparison is the preserved
`review-r3-r5-65fccb09` candidate. `final-d373a1bf` is historical R13 evidence,
not the incremental R6 baseline.

## Results

The first eight-process attempt stopped in its second opening process: the
BF16 MMA arm failed correctness before timing. An independent CPU calculation
sums the original F32 inputs to 8.71657181 and round-to-nearest-even BF16 inputs
to 8.81952667. GGML MMVF produces 8.716572; decode MMA produces 8.819527. The
1.18113941% relative error is the matrix path's documented BF16 activation
rounding and exceeds the unchanged 1% screen. That arm is rejected; BF16 FMA
and vector arms were not reached and remain unmeasured. The initial MXFP8
opening sample is excluded from the accepted bookends.

A fresh six-process screen of the other three pairs used the same payloads,
repeats and criteria, opening `mxfp8 glu rows` and closing in reverse. All six
correctness screens passed: MXFP8 relative RMS 3.48187879e-7, GLU zero, rows
exactly zero. Latencies below are microseconds per launch, median [min,max] of
nine replays of a 64-launch graph in each fresh process.

| Pair | Incumbent opening | Decode opening | Incumbent closing | Decode closing | Decode slower, opening/closing |
| --- | ---: | ---: | ---: | ---: | ---: |
| MXFP8 vector K2560/N10240, 1 row | 111.26 [111.08,116.08] | 112.85 [112.18,117.78] | 112.45 [108.68,113.68] | 114.87 [111.29,115.98] | 1.43% / 2.15% |
| MXFP8 GLU K2560/N640, 1 row | 16.33 [16.23,16.45] | 18.32 [18.23,18.38] | 15.80 [15.68,15.99] | 18.06 [18.00,18.14] | 12.19% / 14.30% |
| NVFP4 rows, 160 values, 16 IDs | 0.90 [0.90,0.93] | 1.16 [1.16,1.19] | 0.90 [0.90,0.91] | 1.18 [1.17,1.19] | 28.89% / 31.11% |

Incumbent bookend drift was +1.07%, -3.25% and zero respectively, within the
fixed 5% bound. None of the decode candidates won; larger bands stop here and
remain unranked. The measurement table records only two exact incumbent
preferences on NVIDIA GB10/sm_121/48 SMs: MXFP8 split MatMul rows 1/K2560/N10240,
and tensor-scaled row-interleaved NVFP4 GetRows with 16 IDs/K160/table 1048576.
The latter query counts gathered IDs, not tokens. Batched operands, other
parameters, shapes or devices do not inherit these ranks. The generic decode
duplicates remain available where their native counterparts are ineligible,
including other compiled devices and supported physical/index/shape domains.

The GLU incumbent resolves to two `llmp.mxfp8.mul_mat_vec` calls and
`ggml.swiglu`; it has no single fused implementation identity to rank. Its win
keeps the primitive recipe. Newly reachable decode GLU automatic rewrites
remain unqualified; direct typed operations remain available. No served winner
is adopted and no frozen digest is re-recorded. Actual before/after serving
choice inventories and affected family controls are collected separately.

### Provenance and memory

Measurements ran exclusively on `spark`, NVIDIA GB10, architecture 121, 48 SMs,
CUDA driver API 13000, runtime 13040, driver 580.178.04 and nvcc 13.4.92, on
2026-10-17. The fixed B source-manifest SHA256 was
`14917d6f65c63bf2e5fe92df83034b128845e90751f125b645f1fe44466133fe`, source-lock
SHA256 `07e4753248f9a829a85f21738e10fd7e724dcd1f6043e38d3d71e7317d43cb8d`,
benchmark source `32350aa1cd9c2f6457544371c2c0adb2c00d58c1d60c5f24cc1971981e7f845e`,
binary `75d8dc4e98e1faa5b505c3cb240eec5df0666e1c747ea6dc73dcf57e5a803550` and
six-process script `6ce95f17cae5b1a001d492734ddfc1470f764b079b4d446eb3dba7df55c0f443`.
The external frozen protocol SHA256 is
`10084b78e2e9d30c9627eca7639aa750a4de1512712f240d741a0a7d70d64345`.
Only diagnostic flushing and independent input-sum reporting were added after
the first failure; inputs, arithmetic, timing and acceptance criteria did not
change. The raw six-process log SHA256 is
`7affeaf07146352b4a7345ee8292fcb9a0ee518d9ce4f49c83d625c0d6b97817`; failed initial
screen and dense diagnostic are retained externally with SHA256
`83871dacbd3a357f39e5b430d9fbb5e82602fb769331cf4cfa1e6d0e732382eb` and
`32478e00cf10a58282240a6f35782b4b6880f67443fd25835ee3b7cbb25c5a46`.

| Case | Resident encoded weights | Input/ID allocations | Decode output | GGML intermediate/output allocations | Launch workspace | Exact harness allocations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MXFP8 | 108134400 | 10240 | 40960 | 163840 | 536870912 | 645220352 |
| GLU | 54067200 | 10240 | 2560 | 122880 | 536870912 | 591073792 |
| Rows | 94372096 | 2048 | 10240 | 81920 | 536870912 | 631337216 |

All memory values are bytes. Rows' tensor-scale allocation is 256 bytes for a
4-byte value; each of eight 64-byte ID sets receives the harness's 256-byte
minimum allocation. CUDA used-memory opening/closing readings were
28597764096/28598165504 (MXFP8),28554760192/28571938816 (GLU), and
28584087552/28594814976 (rows), of 130661203968 total. These unified-domain
counters include system/runtime/graph/library overhead; they are not the
kernel's isolated footprint. Ordinary GPU utilization at the bookend boundaries
was 0/3% and 0/1%; nvidia-smi did not expose memory counters on this host. No
profiler or unmatched performance baseline was used.

## EXL3 paired-product qualification (2026-10-18)

The actual plan audit exposed a separate unqualified rewrite after universal
source registration. On the sealed 145-row EXL3 control, the primitive graph
identity and non-product operation counts matched, but the selected prefill
recipe changed:

| Prefill selection | Accepted baseline80ede618 | R6 before qualification repair |
| --- | ---: | ---: |
| Exact pinned reconstruction products |169|121|
| Occupancy-derived paired packed products |0|24|
| Total selected steps |681|657|

Each new paired product replaced two gate/up reconstruction products. The
normalized scratch requirement was identical: baseline63,183,104 bytes minus
its4,203,008-byte persistent prefix equals candidate58,980,096 scratch bytes.
The scalar plan retained121 exact pinned GEMMs and24 exact pinned paired
products. Both sealed-reference tests stopped on their first145-row numerical
comparison, so their failing run does not establish successful scalar execution
or complete trajectory quality. Input hash guards passed.

This was an automatic-selection qualification gap: the paired matcher performed
a direct-operation eligibility lookup, and device occupancy made a newly
reachable paired implementation executable at prefill sizes. The same output
encoding does not guarantee identical reconstruction/cuBLASLt and packed-kernel
accumulation. Neither a speed nor quality advantage had been measured for that
replacement. The repair applies automatic-rewrite intent to every matcher
lookup and requires measurement for occupancy-derived paired products. It keeps
direct typed access, exact historical scalar pins and measured qualification
available; it introduces no model or shape allowlist. A future bounded paired
product screen and corresponding family quality/speed checks can qualify a new
automatic winner. The old oracle is not a requirement to preserve old bits.

A subsequent source challenge found that qualifying the matcher alone was
insufficient: an exact historical scalar pair could permit the rewrite, then
ordinary final selection could choose the unqualified occupancy-derived pair.
The ordinary backend has no explicit-source preference to mask that choice.
Final selection now retains the rewrite's existing node provenance through
pattern rounds, pruning and fallback decomposition. Common-subexpression
joining keeps explicit and automatic operations separate, preserving direct
typed access and each operation's eligible implementation set. Controls cover
the actual ordinary backend at 1/145/1024 rows, a higher-priority unqualified
competitor, measured qualification, graph transformations and both orders of
otherwise identical explicit/automatic operations. These controls passed on
`spark`, as did both sealed-reference trajectories and the three fixed-teacher
cases. Their ordered selections match the accepted baseline; this is correctness
and selection evidence, not a new paired-product speed result.

Regression evidence: accepted-baseline EXL3 reference log SHA256
`0a467090b726552f24a7006bbecaf49d817646dca1d45fe29084dd5ec7b1ec2b`;
pre-repair R6 reference log SHA256
`ee1ba57b5c77e96e594d9759b3d8c30e265630e8fadd6ad31ca9a840933d222a`.
Post-repair reference log SHA256
`ec2d6b2898c025d67649a250bf1504398a06e95440c465867c38028f8dd3a2c2`;
fixed-teacher log SHA256
`e06a7dd1943285da675e236c5d58248132a8b0796b9404c8f80e569491b96d94`.
The teacher input contains 43 authenticated natural-text tokens, not the
separate 76-token Qwen 2 control.


## Actual selection and memory accounting

The accepted comparison source is `80ede61814ce8697b274bf6c98ea9627628b5cff`.
The final audit union compared 36 CTest wrappers, 60 actual parameterized cases
and 1,913 plan records per arm. Every actual case completed successfully and
had plan records. The comparator joins wrapper, actual case, primitive identity,
stage, phase and owner shape; it preserves selected-step order and multiplicity.
The final candidate versions were production 30 except Qwen 3.8 at production
31, with Qwen Image and both EXL3 controls at production 29. Seven groups had
identical choices. Gemma 2/3 was the only group with choice changes, confined
to Gemma 3's grouped cache stores described below. Across the union, 150 scratch
records decreased, none increased, and 1,763 were unchanged.

| Group | Wrappers | Actual cases | Records per arm |
| --- | ---: | ---: | ---: |
| Gemma 2/3 and assistants | 4 | 8 | 279 |
| Gemma 4 | 3 | 12 | 576 |
| Qwen 3.8 | 7 | 9 | 450 |
| DeepSeek V4/DSpark | 4 | 4 | 46 |
| Qwen Image | 9 | 9 | 54 |
| Qwen 2 FP16/Q8 | 4 | 13 | 490 |
| EXL3 reference trajectories | 2 | 2 | 6 |
| EXL3 fixed-teacher cases | 3 | 3 | 12 |

Qwen 2 uses the seven-key fixture `tests/unit/data/v1_frozen/qwen2_fp16.json`,
SHA256 `211b903a9364cd2eee1d07d4fb5ae5dd30b3a45065624fa2b66a9da33530aa51`,
with recording disabled. Its original two values are unchanged; five additive
pins come from accepted-baseline executions. They supply missing regression
witnesses, not independent M3.6 quality oracles. The bridge compares independent
logits; same-plan and Q8 checks establish lifecycle consistency. Broader
reference coverage remains R12 work.

The verified external union summary has SHA256
`bd34b6fc3fd6b15ebaf2005688b7da96e32fb62da881ecaf8735dd9f8734bbdd`.
The choice comparator source SHA256 is
`f66a7d48a8d6b1f56a28d674997b99306f7c0da6b283909b565b8823ff956898`; the
aggregate summary tool SHA256 is
`13d7c526a7c60d11946e7f5617e8b70b1f095f44c10b456003679b341d2fcfd0`.
Production31's clean 2,580-file source manifest has SHA256
`ce167eef05fbebba2dd1d0011754dc96c3e5dd1726b83102d8812cc947b6596c`; its
clean build log has SHA256
`79410abf48a7f43416b95069628edfdface76e88883dde9debf9f8b228417614`.
The final audit's per-group capture-log SHA256 pairs are:

| Group | Accepted baseline log | Final candidate log |
| --- | --- | --- |
| Gemma 2/3 | `61326959a0ba90d9b29e478797d53d5c37b527ea04ce0883effc5f7bf49590e2` | `ad09ae00f8a78aa363319aff8b06bab7d265b6420e173f54797d3d260083d303` |
| Gemma 4 | `964ff667a6089c72993e95ebfb735a20a701ee5b980380bbe63a81a7046cea29` | `1f7d20471fa8d4b4e00baf32fcd8d28d8980ba5e7792b6da8f9d43b277f5e69a` |
| Qwen 3.8 | `b61e945e08ff2f54dc8b5149b0c1d18175f1a068d9bc91517eae4cd757638c9f` | `821e1417bcf7df26549f7af48830de735b9c4542a4fc95983b252df26cccf1b2` |
| DeepSeek V4 | `46ac2f093a434668f1288992028e3571b805c19f344bd8f7391e6404af0a8fb1` | `662c45809ee30dc15b09d8a62e3dd7467e31bb454f603763bfa2efe47703888e` |
| Qwen Image | `8193842fc012c44a6ea4a1e27d14b7f7ea8596606d99ca1e405e24414d3b34c2` | `6f52e6e4e83bf453b244ff892ba8f77eeef1dfbf56b0af037719e2395737bb2a` |
| Qwen 2 | `522615d0e3c5be42143ae1ede4ebd1fc0c77387c6dd18eec05fccb92e34d64ea` | `15b120e245901199b90346abf8d5645a799f4659ed3207fb8971f51ef5e27386` |
| EXL3 reference | `0a467090b726552f24a7006bbecaf49d817646dca1d45fe29084dd5ec7b1ec2b` | `ec2d6b2898c025d67649a250bf1504398a06e95440c465867c38028f8dd3a2c2` |
| EXL3 fixed teacher | `cf366271be287854575c51399befa10c5c39c7375bab27a5f02b1732b2e8bc5c` | `e06a7dd1943285da675e236c5d58248132a8b0796b9404c8f80e569491b96d94` |

### Backend ownership has a paid cost

Each automatic backend owns five selected-source states, requiring 21,015,040
bytes of device backing, rounded to 23,068,672 bytes (22 MiB) by catalog
allocation. This cost is paid even for a plan using only GGML. Explicit EXL3
reference contexts include six sources: 25,218,048 bytes, rounded to 27,262,976
bytes (26 MiB). The old single-source context required 4,203,008 bytes, rounded
to 6,291,456 bytes. These retained allocations are separate from reclaimable
scratch; neither a scratch remap nor another model may overwrite them.
The executed A–B–A–B control retains two real model runners across shared-scratch
reclamation and remapping. Retirement waits for positive device completion,
then destroys plans and launch state before releasing their backing.

The final lane-contract declaration control on `spark` records 5,168 tuning
facts, 250,155 temporary host bytes, 1,143,979 retained host bytes and 1,394,134
peak/paid host bytes. This measures `Declare`, excludes the separate owner
object's `OverheadBytes`, and is not a whole-model/post-`Seal` host-memory
measurement. Metadata growth is accounted against actual catalog capacity,
including post-factory sealing; the former fixed 1 MiB allowance is gone.

### Actual lanes govern the plan

The earlier audit exposed unused lane funding in Qwen 3.8's serial joined-MTP
recipe: its largest affected scratch band grew from 6,389,760 to 25,655,296
bytes, although selected operations were identical. The latter equals 96,256
plus four copies of 6,389,760. The backend had advertised per-owner stream
support, the planner assigned lanes, and binding subsequently discarded them
because this recipe configured none.

Lane count and funded slice size now belong to the immutable backend contract
and plan identity. A serial backend supplies no hypothetical owner lanes; a
lane backend configures its streams before publishing launch state. Binding
refuses a mismatch instead of silently changing the schedule. DeepSeek retains
its configured four lanes. Its existing conservative overlap between lane
reserve and measured scratch remains a physical-placement follow-up in R7.

The correction also removes an unnecessary split of Gemma 3's grouped cache
stores. Twenty of the 279 Gemma 2/3 records change, and Gemma 3 is the only
actual choice change in the final audit. Each affected plan launches 34 fewer
`llmp.set_rows.grouped` operations: one group per layer contains both owners'
K/V writes, where hypothetical lane tags formerly split it into two. Primitive
identities, all other implementation counts and scratch bands are unchanged.
The grouped-store contract proves independent roots and rejects overlapping
destinations and operands; element conversion and arithmetic do not change. All
eight model cases pass the existing head/state/spill controls. This is a real
launch-count change and is reported separately from decoder choice changes.

### Gemma 3 matched speed and quality closeout

The production31 candidate was compared with the preserved
`review-r3-r5-65fccb09` runtime using the same v1 artifact
`634bc193468f5968ac142c2e5547f72777b182f3f694873075d7a50660f46626` and exact
configuration SHA256
`d031addd78288c92b88076645abb31be884de65ec2d473917cfe47fdbd816901`. Both
arms used empty diagnostic environment maps. The preregistered tier-1 screen
used two ABBA blocks, six measured passes per run
plus its warmup. Each block primes the old runtime, then runs old/new/new/old
in separate processes. The workload uses 11,500-character prompts, 128 generated
tokens, context 4,096, prefill chunks of 128 and two maximum slots. All eight
measured runs completed seven passes and 21 requests each without errors; every
request generated 128 tokens. This is a screen, not the canonical
minimum-seven-block full gate. Within each block the two runs per arm are
arithmetically averaged. The table gives the geometric mean across block
slowdown ratios: candidate/reference for prefill time and reference/candidate
for decode and wall throughput. Positive values indicate slowdown. Every result
is within the fixed +0.5% tier-1 bound. This screen alone does not establish a
statistically significant speed gain.

| Cell | Prefill | Decode | End-to-end wall |
| --- | ---: | ---: | ---: |
| C1 | -0.454% | +0.037% | -0.104% |
| C2 | -0.688% | -0.498% | -0.660% |

The matched quality control scored 1,070 targets. Both arms produced mean NLL
3.668106400 and perplexity 39.177649; all 1,070 per-target scores were
identical. All five preregistered bounds passed: target count 1,070, absolute
PPL delta at most 0.5% (observed 0), mean absolute per-target difference at
most 0.05 (0), maximum absolute difference at most 1.0 (0), and agreement
within 0.1 nats at least 95% (100%). Mean paired peak memory was 4.7007 GiB
old and 4.7163 GiB new, an increase of 0.0157 GiB (0.333%); observed ranges
were 4.694–4.711 GiB old and 4.688–4.764 GiB new.

Provenance: the production31 clean source manifest covers 2,580 files and has
SHA256 `ce167eef05fbebba2dd1d0011754dc96c3e5dd1726b83102d8812cc947b6596c`;
its clean build log SHA256 is
`79410abf48a7f43416b95069628edfdface76e88883dde9debf9f8b228417614`. The
candidate identity file SHA256 is
`100208be1874e370840729e4d5927df0642f61d1b684c93566ca9d9e9bb3cd1f`, its
build receipt SHA256 is
`fbd29180def01f3def6bdb1b3c0cf608440bcd472eef141d23bd81927097d52a`, and its
runtime SHA256 is
`92e3f43d90882cd8eb8716717bfb2f65a4911ed4e7342384e2b33a507c312e15`. The
reference runtime SHA256 is
`3690936aa25550ca64a4d1ca74992e0168cfcbf5ede8d46d4965774b5059853f`. The
speed harness and quality scorer SHA256 is
`60fd224b20242837f9ca38bf993e871fec7cb5b66600b8733e074eee34e14a9e`; the
quality-pair comparator SHA256 is
`9bf1d1597a8d6e97d65a648a9db68d8a51c694cc3772a6f01ba158fc420137bd`; the
quality-bounds preregistration SHA256 is
`4f9e2e748035e622524207df18a9e862a40c20cb864b207dd22aa6b707c6450e`, and
the corpus SHA256 is
`55381941fdb19f99a88c5170f25d7cbc3e091d7d839a8776ac491a593c978ceb`. Only the
registered quality bounds are reused: its historical v0/v1 label does not
describe these two explicitly configured v1 arms. The two approved step files have SHA256 values
`39c9dfc1077ea0b89f24a2d69aa635cd21bbeeec2e39e77168e0c43a49cac00c` and
`b260d88ab541479827e8cee57b27745c850a4b62de89172344165299ae6dc6b9`. The
supervisor logs for blocks 1 and 2 have SHA256 values
`2ae962d1f027ee27164e569c0e69f4c512c9106a9085840446e4d00984008469` and
`35a38249d681fcb71b88b9380116c813da43f16df3cb2c8c83cbaaea00faab5f`. The
speed-run JSON SHA256 values in order B1 p0 old, p1 new, p2 new, p3 old,
then B2 p0 old, p1 new, p2 new, p3 old are
`03a218797cfc3ebbcbf8aee2627020d1b6298cfcf964fd72c0a6dd752e45beb6`,
`7e019b8269c1b1d37aa0a142c2a46055940bdb050fc75a0e0c3502c7c24f22b5`,
`1e4f4d48a5b7a63622c987573a0880873a4c06861a25995e8225a4c867e7e89b`,
`ba9082469f6ca4acc5719ca8569ca110eea346340a17f706aeed569ba67083bf`,
`12b1adce69113b6214d934570670faae2ab47caf757cf9c546281a97e7bb45a0`,
`30422a8fc84f95f454daa781fa792dbce5424c9cbeb352cf1be26167b7f5e0ca`,
`b0136d9c5ee7f7dcf930f5925d30434d3d65ed3deb27c1089ef69738454b4d43`, and
`a0afb4e89da0085b7b2030e29f30b8de2d569317ab5c03ddd4b6a0486d95abca`.
The quality run logs have SHA256 values
`07213d4e7e1252426cbece16129d953d9e8b51a74a553d309d5439a5beb07851` (old)
and `8611ded6c5a7a5b6af159f012113ac59b34609ead2524ef6bb2e390d767196d2`
(new); the quality pair report SHA256 is
`db732a47c0e62d4de17f8032cbbd75ad64b7013b87b28f1a5d3c061a9b1fec87`. These
measurements ran on `spark` (hostname `spark-c4e2`), NVIDIA GB10, with driver
580.178.04 and no competing GPU compute applications. Main integration checks
are recorded separately in the slice commit; this report describes the frozen
candidate measurements.
