<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Literal ds4 HCA attention over native F16 state

The literal ds4 token-tile core takes 7.056 ms on identical captured
first-4K community-GGUF attention operands, versus native attention's
89.030/89.012 ms before/after. Its charged path includes a chronological
F16 raw-ring mirror and original dense causal-record preparation. Four
fresh matched 8K model processes give a 17.88% mean prefill throughput
gain with identical 128 IDs and each arm's full-logit repeat exact.
The production-sized 2,048-row comparison gains 15.76% mean prefill
throughput under the same protocol. The native integration remains a
default-off benchmark option pending larger compressed extents and
fixed-bound quality qualification. Serving and
production chunk sizes remain unchanged.

## One attention product, identical inputs and cache bytes

The current same-community first-4K native profile recorded in the
[Q2 product study](../ds4-q2-d2r/README.md) has 6.572582 s total kernel
duration, including 1.233 s in ordinary D512 attention. The prior Q2
transfer improved mean complete-model prefill throughput by 3.68%; it
does not explain the remaining attention gap. This experiment ports one
existing MIT attention core rather than combining attention, cache and
routing changes.

Source is Entrpi/ds4
`76d51ef82a81b70b78e51a3a6ea11946286de976`, `ds4_cuda.cu` numerical
sections 12,313–12,577 and 12,844–13,476. They are copied verbatim into
the locked `jitllm_ds4_hca` target, including helper names and attributes.
The four-token/G8 core keeps its RN F32-Q to F16 conversion, F32 QK and
PV tensor-core accumulation, F32 online softmax and F16 probabilities.
It differs from native ordinary attention's weighted-accumulator
arithmetic. Identical stored inputs therefore need not yield identical
outputs; numerical differences and quality gates are stated separately.

Both arms use native F16 KV/indexer state. No ds4 FP8/FP4 cache transform,
runtime, routing, quantized product, permanent mirror or alternate state
format is selected. Negative raw-prefix rows are unavailable keys, not
zero-valued keys: the original `raw_row_min` excludes them from softmax.
The original dense record builder is called with positions null; its
visibility is `floor((first + row + 1) / 128)`. Its non-null positions
branch has a different boundary expression and is not used.

## Real operands and complete charged replay

Capture is community artifact
`cd39d504dc2dbfe911a4a521fa8efc8053dc3e80e99738a9b25fa6b70c97a1ac`,
layer three, first position zero, the first 4,096 rows of the immutable
8,192-ID study prompt. Capture is outside timing. The original full
`op_params` bytes, source strides, raw/compressed extents and output
are retained externally.

| Operand | Logical shape | Native byte strides |
| --- | --- | --- |
| F32 Q | `[512,4096,64,1]` | `[4,131072,2048,536870912]` |
| Joined F16 KV, K and V alias | `[512,4608,1,1]` | `[2,1024,1024,4718592]` |
| F16 mask | `[4608,4096,1,1]` | packed |
| F32 sinks | `[64,1,1,1]` | packed |
| F32 output | `[512,64,4096,1]` | packed |

Raw capacity is 4,352 rows and compressed capacity 256. Scale is
`0.04419417306780815`, ALiBi and softcap zero, precision F32 (enum 10),
`n_kv_max` 384 and sparse flag one. The original graph's `n_kv_max`
is the padded bound 128+256, rather than the first chunk's visible
maximum; the exact captured value is carried into the baseline.

Each timing graph contains four complete calls. The candidate includes
raw-ring mirroring, original record/count construction and core launch
within every call. Nine samples run native/candidate/native, with two
identical Q banks totaling 1,073,741,824 bytes, beyond the measured GB10
L2 capacity of 25,165,824 bytes. Both graphs contain 12 nodes.

| Charged arm | Median ms |
| --- | ---: |
| Native ordinary, before | 89.029541 |
| Literal ds4 mirror + records + attention | 7.055536 |
| Native ordinary, after | 89.012154 |

The operator speed ratio is about 12.6×, or 92.1% lower latency.
Native before/after drift is 0.0195%. Candidate planned scratch is
6,425,600 bytes versus native attention's 7,892,992 bytes; the unchanged
KV input occupies 4,718,592 bytes. The core uses 128 registers,
88,576 bytes dynamic shared memory, one barrier resource, eight stack
bytes and eight-byte spill stores/loads. These compiler resources do not
independently explain the measured gain.

The native replay exactly reproduces the captured output. Candidate
versus native NMSE is `9.484399536172616e-8`, maximum absolute difference
`0.0063800811767578125`. A separately compiled strict FP64 scorer on
22 selected token/head rows gives:

| Output/reference | NMSE |
| --- | ---: |
| Native / full F32 Q | `9.65217482737688e-8` |
| Literal ds4 / full F32 Q | `1.65870549908486e-8` |
| Literal ds4 / RN-F16 Q | `2.4396800567845164e-9` |

These selected-row reference results distinguish Q conversion from
accumulator error; they are not full-model quality. The registered
operator bound remains `5e-4`.

Thirty refusal cases cover dimensions, arithmetic/extents, unsupported
scale/bias/softcap, nonfinite parameters, alignment and overlapping
operands/scratch before submission. Thirteen accepted controls cover
first-zero tails through 129 rows; first positions 126, 127 and 128;
nonzero wrapped first positions 400 and 511; and zero KV/dominant sinks.
They check exact mirror/records/counts, guards, each path's fresh repeats,
captured repeats and changed-Q capture reuse. They do not require the
two approximations to be bit-identical.

## Native integration

`jitllm_dsv4_exec --ds4-hca` is the only option that selects
`jitllm.dsv4.hca_tokentile`. It refuses exact plans and chunks other
than 2,048 or 4,096. The device predicate is restricted to measured GB10,
D512/G64, T2048/raw2304 or T4096/raw4352, with 256 compressed cells. Serving, decode,
speculative verify, other devices and unknown shapes keep their previous
selection. The generic direct entry point admits bounded tail fixtures
without broadening the measured graph predicate.

The trusted binder tag vouches for canonical zero finite raw causal
masks and compressed visibility counts; structural checks do not inspect
arbitrary device-mask contents. The validator requires packed interleaved
Q, packed joined F16 KV, shared K/V identity, packed sinks/output,
standard scale, zero ALiBi/softcap, exact sparse bounds and nonoverlapping
read/write ranges. Scratch cannot overlap operands. First position is
immutable plan metadata, with an explicitly bounded position-keyed cache.

Records, counts and mirror use one planned, cataloged workspace scope.
The executor plans the whole aligned payload; insufficient workspace
refuses before submission. Shared-memory opt-in happens before capture.
Foreign launch results are recorded before the next submission, and
unknown completion retains owners. Captured attention followed by
ordinary attention reusing the same pool passes changed-Q and exact
own-repeat controls. The native tail controls report NMSE
`5.62815e-8`, `1.60439e-7` and `2.37771e-7` against the unchanged
ordinary approximation, all within `5e-4`.

Production DeepSeek uses 2,048-row chunks. Its actual charged replay and
model comparison are recorded below. A future default decision still
requires larger compressed extents, unchanged memory headroom and the
existing fixed near-tie/PPL bounds. Neither benchmark option changes
production's chunk size.

## Matched 8K complete-model comparison

Four fresh processes use one final checked binary and community artifact,
the same immutable 8,192 input IDs, context 9,216, chunks of 4,096,
compact expert scheduling, frontier-only head and 128 greedy outputs.
Only `--ds4-hca` differs. The Q2 D2R option is off in all four processes,
so the previous Q2 product's gain is not included here. Each process has
a successful 105-GiB/GPU/container/native-model preflight before loading
and an explicit empty-device gate after retirement.

| Fresh process, in order | Prefill s | Prefill tok/s | 127 decode steps s | Literal HCA launches |
| --- | ---: | ---: | ---: | ---: |
| Ordinary off0 | 13.4006 | 611.32 | 6.8961 | 0 |
| Literal HCA on0 | 11.3943 | 718.96 | 6.9070 | 40 |
| Literal HCA on1 | 11.3859 | 719.49 | 6.8941 | 40 |
| Ordinary off1 | 13.4519 | 608.98 | 6.9182 | 0 |

Mean prefill time falls from 13.42625 to 11.39010 s: 17.8765% higher
throughput, or 15.1654% lower latency. Both native repeats produce
identical full logits; both candidate repeats do too. All 128 IDs also
match between arms. Candidate full logits differ from the ordinary
approximation, as expected from the arithmetic change. Decode products
and choices are unchanged; the observed decode times are consistent
with that scope.

Both arms plan 1,462,430,464 activation bytes and 151,013,376 scratch
bytes. MemAvailable-derived peak occupancy is
91,245,408,256–91,362,074,624 bytes across the four processes; it is a
whole-process estimate, not physical ownership measured by the catalog.
The unchanged maximum workspace is dominated by another operation.

This is one matched 8K ABBA, not a long-context/default quality gate.
The fixed 0.947-nat near-tie and symmetric 3% PPL bounds are unchanged.
Own graph scratch/reuse controls pass; the unchanged full-swap table is
not repeated for this default-off attention arithmetic experiment.

## Production chunk size, separate matched comparison

A fresh capture executes the first 2,048 IDs of the same immutable
community prompt with 2,048-row chunks. It captures layer three's actual
Q, joined F16 KV, mask, sinks, visibility and native output. It is not a
slice of the 4,096-row capture. Raw capacity is 2,304, compressed capacity
256 and total KV capacity 2,560; original operation parameters are
retained. The unchanged literal core, strict scorer and charged replay
run their existing controls.

| Charged arm, nine samples | Median ms |
| --- | ---: |
| Native ordinary, before | 46.559704 |
| Literal mirror + records + attention | 3.556768 |
| Native ordinary, after | 46.553265 |

Native replay exactly matches the captured output. Candidate/native
NMSE is `8.772082589e-8`, maximum absolute difference `0.005705357`.
The strict 22-row full-F32-Q reference gives native NMSE
`8.508874690e-8` and candidate NMSE `2.540958133e-8`; against RN-F16 Q,
candidate NMSE is `2.905187262e-9`. Candidate/native scratch is
3,277,824 / 4,739,072 bytes. The two identical Q banks total 536,870,912
bytes, beyond L2. Operator controls do not establish model quality.

The subsequent fresh-process 8K ABBA changes only the benchmark's
attention opt-in. Context is 9,216, chunk size 2,048, compact scheduling
and frontier head enabled, Q2 D2R disabled, with 128 outputs.

| Fresh process, in order | Prefill s | 127 decode steps s | Literal HCA launches |
| --- | ---: | ---: | ---: |
| Ordinary off0 | 15.0271 | 6.8756 | 0 |
| Literal on0 | 13.0061 | 6.8893 | 80 |
| Literal on1 | 12.9797 | 6.8973 | 80 |
| Ordinary off1 | 15.0552 | 6.9021 | 0 |

Mean prefill falls from 15.04115 to 12.99290 s: 15.7644% higher
throughput, or 13.6176% lower latency. All 128 IDs match between arms;
each arm's fresh full-logit repeat is exact. Planned activation/scratch
allocations are unchanged at 903,872,512 / 96,468,992 bytes;
MemAvailable-derived peak occupancy ranges from 89,652,146,176 to
90,070,847,488 bytes. Each load and retirement passes the successful-query
105-GiB preflight. No long-quality gate or default adoption is claimed.

On `spark`, supervised replay `m3-ds4-hca-2048-replay-r1` completed rc0
2026-09-30, 13:18:14–13:18:51 EDT; model batch
`m3-ds4-hca-2048-model-ab` completed rc0 at 13:25:03–13:27:15 EDT.
Raw source, operands, checks and commands remain externally in
`~/scratch/m3-ds4-hca/source-2048/`, `replay-community-2048-r1/`,
`native-2048-final/` and `native-2048-ab/`.

| 2,048-row identity | SHA-256 |
| --- | --- |
| First-2K prompt TSV | `338b4d6b2962ef931b3cb3e06b90193c7166c2fbcd831c487d5bdad9d5c5a40e` |
| Capture binary | `d858aed16f659c600ffcc06707d0cfd1523465f6054b98b8405761d9db5ef1dd` |
| Replay binary | `b554d0deed80fd7d6b1f141811180ac7bb4596d0ccd673a4176293172d71e707` |
| Captured Q | `8d946cce71f68ed9272a6b5db6c32b3f3f376e98b454f8ab3936c7a71f743e1b` |
| Captured joined KV | `133425dcd227da71af4385414830125432c4b9a2c8ef5decec28c0d3658528ee` |
| Checked/measured native binary | `e3b9f22116b7647c1db13fe78ea7a28df850d71252c9653e562e9e0658132903` |
| Model aggregate | `2e130eb130f16eebd2f195ff01a639680bea5784879096be0903ed388b4162d2` |
| Off0/off1 full logits | `0033c67e3f92e68d9066678ab8533e6b3cebe8981706741a43acbd56ecd0a97a` |
| On0/on1 full logits | `134d427cf962e3cbb59287cef4e2a07f259e8c2b0e1d606c95a36f02c427c91c` |

The two-file eligibility/cache/CLI change passes prepare and all 1,041
locked Spark-native tests (203 GPU, 64.76 s), SDK format and the affected
host translation unit's tidy, 296 boundaries, actual REUSE and 22 headers.
The subsequent comment/report/plan update passes a separate actual REUSE
and 23-header export; six CLI refusals pass before loading. Broad swap
timing is unchanged and is not repeated for this benchmark-only extension.

## Provenance and reproduction of the initial 4,096-row trial

Raw capture, controls and replay are under
`spark:~/scratch/m3-ds4-hca/replay-community-r2/`; supervised job
`m3-ds4-hca-replay-r2` completed rc0 on 2026-09-30,
11:22:14–11:23:01 EDT. Full numerical source remains available from
the source lock and reviewed patch; no upstream runtime is incorporated.
The locked target uses the original `-O3 --use_fast_math -lineinfo`,
sm_121a SASS. The external replay inherits native host `-O2`, with
`--use_fast_math -lineinfo`; both retain NVCC's default O3 device
optimization. The replay's actual commands and compiler-resource output are in
`replay-community-r2/build/build-micro.log`. Its FP64 scorer does
not use fast math. All model/operand inputs, vectors, traces and logs
remain outside Git.

The model batch `m3-ds4-hca-native-ab` completed rc0 on `spark`,
2026-09-30, 12:44:01–12:46:07 EDT. Its four commands, per-process
summaries, full-logit vectors and aggregate analysis are external under
`~/scratch/m3-ds4-hca/native-ab/`. The controller requires the binary
and build receipt to match the final checked identities, then verifies
that those files and the prompt remain unchanged between processes.
Analysis SHA-256 is
`b14b5b40cd37134a03abc8f590351f421e1cc11a862452cdfa76be70e52cbd36`.

| Identity | SHA-256 |
| --- | --- |
| Source archive, 7,794,185 bytes | `731e037da1bed009da5db31e5170681b59f1e22af00a3e56fa29cff8706ed63c` |
| Original `ds4_cuda.cu` | `8d5de76a7aaaf9131ba8f9cf35412863fef88299b39676ea4bcfb07aef7386e3` |
| Original helpers, 8,675 bytes | `41eeb9ec3aeb0fd77e40212efbcea4e8cbede859226849c5333e49c9cec1b26c` |
| Original core, 25,074 bytes | `d953dd3ccc968f1bbe605851366ab78af9f5ba1bb0cdaa6cb7e1fe4bb15f2719` |
| Patch 0002 | `b8f73b3f2aa5a0f01f1cb417361c48e94788332b2e7104ed062490e72604c425` |
| Prepared source tree | `829a29fcc0b9e10560d8da819c6a71d8af8c5456cd56204cbd99a9671913f5ac` |
| Prepared HCA CUDA | `48c1999315695ffdfd72b741f914deb568d0ab914a5c2faebcb7a81d00c30f79` |
| Replay binary | `9c88c430f7aac1ef68fabb3d1f1f3a305df1d9d0e609f7351a0c4de6ff0a427d` |
| Capture binary | `a4f3782c924ad9b9ce5737846166412d92e6f6cba1d17971ee43e0abaab49c77` |
| Strict reference object | `042ec23621fc2e836f77063d80fedbbbf00b607fdb63ca1cf6f390ad28a55bbf` |
| Replay Q | `13ac2cd8f7cb7a5cf72f59e72981fad69974da8dcfca134392a7d4213ad310bc` |
| Replay joined KV | `01d3dbd920f01d29699f9f75389a376bacc1d8881973f90cf5168a87e9fc2230` |
| Original op_params | `460a6dfea38797bf78feff173c5f905dc060c3144b5fbc1951803d77cd509bf4` |
| Native captured output | `423eeffc50beee057da7dd715c74a0802db4f9d04d17ca2b1cad4f013dd53166` |
| Immutable 8K prompt | `c1d7137841d7d594254e5540806b645e8fb7e6b741f8bcee77776bff67256e29` |
| First-4K prompt TSV | `2781d8299bc58782978d9ddd7bc868bf0b40558508e5f539bd67538e4f98177f` |
| Final checked/measured native binary | `91e2f84ef4b53a1e799e7ecd9920d035a5bfa1d1a9a9481a78df86d6870602da` |
| ABBA build receipt | `21c6c430c1f9bb39019b0fc4a30a64a094e85150d0b9a22f18b47a23f6fe66c7` |
| Final metadata build receipt | `6160958029888c65a6d0bcc959f59ea617a76fd78ee0ae29a41b976ddc326353` |
| Off0/off1 full logits | `9979bc75807ad6c378443fa13e40c624e865af996a8f7cbd7f893efadde5147c` |
| On0/on1 full logits | `f175324e40b004671bcce27c928ab563542eaeb12c44b3251a215f5bb9a2b755` |

The shared source parser verifies the archive hash/length, applies both
reviewed patches and verifies the resulting tree. The extraction receipt
also directly verifies that each original numerical byte range occurs
exactly once in the prepared HCA CUDA unit. The root MIT notice and
ds4/Entrpi/GGML holders are retained; the existing product's Marco
Palaferri attribution remains shipped. The
[upstream entry](../../upstream/ds4.md#literal-token-tile-hca-core-over-native-f16-state)
states the standalone interface proposal and native adaptation boundary.

Validation on `spark`, 2026-09-30: prepare plus all 1,041 locked
Spark-native tests passed (203 GPU, 65.23 s); SDK clang-format,
seven eligible C++ translation units' clang-tidy and the boundary
check passed (296 files). CUDA units are formatted, while SDK policy
does not lint NVCC commands. The exact extraction, prepared-tree and
MIT/Marco notice checks passed. Actual REUSE/header checks passed on
the owned export (22 headers) after matching the patch's annotated
holder in its sidecar; four experimental CLI refusals pass before
model loading. Final documentation/source-lock metadata corrections
change the build-receipt hash, while the native, runtime and long-swap
binaries remain byte-identical to the checked build. Workstation tiers
remain deferred by the owner.
