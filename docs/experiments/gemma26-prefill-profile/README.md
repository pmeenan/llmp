<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma26 paid-prefill profile

The traced native prefill has 1.361 s without recorded GPU activity, against
0.085 s in the original engine. GPU activity unions are much closer:
2.469 s native and 2.364 s original. This identifies a substantial interval
for a subsequent host-thread diagnosis; it does not identify busy CPU work,
planning or a scheduler defect. No production policy changes are selected.

The competitive result remains the untraced
[ring-cache comparison](../gemma26-swa-ring-transfer/README.md): native
3.80551 s versus original bookends 2.41568/2.41232 s, +57.6433% latency.
The instrumented spans below are observer diagnostics and cannot replace
that bookend. [Results](results.json), [provenance](provenance.json) and the
unchanged preregistered [protocol](PROTOCOL.md) give the aggregate evidence.
Raw SQLite, logs, tokens, state and event records remain external.

## Fixed workload and fidelity

One approved Gemma26 UD-Q4_K_M artifact, C1, context16384,
max_rows/physical ubatch1024, local2048/global16384 and F16 KV. Both engines
use the same canonical IDs, six warm rows followed by clear, 8192 paid
prefill rows, three anchors and 32 completed forced decode units. Each
unit argmaxes its incoming head. Original `swa_full=false` and
`kv_unified=false` remain explicit. Native uses checked norm and MoE
compound options; last-built plan counters are 60 norm/RoPE, 90 norm/add,
30 route and 30 reduction, with other optional counters zero. Those are
source-plan counts, not observed CUDA launch counts or original eligibility
parity. Native pays eight complete Chunk heads; original publishes one
final head after eight physical ubatches.

Exactly one untraced annotated control and one trace per engine completed.
All four runs reproduce their respective Task40 complete prefill/final
heads and all 32 incoming-head choices. Native reproduces all 592,445,440
initialized state bytes and the same layout. Both original runs report
actual local2048/global16384 cache capacities. Each app exit, profiler exit,
installed supervisor final record and original owned-container absence was
verified. Native/reference head identity with their own prior runs does not
establish cross-engine full-head agreement or a quality pass. No all32
full-vector publication, PPL scan or new noise calibration was performed.

## Paid-range observations

A single public NVTX range encloses the existing paid-prefill timer, including
native last-built policy lookup. Model loading, warm/reset, anchors, decode,
file/state outputs and teardown are outside this interval.

| Clipped observation | Native | Original |
| --- | ---: | ---: |
| NVTX wall span, s | 3.830521312 | 2.449598128 |
| GPU activity union, s | 2.469429759 | 2.364314953 |
| Kernel union, s | 2.458002527 | 2.360788361 |
| Copy union, ms | 11.427232 | 3.526592 |
| Wall minus GPU union, s | 1.361091553 | 0.085283175 |
| CUDA API union, s | 0.903534928 | 2.127590704 |
| CUDA kernels | 12,648 | 11,660 |
| Device copies | 328 | 66 |
| CUDA API records | 142,010 | 19,158 |

Unions and category sums overlap. CUDA API time can include waits for GPU
work. The model OS-runtime union fills each entire range because idle worker
poll/condition waits overlap it; this is not busy-CPU time. The extractor
authenticates actual model process ownership through the atomic child
receipt, SQLite PROCESSES and NVTX global-thread encoding, and excludes one
observer-parent OS-runtime wait per trace. Present tables have zero unknown
process events. The separate driver table is absent in both databases;
CUDA runtime-table records include the listed driver-named API calls. Native
has no memset table; original has zero paid memset records. Neither trace
records a graph API call inside paid prefill; CUDA kernel counts still
represent executed activity, not graph-launch or plan counts.

Actual quantized matrix-product families sum to 1,443.432031 ms over 2,080
native kernels versus 1,460.002621 ms over 2,120 original kernels. F16
attention families sum to 223.034112 ms versus 219.845504 ms, both over 240
kernels. These are clipped kernel-duration sums, not disjoint wall costs or
complete operator attribution.

## Heads, copies and API bodies

The vocabulary projection is the unique 262144-row product from normalized
hidden state and `b.output` in `gemma4_graph.cc`. Its observed Q8_0 MMVQ
symbol is `mul_mat_vec_q<(ggml_type)8,(int)1,(bool)0,(bool)0,(bool)0>`,
grid262144×1×1, block32×4×1. Native executes eight such kernels totaling
29.858208 ms; original executes one totaling 3.780448 ms. Corresponding
1 MiB head D2H copies total 0.201856 ms versus 0.022912 ms. This geometry
and source evidence supports the head label. Frontier-head narrowing also
changes final-block projected/residual rows before FFN, so the eight-versus-one
publication axis includes final-block product work. Turning heads off is not
known to be cheaper or state-only, and was not attempted.

Native additionally records 280 HtoD copies of 2 MiB each: 587,202,560 bytes,
11.141472 ms summed device-copy time. These are device-transport bytes;
they do not prove disk reads, paging I/O or physical peak memory.

| Clipped CUDA API-body sum, ms | Native | Original |
| --- | ---: | ---: |
| cuMemCreate | 23.686304 (280 calls) | 0.962304 (2 calls) |
| cuMemMap | 0.879104 (280 calls) | 0.013600 (2 calls) |
| cuMemSetAccess | 24.200448 (280 calls) | 0.365776 (2 calls) |
| Three VMM APIs together | 48.765856 | 1.341680 |
| Two main kernel-launch APIs together | 811.335744 | 637.697136 |

The VMM body sum is much smaller than native's 1.361 s GPU-unattributed
interval. Launch API bodies overlap GPU execution and other threads; neither
sum can be subtracted from that interval or assigned critical-path CPU cost.
Original `cudaStreamSynchronize` sums to 1,456.734688 ms across 146 calls,
which likewise includes waiting. Native event-query/context-setter counts
are high, but their body sums are 18.891776/5.671872 ms. CheckPlaces,
node.Call, descriptor/plan construction and state commitment remain source-led
hypotheses. Passive critical-thread sampling and wait attribution are needed
to distinguish them; this experiment supplies no CPU-stack evidence.

## Source and reproduction

Measured source base is 36811f4. The frozen14-source environment includes ten
new/changed acquisition files and four unchanged historical imports. All
457 tracked production/source-lock/toolchain files match their authenticated
manifest. The native compiled closure records 231 translation units, including
74 locked prepared-source units and two generated module digests. Eight
external NVTX headers are used; all33 public-header identities are recorded.
Public NVTX is Apache-2.0 WITH LLVM-exception and remains external. No
production target depends on it. Original math remains the digest-pinned
b29 image/libraries, with only a thin-client annotation delta.

The initial `trace.py` and all acquisition sources remain byte-exact to the
pre-run freeze. After actual SQLite process mappings were inspected,
`trace_v2.py` was separately frozen before analysis; it adds process ownership
filtering without changing acquisition or math. Provenance distinguishes
this derivative from the pre-acquisition environment. Later CMake integration
appends only the manual-target stanza onto the newer parent; the measured
CMake and executable identities stay recorded separately.

Supply the authenticated Task40 raw receipt/logs, canonical IDs, approved
prepared artifact/raw model, pinned eight original headers and existing
Nsight public interface in external scratch. The protocol explains the
production-source/ancestry manifests and model-free interface runtime gate.
Build the optional native manual target with the recorded external
`JITLLM_BENCHMARK_NVTX_INCLUDE_DIR`, then run `reference.sh build` before
freezing binaries. The wrappers use the owner's B `m3gm31` tree and external
`~/.local/share/jitllm/gemma26-prefill-profile` directory. Use fresh names;
receipts and outputs refuse overwrites.

Every build, probe, control, trace and analysis runs through installed
`spark-job start --gpu --timeout 600 --stop-on-fail`, followed by official
`spark-job wait`. In that supervised job, the gate sequence is:

```sh
python3 -B validate.py source ROOT SCRATCH BUILD_JOB
bash native.sh control native-control
# After retirement:
python3 -B validate.py result ROOT SCRATCH NATIVE_JOB SOURCE_SHA native-control native
bash reference.sh control original-control
python3 -B validate.py result ROOT SCRATCH ORIGINAL_JOB SOURCE_SHA original-control original
# After both controls authenticate:
bash native.sh trace native-trace
bash reference.sh trace original-trace
# After each trace retires, validate.py result again with its engine/name/job.
python3 -B trace_v2.py --self-test
python3 -B trace_v2.py ROOT SCRATCH SOURCE_SHA native-trace native NATIVE_TRACE_JOB \
  NATIVE_CONTROL_SHA ANALYSIS_SHA NATIVE_TRACE_VALIDATED_SHA
python3 -B trace_v2.py ROOT SCRATCH SOURCE_SHA original-trace original ORIGINAL_TRACE_JOB \
  ORIGINAL_CONTROL_SHA ANALYSIS_SHA ORIGINAL_TRACE_VALIDATED_SHA
```

Scripts above reside in this experiment directory; SHA arguments come from
exclusive earlier receipts. Child completion is published only after
checked successful app exit and receipt write/fsync/close. Docker traps retire
only the recorded ownership-labelled CID; after a killed wrapper run
`reference.sh cleanup NAME` under supervision and verify busy/absence before
another model. Profiler exit or output existence alone is insufficient.

The official model-free probe, locked build, four fidelity runs and separately
frozen analysis completed successfully. Earlier license-assumption, compile
shadow and scratch-marker failures occurred before model acquisition and
remain identified in provenance. This benchmark-only diagnostic uses the
experimental-comparison workflow override; it does not rerun the whole unit
suite or select a production optimization. Memory occupancy and profiler
allocations are not physical peak evidence. Gemma31 and backward transfers
to other families require their own measured controls.
