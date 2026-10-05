<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dense31 natural C4 first-local attention replay

On identical native-origin operands, the existing native four-stream MMA
launcher matches the unchanged original-image GGML backend **byte for byte over
all 32,768 F32 attention outputs**. Four segmented vector launches and four
segmented MMA launches differ. This isolates a real attention kernel/stream
packing arithmetic difference; it does not establish the cause of the complete
model's quality gap or select a production policy.

The manual capture/replay targets add no production selection, graph keep,
configuration or API change. [Protocol](PROTOCOL.md), [aggregate results](results.json)
and [provenance](provenance.json) retain the bounded recipe and identities.

## Inputs and capture fidelity

The approved dense31 prepared artifact is `32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08`.
The measured native source base is c2717bf. Natural fixed prompts have 64+owner
IDs, followed by three forced warm anchors; the first following four-owner
query is at positions 67/68/69/70. Context is 256, maximum rows 128. Ordinary
products and both checked norm policies are selected; row-invariant products
and all other optional policies are off. This is one actual first-local layer,
not synthetic attention inputs or captured original-model operands.

A benchmark-only link wrapper copies actual Q/K/V/mask payloads on the original
native launch stream before attention. It adds no node or keep consumer. Two
independent captures and two capture-disabled controls have identical complete
four-owner vocabulary heads (4,194,304 bytes). All captured operands repeat
exactly. Cached graphs retain fixed-address D2H commands even when the host hook
is not re-executed. Completed mask witnesses show current-query KV included:
68/69/70/71 visible cells, with zero initialized cache padding through readwidth
256 and negative infinity in unused mask rows.

Captured per-owner Q is F32 `[256,1,32,1]`; K/V are F16
`[256,256,16,1]`; mask is F16 `[256,32,1,1]`. Q views have actual shared-root
offsets 0/32768/65536/98304 bytes. The replay concatenates these native payloads
into constructed four-stream roots with the authenticated strides. These roots
are not witnessed original-model cache storage. Parameters preserve scale 1,
F32 precision and absence of softcap, ALiBi or sinks. Output normalization only
concatenates owner/head/dimension; it performs no numerical conversion.

## Same-input arithmetic

| Arm | Dispatch | Byte exact to original A | Maximum raw delta | NMSE against A |
| --- | --- | --- | ---: | ---: |
| A | Original public backend, one four-stream descriptor | yes | 0 | 0 |
| B | Native, four segmented vector descriptors | no | 0.0088415145874 | 4.9994094533e-8 |
| C | Native, four segmented MMA descriptors | no | 0.004547119140625 | 1.5043523063e-8 |
| D | Native, one four-stream MMA descriptor | yes | 0 | 0 |

A and D share whole-output SHA-256
`724ee828a64fe12bd796b7ac5f984ea2f365ea34c7e31bddbfdf6391955d82a5`.
All four attention-coordinate argmaxes agree, including both nonexact arms;
these are not target-vocabulary choices. Per-arm whole-output own repeats are
exact, and the native B/C/D source/output freeze precedes the original A run.
No bound was enlarged after comparison.

The pinned NVIDIA vector kernel retains F32 scaled queries; the MMA family
converts query pairs to F16. B versus C changes the attention implementation,
not merely one rounding instruction. C versus D preserves the MMA family while
changing stream packing and partition/mask handling; its complete output delta
is the C row above. Native plans report tile 4/GQA2, 48 blocks, scratch 399,360
bytes for each segmented MMA descriptor and 399,616 bytes with a mask prepass
for the four-stream descriptor. Vector scratch is zero. Planned API groups are
not CUDA launch counts.

## Actual original launch observation

A separate Nsight Systems 2025.3.2 trace observes the unchanged original backend
selecting `flash_attn_ext_f16<256,256,4,2,false,false,false>` on these constructed
descriptors: grid `[48,1,1]`, block `[32,4,1]`, dynamic shared memory 66,112 bytes.
It also observes the mask prepass (grid `[1,4,1]`, block `[128,1,1]`) and stream-k
fixup (grid `[48,4,2]`, block `[256,1,1]`). This establishes actual MMA selection
for the replay. The original full-model selector remains source inferred.

Default graph-level tracing records one eager kernel triad and 38 CUDA graph
launch calls. The three kernel events are not a total count of replay kernel
launches. Traced initial/repeated/fresh outputs match untraced A exactly, and an
exclusive application receipt follows final synchronization and backend/context
release. Successful profiler exit alone is not the completion proof. No traced
duration enters the paid table.

## Paid one-operation screen

Each untraced arm performs 32 fixed C4 waves (128 completed owner units), with
full 131,072-byte D2H output publication, finite scans and attention argmax
scans inside its wall timer. Upload/setup, warm execution, capture, fresh-query
controls and GPU input witnesses are excluded. The chronological bookend is
A/B/C/D/A.

| Arm | Seconds for 32 waves |
| --- | ---: |
| A before | 0.004157245 |
| B | 0.003609584 |
| C | 0.003449617 |
| D | 0.002774500 |
| A after | 0.004149725 |

These millisecond operand wall times include different native and foreign
submission/completion paths. They are not full-model speed, throughput or
adoption evidence. Every arm also proves complete GPU Q/K/V/mask byte equality
before and after paid work. A changed-Q eager/captured/backend control produces
a changed output, repeats exactly, then restores the initial output exactly.

## Provenance and reproduction

Original math comes from llama.cpp b29c606e28a01b1bc8c1351026a0fa6e616bf6c4
and digest-pinned image
`ghcr.io/ggml-org/llama.cpp@sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7`.
The client links unchanged `libggml-cuda`, `libggml` and `libggml-base` with exact
pinned public headers. It loads no model and rebuilds no math library. Native
kernels use the declared prepared GGML tree `026f1ac94af98011…`. TensorFold was
refreshed at task entry 2026-10-05T10:10:37.952043Z to
[609ca419](https://github.com/ashhart/TensorFold/tree/609ca419abecebdc5a059498a613680bd3aa847f),
version 0.6.5: its Gemma recipe is 26B-A4B/MLX only, not a dense31 CUDA comparator.

Build the two manual native CMake targets with the locked Spark SDK. Supply the
verified artifact and the external 1,024-ID I32 file identified in provenance:

```sh
# Freeze the capture sources/binary/ID recipe before these four runs.
bash reference.sh build
jitllm_gemma_attention_capture ARTIFACT capture-first 31 4 joined norm IDS_I32 capture
jitllm_gemma_attention_capture ARTIFACT control-first 31 4 joined norm IDS_I32 control
# Repeat both in new directories before preparing the immutable replay payload.
python3 prepare.py EXPERIMENT_ROOT SOURCE_ROOT
# Freeze the replay sources, both binaries, headers and packed inputs now.
jitllm_gemma_attention_replay EXPERIMENT_ROOT/packed native-own-B B
jitllm_gemma_attention_replay EXPERIMENT_ROOT/packed native-own-C C
jitllm_gemma_attention_replay EXPERIMENT_ROOT/packed native-own-D D
# Freeze complete native own outputs and successful retirement before A.
bash reference.sh replay original-first
# Paid B/C/D calls in new directories, then:
bash reference.sh replay original-repeat
python3 analyze.py EXPERIMENT_ROOT BOOKEND_LOG comparison.json
bash reference.sh trace
nsys export --type sqlite --output EXPERIMENT_ROOT/original-launches.sqlite EXPERIMENT_ROOT/original-launches.nsys-rep
python3 observe.py EXPERIMENT_ROOT launch-observation.json
```

Use new private directories named `capture-first`, `capture-repeat`,
`control-first`, and `control-repeat` under `EXPERIMENT_ROOT`. Each source/input
freeze uses `{ "bytes": file_size, "sha256": SHA256(exact_file_bytes) }` identities.
Before capture, write `source-freeze-capture1.json` exclusively with `sources`
(path-to-identity for capture .cc/.h/wrapper and CMake), `binary` (capture target),
`input` (IDs), `artifact`, and the exact `policy` string shown in provenance.
`prepare.py` validates that record, all four complete heads and both operand
sets, then exclusively creates `packed/*.bin` and `capture-own-freeze1.json`.

Before native replay, exclusively write `source-freeze-replay1.json`: `sources`
for native replay, shared header, original client, wrapper, prepare/protocol and
CMake; `binaries.native`/`binaries.original`; public `headers`; `capture_own_freeze`;
and `packed` copied from the authenticated capture-own receipt. After successful
B/C/D retirement, exclusively write `native-own-freeze1.json` with `source_freeze`
identity and `outputs.B/C/D.first/repeat/fresh` identities. Check every file has
131,072 finite bytes, first equals repeat, fresh differs, and the native job's
final and all steps report done/rc0. Preserve its log/final/steps identities in
`official_retirement`. The complete measured records in provenance are schema
templates; new acquisitions must compute their own identities and preserve
freeze-before-comparison ordering, not reuse historical hashes as new evidence.

The wrapper uses the documented experiment/source roots and original pinned
headers supplied externally. Its trace branch preserves the installed Nsight
namespace. All Spark work runs through the installed GPU supervisor and is
waited on; raw payloads/logs/traces stay external. Acquisition wall seconds are
explicitly untimed diagnostics. The report's measured source freeze predates
later trace-only wrapper plumbing and protocol/report edits; provenance retains
both wrapper identities and the exact diff. Build/replay branches, math clients
and measured binaries are unchanged.

Native replay pre-funds a 64 MiB host allowance, 32 MiB mapped operand/output
storage, 2 MiB activation and 16 MiB launch workspace, and 8,519,680 pinned
bytes; all registered occupancy participates in startup funding. The original
backend reports a 17,104,896-byte operand/output buffer and the same known host
allowance. Foreign pools/library allocations are not covered by that buffer
number. These are known capacities/occupancy, not physical peak measurements.
Native requests hold mapped roots and stable captured addresses until proved
completion. Any uncertain retirement retains the entire lifetime bundle;
original failures fail-stop rather than destroy borrowed owners.

This single first-local shape establishes reusable-kernel fidelity for D on
these native inputs. It does not prove original-model upstream input equality,
end-to-end quality, recurrent attention, D512, C12, or model/batching support.
All existing defaults and failed full-model quality evidence remain unchanged.

Gemma26 is the immediate backward-transfer lead: its local attention also uses
D256/GQA2, but its Q/KV head counts and stream geometry differ. A real Gemma26
operand and full-model control must establish applicability before any policy
selection. Qwen/DeepSeek attention shape/format eligibility must likewise be
audited; this dense31 replay supplies no automatic cross-family qualification.

The final current-base a6dd81f locked Spark-b compatibility compile passed for
both new targets (11 s); those rebuilt binaries are compile-only evidence, not
the measured executables. The original measured binaries and receipt were
retained externally before rebuilding. Local REUSE/header checks passed 1,440
files, boundary checks 392, changed C++ format/diff and Python/JSON/bash syntax
checks passed, and signed-zero/tie/nonfinite metric controls passed. All capture,
own-freeze, untraced bookend, trace/application retirement and observation jobs
completed under the installed supervisor. Diagnostic-only scope uses the
experimental comparison check override; no routine whole unit suite was run.
