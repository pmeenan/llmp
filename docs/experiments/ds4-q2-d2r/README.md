<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Direct ds4 Q2_K product on the same native inputs

The raw-GGUF loader for ds4's existing Q2_K D2R product reduces the
charged isolated product's latency by about 26% versus current native
compact MMQ. Four fresh same-source 8K model runs give a 3.68% mean
prefill throughput gain, with the same 128 greedy token IDs. The product
is retained as an explicit benchmark-only opt-in; serving, exact plans,
decode and speculative verify keep their previous choices. This result
does not qualify a quality/default change or close the ds4 performance gap.

## Choose one product, keep the surrounding execution fixed

The historical matched 4K profiles in the
[ds4 study](../ds4-study/README.md) identify Q2_K down as the largest
individual product gap: native 0.956 s versus ds4 0.452 s. A fresh native
profile after compact scheduling and the qualified HCA/frontier changes
has 6.572582 s total GPU kernel duration per first 4K chunk, including
0.703324 s Q2 down and 0.885564 s IQ2 gate/up. These are GPU component
sums, distinct from the 6.774 s profiled prefill wall time. The Q8 bucket
is not an isolated ds4 comparison because ds4 separately fuses attention
output work. The current Q2 share is 10.7%, so a 26% product gain is not
a prediction of a 26% whole-model gain.

Only the Q2 product is ported. The native artifact, routing, selected
contributions, post-down route weighting, Q8 activation preparation,
F16 attention/indexer caches and surrounding graph remain the same.
ds4's runtime, cache transforms and fused IQ2/SwiGLU/early-weighting
epilogue are not selected. This native A/B therefore avoids the cache
and stage-precision differences of a full ds4-versus-jitLLM comparison.

The product source is Entrpi/ds4
`76d51ef82a81b70b78e51a3a6ea11946286de976`,
`cuda/mmq/ds4_mmq_d2r.cu`. The original SoA Q2 kernel and raw loader both
use paired-row integer MMA, stored half coefficients, F32 correction
and accumulation, and the original output scatter. The raw loader stages
84-byte GGUF blocks at explicit row/expert strides into the same shared
layout instead of requiring an additional persistent SoA weight array.
The original D2R approximation differs from GGML's coefficient rounding
and reduction order; each path's precision is stated rather than assuming
that identical weight and Q8 bytes imply identical products.

## Real-input capture and isolated replay

The captured layer-zero down product is the first 4,096 rows of the study's
8,192-token prompt. Capture runs outside timing and explicitly proves the
complete resident expert slab before copying all 256 experts. It does not
read potentially absent pages from a paged runner.

| Operand | Logical shape | Native byte strides |
| --- | --- | --- |
| Q2_K weights | `[2048,4096,256,1]` | `[84,672,7081536,1812873216]` |
| F32 per-slot input | `[2048,6,4096,1]` | `[4,8192,49152,201326592]` |
| I32 selected IDs | `[6,4096,1,1]` | `[4,24,98304,98304]` |
| F32 unweighted output | `[4096,6,4096,1]` | `[4,16384,98304,402653184]` |

The source array starts at byte 2,162,688 in a 1,812,873,216-byte backed
slab. The external raw weight fixture strips unused expert padding but
preserves every original quantized block; its expert stride is 2,752,512
and it contains 704,643,072 weight bytes. Every isolated arm consumes the
same logical weight, input and ID bytes. The production wrapper retains
the artifact's padded expert stride, also covered by short controls.
All 256 experts are reached; accessed weights exceed the GB10's measured
25,165,824-byte L2 capacity. Each sample runs behind a 512-MiB L2 flush.

The times below include expert maps, unchanged GGML Q8 preparation,
worklist construction where used, and the product. A graph captures each
complete arm; nine samples run in forward order, then nine in reverse.

| Complete product arm | First median ms | Reverse median ms |
| --- | ---: | ---: |
| Ordinary GGML MMQ | 20.877024 | 21.421791 |
| Native compact MMQ | 16.591520 | 16.589567 |
| Raw-GGUF D2R | 12.273472 | 12.228928 |
| Original SoA D2R control | 10.778304 | 10.778752 |

Raw D2R is 1.352–1.357× compact throughput, or about 26% lower latency.
The SoA control incurs a separate 0.179677 s host repack and 672 MiB of
extra weight storage; those costs are excluded from its product time and
it is not the integrated arm. The integrated raw path has no such copy.

Raw and original SoA D2R are bit-identical on this input. Against compact
MMQ, NMSE is `2.1331352209781956e-7`, maximum absolute difference
`0.0028363466262817383`; 100,657,346 of 100,663,296 output values differ.
The compact control exactly reproduces the captured native output. Each
arm's fresh eager/graph repeats and guards pass. This is operator-error
evidence, not a substitute for fixed-bound model quality.

Six short controls use K512, even M2/18/130, T17/65/129, 16 experts and
six distinct selected experts per token, with repeated experts across
tokens, unused experts, padded rows/expert strides, zero activations and
tails. A duplicate expert within one token violated the original GGML
mapping precondition and was removed from an earlier failed fixture;
that failed run is not performance evidence. The
[upstream log](../../upstream/ggml.md#selected-expert-ids-are-unique-within-a-routed-token)
records the distinction.

The raw and original SoA main kernels use 128 registers, 47,472 bytes
static shared memory and zero spills; the worklist kernel uses 32
registers and 1,040 shared bytes. This describes compiler resources,
not an independent speed explanation. The replay's peak planned scratch
is 56,843,520 bytes in an explicitly bounded 256-MiB pool.

## Native integration and whole-model A/B

`--q2-d2r` on `jitllm_dsv4_exec` enables the new registry identity
`jitllm.mul_mat_id.q2_d2r`. Default selection is off; its opt-in device
predicate admits only measured GB10 K2048/M4096/E256/used6/T4096 shapes.
Row-invariant verify has precedence. Other types, shapes and devices
retain ordinary/compact products. The generic raw interface checks
even output rows, whole-512-element K, valid strides, nonoverlapping
experts, per-slot inputs, contiguous output and bounded worklist/grid
capacity. The original signed expert decode limits E to 32,768; boundary
controls accept that count and refuse 32,769.

Maps, Q8 preparation and worklist scratch live in one planned launch
scope. The executor explicitly accounts for the new implementation's
workspace. A captured pair followed by another product reusing the pool
passes changing-route repeats; insufficient workspace refuses before
submission. Unknown launch completion faults the context and preserves
owned resources. Default and unknown-shape graph controls retain the
previous primitive selection.

The four model processes below use the same checked binary and artifact,
the same 8,192 input IDs, context 9,216, chunks of 4,096, compact experts,
the frontier head and 128 greedy outputs. Order is ordinary/candidate/
candidate/ordinary. These are fresh cold process runs, with one model on
the Spark and a successful 105-GiB/GPU/container/process gate before each.

| Arm | Prefill s | Prefill tok/s | 127 decode steps s | Peak MemAvailable drop GiB |
| --- | ---: | ---: | ---: | ---: |
| Ordinary first | 13.3586 | 613.24 | 6.8789 | 85.24 |
| Raw D2R first | 12.9149 | 634.30 | 6.8828 | 84.93 |
| Raw D2R repeat | 12.8986 | 635.11 | 6.8822 | 84.97 |
| Ordinary last | 13.4055 | 611.09 | 6.8804 | 84.90 |

All 128 output IDs are identical across all four captures. Candidate
prefill throughput gains 3.44% and 3.93% against its paired ordinary
control; the mean gain is 3.68% (3.55% mean latency reduction), below
D-085's 10% material threshold. Decode is unchanged. Planned activation
and pool envelopes are identical at 1,828,716,544 and 190,840,832 bytes;
the actual largest product scratch is 151,013,376 bytes in both arms.
Memory stays within the unchanged 1.1× bound. The 128-token equality does
not establish full-window PPL, long greedy, rollback or swap quality for
this changed approximation. Those gates remain owed before a default.

## Source and run provenance

The community GGUF is 86,720,111,488 bytes, SHA-256
`ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`;
native artifact is
`cd39d504dc2dbfe911a4a521fa8efc8053dc3e80e99738a9b25fa6b70c97a1ac`.
Prompt TSV SHA-256 is
`c1d7137841d7d594254e5540806b645e8fb7e6b741f8bcee77776bff67256e29`.
Host `spark-c4e2`, CUDA 13.4.92, SDK `aarch64-e0a0c85c42806fb1`.
Native base is `af879a0` with this unit's uncommitted port; model A/B
binary SHA-256 is
`5afaa263b7e3a4351ef4cc50497c123fc45a9b40d048d6d371ecf22378ad3448`.

The whole pinned archive was inspected before narrowing the compiled
tree. Archive SHA-256
`731e037da1bed009da5db31e5170681b59f1e22af00a3e56fa29cff8706ed63c`,
length 7,794,185 bytes; raw-loader/build patch SHA-256
`cf5f0b6686cf2917a34165a72c9ce748d6f091ba0c9a8652ea2fe6360b44f72f`;
prepared tree SHA-256
`dc925631f904c032ea2dcafaa94ca759af3c9c1ccaa1adfc8f1ecc3e81f803c7`.
The lock records the full MIT notice, Marco Palaferri's inherited
attribution, kept/compiled/unselected source scope and original compile
flags. The adaptation compiles O3, `sm_121a` SASS, fast math and extended
lambda as the isolated replay, with configured discrete SASS as well.
It links no ds4 model runtime. Workstation/package checks remain deferred.

The slice's Spark set passed 1,039 tests, including 202 GPU tests,
in 64.44 s, with SDK format on all 11 changed C++/CUDA files, tidy on
seven host translation units, 295 clean boundaries and actual REUSE/
24 embedded headers clean. The final identity/constant-parentheses lint
refactor is followed by a passing registry control. It changes metadata
construction and spelling, not the measured numerical sequence; the
explicit absent-callback default likewise preserves the prior empty
`std::function`. Final checked native benchmark SHA-256 is
`3d43c8bff1ce7bdc3699eed0ba47be6e1ee06fc38e270b31ee6ed090a0850dad`,
distinct from the measured model binary above. Final wrapper source
SHA-256 is `751478f710fd2f97f2b52bcb5524ddca2a1ba5b06cf9dbb92ed3ac988ef999cd`;
receipt `4a564d2bd25396d1f6dd476dad6cc0e583ca6511bf41ef41a584e5c90d4a9476`.
The prepared tree's extracted five-line Marco notice is checked explicitly.

External raw files are at `spark:~/scratch/m3-q2-d2r/`: `profile/`,
`capture/q2/metadata.json` and its weight/input/ID/output fixtures,
`controls.jsonl`, `real.jsonl`, `native-model/`, source-lock audit/receipts,
build/check logs and source/binary identities. `m3-q2-d2r-micro3` completed
rc0 at 08:51:25–08:51:43 EDT; native ABBA
`m3-q2-d2r-native-model` at 09:40:31–09:42:38 EDT. The external real-input
replay source SHA-256 is
`6eb301df78a6ab5fbe5ac962d9026bffa2bbbc639369165b483dda5a452918e8`,
binary `188cc783be19134371d85b3c3f12fe07b9c887b16b16c884d1e7120c059f38f9`.
Use `q2-d2r-build.py`, `q2-d2r-micro.sh` and `q2-d2r-native-ab.sh` with the
pinned captured inputs/artifact to reproduce these distinct comparisons.
