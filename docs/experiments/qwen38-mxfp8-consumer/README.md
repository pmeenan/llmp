<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Literal Mia MXFP8 projection on captured native inputs

The actual Mia trace-derived MXFP8 orientation costs **6.2% more cold and
6.3% more warm** than the existing native projection on this eight-input
control. The fresh public API default costs roughly 3×; it selects a different
orientation and cannot stand for Mia's recorded kernel. Neither result
justifies integration. Production arithmetic and defaults are unchanged.
This is a single-request operator diagnostic, with no model quality,
concurrent-request throughput or complete-reference-pipeline claim.

## Same operands and charged work

The retained native capture uses canonical31743 prompt IDs, context33792,
chunk4096, existing selected47172 drafter and two history steps at fixed
depth2/T3 and depth3/T4. QSA layers23/47 contribute their existing attention
HC F32 input and unreshaped query/gate projection output: K2560, N12288.
The eight records use two immutable E4M3/E8M0 weight banks totaling64,880,640B
(61.875MiB). Every code and scale byte matches the authenticated native
prepared artifact and original checkpoint tensor. GDN's imported row
permutation is outside this experiment.

The six-path capture delta retains existing views through the native graph
and copies only after completed Verify. T4 adds475,136B pinned staging.
Both depth processes pass the existing fresh off/on/on/off draft,
probability, full-state/cursor, commit/rollback, captured operand and repeated
logit controls. Capture changes no product node, dispatch or default.

The original arm invokes existing native MXFP8 vector products on F32 input
with F32 output. The literal arm charges F32-to-BF16 conversion, installed
FlashInfer CuTe E4M3/E8M0 activation quantization including padded scales,
installed CUTLASS BF16-output GEMM and widening to F32. It follows the installed
vLLM postload scale permutation and verifies every scale byte against its
preimage. This is a precision-changing operator sequence, not bit-equivalent
arithmetic. Host SDK link libraries and the container's loaded runtime
libraries have separate authenticated inventories.

Each replay passes eight native captured-byte/own-repeat controls, eight
candidate eager output and activation-code/scale own-repeat controls, and
32 native/candidate graph-to-eager byte controls. All output rows are finite.
Each cache condition has four balanced ABBA groups, eight observations per
arm:32 positive events total. A cold sample flushes512MiB outside the event
and times the complete eight-invocation graph. Values below are event elapsed
divided by8, **milliseconds per projection invocation**; casts, quantization,
product and widening are all included. Warm repeats are secondary.

| Selection | Cache | Native median ms/invocation | Literal median ms/invocation | Literal/native cost | Paired-group cost ratio range |
| --- | --- | ---: | ---: | ---: | ---: |
| Fresh public API, returned−1 | Cold | 0.151196 | 0.430954 | 2.85030× | 2.84890–2.94906× |
| Fresh public API, returned−1 | Warm | 0.142994 | 0.432194 | 3.02246× | 3.01569–3.06578× |
| Explicit trace-derived tactic1 | Cold | 0.150876 | 0.160214 | 1.06189× | 1.06093–1.07643× |
| Explicit trace-derived tactic1 | Warm | 0.142734 | 0.151750 | 1.06317× | 1.04507–1.06686× |

The default and traced arms produce exactly the same candidate output,
quantized activation code and scale bytes for all eight operands. Compared
with native, all eight full vectors differ: maximum absolute error
0.0822697–0.139915, RMS absolute error0.00799824–0.0148277, relative L2
0.00678388–0.00835415. These are descriptive operator errors, not a measured
task-quality loss or an acceptance bound. There is no sampled FP64 oracle,
isolated phase timer or full-model candidate run in this diagnostic.

## Actual tactic correspondence

The pinned SM120 binding turns tactic−1 into0. Its original getConfigs emits
tile128×32×128 first with SwapAB=false, then with SwapAB=true; those are
tactics0 and1. The original templated runner and instantiation file map these
to `DeviceGemmMxfp8GemmSm120___nv_bfloat16_128_32_128false` and `...true`.
Mia's existing pure-decode trace names the latter. The follow-up therefore
forces **tactic1 in the same original runner**, while separately recording
that the unchanged public AutoTuner returned−1 for both T3 and T4. This proves
the source-derived tile/orientation correspondence; it does not recover Mia's
per-process tactic cache or match every reference hidden/cache stage.

| Original mapping source | SHA-256 |
| --- | --- |
| SM120 binding CU | `896676c7f8f8238958f3c918c611963a0630729a664f82511ed509e57bb6c962` |
| SM120 configuration header | `57fe994ba800726e4b3414135840f451830f7dd5a950cbe8102ebf9a45fdbbe4` |
| SM120 kernel header | `b7dcdbf2036f4aff130548a67c3ff369283315dcda57c6435f6d0a6b1d0fe17a` |
| Original instantiation Jinja | `692ef56546cb9a50120b0138c5949e90e62c238a578c7232ae2cf7ea7d8ee1ff` |

Actual numerical modules are vLLM linear
`ecc607bdb3998ad0180303b38b4999147badacfb99406825774ed979c62ad45d`,
vLLM utils`227d42fb1e866ea3d187356a3a29b0804a6ede022c1807b5d0eb3004553fbc34`,
FlashInfer GEMM`6c17e99e93f34fb7d51908d87719fca5cf5a89ebbe88af594d2ccc1773db7213`,
FP8 quantization`73d82d7538be184592b0f6afbbeeee98b03dc902d5e32e6c0b624124566dcb9b`
and CuTe quantizer`cff8c63794b725ee3dccedce02c0cfbce552f607fa8b91d214fe3c6556e8876d`.
The immutable image is`fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`;
the actually loaded MXFP8 binding is
`81f680c108ff07d81ba4939aa67d5100b00504e6b27a47e05b56d4c1e54ebe75`.
Original NVIDIA/Apache notices remain with copied source. The trace summary
is`a083c67207d02b4609e1581dd11e283bfd2d2e0dc5768534c05d0fa55402f886`,
from compressed trace`791cff66b082bbc737ea61993b1e4fbd1f11d2b585e28b5784ade9ce0a51ff35`.
This batch records exact imported sources and mapped shared-library hashes;
it does not retain a separate CuTe JIT cubin/cache artifact.

## Provenance and disposition

Raw sources, operands, binaries and receipts remain on A under
`~/scratch/m3-qwen-mxfp8-r1/`; compact local copies are under
`/tmp/llmp-qwen-mxfp8-*`. No raw vectors or binary payloads are in Git.

| Receipt | SHA-256 |
| --- | --- |
| Native target build | `ebd6d66c7f9c43b8e70a62b261ae7f6285b9341e3164cfc5bd5d2346d0d897f4` |
| Native capture | `2e20cdcc65dea8c6dc6353d48008c9d9222ebc7051fbf0e47ed9297e0108f14c` |
| Bridge build | `5b80ec11fcbc759dbddbc8442724ae79a7a4af98171b85a260300565e7d22240` |
| Corrected matrix extraction | `8fa1945d1a37025536757be924c15990f0cbe2eaaae3e0bbb935ddb9697affc7` |
| Default launch/result | `2ae27ab2e97122b3d848f03104e6225607297ac868fe525ed9ffed5e04c70571` / `c734b96e2a330c32e81d77e63c5e7f84c3646eb5b81b6133b9e9e395f605888a` |
| Trace source proof | `954070012487f37754f3f7875e333754733b4e4aaba566d77bb9933c1f42fb62` |
| Traced launch/result | `e9a9ce30fedcd1fc7cdc6f9ebf0a6ce36fdef7d06c08f835e797e7a2a3f9e1d0` / `c6ccd7c2112cf364f7d3d93762a9db433338399567ce14b86f1f2d907bd83287` |

The target binary is`7eceed54e0cdd26e20d80527a79812ccafa015050779e0977c7af37e990f748d`,
bridge`4120da8e9db00a9cc27c1785d81b3c98855754307484d7a9aaf226301c445315`;
source map2505c58e authenticates1241 exported files on main e46ed93 plus the
six capture paths. Native build/capture and all external source/controllers
received root and independent whole review. Target-only build passed; routine
full-suite/style checks are deferred under the owner's diagnostic workflow
override, and are owed for any adopted production change.

`qwen-mxfp8-native-capture-r1` retired rc0 in152.87s. Default operator
`qwen-mxfp8-literal-operator-r1` retired rc0 in27.8172s with receipt-bound
116.683GiB free; traced `qwen-mxfp8-traced-operator-r1`, supervisor1759096,
retired rc0 in17.7157s with117.055GiB free. Both owned launch sessions were
reaped before daemon container absence and the strong retirement probes.
One-time two-bank weight setup is1.03342s default and1.00980s traced,
including separately measured CPU permutation audits0.38894s/0.40950s;
readiness is0.51982s/0.16750s. Candidate Torch allocated642,836,480B and
reserved648,019,968B; native scratch is1MiB. These are operator fixture
allocations, not complete-model memory comparisons. The initial extractor's
wrong tensor-prefix failure is preserved and excluded; corrected extraction
uses the actual `model.language_model.layers` prefix with unchanged byte proofs.

Retain the bounded capture seam for future identical-operand diagnosis; stop
this slower single-request factor. Strict32K decode and concurrent-request
comparisons remain open. No additional head policy, model quality exception,
or default precision change follows from this experiment.
