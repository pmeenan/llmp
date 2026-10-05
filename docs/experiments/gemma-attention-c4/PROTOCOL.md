<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dense31 first-local natural C4 operand replay

One bounded manual benchmark from c2717bf, with no production policy, graph or
kernel changes. Raw arrays, logs and traces stay external. Negative results do
not qualify model support or warrant C12/D512 expansion.

## Source facts and attribution boundary

At original llama.cpp b29c606e28a01b1bc8c1351026a0fa6e616bf6c4,
llama-graph.cpp:2605-2610 splits Q into independent streams then permutes it;
kv_unified=false and n_seq_max=4 retain Q.ne[3]=4. This is not query mixing.
fattn.cu:617 requires Q.ne[1]=1 AND Q.ne[3]=1 for unquantized vector attention;
with aligned masked padded GQA2, lines253 and148 select MMA<256,256,4,2>.
The !gqa_opt fallback at631 can select vector, so actual descriptors/alignment
must be checked. Native Gemma4Graph slices four Q/K/V descriptors with ne3=1;
FlashAttnVec256Selected selects vector for each one-row descriptor.
On NVIDIA the HIP-only V_DOT2 macro is absent (common.cuh:769-774), so
fattn-vec.cuh:229-245 keeps scaled float2 Q; MMA converts Q to half2 at
fattn-mma-f16.cuh:1273. None of this establishes the cause of complete-head
quality failures before the identical-operand experiment.

Read-only git-show b29 copies were byte-identical to historical local prepared
files; the measured build separately records its prepared tree.

- fattn.cu SHA fbf690a15864ad11edac131823c174b05b3432896c14bc87d2c698929860e1d7
- fattn-vec.cuh SHA deb732ee1f1007814d95943ce48f3eec228087bac986c2bfcf8191a65706cfe7
- fattn-mma-f16.cuh SHA a211fc573b9c560688a944bfab491d6c3e4597bc385c4820fd9e8117329deb3f
- fattn-common.cuh SHA 411e3017439046288815a08a47d301dff046a909eb0ab505ac7f22f87257aabc
common.cuh differs only in jitLLM error/PDL environment handling, not the macro.
Final native compiled prepared-tree identity must come from its actual receipt,
not a historical directory name.

## Exact native operand origin/capture seam

Reuse supplied dense31 natural64+owner prompt plus three forced warm anchors,
with ordinary products and the explicitly recorded norm policy. Capture only
layer0 local attention at the first following C4 query, P=67/68/69/70. K/V
include each current-query write; this is target attention, not the assistant's
frozen P-excluded cache. Keep the same native input/source/policy identities as
that capture. Do not mix operands from stock or claim native==stock upstream Q.

The benchmark-only link wrapper around the selected exported native
FlashAttnVec256 entry: same signature, original symbol reached with --wrap;
verify the exact mangled symbol and four descriptor records. Before calling the real
launcher, queue D2H copies of Q/K/V/mask on that same launch stream into stable,
pre-funded pinned destinations. This occurs after original producers and cache
writes, before consumers; it adds no GGML keep node, changes no plan/placement,
and holds no graph producer for fusion. Use captured D2H copies at fixed
addresses if the source graph is captured, then fence before reading. Run only
through this first query for acquisition, so later queries cannot overwrite the
witness. Warm graphs may include these diagnostic copies; this acquisition is
untimed. Compare the complete resulting model heads against an unchanged
same-input run to prove capture fidelity. Refuse if link wrapping bypasses the
actual selected call, stream retrieval requires unsafe ABI guesses, or complete
heads differ. No production observer/API addition and no stock interception.

Record every actual Q/K/V/mask type, ne, nb, view depth/root/relative offset and
parameters, cache initialized read extent and occupied physical mapping.
Readwidth is expected256; authenticate it, rather than infer from P alone.
Four Q views expected F32[256,1,32,1], K/V F16[256,256,16,1], scale1,
no ALiBi/softcap/sinks. Preserve the actual payload bitwise, including padding
and signed zero. Serialize no process addresses. Freeze source/binary/input
identities and two independent operand+head acquisitions before arm comparisons.

## One identical payload, four arms

A: unchanged original-image public GGML CUDA backend, one attention op with
Q[256,1,32,4], K/V[256,256,16,4]. Build input root tensors and permutation/views
through public GGML, use ggml_backend_cuda_init(0), original
backend_alloc_ctx_tensors and backend_graph_compute; no math rebuild or private
CUDA context construction. Link original libggml/libggml-base/libggml-cuda
from image sha256:837fc732fea84b0d795097a3c8c5706bb16774f1722dab0f70bf6093c60aecc7,
with the authenticated exact b29 headers and library digests.

B: four native one-stream views, existing FlashAttnVec256.
C: those same views, existing FlashAttnMmaGqa2; no selector change in production.
D: one native four-stream descriptor, existing FlashAttnMmaGqa2. Its existing
checked contract accepts multiple streams; PlanFlashAttnMmaGqa2 pays the mask
prepass and stream-k/fixup. This isolates stream packing/partition changes from
B-versus-C query conversion. No new kernel is needed.

Preserve original stream pitch where witnessed. Expected original Q strides are
[4,32768,1024,32768]. For the existing native context256, local cache capacity is256 and the
constructed four-stream pitch is256*4096*2=2097152 bytes, hence K/V strides
[2,8192,512,2097152]. Preserve actual captured root metadata; this constructed
packing is not a witnessed stock cache descriptor. If replay root padding is newly zero-initialized, label
that reconstruction and never call its unread padding captured stock bytes.
Fund and validate complete root/view bounds. The acquired native masks already
have 32 rows; replay copies every mask byte, preserving the required full MMA
tile and negative-infinity unused rows. Each real query row must retain precisely
its captured visible cells. Record this packing as a replay construction.

Outputs normalize only layout: A/D [256,32,1,4] become owner/head/dimension;
B/C already one [256,32,1,1] result per owner. No numerical conversion or token
sampling. Compare every 32768 F32 output value and four argmaxes, byte equality,
finite status, maximum raw delta and declared independent attention NMSE.
Own repeat bounds are frozen per arm; no tolerance widening to absorb stock.

## Actual launch observation, paid work, ownership

Record native registry identity and planned query tile, mask-prepass/scratch,
blocks and stream-k/fixup requirements. Occupancy is not directly observed by
the exposed plan record. Original-model family remains a static source inference. A separate, bounded
CUDA launch-name observation establishes the actual family selected by the
unchanged backend on these constructed descriptors. A separate bounded nsys launch-only arm can confirm selection, with no eval callback, fusion env or graph
keep changes; traced duration is not performance. Preserve exact capture versus
eager/replay counts. Do not call API op counts CUDA kernel launch counts.

After correctness, one short untraced A/B/C/D/A bookend:
all arms use the same fixed payload, 32 C4 waves/128 completed owner units,
copy and inspect all
four complete attention outputs each unit. Exclude input upload, cold setup,
warm execution, reset and capture; include actual replay/submission, completion,
D2H publication and full-output inspection. Report A/D one shared op versus B/C
four separate ops per unit and actual CUDA launches from observation separately.
This is operand timing, not full model speed or adoption evidence.

Before allocation reserve a bounded host grant and all device roots, masks,
outputs and worst scratch for the selected arm; include original foreign backend
allocations explicitly rather than count them as native catalog allocations.
Mapped/pinned addresses remain stable across capture. All staging reuse and
output reads follow checked completion. Heap-own provider/streams/contexts,
original backend/buffers, native launch/captured graphs and allocations as one
lifetime bundle; retain all on any unproven retirement. Check Q/K/V/mask witnesses
before/after every arm and fresh-query capture replay, preserve failure records.
No D512 acquisition now: its stream count enters ntiles_dst and stream-k block
partition (fattn-common.cuh:1093-1096,1137-1174); do not infer a changed per-query
partition without actual occupancy/grid evidence.

The trace wrapper requires an exclusive completion receipt written only after
final synchronization and backend buffer/backend/context release, plus exact
first/repeat outputs. Application completion is authenticated independently of
profiler exit. Trace durations are excluded from the untraced bookend.
