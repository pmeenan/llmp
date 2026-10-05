<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dense31 scalar layer0 FFN identical-operand screen (planning only)

Task entry 2026-10-05T11:42:02Z; TensorFold HEAD freshly resolved to
609ca419abecebdc5a059498a613680bd3aa847f, primary files re-fetched at that pin,
version0.6.5. Gemma26 MLX-only recipe, no dense31 CUDA comparator. Dedicated
read-only planning worktree gemma-dense-ffn-replay@c67dd4d. This protocol fixes the benchmark-only acquisition and replay contract.

## Source-backed feasibility

Original b29c606e28a01b1bc8c1351026a0fa6e616bf6c4 ggml-cuda.cu freshly read
with git-show from immutable owner clone (no checkout). Whole source SHA
523470d6604755b82d0208414ce40f1378941b10bc1349763bbdf02edaab9634,
identical to root's historical copy.

* Lines1740-1756 require matching gate/up quantized types/shapes/strides and
  exactly the same F32 input; GLU must be unswapped GeGLU (zero clamp).
* Lines1794-1820 allow quantized/F32->F32 MMVQ fusion on post-Pascal devices,
  subject to padding/view safety, MMVQ batch bound and one destination column.
* Lines3937-3979 recognize adjacent MUL_MAT,MUL_MAT,GLU and call the original
  ggml_cuda_mul_mat_vec_q with gate weights/GeGLU metadata, skipping both
  materialized product outputs. No floating MMVF conversion is involved.
* Native Gemma4Graph at341-343 emits up/gate products and split GeGLU/down.
  Ordinary planning uses existing SelectMulMatQ/PlanMulMatVecQ/MulMatVecQ
  and GeGlu. Native generic floating GeGLU fusion is a different path and
  is off. Existing VecQGlu writer synthetic N704/N2112 evidence does not
  establish real dense31 N21504 behavior.

The existing original public backend/image client seam from c67dd4d can run
this graph against immutable uploaded quant bytes; no original math rebuild,
private CUDA context or changed original evaluation callback is needed.
Full-original-model fusion remains inferred from source until separately
observed; replay fused versus separate selection must be observed in a bounded
launch-only trace or equally direct existing selection logging.

## Closed native origin and metadata admission

One dense31 owner, supplied natural first64 IDs plus three forced warm anchors,
first subsequent scalar query P67. Context256/max_rows128. Ordinary products,
row_invariantOFF, checked normBOTH, all other optional policies OFF: same
upstream policy as the previous attention operand experiment, now genuinely C1.
Target artifact32c92e... and canonical 4096-byte IDs b2d7aaf6... authenticated.
Explicitly scalar, not C4 repacking or an assumption C1 equals a joined policy.

Before allocation/model acquisition, authenticate layer0 binding metadata from
approved prepared artifact: up/gate [5376,21504], down [21504,5376], exact
quant type/block size/ne/nb/logical bytes/readable tails/group/resource identity.
Gate/up are Q4_K (type12), same type/stride; down is Q6_K (type14).
The approved source fixture establishes this mixed format; the prepared metadata
read must confirm it. If any premise differs, stop and report before expanding
the closed contract; no weight conversion, synthesized matrix or copied Q4K704 data.
Installed metadata1/metadata2 both officially retired rc0, no payload/model
loads. Manifest SHA/artifactID32c92e077a6816b54aa988e2dee61a3639c958fd510ea99e25f3621f10b2aa08,
indexSHA9ae365a17ed73528262d7befdb505bf07e6695e03f0fa36f69e69099dbd5f34c.
Group1 layer0 metadata confirms gate/up Q4_K [5376,21504], each65028096
logical/65028240 readable bytes (one144-byte tail); down Q6_K [21504,5376],
94832640 logical/readable bytes. Expected quant row pitches3024 and17640
bytes respectively; capture checks actual nb/root coverage before copying.
Total logical224888832/readable224889120 bytes. Exact offsets remain in
external metadata2.log and the capture manifest.

Benchmark-only --wrap MulMatVecQ captures the first exact layer0 gate/up/down
resource descriptors, identified against independently checked binding indices
and actual addresses observed through a second link wrapper of existing public
PagedWeights::resource_address (not merely first arbitrary matrix of that shape).
Both wrappers remain benchmark-only; compiled symbol/call-path evidence is kept. Validate the
scalar shared F32 input identity, full root/view bounds, immutable weight
resource extents and exact op_params. Each gate/up/down product and GeGLU parameter block is copied
to params.bin and metadata; both clients refuse anything that differs from the
constructed closed ordinary product/unswapped GeGLU descriptors. Queue completed D2H of the actual input
and logical quant weights on the exact existing launch stream, before the real
launcher; before down also copy actual GeGLU activation. Arm the capture only for the final query P67, after earlier prefill/warm
work; match exact layer0 resource identity and scalar column before installing
copies. Immediately after the real down launch returns, queue its complete
5376-F32 output D2H on the same stream before any next consumer can overwrite
activation placement. Do not copy down after the whole query. Validate these seams
before acquisition; no graph keep node, placement change, producer retention,
new production observer or lifetime ABI is introduced.

Keep actual weight row strides and canonical tails; reconstructed original
backend weights use zero-filled extra-row roots with WEIGHTS buffer usage;
public tensor_alloc binds exact-shape leaf descriptors at the root bases, so no
VIEW node interrupts the adjacent products. Complete constructed tensor roots, captured readable tails and constructed
extra-row zeros have GPU byte witnesses. Public allocated sizes are reported
separately; extra backend allocation padding is cleared at setup and is not
claimed as a full-allocation byte witness. This
physical padding is declared separately from captured native root metadata.
Two independent captures + two disabled controls must have byte-exact complete
262144-value model heads. Actual captured activation/down must repeat exactly;
Both target capture/control arms use graphs=false, so the final P67 arm is
observed eagerly and copies execute only on that query. No graph-on target
fidelity is claimed. Copying is an untimed diagnostic; record capture/source/ID/
policy/binary/descriptor/weight hashes and successful official retirement.

## Three identical-input arms

A: original unchanged image/public backend, adjacent gate/up->GeGLU graph,
   original default fusion ON, followed by a separate down graph. The activation
   is the first graph's output in ALL arms, with a stable caller-owned buffer;
   no gate/up producer is kept to obtain an intermediate. This isolates the
   original triple fusion while making both activation and down observable.
A0: same original backend/weights/input, separate gate and up graphs followed
    by a separate GeGLU graph, then the same down graph. This intentionally
    prevents the adjacent triple from fusing; no global fusion-disable
    environment or rebuilt math. Dedicated bound materialized gate/up buffers
    and no host roundtrip between these graphs.
B: native existing ordinary MMVQ gate/up + split GeGlu, then the same down,
   planned scratch/preparation included. This is the native primitive recipe
   from which captured whole-target outputs came; no native fused-writer policy.

The first discriminating comparison is eager completed full activation/down
plus separate actual launch observation. Capture/replay freshness controls use
existing replay seams only when needed for the subsequent short paid block;
no broad eager-versus-graph ladder is a prerequisite.

The dense31 hidden/input/down width is5376;2816 belongs to Gemma26 and must
not appear in dense31 allocation or output bounds.

Before A/A0 original output exposure, freeze B source/binary/input/weights and
complete own repeated activation21504 F32/down5376 F32 outputs, plus source
capture model-head controls. Native B eager reconstruction must equal BOTH
actual captured complete activation/down arrays exactly before oracle exposure. A and A0 each have independent whole-output own
repeats. Compare all activation/down bytes, finite status, maximum raw delta,
reference-energy NMSE; activation/down coordinate argmax is not vocabulary
quality. No inferred noise allowance or source correction after oracle exposure.

GPU-root byte witnesses of input and all three weight buffers before/after
EACH arm, outside timers; changed input eager versus captured/backend replay
must change output and repeat exactly, then restore original input/output.
Original input/weights stay immutable, output/arena/scratch addresses stable.
Native launches/copies/capture execute under held request and node.Jobs; partial
submission reports Unknown, whole lifetime retained on unproved retirement.
Original public-backend failure fail-stops; completion marker after final
synchronization/resource release, independently of profiler exit.

## Paid screen and boundary

After correctness, one short untraced A/B/A0/A bookend, 32 repeated full FFN
chains per arm, fixed actual input, full activation+down D2H/finite/coordinate
scans included. Include Q8 preparation, gate/up, GLU and down; setup/uploads/
warm/reset/capture/root witnesses/disk output excluded. Report backend/native
submission differences and graph/operation groups separately from actual CUDA
launch observations. Trace-only timings never enter performance. Expected
2–3min for supervised builds/capture plus <1min for bounded arms, each installed
--gpu timeout<=600; no new paid job until closed implementation is reviewed.

Pre-fund exact logical weights: gate/up65,028,096 bytes EACH (Q4_K block144),
down94,832,640 bytes (Q6_K block210), total224,888,832 bytes, plus actual
readable row tails/root pitches and full captured
buffers, all input/product/activation/down/output storage, arenas, Q8 staging,
max planned scratch, pinned witnesses and bounded host grants before allocation.
Node catalog protects every native mapped root through completed work; foreign
public buffer occupancy excludes library/pool peak. Whole heap lifetime owns
node/providers/streams/plans/graphs/buffers/borrows, retained on unknown failure.
Host/device witness buffers are completion ordered, not recycled after a failed
copy. Native acquisition similarly pays captured full weight/payload copies.

The screen can distinguish fused-versus-unfused FFN arithmetic on ONE real
native input. It is not an original-target operand capture, whole-model quality
closure, C4/attention diagnosis or policy adoption. Gemma26's shared/expert
shapes and actual weights are the next backward-transfer eligibility check if
results justify it; synthetic routed N704 is not treated as that qualification.
