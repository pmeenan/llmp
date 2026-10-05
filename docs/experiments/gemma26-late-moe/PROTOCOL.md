<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma26 common-input late-prefill MoE probe

Stage B is committed as 23de091bf5c73450a8df8255eefe7b895dff1627.
This additive diagnostic preserves its measured helpers and inputs. Initial source
and narrow builds are approved; capture and operator acquisitions require the
separate authenticated source, binary and own-output admission gates below.

Fresh task entry: 2026-10-05T16:05:40.504638Z. Official TensorFold HEAD
609ca419abecebdc5a059498a613680bd3aa847f/version0.6.5 remains Gemma26 MLX-only;
no same-format CUDA comparator applies. Primary snapshots/URLs/hashes and entry
receipt a1af56617bdbc0e8d43d13ac3d2ddf1310874099d9eb372f8a41458302fae035
are in gemma26-late-moe-planning. A later delayed acquisition refreshes again.

## Question and factual boundary

Stage B's packed norm+route/reduce policy has92/128 exact heads but two positive-
margin choices (wave3/owner1/input71 and wave4/owner3/input74), versus68 control
misses. Each policy repeats its full heads/initialized states exactly. That does
not establish a late-prefill cause or identify the first divergent full block.

Stage A ACTUALLY observed the original physical C4/u128 recipe: each prefilling
64/65/66/67 call routes layers0–27 with fusion, layers28/29 without fusion due to
memory=0 (structural=shape=1); all30 scaled reductions are fused. Decode4 selects
all30 of both. No layer whitelist, alias imitation or reference-bug claim follows.

For actual layer28/rows64, upstream operands recorded:
- ffn_moe_logits-28: packed F32[128,64],32,768B; SOFTMAX probabilities alias this
  storage, so raw logits must be copied BEFORE SOFTMAX, not after the graph.
- first8 ID VIEW: I32[8,64] with nb1=512B/full128 root pitch. Only first8 per row
  are observable in the fused contract; never compare or consume unwritten tail.
- normalized weights: F32[1,8,64],2,048B; exact clamp2^-14 and slot order0..7.
- raw expert down: F32[2816,8,64],5,767,168B, scales F32[1,8,64],2,048B; routed
  weights same shape, destination F32[2816,64],720,896B.
Layer28 down is actual Q5_1; layer29 down is Q8_0, fused gate/up actual Q4_K.
This probe starts AFTER those products and does not substitute a quant format.
Native whole-model frontier-head prefill narrows the final layer before its FFN.
Captured original layer29 routing/reduction therefore has64rows while that
native whole-model prefill path has frontier1. Native64 operator replay is an
identical-original-operand diagnostic, not evidence of native layer29 whole-model
shape or a unique residual cause. Layer28FFN can feed layer29 cached K/V;
layer29FFN does not feed another layer's prefill cache. No additional1row arm
is implied by this scope.

Router products and incoming activations/state may already differ. Common-input
operator results can test the MoE contract hypothesis, not prove whole-model
history parity or explain every head.

## Smallest initial acquisition: owner0 post-reset prefill64, layers28 AND29

Reuse the exact original C4 client01889d8c, inputb2d7/4096B, approved raw/artifact,
image837fc732 and original floating library5a13585e. Context256 per owner/F16KV,
fusionON/graphsallowed, four histories64..67, eight warm/reset, three anchors,
32forcedwaves/full128heads remain identical to Stage A/B. No timing claim from
instrumentation. Only two late-layer operator input/output groups are captured.

Derive an external controller from exact523470... source and preserve its
floating launch arguments, original fusion predicates, allocation hints, tensor
readers/keeps and graph edges. Match concrete descriptor roles/phase/layer,
not an unsafe name substring or a production dispatch policy. Observe full30
route/reduction decisions for each evaluated phase as before. The captured
phase is the second evaluated64-row prefill, authenticated to client reset
chronology; never silently substitute a warm invocation/cached replay. Fail if
phase identification or stream-capture state prevents that exact witness.

Capture uses separate bounded preallocated PINNED diagnostic destinations and
copies on the ORIGINAL producer stream, retired before any reuse/free. Inputs
are copied immediately before the relevant original operation; output copies
immediately after the last relevant operation. All selected copies are queued without an interior synchronization. Final public
backend retirement and device synchronization precede inspection or release. Do not enqueue pageable
copies, inspect host values before retirement, overwrite/repoint any original
operand or add graph keeps. Do not change CUDA graph policy to make capture work:
if the selected call is being captured, first establish a legal unchanged-policy
capture seam; otherwise report an instrumentation limitation and stop.

Routing: capture raw logits before the selected original SOFTMAX call and
selected ID values immediately after ARGSORT/first8VIEW plus normalized weights
immediately after DIV. The source full-root backing/span/pitch is authenticated;
canonical exported IDs copy only each row's8 values, with physical pitch retained
in metadata. No read of the fused-sort tail. Reduction: capture actual rawdown,
selected scales and routing weights before the original fused launcher, and
its complete sum after it. Actual physical input/output descriptors and alias
metadata are recorded, not recreated from expected shapes alone. Both layers
retain actual selected experts and ascending slot mapping.

Require instrumented original128head file EXACT both StageB fresh references
SHA5808306e... and unchanged actual routing/reduction decision tables. Preserve
output blindness through input-only manifests: captured reference heads are
only a fidelity hash; operator outputs remain withheld until native own-freeze.
If final heads or selected decisions change, the capture is intrusive: stop,
retain failure, and do not infer operator cause from it. Budget/retirement proofs
cover original context, pinned input/output groups and library owners, with
unknown completion retaining all owners. Two rawdown inputs total11,534,336B;
logits/scales/weights/IDs/sums are bounded separately. No broad endpoint ladder.

## Two sequential common-input controls, source fixed before output release

1. ROUTING input-only release: exact captured rawlogits64x128 for both layers,
   immutable finite input/descriptor/phase receipts. Native same-input fused and
   actual primitive-fallback chains each run twice, full first8 IDs plus all
   normalized weights frozen exclusively BEFORE reference route outputs appear.
   Native source and both original comparison client branches compile/freeze
   before release. Then compare original untouched-model primitive routing
   outputs, original direct exported fused routing, and an isolated original
   fully separate10-node chain on those exact logits. The isolated primitive chain calls the original single-node exports in its
   actual constructed dependency order and records each implementation. It
   consults no generic fusion gate; the defining full-model capture retains
   untouched stock gates and allocator hints. No sorting-tail outputs.
   Original direct fused must use softmax=true/norm=true/clamp2^-14, null scale/
   bias, same strides/full128 ID-root funding. IDs/ties/order and weight bytes,
   finite values, maxraw and full normalized sums are compared independently.

2. REDUCTION input release follows routing native-own freeze: exact rawdown,
   scale and ORIGINAL normalized weights are now a declared common-input fixture.
   This avoids exposing reference routing output through reduction weights before
   native routing calibration. Native fused plus actual primitive17-node ordered
   scale->weight->eight-view/seven-add chain each run twice and freeze full sums
   before original sum exposure. Original exported fused and isolated fully
   separate17-node diagnostic use precisely the same operands. No DeepSeek
   reducer, omitted scales, reordered sums or artificial padding/row128 promotion.
   The actual rows64/width2816/top8 are preserved. Original generic MUL/ADD fusion
   variations are a separately labelled future arm, not required initially.

Routing and reduction are separate operator comparisons. If IDs differ, do NOT
feed new weights into expert tensors computed for old IDs and call that a full
chain. Reduction uses frozen captured IDs/weights/experts as one coherent fixed
operand set. No upstream product, norm, shared-FFN or cache arithmetic attribution
is made by a reduction-only equality.

Native execution reuses checked wrappers/primitive planner + PagedNode,
RunnerResources, GraphRuns, funded mapped inputs/outputs/scratch/activation roots,
complete closures and completion-aware pinned publication. No foreign backend
runtime in native. Fresh captured replay and input byte witnesses establish that
common inputs are unchanged, with all actual outputs finite before publication.
Original thin external client may use its original backend context ONLY outside
jitLLM; existing original-image standalone MoE oracle already proves both exported
seams. No original kernel rebuild is needed to obtain these outputs.

## Defining translation units and compiler identity, before attribution

Authenticate exports/loaded-object binding (dladdr/maps or equivalent) so direct
reference ggml_cuda_op_topk_moe and ggml_cuda_op_moe_weighted_reduction resolve
in original5a13585e libggml-cuda, not a client/controller replacement. Original
floating definitions are topk-moe.cu and moe-weighted-reduction.cu at b29; native
uses those exact prepared upstream sources through gemma_moe.cu wrappers,
source lock84226f/prepared026f. Record source/export/header hashes and native
compile_commands/buildreceipt flags. Native SDK CUDA13.4.92/Clang22.1.8; immutable
original image's authenticated toolkit environment CUDA13.3.0/CUDART13.3.29-1.
Do not mistake image toolkit metadata for an unavailable exact floating-TU build
command; retain that qualification unless its original build receipt proves more.
The observation controller rebuild also defines the unchanged INTEGER pointer-
preparation kernel. List that separately; it is not a rebuilt floating router/
reducer. The thin clients define no floating operators. Primitive implementation
symbols/true source TUs and selected native steps must likewise be recorded:
whole-file source paths alone do not prove which implementation executed.

## Decision and bounds

Fused native==original on real operands establishes this port's operator fidelity;
original fused!=primitive on same logits would establish a real arithmetic effect
of the observed late-prefill gate. Primitive native==original separately checks
fallback. Fused route equality but modelhead inequality cannot exclude different
incoming router logits/KV/shared work. Reduction equality only excludes that
contract on the captured operand set. Neither result selects a policy, proves
reference wrong, or authorizes per-layer gating/alias imitation.

First scope: one prefix64/layers28+29 capture with full128head fidelity, two native
repeat operator modes and original direct/isolated controls. Narrow builds and
acquisitions use installed A GPU600/stop-on-fail after root source/protocol
approval, roughly10–30s each; compilation/source-proof can take longer and is
announced. No C12, PPL, long-context or wholemodel omission ladder. Expand to
owner1/prefix65 or a real incoming full-block replay ONLY if this first result
leaves a specific decision unresolved and root approves. Raw tensors/logs stay
external; aggregate exact IDs/weights/sums deltas and provenance in final packet.
Independent GENERAL/root ADV/light precede diagnostic commit; no agent commits.

## Implemented capture and operator boundaries

`capture_controller.py` inserts copy-only callbacks over the committed Stage A
observer. Removing both marker sets reconstructs exact controller source
523470d6604755b82d0208414ce40f1378941b10bc1349763bbdf02edaab9634.
It rebuilds the controller translation unit, including its unchanged integer
pointer-preparation kernel. The original floating operators remain in image
libggml-cuda SHA 5a13585ed1dc0263639e47f5154b0ea51c542df555e391a7b753daa73a56934e.
The external clients also compile host SHA-256/check helpers; these define no
floating CUDA operator. The raw capture is not a timing arm.

The host wrapper preallocates 14 independent pinned destinations under a
16 MiB cap before invoking unchanged C4 client source c80b73c85453ece170eda57bc0725cdef5b4e0af3d7cacf3d1a8bee0d1cfa19e.
It counts all actual 64-row graph calls, selects only occurrence two, requires
that call to evaluate or record rather than replay an uninstrumented graph, and
refuses any later 64-row invocation. Both capture policy booleans and each
copy's stream-capture status are recorded. Copies never synchronize inside
capture. The fixed destinations survive every graph/backend until the original
client's explicit public teardown and a final known device synchronization.
Unproven completion retains owners to process exit. The exclusive owner-only
files are flushed only after complete finite/canonical snapshots.

`capture_freeze.py` requires all 128 head bytes to match both untouched controls,
actual canonical input IDs, and complete unchanged per-call/all-phase fusion and
product decisions. Address metadata is retained externally but is not compared
as a policy. Its receipt links official successful supervisor retirement, owned
container cleanup and the pre-capture compiled identity. No operator input is
admitted if capture changes any measured mathematical output or gate table.
Only two logits files are initially released; original route outputs remain
withheld. Reduction input release includes original weights only after native
routing own-freeze. Captured sums remain withheld until native reduction freeze.

The new manual `jitllm_gemma26_late_moe` target shares graph construction with the
external original operator client. Native metadata controls launch nothing and
report actual descriptor/plan counts. Execution maps every complete root, funds
inputs/outputs, host retention, pinned uploads, workspace and captured graphs,
and holds their catalog closure including activation/pool extents. Every upload
retires before pinned staging reuse. Each layer first runs eagerly, then through
three captured replays. Layer 29 replaces the input bytes in stable storage,
providing a fresh captured-input control. Before/after whole-input witnesses
and independent process repeats are mandatory. Full-root ID tails are allocated
and initialized but only the selected eight entries per row are published.

The external primitive client calls the pinned library's actual softmax, argsort,
get-rows, sum-rows, clamp, div, mul and add exports for the constructed computed
nodes; views launch nothing. Every actual call logs its symbol and defining
library. Fused controls call the original top-k and scaled-reduction exports.
No new floating kernel source, primitive arithmetic or quant format is compiled.

`reference.sh` gives every Docker build/capture/operator run its own fixed CID,
name and ownership label. EXIT/INT/TERM cleanup verifies the same owner before
removal and checks absence. A preserved CID permits supervised SIGKILL recovery
with identical ownership checks; failed/missing proof is not successful model
retirement. Installed-supervisor success, app/device completion and container
absence are distinct proofs, all retained in external provenance.

`original_freeze.py` is a required counterpart-admission check. It links the
precompiled source/binary, native own-freeze, captured receipt, exact incoming
manifest and officially retired two-arm original acquisition. It authenticates
app/device markers and both owned container absence proofs before the comparator
accepts any counterpart file. The comparator independently checks that linked
captured role hashes and every original/native output file remain unchanged.

Prior gate identities are fixed outside these receipt files by the root reviewer.
`capture_freeze.py` takes the admitted PRE SHA; `own_freeze.py` takes admitted PRE
and captured-receipt SHAs; `original_freeze.py` takes admitted PRE, capture and
native-own SHAs. Each validates all supplied bounded receipt bytes before parsing
them or reading operator outputs. Input release requires the admitted capture SHA
for either kind and the already admitted routing-own SHA before reduction weights
can be exposed. `compare.py` requires all four externally fixed PRE, native-own,
capture and original-admission SHAs. It reports those identities in the aggregate
in addition to checking internal links. A replaced self-consistent receipt/output
chain cannot replace an earlier admission or bypass the blind release boundary.
