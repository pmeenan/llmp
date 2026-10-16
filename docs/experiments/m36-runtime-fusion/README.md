<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# The run-time fusions of `BoundGraph::Run`

`b37f763f` made the shared Qwen-Image path competitive by pattern-matching
the planned GGML steps of every spatial or region graph at run time
(`src/kernels/ggml/executor.cc`, `BoundGraph::Run`) and launching fused
kernels in place of the steps it matched. The planner never sees these
choices (against D-053/D-107's "the plan selects"): no registry entry, no
plan-vs-GGML comparison, no placement-derived liveness. With them off the
1024²/40-step generation takes 95.70 s against 34.1 s with them on
([m36-baselines](../m36-baselines/README.md#qwen-image-21-on-the-shared-engine)).

This record is sub-task 1 of their move into planner patterns and registry
implementations: what each matcher does, what it depends on, the test that
holds it against the reference executor before any code moves, and where
each lands. `kernels/ggml/executor.h` `RuntimeFusion` named them; a test's
`RuntimeFusionControl` turned any matcher off and counted what applied.

**Status: deleted.** Sub-tasks 2a–2e moved every matcher into planner
patterns and registry implementations, and sub-task 3 made that plan the
image's production policy and deleted the matchers, `RuntimeFusion` and
the control (below, "The run-time path deleted"). `BoundGraph::Run` now
runs its bound steps in order, nothing else. The sections up to
"Findings" are the record of what the matchers did and what replaced
them.

## When it runs

`Run` takes the matcher loop for any graph with `regions` (when the launch
context has no lanes) or with a spatial step (`has_spatial_ops_`: a step
bound to `ggml.attention.f32`, `ggml.conv2d.tiled_f32`, `ggml.l2_norm`,
`llmp.image.spatial_channel_norm`, `ggml.norm` or `ggml.repeat`). Since `d36922cb` only DeepSeek's serving
carves lanes, so every other region graph (LLMs' too) also takes the loop;
none of its matchers is known to match an LLM graph, but nothing prevents
one.

## The matchers

Each matches a run of consecutive planned steps by implementation name and
tensor wiring, checks that the run's intermediates are not read later
(`used_after`), and launches its kernels in place of the run.

| # | `RuntimeFusion` | Matched steps | Launches | Replaces |
|---|---|---|---|---|
| 0B | `split_broadcast` | cont, repeat, cont, repeat, concat of two `[1, C]` vectors | two conts into the concat's last two rows | the broadcast concat (its readers take `SplitBroadcastWhc`) |
| 0 | `spatial_norm` | cont(permute 1203), l2_norm, scale, mul, [SiLU], cont(permute 2013) | `SpatialChannelNormF32` (+SiLU) | channel RMS norm round-trip through rows |
| 1A | `modulate` | cont(x), cont(side), norm, scale(1, 1), mul | `LayerNormModulateF32` (x spatial), optional BF16 copy in workspace, optional skipped F32 | layer_norm(x)·(1+s) with both layouts' conts |
| 1B | `modulate_rows` | cont(side), norm, scale(1, 1), mul | `LayerNormModulateF32` (x rows) | the same with x already rows |
| 2 | `norm_rope` | [convert w], rms_norm·w, 1–3 × (cont(pos), rope.ext) | `RmsNormMulRope3d` | per-axis rotations of a normed head |
| 3A+1A | `gated_residual_modulate` | cont(branch rows), tanh, mul, add, then 1A on the sum | `GatedResidualLayerNormModulateF32` | residual, its layout change and the next modulation in one pass |
| 3A | `gated_residual_rows` | cont(branch rows), tanh, mul, add | `GatedResidualF32` (branch rows, optional BF16 branch/carry) | residual with the branch's layout change |
| 3B | `gated_residual` | tanh, mul, add | `GatedResidualF32` | x + tanh(g)·y |
| 4A | `rows_projections` | cont(rows→spatial), then 1–3 × (cont(permute 1203), mul_mat) of it | the products from the rows directly (`MulMatCublasWithReuse`), or QKV/MLP below | the spatial round trip before projections |
| 4B | `spatial_projections` | cont(permute 1203), mul_mat | the product, reusing a BF16 conversion of the same input already in workspace | a repeated conversion |
| — | `qkv_attention` | (inside 4A/4B) q,k,v products, conts, two `norm_rope` runs, two concats, two conts, `ggml.attention.f32` | BF16 Q/K/V products into reused intermediates, `RmsNormMulRope3dQKInplaceBf16`, `AttentionF32WithOptions` (BF16 row-major QKV, query norm in the kernel, packed mask, BF16 output) | about 30 steps a block |
| — | `mlp` | (inside 4A/4B) gate product, cont, cont, up product, cont, SwiGLU, cont, down product | gate/up products, `SwiGluToBf16`, down product reading BF16 (optionally writing BF16 for 3A) | about 8 steps a block |
| 5 | `conv_bias` | `ggml.conv2d.tiled_f32`, add(bias) in place | `Conv2dTiledF32WithBias` | the separate bias add |
| 6 | `dup_up_add` | cont, repeat, cont(view), cont(permute), cont(permute), add | `DupUpAddF32` | the shortcut's duplicate-up and add |
| 7 | `conv1x1` | cont(transpose(reshape x)), mul_mat, cont(transpose), add | `SpatialConv1x1F32` | a 1×1 convolution through rows |

Counted variants (not separately controlled): `qkv_bf16` (the BF16 QKV
form), `query_norm_in_attention`, `attention_bf16_out`, `mask_prepacked`,
`reuse_converted_input`, `skip_f32_modulation`, `bf16_branch`.

## What they depend on

- **State across matches, within one `Run`.** A BF16 conversion of a
  product's input left in the launch workspace and reused by later
  products of the same input (`ws_bf16_*`, `MulMatCublasWithReuse`'s
  `reuse_converted_src1`); the last packed attention mask and rotary table,
  keyed by tensor pointers (`cached_attn_*`, `cached_rope_*`); BF16 copies of
  an attention output or a down product read by the next match
  (`direct_bf16_rows`, `bf16_branch_tensor`); the rows layout of a residual
  sum (`rows_layout_*`); elided broadcasts (`elided_bcasts`). Each is
  invalidated by heuristics over the next step's kernel name and output
  address, not by the plan.
- **Predictions of a later match.** A matcher writes an output only the next
  matcher's kernel can read when it predicts that match: `skip_f32_modulation`
  (no F32 modulation output), `attention_bf16_out`, `bf16_branch`, the
  combined residual's BF16 rows, `split_broadcast` (no broadcast concat).
  If the predicted match then fails its own checks, the planned step reads
  bytes that were never written. The test control enforces each
  prediction's dependency (a predictor requires its consumer allowed); in
  production nothing checks it.
- **Liveness.** `used_after` scanned later steps' sources only, ignoring
  the step's graph outputs and kept tensors (`LoweredStep::keep`), so an
  intermediate that was also an output was elided silently. Since 2a
  `PlanGraph` records the roots of what it was told to keep
  (`GraphPlan::kept`) and both bind paths (`BoundGraph::Bind` from
  `engine/planned.cc`, and the step programs) hand them to `Run`, whose
  `used_after`, broadcast elision (`can_elide_concat_bcast`) and combined
  residual (`only_used_by_norm_and_next_residual`) treat them as read later.
- **Scratch.** The fused QKV group, the residuals and `conv1x1` write into
  other intermediates' memory chosen at run time (`tmp_q`, `q_buf`,
  `find_scratch`), checked disjoint only from the operands the matcher names,
  not from everything the placement keeps live there. The spatial norm,
  the norm and rotation, the modulation over rows and the plain gated
  residual now also check that their output is exactly, or shares no byte
  with, each operand they read; the last two refuse the step loudly where
  a declined match would leave a predicted output unwritten (finding 2).
- **Profile constants.** `mod->ne[0] == 4096` selects the combined
  residual's BF16 rows; the BF16 QKV form needs head dimension 128 and equal
  rotary grids for Q and K.

## The reference tests

(The record of the run-time path's test; since sub-task 3 the test runs the
planned path only, below.)

[image_fusion_gpu_test](../../../tests/unit/image_fusion_gpu_test.cc)
builds graphs with the production spatial components and lowers them
through the production GGML backend, declared as the image source declares
it (masked F32 attention tuned for each flow's joint rows; the tiled
convolution for the decoder's 3×3 shapes), with synthetic weights:

| Case | Graph | Reference |
|---|---|---|
| `flow_w4096` | one flow block at the source width (32 heads of 128, MLP 1,024), 8 text rows and a 16×16 latent (264 rows) | the reference executor (double) |
| `flow_source` | two blocks of the source's geometry (MLP 12,288), 25 text rows and a 64×64 latent (4,121 rows: the 1024² image) | the same graph with every matcher off (the oracle is out of reach at this size) |
| `decoder` | the decoder's structure at base 16, z 16, one residual block a level: a 4×4 latent to 64×64 | the reference executor |
| `flow_w256` | one block of two heads (MLP 512), 8 text rows and a 6×6 latent | the reference executor; with every matcher on it must refuse or match (finding 2) |

Each case runs with every matcher off, with every matcher on (which must
equal the production run, no control installed, bit for bit), and with each
of the fifteen controlled matchers left out in turn; every run that
completes must be within the case's bound of its reference (relative RMS:
5e-3 for the flows against the oracle and the decoder, whose tiled
convolution rounds its operands to BF16; 6e-3 for `flow_source` against its
unfused run). A matcher turned off never applies, and every matcher and
variant applies in some completed run. Spark A, 2026-10-15:

| Case | All off | All on | Each left out |
|---|---:|---:|---|
| `flow_w4096` (vs oracle) | 2.337e-3 | 2.358e-3 | 2.34e-3 to 2.37e-3; `without_qkv_attention` refused |
| `flow_source` (vs all off) | 0 | 3.893e-3 | 1.26e-3 to 3.89e-3; `without_mlp` refused |
| `decoder` (vs oracle) | 2.901e-3 | 2.898e-3 | 2.898e-3 to 2.901e-3 |
| `flow_w256` (vs oracle) | 2.275e-3 | **0.2099** | 0.20 to 0.22 unless the split broadcast or a gated residual matcher is left out (2.33e-3); `without_qkv_attention` and `without_mlp` refused |

(The first output, the flow's noise prediction; the second, the updated
latent, is about ten times smaller in each row.)

After 2a (the root fix and the two planner patterns below), Spark A,
2026-10-15: `flow_w4096` all on 2.358e-3, `flow_source` 3.888e-3,
`decoder` 2.898e-3, unchanged; `flow_w256` with every matcher on now
refuses ("an elided broadcast gate shares its gated residual's output
bytes") instead of returning a 21% error. The decoder's plan selects
`llmp.image.spatial_channel_norm` 26 times and the flows' plans
`llmp.image.rms_norm_mul_rope_sections` for every QK norm; the two
run-time matchers they replace apply nowhere (the test requires zero
hits), and the QKV matcher takes the planned step in place of its
`norm_rope` run. Every case's "none" arm (all matchers off) must run and
match its reference.

End to end (Spark A, 2026-10-15): the image's pixels become `2ee8f010…`
(the text encoder's QK norms now fuse), at PSNR 40.687 dB and luma SSIM
0.99418 against the retained reference `9de442f1…` (36.456 dB / 0.98944
before; floors 32 dB / 0.98). Speed is unchanged: −0.01% in an ABBA A/B
against `b29380a7`, +2.03% against the frozen legacy arm in a two-block
screen (+2.14% before) ([m36-baselines](../m36-baselines/README.md#qwen-image-21-on-the-shared-engine)).

Checks and where they ran: `image_fusion_gpu_test`,
`norm_rope_sections_gpu_test` (now `image_ops_gpu_test`; the kernel against the decomposition: both
rotation modes, one to three sections, F32 and BF16 weights), the planner,
lowering and reference tests, the shared set, the image model tests and
the Gemma frozen v1 proofs, all on Spark A (`spark`).

## Findings

1. **A matcher declining can make the launch refuse the step.** With the
   QKV matcher (or, at 4,121 rows, the MLP matcher) off, the projections
   matcher falls back to running the first product from the rows input it
   redirects the product to, past that input's planned lifetime, and the
   product's own checks refuse the overlap with the bytes the plan reused:
   "step 58: a stale view, or an output overlapping an operand". The production plan avoids
   this only because, at its shapes, the QKV and MLP matchers always
   apply; whether a fallback collides depends on the placement (with two
   blocks, `flow_source` without the QKV matcher runs). The refusal stays
   loud until 2b and 2c move those matchers into the plan. *Resolved:*
   with the matchers gone nothing launches outside the plan.
2. **A silent miscompute at width 256.** With every matcher on, the
   two-head flow's output is 21% off the oracle (and the latent 1.8%). It
   needs both the split broadcast (0B) and the plain gated residual (3B,
   which the source width never reaches: there 3A with its modulation
   applies): leaving out either restores 2.3e-3. The production width does
   not take this path; the planner patterns must not inherit it. Cause: the
   plain gated residual wrote its output over the bytes of the broadcast
   gate the split broadcast had elided (the plan had reused them, as the
   broadcast was dead to it). 2a's guard refuses that launch. *Resolved:*
   the planned `flow_w256` matches the oracle (2.768e-3).
3. **The projections matcher depends on the attention being tuned.**
   Without the masked F32 attention selected (an untuned backend), the
   attention decomposes into products and a softmax, the QKV matcher never
   applies, and the projections matcher refuses as in finding 1 (40 and
   264 rows tried). *Resolved* with finding 1.
4. **The decoder's matchers are exact to their unfused forms** (spatial
   norm with SiLU, convolution bias, duplicate-up add, 1×1 convolution);
   the decoder's error is the tiled convolution's BF16 operands, with or
   without them.
5. **The flow's matchers change the result by up to 3.9e-3** relative to
   the unfused run at the source geometry (BF16 Q/K/V, BF16 activations into
   products, the attention's BF16 output): the M3 pipeline's arithmetic,
   which the quality check (36.456 dB) already covers.

## Where each lands (sub-task 2)

The IR already has the fused kinds the legacy pipeline's kernels implement
(`ir/op.h`: `kLayerNormModulate`, `kGatedResidual`, `kGatedResidualNorm`,
`kDupUpAdd`, `kEulerStep`; `kRmsNormMulRope`, `kMatMulGlu`). Each group
becomes planner patterns (`planner/pattern.h`) emitting a fused node, a
registry implementation per GGML launcher with its capability (`Check`),
lowering in `kernels/ggml/lower.cc`, and the reference executor's
decomposition as its oracle:

| Group | Matchers | IR destination |
|---|---|---|
| 2a rope/norm (landed) | `norm_rope`, `spatial_norm` | `kRmsNormMulRopeSections` (pattern `rms_norm_mul_rope_sections`: the norm, its weight's widening, one to three rotated sections covering the head and their positions' copies; `RmsNormMulRope3d`) and `kSpatialChannelNorm` (pattern `spatial_channel_norm`; `SpatialChannelNormF32`). The first is opt-in: only the image's planning policy enables it (`spec/image_spatial.h` `kImagePlanningPolicy`), so the only LLM-shaped graph it takes is the image's text encoder (its QK norms and whole-head NEOX rotations); a model whose policy disables `rms_norm_mul_rope` (Gemma 4's, DeepSeek's, Gemma 2's) keeps its plans |
| 2b GLU | `mlp` | pattern `bf16_glu`: the gate and up products into BF16 from their input converted once, the SwiGLU into BF16 (`llmp.image.swiglu_bf16`), the down product from BF16 |
| 2c attention | `qkv_attention` and its variants | pattern `qk_attention` (round 1): BF16 Q/K/V products, the key's sectioned norm and rotation into BF16, and `kQkAttention` (`llmp.image.qk_attention`: the query's norm and rotation inside `FlashForwardBf16RowMajorKernel`, BF16 output where only BF16 products read it) over a packed mask (`kAttentionMaskPack`) and rotation table (`kRopeTable`), each made once per step for every attention that reads it |
| 2d modulation/residual | `modulate`, `modulate_rows`, `gated_residual*` | patterns `split_modulate` (`kLayerNormModulateSplit`, BF16 where only BF16 products read it), `split_gated_residual` (`kGatedResidualSplit`) and, in round 1, `split_gated_residual_modulate` (`kGatedResidualModulateSplit`: the sum and its modulation in one pass) over the conditioning's two vectors; with them `bf16_branch` (a product a residual adds, into BF16) and the option `bf16_residual` (a sum only later residuals carry, in BF16) |
| 2e VAE | `conv_bias`, `dup_up_add`, `conv1x1` | the tiled convolution adds its bias (a third source, one step); `llmp.image.dup_up_add` implements `kDupUpAdd`; the components emit a 1×1 convolution as `Conv2d`, which `llmp.image.conv1x1` implements |
| layout | `split_broadcast`, `rows_projections`, `spatial_projections` | layout sinking (`ir/layout.h`, the option `layout_sinking`): the step's builder follows values as views of packed storage, runs elementwise work in the order its operands are in, re-emits layout-free conditioning rows (repeats of vectors) in their reader's order, and keeps the original graph where it cannot copy fewer bytes; BF16 product inputs are planned conversions (`bf16_inputs`) |

Liveness and scratch then come from the plan's placement (graph outputs and
kept tensors included), and caches are keyed by the graph's identity or a
plan input, not by tensor pointers.

## The planned path (2b–2e)

Every pattern above is opt-in; only the image's policy
(`spec/image_spatial.h` `kImagePlanningPolicy`, production since sub-task
3) enables them, so no LLM's plan changes (checked below). Each fused kind decomposes to the primitives the components
emit (`ir/decompose.cc`), and its GGML implementation is tested against that
decomposition on the reference executor ([image_ops_gpu_test](../../../tests/unit/image_ops_gpu_test.cc):
the sectioned norm and rotation over BF16 heads, the mask pack, the
rotation table, the QK attention, both split modulations, the joined
residual over F32 and BF16 rows, the duplicate-up and the 1×1
convolution). The planner tests hold each pattern's match, its opt-in and
its refusals ([planner_test](../../../tests/unit/planner_test.cc)); two
source-geometry flow blocks plan every attention, MLP, modulation and
residual fused and lower ([ggml_spatial_test](../../../tests/unit/ggml_spatial_test.cc)
`PlannedFlowFusesEveryBlock`); the planned flows and decoder run against
their oracles on the GPU (`image_fusion_gpu_test`, below).

Two choices are on by default in the image's policy: the products a gated
residual adds write BF16 (`bf16_branch`), and a residual's sum that only
later residuals carry stays BF16 between blocks (`bf16_residual`). They
reproduce the shipping path's established BF16 activation numerics (its
run-time residual wrote the same rows at width 4,096, from BF16 down and
output projections), and quality against the diffusers reference is equal
or better (PSNR 41.26 dB / luma SSIM 0.99419 with them, against the
shipping path's 40.687 dB / 0.99418; the final state's numbers are below),
so this is speed over bit-exactness at the same quality, not a quality
trade. Both apply at any geometry and width their kernels take (a BF16
sum up to 8,192 channels); their quality is qualified at 1024² on the
model's width (4,096), the one diffusers reference retained; a second
geometry needs its own reference. The split patterns read the two
conditioning vectors directly and leave the rows they were repeated into
unread; the first block joined its text and latent rows twice, once for its
norm and once for its residual (two 64 MiB row concatenations a step,
about 47 ms a generation). Sub-task 5's planner drops the unread rows and
joins the rows once (below).

The VAE's three matchers now never apply in production either: the
decoder's plan selects the tiled convolution with its bias, the
duplicate-up and the 1×1 convolution, the same kernels (the pixels are
unchanged, `2ee8f010…`); `ggml_spatial_test`'s 1,024² decoder launches 78
fewer steps (7,450) in 0.56 GiB less activation extent.

Measured on Spark A, 2026-10-15, before sub-task 3, with an experiment
tree whose production policy is the planned one and whose run-time matchers are skipped
(1,024², 40 steps, two primed ABBA blocks against the current production
path on the same build): 33.908 s against 34.007 s, −0.29% (−0.28% and
−0.30% by block; an A/A pair of the same binaries read +0.12%), peak memory
0.07 GiB lower. Before the two BF16 choices the planned path read +3.8%:
an nsys pair placed it in the F32 residual rows (the joined residual pass
1.08 ms against the run-time path's 0.59 ms) and the F32 outputs of the
two products a residual adds. Quality on the planned path (the same
request and test as above, pixels `9d0b448f…`): PSNR 41.26 dB, luma SSIM
0.99419 against `9de442f1…` (floors 32 dB / 0.98), passing.

Raw records stay outside Git on `spark`: `~/scratch/m36-image-exp0{2,3,5}`
(+3.8%, the A/A pair, −0.29%),
`~/scratch/m36-image-plan/nsys` and `~/scratch/m36-image-plan/q3`.

## The run-time path deleted (sub-task 3)

`kImagePlanningPolicy` is now the planned policy above (the separate
`kImagePlannedPolicy` is gone), and the image's builders sink layouts
because it enables `layout_sinking`. Deleted: `BoundGraph::Run`'s matcher
loop and its helpers (`kernels/ggml/executor.cc` from 2,436 lines to 500),
`RuntimeFusion`, `RuntimeFusionControl` and `SetRuntimeFusionControl`,
`GraphPlan::kept` and `BoundGraph::kept_` (2a's liveness fix, which only
the matchers read), and the launchers only the matchers called:
`LayerNormModulateF32`, `GatedResidualF32`,
`GatedResidualLayerNormModulateF32`, `RmsNormMulRope3dQKInplaceBf16`,
`SpatialConv1x1F32`, `MulMatCublasWithReuse` (and its reuse, BF16-output
and pinned-cuBLASLt flags, which `llmp.gemm.bf16_lt` replaces) and
`AttentionF32WithOptions` (`ggml.attention.f32` keeps its default form).
`Run` runs the bound steps in order (or on the lanes); region graphs of
LLMs, which used to pass through the matcher loop matching nothing, take
the same steps.

`image_fusion_gpu_test` is now planned-vs-oracle only: each case is
planned with the production policy and layout sinking, run once and held
to its reference, and every fused implementation must appear in some
plan. `flow_source`, out of the oracle's reach, is held to the same graph
planned with no patterns and no layout sinking. Spark A, 2026-10-15:

| Case | Reference | Relative RMS (bound) |
|---|---|---:|
| `flow_w4096` | oracle | 2.905e-3 (5e-3) |
| `flow_source` | primitive plan | 3.845e-3 (6e-3) |
| `decoder` | oracle | 2.963e-3 (5e-3) |
| `flow_w256` | oracle | 2.768e-3 (5e-3; finding 2's case) |

**LLM plans are unchanged.** With a temporary hook logging every plan
(its primitive graph's digest, its policy, the planned graph's digest and
each step's implementation), the LLM lowering tests (`ggml_lower_test`,
`_gemma4`, `_dsv4`, `_qwen38`, `components_qwen2_test`) and the frozen v1
proofs of Gemma 4, Qwen3.8 GGUF and DeepSeek V4 planned 3,018 steps (1,601
distinct) on `7f4b0547` and on this state: identical input digests,
planned-graph digests and implementations, the only difference the new
opt-in patterns' names in policies that list every pattern as disabled.

**Final state, Spark A, 2026-10-15.** The full shared set passes (2,830
tests); the frozen v1 proofs pass (`LLMP_TEST_V1_FROZEN_ONLY=1`: Gemma
native 3, Gemma 2 1, Gemma assistant 2, Qwen3.8 GGUF 1, DeepSeek V4 1,
DSpark 1, Qwen2 FP16 3, Qwen2 EXL3 5); the image model tests pass (11 run,
the six `ImageGenerationGpu` diagnostics skip as they require an explicit
diagnostic). The production image's pixels are `9d0b448f…` (the new pin):
PSNR 41.26 dB, luma SSIM 0.99419 against `9de442f1…` (floors 32 dB /
0.98; the shipping path read 40.687 dB / 0.99418). A two-block screen
against the frozen legacy arm (`1facc957`, primed, ABBA) reads +1.42%
(new 33.674 s, legacy 33.203 s; +1.50% and +1.34% by block), down from
+2.03% at 2a; peak memory +0.81 GiB over legacy (+1.59 GiB before).

Raw records on `spark`: `~/scratch/m36-image-screen09`,
`~/scratch/m36-image-plan/q4`, `~/scratch/m36-image-plan/dg-{old,new}.log`.

## The speed gap (sub-task 5)

An nsys pair of the sub-task 3 state against the frozen legacy arm placed
the +1.42% in: the text encoder's window (246 ms against 133, the same
72.5 ms of kernels: the first generation's plan builds and the cuBLAS and
cuBLASLt libraries its first products load, `cuLibraryLoadData`, which
`CUDA_MODULE_LOADING=EAGER` does not move), each denoising step's
upload of its 68 MB mask and its rebuilt mask pack and rope table, the
VAE's channel norm (126 ms against 68), the first block's input rows
joined twice a step, the output norm's 64 MiB row copy, and the
conditioning products. Each fix is general:

1. **Inputs unchanged since the last run** (planner/plan.h
   `PlanOptions::invariant`, `StepPlan::prologue`;
   `StepProgram::RunWarm`): a caller declares inputs it keeps the same
   between consecutive runs of a plan; the steps that read only them
   (and weights) form a prologue whose results placement keeps whole, and
   a warm run skips the prologue and those inputs' copies (host staging or
   device port). The image's denoiser declares its text rows, positions
   and mask (`ImageStage::invariant`), so a generation uploads the mask,
   packs it, builds the rope table and projects the text rows once. LLMs
   declare none yet.
2. **Plans and recordings at load** (`EngineImage::AfterLoad`;
   `StepRuns::CaptureAhead` takes a run's device copies and its warm
   form; the plans are the load's `PlanStep`'s, so no charge reclaims one
   whose capture is pending, and a plan the load cannot make is left to
   the first request): before the swap is ready the image plans every stage and records
   each run its sequence makes, so the first generation replays them; the
   recording loads the libraries. This moves time, it does not remove it:
   on a first load the model is ready about 0.24 s later and generates as
   much faster (the gate's `generation_s` starts at ready). Plans and
   recordings outlive a swap out, so a prepared swap pays nothing; a
   `swap-table` pair (`qwen3.8:image`, first use, 8K context) read
   LLM→image 3.877 s before and 3.933 s after (setup 0.004 → 0.193 s,
   first output 1.477 → 1.374 s) and image→LLM 6.010 s and 6.002 s.
3. **Repeats and unread nodes** (planner/plan.cc `Repeats`, `Unread`):
   after the patterns a node equal to an earlier one runs once and a node
   nothing needs is dropped. The image's block input rows are joined once
   (one 64 MiB concatenation a step, −23 ms a generation). Neither joins
   nor drops a leaf, a write in place or communication, and no node that
   reads or makes a root written in place (a store's cache, a state) is
   joined (`WrittenRoots`). In LLM plans (a temporary hook logging every
   plan's input digest, policy, planned-graph digest and implementations,
   over the lowering tests, the frozen v1 proofs and Qwen3.8 ModelOpt's
   CUTLASS screens, on main `f09e7994` and on this state: 3,076 plans)
   64 plans change, all Qwen3.8's but 2: ModelOpt CUTLASS 43 of 206,
   native GGML NVFP4 11 of 44, GGUF 8 of 35 (their layers' equal masks
   built once, 20 or 22 a plan, and a few conversions, masks, views or
   row gathers left unread dropped); DeepSeek's 12 and Gemma's are
   unchanged. The tests and proofs pass.
4. **The output norm joined with the last residual**
   (`split_gated_residual_modulate` version 2): the modulation may read a
   range of whole rows of the sum (a view, copied or not): norm and
   modulation are row-wise, so the joined node modulates every row and the
   output projection reads the latent rows of it; the sum nothing else
   reads is recorded in BF16. The 64 MiB row copy and a pass over the sum
   go (−24.5 ms of device copies, −15 ms of kernels).
5. **The VAE's channel norm** (`SpatialChannelNormTiledF32Kernel`, up to
   1,024 channels): 32 pixels a block, their channels dealt over 8 or 16
   warps and held in registers, so the input is read once (1,024²: 6.7 ms
   a call against 10.2). The channel sums' order changes, so the pixels
   change (pin `93356574…`) at the same quality: PSNR 41.26 dB, luma SSIM
   0.99419 against `9de442f1…`. BF16 VAE activations were not needed.
6. The conditioning products were left: the screen was below +0.5%.

What remains against legacy is kernel time (nsys, `item5b` against `old1`,
excluding the products and attention, whose run-to-run spread is larger):
the F32 VAE (convolutions +48 ms, norms +20, adds +23, against legacy's
BF16 activations), the joined residual and modulation (+43), the SwiGLU
(+33) and the conditioning products (`mul_mat_f`, +37; legacy's `Kernel2`
−30).

**Gate, Spark A, 2026-10-16** (`tools/speed_gate.py`, 1,024², 40 steps, 6
primed ABBA blocks against the frozen legacy arm `1facc957`): **PASS**,
`generation_s` +0.22% (upper bound +0.37%, sd 0.14%; legacy 33.792 s, new
33.867 s); peak memory +0.83 GiB (informational). Ready after load is
2.389 s against 2.633 s, so ready plus generation is +0.88%. The two-block
screens read +1.17% after item 1 of this list, +0.60% after item 2,
+0.24% after all five.

After the review's fixes (writes in place kept out of repeats and the
prologue, an invariant input the prologue cannot hold copied every run,
roots placed inside others the prologue's only together, a lowered step
that joins prologue and per-run steps refused, warmness by plan serial and
reset by a newly opened lease, the load's plans its own `PlanStep`'s, no
recording ahead while another model holds the pageable workspace) and on
main `f05323d1` (the workspace paged with the active model), the pixels
are unchanged (`93356574…`). The image now needs 3.52 GiB of activations
and 0.08 GiB of pool (legacy: 2.81 GiB), 32.03 GiB resident while active
(`ActiveFootprint`). **Gate again, Spark A, 2026-10-16** (6 primed ABBA
blocks against `1facc957`; `~/scratch/m36-image-r2gate`): **PASS**,
`generation_s` −0.36% (upper bound −0.08%; legacy 33.418 s, new 33.298 s),
peak memory +0.81 GiB; ready 2.758 s against 2.310 s (the recordings and,
since `f05323d1`, the workspace's page-in), so ready plus generation is
+0.92%. The `qwen3.8:image` swap rows on this build (first use, prepared,
both directions) are all exact, the image→LLM rows 6.25–6.56 s and the
LLM→image rows 3.74–3.94 s. The LLM plan digests (main `f05323d1` against
this state, 3,076 plans) change in the same 64. A one-block `qwen38` screen
against `1facc957` (the v1 ModelOpt pair, `~/scratch/m36-r2-qwen`) is
faster in every cell (C1 decode −1.16%, prefill −3.11%; C2 decode −2.78%;
C4 decode −0.93%, prefill −0.23%), peak memory +9.1 GiB (the ModelOpt v1
gate read +8.3).

Checks on this state (Spark A): the shared set passes; the frozen v1
proofs pass (`LLMP_TEST_V1_FROZEN_ONLY=1`); the image model tests pass (the
six `ImageGenerationGpu` diagnostics skip); the Qwen3.8, DeepSeek, Gemma 4
and Gemma 2/3 engine model tests pass except
`ImportDrafterNormalizeTest.SyntheticGemma4AssistantNormalizesPublishesAndAdmitsWithTargetComposition`
("the drafter keeps no GGUF metadata", in admission code this change does
not touch); the EXL3 and Qwen2 corpus tests skip without their oracle
inputs (the frozen proofs ran their subsets).

Raw records on `spark`: `~/scratch/m36-image-s5gate` (the gate),
`~/scratch/m36-image-s5scr{1,2,3}` (screens), `~/scratch/m36-s5-swap`
(swap rows), `~/scratch/m36-image-plan/nsys/{old1,new1,item4,item5a,item5b}`
and `~/scratch/m36-image-plan/s5/simp*.log` (the LLM plan hook).
