<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Optimization inventory

This inventory covers the current model graph builders, kernel families and
shared execution machinery. It also keeps rejected implementations in view:
a useful preparation, scheduling or reduction step can survive a negative
whole-kernel result. The owner requested this cross-family pass during M3
on 2026-09-29. The linked source declarations are the authoritative lists
of individual implementation identities; this document records techniques,
consumers, eligibility and evidence.

An available implementation is not necessarily selected by a model's plan.
Transfer a technique only after checking its actual call sites, operand
types and strides, shapes, arithmetic, scratch accounting and lifetime.
Measure the eligible shape and the whole model. A lower register count or
fewer launches alone is not a speed measurement. D-085 permits different
reduction order at the same quality; our own rollback, repeat and restored
continuation remain exact. Precision changes need an explicit per-alias
quality/performance mode, off by default.

## Coverage

| Implementation family | Current consumer and source | Techniques to compare |
| --- | --- | --- |
| GGML float primitives and structural fusions | [Registry declarations](../src/kernels/ggml/implementations.cc), [fusion gates](../src/kernels/ggml/fusion.cc), [planner](../src/kernels/ggml/graph_plan.cc); Qwen2, DeepSeek, Qwen3.8 and the EXL3 plan's non-linear operations | Vector versus tiled/cuBLAS products; RMSNorm/scale, bias/product, gate/up/SwiGLU and RoPE/cache-store fusions; activation lifetime placement |
| Ordinary GGUF products | [Quantized products](../src/kernels/ggml/mul_mat_q.cu), [row products](../src/kernels/ggml/mmvq_rows.cu), [DeepSeek fast products](../src/kernels/ggml/dsv4_fast.cu); DeepSeek target and DSpark | Raw expert strides, Q8 input preparation reuse, adjacent gate/up preparation, vector rows that share expert reads, compact scheduling candidate |
| NVFP4/MXFP8 products | [Qwen3.8 graph](../src/kernels/ggml/qwen38_graph.cc), [NVFP4 GEMV](../src/kernels/ggml/jitllm_moe.cu), [grouped GEMM](../src/kernels/ggml/moe_cutlass.cu), [MXFP8](../src/kernels/ggml/mxfp8_cutlass.cu) | Cached input conversions, quantized tensor-core prefill, fused activation/quantization, one-row kernels and shape-specific product choices |
| Attention and selection | [GGML attention](../src/kernels/ggml/fattn_mma.cu), [DeepSeek sparse path](../src/kernels/ggml/dsv4_sparse.cu), [QSA](../src/kernels/ggml/qsa_sparse.cu), [image attention](../src/kernels/image/flash_attention.cu) | Selected-cell gathers, sparse query unions, cached block keys, deterministic index ties, live cell counts, fused normalization/rotation |
| Hyper-connections, routing and recurrence | [DeepSeek](../src/kernels/ggml/dsv4_fast.cu), [Qwen fusions](../src/kernels/ggml/jitllm_fused.cu), [Qwen graph](../src/kernels/ggml/qwen38_graph.cc) | Clustered reductions, PDL, fused small operations, in-place state and commit of accepted verify rows |
| EXL3 packed and reconstruction products | [Registry](../src/kernels/exl3/implementations.cc), [linears](../src/kernels/exl3/linear.cc), [Qwen2 plan](../src/kernels/exl3/qwen2.cc) | Packed GEMV/GEMM, fused gate/up, shared transforms, bounded reconstruction slices, fused reconstruction and pinned cuBLASLt algorithms |
| Image BF16 pipeline | [Registry](../src/kernels/image/implementations.cc), [pipeline](../src/kernels/image/pipeline.cc), [products](../src/kernels/image/gemm.cc), [convolution](../src/kernels/image/conv.cu) | Per-shape GEMM pins, residual/next-norm fusion, implicit GEMM, prefix KV reuse and norm/rotation inside attention |
| Paging and shared engine | [Paging kernels](../src/kernels/paging/), [runner skeleton](engine.md), [serving](runtime-serving.md) | Stable-address graphs, one request lease, sleeping lanes, bounded staging, initialized state extents, sparse spill and turn checkpoints |
| Sampling and draft control | [Sampling](../src/execution/sampling.cc), [draft-head study](experiments/qwen38-draft-head/README.md) | Validated top-k-one greedy shortcut; optional confidence computation; selected head with original IDs; observed acceptance, depth and avoiding unused draft passes |

Qwen2's GGML graph is the FP16 backend fixture, and the native EXL3 Qwen2
plan is its separate packed-format companion. Neither is an M3 serving
model. The image includes its text encoder, denoiser and VAE; convolution
and iterative denoising optimizations are part of the inventory even when
they have no text-model equivalent.

## Transfers and gaps

| Technique | Already used | Missing or conditional consumer | Action and evidence |
| --- | --- | --- | --- |
| Keep only selected attention cells | DeepSeek's window plus indexer selection; Qwen3.8's kept QSA cells | A dense-attention family has no equivalent sparse selection | Shared principle is adopted. Model selection semantics remain authoritative; do not invent sparsity for another architecture. [Long context](experiments/long-context/README.md) |
| Reuse a sparse query tile's union of KV rows | DeepSeek fast prefill through the shared `mma_wide` implementation | Generic D256/D512 attention; Qwen's separate QSA implementation | Capability remains off for unknown/reference plans. D256 synthetic overlap/disjoint gains 4.58×/2.20×; D512 gains 1.50× with overlap but falls to 0.556× disjoint. Shape and overlap matter. [ds4 study](experiments/ds4-study/README.md) |
| Prepare an input once for adjacent products | Qwen prefill reuses BF16/MXFP8 input forms; ordinary GGUF gate/up can share Q8 and route preparation | Other products that consume the same logical input; raw NVFP4 has a different contract | The ordinary pair is generic but default-off outside measured DeepSeek fast plans; its 8K model gain was 0.8%, not a decisive product speedup. Check conversions and map identity before sharing. [ds4 study](experiments/ds4-study/README.md) |
| Reuse decode input quantization | DeepSeek `Q8Of` shares Q8_1 conversions across compatible attention/shared/routed/head inputs | Ordinary primitive products still prepare their own input | Already adopted for DeepSeek decode; this is separate from the paired-MMQ prefill work. EXL3 trellis/Hadamard, MXFP8 and NVFP4 per-16-value Q8 have different representations. |
| Read one expert's weights for several verify rows | DeepSeek fast vector products | Qwen NVFP4 expert GEMV | Qwen tried the grouping already; current paired replication is neutral. Preserve the format-specific choices. [Qwen MTP](experiments/qwen38-mtp/README.md#judgement-calls) |
| Cluster a small reduction and prefetch the next weights | Qwen3.8's hyper-connection preparation, up to eight tokens | DeepSeek's hyper-connection pre-projection | Source-compatible principle, not the same formula or weight layout. A remaining DeepSeek lead is about 1% of a step; measure only after the larger product gap. sm_86 has no clusters. [TensorFold techniques](experiments/tensorfold-techniques/README.md#limits-and-what-remains) |
| Fuse a small operation chain | Qwen short convolution/history and norm/gate; DeepSeek routing/hyper-connections; image residual/next norm; shared RMSNorm/scale | Qwen recurrent alpha/beta chain; other adjacent operations with matching rounding | Qwen's remaining alpha/beta chain is identified in the TensorFold report. Preserve recurrent-state commit/rollback and BF16 rounding points; validate a proposed fusion independently. |
| Compute only the head rows the caller needs | Qwen3.8 fast prefill chooses its final output frontier | DeepSeek's prefill graph retains all-row head output; audit fixture/EXL3 output selection separately | Worth a bounded DeepSeek experiment: the matched first-4K profile's head is about 65 ms, roughly 1% of GPU work, but its all-row logits also occupy substantial workspace. PPL and forced scoring still need their requested rows. |
| Pin library product algorithms by shape | Image BF16 and EXL3 reconstructed GEMM | Qwen BF16 multi-row products and other library products | Share tuning/provenance discipline rather than a pin from another shape. Image's table has BF16 outputs; Qwen hyper-connection down products need F32 output. Operand/output types, dimensions, strides, workspace, device and library version must match; preserve captured addresses. |
| Return device argmax verdicts for greedy verify | Qwen target verify copies IDs; full logits are optional | DeepSeek target verify still copies every full logit row | A bounded lean-greedy candidate; preserve lower-index ties and non-finite policy, plus sampled and diagnostic rows. No isolated DeepSeek speed gain measured. |
| Tensor-core attention for eligible dense shapes | Generic D128/D256/D512 kernels and image BF16 attention | Current Qwen2/EXL3 fixture: masked D64, GQA 14:2 | Existing D128 MMA and image kernels do not satisfy this mask/GQA contract. A new eligible D64 path belongs to the M3.5 comparison, not an unchecked dispatch change. |
| Fuse residual-add with the next norm | Image's fast denoiser | Future eligible dense-text graph | Shared RMSNorm/scale fusion does not already fuse the residual add. Image BF16 intermediates differ from GGML F32 and EXL3 casts; preserve each model's rounding. Unmeasured outside image. |
| State updates in place | Qwen one-row Gated DeltaNet; DeepSeek window ring; image prefix cache | A new recurrent family or verify implementation | Share lifecycle/accounting helpers. Recurrence formulas and rejected-row restoration are family-specific; no in-place update without its rollback proof. |
| Stable-address graphs and request leases | All three M3 runners, including image steps | EXL3/other families when they enter serving | Engine skeleton is shared. M3.5 integrations should reuse it instead of adding a parallel execution lifecycle. |
| Growing state and turn-boundary reuse | DeepSeek and Qwen serving | New recurrent families and their state layouts | The shared helpers already exist; each adapter supplies its footprint/checkpoint semantics. Prefix matching still does not identify a conversation's lifetime. |
| Selected head and confidence-based draft control | Qwen optional externally supplied selected BF16 head; adaptive depth 2–3 | Other learned drafters with their own vocabulary/state contracts | Curated head and depth choices are measured independently; curated is not uniformly faster. Fixed depths 3–5 and confidence windows, including actual incremental draft stopping, did not close the 128K speed gap. Sampled policy stays fixed; the final selected head passes the fresh distribution check. [Calibration and rejected prototypes](experiments/qwen38-mtp-speed/README.md) |

## Rejected kernels: pieces worth retaining

A rejection applies to its tested shape, precision and surrounding work.
An isolated component remains a candidate until measured with a compatible
consumer. Raw prototypes and traces stay in external scratch; durable
aggregate results and upstream source identities belong in the linked reports.

| Whole implementation or choice | Why it was not adopted | Piece that may transfer | Required next proof |
| --- | --- | --- | --- |
| Qwen grouped NVFP4 verify products | Earlier four-row test: 21.5–21.9 ms versus 18.7 ms; current 128K paired model test also neutral | Matching expert/row scheduling and leader selection; L2 already serves repeated weights | Isolate scheduling overhead or a new shape with demonstrably poor cache reuse. Do not repeat the same grouping as a new idea. [Qwen MTP](experiments/qwen38-mtp/README.md) |
| Qwen precomputed Q8 activations for multi-row verify experts | Current paired 128K model test neutral after 96 added preparation launches per verify; one-token products stayed inline | Exact per-16-value Q8 conversion outside the 2–8-token products; product register counts fall from 98/106 to 78/74 | A register reduction is not measured speed. Try the preparation only where a producer can fuse it or enough products reuse it; include transient scratch and the original quantization order. [Calibration](experiments/qwen38-mtp-speed/README.md) |
| Qwen BF16 vector kernel for multi-row verify | Slower than cuBLAS at that shape and two tokens exceeded the existing speculation near-tie bound | Its one-row product/reduction remains adopted for plain decode | New multi-row shape must pass the original quality bound and win end to end; do not widen the bound to accept a kernel. [TensorFold techniques](experiments/tensorfold-techniques/README.md) |
| Confidence window that only shortens verify | No model gain; every draft pass has already run | Candidate confidence and deterministic stopping criterion | Avoid subsequent draft work itself. Compare against the unchanged whole-prefix draft and check state, repeats, swaps and rejection. |
| Incremental confidence stopping after at least two drafts | Actual early stopping remained neutral at 128K: prefix 45.00 and curated 45.56 tok/s versus fixed-depth-four controls 45.61 and 45.76 | Bounded continuation graphs and explicit draft-state catch-up | Short forced-rejection control passed; direct parity with the monolithic draft was not established. Any new consumer needs that proof plus a measured gain. Prototype removed. [Calibration](experiments/qwen38-mtp-speed/README.md) |
| ds4 direct-to-residual products | Different SoA weight layout and earlier route-weight multiplication around input quantization; not a drop-in raw GGUF product | Compact expert-major scheduling, paired gate/up preparation, fused activation, integer-dot/correction techniques | Keep raw artifact strides and current route weighting. Measure each piece before considering a layout change or extra weight replica. [ds4 study](experiments/ds4-study/README.md) |
| ds4 FP8 KV/FP4 indexer caches | A precision change; native M3 keeps its F16 state contract | Scheduling, gathered-cell bounds and attention tiling independent of compression | Same-quality cache precision remains the control. A compression mode needs explicit per-alias opt-in and its own quality/memory evidence. |
| Wider sparse attention everywhere | Disjoint D512 synthetic case regressed, despite other shapes winning | Query-union construction and overlap-aware tile policy | Limit selection to measured eligible consumers; no universal wider-tile default. |
| DeepSeek grouped multi-matrix launches | Existing decode study found no gain | Shared input staging or an eligible adjacent-product preparation | Isolate useful shared work; retain the measured per-product path until an end-to-end comparison supports another choice. [DeepSeek decode](experiments/dsv4-decode/README.md) |
| Image batched QKV/gate-up products | Pinned single products were as fast or faster: 4.40 ms versus 3 × 1.47 for QKV, 8.52 versus 2 × 4.12 for gate/up | Shape-specific product tuning and shared inputs | A different batch or layout needs its own measured algorithm and model result. [Image study](experiments/qwen-image-native/README.md) |
| Image third KV attention stage | 3.43 versus 3.49 ms isolated, but 32.56 versus 32.58 seconds for generation; neutral | Pipeline-stage tuning with matched shared-memory occupancy | The 96 KiB stage is a measured isolated gain, not an end-to-end adoption. A new consumer must account for its shared-memory and occupancy cost. [Image study](experiments/qwen-image-native/README.md) |
| Image prefix cache read in place, alone | Full generation did not move measurably | Stable prefix views became the base of the adopted query-norm fusion | A concrete useful piece of a neutral change; keep combined and isolated outcomes distinct. [Image study](experiments/qwen-image-native/README.md) |

## Measurement order

First finish the active compact ordinary-expert and shape-specific Qwen
library-product experiments, including their correctness controls. Then measure any
inventory candidate that can materially help the open M3 speed or maximum
memory gate. Small unmeasured leads remain named here instead of becoming
automatic model defaults or an unbounded benchmark queue.

For each candidate, record the current plan and the exact missing consumer,
the fragment being transferred, its precision/layout requirements, isolated
timing, whole-model timing and peak memory, correctness/rejection controls,
and the resulting selection policy. Update this inventory and its source
report when the outcome changes.

M3.5 applies the same comparison to its new EXL3 and legacy families. M4's
[Kindling and TensorFold references](m4-references.md) add multi-node
techniques only after their recipes and measured two-Spark behavior are
re-pinned at entry. Their claimed rankings are not local measurements.
