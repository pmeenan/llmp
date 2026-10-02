<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Cross-implementation optimization inventory

Inventory recorded 2026-10-01 alongside the native Qwen request-state work. Availability, selection and measured benefit are different:
a kernel in the registry is not necessarily selected by a model or profitable
at another shape. This inventory includes rejected kernels' reusable pieces.
It proposes bounded comparisons; it does not expand a quality exception.

## Current coverage

| Implementation | Native shape and format contract | Techniques already present | Transfer opportunity |
| --- | --- | --- | --- |
| GGML Qwen2 fixture | Dense attention with F16 KV; ordinary GGML linears and graph operations | Shared planner's supported norm/multiply, RoPE/store and vector linear/activation fusions, when their graph and device guards match | Use the fixture to validate generic fusion/planner changes. Sparse-cell union is irrelevant to its dense attention; an attention replacement needs its own mask/GQA/precision controls. |
| GGML DeepSeek V4 target and DSpark | GGUF quantized weights; sparse window/indexer/compressed attention; hyper-connections and six routed experts | Fast small-row routed vectors, HC preparation/routing, PDL and graph replay; wide compression, native HC primitives, compact expert scheduling, shared gate/up Q8 preparation, ordered wide-row expert reduction and exact Q-head RMSNorm/RoPE fusion | Continue measured wide-prefill chains before extending the small-row fast path. IQ2 loader/MMA scheduling remains a separate operator gap. |
| GGML Qwen3.8 target and MTP | MXFP8 dense products, NVFP4 experts, BF16 head and recurrent Gated DeltaNet/QSA | HC preparation/mixing, quantization epilogues, fused gate/up/SwiGLU, in-place recurrent updates, clustered small BF16 products, PDL, device QSA selection, cached block keys, sparse attention, graph replay and paired-eight cross-request products with guarded BF16 target-head sharing | Carry the shared driver and independent state/recovery seams to other families using their own product guards. Attribute acceptance and milliseconds per step separately when choosing depth or vocabulary. |
| EXL3 fixture companion | Packed EXL3 weights and transform/reconstruction contracts; F16 output/bias behavior | Packed GEMV/GEMM and grouped GEMM paths; fused reconstruction option; pinned cuBLASLt reconstruction algorithms and upstream bias addition | Request-row sharing and paid shape-dependent dispatch are reusable ideas. An MXFP8/NVFP4 kernel cannot replace its different representation or transforms. Run its actual fixture paths at new row counts before changing their selection. |
| Qwen-Image-2.1 | BF16 DiT/text/VAE operations with reference materialization rounding points | Flash attention, normalization/modulation/residual and normalization/RoPE fusions, implicit-GEMM VAE convolution, pinned BF16 products and cached text-prefix inputs | Preserve each documented BF16 rounding point while sharing preparation or fusing epilogues. Large dense image rows do not establish a use for small-row decode kernels. Profile launches before trying PDL. |
| Shared engine and paging | Cataloged weights/state, completion-aware closures, bounded staging and graph plans | Request leases, device-resident extents, growing used-state backing, checkpoints, graph replay, row-wise n-gram reads and per-model plans | Batch requests without duplicating weights or engine owners; retain independent state/cursors and count all copies/plans/reads. Prefix reuse and expiry must be bounded and measured separately from fresh-prefill throughput. |
| Shared sampling | Validated parameters and finite-logit checks; seeded exact sampling/verification | Validated top_k=1 greedy shortcut, candidate filtering and deterministic token tie handling | Already shared across the LLMs. No family-specific shortcut is owed. New GPU sampling should be justified by measured request cost. |

Source anchors: `src/kernels/ggml/{qwen2_graph,dsv4_graph,qwen38_graph,graph_plan,implementations}.cc`,
`src/kernels/ggml/{jitllm_ops,ops_ext}.h`,
`src/kernels/exl3/{implementations,linear}.cc`,
`src/kernels/image/{ops,implementations}.h`,
`src/kernels/image/{flash_attention,conv}.cu`,
`src/engine/{graph_runs,live_state}.cc` and `src/execution/sampling.cc`.

## Techniques and the boundaries on transferring them

| Idea | Already selected or demonstrated | What prevents automatic transfer | First useful comparison |
| --- | --- | --- | --- |
| Quantize one routed input once for gate/up | DeepSeek compact MMQ preparation; Qwen gate/up preparation under its own NVFP4 scale contract | Q8 GGUF and per-16-value NVFP4 codes/scales and nonlinear rounding are different | Keep each family’s exact producer bytes, then compare sharing versus duplicate preparation on captured inputs. |
| Reuse selected weights across rows | DeepSeek small-row fast vectors; Qwen verify vector paths and private C2/C4 paired products | Different requests require separate routing IDs and recurrence/attention state; wider rows can spill or choose another kernel | Paid natural decode, same IDs/acceptance/state, including packing and ragged active masks. |
| Expert-major scheduling and compact row lists | DeepSeek wide MMQ; Qwen has its own sorted/grouped prefill route | Empty/skewed experts, padding, codebooks and row/slot maps affect dispatch and correctness | Captured real routes with every destination row checked; separate scheduling from arithmetic. |
| Hyper-connection preparation and epilogues | Both LLMs have native HC operations; Qwen combines preparation and conversions for its fast path | Their HC definitions and weight formats differ. A small-row guard is not proof that all wide HC work is unfused | Matched eight-chain DeepSeek attribution, then one missing wide operation at a time. |
| Sparse attention sharing across neighboring queries | DeepSeek wide sparse attention; Qwen shares query heads within its QSA attention | Selected-cell overlap, head dimension, per-row masks and dummy cells change cost. D512 disjoint sharing already regresses | Actual selected-cell lists, overlapping and disjoint controls, long-context likelihood and answers. |
| In-place recurrent state | Qwen one-row Gated DeltaNet | Verification needs per-kept-row recovery; dense attention and image layers do not have this recurrence | Preserve forced rejection, cancellation, checkpoints and swap-return controls before altering state storage. |
| Norm/residual/activation/store fusion | Generic GGML planner; Qwen HC/GDN/QSA; image BF16 operation fusions | Algebraically equivalent operations can round at different materialization points or reorder contributions | Fixed-history operand comparison with explicit rounding/order, then whole-model quality and paid timing. |
| Q-head normalization and rotation fusion | DeepSeek's selected wide Q-head path gains 3.17% whole-model prefill with byte-exact full heads | FMA operand order must preserve the materialized path. Qwen QSA already prepares normalized/rotated heads together; the Qwen2/EXL3 fixture lacks this RMSNorm pair | Match the producer/consumer, reduction and rounding contracts before adding a new family guard. |
| PDL and clustered reductions | Native small-row Qwen and DeepSeek paths | Requires correct launch dependencies, hardware shape/resource limits and unchanged stream lifetime. A cluster can worsen occupancy | Actual launch/resource evidence and one isolated shape; pay the surrounding request work. |
| Cached keys or prefixes | Qwen block-key pool; image text prefix; retained LLM turn checkpoints | A block key cache and a conversation prefix cache have different contents and invalidation. Multiple requests need independent state | Cold/fresh and strict-prefix warm request groups, charged tail and copy work, bounded expiry/bytes. |
| Shape-dependent library algorithms | EXL3 reconstruction and image BF16 products; ordinary GGML head selection | Different row counts naturally change selector/arithmetic. A whole-model format comparison cannot isolate that choice | Same weight/input operand and compile/runtime selector evidence, followed by same-work request timing. |

## Rejected or optional kernels with useful pieces

| Whole proposal and outcome | Pieces to retain or investigate | Limit of the evidence |
| --- | --- | --- |
| Original ds4 MoE pipeline is not the native production architecture | Expert-major compact scheduling, shared row maps/Q8 input, raw-layout loaders, frontier-only heads and potentially a fused gate/up epilogue | Original SoA representation, cache precision and early expert weighting are separate factors. Existing component GPU times are not additive end-to-end gains. |
| Native IQ2 shared worklist: no adoption | Its diagnostic proves the second builder can be removed while retaining the two J128 products and every output bit | Rate improvement is only 0.202%, smaller than ordinary bookend movement; it does not close the consumer gap. |
| Native IQ2 J64: retain J128 | Resource-limited J64 is a different potential configuration; loader lookahead/staging remain inspectable | Unchanged J64 was 4.92% slower. Registers, local bytes and dynamic shared memory must be measured for any new configuration; two-block occupancy is not established. |
| DeepSeek wide HC mix/pre and suffix: no adoption | Retain efficient ordinary producers when investigating nonlinear/output-normalization fusion | Complete 8K mix/pre is 2.18% slower with paid weight casts. Preserving ordinary RMS/F16 projection and fusing only the suffix gains 0.48% amid 0.30% bookend movement. Both change heads; neither warrants a wider ladder. Qwen already has its own wide HC path. |
| Qwen BF16 full-target-head sharing: positive controls | Ordinary selector/MMF at six, eight or sixteen columns, paid concatenations and bounded split views over independent states | Paid fixed3 C4 decode gains 8.867%. Fixed3 cooperative C2 HTTP gains 5.40%/4.99%. Extending the shared three-row guard under unchanged adaptive depth gains 2.53%/1.33%, pooled 1.93%, with exact full native vectors and HTTP responses. These head-only factors do not establish cross-engine parity. |
| Qwen fixed-three depth: modest direction, no default change | A fixed depth remains a cheap scheduling candidate; the current adaptive policy stays selected | One fresh C2 screen gains 2.35% amid 1.18% ordinary bookend movement. Both reasoning trajectories change, so later acceptance/work can differ; the capped replies contain no final answer. |
| Qwen shared HC BF16 products: small positive private screen | Combine compatible multirow products over the same weights while preserving separate preparation, nonlinear mixing and request state | Fresh adaptive C2 HTTP gains 1.60% amid 0.28% control movement with identical capped replies. Full native-vector/state controls and attribution of HC pair counts remain open. |
| Qwen ragged BF16 full-head sharing: small positive private screen | The ordinary seven-column MMF can share compatible three-plus-four verification heads, with paid concatenation and independent output views | Fresh adaptive C2 HTTP gains 1.50% amid 0.27% control movement with identical capped replies. Combine with HC sharing for focused full-vector/state controls; separate gains are not additive. |
| DeepSeek inverse-RoPE/output-A/layout fusion: positive private timing and focused quality | Fuse a producer transform with a grouped projection and eliminate intermediate layout work; adapt the useful mechanism to native operations | Paid operator latency falls 45.63%; community 8K throughput improves 4.08% amid 0.42% bookend movement. The original-checkpoint 32K candidate passes the unchanged greedy bound and repeats all full logits exactly; its native baseline fails one step. Broader quality and production integration remain open. Qwen and Qwen2 lack this grouped output-A chain. |
| Exact16 Qwen MXFP8: no adoption | Share work using paired-eight products; investigate BF16 full heads independently | Exact16 is 7.469 times slower despite exact outputs/state. Compiled local-storage growth supports a spill concern but does not measure traffic or prove the sole cause. |
| Literal Mia scheduling/attention consumers: earlier negative controls | Inspect their packing, cache and launch boundaries separately; retain native device selection and sparse work | A faster kernel in its original runtime can lose after adapters or dispatch changes. Do not repeat an unchanged negative factor. |
| Cross-token Qwen sparse-cell union: rejected timing | The native per-KV-head query tile and selected-cell cache remain useful; a cheaper bounded union would be a new experiment | The measured union proposal materially regressed. Synthetic D256 sharing does not approve it for Qwen. |
| DeepSeek original HCA attention: optional, off by default | Query-token tiling and F32 weighted-value accumulation are inspectable independently of cache changes | HCA plus output-A passes the original-checkpoint 32K/128K greedy bounds, with an exact full-logit repeat at 32K. The actual mixed 128K path passes registered held-out PPL at 1.92884 versus 1.9298 (−0.05%) and preserves the frozen 127K-token variable-binding answer. Native integration and broader acceptance remain open. |
| TensorFold adaptive draft window and partial vocabulary | Acceptance-dependent scheduling, model-specific curated IDs and future prompt/turn-frequency vocabulary proposals | Model/format/cache/depth factors differ. Acceptance benefit is workload-specific; quantizing or truncating a draft head requires its own measured tradeoff. |
| TensorFold import-layout and tensor-core decode suggestions | Packing and epilogue techniques at matching native shapes | Existing native vector kernels already have their own measured rate. Checkpoint bytes alone do not establish device traffic or an attributable format-only speedup. |
| Image reduced-precision linears/cache and quantized heads | Optional future per-alias quality/performance modes | Preserve default model quality; no unmeasured precision reduction belongs in the ordinary speed pass. |

Evidence lives in the existing reports:
`docs/experiments/ds4-study/README.md`,
`ds4-native-iq2-consumer/README.md`,
`ds4-native-iq2-j64/README.md`,
`ds4-native-iq2-worklist/README.md`,
`qwen38-request-batching/README.md`,
`qwen38-mxfp8-sixteen/README.md`,
`qwen38-target-head-sharing/README.md`,
`qwen38-depth-policy/README.md`,
`qwen38-hc-sharing/README.md`,
`qwen38-ragged-head-sharing/README.md`,
`dsv4-qhead/README.md`,
`ds4-output-prefix/README.md`,
`qwen38-mxfp8-scheduling/README.md`,
`tensorfold-techniques/README.md` and `ds4-long-context-tasks/README.md`.
Entries marked investigation have no claimed measured gain.

## Priority

Native Qwen shared execution integrates paired-eight products and guarded
BF16 equal three- or four-row heads using independent slots and unchanged
adaptive policy. Extending head sharing to depth-two verification gives a
modest 1.93% pooled gain in two adaptive HTTP screens. The preceding adaptive
shared-versus-serial screen was neutral; cross-engine concurrent qualification
remains open.
The short [fixed-three depth screen](../qwen38-depth-policy/README.md) is a
modest candidate; no context ladder or default change follows from it.
The short [HC-sharing screen](../qwen38-hc-sharing/README.md) gains 1.60%
without changing adaptive depth, independent state or capped replies. It
remains private pending focused numerical/state controls.
The literal/native benchmark pipeline
study is already complete; the remaining DeepSeek attribution concerns the
current production graph, rather than another literal-pipeline parity run.
The [matched production attribution](../ds4-production-prefill-attribution/README.md)
now places the ordered weighted-expert reduction at about 5% of 8K prefill
wall. Its captured-operand screen is byte-exact and 4.19× faster; the complete
8K substitution gains 4.30% in tokens/s with every full head byte unchanged.
The selected native operation handles canonical F32 width 4096 with six
experts, preserving weighting placement and ascending multiply/add order.
Qwen's ten-expert sorted NVFP4 combine already has its own implementation;
other MoE families need matching layout, precision and sum contracts before
transferring this mechanism. Dense and image consumers have no such sum.
The selected [Q-head fusion](../dsv4-qhead/README.md) adds a separate 3.17%
whole-model prefill gain with byte-exact full heads. The private
[inverse-RoPE/output-A/layout screen](../ds4-output-prefix/README.md) gains
4.08% on the community 8K workload but changes logits. At 32K on the original
checkpoint it has zero outside-bound differences in 512 fixed-history rows
and repeats complete logits exactly; the same native baseline has one
exception. The optional HCA interaction also passes that greedy/repeat
control; its likelihood on the same generated continuation is almost
native's. The actual mixed 128K path also passes registered held-out
perplexity at −0.05% relative to the oracle and preserves the saved native
127K-token variable-binding answer exactly. The 128K fixed-history
comparison has 500 exact choices, 12 within-bound differences and no
outside/unresolved rows; former step 315 now agrees. Native integration
and broader acceptance remain open.
Operator gains and separate model factors are not additive.
The first native
Slot-wave C2 screen improves paid decode by 9.91% with exact IDs and traces,
selecting focused state/recovery controls and serving integration.
Measure real concurrent requests on both models. Broader PDL,
layout, attention and vocabulary proposals follow a measured missing budget.

Private A/B iterations need target builds and their causal controls. The
normal full Spark check set belongs to selected production changes; x86
checks remain deferred to the end of the optimization run. Run the final swap
table once for the settled implementation, rather than inside these controls.
