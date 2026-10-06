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

The [current M3 closure status](m3-optimization-status.md) separates completed
review experiments, rejected candidates, selected native changes and open
engine gaps. Individual positive factors do not establish pipeline parity.

The [2026-10-01 implementation snapshot](experiments/optimization-inventory/README.md)
adds the independent native request slots, positive private BF16 head-sharing
factor, rejected shared-worklist/exact16 controls and cross-family boundaries
on batching and weight reuse. Its priority order follows the current M3 gaps.

An available implementation is not necessarily selected by a model's plan.
Transfer a technique only after checking its actual call sites, operand
types and strides, shapes, arithmetic, scratch accounting and lifetime.
Measure the eligible shape and the whole model. A lower register count or
fewer launches alone is not a speed measurement. D-085 permits different
reduction order at the same quality; our own rollback, repeat and restored
continuation remain exact. Changes that trade quality for speed need an
explicit per-alias quality/performance mode, off by default.

Before closing a model's qualification, review its learnings against the
previously implemented models, particularly Gemma 26B (owner, 2026-10-05).
Apply eligible improvements and verify matched correctness, performance,
batching and memory on each recipient. Record ineligible shapes or formats
and measured negative results; availability alone does not establish a
transfer.

The complete literal ds4 path was a temporary benchmark reference for
matching the configured end-to-end performance and restoring native stages
individually to isolate the causes; the mechanisms adopted from it are
recorded in this inventory and the M3 record. Its unused reference code left the
tree on 2026-10-04 (commit `ec8af04` is the last that holds it); only the
original output-A and F16-Q token-tile HCA cores that production selects
remain. [Matched-pipeline study](experiments/ds4-complete-plan/README.md).

The matched 8K reference's [paid-chain profile](experiments/ds4-restoration-profile/README.md)
puts routed FFN at 35.55% and attention output at 17.94% of measured GPU
time, with all eight full heads byte-identical to original ds4. The first
[native output-B restoration](experiments/ds4-native-outputb/README.md)
retains the paid D4 producer and sanitizer: rate ratio 0.9983, all twelve
full heads and three complete original-input operator outputs byte-identical.
The measured output chain also includes output-A and HC expansion.
Separate FFN weight layout, consumer and fusion controls before interactions.
The native consumer remains available in the private benchmark; production
dispatch is unchanged. The first [routed-FFN factor](experiments/ds4-routed-ffn-first-axis/README.md)
compares paid producer/gather + fusion/storage: Materialized is 2.35% slower
than Direct at 8K, with twelve full heads and complete captured D2S6/down
outputs byte-identical. Keep Direct; no new gather adapter is justified by
that bounded gap. The matched 32K pipeline reproduces every original final
logit byte at 0.993× its throughput. The [captured native IQ2 restoration](experiments/ds4-native-iq2/README.md)
combines current compact products with the original suffix and runs at
0.866× Direct. The [preparation capture](experiments/ds4-native-iq2-prepared/README.md)
now proves every native/original D4 payload byte equal; borrowing either
payload reproduces native gate/up exactly. Producer removal saves about
4–6% of this resident pair's time. The [same-D4 standalone consumer control](experiments/ds4-native-iq2-consumer/README.md)
measures native at 0.880357× original resident rate. Its [J128/J64 follow-up](experiments/ds4-native-iq2-j64/README.md)
keeps all outputs exact but reduces rate by 4.92%; retain J128. Weight
loading, paired-launch structure and remaining arithmetic/compiler factors
still need separate causal controls; historic compound times cannot isolate them.
The [query-B consumer factor](experiments/ds4-query-b/README.md) keeps paid
original D4 and every other literal stage, including output-B: native
borrowed-D4 Q8 MMQ is 3.110% slower at T4096/K1024/M32768, with 0.430%
bookend movement. Original full heads remain golden; native repeats exactly
with the same final argmax but different logits. This closes one consumer
screen, not Q/KV producers or a quality gate; no cross-family gain is claimed.

The [M3.5 legacy primitive screen](experiments/m35-legacy-quants/README.md)
extends Q4_0/Q4_1/Q5_0/Q5_1/IQ4_NL row-invariant and joined VecQ mechanics,
with shared input preparation and original per-column arithmetic. Q5_1 at
Gemma's K704/N2816/128-expert/top8 shape shows a P1 gain only for synthetic
disjoint four-row routes; two/eight shared experts favor ordinary MMVQ, and
P4 loses in all measured patterns. No universal Gemma dispatch or format/
model supported status follows. Ordinary partial output-row blocks are
explicitly refused where pinned MMVQ would over-read (RE-045); the own
row-invariant kernel guards them.

The proposed ordinary Gemma row/attention recipe is still gated by actual cohort
quality. Its [first31 C2 comparison](experiments/gemma31-production-c2/README.md)
retains eight positive-margin strict differences, zero exact full-head rows and
a passing+0.791% conditional-loss result over64 within-history targets. Actual
C2 selects independent attention with zero owner steps. The source-backed next
factor uses the existing D256 MMA fallback instead of the local vector family
in a private build; no causal result, production policy or borrowed allowance
is implied. Existing whole-C8/C12 geometry evidence and grouped-store rejection
retain their separate scopes.

## Coverage

The [Gemma26 late-prefill common-input controls](experiments/gemma26-late-moe/README.md)
match native and original primitive/fused bytes for 128-expert/top-eight routing
and width-2,816 scaled reduction at 64 rows. Captured stock uses primitive
routing and fused reduction; switching recipes changes weights/sums by at most
1.19e-7/5.96e-8 without changing selected IDs. This is operator fidelity on actual
inputs, not the cause of the remaining model misses or a production selection.
Final-layer native prefill uses one frontier row versus the captured 64 rows;
Qwen/DeepSeek shapes, scaling and contraction contracts require separate proof.

| Implementation family | Current consumer and source | Techniques to compare |
| --- | --- | --- |
| GGML float primitives and structural fusions | [Registry declarations](../src/kernels/ggml/implementations.cc), [fusion gates](../src/kernels/ggml/fusion.cc), [planner](../src/kernels/ggml/graph_plan.cc); Qwen2, DeepSeek, Qwen3.8 and the EXL3 plan's non-linear operations | Vector versus tiled/cuBLAS products; RMSNorm/scale, bias/product, gate/up/SwiGLU and RoPE/cache-store fusions; activation lifetime placement |
| Ordinary GGUF products | [Quantized products](../src/kernels/ggml/mul_mat_q.cu), [row products](../src/kernels/ggml/mmvq_rows.cu), [DeepSeek fast products](../src/kernels/ggml/dsv4_fast.cu); DeepSeek target and DSpark | Raw expert strides, Q8 input preparation reuse, adjacent gate/up preparation, vector rows that share expert reads, compact expert-major prefill scheduling |
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
| Select the pinned attention arm by head grouping and query shape | Gemma local D256/F16 primitive: GB10 solo vector, group2 MMA for multiple rows/sequences | Gemma graph and whole-model solo/batched execution | Tile4/8/16/32 and paid mask/fixup scratch follow the pinned selector. Independent 16/32-head ring controls and original/capture comparison are [bounded primitive evidence](experiments/gemma-local-attention/README.md); model quality and optimized batching remain owed. Checked per-segment device masks have [separate bounded controls](experiments/gemma-device-masks/README.md). Existing D64 vector and D256/D512 group8 routes retain their identities. |
| Keep only selected attention cells | DeepSeek's window plus indexer selection; Qwen3.8's kept QSA cells | A dense-attention family has no equivalent sparse selection | Shared principle is adopted. Model selection semantics remain authoritative; do not invent sparsity for another architecture. [Long context](experiments/long-context/README.md) |
| Reuse a sparse query tile's union of KV rows | DeepSeek fast CSA/window attention through shared `mma_wide` | Generic D256/D512 attention; Qwen's separate QSA implementation | Unknown/reference defaults stay off. Count-based HCA retains ordinary MMA after a faithful-floor all-wide 32K oracle failure; this semantic split passes fresh 32K/128K repeats and matched PPL. D256 overlap/disjoint gains 4.58×/2.20×; D512 gains 1.50× overlapping but 0.556× disjoint. Fresh sampled top16-plus-other TV passes at 0.0034 / 0.0098 / 0.0186 / 0.0112; subsequent maximum-context timing, neutral retrieval and exact continuing-context swaps are recorded in the final-context report. The real Qwen paired-query transfer regresses despite passing operator bounds; see the rejected-kernel entry. [ds4 study](experiments/ds4-study/README.md), [diagnosis](experiments/dsv4-frontier-head/README.md#head-arithmetic-and-quality), [final context](experiments/m3-final-context/README.md), [QSA transfer](experiments/qwen38-qsa-pair/README.md) |
| Prepare an input once for adjacent products | Qwen prefill reuses BF16/MXFP8 input forms; ordinary GGUF gate/up can share Q8 and route preparation | Other products that consume the same logical input; raw NVFP4 has a different contract | The ordinary pair is generic but default-off outside measured DeepSeek fast plans; its 8K model gain was 0.8%, not a decisive product speedup. Check conversions and map identity before sharing. [ds4 study](experiments/ds4-study/README.md) |
| Capture preparation and borrow the unchanged consumer | Private 4,096-row IQ2_XXS compact pair | Other compact GGML products after separate shape/type qualification | Full native/original D4 bytes match, and either borrowed payload reproduces native gate/up exactly. Removing preparation changes resident pair time by about 4–6%; it does not measure original standalone consumer speed. Keep source-map audit representations distinct: native token-slot inverse versus original pair-to-token forward. Full operand guards, retained ownership and paid adaptation are reusable controls. [Prepared IQ2 factor](experiments/ds4-native-iq2-prepared/README.md) |
| Describe the actual selected quantized-product tile | Private same-D4 native/original IQ2 pair | Other compact types/shapes after separate qualification | Match the real selected tile and defining numerical object rather than infer it from guard capacity. Native runs at 0.880357× original resident rate with complete G/U errors below 1.44e−6; loading, scheduling, launch structure and arithmetic/compiler choices remain combined. Fixed full-byte operands, complete repeats, paid worklists and actual defining-TU attributes provide reusable controls. [Standalone IQ2](experiments/ds4-native-iq2-consumer/README.md) |
| Compact expert-major tile scheduling | DeepSeek fast ordinary-GGUF prefill at 2,048 rows or more | Other ordinary expert families, smaller chunks and formats; Qwen uses its separate grouped tensor-core path | Compact scheduling preserves raw weights, route maps and the full-K arithmetic. Alternating 8K model controls gain 11.1% community / 9.8% original throughput with identical captured logits. Q5_K/Q8_0 regress at smaller shapes, so generic/exact dispatch stays off. [ds4 study](experiments/ds4-study/README.md) |
| Direct original quantized product with a raw-layout loader | Community DeepSeek Q2_K down on GB10, every prefill chunk of 64–4,096 rows (fast-plan default) | Other Q2 consumers/shapes; IQ2 products need a separate format-specific loader | The MIT ds4 D2R kernel reads original raw blocks without a persistent SoA replica. Charged real-input latency falls about 26%, with 2.13e−7 NMSE versus compact MMQ; same-source 8K ABBA gives 3.68% mean prefill throughput gain and identical 128 IDs. Now a fast-plan default: its 32K PPL is within 0.04% of ds4 ([stage mechanisms](experiments/ds4-prefill-stages/README.md)). Keep maps/activation format/route weighting/cache/stage settings fixed when measuring a new piece. [Direct Q2 study](experiments/ds4-q2-d2r/README.md) |
| Write the routed activation from the up product | DeepSeek IQ2_XXS (occupancy-two J64) and IQ2_XS (GGML's J128) compact gate/up pairs, 256–4,096 rows; the Q8 form beside the community's Q2_K down | Other compact pairs once their launch is matched; the Q8 form for other down layouts (D4 for IQ3_XXS) | Byte-exact against the pair and SwiGLU. IQ2_XS's occupancy-two J64 launch measured 29.8 vs 23.6 ms for J128, so each type keeps its own launch. [stage mechanisms](experiments/ds4-prefill-stages/README.md#partial-chunks-and-other-quant-types) |
| Fuse the HC norm and expert sum into the post for any mix type | DeepSeek F16 (community) and F32 (0731) HC mixing weights | Qwen3.8's HC, if its products read a normalized input | F32 rows are rms_norm's own output, so the 0731 GGUF stays byte-exact. |
| Reuse decode input quantization | DeepSeek `Q8Of` shares Q8_1 conversions across compatible attention/shared/routed/head inputs | Ordinary primitive products still prepare their own input | Already adopted for DeepSeek decode; this is separate from the paired-MMQ prefill work. EXL3 trellis/Hadamard, MXFP8 and NVFP4 per-16-value Q8 have different representations. |
| Read one expert's weights for several verify rows | DeepSeek fast vector products | Qwen NVFP4 expert GEMV | Qwen tried the grouping already; current paired replication is neutral. Preserve the format-specific choices. [Qwen MTP](experiments/qwen38-mtp/README.md#judgement-calls) |
| Join one-row quantized products across requests | DeepSeek plain waves and Qwen3.8 GGUF plain waves, dense and routed products plus the full head | Other VecQ families after input/route/state qualification; multirow Qwen GGUF products stay separate | Qwen packs F32 inputs and pays Q8 preparation, preserving one-token sums with `SetVecQOneToken`. Ten experts per row limit joined groups to twelve slots under the 128-pair bound. Full heads and final state are exact against unjoined waves at production alignment, including capture/replay. [GGUF waves](experiments/qwen38-gguf/README.md#one-row-gguf-waves-2026-10-03) |
| Find a routed block's (token, slot) pairs with one warp's ballots | `jitllm.vecq` routed products: DeepSeek waves and verifies, Qwen3.8's GGUF experts | None found: the other routed kernels take expert-major lists or GGML's own ID handling | A serial one-thread scan cost more than short routed rows' products in four-request waves; the ballot scan keeps pair order and every sum. [Wave step](experiments/deepseek-batching/README.md#four-request-wave-step-and-scheduling) |
| Run each request's own operations of a wave on concurrent streams in one graph | DeepSeek and Qwen3.8 waves: each slot's attention, recurrence, compressors, indexer and cache writes (`AssignLanes`, `BoundGraph::Run`, `LaunchContext::ConfigureLanes`) | Qwen waves below four requests stay on one stream; unjoined library products stay on the owning stream | The executor orders every cross-lane read/write hazard (`LaneOrder`), and cuBLAS work stays on the owning stream. Separate lane scratch is charged to the budget. DeepSeek four-request step 86.5 → 82.8 ms, 124-token C4 +4.7–5.1%. Qwen NVFP4 verify 112.299 → 107.196 ms short, 114.020 → 105.141 ms long, with complete logits and state exact; GGUF short HTTP C4 +7.65%. [DeepSeek](experiments/deepseek-batching/README.md#wave-lanes), [Qwen](experiments/qwen38-four-request-waves/README.md#wave-lanes) |
| Join several requests' draft passes into one graph | DeepSeek DSpark waves (`BuildDsparkWaveGraph`): the blocks' products and the head read once, each slot's attention and Markov head its own | Qwen3.8's MTP drafts in shared waves | Bit-exact by the same column-invariant products as a joined verify. A four-request DSpark wave 240.9 → 231.2 ms (end to end level within noise at C4 and C5); the drafts were 13% of it. Measure a family's draft share first. [Joined draft blocks](experiments/deepseek-batching/README.md#joined-draft-blocks) |
| DeepSeek prefill chunks 8,192 | Rejected current output-A/HCA C1 screen: 10.663s versus6.992/6.999s through first output (+52.43% wall); keep4,096 | Shape-specific calibration needs a measured win | Same7,043 literal IDs, zero cache, one returned output with matching text digest; no kernel attribution or expanded quality/state claim. [Prefill chunks](experiments/deepseek-prefill-chunks/README.md) |
| Qwen shared draft caps one and three | Both C4 served screens rejected: cap one −9.75%; cap three 27.86 versus 32.63/31.98 completed tok/s, −13.75%; production keeps cap two | The existing configurable cap permits other workload-specific screens | Each screen completes twelve uncached 8K inputs and 256-output replies; draft-wave limit stays two. Adaptive branch depth can still vary. No kernel/acceptance attribution or cross-arm quality/state claim. [Cap one](experiments/qwen38-shared-depth/README.md), [cap three](experiments/qwen38-shared-depth-three/README.md) |
| Join selected draft heads through per-product MMVF selection | Qwen3.8 MTP BF16 heads (K2560, N16K–65K), immutable prefix views or selected leaves | Other draft-head widths/types require their own arithmetic and model qualification; full target heads retain MMF | Paid C2 HTTP +1.77%; short/8K complete target rows, tokens and initialized target/drafter state exact. Columns1–8 match ordinary MMVF bit for bit. Wider C4 drafting stays unadopted. [Head sharing](experiments/qwen38-draft-head-waves/README.md) |
| Group a wide wave's float products by whole requests up to the column-invariant count | DeepSeek waves and joined drafts (`FloatMm`: eight columns, two four-row verifies a product) | Qwen GGUF F32 routers: rejected four-request screen; native multirow BF16 routers need separate arithmetic qualification | GGML's vector float kernel keeps each column's sums up to eight columns, so grouping whole requests is exact; it halves the router and head-mix launches of a 16-row DeepSeek verify. The private Qwen GGUF one-row C4 transfer preserves all measured histories, IDs, full logits and initialized state, but its captured paid-wave rate gains only 0.245% while bookends change −0.269%; it stays unadopted. [GGUF router screen](experiments/qwen38-gguf-router-wave/README.md) |
| Cluster a small reduction and prefetch the next weights | Qwen3.8's hyper-connection preparation, up to eight tokens | DeepSeek's hyper-connection pre-projection | Source-compatible principle, not the same formula or weight layout. A remaining DeepSeek lead is about 1% of a step; measure only after the larger product gap. sm_86 has no clusters. [TensorFold techniques](experiments/tensorfold-techniques/README.md#limits-and-what-remains) |
| Fuse a small operation chain | Qwen short convolution/history, norm/gate and verify alpha/beta planes; DeepSeek routing/hyper-connections; image residual/next norm; shared RMSNorm/scale | Other phases/shapes and adjacent operations with matching rounding | Qwen fast verify F32 [48, 1..16] replaces four pointwise launches with one while retaining both Linear products, recurrence and commit. GPU chain/save views, full wave heads/states and owed-restore swap remain exact; matched 8K C4 HTTP gains 1.34%. Prefill/exact retain primitives. [Alpha/beta fusion](experiments/qwen38-gdn-gates/README.md) |
| Fuse per-head normalization and rotation | Qwen `QsaPrep`; DeepSeek fast canonical 512-value Q heads with a normal 64-value tail | Other widths/layouts/rotation types after qualification; EXL3 Qwen2 currently has no matching per-head norm | DeepSeek's corrected 8K screen gains 3.17% with all six full heads byte-identical. Preserve the native RMS reduction, the normalization store's F32 rounding, and actual RoPE FMA operand order; equivalent source expressions contracted differently before correction. Direct input dependencies remove the norm intermediate while preserving lifetime placement. [Q-head study](experiments/dsv4-qhead/README.md) |
| Share a full target head across requests | Qwen3.8 equal three/four-row BF16 heads, with independent state and adaptive policy | Other families after representation, shape and state qualification | Paid adaptive C2 HTTP gains 2.53%/1.33% in opposite orders, pooled 1.93%, with byte-exact native full heads and exact replies. This head-only factor does not establish shared-versus-serial or cross-engine parity; the preceding adaptive whole-driver screen was neutral and delayed first completion. [Request batching](experiments/qwen38-request-batching/README.md#three-row-heads-under-unchanged-adaptive-depth) |
| Admit literal requests to native cohorts | Qwen and DeepSeek generation-wave backends, alongside same-model chat requests | Other model families after generation-wave qualification | Prompt likelihoods remain one-row target work interleaved with peers; generated rows use the existing waves. Shared continuation output owns its decoders, scores and host-memory charge, without borrowing a retired exchange. Scored partial prompts and generated scores survive model switches exactly against uninterrupted C1 controls. [Literal cohorts](experiments/literal-batching/README.md) |
| Compute only the head rows the caller needs | Qwen3.8 fast prefill and measured DeepSeek Q4_K production prefill | Fixture/EXL3 output envelopes need separate work | The [frontier study](experiments/dsv4-frontier-head/README.md) narrows only the target head suffix after all recurrent/state work and DSpark feature capture. Fresh 32K/128K head repeats pass the unchanged oracle bound; 24 direct-runner arms preserve full state and common continuation exactly, with forced rejection and swap controls exact. PPL, scoring and verify keep all requested rows; community Q8 stays opt-in. The 1M planned peak is unchanged. |
| Pin library product algorithms by shape | Image BF16 and EXL3 reconstructed GEMM | Qwen BF16 multi-row products and other library products | Qwen's three/four/five-column HC products and full head were tuned with their actual types; gains stayed around 2% or less per product, with a slower router. No production pin was adopted. Share tuning/provenance discipline rather than another shape's pin; preserve output types and captured addresses. [Qwen tuning](experiments/qwen38-gemm-tuning/README.md) |
| Tune vector rows and warps for small output counts | Qwen MXFP8 products with 2,560 inputs and 48/512/640 outputs on GB10 | Other shapes/devices and vector product formats | The measured schedule retains per-lane FMA/reduction order and PDL. Fixed-depth-three 128K decode gains 0.8% prefix / 1.3% curated with identical 512-token trajectories. Large products stay original; 512/640-output columns 2/6/8 showed no gain or regressions. [MXFP8 scheduling](experiments/qwen38-mxfp8-scheduling/README.md) |
| Return device argmax verdicts for greedy verify | Qwen target verify copies IDs; full logits are optional | DeepSeek target verify still copies every full logit row | A bounded lean-greedy candidate; preserve lower-index ties and non-finite policy, plus sampled and diagnostic rows. No isolated DeepSeek speed gain measured. |
| Tensor-core attention for eligible dense shapes | Generic D128/D256/D512 kernels and image BF16 attention | Current Qwen2/EXL3 fixture: masked D64, GQA 14:2 | Existing D128 MMA and image kernels do not satisfy this mask/GQA contract. A new eligible D64 path belongs to the M3.5 comparison, not an unchecked dispatch change. |
| Fuse residual-add with the next norm | Image's fast denoiser | Future eligible dense-text graph | Shared RMSNorm/scale fusion does not already fuse the residual add. Image BF16 intermediates differ from GGML F32 and EXL3 casts; preserve each model's rounding. Unmeasured outside image. |
| State updates in place | Qwen one-row Gated DeltaNet; DeepSeek window ring; image prefix cache | A new recurrent family or verify implementation | Share lifecycle/accounting helpers. Recurrence formulas and rejected-row restoration are family-specific; no in-place update without its rollback proof. |
| Stable-address graphs and request leases | All three M3 runners, including image steps | EXL3/other families when they enter serving | Engine skeleton is shared. M3.5 integrations should reuse it instead of adding a parallel execution lifecycle. |
| Growing state and turn-boundary reuse | DeepSeek and Qwen serving | New recurrent families and their state layouts | The shared helpers already exist; each adapter supplies its footprint/checkpoint semantics. Prefix matching still does not identify a conversation's lifetime. |
| Selected head and confidence-based draft control | Qwen optional externally supplied selected BF16 head; adaptive depth 2–3 | Other learned drafters with their own vocabulary/state contracts | Curated head and depth choices are measured independently; curated is not uniformly faster. Fixed depths 3–5 and confidence windows, including actual incremental draft stopping, did not close the 128K speed gap. Sampled policy stays fixed; the final selected head passes the fresh distribution check. [Calibration and rejected prototypes](experiments/qwen38-mtp-speed/README.md) |
| Match a draft head's input rounding | Qwen's bounded retained-input diagnostic and external replay; serving remains F32 input | Learned draft heads with a proved rounding difference | On 12 real inputs per head, BF16 RN/widen plus the original vector product changed no choices and added 2–3 µs; the charged BF16 library product was about 40% slower. Neither arm was integrated or tested for full-model acceptance. Selected IDs/depth remain separate factors. [Draft-input experiment](experiments/qwen38-draft-input/README.md) |
| Portable text seeds and local draft-vocabulary adaptation | No implementation; deferred by the owner to later optimization passes | Models with a selectable draft head and full target verification | Tokenize curated prose/code/output-format strings with each model's tokenizer, including boundary variants and actual output/stop delimiters; do not share token IDs or match decoded labels alone. Combine seeds with representative answer frequencies, then separately test bounded local output/verifier counts and charged between-request updates. Keep frozen held-out data independent, a broad fallback and unchanged target/distribution gates. [Study and follow-up](experiments/qwen38-own-draft-vocab/protocol.md) |

Gemma's [Q8 assistant component](experiments/gemma-assistant-execution/README.md)
reuses the target stream/cohort/catalog, stable PlanCache/GraphRuns, pinned
staging and dedicated recurrence storage. Ordinary primitive C1 and independent
C2 execution exist; one frozen original-input C1 chain has complete arithmetic
byte identity. Same-slot cache/feature protection is explicit, with peer
progress and completed payload witnesses. No optimized assistant join is
selected. Width1,024 norm/residual fusion needs a new checked eligibility
contract; Q8 joins/shared preparation/row-invariant products require actual
1,024/8,192 and projection/head qualification. Device frozen-prefix masks,
lean canonical heads and adaptive depth remain measured transfer work, not
inherited target defaults. The default-off
[engine-only target verifier](experiments/gemma-target-verify/README.md) now
shares LiveState row snapshots/accept/rollback, explicit host metadata funding
and separate retained-feature storage. Both profiles pass focused C1
transaction controls, including ring wrap and rejected bytes; assistant
acceptance/serving and scalar-prefix parity remain unqualified. Full heads and
funded host masks are the present diagnostic baseline.

The [Gemma26 assistant C2 screen](experiments/gemma-assistant-c2/README.md)
checks ordinary joined products and per-segment attention on distinct frozen
histories. Native serial and joined policies reproduce all six original
serial-reference heads/features exactly. Both retain the measured differences
from original physical batch two, with all six greedy IDs agreeing.
For 32 two-owner waves, native joined takes 0.09927 s against original serial
0.18207/0.18429 s, and 0.10118 s against original batch two
0.12050/0.11940 s. These paid component screens include full host publication;
they do not select a serving policy or establish full-model qualification.
Larger batches, device recurrence/masks and other widths/formats still need
their own transfer checks.

Gemma's native GELU-tanh and split GeGLU now have primitive fallbacks.
The [floating GeGLU screen](experiments/gemma-activations/README.md) checks
matched F32 accumulation at K2816 with F16 weights: N704 gains while N2112
loses. GeGLU fusion has its own eligibility callback, off by default;
neither approved Gemma GGUF's quantized FFN weights take this MMVF path.
The explicit quantized `VecQGlu::kGeGlu` writer now calls the pinned GELU
helper at Q4_K expert704 and Q8_0 shared2112 widths. Its [paid complete expert
chain](experiments/gemma-quant-geglu/README.md) separates input-preparation
reuse from writer fusion; no universal writer default is justified. It keeps
full runtime slab strides and canonical tails, including pitches not aligned
to 256. Standard activation/Q8 writeback remains separate; no Q4_K shared
block decode rewrite or paired-MMQ transfer is adopted.

Gemma's [standalone routing and scaled-reduction contracts](experiments/gemma-moe-primitives/README.md)
expose the unchanged pinned GGML launchers with checked 128-expert/top-eight
routing and width-2,816 three-input ordered reduction. The existing full
ARGSORT-root backing remains charged; only its first eight IDs are written.
This availability closes an operator gap without selecting a graph policy.
Readability of kept intermediates, whole-model reference arithmetic and paid
solo/joined execution still govern any later adoption. The DeepSeek strict
rounded two-input reducer is a different contract and cannot substitute.

The [structural matchers](gemma-moe-matchers.md) recognize the exact routing
chain and contiguous ordered reduction, preserving all descriptors for
placement and refusing kept or externally read elided values. The
[native default-off dispatch](experiments/gemma-native-moe/README.md) expands
the ordered raw routed sum before its post-norm, yielding all 30 routing and
30 reduction matches on actual 1/2/4-segment graphs. All descriptors/full-sort
backing remain funded. Compound norm/MoE has nine out-of-noise representative
argmax differences; a 3.03% short native timer gain does not select a default.
Whole-model/reference, optimized joining and long-context gates remain owed.

Gemma's historical default-off [joined runtime diagnostic](experiments/gemma-joined-serving/README.md)
reuses independent completed-unit ownership and one-row invariant products,
bounded to eight owners per shared group. Its C12 uses ordered8+4 with funded full
heads and per-group refusal isolation. Solo/joined heads and state agree, while
natural-prefix stock quality and C12 competitive performance fail; no default
is selected. Its [transfer audit](experiments/gemma-joined-serving/transfer-audit.md)
records the eligible Qwen/DeepSeek mechanisms and unselected math contracts.

Gemma's [checked norm chains](experiments/gemma-native-norm/README.md) reuse the
original D256/D512 F32 norm/NEOX and 2816/5376 residual-add launchers through
separate default-off policies. Kept/view-read intermediates retain primitives;
final residual GET_ROWS stays paid before a deferred norm/add step. Dense31's
measured 128-row complete heads match stock, but its paid 8K comparison against
reference ubatch256 still differs and gives no stable speed winner. This does
not select a default or qualify routed26, optimized batching or long context.
The [natural C4 norm/row first screen](experiments/gemma-joined-norm/README.md)
selects both checked norm chains during prefill and decode and keeps native
scalar/joined heads exact. Fresh stock has 33/128 strict positive-margin
differences; candidate latency is 1.7523% above the fresh stock bookends.
This negative candidate remains unselected; the earlier 128-row proof does not
transfer to this shape.
The [ordinary-product norm C4 screen](experiments/gemma-joined-norm-ordinary/README.md)
restores column-dependent product selection with both norm chains. Fresh
recipe-aligned stock ubatch128 still has 15/128 strict positive-margin
differences; candidate latency is 1.1161% higher. Own C4 repeats remain exact,
but no C1=C4 equivalence or selected optimization follows.

The [identical-operand C4 attention replay](experiments/gemma-attention-c4/README.md)
isolates two differences on dense31's actual first-local inputs: native
segmented vector attention differs from MMA, and segmented MMA differs from
four-stream MMA. Native four-stream MMA matches the unchanged original
backend byte-for-byte across all 32,768 attention values, with its original
launch family and geometry observed separately. This establishes kernel
fidelity on these inputs, not the full-logit cause or a selected model policy.
Gemma26's D256/GQA2 local attention is the first backward-transfer candidate;
its different head counts and stream geometry require measured controls.
Audit other models' actual dispatch and reduction shapes before transfer.

The [packed C4 full-head screen](experiments/gemma-packed-attention-c4/README.md)
then joins local and global attention streams in a benchmark-only dense31 graph.
All 128 complete heads match both fresh original bookends byte-for-byte;
the unchanged norm/ordinary-product control retains 15 positive-margin misses.
This closes the tested joint dispatch/stream geometry explanation, without
isolating local query precision from reduction geometry. Paid packing raises
latency 12.2774% above the reference mean. Avoiding those copies is the next
performance lead; no production policy is selected. Gemma26's independent
first transfer screen below fails. Other contexts, query shapes, assistants
and representative quality remain unqualified.

The [Gemma26 packed-attention transfer](experiments/gemma26-packed-attention-c4/README.md)
uses its actual 16-head/local-KV8/global-KV2 profile and unchanged routed
Q4_K/Q5_1/Q8_0 products. Complete heads and initialized states repeat exactly,
but 68/128 choices retain positive reference margins and latency rises 9.1881%.
The unchanged segmented control has 77 misses; candidate early-wave misses
increase, so fewer total misses do not qualify the transfer. Both optional
routing/reduction policies are off. The older layer-28 allocation eligibility
observation does not explain this result. No production policy is selected.

The [actual Gemma26 stock dispatch observation](experiments/gemma26-dispatch-observation/README.md)
preserves all 128 complete stock heads while logging routing/reduction gates.
Stock selects all 30 routing and scaled-reduction fusions in four-row decode;
each 64/65/66/67-row prefill refuses routing in layers 28 and 29 at the memory
gate, while scaled reduction stays fused. The packed native screen above
leaves both families off. These phase-specific observations identify another
recipe difference, not its complete numerical cause or a production policy.
Do not infer eligibility from the older 128-row observation, imitate incidental
aliases, or adopt a layer whitelist. Future compound tests must account for
the mixed prefill history. Routed products select MMVQ at decode and MMQ at
prefill for the actual Q4_K/Q5_1/Q8_0 weights.

The [packed Gemma26 compound screen](experiments/gemma26-compound-packed-c4/README.md)
adds the existing checked routing and reduction to that packed norm control.
Positive-margin disagreements fall from 68 to 2 of 128 heads, and 92 complete
heads become byte-exact. Paid latency remains 7.15% above the reference mean.
Native selects all 30 routing fusions during prefill, so the stock late-layer
memory refusals remain a recipe difference. The residual misses require a
common-input probe; neither a specific cause nor a production policy is selected.

The [actual-input scalar FFN replay](experiments/gemma-dense-ffn/README.md)
checks dense31's layer-zero Q4_K gate/up and Q6_K down products at width 5,376.
Observed original quantized GeGLU fusion, separate original products/GeGLU
and native primitives match all 21,504 activation and 5,376 down-output values
byte-for-byte. This input rejects that arithmetic-gap lead; it does not qualify
other FFN inputs, select fusion or explain the whole C1 gap. The short component
timing differences are within reference bookend movement. Standalone binders
must preserve the checked readable-tail marker for these padded Q4_K rows.

The [solo checked-norm screen](experiments/gemma-dense31-c1-norm/README.md)
then matches all 32 dense31 full heads to both fresh C1/u128 original bookends
byte-for-byte, using ordinary products and both checked norm chains. Ordinary
control has nine strict differences, with its own noise unmeasured. Candidate
latency is 1.9202% higher, so the short math result selects no production policy
and supplies no broader state, context or performance qualification. The
unchanged reference uses its C-API full sliding-window cache default; this
short prefix never wraps. Other models need independent transfer controls.

The [8K ring-cache reference screen](experiments/gemma-swa-ring-h1/README.md)
then explicitly matches the original CLI/server sliding-window policy to
native's bounded cache, leaving native math unchanged. All 32 incoming-head
choices agree and the final complete head is byte-exact, while the prefill
head still differs. Native prefill/decode take 16.1957%/2.0504% more time.
Historical full-cache ratios do not establish representative ring-cache
speed or memory parity. Match effective cache topology, capacity and read
policy in other families' comparisons. No policy is selected.

The independent [Gemma26 ring-cache transfer](experiments/gemma26-swa-ring-transfer/README.md)
uses its actual 1,024-row compound policy and 2,048-cell local ring. All 32
incoming-head choices agree, but both retained complete heads still differ;
dense31's byte-exact final head did not transfer. Native prefill/decode take
57.6433%/0.4880% more time. Native repeats and the bookend preserve exact own
initialized state. The remaining full-head and prefill-speed gaps need their
own diagnosis; this screen selects no policy or full quality/performance pass.

The [matched Gemma26 prefill profile](experiments/gemma26-prefill-profile/README.md)
retains the earlier complete heads/choices and initialized native state.
Instrumented GPU activity unions are close (2.469 s native, 2.364 s stock),
while wall time outside recorded GPU activity is 1.361 s versus 0.085 s.
Extra head-shaped kernels account for about 26 ms; VMM API bodies total about
49 ms. Neither establishes critical-path cost or explains the residual interval.
CPU and wait attribution must precede changes to planning, staging or residency;
no optimization or cross-family transfer is selected from this profile.

The [coarse Gemma26 diagnosis](experiments/gemma26-prefill-coarse/README.md)
measures about 1.093 s of charged caller CPU in the ordered first/second
graph-plan passes, with no recorded GPU overlap and exact prior head/state
fidelity. The [call-local root/read index](experiments/gemma-plan-index/README.md) now
removes repeated whole-graph reader scans while retaining both passes, SamePlan,
primitive fallback, malformed-view, keep/readability and post-placement guards.
Gemma26 prefill is 27.3% faster than its retained baseline; matched Gemma26/31
initial prefill gaps were 15.10%/13.83%. Both preserve exact prior native heads/state.
Qwen/DeepSeek planner regressions pass; unchanged primitive policies skip the
index. Full quality/performance qualification remains open; no CPU stack or
matcher-specific attribution is claimed.

The [exact source-use extension](experiments/gemma-use-index/README.md) also
removes repeated full-graph edge counts from generic fusion gates. It preserves
exact pointer, duplicate/self-edge, local-subgraph and standalone semantics.
Fresh Gemma26/31 prefill improves 2.80%/2.31% against unchanged controls; remaining
reference gaps are 12.38%/11.40%. Prior-family planner controls pass; no kernel
math, selected policy or full qualification changes.

The [cold/retained-plan diagnostic](experiments/gemma-retained-plan/README.md)
verifies all 8/32 Gemma26/31 prefill cache hits with capture disabled and identical
state reset. Full cold planning costs 126/526 ms; state growth costs 64/164 ms.
Retained prefill improves 7.12%/4.77% in one pair, while execution also changes.
Planning is material but does not explain the whole remaining gap. Existing
plan retention helps repeated shapes; no new production reuse mechanism or
competitive batching/quality pass is established. Its intermediate-head lead
was addressed by the state-only path below, which preserves final KV writes
while pruning downstream work, rather than only disabling the head.

The [plain RMSNorm/Mul screen](experiments/gemma-normmul-screen/README.md)
selects 121 existing fusions on Gemma31 while preserving exact native heads,
state and choices. Two candidate runs show no resolved end-to-end gain against
the fresh off control; the benchmark flag defaults off and production keeps
the selector unselected. No Gemma26 transfer or broader ladder was run.

The [current Gemma31 timeline](experiments/gemma31-current-timeline/README.md)
measures 592 ms in 32 internal GPU-idle gaps over 10 ms and 145 ms of extra
vocabulary projections. Product and attention duration sums are close to stock.
Native also executes 3,872 extra standalone MUL kernels, while final-block FFN
row shapes differ because native narrows earlier. Those differences prevent
an equal-work kernel-parity claim; gaps alone do not identify caller CPU work.
This strengthens the state-only prefill lead and motivates further planning
work. It does not override the unresolved plain-norm screen or establish a
Gemma26 transfer result.

The [state-only prefill path](experiments/gemma-state-only-prefill/README.md)
omits unused final-layer query/attention/output/FFN/head work only from
non-final, non-scoring Gemma prompt chunks, preserving all KV stores. Both
approved Gemmas retain exact initialized state and continuation under ordinary
and optional policies, including captured scalar and unequal-row waves. The
measured all1024/both256 recipes improve 3.04%/2.42%; the 26B reference drift
prevents a resolved competitive gap. Qwen/DeepSeek retain the default full-head
hook: their MTP/features, hyper-connections and recurrent state need separate
dependency-cut qualification before this transfer. No batching or quality gate
is inferred from head omission.

The [bounded graph traversal](experiments/ggml-graph-order/README.md) replaces
quadratic visited-vector membership in all nine Qwen/DeepSeek/Gemma graph
factories with an arena-owned pointer table. DFS order, cycles, PARAM leaves,
capacity refusal and explicit scratch/retained-host charges remain checked.
Standalone callers keep the original overload. Exact Gemma26/31 head/state
controls pass; fresh state-only prefill improves 1.93%/0.99%, leaving
8.19%/7.54% reference latency gaps. Previous-family graph/plan controls pass;
no previous-family model speed or planning-only gain is claimed.

The [bounded Gemma lookahead](experiments/gemma-prefill-lookahead/README.md)
builds only the next structural CPU plan during current execution. Host funding
precedes submission; binding, coverage and cache insertion follow proven
completion. Optional refusal, an abandoned hint and inline execution retain
normal behavior. Same-binary state-only prefill improves 1.55%/2.36% on
Gemma26/31 with exact native heads/state. This transfers the overlap opportunity
from the earlier runners without changing future KV ownership. Previous-family
default prompt hooks ignore the hint; no additional Qwen/DeepSeek speed claim
is made. Scoring, features and final prompt heads retain their full paths.

The [plain norm adoption](experiments/gemma-state-only-norm-policy/README.md)
now selects existing checked RMSNorm/Mul by default for both approved Gemmas.
Ordinary off/on controls preserve complete heads, initialized state and choices;
new default scalar/unequal-wave continuation and capture controls pass. Research
state-only/lookahead comparisons favor fusion at both profiles, with noisy 31B
effect magnitude. This supersedes the earlier unresolved full-head screen for
default selection, while broader corpus quality and batching remain owed.
Other arithmetic experiment defaults remain off.

The [current post-lookahead accounting](experiments/gemma-current-phases/README.md)
finds 31/32 required plans cached and about 12 ms synchronous planning. Nested
binding costs about 175 ms across 32 insertions, versus 17 ms coverage and
5 ms insertion; state growth costs 162–185 ms. These nested elapsed spans
are not additive with the outer phases or evidence of active GPU arithmetic.
Attention occupancy queries and execution-plan identity construction remain
unmeasured binding subcomponents; no new optimization is selected here.

The [per-bind wrapper cache](experiments/gemma-binding-wrapper-cache/README.md)
removes repeated declaration validation within the shared GGML executor. Keys
are immutable resolved registry pointers, values are prior bound step indices;
every occurrence still validates current operands, arity and lanes. Temporary
metadata is bounded within the existing per-node binding allowance. Measured
Gemma31/26 binding falls 73.98%/69.29%, with exact heads/state and focused
late-failure, stale-identity and capture controls. Qwen/DeepSeek share this
executor change; their model speed is unmeasured here. Wall timing favors the
candidate but has material spread, and competitive/quality gates stay open.

The [grouped state diagnostic](experiments/gemma-state-preparation/README.md)
uses the existing LiveState-backed preparation API once for the known 8K prompt,
inside the paid timer. Later chunk calls still occur and find initialized cells.
Gemma31 state time falls 61.1245 ms / 33.45%, combining closure/acquire and
materialization grouping; execution spread prevents a wall-speed conclusion.
Exact heads/state/choices remain. This is a transferable state-path lead, not
an adopted policy: pressure, partial refusal and continuation lifetime need
qualification before earlier full-prompt backing is selected. The [26B transfer](experiments/gemma26-state-preparation/README.md)
reduces paid prefill by 20.375 ms / 0.822% and state preparation by 18.345 ms,
with exact heads/state/choices. Neither diagnostic establishes a fresh
competitive comparison or production adoption.

The [independent C4 attention owner-root proof](experiments/gemma-owner-root-attention/README.md)
reuses the pinned MMA tile/fixup arithmetic with eight checked K/V addresses,
keeping the packed kernel's grid and reduction partitions. All 32,768 real
D25631 outputs, fresh/restored controls and captured repeats match byte for byte.
The operator timer excludes packing, so it establishes no copy-removal gain.
The [first-global D512 proof](experiments/gemma-owner-root-global/README.md)
adds all 65,536 native-origin output values, fresh/restored controls and captured
repeats, preserving the original 96-block grid. The [closed C4 consumer](experiments/gemma-owner-root-c4/README.md)
uses a checked ten-source operation over the exact cache writers, removing only
K/V CONCAT while retaining Q/mask packing and original grid/partitions. Paid
latency falls 8.93%; all 128 heads/four initialized states match packed repeats.
Plain-norm-on remains 2.10% slower than fresh original bookends with all 128 heads
byte-exact. The [Gemma26 backward transfer](experiments/gemma26-owner-root-c4/README.md)
uses 30 layers / 16 query heads at context/read 256, preserving measured local
48-block and global 64-block geometry. K/V copy removal lowers paid latency
7.13%, with all 128 native heads/four states unchanged; the fresh stock gap is
0.265%, but two positive-margin choices still fail strict quality. Other read
widths/window crossings and assistant transfer need actual controls. The
[variable-width native owner option](experiments/gemma-owner-variable/README.md)
now defaults off and supports checked C4/query1/real-parent spans through 16384
read cells, retaining original compiled grid/partitions and whole-graph fallback
for unequal widths or unsupported shapes. At 8K, eliminating K/V CONCAT lowers
paid C4 latency 53.95%/43.71% for 31B/26B with exact native heads/four states/choices.
Fresh reference timing is +2.31%/−5.59%, but strict quality fails 4/128 and 25/128
positive-margin choices. Native 128-row untimed prefill differs from stock 256/1024;
26B ring capacities also differ 1280/2048. This evidence does not qualify defaults,
wider cohorts, assistants or corpus quality.

The [current C4 phase split](experiments/gemma-owner-c4-phases/README.md)
records 32 plan hits, 24.74 ms of checks and 78.73 ms outside execution,
with exact native heads/state/choices. The latter includes 43.57 ms outside
all runner phases. Repeated immutable-weight placement scans are a lead;
this diagnostic establishes neither their share of the checks phase nor
active GPU time or a fresh competitive gap.

The [successful weight placement memo](experiments/weights-placement-memo/README.md)
reuses an immutable-weight check under the same scheduler lifetime and
source/pin epoch. Its original control kept mutable-state checks fresh; every
scheduler Call remains.
Current 31B C4 checks fall 23.50→2.92 ms and paid elapsed falls 0.77%, with exact
heads/state/choices and 17 focused lifetime/mutation controls. The shared caller
optimization applies to earlier runners, but their speed is not measured here.
Token/epoch saturation refuses caching; no persistent per-extent cache or math
policy changes. The full suite remains owner-deferred.

The [successful live-state memo](experiments/gemma-state-placement-memo/README.md)
extends that stamp to the existing Gemma/Qwen3.8/DeepSeek state checks, with
invalidation before mutable source/binding/growth/restore/clear/release changes,
including refusals. Only successful local source/pin validation is cached;
residency, leases and every scheduler Call remain independent. Nineteen focused
controls prove hits, invalidations, saturation and actual eviction/restore;
complete C8 heads/states/layouts/choices are exact for both Gemmas. The short
26B same-host factor falls 29.195 ms/1.7661%; 31B latency gain is unestablished
within spread. No other-family speed, phase or fresh-reference claim. Sixteen
inline bytes per state are included in pre-probe occupancy; the manual helper
funds its 192-byte maximum increment inside its existing metadata envelope.

The [immutable Gemma head capacity](experiments/gemma-head-capacity/README.md)
separates pinned publication from input capacity: serving uses owner slots,
manual teacher-forcing keeps its full-row default. Eighteen focused controls
cover both profiles and validate at least 124 MiB less pinned staging for cap4
versus max128. Input-row policy stays unchanged. Earlier Qwen/DeepSeek runners
already bound ordinary solo head storage and retain their existing output
envelopes; any further reduction needs separate feature, scoring and MTP
publication accounting rather than automatic transfer.

The [fresh input qualification](experiments/gemma-input-preparation/README.md)
independently checks both checkpoint tokenizers on twelve fixed prose slices
and four natural prompts. All supplied 1,024-token prefixes and complete chat
IDs agree with the public tokenizer; actual native/Jinja renders agree, with
one BOS each. The checkpoint-specific manifests bind the inputs even though
their carrier bytes match. Corpus scoring and cohort frontier targets remain
separate: 1,023 within-history transitions, no target beyond position 1,023.
This reusable input harness changes no arithmetic policy or quality allowance
and establishes no model quality, assistant or batching gate.

The [current corpus screen](experiments/gemma-current-quality/README.md)
checks the actually measured solo research policies with plain norm fusion:
31B both256 matches all 1,024 complete fresh ring-reference heads byte for byte;
26B all1024 has nine positive reference-margin choice differences and only
15 exact heads, failing strict quality. Its +0.0528% relative PPL change does not
waive that gate. Each model freezes complete same-policy own repeats before
stock. These new shapes use no inherited 128-row allowance and do not qualify
frontier-only 8K work, whole contexts or optimized batching.

The [current 1,024-row Gemma26 dispatch observation](experiments/gemma-current-dispatch/README.md)
reuses an existing stock-policy controller without changing floating math.
All 1,024 complete heads match the frozen reference. Actual stock routing is
selected in layers 0–27 and 29, versus native's 30 layers; norm and reduction
counts match. The logger gives no layer-28 refusal reason. This is a concrete
recipe difference to isolate, without attributing the nine failed quality choices
or selecting a production layer policy.

The [Gemma26 routing keep factor](experiments/gemma-keep28-routing/README.md)
resolves the fixed 1,024-row corpus difference: retaining only layer 28's
routing probabilities makes all complete heads byte-identical to stock.
This establishes the routing-policy cause of the earlier nine disagreements
for that recipe. The diagnostic does not select a production layer whitelist;
the [actual refusal observation](experiments/gemma26-routing-gate/README.md)
identifies 32,768 bytes of weights/logits overlap at 1,024 rows. The original
memory gate refuses that allocation, while all structure/shape gates pass.
A general native policy still requires qualification; no static layer rule follows.

The [opt-in owner cohorts](experiments/gemma-owner-cohorts/README.md) preserve
real cache roots while matching the whole-eight stream-K partition for eligible
C8 waves. Dense31's 256 paid heads become byte-exact; 26B passes its unchanged
prior margin/conditional-score gates. The checked parent-storage bound is now
1 GiB while executed operands remain 64 MiB and reads at most 16K, admitting
short reads with configured 262K backing. Packed attention keeps its previous
limits. General serving selection, C12, depth and other family transfers remain
unqualified; no shader arithmetic or model default changes follow.

The [ordinary whole-C12 factor](experiments/gemma-c12-single-wave/README.md)
uses one twelve-column product group and three real-root attention quads,
matching whole-twelve stream-K partitions when divisible and otherwise retaining
four-owner geometry. Invariant product waves retain their eight-row limit.
Dense31 matches all 384 paid stock heads; 26B passes its unchanged prior bound
and conditional-score gates with three strict differences. Both own repeats
retain full initialized states and finite heads. Short before/after means fall
30.98%/28.03%, with landed memo/parent-validation source changes and timing
spread disclosed. Original 8+4 failures and broader gates remain open; owner and
joining defaults stay off. Other families need their own product/attention rules.

The [fresh fixed-capacity schedule control](experiments/gemma-fresh-quality-validation/README.md)
keeps Gemma26 maximum rows 1,024 and F16 capacities 2,048/4,096 identical while
changing only supported host chunk schedules. History 13 freezes genuine
operational p99 margin movement 9.5026, with 209 differing native choices;
exact own repeats remain a separate zero requirement. Independent history 14
has 17 strict disagreements, none outside that unchanged bound, and +0.1970%
PPL. The declared gates pass while strict zero-difference fails. This manual
control adopts no arithmetic policy, layer whitelist or runtime default and
supplies no allowance for batching or other histories.

The [fresh current reference bookends](experiments/gemma-current-reference/README.md)
measure the accumulated solo recipe: stable 26B latency gaps are 2.43% prefill
and 0.92% decode. Native 31B prefill is stable while stock moves 1.3355 s;
native is 2.08% slower than the closing stock arm, so no parity or gain follows
from the reference mean. Native execution spans fit within reference whole
prefill times; remaining state/host spans are a lead, not proof of a particular
allocation subcallee or GPU arithmetic cost. Complete retained native
heads/state/choices stay exact. Corpus and C4 gates remain independent.

Gemma 4's [foundation transfer checklist](gemma4.md#required-execution-and-optimization-qualification)
maps these selected techniques to its actual GGUF operand contracts and
independent request segments. It records the checked Q5_1 primitive controls
and GELU/GeGLU primitive controls, including the explicit quantized writer.
Model selection and model-execution gaps remain; solo/batched execution
qualification remains owed before either checkpoint is supported.

Gemma's optional graph-owned causal/ring mask producer replaces mandatory
host O(rows × context) mask construction/staging with fresh I32 positions.
One global/local mask pair per independent segment is reused across layers;
all query/cell padding is initialized to negative infinity. The funded host
reference path remains available, and source checks still authenticate tokens,
positions and cache indices. [Bounded mask controls](experiments/gemma-device-masks/README.md)
compare complete paid layers and captured execution. This transfer closes a
primitive/input staging gap, not whole-model serving or batching qualification.

Gemma's explicit default-off per-segment RoPE/cache-store policy transfers the
existing checked factor-aware fused launcher without enabling generic fusion.
Joined learned K normalization, Q rotation and global raw-K-as-V remain intact.
Complete zero-offset flattening views make each independent store eligible;
unsupported geometry and diagnostic keeps retain the ordinary producer.
Current activation placement still funds rotated intermediate storage, so the
transfer saves writes/launches rather than claiming physical memory elision.
[Actual-width complete-layer and cache controls](experiments/gemma-rope-store/README.md)
show byte agreement and no decisive paid solo/four-segment gain. This is
conditional primitive availability; whole-model adoption and optimized-batch
performance qualification remain separate.

## Rejected kernels: pieces worth retaining

Gemma's [grouped independent cache-write experiment](experiments/gemma-grouped-cache-writes/README.md)
was rejected without adopting its source or defaults. Both six-control proofs
retain four original outputs/dependencies and exact full padded destinations;
31B C12 original/fast-divider means increased 7.18%/3.22% with complete native
heads/state/choices exact. Timing movement prevents a divider-cause claim.
Compiled kernels have zero local/stack memory and direct parameter loads,
rejecting the per-thread argument-copy hypothesis. Zero extra scratch does not
establish unchanged peak; delayed activation lifetimes and untimed prefill
matching cost remain caveats. Qwen/DeepSeek views, formats, consumer ordering
and lane tags exclude a claimed current quartet transfer. No 26B model transfer
or new reference/quality allowance was introduced.

A rejection applies to its tested shape, precision and surrounding work.
An isolated component remains a candidate until measured with a compatible
consumer. Raw prototypes and traces stay in external scratch; durable
aggregate results and upstream source identities belong in the linked reports.

| Whole implementation or choice | Why it was not adopted | Piece that may transfer | Required next proof |
| --- | --- | --- | --- |
| Qwen four-request GDN recurrence in native waves | The exact captured operator gained 26.51% against four serial launches, but the integrated lane-enabled verify is neutral: 107.399 ms versus 107.228 / 107.018 ms disabled bookends | Unchanged F32 recurrence body, bounded by-value per-slot records and independent state/output checks; full target-row/state authentication with capture/replay | A different shape or schedule needs its own paid model gain. Fewer launches alone do not justify adoption; HTTP, swap and memory gates were not run. [Integration screen](experiments/qwen38-gdn-cohort/README.md#native-wave-integration-screen--2026-10-04) |
| DeepSeek routed expert lists prepared once per product | Representative C4 DSpark median falls 0.544% while controls move −0.984%; no stable gain or adoption | Same first owner and pair order, charged stream scratch, original sums, complete solo/wave and discard/departure controls | Keep the inline ballot scan. This does not reject a shared graph node amortized across adjacent products or other schedules. [List screen](experiments/deepseek-expert-worklists/README.md) |
| DeepSeek wide IQ2_XXS gate/up with multi-token passes | Four-token and two-token passes raise representative C4 DSpark wave median latency by 0.512% / 1.400% versus their bookend means; neither is selected | Exact shared sign/grid/scale decoding, unchanged R2/W4 warp-row reduction and bounded token tails; existing scalar/wave, discard and departed-slot controls | A different consumer or schedule needs a paid model gain. This does not reject routed-down or worklist factors. [Expert-pass screens](experiments/deepseek-expert-passes/README.md) |
| Qwen NVFP4 expert output-tile counts | Doubling gate/up/down output tiles from 4/8 to 8/16 raises representative C4 verify latency 1.501%; halving to 2/4 raises it 0.326%, so neither is selected | Exact per-output arithmetic, fixed routing and sharing cap; full verify-row/final-state checks with capture/replay | Retain 4/8. A different shape or layout needs its own model screen; this does not reject a separate worklist factor. [Tile screens](experiments/qwen38-expert-tiles/README.md) |
| Qwen grouped NVFP4 verify products | Earlier four-row test: 21.5–21.9 ms versus 18.7 ms; current 128K paired model test also neutral | Matching expert/row scheduling and leader selection; L2 already serves repeated weights | Isolate scheduling overhead or a new shape with demonstrably poor cache reuse. Do not repeat the same grouping as a new idea. [Qwen MTP](experiments/qwen38-mtp/README.md) |
| Qwen FP4/BF16 grouped chain in small verify | Existing chain loses19.0–19.2% on fixed3/curated128K; the actual Mia cooperative tile/stage pair loses33.3–33.6%. Prototypes removed | Existing FP4 preparation, BF16 product/GLU contracts, shape-bounded grouped schedules and common-input controls | Mia's static a1/a2 scaling and fused row-expansion/scatter still differ. A literal consumer piece needs charged preparation/product/combine costs, same-history quality and model gain; a tile name alone does not establish equivalence. [Grouped factor](experiments/qwen38-grouped-verify/README.md) |
| Literal FlashInfer static-a2 G2 consumer | Complete captured-input cold latency rises20.95–27.64%, warm28.77–29.27%; no model candidate | Bounded native operand capture, source-visible BF16/static-FP4 conversion, per-expert pre-BF16 alpha and actual CUDA scalar-bit proof | Preparation alone costs47.4µs versus the complete original151.96–160.48µs. This standalone factor does not reject the actual fused whole consumer or establish model quality loss. Initial private vLLM quantization and fused preparation/finalization remain separate pieces. [Literal G2 study](experiments/qwen38-fi-down-stage/README.md) |
| Actual FlashInfer complete routed consumer | Traced fused T4 pair is2.32% slower cold/effectively neutral warm; ordered is1.75%/1.73% slower, so no native port or model candidate | Existing-view first-product capture; authenticated ModelOpt row/SF permutations; actual private-quantizer and source/trace tactic proof; charged complete-chain graph controls | All24 native T4/T3 byte controls pass. Native/ordered/default graph outputs match12/12 each; fused12/12 vary from eager despite matching private operands and finite output. Traced atomic finalization also fails exact own repeat on23/24 inputs; ordered repeats pass but do not gain speed. Operator relative L2 is not model quality loss. These controls are reusable across packed-expert consumers when actual loader/scale contracts match. [Complete consumer](experiments/qwen38-fi-down-stage/README.md#complete-consumer-follow-up) |
| Qwen draft-head BF16 operand/output matching | RN plus original MMVF is neutral; cuBLAS F32 and actual BF16-output arms are about41% slower on12real rows per head | Completion-aware kept operands, original-logit byte checks, full-vector ranking/TV and charged widening controls | Captured dtypes alone do not reproduce Mia's MTP state or whole library path. No speculative acceptance batch follows a slower head; reuse identical operands for a new source-backed piece. [Draft input](experiments/qwen38-draft-input/README.md) |
| Qwen precomputed Q8 activations for multi-row verify experts | Current paired 128K model test neutral after 96 added preparation launches per verify; one-token products stayed inline | Exact per-16-value Q8 conversion outside the 2–8-token products; product register counts fall from 98/106 to 78/74 | A register reduction is not measured speed. Try the preparation only where a producer can fuse it or enough products reuse it; include transient scratch and the original quantization order. [Calibration](experiments/qwen38-mtp-speed/README.md) |
| Qwen NVFP4 expert rows/warps | Rows8/warps4 improves isolated overlapping down products by 11.4–11.5% in speed ratio, but adaptive128K prefix is neutral and curated loses about 0.8–0.9% | Six exact schedules, actual routed working-set rotation, packed512-expert controls and unchanged per-slot arithmetic/PDL | Lower registers alone did not win. Real3/4-row launches enter the attempted selector; it remains unadopted. [NVFP4 scheduling](experiments/qwen38-nvfp4-scheduling/README.md) |
| Qwen GLU-producer down-Q8 export | Complete pair gains only 2.4–2.9% in speed ratio with overlapping routes; disjoint latency rises 0.64–1.02%, so no production fusion or model trial | Exact per-16-value producer export without an extra launch; stored-F32/Q8-byte controls and complete-pair graph accounting | A different consumer needs its own complete-pair and model gain, plus explicit planned scratch and captured pool-reuse lifetime proof. Registers fall 106→98 for GLU and 98→78 for down, with one new barrier; those counts alone are not speed evidence. [Producer study](experiments/qwen38-nvfp4-scheduling/README.md#producer-fused-down-input-q8) |
| Qwen BF16 vector kernel for multi-row verify | Slower than cuBLAS at that shape and two tokens exceeded the existing speculation near-tie bound | Its one-row product/reduction remains adopted for plain decode | New multi-row shape must pass the original quality bound and win end to end; do not widen the bound to accept a kernel. [TensorFold techniques](experiments/tensorfold-techniques/README.md) |
| Qwen cuBLASLt small-column products | HC products were neutral; router conversion plus Lt was slower; full-head gain was 1.3–1.6%, about 0.1–0.2% of verify | Bounded candidate search, original F32-to-BF16 rounding, full working-set rotation and actual F32/BF16 output contracts | A new product shape needs its own isolated and model gain. Direct BF16-weight/F32-input Lt was unsupported on the measured head/router shapes by the pinned library. Production stays unchanged. [Qwen tuning](experiments/qwen38-gemm-tuning/README.md) |
| MXFP8 vector input columns loaded one at a time | Lower registers did not produce the best small-output timing | Changed input/decoded-weight liveness without new quantization, scratch or launches | At columns 4 / rows 1 / warps 8, registers fall from 91 to 56, but the measured production transfer retains the original load order. A new consumer needs isolated and model evidence. [MXFP8 scheduling](experiments/qwen38-mxfp8-scheduling/README.md) |
| Qwen twelve-column QKV MXFP8 CTA size | Representative C4 verify latency −0.116% while controls move −2.417%; keep 16 warps | Existing staged two-row kernel, unchanged per-lane FMA/scale/XOR sums and PDL, actual twelve-column dispatch proof | The GB10 K2560/N10240 16→8-warps factor is inconclusive and rejected. Complete operand outputs, wave target rows/tokens and initialized states remain exact. This does not reject other shapes or tensor-core crossover. [Twelve-column screen](experiments/qwen38-mxfp8-twelve/README.md) |
| Qwen wave read alignment 1024 | Fixed-history C4 verify latency −0.851% with exact full verify rows/token histories; HTTP apparent rate +6.845% versus controls’ −14.693% wall movement; keep 2048 | Existing bounded cache/indexer padding, literal-history/geometry metadata and resolved/registered setting proof | The small phase benefit does not establish serving benefit. HTTP text differs between controls; this rung crosses no alignment boundary, and no cross-alignment logical-state/swap quality claim follows. [Read-alignment screen](experiments/qwen38-wave-read-align/README.md) |
| Qwen paired shared-expert MXFP8 plus SwiGLU | Complete C12/K2560/N640 operator latency −28.64%, but representative C4 verify latency +0.091% with controls moving −0.447%; no adoption | Exact independent lane sums/reductions, pinned fast unary epilogue, complete operator byte controls and bounded post-join fusion with both input concat trees paid | Keep the original chain. All native wave histories/full rows/initialized states match; the isolated gain does not establish a model or serving gain. Device/contract expansion was not qualified. [Paired shared-expert screen](experiments/qwen38-shared-mxfp8/README.md) |
| MXFP8 tensor-core crossover | All 24 three/four/five-column products and the C4 twelve-column K2560/N10240 shape are slower after charging F32 quantization, scale swizzle and GEMM; twelve columns +1.493% latency | Existing quantize/swizzle/GEMM wrappers, passing padded/tail controls and generated working sets larger than L2 | Measure pieces separately before attributing the slowdown. A new consumer must include every conversion/storage cost and pass original quality gates; the input rounding differs. Global threshold unchanged. [Depth and crossover](experiments/qwen38-depth-crossover/README.md) |
| Paired-prefix adaptive-depth estimator | Prefix 128K regresses 3.75–4.23%; curated gains only 0.62–1.76% | Shared recent-prefix prediction; independent policy traces and measured phase costs | Different verify widths need not give exact counterfactual rewards. A new estimator must win on both measured heads and preserve deterministic checkpoint choices. Prototype removed; telemetry retained. [Depth and crossover](experiments/qwen38-depth-crossover/README.md) |
| Confidence window that only shortens verify | No model gain; every draft pass has already run | Candidate confidence and deterministic stopping criterion | Avoid subsequent draft work itself. Compare against the unchanged whole-prefix draft and check state, repeats, swaps and rejection. |
| Incremental confidence stopping after at least two drafts | Actual early stopping remained neutral at 128K: prefix 45.00 and curated 45.56 tok/s versus fixed-depth-four controls 45.61 and 45.76 | Bounded continuation graphs and explicit draft-state catch-up | Short forced-rejection control passed; direct parity with the monolithic draft was not established. Any new consumer needs that proof plus a measured gain. Prototype removed. [Calibration](experiments/qwen38-mtp-speed/README.md) |
| ds4 direct-to-residual products | Different SoA weight layout and earlier route-weight multiplication around input quantization; not a drop-in raw GGUF product | Compact expert-major scheduling, paired gate/up preparation, fused activation, integer-dot/correction techniques | Keep raw artifact strides and current route weighting. Measure each piece before considering a layout change or extra weight replica. [ds4 study](experiments/ds4-study/README.md) |
| Current native compact IQ2 pair plus original suffix | Captured 4,096-row layer: 0.865911× Direct / 0.920092× Materialized resident speed; no full-model candidate | Full data-only aligned/raw inverse, complete original-map and suffix calibration, paid native-map adapter, full output/FP64 controls and actual compiler/static-link proof | Gate/up maximum errors are 1.19e−6 / 1.43e−6; D2S6 changes amplify summed maximum error to 0.001705 with NMSE 8.12e−9. Full D4 capture excludes activation preparation as the arithmetic cause; the matched standalone consumer is 0.880357× original rate, while native J64 also regresses. Remaining loader/launch/arithmetic factors need separate controls. Preparation and duplicate mapping remain paid in this compound arm. Operator errors establish no model-quality gate. [Captured IQ2 study](experiments/ds4-native-iq2/README.md), [prepared factor](experiments/ds4-native-iq2-prepared/README.md), [standalone consumer](experiments/ds4-native-iq2-consumer/README.md) |
| Native IQ2 J64 instead of selected J128 | Same captured pair: 0.950775× J128 resident rate, all complete output bytes equal | Full-byte matched inputs, actual O3 defining-TU runtime attributes and explicit source-override proof | J64 reduces registers 254→244 and dynamic shared bytes 57,856→48,384 but raises live work items 358→533. Occupancy was not measured. Keep J128; resource counts alone do not choose a faster schedule. The benchmark bridge is external, uses a truthful unofficial override and changes no default. [J64 control](experiments/ds4-native-iq2-j64/README.md) |
| ds4 FP8 KV/FP4 indexer caches | Different transforms/storage; native M3 keeps its F16 state contract | Scheduling, gathered-cell bounds and attention tiling independent of compression | The ds4 F32-storage control retains FP8/FP4-rounded values; it is not an unrounded oracle. Quantization-aware-training transforms exist in the official model too. Establish same-quality evidence before choosing a default; a quality sacrifice needs per-alias opt-in. [Precision clarification](experiments/ds4-study/README.md#cold-context-results) |
| ds4 token-tile attention arithmetic | Default-off native benchmark HCA transfer, with original numerical helpers/core verbatim | Other compressed extents and shapes need separate qualification | Charged first-4K/2K latency is 7.056/3.557 ms versus ordinary 89.030/46.560 ms. Fresh 8K ABBAs gain 17.88%/15.76%, with identical 128 IDs and exact own repeats. Late real 2K chunks gain about 2.5× at 32K and 3.9× near 128K; strict selected-row FP64 controls pass. Selection admits GB10 T2048/raw2304 with 256/1,024 compressed cells and T4096/raw4352 with 256, off by default; since 2026-10-03 also any chunk of 64 rows or more whose ring holds rows + 256 cells ([partial chunks](experiments/ds4-prefill-stages/README.md#partial-chunks-and-other-quant-types)). Two original-GGUF 32K repeats still fail step 249: oracle margin 2.616249 versus fixed 0.947 nats. Mixed eligible 128K PPL passes at −0.1435%, without reversing the greedy failure. The original core does not eliminate the prior failure; operator accuracy is not model quality. With output-A it is now serving's default on every prefill chunk of 64 rows or more, plans keyed by shape with the position set each run, never captured; HCA alone stays off. Under the owner's tie-aware rule (2026-10-03) step 249 is a tie flip ([tie-aware re-scoring](experiments/ds4-output-prefix/README.md#tie-aware-re-scoring-and-partial-chunks)). [Literal HCA study](experiments/ds4-hca-tokentile/README.md) |
| Qwen two-query shared-KV QSA union | All six real-input 32K/near128K cases regress: 0.524–0.668× native speed warm, 0.548–0.641× displaced, charging preparation; prototype removed | Completion-aware native operand capture, full native/separate byte controls, fixed FP64 probes, current-operand graph checks, bounded integer union records and charged prep/cache controls | Native and separate controls are byte-exact; shared NMSE 2.82e−11–1.64e−10 and fixed rounded-query FP64 probes pass. Median overlap is 1,147–1,339, union 2,763–2,956 versus about 2,051 cells per query; added masked MMA work is an inference from the source, not profiler attribution. QSA already has F32 weighted-value accumulation. No model trial or decode/MTP gain is claimed. [Real-QSA transfer](experiments/qwen38-qsa-pair/README.md) |
| Wider sparse attention everywhere | Disjoint D512 synthetic case regressed; faithful-floor all-wide 32K fails one oracle row | Query-union construction and semantic tile policy | Count-based HCA uses ordinary MMA uniformly; CSA/window retain sharing. Both 32K/128K fresh controls and matched PPL pass without context/oracle-dependent selection. Experimental all-narrow compact-tail 128K also failed, so it is not evidence for a blanket rollback. |
| DeepSeek grouped multi-matrix launches | Existing decode study found no gain | Shared input staging or an eligible adjacent-product preparation | Isolate useful shared work; retain the measured per-product path until an end-to-end comparison supports another choice. [DeepSeek decode](experiments/dsv4-decode/README.md) |
| Image batched QKV/gate-up products | Pinned single products were as fast or faster: 4.40 ms versus 3 × 1.47 for QKV, 8.52 versus 2 × 4.12 for gate/up | Shape-specific product tuning and shared inputs | A different batch or layout needs its own measured algorithm and model result. [Image study](experiments/qwen-image-native/README.md) |
| Image third KV attention stage | 3.43 versus 3.49 ms isolated, but 32.56 versus 32.58 seconds for generation; neutral | Pipeline-stage tuning with matched shared-memory occupancy | The 96 KiB stage is a measured isolated gain, not an end-to-end adoption. A new consumer must account for its shared-memory and occupancy cost. [Image study](experiments/qwen-image-native/README.md) |
| Image prefix cache read in place, alone | Full generation did not move measurably | Stable prefix views became the base of the adopted query-norm fusion | A concrete useful piece of a neutral change; keep combined and isolated outcomes distinct. [Image study](experiments/qwen-image-native/README.md) |

## Measurement order

The compact ordinary-expert experiment is complete and its measured
DeepSeek selection is retained. Qwen's library-product sweep was neutral;
its measured small-output MXFP8 schedules give a modest incremental gain.
The complete small-row tensor-core crossover is slower on every measured
shape, and the paired-prefix depth estimator regresses the default prefix
head. Both remain unadopted; the existing kernels and deterministic policy
stay in production. Measure an inventory candidate that can materially
help the open M3 speed or maximum
memory gate. Small unmeasured leads remain named here instead of becoming
automatic model defaults or an unbounded benchmark queue.

For each candidate, record the current plan and the exact missing consumer,
the fragment being transferred, its precision/layout requirements, isolated
timing, whole-model timing and peak memory, correctness/rejection controls,
and the resulting selection policy. Update this inventory and its source
report when the outcome changes.

M3.5 applies the same comparison to its new EXL3 and legacy families.
The owner requires every new family and quantization to adopt the applicable
optimizations selected by the Qwen and DeepSeek paths and to support optimized
batching (2026-10-04). Each bring-up records which techniques apply, where
its plan selects them, any shape or format limits, and its solo and batched
correctness, speed and memory evidence. A supported status includes that
batching qualification; a serial bring-up is an intermediate control.
Use the latest TensorFold at each applicable family, quantization or
optimization task under [reference comparisons](reference-comparisons.md),
including supported single-Spark paths in M3.5. M4's
[Kindling and TensorFold references](m4-references.md) add multi-node
techniques after their recipes and measured two-Spark behavior are
re-pinned; TensorFold refreshes per task as well as at entry. Their claimed
rankings are not local measurements.


### Qwen expert-major group cap — 2026-10-04

The private NVFP4/MTP C4 reduction from eight to four matching slots is
neutral/slower: 106.947 ms versus 106.667 / 106.060 ms original verify
bookends, +0.549% against their mean. All per-slot tokens, full verify rows
and initialized final state hashes match. Keep the production cap eight;
this is no measured register/occupancy result. See the
[screen](experiments/qwen38-four-request-waves/README.md#smaller-expert-sharing-groups--2026-10-04).
