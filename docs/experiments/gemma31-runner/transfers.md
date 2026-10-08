<!-- SPDX-FileCopyrightText: 2026 llmpalooza contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Gemma 31B optimization transfer audit

This task audits the [inventory](../../optimization-inventory.md) against the
approved dense 31B tensor and state contract. Availability is not adoption.
`Gemma4Runner::Choices` begins with `DeviceChoicesOf` and explicitly enables
only requested diagnostic norm/store/row policies. `PlanGemma4Chunk` plans
without generic upstream fusion. Ordinary 31B options request none of those
policies; `last_built_policy` is checked in real-model controls. No quality or
speed result from Qwen, DeepSeek, a synthetic primitive, or Gemma 26B qualifies 31B.

| Existing mechanism | Dense 31B eligibility and actual ordinary choice |
| --- | --- |
| Pinned device product selector and ordinary MMVQ/MMQ | Reused for the actual Q4_K/Q5_K/Q6_K operands. One-row and multirow products keep the pinned ordinary selector, with existing row padding and full readable-span checks. No exotic format conversion is introduced. |
| Input preparation shared across adjacent products; shared decode Q8_1 | Quantized dense gate/up and compatible attention/head inputs are potential consumers of the existing `shared_q8` builder. It stays off; ordinary products prepare their own inputs. Same logical input, exact conversion and consumer guards remain prerequisites for a later experiment. |
| DeepSeek compact pairs, prepared IQ2_XXS consumers and selected tile descriptions | Ineligible: dense 31B has no expert arrays, IQ2_XS/XXS/IQ3 or compact expert-major representation. Real operand byte captures remain a useful experimental method, not an applicable kernel choice. |
| Direct raw Q2_K D2R loader, expert-major tiles and pair SwiGLU/Q8 writer | Ineligible: no Q2_K/IQ2 routed products, no routing maps or expert list, and the activation is GeGLU. No DeepSeek compact scheduling, writer or launch attributes are transferred. |
| NVFP4 grouped GEMM/GEMV, MXFP8 cached conversion and fused activation quantization | Ineligible for Q4_K/Q5_K/Q6_K raw blocks. These Qwen kernels require block-scaled weights/activation representations and their own layout. No NVFP4/MXFP8/EXL3 conversion, companion backend or dispatch is selected. |
| Read expert weights once for several verify rows; ballot-based token/slot maps | Ineligible: the dense 31B profile has zero experts. `ReserveWeights` creates no expert slabs, shard padding, routing closure or expert-pair map. |
| Column-invariant VecQ joins and row-invariant float products | Existing implementations have bounded shape/type contracts and may be screened separately. `shared_q8` and `row_invariant` stay off. Ordinary waves use their actual joined product shape; this task proves exact same-shape replay, not equivalence to scalar arithmetic or optimized batching. |
| Whole-request float grouping and per-node MMVF selection | No draft/router head mixes exist in dense 31B. Generic F32 learned norms remain pointwise; the quantized target head is not a BF16 draft head. No vector-float override is selected. |
| Concurrent wave lanes and separate lane scratch | Independent segments can support a future lane experiment through the shared executor, but current Gemma planning does not assign lanes. This task retains the owning stream and reports `lane_steps=false`. |
| Pinned FlashAttention arm by D/GQA/query shape | Reused: D256/GQA2 local and D512/GQA8 global attention, now at 32 query heads and 16/4 KV heads. Solo local uses the pinned eligible vector arm; multirow local uses group2 MMA; global retains its group8 path. State/input/plan checks authenticate each independent segment. No sparse kernel is forced. |
| Selected sparse cells, sparse-query tile unions and QSA | Ineligible: Gemma uses its canonical dense global and sliding-window local masks, with no DeepSeek indexer, compressed HCA/CSA cells or Qwen QSA semantics. Neither similarity of D256/D512 nor a sparse primitive gain changes that selection. |
| Graph-owned causal/ring masks | Reused and selected by the ordinary native runner: fresh checked I32 positions produce per-segment F16 masks, shared across matching layers. Diagnostic host masks remain separately funded. No claim that device-mask staging closes model quality follows. |
| Per-head norm/rotation; DeepSeek fast Q head and Qwen QsaPrep | Foreign launchers are ineligible: Gemma learned head norms and D256/full256 or D512/full512 factor-aware NEOX rotation differ from the DeepSeek512/64-tail and QSA contracts. Ordinary separate norm, rotation and cache write remain selected. |
| Checked per-segment RoPE/cache-store fusion | Existing factor-aware implementation is conditionally available, with complete contiguous-view, alias/span and generation guards. `rope_store` stays off. Intermediate allocation elision is not claimed. |
| RMSNorm/weight product fusion | The existing generic RMSNorm → MUL mechanism is eligible at matching F32 learned-normalization shapes. `fuse_norms` stays off for ordinary 31B; a separately requested norm arm only calibrates native numerical movement. Its exactness alone does not justify adoption. |
| Norm/MUL/residual ADD and norm/MUL/RoPE | These additional pinned-reference families lack native checked planner/registry launch contracts. Reusing compiled upstream source is not sufficient without those contracts, numerical controls and paid full-model evidence. The independent 26B reference diagnosis does not identify their 31B effect. |
| Floating gate/up GeGLU; quantized GeGLU writer | Floating MMVF GeGLU fusion is ineligible for the approved quantized dense FFN weights. The explicit GeGLU/Q8 writer is not selected by ordinary products and needs a matched consumer experiment; the GELU-tanh primitive fallback remains. No SwiGLU substitution is allowed. |
| HC norm/expert sum, clustered reductions and next-weight prefetch | Ineligible: no hyper-connections, HC matrices, expert mixture or DeepSeek post-projection formula. Qwen's reduction widths/weight layout do not match Gemma's learned norms. |
| Qwen short convolution, DeltaNet recurrence and verify alpha/beta planes | Ineligible: dense 31B has no such recurrent state or operations. The shared lifetime/accounting helpers apply; recurrent arithmetic and accepted-row commit do not. |
| In-place state, growing extents, sparse spill and turn reuse | Reuse the existing Gemma ring/append state layout and shared `LiveState`, funding ledger, checkpoint completion guards, catalog reclaim and spill/restore paths. No independent state or scheduler is added. Local rings do not advertise arbitrary rollback. |
| Stable-address graphs and one request lease | Reused from the same runner resources, plan cache, graph runs and request cohort. Real controls require capture/replay and exact restored continuation. Graph state and host plan capacity remain charged. |
| Narrow frontier head after required state work | Existing `frontier_head` remains selected for frontier requests; all-head likelihood requests retain every requested row. Paid 8K counts its native intermediate chunk heads. Feature capture for a future assistant is not silently omitted or implemented here. |
| Joined draft blocks, selected draft heads, confidence/depth caps, context-copy drafts and portable vocabulary seeds | Not implemented for 31B in this task. The approved assistant artifact needs separate binding, feature-row, state, vocabulary and target-verification contracts. Qwen MTP/DeepSeek DSpark acceptance or head-rounding evidence does not transfer. |
| Greedy/top-k sampling shortcut | The engine exposes full logits; the 31B diagnostic chooses/records strict argmax itself. No 31B serving sampler/request/stop ownership is admitted. Existing 26B serving behavior remains restricted and checked. |
| Prefill chunk sizes and discarded speed experiments | Native 31B's bounded 128-row envelope stays explicit. The paid 8K reference screen varies supported ubatches independently, including 1024/2048. Earlier DeepSeek 8192 regression and Qwen draft-cap results are not 31B measurements. |

This covers existing adopted and rejected Qwen/DeepSeek mechanisms named in the
inventory, including reusable parts of rejected products, sparse-query pairing,
HC reduction variants and draft-head rounding experiments. Their numerical and
launch contracts remain separate. The ordinary plan's real policy records,
matched oracle screen and paid bookends belong in the task report; unsupported
transfers and any measured gap remain open work.
