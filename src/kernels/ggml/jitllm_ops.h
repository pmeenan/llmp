// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// jitLLM's own operations on GGML tensors, for the formats GGML has no type
// for (M3, Qwen3.8 Flash Next's ModelOpt checkpoint; the A/B that chose them
// is docs/experiments/qwen38-native/README.md):
//
//   jitllm.mxfp8.mul_mat_vec  y[n, t] = W[n, :] · x[:, t] for up to 8
//                             columns t, W in MXFP8 (E4M3 codes, one E8M0
//                             scale per 32 along k), each 32-element block's
//                             dot product in F32, then scaled, the blocks
//                             summed in F32; memory-bound decode products.
//   jitllm.mxfp8.dequant      W as BF16 [k, n], exactly (an E4M3 value times
//                             a power of two) wherever that is a BF16 value,
//                             for GGML's float products (cuBLAS) over wider
//                             batches. E8M0 0xFF and E4M3 0x7F/0xFF are NaN.
//   jitllm.nvfp4.get_rows     rows of a table in ModelOpt NVFP4, each row its
//                             v/2 code bytes (element 2i in the low nibble)
//                             then its v/16 E4M3 scales, times the table's
//                             global F32 scale, as F32 [v, ids]: Qwen3.8's
//                             n-gram embedding lookup.
//   jitllm.argmax             I32 [rows]: each F32 row's highest value's
//                             index, the lowest index among equals (the rule
//                             of execution/sampling.h's Greedy); a NaN never
//                             wins, and a row of NaN only gives 0 (the
//                             index feeds unchecked row lookups, so it is
//                             always in the row). DeepSeek's DSpark drafter
//                             chains its Markov head on it (GGML's argmax
//                             is not in jitLLM's subset).
//
// and, for Qwen3.8's prefill, fusions of the elementwise work the
// hyper-connections and the MoE output do at four streams' width, each the
// same arithmetic as the GGML nodes it replaces (qwen38_graph.cc), in the
// same order and with GGML's device flags (-use_fast_math), so that its
// result equals theirs bit for bit (tests/unit/qwen38_ops_test.cc):
//
//   jitllm.hc.combine   res[c, k, t] + out[c, t] · (2 · sigmoid(inj[k, t] / hc)):
//                       build_hc_combine's sigmoid, scales, repeat, mul
//                       and add;
//   jitllm.hc.norm      rms_norm over each stream's `width` elements, then
//                       times the norm weight, in F32 or rounded to BF16
//                       (the input of the mixer's products);
//   jitllm.hc.mix       build_hc_mix's gate and fold: the norm again (each
//                       stream's scale recomputed, as GGML's rms_norm
//                       reduces), times the weight and sigmoid(g), summed
//                       over the streams in order, times 1 / hc;
//   jitllm.moe.glu      silu(gate · s_gate[e]) · (up · s_up[e]): the
//                       experts' global scales and SwiGLU;
//   jitllm.moe.combine  Σ_i (down_i · s_down[e_i]) · w_i in expert order,
//                       plus shared · sigmoid(shared gate): the routed
//                       experts' weighted sum and the gated shared expert;
//   jitllm.bf16         F32 to BF16, rounded to nearest (as GGML converts
//                       cuBLAS operands);
//   jitllm.gemm.bf16    y[n, t] = W[k, n] · x[k, t] with BF16 W and x, F32
//                       out, through the lent cuBLAS handle: the call GGML's
//                       cuBLAS product makes for BF16 weights after
//                       converting F32 activations, so one conversion
//                       serves every product of an input.
//
// Those are the fused graph's reference form. Its fast form, the default
// (D-085: speed before bit exactness), adds operations below whose
// arithmetic is not GGML's nodes', each checked against an FP64 reference
// (tests/unit/qwen38_fast_test.cc): the MXFP8 products on tensor cores
// (jitllm.mxfp8.*), the hyper-connections' prep and mix (jitllm.hc.prep,
// .lo, .mix_bf16), routing (jitllm.moe.router), Gated DeltaNet's history
// (jitllm.gdn.history, and its conv and gated norm over BF16 rows and into
// MXFP8), and QSA's prep, selection and gate (jitllm.qsa.*). Several write
// more than one output into one byte blob, whose layout struct names each
// part's offset; the graph views each part as a tensor of its type.
//
// Each is a GGML_OP_CUSTOM node (ggml_custom_4d) whose function pointer
// names the operation; the function itself is never called (GGML's CPU
// backend never runs these graphs). The builders make the nodes; the plan
// (graph_plan.h) names the implementation from the kind; the checks here
// are what each implementation refuses on the host before anything is
// queued (D-086), and the launchers are in jitllm_ops.cu.
//
// Byte tensors are GGML_TYPE_I8: MXFP8 codes [k, n] and scales [k/32, n],
// NVFP4 table rows [v/2 + v/16, rows].

#ifndef JITLLM_KERNELS_GGML_JITLLM_OPS_H_
#define JITLLM_KERNELS_GGML_JITLLM_OPS_H_

#include <cstdint>
#include <expected>

#include "ggml.h"
#include "kernels/ggml/tensors.h"

namespace jitllm::kernels::ggml {

enum class JitllmOp : std::uint8_t {
  kNone,
  kMxfp8MulMatVec,
  kMxfp8Dequant,
  kNvfp4Rows,
  kArgmax,
  kHcCombine,
  kHcNorm,
  kHcMix,
  kMoeGlu,
  kMoeCombine,
  kBf16,
  kGemmBf16,
  kMoeRoute,
  kMoeQuantize,
  kMoeGemm,
  kMoeGluQuantize,
  kMoeCombineSorted,
  kMoeGemv,
  kGdnConv,
  kGdnNormGate,
  kMxfp8Quantize,
  kMxfp8Swizzle,
  kMxfp8Gemm,
  kHcPrep,
  kHcLo,
  kHcMixBf16,
  kMoeRouter,
  kGdnHistory,
  kQsaPrep,
  kQsaGateQuantize,
  kQsaPool,
  kQsaTopK,
  kQsaAttn,
  kQuantizeQ8,
  kVecQ,
  kDsv4Route,
  kDsv4Combine,
  kDsv4HcMix,
  kDsv4HcPre,
  kDsv4Compress,
  kGdnStep,
  kGdnGates,
  kDsv4LidTopK,
  kDsv4SparseMask,
  kDsv4WeightedReduce,
  kDsv4QHead,
  kDsv4OutA,
  kDsv4HcNormF16,
  kDsv4F16Copy,
  kQRows,
};

// The operation a GGML_OP_CUSTOM node names, or kNone.
JitllmOp JitllmOpOf(const ggml_tensor* node);
// The norms' epsilon, which their builders store after GGML's custom
// parameters.
float JitllmOpEps(const ggml_tensor* node);
// The routed-expert operations' integer parameters, stored in the same
// place: index 0 to 7.
std::int32_t JitllmOpInt(const ggml_tensor* node, int index);
float JitllmOpFloat(const ggml_tensor* node, int index);

// The most columns a graph's own jitllm.mxfp8.mul_mat_vec takes: past them
// a chunk's MXFP8 products run on tensor cores (qwen38_graph.cc).
inline constexpr std::int64_t kMxfp8VecColumns = 8;
// The most columns the kernel takes: a Qwen3.8 wave's shared products
// (engine/qwen38_wave_plan.h) join up to four requests' verify rows. Each
// column's arithmetic is the eight-column kernel's.
inline constexpr std::int64_t kMxfp8VecWaveColumns = 16;

// Builders. `codes` I8 [k, n], `scales` I8 [k / 32, n], `x` F32 [k, t].
ggml_tensor* Mxfp8MulMatVec(ggml_context* context, ggml_tensor* codes, ggml_tensor* scales,
                            ggml_tensor* x);
ggml_tensor* Mxfp8Dequant(ggml_context* context, ggml_tensor* codes, ggml_tensor* scales);
// `table` I8 [values / 2 + values / 16, rows], `ids` I32 [n], `scale` F32
// [1]: F32 [values, n].
ggml_tensor* Nvfp4Rows(ggml_context* context, ggml_tensor* table, ggml_tensor* ids,
                       ggml_tensor* scale, std::int64_t values);
// jitllm.qrows.get_rows: rows of a table in one of GGML's 32-value block
// types (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ4_NL) as F32, each value as GGML's
// dequantization gives it (dequantize.cuh; IQ4_NL's kvalues_iq4nl times
// its block's scale): a GGUF checkpoint's n-gram table, whose 160-value
// rows are not the whole 256-value super-blocks GGML's own get_rows
// dequantizes these i-quants in (getrows.cu k_get_rows_kq). `table` [values,
// rows] of such a type (packed), `ids` I32 [n]: F32 [values, n]. Rows of a
// multiple of 32 values, at most 1,024; an id outside the table gives NaN
// rather than a read out of bounds.
ggml_tensor* QRows(ggml_context* context, ggml_tensor* table, ggml_tensor* ids);
// Whether jitllm.qrows.get_rows takes a table of `type`.
bool QRowsType(ggml_type type);
// `x` F32 [n, rows]: I32 [rows]; with `probability`, I32 [2 · rows]: the
// indices, then each row's softmax probability of its highest value (F32
// bits; 0 for a row of NaN), the drafter's confidence that TensorFold's
// adaptive window reads (docs/experiments/tensorfold-techniques/).
ggml_tensor* Argmax(ggml_context* context, ggml_tensor* x, bool probability = false);

// The fusions. `res` F32 [width, hc, t], `out` F32 [width, t], `inject` F32
// [hc, t]: F32 [width, hc, t].
ggml_tensor* HcCombine(ggml_context* context, ggml_tensor* res, ggml_tensor* out,
                       ggml_tensor* inject);
// `x` F32 [width, hc, t], `weight` F32 [width · hc]: [width · hc, t] of
// `type` (F32 or BF16).
ggml_tensor* HcNorm(ggml_context* context, ggml_tensor* x, ggml_tensor* weight, float eps,
                    ggml_type type);
// `x` and `weight` as HcNorm's, `gate` F32 [width · hc, t] (before its
// sigmoid): F32 [width, t].
ggml_tensor* HcMix(ggml_context* context, ggml_tensor* x, ggml_tensor* weight, ggml_tensor* gate,
                   float eps);
// `gate` and `up` F32 [n, used, t] (the experts' raw products), `ids` I32
// [used, t] (rows may be strided), `gate_scale` and `up_scale` F32
// [experts]: F32 [n, used, t].
ggml_tensor* MoeGlu(ggml_context* context, ggml_tensor* gate, ggml_tensor* up, ggml_tensor* ids,
                    ggml_tensor* gate_scale, ggml_tensor* up_scale);
// `down` F32 [width, used, t], `ids` as MoeGlu's, `down_scale` F32
// [experts] (or null: experts without global scales, a GGUF checkpoint's,
// whose sum is GGML's unfused nodes' without the scale's product),
// `weights` F32 [1, used, t], `shared` F32 [width, t], `shared_gate` F32
// [1, t] (before its sigmoid): F32 [width, t].
ggml_tensor* MoeCombine(ggml_context* context, ggml_tensor* down, ggml_tensor* ids,
                        ggml_tensor* down_scale, ggml_tensor* weights, ggml_tensor* shared,
                        ggml_tensor* shared_gate);
// `x` F32: BF16 of its shape.
ggml_tensor* ToBf16(ggml_context* context, ggml_tensor* x);
// `weights` BF16 [k, n], `x` BF16 [k, t]: `type` (F32, or BF16 for the fast
// path's hyper-connection gate) [n, t].
ggml_tensor* GemmBf16(ggml_context* context, ggml_tensor* weights, ggml_tensor* x,
                      ggml_type type = GGML_TYPE_F32);
// The same product, which up to kGemvBf16FastColumns columns (and k at most
// kGemvBf16MaxK) runs jitLLM's BF16 vector kernel instead of cuBLAS (the
// fast graph's hyper-connection products; other sums' order than cuBLAS's).
// The kernel itself takes up to kGemvBf16Columns.
inline constexpr std::int64_t kGemvBf16Columns = 8;
// The columns up to which GemvBf16 nodes take the vector kernel: one (a
// decode step). At a 3-row verify its down product read x 3 times a block
// and took 4.63 ms a verify against cuBLAS's 3.1 (nsys, spark-b,
// 2026-09-28), so wider nodes run cuBLAS.
inline constexpr std::int64_t kGemvBf16FastColumns = 1;
inline constexpr std::int64_t kGemvBf16MaxK = 20480;
ggml_tensor* GemvBf16(ggml_context* context, ggml_tensor* weights, ggml_tensor* x,
                      ggml_type type = GGML_TYPE_F32);
// Whether a jitllm.gemm.bf16 node was built by GemvBf16.
bool IsGemvBf16(const ggml_tensor* node);

// The host checks: operands bound, typed and shaped as above, packed where
// the kernels read them with vector loads (codes and x rows 16-byte
// aligned), extents within the kernels' 32-bit indexing, and outputs
// disjoint from their operands. The row lookup's ids are not read here:
// the caller builds them within the table (model/qwen38.h's n-gram hash
// over offsets and sizes the binding checked); the kernel writes NaN for
// an id outside the table rather than read out of bounds.
std::expected<void, KernelFailure> CheckMxfp8MulMatVec(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckMxfp8Dequant(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckNvfp4Rows(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckQRows(const ggml_tensor* node);
// Packed F32 rows of at most 2^31 - 1 values into a packed I32 row of one
// index a row, at most 65,535 rows.
std::expected<void, KernelFailure> CheckArgmax(const ggml_tensor* node);
// The fusions' checks: every operand bound, typed and shaped as its
// builder's, packed (the expert ids may have a longer row stride), 16-byte
// aligned where a kernel loads four floats, within 32-bit grids, and the
// output disjoint from every operand. The expert ids are not read here: the
// kernels write NaN for an id outside the scales rather than read out of
// bounds. The norms take streams of at least 1,024 elements (GGML's
// rms_norm reduces those over 1,024 threads, which the kernels repeat).
std::expected<void, KernelFailure> CheckHcCombine(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckHcNorm(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckHcMix(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckMoeGlu(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckMoeCombine(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckBf16(const ggml_tensor* node);
// And the product's: cuBLAS's int extents, and both operands' rows k
// elements apart (packed).
std::expected<void, KernelFailure> CheckGemmBf16(const ggml_tensor* node);

// jitllm.gated_delta_net.columns: a second implementation of GGML's
// gated_delta_net node (the GGML implementation is ggml.gated_delta_net,
// ops_ext.h), the same arithmetic per value column as upstream's kernel at
// 128-wide heads, with several columns a warp (jitllm_fused.cu), for one
// sequence, a scalar gate and no state snapshots. Its checks are GGML's
// (validate_ext.h CheckGatedDeltaNet) and those restrictions;
// GatedDeltaNetColumnsFits says whether the node has that shape (the plan's
// structural choice).
bool GatedDeltaNetColumnsFits(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckGatedDeltaNetColumns(const ggml_tensor* node);
// jitllm.gated_delta_net.lanes: the same recurrence with each value column
// split over 8 lanes of 16 rows, for prefill (more than
// kGatedDeltaNetLanesTokens tokens; the plan's structural choice): fewer
// shuffles, its F32 sums in another order than upstream's, so not bit for
// bit (NMSE against FP64 as upstream's; tests/unit/qwen38_fused_test.cc).
// Its checks are the columns kernel's, and q, k, v and the state 16-byte
// aligned at 16-byte strides for its vector loads.
inline constexpr std::int64_t kGatedDeltaNetLanesTokens = 16;
bool GatedDeltaNetLanesFits(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckGatedDeltaNetLanes(const ggml_tensor* node);

// And two more fusions of Qwen3.8's Gated DeltaNet nodes, the same
// arithmetic in the same order:
//   jitllm.gdn.conv       the causal convolution over the conv state's
//                         history then the chunk's rows (ssm_conv), its
//                         silu, and the query and key heads' L2 norm
//                         (rms_norm with eps / d, times 1 / sqrt(d)), read
//                         straight from the rows (no transposed copy or
//                         concatenation);
//   jitllm.gdn.norm_gate  each head's rms_norm times the norm weight times
//                         sigmoid(z), in F32 or rounded to BF16 (the output
//                         projection's input), or (the fast graph's)
//                         quantized to MXFP8 for the tensor-core product
//                         (mxfp8_cutlass.h RowsLayout, as
//                         jitllm.mxfp8.quantize quantizes).
// The fast graph's QKV and z rows may be BF16 (its tensor-core products'
// output), and its convolution history is stored by
//   jitllm.gdn.history    the chunk's last k - 1 rows transposed into the
//                         conv state's layout, in F32.
// `x` F32 or BF16 [channels, t] (the QKV rows), `history` F32 [(k - 1) ·
// channels] (the conv state: tap j of channel c at c · (k - 1) + j),
// `weight` F32 [k, channels], `qk_channels` the leading channels (query and
// key heads of `head` values) that are normalized: F32 [channels, t].
ggml_tensor* GdnConv(ggml_context* context, ggml_tensor* x, ggml_tensor* history,
                     ggml_tensor* weight, std::int64_t qk_channels, std::int64_t head, float eps,
                     float scale);
// `o` F32 [d, heads, t] (packed), `weight` F32 [d], `z` F32 or BF16 [d ·
// heads, t]: [d · heads, t] of `type` (F32 or BF16), or with I8 the MXFP8
// rows' bytes.
ggml_tensor* GdnNormGate(ggml_context* context, ggml_tensor* o, ggml_tensor* weight, ggml_tensor* z,
                         float eps, ggml_type type);
// `x` F32 or BF16 [channels, t], t at least k - 1: F32 [(k - 1) · channels,
// 1]; or, given the old `history` (the conv state, F32 [(k - 1) ·
// channels]), any t: the last k - 1 of its taps followed by the rows (the
// fast graph's decode, whose chunks are shorter than the history).
ggml_tensor* GdnHistory(ggml_context* context, ggml_tensor* x, std::int64_t taps,
                        ggml_tensor* history = nullptr);
std::expected<void, KernelFailure> CheckGdnConv(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckGdnNormGate(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckGdnHistory(const ggml_tensor* node);
//   jitllm.gdn.gates     Qwen verify's two pointwise chains, with the
//                        original F32 rounding: sigmoid(beta) and
//                        softplus(alpha + dt_bias) * ssm_a.
// `alpha`, `beta` F32 [48, t], `dt_bias`, `ssm_a` F32 [48], 1 <= t <= 16:
// F32 [48, t, 2], gate in the first plane and beta in the second.
ggml_tensor* GdnGates(ggml_context* context, ggml_tensor* alpha, ggml_tensor* beta,
                      ggml_tensor* dt_bias, ggml_tensor* ssm_a);
std::expected<void, KernelFailure> CheckGdnGates(const ggml_tensor* node);
//   jitllm.gdn.step      the fast graph's decode form of the gated delta rule
//                        (up to kGatedDeltaNetLanesTokens tokens): the
//                        jitllm.gated_delta_net.columns recurrence, the same
//                        arithmetic, but the new state written over `state`
//                        in place, so decode neither writes the state into
//                        the node's output nor copies it back with set_rows
//                        (TensorFold's double-buffered state, taken as an
//                        in-place update: half the state traffic); without
//                        `write_state` (a speculative verify's, whose kept
//                        rows' state the commit writes) the state is read
//                        and not written at all.
// `q`, `k` F32 [128, qk heads, t] and `v` F32 [128, heads, t] (views with
// contiguous rows), `g` and `beta` F32 [1, heads, t], `state` F32 [128, 128,
// heads] (packed, written): F32 [128, heads, t], the attention output.
ggml_tensor* GdnStep(ggml_context* context, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                     ggml_tensor* g, ggml_tensor* beta, ggml_tensor* state,
                     bool write_state = true);
std::expected<void, KernelFailure> CheckGdnStep(const ggml_tensor* node);
class LaunchContext;
std::expected<void, KernelFailure> RunGdnGates(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunGdnStep(LaunchContext& launch, ggml_tensor* node);

// The routed experts over the CUTLASS layout (moe_layout.h), for
// Qwen3.8's prefill and decode (jitllm_moe.cu; the grouped GEMM is
// CUTLASS's, moe_cutlass.h):
//
//   jitllm.moe.route          the routing's ids sorted by expert: offsets,
//                             each slot's row, each row's token and expert
//                             (moe_layout.h RouteLayout), deterministic
//                             (slots in token order within an expert);
//   jitllm.moe.quantize       each token's activations quantized once to
//                             NVFP4 as GGML's MMQ quantizes them
//                             (quantize_mmq_nvfp4: a row scale amax / 2688,
//                             E4M3 block scales searched as upstream does),
//                             written to the token's sorted rows
//                             (moe_layout.h QuantLayout);
//   jitllm.moe.gemm           CUTLASS's block-scaled grouped GEMM over the
//                             sorted rows, BF16 out;
//   jitllm.moe.glu_quantize   each sorted row's gate and up products times
//                             its row scale and the experts' global scales,
//                             SwiGLU (upstream's silu), quantized again for
//                             the down projection;
//   jitllm.moe.combine_sorted each token's experts' down products (times
//                             their row and global scales and routing
//                             weights) summed in expert order, plus the
//                             gated shared expert: jitllm.moe.combine's
//                             arithmetic over the sorted rows;
//   jitllm.moe.gemv           up to 8 tokens' routed products straight from
//                             the layout, the F32 activations quantized to
//                             8 bits as MMVQ quantizes them, F32 out as
//                             mul_mat_id's (decode's); in its SwiGLU form
//                             (MoeGemvSwiglu) the gate and up rows together,
//                             out jitllm.moe.glu's result.
//
// Each writes and reads the layouts moe_layout.h describes; the builders
// store the extents (experts, experts used, tokens, n, k and offsets) in
// the nodes' parameters, and the checks hold every operand to them.
ggml_tensor* MoeRoute(ggml_context* context, ggml_tensor* ids, std::int64_t experts);
ggml_tensor* MoeQuantize(ggml_context* context, ggml_tensor* x, ggml_tensor* route);
ggml_tensor* MoeGemm(ggml_context* context, ggml_tensor* a, ggml_tensor* route,
                     ggml_tensor* weights, std::int64_t n, std::uint64_t codes_offset,
                     std::uint64_t scales_offset);
ggml_tensor* MoeGluQuantize(ggml_context* context, ggml_tensor* d, ggml_tensor* a,
                            ggml_tensor* route, ggml_tensor* gate_scale, ggml_tensor* up_scale);
ggml_tensor* MoeCombineSorted(ggml_context* context, ggml_tensor* d, ggml_tensor* a,
                              ggml_tensor* route, ggml_tensor* down_scale, ggml_tensor* weights,
                              ggml_tensor* shared, ggml_tensor* shared_gate);
ggml_tensor* MoeGemv(ggml_context* context, ggml_tensor* weights, ggml_tensor* x, ggml_tensor* ids,
                     std::int64_t n, std::int64_t row0, std::int64_t rows,
                     std::uint64_t codes_offset, std::uint64_t scales_offset);
// jitllm.moe.gemv's SwiGLU form over a block of 2 · f rows (gate rows,
// then up rows): silu(gate · gate_scale[e]) · (up · up_scale[e]), F32
// [f, used, t].
ggml_tensor* MoeGemvSwiglu(ggml_context* context, ggml_tensor* weights, ggml_tensor* x,
                           ggml_tensor* ids, std::int64_t f, ggml_tensor* gate_scale,
                           ggml_tensor* up_scale, std::uint64_t codes_offset,
                           std::uint64_t scales_offset);
// Whether a jitllm.moe.gemv node is the SwiGLU form.
bool IsMoeGemvSwiglu(const ggml_tensor* node);
// The most tokens a graph's own jitllm.moe.gemv takes (past them, the
// grouped GEMM), and the most the kernel takes: a wave's shared products.
inline constexpr std::int64_t kMoeGemvTokens = 8;
inline constexpr std::int64_t kMoeGemvWaveTokens = 16;

std::expected<void, KernelFailure> CheckMoeRoute(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckMoeQuantize(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckMoeGemm(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckMoeGluQuantize(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckMoeCombineSorted(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckMoeGemv(const ggml_tensor* node);

// The launchers (CUDA builds, jitllm_ops.cu and jitllm_fused.cu): the
// check, then one kernel (or one cuBLAS call) on the context's stream. None
// draws scratch; the product needs the context's cuBLAS handle and writes
// its workspace.
class LaunchContext;
std::expected<void, KernelFailure> RunMxfp8MulMatVec(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunMxfp8Dequant(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunNvfp4Rows(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunQRows(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunArgmax(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunHcCombine(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunHcNorm(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunHcMix(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunMoeGlu(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunMoeCombine(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunBf16(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunGemmBf16(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunGatedDeltaNetColumns(LaunchContext& launch,
                                                           ggml_tensor* node);
std::expected<void, KernelFailure> RunGatedDeltaNetLanes(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunGdnConv(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunGdnNormGate(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunGdnHistory(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunMoeRoute(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunMoeQuantize(LaunchContext& launch, ggml_tensor* node);
// The grouped GEMM draws its arguments and CUTLASS's workspace from the
// pool: PlanMoeGemm's bytes.
std::expected<std::uint64_t, KernelFailure> PlanMoeGemm(const LaunchContext& launch,
                                                        const ggml_tensor* node);
std::expected<void, KernelFailure> RunMoeGemm(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunMoeGluQuantize(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunMoeCombineSorted(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunMoeGemv(LaunchContext& launch, ggml_tensor* node);

// Qwen3.8's fast path (the graph's default, qwen38_graph.h; D-085: speed
// before bit exactness): the MXFP8 products on tensor cores (CUTLASS's
// block-scaled GEMM, mxfp8_cutlass.h), their activations quantized to MXFP8
// as Mia's vLLM quantizes the checkpoint's MXFP8 linears' (the oracle,
// docs/experiments/qwen38-native/README.md):
//
//   jitllm.mxfp8.quantize   F32 or BF16 rows [k, t] to MXFP8: each 32-value
//                           block's E8M0 scale 2^ceil(log2(amax / 448)) (so
//                           no value saturates), its values divided by it
//                           and rounded to the nearest E4M3; the codes and
//                           swizzled scales laid out as mxfp8_cutlass.h's
//                           RowsLayout;
//   jitllm.mxfp8.swizzle    a weight's E8M0 scales [k / 32, n] (the
//                           artifact's) into the product's swizzled layout
//                           (SwizzledScaleBytes), each chunk: 1/32 of the
//                           weight's bytes;
//   jitllm.mxfp8.gemm       y[n, t] = W x from the two, F32 or BF16 out.
//
// `x` F32 or BF16 [k, t], k a multiple of 128, its rows at a 16-byte
// aligned stride: I8 [RowsLayout{k, t}.bytes()].
ggml_tensor* Mxfp8Quantize(ggml_context* context, ggml_tensor* x);
// `scales` I8 [k / 32, n]: I8 [SwizzledScaleBytes(n, k)].
ggml_tensor* Mxfp8Swizzle(ggml_context* context, ggml_tensor* scales);
// `a` the MXFP8 rows of [k, t] (jitllm.mxfp8.quantize's, or a fusion's
// ending in the same quantization: RowsLayout's bytes), `codes` I8 [k, n],
// `scales` a jitllm.mxfp8.swizzle node of the weight's scales: `type` (F32
// or BF16) [n, t].
ggml_tensor* Mxfp8Gemm(ggml_context* context, ggml_tensor* a, ggml_tensor* codes,
                       ggml_tensor* scales, ggml_type type, std::int64_t t);
// The checks: operands bound, typed and shaped as the builders', packed
// (the quantization's input rows at any 16-byte aligned stride of at least
// a row), 16-byte aligned, within the kernels' 32-bit grids, the output
// disjoint from every operand; the product's operands the nodes of those
// builders with its extents.
std::expected<void, KernelFailure> CheckMxfp8Quantize(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckMxfp8Swizzle(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckMxfp8Gemm(const ggml_tensor* node);
std::expected<void, KernelFailure> RunMxfp8Quantize(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunMxfp8Swizzle(LaunchContext& launch, ggml_tensor* node);
// The product runs on a compute capability 12.1 device only, and draws
// CUTLASS's workspace from the pool: PlanMxfp8Gemm's bytes.
std::expected<std::uint64_t, KernelFailure> PlanMxfp8Gemm(const LaunchContext& launch,
                                                          const ggml_tensor* node);
std::expected<void, KernelFailure> RunMxfp8Gemm(LaunchContext& launch, ggml_tensor* node);

// And the fast path's hyper-connections, which read and write the four
// streams once a mix instead of three times (the reference form's
// jitllm.hc.combine, jitllm.hc.norm and jitllm.hc.mix):
//
//   jitllm.hc.prep      a token a block: optionally the previous block's
//                       output combined into the streams first (as
//                       jitllm.hc.combine), then each stream's rms_norm
//                       times the norm weight, stored in BF16 for the
//                       mixer's products, and, given an inject weight
//                       (BF16 [width · hc, hc]), the next combine's logits
//                       from the normalized streams in F32: a blob of the
//                       new streams, the normalized streams and the logits
//                       (HcPrepLayout);
//   jitllm.hc.lo        the mixer's rank-wide activation, silu(lo / hc), in
//                       BF16 for its up product;
//   jitllm.hc.mix_bf16  (1 / hc) Σ_k xn_k · sigmoid(g_k) over the BF16
//                       normalized streams and the up product's BF16 logits.
struct HcPrepLayout {
  std::int64_t width = 0;
  std::int64_t hc = 0;
  std::int64_t t = 0;
  bool combine = false;  // the blob starts with the combined streams
  bool inject = false;   // and ends with the logits
  static constexpr std::uint64_t Align(std::uint64_t b) { return (b + 255) / 256 * 256; }
  std::uint64_t streams_bytes() const {
    return combine ? static_cast<std::uint64_t>(width * hc * t) * sizeof(float) : 0;
  }
  static constexpr std::uint64_t streams() { return 0; }
  std::uint64_t normed() const { return Align(streams_bytes()); }
  std::uint64_t logits() const {
    return Align(normed() + (static_cast<std::uint64_t>(width * hc * t) * 2));
  }
  std::uint64_t bytes() const {
    return logits() + (inject ? static_cast<std::uint64_t>(hc * t) * sizeof(float) : 0);
  }
};
// `x` F32 [width, hc, t] (packed), `norm` F32 [width · hc], `inject` BF16
// [width · hc, hc] or null; with `out` F32 [width, t] and `logits` F32 [hc,
// t] (packed rows), the combine first. I8 [HcPrepLayout.bytes()].
ggml_tensor* HcPrep(ggml_context* context, ggml_tensor* x, ggml_tensor* norm, ggml_tensor* inject,
                    ggml_tensor* out, ggml_tensor* logits, float eps);
// `lo` F32 [rank, t], `hc` the streams: BF16 [rank, t].
ggml_tensor* HcLo(ggml_context* context, ggml_tensor* lo, std::int64_t hc);
// `normed` and `gate` BF16 [width · hc, t] (packed rows): F32 [width, t];
// or, with `mxfp8`, a blob of that F32 output, its MXFP8 quantization (as
// jitllm.mxfp8.quantize's, for the products that read it) and, with
// `bf16`, it rounded to BF16 (HcMixLayout).
struct HcMixLayout {
  std::int64_t width = 0;
  std::int64_t t = 0;
  bool bf16 = false;
  static constexpr std::uint64_t Align(std::uint64_t b) { return (b + 255) / 256 * 256; }
  static constexpr std::uint64_t mixed() { return 0; }
  std::uint64_t quantized() const {
    return Align(static_cast<std::uint64_t>(width * t) * sizeof(float));
  }
  std::uint64_t quantized_bytes() const;
  std::uint64_t rounded() const { return Align(quantized() + quantized_bytes()); }
  std::uint64_t bytes() const {
    return rounded() + (bf16 ? static_cast<std::uint64_t>(width * t) * 2 : 0);
  }
};
ggml_tensor* HcMixBf16(ggml_context* context, ggml_tensor* normed, ggml_tensor* gate,
                       std::int64_t hc, bool mxfp8 = false, bool bf16 = false);
std::expected<void, KernelFailure> CheckHcPrep(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckHcLo(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckHcMixBf16(const ggml_tensor* node);
std::expected<void, KernelFailure> RunHcPrep(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunHcLo(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunHcMixBf16(LaunchContext& launch, ggml_tensor* node);

// And the fast path's routing:
//
//   jitllm.moe.router   a token a warp: the router logits' softmax, the
//                       `used` most probable experts (descending, the lower
//                       index first among equals) and their probabilities
//                       renormalized (their sum clamped below at 2^-14, as
//                       build_moe_ffn clamps it), and the shared expert's gate
//                       logit (the gate row · x): a blob of I32 ids [used,
//                       t], then F32 weights [used, t], then F32 gate logits
//                       [t] (MoeRouterLayout).
struct MoeRouterLayout {
  std::int64_t used = 0;
  std::int64_t t = 0;
  static constexpr std::uint64_t ids() { return 0; }
  std::uint64_t weights() const { return static_cast<std::uint64_t>(used * t) * 4; }
  std::uint64_t gate() const { return static_cast<std::uint64_t>(2 * used * t) * 4; }
  std::int64_t ints() const { return (2 * used * t) + t; }
};
// `logits` F32 [experts, t], `x` F32 [width, t], `gate_row` BF16 [width]
// (the ModelOpt checkpoint's) or F32 [width] (a GGUF checkpoint's): I32
// [MoeRouterLayout.ints()].
ggml_tensor* MoeRouter(ggml_context* context, ggml_tensor* logits, ggml_tensor* x,
                       ggml_tensor* gate_row, std::int64_t used);
std::expected<void, KernelFailure> CheckMoeRouter(const ggml_tensor* node);
std::expected<void, KernelFailure> RunMoeRouter(LaunchContext& launch, ggml_tensor* node);

// And QSA's attention inputs and output:
//
//   jitllm.qsa.prep            a head of a token a warp: its rms_norm times
//                              the norm weight, then the text-only
//                              interleaved mrope (every section at the
//                              token's position: NEOX pairs (i, i + 32) of
//                              the first 64 dimensions, theta = pos ·
//                              theta_scale^i, as GGML's rope_multi computes
//                              it), packed [d, heads, t];
//   jitllm.qsa.gate_quantize   the attention's output times sigmoid of its
//                              gate (the second half of each head's query
//                              projection), quantized to MXFP8 for the output
//                              projection (RowsLayout).
// `x` F32 [.., t] whose heads of d values sit `stride` values apart from
// its row's start, `weight` F32 [d], `positions` I32 [4 · t] (the first t
// read): F32 [d, heads, t]. d a multiple of 32, at most 512; 64 rotated
// dimensions.
ggml_tensor* QsaPrep(ggml_context* context, ggml_tensor* x, ggml_tensor* weight,
                     ggml_tensor* positions, std::int64_t d, std::int64_t heads,
                     std::int64_t stride, float eps, float theta_scale);
// `attn` F32 [d · heads, t] (packed), `q_full` F32 [2 · d · heads, t] (each
// head's query then its gate): I8 [RowsLayout{d · heads, t}.bytes()].
ggml_tensor* QsaGateQuantize(ggml_context* context, ggml_tensor* attn, ggml_tensor* q_full,
                             std::int64_t d);
std::expected<void, KernelFailure> CheckQsaPrep(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckQsaGateQuantize(const ggml_tensor* node);
std::expected<void, KernelFailure> RunQsaPrep(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunQsaGateQuantize(LaunchContext& launch, ggml_tensor* node);

// And QSA at any depth (qsa_sparse.cu; the fast graph's, qwen38_graph.h),
// which never builds a tensor of every cell by every row:
//
//   jitllm.qsa.pool   the indexer's block keys, cached: for each block of
//                     `ratio` cells that the chunk completes (all of its
//                     cells at or before the chunk's last position, not
//                     before its first), its raw keys summed in order and
//                     scaled by 1 / ratio, then jitllm.qsa.prep's norm and
//                     rotation at the block's first position, rounded to
//                     BF16 and written in place into the state's block keys
//                     (a warp a block). Its output is a marker the
//                     selection reads, so it runs first.
//   jitllm.qsa.topk   build_qsa_top_k's selection as a list of cells: each
//                     complete block's score is its heads' relu scores
//                     (the query rounded to BF16 times the block key, F32
//                     sums; a tensor-core product past kQsaTopKVecRows
//                     rows) summed in head order; the token's own
//                     incomplete block is always kept (build_qsa_top_k's
//                     1e9 bias) and later cells are never visible. The
//                     `width` best visible cells are kept, the lower cell
//                     first among equals (where GGML's top_k keeps any, RE-031),
//                     deterministically at any depth: a byte-wise radix
//                     select for the width-th largest order-preserving key
//                     within tiles of kQsaTopKTile blocks, the tiles'
//                     candidates then selected again (TensorFold #93's
//                     technique). Out: each token's kept cells in
//                     ascending order, then -1 to the row's end.
//   jitllm.qsa.attn   attention over those cells alone: a warp a token's
//                     KV head (its query heads as one tensor-core tile) and
//                     a share of its cells, which it gathers 16 at a time
//                     (K and V double-buffered), with the online softmax;
//                     shares of one token combined in order. F16 products
//                     with F32 sums and softmax, the query scaled before it
//                     is rounded, as GGML's MMA flash attention computes.
//
// `raw` F32 [d, cells] (the raw key cache, as its set_rows node),
// `blocks` BF16 [d, cells / ratio] (the block keys, written in place),
// `weight` F32 [d], `positions` I32 [4 · t] (the first t read): I32 [1]. d
// a multiple of 32, at most 512, 64 rotated dimensions.
ggml_tensor* QsaPool(ggml_context* context, ggml_tensor* raw, ggml_tensor* blocks,
                     ggml_tensor* weight, ggml_tensor* positions, std::int64_t ratio, float eps,
                     float theta_scale);
// `q` F32 [128, 4, t] (jitllm.qsa.prep's indexer queries), `blocks` BF16
// [128, at least n_blocks] (the block keys), `positions` as the pool's,
// `pool` its jitllm.qsa.pool node: I32 [QsaTopKRow(width), t]. Draws scratch
// (PlanQsaTopK). At most kQsaTopKMaxTiles tiles.
inline constexpr std::int64_t kQsaTopKVecRows = 16;
inline constexpr std::int64_t kQsaTopKTile = 8192;
// (Tiles to 262,144 cells at a ratio of 4, Qwen3.8's configured maximum.)
inline constexpr std::int64_t kQsaTopKMaxTiles = 8;
// The candidates the second selection holds: 256 threads, 32 each.
inline constexpr std::int64_t kQsaTopKCandidates = 8192;
inline constexpr std::int64_t kQsaIndexDim = 128;
inline constexpr std::int64_t kQsaIndexHeads = 4;
inline constexpr std::int64_t kQsaAttnHead = 256;
// A token's row of kept cells: the width rounded up to whole 16-cell
// gathers.
constexpr std::int64_t QsaTopKRow(std::int64_t width) { return (width + 15) / 16 * 16; }
// The selection's scratch: each token's block keys (U32 [n_blocks, t]), the
// queries in BF16 ([128, 4, t]), and past one tile each tile's candidates
// (U32 key and block pairs, `candidates` a tile) and their counts.
struct QsaTopKLayout {
  std::int64_t t = 0;
  std::int64_t n_blocks = 0;
  std::int64_t width = 0;
  std::int64_t ratio = 0;
  static constexpr std::uint64_t Align(std::uint64_t b) { return (b + 255) / 256 * 256; }
  std::int64_t tiles() const { return (n_blocks + kQsaTopKTile - 1) / kQsaTopKTile; }
  // A tile keeps at most every block with a kept cell: whole blocks but the
  // token's own and the last one kept in part.
  std::int64_t candidates() const { return ((width + ratio - 1) / ratio) + 2; }
  static constexpr std::uint64_t keys() { return 0; }
  std::uint64_t query() const {
    return Align(static_cast<std::uint64_t>(t) * static_cast<std::uint64_t>(n_blocks) * 4);
  }
  std::uint64_t pairs() const {
    return query() + Align(static_cast<std::uint64_t>(t * kQsaIndexDim * kQsaIndexHeads) * 2);
  }
  std::uint64_t counts() const {
    return pairs() +
           Align(tiles() > 1 ? static_cast<std::uint64_t>(t * tiles() * candidates()) * 8 : 0);
  }
  std::uint64_t bytes() const {
    return counts() + Align(tiles() > 1 ? static_cast<std::uint64_t>(t * tiles()) * 4 : 0);
  }
};
// The attention's shares of a token's KV head: enough warps for a decode
// step or a verify, each at least 4 gathers; one past about 96 tokens.
constexpr std::int64_t QsaAttnShares(std::int64_t t, std::int64_t kv_heads, std::int64_t row) {
  const std::int64_t gathers = row / 16;
  const std::int64_t items = t * kv_heads;
  const std::int64_t wanted = (192 + items - 1) / items;
  const std::int64_t most = (gathers + 3) / 4;
  const std::int64_t shares = wanted < most ? wanted : most;
  return shares < 1 ? 1 : shares;
}
// A share's partial result: each query head's running max and sum, then its
// unnormalized output (F32).
constexpr std::uint64_t QsaAttnPartial(std::int64_t group) {
  return static_cast<std::uint64_t>(group) * (2 + kQsaAttnHead) * 4;
}
ggml_tensor* QsaTopK(ggml_context* context, ggml_tensor* q, ggml_tensor* blocks,
                     ggml_tensor* positions, ggml_tensor* pool, std::int64_t n_blocks,
                     std::int64_t width, std::int64_t ratio);
// `q` F32 [256, heads, t] (jitllm.qsa.prep's), `k` and `v` F16 [256 ·
// kv_heads, cells] (the caches, as their set_rows nodes, with equal
// 16-byte-aligned row strides), `cells` a
// jitllm.qsa.topk node: F32 [256 · heads, t]. At most 16 query heads per KV
// head. Draws scratch past one share per token's head (PlanQsaAttn).
ggml_tensor* QsaAttn(ggml_context* context, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                     ggml_tensor* cells, float scale);
std::expected<void, KernelFailure> CheckQsaPool(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckQsaTopK(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckQsaAttn(const ggml_tensor* node);
// Scratch bytes: the keys of every token's blocks, and past one tile the
// tiles' candidates; the attention's shares' partial results.
std::uint64_t PlanQsaTopK(const ggml_tensor* node);
std::uint64_t PlanQsaAttn(const ggml_tensor* node);
std::expected<void, KernelFailure> RunQsaPool(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunQsaTopK(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunQsaAttn(LaunchContext& launch, ggml_tensor* node);

// Byte ranges copied device to device in one kernel, not a graph node: a
// speculative verify's snapshot of the state rows it will write, and the
// restore of the rows a rejected draft wrote (D-068's truncation at a
// snapshot; engine/live_state.h). Each range is 16-byte aligned at
// both ends and a multiple of 16 bytes; no destination overlaps another
// range. `ranges` is memory the device reads (pinned host memory, or
// device memory), valid until the copy completes; at most kMaxRangeCopies.
struct RangeCopy {
  std::uint64_t from = 0;
  std::uint64_t to = 0;
  std::uint64_t bytes = 0;
};
inline constexpr std::uint32_t kMaxRangeCopies = 65535;
std::expected<void, KernelFailure> CopyRanges(LaunchContext& launch, const RangeCopy* ranges,
                                              std::uint32_t count);

// DeepSeek V4's fast decode plan (the owner's policy, 2026-09-28: speed
// before bit-exactness; dsv4_graph.h Dsv4GraphOptions::fused), jitLLM's own
// kernels (dsv4_fast.cu), none of them GGML's arithmetic bit for bit:
//
//   jitllm.q8_1         F32 rows quantized once to GGML's Q8_1 blocks
//                       (quantize_row_q8_1_cuda), each row padded to 512
//                       values, for every product that reads them;
//   jitllm.vecq         a quantized vector product over Q8_1 activations,
//                       up to kVecQTokens tokens: dense (each weight row
//                       read once for every token), or routed experts
//                       (ids), each distinct expert of the chunk read once
//                       for every token that selected it; with gate weights
//                       its SwiGLU (clamped) in the same kernel. Under one
//                       launch configuration a token's sums take the same
//                       arithmetic however many tokens share the read; the
//                       default configuration follows the token count, so
//                       a verify's rows need not equal their one-row
//                       steps' bit for bit (D-092's exact mode does);
//   jitllm.dsv4.route   the routing: sqrt(softplus(logits)), the top
//                       experts by it plus the bias (or the hash layers'
//                       table's), their weights normalized and scaled: I32
//                       [2 · used, tokens], the ids then the weights' bits;
//   jitllm.dsv4.combine the routed experts' weighted sum plus the shared
//                       expert's output;
//   jitllm.dsv4.hc_mix  the hyper-connection mixes' partial sums (each
//                       stream group's dot products with the mixing
//                       weights, and its sum of squares), over chunks of
//                       the flattened streams;
//   jitllm.dsv4.hc_pre  the mixes, pre and post weights and the Sinkhorn
//                       combination, the streams' weighted sum and its
//                       RMSNorm times the layer's norm weight: F32
//                       [(width + 32) · tokens], the normed rows packed,
//                       then each token's 32: post (4) and comb (16,
//                       dst-major as dsv4_hc_comb writes it).
//
// `q8` of a vecq node is a jitllm.q8_1 node. Routed products read `ids`
// (I32 [used, tokens], rows may be strided, as a jitllm.dsv4.route node's
// first columns); an id is not checked on the device against the experts
// (the routing and the hash table's check bound it, as for mul_mat_id).
// The fast plan's chunks take vecq to kVecQTokens rows; a wave of several
// sequences' rows (dsv4_graph.h Dsv4WaveGraph) to kVecQMaxTokens, routed
// products to 128 (token, slot) pairs.
inline constexpr std::int64_t kVecQTokens = 8;
inline constexpr std::int64_t kVecQMaxTokens = 16;
inline constexpr std::int64_t kDsv4HcChunks = 256;
inline constexpr std::int64_t kDsv4HcChunkThreads = 64;  // a hc_mix block's threads
enum class VecQGlu : std::int32_t { kNone = 0, kSwiglu = 1, kSwigluClamp = 2, kGeGlu = 3 };
// `x` F32 [k, rows, planes]: I8, the Q8_1 blocks of rows · planes rows.
ggml_tensor* QuantizeQ8(ggml_context* context, ggml_tensor* x);
// Dense: `weights` [k, n], `q8` of [k, tokens]: F32 [n, tokens]. Grouped
// (`weights` [k, n, groups] and no ids, `per_slot`): matrix g over row g of
// `q8` of [k, groups, tokens]: F32 [n, groups, tokens]. Routed:
// `weights` [k, n, experts] at their stride, `ids` I32 [used, tokens]; `q8`
// of [k, tokens] (each token's row serves its slots) or, with `per_slot`,
// of [k, used, tokens]: F32 [n, used, tokens]. `gate` (optional): weights of
// `weights`' type, shape and strides; the result is glu(gate · x, weights · x).
// kGeGlu uses the pinned GELU-tanh and requires a zero clamp limit.
// Callers opt into it; ordinary products plus split GeGLU remain the fallback.
ggml_tensor* VecQ(ggml_context* context, ggml_tensor* weights, ggml_tensor* q8, ggml_tensor* ids,
                  std::int64_t tokens, bool per_slot, ggml_tensor* gate = nullptr,
                  VecQGlu glu = VecQGlu::kNone, float limit = 0.0f);
// A vecq node takes the configuration a one-token product of its shape
// takes (its rows, warps and reduction), whatever its tokens: each token's
// sums then equal a one-token step's bit for bit (a wave of one-row steps,
// dsv4_graph.h Dsv4WaveGraph). A dense product without a GLU computes four
// tokens a pass, reading each weight block once for them; routed and GLU
// products one token a pass, their weights read again from cache.
void SetVecQOneToken(ggml_tensor* node);
bool VecQOneToken(const ggml_tensor* node);
// `logits` F32 [experts, tokens]; `bias` F32 [experts] or null; `table`
// I32 [used, vocab] and `tokens` I32 [tokens], or both null.
ggml_tensor* Dsv4Route(ggml_context* context, ggml_tensor* logits, ggml_tensor* bias,
                       ggml_tensor* table, ggml_tensor* tokens, std::int64_t used, bool norm,
                       float clamp, float scale);
// `down` F32 [n, used, tokens], `route` a jitllm.dsv4.route node, `shared`
// F32 [n, tokens]: F32 [n, tokens].
ggml_tensor* Dsv4Combine(ggml_context* context, ggml_tensor* down, ggml_tensor* route,
                         ggml_tensor* shared);
// `x` F32 [width, hc, tokens] packed, `fn` [width · hc, mixes] of a type
// Dsv4HcMixWeightType takes: F32 [mixes + 1, kDsv4HcChunks, tokens]. Each
// weight is widened to F32 exactly, so every type's sums take one order.
ggml_tensor* Dsv4HcMix(ggml_context* context, ggml_tensor* x, ggml_tensor* fn);
// The mixing weights' types jitllm.dsv4.hc_mix reads: F32, F16 and BF16.
bool Dsv4HcMixWeightType(ggml_type type);
// `partials` a jitllm.dsv4.hc_mix node over `x`; `scale` F32 [3], `base`
// F32 [(2 + hc) · hc], `norm` F32 [width].
ggml_tensor* Dsv4HcPre(ggml_context* context, ggml_tensor* partials, ggml_tensor* x,
                       ggml_tensor* scale, ggml_tensor* base, ggml_tensor* norm, float rms_eps,
                       float hc_eps, std::int32_t iterations);
inline constexpr std::int64_t kDsv4HcTail = 32;  // hc_pre's floats after the normed row
// jitllm.dsv4.compress: a compressor's blocks before their norm, as
// deepseek4.cpp's build_overlap_compressed_kv_from_state (overlap) or
// build_hca_compressed_kv_from_state computes them from the state ring and
// the chunk's rows: each block's `ratio` (overlap: twice `ratio`, the
// previous block's first half and this one's second) rows named by
// `read_idxs`, softmax-weighted by their scores per channel and summed.
// `state_kv` and `state_score` F32 [c, S], `kv` and `score` F32 [c, tokens]
// (rows may be strided): row s < S of the source is the state's, S + i the
// chunk's row i, and with overlap S + tokens a zero row whose score is
// -inf. c is the head (overlap: twice the head). F32 [head, 1, blocks]; a
// block whose rows' scores are all -inf gives zeros.
ggml_tensor* Dsv4Compress(ggml_context* context, ggml_tensor* state_kv, ggml_tensor* state_score,
                          ggml_tensor* kv, ggml_tensor* score, ggml_tensor* read_idxs,
                          std::int64_t ratio, bool overlap);

std::expected<void, KernelFailure> CheckQuantizeQ8(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckVecQ(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckDsv4Route(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckDsv4Combine(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckDsv4HcMix(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckDsv4HcPre(const ggml_tensor* node);
// The read indices are not checked on the host: an index outside the
// source reads as the zero row (a score of -inf), never out of bounds.
std::expected<void, KernelFailure> CheckDsv4Compress(const ggml_tensor* node);
// The Q8_1 blocks a jitllm.q8_1 node of `k` values by `rows` rows takes.
std::int64_t Q8Bytes(std::int64_t k, std::int64_t rows);
// Whether jitllm.vecq has a kernel for a weight type.
bool VecQType(ggml_type type);

// DeepSeek V4's sparse attention at depth (the fast plan over a ring
// window, model/dsv4.h Dsv4Window::kRing; dsv4_sparse.cu), deterministic,
// so a run repeats bit for bit (RE-031):
//
//   jitllm.dsv4.lid_topk     the lightning indexer and its selection: each
//                            row's scores over the compressed rows it sees
//                            (its visible count), Σ_h w[h] · relu(q_h · k)
//                            on tensor cores (q rounded to F16 as GGML's
//                            WMMA indexer rounds it, F32 sums), and the
//                            `top` best of them, ties to the lower row,
//                            listed in ascending order and padded with -1
//                            (all it sees when that is fewer). Rows share
//                            each key read four at a time (a verify's rows
//                            one pass); their scores go to pool scratch of
//                            at most kDsv4LidScratch bytes, the rows taken
//                            in groups that fit it.
//   jitllm.dsv4.sparse_mask  the attention mask of a compressed layer over
//                            [window cells | compressed rows]: the window's
//                            mask copied (-inf past it to the compressed
//                            rows' start), then 0 at the rows the indexer
//                            selected (CSA: build_top_k_mask's result) or
//                            at the rows each row sees (HCA), -inf
//                            elsewhere; the concatenation's result, in one
//                            pass, with no host-built compressed mask.
inline constexpr std::int64_t kDsv4LidScratch = std::int64_t{128} << 20;
// And its attention: a flash_attn_ext node marked here (op_params[5] = 1,
// a slot GGML leaves free) takes the MMA kernel's sparse gather of its
// unmasked cells (ops_ext.h FlashAttnMma) whenever its n_kv_max cells are
// at most half of K's, not only past upstream's 4,096 cells; the cells
// gathered are the mask's either way.
inline constexpr int kFlashAttnSparseParam = 5;
// Dispatch-local only: a copied node records the explicit launch choice.
// Builders never set it; primitive/reference launches always clear it.
inline constexpr int kFlashAttnWideSparseParam = 6;
inline void SetFlashAttnSparseAny(ggml_tensor* node) { node->op_params[kFlashAttnSparseParam] = 1; }
// `q` F32 [128, 64, rows] (rows and heads 8-byte aligned), `k` F16 [128,
// n_kv] (the indexer's cache, rows 16-byte aligned), `w` F32 [64, rows],
// `visible` I32 [rows]: I32 [top, rows].
ggml_tensor* Dsv4LidTopK(ggml_context* context, ggml_tensor* q, ggml_tensor* k, ggml_tensor* w,
                         ggml_tensor* visible, std::int64_t top);
// `window` F16 [w, rows] (packed rows), and either `top` a
// jitllm.dsv4.lid_topk node or `visible` I32 [rows]; `cells` (at least w)
// the compressed rows' start: F16 [cells + n_kv, rows, 1, 1].
ggml_tensor* Dsv4SparseMask(ggml_context* context, ggml_tensor* window, ggml_tensor* top,
                            ggml_tensor* visible, std::int64_t cells, std::int64_t n_kv);
// The checks: operands bound, typed and shaped as above, within the
// kernels' 32-bit extents, the output disjoint from every operand. The
// visible counts are clamped to [0, n_kv] on the device; the selection's
// entries are rows below n_kv or -1, which the mask kernel also bounds.
std::expected<void, KernelFailure> CheckDsv4LidTopK(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckDsv4SparseMask(const ggml_tensor* node);
std::expected<std::uint64_t, KernelFailure> PlanDsv4LidTopK(const LaunchContext& launch,
                                                            const ggml_tensor* node);
std::expected<void, KernelFailure> RunDsv4LidTopK(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunDsv4SparseMask(LaunchContext& launch, ggml_tensor* node);

std::expected<void, KernelFailure> RunQuantizeQ8(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunVecQ(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunDsv4Route(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunDsv4Combine(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunDsv4HcMix(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunDsv4HcPre(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunDsv4Compress(LaunchContext& launch, ggml_tensor* node);

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_JITLLM_OPS_H_
