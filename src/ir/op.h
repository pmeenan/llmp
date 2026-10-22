// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The graph IR's op set, version 1 (D-107;
// docs/engine-components.md#the-graph-ir): the operations a model's step is
// built from, the fused kinds patterns rewrite subgraphs into, and the
// parameters a node carries. Changing the op set is a reviewed change, like an
// artifact schema: it bumps kOpSetVersion once the version has been released.
//
// An operation's semantics are its op, its parameters and its operands'
// formats; an implementation (execution/registry.h) declares which of
// those it computes. Every fused kind and every composite op has a
// primitive decomposition (ir/decompose.h), so a graph without its pattern
// is still correct (portability.md's primitive-fallback rule). Parameter
// structs are trivially copyable, fit a node's parameter bytes and have no
// implicit padding (their reserved bytes stay zero), so a node's parameter
// bytes are canonical (ir/graph.h's digest hashes them as they are).
//
// Shapes are GGML's order, ne[0] innermost: a row of ne[0] values, ne[1]
// rows, ne[2] · ne[3] matrices. Unless an op says otherwise, a source may
// be any readable format (a quantized operand is read through its
// decoder), a result is F32 and may be recorded in another dense float
// encoding (the value rounded once), and arithmetic is defined in exact
// real arithmetic rounded to the result (kernels/reference/exec.h computes
// in F32 values with wider sums). Where GGML has the op, its semantics are
// GGML's at the locked pin (ggml.c's shape rules, ggml-cuda's kernels);
// llmpalooza's own definitions cite their source. "b broadcasts over a"
// means a.ne[d] % b.ne[d] == 0 and a's index i reads b's i % b.ne[d].
//
// Sources are listed in position order; '?' marks an optional one (an
// absent TensorId at its position). Integer sources (ids, positions,
// indices) are I32 unless stated; an index outside its table is an error.
//
// Structure (no arithmetic):
//   kInput, kWeight, kState  a leaf: no sources; the tensor is bound from
//       outside (an input or constant each step, a weight, a state tensor).
//   kView     (a): a view of a's bytes with its own shape, byte strides and
//       offset (the output's TensorDesc): GGML's ggml_view_*.
//   kReshape  (a): a contiguous a, viewed with another shape of the same
//       element count.
//   kPermute  (a): PermuteParams: a view, source axis i at output axis axes[i]
//       (ggml_permute).
//   kCont     (a): a's values in logical order, contiguous, in a's
//       encoding and with any shape of the same element count (ggml_cont_4d).
//   kConcat   (a, b): ConcatParams: along dim, other dims equal, one encoding.
//   kRepeat   (a): a tiled to the output's shape, a multiple of a's.
//   kConvert  (a): a dense tensor into another dense encoding (a cast).
//   kGetRows  (a [n, R, A2, A3], ids [N, A2, A3]): [n, N, A2, A3], row
//       ids[i, j, k] of matrix (j, k) (ggml_get_rows); I32 when a is I32.
// Products (ProductParams):
//   kMatMul   (a [K, N, A2, A3], b [K, M, B2, B3]): [N, M, B2, B3],
//       out[n, m] = Σ_k a[k, n] · b[k, m]; a broadcasts over b in dims 2 and
//       3 (B2 % A2 == 0: b's matrix i reads a's i / (B2 / A2), ggml_mul_mat).
//   kMatMulId (as [K, N, E], b [K, B1, T], ids [U, T]): [N, U, T], out[:, u,
//       t] = as[ids[u, t]] · b[:, u % B1, t] (ggml_mul_mat_id).
//   A product's accumulation and hint (and attention's, AttnParams) do not
//   change its definition: they are the choices production makes per node
//   (GGML's ggml_prec_set_acc and ggml_mul_mat_set_hint), hashed so that a
//   lowering reproduces them node for node. Accumulate::kF32 requires an
//   implementation to sum in F32 or wider (GGML_PREC_F32); kDefault lets it
//   choose (GGML's F16 tensor-core sums where it takes them). Hint
//   kHadamard says a is a square Hadamard matrix (GGML_HINT_SRC0_IS_HADAMARD,
//   llama.cpp's llama_mul_mat_hadamard), which a backend may apply as a
//   fast transform. kernels/reference/exec.h ignores both.
//   kPrepare  (a): a in the output's encoding, quantized row by row (the
//       encoder's rounding): a product's input form.
//   kExpertSort (ids [U, T]): ExpertSortParams: outputs I32 [U · T], the
//       (token, slot) pairs t · U + u in expert order, token order within an
//       expert; and I32 [E + 1], each expert's first position, then U · T.
// Attention and masks:
//   kAttention (q [D, Nq, H, B], k [D, Nk, Hk, B], v [Dv, Nk, Hk, B],
//       mask? [Nk, >= Nq, M2, M3], sinks? [H]): AttnParams: [Dv, H, Nq, B]
//       (ggml_flash_attn_ext): head h reads KV head h / (H / Hk); a score is
//       q · k · scale, with softcap c: c · tanh(score / c); plus slope · mask
//       (slope 1, or ALiBi's for max_bias); softmax over cells, a sink's
//       exp(sink[h]) joining the denominator. A row that sees no cell and
//       has no sink is zero (GGML's kernels divide 0 by 0 there: a port's
//       comparison skips such rows or the port guards them). AttnParams'
//       accumulate is a product's (above); n_kv_max promises that no mask
//       row leaves more than that many cells finite, so an implementation
//       may gather them sparsely (ggml_flash_attn_ext_set_n_kv_max; a row
//       past the bound is undefined), and sparse_any lets it take that
//       gather at any cache length (llmp_ops.h SetFlashAttnSparseAny). The
//       reference executor ignores all three.
//   kMaskBuild (q_pos [R], cell_pos [C]): MaskParams: F16 or F32 [C, >= R],
//       0 where a query sees a cell, else -inf. A cell holding position c >=
//       0 is seen by a query at p when c <= p (kCausal), and p - c < window
//       (kWindow), or c < (p / block + 1) · block (kBlock: its own block's
//       later positions too). kAhead (amended 2026-10-10, slice G4a): c > p
//       (a later row of the step's own: no cell holds a position past the
//       step's last row) or, with a window, p - c < window (none: every
//       earlier cell): a block drafter's rows, which see one another and
//       the committed positions within their window (DSpark's, llama.cpp's
//       non-causal is_masked_swa). Rows past R and cells below 0 are -inf.
//   kCacheStore (cache [n, C, A2, A3], src [n, R, A2, A3], idx [R, I1, I2],
//       I32 or I64): the output is a view of cache with src's rows stored
//       at rows idx[r, j % I1, k % I2] of matrix (j, k), in cache's encoding
//       (ggml_set_rows).
//   kIndexScore (q [D, H, T, S], k [D, 1, C, S], w [H, T, 1, S], mask? [C,
//       T, 1, M3]): [C, T, 1, S] = Σ_h w[h, t] · max(q_h · k_c, 0) + mask
//       (ggml_lightning_indexer, DeepSeek V4's lightning indexer).
//   kRoundedIndexScore (q F32 [D,H,T], k F16 [D,1,C], w F32 [H,T],
//       counts I32 [T]): RoundedIndexScoreParams, exact F32 [C,T]. Round q
//       to F16, then sum head weights times ReLU(dot(q_head,k_cell));
//       ReLU maps a NaN dot to +0. Clamp count to [0,C]; cells outside its
//       prefix are literal -Inf. Products/reductions follow the ordinary
//       F32-or-wider numerical convention, not one universal native bit
//       sequence. The independent scalar uses wider sums. The primitive
//       fallback's F32->F16->F32 cast preserves query rounding, with F32
//       intermediates; a differently selected numerical implementation can
//       change scores and near-cutoff selected sets (D-053).
//   kPrefixOverwrite (values F32 [C,T], counts I32 [T]): PrefixOverwriteParams,
//       exact F32 [C,T]. Bit-copy cells below clamp(count,0,C); write the
//       literal fill_bits representation elsewhere. No arithmetic; every
//       F32 bit pattern, including signaling NaNs and signed zero, is valid.
//   kPool     (x [C, R · N, A2, A3], scores? like x): PoolParams: [C, N, A2,
//       A3], each run of R = ratio rows pooled: kMean their sum / R
//       (block keys, llmp.qsa.pool), kSoftmax their sum
//       weighted by softmax(scores) over the run per channel (DeepSeek V4's
//       CSA/HCA compressor, deepseek4.cpp's build_*_compressed_kv; a run of
//       -inf scores pools to 0).
// Norms and rotations (NormParams; over each row of ne[0] unless stated):
//   kRmsNorm   x / sqrt(mean(x²) + eps).
//   kL2Norm    x / max(sqrt(Σ x²), eps).
//   kLayerNorm (x - mean) / sqrt(mean((x - mean)²) + eps).
//   kChannelNorm x / max(sqrt(Σ x²), eps) over ne[1] for each (i0, i2, i3):
//       a pixel's channels (F.normalize; Qwen-Image's VAE).
//   kRope (x [D, H, T, B], pos, freq? [>= dims / 2]): RopeParams: x's shape
//       (ggml_rope_ext, _multi and _ext_back, with ggml_rope_set_offset):
//       channels [offset, offset + dims) rotate, the others pass through.
//       offset is even; freq has one factor a pair (dims for kVision).
//       pos is I32 [T] (kMulti, kInterleavedMulti: [4 · T], a section's
//       positions after another's), or with `table` F32 [2, dims / 2, T]
//       (cos, sin per pair: a precomputed rotation).
//       The angle of pair i (its exponent e: i, or a vision section's own
//       index) is formed in F32 as GGML's CUDA kernel forms it (rope.cu,
//       rope_yarn), each step rounded to F32: s = (float) base^((float)
//       (-2 / dims)) (the host's powf(base, -2.0f / dims)), f = (float)
//       s^e, t = (float) p · f, then t / freq[i], then x = scale · t;
//       with YaRN (ext_factor != 0) r = (1 - clamp((i - lo) / max(
//       0.001, hi - lo), 0, 1)) · ext_factor over ggml_rope_yarn_corr_dims'
//       float lo and hi, x = x · (1 - r) + t · r and the magnitude m =
//       attn_factor · (1 + 0.1 · logf(1 / scale)), else m = attn_factor.
//       The rotation is then exact: cos(x) · m and sin(x) · m (negated sin
//       when inverse) over the F32 x. At long context the F32 angle is the
//       definition: its rounding (about 0.03 radians at position 2^20) is
//       production's, not an error. GGML's own units build with
//       -use_fast_math, so the device's powf (s^e), its FMA contraction of
//       the YaRN mix and its sinf/cosf are approximations of these steps; a
//       port's comparison allows for those, not for another formation.
// Elementwise (b broadcasts over a; the result has a's shape):
//   kAdd, kSub, kMul, kDiv (a, b).
//   kSelect (condition I32, yes, no): exact byte copy from yes where the
//       condition is nonzero, otherwise no. Condition and no broadcast over
//       yes; branches have the same dense scalar encoding. The output has
//       yes's shape/encoding. No arithmetic, conversion or inspection of the
//       unselected value; NaN payloads, signed zero and integers retain bits.
//   kScale (x): ScaleParams: x · scale + bias.
//   kUnary (x): UnaryParams.       kClamp (x): ClampParams (NaN stays NaN;
//       GGML's fminf/fmaxf give a bound, as kSwiGluClamp's do).
//   kFill  (): FillParams: the output's shape, every value one value.
//   kGlu   (gate, up?): GluParams: act(gate) · up; with no up, gate's rows
//       hold gate then up as halves (up first when swapped). kSwiGluClamp:
//       silu(min(gate, limit)) · clamp(up, -limit, limit) (ggml_swiglu_clamp).
//   kSoftcap (x): SoftcapParams: cap · tanh(x / cap).
// Reductions and selection (over each row; ties to the lower index; NaN
// below every number):
//   kSumRows  (x): [1, n1, n2, n3].
//   kSoftMax  (x, mask?, sinks? [n2]): SoftMaxParams: softmax(x · scale +
//       slope · mask) (ggml_soft_max_ext), mask broadcasting over x's
//       matrices and reading its first n1 rows; a row of -inf is zero
//       (GGML's soft_max gives NaN, as attention's).
//   kArgmax   (x [n, R]): ArgmaxParams: I32 [R], the highest value's
//       index. kIgnore: NaN ignored, a row of NaN gives 0 (llmp.argmax's
//       native flavor, draft and verify heads); kFirstColumn: std::
//       max_element's, a NaN in column 0 gives 0 and a later NaN never
//       replaces a number (the host-greedy flavor every greedy head uses).
//   kArgsort  (x): ArgsortParams: I32 of x's shape, a stable sort's order.
//   kTopK     (x): TopKParams: I32 [k, n1, n2, n3], the k highest, highest
//       first. GGML's top_k returns them in no particular order and its
//       argsort is not stable: compare a port's selection by the multiset of
//       selected values (ties at the k-th value, ReLU's zeros, padded -inf
//       cells, a QSA block's shared score, pick either index).
//   kVisibleTopK (scores F32 [C,T], counts I32 [T]): TopKParams, I32
//       [k,T]. Clamp count to [0,C]; rank that prefix by descending numeric
//       value (NaN below every number), ties including signed zero and NaNs
//       by lower cell. Return the chosen IDs in ascending cell order, then
//       -1 to k. Every F32 bit pattern is supported; k may exceed C.
//   kSelectedMask (raw F16 [window,T], ids I32 [k,T]): SelectedMaskParams,
//       F16 [raw_cells + cells,T]. Copy raw half bits exactly, fill all
//       remaining cells with -Inf, set compressed IDs in [0,cells) to +0.
//       Duplicate IDs are idempotent; all other integers are ignored.
// Routing:
//   kRoute (logits [E, T], bias? [E], table? [U, V], tokens? [T]):
//       RouteParams: outputs ids I32 [U, T] and weights F32 [U, T].
//       Probabilities p: softmax (kSoftmaxTopK), sigmoid (kSigmoidTopK),
//       sqrt(softplus) (kSqrtSoftplusBias, kHash). The U highest p + bias
//       (bias optional for kSigmoidTopK, required for kSqrtSoftplusBias),
//       highest first; kHash's ids are table[:, tokens[t]], or without
//       tokens the table itself, [U, T]: each row's ids looked up already
//       (a token-hash side input's; amended 2026-10-10, slice G4a). Weights are
//       p[ids], with normalize divided by max(Σ, 2^-14) (llama.cpp's
//       build_moe_ffn), then times scale. scale is the factor itself and
//       above zero: a checkpoint's "unset" 0 (llama.cpp's
//       expert_weights_scale, which DeepSeek V4's production graph reads as
//       1) is written 1 here, and the verifier refuses 0.
//   kExpertReduce (x [n, U, T], w [U, T], ids? [U, T], scale? [E] or
//       [1, U, T]): [n, T] = Σ_u (x[:, u, t] · s[u, t]) · w[u, t], in slot
//       order, where s is scale[ids[u, t]] for a scale per expert (which
//       needs the ids), scale[0, u, t] for one already gathered per slot
//       (llama.cpp's get_rows of the experts' scales; the ids then only
//       name each slot's expert, as expert parallelism reads them), else 1.
// Recurrences and convolutions:
//   kSsmConv (sx [K - 1 + T, C, S], c [K, C]): [C, T, S], out[ch, t, s] =
//       Σ_j sx[t + j, ch, s] · c[j, ch] (ggml_ssm_conv).
//   kGatedDeltaNet (q [Sk, Hk, T, Bq], k like q, v [Sv, H, T, B], g [1 or
//       Sk, H, T, B], beta [1, H, T, B], state [Sk, Sv, H, B]): outputs the
//       attention [Sv, H, T, B] and the final state [Sk, Sv, H, B]
//       (ggml_gated_delta_net with one snapshot, gated_delta_net.cu). Head
//       h reads q and k head h % Hk of sequence b / (B / Bq). Per token,
//       with S[i][c] = state[i, c, h, b] and d = exp(g) (one value, or one
//       per i): S ← d ⊙ S, then S[i][c] += k[i] · beta · (v[c] - Σ_i S[i][c]
//       · k[i]), and the attention is Σ_i S[i][c] · q[i] / sqrt(Sk).
//   kConv2d (kernel [KW, KH, IC, OC], x [W, H, IC, N], bias? [OC]):
//       Conv2dParams: [W', H', OC, N] (ggml_conv_2d_direct, plus the bias).
//   kUpsample (x [W, H, C, N]): UpsampleParams: nearest, [f · W, f · H, C, N].
// Hyper-connections (HcParams; hc = streams):
//   kHcPre (x [W, hc, T], w): [W, T]. kWeights: w [hc, T], Σ_h x[:, h] · w[h]
//       (ggml_dsv4_hc_pre; scale stays 1); kGated: w [W, hc, T], scale ·
//       Σ_h x[:, h] · sigmoid(w[:, h]) (ggml_dsv4_hc_pre_gated; Qwen3.8's
//       fold, whose x is kHcNorm's output as [W, hc, T] and scale 1 / hc).
//   kHcPost (y [W, T], x [W, hc, T], post [hc, T], comb? [hc, hc, T]): [W,
//       hc, T], out[:, d] = y · post[d] + Σ_s x[:, s] · comb[d, s]
//       (ggml_dsv4_hc_post); without comb, + x[:, d] (Qwen3.8's combine).
//   kHcMix (mixes [(2 + hc) · hc, T], scale [>= 3], base [(2 + hc) · hc]):
//       kSinkhorn: [hc, hc, T], mixes' last hc² (times scale[2], plus base)
//       softmaxed over destinations, plus eps, then normalized over sources
//       and destinations alternately, iterations times
//       (ggml_dsv4_hc_comb); kSigmoid: [hc, 2, T], pre = sigmoid(m · scale[0]
//       + base) + eps and post = 2 · sigmoid(m · scale[1] + base) of the
//       first and second hc (deepseek4.cpp's build_hc_pre).
// Sampling and collectives:
//   kSample (logits [V, R], keys? I64 [3, R]): SampleParams: I32 [R],
//       execution/sampling.h's Greedy, or Sample keyed by (seed, stream,
//       position) per row.
//   kAllReduce (x): the sum over the device group, in member order.
//   kAllGather (x): the members' x concatenated along dim, in member order.
//   kAllToAll (x): x split along dim into one part per member; member r
//       gets part r of every member, concatenated in member order.
//   kSend (x): no output; x to peer.  kRecv (): the output from peer.
// Fused kinds: kFused with FusedKind, sources and parameters per kind
// (ir/decompose.h lists each one's decomposition, which is its meaning).

#ifndef LLMP_IR_OP_H_
#define LLMP_IR_OP_H_

#include <array>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>

namespace llmp::ir {

inline constexpr std::uint32_t kOpSetVersion = 1;

enum class Op : std::uint16_t {
#define LLMP_PRIMITIVE(op, ...) op,
#define LLMP_FUSED(kind, ...)
#include "ir/op_traits.inc"
#undef LLMP_PRIMITIVE
#undef LLMP_FUSED
};

// Each fused kind's sources (in order), outputs and parameters; its
// decomposition (ir/decompose.h) is its meaning.
enum class FusedKind : std::uint16_t {
#define LLMP_PRIMITIVE(op, ...)
#define LLMP_FUSED(kind, ...) kind,
#include "ir/op_traits.inc"
#undef LLMP_PRIMITIVE
#undef LLMP_FUSED
};
// Counts are generated too: no separate enum or last-kind edit for a new op.
inline constexpr std::size_t kOpCount = 0
#define LLMP_PRIMITIVE(...) +1
#define LLMP_FUSED(...)
#include "ir/op_traits.inc"
#undef LLMP_PRIMITIVE
#undef LLMP_FUSED
    ;
inline constexpr std::size_t kFusedKindCount = 0
#define LLMP_PRIMITIVE(...)
#define LLMP_FUSED(...) +1
#include "ir/op_traits.inc"
#undef LLMP_PRIMITIVE
#undef LLMP_FUSED
    ;
inline constexpr FusedKind kLastFusedKind = static_cast<FusedKind>(kFusedKindCount - 1);

struct OpKey {
  Op op = Op::kInput;
  FusedKind fused = FusedKind::kNone;
  auto operator<=>(const OpKey&) const = default;
};

constexpr OpKey Primitive(Op op) { return {.op = op, .fused = FusedKind::kNone}; }
constexpr OpKey Fused(FusedKind kind) { return {.op = Op::kFused, .fused = kind}; }

std::string_view OpName(Op op);
std::string_view FusedKindName(FusedKind kind);
// "mul_mat", or "fused.rms_norm_mul".
std::string OpKeyName(OpKey key);

// ---------------------------------------------------------------- parameters

inline constexpr std::size_t kMaxParamBytes = 68;

enum class QueryRounding : std::uint8_t { kF16 };
struct RoundedIndexScoreParams {
  QueryRounding query = QueryRounding::kF16;
  std::array<std::uint8_t, 3> reserved{};
};
struct PrefixOverwriteParams {
  std::uint32_t fill_bits = 0;
};
static_assert(sizeof(RoundedIndexScoreParams) == 4);
static_assert(sizeof(PrefixOverwriteParams) == 4);

struct NormParams {
  float eps = 0.0f;
};

// Semantic output activation, preserved by fusion and primitive decomposition.
enum class NormGateFn : std::uint32_t { kSigmoid, kSilu };
struct NormGateParams {
  float eps = 0.0f;
  NormGateFn gate = NormGateFn::kSigmoid;
};
static_assert(sizeof(NormGateParams) == 8);

enum class RopeMode : std::uint8_t {
  kNormal,            // pairs (2i, 2i + 1)
  kNeox,              // pairs (i, i + dims / 2)
  kMulti,             // GGML's MROPE: NEOX pairs, sections' positions in turn
  kVision,            // GGML's VISION: pairs (i, i + dims), two sections
  kInterleavedMulti,  // GGML's IMROPE: NEOX pairs, sections interleaved
};
// GGML's rope_ext's parameters, its offset (ggml_rope_set_offset) and its
// inverse (ggml_rope_ext_back).
struct RopeParams {
  RopeMode mode = RopeMode::kNormal;
  bool inverse = false;      // rotate by -theta
  bool table = false;        // pos is an F32 cos/sin table, not positions
  bool llama3 = false;       // Llama 3 smooth frequency scaling
  std::uint32_t dims = 0;    // rotated values a head
  std::uint32_t offset = 0;  // the first rotated value (GGML's n_offs)
  std::uint32_t original_context = 0;
  float base = 10000.0f;
  float scale = 1.0f;
  float ext_factor = 0.0f;
  float attn_factor = 1.0f;
  float beta_fast = 32.0f;
  float beta_slow = 1.0f;
  std::array<std::uint32_t, 4> sections{};  // the multi-section modes'
  // UINT32_MAX: rotate every pair. Otherwise rotate the first rotated_dims/2
  // pairs with the full dims denominator and pairing span; other values pass
  // through unchanged (proportional RoPE, including zero active pairs).
  std::uint32_t rotated_dims = UINT32_MAX;
};

enum class MaskKind : std::uint8_t { kCausal, kWindow, kBlock, kAhead };
struct MaskParams {
  MaskKind kind = MaskKind::kCausal;
  // The mask has exactly the segment's rows (an attention that reads masks
  // of the query's rows, as llmp's QSA kernels and GGML's flash attention
  // at the pin do); otherwise its producer may pad them (op.h kMaskBuild).
  bool exact_rows = false;
  std::array<std::uint8_t, 2> reserved{};
  std::uint32_t window = 0;  // kWindow, kAhead: positions seen (kAhead: 0 for every one)
  std::uint32_t block = 0;   // kBlock: positions a block
};

// A product's accumulation (see the op set's products above).
enum class Accumulate : std::uint8_t {
  kDefault,  // the implementation's choice (GGML_PREC_UNDEFINED)
  kF32,      // F32 or wider (GGML_PREC_F32)
};
enum class ProductHint : std::uint8_t {
  kNone,
  kHadamard,  // a is a square Hadamard matrix (GGML_HINT_SRC0_IS_HADAMARD)
};
// Numerical contract preserved through sharing, CSE and final selection.
// Convert/Prepare still explicitly describes the operand's representation.
enum class ProductArithmetic : std::uint8_t {
  kImplementation,
  kBf16F32,          // BF16 RNE input, F32 accumulation, declared output storage
  kBlock32Q8SumF32,  // Block32 max-abs, FP16 scale/sum, roundf int8,
                     // pad512 zero tail and original vector reduction into F32
};
std::string_view ProductArithmeticName(ProductArithmetic arithmetic);
struct ProductParams {
  Accumulate accumulate = Accumulate::kDefault;
  ProductHint hint = ProductHint::kNone;
  std::array<std::uint8_t, 2> reserved{};
  ProductArithmetic arithmetic = ProductArithmetic::kImplementation;
  std::array<std::uint8_t, 3> arithmetic_reserved{};
};

struct AttnParams {
  float scale = 1.0f;
  float softcap = 0.0f;   // 0: none
  float max_bias = 0.0f;  // ALiBi's; 0: none
  // The cells a row may see, which its mask already enforces (so a mask is
  // required); an implementation may skip the cells the mask hides. 0:
  // the whole cache.
  std::uint32_t window = 0;
  bool sinks = false;  // a learned sink per head (source 4)
  Accumulate accumulate = Accumulate::kDefault;
  bool sparse_any = false;  // the sparse gather at any length (needs n_kv_max)
  std::uint8_t reserved = 0;
  // At most this many finite cells in any mask row (needs the mask); 0: no
  // bound.
  std::uint32_t n_kv_max = 0;
};

enum class UnaryFn : std::uint8_t {
  kSilu,
  // GGML's GELU is the tanh form; the erf form is its GELU_ERF. Named by
  // form so a GGML name never maps to the other one.
  kGeluErf,   // x · Φ(x) (GGML's GELU_ERF)
  kGeluTanh,  // the tanh approximation (GGML's GELU, Gemma's, Qwen-Image's)
  kSigmoid,
  kTanh,
  kRelu,
  kExp,
  kSoftplus,  // log(1 + exp(x))
  kSqrt,
  kSqr,
  kNeg,
  kAbs,
  kSgn,
};
struct UnaryParams {
  UnaryFn fn = UnaryFn::kSilu;
};

// kGeGluErf is GGML's GEGLU_ERF; kGeGluTanh is GGML's GEGLU (ggml_geglu_split,
// every Gemma FFN). The typed GluParams distinguish the arithmetic.
enum class GluKind : std::uint8_t { kSwiGlu, kGeGluErf, kGeGluTanh, kSwiGluClamp };
struct GluParams {
  GluKind kind = GluKind::kSwiGlu;
  bool swapped = false;  // a single source's halves: up, then gate
  std::array<std::uint8_t, 2> reserved{};
  float limit = 0.0f;  // kSwiGluClamp's
};

struct ScaleParams {
  float scale = 1.0f;
  float bias = 0.0f;
};

struct ClampParams {
  float min = 0.0f;
  float max = 0.0f;
};

struct FillParams {
  float value = 0.0f;
};

struct SoftcapParams {
  float cap = 0.0f;
};

struct SoftMaxParams {
  float scale = 1.0f;
  float max_bias = 0.0f;
};

enum class ArgmaxNan : std::uint8_t { kIgnore, kFirstColumn };
struct ArgmaxParams {
  ArgmaxNan nan = ArgmaxNan::kIgnore;
};

struct ArgsortParams {
  bool descending = false;
};

struct TopKParams {
  std::uint32_t k = 0;
};

struct RoundedIndexMaskParams {
  std::uint32_t top = 0;
  std::uint32_t raw_cells = 0;
  std::uint32_t cells = 0;
};

struct SelectedMaskParams {
  std::uint32_t raw_cells = 0;
  std::uint32_t cells = 0;
};

struct ExpertSortParams {
  std::uint32_t experts = 0;
};

enum class RouteKind : std::uint8_t { kSoftmaxTopK, kSigmoidTopK, kSqrtSoftplusBias, kHash };
struct RouteParams {
  RouteKind kind = RouteKind::kSoftmaxTopK;
  bool normalize = false;  // the selected weights scaled to sum to 1
  std::array<std::uint8_t, 2> reserved{};
  std::uint32_t experts = 0;
  std::uint32_t used = 0;
  float scale = 1.0f;  // then multiplied by this
};

enum class HcMixKind : std::uint8_t { kSigmoid, kSinkhorn };
enum class HcFold : std::uint8_t { kWeights, kGated };
struct HcParams {
  HcMixKind mix = HcMixKind::kSigmoid;  // kHcMix's
  HcFold fold = HcFold::kWeights;       // kHcPre's
  std::array<std::uint8_t, 2> reserved{};
  std::uint32_t streams = 0;
  std::uint32_t iterations = 0;   // kSinkhorn's
  float eps = 0.0f;               // kHcMix's
  float scale = 1.0f;             // kHcPre's
  std::uint16_t mix_offset = 0;   // kHcMix's offset in mixes/base (0: auto)
  std::uint16_t scale_index = 0;  // kHcMix's index in scale (0: auto)
};

struct ConcatParams {
  std::uint8_t dim = 0;
};

struct PermuteParams {
  std::array<std::uint8_t, 4> axes{0, 1, 2, 3};
};

struct Conv2dParams {
  std::uint32_t stride = 1;
  std::uint32_t padding = 0;
  std::uint32_t dilation = 1;
};

struct UpsampleParams {
  std::uint32_t factor = 2;
};

enum class PoolKind : std::uint8_t { kMean, kSoftmax };
struct PoolParams {
  std::uint32_t ratio = 1;  // cache rows pooled into one
  PoolKind kind = PoolKind::kMean;
  std::array<std::uint8_t, 3> reserved{};
};

enum class SampleKind : std::uint8_t { kGreedy, kSeeded };
struct SampleParams {
  SampleKind kind = SampleKind::kGreedy;
  std::array<std::uint8_t, 3> reserved{};
  float temperature = 1.0f;
  std::uint32_t top_k = 0;
  float top_p = 1.0f;
  float min_p = 0.0f;
};

struct CollectiveParams {
  std::uint32_t group = 0;  // the partition's device group
  std::uint32_t peer = 0;   // kSend's and kRecv's
  std::uint8_t dim = 0;     // kAllGather's and kAllToAll's
  std::array<std::uint8_t, 3> reserved{};
};

// The fused kinds' own parameters.
struct NormRopeParams {
  float eps = 0.0f;
  RopeParams rope{};
};

struct CompressorParams {
  std::uint32_t ratio = 0;
  bool overlap = false;
  std::array<std::uint8_t, 3> reserved{};
};

// The most owners a step joins under kOwnerAttention's groups.
inline constexpr std::uint32_t kMaxOwnerCohort = 64;
struct OwnerAttnParams {
  AttnParams attn{};
  std::uint32_t owners = 0;
  // The joined owners the node is a group of (0: its own owners) and its
  // first owner's place among them; offset + owners <= cohort.
  std::uint32_t cohort = 0;
  std::uint32_t offset = 0;
};

struct HcNormPostParams {
  float eps = 0.0f;
  std::uint32_t streams = 0;
};

struct SinkhornPreParams {
  std::uint32_t streams = 0;
  float eps = 0.0F;
  float hc_eps = 0.0F;
  std::uint32_t iterations = 0;
};

struct DupUpParams {
  std::uint32_t factor_t = 1;
};

struct SpatialNormParams {
  float eps = 0.0f;
  float scale = 1.0f;
  bool silu = false;
  std::array<std::uint8_t, 3> reserved{};
};

struct RopeSectionsParams {
  float eps = 0.0f;
  float base = 10000.0f;
  RopeMode mode = RopeMode::kNormal;  // kNormal or kNeox, every section's
  std::uint8_t sections = 0;          // 1 to 3
  std::array<std::uint8_t, 2> reserved{};
  std::array<std::uint32_t, 3> dims{};  // each section's rotated values
};

struct MaskPackParams {
  std::uint32_t cells = 0;
  std::uint32_t rows = 0;
};

struct SplitModulateParams {
  float eps = 0.0f;
  std::uint32_t split = 0;
};

struct QkAttentionParams {
  float eps = 0.0f;
  float scale = 1.0f;
};

struct ScaledUnaryParams {
  float scale = 1.0f;
  UnaryFn fn = UnaryFn::kSilu;
  std::array<std::uint8_t, 3> reserved{};
};

struct HcFoldParams {
  std::uint32_t streams = 0;
  bool prepared = false;  // an output in MXFP8 rows (format::kMxfp8 scale atoms)
  bool rounded = false;   // an output in BF16
  std::array<std::uint8_t, 2> reserved{};
  float eps = 0.0f;  // 3-source norm's eps
};

struct GdnConvParams {
  float eps = 0.0f;
  float scale = 1.0f;
  std::uint32_t qk_channels = 0;
  std::uint32_t head = 0;
};

struct GdnHistoryParams {
  std::uint32_t taps = 0;
};

struct GdnStepParams {
  bool write = false;
  std::array<std::uint8_t, 3> reserved{};
};

struct HeadGateParams {
  std::uint32_t head = 0;
};

struct BlockPoolParams {
  float eps = 0.0f;
  std::uint32_t ratio = 0;
  RopeParams rope{};
};

struct BlockSparseAttnParams {
  AttnParams attn{};
  std::uint32_t width = 0;  // cells kept
  std::uint32_t ratio = 0;  // cells a block
};

struct MoeExpertsParams {
  GluParams glu{};
  std::uint32_t hidden = 0;  // F
};

// What a step input (a kInput leaf of Role::kInput) holds where a pass
// reads it as more than its values: its parameters, which the graph's
// digest includes (names it does not). kNone: nothing beyond its format.
enum class InputFact : std::uint8_t {
  kNone,
  // I32 [sections · rows]: each row's position, `sections` copies one
  // after another (a multi-section rotation's positions).
  kPositions,
};
inline constexpr InputFact kLastInputFact = InputFact::kPositions;
struct InputParams {
  InputFact fact = InputFact::kNone;
  std::uint8_t sections = 0;
  std::array<std::uint8_t, 2> reserved{};
  bool operator==(const InputParams&) const = default;
};

template <typename P>
inline constexpr bool kIsParams = std::is_trivially_copyable_v<P> && sizeof(P) <= kMaxParamBytes;

static_assert(kIsParams<NormParams> && kIsParams<RopeParams> && kIsParams<MaskParams> &&
              kIsParams<ProductParams> && kIsParams<AttnParams> && kIsParams<UnaryParams> &&
              kIsParams<GluParams> && kIsParams<ScaleParams> && kIsParams<ClampParams> &&
              kIsParams<FillParams> && kIsParams<SoftcapParams> && kIsParams<SoftMaxParams> &&
              kIsParams<ArgmaxParams> && kIsParams<ArgsortParams> && kIsParams<TopKParams> &&
              kIsParams<RoundedIndexMaskParams> && kIsParams<SelectedMaskParams> &&
              kIsParams<RoundedIndexScoreParams> && kIsParams<PrefixOverwriteParams> &&
              kIsParams<ExpertSortParams> && kIsParams<RouteParams> && kIsParams<HcParams> &&
              kIsParams<ConcatParams> && kIsParams<PermuteParams> && kIsParams<Conv2dParams> &&
              kIsParams<UpsampleParams> && kIsParams<PoolParams> && kIsParams<SampleParams> &&
              kIsParams<CollectiveParams> && kIsParams<NormRopeParams> &&
              kIsParams<OwnerAttnParams> && kIsParams<HcNormPostParams> && kIsParams<DupUpParams> &&
              kIsParams<ScaledUnaryParams> && kIsParams<HcFoldParams> && kIsParams<GdnConvParams> &&
              kIsParams<GdnHistoryParams> && kIsParams<GdnStepParams> &&
              kIsParams<HeadGateParams> && kIsParams<BlockPoolParams> &&
              kIsParams<BlockSparseAttnParams> && kIsParams<MoeExpertsParams> &&
              kIsParams<CompressorParams> && kIsParams<SinkhornPreParams> &&
              kIsParams<SpatialNormParams> && kIsParams<RopeSectionsParams> &&
              kIsParams<MaskPackParams> && kIsParams<QkAttentionParams> &&
              kIsParams<SplitModulateParams> && kIsParams<InputParams>);
static_assert(sizeof(CompressorParams) == 8 && sizeof(ScaledUnaryParams) == 8 &&
              sizeof(HcFoldParams) == 12 && sizeof(GdnConvParams) == 16 &&
              sizeof(GdnStepParams) == 4 && sizeof(BlockPoolParams) == 8 + sizeof(RopeParams) &&
              sizeof(BlockSparseAttnParams) == sizeof(AttnParams) + 8 &&
              sizeof(MoeExpertsParams) == sizeof(GluParams) + 4);
static_assert(sizeof(SinkhornPreParams) == 16 && sizeof(SpatialNormParams) == 12 &&
              sizeof(RopeSectionsParams) == 24 && sizeof(MaskPackParams) == 8 &&
              sizeof(QkAttentionParams) == 8 && sizeof(SplitModulateParams) == 8);

// No implicit padding: each struct's size is the sum of its fields'.
static_assert(sizeof(RopeParams) == 4 + (4 * 4) + (6 * 4) + 16);
static_assert(sizeof(MaskParams) == 4 + 4 + 4);
static_assert(sizeof(ProductParams) == 8);
static_assert(sizeof(AttnParams) == (4 * 4) + 4 + 4);
static_assert(sizeof(GluParams) == 4 + 4);
static_assert(sizeof(RouteParams) == 4 + (3 * 4));
static_assert(sizeof(HcParams) == 4 + (5 * 4));
static_assert(sizeof(PoolParams) == 4 + 4);
static_assert(sizeof(SampleParams) == 4 + (4 * 4));
static_assert(sizeof(CollectiveParams) == (2 * 4) + 4);
static_assert(sizeof(NormRopeParams) == 4 + sizeof(RopeParams));
static_assert(sizeof(OwnerAttnParams) == sizeof(AttnParams) + (3 * 4));
static_assert(sizeof(InputParams) == 4);

}  // namespace llmp::ir

#endif  // LLMP_IR_OP_H_
