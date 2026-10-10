// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The graph IR's op set, version 1 (D-107; docs/engine-components.md#the-graph-ir):
// the operations a model's step is built from, the fused kinds patterns
// rewrite subgraphs into, and the parameters a node carries. Changing the
// op set is a reviewed change, like an artifact schema: it bumps
// kOpSetVersion once the version has been released.
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
//   kPool     (x [C, R · N, A2, A3], scores? like x): PoolParams: [C, N, A2,
//       A3], each run of R = ratio rows pooled: kMean their sum / R
//       (Qwen3.8's QSA block keys, llmp.qsa.pool), kSoftmax their sum
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
  // Structure.
  kInput,
  kWeight,
  kState,
  kView,
  kReshape,
  kPermute,
  kCont,
  kConcat,
  kRepeat,
  kConvert,  // a dense tensor into another encoding (a cast)
  kGetRows,  // rows gathered by index
  // Products.
  kMatMul,
  kMatMulId,    // grouped: each row times its selected matrices
  kPrepare,     // activations into a product's input form (a quantization)
  kExpertSort,  // routed rows sorted by matrix: a grouped product's worklist
  // Attention and masks.
  kAttention,   // AttnParams
  kMaskBuild,   // MaskParams
  kCacheStore,  // rows stored at indices (set_rows)
  kIndexScore,  // sparse attention's cell scores
  kPool,        // compressed KV pooling
  // Norms and rotations.
  kRmsNorm,
  kL2Norm,
  kLayerNorm,
  kChannelNorm,
  kRope,  // RopeParams
  // Elementwise.
  kAdd,
  kSub,
  kMul,
  kDiv,
  kScale,  // x · scale + bias
  kUnary,  // UnaryParams
  kClamp,
  kFill,
  kGlu,  // GluParams
  kSoftcap,
  // Reductions and selection.
  kSumRows,
  kSoftMax,
  kArgmax,
  kArgsort,
  kTopK,
  // Routing.
  kRoute,         // RouteParams: scores into selected experts and weights
  kExpertReduce,  // the selected experts' outputs, weighted and summed
  // Recurrences and convolutions.
  kSsmConv,
  kGatedDeltaNet,
  kConv2d,
  kUpsample,
  // Hyper-connections: composite ops with a mandatory primitive
  // decomposition.
  kHcPre,   // streams folded into one
  kHcPost,  // one output spread back over the streams
  kHcMix,   // HcParams: the mixing weights
  // Sampling and collectives.
  kSample,
  kAllReduce,
  kAllGather,
  kAllToAll,
  kSend,
  kRecv,
  // A pattern's result; FusedKind names it.
  kFused,
  // Exact visibility-bounded selection and its bit-preserving mask.
  kVisibleTopK,
  kSelectedMask,
};

// Each fused kind's sources (in order), outputs and parameters; its
// decomposition (ir/decompose.h) is its meaning.
enum class FusedKind : std::uint16_t {
  kNone,
  kRmsNormMul,      // (x, w): NormParams: rms_norm(x) · w
  kRmsNormMulRope,  // (x, w, pos, freq?): NormRopeParams: ... then rope
  kRmsNormMulAdd,   // (x, w, r): NormParams: rms_norm(x) · w + r
  // (x [D, H, T], pos, cache [D · H, C], idx [T], freq?): RopeParams: the
  // rotated rows stored; the output is a view of cache.
  kRopeStore,
  // (w_gate, w_up, x): GluParams: glu(w_gate · x, w_up · x), the gate's
  // product the GLU's gate (GGML's ggml_glu_split(gate, up) over its
  // mul_mat(w_gate, x) and mul_mat(w_up, x); the weights first, as kMatMulAdd's).
  kMatMulGlu,
  kMatMulAdd,  // (w, x, b): w · x + b
  // (q [D, R, H, owners], mask [C, >= R, 1, owners], k0, ..., v0, ...):
  // OwnerAttnParams: joined requests' rows, each owner's R rows attending to
  // its own cache [D, C_o, Hk, 1] (C_o <= C): owner o's queries q[:, :, :,
  // o] and its mask mask[:C_o, :R, 0, o] (the first C_o cells and R rows
  // of its own matrix, so each owner keeps its mask's row padding);
  // [Dv, H, R, owners], owner o's attention at o. One to four owners.
  // Its cohort and offset (amended 2026-10-09) place the node among a
  // step's joined owners when they run as several nodes of up to four
  // (GGML's four-root carriers): the cohort is every joined owner, the
  // offset the node's first one there. Like a product's accumulation, they
  // are hashed choices that do not change the definition (an
  // implementation partitions its grid by them); 0 and 0 is a lone node.
  kOwnerAttention,
  // A batch key, never a node (the builder and verifier refuse one, and it
  // has no decomposition): consecutive independent kCacheStore nodes that
  // one implementation runs as one launch (planner/pattern.h
  // grouped_stores). A capability of it describes one member store
  // (cache, src, idx) and declares its batch bound (execution/registry.h
  // Capability::batch). Its first form, a node of two or three stores,
  // held none of GGML's groups (two to sixteen) and was dropped
  // (2026-10-09).
  kGroupedStores,
  // (gate [n, U, T], up [n, U, T], ids [U, T], s_gate [E], s_up [E]):
  // GluParams: glu(gate · s_gate[ids], up · s_up[ids]).
  kMoeGlu,
  // (down [n, U, T], ids, weights [U, T], shared [n, T], shared_gate? [1, T],
  // s_down? [E]): expert_reduce(down, weights, ids, s_down) + shared ·
  // sigmoid(shared_gate), or + shared where the shared expert has no gate
  // (DeepSeek V4's).
  kMoeCombine,
  // (x [W, hc, T], w [W · hc]): NormParams: each stream's rms_norm (over W,
  // Qwen3.8's llmp.hc.norm), then as [W · hc, T], · w. DeepSeek V4's flat
  // norm over all W · hc is kHcNormPost's second output, or rms_norm of the
  // reshape.
  kHcNorm,
  // (y, x, post, comb?): HcNormPostParams: two outputs, hc_post's p and
  // rms_norm(p as [W · hc, T]).
  kHcNormPost,
  // (q, wq, k, wk, pos, freq?): NormRopeParams: two outputs, each
  // rope(rms_norm(·) · w).
  kQkPrep,
  // (w0, w1, ..., x): an output per product wi · x, up to four; the
  // weights first, as kMatMulGlu's and kMatMulAdd's.
  kPreparedProducts,
  kNormGate,           // (x, w, z): NormParams: rms_norm(x) · w · sigmoid(z)
  kLayerNormModulate,  // (x, s): NormParams: layer_norm(x) · (1 + s)
  kGatedResidual,      // (x, gate, y): x + tanh(gate) · y
  // (x, gate, y, s): NormParams: two outputs, x' = x + tanh(gate) · y and
  // layer_norm(x') · (1 + s).
  kGatedResidualNorm,
  kEulerStep,  // (sample, noise, dt [1]): sample + dt · noise
  // (x [2W, 2H, Co], shortcut [W, H, Ci]): DupUpParams: x plus Qwen-Image's
  // DupUp3D of the shortcut at its last kept frame (kernels/image/ops.h
  // AddDupUp): output channel o's sub-pixel (b, d) reads input channel
  // (((o · ft + ft - 1) · 2 + b) · 2 + d) / (Co · ft · 4 / Ci).
  kDupUpAdd,
  // Qwen3.8's fast form (slice G3b, 2026-10-10; still op set v1, nothing
  // released): the hyper-connections, Gated DeltaNet, QSA and MoE pieces
  // llmpalooza's kernels fuse (kernels/ggml/llmp_ops.h), each defined by its
  // decomposition. Several round their outputs where the decomposition's are
  // F32 (a result's recorded encoding, the builder's `encodings`): kHcPrep's
  // normed streams and kScaledUnary's result in BF16 for BF16 products.
  //
  // (x [W, hc, T], norm [W · hc], inject? [W · hc, hc], out? [W, T],
  // logits? [hc, T]): HcNormPostParams: the streams s = x, or with out and
  // logits x combined with the last block's output first (kHcCombine); the
  // normed streams xn = kHcNorm(s, norm), and with inject the next combine's
  // logits inject · xn (from xn in F32). Outputs: xn, then the logits (with
  // inject), then s (with out).
  kHcPrep,
  // (x): ScaledUnaryParams: fn(x · scale).
  kScaledUnary,
  // (xn [W · hc, T], gate [W · hc, T]): HcFoldParams: (1 / hc) Σ_k xn_k ·
  // sigmoid(gate_k) over the streams in order (kHcPre's gated fold of xn
  // as [W, hc, T]), [W, T]; with `prepared` also the result prepared into
  // MXFP8 rows (kPrepare's), with `rounded` also rounded to BF16, in that
  // order after it.
  kHcFold,
  // (x [W, hc, T], out [W, T], logits [hc, T]): HcParams: x + out ·
  // 2 · sigmoid(logits / hc) on each stream (hc.combine's).
  kHcCombine,
  // (x [W, hc, T], norm [W · hc], gate [W · hc, T]): HcNormPostParams:
  // kHcFold of kHcNorm(x, norm) and gate (the norm recomputed: GGML-form
  // checkpoints' llmp.hc.mix).
  kHcMixNorm,
  // (rows [C, T], history [(K - 1) · C], w [K, C]): GdnConvParams: the
  // causal depthwise convolution of the rows after the history (tap j of
  // channel c at c · (K - 1) + j) and SiLU, [C, T]; its first qk_channels
  // channels, heads of `head`, L2-normed as rms_norm(x, eps) · scale.
  kGdnConv,
  // (rows [C, T], history? [(K - 1) · C]): GdnHistoryParams: the
  // convolution's next history, [taps · C, 1]: the last `taps` of the
  // history (when given) followed by the rows, each channel's taps
  // together.
  kGdnHistory,
  // (q, k, v, g, beta, state [S], idx? [1]): GdnStepParams: kGatedDeltaNet
  // over state viewed as [Sk, Sv, H, 1]: its attention, and with `write`
  // the new state stored over state at idx (an output that is a view of
  // state).
  kGdnStep,
  // (alpha [H, T], beta [H, T], dt_bias [H], a [H]): two outputs, the
  // decay's log softplus(alpha + dt_bias) · a and sigmoid(beta), each [1,
  // H, T, 1].
  kGdnGates,
  // (attn [d · H, T], q [2 · d · H, T]): HeadGateParams: attn times
  // sigmoid of each head's gate (q's second d of each head's 2 · d).
  kHeadGate,
  // (raw [D, C], blocks [D, N], w [D], block_pos [4 · B], cells [B],
  // positions? [4 · T]): QsaPoolParams: the block keys of the first B
  // blocks of `ratio` raw rows, each their mean, rms_norm · w, rotated at
  // its block's position and rounded to BF16, stored in blocks at cells:
  // a view of blocks. An implementation may write only the blocks the
  // step's positions complete (the others hold what earlier steps stored):
  // a block's key never changes once its rows are written.
  kQsaPool,
  // (q [d, H, T], k [d, C, Hk], v [d, C, Hk], qi [Di, Hi, T], blocks [Di,
  // B], causal [C, T] F32, seen [B, T] F32, mask [C, >= T], positions? [4 ·
  // T]): QsaAttnParams: QSA's selection and attention (components/qsa.cc):
  // each block's score Σ_h relu(bf16(qi_h) · block), plus 1e9 where `seen`
  // hides the block's last position, spread over its `ratio` cells, plus
  // `causal`; the `width` best cells kept; attention of q over k and v
  // under `mask` and -inf but at the kept cells. [d · H, T].
  kQsaAttention,
  // (logits [E, T], x [W, T], gate [W]): RouteParams: kRoute's ids and
  // weights, and the shared expert's gate logit Σ x · gate [1, T].
  kMoeRouter,
  // (gate_up [W, 2F, E], down [F, W, E], x [W, T], ids [U, T], weights [U,
  // T], shared [W, T], gate [1, T], s_gate [E], s_up [E], s_down [E]):
  // MoeExpertsParams: the routed experts (gate_up's halves, each by its
  // expert's scale, the GLU, down) reduced with s_down and the weights,
  // plus shared · sigmoid(gate): kMoeCombine over them.
  kMoeExperts,
  // (gate [W, F, E], up [W, F, E], x [W, T], ids [U, T]): MoeExpertsParams:
  // glu(gate[ids] · x, up[ids] · x), [F, U, T].
  kMatMulIdGlu,
  // (q [D, H, T], positions [T]): NormRopeParams: the Q heads'
  // rope(rms_norm(q, eps)); no learned norm weight. The primitive formula
  // is general; the fast implementation's capability restricts its rotary
  // tail and head width.
  kDsv4QHead,
  // (state_kv [C, S], state_score [C, S], kv [C, T], score [C, T],
  // read_idxs [ratio * B * (overlap ? 2 : 1)] I32): CompressorParams:
  // each block's rows from [state | current] softmax pooled per channel.
  // Without overlap C = D; with overlap C = 2D, each block takes its
  // previous rows' first D channels, then its current rows' second D.
  // Overlap's source also has a final zero row with score -inf. The
  // indices address this source; [D, 1, B] F32 before norm or rotation.
  kDsv4Compress,
  // (x [W, hc, T], fn [W * hc, (2 + hc) * hc], scale [3],
  // base [(2 + hc) * hc], norm [W]): SinkhornPreParams: the full-carry
  // Sinkhorn block mix. Outputs are the folded stream's weighted RMSNorm
  // [W, T], post [hc, T], and combination [hc, hc, T], all packed F32.
  kDsv4HcPre,
  // (q [D, T, H], raw [D, 1, A], compressed [D, 1, B], mask?, sinks?):
  // AttnParams, unchanged: Attention(q, permute(concat(raw, compressed, 2)),
  // the same K as V, mask, sinks). The formula needs no layout fact;
  // only zero-copy implementations require a proven contiguous state span.
  kStateBankAttention,
};
// The op set's last fused kind (the kinds are 1 to it).
inline constexpr FusedKind kLastFusedKind = FusedKind::kStateBankAttention;

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

inline constexpr std::size_t kMaxParamBytes = 64;

struct NormParams {
  float eps = 0.0f;
};

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
  bool inverse = false;  // rotate by -theta
  bool table = false;    // pos is an F32 cos/sin table, not positions
  std::uint8_t reserved = 0;
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
struct ProductParams {
  Accumulate accumulate = Accumulate::kDefault;
  ProductHint hint = ProductHint::kNone;
  std::array<std::uint8_t, 2> reserved{};
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
// every Gemma FFN) and execution::Operation::kGeGlu.
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
  std::uint32_t iterations = 0;  // kSinkhorn's
  float eps = 0.0f;              // kHcMix's
  float scale = 1.0f;            // kHcPre kGated's
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

struct QsaPoolParams {
  float eps = 0.0f;
  std::uint32_t ratio = 0;
  RopeParams rope{};
};

struct QsaAttnParams {
  AttnParams attn{};
  std::uint32_t width = 0;  // cells kept
  std::uint32_t ratio = 0;  // cells a block
};

struct MoeExpertsParams {
  GluParams glu{};
  std::uint32_t hidden = 0;  // F
};

template <typename P>
inline constexpr bool kIsParams = std::is_trivially_copyable_v<P> && sizeof(P) <= kMaxParamBytes;

static_assert(kIsParams<NormParams> && kIsParams<RopeParams> && kIsParams<MaskParams> &&
              kIsParams<ProductParams> && kIsParams<AttnParams> && kIsParams<UnaryParams> &&
              kIsParams<GluParams> && kIsParams<ScaleParams> && kIsParams<ClampParams> &&
              kIsParams<FillParams> && kIsParams<SoftcapParams> && kIsParams<SoftMaxParams> &&
              kIsParams<ArgmaxParams> && kIsParams<ArgsortParams> && kIsParams<TopKParams> &&
              kIsParams<SelectedMaskParams> && kIsParams<ExpertSortParams> &&
              kIsParams<RouteParams> && kIsParams<HcParams> && kIsParams<ConcatParams> &&
              kIsParams<PermuteParams> && kIsParams<Conv2dParams> && kIsParams<UpsampleParams> &&
              kIsParams<PoolParams> && kIsParams<SampleParams> && kIsParams<CollectiveParams> &&
              kIsParams<NormRopeParams> && kIsParams<OwnerAttnParams> &&
              kIsParams<HcNormPostParams> && kIsParams<DupUpParams> &&
              kIsParams<ScaledUnaryParams> && kIsParams<HcFoldParams> && kIsParams<GdnConvParams> &&
              kIsParams<GdnHistoryParams> && kIsParams<GdnStepParams> &&
              kIsParams<HeadGateParams> && kIsParams<QsaPoolParams> && kIsParams<QsaAttnParams> &&
              kIsParams<MoeExpertsParams> && kIsParams<CompressorParams> &&
              kIsParams<SinkhornPreParams>);
static_assert(sizeof(CompressorParams) == 8 && sizeof(ScaledUnaryParams) == 8 &&
              sizeof(HcFoldParams) == 8 && sizeof(GdnConvParams) == 16 &&
              sizeof(GdnStepParams) == 4 && sizeof(QsaPoolParams) == 8 + sizeof(RopeParams) &&
              sizeof(QsaAttnParams) == sizeof(AttnParams) + 8 &&
              sizeof(MoeExpertsParams) == sizeof(GluParams) + 4);
static_assert(sizeof(SinkhornPreParams) == 16);

// No implicit padding: each struct's size is the sum of its fields'.
static_assert(sizeof(RopeParams) == 4 + (3 * 4) + (6 * 4) + 16);
static_assert(sizeof(MaskParams) == 4 + 4 + 4);
static_assert(sizeof(ProductParams) == 4);
static_assert(sizeof(AttnParams) == (4 * 4) + 4 + 4);
static_assert(sizeof(GluParams) == 4 + 4);
static_assert(sizeof(RouteParams) == 4 + (3 * 4));
static_assert(sizeof(HcParams) == 4 + (4 * 4));
static_assert(sizeof(PoolParams) == 4 + 4);
static_assert(sizeof(SampleParams) == 4 + (4 * 4));
static_assert(sizeof(CollectiveParams) == (2 * 4) + 4);
static_assert(sizeof(NormRopeParams) == 4 + sizeof(RopeParams));
static_assert(sizeof(OwnerAttnParams) == sizeof(AttnParams) + (3 * 4));

}  // namespace llmp::ir

#endif  // LLMP_IR_OP_H_
