// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// A port of llama.cpp b29c606e2's src/models/qwen4exp.cpp graph (with the
// llm_graph_context and delta-net parts it calls), which is MIT: its
// structure, helpers and parameters follow upstream's closely, so the file
// carries GGML's notice (docs/licensing.md). qwen38_graph.h lists where it
// departs from upstream.

#include "kernels/ggml/qwen38_graph.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ggml.h"
#include "kernels/ggml/dsv4_graph.h"
#include "kernels/ggml/fusion.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/moe_layout.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate_ext.h"
#include "model/qwen38.h"

namespace jitllm::kernels::ggml {
namespace {

std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

// A non-negative extent as a byte multiplier.
std::size_t U(std::int64_t n) { return static_cast<std::size_t>(n); }

class Builder {
 public:
  Builder(ggml_context* c, const model::Qwen38Profile& p, const model::Qwen38Binding& b,
          const Qwen38ChunkShape& s, Qwen38Graph& g, bool fused, bool exact, bool cutlass,
          bool verify = false, bool export_streams = false, std::uint64_t capture_routed = 0)
      : c_(c),
        p_(p),
        b_(b),
        s_(s),
        g_(g),
        fused_(fused),
        fast_(fused && !exact),
        gguf_(b.gguf()),
        cutlass_(cutlass),
        verify_(verify),
        export_(export_streams),
        capture_routed_(capture_routed) {}

  std::expected<void, KernelFailure> Leaves(const Qwen38GraphOptions& options);
  void Build();
  // One MTP drafter pass over this builder's shape and inputs (g_'s, its
  // layers[0] the drafter's layer): `tokens` I32 [rows] (the next tokens,
  // or the previous pass's draft), `hidden` F32 [hc_width, rows] the
  // streams it reads. With `head`, the combined streams and the last row's
  // draft over the head's first `head_rows` rows (0: all); without, only
  // its caches' writes (a prefill pass), both null.
  struct MtpOut {
    ggml_tensor* streams = nullptr;      // F32 [width, hc, rows]
    ggml_tensor* draft = nullptr;        // I32 [1]
    ggml_tensor* probability = nullptr;  // I32 [1]: the draft's softmax probability's F32 bits
    ggml_tensor* head_input = nullptr;
    ggml_tensor* head_logits = nullptr;
  };
  MtpOut MtpPass(const Qwen38MtpGraph& m, ggml_tensor* tokens, ggml_tensor* hidden, bool head,
                 std::int64_t head_rows, bool confidence);
  const std::vector<ggml_tensor*>& expanded() const { return expanded_; }
  // Whether this shape's QSA selection runs on the device (no host masks).
  bool SelectsOnDevice() const { return DeviceSelect(); }

 private:
  void Name(ggml_tensor* t, std::string_view name, int il) {
    g_.named.emplace_back(il >= 0 ? std::format("{}-{}", name, il) : std::string(name), t);
  }
  void Expand(ggml_tensor* t) { expanded_.push_back(t); }

  // GGML's row norms take packed input: a view of heads is made packed first
  // (every stride a dense tensor's, even over extents of 1, which
  // ggml_is_contiguous overlooks).
  ggml_tensor* Packed(ggml_tensor* x) {
    bool packed = x->nb[0] == ggml_type_size(x->type);
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
      packed = packed && x->nb[i] == x->nb[i - 1] * static_cast<std::size_t>(x->ne[i - 1]);
    }
    return packed ? x : ggml_cont(c_, x);
  }
  // build_norm with LLM_NORM_RMS: rows scaled to unit RMS, then the weight.
  ggml_tensor* Norm(ggml_tensor* x, ggml_tensor* weight) {
    return ggml_mul(c_, ggml_rms_norm(c_, Packed(x), p_.rms_eps), weight);
  }
  // build_gdn_l2_norm (models.h): rms_norm(x, eps / n) / sqrt(n).
  ggml_tensor* L2Norm(ggml_tensor* x) {
    const auto n = static_cast<float>(x->ne[0]);
    return ggml_scale(c_, ggml_rms_norm(c_, Packed(x), p_.rms_eps / n), 1.0f / std::sqrt(n));
  }
  ggml_tensor* Rope(ggml_tensor* x, ggml_tensor* pos) {
    std::array<int, 4> sections = p_.rope_sections;
    return ggml_rope_multi(c_, x, pos, nullptr, static_cast<int>(p_.rope_dims), sections.data(),
                           GGML_ROPE_TYPE_IMROPE, 262144, p_.rope_base, 1.0f, 0.0f, 1.0f, 32.0f,
                           1.0f);
  }
  // A product's input x [k, t] (packed) and its conversions, each made once
  // for every product that reads it: BF16 (the fused graph's float products
  // past kQwen38Bf16Rows) and MXFP8 (the fast graph's tensor-core products).
  struct Input {
    ggml_tensor* x = nullptr;
    ggml_tensor* bf16 = nullptr;
    ggml_tensor* mxfp8 = nullptr;
  };
  Input In(ggml_tensor* x) {
    if (!ggml_is_contiguous(x)) {
      x = ggml_cont(c_, x);
    }
    if (x->ne[2] != 1 || x->ne[3] != 1) {
      x = ggml_reshape_2d(c_, x, x->ne[0], ggml_nelements(x) / x->ne[0]);
    }
    return {.x = x};
  }
  // x in BF16 for the float products, or null where GGML's product takes x.
  ggml_tensor* Bf16Of(Input& in) {
    if (in.bf16 == nullptr && Bf16Inputs(in.x->ne[1])) {
      in.bf16 = ToBf16(c_, in.x);
    }
    return in.bf16;
  }
  // An MXFP8 product y[n, t] = W x: jitLLM's vector product up to its
  // column bound; past it, in the fast graph, the tensor-core product
  // (`out` F32 or BF16), else the weights dequantized to BF16 for jitLLM's
  // BF16 product (the fused graph) or GGML's (the unfused one). F32 unless
  // `out` is BF16 and the tensor-core product runs.
  ggml_tensor* Linear(const Qwen38Mxfp8Tensors& w, Input& in, ggml_type out = GGML_TYPE_F32) {
    if (w.bf16 != nullptr) {
      return LinearBf16(w.bf16, in, out);
    }
    if (w.matrix != nullptr) {
      // A GGUF checkpoint's matrix: GGML's product of its type (F32 out),
      // a BF16 one's in the fused graph as the drafter's.
      if (w.matrix->type == GGML_TYPE_BF16 && fused_) {
        return LinearBf16(w.matrix, in, out);
      }
      return Mm(w.matrix, in.x);
    }
    if (in.x->ne[1] <= kMxfp8VecColumns) {
      return Mxfp8MulMatVec(c_, w.codes, w.scales, in.x);
    }
    if (fast_) {
      if (in.mxfp8 == nullptr) {
        in.mxfp8 = Mxfp8Quantize(c_, in.x);
      }
      return LinearMxfp8(w, in.mxfp8, in.x->ne[1], out);
    }
    if (ggml_tensor* x_bf16 = Bf16Of(in); x_bf16 != nullptr) {
      return GemmBf16(c_, Mxfp8Dequant(c_, w.codes, w.scales), x_bf16);
    }
    return ggml_mul_mat(c_, Mxfp8Dequant(c_, w.codes, w.scales), in.x);
  }
  ggml_tensor* Linear(const Qwen38Mxfp8Tensors& w, ggml_tensor* x) {
    Input in = In(x);
    return Linear(w, in);
  }
  // A BF16 linear (the MTP drafter's): GGML's float product (MMVF or MMF)
  // up to kQwen38Bf16Rows rows, F32 out; past them jitllm.gemm.bf16 over x
  // in BF16 (made once for every product that reads it), `out` out.
  ggml_tensor* LinearBf16(ggml_tensor* w, Input& in, ggml_type out = GGML_TYPE_F32) {
    if (in.x->ne[1] <= kQwen38Bf16Rows) {
      return ggml_mul_mat(c_, w, in.x);
    }
    if (in.bf16 == nullptr) {
      in.bf16 = ToBf16(c_, in.x);
    }
    return GemmBf16(c_, w, in.bf16, out);
  }
  // The tensor-core product over `rows` activations already quantized to
  // MXFP8 (a fusion's output).
  ggml_tensor* LinearMxfp8(const Qwen38Mxfp8Tensors& w, ggml_tensor* mxfp8, std::int64_t rows,
                           ggml_type out = GGML_TYPE_F32) {
    return Mxfp8Gemm(c_, mxfp8, w.codes, Mxfp8Swizzle(c_, w.scales), out, rows);
  }
  // The type the fast graph's products into the recurrences' rows give:
  // BF16 past the vector product's columns.
  ggml_type RowsType(std::int64_t rows) const {
    return fast_ && rows > kMxfp8VecColumns ? GGML_TYPE_BF16 : GGML_TYPE_F32;
  }
  // A product with BF16 weights: jitLLM's over x in BF16 where the fused
  // graph converts it, else GGML's (and GGML's for any other type, a GGUF
  // checkpoint's).
  ggml_tensor* MulMat(ggml_tensor* w, Input& in) {
    ggml_tensor* x_bf16 = w->type == GGML_TYPE_BF16 ? Bf16Of(in) : nullptr;
    return x_bf16 != nullptr ? GemmBf16(c_, w, x_bf16) : ggml_mul_mat(c_, w, in.x);
  }
  // Whether a linear is the ModelOpt checkpoint's MXFP8 (the fast form's
  // tensor-core products and their fused quantizations read it).
  static bool Mxfp8(const Qwen38Mxfp8Tensors& w) { return w.codes != nullptr; }
  // A GGUF checkpoint's fast form up to kVecQTokens rows (decode): a
  // quantized matrix of a type jitllm.vecq takes is its product over the
  // input's one Q8_1 quantization (DeepSeek's fast plan's, jitllm_ops.h),
  // every product of an input sharing it; anything else GGML's product.
  bool VecQFits(const ggml_tensor* w, const ggml_tensor* x) const {
    return gguf_ && fast_ && w != nullptr && VecQType(w->type) && w->ne[2] == 1 && w->ne[3] == 1 &&
           x->type == GGML_TYPE_F32 && x->ne[0] == w->ne[0] && x->ne[1] <= kVecQTokens &&
           x->ne[2] == 1 && x->ne[3] == 1 && ggml_is_contiguous(x) && x->ne[0] % 32 == 0;
  }
  ggml_tensor* Q8Of(ggml_tensor* x) {
    if (const auto found = q8_.find(x); found != q8_.end()) {
      return found->second;
    }
    ggml_tensor* q = QuantizeQ8(c_, x);
    q8_.emplace(x, q);
    return q;
  }
  ggml_tensor* Mm(ggml_tensor* w, ggml_tensor* x) {
    if (!VecQFits(w, x)) {
      return ggml_mul_mat(c_, w, x);
    }
    return VecQ(c_, w, Q8Of(x), nullptr, x->ne[1], false);
  }
  // Whether the fused graph gives a float product of `rows` BF16 inputs.
  bool Bf16Inputs(std::int64_t rows) const { return fused_ && rows > kQwen38Bf16Rows; }
  // Whether the fast graph's QSA fusions (jitllm.qsa.prep and .select) take
  // the profile's heads: a 64-dimension rotation, heads of whole warps.
  bool FastSelect() const {
    return fast_ && p_.rope_dims == 64 && p_.indexer_head_dim % 32 == 0 &&
           p_.indexer_head_dim >= 64 && p_.indexer_head_dim <= 512 && p_.head_dim % 32 == 0 &&
           p_.head_dim >= 64 && p_.head_dim <= 512;
  }
  // Whether the fast graph caches block keys, selects and attends sparsely
  // (jitllm.qsa.pool, .topk and .attn): the profile's heads are the kernels'.
  bool Sparse() const {
    return FastSelect() && p_.head_dim == kQsaAttnHead && p_.indexer_head_dim == kQsaIndexDim &&
           p_.indexer_heads == kQsaIndexHeads && p_.kv_heads > 0 && p_.heads % p_.kv_heads == 0 &&
           p_.heads / p_.kv_heads <= 16 && p_.indexer_ratio > 0 && p_.indexer_ratio <= 32;
  }
  // Whether this chunk's QSA selection runs on the device (jitllm.qsa.topk,
  // then jitllm.qsa.attn over the kept cells): the sparse graph's, within
  // the selection's tiles. Otherwise GGML's top-k selects over the host's
  // masks and attention reads every cell under them.
  bool DeviceSelect() const {
    const QsaTopKLayout layout{.t = s_.rows,
                               .n_blocks = s_.qsa_blocks,
                               .width = std::int64_t{p_.indexer_budget} + p_.indexer_ratio - 1,
                               .ratio = p_.indexer_ratio};
    return s_.qsa_select && Sparse() && layout.tiles() <= kQsaTopKMaxTiles &&
           layout.tiles() * layout.candidates() <= kQsaTopKCandidates;
  }
  // build_lora_mm_id with a per-expert scale (llama-graph.cpp:1545-1581),
  // or none (a GGUF checkpoint's experts).
  ggml_tensor* MulMatId(ggml_tensor* w, ggml_tensor* x, ggml_tensor* ids, ggml_tensor* scale) {
    ggml_tensor* res = ggml_mul_mat_id(c_, w, x, ids);
    if (scale == nullptr) {
      return res;
    }
    const std::int64_t n_expert = scale->ne[0];
    const std::int64_t nt = x->ne[2];
    ggml_tensor* s = ggml_reshape_3d(c_, scale, 1, n_expert, 1);
    s = ggml_repeat_4d(c_, s, 1, n_expert, nt, 1);
    s = ggml_get_rows(c_, s, ids);
    return ggml_mul(c_, res, s);
  }
  // A state tensor's single row rewritten from `src` (packed, the row's size).
  ggml_tensor* StoreState(ggml_tensor* state, ggml_tensor* src) {
    ggml_tensor* rows = ggml_reshape_2d(c_, src, state->ne[0], 1);
    return ggml_set_rows(c_, state, rows, g_.state_row);
  }

  // A packed `type` [ne0, ne1] view of a byte blob a jitLLM operation wrote
  // (several outputs laid out in one tensor), from `offset`.
  ggml_tensor* TypedView(ggml_tensor* blob, ggml_type type, std::int64_t ne0, std::int64_t ne1,
                         std::size_t offset) {
    const std::size_t row = ggml_row_size(type, ne0);
    // ggml_view_1d counts the blob's own elements (bytes, or I32 words).
    ggml_tensor* v = ggml_view_1d(
        c_, blob, static_cast<std::int64_t>(row * U(ne1) / ggml_type_size(blob->type)), offset);
    v->type = type;
    v->ne[0] = ne0;
    v->ne[1] = ne1;
    v->nb[0] = ggml_type_size(type);
    v->nb[1] = row;
    v->nb[2] = row * U(ne1);
    v->nb[3] = v->nb[2];
    return v;
  }
  ggml_tensor* HcMix(ggml_tensor* x, ggml_tensor* w_norm, ggml_tensor* w_down, ggml_tensor* w_up,
                     ggml_tensor* w_inject, ggml_tensor** inject, int il);
  // The fast graph's mix (jitllm.hc.prep, .lo, .mix_bf16): `res` the streams,
  // or, given the previous block's output `out` and its combine logits
  // `logits`, the streams before that combine, which is done first and
  // `res` set to its result. Returns the mixed input; `inject`, if given,
  // the next combine's logits.
  // Past the vector product's columns the mix also gives its output in
  // MXFP8 for the products and, with `bf16`, in BF16 (the router's).
  Input HcFast(ggml_tensor*& res, ggml_tensor* out, ggml_tensor* logits, ggml_tensor* w_norm,
               ggml_tensor* w_down, ggml_tensor* w_up, ggml_tensor* w_inject, ggml_tensor** inject,
               bool bf16, int il);
  void BuildFast(ggml_tensor* res, ggml_tensor* ple);
  ggml_tensor* HcCombine(ggml_tensor* residual, ggml_tensor* block_out, ggml_tensor* inject);
  // With `store` false (a verify's) the new history is not stored.
  ggml_tensor* ConvStateAt(ggml_tensor* state, ggml_tensor* x, std::int64_t cols,
                           std::int64_t channels, bool store = true);
  // A verify's save of `src` [width, rows] into `into` at its rows.
  void Save(ggml_tensor* into, ggml_tensor* src, std::int64_t width) {
    Expand(ggml_set_rows(c_, into, ggml_reshape_2d(c_, Packed(src), width, s_.rows), g_.row_ids));
  }
  ggml_tensor* Ple(const Qwen38LayerTensors& l, ggml_tensor* emb, ggml_tensor* hidden, int il);
  // The blocks read `cur`, or with `pre` its conversions made already.
  ggml_tensor* LinearAttention(const Qwen38LayerTensors& l, ggml_tensor* cur, int il,
                               const Input* pre = nullptr);
  ggml_tensor* QsaTopK(const Qwen38LayerTensors& l, Input& cur, int il);
  ggml_tensor* Attention(const Qwen38LayerTensors& l, ggml_tensor* cur, int il,
                         const Input* pre = nullptr);
  ggml_tensor* Moe(const Qwen38LayerTensors& l, ggml_tensor* cur, int il,
                   const Input* pre = nullptr);

  ggml_context* c_;
  const model::Qwen38Profile& p_;
  const model::Qwen38Binding& b_;
  const Qwen38ChunkShape& s_;
  Qwen38Graph& g_;
  bool fused_;
  bool fast_;
  // A GGUF checkpoint's artifact (qwen38_graph.h): GGML's products, the
  // reference form's hyper-connection fusions, no MXFP8 or NVFP4 operation.
  bool gguf_;
  bool cutlass_;
  bool verify_;
  bool export_;
  std::uint64_t capture_routed_;
  ggml_tensor* capture_attention_input_ = nullptr;
  ggml_tensor* capture_attention_projection_ = nullptr;
  std::vector<ggml_tensor*> expanded_;
  // Each input's Q8_1 quantization (Q8Of), made once.
  std::unordered_map<const ggml_tensor*, ggml_tensor*> q8_;
};

std::expected<ggml_tensor*, KernelFailure> Leaf(ggml_context* c, const model::Qwen38Tensor& t,
                                                std::string_view role) {
  ggml_type type = GGML_TYPE_COUNT;
  if (t.plain) {
    // Checkpoint-layout bytes: I8 to GGML; F32 and I64 keep their type.
    if (t.type == "F8_E4M3" || t.type == "U8") {
      type = GGML_TYPE_I8;
    } else if (t.type == "F32") {
      type = GGML_TYPE_F32;
    } else if (t.type == "I64") {
      type = GGML_TYPE_I64;
    } else {
      return Rejected(std::format("{}: plain {} is not a type the graph reads", role, t.type));
    }
  } else {
    auto parsed = GgmlTypeOf(t.type);
    if (!parsed) {
      return std::unexpected(parsed.error());
    }
    type = *parsed;
  }
  std::array<std::int64_t, 4> ne = {1, 1, 1, 1};
  if (t.ne.empty() || t.ne.size() > 4) {
    return Rejected(std::format("{}: not a GGML shape", role));
  }
  for (std::size_t i = 0; i < t.ne.size(); ++i) {
    ne[i] = static_cast<std::int64_t>(t.ne[i]);
  }
  if (ne[0] % ggml_blck_size(type) != 0) {
    return Rejected(std::format("{}: rows are not whole blocks", role));
  }
  ggml_tensor* out = ggml_new_tensor(c, type, static_cast<int>(t.ne.size()), ne.data());
  // A quantized matrix whose rows are not whole 512-element steps (a GGUF
  // checkpoint's 320- and 640-element rows): GGML's products read past its
  // last row up to the next step. Marked where the artifact reserves and
  // zeroes those bytes in the resource's readable range
  // (docs/artifact-format.md, the GGML over-read rule); otherwise the
  // products refuse the short rows (validate_ext.h).
  if (!t.plain && ggml_is_quantized(type) && ne[0] % 512 != 0 && t.readable != 0) {
    const std::uint64_t over = ggml_row_size(type, 512 - (ne[0] % 512));
    if (t.readable >= ggml_nbytes(out) + over) {
      MarkRowPaddingReadable(out);
    }
  }
  return out;
}

// A GGUF checkpoint's matrix (`t` any GGML type): refused unless this
// build's products take its type (validate_ext.h's quantized types, or
// F32, F16 and BF16).
std::expected<ggml_tensor*, KernelFailure> MatrixLeaf(ggml_context* c, const model::Qwen38Tensor& t,
                                                      std::string_view role) {
  auto made = Leaf(c, t, role);
  if (!made) {
    return made;
  }
  const ggml_type type = (*made)->type;
  if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16 &&
      !IsQuantizedWeightType(type)) {
    return Rejected(
        std::format("{}: {} is not a type this build's matrix products take", role, t.type));
  }
  return made;
}

std::expected<ggml_tensor*, KernelFailure> LinearLeaf(ggml_context* c, const model::Qwen38Mxfp8& t,
                                                      Qwen38Mxfp8Tensors& into,
                                                      std::string_view role) {
  if (t.is_bf16()) {
    auto made = Leaf(c, t.bf16, role);
    if (made) {
      into.bf16 = *made;
    }
    return made;
  }
  if (t.is_matrix()) {
    auto made = MatrixLeaf(c, t.matrix, role);
    if (made) {
      into.matrix = *made;
    }
    return made;
  }
  auto codes = Leaf(c, t.codes, role);
  if (!codes) {
    return codes;
  }
  into.codes = *codes;
  auto scales = Leaf(c, t.scales, role);
  if (scales) {
    into.scales = *scales;
  }
  return scales;
}

// One layer's weights (and, for a QSA layer, its caches over `cells`): the
// target's layer `il`, or the MTP drafter's (`ple` false), its experts at
// `stride`.
std::expected<void, KernelFailure> LayerLeaves(ggml_context* c, const model::Qwen38Profile& p,
                                               const model::Qwen38Layer& w, Qwen38LayerTensors& l,
                                               bool ple, bool cutlass, bool cutlass_artifact,
                                               std::uint64_t stride, std::int64_t cells,
                                               std::uint32_t il) {
  const auto leaf = [&](ggml_tensor*& into, const model::Qwen38Tensor& t,
                        std::string_view role) -> std::expected<void, KernelFailure> {
    auto made = Leaf(c, t, role);
    if (!made) {
      return std::unexpected(made.error());
    }
    into = *made;
    return {};
  };
  const auto mx = [&](Qwen38Mxfp8Tensors& into, const model::Qwen38Mxfp8& t,
                      std::string_view role) -> std::expected<void, KernelFailure> {
    auto made = LinearLeaf(c, t, into, role);
    if (!made) {
      return std::unexpected(made.error());
    }
    return {};
  };
#define JITLLM_LEAF(into, tensor)                       \
  if (auto made = leaf(into, tensor, #tensor); !made) { \
    return made;                                        \
  }
#define JITLLM_MX(into, tensor)                       \
  if (auto made = mx(into, tensor, #tensor); !made) { \
    return made;                                      \
  }
  // A product's weights: any type the products take (MatrixLeaf), which
  // the ModelOpt binding's fixed BF16 and F32 types always are.
  const auto matrix = [&](ggml_tensor*& into, const model::Qwen38Tensor& t,
                          std::string_view role) -> std::expected<void, KernelFailure> {
    auto made = MatrixLeaf(c, t, role);
    if (!made) {
      return std::unexpected(made.error());
    }
    into = *made;
    return {};
  };
#define JITLLM_MATRIX(into, tensor)                       \
  if (auto made = matrix(into, tensor, #tensor); !made) { \
    return made;                                          \
  }
  JITLLM_LEAF(l.hc_attn_norm, w.hc_attn_norm)
  JITLLM_MATRIX(l.hc_attn_down, w.hc_attn_down)
  JITLLM_MATRIX(l.hc_attn_up, w.hc_attn_up)
  JITLLM_MATRIX(l.hc_attn_inject, w.hc_attn_inject)
  JITLLM_LEAF(l.hc_ffn_norm, w.hc_ffn_norm)
  JITLLM_MATRIX(l.hc_ffn_down, w.hc_ffn_down)
  JITLLM_MATRIX(l.hc_ffn_up, w.hc_ffn_up)
  JITLLM_MATRIX(l.hc_ffn_inject, w.hc_ffn_inject)
  if (w.linear) {
    JITLLM_MX(l.qkv, w.qkv)
    JITLLM_MX(l.z, w.z)
    JITLLM_MX(l.beta, w.beta)
    JITLLM_MX(l.alpha, w.alpha)
    JITLLM_MX(l.ssm_out, w.ssm_out)
    JITLLM_LEAF(l.dt_bias, w.dt_bias)
    JITLLM_LEAF(l.ssm_a, w.ssm_a)
    JITLLM_LEAF(l.conv1d, w.conv1d)
    JITLLM_LEAF(l.ssm_norm, w.ssm_norm)
    const std::int64_t d = p.lin_head_dim;
    l.conv_state =
        ggml_new_tensor_2d(c, GGML_TYPE_F32, std::int64_t{p.conv - 1} * p.conv_channels(), 1);
    l.recurrent = ggml_new_tensor_2d(c, GGML_TYPE_F32, d * d * p.lin_v_heads, 1);
  } else {
    JITLLM_MX(l.q, w.q)
    JITLLM_MX(l.k, w.k)
    JITLLM_MX(l.v, w.v)
    JITLLM_MX(l.o, w.o)
    if (w.idx_q.is_matrix()) {
      // A GGUF checkpoint's split indexer projection.
      JITLLM_MX(l.idx_q, w.idx_q)
      JITLLM_MX(l.idx_k, w.idx_k)
    } else {
      JITLLM_MX(l.idx_qk, w.idx_qk)
    }
    JITLLM_LEAF(l.q_norm, w.q_norm)
    JITLLM_LEAF(l.k_norm, w.k_norm)
    JITLLM_LEAF(l.idx_q_norm, w.idx_q_norm)
    JITLLM_LEAF(l.idx_k_norm, w.idx_k_norm)
    const std::int64_t kv = std::int64_t{p.head_dim} * p.kv_heads;
    l.cache_k = ggml_new_tensor_2d(c, GGML_TYPE_F16, kv, cells);
    l.cache_v = ggml_new_tensor_2d(c, GGML_TYPE_F16, kv, cells);
    l.cache_idx = ggml_new_tensor_2d(c, GGML_TYPE_F32, p.indexer_head_dim, cells);
    l.cache_pool = ggml_new_tensor_2d(c, GGML_TYPE_BF16, p.indexer_head_dim,
                                      (cells + p.indexer_ratio - 1) / p.indexer_ratio);
  }
  if (ple) {
    JITLLM_MATRIX(l.ple_key, w.ple_key)
    JITLLM_MATRIX(l.ple_value, w.ple_value)
    JITLLM_LEAF(l.ple_norm_key, w.ple_norm_key)
    JITLLM_LEAF(l.ple_norm_query, w.ple_norm_query)
    JITLLM_LEAF(l.ple_norm_conv, w.ple_norm_conv)
    JITLLM_LEAF(l.ple_conv1d, w.ple_conv1d)
    l.ple_state =
        ggml_new_tensor_2d(c, GGML_TYPE_F32, std::int64_t{p.ple_history()} * p.hc_width(), 1);
  }
  JITLLM_MATRIX(l.router, w.router)
  JITLLM_LEAF(l.shared_gate, w.shared_gate)
  if (l.shared_gate->type != GGML_TYPE_F32 && l.shared_gate->type != GGML_TYPE_BF16) {
    return Rejected(std::format("layer {}: the shared expert's gate is {}, not F32 or BF16", il,
                                w.shared_gate.type));
  }
  JITLLM_MX(l.gate_shexp, w.gate_shexp)
  JITLLM_MX(l.up_shexp, w.up_shexp)
  JITLLM_MX(l.down_shexp, w.down_shexp)
  // NVFP4's per-expert global scales; a GGUF checkpoint's experts have none.
  if (!w.gate_exps_scale.type.empty()) {
    JITLLM_LEAF(l.gate_exps_scale, w.gate_exps_scale)
    JITLLM_LEAF(l.up_exps_scale, w.up_exps_scale)
    JITLLM_LEAF(l.down_exps_scale, w.down_exps_scale)
  }
#undef JITLLM_LEAF
#undef JITLLM_MX
#undef JITLLM_MATRIX
  const auto experts = [&](ggml_tensor*& into,
                           const model::Qwen38Tensor& t) -> std::expected<void, KernelFailure> {
    auto type = GgmlTypeOf(t.type);
    if (!type) {
      return std::unexpected(type.error());
    }
    if (!IsQuantizedWeightType(*type)) {
      return Rejected(
          std::format("layer {}: routed experts of {}, which this build's expert "
                      "products do not take",
                      il, t.type));
    }
    into = ggml_new_tensor_3d(c, *type, static_cast<std::int64_t>(t.ne[0]),
                              static_cast<std::int64_t>(t.ne[1]), p.experts);
    const std::size_t slice = into->nb[2];
    if (stride != 0) {
      if (stride < slice || stride % ggml_type_size(*type) != 0 || stride % 16 != 0) {
        return Rejected(std::format(
            "layer {}: an expert stride of {} bytes does not hold whole, 16-byte aligned {} "
            "slices",
            il, stride, t.type));
      }
      into->nb[2] = stride;
      into->nb[3] = stride * static_cast<std::size_t>(p.experts);
    }
    // Rows short of a 512-element step read past the slice: into the
    // next slice of the group or, after the last row, into the readable
    // bytes the artifact reserves (and zeroes) for it. Marked only where
    // the stride provably holds them for every expert, the last one's
    // included; otherwise the quantized products refuse the short rows.
    const std::int64_t k = into->ne[0];
    if (k % 512 != 0) {
      const std::uint64_t over = ggml_row_size(*type, 512 - (k % 512));
      if (stride != 0 && t.readable >= slice + over && t.group_offset <= stride &&
          t.readable <= stride - t.group_offset) {
        MarkRowPaddingReadable(into);
      }
    }
    return {};
  };
  if (cutlass) {
    // The slab as bytes: every slot must hold the CUTLASS layout.
    const moe::ExpertLayout layout{.ffn = p.expert_ffn, .width = p.width};
    if (stride == 0 || stride < layout.bytes() || stride % 16 != 0) {
      return Rejected(std::format(
          "layer {}: an expert stride of {} bytes does not hold the CUTLASS layout's {}", il,
          stride, layout.bytes()));
    }
    // An artifact in the CUTLASS layout: its arrays at the layout's
    // offsets, byte for byte (else the slots were converted at load).
    if (cutlass_artifact &&
        (w.gate_up_codes.group_offset != moe::ExpertLayout::gate_up_codes() ||
         w.gate_up_scales.group_offset != layout.gate_up_scales() ||
         w.down_codes.group_offset != layout.down_codes() ||
         w.down_scales.group_offset != layout.down_scales() ||
         w.down_scales.group_offset + (w.down_scales.ne[0] * w.down_scales.ne[1]) !=
             layout.bytes())) {
      return Rejected(std::format(
          "layer {}: the artifact's CUTLASS expert arrays are not at the layout's offsets", il));
    }
    l.experts = ggml_new_tensor_2d(c, GGML_TYPE_I8, static_cast<std::int64_t>(stride), p.experts);
    return {};
  }
  if (cutlass_artifact) {
    return Rejected(
        "the artifact's experts are in the CUTLASS layout, which GGML's products "
        "do not read");
  }
  if (auto e = experts(l.gate_exps, w.gate_exps); !e) {
    return e;
  }
  if (auto e = experts(l.up_exps, w.up_exps); !e) {
    return e;
  }
  return experts(l.down_exps, w.down_exps);
}

std::expected<void, KernelFailure> Builder::Leaves(const Qwen38GraphOptions& options) {
  const std::int64_t n = s_.rows;
  g_.tokens = ggml_new_tensor_1d(c_, GGML_TYPE_I32, n);
  g_.positions = ggml_new_tensor_1d(c_, GGML_TYPE_I32, 4 * n);
  g_.cells = ggml_new_tensor_1d(c_, GGML_TYPE_I64, n);
  // The fast graph's selection makes the attention's mask itself.
  const bool fast_select = DeviceSelect();
  if (!fast_select) {
    g_.mask = ggml_new_tensor_4d(c_, GGML_TYPE_F16, s_.n_kv, n, 1, 1);
  }
  g_.ple_rows = ggml_new_tensor_1d(c_, GGML_TYPE_I32, std::int64_t{p_.ple_heads()} * n);
  g_.state_row = ggml_new_tensor_1d(c_, GGML_TYPE_I64, 1);
  g_.row_zero = ggml_new_tensor_1d(c_, GGML_TYPE_I32, 1);
  g_.out_ids = ggml_new_tensor_1d(c_, GGML_TYPE_I32, s_.outputs);
  if (s_.qsa_select && !fast_select) {
    const std::int64_t blocks = s_.qsa_blocks;
    g_.mask_f32 = ggml_new_tensor_2d(c_, GGML_TYPE_F32, s_.n_kv, n);
    g_.cell_block = ggml_new_tensor_1d(c_, GGML_TYPE_I32, s_.n_kv);
    g_.block_cells = ggml_new_tensor_1d(c_, GGML_TYPE_I32, std::int64_t{p_.indexer_ratio} * blocks);
    g_.block_pos = ggml_new_tensor_1d(c_, GGML_TYPE_I32, 4 * blocks);
    g_.block_bias = ggml_new_tensor_2d(c_, GGML_TYPE_F32, blocks, n);
  }
  if (verify_) {
    g_.row_ids = ggml_new_tensor_1d(c_, GGML_TYPE_I64, n);
  }
  if (export_) {
    if (options.stream_rows < n) {
      return Rejected("the streams' rows cannot hold the chunk's");
    }
    g_.stream_rows = ggml_new_tensor_1d(c_, GGML_TYPE_I64, n);
    g_.streams = ggml_new_tensor_2d(c_, GGML_TYPE_F32, p_.hc_width(), options.stream_rows);
  }

  const auto leaf = [&](ggml_tensor*& into, const model::Qwen38Tensor& t,
                        std::string_view role) -> std::expected<void, KernelFailure> {
    auto made = Leaf(c_, t, role);
    if (!made) {
      return std::unexpected(made.error());
    }
    into = *made;
    return {};
  };
#define JITLLM_LEAF(into, tensor)                       \
  if (auto made = leaf(into, tensor, #tensor); !made) { \
    return made;                                        \
  }
  JITLLM_LEAF(g_.token_embd, b_.token_embd)
  JITLLM_LEAF(g_.ple_table, b_.ple_table)
  if (gguf_) {
    // The tables' lookups: GGML's get_rows of float rows, or of quantized
    // ones in whole 256-value super-blocks (validate_ext.h
    // CheckGetRowsExt); the n-gram table's (160-value rows) by
    // jitllm.qrows.get_rows in a 32-value block type.
    // (Not NVFP4: GGML's get_rows has no case for it.)
    const auto gathers = [](const ggml_tensor* t) {
      return t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16 ||
             (IsQuantizedWeightType(t->type) && t->type != GGML_TYPE_NVFP4 && t->ne[0] % 256 == 0);
    };
    if (!gathers(g_.token_embd)) {
      return Rejected(
          std::format("token_embd: a table of {} rows of {} values, which no row "
                      "lookup here takes",
                      b_.token_embd.type, g_.token_embd->ne[0]));
    }
    if (!QRowsType(g_.ple_table->type) && !gathers(g_.ple_table)) {
      return Rejected(
          std::format("per_layer_token_embd: a table of {} rows of {} values, which "
                      "no row lookup here takes",
                      b_.ple_table.type, g_.ple_table->ne[0]));
    }
  } else {
    JITLLM_LEAF(g_.ple_table_scale, b_.ple_table_scale)
  }
  JITLLM_LEAF(g_.output, b_.output)
  JITLLM_LEAF(g_.output_hc_norm, b_.output_hc_norm)
  JITLLM_LEAF(g_.output_hc_down, b_.output_hc_down)
  JITLLM_LEAF(g_.output_hc_up, b_.output_hc_up)
  for (const auto& [t, role] :
       {std::pair{g_.output, "output"}, std::pair{g_.output_hc_down, "output_hc_down"},
        std::pair{g_.output_hc_up, "output_hc_up"}}) {
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16 &&
        !IsQuantizedWeightType(t->type)) {
      return Rejected(std::format("{}: {} is not a type this build's matrix products take", role,
                                  ggml_type_name(t->type)));
    }
  }
  if (!options.expert_stride.empty() && options.expert_stride.size() != p_.layers) {
    return Rejected("an expert stride per layer, or none");
  }
#undef JITLLM_LEAF
  g_.layers.resize(p_.layers);
  const bool cutlass = options.experts == Qwen38GraphOptions::Experts::kCutlass;
  for (std::uint32_t il = 0; il < p_.layers; ++il) {
    const model::Qwen38Layer& w = b_.layers[il];
    Qwen38LayerTensors& l = g_.layers[il];
    const std::uint64_t stride = options.expert_stride.empty() ? 0 : options.expert_stride[il];
    if (auto made = LayerLeaves(c_, p_, w, l, il == p_.ple_layer, cutlass, b_.cutlass(), stride,
                                s_.cells, il);
        !made) {
      return made;
    }
    if (verify_ && w.linear) {
      const std::int64_t channels = p_.conv_channels();
      l.commit_conv = ggml_new_tensor_2d(c_, GGML_TYPE_F32, channels, n);
      l.commit_qkv = ggml_new_tensor_2d(c_, GGML_TYPE_F32, channels, n);
      l.commit_gate = ggml_new_tensor_2d(c_, GGML_TYPE_F32, p_.lin_v_heads, n);
      l.commit_beta = ggml_new_tensor_2d(c_, GGML_TYPE_F32, p_.lin_v_heads, n);
    }
    if (verify_ && il == p_.ple_layer) {
      l.commit_ple = ggml_new_tensor_2d(c_, GGML_TYPE_F32, p_.hc_width(), n);
    }
  }
  return {};
}

// build_hc_mix (qwen4exp.cpp:266-312).
ggml_tensor* Builder::HcMix(ggml_tensor* x, ggml_tensor* w_norm, ggml_tensor* w_down,
                            ggml_tensor* w_up, ggml_tensor* w_inject, ggml_tensor** inject,
                            int il) {
  const std::int64_t hc = p_.hc;
  const std::int64_t hc_dim = p_.hc_width();
  const std::int64_t nt = x->ne[2];
  const std::int64_t n_embd = p_.width;
  if (fused_) {
    // The norm (in BF16 for cuBLAS's products), the products, then the
    // gate and fold over the streams, the norm recomputed (jitllm.hc.mix).
    // A GGUF checkpoint's mixer weights (Q8_0, an F32 inject) take GGML's
    // products over the F32 norm.
    const bool bf16 = Bf16Inputs(nt) && w_down->type == GGML_TYPE_BF16 &&
                      (w_inject == nullptr || w_inject->type == GGML_TYPE_BF16);
    ggml_tensor* xn = HcNorm(c_, x, w_norm, p_.rms_eps, bf16 ? GGML_TYPE_BF16 : GGML_TYPE_F32);
    ggml_tensor* lo = bf16 ? GemmBf16(c_, w_down, xn) : Mm(w_down, xn);
    lo = ggml_silu(c_, ggml_scale(c_, lo, 1.0f / static_cast<float>(hc)));
    ggml_tensor* gate = Mm(w_up, lo);
    ggml_tensor* mixed = ggml::HcMix(c_, x, w_norm, gate, p_.rms_eps);
    Name(mixed, "hc_mixed", il);
    if (inject != nullptr) {
      *inject = bf16 ? GemmBf16(c_, w_inject, xn) : Mm(w_inject, xn);
    }
    return mixed;
  }
  ggml_tensor* xn = ggml_rms_norm(c_, x, p_.rms_eps);
  xn = ggml_reshape_2d(c_, xn, hc_dim, nt);
  xn = ggml_mul(c_, xn, w_norm);
  ggml_tensor* lo = ggml_mul_mat(c_, w_down, xn);
  lo = ggml_silu(c_, ggml_scale(c_, lo, 1.0f / static_cast<float>(hc)));
  ggml_tensor* gate = ggml_sigmoid(c_, ggml_mul_mat(c_, w_up, lo));
  ggml_tensor* gated = ggml_mul(c_, xn, gate);
  gated = ggml_reshape_3d(c_, gated, n_embd, hc, nt);
  const std::size_t row = ggml_row_size(gated->type, n_embd);
  const std::size_t stride = row * U(hc);
  ggml_tensor* mixed = ggml_cont(c_, ggml_view_2d(c_, gated, n_embd, nt, stride, 0));
  for (std::int64_t k = 1; k < hc; ++k) {
    ggml_tensor* s = ggml_view_2d(c_, gated, n_embd, nt, stride, row * U(k));
    mixed = ggml_add(c_, mixed, s);
  }
  mixed = ggml_scale(c_, mixed, 1.0f / static_cast<float>(hc));
  Name(mixed, "hc_mixed", il);
  if (inject != nullptr) {
    *inject = ggml_mul_mat(c_, w_inject, xn);
  }
  return mixed;
}

// build_hc_combine (qwen4exp.cpp:314-334).
ggml_tensor* Builder::HcCombine(ggml_tensor* residual, ggml_tensor* block_out,
                                ggml_tensor* inject) {
  const std::int64_t hc = p_.hc;
  const std::int64_t nt = residual->ne[2];
  if (fused_) {
    return ggml::HcCombine(c_, residual, ggml_reshape_2d(c_, block_out, p_.width, nt), inject);
  }
  ggml_tensor* w = ggml_sigmoid(c_, ggml_scale(c_, inject, 1.0f / static_cast<float>(hc)));
  w = ggml_scale(c_, w, 2.0f);
  w = ggml_reshape_3d(c_, w, 1, hc, nt);
  ggml_tensor* b = ggml_reshape_3d(c_, block_out, p_.width, 1, nt);
  b = ggml_repeat_4d(c_, b, p_.width, hc, nt, 1);
  return ggml_add(c_, residual, ggml_mul(c_, b, w));
}

Builder::Input Builder::HcFast(ggml_tensor*& res, ggml_tensor* out, ggml_tensor* logits,
                               ggml_tensor* w_norm, ggml_tensor* w_down, ggml_tensor* w_up,
                               ggml_tensor* w_inject, ggml_tensor** inject, bool bf16, int il) {
  const std::int64_t hc = p_.hc;
  const std::int64_t hc_dim = p_.hc_width();
  const std::int64_t nt = res->ne[2];
  const std::int64_t n_embd = p_.width;
  ggml_tensor* blob =
      HcPrep(c_, res, w_norm, inject != nullptr ? w_inject : nullptr,
             out != nullptr ? ggml_reshape_2d(c_, out, n_embd, nt) : nullptr, logits, p_.rms_eps);
  const HcPrepLayout layout{
      .width = n_embd, .hc = hc, .t = nt, .combine = out != nullptr, .inject = inject != nullptr};
  if (out != nullptr) {
    ggml_tensor* streams = TypedView(blob, GGML_TYPE_F32, hc_dim, nt, HcPrepLayout::streams());
    res = ggml_reshape_3d(c_, streams, n_embd, hc, nt);
  }
  ggml_tensor* xn = TypedView(blob, GGML_TYPE_BF16, hc_dim, nt, layout.normed());
  // At one row (a decode step) the products run jitLLM's BF16 vector kernel
  // (GemvBf16, kGemvBf16FastColumns), wider cuBLAS.
  ggml_tensor* lo = HcLo(c_, GemvBf16(c_, w_down, xn), hc);
  ggml_tensor* gate = GemvBf16(c_, w_up, lo, GGML_TYPE_BF16);
  if (inject != nullptr) {
    *inject = TypedView(blob, GGML_TYPE_F32, hc, nt, layout.logits());
  }
  // (The head's mix, which has no inject weight, feeds no MXFP8 product.)
  if (nt <= kMxfp8VecColumns || n_embd % 128 != 0 || inject == nullptr) {
    ggml_tensor* mixed = HcMixBf16(c_, xn, gate, hc);
    Name(mixed, "hc_mixed", il);
    return {.x = mixed};
  }
  ggml_tensor* both = HcMixBf16(c_, xn, gate, hc, true, bf16);
  const HcMixLayout mix{.width = n_embd, .t = nt, .bf16 = bf16};
  Input in{.x = TypedView(both, GGML_TYPE_F32, n_embd, nt, HcMixLayout::mixed()),
           .bf16 = bf16 ? TypedView(both, GGML_TYPE_BF16, n_embd, nt, mix.rounded()) : nullptr,
           .mxfp8 = TypedView(both, GGML_TYPE_I8, static_cast<std::int64_t>(mix.quantized_bytes()),
                              1, mix.quantized())};
  Name(in.x, "hc_mixed", il);
  return in;
}

// The fast graph's layers: each block's output combined into the streams by
// the next mix's jitllm.hc.prep, so the streams are read once a mix; the
// n-gram layer and the head take the streams combined on their own.
void Builder::BuildFast(ggml_tensor* res, ggml_tensor* ple) {
  const std::int64_t nt = s_.rows;
  const std::int64_t hc = p_.hc;
  ggml_tensor* out = nullptr;     // the last block's output, not yet combined
  ggml_tensor* logits = nullptr;  // its combine logits
  const auto combine = [&](int il) {
    res = ggml::HcCombine(c_, res, ggml_reshape_2d(c_, out, p_.width, nt), logits);
    Name(res, "l_last", il);
    out = nullptr;
  };
  for (std::uint32_t il_u = 0; il_u < p_.layers; ++il_u) {
    const int il = static_cast<int>(il_u);
    const Qwen38LayerTensors& l = g_.layers[il_u];
    if (il_u == p_.ple_layer) {
      if (out != nullptr) {
        combine(il - 1);
      }
      res = Ple(l, ple, res, il);
      Name(res, "ple_out", il);
    }
    ggml_tensor* inject = nullptr;
    Input cur = HcFast(res, out, logits, l.hc_attn_norm, l.hc_attn_down, l.hc_attn_up,
                       l.hc_attn_inject, &inject, false, il);
    Expand(cur.x);
    out = b_.layers[il_u].linear ? LinearAttention(l, cur.x, il, &cur)
                                 : Attention(l, cur.x, il, &cur);
    logits = inject;
    cur = HcFast(res, out, logits, l.hc_ffn_norm, l.hc_ffn_down, l.hc_ffn_up, l.hc_ffn_inject,
                 &inject, true, il);
    out = Moe(l, cur.x, il, &cur);
    Name(out, "ffn_out", il);
    logits = inject;
  }
  combine(static_cast<int>(p_.layers) - 1);
  // The rows the head computes.
  ggml_tensor* flat = ggml_reshape_2d(c_, res, p_.hc_width(), nt);
  if (export_) {
    // Every row's streams for the MTP drafter (vLLM's scheme A: the
    // pre-final-mixer multi stream).
    Expand(ggml_set_rows(c_, g_.streams, flat, g_.stream_rows));
  }
  flat = ggml_get_rows(c_, flat, g_.out_ids);
  res = ggml_reshape_3d(c_, flat, p_.width, hc, s_.outputs);
  ggml_tensor* cur = HcFast(res, nullptr, nullptr, g_.output_hc_norm, g_.output_hc_down,
                            g_.output_hc_up, nullptr, nullptr, false, -1)
                         .x;
  Name(cur, "result_norm", -1);
  g_.logits = ggml_mul_mat(c_, g_.output, cur);
  Name(g_.logits, "result_output", -1);
  Expand(g_.logits);
  if (verify_) {
    // The greedy verdict's argmaxes on the device: the host reads a row's
    // logits only when it samples.
    g_.argmax = Argmax(c_, g_.logits);
    Expand(g_.argmax);
  }
  g_.nodes = GraphOrder(expanded_);
}

// build_conv_state_at (qwen4exp.cpp:1117-1169), one sequence, no rollback
// slots: the history [cols, channels] then the chunk's rows, time fastest;
// the last `cols` columns become the new history.
ggml_tensor* Builder::ConvStateAt(ggml_tensor* state, ggml_tensor* x, std::int64_t cols,
                                  std::int64_t channels, bool store) {
  ggml_tensor* history = ggml_reshape_3d(c_, state, cols, channels, 1);
  // The chunk's rows transposed into packed rows first: the concatenation
  // takes contiguous rows only.
  ggml_tensor* input = ggml_concat(c_, history, ggml_cont(c_, ggml_transpose(c_, x)), 0);
  if (store) {
    const std::int64_t start = input->ne[0] - cols;
    ggml_tensor* tail = ggml_view_3d(c_, input, cols, channels, 1, input->nb[1], input->nb[2],
                                     ggml_row_size(input->type, start));
    Expand(StoreState(state, ggml_cont(c_, tail)));
  }
  return input;
}

// build_ple (qwen4exp.cpp:1192-1283).
ggml_tensor* Builder::Ple(const Qwen38LayerTensors& l, ggml_tensor* emb, ggml_tensor* hidden,
                          int il) {
  const std::int64_t hc = p_.hc;
  const std::int64_t hc_dim = p_.hc_width();
  const std::int64_t n_embd = p_.width;
  const std::int64_t nt = hidden->ne[2];
  ggml_tensor* key = Mm(l.ple_key, emb);
  ggml_tensor* value = Mm(l.ple_value, emb);
  const auto grouped_norm = [&](ggml_tensor* x, ggml_tensor* w) {
    ggml_tensor* t = ggml_reshape_3d(c_, x, n_embd, hc, nt);
    t = ggml_rms_norm(c_, t, p_.rms_eps);
    t = ggml_reshape_2d(c_, t, hc_dim, nt);
    t = ggml_mul(c_, t, w);
    return ggml_reshape_3d(c_, t, n_embd, hc, nt);
  };
  key = grouped_norm(key, l.ple_norm_key);
  ggml_tensor* query = grouped_norm(hidden, l.ple_norm_query);
  ggml_tensor* s = ggml_sum_rows(c_, ggml_mul(c_, key, query));
  s = ggml_scale(c_, s, 1.0f / std::sqrt(static_cast<float>(n_embd)));
  ggml_tensor* mag = ggml_sqrt(c_, ggml_clamp(c_, ggml_abs(c_, s), 1e-6f, INFINITY));
  ggml_tensor* gate = ggml_sigmoid(c_, ggml_mul(c_, ggml_sgn(c_, s), mag));
  Name(gate, "ple_gate", il);
  ggml_tensor* v3 = ggml_reshape_3d(c_, value, n_embd, 1, nt);
  v3 = ggml_repeat_4d(c_, v3, n_embd, hc, nt, 1);
  ggml_tensor* gated = ggml_mul(c_, v3, gate);
  ggml_tensor* normalized = grouped_norm(ggml_reshape_2d(c_, gated, hc_dim, nt), l.ple_norm_conv);
  normalized = ggml_reshape_2d(c_, normalized, hc_dim, nt);
  const std::int64_t kern = p_.ple_conv;
  const std::int64_t dil = p_.ngram;
  const std::int64_t hist = (kern - 1) * dil;
  // A verify saves its rows for the commit instead of storing the history.
  if (verify_) {
    Save(l.commit_ple, normalized, hc_dim);
  }
  ggml_tensor* padded = ConvStateAt(l.ple_state, ggml_reshape_3d(c_, normalized, hc_dim, nt, 1),
                                    hist, hc_dim, !verify_);
  ggml_tensor* conv_out = nullptr;
  for (std::int64_t k = 0; k < kern; ++k) {
    const std::int64_t start = hist - ((kern - 1 - k) * dil);
    ggml_tensor* shifted = ggml_cont(
        c_, ggml_transpose(c_, ggml_view_3d(c_, padded, nt, hc_dim, 1, padded->nb[1], padded->nb[2],
                                            ggml_row_size(padded->type, start))));
    ggml_tensor* wk =
        ggml_cont(c_, ggml_view_2d(c_, l.ple_conv1d, 1, hc_dim, l.ple_conv1d->nb[1],
                                   static_cast<std::size_t>(k) * l.ple_conv1d->nb[0]));
    wk = ggml_reshape_1d(c_, wk, hc_dim);
    ggml_tensor* term = ggml_mul(c_, shifted, wk);
    conv_out = conv_out != nullptr ? ggml_add(c_, conv_out, term) : term;
  }
  conv_out = ggml_silu(c_, conv_out);
  conv_out = ggml_reshape_3d(c_, ggml_cont(c_, conv_out), n_embd, hc, nt);
  Name(conv_out, "ple_conv_out", il);
  return ggml_add(c_, hidden, ggml_add(c_, gated, conv_out));
}

// build_layer_attn_linear (qwen4exp.cpp:847-972) with the fused gated delta
// rule (delta-net-base.cpp build_delta_net_fused, K = 1).
ggml_tensor* Builder::LinearAttention(const Qwen38LayerTensors& l, ggml_tensor* cur, int il,
                                      const Input* pre) {
  capture_attention_input_ = nullptr;
  capture_attention_projection_ = nullptr;
  const std::int64_t nt = cur->ne[1];
  const std::int64_t d = p_.lin_head_dim;
  const std::int64_t hk = p_.lin_k_heads;
  const std::int64_t hv = p_.lin_v_heads;
  const std::int64_t channels = p_.conv_channels();
  Input in = pre != nullptr ? *pre : In(cur);
  ggml_tensor* qkv = Linear(l.qkv, in, RowsType(nt));
  qkv = ggml_reshape_3d(c_, qkv, qkv->ne[0], nt, 1);
  ggml_tensor* z = Linear(l.z, in, RowsType(nt));
  ggml_tensor* beta = Linear(l.beta, in);
  ggml_tensor* alpha = Linear(l.alpha, in);
  ggml_tensor* gate = nullptr;
  if (fast_ && verify_ && d == 128 && hv == 48 && nt <= 16) {
    // Keep both products and the recurrence's commit/saved rows unchanged.
    // Prefill keeps the original pointwise chains.
    ggml_tensor* gates = GdnGates(c_, alpha, beta, l.dt_bias, l.ssm_a);
    const std::size_t row = ggml_row_size(GGML_TYPE_F32, hv);
    const std::size_t plane = row * U(nt);
    gate = ggml_view_4d(c_, gates, 1, hv, nt, 1, sizeof(float), row, plane, 0);
    beta = ggml_view_4d(c_, gates, 1, hv, nt, 1, sizeof(float), row, plane, plane);
  } else {
    beta = ggml_reshape_4d(c_, beta, 1, hv, nt, 1);
    beta = ggml_sigmoid(c_, beta);
    alpha = ggml_reshape_3d(c_, alpha, hv, nt, 1);
    ggml_tensor* alpha_sp = ggml_softplus(c_, ggml_add(c_, alpha, l.dt_bias));
    gate = ggml_mul(c_, alpha_sp, l.ssm_a);
    gate = ggml_reshape_4d(c_, gate, 1, hv, nt, 1);
  }
  ggml_tensor* state = ggml_reshape_4d(c_, l.recurrent, d, d, hv, 1);
  const std::int64_t history = p_.conv - 1;
  // jitLLM's convolution (jitllm.gdn.conv) reads the history and the rows
  // in place and normalizes the query and key heads itself; it takes whole
  // histories' worth of rows, so that the new history is the last rows (a
  // verify, which stores no history, takes it at every width).
  // The fast graph takes it at every width too: its history kernel reads the
  // old history where the rows are fewer (decode).
  const bool fused_conv = fused_ && d == 128 && p_.conv == 4 && (nt >= history || verify_ || fast_);
  ggml_tensor* conv = nullptr;
  if (fused_conv) {
    const auto n = static_cast<float>(d);
    ggml_tensor* rows = ggml_reshape_2d(c_, qkv, channels, nt);
    conv = GdnConv(c_, rows, l.conv_state, l.conv1d, 2 * d * hk, d, p_.rms_eps / n,
                   1.0f / std::sqrt(n));
    // The convolution reads the old history before the new one is stored.
    Expand(conv);
    if (verify_) {
      // The commit replays the accepted rows' recurrence from the
      // convolution's output and their history from its input.
      Save(l.commit_conv, conv, channels);
      Save(l.commit_qkv, rows, channels);
    } else if (fast_) {
      Expand(StoreState(l.conv_state,
                        GdnHistory(c_, rows, history, nt < history ? l.conv_state : nullptr)));
    } else {
      ggml_tensor* tail = ggml_view_2d(c_, qkv, channels, history, qkv->nb[1],
                                       static_cast<std::size_t>(nt - history) * qkv->nb[1]);
      Expand(StoreState(l.conv_state, ggml_cont(c_, ggml_transpose(c_, tail))));
    }
  } else {
    ggml_tensor* conv_input = ConvStateAt(l.conv_state, qkv, history, channels);
    // Past 32 tokens upstream's convolution loads whole 32-token windows,
    // which would read past the window when the tokens are not a multiple of
    // 32 (RE-032; CheckSsmConv refuses that): the window is padded to whole
    // windows with copies of its first columns, whose outputs are dropped.
    const std::int64_t pad = nt > 32 && nt % 32 != 0 ? 32 - (nt % 32) : 0;
    if (pad != 0) {
      ggml_tensor* first =
          ggml_view_3d(c_, conv_input, pad, channels, 1, conv_input->nb[1], conv_input->nb[2], 0);
      conv_input = ggml_concat(c_, conv_input, ggml_cont(c_, first), 0);
    }
    ggml_tensor* windows = ggml_ssm_conv(c_, conv_input, l.conv1d);
    if (pad != 0) {
      // The first nt tokens' outputs, packed (one sequence).
      windows = ggml_view_3d(c_, windows, channels, nt, 1, windows->nb[1],
                             windows->nb[1] * static_cast<std::size_t>(nt), 0);
    }
    conv = ggml_silu(c_, windows);
  }
  // A head's stride (views' nb1), a token's (nb2) and the chunk's (nb3).
  const std::size_t head_stride = ggml_row_size(conv->type, d);
  const std::size_t token_stride = ggml_row_size(conv->type, channels);
  const std::size_t chunk_stride = token_stride * U(nt);
  ggml_tensor* q = ggml_view_4d(c_, conv, d, hk, nt, 1, head_stride, token_stride, chunk_stride, 0);
  ggml_tensor* k = ggml_view_4d(c_, conv, d, hk, nt, 1, head_stride, token_stride, chunk_stride,
                                ggml_row_size(conv->type, d * hk));
  ggml_tensor* v = ggml_view_4d(c_, conv, d, hv, nt, 1, head_stride, token_stride, chunk_stride,
                                ggml_row_size(conv->type, 2 * d * hk));
  if (!fused_conv) {
    q = L2Norm(q);
    k = L2Norm(k);
  }
  ggml_tensor* output = nullptr;
  if (fast_ && d == 128 && nt <= kGatedDeltaNetLanesTokens) {
    // Decode's recurrence writes the new state over the old in place
    // (jitllm.gdn.step): no state rows in the output, no set_rows copy. A
    // verify's reads it and writes none (the commit replays the kept rows).
    output = GdnStep(c_, q, k, v, gate, beta, state, !verify_);
    Expand(output);
    if (verify_) {
      Save(l.commit_gate, gate, hv);
      Save(l.commit_beta, beta, hv);
    }
  } else {
    ggml_tensor* result = ggml_gated_delta_net(c_, q, k, v, gate, beta, state, 1);
    output = ggml_view_4d(c_, result, d, hv, nt, 1, ggml_row_size(result->type, d),
                          ggml_row_size(result->type, d * hv),
                          ggml_row_size(result->type, d * hv * nt), 0);
    ggml_tensor* new_state = ggml_view_4d(
        c_, result, d, d, hv, 1, ggml_row_size(result->type, d), ggml_row_size(result->type, d * d),
        ggml_row_size(result->type, d * d * hv), ggml_row_size(result->type, d * hv * nt));
    if (verify_) {
      Save(l.commit_gate, gate, hv);
      Save(l.commit_beta, beta, hv);
    } else {
      Expand(StoreState(l.recurrent, new_state));
    }
  }
  ggml_tensor* out = nullptr;
  if (fast_ && d == 128) {
    // build_norm_gated as jitllm.gdn.norm_gate, quantized to MXFP8 for the
    // tensor-core product (in F32 for a GGUF checkpoint's GGML product).
    const bool wide = nt > kMxfp8VecColumns && Mxfp8(l.ssm_out);
    ggml_tensor* normed = GdnNormGate(c_, output, l.ssm_norm, ggml_reshape_2d(c_, z, d * hv, nt),
                                      p_.rms_eps, wide ? GGML_TYPE_I8 : GGML_TYPE_F32);
    out = wide ? LinearMxfp8(l.ssm_out, normed, nt) : Linear(l.ssm_out, normed);
  } else if (fused_ && d == 128) {
    // build_norm_gated as jitllm.gdn.norm_gate, in BF16 for cuBLAS's product
    // (F32 for a GGUF checkpoint's GGML product).
    const bool bf16 = Bf16Inputs(nt) && Mxfp8(l.ssm_out);
    ggml_tensor* normed = GdnNormGate(c_, output, l.ssm_norm, ggml_reshape_2d(c_, z, d * hv, nt),
                                      p_.rms_eps, bf16 ? GGML_TYPE_BF16 : GGML_TYPE_F32);
    out = bf16 ? GemmBf16(c_, Mxfp8Dequant(c_, l.ssm_out.codes, l.ssm_out.scales), normed)
               : Linear(l.ssm_out, normed);
  } else {
    ggml_tensor* z4 = ggml_reshape_4d(c_, z, d, hv, nt, 1);
    // build_norm_gated: the sigmoid output gate.
    ggml_tensor* normed = ggml_mul(c_, Norm(output, l.ssm_norm), ggml_sigmoid(c_, z4));
    ggml_tensor* flat = ggml_reshape_2d(c_, normed, d * hv, nt);
    out = Linear(l.ssm_out, flat);
  }
  Name(out, "linear_attn_out", il);
  return out;
}

// build_qsa_top_k (qwen4exp.cpp:525-674), one stream, the per-block bias.
ggml_tensor* Builder::QsaTopK(const Qwen38LayerTensors& l, Input& cur, int il) {
  const std::int64_t idx_dim = p_.indexer_head_dim;
  const std::int64_t n_idx_h = p_.indexer_heads;
  const std::int64_t r = p_.indexer_ratio;
  const std::int64_t n_kv = s_.n_kv;
  const std::int64_t n_blocks = s_.qsa_blocks;
  const std::int64_t nt = cur.x->ne[1];
  // [q heads · dim | dim, nt], or a GGUF checkpoint's two products: the
  // queries [q heads · dim, nt] and the key [dim, nt].
  ggml_tensor* qk = nullptr;
  ggml_tensor* k_raw = nullptr;
  if (l.idx_q.matrix != nullptr) {
    qk = Linear(l.idx_q, cur);
    k_raw = Linear(l.idx_k, cur);
  } else {
    qk = Linear(l.idx_qk, cur);
    k_raw =
        ggml_view_2d(c_, qk, idx_dim, nt, qk->nb[1], ggml_row_size(qk->type, n_idx_h * idx_dim));
  }
  // The cached keys are raw: pooling precedes their norm and rotation.
  ggml_tensor* raw = ggml_set_rows(c_, l.cache_idx, Packed(k_raw), g_.cells);
  Expand(raw);
  const float theta_scale = std::pow(p_.rope_base, -2.0f / static_cast<float>(p_.rope_dims));
  ggml_tensor* pool = nullptr;
  if (Sparse()) {
    // The sparse graph keeps each block's key from the chunk that completes
    // it, every chunk, whether it selects or not (a later one will).
    pool = QsaPool(c_, raw, l.cache_pool, l.idx_k_norm, g_.positions, r, p_.rms_eps, theta_scale);
    Expand(pool);
  }
  if (!s_.qsa_select) {
    return nullptr;
  }
  if (DeviceSelect()) {
    // The queries' norm and rotation by jitllm.qsa.prep, then the selection
    // over the cached block keys (jitllm.qsa.topk, after the pool), whose
    // kept cells Attention reads alone.
    ggml_tensor* q = QsaPrep(c_, qk, l.idx_q_norm, g_.positions, idx_dim, n_idx_h, idx_dim,
                             p_.rms_eps, theta_scale);
    const std::int64_t width =
        std::min<std::int64_t>(n_kv, std::int64_t{p_.indexer_budget} + r - 1);
    return ggml::QsaTopK(c_, q, l.cache_pool, g_.positions, pool, n_blocks, width, r);
  }
  ggml_tensor* k_all = ggml_view_2d(c_, l.cache_idx, idx_dim, n_kv, l.cache_idx->nb[1], 0);
  ggml_tensor* members = ggml_get_rows(c_, k_all, g_.block_cells);
  members = ggml_reshape_4d(c_, members, idx_dim, r, n_blocks, 1);
  ggml_tensor* pooled = nullptr;
  for (std::int64_t i = 0; i < r; ++i) {
    ggml_tensor* slice =
        ggml_cont(c_, ggml_view_3d(c_, members, idx_dim, n_blocks, 1, members->nb[2],
                                   members->nb[3], static_cast<std::size_t>(i) * members->nb[1]));
    pooled = pooled != nullptr ? ggml_add(c_, pooled, slice) : slice;
  }
  pooled = ggml_scale(c_, pooled, 1.0f / static_cast<float>(r));
  pooled = ggml_reshape_3d(c_, pooled, idx_dim, n_blocks, 1);
  pooled = Norm(pooled, l.idx_k_norm);
  pooled = ggml_reshape_3d(c_, pooled, idx_dim, 1, n_blocks);
  pooled = Rope(pooled, g_.block_pos);
  pooled = ggml_reshape_3d(c_, pooled, idx_dim, n_blocks, 1);
  ggml_tensor* q =
      ggml_view_3d(c_, qk, idx_dim, n_idx_h, nt, ggml_row_size(qk->type, idx_dim), qk->nb[1], 0);
  q = Norm(q, l.idx_q_norm);
  q = Rope(q, g_.positions);
  ggml_tensor* score = ggml_mul_mat(c_, pooled, ggml_reshape_3d(c_, q, idx_dim, n_idx_h * nt, 1));
  score = ggml_reshape_4d(c_, score, n_blocks, n_idx_h, nt, 1);
  score = ggml_relu(c_, score);
  ggml_tensor* summed = nullptr;
  for (std::int64_t h = 0; h < n_idx_h; ++h) {
    ggml_tensor* slice = ggml_view_3d(c_, score, n_blocks, nt, 1, score->nb[2], score->nb[3],
                                      static_cast<std::size_t>(h) * score->nb[1]);
    summed = summed != nullptr ? ggml_add(c_, summed, slice) : ggml_cont(c_, slice);
  }
  score = ggml_add(c_, summed, ggml_reshape_3d(c_, g_.block_bias, n_blocks, nt, 1));
  Name(score, "indexer_score", il);
  ggml_tensor* expanded =
      ggml_get_rows(c_, ggml_cont(c_, ggml_permute(c_, score, 1, 0, 2, 3)), g_.cell_block);
  expanded = ggml_cont(c_, ggml_permute(c_, expanded, 1, 0, 2, 3));
  expanded = ggml_add(c_, expanded, ggml_reshape_3d(c_, g_.mask_f32, n_kv, nt, 1));
  const std::int64_t width = std::min<std::int64_t>(n_kv, std::int64_t{p_.indexer_budget} + r - 1);
  ggml_tensor* top_k = ggml_cont(c_, ggml_top_k(c_, expanded, static_cast<int>(width)));
  return ggml_reshape_4d(c_, top_k, width, nt, 1, 1);
}

// build_layer_attn with build_attn_qsa (qwen4exp.cpp:676-845).
ggml_tensor* Builder::Attention(const Qwen38LayerTensors& l, ggml_tensor* cur, int il,
                                const Input* pre) {
  capture_attention_input_ = nullptr;
  capture_attention_projection_ = nullptr;
  const std::int64_t d = p_.head_dim;
  const std::int64_t heads = p_.heads;
  const std::int64_t kvh = p_.kv_heads;
  const std::int64_t nt = cur->ne[1];
  const std::int64_t n_kv = s_.n_kv;
  Input in = pre != nullptr ? *pre : In(cur);
  ggml_tensor* top_k = QsaTopK(l, in, il);
  // [(d · 2) · heads, nt]: per head, q then its gate
  ggml_tensor* q_full = Linear(l.q, in);
  if (il < 64 && (capture_routed_ & (std::uint64_t{1} << il)) != 0) {
    capture_attention_input_ = in.x;
    capture_attention_projection_ = q_full;
  }
  const std::size_t f = ggml_element_size(q_full);
  const std::size_t per_head = f * U(d) * 2;  // q then its gate
  ggml_tensor* k = Linear(l.k, in);
  ggml_tensor* v = Linear(l.v, in);
  ggml_tensor* q = nullptr;
  // The fast graph's query and key heads: norm and rotation in one pass
  // (jitllm.qsa.prep), where the rotation is 64 dimensions.
  if (FastSelect()) {
    const float theta_scale = std::pow(p_.rope_base, -2.0f / static_cast<float>(p_.rope_dims));
    q = QsaPrep(c_, q_full, l.q_norm, g_.positions, d, heads, 2 * d, p_.rms_eps, theta_scale);
    k = QsaPrep(c_, k, l.k_norm, g_.positions, d, kvh, d, p_.rms_eps, theta_scale);
  } else {
    q = ggml_view_3d(c_, q_full, d, heads, nt, per_head, per_head * U(heads), 0);
    q = Norm(q, l.q_norm);
    k = ggml_reshape_3d(c_, k, d, kvh, nt);
    k = Norm(k, l.k_norm);
    q = Rope(q, g_.positions);
    k = Rope(k, g_.positions);
  }
  v = ggml_reshape_3d(c_, v, d, kvh, nt);
  Expand(q);
  Expand(v);
  Expand(k);
  // llama_kv_cache::cpy_k and cpy_v: merge the heads, store at the cells.
  ggml_tensor* stored_k =
      ggml_set_rows(c_, l.cache_k, ggml_reshape_2d(c_, k, d * kvh, nt), g_.cells);
  ggml_tensor* stored_v =
      ggml_set_rows(c_, l.cache_v, ggml_reshape_2d(c_, v, d * kvh, nt), g_.cells);
  Expand(stored_k);
  Expand(stored_v);
  const float scale = 1.0f / std::sqrt(static_cast<float>(d));
  ggml_tensor* attn = nullptr;
  ggml_tensor* kq_mask = g_.mask;
  if (top_k != nullptr && JitllmOpOf(top_k) == JitllmOp::kQsaTopK) {
    // The sparse graph: attention over each row's kept cells alone, read
    // from the caches after this chunk's cells are stored.
    attn = QsaAttn(c_, q, stored_k, stored_v, top_k, scale);
  } else if (top_k != nullptr) {
    // build_attn_qsa's mask: -inf everywhere but the selected cells, plus
    // the causal mask.
    ggml_tensor* all = ggml_fill(c_, kq_mask, -INFINITY);
    all = ggml_view_4d(c_, all, 1, all->ne[0], all->ne[1], all->ne[3], all->nb[0], all->nb[1],
                       all->nb[2], 0);
    ggml_tensor* top_k_3d =
        ggml_view_4d(c_, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1],
                     top_k->nb[2], static_cast<std::size_t>(top_k->ne[3]) * top_k->nb[3], 0);
    ggml_tensor* zeros =
        ggml_new_tensor_4d(c_, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(c_, zeros, 0.0f);
    ggml_tensor* masked = ggml_set_rows(c_, all, zeros, top_k_3d);
    masked = ggml_view_4d(c_, masked, masked->ne[1], masked->ne[2], 1, masked->ne[3], masked->nb[2],
                          masked->nb[3], masked->nb[3], 0);
    kq_mask = ggml_add(c_, masked, kq_mask);
  }
  if (attn == nullptr) {
    // get_k / get_v over the first n_kv cells: [d, n_kv, kv heads].
    const std::size_t hrow = ggml_row_size(l.cache_k->type, d);
    const std::size_t crow = l.cache_k->nb[1];
    ggml_tensor* kc = ggml_view_3d(c_, l.cache_k, d, n_kv, kvh, crow, hrow, 0);
    ggml_tensor* vc = ggml_view_3d(c_, l.cache_v, d, n_kv, kvh, crow, hrow, 0);
    ggml_tensor* qp = ggml_permute(c_, q, 0, 2, 1, 3);
    attn = ggml_flash_attn_ext(c_, qp, kc, vc, kq_mask, scale, 0.0f, 0.0f);
    ggml_prec_set_acc(attn, GGML_PREC_F32);
    attn = ggml_reshape_2d(c_, attn, attn->ne[0] * attn->ne[1], attn->ne[2] * attn->ne[3]);
  }
  Name(attn, "attn_pregate", il);
  ggml_tensor* out = nullptr;
  if (fast_ && nt > kMxfp8VecColumns && Mxfp8(l.o)) {
    // The gate and the output projection's quantization in one pass.
    out = LinearMxfp8(l.o, QsaGateQuantize(c_, attn, q_full, d), nt);
  } else {
    ggml_tensor* gate =
        ggml_view_3d(c_, q_full, d, heads, nt, per_head, per_head * U(heads), f * U(d));
    gate = ggml_cont_2d(c_, gate, d * heads, nt);
    attn = ggml_mul(c_, attn, ggml_sigmoid(c_, gate));
    out = Linear(l.o, attn);
  }
  Name(out, "attn_output", il);
  return out;
}

// build_layer_ffn with build_moe_ffn and build_ffn (qwen4exp.cpp:974-1022,
// llama-graph.cpp:1748-2380): softmax routing, the top experts' weights
// renormalized, SwiGLU experts, and the shared expert behind its sigmoid
// gate.
ggml_tensor* Builder::Moe(const Qwen38LayerTensors& l, ggml_tensor* cur, int il, const Input* pre) {
  const std::int64_t n_embd = cur->ne[0];
  const std::int64_t nt = cur->ne[1];
  const std::int64_t n_expert = p_.experts;
  const std::int64_t used = p_.experts_used;
  Input in = pre != nullptr ? *pre : In(cur);
  ggml_tensor* logits = MulMat(l.router, in);
  ggml_tensor* selected = nullptr;
  ggml_tensor* weights = nullptr;
  ggml_tensor* fast_gate = nullptr;
  if (fast_) {
    // jitllm.moe.router: the softmax, top experts and their renormalized
    // weights, and the shared expert's gate logit, in one pass.
    ggml_tensor* routed = MoeRouter(c_, logits, in.x, l.shared_gate, used);
    const MoeRouterLayout layout{.used = used, .t = nt};
    selected = TypedView(routed, GGML_TYPE_I32, used, nt, MoeRouterLayout::ids());
    weights = TypedView(routed, GGML_TYPE_F32, used, nt, layout.weights());
    fast_gate = TypedView(routed, GGML_TYPE_F32, 1, nt, layout.gate());
  } else {
    ggml_tensor* probs = ggml_soft_max(c_, logits);
    selected = ggml_argsort_top_k(c_, probs, static_cast<int>(used));
    probs = ggml_reshape_3d(c_, probs, 1, n_expert, nt);
    weights = ggml_get_rows(c_, probs, selected);
    weights = ggml_reshape_2d(c_, weights, used, nt);
    ggml_tensor* sum = ggml_sum_rows(c_, weights);
    sum = ggml_clamp(c_, sum, 6.103515625e-5f, INFINITY);
    weights = ggml_div(c_, weights, sum);
  }
  Name(selected, "ffn_moe_topk", il);
  weights = ggml_reshape_3d(c_, weights, 1, used, nt);
  Expand(weights);
  ggml_tensor* x = ggml_reshape_3d(c_, cur, n_embd, 1, nt);
  // The shared expert's gate, one value a token: GGML's products refuse a
  // one-row output (its rows are not 8-byte aligned), so the dot product is
  // the gate row (read as F32) times each token, summed (the fast graph's
  // router computes it).
  const auto shared_gate_dot = [&] {
    if (fast_gate != nullptr) {
      return fast_gate;
    }
    ggml_tensor* gate_row =
        ggml_get_rows(c_, ggml_reshape_2d(c_, l.shared_gate, n_embd, 1), g_.row_zero);
    return ggml_sum_rows(c_, ggml_mul(c_, cur, gate_row));
  };
  if (cutlass_) {
    // The CUTLASS layout: moe_layout.h's parts of each slot.
    const moe::ExpertLayout layout{.ffn = p_.expert_ffn, .width = p_.width};
    const std::int64_t f = p_.expert_ffn;
    ggml_tensor* sh_up = Linear(l.up_shexp, in);
    ggml_tensor* sh_gate = Linear(l.gate_shexp, in);
    ggml_tensor* sh = Linear(l.down_shexp, ggml_swiglu_split(c_, sh_gate, sh_up));
    if (nt <= kMoeGemvTokens) {
      // Decode: each slot's products straight from the layout (gate and up
      // with their SwiGLU in one), then the same weighted sum as GGML's
      // products feed.
      ggml_tensor* act =
          MoeGemvSwiglu(c_, l.experts, x, selected, f, l.gate_exps_scale, l.up_exps_scale,
                        moe::ExpertLayout::gate_up_codes(), layout.gate_up_scales());
      ggml_tensor* down = MoeGemv(c_, l.experts, act, selected, n_embd, 0, n_embd,
                                  layout.down_codes(), layout.down_scales());
      ggml_tensor* shared_gate = shared_gate_dot();
      ggml_tensor* combined =
          MoeCombine(c_, down, selected, l.down_exps_scale, weights, sh, shared_gate);
      if (il < 64 && (capture_routed_ & (std::uint64_t{1} << il)) != 0) {
        g_.routed.push_back({.layer = static_cast<std::uint32_t>(il),
                             .input = x,
                             .activation = act,
                             .down = down,
                             .shared = sh,
                             .gate = shared_gate,
                             .weights = weights,
                             .ids = selected,
                             .combined = combined,
                             .attention_input = capture_attention_input_,
                             .attention_projection = capture_attention_projection_});
      }
      return combined;
    }
    // Prefill: the rows sorted by expert, quantized once, CUTLASS's grouped
    // GEMMs (gate and up as one), SwiGLU with the next quantization, and
    // the weighted sum.
    ggml_tensor* route = MoeRoute(c_, selected, n_expert);
    ggml_tensor* a1 = MoeQuantize(c_, cur, route);
    ggml_tensor* d1 = MoeGemm(c_, a1, route, l.experts, 2 * f, moe::ExpertLayout::gate_up_codes(),
                              layout.gate_up_scales());
    ggml_tensor* a2 = MoeGluQuantize(c_, d1, a1, route, l.gate_exps_scale, l.up_exps_scale);
    ggml_tensor* d2 =
        MoeGemm(c_, a2, route, l.experts, n_embd, layout.down_codes(), layout.down_scales());
    return MoeCombineSorted(c_, d2, a2, route, l.down_exps_scale, weights, sh, shared_gate_dot());
  }
  // A GGUF checkpoint's fast form at decode: the routed experts' gate and up
  // products with their SwiGLU in one jitllm.vecq (each selected expert read
  // once), the down product over the activations' one quantization, the
  // shared expert likewise (dsv4_graph.h's fast plan's MoE), then the
  // unscaled combine.
  if (gguf_ && fast_ && nt * used <= 64 && VecQType(l.up_exps->type) &&
      VecQType(l.down_exps->type) && l.gate_exps->type == l.up_exps->type &&
      in.x->type == GGML_TYPE_F32 && ggml_is_contiguous(in.x) && in.x->ne[1] == nt &&
      n_embd % 32 == 0 && p_.expert_ffn % 32 == 0) {
    ggml_tensor* q = Q8Of(in.x);
    ggml_tensor* act = VecQ(c_, l.up_exps, q, selected, nt, false, l.gate_exps, VecQGlu::kSwiglu);
    ggml_tensor* down = VecQ(c_, l.down_exps, QuantizeQ8(c_, act), selected, nt, true);
    ggml_tensor* sh = nullptr;
    const ggml_tensor* sg = l.gate_shexp.matrix;
    const ggml_tensor* su = l.up_shexp.matrix;
    if (VecQFits(su, in.x) && sg != nullptr && sg->type == su->type &&
        ggml_are_same_shape(sg, su)) {
      sh =
          VecQ(c_, l.up_shexp.matrix, q, nullptr, nt, false, l.gate_shexp.matrix, VecQGlu::kSwiglu);
    } else {
      sh = ggml_swiglu_split(c_, Linear(l.gate_shexp, in), Linear(l.up_shexp, in));
    }
    sh = Linear(l.down_shexp, sh);
    return MoeCombine(c_, down, selected, nullptr, weights, sh, shared_gate_dot());
  }
  if (fused_) {
    // jitllm.moe.glu and jitllm.moe.combine over the same products; experts
    // without global scales (a GGUF checkpoint's) take GGML's SwiGLU and the
    // combine's unscaled form.
    ggml_tensor* up = ggml_mul_mat_id(c_, l.up_exps, x, selected);
    ggml_tensor* gate = ggml_mul_mat_id(c_, l.gate_exps, x, selected);
    ggml_tensor* act = l.gate_exps_scale != nullptr
                           ? MoeGlu(c_, gate, up, selected, l.gate_exps_scale, l.up_exps_scale)
                           : ggml_swiglu_split(c_, gate, up);
    ggml_tensor* down = ggml_mul_mat_id(c_, l.down_exps, act, selected);
    ggml_tensor* sh_up = Linear(l.up_shexp, in);
    ggml_tensor* sh_gate = Linear(l.gate_shexp, in);
    ggml_tensor* sh = Linear(l.down_shexp, ggml_swiglu_split(c_, sh_gate, sh_up));
    // (Named ffn_out by the caller: the fusion includes the shared expert.)
    return MoeCombine(c_, down, selected, l.down_exps_scale, weights, sh, shared_gate_dot());
  }
  ggml_tensor* up = MulMatId(l.up_exps, x, selected, l.up_exps_scale);
  ggml_tensor* gate = MulMatId(l.gate_exps, x, selected, l.gate_exps_scale);
  ggml_tensor* act = ggml_swiglu_split(c_, gate, up);
  ggml_tensor* experts = MulMatId(l.down_exps, act, selected, l.down_exps_scale);
  experts = ggml_mul(c_, experts, weights);
  Expand(experts);
  std::vector<ggml_tensor*> views;
  for (std::int64_t i = 0; i < used; ++i) {
    views.push_back(ggml_view_2d(c_, experts, n_embd, nt, experts->nb[2],
                                 static_cast<std::size_t>(i) * experts->nb[1]));
    Expand(views.back());
  }
  ggml_tensor* moe_out = views[0];
  for (std::size_t i = 1; i < views.size(); ++i) {
    moe_out = ggml_add(c_, moe_out, views[i]);
    Expand(moe_out);
  }
  if (used == 1) {
    moe_out = ggml_cont(c_, moe_out);
  }
  Name(moe_out, "ffn_moe_out", il);
  ggml_tensor* sh_up = Linear(l.up_shexp, cur);
  ggml_tensor* sh_gate = Linear(l.gate_shexp, cur);
  ggml_tensor* sh = Linear(l.down_shexp, ggml_swiglu_split(c_, sh_gate, sh_up));
  ggml_tensor* shared_gate = ggml_sigmoid(c_, shared_gate_dot());
  sh = ggml_mul(c_, sh, shared_gate);
  Name(sh, "ffn_shexp_gated", il);
  return ggml_add(c_, moe_out, sh);
}

void Builder::Build() {
  const std::int64_t nt = s_.rows;
  const std::int64_t hc = p_.hc;
  ggml_tensor* inpl = ggml_get_rows(c_, g_.token_embd, g_.tokens);
  Expand(inpl);
  ggml_tensor* ple = nullptr;
  if (!gguf_) {
    ple = Nvfp4Rows(c_, g_.ple_table, g_.ple_rows, g_.ple_table_scale, p_.ple_row);
  } else if (QRowsType(g_.ple_table->type)) {
    ple = QRows(c_, g_.ple_table, g_.ple_rows);
  } else {
    ple = ggml_get_rows(c_, g_.ple_table, g_.ple_rows);
  }
  ple = ggml_reshape_2d(c_, ple, p_.ple_width(), nt);
  Name(ple, "ple_embd", -1);
  Expand(ple);
  ggml_tensor* res =
      ggml_repeat_4d(c_, ggml_reshape_3d(c_, inpl, p_.width, 1, nt), p_.width, hc, nt, 1);
  Name(res, "hc_init", -1);
  // (A GGUF checkpoint's mixer weights are not the fast mix's BF16: its
  // layers take the loop below, whose blocks keep their fast forms.)
  if (fast_ && !gguf_) {
    BuildFast(res, ple);
    return;
  }
  for (std::uint32_t il_u = 0; il_u < p_.layers; ++il_u) {
    const int il = static_cast<int>(il_u);
    const Qwen38LayerTensors& l = g_.layers[il_u];
    if (il_u == p_.ple_layer) {
      res = Ple(l, ple, res, il);
      Name(res, "ple_out", il);
    }
    ggml_tensor* inject = nullptr;
    ggml_tensor* cur =
        HcMix(res, l.hc_attn_norm, l.hc_attn_down, l.hc_attn_up, l.hc_attn_inject, &inject, il);
    Expand(cur);
    cur = b_.layers[il_u].linear ? LinearAttention(l, cur, il) : Attention(l, cur, il);
    res = HcCombine(res, cur, inject);
    cur = HcMix(res, l.hc_ffn_norm, l.hc_ffn_down, l.hc_ffn_up, l.hc_ffn_inject, &inject, il);
    cur = Moe(l, cur, il);
    Name(cur, "ffn_out", il);
    res = HcCombine(res, cur, inject);
    Name(res, "l_last", il);
  }
  // The rows the head computes.
  ggml_tensor* flat = ggml_reshape_2d(c_, res, p_.hc_width(), nt);
  flat = ggml_get_rows(c_, flat, g_.out_ids);
  res = ggml_reshape_3d(c_, flat, p_.width, hc, s_.outputs);
  ggml_tensor* cur =
      HcMix(res, g_.output_hc_norm, g_.output_hc_down, g_.output_hc_up, nullptr, nullptr, -1);
  Name(cur, "result_norm", -1);
  g_.logits = Mm(g_.output, cur);
  Name(g_.logits, "result_output", -1);
  Expand(g_.logits);
  g_.nodes = GraphOrder(expanded_);
}

// Qwen3_8FlashNextMultiTokenPredictor.forward (mtp.py:262-329) for one
// pass, the decoder layer's delayed combine done before the final mixer
// (GatedResidual.combine_and_mix).
Builder::MtpOut Builder::MtpPass(const Qwen38MtpGraph& m, ggml_tensor* tokens, ggml_tensor* hidden,
                                 bool head, std::int64_t head_rows, bool confidence) {
  const std::int64_t nt = s_.rows;
  const std::int64_t width = p_.width;
  const std::int64_t hc = p_.hc;
  const Qwen38LayerTensors& l = g_.layers[0];
  // The embedding branch: pre_fc_norm_embedding, then fc_embedding.
  ggml_tensor* e = ggml_get_rows(c_, m.token_embd, tokens);
  e = Norm(e, m.norm_embd);
  Input embedded = In(e);
  e = LinearBf16(m.fc_embd, embedded);
  // The streams: pre_fc_norm_hidden over all of them, then fc_hidden on each.
  ggml_tensor* h = Norm(hidden, m.norm_hidden);
  Input streams = In(ggml_reshape_2d(c_, h, width, hc * nt));
  h = LinearBf16(m.fc_hidden, streams);
  // The embedding added to every stream.
  ggml_tensor* res =
      ggml_add(c_, ggml_reshape_3d(c_, h, width, hc, nt), ggml_reshape_3d(c_, e, width, 1, nt));
  ggml_tensor* inject = nullptr;
  Input cur = HcFast(res, nullptr, nullptr, l.hc_attn_norm, l.hc_attn_down, l.hc_attn_up,
                     l.hc_attn_inject, &inject, false, 0);
  ggml_tensor* attn = Attention(l, cur.x, 0, &cur);
  if (!head) {
    return {};  // a prefill pass: its caches' writes only
  }
  ggml_tensor* logits = inject;
  cur = HcFast(res, attn, logits, l.hc_ffn_norm, l.hc_ffn_down, l.hc_ffn_up, l.hc_ffn_inject,
               &inject, true, 0);
  ggml_tensor* out = Moe(l, cur.x, 0, &cur);
  res = ggml::HcCombine(c_, res, ggml_reshape_2d(c_, out, width, nt), inject);
  Name(res, "mtp_streams", 0);
  // The last row's final mix and the head over the draft vocabulary.
  ggml_tensor* last =
      nt == 1 ? res
              : ggml_view_3d(c_, res, width, hc, 1, res->nb[1], res->nb[2], U(nt - 1) * res->nb[2]);
  ggml_tensor* mixed = HcFast(last, nullptr, nullptr, m.output_hc_norm, m.output_hc_down,
                              m.output_hc_up, nullptr, nullptr, false, -1)
                           .x;
  ggml_tensor* head_input = mixed;
  ggml_tensor* w =
      head_rows > 0 ? ggml_view_2d(c_, m.output, width, head_rows, m.output->nb[1], 0) : m.output;
  // The argmax and its probability (the drafter's confidence, which an
  // adaptive window reads): the draft is the first I32, the probability's
  // bits the second.
  ggml_tensor* head_logits = ggml_mul_mat(c_, w, mixed);
  ggml_tensor* both = Argmax(c_, head_logits, confidence);
  Expand(res);
  Expand(both);
  ggml_tensor* draft = ggml_view_1d(c_, both, 1, 0);
  ggml_tensor* probability = confidence ? ggml_view_1d(c_, both, 1, sizeof(std::int32_t)) : nullptr;
  if (m.draft_ids != nullptr) {
    draft = ggml_reshape_1d(c_, ggml_get_rows(c_, m.draft_ids, draft), 1);
  }
  Name(draft, "mtp_draft", 0);
  Expand(draft);
  if (probability != nullptr) {
    Expand(probability);
  }
  return {.streams = last,
          .draft = draft,
          .probability = probability,
          .head_input = head_input,
          .head_logits = head_logits};
}

}  // namespace

std::vector<ggml_tensor*> Qwen38MtpGraph::inputs() const {
  std::vector<ggml_tensor*> all = {state_row, row_zero};
  for (const Qwen38MtpPass& p : passes) {
    for (ggml_tensor* t : {p.tokens, p.positions, p.cells, p.mask, p.out_ids}) {
      if (t != nullptr) {
        all.push_back(t);
      }
    }
  }
  return all;
}

std::size_t Qwen38MtpGraphTensors(const model::Qwen38Profile& profile, std::int64_t passes) {
  (void)profile;
  // Leaves about 60; a pass at most about 400 nodes (a QSA layer with its
  // selection, the MoE, the mixes and the head).
  return 256 + (static_cast<std::size_t>(std::max<std::int64_t>(passes, 1)) * 512);
}

std::expected<Qwen38MtpGraph, KernelFailure> BuildQwen38MtpGraph(
    TensorArena& arena, const model::Qwen38Profile& profile, const model::Qwen38Binding& target,
    const model::Qwen38MtpBinding& drafter, const Qwen38MtpShape& shape,
    std::uint64_t expert_stride) {
  const Qwen38MtpShape& s = shape;
  const std::int64_t ratio = profile.indexer_ratio;
  const auto head_rows = drafter.selected_head() ? drafter.draft_output.ne[1] : profile.vocab;
  if (s.rows <= 0 || s.passes <= 0 || s.passes > 8 || s.cells <= 0 || s.n_kv < 256 ||
      s.n_kv > s.cells || s.n_kv % 256 != 0 || s.hidden_row < 0 || s.hidden_rows <= 0 ||
      s.hidden_row + s.rows > s.hidden_rows || (s.passes > 1 && !s.head) || s.head_rows < 0 ||
      std::cmp_greater(s.head_rows, head_rows) || ratio <= 0) {
    return Rejected("not an MTP drafter shape its state holds");
  }
  if (s.capture_head &&
      (!s.head ||
       (s.head_rows == 0 ? head_rows : static_cast<std::uint64_t>(s.head_rows)) > 65536)) {
    return Rejected("a draft-head capture requires a head of at most 65,536 rows");
  }
  const bool past_budget = s.n_kv > std::int64_t{profile.indexer_budget} + ratio - 1;
  if (s.qsa_select != past_budget ||
      s.qsa_blocks != (s.qsa_select ? (s.n_kv + ratio - 1) / ratio : s.qsa_blocks)) {
    return Rejected("a QSA selection that is not the indexer budget's");
  }
  if (auto room = arena.Reserve(Qwen38MtpGraphTensors(profile, s.passes)); !room) {
    return std::unexpected(room.error());
  }
  ggml_context* c = arena.context();
  Qwen38MtpGraph m;
  m.state_row = ggml_new_tensor_1d(c, GGML_TYPE_I64, 1);
  m.row_zero = ggml_new_tensor_1d(c, GGML_TYPE_I32, 1);
  if (auto made = LayerLeaves(c, profile, drafter.layer, m.layer, false, true, true, expert_stride,
                              s.cells, 0);
      !made) {
    return std::unexpected(made.error());
  }
  for (const auto& [into, t, role] :
       {std::tuple{&m.fc_embd, &drafter.fc_embd, "fc_embd"},
        std::tuple{&m.fc_hidden, &drafter.fc_hidden, "fc_hidden"},
        std::tuple{&m.norm_embd, &drafter.norm_embd, "norm_embd"},
        std::tuple{&m.norm_hidden, &drafter.norm_hidden, "norm_hidden"},
        std::tuple{&m.output_hc_norm, &drafter.output_hc_norm, "output_hc_norm"},
        std::tuple{&m.output_hc_down, &drafter.output_hc_down, "output_hc_down"},
        std::tuple{&m.output_hc_up, &drafter.output_hc_up, "output_hc_up"},
        std::tuple{&m.token_embd, &target.token_embd, "token_embd"},
        std::tuple{&m.output, drafter.selected_head() ? &drafter.draft_output : &target.output,
                   "output"}}) {
    auto made = Leaf(c, *t, role);
    if (!made) {
      return std::unexpected(made.error());
    }
    *into = *made;
  }
  if (drafter.selected_head()) {
    auto made = Leaf(c, drafter.draft_ids, "draft_ids");
    if (!made) {
      return std::unexpected(made.error());
    }
    m.draft_ids = *made;
  }
  m.streams = ggml_new_tensor_2d(c, GGML_TYPE_F32, profile.hc_width(), s.hidden_rows);
  const std::size_t row = m.streams->nb[1];
  ggml_tensor* hidden =
      ggml_view_2d(c, m.streams, profile.hc_width(), s.rows, row, U(s.hidden_row) * row);
  ggml_tensor* tokens = nullptr;
  std::vector<ggml_tensor*> expanded;
  m.passes.resize(static_cast<std::size_t>(s.passes));
  for (std::int64_t p = 0; p < s.passes; ++p) {
    const std::int64_t n = p == 0 ? s.rows : 1;
    const Qwen38ChunkShape ps{.rows = n,
                              .n_kv = s.n_kv,
                              .cells = s.cells,
                              .outputs = 1,
                              .qsa_select = s.qsa_select,
                              .qsa_blocks = s.qsa_blocks};
    Qwen38Graph pg;
    pg.layers = {m.layer};
    pg.state_row = m.state_row;
    pg.row_zero = m.row_zero;
    Builder b(c, profile, target, ps, pg, true, false, true);
    Qwen38MtpPass& in = m.passes[static_cast<std::size_t>(p)];
    if (p == 0) {
      in.tokens = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
      tokens = in.tokens;
    }
    in.positions = pg.positions = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4 * n);
    in.cells = pg.cells = ggml_new_tensor_1d(c, GGML_TYPE_I64, n);
    if (!b.SelectsOnDevice()) {
      in.mask = pg.mask = ggml_new_tensor_4d(c, GGML_TYPE_F16, s.n_kv, n, 1, 1);
    }
    // The drafter selects on the device (from its cached block keys), at
    // any depth its selection's tiles hold.
    if (s.qsa_select && !b.SelectsOnDevice()) {
      return Rejected("the drafter selects on the device only (its shapes' blocks)");
    }
    const Builder::MtpOut out = b.MtpPass(m, tokens, hidden, s.head, s.head_rows, s.confidence);
    expanded.insert(expanded.end(), b.expanded().begin(), b.expanded().end());
    if (s.head) {
      m.drafts.push_back(out.draft);
      if (out.probability != nullptr) {
        m.probabilities.push_back(out.probability);
      }
      if (s.capture_head) {
        m.head_inputs.push_back(out.head_input);
        m.head_logits.push_back(out.head_logits);
      }
      hidden = ggml_reshape_2d(c, out.streams, profile.hc_width(), 1);
      tokens = out.draft;
    }
  }
  m.nodes = GraphOrder(expanded);
  return m;
}

Qwen38ChunkShape Qwen38ShapeOf(const model::Qwen38StateLayout& state,
                               const model::Qwen38ChunkInputs& chunk, std::int64_t outputs) {
  return {.rows = chunk.rows,
          .n_kv = chunk.n_kv,
          .cells = state.cells,
          .outputs = outputs,
          .qsa_select = chunk.qsa_select,
          .qsa_blocks = chunk.qsa_select ? chunk.qsa.blocks : 0};
}

std::vector<ggml_tensor*> Qwen38Graph::inputs() const {
  std::vector<ggml_tensor*> all = {tokens, positions, cells};
  if (mask != nullptr) {
    all.push_back(mask);
  }
  for (ggml_tensor* t : {ple_rows, state_row, row_zero, out_ids}) {
    all.push_back(t);
  }
  for (ggml_tensor* t :
       {mask_f32, cell_block, block_cells, block_pos, block_bias, row_ids, stream_rows}) {
    if (t != nullptr) {
      all.push_back(t);
    }
  }
  return all;
}

ggml_tensor* Qwen38Graph::Named(std::string_view name) const {
  for (const auto& [n, t] : named) {
    if (n == name) {
      return t;
    }
  }
  return nullptr;
}

std::size_t Qwen38GraphTensors(const model::Qwen38Profile& profile) {
  // Leaves: about 40 per layer; nodes: at most about 200 per layer (a QSA
  // layer with its selection, MoE included). Rounded up generously.
  return 512 + (std::size_t{profile.layers} * 384);
}

std::expected<Qwen38Graph, KernelFailure> BuildQwen38Graph(TensorArena& arena,
                                                           const model::Qwen38Profile& profile,
                                                           const model::Qwen38Binding& binding,
                                                           const Qwen38ChunkShape& shape,
                                                           const Qwen38GraphOptions& options) {
  const Qwen38ChunkShape& s = shape;
  if (s.rows <= 0 || s.cells <= 0 || s.n_kv < 256 || s.n_kv > s.cells || s.n_kv % 256 != 0 ||
      s.outputs <= 0 || s.outputs > s.rows || (s.qsa_select && s.qsa_blocks <= 0)) {
    return Rejected("not a Qwen3.8 chunk shape the state holds");
  }
  // QSA selects exactly when the cells pass the budget (plus the tail block
  // model/qwen38.cc's Qwen38Chunk keeps), over every cell's pooled block.
  const std::int64_t ratio = profile.indexer_ratio;
  const bool past_budget = ratio > 0 && s.n_kv > std::int64_t{profile.indexer_budget} + ratio - 1;
  if (ratio <= 0 || s.qsa_select != past_budget ||
      s.qsa_blocks != (s.qsa_select ? (s.n_kv + ratio - 1) / ratio : s.qsa_blocks)) {
    return Rejected("a QSA selection that is not the indexer budget's");
  }
  if (binding.layers.size() != profile.layers) {
    return Rejected("the binding is not the profile's");
  }
  if (auto room = arena.Reserve(Qwen38GraphTensors(profile)); !room) {
    return std::unexpected(room.error());
  }
  Qwen38Graph g;
  const bool cutlass = options.experts == Qwen38GraphOptions::Experts::kCutlass;
  if (cutlass && (!options.fused || options.expert_stride.empty())) {
    return Rejected("the CUTLASS expert layout takes the fused graph and an expert stride");
  }
  // A GGUF checkpoint's experts are GGML's types (no CUTLASS layout), and
  // speculation is built for the ModelOpt artifact alone (its drafter).
  if (binding.gguf() &&
      (cutlass || options.verify || options.export_streams || options.capture_routed != 0)) {
    return Rejected(
        "a GGUF artifact's graph takes GGML's expert layout and no verify, drafter streams or "
        "routed capture");
  }
  // A verify's rows are the vector products' (their F32 rows are saved).
  if (options.verify && (!options.fused || options.exact || s.rows > kMxfp8VecColumns ||
                         s.outputs != s.rows || profile.lin_head_dim != 128 || profile.conv != 4)) {
    return Rejected("a verify takes the fast form, at most 8 rows, every row's logits");
  }
  if (options.export_streams && (!options.fused || options.exact)) {
    return Rejected("the streams are exported by the fast form");
  }
  if (options.capture_routed != 0 &&
      (!Qwen38RoutedCaptureFits(options.capture_routed, profile.layers) || !options.verify ||
       !cutlass || s.rows > 4)) {
    return Rejected(
        "routed capture takes at most three layers of a CUTLASS fast verify of 1..4 rows");
  }
  Builder builder(arena.context(), profile, binding, shape, g, options.fused, options.exact,
                  cutlass, options.verify, options.export_streams, options.capture_routed);
  // A selection over the host's masks builds F32 [n_kv, rows] tensors, whose
  // plane GGML strides in 32 bits (RE-037; model/qwen38.h Qwen38State).
  if (s.qsa_select && !builder.SelectsOnDevice() &&
      s.n_kv * s.rows * 4 > std::int64_t{std::numeric_limits<std::int32_t>::max()}) {
    return Rejected(
        "a chunk whose masks of every cell by every row pass GGML's 32-bit strides (RE-037)");
  }
  if (auto leaves = builder.Leaves(options); !leaves) {
    return std::unexpected(leaves.error());
  }
  builder.Build();
  return g;
}

}  // namespace jitllm::kernels::ggml
