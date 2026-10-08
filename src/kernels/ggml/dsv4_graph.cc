// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// A port of llama.cpp b29c606e2's src/models/deepseek4.cpp graph (and the
// llm_graph_context and llama_kv_cache parts it calls, the Hadamard matrix
// among them), which is MIT: its structure, helpers and parameters follow
// upstream's closely, so the file carries GGML's notice (docs/licensing.md).

#include "kernels/ggml/dsv4_graph.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ggml.h"
#include "kernels/ggml/dsv4_hc_norm.h"
#include "kernels/ggml/dsv4_outa.h"
#include "kernels/ggml/dsv4_qhead.h"
#include "kernels/ggml/dsv4_weighted_reduce.h"
#include "kernels/ggml/fusion.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/tensors.h"
#include "model/dspark.h"
#include "model/dsv4.h"

namespace jitllm::kernels::ggml {
namespace {

std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

// dsv4_rope_attn_factor (deepseek4.cpp:11-17).
float RopeAttnFactor(float freq_scale, float ext_factor) {
  if (ext_factor == 0.0f) {
    return 1.0f;
  }
  return 1.0f / (1.0f + (0.1f * logf(1.0f / freq_scale)));
}

// The RoPE parameters of llm_graph_context and deepseek4.cpp for this
// profile: YaRN (ext_factor 1, freq_scale 1 / factor) on the compressed
// layers, the indexer and the compressed rows; none on the window-only
// layers.
struct Rope {
  int n_ctx_orig = 0;
  float base = 0;
  float scale = 1;
  float ext = 0;
  float attn = 1;
  float beta_fast = 0;
  float beta_slow = 0;
};

Rope CompressedRope(const model::Dsv4Profile& p) {
  const float scale = 1.0f / p.rope_scale;
  return {.n_ctx_orig = static_cast<int>(p.yarn_original_context),
          .base = p.compress_rope_base,
          .scale = scale,
          .ext = 1.0f,
          .attn = RopeAttnFactor(scale, 1.0f),
          .beta_fast = p.yarn_beta_fast,
          .beta_slow = p.yarn_beta_slow};
}

Rope LayerRope(const model::Dsv4Profile& p, std::uint32_t ratio) {
  if (ratio != 0) {
    return CompressedRope(p);
  }
  return {.n_ctx_orig = 0,
          .base = p.rope_base,
          .scale = 1.0f,
          .ext = 0.0f,
          .attn = RopeAttnFactor(1.0f, 0.0f),
          .beta_fast = 0.0f,
          .beta_slow = 0.0f};
}

// LLAMA_ROPE_TYPE_NORM: consecutive pairs.
constexpr int kRopeMode = 0;

// The fast plan's fused form's conditions (Builder::Fused, Dsv4WaveSupport):
// the profile's, which the hyper-connection kernels and routed products
// take, and a layer's expert weights', each a jitllm.vecq type with gate and
// up alike.
bool FusedProfile(const model::Dsv4Profile& p) {
  return p.hc == 4 && (std::int64_t{p.width} * p.hc) % (kDsv4HcChunks * kDsv4HcChunkThreads) == 0 &&
         p.width % 1024 == 0 && p.width <= 8192 && p.experts == 256;
}

bool FusedTypes(ggml_type up, ggml_type gate, ggml_type down, ggml_type up_shared,
                ggml_type gate_shared, ggml_type down_shared) {
  return VecQType(up) && gate == up && VecQType(down) && VecQType(up_shared) &&
         gate_shared == up_shared && VecQType(down_shared);
}

// A wave's slot (BuildDsv4WaveGraph): its inputs and state, its shape, and
// its rows among the wave's.
struct Segment {
  Dsv4Graph* g = nullptr;
  const Dsv4ChunkShape* s = nullptr;
  std::int64_t first = 0;
  std::int64_t inject_rows = 0;
};

class Builder {
 public:
  // `sparse`: a target chunk's fast plan (Dsv4GraphOptions::fused), its
  // attention at depth sparse; never a draft block's. With `segments`, a
  // wave: `s` the joined shape (its rows every slot's), `g` the joined graph.
  Builder(ggml_context* c, const model::Dsv4Profile& p, const model::Dsv4Binding& b,
          const Dsv4ChunkShape& s, Dsv4Graph& g, const Dsv4GraphOptions& options, bool sparse,
          std::span<Segment> segments = {})
      : c_(c), p_(p), b_(b), s_(s), g_(g), o_(options), sparse_(sparse), segs_(segments) {}

  // A target chunk's inputs.
  void Inputs();
  // A wave's: the joined rows' and each slot's own.
  void WaveInputs();
  // Whether every layer takes the fast plan's fused form (a wave's
  // requirement).
  bool AllFused() const {
    for (std::uint32_t il = 0; il < p_.layers; ++il) {
      if (!Fused(il)) {
        return false;
      }
    }
    return true;
  }
  // The weights (and every layer's state) of the binding's blocks and head.
  std::expected<void, KernelFailure> Weights();
  // A target chunk's DSpark injection leaves: the drafter's weights and ring.
  std::expected<void, KernelFailure> InjectLeaves(const Dsv4Injection& inject);
  void Build();
  const std::vector<ggml_tensor*>& expanded() const { return expanded_; }
  // A wave's concurrent lanes (Dsv4WaveGraph::lanes), tagged as Build makes
  // the tensors.
  void SetLanes(LaneTags* lanes) { lanes_ = lanes; }
  // A DSpark draft block (the drafter's blocks over the binding's head).
  std::expected<void, KernelFailure> DraftInputs(DsparkGraph& d, const model::DsparkBinding& b);
  void BuildDraft(DsparkGraph& d);
  // A joined draft (BuildDsparkWaveGraph): the joined rows' inputs and each
  // slot's, then the blocks over every slot's rows and each slot's Markov
  // head.
  std::expected<void, KernelFailure> DraftWaveInputs(DsparkWaveGraph& d,
                                                     const model::DsparkBinding& b);
  void BuildDraftWave(DsparkWaveGraph& d);

 private:
  // The drafter's blocks over the rows (each slot's attention over its own
  // ring in a joined draft), its hyper-connection head and final norm, and
  // the target's head: the logits, [vocab, rows].
  ggml_tensor* DraftTrunk();
  // The Markov head over `nt` rows of `logits` from row `first`, whose
  // block tokens (the anchor first) `tokens` holds from `first`: each row's
  // logits biased by the row before it's argmax (the anchor's for the
  // first), the biased rows joined.
  ggml_tensor* Markov(ggml_tensor* logits, ggml_tensor* tokens, ggml_tensor* w1, ggml_tensor* w2,
                      std::int64_t first, std::int64_t nt);
  // A named intermediate, as llama.cpp's callback names it.
  void Name(ggml_tensor* t, std::string_view name, int il) {
    g_.named.emplace_back(il >= 0 ? std::format("{}-{}", name, il) : std::string(name), t);
  }
  void Expand(ggml_tensor* t) { expanded_.push_back(t); }

  ggml_tensor* Norm(ggml_tensor* x, ggml_tensor* weight) {
    return ggml_mul(c_, ggml_rms_norm(c_, x, p_.rms_eps), weight);
  }
  ggml_tensor* RopeExt(ggml_tensor* x, ggml_tensor* pos, const Rope& r, int n_ctx_orig) {
    return ggml_rope_ext(c_, x, pos, nullptr, static_cast<int>(p_.rope_dims), kRopeMode, n_ctx_orig,
                         r.base, r.scale, r.ext, r.attn, r.beta_fast, r.beta_slow);
  }
  // llama_mul_mat_hadamard (llama-impl.h).
  ggml_tensor* Hadamard(ggml_tensor* cur, ggml_tensor* rot) {
    const std::int64_t n = rot->ne[0];
    ggml_tensor* res = ggml_is_contiguous(cur)
                           ? ggml_reshape_2d(c_, cur, n, ggml_nelements(cur) / n)
                           : ggml_cont_2d(c_, cur, n, ggml_nelements(cur) / n);
    res = ggml_mul_mat(c_, rot, res);
    ggml_mul_mat_set_hint(res, GGML_HINT_SRC0_IS_HADAMARD);
    return ggml_reshape_4d(c_, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
  }

  ggml_tensor* HcPre(ggml_tensor* x, ggml_tensor* fn, ggml_tensor* scale, ggml_tensor* base,
                     ggml_tensor** post, ggml_tensor** comb, int il);
  ggml_tensor* HcHead(ggml_tensor* x);
  ggml_tensor* OverlapCompress(ggml_tensor* kv_state, ggml_tensor* score_state,
                               ggml_tensor* read_idxs, ggml_tensor* comp_pos, ggml_tensor* norm,
                               std::int64_t ratio, std::int64_t head);
  ggml_tensor* HcaCompress(ggml_tensor* kv_state, ggml_tensor* score_state, ggml_tensor* read_idxs,
                           ggml_tensor* comp_pos, ggml_tensor* norm, std::int64_t head);
  ggml_tensor* CompressFused(ggml_tensor* state_kv, ggml_tensor* state_score, ggml_tensor* kv,
                             ggml_tensor* score, ggml_tensor* read_idxs, ggml_tensor* comp_pos,
                             ggml_tensor* norm, std::int64_t ratio, std::int64_t head,
                             bool overlap);
  ggml_tensor* AppendZeroRow(ggml_tensor* t, bool neg_inf);
  ggml_tensor* LidTopK(const Dsv4LayerTensors& l, ggml_tensor* qr, ggml_tensor* cur, int il);
  ggml_tensor* TopKMask(ggml_tensor* kq_mask, ggml_tensor* top_k);
  ggml_tensor* AttnMha(ggml_tensor* q, ggml_tensor* k, ggml_tensor* kq_mask, ggml_tensor* sinks,
                       std::int64_t n_kv_max);
  ggml_tensor* AttnMhaRow(ggml_tensor* q, ggml_tensor* k, ggml_tensor* kq_mask, ggml_tensor* sinks,
                          std::int64_t n_kv_max, bool sparse_any = false);
  // The fast plan's sparse attention and indexer (sparse_).
  ggml_tensor* AttentionSparse(std::uint32_t il, ggml_tensor* q, ggml_tensor* kv, ggml_tensor* qr,
                               ggml_tensor* cur);
  ggml_tensor* LidTopKSparse(const Dsv4LayerTensors& l, ggml_tensor* qr, ggml_tensor* cur, int il);
  // dsv4_hc_mean (deepseek4.cpp:270-278): the mean of x's streams.
  ggml_tensor* HcMean(ggml_tensor* x);
  void Inject(const DsparkInjectTensors& t, const Dsv4Injection& inject);
  ggml_tensor* CpyK(ggml_tensor* cache, ggml_tensor* k_cur, ggml_tensor* idxs);
  ggml_tensor* GetK(ggml_tensor* cache, std::int64_t n_kv);
  ggml_tensor* Attention(std::uint32_t il, ggml_tensor* cur);
  // A wave's attention: the row-local projections over every slot's rows,
  // each slot's compressors, indexer and attention over its own state.
  ggml_tensor* AttentionWave(std::uint32_t il, ggml_tensor* cur);
  // A wave's DSpark injection: the projections joined, each slot's rows
  // stored in its own ring.
  void InjectWave(const DsparkInjectTensors& t, const Dsv4Injection& inject);
  // A slot's rows [first, first + rows) of a joined [ne0, rows, ...] tensor.
  ggml_tensor* Rows2(ggml_tensor* t, const Segment& seg) const {
    return ggml_view_2d(c_, t, t->ne[0], seg.s->rows, t->nb[1],
                        static_cast<std::size_t>(seg.first) * t->nb[1]);
  }
  ggml_tensor* Rows3(ggml_tensor* t, const Segment& seg) const {
    return ggml_view_3d(c_, t, t->ne[0], t->ne[1], seg.s->rows, t->nb[1], t->nb[2],
                        static_cast<std::size_t>(seg.first) * t->nb[2]);
  }
  void StateLeaves(Dsv4LayerTensors& l, std::uint32_t ratio, const Dsv4ChunkShape& s);
  ggml_tensor* Moe(std::uint32_t il, ggml_tensor* cur);
  // The fast plan's forms (Dsv4GraphOptions::fused), where Fused() holds.
  bool Fused(std::uint32_t il) const;
  // HcPre then Norm(·, norm): the normed row, and post and comb as views.
  ggml_tensor* HcPreFused(ggml_tensor* x, ggml_tensor* fn, ggml_tensor* scale, ggml_tensor* base,
                          ggml_tensor* norm, ggml_tensor** post, ggml_tensor** comb);
  ggml_tensor* MoeFused(std::uint32_t il, ggml_tensor* cur);
  // HcPre and Norm: in a fused layer whose mixing weights jitllm.dsv4.hc_mix
  // reads, HcPreFused; otherwise (a quantized mix, or an unfused layer)
  // GGML's product, a wave's per slot as FloatMm runs it.
  ggml_tensor* PreNorm(std::uint32_t il, ggml_tensor* x, ggml_tensor* fn, ggml_tensor* scale,
                       ggml_tensor* base, ggml_tensor* norm, ggml_tensor** post,
                       ggml_tensor** comb) {
    if (Fused(il) && Dsv4HcMixWeightType(fn->type)) {
      return HcPreFused(x, fn, scale, base, norm, post, comb);
    }
    return Norm(HcPre(x, fn, scale, base, post, comb, static_cast<int>(il)), norm);
  }

  ggml_context* c_;
  const model::Dsv4Profile& p_;
  const model::Dsv4Binding& b_;
  const Dsv4ChunkShape& s_;
  Dsv4Graph& g_;
  const Dsv4GraphOptions& o_;
  bool sparse_;
  std::span<Segment> segs_;  // a wave's slots; empty for one chunk
  std::vector<ggml_tensor*> expanded_;
  LaneTags* lanes_ = nullptr;
  std::uint32_t regions_ = 0;
  ggml_tensor* cursor_ = nullptr;  // the context's last tensor Last found

  // The context's last tensor so far (null if none).
  ggml_tensor* Last() {
    ggml_tensor* t = cursor_ != nullptr ? cursor_ : ggml_get_first_tensor(c_);
    if (t == nullptr) {
      return nullptr;
    }
    for (ggml_tensor* next = ggml_get_next_tensor(c_, t); next != nullptr;
         next = ggml_get_next_tensor(c_, next)) {
      t = next;
    }
    cursor_ = t;
    return t;
  }
  // Tags the tensors made since `mark` (Last before them) with a lane and
  // region.
  void Tag(ggml_tensor* mark, std::uint8_t lane, std::uint32_t region) {
    if (lanes_ == nullptr) {
      return;
    }
    for (ggml_tensor* t = mark != nullptr ? ggml_get_next_tensor(c_, mark)
                                          : ggml_get_first_tensor(c_);
         t != nullptr; t = ggml_get_next_tensor(c_, t)) {
      lanes_->emplace_back(t, LaneTag{.lane = lane, .region = region});
    }
  }
  // The fast plan's activations quantized once (Q8Of), by input.
  std::unordered_map<const ggml_tensor*, ggml_tensor*> q8_;

  ggml_tensor* Q8Of(ggml_tensor* x) {
    if (const auto found = q8_.find(x); found != q8_.end()) {
      return found->second;
    }
    ggml_tensor* q = QuantizeQ8(c_, x);
    q8_.emplace(x, q);
    return q;
  }
  // The most rows jitllm.vecq takes here: a chunk's kVecQTokens, a wave's
  // kDsv4WaveRows (each token's sums the same at any count of two or more).
  std::int64_t VecQRows() const { return segs_.empty() ? kVecQTokens : kDsv4WaveRows; }
  // GGML's float vector kernel keeps each column's sums independent of the
  // count up to this many columns (graph_plan.h kRowInvariantColumns); a
  // wider wave runs a float product over each group of whole slots whose
  // rows fit that count (each slot's columns as its own chunk's), and joins
  // the results. A quantized weight's GGML
  // product (MMVQ) picks its launch by the column count, so a wave runs it
  // per slot at any width.
  static constexpr std::int64_t kVectorFloatColumns = 8;
  ggml_tensor* FloatMm(ggml_tensor* w, ggml_tensor* x, bool f32_acc = false) {
    if (segs_.empty() || (x->ne[1] <= kVectorFloatColumns && !ggml_is_quantized(w->type)) ||
        x->ne[2] != 1 || x->ne[3] != 1) {
      ggml_tensor* out = ggml_mul_mat(c_, w, x);
      if (f32_acc) {
        ggml_prec_set_acc(out, GGML_PREC_F32);
      }
      return out;
    }
    // Whole slots together while their rows fit the invariant count (a
    // float weight's product; a quantized one's a slot at a time).
    ggml_tensor* joined = nullptr;
    for (std::size_t i = 0; i < segs_.size();) {
      std::size_t j = i + 1;
      std::int64_t rows = segs_[i].s->rows;
      while (!ggml_is_quantized(w->type) && j < segs_.size() &&
             rows + segs_[j].s->rows <= kVectorFloatColumns) {
        rows += segs_[j].s->rows;
        ++j;
      }
      ggml_tensor* part = ggml_view_2d(c_, x, x->ne[0], rows, x->nb[1],
                                       static_cast<std::size_t>(segs_[i].first) * x->nb[1]);
      ggml_tensor* one = ggml_mul_mat(c_, w, part);
      if (f32_acc) {
        ggml_prec_set_acc(one, GGML_PREC_F32);
      }
      joined = joined != nullptr ? ggml_concat(c_, joined, one, 1) : one;
      i = j;
    }
    return joined;
  }
  // A product: in the fast plan, a quantized 2D weight over at most
  // VecQRows() rows of F32 activations is jitllm.vecq over the input's one
  // quantization; anything else GGML's mul_mat (a wide wave's float
  // products a slot at a time, FloatMm).
  bool VecQInput(const ggml_tensor* x) const {
    return o_.fused && x->type == GGML_TYPE_F32 && x->ne[1] <= VecQRows() && x->ne[2] == 1 &&
           x->ne[3] == 1 && x->nb[0] == sizeof(float) && x->ne[0] % 32 == 0;
  }
  static bool VecQWeight(const ggml_tensor* w, const ggml_tensor* x) {
    return w != nullptr && w->ne[2] == 1 && w->ne[3] == 1 && VecQType(w->type) &&
           w->ne[0] == x->ne[0];
  }
  ggml_tensor* Mm(ggml_tensor* w, ggml_tensor* x) {
    if (!VecQInput(x) || !VecQWeight(w, x)) {
      return segs_.empty() ? ggml_mul_mat(c_, w, x) : FloatMm(w, x);
    }
    return VecQ(c_, w, Q8Of(x), nullptr, x->ne[1], false);
  }
  // Mm, but with shared_f16_inputs an F16-weight product of a wide F32 input
  // reads one F16 copy of it, made once for every such product of that input.
  ggml_tensor* MmShared(ggml_tensor* w, ggml_tensor* x) {
    if (!o_.fused || !o_.shared_f16_inputs || w->type != GGML_TYPE_F16 || x->ne[1] < 64 ||
        (VecQInput(x) && VecQWeight(w, x)) || !Dsv4F16CopyFits(x)) {
      return Mm(w, x);
    }
    if (f16_source_ != x) {
      f16_source_ = x;
      f16_copy_ = Dsv4F16Copy(c_, x);
    }
    return ggml_mul_mat(c_, w, f16_copy_);
  }
  ggml_tensor* f16_source_ = nullptr;
  ggml_tensor* f16_copy_ = nullptr;
};

std::expected<ggml_tensor*, KernelFailure> Leaf(ggml_context* c, const model::Dsv4Tensor& t,
                                                std::string_view role) {
  auto type = GgmlTypeOf(t.type);
  if (!type) {
    return std::unexpected(type.error());
  }
  std::array<std::int64_t, 4> ne = {1, 1, 1, 1};
  if (t.ne.empty() || t.ne.size() > 4) {
    return Rejected(std::format("{}: not a GGML shape", role));
  }
  for (std::size_t i = 0; i < t.ne.size(); ++i) {
    ne[i] = static_cast<std::int64_t>(t.ne[i]);
  }
  if (ne[0] % ggml_blck_size(*type) != 0) {
    return Rejected(std::format("{}: rows are not whole blocks", role));
  }
  return ggml_new_tensor(c, *type, static_cast<int>(t.ne.size()), ne.data());
}

void Builder::Inputs() {
  const std::int64_t n = s_.rows;
  g_.embd = ggml_new_tensor_2d(c_, GGML_TYPE_F32, p_.width, n);
  g_.tokens = ggml_new_tensor_1d(c_, GGML_TYPE_I32, n);
  g_.positions = ggml_new_tensor_1d(c_, GGML_TYPE_I32, n);
  g_.raw_k_idxs = ggml_new_tensor_1d(c_, GGML_TYPE_I64, n);
  g_.device_raw_mask = o_.raw_mask_context != 0;
  g_.raw_mask =
      g_.device_raw_mask
          ? CausalRingMask(c_, g_.positions, s_.raw_n_kv, 0, static_cast<std::int32_t>(n),
                           static_cast<std::int32_t>(s_.raw_cells),
                           static_cast<std::int32_t>(p_.window),
                           static_cast<std::int32_t>(o_.raw_mask_context), CausalMaskRows::kExact)
          : ggml_new_tensor_4d(c_, GGML_TYPE_F16, s_.raw_n_kv, n, 1, 1);
  g_.out_ids = ggml_new_tensor_1d(c_, GGML_TYPE_I32, s_.outputs == 0 ? n : s_.outputs);
  const auto comp = [&](Dsv4CompInputs& in, std::int64_t blocks, std::int64_t persist,
                        std::int64_t reads, std::int64_t n_kv) {
    in.state_pos = ggml_new_tensor_1d(c_, GGML_TYPE_I32, n);
    in.persist_src = ggml_new_tensor_1d(c_, GGML_TYPE_I32, persist);
    in.persist_dst = ggml_new_tensor_1d(c_, GGML_TYPE_I32, persist);
    in.read_idxs = ggml_new_tensor_1d(c_, GGML_TYPE_I32, reads);
    in.write_idxs = ggml_new_tensor_1d(c_, GGML_TYPE_I64, blocks);
    in.write_pos = ggml_new_tensor_1d(c_, GGML_TYPE_I32, blocks);
    // The fast plan masks the compressed rows on the device from each
    // row's count.
    if (!sparse_) {
      in.mask = ggml_new_tensor_4d(c_, GGML_TYPE_F16, n_kv, n, 1, 1);
    }
  };
  const std::int64_t csa_reads = 2 * std::int64_t{model::kDsv4CsaRatio} * s_.csa_blocks;
  comp(g_.csa, s_.csa_blocks, s_.csa_persist, csa_reads, s_.csa_n_kv);
  comp(g_.hca, s_.hca_blocks, s_.hca_persist, model::kDsv4HcaRatio * s_.hca_blocks, s_.hca_n_kv);
  comp(g_.lid, s_.csa_blocks, s_.csa_persist, csa_reads, s_.csa_n_kv);
  g_.lid_rot = ggml_new_tensor_2d(c_, GGML_TYPE_F32, p_.indexer_head_dim, p_.indexer_head_dim);
  if (sparse_) {
    g_.csa_visible = ggml_new_tensor_1d(c_, GGML_TYPE_I32, n);
    g_.hca_visible = ggml_new_tensor_1d(c_, GGML_TYPE_I32, n);
    return;
  }
  const std::int64_t top_k = std::min<std::int64_t>(s_.csa_n_kv, p_.indexer_top_k);
  g_.top_k_zeros = ggml_new_tensor_4d(c_, GGML_TYPE_F16, 1, top_k, n, 1);
}

void Builder::WaveInputs() {
  // The joined rows: what row-local operations read.
  const std::int64_t n = s_.rows;
  g_.embd = ggml_new_tensor_2d(c_, GGML_TYPE_F32, p_.width, n);
  g_.tokens = ggml_new_tensor_1d(c_, GGML_TYPE_I32, n);
  g_.positions = ggml_new_tensor_1d(c_, GGML_TYPE_I32, n);
  for (Dsv4CompInputs* in : {&g_.csa, &g_.hca, &g_.lid}) {
    in->state_pos = ggml_new_tensor_1d(c_, GGML_TYPE_I32, n);
  }
  g_.lid_rot = ggml_new_tensor_2d(c_, GGML_TYPE_F32, p_.indexer_head_dim, p_.indexer_head_dim);
  // Each slot's own: its cells, window mask, compressor lists and counts.
  for (Segment& seg : segs_) {
    Dsv4Graph& sg = *seg.g;
    const Dsv4ChunkShape& s = *seg.s;
    sg.raw_k_idxs = ggml_new_tensor_1d(c_, GGML_TYPE_I64, s.rows);
    sg.device_raw_mask = o_.raw_mask_context != 0;
    sg.raw_mask =
        sg.device_raw_mask
            ? CausalRingMask(c_, g_.positions, s.raw_n_kv, static_cast<std::int32_t>(seg.first),
                             static_cast<std::int32_t>(s.rows),
                             static_cast<std::int32_t>(s.raw_cells),
                             static_cast<std::int32_t>(p_.window),
                             static_cast<std::int32_t>(o_.raw_mask_context), CausalMaskRows::kExact)
            : ggml_new_tensor_4d(c_, GGML_TYPE_F16, s.raw_n_kv, s.rows, 1, 1);
    const auto comp = [&](Dsv4CompInputs& in, std::int64_t blocks, std::int64_t persist,
                          std::int64_t reads) {
      in.persist_src = ggml_new_tensor_1d(c_, GGML_TYPE_I32, persist);
      in.persist_dst = ggml_new_tensor_1d(c_, GGML_TYPE_I32, persist);
      in.read_idxs = ggml_new_tensor_1d(c_, GGML_TYPE_I32, reads);
      in.write_idxs = ggml_new_tensor_1d(c_, GGML_TYPE_I64, blocks);
      in.write_pos = ggml_new_tensor_1d(c_, GGML_TYPE_I32, blocks);
    };
    const std::int64_t csa_reads = 2 * std::int64_t{model::kDsv4CsaRatio} * s.csa_blocks;
    comp(sg.csa, s.csa_blocks, s.csa_persist, csa_reads);
    comp(sg.hca, s.hca_blocks, s.hca_persist, model::kDsv4HcaRatio * s.hca_blocks);
    comp(sg.lid, s.csa_blocks, s.csa_persist, csa_reads);
    sg.csa_visible = ggml_new_tensor_1d(c_, GGML_TYPE_I32, s.rows);
    sg.hca_visible = ggml_new_tensor_1d(c_, GGML_TYPE_I32, s.rows);
  }
}

std::expected<void, KernelFailure> Builder::Weights() {
  const Dsv4GraphOptions& options = o_;
  const auto leaf = [&](ggml_tensor*& into, const model::Dsv4Tensor& t,
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
  JITLLM_LEAF(g_.output_norm, b_.output_norm)
  JITLLM_LEAF(g_.output, b_.output)
  JITLLM_LEAF(g_.hc_head_fn, b_.hc_head_fn)
  JITLLM_LEAF(g_.hc_head_base, b_.hc_head_base)
  JITLLM_LEAF(g_.hc_head_scale, b_.hc_head_scale)
  g_.layers.resize(p_.layers);
  for (Segment& seg : segs_) {
    seg.g->layers.resize(p_.layers);
  }
  if (!options.expert_stride.empty() && options.expert_stride.size() != p_.layers) {
    return Rejected("an expert stride per layer, or none");
  }
  for (std::uint32_t il = 0; il < p_.layers; ++il) {
    const model::Dsv4Layer& w = b_.layers[il];
    Dsv4LayerTensors& l = g_.layers[il];
    JITLLM_LEAF(l.attn_norm, w.attn_norm)
    JITLLM_LEAF(l.attn_sinks, w.attn_sinks)
    JITLLM_LEAF(l.q_a, w.q_a)
    JITLLM_LEAF(l.q_a_norm, w.q_a_norm)
    JITLLM_LEAF(l.q_b, w.q_b)
    JITLLM_LEAF(l.kv, w.kv)
    JITLLM_LEAF(l.kv_norm, w.kv_norm) {
      // llama.cpp loads wo_a as [heads·head / groups, o_lora, groups].
      auto type = GgmlTypeOf(w.out_a.type);
      if (!type) {
        return std::unexpected(type.error());
      }
      l.out_a = ggml_new_tensor_3d(c_, *type, static_cast<std::int64_t>(w.out_a.ne[0]), p_.o_lora,
                                   p_.o_groups);
    }
    JITLLM_LEAF(l.out_b, w.out_b)
    JITLLM_LEAF(l.hc_attn_fn, w.hc_attn_fn)
    JITLLM_LEAF(l.hc_attn_base, w.hc_attn_base)
    JITLLM_LEAF(l.hc_attn_scale, w.hc_attn_scale)
    JITLLM_LEAF(l.hc_ffn_fn, w.hc_ffn_fn)
    JITLLM_LEAF(l.hc_ffn_base, w.hc_ffn_base)
    JITLLM_LEAF(l.hc_ffn_scale, w.hc_ffn_scale)
    if (w.ratio != 0) {
      JITLLM_LEAF(l.comp_kv, w.comp_kv)
      JITLLM_LEAF(l.comp_gate, w.comp_gate)
      JITLLM_LEAF(l.comp_ape, w.comp_ape)
      JITLLM_LEAF(l.comp_norm, w.comp_norm)
    }
    if (w.ratio == model::kDsv4CsaRatio) {
      JITLLM_LEAF(l.idx_q_b, w.idx_q_b)
      JITLLM_LEAF(l.idx_proj, w.idx_proj)
      JITLLM_LEAF(l.idx_comp_kv, w.idx_comp_kv)
      JITLLM_LEAF(l.idx_comp_gate, w.idx_comp_gate)
      JITLLM_LEAF(l.idx_comp_ape, w.idx_comp_ape)
      JITLLM_LEAF(l.idx_comp_norm, w.idx_comp_norm)
    }
    JITLLM_LEAF(l.ffn_norm, w.ffn_norm)
    JITLLM_LEAF(l.router, w.router)
    if (w.hash) {
      JITLLM_LEAF(l.tid2eid, w.tid2eid)
    } else {
      JITLLM_LEAF(l.router_bias, w.router_bias)
    }
    JITLLM_LEAF(l.up_shexp, w.up_shexp)
    JITLLM_LEAF(l.gate_shexp, w.gate_shexp)
    JITLLM_LEAF(l.down_shexp, w.down_shexp)
    // Routed experts: 3D at the layer's stride.
    const std::uint64_t stride = options.expert_stride.empty() ? 0 : options.expert_stride[il];
    const auto experts = [&](ggml_tensor*& into,
                             const model::Dsv4Tensor& t) -> std::expected<void, KernelFailure> {
      auto type = GgmlTypeOf(t.type);
      if (!type) {
        return std::unexpected(type.error());
      }
      into = ggml_new_tensor_3d(c_, *type, static_cast<std::int64_t>(t.ne[0]),
                                static_cast<std::int64_t>(t.ne[1]), p_.experts);
      if (stride != 0) {
        const std::size_t slice = into->nb[2];
        if (stride < slice || stride % ggml_type_size(*type) != 0) {
          return Rejected(
              std::format("layer {}: an expert stride of {} bytes does not hold "
                          "whole {} slices",
                          il, stride, t.type));
        }
        into->nb[2] = stride;
        into->nb[3] = stride * static_cast<std::size_t>(p_.experts);
      }
      return {};
    };
    if (auto e = experts(l.up_exps, w.up_exps); !e) {
      return e;
    }
    if (auto e = experts(l.gate_exps, w.gate_exps); !e) {
      return e;
    }
    if (auto e = experts(l.down_exps, w.down_exps); !e) {
      return e;
    }
    if (segs_.empty()) {
      StateLeaves(l, w.ratio, s_);
    } else {
      // A wave: each slot's state its own (its layout's shape).
      for (Segment& seg : segs_) {
        StateLeaves(seg.g->layers[il], w.ratio, *seg.s);
      }
    }
  }
#undef JITLLM_LEAF
  return {};
}

void Builder::StateLeaves(Dsv4LayerTensors& l, std::uint32_t ratio, const Dsv4ChunkShape& s) {
  // State. In the fast plan a compressed layer reads its window cells and
  // its compressed rows as one tensor: the compressed cache follows the
  // window's in the state (model/dsv4.h), a view of the whole.
  const std::int64_t head = p_.head_dim;
  std::int64_t joined = 0;
  if (sparse_ && ratio == model::kDsv4CsaRatio) {
    joined = s.csa_cells;
  } else if (sparse_ && ratio == model::kDsv4HcaRatio) {
    joined = s.hca_cells;
  }
  l.raw_k = ggml_new_tensor_3d(c_, GGML_TYPE_F16, head, s.raw_cells + joined, 1);
  const auto after_window = [&](std::int64_t cells) {
    return ggml_view_3d(c_, l.raw_k, head, cells, 1, l.raw_k->nb[1],
                        l.raw_k->nb[1] * static_cast<std::size_t>(cells),
                        l.raw_k->nb[1] * static_cast<std::size_t>(s.raw_cells));
  };
  if (ratio == model::kDsv4CsaRatio) {
    l.csa_k = joined != 0 ? after_window(s.csa_cells)
                          : ggml_new_tensor_3d(c_, GGML_TYPE_F16, head, s.csa_cells, 1);
    l.csa_state_kv = ggml_new_tensor_2d(c_, GGML_TYPE_F32, 2 * head, s.csa_state_rows);
    l.csa_state_score = ggml_new_tensor_2d(c_, GGML_TYPE_F32, 2 * head, s.csa_state_rows);
    l.lid_k = ggml_new_tensor_3d(c_, GGML_TYPE_F16, p_.indexer_head_dim, s.csa_cells, 1);
    const std::int64_t lid_ring = 2 * std::int64_t{p_.indexer_head_dim};
    l.lid_state_kv = ggml_new_tensor_2d(c_, GGML_TYPE_F32, lid_ring, s.csa_state_rows);
    l.lid_state_score = ggml_new_tensor_2d(c_, GGML_TYPE_F32, lid_ring, s.csa_state_rows);
  } else if (ratio == model::kDsv4HcaRatio) {
    l.hca_k = joined != 0 ? after_window(s.hca_cells)
                          : ggml_new_tensor_3d(c_, GGML_TYPE_F16, head, s.hca_cells, 1);
    l.hca_state_kv = ggml_new_tensor_2d(c_, GGML_TYPE_F32, head, s.hca_state_rows);
    l.hca_state_score = ggml_new_tensor_2d(c_, GGML_TYPE_F32, head, s.hca_state_rows);
  }
}

std::expected<void, KernelFailure> Builder::InjectLeaves(const Dsv4Injection& inject) {
  if (inject.profile == nullptr || inject.binding == nullptr ||
      (segs_.empty() && (inject.rows <= 0 || inject.rows > s_.rows)) || inject.ring <= 0) {
    return Rejected("not a DSpark injection of this chunk");
  }
  for (const Segment& seg : segs_) {
    if (seg.inject_rows <= 0 || seg.inject_rows > seg.s->rows) {
      return Rejected("not a DSpark injection of this wave's slot");
    }
  }
  const model::Dsv4Profile& dp = inject.profile->blocks;
  const model::DsparkBinding& db = *inject.binding;
  if (db.blocks.layers.size() != dp.layers || dp.width != p_.width) {
    return Rejected("the injection's binding is not its drafter's");
  }
  DsparkInjectTensors t;
  if (segs_.empty()) {
    t.cells = ggml_new_tensor_1d(c_, GGML_TYPE_I64, inject.rows);
  }
  auto fc = Leaf(c_, db.fc, "fc");
  auto enc_norm = Leaf(c_, db.enc_norm, "enc_norm");
  if (!fc || !enc_norm) {
    return std::unexpected(!fc ? fc.error() : enc_norm.error());
  }
  t.fc = *fc;
  t.enc_norm = *enc_norm;
  for (std::uint32_t il = 0; il < dp.layers; ++il) {
    auto kv = Leaf(c_, db.blocks.layers[il].kv, "drafter kv");
    auto kv_norm = Leaf(c_, db.blocks.layers[il].kv_norm, "drafter kv_norm");
    if (!kv || !kv_norm) {
      return std::unexpected(!kv ? kv.error() : kv_norm.error());
    }
    t.kv.push_back(*kv);
    t.kv_norm.push_back(*kv_norm);
    if (segs_.empty()) {
      t.ring.push_back(ggml_new_tensor_3d(c_, GGML_TYPE_F16, dp.head_dim, inject.ring, 1));
    }
  }
  // A wave: each slot's cells and ring its own; the weights the joined graph's.
  for (Segment& seg : segs_) {
    DsparkInjectTensors own;
    own.cells = ggml_new_tensor_1d(c_, GGML_TYPE_I64, seg.inject_rows);
    for (std::uint32_t il = 0; il < dp.layers; ++il) {
      own.ring.push_back(ggml_new_tensor_3d(c_, GGML_TYPE_F16, dp.head_dim, inject.ring, 1));
    }
    seg.g->inject = std::move(own);
  }
  g_.inject = std::move(t);
  return {};
}

// build_hc_pre with the fused comb and pre (deepseek4.cpp:354-410).
ggml_tensor* Builder::HcPre(ggml_tensor* x, ggml_tensor* fn, ggml_tensor* scale, ggml_tensor* base,
                            ggml_tensor** post, ggml_tensor** comb, int il) {
  const std::int64_t hc = p_.hc;
  const std::int64_t nt = x->ne[2];
  ggml_tensor* flat = ggml_reshape_2d(c_, x, p_.hc_width(), nt);
  // The F16 rows are exactly what the product would convert the F32 norm
  // to; for F32 mixing weights, F32 rows are the norm itself. Either way
  // the post before them writes them (dsv4_hc_norm.h).
  const bool rows = o_.fused && o_.hc_f16_rows && nt >= 64 &&
                    (fn->type == GGML_TYPE_F16 || fn->type == GGML_TYPE_F32) &&
                    Dsv4HcNormF16Fits(flat, p_.rms_eps);
  ggml_tensor* flat_norm =
      rows ? Dsv4HcNormF16(c_, flat, p_.rms_eps, fn->type) : ggml_rms_norm(c_, flat, p_.rms_eps);
  ggml_tensor* mixes = FloatMm(fn, flat_norm);
  ggml_tensor* scale_pre = ggml_view_1d(c_, scale, 1, ggml_row_size(scale->type, 0));
  ggml_tensor* scale_post = ggml_view_1d(c_, scale, 1, ggml_row_size(scale->type, 1));
  ggml_tensor* base_pre = ggml_view_1d(c_, base, hc, ggml_row_size(base->type, 0));
  ggml_tensor* base_post = ggml_view_1d(c_, base, hc, ggml_row_size(base->type, hc));
  ggml_tensor* pre = ggml_view_2d(c_, mixes, hc, nt, mixes->nb[1], ggml_row_size(mixes->type, 0));
  pre = ggml_add(c_, ggml_mul(c_, pre, scale_pre), base_pre);
  pre = ggml_sigmoid(c_, pre);
  pre = ggml_scale_bias(c_, pre, 1.0f, p_.hc_eps);
  *post = ggml_view_2d(c_, mixes, hc, nt, mixes->nb[1], ggml_row_size(mixes->type, hc));
  *post = ggml_add(c_, ggml_mul(c_, *post, scale_post), base_post);
  *post = ggml_sigmoid(c_, *post);
  *post = ggml_scale(c_, *post, 2.0f);
  *comb = ggml_dsv4_hc_comb(c_, mixes, scale, base, p_.hc_eps,
                            static_cast<std::int32_t>(p_.sinkhorn_iterations));
  (void)il;
  return ggml_dsv4_hc_pre(c_, x, pre);
}

// build_hc_head (deepseek4.cpp:449-469), its pre fused (il -1 is not fused
// upstream: build_hc_pre's weighted sum runs as views, muls and adds).
ggml_tensor* Builder::HcHead(ggml_tensor* x) {
  const std::int64_t hc = p_.hc;
  const std::int64_t nt = x->ne[2];
  ggml_tensor* flat = ggml_reshape_2d(c_, x, p_.hc_width(), nt);
  ggml_tensor* flat_norm = ggml_rms_norm(c_, flat, p_.rms_eps);
  ggml_tensor* mixes = FloatMm(g_.hc_head_fn, flat_norm);
  ggml_tensor* pre = ggml_add(c_, ggml_mul(c_, mixes, g_.hc_head_scale), g_.hc_head_base);
  pre = ggml_sigmoid(c_, pre);
  pre = ggml_scale_bias(c_, pre, 1.0f, p_.hc_eps);
  // build_hc_pre(x, pre, -1): cparams.fused_dsv4_hc_pre && il >= 0 is false.
  ggml_tensor* result = nullptr;
  for (std::int64_t ih = 0; ih < hc; ++ih) {
    const auto h = static_cast<std::size_t>(ih);
    ggml_tensor* xh = ggml_view_2d(c_, x, p_.width, nt, x->nb[2], h * x->nb[1]);
    ggml_tensor* wh = ggml_view_2d(c_, pre, 1, nt, pre->nb[1], h * pre->nb[0]);
    ggml_tensor* cur = ggml_mul(c_, xh, wh);
    result = result != nullptr ? ggml_add(c_, result, cur) : cur;
  }
  return result;
}

// dsv4_append_zero_row.
ggml_tensor* Builder::AppendZeroRow(ggml_tensor* t, bool neg_inf) {
  ggml_tensor* row = ggml_view_1d(c_, t, t->ne[0], 0);
  row = neg_inf ? ggml_scale_bias(c_, row, 0.0f, -INFINITY) : ggml_scale(c_, row, 0.0f);
  row = ggml_reshape_2d(c_, row, t->ne[0], 1);
  return ggml_concat(c_, t, row, 1);
}

// build_overlap_compressed_kv_from_state (deepseek4.cpp:518-589).
ggml_tensor* Builder::OverlapCompress(ggml_tensor* kv_state, ggml_tensor* score_state,
                                      ggml_tensor* read_idxs, ggml_tensor* comp_pos,
                                      ggml_tensor* norm, std::int64_t ratio, std::int64_t head) {
  const std::int64_t n_blocks = comp_pos->ne[0];
  kv_state = AppendZeroRow(kv_state, false);
  score_state = AppendZeroRow(score_state, true);
  const std::int64_t n_read = ratio * n_blocks;
  const auto reads = static_cast<std::size_t>(n_read);
  ggml_tensor* kv_rows = ggml_get_rows(c_, kv_state, read_idxs);
  ggml_tensor* score_rows = ggml_get_rows(c_, score_state, read_idxs);
  ggml_tensor* kv_prev = ggml_cont(c_, ggml_view_2d(c_, kv_rows, head, n_read, kv_rows->nb[1], 0));
  kv_prev = ggml_reshape_3d(c_, kv_prev, head, ratio, n_blocks);
  ggml_tensor* score_prev =
      ggml_cont(c_, ggml_view_2d(c_, score_rows, head, n_read, score_rows->nb[1], 0));
  score_prev = ggml_reshape_3d(c_, score_prev, head, ratio, n_blocks);
  ggml_tensor* kv_cur =
      ggml_cont(c_, ggml_view_2d(c_, kv_rows, head, n_read, kv_rows->nb[1],
                                 (reads * kv_rows->nb[1]) + ggml_row_size(kv_rows->type, head)));
  kv_cur = ggml_reshape_3d(c_, kv_cur, head, ratio, n_blocks);
  ggml_tensor* score_cur = ggml_cont(
      c_, ggml_view_2d(c_, score_rows, head, n_read, score_rows->nb[1],
                       (reads * score_rows->nb[1]) + ggml_row_size(score_rows->type, head)));
  score_cur = ggml_reshape_3d(c_, score_cur, head, ratio, n_blocks);
  ggml_tensor* values = ggml_concat(c_, kv_prev, kv_cur, 1);
  ggml_tensor* scores = ggml_concat(c_, score_prev, score_cur, 1);
  values = ggml_cont(c_, ggml_permute(c_, values, 1, 0, 2, 3));
  scores = ggml_cont(c_, ggml_permute(c_, scores, 1, 0, 2, 3));
  ggml_tensor* weights = ggml_soft_max(c_, scores);
  ggml_tensor* comp = ggml_mul(c_, values, weights);
  comp = ggml_sum_rows(c_, comp);
  comp = ggml_cont(c_, ggml_permute(c_, comp, 1, 0, 2, 3));
  comp = Norm(comp, norm);
  const Rope r = CompressedRope(p_);
  comp = RopeExt(comp, comp_pos, r, r.n_ctx_orig);
  return ggml_rope_set_offset(comp, static_cast<int>(head - p_.rope_dims));
}

// The fast plan's compressor (jitllm.dsv4.compress over the state and the
// chunk's rows, no source copy), then the norm and RoPE as above.
ggml_tensor* Builder::CompressFused(ggml_tensor* state_kv, ggml_tensor* state_score,
                                    ggml_tensor* kv, ggml_tensor* score, ggml_tensor* read_idxs,
                                    ggml_tensor* comp_pos, ggml_tensor* norm, std::int64_t ratio,
                                    std::int64_t head, bool overlap) {
  ggml_tensor* comp = Dsv4Compress(c_, state_kv, state_score, kv, score, read_idxs, ratio, overlap);
  comp = Norm(comp, norm);
  const Rope r = CompressedRope(p_);
  comp = RopeExt(comp, comp_pos, r, r.n_ctx_orig);
  return ggml_rope_set_offset(comp, static_cast<int>(head - p_.rope_dims));
}

// build_hca_compressed_kv_from_state (deepseek4.cpp:471-516).
ggml_tensor* Builder::HcaCompress(ggml_tensor* kv_state, ggml_tensor* score_state,
                                  ggml_tensor* read_idxs, ggml_tensor* comp_pos, ggml_tensor* norm,
                                  std::int64_t head) {
  const std::int64_t n_blocks = comp_pos->ne[0];
  const std::int64_t ratio = model::kDsv4HcaRatio;
  ggml_tensor* kv = ggml_get_rows(c_, kv_state, read_idxs);
  kv = ggml_reshape_3d(c_, kv, head, ratio, n_blocks);
  ggml_tensor* score = ggml_get_rows(c_, score_state, read_idxs);
  score = ggml_reshape_3d(c_, score, head, ratio, n_blocks);
  ggml_tensor* values = ggml_cont(c_, ggml_permute(c_, kv, 1, 0, 2, 3));
  ggml_tensor* scores = ggml_cont(c_, ggml_permute(c_, score, 1, 0, 2, 3));
  ggml_tensor* weights = ggml_soft_max(c_, scores);
  ggml_tensor* comp = ggml_mul(c_, values, weights);
  comp = ggml_sum_rows(c_, comp);
  comp = ggml_cont(c_, ggml_permute(c_, comp, 1, 0, 2, 3));
  comp = Norm(comp, norm);
  const Rope r = CompressedRope(p_);
  comp = RopeExt(comp, comp_pos, r, r.n_ctx_orig);
  return ggml_rope_set_offset(comp, static_cast<int>(head - p_.rope_dims));
}

// llama_kv_cache::cpy_k: merge the heads, store the rows at the cells.
ggml_tensor* Builder::CpyK(ggml_tensor* cache, ggml_tensor* k_cur, ggml_tensor* idxs) {
  const std::int64_t n_embd_gqa = k_cur->ne[0] * k_cur->ne[1];
  k_cur = ggml_view_2d(c_, k_cur, n_embd_gqa, k_cur->ne[2], k_cur->nb[2], 0);
  return ggml_set_rows(c_, cache, k_cur, idxs);
}

// llama_kv_cache::get_k over the first n_kv cells (one KV head, one stream).
ggml_tensor* Builder::GetK(ggml_tensor* cache, std::int64_t n_kv) {
  const std::int64_t head = cache->ne[0];
  return ggml_view_4d(c_, cache, head, 1, n_kv, 1, ggml_row_size(cache->type, head),
                      ggml_row_size(cache->type, head),
                      ggml_row_size(cache->type, head * cache->ne[1]), 0);
}

// A speculative verify's attention (D-092): each query row alone, over
// views of its q and mask rows, as a one-row chunk at its position runs it
// (the MMA kernel's column tiles and stream-k split follow the query rows),
// the rows' outputs concatenated.
ggml_tensor* Builder::AttnMha(ggml_tensor* q, ggml_tensor* k, ggml_tensor* kq_mask,
                              ggml_tensor* sinks, std::int64_t n_kv_max) {
  const std::int64_t nt = q->ne[2];
  if (!o_.row_invariant || nt == 1) {
    return AttnMhaRow(q, k, kq_mask, sinks, n_kv_max);
  }
  ggml_tensor* out = nullptr;
  for (std::int64_t t = 0; t < nt; ++t) {
    const auto row = static_cast<std::size_t>(t);
    ggml_tensor* q_t =
        ggml_view_3d(c_, q, q->ne[0], q->ne[1], 1, q->nb[1], q->nb[2], row * q->nb[2]);
    ggml_tensor* mask_t = ggml_view_4d(c_, kq_mask, kq_mask->ne[0], 1, 1, 1, kq_mask->nb[1],
                                       kq_mask->nb[2], kq_mask->nb[3], row * kq_mask->nb[1]);
    ggml_tensor* one = AttnMhaRow(q_t, k, mask_t, sinks, n_kv_max);
    out = out != nullptr ? ggml_concat(c_, out, one, 1) : one;
  }
  return out;
}

// build_attn_mha with flash attention, one stream (llama-graph.cpp:2591-2650).
ggml_tensor* Builder::AttnMhaRow(ggml_tensor* q, ggml_tensor* k, ggml_tensor* kq_mask,
                                 ggml_tensor* sinks, std::int64_t n_kv_max, bool sparse_any) {
  ggml_tensor* v = k;
  q = ggml_view_4d(c_, q, q->ne[0], q->ne[1], q->ne[2], 1, q->nb[1], q->nb[2], q->nb[3], 0);
  q = ggml_permute(c_, q, 0, 2, 1, 3);
  k = ggml_permute(c_, k, 0, 2, 1, 3);
  v = ggml_permute(c_, v, 0, 2, 1, 3);
  const float scale = 1.0f / std::sqrt(static_cast<float>(p_.head_dim));
  ggml_tensor* cur = ggml_flash_attn_ext(c_, q, k, v, kq_mask, scale, 0.0f, 0.0f);
  ggml_flash_attn_ext_add_sinks(cur, sinks);
  ggml_flash_attn_ext_set_n_kv_max(cur, static_cast<std::int32_t>(n_kv_max));
  if (sparse_any) {
    SetFlashAttnSparseAny(cur);
  }
  ggml_prec_set_acc(cur, GGML_PREC_F32);
  return ggml_reshape_2d(c_, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);
}

// build_lid_top_k (deepseek4.cpp:591-677), the indexer fused.
ggml_tensor* Builder::LidTopK(const Dsv4LayerTensors& l, ggml_tensor* qr, ggml_tensor* cur,
                              int il) {
  const std::int64_t ih = p_.indexer_head_dim;
  const std::int64_t heads = p_.indexer_heads;
  const std::int64_t nt = cur->ne[1];
  ggml_tensor* q = Mm(l.idx_q_b, qr);
  q = ggml_reshape_3d(c_, q, ih, heads, nt);
  const Rope r = CompressedRope(p_);
  q = RopeExt(q, g_.positions, r, r.n_ctx_orig);
  q = ggml_rope_set_offset(q, static_cast<int>(ih - p_.rope_dims));
  q = Hadamard(q, g_.lid_rot);
  ggml_tensor* weights = MmShared(l.idx_proj, cur);
  weights = ggml_scale(c_, weights, 1.0f / sqrtf(static_cast<float>(ih * heads)));
  ggml_tensor* k = GetK(l.lid_k, s_.csa_n_kv);
  q = ggml_view_4d(c_, q, q->ne[0], q->ne[1], q->ne[2], 1, q->nb[1], q->nb[2], q->nb[3], 0);
  weights = ggml_view_4d(c_, weights, weights->ne[0], weights->ne[1], weights->ne[2], 1,
                         weights->nb[1], weights->nb[2], weights->nb[3], 0);
  ggml_tensor* score = ggml_lightning_indexer(c_, q, k, weights, g_.lid.mask);
  Name(score, "lid_score_masked", il);
  const std::int64_t top = std::min<std::int64_t>(score->ne[0], p_.indexer_top_k);
  ggml_tensor* top_k = ggml_cont(c_, ggml_top_k(c_, score, static_cast<int>(top)));
  Name(top_k, "lid_topk", il);
  return top_k;
}

// The fast plan's attention: each layer's window cells and, in a
// compressed layer, its compressed rows read in place as one K (the state
// lays them out together; no concatenation), under one mask built on the
// device (jitllm.dsv4.sparse_mask: the window's cells, then the indexer's
// selection for CSA or the visible rows for HCA), through the MMA kernel's
// sparse gather of the unmasked cells. So a row's work is its window's 128
// cells and its compressed layer's selected (CSA: the indexer's top 512)
// or visible (HCA: one per 128 positions) rows, however long the context
// (the window a ring, model/dsv4.h Dsv4Window::kRing; over the full window
// only the masks grow). The cells a row attends are llama.cpp's; only the
// order the kernel reduces them in differs.
ggml_tensor* Builder::AttentionSparse(std::uint32_t il_u, ggml_tensor* q, ggml_tensor* kv,
                                      ggml_tensor* qr, ggml_tensor* cur) {
  const int il = static_cast<int>(il_u);
  const Dsv4LayerTensors& l = g_.layers[il_u];
  const std::uint32_t ratio = p_.compress_ratios[il_u];
  const std::int64_t window = std::min<std::int64_t>(s_.raw_n_kv, p_.window);
  ggml_tensor* top_k = nullptr;
  if (ratio == model::kDsv4CsaRatio) {
    top_k = LidTopKSparse(l, qr, cur, il);
  }
  Expand(q);
  Expand(kv);
  Expand(CpyK(l.raw_k, kv, g_.raw_k_idxs));
  ggml_tensor* out = nullptr;
  if (ratio == model::kDsv4CsaRatio) {
    ggml_tensor* k = GetK(l.raw_k, s_.raw_cells + s_.csa_n_kv);
    ggml_tensor* kq_mask =
        Dsv4SparseMask(c_, g_.raw_mask, top_k, nullptr, s_.raw_cells, s_.csa_n_kv);
    Name(kq_mask, "kq_mask", il);
    out = AttnMhaRow(q, k, kq_mask, l.attn_sinks, window + top_k->ne[0], true);
    Name(out, "attn_csa_lid", il);
  } else if (ratio == model::kDsv4HcaRatio) {
    ggml_tensor* k = GetK(l.raw_k, s_.raw_cells + s_.hca_n_kv);
    ggml_tensor* kq_mask =
        Dsv4SparseMask(c_, g_.raw_mask, nullptr, g_.hca_visible, s_.raw_cells, s_.hca_n_kv);
    out = AttnMhaRow(q, k, kq_mask, l.attn_sinks, window + s_.hca_n_kv, true);
    Name(out, "attn_hca", il);
  } else {
    ggml_tensor* k = GetK(l.raw_k, s_.raw_n_kv);
    out = AttnMhaRow(q, k, g_.raw_mask, l.attn_sinks, window, true);
    Name(out, "attn_raw", il);
  }
  return out;
}

// The fast plan's lightning indexer: build_lid_top_k's query, rotation and
// weights, then jitllm.dsv4.lid_topk's scores and selection over the rows
// each row sees.
ggml_tensor* Builder::LidTopKSparse(const Dsv4LayerTensors& l, ggml_tensor* qr, ggml_tensor* cur,
                                    int il) {
  const std::int64_t ih = p_.indexer_head_dim;
  const std::int64_t heads = p_.indexer_heads;
  const std::int64_t nt = cur->ne[1];
  ggml_tensor* q = Mm(l.idx_q_b, qr);
  q = ggml_reshape_3d(c_, q, ih, heads, nt);
  const Rope r = CompressedRope(p_);
  q = RopeExt(q, g_.positions, r, r.n_ctx_orig);
  q = ggml_rope_set_offset(q, static_cast<int>(ih - p_.rope_dims));
  q = Hadamard(q, g_.lid_rot);
  ggml_tensor* weights = MmShared(l.idx_proj, cur);
  weights = ggml_scale(c_, weights, 1.0f / sqrtf(static_cast<float>(ih * heads)));
  ggml_tensor* k = ggml_view_2d(c_, l.lid_k, ih, s_.csa_n_kv, l.lid_k->nb[1], 0);
  const std::int64_t top = std::min<std::int64_t>(s_.csa_n_kv, p_.indexer_top_k);
  ggml_tensor* top_k = Dsv4LidTopK(c_, q, k, weights, g_.csa_visible, top);
  Name(top_k, "lid_topk", il);
  return top_k;
}

// build_top_k_mask (deepseek4.cpp:679-706).
ggml_tensor* Builder::TopKMask(ggml_tensor* kq_mask, ggml_tensor* top_k) {
  ggml_tensor* all = ggml_fill(c_, kq_mask, -INFINITY);
  all = ggml_view_4d(c_, all, 1, all->ne[0], all->ne[1], all->ne[3], all->nb[0], all->nb[1],
                     all->nb[2], 0);
  ggml_tensor* top_k_3d =
      ggml_view_4d(c_, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1],
                   top_k->nb[2], static_cast<std::size_t>(top_k->ne[3]) * top_k->nb[3], 0);
  ggml_tensor* zeros = ggml_fill(c_, g_.top_k_zeros, 0.0f);
  ggml_tensor* masked = ggml_set_rows(c_, all, zeros, top_k_3d);
  masked = ggml_view_4d(c_, masked, masked->ne[1], masked->ne[2], 1, masked->ne[3], masked->nb[2],
                        masked->nb[3], masked->nb[3], 0);
  return ggml_add(c_, masked, kq_mask);
}

// build_attention_impl (deepseek4.cpp:879-1221) for one layer.
ggml_tensor* Builder::Attention(std::uint32_t il_u, ggml_tensor* cur) {
  const int il = static_cast<int>(il_u);
  const Dsv4LayerTensors& l = g_.layers[il_u];
  const std::uint32_t ratio = p_.compress_ratios[il_u];
  const std::int64_t head = p_.head_dim;
  const std::int64_t nope = head - p_.rope_dims;
  const std::int64_t heads = p_.heads;
  const std::int64_t groups = p_.o_groups;
  const std::int64_t nt = cur->ne[1];
  const Rope rl = LayerRope(p_, ratio);

  ggml_tensor* qr = Mm(l.q_a, cur);
  qr = Norm(qr, l.q_a_norm);
  ggml_tensor* q = Mm(l.q_b, qr);
  q = ggml_reshape_3d(c_, q, head, heads, nt);
  const Dsv4QHeadParams q_params{.eps = p_.rms_eps,
                                 .original_context = rl.n_ctx_orig,
                                 .base = rl.base,
                                 .scale = rl.scale,
                                 .extension = rl.ext,
                                 .attention = rl.attn,
                                 .beta_fast = rl.beta_fast,
                                 .beta_slow = rl.beta_slow};
  if (o_.fused && p_.rope_dims == 64 && Dsv4QHeadFits(q, g_.positions, q_params)) {
    q = Dsv4QHead(c_, q, g_.positions, q_params,
                  o_.f16_q && nt >= 64 ? GGML_TYPE_F16 : GGML_TYPE_F32);
  } else {
    q = ggml_rms_norm(c_, q, p_.rms_eps);
    q = RopeExt(q, g_.positions, rl, rl.n_ctx_orig);
    q = ggml_rope_set_offset(q, static_cast<int>(nope));
  }
  Name(q, "q", il);

  ggml_tensor* kv = Mm(l.kv, cur);
  kv = Norm(kv, l.kv_norm);
  kv = ggml_reshape_3d(c_, kv, head, 1, nt);
  kv = RopeExt(kv, g_.positions, rl, rl.n_ctx_orig);
  kv = ggml_rope_set_offset(kv, static_cast<int>(nope));
  Name(kv, "kv", il);

  ggml_tensor* hca_state_kv = nullptr;
  ggml_tensor* hca_state_score = nullptr;
  if (ratio == model::kDsv4HcaRatio) {
    hca_state_kv = MmShared(l.comp_kv, cur);
    hca_state_score = MmShared(l.comp_gate, cur);
    ggml_tensor* ape_rows = ggml_get_rows(c_, l.comp_ape, g_.hca.state_pos);
    hca_state_score = ggml_add(c_, hca_state_score, ape_rows);
  }
  if (ratio == model::kDsv4CsaRatio) {
    ggml_tensor* csa_kv = MmShared(l.comp_kv, cur);
    ggml_tensor* csa_score = MmShared(l.comp_gate, cur);
    ggml_tensor* ape_rows = ggml_get_rows(c_, l.comp_ape, g_.csa.state_pos);
    csa_score = ggml_add(c_, csa_score, ape_rows);
    // The ring state, read in place (no rollback planes to restore from).
    ggml_tensor* base_kv = ggml_view_2d(c_, l.csa_state_kv, l.csa_state_kv->ne[0],
                                        s_.csa_state_rows, l.csa_state_kv->nb[1], 0);
    ggml_tensor* base_score = ggml_view_2d(c_, l.csa_state_score, l.csa_state_score->ne[0],
                                           s_.csa_state_rows, l.csa_state_score->nb[1], 0);
    ggml_tensor* comp = nullptr;
    if (o_.fused) {
      comp = CompressFused(base_kv, base_score, csa_kv, csa_score, g_.csa.read_idxs,
                           g_.csa.write_pos, l.comp_norm, model::kDsv4CsaRatio, head, true);
    } else {
      ggml_tensor* source_kv = ggml_concat(c_, base_kv, csa_kv, 1);
      ggml_tensor* source_score = ggml_concat(c_, base_score, csa_score, 1);
      comp = OverlapCompress(source_kv, source_score, g_.csa.read_idxs, g_.csa.write_pos,
                             l.comp_norm, model::kDsv4CsaRatio, head);
    }
    Name(comp, "csa_state_compress", il);
    Expand(CpyK(l.csa_k, comp, g_.csa.write_idxs));
    ggml_tensor* persist_kv = ggml_get_rows(c_, csa_kv, g_.csa.persist_src);
    ggml_tensor* persist_score = ggml_get_rows(c_, csa_score, g_.csa.persist_src);
    Expand(ggml_set_rows(c_, l.csa_state_kv, persist_kv, g_.csa.persist_dst));
    Expand(ggml_set_rows(c_, l.csa_state_score, persist_score, g_.csa.persist_dst));

    const std::int64_t ih = p_.indexer_head_dim;
    ggml_tensor* lid_kv = MmShared(l.idx_comp_kv, cur);
    ggml_tensor* lid_score = MmShared(l.idx_comp_gate, cur);
    ggml_tensor* lid_ape_rows = ggml_get_rows(c_, l.idx_comp_ape, g_.lid.state_pos);
    lid_score = ggml_add(c_, lid_score, lid_ape_rows);
    ggml_tensor* lid_base_kv = ggml_view_2d(c_, l.lid_state_kv, l.lid_state_kv->ne[0],
                                            s_.csa_state_rows, l.lid_state_kv->nb[1], 0);
    ggml_tensor* lid_base_score = ggml_view_2d(c_, l.lid_state_score, l.lid_state_score->ne[0],
                                               s_.csa_state_rows, l.lid_state_score->nb[1], 0);
    ggml_tensor* lid_comp = nullptr;
    if (o_.fused) {
      lid_comp = CompressFused(lid_base_kv, lid_base_score, lid_kv, lid_score, g_.lid.read_idxs,
                               g_.lid.write_pos, l.idx_comp_norm, model::kDsv4CsaRatio, ih, true);
    } else {
      ggml_tensor* lid_source_kv = ggml_concat(c_, lid_base_kv, lid_kv, 1);
      ggml_tensor* lid_source_score = ggml_concat(c_, lid_base_score, lid_score, 1);
      lid_comp = OverlapCompress(lid_source_kv, lid_source_score, g_.lid.read_idxs,
                                 g_.lid.write_pos, l.idx_comp_norm, model::kDsv4CsaRatio, ih);
    }
    lid_comp = Hadamard(lid_comp, g_.lid_rot);
    Name(lid_comp, "lid_state_compress_rot", il);
    Expand(CpyK(l.lid_k, lid_comp, g_.lid.write_idxs));
    ggml_tensor* lid_persist_kv = ggml_get_rows(c_, lid_kv, g_.lid.persist_src);
    ggml_tensor* lid_persist_score = ggml_get_rows(c_, lid_score, g_.lid.persist_src);
    Expand(ggml_set_rows(c_, l.lid_state_kv, lid_persist_kv, g_.lid.persist_dst));
    Expand(ggml_set_rows(c_, l.lid_state_score, lid_persist_score, g_.lid.persist_dst));
  }
  if (ratio == model::kDsv4HcaRatio) {
    ggml_tensor* base_kv = ggml_view_2d(c_, l.hca_state_kv, l.hca_state_kv->ne[0],
                                        s_.hca_state_rows, l.hca_state_kv->nb[1], 0);
    ggml_tensor* base_score = ggml_view_2d(c_, l.hca_state_score, l.hca_state_score->ne[0],
                                           s_.hca_state_rows, l.hca_state_score->nb[1], 0);
    ggml_tensor* comp = nullptr;
    if (o_.fused) {
      comp = CompressFused(base_kv, base_score, hca_state_kv, hca_state_score, g_.hca.read_idxs,
                           g_.hca.write_pos, l.comp_norm, model::kDsv4HcaRatio, head, false);
    } else {
      ggml_tensor* source_kv = ggml_concat(c_, base_kv, hca_state_kv, 1);
      ggml_tensor* source_score = ggml_concat(c_, base_score, hca_state_score, 1);
      comp = HcaCompress(source_kv, source_score, g_.hca.read_idxs, g_.hca.write_pos, l.comp_norm,
                         head);
    }
    Name(comp, "hca_state_compress", il);
    Expand(CpyK(l.hca_k, comp, g_.hca.write_idxs));
    ggml_tensor* persist_kv = ggml_get_rows(c_, hca_state_kv, g_.hca.persist_src);
    ggml_tensor* persist_score = ggml_get_rows(c_, hca_state_score, g_.hca.persist_src);
    Expand(ggml_set_rows(c_, l.hca_state_kv, persist_kv, g_.hca.persist_dst));
    Expand(ggml_set_rows(c_, l.hca_state_score, persist_score, g_.hca.persist_dst));
  }

  ggml_tensor* out = nullptr;
  if (sparse_) {
    out = AttentionSparse(il_u, q, kv, qr, cur);
  } else if (ratio == model::kDsv4CsaRatio) {
    // build_csa_lid_attention.
    ggml_tensor* top_k = LidTopK(l, qr, cur, il);
    Expand(q);
    Expand(kv);
    Expand(CpyK(l.raw_k, kv, g_.raw_k_idxs));
    ggml_tensor* raw_k = GetK(l.raw_k, s_.raw_n_kv);
    ggml_tensor* csa_k = GetK(l.csa_k, s_.csa_n_kv);
    ggml_tensor* k_all = ggml_concat(c_, raw_k, csa_k, 2);
    ggml_tensor* csa_mask = TopKMask(g_.csa.mask, top_k);
    ggml_tensor* kq_mask = ggml_concat(c_, g_.raw_mask, csa_mask, 0);
    Name(kq_mask, "kq_mask", il);
    const std::int64_t n_kv_max =
        std::min<std::int64_t>(g_.raw_mask->ne[0], p_.window) + top_k->ne[0];
    out = AttnMha(q, k_all, kq_mask, l.attn_sinks, n_kv_max);
    Name(out, "attn_csa_lid", il);
  } else if (ratio == model::kDsv4HcaRatio) {
    Expand(q);
    Expand(kv);
    Expand(CpyK(l.raw_k, kv, g_.raw_k_idxs));
    ggml_tensor* raw_k = GetK(l.raw_k, s_.raw_n_kv);
    ggml_tensor* hca_k = GetK(l.hca_k, s_.hca_n_kv);
    ggml_tensor* k_all = ggml_concat(c_, raw_k, hca_k, 2);
    ggml_tensor* kq_mask = ggml_concat(c_, g_.raw_mask, g_.hca.mask, 0);
    out = AttnMha(q, k_all, kq_mask, l.attn_sinks, 0);
    Name(out, "attn_hca", il);
  } else {
    Expand(q);
    Expand(kv);
    Expand(CpyK(l.raw_k, kv, g_.raw_k_idxs));
    ggml_tensor* k = GetK(l.raw_k, s_.raw_n_kv);
    out = AttnMha(q, k, g_.raw_mask, l.attn_sinks, 0);
    Name(out, "attn_raw", il);
  }

  out = ggml_reshape_3d(c_, out, head, heads, nt);
  const Dsv4OutAParams outa_params = {.original_context = rl.n_ctx_orig,
                                      .base = rl.base,
                                      .scale = rl.scale,
                                      .extension = rl.ext,
                                      .attention = rl.attn,
                                      .beta_fast = rl.beta_fast,
                                      .beta_slow = rl.beta_slow};
  if (o_.fused && o_.outa_prefill && p_.rope_dims == 64 && nope == 448 &&
      Dsv4OutAFits(l.out_a, out, g_.positions, outa_params)) {
    // The direct dependency retains unrotated heads through the disjoint
    // canonical output write. No inverse-RoPE tensor or output layout copy.
    // Its output's rows are the chunk's rounded up to 16 (Dsv4OutARows):
    // out_b reads the chunk's.
    ggml_tensor* low = Dsv4OutA(c_, l.out_a, out, g_.positions, outa_params);
    if (low->ne[1] != nt) {
      low = ggml_view_2d(c_, low, low->ne[0], nt, low->nb[1], 0);
    }
    out = Mm(l.out_b, low);
    Name(out, "attn_out", il);
    return out;
  }
  out = ggml_rope_ext_back(c_, out, g_.positions, nullptr, static_cast<int>(p_.rope_dims),
                           kRopeMode, rl.n_ctx_orig, rl.base, rl.scale, rl.ext, rl.attn,
                           rl.beta_fast, rl.beta_slow);
  out = ggml_rope_set_offset(out, static_cast<int>(nope));
  out = ggml_reshape_3d(c_, out, (heads / groups) * head, groups, nt);
  ggml_tensor* oa = nullptr;
  if (o_.fused && nt <= kVecQTokens && VecQType(l.out_a->type) && ggml_is_contiguous(out) &&
      out->ne[0] % 32 == 0) {
    // The fast plan: each group's matrix over its rows, straight into the
    // [o_lora, groups, tokens] layout the permuted copy would give.
    oa = VecQ(c_, l.out_a, Q8Of(out), nullptr, nt, true);
    oa = ggml_reshape_2d(c_, oa, std::int64_t{p_.o_lora} * groups, nt);
  } else {
    out = ggml_permute(c_, out, 0, 2, 1, 3);
    oa = ggml_mul_mat(c_, l.out_a, out);
    oa = ggml_permute(c_, oa, 0, 2, 1, 3);
    oa = ggml_cont_2d(c_, oa, std::int64_t{p_.o_lora} * groups, nt);
  }
  out = Mm(l.out_b, oa);
  Name(out, "attn_out", il);
  return out;
}

// A wave's attention for one layer (BuildDsv4WaveGraph): Attention's and
// AttentionSparse's operations, the row-local ones over every slot's rows
// and the rest per slot over its rows' views, its own inputs and state.
ggml_tensor* Builder::AttentionWave(std::uint32_t il_u, ggml_tensor* cur) {
  const int il = static_cast<int>(il_u);
  const Dsv4LayerTensors& l = g_.layers[il_u];
  const std::uint32_t ratio = p_.compress_ratios[il_u];
  const std::int64_t head = p_.head_dim;
  const std::int64_t nope = head - p_.rope_dims;
  const std::int64_t heads = p_.heads;
  const std::int64_t groups = p_.o_groups;
  const std::int64_t nt = cur->ne[1];
  const Rope rl = LayerRope(p_, ratio);

  ggml_tensor* qr = Mm(l.q_a, cur);
  qr = Norm(qr, l.q_a_norm);
  ggml_tensor* q = Mm(l.q_b, qr);
  q = ggml_reshape_3d(c_, q, head, heads, nt);
  const Dsv4QHeadParams q_params{.eps = p_.rms_eps,
                                 .original_context = rl.n_ctx_orig,
                                 .base = rl.base,
                                 .scale = rl.scale,
                                 .extension = rl.ext,
                                 .attention = rl.attn,
                                 .beta_fast = rl.beta_fast,
                                 .beta_slow = rl.beta_slow};
  if (o_.fused && p_.rope_dims == 64 && Dsv4QHeadFits(q, g_.positions, q_params)) {
    q = Dsv4QHead(c_, q, g_.positions, q_params,
                  o_.f16_q && nt >= 64 ? GGML_TYPE_F16 : GGML_TYPE_F32);
  } else {
    q = ggml_rms_norm(c_, q, p_.rms_eps);
    q = RopeExt(q, g_.positions, rl, rl.n_ctx_orig);
    q = ggml_rope_set_offset(q, static_cast<int>(nope));
  }
  Name(q, "q", il);
  ggml_tensor* kv = Mm(l.kv, cur);
  kv = Norm(kv, l.kv_norm);
  kv = ggml_reshape_3d(c_, kv, head, 1, nt);
  kv = RopeExt(kv, g_.positions, rl, rl.n_ctx_orig);
  kv = ggml_rope_set_offset(kv, static_cast<int>(nope));
  Name(kv, "kv", il);

  // The compressors' and the indexer's projections, every slot's rows.
  const std::int64_t ih = p_.indexer_head_dim;
  ggml_tensor* comp_kv = nullptr;
  ggml_tensor* comp_score = nullptr;
  ggml_tensor* lid_kv = nullptr;
  ggml_tensor* lid_score = nullptr;
  ggml_tensor* lid_q = nullptr;
  ggml_tensor* lid_w = nullptr;
  if (ratio == model::kDsv4HcaRatio || ratio == model::kDsv4CsaRatio) {
    const Dsv4CompInputs& own = ratio == model::kDsv4HcaRatio ? g_.hca : g_.csa;
    comp_kv = MmShared(l.comp_kv, cur);
    comp_score = MmShared(l.comp_gate, cur);
    comp_score = ggml_add(c_, comp_score, ggml_get_rows(c_, l.comp_ape, own.state_pos));
  }
  if (ratio == model::kDsv4CsaRatio) {
    lid_kv = MmShared(l.idx_comp_kv, cur);
    lid_score = MmShared(l.idx_comp_gate, cur);
    lid_score = ggml_add(c_, lid_score, ggml_get_rows(c_, l.idx_comp_ape, g_.lid.state_pos));
    lid_q = Mm(l.idx_q_b, qr);
    lid_q = ggml_reshape_3d(c_, lid_q, ih, p_.indexer_heads, nt);
    const Rope r = CompressedRope(p_);
    lid_q = RopeExt(lid_q, g_.positions, r, r.n_ctx_orig);
    lid_q = ggml_rope_set_offset(lid_q, static_cast<int>(ih - p_.rope_dims));
    lid_q = Hadamard(lid_q, g_.lid_rot);
    lid_w = MmShared(l.idx_proj, cur);
    lid_w = ggml_scale(c_, lid_w, 1.0f / sqrtf(static_cast<float>(ih * p_.indexer_heads)));
  }
  Expand(q);
  Expand(kv);

  // Each slot: its compressors' blocks and ring rows, its window cells, its
  // indexer's selection and its attention, over its own state; with lanes,
  // each slot's on a lane of its own, the layer a region.
  ggml_tensor* out = nullptr;
  const std::uint32_t region = lanes_ != nullptr ? ++regions_ : 0;
  // Slots past kMaxLanes share lanes in turn (a lane's steps run in order).
  std::uint32_t slot = 0;
  for (const Segment& seg : segs_) {
    const auto lane = static_cast<std::uint8_t>((slot++ % kMaxLanes) + 1);
    ggml_tensor* const mark = lanes_ != nullptr ? Last() : nullptr;
    const Dsv4ChunkShape& s = *seg.s;
    Dsv4Graph& sg = *seg.g;
    const Dsv4LayerTensors& sl = sg.layers[il_u];
    const auto ring = [&](ggml_tensor* t, std::int64_t rows) {
      return ggml_view_2d(c_, t, t->ne[0], rows, t->nb[1], 0);
    };
    const auto persist = [&](ggml_tensor* rows, const Dsv4CompInputs& in, ggml_tensor* kv_state,
                             ggml_tensor* score_state, ggml_tensor* score) {
      Expand(ggml_set_rows(c_, kv_state, ggml_get_rows(c_, Rows2(rows, seg), in.persist_src),
                           in.persist_dst));
      Expand(ggml_set_rows(c_, score_state, ggml_get_rows(c_, Rows2(score, seg), in.persist_src),
                           in.persist_dst));
    };
    ggml_tensor* top_k = nullptr;
    if (ratio == model::kDsv4CsaRatio) {
      ggml_tensor* comp = CompressFused(
          ring(sl.csa_state_kv, s.csa_state_rows), ring(sl.csa_state_score, s.csa_state_rows),
          Rows2(comp_kv, seg), Rows2(comp_score, seg), sg.csa.read_idxs, sg.csa.write_pos,
          l.comp_norm, model::kDsv4CsaRatio, head, true);
      Expand(CpyK(sl.csa_k, comp, sg.csa.write_idxs));
      persist(comp_kv, sg.csa, sl.csa_state_kv, sl.csa_state_score, comp_score);
      ggml_tensor* lid_comp = CompressFused(
          ring(sl.lid_state_kv, s.csa_state_rows), ring(sl.lid_state_score, s.csa_state_rows),
          Rows2(lid_kv, seg), Rows2(lid_score, seg), sg.lid.read_idxs, sg.lid.write_pos,
          l.idx_comp_norm, model::kDsv4CsaRatio, ih, true);
      lid_comp = Hadamard(lid_comp, g_.lid_rot);
      Expand(CpyK(sl.lid_k, lid_comp, sg.lid.write_idxs));
      persist(lid_kv, sg.lid, sl.lid_state_kv, sl.lid_state_score, lid_score);
      ggml_tensor* k = ggml_view_2d(c_, sl.lid_k, ih, s.csa_n_kv, sl.lid_k->nb[1], 0);
      const std::int64_t top = std::min<std::int64_t>(s.csa_n_kv, p_.indexer_top_k);
      top_k = Dsv4LidTopK(c_, Rows3(lid_q, seg), k, Rows2(lid_w, seg), sg.csa_visible, top);
    } else if (ratio == model::kDsv4HcaRatio) {
      ggml_tensor* comp = CompressFused(
          ring(sl.hca_state_kv, s.hca_state_rows), ring(sl.hca_state_score, s.hca_state_rows),
          Rows2(comp_kv, seg), Rows2(comp_score, seg), sg.hca.read_idxs, sg.hca.write_pos,
          l.comp_norm, model::kDsv4HcaRatio, head, false);
      Expand(CpyK(sl.hca_k, comp, sg.hca.write_idxs));
      persist(comp_kv, sg.hca, sl.hca_state_kv, sl.hca_state_score, comp_score);
    }
    Expand(CpyK(sl.raw_k, Rows3(kv, seg), sg.raw_k_idxs));
    const std::int64_t window = std::min<std::int64_t>(s.raw_n_kv, p_.window);
    ggml_tensor* q_s = Rows3(q, seg);
    ggml_tensor* one = nullptr;
    if (ratio == model::kDsv4CsaRatio) {
      ggml_tensor* k = GetK(sl.raw_k, s.raw_cells + s.csa_n_kv);
      ggml_tensor* kq_mask =
          Dsv4SparseMask(c_, sg.raw_mask, top_k, nullptr, s.raw_cells, s.csa_n_kv);
      one = AttnMhaRow(q_s, k, kq_mask, l.attn_sinks, window + top_k->ne[0], true);
    } else if (ratio == model::kDsv4HcaRatio) {
      ggml_tensor* k = GetK(sl.raw_k, s.raw_cells + s.hca_n_kv);
      ggml_tensor* kq_mask =
          Dsv4SparseMask(c_, sg.raw_mask, nullptr, sg.hca_visible, s.raw_cells, s.hca_n_kv);
      one = AttnMhaRow(q_s, k, kq_mask, l.attn_sinks, window + s.hca_n_kv, true);
    } else if (sparse_) {
      ggml_tensor* k = GetK(sl.raw_k, s.raw_n_kv);
      one = AttnMhaRow(q_s, k, sg.raw_mask, l.attn_sinks, window, true);
    } else {
      // A joined draft's block (BuildDsparkWaveGraph): Attention's window
      // attention over the slot's whole ring, as its own block runs it.
      ggml_tensor* k = GetK(sl.raw_k, s.raw_n_kv);
      one = AttnMhaRow(q_s, k, sg.raw_mask, l.attn_sinks, 0);
    }
    if (lanes_ != nullptr) {
      Tag(mark, lane, region);
    }
    ggml_tensor* const joined = lanes_ != nullptr ? Last() : nullptr;
    out = out != nullptr ? ggml_concat(c_, out, one, 1) : one;
    if (lanes_ != nullptr) {
      Tag(joined, 0, region);  // the stream's, in the region
    }
  }

  out = ggml_reshape_3d(c_, out, head, heads, nt);
  out = ggml_rope_ext_back(c_, out, g_.positions, nullptr, static_cast<int>(p_.rope_dims),
                           kRopeMode, rl.n_ctx_orig, rl.base, rl.scale, rl.ext, rl.attn,
                           rl.beta_fast, rl.beta_slow);
  out = ggml_rope_set_offset(out, static_cast<int>(nope));
  out = ggml_reshape_3d(c_, out, (heads / groups) * head, groups, nt);
  ggml_tensor* oa = nullptr;
  if (o_.fused && nt <= VecQRows() && VecQType(l.out_a->type) && ggml_is_contiguous(out) &&
      out->ne[0] % 32 == 0) {
    oa = VecQ(c_, l.out_a, Q8Of(out), nullptr, nt, true);
    oa = ggml_reshape_2d(c_, oa, std::int64_t{p_.o_lora} * groups, nt);
  } else {
    // A weight jitllm.vecq has no kernel for: GGML's grouped product, whose
    // launch follows the column count, a slot at a time over its own rows,
    // as each slot's own chunk runs it (Attention), joined.
    for (const Segment& seg : segs_) {
      ggml_tensor* one = ggml_mul_mat(c_, l.out_a, ggml_permute(c_, Rows3(out, seg), 0, 2, 1, 3));
      one = ggml_permute(c_, one, 0, 2, 1, 3);
      one = ggml_cont_2d(c_, one, std::int64_t{p_.o_lora} * groups, seg.s->rows);
      oa = oa != nullptr ? ggml_concat(c_, oa, one, 1) : one;
    }
  }
  out = Mm(l.out_b, oa);
  Name(out, "attn_out", il);
  return out;
}

// The drafter's projections run per slot, over its injected rows alone, as
// Inject runs them for its chunk: they are GGML's quantized products, whose
// launch (and so each column's arithmetic) follows the column count, so a
// slot's ring rows equal its own chunk's bit for bit. The drafter's few
// projection weights are read once a slot.
void Builder::InjectWave(const DsparkInjectTensors& t, const Dsv4Injection& inject) {
  const model::Dsv4Profile& dp = inject.profile->blocks;
  const Rope rope = LayerRope(dp, 0);
  const std::int64_t head = dp.head_dim;
  for (const Segment& seg : segs_) {
    const std::int64_t r = seg.inject_rows;
    const auto skip = static_cast<std::size_t>(seg.first + seg.s->rows - r);
    ggml_tensor* features = ggml_view_2d(c_, g_.features, g_.features->ne[0], r, g_.features->nb[1],
                                         skip * g_.features->nb[1]);
    ggml_tensor* positions = ggml_view_1d(c_, g_.positions, r, skip * g_.positions->nb[0]);
    ggml_tensor* inp_g = ggml_mul_mat(c_, t.fc, features);
    inp_g = ggml_mul(c_, ggml_rms_norm(c_, inp_g, dp.rms_eps), t.enc_norm);
    for (std::uint32_t il = 0; il < dp.layers; ++il) {
      ggml_tensor* kv = ggml_mul_mat(c_, t.kv[il], inp_g);
      kv = ggml_mul(c_, ggml_rms_norm(c_, kv, dp.rms_eps), t.kv_norm[il]);
      kv = ggml_reshape_3d(c_, kv, head, 1, r);
      kv = ggml_rope_ext(c_, kv, positions, nullptr, static_cast<int>(dp.rope_dims), kRopeMode,
                         rope.n_ctx_orig, rope.base, rope.scale, rope.ext, rope.attn,
                         rope.beta_fast, rope.beta_slow);
      kv = ggml_rope_set_offset(kv, static_cast<int>(head - dp.rope_dims));
      // InjectLeaves gave every slot its own cells and ring.
      if (const std::optional<DsparkInjectTensors>& own = seg.g->inject; own.has_value()) {
        Expand(CpyK(own->ring[il], kv, own->cells));
      }
    }
  }
}

// build_moe_ffn and build_ffn for one layer (llama-graph.cpp:1748-2380).
ggml_tensor* Builder::Moe(std::uint32_t il_u, ggml_tensor* cur) {
  const int il = static_cast<int>(il_u);
  const Dsv4LayerTensors& l = g_.layers[il_u];
  const std::int64_t n_embd = cur->ne[0];
  const std::int64_t nt = cur->ne[1];
  const std::int64_t n_expert = p_.experts;
  const std::int64_t used = p_.experts_used;

  ggml_tensor* selected = nullptr;
  ggml_tensor* bias = l.router_bias;
  if (l.tid2eid != nullptr) {
    selected = ggml_get_rows(c_, l.tid2eid, g_.tokens);
    bias = nullptr;
  }
  ggml_tensor* logits = ggml_mul_mat(c_, l.router, cur);
  ggml_prec_set_acc(logits, GGML_PREC_F32);
  Name(logits, "ffn_moe_logits", il);
  ggml_tensor* probs = ggml_sqrt(c_, ggml_softplus(c_, logits));
  Name(probs, "ffn_moe_probs", il);
  ggml_tensor* selection = probs;
  if (bias != nullptr) {
    selection = ggml_add(c_, probs, bias);
    Name(selection, "ffn_moe_selection", il);
  }
  if (selected == nullptr) {
    selected = ggml_argsort_top_k(c_, selection, static_cast<int>(used));
  }
  Name(selected, "ffn_moe_topk", il);
  probs = ggml_reshape_3d(c_, probs, 1, n_expert, nt);
  ggml_tensor* weights = ggml_get_rows(c_, probs, selected);
  if (p_.expert_weights_norm) {
    weights = ggml_reshape_2d(c_, weights, used, nt);
    ggml_tensor* sum = ggml_sum_rows(c_, weights);
    sum = ggml_clamp(c_, sum, 6.103515625e-5f, INFINITY);
    weights = ggml_div(c_, weights, sum);
    weights = ggml_reshape_3d(c_, weights, 1, used, nt);
  }
  if (p_.expert_weights_scale != 0.0f && p_.expert_weights_scale != 1.0f) {
    weights = ggml_scale(c_, weights, p_.expert_weights_scale);
  }
  Name(weights, "ffn_moe_weights_scaled", il);
  Expand(weights);
  ggml_tensor* x = ggml_reshape_3d(c_, cur, n_embd, 1, nt);
  ggml_tensor* up = ggml_mul_mat_id(c_, l.up_exps, x, selected);
  ggml_tensor* gate = ggml_mul_mat_id(c_, l.gate_exps, x, selected);
  ggml_tensor* act = nullptr;
  if (p_.swiglu_limit > 1e-6f) {
    act = ggml_swiglu_clamp(c_, gate, up, p_.swiglu_limit);
  } else {
    act = ggml_swiglu_split(c_, gate, up);
  }
  ggml_tensor* experts = ggml_mul_mat_id(c_, l.down_exps, act, selected);
  ggml_tensor* moe_out = nullptr;
  if (o_.fused && nt > kVecQTokens && Dsv4WeightedReduceFits(experts, weights)) {
    // The ordinary graph dependencies make down, weights and output live
    // together. Exact, small-row and outside-contract paths stay ordinary.
    moe_out = Dsv4OrderedReduce(c_, experts, weights);
  } else {
    experts = ggml_mul(c_, experts, weights);
    Expand(experts);
    std::vector<ggml_tensor*> views;
    const auto n_used = static_cast<std::size_t>(used);
    for (std::size_t i = 0; i < n_used; ++i) {
      views.push_back(ggml_view_2d(c_, experts, n_embd, nt, experts->nb[2], i * experts->nb[1]));
      Expand(views.back());
    }
    moe_out = views[0];
    for (std::size_t i = 1; i < n_used; ++i) {
      moe_out = ggml_add(c_, moe_out, views[i]);
      Expand(moe_out);
    }
    if (used == 1) {
      moe_out = ggml_cont(c_, moe_out);
    }
  }
  Name(moe_out, "ffn_moe_out", il);

  // build_ffn: up, then gate (parallel), the clamped SwiGLU, down.
  ggml_tensor* sh_up = ggml_mul_mat(c_, l.up_shexp, cur);
  ggml_tensor* sh_gate = ggml_mul_mat(c_, l.gate_shexp, cur);
  ggml_tensor* sh = p_.swiglu_limit_shared > 1e-6f
                        ? ggml_swiglu_clamp(c_, sh_gate, sh_up, p_.swiglu_limit_shared)
                        : ggml_swiglu_split(c_, sh_gate, sh_up);
  sh = ggml_mul_mat(c_, l.down_shexp, sh);
  Name(sh, "ffn_shexp", il);
  ggml_tensor* out = ggml_add(c_, moe_out, sh);
  Name(out, "ffn_out", il);
  return out;
}

bool Builder::Fused(std::uint32_t il) const {
  if (!o_.fused || s_.rows > VecQRows() || !FusedProfile(p_)) {
    return false;
  }
  // The mixing weights' type is not a condition: PreNorm takes GGML's
  // product for a type jitllm.dsv4.hc_mix does not read.
  const Dsv4LayerTensors& l = g_.layers[il];
  return FusedTypes(l.up_exps->type, l.gate_exps->type, l.down_exps->type, l.up_shexp->type,
                    l.gate_shexp->type, l.down_shexp->type) &&
         ggml_are_same_shape(l.up_exps, l.gate_exps) &&
         ggml_are_same_stride(l.up_exps, l.gate_exps) &&
         ggml_are_same_shape(l.up_shexp, l.gate_shexp);
}

ggml_tensor* Builder::HcPreFused(ggml_tensor* x, ggml_tensor* fn, ggml_tensor* scale,
                                 ggml_tensor* base, ggml_tensor* norm, ggml_tensor** post,
                                 ggml_tensor** comb) {
  const std::int64_t hc = p_.hc;
  const std::int64_t width = p_.width;
  const std::int64_t nt = x->ne[2];
  ggml_tensor* partials = Dsv4HcMix(c_, x, fn);
  ggml_tensor* pre = Dsv4HcPre(c_, partials, x, scale, base, norm, p_.rms_eps, p_.hc_eps,
                               static_cast<std::int32_t>(p_.sinkhorn_iterations));
  const std::size_t f = sizeof(float);
  const auto tails = static_cast<std::size_t>(width * nt) * f;
  *post = ggml_view_2d(c_, pre, hc, nt, kDsv4HcTail * f, tails);
  *comb = ggml_view_3d(c_, pre, hc, hc, nt, static_cast<std::size_t>(hc) * f, kDsv4HcTail * f,
                       tails + (static_cast<std::size_t>(hc) * f));
  return ggml_view_2d(c_, pre, width, nt, static_cast<std::size_t>(width) * f, 0);
}

// Moe's routing, experts and shared expert in the fast plan's operations.
ggml_tensor* Builder::MoeFused(std::uint32_t il_u, ggml_tensor* cur) {
  const int il = static_cast<int>(il_u);
  const Dsv4LayerTensors& l = g_.layers[il_u];
  const std::int64_t nt = cur->ne[1];
  const std::int64_t used = p_.experts_used;
  ggml_tensor* logits = FloatMm(l.router, cur, true);
  Name(logits, "ffn_moe_logits", il);
  const float scale = p_.expert_weights_scale != 0.0f && p_.expert_weights_scale != 1.0f
                          ? p_.expert_weights_scale
                          : 1.0f;
  constexpr float kClamp = 6.103515625e-5f;
  ggml_tensor* route = l.tid2eid != nullptr
                           ? Dsv4Route(c_, logits, nullptr, l.tid2eid, g_.tokens, used,
                                       p_.expert_weights_norm, kClamp, scale)
                           : Dsv4Route(c_, logits, l.router_bias, nullptr, nullptr, used,
                                       p_.expert_weights_norm, kClamp, scale);
  Name(route, "ffn_moe_route", il);
  ggml_tensor* ids = ggml_view_2d(c_, route, used, nt, route->nb[1], 0);
  ggml_tensor* q = Q8Of(cur);
  const auto glu = [](float limit) {
    return limit > 1e-6f ? VecQGlu::kSwigluClamp : VecQGlu::kSwiglu;
  };
  ggml_tensor* act =
      VecQ(c_, l.up_exps, q, ids, nt, false, l.gate_exps, glu(p_.swiglu_limit), p_.swiglu_limit);
  Name(act, "ffn_moe_act", il);
  ggml_tensor* down = VecQ(c_, l.down_exps, QuantizeQ8(c_, act), ids, nt, true);
  Name(down, "ffn_moe_down", il);
  ggml_tensor* sh = VecQ(c_, l.up_shexp, q, nullptr, nt, false, l.gate_shexp,
                         glu(p_.swiglu_limit_shared), p_.swiglu_limit_shared);
  sh = VecQ(c_, l.down_shexp, QuantizeQ8(c_, sh), nullptr, nt, false);
  Name(sh, "ffn_shexp", il);
  ggml_tensor* out = Dsv4Combine(c_, down, route, sh);
  Name(out, "ffn_out", il);
  return out;
}

ggml_tensor* Builder::HcMean(ggml_tensor* x) {
  const std::int64_t hc = x->ne[1];
  ggml_tensor* acc = ggml_view_2d(c_, x, x->ne[0], x->ne[2], x->nb[2], 0);
  for (std::int64_t s = 1; s < hc; ++s) {
    acc = ggml_add(
        c_, acc,
        ggml_view_2d(c_, x, x->ne[0], x->ne[2], x->nb[2], static_cast<std::size_t>(s) * x->nb[1]));
  }
  return ggml_scale(c_, acc, 1.0f / static_cast<float>(hc));
}

// graph_dsv4's embd batch (dflash.cpp:844-883) over the chunk's features:
// fc and the encoder's norm, then each drafter block's wkv, kv norm and
// uncompressed RoPE, stored at the injected rows' ring cells.
void Builder::Inject(const DsparkInjectTensors& t, const Dsv4Injection& inject) {
  const model::Dsv4Profile& dp = inject.profile->blocks;
  const std::int64_t n = s_.rows;
  const std::int64_t r = inject.rows;
  ggml_tensor* features = g_.features;
  ggml_tensor* positions = g_.positions;
  if (r < n) {
    const auto skip = static_cast<std::size_t>(n - r);
    features =
        ggml_view_2d(c_, features, features->ne[0], r, features->nb[1], skip * features->nb[1]);
    positions = ggml_view_1d(c_, positions, r, skip * positions->nb[0]);
  }
  ggml_tensor* inp_g = ggml_mul_mat(c_, t.fc, features);
  inp_g = ggml_mul(c_, ggml_rms_norm(c_, inp_g, dp.rms_eps), t.enc_norm);
  Name(inp_g, "inp_g_embeddings", -1);
  const Rope rope = LayerRope(dp, 0);
  const std::int64_t head = dp.head_dim;
  for (std::uint32_t il = 0; il < dp.layers; ++il) {
    ggml_tensor* kv = ggml_mul_mat(c_, t.kv[il], inp_g);
    kv = ggml_mul(c_, ggml_rms_norm(c_, kv, dp.rms_eps), t.kv_norm[il]);
    kv = ggml_reshape_3d(c_, kv, head, 1, r);
    kv = ggml_rope_ext(c_, kv, positions, nullptr, static_cast<int>(dp.rope_dims), kRopeMode,
                       rope.n_ctx_orig, rope.base, rope.scale, rope.ext, rope.attn, rope.beta_fast,
                       rope.beta_slow);
    kv = ggml_rope_set_offset(kv, static_cast<int>(head - dp.rope_dims));
    Name(kv, "kv_injected", static_cast<int>(il));
    Expand(CpyK(t.ring[il], kv, t.cells));
  }
}

void Builder::Build() {
  const std::int64_t nt = s_.rows;
  const std::int64_t hc = p_.hc;
  ggml_tensor* inp = ggml_reshape_3d(c_, g_.embd, p_.width, 1, nt);
  ggml_tensor* inpl = ggml_repeat_4d(c_, inp, p_.width, hc, nt, 1);
  Name(inpl, "hc_init", -1);
  // The streams entering each feature layer (DSpark's target layers).
  std::vector<ggml_tensor*> features(o_.features.size(), nullptr);
  const auto capture = [&](std::uint32_t layer, ggml_tensor* streams) {
    for (std::size_t k = 0; k < o_.features.size(); ++k) {
      if (o_.features[k] == layer) {
        features[k] = streams;
      }
    }
  };
  for (std::uint32_t il_u = 0; il_u < p_.layers; ++il_u) {
    const int il = static_cast<int>(il_u);
    const Dsv4LayerTensors& l = g_.layers[il_u];
    capture(il_u, inpl);
    ggml_tensor* residual = inpl;
    ggml_tensor* post = nullptr;
    ggml_tensor* comb = nullptr;
    if (Fused(il_u)) {
      ggml_tensor* cur = PreNorm(il_u, inpl, l.hc_attn_fn, l.hc_attn_scale, l.hc_attn_base,
                                 l.attn_norm, &post, &comb);
      Name(cur, "attn_norm", il);
      cur = segs_.empty() ? Attention(il_u, cur) : AttentionWave(il_u, cur);
      inpl = ggml_dsv4_hc_post(c_, cur, residual, post, comb);
      Name(inpl, "hc_attn_post", il);
      residual = inpl;
      cur =
          PreNorm(il_u, inpl, l.hc_ffn_fn, l.hc_ffn_scale, l.hc_ffn_base, l.ffn_norm, &post, &comb);
      Expand(residual);
      Name(cur, "ffn_norm", il);
      cur = MoeFused(il_u, cur);
      inpl = ggml_dsv4_hc_post(c_, cur, residual, post, comb);
      Name(inpl, "l_last", il);
      continue;
    }
    ggml_tensor* cur = HcPre(inpl, l.hc_attn_fn, l.hc_attn_scale, l.hc_attn_base, &post, &comb, il);
    Name(cur, "hc_attn_pre", il);
    cur = Norm(cur, l.attn_norm);
    Name(cur, "attn_norm", il);
    cur = Attention(il_u, cur);
    inpl = ggml_dsv4_hc_post(c_, cur, residual, post, comb);
    Name(inpl, "hc_attn_post", il);
    residual = inpl;
    cur = HcPre(inpl, l.hc_ffn_fn, l.hc_ffn_scale, l.hc_ffn_base, &post, &comb, il);
    Name(cur, "hc_ffn_pre", il);
    Expand(residual);
    Expand(post);
    Expand(comb);
    cur = Norm(cur, l.ffn_norm);
    Name(cur, "ffn_norm", il);
    cur = Moe(il_u, cur);
    inpl = ggml_dsv4_hc_post(c_, cur, residual, post, comb);
    Name(inpl, "l_last", il);
  }
  capture(p_.layers, inpl);
  // Capture above retains every feature row. Only the requested trailing
  // rows enter the final mix, norm and vocabulary head (a wave's: all).
  if (segs_.empty()) {
    ggml_tensor* flat = ggml_reshape_2d(c_, inpl, p_.hc_width(), nt);
    ggml_tensor* flat_out = ggml_get_rows(c_, flat, g_.out_ids);
    inpl = ggml_reshape_3d(c_, flat_out, p_.width, hc, g_.out_ids->ne[0]);
  }
  ggml_tensor* cur = HcHead(inpl);
  Name(cur, "hc_head", -1);
  cur = Norm(cur, g_.output_norm);
  Name(cur, "result_norm", -1);
  g_.logits = Mm(g_.output, cur);
  Name(g_.logits, "result_output", -1);
  Expand(g_.logits);
  // The drafter's part after the target's own, so the target's nodes keep
  // their order: the features (the streams' means, llama.cpp's layer_inp
  // extraction), and the injection.
  for (ggml_tensor* streams : features) {
    ggml_tensor* mean = HcMean(streams);
    g_.features = g_.features != nullptr ? ggml_concat(c_, g_.features, mean, 0) : mean;
  }
  if (g_.features != nullptr) {
    Name(g_.features, "layer_inp", -1);
    Expand(g_.features);
    if (g_.inject && o_.inject) {
      if (segs_.empty()) {
        Inject(*g_.inject, *o_.inject);
      } else {
        InjectWave(*g_.inject, *o_.inject);
      }
    }
  }
}

std::expected<void, KernelFailure> Builder::DraftInputs(DsparkGraph& d,
                                                        const model::DsparkBinding& b) {
  const std::int64_t n = s_.rows;
  g_.embd = ggml_new_tensor_2d(c_, GGML_TYPE_F32, p_.width, n);
  d.tokens = ggml_new_tensor_1d(c_, GGML_TYPE_I32, n);
  g_.positions = ggml_new_tensor_1d(c_, GGML_TYPE_I32, n);
  g_.raw_k_idxs = ggml_new_tensor_1d(c_, GGML_TYPE_I64, n);
  g_.raw_mask = ggml_new_tensor_4d(c_, GGML_TYPE_F16, s_.raw_n_kv, n, 1, 1);
  auto w1 = Leaf(c_, b.markov_w1, "markov_w1");
  auto w2 = Leaf(c_, b.markov_w2, "markov_w2");
  if (!w1 || !w2) {
    return std::unexpected(!w1 ? w1.error() : w2.error());
  }
  d.markov_w1 = *w1;
  d.markov_w2 = *w2;
  return {};
}

std::expected<void, KernelFailure> Builder::DraftWaveInputs(DsparkWaveGraph& d,
                                                            const model::DsparkBinding& b) {
  const std::int64_t n = s_.rows;
  g_.embd = ggml_new_tensor_2d(c_, GGML_TYPE_F32, p_.width, n);
  d.tokens = ggml_new_tensor_1d(c_, GGML_TYPE_I32, n);
  g_.positions = ggml_new_tensor_1d(c_, GGML_TYPE_I32, n);
  for (Segment& seg : segs_) {
    Dsv4Graph& sg = *seg.g;
    sg.raw_k_idxs = ggml_new_tensor_1d(c_, GGML_TYPE_I64, seg.s->rows);
    sg.raw_mask = ggml_new_tensor_4d(c_, GGML_TYPE_F16, seg.s->raw_n_kv, seg.s->rows, 1, 1);
  }
  auto w1 = Leaf(c_, b.markov_w1, "markov_w1");
  auto w2 = Leaf(c_, b.markov_w2, "markov_w2");
  if (!w1 || !w2) {
    return std::unexpected(!w1 ? w1.error() : w2.error());
  }
  d.markov_w1 = *w1;
  d.markov_w2 = *w2;
  return {};
}

// graph_dsv4's token batch (dflash.cpp:886-1001): the blocks, the
// hyper-connection head, the norm and the target's head.
ggml_tensor* Builder::DraftTrunk() {
  const std::int64_t nt = s_.rows;
  const std::int64_t hc = p_.hc;
  ggml_tensor* inp = ggml_reshape_3d(c_, g_.embd, p_.width, 1, nt);
  ggml_tensor* inpl = ggml_repeat_4d(c_, inp, p_.width, hc, nt, 1);
  Name(inpl, "hc_init", -1);
  for (std::uint32_t il_u = 0; il_u < p_.layers; ++il_u) {
    const int il = static_cast<int>(il_u);
    const Dsv4LayerTensors& l = g_.layers[il_u];
    ggml_tensor* residual = inpl;
    ggml_tensor* post = nullptr;
    ggml_tensor* comb = nullptr;
    ggml_tensor* cur = PreNorm(il_u, inpl, l.hc_attn_fn, l.hc_attn_scale, l.hc_attn_base,
                               l.attn_norm, &post, &comb);
    cur = segs_.empty() ? Attention(il_u, cur) : AttentionWave(il_u, cur);
    inpl = ggml_dsv4_hc_post(c_, cur, residual, post, comb);
    residual = inpl;
    if (Fused(il_u)) {
      cur =
          PreNorm(il_u, inpl, l.hc_ffn_fn, l.hc_ffn_scale, l.hc_ffn_base, l.ffn_norm, &post, &comb);
      Expand(residual);
      cur = MoeFused(il_u, cur);
    } else {
      cur = HcPre(inpl, l.hc_ffn_fn, l.hc_ffn_scale, l.hc_ffn_base, &post, &comb, il);
      Expand(residual);
      Expand(post);
      Expand(comb);
      cur = Norm(cur, l.ffn_norm);
      cur = Moe(il_u, cur);
    }
    inpl = ggml_dsv4_hc_post(c_, cur, residual, post, comb);
    Name(inpl, "l_out", il);
  }
  ggml_tensor* cur = HcHead(inpl);
  cur = Norm(cur, g_.output_norm);
  g_.logits = Mm(g_.output, cur);  // [vocab, rows]
  Expand(g_.logits);
  return g_.logits;
}

// build_dspark_markov_head (dflash.cpp:293-404): each block row's logits
// biased by the row before it, the anchor before the first.
ggml_tensor* Builder::Markov(ggml_tensor* logits, ggml_tensor* tokens, ggml_tensor* w1,
                             ggml_tensor* w2, std::int64_t first, std::int64_t nt) {
  const std::int64_t vocab = logits->ne[0];
  ggml_tensor* prev = ggml_view_1d(c_, tokens, 1, static_cast<std::size_t>(first) * tokens->nb[0]);
  ggml_tensor* cat = nullptr;
  for (std::int64_t i = 0; i < nt; ++i) {
    ggml_tensor* w1_prev = ggml_get_rows(c_, w1, prev);  // [rank, 1]
    ggml_tensor* bias = ggml_mul_mat(c_, w2, w1_prev);   // [vocab, 1]
    ggml_tensor* base = ggml_view_2d(c_, logits, vocab, 1, logits->nb[1],
                                     static_cast<std::size_t>(first + i) * logits->nb[1]);
    ggml_tensor* col = ggml_add(c_, base, bias);
    cat = cat != nullptr ? ggml_concat(c_, cat, col, 1) : col;
    if (i + 1 < nt) {
      prev = Argmax(c_, col);
    }
  }
  return cat;
}

// One block, anchor first (sample_from_anchor), without the confidence
// head.
void Builder::BuildDraft(DsparkGraph& d) {
  ggml_tensor* logits = DraftTrunk();
  d.logits = Markov(logits, d.tokens, d.markov_w1, d.markov_w2, 0, s_.rows);
  Name(d.logits, "dspark_logits", -1);
  d.drafts = Argmax(c_, d.logits);
  Expand(d.drafts);
}

// Every slot's block: the trunk over the joined rows, then each slot's
// Markov head over its rows, on a lane of its own with lanes (the chains
// are independent; one region).
void Builder::BuildDraftWave(DsparkWaveGraph& d) {
  ggml_tensor* logits = DraftTrunk();
  const std::uint32_t region = lanes_ != nullptr ? ++regions_ : 0;
  std::uint32_t slot = 0;
  for (const Segment& seg : segs_) {
    const auto lane = static_cast<std::uint8_t>((slot++ % kMaxLanes) + 1);
    ggml_tensor* const mark = lanes_ != nullptr ? Last() : nullptr;
    ggml_tensor* biased =
        Markov(logits, d.tokens, d.markov_w1, d.markov_w2, seg.first, seg.s->rows);
    ggml_tensor* drafts = Argmax(c_, biased);
    Expand(drafts);
    d.drafts.push_back(drafts);
    if (lanes_ != nullptr) {
      Tag(mark, lane, region);
    }
  }
}

}  // namespace

Dsv4ChunkShape Dsv4ShapeOf(const model::Dsv4StateLayout& state, const model::Dsv4ChunkInputs& chunk,
                           std::int64_t outputs) {
  return {.rows = chunk.rows,
          .outputs = outputs,
          .raw_n_kv = chunk.raw_n_kv,
          .raw_cells = state.raw_cells,
          .csa_n_kv = chunk.csa.n_kv,
          .hca_n_kv = chunk.hca.n_kv,
          .csa_cells = state.csa_cells,
          .hca_cells = state.hca_cells,
          .csa_blocks = chunk.csa.blocks(),
          .hca_blocks = chunk.hca.blocks(),
          .csa_persist = static_cast<std::int64_t>(chunk.csa.persist_src.size()),
          .hca_persist = static_cast<std::int64_t>(chunk.hca.persist_src.size()),
          .csa_state_rows = state.csa_state_rows,
          .hca_state_rows = state.hca_state_rows};
}

std::vector<ggml_tensor*> Dsv4Graph::inputs() const {
  std::vector<ggml_tensor*> all = {embd, tokens, positions, raw_k_idxs};
  if (!device_raw_mask) all.push_back(raw_mask);
  all.push_back(out_ids);
  for (const Dsv4CompInputs* in : {&csa, &hca, &lid}) {
    all.insert(all.end(), {in->state_pos, in->persist_src, in->persist_dst, in->read_idxs,
                           in->write_idxs, in->write_pos});
    if (in->mask != nullptr) {
      all.push_back(in->mask);
    }
  }
  all.push_back(lid_rot);
  if (top_k_zeros != nullptr) {
    all.push_back(top_k_zeros);
  } else {
    all.push_back(csa_visible);
    all.push_back(hca_visible);
  }
  if (inject) {
    all.push_back(inject->cells);
  }
  return all;
}

std::vector<ggml_tensor*> DsparkGraph::inputs() const {
  return {core.embd, tokens, core.positions, core.raw_k_idxs, core.raw_mask};
}

std::vector<ggml_tensor*> DsparkWaveGraph::inputs() const {
  std::vector<ggml_tensor*> all = {joined.embd, tokens, joined.positions};
  for (const Dsv4Graph& s : slots) {
    all.insert(all.end(), {s.raw_k_idxs, s.raw_mask});
  }
  return all;
}

ggml_tensor* Dsv4Graph::Named(std::string_view name) const {
  for (const auto& [n, t] : named) {
    if (n == name) {
      return t;
    }
  }
  return nullptr;
}

std::size_t Dsv4GraphTensors(const model::Dsv4Profile& profile) {
  // Leaves: about 60 per layer; nodes: at most about 330 per layer (a CSA
  // layer with its indexer), 40 for the head; a row-invariant verify adds
  // about 10 per row and layer for its attention (up to 8 rows), a
  // drafter's features and injection about 60. Rounded up generously.
  return 512 + (std::size_t{profile.layers} * (512 + 96));
}

std::vector<ggml_tensor*> Dsv4WaveGraph::inputs() const {
  std::vector<ggml_tensor*> all = {joined.embd,          joined.tokens,        joined.positions,
                                   joined.csa.state_pos, joined.hca.state_pos, joined.lid.state_pos,
                                   joined.lid_rot};
  for (const Dsv4Graph& s : slots) {
    all.push_back(s.raw_k_idxs);
    if (!s.device_raw_mask) all.push_back(s.raw_mask);
    for (const Dsv4CompInputs* in : {&s.csa, &s.hca, &s.lid}) {
      all.insert(all.end(),
                 {in->persist_src, in->persist_dst, in->read_idxs, in->write_idxs, in->write_pos});
    }
    all.insert(all.end(), {s.csa_visible, s.hca_visible});
    if (const std::optional<DsparkInjectTensors>& injected = s.inject; injected.has_value()) {
      all.push_back(injected->cells);
    }
  }
  return all;
}

std::size_t Dsv4WaveGraphTensors(const model::Dsv4Profile& profile, std::size_t slots) {
  // A chunk's, and each slot's own inputs, state and per-slot operations
  // (about 30 leaves and 60 nodes a layer, its share of the drafter's ring).
  return Dsv4GraphTensors(profile) + (slots * (128 + (std::size_t{profile.layers} * 128)));
}

std::size_t DsparkGraphTensors(const model::DsparkProfile& profile, std::int64_t rows) {
  // The blocks as a target chunk's layers, and the Markov head's six a slot.
  return 256 + (std::size_t{profile.blocks.layers} * 512) + (static_cast<std::size_t>(rows) * 8);
}

std::size_t DsparkWaveGraphTensors(const model::DsparkProfile& profile, std::int64_t rows,
                                   std::size_t slots) {
  // The joined blocks, and each slot's inputs, ring, attention (with a float
  // product a slot past the column-invariant count) and Markov head.
  return DsparkGraphTensors(profile, rows * static_cast<std::int64_t>(slots)) +
         (slots * (128 + (std::size_t{profile.blocks.layers} * 128) +
                   (static_cast<std::size_t>(rows) * 16)));
}

std::expected<ggml_type, KernelFailure> GgmlTypeOf(std::string_view name) {
  // GGML names its types in lower case ("q8_0"); the artifact's index in
  // upper case ("Q8_0"), as GGUF tools print them.
  const auto same = [](std::string_view a, std::string_view b) {
    return a.size() == b.size() && std::ranges::equal(a, b, [](char x, char y) {
             const auto lower = [](char c) {
               return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c;
             };
             return lower(x) == lower(y);
           });
  };
  for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
    const auto type = static_cast<ggml_type>(t);
    const char* type_name = ggml_type_name(type);
    if (ggml_blck_size(type) > 0 && ggml_type_size(type) > 0 && type_name != nullptr &&
        same(name, type_name)) {
      return type;
    }
  }
  return Rejected(std::format("{} is not a GGML type", name));
}

std::vector<float> HadamardMatrix(std::int64_t n) {
  std::vector<float> data(static_cast<std::size_t>(n * n), 0.0f);
  if (n <= 0 || (n & (n - 1)) != 0) {
    return data;
  }
  data[0] = 1.0f / sqrtf(static_cast<float>(n));
  for (std::int64_t s = 1; s < n; s *= 2) {
    for (std::int64_t i = 0; i < s; ++i) {
      for (std::int64_t j = 0; j < s; ++j) {
        const float v = data[static_cast<std::size_t>((i * n) + j)];
        data[static_cast<std::size_t>(((i + s) * n) + j)] = v;
        data[static_cast<std::size_t>((i * n) + j + s)] = v;
        data[static_cast<std::size_t>(((i + s) * n) + j + s)] = -v;
      }
    }
  }
  return data;
}

std::expected<Dsv4Graph, KernelFailure> BuildDsv4Graph(TensorArena& arena,
                                                       const model::Dsv4Profile& profile,
                                                       const model::Dsv4Binding& binding,
                                                       const Dsv4ChunkShape& shape,
                                                       const Dsv4GraphOptions& options) {
  if (options.raw_mask_context > INT32_MAX ||
      (options.raw_mask_context != 0 &&
       (profile.window > INT32_MAX || shape.raw_cells > INT32_MAX || shape.rows > INT32_MAX - 31)))
    return Rejected("raw causal/ring mask parameters exceed int32");
  const Dsv4ChunkShape& s = shape;
  if (s.rows <= 0 || s.outputs < 0 || s.outputs > s.rows || s.raw_cells <= 0 || s.raw_n_kv < 256 ||
      s.raw_n_kv > s.raw_cells || s.raw_n_kv % 256 != 0 || s.csa_n_kv < 256 ||
      s.csa_n_kv > s.csa_cells || s.csa_n_kv % 256 != 0 || s.hca_n_kv < 256 ||
      s.hca_n_kv > s.hca_cells || s.hca_n_kv % 256 != 0 || s.csa_blocks <= 0 || s.hca_blocks <= 0 ||
      s.csa_persist <= 0 || s.hca_persist <= 0 ||
      s.csa_state_rows != 2 * std::int64_t{model::kDsv4CsaRatio} ||
      s.hca_state_rows != std::int64_t{model::kDsv4HcaRatio}) {
    return Rejected("not a DeepSeek V4 chunk shape the state holds");
  }
  if (binding.layers.size() != profile.layers || profile.compress_ratios.size() != profile.layers ||
      profile.hc != 4 || profile.heads % profile.o_groups != 0) {
    return Rejected("the binding is not the profile's");
  }
  if (options.inject && options.features.empty()) {
    return Rejected("a DSpark injection reads the chunk's features");
  }
  if (options.fused && options.row_invariant) {
    return Rejected("the fast plan's sparse attention has no row-invariant form (D-092)");
  }
  for (const std::uint32_t layer : options.features) {
    if (layer > profile.layers) {
      return Rejected("a feature layer past the stream leaving the last layer");
    }
  }
  if (auto room = arena.Reserve(Dsv4GraphTensors(profile)); !room) {
    return std::unexpected(room.error());
  }
  Dsv4Graph g;
  Builder builder(arena.context(), profile, binding, shape, g, options, options.fused);
  builder.Inputs();
  if (auto weights = builder.Weights(); !weights) {
    return std::unexpected(weights.error());
  }
  if (options.inject) {
    if (auto leaves = builder.InjectLeaves(*options.inject); !leaves) {
      return std::unexpected(leaves.error());
    }
  }
  builder.Build();
  auto ordered = GraphOrder(builder.expanded(), arena);
  if (!ordered) return std::unexpected(ordered.error());
  g.nodes = std::move(*ordered);
  return g;
}

std::expected<void, KernelFailure> Dsv4WaveSupport(const model::Dsv4Profile& profile,
                                                   const model::Dsv4Binding& binding) {
  if (!FusedProfile(profile) || binding.layers.size() != profile.layers) {
    return Rejected("the profile's widths do not take the fast plan's fused form");
  }
  for (std::uint32_t il = 0; il < profile.layers; ++il) {
    const model::Dsv4Layer& w = binding.layers[il];
    std::array<ggml_type, 6> types{};
    const std::array<const model::Dsv4Tensor*, 6> tensors = {
        &w.up_exps, &w.gate_exps, &w.down_exps, &w.up_shexp, &w.gate_shexp, &w.down_shexp};
    for (std::size_t i = 0; i < tensors.size(); ++i) {
      auto type = GgmlTypeOf(tensors[i]->type);
      if (!type) {
        return std::unexpected(type.error());
      }
      types[i] = *type;
    }
    if (!FusedTypes(types[0], types[1], types[2], types[3], types[4], types[5]) ||
        w.up_exps.ne != w.gate_exps.ne || w.up_shexp.ne != w.gate_shexp.ne) {
      return Rejected(std::format(
          "layer {}: the routed experts' up/gate/down ({}/{}/{}) and the shared expert's "
          "({}/{}/{}) are not jitllm.vecq types with gate and up alike in type and shape",
          il, w.up_exps.type, w.gate_exps.type, w.down_exps.type, w.up_shexp.type,
          w.gate_shexp.type, w.down_shexp.type));
    }
  }
  return {};
}

std::expected<Dsv4WaveGraph, KernelFailure> BuildDsv4WaveGraph(TensorArena& arena,
                                                               const model::Dsv4Profile& profile,
                                                               const model::Dsv4Binding& binding,
                                                               const Dsv4WaveShape& shape,
                                                               const Dsv4GraphOptions& options) {
  if (options.raw_mask_context > INT32_MAX || profile.window > INT32_MAX ||
      std::ranges::any_of(shape.slots, [](const auto& slot) { return slot.raw_cells > INT32_MAX; }))
    return Rejected("raw causal/ring mask parameters exceed int32");
  const std::size_t count = shape.slots.size();
  if (count == 0 || count > kDsv4WaveSlots ||
      (options.inject ? shape.inject_rows.size() != count
                      : std::ranges::any_of(shape.inject_rows, [](auto r) { return r != 0; }))) {
    return Rejected(std::format("a DeepSeek V4 wave takes one to {} slots, each injected or none",
                                kDsv4WaveSlots));
  }
  if (!options.fused || options.row_invariant || options.outa_prefill) {
    return Rejected("a DeepSeek V4 wave runs the fast plan's fused form alone");
  }
  const Dsv4ChunkShape& lead = shape.slots.front();
  std::int64_t rows = 0;
  for (const Dsv4ChunkShape& s : shape.slots) {
    if (s.rows <= 0 || s.outputs != 0 || s.raw_cells <= 0 || s.raw_n_kv < 256 ||
        s.raw_n_kv > s.raw_cells || s.raw_n_kv % 256 != 0 || s.csa_n_kv < 256 ||
        s.csa_n_kv > s.csa_cells || s.csa_n_kv % 256 != 0 || s.hca_n_kv < 256 ||
        s.hca_n_kv > s.hca_cells || s.hca_n_kv % 256 != 0 || s.csa_blocks <= 0 ||
        s.hca_blocks <= 0 || s.csa_persist <= 0 || s.hca_persist <= 0 ||
        s.csa_state_rows != 2 * std::int64_t{model::kDsv4CsaRatio} ||
        s.hca_state_rows != std::int64_t{model::kDsv4HcaRatio} || s.raw_cells != lead.raw_cells ||
        s.csa_cells != lead.csa_cells || s.hca_cells != lead.hca_cells) {
      return Rejected("not a DeepSeek V4 wave slot shape the state holds");
    }
    rows += s.rows;
  }
  if (rows > kDsv4WaveRows) {
    return Rejected("a DeepSeek V4 wave's rows exceed its column-invariant products");
  }
  if (binding.layers.size() != profile.layers || profile.compress_ratios.size() != profile.layers ||
      profile.hc != 4 || profile.heads % profile.o_groups != 0) {
    return Rejected("the binding is not the profile's");
  }
  if (options.inject && options.features.empty()) {
    return Rejected("a DSpark injection reads the chunk's features");
  }
  for (const std::uint32_t layer : options.features) {
    if (layer > profile.layers) {
      return Rejected("a feature layer past the stream leaving the last layer");
    }
  }
  if (auto room = arena.Reserve(Dsv4WaveGraphTensors(profile, count)); !room) {
    return std::unexpected(room.error());
  }
  Dsv4WaveGraph wave;
  wave.slots.resize(count);
  std::vector<Segment> segments(count);
  std::int64_t first = 0;
  for (std::size_t i = 0; i < count; ++i) {
    segments[i] = {.g = &wave.slots[i],
                   .s = &shape.slots[i],
                   .first = first,
                   .inject_rows = options.inject ? shape.inject_rows[i] : 0};
    wave.first.push_back(first);
    first += shape.slots[i].rows;
  }
  Dsv4ChunkShape joined = lead;
  joined.rows = rows;
  Builder builder(arena.context(), profile, binding, joined, wave.joined, options, true, segments);
  builder.SetLanes(&wave.lanes);
  builder.WaveInputs();
  if (auto weights = builder.Weights(); !weights) {
    return std::unexpected(weights.error());
  }
  if (!builder.AllFused()) {
    return Rejected("a DeepSeek V4 wave needs every layer in the fast plan's fused form");
  }
  if (options.inject) {
    if (auto leaves = builder.InjectLeaves(*options.inject); !leaves) {
      return std::unexpected(leaves.error());
    }
  }
  builder.Build();
  auto ordered = GraphOrder(builder.expanded(), arena);
  if (!ordered) return std::unexpected(ordered.error());
  wave.joined.nodes = std::move(*ordered);
  // A wave of one-row steps: each vector product takes a one-row step's
  // launch, so every slot's row equals its step alone bit for bit (the
  // multi-token launch's reduction differs). Verify waves keep the
  // multi-token launch their verifies take alone.
  if (std::ranges::all_of(shape.slots, [](const Dsv4ChunkShape& s) { return s.rows == 1; })) {
    for (ggml_tensor* node : wave.joined.nodes) {
      if (JitllmOpOf(node) == JitllmOp::kVecQ) {
        SetVecQOneToken(node);
      }
    }
  }
  return wave;
}

std::expected<DsparkGraph, KernelFailure> BuildDsparkGraph(TensorArena& arena,
                                                           const model::DsparkProfile& profile,
                                                           const model::DsparkBinding& binding,
                                                           std::int64_t rows, std::int64_t ring,
                                                           const Dsv4GraphOptions& options) {
  const model::Dsv4Profile& p = profile.blocks;
  if (rows <= 0 || std::cmp_greater(rows, profile.block_size) ||
      std::cmp_not_equal(ring, profile.ring) || ring % 256 != 0 ||
      binding.blocks.layers.size() != p.layers || p.hc != 4 || p.heads % p.o_groups != 0 ||
      !options.features.empty() || options.inject || options.row_invariant ||
      options.raw_mask_context != 0) {
    return Rejected("not a DSpark draft block this drafter runs");
  }
  if (auto room = arena.Reserve(DsparkGraphTensors(profile, rows)); !room) {
    return std::unexpected(room.error());
  }
  // The blocks' attention reads the whole ring, window only.
  const Dsv4ChunkShape shape{.rows = rows, .raw_n_kv = ring, .raw_cells = ring};
  DsparkGraph d;
  Builder builder(arena.context(), p, binding.blocks, shape, d.core, options, false);
  if (auto inputs = builder.DraftInputs(d, binding); !inputs) {
    return std::unexpected(inputs.error());
  }
  if (auto weights = builder.Weights(); !weights) {
    return std::unexpected(weights.error());
  }
  builder.BuildDraft(d);
  auto ordered = GraphOrder(builder.expanded(), arena);
  if (!ordered) return std::unexpected(ordered.error());
  d.core.nodes = std::move(*ordered);
  return d;
}

std::expected<DsparkWaveGraph, KernelFailure> BuildDsparkWaveGraph(
    TensorArena& arena, const model::DsparkProfile& profile, const model::DsparkBinding& binding,
    std::int64_t rows, std::size_t slots, std::int64_t ring, const Dsv4GraphOptions& options) {
  const model::Dsv4Profile& p = profile.blocks;
  if (rows <= 0 || std::cmp_greater(rows, profile.block_size) ||
      std::cmp_not_equal(ring, profile.ring) || ring % 256 != 0 ||
      binding.blocks.layers.size() != p.layers || p.hc != 4 || p.heads % p.o_groups != 0 ||
      !options.features.empty() || options.inject || options.row_invariant ||
      options.raw_mask_context != 0) {
    return Rejected("not a DSpark draft block this drafter runs");
  }
  if (slots < 2 || slots > kDsv4WaveSlots || !options.fused ||
      rows * static_cast<std::int64_t>(slots) > kDsv4WaveRows) {
    return Rejected(
        "a joined draft takes two blocks or more of the fast plan, within a wave's rows");
  }
  if (auto room = arena.Reserve(DsparkWaveGraphTensors(profile, rows, slots)); !room) {
    return std::unexpected(room.error());
  }
  // Each slot's blocks read its whole ring, window only, as its own block.
  const Dsv4ChunkShape shape{.rows = rows, .raw_n_kv = ring, .raw_cells = ring};
  DsparkWaveGraph d;
  d.slots.resize(slots);
  std::vector<Segment> segments(slots);
  for (std::size_t i = 0; i < slots; ++i) {
    const std::int64_t first = rows * static_cast<std::int64_t>(i);
    segments[i] = {.g = &d.slots[i], .s = &shape, .first = first, .inject_rows = 0};
    d.first.push_back(first);
  }
  const Dsv4ChunkShape joined{
      .rows = rows * static_cast<std::int64_t>(slots), .raw_n_kv = ring, .raw_cells = ring};
  Builder builder(arena.context(), p, binding.blocks, joined, d.joined, options, false, segments);
  builder.SetLanes(&d.lanes);
  if (auto inputs = builder.DraftWaveInputs(d, binding); !inputs) {
    return std::unexpected(inputs.error());
  }
  if (auto weights = builder.Weights(); !weights) {
    return std::unexpected(weights.error());
  }
  if (!builder.AllFused()) {
    return Rejected("a joined draft needs every block in the fast plan's fused form");
  }
  builder.BuildDraftWave(d);
  auto ordered = GraphOrder(builder.expanded(), arena);
  if (!ordered) return std::unexpected(ordered.error());
  d.joined.nodes = std::move(*ordered);
  return d;
}

}  // namespace jitllm::kernels::ggml
