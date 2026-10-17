// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// attention.mqa, version 1: DeepSeek V4's attention (llama.cpp's
// deepseek4.cpp build_attention_impl, build_csa_lid_attention, build_*_
// compressed_kv_from_state and build_lid_top_k; the retired kernels/ggml/dsv4_graph.cc's
// reference form), inside hyper-connections (its input norm is its mix's).
// From the stream x:
//
//   - the query through a low rank: q_a · x, RMS-normed with its weight,
//     q_b · that ([head_dim, heads] a row); each head RMS-normed
//     (unweighted) and its last rope_dims values rotated (NORMAL pairs,
//     from value head_dim - rope_dims);
//   - one key that is also the value (multi-query, K = V): kv · x,
//     RMS-normed with its weight, its tail rotated the same way, stored F16
//     at the row's cell of the window cache (`cache`: "ring", a window
//     ring read whole; "full", a cell per position read through the step's
//     read cells; "injected", a block drafter's ring of `cells` cells its
//     target's steps also write, below);
//   - with a compression `ratio` (CSA or HCA): a compressor. Each row's
//     projections (comp_kv · x, and comp_gate · x plus its position's row of
//     the learned position embedding, `ape`, by p mod ratio) join the
//     compressor's ring of recent positions' (F32, ratio positions, twice
//     that with `overlap`); each block of `ratio` positions the rows
//     complete is pooled (ir::kPool's softmax: per channel, the softmax of
//     its scores weighting its values; with overlap a block pools the
//     previous block's positions' first halves beside its own positions'
//     second halves, a position before 0 reading a zero value under a -inf
//     score), RMS-normed with its weight, its tail rotated at the block's
//     first position, and stored F16 at the block's cell of the compressed
//     cache; a chunk completing fewer blocks than its plan has
//     (components::CompressorBlocks) pools a dummy block into the cache's
//     last cell, which no row sees; the rows' projections are kept in the
//     ring (its last min(rows, ring) rows);
//   - with the `indexer` (CSA's lightning indexer, which needs overlap):
//     the same compressor of its own weights over index_dim values, the
//     pooled rows multiplied by the normalized Hadamard matrix
//     (InputKind::kHadamard) before they are stored; its queries idx_q ·
//     (q_a · x normed) as indexer_heads heads, rotated and multiplied by
//     the Hadamard matrix, its head weights idx_proj · x scaled by 1 /
//     sqrt(index_dim · indexer_heads); each compressed row's score
//     Σ_h w_h · max(q_h · k, 0) (ir::kIndexScore) under the compressed
//     rows' causal mask, and the min(read, top_k) best (ir::kTopK);
//   - attention over the window cells (the window mask; a block drafter's:
//     its own block's rows too, ir::MaskKind::kAhead) and, with a ratio,
//     the compressed rows each row sees (with the indexer, those it
//     selects: a -inf fill with zeros stored at the selected cells, plus
//     the causal mask), at scale 1 / sqrt(head_dim) with each head's
//     learned sink in the softmax and F32 sums, K and V the same cells;
//   - each head's output rotated back, the heads in `o_groups` groups,
//     each group's values through its own matrix of out_a (to o_rank), the
//     groups' results joined and projected by out_b.
//
// Rotation: base `rope_base`; with a `rope_factor` above 1, YaRN (its
// original context `yarn_context` and betas); a mapping gives a
// compressed layer its compressed base and YaRN, a window layer neither.
//
// A block drafter's attention (`cache` "injected", window only; DSpark's)
// has an injection (ComponentDef::inject): in its target's steps it writes
// its ring at each segment's last rows: kv · (its injected input),
// RMS-normed, rotated at the rows' positions (the drafter's own blocks
// write the ring as above).
//
// The fragment emits what the retired dsv4_graph.cc's reference form recorded, in
// GGML's graph order: the compressors' stores and ring rows, the query,
// the key and its store, then attention (its key views, the indexer's
// selection within the mask) and the output; the multi-segment form runs
// the row-local products over every owner's rows and the stores,
// pooling, selection and attention per owner. The pooling is ir::kPool
// (whose primitive decomposition is the reference form's permutes,
// softmax, product and sum). The fast plan's fusions (llmp.dsv4.qhead,
// compress, lid_topk, sparse_mask and the sparse gather, output-A) are
// the planner's patterns (G4b). It splits no way: one KV head, and its
// compressors' and indexer's state, whole on every device.
//
// Also here: components::CompressorBlocks and PlanCompressor (step.h), the
// compressor plan's port of llama.cpp's dsv4_build_comp_plan, which the
// step's compressor inputs use; and defs::YarnRotation and RotateNormal
// (defs.h), the references' rotation, ported from GGML's.

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/check.h"
#include "components/component.h"
#include "components/defs.h"
#include "components/step.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "model/state_contract.h"

namespace llmp::components::defs {
namespace {

using format::Encoding;

constexpr std::array<std::string_view, 3> kCaches{"ring", "full", "injected"};
constexpr std::array<std::string_view, 2> kKvEncodings{"f16", "q8_0"};

constexpr ParamSpec Optional(std::string_view name, ParamType type, double least, double most,
                             double fallback) {
  return {.name = name,
          .type = type,
          .required = false,
          .least = least,
          .most = most,
          .fallback = fallback};
}

constexpr std::array kParams{
    ParamSpec{.name = "eps", .type = ParamType::kReal, .least = 0.0, .most = 1.0},
    ParamSpec{.name = "heads", .type = ParamType::kInt, .least = 1, .most = 1024},
    ParamSpec{.name = "head_dim", .type = ParamType::kInt, .least = 2, .most = 4096},
    ParamSpec{.name = "rope_dims", .type = ParamType::kInt, .least = 2, .most = 4096},
    // The query's and the output's low ranks, and the output's groups.
    ParamSpec{.name = "q_rank", .type = ParamType::kInt, .least = 1, .most = 1 << 16},
    ParamSpec{.name = "o_rank", .type = ParamType::kInt, .least = 1, .most = 1 << 16},
    ParamSpec{.name = "o_groups", .type = ParamType::kInt, .least = 1, .most = 1024},
    // Positions a row sees of the window cache (its own included).
    ParamSpec{.name = "window", .type = ParamType::kInt, .least = 1, .most = 1 << 24},
    ParamSpec{.name = "rope_base", .type = ParamType::kReal, .least = 1e-30, .most = 1e12},
    Optional("rope_factor", ParamType::kReal, 1.0, 1e6, 1.0),
    Optional("yarn_context", ParamType::kInt, 0, 1 << 30, 0),
    Optional("beta_fast", ParamType::kReal, 0.0, 1e6, 0.0),
    Optional("beta_slow", ParamType::kReal, 0.0, 1e6, 0.0),
    // The compression ratio (0: the window alone), the overlap of its
    // blocks and the lightning indexer.
    Optional("ratio", ParamType::kInt, 0, 1024, 0),
    Optional("overlap", ParamType::kBool, 0, 1, 0),
    Optional("indexer", ParamType::kBool, 0, 1, 0),
    Optional("indexer_heads", ParamType::kInt, 0, 1024, 0),
    Optional("index_dim", ParamType::kInt, 0, 4096, 0),
    Optional("top_k", ParamType::kInt, 0, 1 << 24, 0),
    ParamSpec{.name = "cache", .type = ParamType::kName, .required = false, .choices = kCaches},
    // An injected ring's cells.
    Optional("cells", ParamType::kInt, 0, 1 << 24, 0),
    ParamSpec{.name = "kv_encoding",
              .type = ParamType::kName,
              .required = false,
              .choices = kKvEncodings},
};

enum : std::uint16_t {
  kQa,
  kQNorm,
  kQb,
  kKv,
  kKvNorm,
  kSinks,
  kOutA,
  kOutB,
  kCompKv,
  kCompGate,
  kCompApe,
  kCompNorm,
  kIdxQ,
  kIdxProj,
  kIdxKv,
  kIdxGate,
  kIdxApe,
  kIdxNorm,
};
constexpr std::array<Encoding, 3> kApe{Encoding::kF32, Encoding::kF16, Encoding::kBF16};
constexpr std::array kRoles{
    WeightRole{.name = "q_a"},
    WeightRole{.name = "q_norm", .encodings = kF32Only},
    WeightRole{.name = "q_b"},
    WeightRole{.name = "kv"},
    WeightRole{.name = "kv_norm", .encodings = kF32Only},
    WeightRole{.name = "sinks", .encodings = kF32Only},
    WeightRole{.name = "out_a"},
    WeightRole{.name = "out_b"},
    WeightRole{.name = "comp_kv"},
    WeightRole{.name = "comp_gate"},
    WeightRole{.name = "comp_ape", .encodings = kApe},
    WeightRole{.name = "comp_norm", .encodings = kF32Only},
    WeightRole{.name = "idx_q"},
    WeightRole{.name = "idx_proj"},
    WeightRole{.name = "idx_kv"},
    WeightRole{.name = "idx_gate"},
    WeightRole{.name = "idx_ape", .encodings = kApe},
    WeightRole{.name = "idx_norm", .encodings = kF32Only},
};

// The state's tensors, in its contract's order.
enum : std::uint16_t {
  kWindowKv,
  kCompressed,
  kCompressedKv,
  kCompressedScore,
  kIndex,
  kIndexKv,
  kIndexScore
};

enum class Cache : std::uint8_t { kRing, kFull, kInjected };

struct Dims {
  std::uint64_t width, heads, d, rope_dims, q_rank, o_rank, groups, window;
  std::uint32_t ratio;
  bool overlap, indexer;
  std::uint64_t ih, idim, top_k;
  Cache cache;
  std::uint64_t cells;
  float eps, base;
  Yarn yarn;
  Encoding kv_encoding;
};
Dims DimsOf(const Instance& in) {
  const Params& p = in.params;
  const std::string_view cache = p.Name("cache");
  return {.width = in.stage.width,
          .heads = static_cast<std::uint64_t>(p.Int("heads")),
          .d = static_cast<std::uint64_t>(p.Int("head_dim")),
          .rope_dims = static_cast<std::uint64_t>(p.Int("rope_dims")),
          .q_rank = static_cast<std::uint64_t>(p.Int("q_rank")),
          .o_rank = static_cast<std::uint64_t>(p.Int("o_rank")),
          .groups = static_cast<std::uint64_t>(p.Int("o_groups")),
          .window = static_cast<std::uint64_t>(p.Int("window")),
          .ratio = static_cast<std::uint32_t>(p.Int("ratio")),
          .overlap = p.Flag("overlap"),
          .indexer = p.Flag("indexer"),
          .ih = static_cast<std::uint64_t>(p.Int("indexer_heads")),
          .idim = static_cast<std::uint64_t>(p.Int("index_dim")),
          .top_k = static_cast<std::uint64_t>(p.Int("top_k")),
          .cache = cache == "full"       ? Cache::kFull
                   : cache == "injected" ? Cache::kInjected
                                         : Cache::kRing,
          .cells = static_cast<std::uint64_t>(p.Int("cells")),
          .eps = F32(p.Real("eps")),
          .base = F32(p.Real("rope_base")),
          .yarn = {.factor = F32(p.Real("rope_factor")),
                   .context = static_cast<std::uint32_t>(p.Int("yarn_context")),
                   .beta_fast = F32(p.Real("beta_fast")),
                   .beta_slow = F32(p.Real("beta_slow"))},
          .kv_encoding = p.Name("kv_encoding") == "q8_0" ? Encoding::kQ8_0 : Encoding::kF16};
}

std::uint64_t Coff(const Dims& d) { return d.overlap ? 2 : 1; }
bool Yarned(const Dims& d) { return d.yarn.factor > 1.0F; }

std::expected<std::vector<RoleShape>, std::string> Shapes(const Instance& in) {
  const Dims d = DimsOf(in);
  if (d.kv_encoding == Encoding::kQ8_0 && (d.d != 256 && d.d != 512)) {
    return std::unexpected("Q8_0 KV requires head_dim 256 or 512");
  }
  if (d.heads % d.groups != 0) return std::unexpected("heads are not a multiple of o_groups");
  if (d.rope_dims % 2 != 0 || d.rope_dims > d.d) {
    return std::unexpected("rope_dims is odd or past a head");
  }
  if (Yarned(d) && d.yarn.context == 0) return std::unexpected("YaRN without its original context");
  if (d.ratio == 0 && (d.overlap || d.indexer)) {
    return std::unexpected("overlap or an indexer without a compression ratio");
  }
  if (d.indexer && (!d.overlap || d.ih == 0 || d.idim < d.rope_dims ||
                    !std::has_single_bit(d.idim) || d.top_k == 0)) {
    return std::unexpected(
        "an indexer needs overlapping blocks, heads, a power-of-two index_dim of at least "
        "rope_dims, and top_k");
  }
  if (!d.indexer && (in.params.Given("indexer_heads") || in.params.Given("index_dim") ||
                     in.params.Given("top_k"))) {
    return std::unexpected("an indexer's parameters without the indexer");
  }
  if ((d.cache == Cache::kInjected) != in.params.Given("cells") ||
      (d.cache == Cache::kInjected && (d.ratio != 0 || d.cells <= d.window))) {
    return std::unexpected("an injected ring's cells, above the window, of a window-only layer");
  }
  const std::uint64_t w = d.width;
  const std::uint64_t hd = d.heads * d.d;
  std::vector<RoleShape> out{Role(kQa, {w, d.q_rank}),
                             Role(kQNorm, {d.q_rank}),
                             Role(kQb, {d.q_rank, hd}),
                             Role(kKv, {w, d.d}),
                             Role(kKvNorm, {d.d}),
                             Role(kSinks, {d.heads}),
                             Role(kOutA, {hd / d.groups, d.o_rank * d.groups}),
                             Role(kOutB, {d.o_rank * d.groups, w})};
  if (d.ratio != 0) {
    const std::uint64_t c = Coff(d) * d.d;
    out.push_back(Role(kCompKv, {w, c}));
    out.push_back(Role(kCompGate, {w, c}));
    out.push_back(Role(kCompApe, {c, d.ratio}));
    out.push_back(Role(kCompNorm, {d.d}));
  }
  if (d.indexer) {
    out.push_back(Role(kIdxQ, {d.q_rank, d.ih * d.idim}));
    out.push_back(Role(kIdxProj, {w, d.ih}));
    out.push_back(Role(kIdxKv, {w, 2 * d.idim}));
    out.push_back(Role(kIdxGate, {w, 2 * d.idim}));
    out.push_back(Role(kIdxApe, {2 * d.idim, d.ratio}));
    out.push_back(Role(kIdxNorm, {d.idim}));
  }
  return out;
}

model::StateContract State(const Instance& in) {
  const Dims d = DimsOf(in);
  const auto kv_cell_bytes = model::CellBytes(d.kv_encoding, d.d);
  base::Check(kv_cell_bytes.has_value(), "a cell of any width");
  model::StateContract c;
  switch (d.cache) {
    case Cache::kFull:
      c.tensors.push_back({.name = "kv",
                           .kind = model::StateKind::kKv,
                           .encoding = d.kv_encoding,
                           .cell_bytes = *kv_cell_bytes,
                           .rollback = model::Rollback::kSnapshot});
      break;
    case Cache::kRing:
      c.tensors.push_back({.name = "kv",
                           .kind = model::StateKind::kRing,
                           .encoding = d.kv_encoding,
                           .cell_bytes = *kv_cell_bytes,
                           .cells = model::CellRule::kRing,
                           .window_or_ratio = static_cast<std::uint32_t>(d.window),
                           .ring_reads_whole = true,
                           .rollback = model::Rollback::kSnapshot});
      break;
    case Cache::kInjected:
      // Its target's verify saves the cells its injection writes.
      c.tensors.push_back({.name = "kv",
                           .kind = model::StateKind::kDrafterRing,
                           .encoding = d.kv_encoding,
                           .cell_bytes = *kv_cell_bytes,
                           .cells = model::CellRule::kPositionRing,
                           .window_or_ratio = static_cast<std::uint32_t>(d.cells),
                           .rollback = model::Rollback::kSnapshot,
                           .target_writes = true});
      break;
  }
  const auto compressor = [&](std::string name, model::StateKind kind, std::uint64_t dim) {
    const Encoding enc = kind == model::StateKind::kCompressed ? d.kv_encoding : Encoding::kF16;
    const auto bytes = model::CellBytes(enc, dim);
    base::Check(bytes.has_value(), "a cell of any width");
    c.tensors.push_back({.name = name,
                         .kind = kind,
                         .encoding = enc,
                         .cell_bytes = *bytes,
                         .cells = model::CellRule::kPerBlock,
                         .window_or_ratio = d.ratio,
                         .basis = model::BlockBasis::kPositions,
                         .dummy_last = true,
                         .rollback = model::Rollback::kSnapshot,
                         // The compressed cache follows the raw window: one bank.
                         .bank_with_previous = kind == model::StateKind::kCompressed});
    const std::uint64_t coff = kind == model::StateKind::kIndex ? 2 : Coff(d);
    for (const char* part : {"_kv", "_score"}) {
      c.tensors.push_back({.name = name + part,
                           .kind = model::StateKind::kCompressor,
                           .encoding = Encoding::kF32,
                           .cell_bytes = coff * dim * 4,
                           .cells = model::CellRule::kPositionRing,
                           .window_or_ratio = static_cast<std::uint32_t>(coff * d.ratio),
                           .rollback = model::Rollback::kSnapshot});
    }
  };
  if (d.ratio != 0) compressor("compressed", model::StateKind::kCompressed, d.d);
  if (d.indexer) compressor("index", model::StateKind::kIndex, d.idim);
  return c;
}

std::vector<MaskUse> Masks(const Instance& in) {
  const Dims d = DimsOf(in);
  std::vector<MaskUse> out{{.tensor = kWindowKv,
                            .window = static_cast<std::uint32_t>(d.window),
                            .ahead = d.cache == Cache::kInjected}};
  if (d.ratio != 0)
    out.push_back({.tensor = kCompressed, .window = 0, .indexer_selected = d.indexer});
  return out;
}

constexpr std::array kBatchClasses{
    BatchClass{.op = ir::Op::kRmsNorm, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kMatMul, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kRope, .batch = ir::Batch::kJoined},
    BatchClass{.op = ir::Op::kCacheStore, .batch = ir::Batch::kPerOwner},
    BatchClass{.op = ir::Op::kPool, .batch = ir::Batch::kPerOwner},
    BatchClass{.op = ir::Op::kIndexScore, .batch = ir::Batch::kPerOwner},
    BatchClass{.op = ir::Op::kTopK, .batch = ir::Batch::kPerOwner},
    BatchClass{.op = ir::Op::kAttention, .batch = ir::Batch::kPerOwner},
};

constexpr std::array<std::string_view, 6> kPatterns{"dsv4_qhead",    "dsv4_compress",
                                                    "dsv4_lid_topk", "dsv4_sparse_attention",
                                                    "dsv4_out_a",    "rms_norm_mul"};

// The layer's rotation (the retired dsv4_graph.cc's Rope: YaRN where the mapping gives
// a factor, its attention factor reciprocal to YaRN's magnitude), over
// `head` values from head - rope_dims.
ir::RopeParams Rotation(const Dims& d, std::uint64_t head, bool inverse = false) {
  if (!Yarned(d)) {
    return {.mode = ir::RopeMode::kNormal,
            .inverse = inverse,
            .dims = static_cast<std::uint32_t>(d.rope_dims),
            .offset = static_cast<std::uint32_t>(head - d.rope_dims),
            .original_context = 0,
            .base = d.base,
            .scale = 1.0F,
            .ext_factor = 0.0F,
            .attn_factor = 1.0F,
            .beta_fast = 0.0F,
            .beta_slow = 0.0F};
  }
  const float scale = 1.0F / d.yarn.factor;
  return {.mode = ir::RopeMode::kNormal,
          .inverse = inverse,
          .dims = static_cast<std::uint32_t>(d.rope_dims),
          .offset = static_cast<std::uint32_t>(head - d.rope_dims),
          .original_context = d.yarn.context,
          .base = d.base,
          .scale = scale,
          .ext_factor = 1.0F,
          .attn_factor = 1.0F / (1.0F + (0.1F * std::log(1.0F / scale))),
          .beta_fast = d.yarn.beta_fast,
          .beta_slow = d.yarn.beta_slow};
}

// ------------------------------------------------------------------ emit

class Emitter {
 public:
  explicit Emitter(Fragment& f) : f_(f), b_(f.b), d_(DimsOf(f.instance)) {}

  std::expected<FragmentOut, std::string> Emit();
  std::expected<FragmentOut, std::string> Inject();

 private:
  // A segment's rows of a [n, rows] tensor (the tensor itself for one
  // segment), or of a [n, m, rows] one's.
  ir::TensorId Rows(ir::TensorId t, std::size_t s) {
    if (f_.step.segments().size() == 1) return t;
    const ir::TensorDesc td = Desc(b_, t);
    const SegmentRows& seg = f_.step.segments()[s];
    const bool three = td.ne(2) > 1 || td.format.shape.rank == 3;
    const format::Shape shape =
        three ? Shape({td.ne(0), td.ne(1), seg.rows}) : Shape({td.ne(0), seg.rows});
    return b_.View(t, shape, ViewStrides(td.nb[0], td.nb[1], three ? td.nb[2] : 0, shape),
                   std::uint64_t{seg.first_row} * (three ? td.nb[2] : td.nb[1]));
  }
  CellKey KeyOf(std::size_t s, std::uint16_t tensor) const {
    const std::uint64_t cap = Desc(b_, f_.State(s, tensor)).ne(1);
    switch (tensor) {
      case kWindowKv:
        return d_.cache == Cache::kFull ? CellKey{.capacity = cap}
               : d_.cache == Cache::kInjected
                   ? CellKey{.capacity = cap, .ring = true, .recent = true}
                   : CellKey{.capacity = cap, .ring = true};
      case kCompressed:
      case kIndex:
        return {.capacity = cap,
                .block = d_.ratio,
                .positions = true,
                .cell_pad = State(f_.instance).tensors[tensor].cell_pad};
      default:
        return {.capacity = cap, .ring = true, .recent = true};
    }
  }
  // A cache's first `cells` cells as attention reads them: [head, 1,
  // cells, 1] (llama_kv_cache::get_k).
  ir::TensorId CacheView(ir::TensorId cache, std::uint64_t cells) {
    const ir::TensorDesc cd = Desc(b_, cache);
    if (b_.tensor(cache).format.encoding == Encoding::kQ8_0) {
      const format::Shape shape = Shape({cd.ne(0), 1, cells, 1});
      auto format = b_.tensor(cache).format;
      format.shape = shape;
      return b_.View(cache, shape, ir::NaturalStrides(format), 0);
    }
    const std::uint64_t row = cd.nb[1];
    return b_.View(cache, Shape({cd.ne(0), 1, cells, 1}), {cd.nb[0], row, row, row * cd.ne(1)}, 0);
  }
  // A [n, rows] mask (or its exact rows of a padded one).
  ir::TensorId MaskRows(ir::TensorId mask, std::uint64_t rows) {
    const ir::TensorDesc md = Desc(b_, mask);
    const format::Shape shape = Shape({md.ne(0), rows});
    return b_.View(mask, shape, ViewStrides(md.nb[0], md.nb[1], 0, shape), 0);
  }

  // A compressor (its weights, its state tensors' indices and its value
  // width): each segment's blocks pooled and stored, then its ring rows.
  struct Comp {
    std::uint16_t kv, gate, ape, norm;
    std::uint16_t cache, ring_kv, ring_score;
    std::uint64_t dim;
    bool hadamard;
  };
  void Compressor(const Comp& c, std::size_t s);
  void StoreWindow(std::size_t s);
  ir::TensorId Query();
  ir::TensorId Key();
  ir::TensorId IndexerQueries();
  std::expected<ir::TensorId, std::string> Selection(std::size_t s, ir::TensorId mask,
                                                     ir::TensorId raw = {});
  std::expected<ir::TensorId, std::string> Attend(std::size_t s);
  ir::TensorId Hadamard(ir::TensorId t, std::uint64_t n);

  Fragment& f_;
  ir::Builder& b_;
  Dims d_;
  ir::TensorId q_{}, qr_{}, kv_{}, iq_{}, iw_{}, comp_kv_{}, comp_score_{}, idx_kv_{}, idx_score_{};
};

ir::TensorId Emitter::Hadamard(ir::TensorId t, std::uint64_t n) {
  // llama_mul_mat_hadamard: the rows as [n, *], the product with the
  // Hadamard matrix, its shape back.
  const ir::TensorDesc td = Desc(b_, t);
  const std::uint64_t count = td.ne(0) * td.ne(1) * td.ne(2) * td.ne(3);
  const ir::TensorId flat = b_.Reshape(t, Shape({n, count / n}));
  const ir::TensorId h = f_.step.Hadamard(n);
  const ir::TensorId product = b_.MatMul(h, flat, {.hint = ir::ProductHint::kHadamard});
  return b_.Reshape(product, Shape({td.ne(0), td.ne(1), td.ne(2), td.ne(3)}));
}

void Emitter::Compressor(const Comp& c, std::size_t s) {
  const ir::TensorId x = f_.stream;
  const std::uint32_t ratio = d_.ratio;
  const bool overlap = c.cache == kIndex || d_.overlap;
  const std::uint64_t coff = overlap ? 2 : 1;
  const std::uint64_t dim = c.dim;
  const ir::RopeParams rope = Rotation(d_, dim);
  ir::TensorId kv = c.cache == kIndex ? idx_kv_ : comp_kv_;
  ir::TensorId score = c.cache == kIndex ? idx_score_ : comp_score_;
  {
    const SegmentRows& seg = f_.step.segments()[s];
    b_.SetOwner(static_cast<std::uint8_t>(s));
    const CellKey ring = KeyOf(s, c.ring_kv);
    const std::uint64_t blocks = f_.step.Blocks(s, ratio, overlap);
    const std::uint64_t reads = ratio * blocks;
    // [ring | the segment's rows (| a zero row)] of one side's values.
    const auto source = [&](std::uint16_t tensor, bool scores) {
      const ir::TensorId state = f_.State(s, tensor);
      const ir::TensorDesc sd = Desc(b_, state);
      const ir::TensorId whole =
          b_.View(state, Shape({sd.ne(0), sd.ne(1)}),
                  ViewStrides(sd.nb[0], sd.nb[1], 0, Shape({sd.ne(0), sd.ne(1)})), 0);
      if (scores && !score.valid()) {
        b_.SetJoined();
        const ir::TensorId gate = b_.MatMul(f_.Weight(c.gate), x);
        const ir::TensorId ape = b_.GetRows(f_.Weight(c.ape), f_.step.PositionPhases(ratio));
        score = b_.Add(gate, ape);
        b_.SetOwner(static_cast<std::uint8_t>(s));
      } else if (!scores && !kv.valid()) {
        b_.SetJoined();
        kv = b_.MatMul(f_.Weight(c.kv), x);
        b_.SetOwner(static_cast<std::uint8_t>(s));
      }
      ir::TensorId joined = b_.Concat(whole, Rows(scores ? score : kv, s), 1);
      if (overlap) {
        // dsv4_append_zero_row: a zero value (a -inf score) past the rows.
        const ir::TensorDesc jd = Desc(b_, joined);
        const ir::TensorId row =
            b_.Fill(Shape({jd.ne(0), 1}), scores ? -std::numeric_limits<float>::infinity() : 0.0F,
                    Encoding::kF32);
        joined = b_.Concat(joined, row, 1);
      }
      const ir::TensorId picked = b_.GetRows(joined, f_.step.PoolReads(s, ring, ratio, overlap));
      if (!overlap) return b_.Reshape(picked, Shape({dim, ratio, blocks}));
      // The previous blocks' first halves, then the blocks' own second
      // halves, side by side per block.
      const ir::TensorDesc pd = Desc(b_, picked);
      const format::Shape half = Shape({dim, reads});
      ir::TensorId prev =
          b_.Cont(b_.View(picked, half, ViewStrides(pd.nb[0], pd.nb[1], 0, half), 0));
      prev = b_.Reshape(prev, Shape({dim, ratio, blocks}));
      ir::TensorId cur = b_.Cont(b_.View(picked, half, ViewStrides(pd.nb[0], pd.nb[1], 0, half),
                                         (reads * pd.nb[1]) + (dim * pd.nb[0])));
      cur = b_.Reshape(cur, Shape({dim, ratio, blocks}));
      return b_.Concat(prev, cur, 1);
    };
    const ir::TensorId values = source(c.ring_kv, false);
    const ir::TensorId scores = source(c.ring_score, true);
    ir::TensorId pooled = b_.Pool(
        values, scores,
        {.ratio = static_cast<std::uint32_t>(coff * ratio), .kind = ir::PoolKind::kSoftmax});
    pooled = b_.Mul(b_.RmsNorm(pooled, d_.eps), f_.Weight(c.norm));
    pooled = b_.Rope(pooled, f_.step.BlockStarts(s, ratio, overlap), {}, rope);
    if (c.hadamard) pooled = Hadamard(pooled, dim);
    const ir::TensorDesc cd = Desc(b_, pooled);
    const format::Shape rows_shape = Shape({cd.ne(0) * cd.ne(1), cd.ne(2)});
    ir::TensorId stored =
        b_.View(pooled, rows_shape, ViewStrides(cd.nb[0], cd.nb[2], 0, rows_shape), 0);
    if (c.cache == kCompressed && d_.kv_encoding == Encoding::kQ8_0 &&
        b_.tensor(stored).format.encoding != Encoding::kF32) {
      stored = b_.Convert(stored, Encoding::kF32);
    }
    b_.CacheStore(f_.State(s, c.cache), stored,
                  f_.step.BlockCells(s, KeyOf(s, c.cache), ratio, overlap));
    // The segment's last rows kept in the ring.
    b_.CacheStore(f_.State(s, c.ring_kv), b_.GetRows(Rows(kv, s), f_.step.KeptRows(s, ring)),
                  f_.step.KeptCells(s, ring));
    b_.CacheStore(f_.State(s, c.ring_score), b_.GetRows(Rows(score, s), f_.step.KeptRows(s, ring)),
                  f_.step.KeptCells(s, ring));
    b_.SetJoined();
    (void)seg;
  }
}

void Emitter::StoreWindow(std::size_t s) {
  const ir::TensorDesc kd = Desc(b_, kv_);
  const SegmentRows& seg = f_.step.segments()[s];
  b_.SetOwner(static_cast<std::uint8_t>(s));
  const format::Shape shape = Shape({d_.d, seg.rows});
  ir::TensorId rows_of = b_.View(kv_, shape, ViewStrides(kd.nb[0], kd.nb[2], 0, shape),
                                 std::uint64_t{seg.first_row} * kd.nb[2]);
  if (d_.kv_encoding == Encoding::kQ8_0 && b_.tensor(rows_of).format.encoding != Encoding::kF32) {
    rows_of = b_.Convert(rows_of, Encoding::kF32);
  }
  b_.CacheStore(f_.State(s, kWindowKv), rows_of, f_.step.StoreCells(s, KeyOf(s, kWindowKv)));
  b_.SetJoined();
}

ir::TensorId Emitter::Query() {
  const std::uint64_t rows = f_.step.rows();
  qr_ = b_.MatMul(f_.Weight(kQa), f_.stream);
  qr_ = b_.Mul(b_.RmsNorm(qr_, d_.eps), f_.Weight(kQNorm));
  ir::TensorId q = b_.Reshape(b_.MatMul(f_.Weight(kQb), qr_), Shape({d_.d, d_.heads, rows}));
  q = b_.RmsNorm(q, d_.eps);
  return b_.Rope(q, f_.step.Positions(), {}, Rotation(d_, d_.d));
}

ir::TensorId Emitter::Key() {
  const std::uint64_t rows = f_.step.rows();
  ir::TensorId kv = b_.MatMul(f_.Weight(kKv), f_.stream);
  kv = b_.Reshape(b_.Mul(b_.RmsNorm(kv, d_.eps), f_.Weight(kKvNorm)), Shape({d_.d, 1, rows}));
  return b_.Rope(kv, f_.step.Positions(), {}, Rotation(d_, d_.d));
}

ir::TensorId Emitter::IndexerQueries() {
  if (iq_.valid()) return iq_;
  const std::uint64_t rows = f_.step.rows();
  ir::TensorId q = b_.Reshape(b_.MatMul(f_.Weight(kIdxQ), qr_), Shape({d_.idim, d_.ih, rows}));
  q = b_.Rope(q, f_.step.Positions(), {}, Rotation(d_, d_.idim));
  iq_ = Hadamard(q, d_.idim);
  return iq_;
}

std::expected<ir::TensorId, std::string> Emitter::Selection(std::size_t s, ir::TensorId mask,
                                                            ir::TensorId raw) {
  const SegmentRows& seg = f_.step.segments()[s];
  const CellKey cells = KeyOf(s, kIndex);
  const std::uint64_t read = f_.step.ReadCells(s, cells);
  const std::uint64_t top = std::min(read, d_.top_k);
  const float neg_inf = -std::numeric_limits<float>::infinity();
  // build_top_k_mask: zeros stored at the selected cells of a -inf fill of
  // the mask's shape, the causal mask added (in GGML's order: the stored
  // zeros, the selection, then the fill they are stored into).
  const bool rounded = f_.step.rounded_indexer_context() != 0;
  const ir::TensorId zeros =
      rounded ? ir::TensorId{} : b_.Fill(Shape({1, top, seg.rows, 1}), 0.0F, Encoding::kF16);
  // The indexer's selection: its queries (every owner's rows) and head
  // weights, the scores over this owner's index cells, the best.
  b_.SetJoined();
  const ir::TensorId queries = IndexerQueries();
  b_.SetOwner(static_cast<std::uint8_t>(s));
  const ir::TensorDesc qd = Desc(b_, queries);
  const format::Shape q_shape = Shape({d_.idim, d_.ih, seg.rows, 1});
  const ir::TensorId q =
      b_.View(queries, q_shape, {qd.nb[0], qd.nb[1], qd.nb[2], qd.nb[2] * seg.rows},
              std::uint64_t{seg.first_row} * qd.nb[2]);
  const ir::TensorId keys = CacheView(f_.State(s, kIndex), read);
  if (!iw_.valid()) {
    b_.SetJoined();
    iw_ = b_.Scale(b_.MatMul(f_.Weight(kIdxProj), f_.stream),
                   1.0F / std::sqrt(static_cast<float>(d_.idim * d_.ih)));
    b_.SetOwner(static_cast<std::uint8_t>(s));
  }
  const ir::TensorDesc wd = Desc(b_, iw_);
  const format::Shape w_shape = Shape({d_.ih, seg.rows, 1, 1});
  const ir::TensorId weights = b_.View(iw_, w_shape, ViewStrides(wd.nb[0], wd.nb[1], 0, w_shape),
                                       std::uint64_t{seg.first_row} * wd.nb[1]);
  if (rounded) {
    // Completed blocks exclude the logical tail and all padding/dummy cells.
    // The selected IDs belong to these scores, not to the ordinary TopK path.
    auto counts = f_.step.VisibleCells(s, cells, read, f_.step.rounded_indexer_context() / 4);
    if (!counts) return std::unexpected(counts.error());
    const auto score = b_.RoundedIndexScore(q, keys, weights, *counts);
    const auto best = b_.VisibleTopK(score, *counts, static_cast<std::uint32_t>(top));
    return b_.SelectedMask(raw, best,
                           {.raw_cells = static_cast<std::uint32_t>(Desc(b_, raw).ne(0)),
                            .cells = static_cast<std::uint32_t>(read)});
  }
  const ir::TensorId score = b_.IndexScore(q, keys, weights, mask);
  const ir::TensorId best = b_.Cont(b_.TopK(score, static_cast<std::uint32_t>(top)));
  const ir::TensorDesc bd = Desc(b_, best);
  const ir::TensorId best3 =
      b_.View(best, Shape({top, seg.rows, 1, 1}), {bd.nb[0], bd.nb[1], bd.nb[2], bd.nb[3]}, 0);
  const ir::TensorId all = b_.Fill(Shape({read, seg.rows}), neg_inf, Encoding::kF16);
  const ir::TensorDesc ad = Desc(b_, all);
  const ir::TensorId all4 =
      b_.View(all, Shape({1, read, seg.rows, 1}), {ad.nb[0], ad.nb[0], ad.nb[1], ad.nb[2]}, 0);
  const ir::TensorId masked = b_.CacheStore(all4, zeros, best3);
  const ir::TensorDesc md = Desc(b_, masked);
  const ir::TensorId rows =
      b_.View(masked, Shape({read, seg.rows, 1, 1}), {md.nb[0], md.nb[2], md.nb[3], md.nb[3]}, 0);
  return b_.Add(rows, mask);
}

std::expected<ir::TensorId, std::string> Emitter::Attend(std::size_t s) {
  const SegmentRows& seg = f_.step.segments()[s];
  b_.SetOwner(static_cast<std::uint8_t>(s));
  // Its rows' queries.
  const ir::TensorDesc qd = Desc(b_, q_);
  const ir::TensorId q = b_.View(q_, Shape({d_.d, d_.heads, seg.rows, 1}),
                                 {qd.nb[0], qd.nb[1], qd.nb[2], qd.nb[2] * seg.rows},
                                 std::uint64_t{seg.first_row} * qd.nb[2]);
  const ir::TensorId queries = b_.Permute(q, {0, 2, 1, 3});
  // Its window cells, then its compressed rows, as one K (and V).
  const CellKey window_cells = KeyOf(s, kWindowKv);
  const std::uint64_t window_read = f_.step.ReadCells(s, window_cells);
  if (d_.cache == Cache::kInjected && std::uint64_t{seg.rows} + d_.window > window_cells.capacity) {
    return std::unexpected(std::format("{} rows and a window of {} in a ring of {} cells", seg.rows,
                                       d_.window, window_cells.capacity));
  }
  ir::TensorId k = CacheView(f_.State(s, kWindowKv), window_read);
  std::uint64_t compressed_read = 0;
  if (d_.ratio != 0) {
    const CellKey cells = KeyOf(s, kCompressed);
    compressed_read = f_.step.ReadCells(s, cells);
    k = b_.Concat(k, CacheView(f_.State(s, kCompressed), compressed_read), 2);
  }
  const ir::TensorId keys = b_.Permute(k, {0, 2, 1, 3});
  const ir::TensorId values = b_.Permute(k, {0, 2, 1, 3});
  auto window_mask = f_.step.Mask(s, window_cells, static_cast<std::uint32_t>(d_.window),
                                  d_.cache == Cache::kInjected);
  if (!window_mask) return std::unexpected(window_mask.error());
  ir::TensorId mask = *window_mask;
  std::uint32_t n_kv_max = 0;
  const bool rounded = f_.step.rounded_indexer_context() != 0;
  if (d_.ratio != 0) {
    if (d_.indexer) {
      n_kv_max = static_cast<std::uint32_t>(std::min(window_read, d_.window) +
                                            std::min(compressed_read, d_.top_k));
    } else if (rounded) {
      n_kv_max = static_cast<std::uint32_t>(std::min(window_read, d_.window) + compressed_read);
    }
    if (d_.indexer && rounded) {
      auto selected = Selection(s, ir::TensorId{}, MaskRows(*window_mask, seg.rows));
      if (!selected) return std::unexpected(selected.error());
      const ir::TensorId rows_mask = *selected;
      if (Desc(b_, rows_mask).ne(0) != window_read + compressed_read)
        return std::unexpected("rounded indexer mask differs from the actual attention bank");
      mask = rows_mask;
    } else {
      auto compressed_mask = f_.step.Mask(s, KeyOf(s, kCompressed), 0);
      if (!compressed_mask) return std::unexpected(compressed_mask.error());
      ir::TensorId rows_mask = MaskRows(*compressed_mask, seg.rows);
      if (d_.indexer) {
        auto selected = Selection(s, rows_mask, ir::TensorId{});
        if (!selected) return std::unexpected(selected.error());
        rows_mask = *selected;
      }
      mask = b_.Concat(MaskRows(*window_mask, seg.rows), rows_mask, 0);
    }
  } else if (rounded && d_.cache == Cache::kRing) {
    n_kv_max = static_cast<std::uint32_t>(std::min(window_read, d_.window));
  }
  const bool sparse_any = rounded && d_.d == 512 && n_kv_max != 0;
  const ir::TensorId attention = b_.Attention(queries, keys, values, mask, f_.Weight(kSinks),
                                              {.scale = 1.0F / std::sqrt(static_cast<float>(d_.d)),
                                               .sinks = true,
                                               .accumulate = ir::Accumulate::kF32,
                                               .sparse_any = sparse_any,
                                               .n_kv_max = n_kv_max});
  return b_.Reshape(attention, Shape({d_.d * d_.heads, seg.rows}));
}

std::expected<FragmentOut, std::string> Emitter::Emit() {
  const std::uint64_t rows = f_.step.rows();
  const auto segments = f_.step.segments();
  const Comp comp_c{.kv = kCompKv,
                    .gate = kCompGate,
                    .ape = kCompApe,
                    .norm = kCompNorm,
                    .cache = kCompressed,
                    .ring_kv = kCompressedKv,
                    .ring_score = kCompressedScore,
                    .dim = d_.d,
                    .hadamard = false};
  const Comp idx_c{.kv = kIdxKv,
                   .gate = kIdxGate,
                   .ape = kIdxApe,
                   .norm = kIdxNorm,
                   .cache = kIndex,
                   .ring_kv = kIndexKv,
                   .ring_score = kIndexScore,
                   .dim = d_.idim,
                   .hadamard = true};
  std::vector<ir::TensorId> outs;
  outs.reserve(segments.size());
  if (segments.size() > 1) {
    b_.SetJoined();
    if (d_.ratio != 0) {
      comp_kv_ = b_.MatMul(f_.Weight(kCompKv), f_.stream);
      const ir::TensorId gate = b_.MatMul(f_.Weight(kCompGate), f_.stream);
      const ir::TensorId ape = b_.GetRows(f_.Weight(kCompApe), f_.step.PositionPhases(d_.ratio));
      comp_score_ = b_.Add(gate, ape);
    }
    if (d_.indexer) {
      idx_kv_ = b_.MatMul(f_.Weight(kIdxKv), f_.stream);
      const ir::TensorId gate = b_.MatMul(f_.Weight(kIdxGate), f_.stream);
      const ir::TensorId ape = b_.GetRows(f_.Weight(kIdxApe), f_.step.PositionPhases(d_.ratio));
      idx_score_ = b_.Add(gate, ape);
    }
    if (f_.stream_needed) q_ = Query();
    kv_ = Key();
    if (f_.stream_needed && d_.ratio != 0 && d_.indexer) {
      (void)IndexerQueries();
      iw_ = b_.Scale(b_.MatMul(f_.Weight(kIdxProj), f_.stream),
                     1.0F / std::sqrt(static_cast<float>(d_.idim * d_.ih)));
    }
    for (std::size_t s = 0; s < segments.size(); ++s) {
      if (d_.ratio != 0) Compressor(comp_c, s);
      if (d_.indexer) Compressor(idx_c, s);
      StoreWindow(s);
      if (f_.stream_needed) {
        auto one = Attend(s);
        b_.SetJoined();
        if (!one) return std::unexpected(one.error());
        outs.push_back(*one);
      }
    }
    if (!f_.stream_needed) return FragmentOut{.stream = f_.stream};
  } else {
    // The compressors' stores and ring rows first.
    if (d_.ratio != 0) Compressor(comp_c, 0);
    if (d_.indexer) Compressor(idx_c, 0);
    if (f_.stream_needed && !q_.valid()) q_ = Query();
    if (!kv_.valid()) kv_ = Key();
    StoreWindow(0);
    if (!f_.stream_needed) return FragmentOut{.stream = f_.stream};
    auto one = Attend(0);
    b_.SetJoined();
    if (!one) return std::unexpected(one.error());
    outs.push_back(*one);
  }
  ir::TensorId joined = outs.front();
  for (std::size_t s = 1; s < outs.size(); ++s) {
    joined = b_.Concat(joined, outs[s], 1);
  }
  // Each head rotated back, then the grouped output product.
  ir::TensorId out = b_.Reshape(joined, Shape({d_.d, d_.heads, rows}));
  out = b_.Rope(out, f_.step.Positions(), {}, Rotation(d_, d_.d, true));
  const std::uint64_t per = d_.heads / d_.groups * d_.d;
  out = b_.Permute(b_.Reshape(out, Shape({per, d_.groups, rows})), {0, 2, 1, 3});
  const ir::TensorId groups = b_.Reshape(f_.Weight(kOutA), Shape({per, d_.o_rank, d_.groups}));
  ir::TensorId low = b_.Permute(b_.MatMul(groups, out), {0, 2, 1, 3});
  low = b_.Cont(low, Shape({d_.o_rank * d_.groups, rows}));
  return FragmentOut{.stream = b_.MatMul(f_.Weight(kOutB), low)};
}

std::expected<FragmentOut, std::string> Emitter::Inject() {
  if (d_.cache != Cache::kInjected) return std::unexpected("an injection into no injected ring");
  const auto segments = f_.step.segments();
  const ir::TensorId input = f_.stream;
  const std::uint64_t injected = Desc(b_, input).ne(1);
  // The injected rows' positions: each segment's last rows.
  ir::TensorId positions = f_.step.Positions();
  std::uint64_t total = 0;
  std::vector<std::uint64_t> kept(segments.size());
  for (std::size_t s = 0; s < segments.size(); ++s) {
    kept[s] = f_.step.Kept(s, KeyOf(s, kWindowKv).capacity);
    total += kept[s];
  }
  if (total != injected) return std::unexpected("an injected input of other rows than its rings'");
  if (total != f_.step.rows()) {
    ir::TensorId joined{};
    for (std::size_t s = 0; s < segments.size(); ++s) {
      const SegmentRows& seg = segments[s];
      const format::Shape shape = Shape({kept[s]});
      const ir::TensorId one =
          b_.View(f_.step.Positions(), shape, {4, 4 * kept[s], 4 * kept[s], 4 * kept[s]},
                  4 * (std::uint64_t{seg.first_row} + seg.rows - kept[s]));
      joined = joined.valid() ? b_.Concat(joined, one, 0) : one;
    }
    positions = joined;
  }
  ir::TensorId kv = b_.MatMul(f_.Weight(kKv), input);
  kv = b_.Reshape(b_.Mul(b_.RmsNorm(kv, d_.eps), f_.Weight(kKvNorm)), Shape({d_.d, 1, injected}));
  kv = b_.Rope(kv, positions, {}, Rotation(d_, d_.d));
  const ir::TensorDesc kd = Desc(b_, kv);
  std::uint64_t first = 0;
  for (std::size_t s = 0; s < segments.size(); ++s) {
    b_.SetOwner(static_cast<std::uint8_t>(s));
    const format::Shape shape = Shape({d_.d, kept[s]});
    ir::TensorId rows =
        b_.View(kv, shape, ViewStrides(kd.nb[0], kd.nb[2], 0, shape), first * kd.nb[2]);
    if (d_.kv_encoding == Encoding::kQ8_0 && b_.tensor(rows).format.encoding != Encoding::kF32) {
      rows = b_.Convert(rows, Encoding::kF32);
    }
    const CellKey ring = KeyOf(s, kWindowKv);
    b_.CacheStore(f_.State(s, kWindowKv), rows, f_.step.KeptCells(s, ring));
    b_.SetJoined();
    first += kept[s];
  }
  return FragmentOut{.stream = input};
}

std::expected<FragmentOut, std::string> Emit(Fragment& f) {
  Emitter e(f);
  return e.Emit();
}

std::expected<FragmentOut, std::string> Inject(Fragment& f) {
  Emitter e(f);
  return e.Inject();
}

// ------------------------------------------------------------- reference

// A position's cell in a ring of `cap` cells (or its own: a cell per
// position), and the position a cell holds once `end` positions are
// stored (-1: none).
std::int64_t Held(std::uint64_t c, std::uint64_t end, std::uint64_t cap, bool ring) {
  if (c >= end) return -1;
  if (!ring) return static_cast<std::int64_t>(c);
  return static_cast<std::int64_t>(c + (((end - 1 - c) / cap) * cap));
}

// The normalized Hadamard product of each run of `n` values.
void HadamardRows(std::span<double> v, std::uint64_t n) {
  const double scale = 1.0 / std::sqrt(static_cast<double>(n));
  std::vector<double> out(n);
  for (std::size_t r = 0; r + n <= v.size(); r += n) {
    for (std::uint64_t i = 0; i < n; ++i) {
      double s = 0.0;
      for (std::uint64_t k = 0; k < n; ++k) {
        s += (std::popcount(i & k) % 2 == 0 ? scale : -scale) * v[r + k];
      }
      out[i] = s;
    }
    std::ranges::copy(out, v.begin() + static_cast<std::ptrdiff_t>(r));
  }
}

// One compressor's reference over a segment: its blocks pooled, normed,
// rotated (and multiplied by the Hadamard matrix) into its cache, then its
// ring rows kept. `kv` and `score` are the segment's rows' projections
// (the score with its position embedding's row).
std::expected<void, std::string> CompressorReference(
    const Dims& d, std::span<const double> kv, std::span<const double> score, std::uint64_t dim,
    bool overlap, bool hadamard, std::span<const float> norm, const RefSegment& seg,
    std::vector<double>& cache, std::uint64_t cache_cells, std::vector<double>& ring_kv,
    std::vector<double>& ring_score, std::uint64_t ring_cells,
    Encoding cache_encoding = Encoding::kF16,
    std::expected<void, std::string> (*store_row)(Encoding, std::span<const double>,
                                                  std::span<double>) = nullptr) {
  const std::uint64_t ratio = d.ratio;
  const std::uint64_t width = (overlap ? 2 : 1) * dim;
  const std::uint64_t past = seg.past;
  const std::uint64_t end = past + seg.rows;
  const double neg_inf = -std::numeric_limits<double>::infinity();
  // A position's projection row: the segment's, the ring's, or (before 0)
  // a zero value under a -inf score.
  const auto value = [&](std::int64_t pos, std::uint64_t i, bool scores) -> double {
    if (pos < 0) return scores ? neg_inf : 0.0;
    const auto p = static_cast<std::uint64_t>(pos);
    if (p >= past && p < end) return (scores ? score : kv)[((p - past) * width) + i];
    return (scores ? ring_score : ring_kv)[((p % ring_cells) * width) + i];
  };
  // The blocks: those completed, then the dummy where the plan wants one.
  struct Block {
    std::int64_t start;
    std::uint64_t cell;
    bool dummy;
  };
  std::vector<Block> blocks;
  for (std::uint64_t pos = past; pos < end; ++pos) {
    if ((pos + 1) % ratio == 0) {
      blocks.push_back({.start = static_cast<std::int64_t>(pos + 1 - ratio),
                        .cell = pos / ratio,
                        .dummy = false});
    }
  }
  const std::uint64_t want = CompressorBlocks(seg.rows, seg.past, d.ratio, overlap);
  if (blocks.size() + 1 == want) {
    blocks.push_back({.start = 0, .cell = cache_cells - 1, .dummy = true});
  } else if (blocks.size() != want) {
    return std::unexpected("compressor reference: the blocks are not the plan's");
  }
  for (const Block& block : blocks) {
    if (block.cell >= cache_cells) return std::unexpected("compressor reference: past the cache");
    std::vector<double> pooled(dim);
    for (std::uint64_t c = 0; c < dim; ++c) {
      // Each pooled position's value and score of channel c.
      std::vector<std::pair<double, double>> runs;
      // With overlap, the previous block's positions' first halves.
      for (std::uint64_t j = 0; overlap && j < ratio; ++j) {
        const std::int64_t prev = block.dummy ? static_cast<std::int64_t>(past)
                                              : block.start - static_cast<std::int64_t>(ratio) +
                                                    static_cast<std::int64_t>(j);
        runs.emplace_back(value(prev, c, false), value(prev, c, true));
      }
      // The block's own positions' (second halves, with overlap).
      for (std::uint64_t j = 0; j < ratio; ++j) {
        const std::int64_t own = block.dummy ? static_cast<std::int64_t>(past)
                                             : block.start + static_cast<std::int64_t>(j);
        const std::uint64_t at = overlap ? dim + c : c;
        runs.emplace_back(value(own, at, false), value(own, at, true));
      }
      double top = neg_inf;
      for (const auto& r : runs) top = std::max(top, r.second);
      if (std::isinf(top)) continue;  // every score -inf: 0
      double sum = 0.0;
      double acc = 0.0;
      for (const auto& [v, sc] : runs) {
        const double e = std::exp(sc - top);
        sum += e;
        acc += e * v;
      }
      pooled[c] = acc / sum;
    }
    RmsRows(pooled, dim, d.eps, norm);
    RotateNormal(pooled, 1, dim, dim - d.rope_dims, static_cast<std::uint32_t>(d.rope_dims), d.base,
                 d.yarn, block.start);
    if (hadamard) HadamardRows(pooled, dim);
    if (cache_encoding == Encoding::kQ8_0) {
      if (store_row == nullptr) {
        return std::unexpected("Q8_0 compressor reference state requires the block row codec");
      }
      auto dst = std::span(cache).subspan(block.cell * dim, dim);
      if (auto stored = store_row(cache_encoding, pooled, dst); !stored) return stored;
    } else {
      for (std::uint64_t c = 0; c < dim; ++c) {
        cache[(block.cell * dim) + c] = Stored(cache_encoding, pooled[c]);
      }
    }
  }
  // The segment's last rows into the ring.
  const std::uint64_t kept = std::min<std::uint64_t>(seg.rows, ring_cells);
  for (std::uint64_t i = seg.rows - kept; i < seg.rows; ++i) {
    const std::uint64_t cell = (past + i) % ring_cells;
    for (std::uint64_t c = 0; c < width; ++c) {
      ring_kv[(cell * width) + c] = Stored(Encoding::kF32, kv[(i * width) + c]);
      ring_score[(cell * width) + c] = Stored(Encoding::kF32, score[(i * width) + c]);
    }
  }
  return {};
}

std::expected<void, std::string> Reference(RefIO& io) {
  const Instance& in = *io.instance;
  const Dims d = DimsOf(in);
  if (d.kv_encoding == Encoding::kQ8_0 && io.store_row == nullptr) {
    return std::unexpected("Q8_0 reference state requires the block row codec");
  }
  const std::uint64_t w = d.width;
  const std::uint64_t rows = io.rows();
  const std::size_t tensors = d.ratio == 0 ? 1 : (d.indexer ? 7 : 4);
  if (io.stream.size() != w * rows || io.state.size() != io.segments.size() ||
      io.state_cells.size() != tensors) {
    return std::unexpected("mqa reference: a stream or state of another shape");
  }
  const auto weight = [&](std::uint16_t role) {
    return std::span<const float>(io.weights.at(role));
  };
  // Every row's projections.
  std::vector<double> qr = Product(weight(kQa), w, d.q_rank, io.stream);
  RmsRows(qr, d.q_rank, d.eps, weight(kQNorm));
  std::vector<double> q = Product(weight(kQb), d.q_rank, d.heads * d.d, qr);
  RmsRows(q, d.d, d.eps, {});
  std::vector<double> kv = Product(weight(kKv), w, d.d, io.stream);
  RmsRows(kv, d.d, d.eps, weight(kKvNorm));
  const std::uint64_t coff = Coff(d);
  std::vector<double> ckv, cscore, ikv, iscore, iq, iw;
  const auto projections = [&](std::uint16_t kv_role, std::uint16_t gate_role,
                               std::uint16_t ape_role, std::uint64_t width,
                               std::vector<double>& kv_out, std::vector<double>& score_out) {
    kv_out = Product(weight(kv_role), w, width, io.stream);
    score_out = Product(weight(gate_role), w, width, io.stream);
    const std::span<const float> ape = weight(ape_role);
    std::uint64_t r = 0;
    for (const RefSegment& s : io.segments) {
      for (std::uint64_t i = 0; i < s.rows; ++i, ++r) {
        const std::uint64_t phase = (s.past + i) % d.ratio;
        for (std::uint64_t c = 0; c < width; ++c) {
          score_out[(r * width) + c] += static_cast<double>(ape[(phase * width) + c]);
        }
      }
    }
  };
  if (d.ratio != 0) projections(kCompKv, kCompGate, kCompApe, coff * d.d, ckv, cscore);
  if (d.indexer) {
    projections(kIdxKv, kIdxGate, kIdxApe, 2 * d.idim, ikv, iscore);
    iq = Product(weight(kIdxQ), d.q_rank, d.ih * d.idim, qr);
    iw = Product(weight(kIdxProj), w, d.ih, io.stream);
    const double scale = 1.0F / std::sqrt(static_cast<float>(d.idim * d.ih));
    for (double& v : iw) v *= scale;
  }
  std::vector<double> attended(rows * d.heads * d.d, 0.0);
  std::uint64_t first = 0;
  for (std::size_t si = 0; si < io.segments.size(); ++si) {
    const RefSegment& seg = io.segments[si];
    auto& st = io.state[si];
    const std::uint64_t end = std::uint64_t{seg.past} + seg.rows;
    const auto rows_of = [&](const std::vector<double>& t, std::uint64_t width) {
      return std::span<const double>(t).subspan(first * width, std::uint64_t{seg.rows} * width);
    };
    if (d.ratio != 0) {
      auto ok = CompressorReference(d, rows_of(ckv, coff * d.d), rows_of(cscore, coff * d.d), d.d,
                                    d.overlap, false, weight(kCompNorm), seg, st.at(kCompressed),
                                    io.state_cells.at(kCompressed), st.at(kCompressedKv),
                                    st.at(kCompressedScore), io.state_cells.at(kCompressedKv),
                                    d.kv_encoding, io.store_row);
      if (!ok) return ok;
    }
    if (d.indexer) {
      auto ok = CompressorReference(d, rows_of(ikv, 2 * d.idim), rows_of(iscore, 2 * d.idim),
                                    d.idim, true, true, weight(kIdxNorm), seg, st.at(kIndex),
                                    io.state_cells.at(kIndex), st.at(kIndexKv), st.at(kIndexScore),
                                    io.state_cells.at(kIndexKv));
      if (!ok) return ok;
    }
    // Rotations, then the key's store.
    const std::uint64_t window_cap = io.state_cells.at(kWindowKv);
    for (std::uint64_t t = 0; t < seg.rows; ++t) {
      const std::uint64_t r = first + t;
      const auto pos = static_cast<std::int64_t>(seg.past + t);
      RotateNormal(std::span(q).subspan(r * d.heads * d.d, d.heads * d.d), d.heads, d.d,
                   d.d - d.rope_dims, static_cast<std::uint32_t>(d.rope_dims), d.base, d.yarn, pos);
      RotateNormal(std::span(kv).subspan(r * d.d, d.d), 1, d.d, d.d - d.rope_dims,
                   static_cast<std::uint32_t>(d.rope_dims), d.base, d.yarn, pos);
      if (d.indexer) {
        const auto row = std::span(iq).subspan(r * d.ih * d.idim, d.ih * d.idim);
        RotateNormal(row, d.ih, d.idim, d.idim - d.rope_dims,
                     static_cast<std::uint32_t>(d.rope_dims), d.base, d.yarn, pos);
        HadamardRows(row, d.idim);
      }
      const std::uint64_t cell =
          d.cache == Cache::kFull ? seg.past + t : (seg.past + t) % window_cap;
      if (d.cache == Cache::kFull && cell >= window_cap) {
        return std::unexpected("mqa reference: a row past the cache");
      }
      if (d.kv_encoding == Encoding::kQ8_0) {
        const auto src = std::span(kv).subspan(r * d.d, d.d);
        auto dst = std::span(st.at(kWindowKv)).subspan(cell * d.d, d.d);
        if (auto stored = io.store_row(d.kv_encoding, src, dst); !stored) return stored;
      } else {
        for (std::uint64_t j = 0; j < d.d; ++j) {
          st.at(kWindowKv)[(cell * d.d) + j] = Stored(Encoding::kF16, kv[(r * d.d) + j]);
        }
      }
    }
    // The cells each row reads: its window's, then its compressed rows'.
    const std::uint64_t window_read =
        d.cache == Cache::kFull ? std::min<std::uint64_t>(window_cap, seg.read_cells) : window_cap;
    std::uint64_t compressed_read = 0;
    if (d.ratio != 0) {
      compressed_read = std::min<std::uint64_t>(
          io.state_cells.at(kCompressed),
          CompressedReadCells(end, d.ratio, State(*io.instance).tensors[kCompressed].cell_pad));
    }
    for (std::uint64_t t = 0; t < seg.rows; ++t) {
      const std::uint64_t r = first + t;
      const std::uint64_t pos = seg.past + t;
      std::vector<std::pair<const std::vector<double>*, std::uint64_t>> cells;
      for (std::uint64_t c = 0; c < window_read; ++c) {
        const std::int64_t held = Held(c, end, window_cap, d.cache != Cache::kFull);
        if (held < 0) continue;
        const auto h = static_cast<std::uint64_t>(held);
        const bool seen = d.cache == Cache::kInjected ? (h > pos || pos - h < d.window)
                                                      : (h <= pos && pos - h < d.window);
        if (seen) cells.emplace_back(&st.at(kWindowKv), c);
      }
      if (d.ratio != 0) {
        const std::uint64_t visible = std::min<std::uint64_t>((pos + 1) / d.ratio, compressed_read);
        std::vector<bool> keep(visible, true);
        if (d.indexer && visible > d.top_k) {
          // The indexer's best top_k of the visible rows (the lower first
          // among equals).
          std::vector<std::pair<double, std::uint64_t>> scored;
          const auto& keys = st.at(kIndex);
          for (std::uint64_t c = 0; c < visible; ++c) {
            double s = 0.0;
            for (std::uint64_t h = 0; h < d.ih; ++h) {
              double dot = 0.0;
              for (std::uint64_t j = 0; j < d.idim; ++j) {
                dot += iq[(r * d.ih * d.idim) + (h * d.idim) + j] * keys[(c * d.idim) + j];
              }
              s += iw[(r * d.ih) + h] * std::max(dot, 0.0);
            }
            scored.emplace_back(s, c);
          }
          std::ranges::stable_sort(scored,
                                   [](const auto& a, const auto& b) { return a.first > b.first; });
          std::ranges::fill(keep, false);
          for (std::uint64_t i = 0; i < d.top_k; ++i) keep[scored[i].second] = true;
        }
        for (std::uint64_t c = 0; c < visible; ++c) {
          if (keep[c]) cells.emplace_back(&st.at(kCompressed), c);
        }
      }
      for (std::uint64_t h = 0; h < d.heads; ++h) {
        const double* qh = &q[(r * d.heads * d.d) + (h * d.d)];
        const double sink = static_cast<double>(weight(kSinks)[h]);
        std::vector<double> scores;
        double top = sink;
        for (const auto& [cache, c] : cells) {
          double dot = 0.0;
          for (std::uint64_t j = 0; j < d.d; ++j) dot += qh[j] * (*cache)[(c * d.d) + j];
          scores.push_back(dot / std::sqrt(static_cast<double>(d.d)));
          top = std::max(top, scores.back());
        }
        double sum = std::exp(sink - top);
        for (double& sc : scores) {
          sc = std::exp(sc - top);
          sum += sc;
        }
        double* out = &attended[(r * d.heads * d.d) + (h * d.d)];
        for (std::size_t i = 0; i < cells.size(); ++i) {
          const auto& [cache, c] = cells[i];
          for (std::uint64_t j = 0; j < d.d; ++j)
            out[j] += scores[i] / sum * (*cache)[(c * d.d) + j];
        }
      }
      RotateNormal(std::span(attended).subspan(r * d.heads * d.d, d.heads * d.d), d.heads, d.d,
                   d.d - d.rope_dims, static_cast<std::uint32_t>(d.rope_dims), d.base, d.yarn,
                   static_cast<std::int64_t>(pos), true);
    }
    first += seg.rows;
  }
  // The grouped output product, then out_b.
  const std::uint64_t per = d.heads / d.groups * d.d;
  const std::span<const float> out_a = weight(kOutA);
  std::vector<double> low(rows * d.o_rank * d.groups, 0.0);
  for (std::uint64_t r = 0; r < rows; ++r) {
    for (std::uint64_t g = 0; g < d.groups; ++g) {
      const double* x = &attended[(r * d.heads * d.d) + (g * per)];
      for (std::uint64_t j = 0; j < d.o_rank; ++j) {
        const float* m = &out_a[((g * d.o_rank) + j) * per];
        double s = 0.0;
        for (std::uint64_t k = 0; k < per; ++k) s += static_cast<double>(m[k]) * x[k];
        low[(r * d.o_rank * d.groups) + (g * d.o_rank) + j] = s;
      }
    }
  }
  io.stream = Product(weight(kOutB), d.o_rank * d.groups, w, low);
  return {};
}

}  // namespace

// rope.cu's rope_yarn and rope_yarn_ramp, ggml.c's ggml_rope_yarn_corr_dims.
std::pair<double, double> YarnRotation(float base, std::uint32_t dims, std::uint64_t pair,
                                       std::int64_t position, const Yarn& y) {
  const float theta_scale = static_cast<float>(
      std::pow(static_cast<double>(base), static_cast<double>(-2.0F / static_cast<float>(dims))));
  const float power =
      static_cast<float>(std::pow(static_cast<double>(theta_scale), static_cast<double>(pair)));
  const float extrap = static_cast<float>(position) * power;
  if (!(y.factor > 1.0F)) return {extrap, 1.0};
  const float scale = 1.0F / y.factor;
  const float interp = scale * extrap;
  const auto corr = [&](float beta) {
    return static_cast<float>(dims) *
           std::log(static_cast<float>(y.context) / (beta * 2.0F * std::numbers::pi_v<float>)) /
           (2.0F * std::log(base));
  };
  const float lo = std::max(0.0F, std::floor(corr(y.beta_fast)));
  const float hi = std::min(static_cast<float>(dims - 1), std::ceil(corr(y.beta_slow)));
  const float t = (static_cast<float>(pair) - lo) / std::max(0.001F, hi - lo);
  const float ramp = 1.0F - std::min(1.0F, std::max(0.0F, t));
  const float theta = (interp * (1.0F - ramp)) + (extrap * ramp);
  const float attn = 1.0F / (1.0F + (0.1F * std::log(1.0F / scale)));
  const float magnitude = attn * (1.0F + (0.1F * std::log(1.0F / scale)));
  return {theta, magnitude};
}

void RotateNormal(std::span<double> row, std::uint64_t heads, std::uint64_t dim,
                  std::uint64_t offset, std::uint32_t dims, float base, const Yarn& yarn,
                  std::int64_t position, bool inverse) {
  for (std::uint64_t h = 0; h < heads; ++h) {
    double* x = &row[(h * dim) + offset];
    for (std::uint64_t i = 0; i < dims / 2; ++i) {
      const auto [theta, m] = YarnRotation(base, dims, i, position, yarn);
      const double c = std::cos(theta) * m;
      const double s = (inverse ? -std::sin(theta) : std::sin(theta)) * m;
      const double x0 = x[2 * i];
      const double x1 = x[(2 * i) + 1];
      x[2 * i] = (x0 * c) - (x1 * s);
      x[(2 * i) + 1] = (x0 * s) + (x1 * c);
    }
  }
}

void ShapeParameters(const Instance& in, std::uint32_t rows, std::uint32_t past,
                     std::uint32_t /*read_align*/, std::vector<std::uint64_t>& out) {
  const Dims d = DimsOf(in);
  if (d.ratio == 0) return;
  out.push_back(CompressorBlocks(rows, past, d.ratio, d.overlap));
  out.push_back(CompressedReadCells(std::uint64_t{past} + rows, d.ratio,
                                    State(in).tensors[kCompressed].cell_pad));
}

const ComponentDef& AttentionMqa() {
  static const ComponentDef def{
      .kind = "attention.mqa",
      .version = 1,
      .params = kParams,
      .roles = kRoles,
      .shapes = Shapes,
      .state = State,
      .batch_classes = kBatchClasses,
      .patterns = kPatterns,
      .masks = Masks,
      .emit = Emit,
      .reference = Reference,
      .inject = Inject,
      .shape_parameters = ShapeParameters,
  };
  return def;
}

}  // namespace llmp::components::defs

namespace llmp::components {

std::uint64_t CompressorBlocks(std::uint32_t rows, std::uint32_t past, std::uint32_t ratio,
                               bool overlap) {
  if (ratio == 0 || rows == 0) return 0;
  const std::uint64_t end = std::uint64_t{past} + rows;
  const std::uint64_t complete = (end / ratio) - (std::uint64_t{past} / ratio);
  if (overlap) return (std::uint64_t{rows} + ratio - 1) / ratio;
  return std::max<std::uint64_t>(complete, 1);
}

std::uint64_t CompressedReadCells(std::uint64_t end, std::uint32_t ratio, std::uint32_t cell_pad) {
  if (ratio == 0) return 0;
  const std::uint64_t align = std::max<std::uint32_t>(cell_pad, 1);
  const std::uint64_t blocks = end / ratio;
  return std::max(align, ((blocks + align - 1) / align) * align);
}

// llama.cpp's dsv4_build_comp_plan.
std::expected<CompressorPlan, std::string> PlanCompressor(std::uint32_t rows, std::uint32_t past,
                                                          std::uint32_t ratio, bool overlap,
                                                          std::uint64_t ring, std::uint64_t cache) {
  if (ratio == 0 || rows == 0 || ring < std::uint64_t{ratio} * (overlap ? 2 : 1) || cache == 0) {
    return std::unexpected<std::string>("not a compressor's geometry");
  }
  const std::uint64_t end = std::uint64_t{past} + rows;
  // A position's row of [ring | rows | zero row].
  const auto source = [&](std::int64_t pos) -> std::int64_t {
    if (pos < 0) return static_cast<std::int64_t>(ring + rows);
    const auto p = static_cast<std::uint64_t>(pos);
    if (p >= past && p < end) return static_cast<std::int64_t>(ring + (p - past));
    return static_cast<std::int64_t>(p % ring);
  };
  CompressorPlan plan;
  std::vector<std::int64_t> prev;
  std::vector<std::int64_t> cur;
  const auto read_block = [&](std::int64_t start, bool dummy) {
    for (std::uint32_t j = 0; j < ratio; ++j) {
      const std::int64_t own = dummy ? source(past) : source(start + j);
      if (overlap) {
        prev.push_back(dummy ? own : source(start - ratio + j));
        cur.push_back(own);
      } else {
        plan.reads.push_back(own);
      }
    }
  };
  for (std::uint64_t pos = past; pos < end; ++pos) {
    if ((pos + 1) % ratio != 0) continue;
    const std::uint64_t block = pos / ratio;
    if (block >= cache)
      return std::unexpected<std::string>(
          std::format("block {} past a {}-cell cache", block, cache));
    const auto start = static_cast<std::int64_t>(pos + 1 - ratio);
    plan.cells.push_back(static_cast<std::int64_t>(block));
    plan.starts.push_back(start);
    read_block(start, false);
  }
  const std::uint64_t want = CompressorBlocks(rows, past, ratio, overlap);
  if (plan.cells.size() + 1 == want) {
    // The dummy block: the cache's last cell, which no row sees.
    if (end / ratio >= cache)
      return std::unexpected<std::string>("the dummy block would overwrite a visible cell");
    plan.cells.push_back(static_cast<std::int64_t>(cache - 1));
    plan.starts.push_back(0);
    read_block(0, true);
  } else if (plan.cells.size() != want) {
    return std::unexpected<std::string>("a compressor's blocks are not its plan's");
  }
  if (overlap) {
    plan.reads = std::move(prev);
    plan.reads.insert(plan.reads.end(), cur.begin(), cur.end());
  }
  return plan;
}

}  // namespace llmp::components
