// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// attention.qsa, version 1: Qwen3.8's full-attention layers (llama.cpp's
// qwen4exp build_layer_attn with build_attn_qsa and build_qsa_top_k,
// kernels/ggml/qwen38_graph.cc's Attention and QsaTopK) inside hyper-
// connections: no norm or residual of its own. From the stream x:
//
//   - the indexer's keys (one fused projection of its query heads' rows
//     then the key's, or two: `indexer` fused or split), stored raw, F32, a
//     cell per position;
//   - the query (each head's q then its output gate, one product), key and
//     value products; q and k each head RMS-normed with their weights and
//     rotated (the interleaved multi-section rotation, every section the
//     position), K and V stored F16, a cell per position;
//   - attention over the read cells under the causal mask at scale
//     1 / sqrt(head_dim), F32 sums; past the indexer's budget (when the
//     read cells pass budget + ratio - 1) over the cells the indexer
//     selects alone (below);
//   - each head's output times sigmoid(its gate), and the output product.
//
// The selection (build_qsa_top_k at llama.cpp b29c606e2, with the fast
// form's rounding: llmp.qsa.pool, .topk): every block of `ratio` read
// cells pooled (the mean of its raw keys), RMS-normed with the indexer's
// key weight and rotated at the block's first position, rounded to BF16
// (a block key); the indexer's query heads normed and rotated as q and
// rounded to BF16; each block's score Σ_h relu(q_h · key) (ir::
// kIndexScore), plus a bias that keeps the token's own block: 0 for a
// block whose last position the token sees, 1e9 otherwise (set_input_qsa's
// table, whose later blocks the causal mask then hides); the score spread
// over the block's cells, the causal mask added, and the `width` = min(read
// cells, budget + ratio - 1) best cells kept (ties to the lower cell). The
// attention's mask is -inf but at the kept cells, plus the causal mask; it
// promises at most `width` finite cells a row (AttnParams::n_kv_max,
// sparse_any), which a sparse attention takes (llmp.qsa.attn).
//
// The block keys are a cache an implementation may keep (llmp.qsa.pool
// writes each once, when its block completes, and the selection reads
// them): they are a function of the raw keys and their positions, so the
// state contract holds them (indexer_blocks, BF16, a cell per block of the
// read cells) for the implementations that keep them, and the primitive
// fragment, which pools every block of the read cells each step, never
// writes or reads them.
//
// The fragment emits what the unfused form records, in GGML's graph order
// (the indexer's key product and store, the query's product, norm and
// rotation, V's, K's, the stores, attention, the gate, the output
// product); its fusions (llmp.qsa.prep, .pool, .topk, .attn and the gate's
// quantization) are the planner's patterns (G3b).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <expected>
#include <limits>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "base/check.h"
#include "components/attention_core.h"
#include "components/component.h"
#include "components/defs.h"
#include "components/rotation_reference.h"
#include "components/step.h"
#include "ir/graph.h"
#include "ir/op.h"
#include "model/state_contract.h"

namespace llmp::components::defs {
namespace {

using format::Encoding;

constexpr std::array<std::string_view, 2> kIndexers{"fused", "split"};
constexpr std::array<std::string_view, 2> kRollbacks{"snapshot", "truncate"};
constexpr std::array<std::string_view, 2> kKvEncodings{"f16", "q8_0"};

constexpr std::array kParams{
    kNormZeroCentered,
    ParamSpec{.name = "eps", .type = ParamType::kReal, .least = 0.0, .most = 1.0},
    ParamSpec{.name = "heads", .type = ParamType::kInt, .least = 1, .most = 1024},
    ParamSpec{.name = "kv_heads", .type = ParamType::kInt, .least = 1, .most = 1024},
    ParamSpec{.name = "head_dim", .type = ParamType::kInt, .least = 2, .most = 1024},
    ParamSpec{.name = "rope_dims", .type = ParamType::kInt, .least = 2, .most = 1024},
    ParamSpec{.name = "rope_base", .type = ParamType::kReal, .least = 1e-30, .most = 1e12},
    // The rotation's four sections (pairs a section).
    ParamSpec{.name = "rope_sections", .type = ParamType::kInts, .least = 0, .most = 1024},
    ParamSpec{.name = "indexer_heads", .type = ParamType::kInt, .least = 1, .most = 64},
    ParamSpec{.name = "indexer_dim", .type = ParamType::kInt, .least = 2, .most = 1024},
    // Positions a block pools.
    ParamSpec{.name = "ratio", .type = ParamType::kInt, .least = 1, .most = 1024},
    // Positions attention keeps past the budget (whole blocks, plus the tail).
    ParamSpec{.name = "budget", .type = ParamType::kInt, .least = 1, .most = 1 << 24},
    ParamSpec{.name = "indexer", .type = ParamType::kName, .choices = kIndexers},
    // How a verify's rejected rows are undone: saved (a target's), or
    // rewritten before they are read again (a drafter's own caches).
    ParamSpec{
        .name = "rollback", .type = ParamType::kName, .required = false, .choices = kRollbacks},
    ParamSpec{.name = "kv_encoding",
              .type = ParamType::kName,
              .required = false,
              .choices = kKvEncodings},
    ParamSpec{
        .name = "replayable", .type = ParamType::kBool, .required = false, .least = 0, .most = 1},
};

enum : std::uint16_t {
  kQ,
  kK,
  kV,
  kOutput,
  kQNorm,
  kKNorm,
  kIndexerQk,
  kIndexerQ,
  kIndexerK,
  kIndexerQNorm,
  kIndexerKNorm,
};
constexpr std::array kRoles{
    WeightRole{.name = "q"},
    WeightRole{.name = "k"},
    WeightRole{.name = "v"},
    WeightRole{.name = "output"},
    WeightRole{.name = "q_norm", .encodings = kNormEncodings, .derived = kZeroCenteredNorm},
    WeightRole{.name = "k_norm", .encodings = kNormEncodings, .derived = kZeroCenteredNorm},
    WeightRole{.name = "indexer_qk"},
    WeightRole{.name = "indexer_q"},
    WeightRole{.name = "indexer_k"},
    WeightRole{.name = "indexer_q_norm", .encodings = kNormEncodings, .derived = kZeroCenteredNorm},
    WeightRole{.name = "indexer_k_norm", .encodings = kNormEncodings, .derived = kZeroCenteredNorm},
};

// The state's tensors, in its contract's order.
enum : std::uint16_t { kCacheK, kCacheV, kCacheIndex, kCacheBlocks };

// The bias that keeps a token's own block in its selection (set_input_qsa).
constexpr float kKeep = 1e9F;

struct Dims {
  std::uint64_t width, heads, kvh, d, rope_dims, ih, idim, ratio, budget;
  std::array<std::uint32_t, 4> sections;
  bool fused, snapshot, replayable;
  float eps, base;
  std::uint32_t context;
  Encoding kv_encoding;
};
Dims DimsOf(const Instance& in) {
  const Params& p = in.params;
  Dims d{.width = in.stage.width,
         .heads = static_cast<std::uint64_t>(p.Int("heads")),
         .kvh = static_cast<std::uint64_t>(p.Int("kv_heads")),
         .d = static_cast<std::uint64_t>(p.Int("head_dim")),
         .rope_dims = static_cast<std::uint64_t>(p.Int("rope_dims")),
         .ih = static_cast<std::uint64_t>(p.Int("indexer_heads")),
         .idim = static_cast<std::uint64_t>(p.Int("indexer_dim")),
         .ratio = static_cast<std::uint64_t>(p.Int("ratio")),
         .budget = static_cast<std::uint64_t>(p.Int("budget")),
         .sections = {},
         .fused = p.Name("indexer") == "fused",
         .snapshot = p.Name("rollback") == "snapshot",
         .replayable = p.Flag("replayable"),
         .eps = F32(p.Real("eps")),
         .base = F32(p.Real("rope_base")),
         .context = in.stage.context,
         .kv_encoding = p.Name("kv_encoding") == "q8_0" ? Encoding::kQ8_0 : Encoding::kF16};
  const auto s = p.Ints("rope_sections");
  for (std::size_t i = 0; i < s.size() && i < 4; ++i)
    d.sections.at(i) = static_cast<std::uint32_t>(s[i]);
  return d;
}

std::expected<std::vector<RoleShape>, std::string> Shapes(const Instance& in) {
  if (NormOffsetContradictsFold(in))
    return std::unexpected("zero-centered norms contradict the norm.plus_one fold");
  const Dims d = DimsOf(in);
  if (d.kv_encoding == Encoding::kQ8_0 && (d.d != 64 && d.d != 256 && d.d != 512)) {
    return std::unexpected("Q8_0 KV requires head_dim 64, 256, or 512");
  }
  if (d.heads % d.kvh != 0) return std::unexpected("heads are not a multiple of kv_heads");
  if (in.params.Ints("rope_sections").size() != 4) return std::unexpected("four rope sections");
  if (d.rope_dims % 2 != 0 || d.rope_dims > d.d || d.rope_dims > d.idim) {
    return std::unexpected("rope_dims is odd or past a head");
  }
  const std::uint64_t w = d.width;
  std::vector<RoleShape> out{
      Role(kQ, {w, 2 * d.heads * d.d}),  Role(kK, {w, d.kvh * d.d}),   Role(kV, {w, d.kvh * d.d}),
      Role(kOutput, {d.heads * d.d, w}), Role(kQNorm, {d.d}),          Role(kKNorm, {d.d}),
      Role(kIndexerQNorm, {d.idim}),     Role(kIndexerKNorm, {d.idim})};
  if (d.fused) {
    out.push_back(Role(kIndexerQk, {w, (d.ih + 1) * d.idim}));
  } else {
    out.push_back(Role(kIndexerQ, {w, d.ih * d.idim}));
    out.push_back(Role(kIndexerK, {w, d.idim}));
  }
  return out;
}

model::StateContract State(const Instance& in) {
  const Dims d = DimsOf(in);
  const model::Rollback rollback =
      d.snapshot ? model::Rollback::kSnapshot : model::Rollback::kTruncate;
  const auto kv = model::CellBytes(d.kv_encoding, d.kvh * d.d);
  base::Check(kv.has_value(), "a cell of any width");
  return {.tensors = {{.name = "k",
                       .kind = model::StateKind::kKv,
                       .encoding = d.kv_encoding,
                       .cell_bytes = *kv,
                       .rollback = rollback,
                       .replayable = d.replayable},
                      {.name = "v",
                       .kind = model::StateKind::kKv,
                       .encoding = d.kv_encoding,
                       .cell_bytes = *kv,
                       .rollback = rollback,
                       .replayable = d.replayable},
                      {.name = "indexer_k",
                       .kind = model::StateKind::kIndex,
                       .encoding = Encoding::kF32,
                       .cell_bytes = d.idim * 4,
                       .rollback = rollback,
                       .replayable = d.replayable},
                      {.name = "indexer_blocks",
                       .kind = model::StateKind::kBlockKeys,
                       .encoding = Encoding::kBF16,
                       .cell_bytes = d.idim * 2,
                       .cells = model::CellRule::kPerBlock,
                       .block_ratio = static_cast<std::uint32_t>(d.ratio),
                       .basis = model::BlockBasis::kReadCells,
                       .rollback = rollback,
                       .replayable = d.replayable}}};
}

std::vector<MaskUse> Masks(const Instance& in) {
  const Dims d = DimsOf(in);
  return {{.tensor = kCacheK,
           .window = 0,
           .selection_bytes = 4,
           .selection_above = static_cast<std::uint32_t>(d.budget + d.ratio - 1)}};
}

// By KV heads, as attention.dense: Q (each head with its gate), K, V by
// their rows, the output product by its columns, K and V by their heads;
// the indexer and its caches whole on every device (the selection is the
// same on each).
constexpr std::array kHeadRoles{RoleSplit{.role = kQ, .dim = 1}, RoleSplit{.role = kK, .dim = 1},
                                RoleSplit{.role = kV, .dim = 1},
                                RoleSplit{.role = kOutput, .dim = 0}};
constexpr std::array kHeadState{StateSplit{.tensor = kCacheK, .dim = 0},
                                StateSplit{.tensor = kCacheV, .dim = 0}};
constexpr std::array kSplits{SplitOption{
    .kind = SplitKind::kHeads,
    .collective = ir::Op::kAllReduce,
    .units =
        [](const Instance& in) { return static_cast<std::uint32_t>(in.params.Int("kv_heads")); },
    .bytes_per_step = [](const SplitStep& s) -> std::uint64_t {
      return s.instance->stage.width * s.rows * 4;
    },
    .weights = kHeadRoles,
    .state = kHeadState}};

ir::RopeParams Rotation(const Dims& d) {
  return MakeRopeParams(ir::RopeMode::kInterleavedMulti, static_cast<std::uint32_t>(d.rope_dims), 0,
                        d.context, d.base, 1.0F, {}, d.sections);
}

// Whether the read cells pass the indexer's budget (plus the tail block).
bool Selects(const Dims& d, std::uint64_t read) { return read > d.budget + d.ratio - 1; }

std::expected<FragmentOut, std::string> Emit(Fragment& f) {
  ir::Builder& b = f.b;
  const Dims d = DimsOf(f.instance);
  F32Parameters<kRoles.size()> parameters(f);
  const auto w = [&](std::uint16_t role) {
    return role == kQNorm || role == kKNorm || role == kIndexerQNorm || role == kIndexerKNorm
               ? parameters.Norm(role)
               : f.Weight(role);
  };
  const ir::TensorId x = f.stream;
  const std::uint64_t rows = f.step.rows();
  const auto segments = f.step.segments();
  const bool one = segments.size() == 1;
  const ir::RopeParams rope = Rotation(d);
  const ir::TensorId positions = f.step.Positions(4);
  const CellKey kv_cells{.capacity = Desc(b, f.State(0, kCacheK)).ne(1)};
  const CellKey index_cells{.capacity = Desc(b, f.State(0, kCacheIndex)).ne(1)};
  // A segment's rows of a [width, rows] tensor: the tensor itself alone.
  const auto rows_of = [&](ir::TensorId t, std::size_t s) {
    if (one) return t;
    const ir::TensorDesc td = Desc(b, t);
    const format::Shape shape = Shape({td.ne(0), segments[s].rows});
    return b.View(t, shape, ViewStrides(td.nb[0], td.nb[1], 0, shape),
                  std::uint64_t{segments[s].first_row} * td.nb[1]);
  };
  // The indexer's raw keys, stored.
  ir::TensorId qk{};
  ir::TensorId raw{};
  if (d.fused) {
    qk = b.MatMul(w(kIndexerQk), x);
    const ir::TensorDesc qt = Desc(b, qk);
    const format::Shape shape = Shape({d.idim, rows});
    raw = b.Cont(b.View(qk, shape, ViewStrides(qt.nb[0], qt.nb[1], 0, shape), d.ih * d.idim * 4));
  } else {
    raw = b.MatMul(w(kIndexerK), x);
  }
  // Each owner's raw keys stored, then the block keys of its read cells'
  // blocks (the indexer_blocks cache): every block's raw keys pooled,
  // normed, rotated at the block's position and rounded, stored at its
  // cell. A block's key is a function of its rows alone, so the store
  // rewrites what earlier steps stored for the blocks they completed, and
  // an implementation may write only the blocks this step completes
  // (llmp.qsa.pool). The selection reads the keys from the cache.
  std::vector<ir::TensorId> block_keys(segments.size());
  for (std::size_t s = 0; s < segments.size(); ++s) {
    b.SetOwner(static_cast<std::uint8_t>(s));
    const ir::TensorId mine = rows_of(raw, s);
    const ir::TensorId cells = f.step.StoreCells(s, index_cells);
    const ir::TensorId stored = b.CacheStore(f.State(s, kCacheIndex), mine, cells);
    const ir::TensorId blocks_state = f.State(s, kCacheBlocks);
    const CellKey block_cells{.capacity = Desc(b, blocks_state).ne(1),
                              .block = static_cast<std::uint32_t>(d.ratio)};
    const std::uint64_t blocks = f.step.ReadCells(s, block_cells);
    if (blocks * d.ratio > index_cells.capacity) {
      return std::unexpected("block keys of cells past the indexer's cache");
    }
    const format::Shape cells_shape = Shape({d.idim, blocks * d.ratio});
    const ir::TensorId read =
        b.View(stored, cells_shape, ViewStrides(4, d.idim * 4, 0, cells_shape), 0);
    ir::TensorId keys = b.Pool(
        read, {}, {.ratio = static_cast<std::uint32_t>(d.ratio), .kind = ir::PoolKind::kMean});
    keys = EmitHeadNorm(b, keys, d.eps, w(kIndexerKNorm));
    const ir::TensorId heads = b.Reshape(keys, Shape({d.idim, 1, blocks}));
    const ir::TensorId block_positions = f.step.BlockPositions(s, block_cells, blocks, 4);
    keys = b.Rope(heads, block_positions, {}, rope);
    const ir::TensorId rounded = b.Convert(keys, Encoding::kBF16);
    keys = b.Reshape(rounded, Shape({d.idim, blocks}));
    const ir::TensorId block_cell = f.step.BlockCells(s, block_cells, blocks);
    block_keys[s] = b.CacheStore(blocks_state, keys, block_cell);
    b.SetJoined();
  }
  // Q (each head's q, then its gate), V, K.
  ir::TensorId q_full{};
  ir::TensorId q{};
  if (f.stream_needed) {
    q_full = b.MatMul(w(kQ), x);
    q = EmitInterleavedHeadSlice(b, q_full, d.d, 2 * d.d, d.heads, rows, 0);
    q = EmitHeadNormRope(b, q, d.eps, w(kQNorm), positions, {}, rope);
  }
  const ir::TensorId v = b.Reshape(b.MatMul(w(kV), x), Shape({d.d, d.kvh, rows}));
  ir::TensorId k = b.Reshape(b.MatMul(w(kK), x), Shape({d.d, d.kvh, rows}));
  k = EmitHeadNormRope(b, k, d.eps, w(kKNorm), positions, {}, rope);
  const std::uint64_t kv = d.d * d.kvh;
  const ir::TensorId k_rows = b.Reshape(k, Shape({kv, rows}));
  for (std::size_t s = 0; s < segments.size(); ++s) {
    b.SetOwner(static_cast<std::uint8_t>(s));
    const ir::TensorId seg_k = rows_of(k_rows, s);
    EmitKvCacheStore(b, f.State(s, kCacheK), seg_k, f.step.StoreCells(s, kv_cells), Encoding::kF16);
    b.SetJoined();
  }
  const ir::TensorId v_rows = b.Reshape(v, Shape({kv, rows}));
  for (std::size_t s = 0; s < segments.size(); ++s) {
    b.SetOwner(static_cast<std::uint8_t>(s));
    const ir::TensorId seg_v = rows_of(v_rows, s);
    EmitKvCacheStore(b, f.State(s, kCacheV), seg_v, f.step.StoreCells(s, kv_cells), d.kv_encoding);
    b.SetJoined();
  }
  if (!f.stream_needed) return FragmentOut{.stream = x};

  // The indexer's queries, made once for the owners whose reads select.
  ir::TensorId index_q{};
  const auto indexer_queries = [&] {
    if (index_q.valid()) return index_q;
    ir::TensorId heads{};
    if (d.fused) {
      const ir::TensorDesc qt = Desc(b, qk);
      heads = b.Cont(
          b.View(qk, Shape({d.idim, d.ih, rows}), {4, d.idim * 4, qt.nb[1], qt.nb[1] * rows}, 0));
    } else {
      heads = b.Reshape(b.MatMul(w(kIndexerQ), x), Shape({d.idim, d.ih, rows}));
    }
    heads = EmitHeadNormRope(b, heads, d.eps, w(kIndexerQNorm), positions, {}, rope);
    index_q = b.Convert(heads, Encoding::kBF16);
    return index_q;
  };

  const float scale = 1.0F / std::sqrt(static_cast<float>(d.d));
  std::vector<ir::TensorId> outputs;
  for (std::size_t s = 0; s < segments.size(); ++s) {
    const SegmentRows& seg = segments[s];
    const std::uint64_t read = f.step.ReadCells(s, kv_cells);
    auto mask = f.step.Mask(s, kv_cells, 0, false, true);
    if (!mask) return std::unexpected(mask.error());
    ir::AttnParams params{.scale = scale, .accumulate = ir::Accumulate::kF32};
    ir::TensorId attn_mask = *mask;
    if (Selects(d, read)) {
      if (read % d.ratio != 0) return std::unexpected("read cells that are not whole blocks");
      const std::uint64_t blocks = read / d.ratio;
      const std::uint64_t width = std::min<std::uint64_t>(read, d.budget + d.ratio - 1);
      const ir::TensorId queries = indexer_queries();
      b.SetOwner(static_cast<std::uint8_t>(s));
      // Every block of the read cells' key, from the cache (stored above).
      const CellKey block_cells{.capacity = Desc(b, f.State(s, kCacheBlocks)).ne(1),
                                .block = static_cast<std::uint32_t>(d.ratio)};
      if (blocks != f.step.ReadCells(s, block_cells)) {
        return std::unexpected("a selection over other blocks than the cache's");
      }
      const format::Shape keys_shape = Shape({d.idim, 1, blocks, 1});
      const ir::TensorId keys =
          b.View(block_keys[s], keys_shape, ViewStrides(2, d.idim * 2, d.idim * 2, keys_shape), 0);
      // Its rows' queries.
      const ir::TensorDesc qt = Desc(b, queries);
      const format::Shape q_shape = Shape({d.idim, d.ih, seg.rows, 1});
      const ir::TensorId mine =
          one ? queries
              : b.View(queries, q_shape, {qt.nb[0], qt.nb[1], qt.nb[2], qt.nb[2] * seg.rows},
                       std::uint64_t{seg.first_row} * qt.nb[2]);
      // The bias: 0 where the block's last position is seen, else kKeep.
      const ir::TensorId at = f.step.SegmentPositions(s);
      const ir::TensorId held = f.step.HeldPositions(s, block_cells);
      const ir::TensorId seen =
          b.MaskBuild(at, held, seg.rows, {.kind = ir::MaskKind::kCausal}, Encoding::kF32);
      const ir::TensorId bias = b.Scale(b.Clamp(seen, -kKeep, 0.0F), -1.0F);
      const ir::TensorId ones = b.Fill(Shape({d.ih, seg.rows, 1, 1}), 1.0F);
      const ir::TensorId bias_rows = b.Reshape(bias, Shape({blocks, seg.rows, 1, 1}));
      ir::TensorId score = b.IndexScore(mine, keys, ones, bias_rows);
      // Spread over the blocks' cells, the causal mask added; the best kept.
      score = b.Repeat(b.Reshape(score, Shape({1, blocks, seg.rows})),
                       Shape({d.ratio, blocks, seg.rows}));
      const ir::TensorId cells_held = f.step.HeldPositions(s, kv_cells);
      const ir::TensorId causal =
          b.MaskBuild(at, cells_held, seg.rows, {.kind = ir::MaskKind::kCausal}, Encoding::kF32);
      score = b.Add(b.Reshape(score, Shape({read, seg.rows})), causal);
      const ir::TensorId kept = b.TopK(score, static_cast<std::uint32_t>(width));
      // The attention's mask: -inf but at the kept cells, then causal.
      ir::TensorId selected =
          b.Fill(Shape({1, read, seg.rows, 1}), -std::numeric_limits<float>::infinity());
      const ir::TensorId zeros = b.Fill(Shape({1, width, seg.rows, 1}), 0.0F);
      selected = b.CacheStore(selected, zeros, kept);
      const ir::TensorDesc mt = Desc(b, *mask);
      const format::Shape rows_shape = Shape({read, seg.rows});
      selected = b.Reshape(selected, rows_shape);
      const ir::TensorId causal_rows =
          b.View(*mask, rows_shape, ViewStrides(mt.nb[0], mt.nb[1], 0, rows_shape), 0);
      attn_mask = b.Add(selected, causal_rows);
      params.n_kv_max = static_cast<std::uint32_t>(width);
      params.sparse_any = true;
      b.SetJoined();
    }
    b.SetOwner(static_cast<std::uint8_t>(s));
    const ir::TensorId queries = EmitSegmentQueries(b, q, d.d, d.heads, seg, /*elide_single=*/one);
    const ir::TensorId keys_view =
        EmitKvCacheView(b, f.State(s, kCacheK), d.d, d.kvh, read, /*direct_3d_f16=*/true);
    const ir::TensorId values_view =
        EmitKvCacheView(b, f.State(s, kCacheV), d.d, d.kvh, read, /*direct_3d_f16=*/true);
    outputs.push_back(EmitSegmentAttention(b, queries, keys_view, values_view, attn_mask, {},
                                           params, d.d, d.heads, seg.rows));
    b.SetJoined();
  }
  const ir::TensorId joined = JoinSegmentOutputs(b, outputs);
  // Each head's output times sigmoid(its gate).
  const ir::TensorId gated = EmitSigmoidOutputGate(b, joined, q_full, d.d, d.d, d.heads, rows);
  return FragmentOut{.stream = b.MatMul(w(kOutput), gated)};
}

// ------------------------------------------------------------- reference

double Bf16(double x) {
  // The nearest BF16 (ties to even) of the F32 value.
  const float f = static_cast<float>(x);
  std::uint32_t u = 0;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7F800000U) != 0x7F800000U) {
    u += 0x7FFFU + ((u >> 16U) & 1U);
  }
  u &= 0xFFFF0000U;
  float r = 0.0F;
  std::memcpy(&r, &u, 4);
  return static_cast<double>(r);
}

std::expected<void, std::string> Reference(RefIO& io) {
  const Instance& in = *io.instance;
  const Dims d = DimsOf(in);
  if (d.kv_encoding == Encoding::kQ8_0 && io.store_row == nullptr) {
    return std::unexpected("Q8_0 reference state requires the block row codec");
  }
  const std::uint64_t w = d.width;
  const std::uint64_t rows = io.rows();
  if (io.stream.size() != w * rows || io.state.size() != io.segments.size() ||
      io.state_cells.size() != 4) {
    return std::unexpected("qsa reference: a stream or state of another shape");
  }
  std::array<std::vector<float>, kRoles.size()> adjusted;
  if (in.params.Flag("norm_zero_centered")) {
    for (const auto role : {kQNorm, kKNorm, kIndexerQNorm, kIndexerKNorm}) {
      adjusted[role] = io.weights.at(role);
      for (float& value : adjusted[role]) value += 1.0F;
    }
  }
  const auto weight = [&](std::uint16_t role) {
    return std::span<const float>(adjusted[role].empty() ? io.weights.at(role) : adjusted[role]);
  };
  std::vector<double> raw;
  std::vector<double> iq;
  if (d.fused) {
    const std::vector<double> both = Product(weight(kIndexerQk), w, (d.ih + 1) * d.idim, io.stream);
    for (std::uint64_t r = 0; r < rows; ++r) {
      const auto row = std::span(both).subspan(r * (d.ih + 1) * d.idim, (d.ih + 1) * d.idim);
      iq.insert(iq.end(), row.begin(), row.begin() + static_cast<std::ptrdiff_t>(d.ih * d.idim));
      raw.insert(raw.end(), row.begin() + static_cast<std::ptrdiff_t>(d.ih * d.idim), row.end());
    }
  } else {
    iq = Product(weight(kIndexerQ), w, d.ih * d.idim, io.stream);
    raw = Product(weight(kIndexerK), w, d.idim, io.stream);
  }
  const std::vector<double> qg = Product(weight(kQ), w, 2 * d.heads * d.d, io.stream);
  std::vector<double> k = Product(weight(kK), w, d.kvh * d.d, io.stream);
  const std::vector<double> v = Product(weight(kV), w, d.kvh * d.d, io.stream);
  std::vector<double> q(rows * d.heads * d.d);
  for (std::uint64_t r = 0; r < rows; ++r) {
    for (std::uint64_t h = 0; h < d.heads; ++h) {
      for (std::uint64_t i = 0; i < d.d; ++i) {
        q[(((r * d.heads) + h) * d.d) + i] = qg[(r * 2 * d.heads * d.d) + (h * 2 * d.d) + i];
      }
    }
  }
  RmsRows(q, d.d, d.eps, weight(kQNorm));
  RmsRows(k, d.d, d.eps, weight(kKNorm));
  // The indexer's queries rounded to F32 at each of the graph's nodes (its
  // selection's BF16 rounding then sees the graph's values).
  RmsRows(iq, d.idim, d.eps, {});
  for (double& x : iq) x = static_cast<double>(static_cast<float>(x));
  for (std::size_t j = 0; j < iq.size(); ++j) {
    iq[j] = static_cast<double>(
        static_cast<float>(iq[j] * static_cast<double>(weight(kIndexerQNorm)[j % d.idim])));
  }
  const ir::RopeParams rope = Rotation(d);
  const std::uint64_t kv = d.kvh * d.d;
  std::vector<double> attended(rows * d.heads * d.d, 0.0);
  std::uint64_t first = 0;
  for (std::size_t si = 0; si < io.segments.size(); ++si) {
    const RefSegment& seg = io.segments[si];
    auto& kc = io.state[si].at(kCacheK);
    auto& vc = io.state[si].at(kCacheV);
    auto& ic = io.state[si].at(kCacheIndex);
    const std::uint64_t capacity = io.state_cells.at(kCacheK);
    for (std::uint64_t t = 0; t < seg.rows; ++t) {
      const std::uint64_t r = first + t;
      const auto pos = static_cast<std::int64_t>(seg.past + t);
      RotateHeads(std::span(q).subspan(r * d.heads * d.d, d.heads * d.d), d.heads, d.d, rope, pos);
      RotateHeads(std::span(k).subspan(r * kv, kv), d.kvh, d.d, rope, pos);
      RotateHeads(std::span(iq).subspan(r * d.ih * d.idim, d.ih * d.idim), d.ih, d.idim, rope, pos);
      const std::uint64_t cell = seg.past + t;
      if (cell >= capacity) return std::unexpected("qsa reference: a row past the cache");
      if (auto stored = StoreKvCacheRow(kc, cell, k, r, d.kvh, d.d, d.kv_encoding, io.store_row);
          !stored) {
        return stored;
      }
      if (auto stored = StoreKvCacheRow(vc, cell, v, r, d.kvh, d.d, d.kv_encoding, io.store_row);
          !stored) {
        return stored;
      }
      for (std::uint64_t j = 0; j < d.idim; ++j) {
        ic[(cell * d.idim) + j] = Stored(Encoding::kF32, raw[(r * d.idim) + j]);
      }
    }
    const std::uint64_t read = std::min<std::uint64_t>(capacity, seg.read_cells);
    const std::uint64_t end = seg.past + seg.rows;
    // The block keys of every block of the read cells, stored in the cache
    // (each step rounded to F32 as the graph's nodes round their results,
    // so the BF16 keys are the graph's).
    auto& bc = io.state[si].at(kCacheBlocks);
    const std::uint64_t pooled = std::min<std::uint64_t>(io.state_cells.at(kCacheBlocks),
                                                         (seg.read_cells + d.ratio - 1) / d.ratio);
    if (pooled * d.ratio > io.state_cells.at(kCacheIndex) || bc.size() < pooled * d.idim) {
      return std::unexpected("qsa reference: blocks past the caches");
    }
    std::vector<double> keys(pooled * d.idim, 0.0);
    {
      const auto f32 = [](std::vector<double>& v) {
        for (double& x : v) x = static_cast<double>(static_cast<float>(x));
      };
      for (std::uint64_t bl = 0; bl < pooled; ++bl) {
        double* key = &keys[bl * d.idim];
        for (std::uint64_t c = 0; c < d.ratio; ++c) {
          for (std::uint64_t j = 0; j < d.idim; ++j)
            key[j] += ic[(((bl * d.ratio) + c) * d.idim) + j];
        }
        for (std::uint64_t j = 0; j < d.idim; ++j) key[j] /= static_cast<double>(d.ratio);
      }
      f32(keys);
      RmsRows(keys, d.idim, d.eps, {});
      f32(keys);
      for (std::size_t j = 0; j < keys.size(); ++j) {
        keys[j] *= static_cast<double>(weight(kIndexerKNorm)[j % d.idim]);
      }
      f32(keys);
      for (std::uint64_t bl = 0; bl < pooled; ++bl) {
        RotateHeads(std::span(keys).subspan(bl * d.idim, d.idim), 1, d.idim, rope,
                    static_cast<std::int64_t>(bl * d.ratio));
      }
      f32(keys);
      for (std::size_t j = 0; j < keys.size(); ++j) {
        keys[j] = Bf16(keys[j]);
        bc[j] = keys[j];
      }
    }
    // The selection, as set_input_qsa's tables and build_qsa_top_k give it.
    const bool selects = Selects(d, read);
    std::vector<std::vector<bool>> kept(seg.rows, std::vector<bool>(read, true));
    if (selects) {
      const std::uint64_t blocks = read / d.ratio;
      const std::uint64_t width = std::min<std::uint64_t>(read, d.budget + d.ratio - 1);
      if (blocks > pooled) return std::unexpected("qsa reference: a selection past the blocks");
      for (std::uint64_t t = 0; t < seg.rows; ++t) {
        const std::uint64_t r = first + t;
        const std::uint64_t pos = seg.past + t;
        std::vector<double> qb(d.ih * d.idim);
        for (std::size_t j = 0; j < qb.size(); ++j) qb[j] = Bf16(iq[(r * d.ih * d.idim) + j]);
        // Cell scores: the block's, its bias, the causal mask.
        std::vector<std::pair<double, std::uint64_t>> cells;
        for (std::uint64_t c = 0; c < read; ++c) {
          if (c > pos || c >= end) continue;  // never visible
          const std::uint64_t bl = c / d.ratio;
          double score = 0.0;
          for (std::uint64_t h = 0; h < d.ih; ++h) {
            double dot = 0.0;
            for (std::uint64_t j = 0; j < d.idim; ++j)
              dot += qb[(h * d.idim) + j] * keys[(bl * d.idim) + j];
            score += std::max(dot, 0.0);
          }
          const bool complete = (bl * d.ratio) + d.ratio - 1 <= pos;
          score = static_cast<double>(static_cast<float>(score) + (complete ? 0.0F : kKeep));
          cells.emplace_back(score, c);
        }
        // The best `width` visible cells, the lower cell first among equals.
        std::ranges::stable_sort(cells,
                                 [](const auto& a, const auto& b) { return a.first > b.first; });
        std::vector<bool> keep(read, false);
        for (std::size_t i = 0; i < cells.size() && i < width; ++i) keep[cells[i].second] = true;
        kept[t] = std::move(keep);
      }
    }
    const double score_scale = 1.0 / std::sqrt(static_cast<double>(d.d));
    std::vector<std::uint8_t> visible(read);
    for (std::uint64_t t = 0; t < seg.rows; ++t) {
      const std::uint64_t r = first + t;
      const std::uint64_t pos = seg.past + t;
      for (std::uint64_t c = 0; c < read; ++c) {
        visible[c] = c <= pos && c < end && kept[t][c] ? 1U : 0U;
      }
      AttendCacheRow(std::span(q).subspan(r * d.heads * d.d, d.heads * d.d), kc, vc, read, visible,
                     d.heads, d.kvh, d.d, d.d, 1.0, score_scale, 0.0, {},
                     std::span(attended).subspan(r * d.heads * d.d, d.heads * d.d));
    }
    first += seg.rows;
  }
  ApplySigmoidOutputGateRef(attended, qg, rows, d.heads, d.d, d.d);
  io.stream = Product(weight(kOutput), d.heads * d.d, w, attended);
  return {};
}

}  // namespace

const ComponentDef& AttentionQsa() {
  static const ComponentDef def{
      .kind = "attention.qsa",
      .version = 1,
      .params = kParams,
      .roles = kRoles,
      .shapes = Shapes,
      .state = State,
      .max_state_tensors = 4,

      .splits = kSplits,

      .masks = Masks,
      .exact_masks = true,
      .emit = Emit,
      .reference = Reference,
  };
  return def;
}

}  // namespace llmp::components::defs
