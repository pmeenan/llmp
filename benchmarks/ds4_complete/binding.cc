// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "benchmarks/ds4_complete/binding.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "engine/dsv4_ds4_paged_weights.h"
#include "engine/dsv4_ds4_weights.h"
#include "engine/paged_weights.h"

namespace jitllm::benchmarks::ds4_complete {
namespace {
using Buffer = kg::Ds4CacheBuffer;
using StateKind = model::Ds4BaselineStateKind;
constexpr std::uint32_t kContext = 8192;
constexpr std::uint64_t kAlignment = 256;

bool SameProfile(const model::Dsv4Profile& p) {
  const auto& expected = model::Dsv4Flash();
  return p.layers == 43 && p.width == 4096 && p.hc == 4 && p.heads == 64 && p.head_dim == 512 &&
         p.q_lora == 1024 && p.o_lora == 1024 && p.o_groups == 8 && p.rope_dims == 64 &&
         p.window == 128 && p.experts == 256 && p.experts_used == 6 && p.expert_ffn == 2048 &&
         p.shared_experts == 1 && p.hash_layers == 3 && p.sinkhorn_iterations == 20 &&
         p.indexer_heads == 64 && p.indexer_head_dim == 128 && p.indexer_top_k == 512 &&
         p.vocab == 129280 && p.rms_eps == expected.rms_eps && p.hc_eps == expected.hc_eps &&
         p.yarn_original_context == expected.yarn_original_context &&
         p.rope_base == expected.rope_base && p.compress_rope_base == expected.compress_rope_base &&
         p.rope_scale == expected.rope_scale && p.yarn_beta_fast == expected.yarn_beta_fast &&
         p.yarn_beta_slow == expected.yarn_beta_slow &&
         p.expert_weights_scale == expected.expert_weights_scale &&
         p.expert_weights_norm == expected.expert_weights_norm &&
         p.swiglu_limit == expected.swiglu_limit &&
         p.swiglu_limit_shared == expected.swiglu_limit_shared &&
         p.compress_ratios == expected.compress_ratios;
}

bool Valid(Buffer b, std::uint64_t bytes, std::uint64_t alignment = kAlignment) {
  return b.address != 0 && b.address % alignment == 0 && b.bytes >= bytes &&
         b.bytes <= std::numeric_limits<std::uint64_t>::max() - b.address;
}

bool Overlap(Buffer a, Buffer b) {
  return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}

std::uint64_t RowBytes(std::uint32_t columns, std::uint32_t element) {
  return std::uint64_t{columns} * element;
}

std::uint64_t Values(std::uint32_t rows, std::uint32_t columns, std::uint32_t element = 4) {
  return std::uint64_t{rows} * columns * element;
}

const void* ReadPointer(std::uint64_t address) {
  return std::bit_cast<const void*>(static_cast<std::uintptr_t>(address));
}

void* WritePointer(std::uint64_t address) {
  return std::bit_cast<void*>(static_cast<std::uintptr_t>(address));
}

kg::Ds4ProductRead Read(Buffer b) { return {.data = ReadPointer(b.address), .bytes = b.bytes}; }
kg::Ds4ProductWrite Write(Buffer b) { return {.data = WritePointer(b.address), .bytes = b.bytes}; }

kg::Ds4ProductMatrix Matrix(Buffer b, std::uint32_t rows, std::uint32_t columns,
                            std::uint32_t element = 4) {
  return {.storage = Read(b),
          .rows = rows,
          .columns = columns,
          .row_stride = RowBytes(columns, element)};
}

kg::Ds4ProductOutput Output(Buffer b, std::uint32_t rows, std::uint32_t columns,
                            std::uint32_t element = 4) {
  return {.storage = Write(b),
          .rows = rows,
          .columns = columns,
          .row_stride = RowBytes(columns, element)};
}

Buffer Slice(Buffer b, std::uint64_t offset, std::uint64_t bytes) {
  if (offset > b.bytes || bytes > b.bytes - offset) return {};
  return {.address = b.address + offset, .bytes = bytes};
}

bool SameShape(const kg::Ds4AlignedShape& a, const kg::Ds4AlignedShape& b) {
  return a.kind == b.kind && a.input == b.input && a.output == b.output && a.groups == b.groups;
}

bool SameLayout(const kg::Ds4AlignedLayout& a, const kg::Ds4AlignedLayout& b) {
  return a.raw_bytes == b.raw_bytes && a.packed_bytes == b.packed_bytes && a.blocks == b.blocks &&
         a.scales_offset == b.scales_offset && a.codes_offset == b.codes_offset &&
         a.blocks_per_row == b.blocks_per_row && a.raw_block_bytes == b.raw_block_bytes;
}

}  // namespace

std::expected<ScratchPlan, std::string> PlanScratch(const model::Dsv4Profile& profile,
                                                    std::uint32_t context, std::uint32_t device_sms,
                                                    bool output_b_study, bool routed_ffn_study) {
  if (!SameProfile(profile) || context != kContext || device_sms == 0 || device_sms > 65535) {
    return std::unexpected("original binding requires Flash/context8192 and valid SM count");
  }
  if (output_b_study && routed_ffn_study)
    return std::unexpected("private consumer studies require separate matched arms");
  ScratchPlan result;
  const auto add = [&](std::string name, std::uint64_t bytes) -> bool {
    if (bytes == 0 || bytes > (std::uint64_t{4} << 30U) ||
        __builtin_add_overflow(result.logical_bytes, bytes, &result.logical_bytes))
      return false;
    result.ranges.push_back({.name = std::move(name), .bytes = bytes, .alignment = kAlignment});
    return true;
  };
  bool good = true;
  const auto values = [&](std::string name, std::uint32_t width, std::uint32_t rows = kRows,
                          std::uint32_t element = 4) {
    good = good && add(std::move(name), Values(rows, width, element));
  };
  values("tokens", 1);
  // The preceding FFN output is the following layer's residual. Ping/pong
  // disjointness also retains the attention residual through its FFN half.
  values("hc.even", 16384);
  values("hc.odd", 16384);
  values("hc.attention", 16384);
  values("hc.attention.f16", 16384, kRows, 2);
  values("hc.ffn.f16", 16384, kRows, 2);
  values("hc.attention.mix", 24);
  values("hc.attention.split", 24);
  values("hc.ffn.mix", 24);
  values("hc.ffn.split", 24);
  values("attention.input", 4096);
  values("attention.norm", 4096);
  values("attention.norm.f16", 4096, kRows, 2);
  values("query.rank", 1024);
  values("query.rank.f16", 1024, kRows, 2);
  values("query.heads", 32768);
  values("kv", 512);
  values("attention.heads", 32768);
  values("attention.low", 8192);
  values("attention.output", 4096);
  values("attention.rope", 64, kRows, 4);  // float2[4096][32]
  values("compression.kv", 1024);
  values("compression.score", 1024);
  values("compression.refresh.kv", 1024, 4);
  values("compression.refresh.score", 1024, 4);
  values("attention.proxy", 512, kContext / 4);
  values("indexer.compression.kv", 256);
  values("indexer.compression.score", 256);
  values("indexer.refresh.kv", 256, 4);
  values("indexer.refresh.score", 256, 4);
  values("indexer.proxy", 128, kContext / 4);
  values("indexer.query", 8192);
  values("indexer.weights", 64);
  values("indexer.query.codes", 64, kRows * 64, 1);
  values("indexer.query.scales", 4, kRows * 64);
  values("indexer.scores", kContext / 4);
  values("indexer.selected", 512);
  good = good && add("indexer.score.scratch", std::uint64_t{kRows} * 64 * 4);
  good = good && add("indexer.diagnostics", sizeof(kg::Ds4IndexerDiagnostics));
  good = good && add("attention.diagnostics", sizeof(kg::Ds4AttentionDiagnostics));
  // Worst dense/indexed TT chain across the two chunks: 2048 compressed
  // records per four-query tile plus paid chronological/F16 mirrors.
  kg::Ds4Attention a{};
  a.tokens = kRows;
  a.first = kRows;
  a.raw_cells = 4352;
  a.raw_count = 4352;
  a.compressed_cells = kContext / 4;
  a.compressed_count = kContext / 4;
  a.consecutive_first = kRows;
  a.domain = kg::Ds4AttentionDomain::kMixedRing;
  auto attention = kg::PlanDs4AttentionScratch(a, kg::Ds4AttentionKind::kTokenTile, device_sms);
  if (!attention) return std::unexpected(attention.error().detail);
  good = good && add("attention.scratch", attention->bytes);
  values("ffn.input", 4096);
  values("ffn.norm", 4096);
  values("ffn.norm.f16", 4096, kRows, 2);
  values("router.logits", 256);
  values("router.probabilities", 256);
  values("router.ids", 6);
  values("router.weights", 6);
  values("shared.gate", 2048);
  values("shared.up", 2048);
  values("shared.middle", 2048);
  values("shared.output", 4096);
  auto routed = kg::Ds4MoeLayoutOf({.rows = kRows, .input = 4096, .middle = 2048, .output = 4096},
                                   kg::Ds4MoeTier::kDirect);
  if (!routed) return std::unexpected(routed.error().detail);
  values("routed.ids.source", 6);
  values("routed.ids.destination", 6);
  good = good && add("routed.bounds", 257ULL * 4);
  good = good && add("routed.work", routed->work_bytes);
  good = good && add("routed.input.quant", routed->input_quant_bytes);
  good = good && add("routed.down.quant", routed->down_quant_bytes);
  good = good && add("routed.down", routed->down_bytes);
  values("routed.sum", 4096);
  if (routed_ffn_study) {
    // These allocations, including probe-only ranges, have the same paid
    // lifetime in Direct and Materialized samples. No full expert replica.
    values("routed.materialized.gate", 2048, 6 * kRows);
    values("routed.materialized.up", 2048, 6 * kRows);
    values("routed.materialized.middle", 2048, 6 * kRows);
    values("routed.control.ids.source", 6);
    values("routed.control.ids.destination", 6);
    good = good && add("routed.control.bounds", 257ULL * 4);
    good = good && add("routed.control.work", routed->work_bytes);
    good = good && add("routed.control.input.quant", routed->input_quant_bytes);
    good = good && add("routed.control.down.quant", routed->down_quant_bytes);
    good = good && add("routed.control.down", routed->down_bytes);
  }
  // Each simultaneously live quantized source gets its own authenticated
  // producer. Q-a/KV share the norm source; other sources remain distinct.
  for (const auto& [name, width] :
       std::array{std::pair{"attention.norm.d4", 4096U}, std::pair{"query.rank.d4", 1024U},
                  std::pair{"attention.low.d4", 8192U}, std::pair{"ffn.norm.d4", 4096U},
                  std::pair{"shared.middle.d4", 2048U}}) {
    auto bytes = kg::Ds4D4Bytes(kRows, width);
    if (!bytes) return std::unexpected(bytes.error().detail);
    good = good && add(name, *bytes);
  }
  values("head.hc.norm", 16384, 1);
  values("head.hc.mix", 4, 1);
  values("head.hc.weights", 4, 1);
  values("head.hidden", 4096, 1);
  values("head.norm", 4096, 1);
  values("head.logits", profile.vocab, 1);
  if (output_b_study) values("output-b.control", 4096);
  auto q81 = kg::Ds4Q81Bytes(1, 4096);
  if (!q81) return std::unexpected(q81.error().detail);
  good = good && add("head.q81", *q81);
  // Whole output-sized bound covers original stream-K tile fixup and the
  // tiny split-K vector head/refresh pool, with no assumed free library pool.
  result.launch_workspace_bytes = Values(kRows, 32768);
  good = good && add("workspace.launch", result.launch_workspace_bytes);
  good = good && add("workspace.cublas", result.cublas_workspace_bytes);
  if (!good) return std::unexpected("original named scratch byte inventory overflows");
  return result;
}

namespace {
bool SameState(const model::Ds4BaselineStateLayout& a, const model::Ds4BaselineStateLayout& b) {
  if (a.context != b.context || a.max_rows != b.max_rows || a.raw_cells != b.raw_cells ||
      a.packed_kv != b.packed_kv || a.packed_indexer != b.packed_indexer ||
      a.granularity != b.granularity || a.fixed_bytes != b.fixed_bytes ||
      a.virtual_bytes != b.virtual_bytes || a.tensors.size() != b.tensors.size())
    return false;
  for (std::size_t i = 0; i < a.tensors.size(); ++i) {
    const auto& x = a.tensors[i];
    const auto& y = b.tensors[i];
    if (x.kind != y.kind || x.layer != y.layer || x.ratio != y.ratio || x.capacity != y.capacity ||
        x.row_bytes != y.row_bytes || x.offset != y.offset || x.bytes != y.bytes)
      return false;
  }
  return true;
}

bool SamePreparation(const engine::Ds4PreparedWeightSet& a, const engine::Ds4PreparedWeightSet& b) {
  if (a.artifact_id != b.artifact_id || a.replaced_groups != b.replaced_groups ||
      a.aligned_expert_bytes != b.aligned_expert_bytes ||
      a.additive_dense_bytes != b.additive_dense_bytes ||
      a.largest_tensor_bytes != b.largest_tensor_bytes || a.tensors.size() != b.tensors.size())
    return false;
  for (std::size_t i = 0; i < a.tensors.size(); ++i) {
    const auto& x = a.tensors[i];
    const auto& y = b.tensors[i];
    if (x.name != y.name || x.index != y.index || x.expert_array != y.expert_array ||
        !SameShape(x.shape, y.shape) || !SameLayout(x.layout, y.layout) ||
        x.sources.size() != y.sources.size())
      return false;
    for (std::size_t j = 0; j < x.sources.size(); ++j) {
      const auto& xs = x.sources[j];
      const auto& ys = y.sources[j];
      if (xs.shard != ys.shard || xs.file_offset != ys.file_offset || xs.bytes != ys.bytes ||
          xs.file_end != ys.file_end)
        return false;
    }
  }
  return true;
}

// Only host descriptors are constructed here. A missing view poisons this
// construction; it never triggers allocation, fallback or GPU submission.
class Binder {
 public:
  Binder(const BindingInputs& inputs, Chunk& chunk) : inputs_(inputs), chunk_(chunk) {}

  Buffer Scratch(std::string_view name) {
    const auto found = std::ranges::find(inputs_.scratch, name, &NamedScratch::name);
    if (found == inputs_.scratch.end()) {
      Fail("missing scratch " + std::string(name));
      return {};
    }
    return found->storage;
  }

  Buffer Raw(const model::Dsv4Tensor& tensor, std::string_view type) {
    if (tensor.type != type || tensor.index >= inputs_.artifact->resources().size()) {
      Fail("original tensor type differs: " + std::string(type));
      return {};
    }
    const auto& r = inputs_.artifact->resources()[tensor.index];
    if (inputs_.raw_weights->group_address(r.group) == 0) {
      Fail("canonical tensor group is not placed: " + r.name);
      return {};
    }
    Buffer result{inputs_.raw_weights->resource_address(tensor.index), r.bytes.value()};
    if (!Valid(result, result.bytes) || result.bytes == 0) {
      Fail("canonical tensor has no valid mapped address: " + r.name);
      return {};
    }
    Immutable(result);
    return result;
  }

  Buffer Aligned(const model::Dsv4Tensor& tensor, bool expert, kg::Ds4AlignedKind kind,
                 std::uint32_t input, std::uint32_t output) {
    const auto* v = inputs_.aligned_weights->Find(tensor.index, expert);
    const kg::Ds4AlignedShape shape{
        .kind = kind, .input = input, .output = output, .groups = expert ? 256U : 1U};
    if (v == nullptr || !SameShape(v->shape, shape)) {
      Fail("missing original aligned tensor");
      return {};
    }
    Buffer result{v->address, v->layout.packed_bytes.value()};
    if (!Valid(result, result.bytes) || result.bytes == 0) {
      Fail("aligned tensor has no valid mapped address");
      return {};
    }
    Immutable(result);
    return result;
  }

  Buffer State(StateKind kind, std::uint32_t layer, std::uint32_t rows = 0) {
    const auto found = std::ranges::find_if(
        inputs_.state->tensors, [&](const auto& t) { return t.kind == kind && t.layer == layer; });
    if (found == inputs_.state->tensors.end() || (rows != 0 && rows > found->capacity)) {
      Fail("missing original state plane");
      return {};
    }
    const auto bytes = rows == 0 ? found->bytes : std::uint64_t{rows} * found->row_bytes;
    return Slice(inputs_.state_storage, found->offset, bytes);
  }

  kg::Ds4Q8Weights Q8(const model::Dsv4Tensor& tensor, std::uint32_t input, std::uint32_t output,
                      bool raw = true) {
    const auto packed = Aligned(tensor, false, kg::Ds4AlignedKind::kQ8Dense, input, output);
    const auto* view = inputs_.aligned_weights->Find(tensor.index, false);
    if (view == nullptr) return {};
    const auto scales = Slice(packed, 0, view->layout.blocks * 2);
    const auto codes = Slice(packed, view->layout.codes_offset, view->layout.blocks * 32);
    return {.raw = raw ? Read(Raw(tensor, "Q8_0")) : kg::Ds4ProductRead{},
            .raw_row_stride = raw ? std::uint64_t{input / 32} * 34 : 0,
            .scales = Read(scales),
            .scale_row_stride = RowBytes(input / 32, 2),
            .codes = Read(codes),
            .code_row_stride = input,
            .rows = output,
            .columns = input};
  }

  kg::Ds4F16Product F16(const model::Dsv4Tensor& weight, std::uint32_t input, std::uint32_t output,
                        Buffer source, Buffer destination) {
    return {.weights = Matrix(Raw(weight, "F16"), output, input, 2),
            .input = Matrix(source, kRows, input, 2),
            .output = Output(destination, kRows, output)};
  }

  kg::Ds4D4Sidecar D4(Buffer source, Buffer storage, std::uint32_t width) const {
    return {.storage = Write(storage),
            .source = ReadPointer(source.address),
            .generation = inputs_.storage_generation,
            .rows = kRows,
            .columns = width};
  }

  kg::Ds4Q8Product Product(const model::Dsv4Tensor& weight, std::uint32_t input,
                           std::uint32_t output, Buffer source, Buffer destination, Buffer sidecar,
                           kg::Ds4Q8Path path, bool prepared) {
    return {.weights = Q8(weight, input, output),
            .input = Matrix(source, kRows, input),
            .output = Output(destination, kRows, output),
            .quantized = D4(source, sidecar, input),
            .generation = inputs_.storage_generation,
            .path = path,
            .prepared = prepared};
  }

  kg::Ds4HcPre Pre(const model::Dsv4Tensor& scale, const model::Dsv4Tensor& base, Buffer residual,
                   Buffer mix, Buffer split, Buffer values) {
    return {.coefficients = {.mix = mix,
                             .scale = Raw(scale, "F32"),
                             .base = Raw(base, "F32"),
                             .split = split,
                             .rows = kRows,
                             .iterations = 20,
                             .epsilon = 1.0e-6F},
            .residual = residual,
            .values = values,
            .width = 4096};
  }

  Compression Compress(const model::Dsv4Tensor& kv_weight, const model::Dsv4Tensor& score_weight,
                       const model::Dsv4Tensor& ape, const model::Dsv4Tensor& norm,
                       std::uint32_t layer, std::uint32_t ratio, kg::Ds4CacheKind kind,
                       const kg::Ds4ProductRope& rope) {
    const bool indexer = kind == kg::Ds4CacheKind::kIndexer128;
    const auto width = indexer ? 128U : 512U;
    const auto columns = width * (ratio == 4 ? 2U : 1U);
    const auto before = inputs_.first / ratio;
    const auto after = (inputs_.first + kRows) / ratio;
    const auto prefix = indexer ? std::string{"indexer."} : std::string{};
    const auto kv = Slice(Scratch(prefix + "compression.kv"), 0, Values(kRows, columns));
    const auto score = Slice(Scratch(prefix + "compression.score"), 0, Values(kRows, columns));
    const auto proxy = Slice(Scratch(indexer ? "indexer.proxy" : "attention.proxy"),
                             Values(before, width), Values(kRows / ratio, width));
    const auto codes_kind = indexer ? StateKind::kIndexerCodes : StateKind::kAttentionCodes;
    const auto scales_kind = indexer ? StateKind::kIndexerScales : StateKind::kAttentionScales;
    const auto code_row = indexer ? 64U : 704U;
    const auto scale_row = indexer ? 16U : 28U;
    Compression result{};
    result.kv_projection = F16(kv_weight, 4096, columns, Scratch("attention.norm.f16"), kv);
    result.score_projection =
        F16(score_weight, 4096, columns, Scratch("attention.norm.f16"), score);
    result.chunk = {
        .state = {.kv = State(indexer ? StateKind::kIndexerKv : StateKind::kAttentionKv, layer),
                  .score =
                      State(indexer ? StateKind::kIndexerScore : StateKind::kAttentionScore, layer),
                  .kind = kind,
                  .ratio = ratio},
        .kv = kv,
        .score = score,
        .ape = Raw(ape, ape.type),
        .norm = Raw(norm, "F32"),
        .values = proxy,
        .codes = Slice(State(codes_kind, layer, after), std::uint64_t{before} * code_row,
                       std::uint64_t{kRows / ratio} * code_row),
        .scales = Slice(State(scales_kind, layer, after), std::uint64_t{before} * scale_row,
                        std::uint64_t{kRows / ratio} * scale_row),
        .ape_format = ape.type == "F16" ? kg::Ds4CompApe::kF16 : kg::Ds4CompApe::kF32,
        .rope = {.original_context = rope.original_context,
                 .frequency_base = rope.base,
                 .frequency_scale = rope.scale,
                 .extension = rope.extension,
                 .attention_factor = rope.attention,
                 .beta_fast = rope.beta_fast,
                 .beta_slow = rope.beta_slow},
        .first = inputs_.first,
        .tokens = kRows,
        .before = before,
        .capacity = (kContext / ratio) + 2,
        .rms_epsilon = 1.0e-6F};
    if (ape.type != "F16" && ape.type != "F32") Fail("unknown original compressor APE format");
    if (ratio == 4) {
      const auto tail = Slice(Scratch("attention.norm"), Values(kRows - 4, 4096), Values(4, 4096));
      const auto refresh_kv = Scratch(indexer ? "indexer.refresh.kv" : "compression.refresh.kv");
      const auto refresh_score =
          Scratch(indexer ? "indexer.refresh.score" : "compression.refresh.score");
      result.refresh_kv = kg::Ds4F16Vector{.weights = result.kv_projection.weights,
                                           .input = Matrix(tail, 4, 4096),
                                           .output = Output(refresh_kv, 4, columns)};
      result.refresh_score = kg::Ds4F16Vector{.weights = result.score_projection.weights,
                                              .input = Matrix(tail, 4, 4096),
                                              .output = Output(refresh_score, 4, columns)};
      result.refresh = kg::Ds4CompRefresh{.state = result.chunk.state,
                                          .kv = refresh_kv,
                                          .score = refresh_score,
                                          .ape = result.chunk.ape,
                                          .ape_format = result.chunk.ape_format,
                                          .first = inputs_.first + kRows - 4};
    }
    return result;
  }

  void Immutable(Buffer b) {
    immutable_.push_back(b);
    if (std::ranges::none_of(chunk_.paid_ranges, [b](Buffer r) {
          return r.address == b.address && r.bytes == b.bytes;
        }))
      chunk_.paid_ranges.push_back(b);
  }
  void Fail(std::string what) {
    if (error_.empty()) error_ = std::move(what);
  }
  const std::string& error() const { return error_; }
  std::span<const Buffer> immutable() const { return immutable_; }

 private:
  const BindingInputs& inputs_;
  Chunk& chunk_;
  std::vector<Buffer> immutable_;
  std::string error_;
};
}  // namespace

std::expected<Chunk, std::string> BindChunk(const model::Dsv4Profile& profile,
                                            const BindingInputs& in) {
  if (!SameProfile(profile) || in.artifact == nullptr || in.raw_weights == nullptr ||
      in.prepared_plan == nullptr || in.aligned_weights == nullptr || in.state == nullptr ||
      !in.raw_weights->opened() || &in.raw_weights->artifact() != in.artifact ||
      in.raw_weights->bytes() == 0 || in.raw_weights->extents().empty() ||
      in.artifact->id() != kCommunityArtifact || (in.first != 0 && in.first != kRows) ||
      in.storage_generation == 0 || in.model_generation == 0) {
    return std::unexpected("incomplete original-model binding inputs");
  }
  if ((!in.output_b_study &&
       (in.output_b_consumer != OutputBConsumer::kOriginal || in.capture_output_b)) ||
      (in.capture_output_b &&
       (in.first != kRows || in.output_b_consumer != OutputBConsumer::kOriginal)))
    return std::unexpected("output-B diagnostic requires original late-chunk study operands");
  if ((in.routed_ffn_tier != RoutedFfnTier::kDirect &&
       in.routed_ffn_tier != RoutedFfnTier::kMaterialized) ||
      (in.routed_ffn_study && in.output_b_study) ||
      (!in.routed_ffn_study &&
       (in.routed_ffn_tier != RoutedFfnTier::kDirect || in.capture_routed_ffn)) ||
      (in.capture_routed_ffn &&
       (in.first != kRows || in.routed_ffn_tier != RoutedFfnTier::kDirect)))
    return std::unexpected("routed-FFN diagnostic requires original late-chunk Direct operands");
  auto scratch =
      PlanScratch(profile, kContext, in.device_sms, in.output_b_study, in.routed_ffn_study);
  if (!scratch) return std::unexpected(scratch.error());
  auto state = model::LayoutDs4BaselineState(profile, kContext, kRows, in.state->granularity);
  if (!state || !SameState(*state, *in.state) ||
      !Valid(in.state_storage, in.state->virtual_bytes, in.state->granularity))
    return std::unexpected("state is not the original packed two-chunk layout");
  auto preparation = engine::PlanDs4PreparedWeights(*in.artifact);
  if (!preparation || !SamePreparation(*preparation, *in.prepared_plan) ||
      in.aligned_weights->views().size() != preparation->tensors.size())
    return std::unexpected("aligned preparation differs from the authenticated artifact");
  for (const auto& t : preparation->tensors) {
    const auto* v = in.aligned_weights->Find(t.index, t.expert_array);
    if (v == nullptr || !SameShape(v->shape, t.shape) || !SameLayout(v->layout, t.layout) ||
        !Valid({v->address, v->layout.packed_bytes.value()}, t.layout.packed_bytes.value()))
      return std::unexpected("mapped aligned tensor does not match its prepared plan");
  }
  auto binding = model::BindDsv4(profile, *in.artifact);
  if (!binding) return std::unexpected(binding.error());
  if (in.scratch.size() != scratch->ranges.size())
    return std::unexpected("named scratch inventory is incomplete or has extra ranges");
  Chunk result;
  result.first = in.first;
  result.context = kContext;
  result.device_sms = in.device_sms;
  result.storage_generation = in.storage_generation;
  result.model_generation = in.model_generation;
  result.artifact_id = in.artifact->id();
  result.output_b_consumer = in.output_b_consumer;
  result.output_b_study = in.output_b_study;
  result.routed_ffn_tier = in.routed_ffn_tier;
  result.routed_ffn_study = in.routed_ffn_study;
  std::vector<Buffer> scratch_ranges;
  for (const auto& required : scratch->ranges) {
    const auto found = std::ranges::find(in.scratch, required.name, &NamedScratch::name);
    if (found == in.scratch.end() ||
        std::ranges::count(in.scratch, required.name, &NamedScratch::name) != 1 ||
        !Valid(found->storage, required.bytes, required.alignment) ||
        Overlap(found->storage, in.state_storage))
      return std::unexpected("named scratch is missing, duplicated, short or aliases state");
    for (const auto& other : scratch_ranges) {
      if (Overlap(found->storage, other)) return std::unexpected("named scratch mappings overlap");
    }
    scratch_ranges.push_back(found->storage);
    result.paid_ranges.push_back(found->storage);
  }
  auto used = model::Ds4BaselineStateThrough(*state, in.first + kRows);
  if (!used) return std::unexpected(used.error());
  for (const auto& r : *used)
    result.paid_ranges.push_back(Slice(in.state_storage, r.offset, r.bytes));
  Binder b(in, result);
  const auto get = [&b](std::string_view name) { return b.Scratch(name); };
  if (in.capture_output_b) result.output_b_control = get("output-b.control");
  if (in.routed_ffn_study)
    result.routed_intermediates = {get("routed.materialized.gate"), get("routed.materialized.up"),
                                   get("routed.materialized.middle")};
  result.embedding = {.tokens = Read(get("tokens")),
                      .weights = Matrix(b.Raw(binding->token_embd, "F16"), profile.vocab, 4096, 2),
                      .output = Output(get("hc.even"), kRows, 16384),
                      .hyper_connections = 4};
  for (std::uint32_t i = 0; i < kLayers; ++i) {
    const auto& w = binding->layers[i];
    auto rope = OriginalRope(profile, w.ratio, in.first);
    auto inverse = OriginalRope(profile, w.ratio, in.first, true);
    if (!rope || !inverse) return std::unexpected("invalid model-derived original RoPE");
    Layer l{};
    l.index = i;
    l.ratio = w.ratio;
    const auto residual = get(i % 2 == 0 ? "hc.even" : "hc.odd");
    const auto next = get(i % 2 == 0 ? "hc.odd" : "hc.even");
    l.hc_attention_rms = {.source = residual,
                          .values_f16 = get("hc.attention.f16"),
                          .width = 16384,
                          .rows = kRows,
                          .epsilon = profile.rms_eps};
    l.hc_attention_projection =
        b.F16(w.hc_attn_fn, 16384, 24, get("hc.attention.f16"), get("hc.attention.mix"));
    l.hc_attention_pre = b.Pre(w.hc_attn_scale, w.hc_attn_base, residual, get("hc.attention.mix"),
                               get("hc.attention.split"), get("attention.input"));
    l.attention_norm = {.source = get("attention.input"),
                        .weights = b.Raw(w.attn_norm, "F32"),
                        .values = get("attention.norm"),
                        .values_f16 = get("attention.norm.f16"),
                        .q8_d4 = get("attention.norm.d4"),
                        .width = 4096,
                        .rows = kRows,
                        .epsilon = profile.rms_eps};
    l.query_a = b.Product(w.q_a, 4096, 1024, get("attention.norm"), get("query.rank"),
                          get("attention.norm.d4"), kg::Ds4Q8Path::kMmq, true);
    l.kv_projection = b.Product(w.kv, 4096, 512, get("attention.norm"), get("kv"),
                                get("attention.norm.d4"), kg::Ds4Q8Path::kMmq, true);
    l.qkv_norm = {.query = Matrix(get("query.rank"), kRows, 1024),
                  .query_weight = Read(b.Raw(w.q_a_norm, "F32")),
                  .query_output = Output(get("query.rank"), kRows, 1024),
                  .kv = Matrix(get("kv"), kRows, 512),
                  .kv_weight = Read(b.Raw(w.kv_norm, "F32")),
                  .kv_output = Output(get("kv"), kRows, 512),
                  .epsilon = profile.rms_eps};
    l.query_b = b.Product(w.q_b, 1024, 32768, get("query.rank"), get("query.heads"),
                          get("query.rank.d4"), kg::Ds4Q8Path::kDenseD2r, false);
    l.query_rope = {.input = Output(get("query.heads"), kRows, 32768),
                    .heads = 64,
                    .head_width = 512,
                    .rope = *rope,
                    .normalize = true,
                    .epsilon = profile.rms_eps};
    l.kv_rope = {.input = Output(get("kv"), kRows, 512),
                 .heads = 1,
                 .head_width = 512,
                 .rope = *rope,
                 .normalize = false,
                 .epsilon = profile.rms_eps};
    l.raw_qat = {.kind = kg::Ds4CacheKind::kKv512,
                 .values = get("kv"),
                 .codes = {},
                 .scales = {},
                 .rows = kRows};
    l.raw_store = {.source = get("kv"),
                   .ring = b.State(StateKind::kRaw, i),
                   .first = in.first,
                   .rows = kRows,
                   .cells = state->raw_cells};
    if (w.ratio != 0) {
      l.compression = b.Compress(w.comp_kv, w.comp_gate, w.comp_ape, w.comp_norm, i, w.ratio,
                                 kg::Ds4CacheKind::kKv512, *rope);
    }
    const auto cells = w.ratio == 0 ? 0 : (in.first + kRows) / w.ratio;
    const bool static_hca = in.first == 0 && w.ratio == 128;
    l.attention.query = get("query.heads");
    l.attention.output = get("attention.heads");
    l.attention.sinks = b.Raw(w.attn_sinks, "F32");
    l.attention.raw = static_hca ? get("kv") : l.raw_store.ring;
    l.attention.scratch = get("attention.scratch");
    l.attention.diagnostics = get("attention.diagnostics");
    l.attention.tokens = kRows;
    l.attention.first = in.first;
    l.attention.raw_cells = static_hca ? kRows : state->raw_cells;
    l.attention.raw_count = in.first == 0 ? kRows : kRows + profile.window;
    l.attention.raw_start = in.first == 0 ? 0 : (in.first - profile.window) % state->raw_cells;
    l.attention.compressed_cells = cells;
    l.attention.compressed_count = cells;
    l.attention.ratio = w.ratio == 0 ? 1 : w.ratio;
    l.attention.consecutive_first = in.first;
    l.attention.domain = kg::Ds4AttentionDomain::kMixedRing;
    if (w.ratio == 4)
      l.attention.domain = kg::Ds4AttentionDomain::kIndexedRing;
    else if (static_hca)
      l.attention.domain = kg::Ds4AttentionDomain::kMixedPrefill;
    if (cells != 0) {
      l.attention.compressed = Slice(get("attention.proxy"), 0, Values(cells, 512));
      l.attention.compressed_codes = b.State(StateKind::kAttentionCodes, i, cells);
      l.attention.compressed_scales = b.State(StateKind::kAttentionScales, i, cells);
      l.attention.decode_table = b.State(StateKind::kDecodeTable, 0);
    }
    if (w.ratio == 4) {
      Indexer x{};
      x.compression = b.Compress(w.idx_comp_kv, w.idx_comp_gate, w.idx_comp_ape, w.idx_comp_norm, i,
                                 4, kg::Ds4CacheKind::kIndexer128, *rope);
      x.query_conversion = {.input = Matrix(get("query.rank"), kRows, 1024),
                            .output = Output(get("query.rank.f16"), kRows, 1024, 2)};
      x.query_projection =
          b.F16(w.idx_q_b, 1024, 8192, get("query.rank.f16"), get("indexer.query"));
      x.weight_projection =
          b.F16(w.idx_proj, 4096, 64, get("attention.norm.f16"), get("indexer.weights"));
      x.query_rope = {.input = Output(get("indexer.query"), kRows, 8192),
                      .heads = 64,
                      .head_width = 128,
                      .rope = *rope,
                      .normalize = false,
                      .epsilon = profile.rms_eps};
      x.query_qat = {.kind = kg::Ds4CacheKind::kIndexer128,
                     .values = get("indexer.query"),
                     .codes = get("indexer.query.codes"),
                     .scales = get("indexer.query.scales"),
                     .rows = kRows * 64};
      x.scores.query = x.query_qat.values;
      x.scores.weights = get("indexer.weights");
      x.scores.keys = Slice(get("indexer.proxy"), 0, Values(cells, 128));
      x.scores.key_codes = b.State(StateKind::kIndexerCodes, i, cells);
      x.scores.key_scales = b.State(StateKind::kIndexerScales, i, cells);
      x.scores.query_codes = x.query_qat.codes;
      x.scores.query_scales = x.query_qat.scales;
      x.scores.scores = Slice(get("indexer.scores"), 0, Values(kRows, cells));
      x.scores.scratch = get("indexer.score.scratch");
      x.scores.diagnostics = get("indexer.diagnostics");
      x.scores.tokens = kRows;
      x.scores.cells = cells;
      x.scores.cells_per_bank = cells;
      x.scores.first = in.first;
      x.scores.score_band = cells;
      x.scores.scale = 1.0F / std::sqrt(128.0F * 64.0F);
      x.select.scores = x.scores.scores;
      x.select.selected = get("indexer.selected");
      x.select.diagnostics = x.scores.diagnostics;
      x.select.tokens = kRows;
      x.select.cells = cells;
      x.select.score_band = cells;
      l.attention.selected = x.select.selected;
      l.indexer = x;
    }
    l.output_a = {.weights = b.Q8(w.out_a, 4096, 8192, false),
                  .heads = Matrix(get("attention.heads"), kRows, 32768),
                  .low = Output(get("attention.low"), kRows, 8192),
                  .rope_table = Write(get("attention.rope")),
                  .rope = *inverse,
                  .quantized = b.D4(get("attention.low"), get("attention.low.d4"), 8192),
                  .generation = in.storage_generation};
    l.output_b = b.Product(w.out_b, 8192, 4096, get("attention.low"), get("attention.output"),
                           get("attention.low.d4"), kg::Ds4Q8Path::kMmq, true);
    l.attention_expand = {.block = get("attention.output"),
                          .residual = residual,
                          .split = get("hc.attention.split"),
                          .values = get("hc.attention"),
                          .values_f16 = get("hc.ffn.f16"),
                          .width = 4096,
                          .rows = kRows,
                          .epsilon = profile.rms_eps};
    l.hc_ffn_projection = b.F16(w.hc_ffn_fn, 16384, 24, get("hc.ffn.f16"), get("hc.ffn.mix"));
    l.hc_ffn_pre = b.Pre(w.hc_ffn_scale, w.hc_ffn_base, get("hc.attention"), get("hc.ffn.mix"),
                         get("hc.ffn.split"), get("ffn.input"));
    l.ffn_norm = {.source = get("ffn.input"),
                  .weights = b.Raw(w.ffn_norm, "F32"),
                  .values = get("ffn.norm"),
                  .values_f16 = get("ffn.norm.f16"),
                  .q8_d4 = get("ffn.norm.d4"),
                  .width = 4096,
                  .rows = kRows,
                  .epsilon = profile.rms_eps};
    l.router_projection = b.F16(w.router, 4096, 256, get("ffn.norm.f16"), get("router.logits"));
    l.router = {.logits = get("router.logits"),
                .bias = i < profile.hash_layers ? Buffer{} : b.Raw(w.router_bias, "F32"),
                .hash = i < profile.hash_layers ? b.Raw(w.tid2eid, "I32") : Buffer{},
                .tokens = i < profile.hash_layers ? get("tokens") : Buffer{},
                .selected = get("router.ids"),
                .weights = get("router.weights"),
                .probabilities = get("router.probabilities"),
                .rows = kRows,
                .hash_rows = i < profile.hash_layers ? profile.vocab : 0};
    l.routed = {
        .shape = {.rows = kRows, .input = 4096, .middle = 2048, .output = 4096},
        .tier = kg::Ds4MoeTier::kDirect,
        .generation = in.storage_generation,
        .producer = {.storage = get("ffn.norm.d4"),
                     .source_address = get("ffn.norm").address,
                     .generation = in.storage_generation,
                     .rows = kRows,
                     .width = 4096,
                     .kind = kg::Ds4MoeQuant::kD4},
        .input = get("ffn.norm"),
        .gate_weights = b.Aligned(w.gate_exps, true, kg::Ds4AlignedKind::kIq2Xxs, 4096, 2048),
        .up_weights = b.Aligned(w.up_exps, true, kg::Ds4AlignedKind::kIq2Xxs, 4096, 2048),
        .down_weights = b.Aligned(w.down_exps, true, kg::Ds4AlignedKind::kQ2K, 2048, 4096),
        .selected = get("router.ids"),
        .weights = get("router.weights"),
        .ids_source = get("routed.ids.source"),
        .ids_destination = get("routed.ids.destination"),
        .expert_bounds = get("routed.bounds"),
        .work = get("routed.work"),
        .input_quant = get("routed.input.quant"),
        .down_quant = get("routed.down.quant"),
        .down = get("routed.down")};
    if (in.routed_ffn_tier == RoutedFfnTier::kMaterialized) {
      l.routed.tier = kg::Ds4MoeTier::kMaterialized;
      l.routed.producer = {};
      l.routed.gate = result.routed_intermediates[0];
      l.routed.up = result.routed_intermediates[1];
      l.routed.middle = result.routed_intermediates[2];
    }
    if (in.capture_routed_ffn && CaptureRoutedFfnAt(in.first, i)) {
      auto control = l.routed;
      control.tier = kg::Ds4MoeTier::kMaterialized;
      control.producer = {};
      control.gate = result.routed_intermediates[0];
      control.up = result.routed_intermediates[1];
      control.middle = result.routed_intermediates[2];
      control.ids_source = get("routed.control.ids.source");
      control.ids_destination = get("routed.control.ids.destination");
      control.expert_bounds = get("routed.control.bounds");
      control.work = get("routed.control.work");
      control.input_quant = get("routed.control.input.quant");
      control.down_quant = get("routed.control.down.quant");
      control.down = get("routed.control.down");
      l.routed_control = control;
    }
    l.shared_gate = b.Product(w.gate_shexp, 4096, 2048, get("ffn.norm"), get("shared.gate"),
                              get("ffn.norm.d4"), kg::Ds4Q8Path::kDenseD2r, true);
    l.shared_up = b.Product(w.up_shexp, 4096, 2048, get("ffn.norm"), get("shared.up"),
                            get("ffn.norm.d4"), kg::Ds4Q8Path::kDenseD2r, true);
    l.shared_swiglu = {.gate = get("shared.gate"),
                       .up = get("shared.up"),
                       .output = get("shared.middle"),
                       .rows = kRows,
                       .width = 2048};
    l.shared_down = b.Product(w.down_shexp, 2048, 4096, get("shared.middle"), get("shared.output"),
                              get("shared.middle.d4"), kg::Ds4Q8Path::kDenseD2r, false);
    l.ffn_expand = {.add = get("shared.output"),
                    .residual = get("hc.attention"),
                    .split = get("hc.ffn.split"),
                    .values = next,
                    .width = 4096,
                    .rows = kRows,
                    .epsilon = profile.rms_eps};
    if (i + 1 == kLayers) {
      l.final_sum = kg::Ds4MoeSum{
          .slots = get("routed.down"), .output = get("routed.sum"), .rows = kRows, .width = 4096};
      l.ffn_expand.block = get("routed.sum");
    } else {
      l.ffn_expand.values_f16 = get("hc.attention.f16");
      l.ffn_expand.moe_unsummed = get("routed.down");
    }
    result.layers.push_back(l);
  }
  if (in.first == kRows) {
    Frontier head{};
    const auto last =
        Slice(result.layers.back().ffn_expand.values, Values(kRows - 1, 16384), Values(1, 16384));
    head.hc_rms = {.source = last,
                   .values = get("head.hc.norm"),
                   .width = 16384,
                   .rows = 1,
                   .epsilon = profile.rms_eps};
    head.hc_projection = {.weights = Matrix(b.Raw(binding->hc_head_fn, "F16"), 4, 16384, 2),
                          .input = Matrix(get("head.hc.norm"), 1, 16384),
                          .output = Output(get("head.hc.mix"), 1, 4)};
    head.hc_weights = {.pre = get("head.hc.mix"),
                       .scale = b.Raw(binding->hc_head_scale, "F32"),
                       .base = b.Raw(binding->hc_head_base, "F32"),
                       .values = get("head.hc.weights"),
                       .rows = 1,
                       .epsilon = profile.hc_eps};
    head.collapse = {.residual = last,
                     .weights = get("head.hc.weights"),
                     .values = get("head.hidden"),
                     .width = 4096,
                     .rows = 1,
                     .weight_stride = 4};
    head.norm = {.source = get("head.hidden"),
                 .weights = b.Raw(binding->output_norm, "F32"),
                 .values = get("head.norm"),
                 .width = 4096,
                 .rows = 1,
                 .epsilon = profile.rms_eps};
    head.projection = {.weights = b.Q8(binding->output, 4096, profile.vocab),
                       .input = Matrix(get("head.norm"), 1, 4096),
                       .output = Output(get("head.logits"), 1, profile.vocab),
                       .quantized = {.storage = Write(get("head.q81")),
                                     .source = ReadPointer(get("head.norm").address),
                                     .generation = in.storage_generation,
                                     .rows = 1,
                                     .columns = 4096},
                       .generation = in.storage_generation};
    result.frontier = head;
  }
  if (!b.error().empty()) return std::unexpected(b.error());
  for (const auto& immutable : b.immutable()) {
    if (Overlap(immutable, in.state_storage))
      return std::unexpected("immutable weights overlap state reservation");
    for (const auto& temporary : scratch_ranges) {
      if (Overlap(immutable, temporary))
        return std::unexpected("scratch mapping overlaps model weights");
    }
  }
  if (auto checked = CheckChunk(profile, result); !checked) return std::unexpected(checked.error());
  if (auto checked = CheckHashTables(profile, result, in.hash_tables); !checked)
    return std::unexpected(checked.error());
  return result;
}

}  // namespace jitllm::benchmarks::ds4_complete
