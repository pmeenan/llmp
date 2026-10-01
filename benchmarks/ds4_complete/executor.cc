// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "benchmarks/ds4_complete/executor.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <string_view>
#include <type_traits>
#include <utility>

#ifdef JITLLM_DS4_COMPLETE_CUDA
#include "engine/paged_node.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/launch.h"
#include "providers/device_runtime.h"
#include "scheduler/commands.h"
#endif

namespace jitllm::benchmarks::ds4_complete {
struct CompletionToken {
  const kg::LaunchContext* owner = nullptr;
  std::uint64_t serial = 0;
  bool poisoned = false;
};
namespace {
using Buffer = kg::Ds4CacheBuffer;
using Checked = std::expected<void, kg::KernelFailure>;

Result Fail(std::string_view what) { return std::unexpected(std::string(what)); }
Result CheckedStages(std::initializer_list<Checked> checks) {
  for (const auto& check : checks) {
    if (!check) return Fail(check.error().detail);
  }
  return {};
}
std::uint64_t Address(const void* p) {
  return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(p));
}
Buffer View(kg::Ds4ProductRead b) { return {Address(b.data), b.bytes}; }
Buffer View(kg::Ds4ProductWrite b) { return {Address(b.data), b.bytes}; }
bool Same(Buffer a, Buffer b) { return a.address != 0 && a.address == b.address; }
bool Same(Buffer a, kg::Ds4ProductRead b) { return Same(a, View(b)); }
bool Same(Buffer a, kg::Ds4ProductWrite b) { return Same(a, View(b)); }
bool Range(Buffer b) { return b.address != 0 && b.bytes != 0 && b.address <= UINT64_MAX - b.bytes; }
bool Covers(Buffer outer, Buffer inner) {
  return Range(outer) && Range(inner) && inner.address >= outer.address &&
         inner.address - outer.address <= outer.bytes &&
         inner.bytes <= outer.bytes - (inner.address - outer.address);
}
bool Overlap(Buffer a, Buffer b) {
  return Range(a) && Range(b) && a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}
bool Absent(Buffer b) { return b.address == 0 && b.bytes == 0; }
bool Absent(kg::Ds4ProductRead b) { return b.data == nullptr && b.bytes == 0; }
bool Matrix(kg::Ds4ProductMatrix m, std::uint32_t rows, std::uint32_t columns,
            std::uint32_t bytes) {
  return m.rows == rows && m.columns == columns &&
         m.row_stride == static_cast<std::uint64_t>(columns) * bytes;
}
bool Matrix(kg::Ds4ProductOutput m, std::uint32_t rows, std::uint32_t columns,
            std::uint32_t bytes) {
  return m.rows == rows && m.columns == columns &&
         m.row_stride == static_cast<std::uint64_t>(columns) * bytes;
}
bool Linked(kg::Ds4ProductOutput a, kg::Ds4ProductMatrix b) {
  return Same(View(a.storage), b.storage) && a.rows == b.rows && a.columns == b.columns;
}
bool Rope(const kg::Ds4ProductRope& a, const kg::Ds4ProductRope& b) {
  return Absent(a.positions) && a.first == b.first && a.step == b.step &&
         a.original_context == b.original_context && a.rotary == b.rotary && a.base == b.base &&
         a.scale == b.scale && a.extension == b.extension && a.attention == b.attention &&
         a.beta_fast == b.beta_fast && a.beta_slow == b.beta_slow && a.inverse == b.inverse;
}
bool CompRope(const kg::Ds4CompRope& a, const kg::Ds4ProductRope& b) {
  return a.original_context == b.original_context && a.frequency_base == b.base &&
         a.frequency_scale == b.scale && a.extension == b.extension &&
         a.attention_factor == b.attention && a.beta_fast == b.beta_fast &&
         a.beta_slow == b.beta_slow;
}
std::uint32_t OriginalRatio(std::uint32_t layer) {
  if (layer < 2) return 0;
  return layer % 2U == 0 ? 4U : 128U;
}
Result Profile(const model::Dsv4Profile& p) {
  if (p.layers != kLayers || p.width != 4096 || p.hc != 4 || p.heads != 64 || p.head_dim != 512 ||
      p.rope_dims != 64 || p.q_lora != 1024 || p.o_lora != 1024 || p.o_groups != 8 ||
      p.window != 128 || p.experts != 256 || p.experts_used != 6 || p.expert_ffn != 2048 ||
      p.shared_experts != 1 || p.hash_layers != 3 || p.sinkhorn_iterations != 20 ||
      p.indexer_heads != 64 || p.indexer_head_dim != 128 || p.indexer_top_k != 512 ||
      p.vocab != 129280 || p.rms_eps != 1.0e-6F || p.hc_eps != 1.0e-6F ||
      p.expert_weights_scale != 1.5F || !p.expert_weights_norm || p.swiglu_limit != 10.0F ||
      p.swiglu_limit_shared != 10.0F || p.rope_base != 10000.0F ||
      p.compress_rope_base != 160000.0F || p.rope_scale != 16.0F ||
      p.yarn_original_context != 65536 || p.yarn_beta_fast != 32.0F || p.yarn_beta_slow != 1.0F ||
      p.compress_ratios.size() != kLayers)
    return Fail("not the pinned Flash wide-prefill profile");
  for (std::uint32_t i = 0; i < kLayers; ++i) {
    const auto ratio = OriginalRatio(i);
    if (p.compress_ratios[i] != ratio)
      return Fail("compression schedule differs from original Flash");
  }
  if (!std::isfinite(p.rope_base) || p.rope_base <= 0 || !std::isfinite(p.compress_rope_base) ||
      p.compress_rope_base <= 0 || !std::isfinite(p.rope_scale) || p.rope_scale <= 0 ||
      p.yarn_original_context == 0 || !std::isfinite(p.yarn_beta_fast) ||
      !std::isfinite(p.yarn_beta_slow) || p.yarn_beta_fast <= 0 || p.yarn_beta_slow <= 0)
    return Fail("invalid model-derived YaRN parameters");
  return {};
}

// Enumerate every borrowed range explicitly; no object padding enters the
// resolve identity. Weights, alternate physical planes, state, optional
// scratch and producer mirrors all participate, including read-only views.
struct Inventory {
  std::vector<Buffer> ranges;
  std::vector<Buffer> immutable;
  std::vector<Buffer> writes;
  std::vector<Buffer> scratch_writes;
  std::vector<Buffer> persistent;
  std::vector<std::uint64_t> scalars;
  template <class T>
  void Value(T v) {
    if constexpr (std::is_same_v<T, float>)
      scalars.push_back(std::bit_cast<std::uint32_t>(v));
    else
      scalars.push_back(static_cast<std::uint64_t>(v));
  }
  template <class... T>
  void Values(T... values) {
    (Value(values), ...);
  }
  void Meta(const kg::Ds4ProductMatrix& d) { Values(d.rows, d.columns, d.row_stride); }
  void Meta(const kg::Ds4ProductOutput& d) { Values(d.rows, d.columns, d.row_stride); }
  template <class T>
  void Sidecar(const T& d) {
    Add(d.storage);
    Values(Address(d.source), d.generation, d.rows, d.columns);
  }
  void RopeMeta(const kg::Ds4ProductRope& d) {
    Add(d.positions);
    Values(d.first, d.step, d.original_context, d.rotary, d.base, d.scale, d.extension, d.attention,
           d.beta_fast, d.beta_slow, d.inverse);
  }
  void RopeMeta(const kg::Ds4HeadRope& d) {
    Meta(d.input);
    RopeMeta(d.rope);
    Values(d.heads, d.head_width, d.normalize, d.epsilon);
  }
  void Add(Buffer b) {
    if (!Absent(b)) ranges.push_back(b);
  }
  void Immutable(Buffer b) {
    Add(b);
    if (!Absent(b)) immutable.push_back(b);
  }
  void Write(Buffer b) {
    Add(b);
    if (!Absent(b)) {
      writes.push_back(b);
      scratch_writes.push_back(b);
    }
  }
  void Persist(Buffer b) {
    Add(b);
    if (!Absent(b)) persistent.push_back(b);
  }
  void StateWrite(Buffer b) {
    Persist(b);
    if (!Absent(b)) writes.push_back(b);
  }
  void Add(kg::Ds4ProductRead b) { Add(View(b)); }
  void Add(kg::Ds4ProductWrite b) { Write(View(b)); }
  void Add(const kg::Ds4Rms& d) {
    Values(d.width, d.rows, d.epsilon);
    Add(d.source);
    Immutable(d.weights);
    Write(d.values);
    Write(d.values_f16);
    Write(d.q8_d4);
  }
  void Add(const kg::Ds4HcPre& d) {
    Values(d.width, d.coefficients.rows, d.coefficients.iterations, d.coefficients.epsilon);
    Add(d.coefficients.mix);
    Immutable(d.coefficients.scale);
    Immutable(d.coefficients.base);
    Write(d.coefficients.split);
    Add(d.residual);
    Write(d.values);
  }
  void Add(const kg::Ds4HcExpand& d) {
    Values(d.width, d.rows, d.epsilon);
    Add(d.block);
    Add(d.add);
    Add(d.residual);
    Add(d.split);
    Write(d.values);
    Write(d.values_f16);
    Add(d.moe_unsummed);
  }
  void Add(const kg::Ds4F16Product& d) {
    Meta(d.weights);
    Meta(d.input);
    Meta(d.output);
    Immutable(View(d.weights.storage));
    Add(d.input.storage);
    Add(d.output.storage);
  }
  void Add(const kg::Ds4F16Vector& d) {
    Meta(d.weights);
    Meta(d.input);
    Meta(d.output);
    Immutable(View(d.weights.storage));
    Add(d.input.storage);
    Add(d.output.storage);
  }
  void Add(const kg::Ds4Q8Weights& d) {
    Values(d.rows, d.columns, d.raw_row_stride, d.scale_row_stride, d.code_row_stride);
    Immutable(View(d.raw));
    Immutable(View(d.scales));
    Immutable(View(d.codes));
  }
  void Add(const kg::Ds4Q8Product& d) {
    Values(d.generation, d.path, d.prepared);
    Meta(d.input);
    Meta(d.output);
    Add(d.weights);
    Add(d.input.storage);
    Add(d.output.storage);
    Sidecar(d.quantized);
  }
  void Add(const kg::Ds4Q8Vector& d) {
    Values(d.generation, d.path, d.prepared);
    Meta(d.input);
    Meta(d.output);
    Add(d.weights);
    Add(d.input.storage);
    Add(d.output.storage);
    Sidecar(d.quantized);
  }
  void Add(const kg::Ds4CacheQat& d) {
    Values(d.kind, d.rows);
    Write(d.values);
    Write(d.codes);
    Write(d.scales);
  }
  void Add(const Compression& d) {
    Values(d.chunk.state.kind, d.chunk.state.ratio, d.chunk.ape_format, d.chunk.first,
           d.chunk.tokens, d.chunk.before, d.chunk.capacity, d.chunk.rms_epsilon,
           d.chunk.rope.original_context, d.chunk.rope.frequency_base, d.chunk.rope.frequency_scale,
           d.chunk.rope.extension, d.chunk.rope.attention_factor, d.chunk.rope.beta_fast,
           d.chunk.rope.beta_slow);
    Add(d.kv_projection);
    Add(d.score_projection);
    StateWrite(d.chunk.state.kv);
    StateWrite(d.chunk.state.score);
    Add(d.chunk.kv);
    Add(d.chunk.score);
    Immutable(d.chunk.ape);
    Immutable(d.chunk.norm);
    Write(d.chunk.values);
    StateWrite(d.chunk.codes);
    StateWrite(d.chunk.scales);
    if (d.refresh_kv) Add(*d.refresh_kv);
    if (d.refresh_score) Add(*d.refresh_score);
    if (d.refresh) {
      Values(d.refresh->state.kind, d.refresh->state.ratio, d.refresh->ape_format,
             d.refresh->first);
      StateWrite(d.refresh->state.kv);
      StateWrite(d.refresh->state.score);
      Add(d.refresh->kv);
      Add(d.refresh->score);
      Immutable(d.refresh->ape);
    }
  }
  void Add(const Indexer& d) {
    Meta(d.query_conversion.input);
    Meta(d.query_conversion.output);
    RopeMeta(d.query_rope);
    Add(d.compression);
    Add(d.query_conversion.input.storage);
    Add(d.query_conversion.output.storage);
    Add(d.query_projection);
    Add(d.weight_projection);
    Add(d.query_rope.input.storage);
    Add(d.query_qat);
    const auto& s = d.scores;
    Values(s.tokens, s.cells, s.cells_per_bank, s.banks, s.first, s.ratio, s.score_band,
           s.consecutive_bank, s.scale, s.causal, s.quality_mode, s.kind);
    Values(d.select.tokens, d.select.cells, d.select.score_band, d.select.top_k, d.select.kind);
    Add(s.query);
    Add(s.weights);
    Add(s.keys);
    Persist(s.key_codes);
    Persist(s.key_scales);
    Add(s.query_codes);
    Add(s.query_scales);
    Add(s.positions);
    Add(s.bank_ids);
    Add(s.layer_scalars);
    Write(s.scores);
    Write(s.scratch);
    Write(s.diagnostics);
    Add(d.select.scores);
    Write(d.select.selected);
    Add(d.select.layer_scalars);
    Write(d.select.scratch);
    Write(d.select.diagnostics);
  }
  void Add(const Layer& d) {
    Values(d.index, d.ratio);
    Meta(d.qkv_norm.query);
    Meta(d.qkv_norm.query_output);
    Meta(d.qkv_norm.kv);
    Meta(d.qkv_norm.kv_output);
    Value(d.qkv_norm.epsilon);
    RopeMeta(d.query_rope);
    RopeMeta(d.kv_rope);
    Values(d.raw_store.first, d.raw_store.rows, d.raw_store.cells);
    Add(d.hc_attention_rms);
    Add(d.hc_attention_projection);
    Add(d.hc_attention_pre);
    Add(d.attention_norm);
    Add(d.query_a);
    Add(d.kv_projection);
    Add(d.query_b);
    Add(d.qkv_norm.query.storage);
    Immutable(View(d.qkv_norm.query_weight));
    Add(d.qkv_norm.query_output.storage);
    Add(d.qkv_norm.kv.storage);
    Immutable(View(d.qkv_norm.kv_weight));
    Add(d.qkv_norm.kv_output.storage);
    Add(d.query_rope.input.storage);
    Add(d.kv_rope.input.storage);
    Add(d.raw_qat);
    Add(d.raw_store.source);
    StateWrite(d.raw_store.ring);
    if (d.compression) Add(*d.compression);
    if (d.indexer) Add(*d.indexer);
    const auto& a = d.attention;
    Values(a.tokens, a.first, a.raw_cells, a.raw_count, a.raw_start, a.compressed_cells,
           a.compressed_count, a.banks, a.window, a.ratio, a.top_k, a.consecutive_first,
           a.allow_multisequence_heads8, a.quality_mode, a.domain, a.kind);
    Add(a.query);
    Write(a.output);
    Immutable(a.sinks);
    Add(a.raw);
    Add(a.compressed);
    Persist(a.compressed_codes);
    Persist(a.compressed_scales);
    Immutable(a.decode_table);
    Add(a.selected);
    Add(a.mask);
    Add(a.positions);
    Add(a.bank_ids);
    Add(a.draft_raw_count);
    Add(a.decode_scalars);
    Add(a.layer_scalars);
    Write(a.scratch);
    Write(a.diagnostics);
    Add(d.output_a.weights);
    Add(d.output_a.heads.storage);
    Add(d.output_a.low.storage);
    Meta(d.output_a.heads);
    Meta(d.output_a.low);
    RopeMeta(d.output_a.rope);
    Value(d.output_a.generation);
    Add(d.output_a.rope_table);
    Sidecar(d.output_a.quantized);
    Add(d.output_b);
    Add(d.attention_expand);
    Add(d.hc_ffn_projection);
    Add(d.hc_ffn_pre);
    Add(d.ffn_norm);
    Add(d.router_projection);
    Values(d.router.rows, d.router.hash_rows, d.router.select);
    Add(d.router.logits);
    Immutable(d.router.bias);
    Immutable(d.router.hash);
    Add(d.router.tokens);
    Write(d.router.selected);
    Write(d.router.weights);
    Write(d.router.probabilities);
    const auto& m = d.routed;
    Values(m.shape.rows, m.shape.input, m.shape.middle, m.shape.output, m.tier, m.generation,
           m.selected_stride, m.weight_stride, m.producer.source_address, m.producer.generation,
           m.producer.rows, m.producer.width, m.producer.kind);
    Add(m.producer.storage);
    Add(m.input);
    Immutable(m.gate_weights);
    Immutable(m.up_weights);
    Immutable(m.down_weights);
    Add(m.selected);
    Add(m.weights);
    Write(m.compact_ids);
    Write(m.compact_weights);
    Write(m.ids_source);
    Write(m.ids_destination);
    Write(m.expert_bounds);
    Write(m.work);
    Write(m.input_quant);
    Write(m.down_quant);
    Write(m.gate);
    Write(m.up);
    Write(m.middle);
    Write(m.down);
    Write(m.sum);
    Add(d.shared_gate);
    Add(d.shared_up);
    Add(d.shared_down);
    Values(d.shared_swiglu.rows, d.shared_swiglu.width);
    Add(d.shared_swiglu.gate);
    Add(d.shared_swiglu.up);
    Write(d.shared_swiglu.output);
    if (d.final_sum) {
      Values(d.final_sum->rows, d.final_sum->width);
      Add(d.final_sum->slots);
      Write(d.final_sum->output);
    }
    Add(d.ffn_expand);
  }
  void Add(const Frontier& d) {
    Values(d.hc_weights.rows, d.hc_weights.epsilon, d.collapse.width, d.collapse.rows,
           d.collapse.weight_stride);
    Add(d.hc_rms);
    Add(d.hc_projection);
    Add(d.hc_weights.pre);
    Immutable(d.hc_weights.scale);
    Immutable(d.hc_weights.base);
    Write(d.hc_weights.values);
    Add(d.collapse.residual);
    Add(d.collapse.weights);
    Write(d.collapse.values);
    Add(d.norm);
    Add(d.projection);
  }
};
Inventory Operands(const Chunk& c) {
  Inventory out;
  out.Meta(c.embedding.weights);
  out.Meta(c.embedding.output);
  out.Value(c.embedding.hyper_connections);
  out.Immutable(View(c.embedding.tokens));
  out.Immutable(View(c.embedding.weights.storage));
  out.Add(c.embedding.output.storage);
  for (const auto& l : c.layers) out.Add(l);
  if (c.frontier) out.Add(*c.frontier);
  return out;
}
std::vector<std::uint64_t> Identity(const Chunk& c) {
  const auto ranges = Operands(c);
  std::vector<std::uint64_t> result{c.first,
                                    c.context,
                                    c.device_sms,
                                    c.storage_generation,
                                    c.model_generation,
                                    ranges.ranges.size()};
  for (const auto& b : ranges.ranges) {
    result.push_back(b.address);
    result.push_back(b.bytes);
  }
  result.push_back(ranges.scalars.size());
  result.insert(result.end(), ranges.scalars.begin(), ranges.scalars.end());
  result.push_back(c.paid_ranges.size());
  for (const auto& b : c.paid_ranges) {
    result.push_back(b.address);
    result.push_back(b.bytes);
  }
  for (const auto& l : c.layers) {
    result.push_back(static_cast<std::uint64_t>(l.attention.kind));
    if (l.indexer) {
      result.push_back(static_cast<std::uint64_t>(l.indexer->scores.kind));
      result.push_back(static_cast<std::uint64_t>(l.indexer->select.kind));
    }
  }
  return result;
}
std::vector<std::uint64_t> ProgressIdentity(const Progress& x) {
  std::vector<std::uint64_t> result{x.first, x.next_layer, x.storage_generation, x.model_generation,
                                    static_cast<std::uint64_t>(x.phase)};
  result.insert(result.end(), x.compressed.begin(), x.compressed.end());
  result.insert(result.end(), x.indexed.begin(), x.indexed.end());
  return result;
}
Result Paid(const Chunk& c) {
  if (c.paid_ranges.empty()) return Fail("no mapped catalog inventory for paid operands");
  for (auto b : c.paid_ranges)
    if (!Range(b)) return Fail("invalid paid catalog byte range");
  const auto inventory = Operands(c);
  for (const auto& b : inventory.ranges) {
    if (!Range(b) || std::ranges::none_of(c.paid_ranges, [b](Buffer r) { return Covers(r, b); }))
      return Fail("borrowed operand is outside paid mapped catalog ranges");
  }
  for (const auto& w : inventory.writes) {
    for (const auto& r : inventory.immutable) {
      if (Overlap(w, r))
        return Fail("recipe output/scratch overwrites immutable weights, tokens or constants");
    }
  }
  for (const auto& w : inventory.scratch_writes) {
    for (const auto& state : inventory.persistent) {
      if (Overlap(w, state))
        return Fail("scratch producer overwrites persistent original cache/state");
    }
  }
  for (std::uint32_t i = 0; i < c.layers.size(); ++i) {
    Inventory a;
    a.Add(c.layers[i]);
    for (std::uint32_t j = 0; j < i; ++j) {
      Inventory b;
      b.Add(c.layers[j]);
      for (const auto& x : a.persistent)
        for (const auto& y : b.persistent) {
          if (Overlap(x, y)) return Fail("distinct layers share mutable original cache/state");
        }
    }
  }
  return {};
}
Result Disjoint(std::initializer_list<Buffer> writes, std::initializer_list<Buffer> live) {
  for (const auto& w : writes) {
    if (Absent(w)) continue;
    for (const auto& r : live)
      if (!Absent(r) && Overlap(w, r))
        return Fail("stage output overwrites a still-live earlier producer");
  }
  return {};
}
Result Q8(const kg::Ds4Q8Product& d, std::uint32_t input, std::uint32_t output, kg::Ds4Q8Path path,
          std::uint64_t generation, bool prepared) {
  if (!Matrix(d.input, kRows, input, 4) || !Matrix(d.output, kRows, output, 4) ||
      d.weights.columns != input || d.weights.rows != output || d.path != path ||
      d.generation != generation || d.prepared != prepared ||
      d.quantized.generation != generation || d.quantized.source != d.input.storage.data ||
      d.quantized.rows != kRows || d.quantized.columns != input ||
      d.weights.codes.data == nullptr || d.weights.scales.data == nullptr)
    return Fail("Q8 product differs from original aligned wide dispatch/producer");
  return CheckedStages({kg::CheckDs4Q8Product(d)});
}
Result F16(const kg::Ds4F16Product& d, std::uint32_t input, std::uint32_t output) {
  if (!Matrix(d.input, kRows, input, 2) || !Matrix(d.output, kRows, output, 4) ||
      !Matrix(d.weights, output, input, 2))
    return Fail("wrong original wide F16 product geometry");
  return CheckedStages({kg::CheckDs4F16Product(d)});
}
Result Pre(const kg::Ds4HcPre& d, const kg::Ds4F16Product& p) {
  if (d.width != 4096 || d.coefficients.rows != kRows || d.coefficients.iterations != 20 ||
      d.coefficients.epsilon != 1.0e-6F || !Same(d.coefficients.mix, p.output.storage))
    return Fail("HC coefficients are not the current original projection");
  return CheckedStages({kg::CheckDs4HcPre(d)});
}
Result Norm(const kg::Ds4Rms& d, std::uint32_t width, bool plain, bool dual) {
  if (d.rows != kRows || d.width != width || d.epsilon != 1.0e-6F || (plain != Absent(d.weights)) ||
      (plain && (!Absent(d.values) || !Absent(d.q8_d4))) ||
      (dual && (Absent(d.values) || Absent(d.values_f16) || Absent(d.q8_d4))))
    return Fail("missing original wide RMS producer outputs");
  return CheckedStages({kg::CheckDs4Rms(d)});
}
Result CompressionOf(const model::Dsv4Profile& p, const Chunk& c, const Compression& d,
                     std::uint32_t ratio, kg::Ds4CacheKind kind, const kg::Ds4Rms& input) {
  const auto width = kind == kg::Ds4CacheKind::kKv512 ? 512U : 128U;
  const auto coff = ratio == 4 ? 2U : 1U;
  auto rope = OriginalRope(p, ratio, c.first);
  if (!rope) return std::unexpected(rope.error());
  if (auto checked = F16(d.kv_projection, 4096, coff * width); !checked) return checked;
  if (auto checked = F16(d.score_projection, 4096, coff * width); !checked) return checked;
  const auto& x = d.chunk;
  if (x.state.kind != kind || x.state.ratio != ratio || x.first != c.first || x.tokens != kRows ||
      x.before != c.first / ratio || x.capacity < (c.first + kRows) / ratio ||
      x.rms_epsilon != p.rms_eps || !CompRope(x.rope, *rope) ||
      !Same(input.values_f16, d.kv_projection.input.storage) ||
      !Same(input.values_f16, d.score_projection.input.storage) ||
      !Same(x.kv, d.kv_projection.output.storage) ||
      !Same(x.score, d.score_projection.output.storage) || Absent(x.codes) || Absent(x.scales))
    return Fail("compressor is not a complete current packed producer");
  auto plan = kg::PlanDs4Comp(x);
  if (!plan) return Fail(plan.error().detail);
  if (plan->emitted != kRows / ratio || plan->after != (c.first + kRows) / ratio ||
      plan->refresh_required != (ratio == 4))
    return Fail("unexpected original compressor frontier");
  if (ratio == 4) {
    if (!d.refresh || !d.refresh_kv || !d.refresh_score)
      return Fail("ratio4 requires both original last-four small products and refresh");
    const auto& r = *d.refresh;
    const auto tail = input.values.address + (static_cast<std::uint64_t>(kRows - 4) * 4096 * 4);
    for (const auto* v : {&*d.refresh_kv, &*d.refresh_score}) {
      if (!Matrix(v->input, 4, 4096, 4) || Address(v->input.storage.data) != tail ||
          !Matrix(v->weights, coff * width, 4096, 2) || !Matrix(v->output, 4, coff * width, 4))
        return Fail("refresh substituted wide rounded projections for original F32 small products");
      if (auto checked = kg::CheckDs4F16Vector(*v); !checked) return Fail(checked.error().detail);
    }
    if (!Same(View(d.refresh_kv->weights.storage), d.kv_projection.weights.storage) ||
        !Same(View(d.refresh_score->weights.storage), d.score_projection.weights.storage) ||
        !Same(r.kv, d.refresh_kv->output.storage) ||
        !Same(r.score, d.refresh_score->output.storage) || !Same(r.state.kv, x.state.kv) ||
        !Same(r.state.score, x.state.score) || r.state.kind != kind || r.state.ratio != ratio ||
        !Same(r.ape, x.ape) || r.ape_format != x.ape_format || r.first != c.first + kRows - 4)
      return Fail("refresh does not update the same original compressor state");
    if (auto checked = kg::CheckDs4CompRefresh(r); !checked) return Fail(checked.error().detail);
  } else if (d.refresh || d.refresh_kv || d.refresh_score) {
    return Fail("ratio128 has no ratio4 refresh stage");
  }
  return {};
}
}  // namespace

std::expected<kg::Ds4ProductRope, std::string> OriginalRope(const model::Dsv4Profile& p,
                                                            std::uint32_t ratio,
                                                            std::uint32_t first, bool inverse) {
  if (auto checked = Profile(p); !checked) return std::unexpected(checked.error());
  if (ratio != 0 && ratio != 4 && ratio != 128) return std::unexpected("unknown Flash RoPE domain");
  const bool compressed = ratio != 0;
  const float scale = compressed ? 1.0F / p.rope_scale : 1.0F;
  const float extension = compressed && p.rope_scale > 1.0F ? 1.0F : 0.0F;
  float attention = 1.0F;
  if (extension != 0.0F && scale > 0.0F) attention /= 1.0F + (0.1F * std::log(1.0F / scale));
  if (!std::isfinite(attention) || attention <= 0)
    return std::unexpected("invalid original YaRN factor");
  return kg::Ds4ProductRope{.first = first,
                            .step = 1,
                            .original_context = compressed ? p.yarn_original_context : 0,
                            .rotary = 64,
                            .base = compressed ? p.compress_rope_base : p.rope_base,
                            .scale = scale,
                            .extension = extension,
                            .attention = attention,
                            .beta_fast = p.yarn_beta_fast,
                            .beta_slow = p.yarn_beta_slow,
                            .inverse = inverse};
}

namespace {
Result IndexerOf(const model::Dsv4Profile& p, const Chunk& c, const Layer& l) {
  if (!l.indexer) return Fail("missing original indexer recipe");
  const auto& d = *l.indexer;
  if (auto check =
          CompressionOf(p, c, d.compression, 4, kg::Ds4CacheKind::kIndexer128, l.attention_norm);
      !check)
    return check;
  auto rope = OriginalRope(p, 4, c.first);
  if (!rope) return std::unexpected(rope.error());
  const auto cells = (c.first + kRows) / 4;
  if (!Matrix(d.query_conversion.input, kRows, 1024, 4) ||
      !Matrix(d.query_conversion.output, kRows, 1024, 2) ||
      !Same(View(d.query_conversion.input.storage), l.qkv_norm.query_output.storage) ||
      !Linked(d.query_conversion.output, d.query_projection.input) ||
      !Same(l.attention_norm.values_f16, d.weight_projection.input.storage) ||
      !Same(View(d.query_projection.output.storage), d.query_rope.input.storage) ||
      !Matrix(d.query_rope.input, kRows, 8192, 4) || d.query_rope.heads != 64 ||
      d.query_rope.head_width != 128 || d.query_rope.normalize || !Rope(d.query_rope.rope, *rope) ||
      d.query_qat.kind != kg::Ds4CacheKind::kIndexer128 || d.query_qat.rows != kRows * 64 ||
      !Same(d.query_qat.values, d.query_rope.input.storage) ||
      !Same(d.scores.query, d.query_qat.values) || !Same(d.scores.query_codes, d.query_qat.codes) ||
      !Same(d.scores.query_scales, d.query_qat.scales) ||
      !Same(d.scores.weights, d.weight_projection.output.storage) || d.scores.tokens != kRows ||
      d.scores.cells != cells || d.scores.cells_per_bank != cells || d.scores.banks != 1 ||
      d.scores.first != c.first || d.scores.ratio != 4 || d.scores.score_band != cells ||
      d.scores.consecutive_bank != UINT32_MAX ||
      d.scores.scale != 1.0F / std::sqrt(128.0F * 64.0F) || !d.scores.causal ||
      d.scores.quality_mode || !Absent(d.scores.positions) || !Absent(d.scores.bank_ids) ||
      !Absent(d.scores.layer_scalars) || Absent(d.scores.key_codes) ||
      Absent(d.scores.key_scales) ||
      (d.scores.kind != kg::Ds4IndexerScoreKind::kOriginalDefault &&
       d.scores.kind != kg::Ds4IndexerScoreKind::kMxf4) ||
      d.select.tokens != kRows || d.select.cells != cells || d.select.score_band != cells ||
      d.select.top_k != 512 || !Same(d.select.scores, d.scores.scores) ||
      !Same(d.select.diagnostics, d.scores.diagnostics) || !Absent(d.select.layer_scalars) ||
      !Same(d.select.selected, l.attention.selected))
    return Fail("indexer does not consume the complete original current producer chain");
  if (d.compression.chunk.codes.address !=
          d.scores.key_codes.address + (static_cast<std::uint64_t>(c.first / 4) * 64) ||
      d.compression.chunk.scales.address !=
          d.scores.key_scales.address + (static_cast<std::uint64_t>(c.first / 4) * 16))
    return Fail("indexer packed emission does not advance the same cache");
  if (auto check = F16(d.query_projection, 1024, 8192); !check) return check;
  if (auto check = F16(d.weight_projection, 4096, 64); !check) return check;
  if (auto check =
          CheckedStages({kg::CheckDs4F16Conversion(d.query_conversion),
                         kg::CheckDs4HeadRope(d.query_rope), kg::CheckDs4CacheQat(d.query_qat),
                         kg::CheckDs4IndexerScores(d.scores), kg::CheckDs4IndexerSelect(d.select)});
      !check)
    return check;
  auto scores = kg::Ds4IndexerScoreScratchBytes(d.scores);
  auto selected = kg::Ds4IndexerSelectScratchBytes(d.select);
  if (!scores || !selected || d.scores.scratch.bytes < *scores ||
      d.select.scratch.bytes < *selected)
    return Fail("missing paid complete indexer score/select scratch");
  return {};
}
Result LayerOf(const model::Dsv4Profile& p, const Chunk& c, const Layer& l) {
  const auto ratio = p.compress_ratios[l.index];
  auto rope = OriginalRope(p, ratio, c.first);
  auto inverse = OriginalRope(p, ratio, c.first, true);
  if (!rope || !inverse) return Fail("invalid model RoPE");
  if (l.ratio != ratio || l.compression.has_value() != (ratio != 0) ||
      l.indexer.has_value() != (ratio == 4) || l.final_sum.has_value() != (l.index == kLayers - 1))
    return Fail("missing or surplus original layer stage");
  if (auto check = Norm(l.hc_attention_rms, 16384, true, false); !check) return check;
  if (auto check = F16(l.hc_attention_projection, 16384, 24); !check) return check;
  if (auto check = Pre(l.hc_attention_pre, l.hc_attention_projection); !check) return check;
  if (auto check = Norm(l.attention_norm, 4096, false, true); !check) return check;
  if (!Same(l.hc_attention_rms.source, l.hc_attention_pre.residual) ||
      !Same(l.hc_attention_rms.values_f16, l.hc_attention_projection.input.storage) ||
      !Same(l.attention_norm.source, l.hc_attention_pre.values) ||
      !Same(l.attention_norm.values, l.query_a.input.storage) ||
      !Same(l.attention_norm.values, l.kv_projection.input.storage) ||
      !Same(l.attention_norm.q8_d4, l.query_a.quantized.storage) ||
      !Same(l.attention_norm.q8_d4, l.kv_projection.quantized.storage))
    return Fail("attention HC/norm products have stale producer wiring");
  if (auto check = Q8(l.query_a, 4096, 1024, kg::Ds4Q8Path::kMmq, c.storage_generation, true);
      !check)
    return check;
  if (auto check = Q8(l.kv_projection, 4096, 512, kg::Ds4Q8Path::kMmq, c.storage_generation, true);
      !check)
    return check;
  if (auto check =
          Q8(l.query_b, 1024, 32768, kg::Ds4Q8Path::kDenseD2r, c.storage_generation, false);
      !check)
    return check;
  if (!Linked(l.query_a.output, l.qkv_norm.query) ||
      !Linked(l.kv_projection.output, l.qkv_norm.kv) ||
      !Same(View(l.qkv_norm.query_output.storage), l.query_b.input.storage) ||
      !Matrix(l.qkv_norm.query_output, kRows, 1024, 4) ||
      !Matrix(l.qkv_norm.kv_output, kRows, 512, 4) || l.qkv_norm.epsilon != p.rms_eps ||
      !Same(View(l.query_b.output.storage), l.query_rope.input.storage) ||
      !Same(View(l.qkv_norm.kv_output.storage), l.kv_rope.input.storage) ||
      !Matrix(l.query_rope.input, kRows, 32768, 4) || !Matrix(l.kv_rope.input, kRows, 512, 4) ||
      l.query_rope.heads != 64 || l.query_rope.head_width != 512 || !l.query_rope.normalize ||
      l.kv_rope.heads != 1 || l.kv_rope.head_width != 512 || l.kv_rope.normalize ||
      l.query_rope.epsilon != p.rms_eps || !Rope(l.query_rope.rope, *rope) ||
      !Rope(l.kv_rope.rope, *rope) || l.raw_qat.kind != kg::Ds4CacheKind::kKv512 ||
      l.raw_qat.rows != kRows || !Same(l.raw_qat.values, l.kv_rope.input.storage) ||
      !Absent(l.raw_qat.codes) || !Absent(l.raw_qat.scales) ||
      !Same(l.raw_store.source, l.raw_qat.values) || l.raw_store.first != c.first ||
      l.raw_store.rows != kRows || l.raw_store.cells != 4352)
    return Fail("Q/KV RMS, RoPE, QAT and ring store are not the original complete chain");
  if (auto check =
          CheckedStages({kg::CheckDs4QkvNorm(l.qkv_norm), kg::CheckDs4HeadRope(l.query_rope),
                         kg::CheckDs4HeadRope(l.kv_rope), kg::CheckDs4CacheQat(l.raw_qat),
                         kg::CheckDs4CacheRawStore(l.raw_store)});
      !check)
    return check;
  if (l.compression) {
    if (auto check =
            CompressionOf(p, c, *l.compression, ratio, kg::Ds4CacheKind::kKv512, l.attention_norm);
        !check)
      return check;
  }
  if (l.indexer) {
    if (auto check = IndexerOf(p, c, l); !check) return check;
  }
  const auto& a = l.attention;
  const auto after = ratio != 0 ? (c.first + kRows) / ratio : 0;
  const auto raw_count = c.first == 0 ? kRows : kRows + 128;
  const auto raw_start = c.first == 0 ? 0U : c.first - 128;
  const bool static_mixed = c.first == 0 && ratio == 128;
  auto domain = kg::Ds4AttentionDomain::kMixedRing;
  if (ratio == 4)
    domain = kg::Ds4AttentionDomain::kIndexedRing;
  else if (static_mixed)
    domain = kg::Ds4AttentionDomain::kMixedPrefill;
  if (a.tokens != kRows || a.first != c.first || a.consecutive_first != c.first || a.banks != 1 ||
      a.window != 128 || a.raw_cells != (static_mixed ? kRows : 4352U) ||
      a.raw_count != raw_count || a.raw_start != raw_start || a.compressed_count != after ||
      a.compressed_cells != after || a.ratio != (ratio == 0 ? 1U : ratio) || a.top_k != 512 ||
      a.domain != domain ||
      (a.kind != kg::Ds4AttentionKind::kOriginalDefault &&
       a.kind != kg::Ds4AttentionKind::kTokenTile) ||
      a.allow_multisequence_heads8 || a.quality_mode ||
      !Same(a.query, l.query_rope.input.storage) ||
      !Same(a.raw, static_mixed ? l.raw_qat.values : l.raw_store.ring) || !Absent(a.mask) ||
      !Absent(a.positions) || !Absent(a.bank_ids) || !Absent(a.draft_raw_count) ||
      !Absent(a.decode_scalars) || !Absent(a.layer_scalars))
    return Fail("attention recipe differs from original consecutive single-bank wide dispatch");
  if (ratio != 0) {
    if (!l.compression) return Fail("missing original compressed KV producer");
    if (Absent(a.compressed_codes) || Absent(a.compressed_scales) || Absent(a.decode_table) ||
        l.compression->chunk.codes.address !=
            a.compressed_codes.address + (static_cast<std::uint64_t>(c.first / ratio) * 704) ||
        l.compression->chunk.scales.address !=
            a.compressed_scales.address + (static_cast<std::uint64_t>(c.first / ratio) * 28))
      return Fail("attention is not bound to the same original packed KV producer");
  } else if (!Absent(a.compressed) || !Absent(a.compressed_codes) || !Absent(a.compressed_scales) ||
             !Absent(a.selected)) {
    return Fail("raw-only layer has a compressed-cache consumer");
  }
  auto scratch = kg::PlanDs4AttentionScratch(a, kg::Ds4AttentionKind::kTokenTile, c.device_sms);
  if (!scratch || a.scratch.bytes < scratch->bytes)
    return Fail("unpaid original full token-tile chain");
  if (auto check = kg::CheckDs4Attention(a); !check) return Fail(check.error().detail);
  if (!Same(a.output, l.output_a.heads.storage) || !Matrix(l.output_a.heads, kRows, 32768, 4) ||
      !Matrix(l.output_a.low, kRows, 8192, 4) || !Rope(l.output_a.rope, *inverse) ||
      l.output_a.generation != c.storage_generation ||
      l.output_a.quantized.generation != c.storage_generation ||
      l.output_a.quantized.source != l.output_a.low.storage.data ||
      !Linked(l.output_a.low, l.output_b.input) ||
      !Same(View(l.output_a.quantized.storage), l.output_b.quantized.storage))
    return Fail("out-a inverse RoPE/low/D4 producer is missing");
  if (auto check = kg::CheckDs4OutA(l.output_a); !check) return Fail(check.error().detail);
  if (auto check = Q8(l.output_b, 8192, 4096, kg::Ds4Q8Path::kMmq, c.storage_generation, true);
      !check)
    return check;
  if (!Same(l.attention_expand.block, l.output_b.output.storage) ||
      !Same(l.attention_expand.residual, l.hc_attention_pre.residual) ||
      !Same(l.attention_expand.split, l.hc_attention_pre.coefficients.split) ||
      !Absent(l.attention_expand.add) || !Absent(l.attention_expand.moe_unsummed) ||
      l.attention_expand.rows != kRows || l.attention_expand.width != 4096 ||
      l.attention_expand.epsilon != p.rms_eps ||
      !Same(l.attention_expand.values_f16, l.hc_ffn_projection.input.storage) ||
      !Same(l.attention_expand.values, l.hc_ffn_pre.residual))
    return Fail("attention expansion lost original fused current HC mirror");
  if (auto check = kg::CheckDs4HcExpand(l.attention_expand); !check)
    return Fail(check.error().detail);
  if (auto check = F16(l.hc_ffn_projection, 16384, 24); !check) return check;
  if (auto check = Pre(l.hc_ffn_pre, l.hc_ffn_projection); !check) return check;
  if (auto check = Norm(l.ffn_norm, 4096, false, true); !check) return check;
  if (auto check = F16(l.router_projection, 4096, 256); !check) return check;
  if (!Same(l.ffn_norm.source, l.hc_ffn_pre.values) ||
      !Same(l.ffn_norm.values_f16, l.router_projection.input.storage) ||
      !Same(l.router.logits, l.router_projection.output.storage) ||
      (l.index < p.hash_layers ? !Same(l.router.tokens, c.embedding.tokens)
                               : !Absent(l.router.tokens)) ||
      l.router.rows != kRows || l.router.select != kg::Ds4RouterSelect::kWarp ||
      l.router.hash_rows != (l.index < p.hash_layers ? p.vocab : 0) ||
      (Absent(l.router.hash) != (l.index >= p.hash_layers)))
    return Fail("router is not original F16 projection/current token hash/bias-only selection");
  const auto& m = l.routed;
  if (m.shape.rows != kRows || m.shape.input != 4096 || m.shape.middle != 2048 ||
      m.shape.output != 4096 || m.tier != kg::Ds4MoeTier::kDirect ||
      m.generation != c.storage_generation || !Same(m.input, l.ffn_norm.values) ||
      !Same(m.selected, l.router.selected) || !Same(m.weights, l.router.weights) ||
      m.selected_stride != 6 || m.weight_stride != 6 || !Absent(m.compact_ids) ||
      !Absent(m.compact_weights) || !Absent(m.sum) || m.producer.kind != kg::Ds4MoeQuant::kD4 ||
      m.producer.generation != c.storage_generation ||
      m.producer.source_address != l.ffn_norm.values.address || m.producer.rows != kRows ||
      m.producer.width != 4096 || !Same(m.producer.storage, l.ffn_norm.q8_d4))
    return Fail("routed path substituted original Direct G1/D2S6/Q2 or its current D4 producer");
  if (auto check = CheckedStages({kg::CheckDs4Router(l.router), kg::CheckDs4Moe(m),
                                  kg::CheckDs4SharedSwiglu(l.shared_swiglu)});
      !check)
    return check;
  if (auto check =
          Q8(l.shared_gate, 4096, 2048, kg::Ds4Q8Path::kDenseD2r, c.storage_generation, true);
      !check)
    return check;
  if (auto check =
          Q8(l.shared_up, 4096, 2048, kg::Ds4Q8Path::kDenseD2r, c.storage_generation, true);
      !check)
    return check;
  if (auto check =
          Q8(l.shared_down, 2048, 4096, kg::Ds4Q8Path::kDenseD2r, c.storage_generation, false);
      !check)
    return check;
  if (!Same(l.ffn_norm.values, l.shared_gate.input.storage) ||
      !Same(l.ffn_norm.values, l.shared_up.input.storage) ||
      !Same(l.ffn_norm.q8_d4, l.shared_gate.quantized.storage) ||
      !Same(l.ffn_norm.q8_d4, l.shared_up.quantized.storage) ||
      !Same(l.shared_swiglu.gate, l.shared_gate.output.storage) ||
      !Same(l.shared_swiglu.up, l.shared_up.output.storage) ||
      !Same(l.shared_swiglu.output, l.shared_down.input.storage) || l.shared_swiglu.rows != kRows ||
      l.shared_swiglu.width != 2048 || !Same(l.ffn_expand.add, l.shared_down.output.storage) ||
      !Same(l.ffn_expand.residual, l.hc_ffn_pre.residual) ||
      !Same(l.ffn_expand.split, l.hc_ffn_pre.coefficients.split) || l.ffn_expand.width != 4096 ||
      l.ffn_expand.rows != kRows || l.ffn_expand.epsilon != p.rms_eps)
    return Fail("shared FFN/HC expansion does not consume original current operands");
  if (l.final_sum) {
    if (!Same(l.final_sum->slots, m.down) || l.final_sum->rows != kRows ||
        l.final_sum->width != 4096 || !Same(l.final_sum->output, l.ffn_expand.block) ||
        !Absent(l.ffn_expand.values_f16) || !Absent(l.ffn_expand.moe_unsummed))
      return Fail("last layer requires explicit guarded six-slot sum/plain HC expansion");
    if (auto check = kg::CheckDs4MoeSum(*l.final_sum); !check) return Fail(check.error().detail);
  } else if (!Same(l.ffn_expand.moe_unsummed, m.down) || Absent(l.ffn_expand.values_f16) ||
             !Absent(l.ffn_expand.block)) {
    return Fail("preceding layer requires original guarded MoE/shared fused next-HC/RMS");
  }
  if (auto check = kg::CheckDs4HcExpand(l.ffn_expand); !check) return Fail(check.error().detail);
  // The residual and split survive both attention and FFN. Distinct
  // temporary stages may reuse scratch only after their last consumer.
  if (auto check =
          Disjoint({l.hc_attention_pre.values, l.attention_norm.values, l.attention_norm.values_f16,
                    l.attention_norm.q8_d4, View(l.query_a.output.storage),
                    View(l.kv_projection.output.storage), View(l.query_b.output.storage), a.output,
                    View(l.output_a.low.storage)},
                   {l.hc_attention_pre.residual, l.hc_attention_pre.coefficients.split});
      !check)
    return check;
  if (auto check = Disjoint({l.ffn_norm.values, l.ffn_norm.values_f16, l.ffn_norm.q8_d4,
                             View(l.router_projection.output.storage), m.down,
                             View(l.shared_gate.output.storage), View(l.shared_up.output.storage),
                             l.shared_swiglu.output, View(l.shared_down.output.storage)},
                            {l.hc_ffn_pre.residual, l.hc_ffn_pre.coefficients.split});
      !check)
    return check;
  if (auto check =
          Disjoint({View(l.query_b.output.storage), a.output, View(l.output_a.low.storage)},
                   {l.attention_norm.values, l.attention_norm.values_f16});
      !check)
    return check;
  if (auto check =
          Disjoint({View(l.kv_projection.output.storage)}, {View(l.query_a.output.storage)});
      !check)
    return check;
  if (auto check = Disjoint({View(l.query_b.output.storage)}, {View(l.qkv_norm.kv_output.storage)});
      !check)
    return check;
  if (auto check = Disjoint({a.scratch, a.output, View(l.output_a.low.storage)},
                            {View(l.qkv_norm.query_output.storage), l.attention_norm.values,
                             l.attention_norm.values_f16});
      !check)
    return check;
  if (l.indexer) {
    const auto& d = *l.indexer;
    if (auto check = Disjoint({d.scores.scores, d.scores.scratch, d.select.scratch},
                              {d.scores.query, d.scores.weights, d.scores.query_codes,
                               d.scores.query_scales, d.scores.key_codes, d.scores.key_scales});
        !check)
      return check;
  }
  return {};
}
Result FrontierOf(const model::Dsv4Profile& p, const Chunk& c) {
  if (!c.frontier) return Fail("missing original final head recipe");
  const auto& d = *c.frontier;
  const auto& last = c.layers.back();
  const auto row =
      last.ffn_expand.values.address + (static_cast<std::uint64_t>(kRows - 1) * 16384 * 4);
  if (d.hc_rms.source.address != row || d.hc_rms.rows != 1 || d.hc_rms.width != 16384 ||
      !Absent(d.hc_rms.weights) || Absent(d.hc_rms.values) || !Absent(d.hc_rms.values_f16) ||
      !Absent(d.hc_rms.q8_d4) || d.hc_rms.epsilon != p.rms_eps ||
      !Same(d.hc_rms.values, d.hc_projection.input.storage) ||
      !Matrix(d.hc_projection.input, 1, 16384, 4) ||
      !Matrix(d.hc_projection.weights, 4, 16384, 2) || !Matrix(d.hc_projection.output, 1, 4, 4) ||
      !Same(d.hc_weights.pre, d.hc_projection.output.storage) || d.hc_weights.rows != 1 ||
      d.hc_weights.epsilon != p.hc_eps || !Same(d.collapse.weights, d.hc_weights.values) ||
      d.collapse.residual.address != row || d.collapse.rows != 1 || d.collapse.width != 4096 ||
      d.collapse.weight_stride != 4 || !Same(d.norm.source, d.collapse.values) ||
      d.norm.width != 4096 || d.norm.rows != 1 || d.norm.epsilon != p.rms_eps ||
      Absent(d.norm.weights) || Absent(d.norm.values) || !Absent(d.norm.values_f16) ||
      !Absent(d.norm.q8_d4) || !Same(d.norm.values, d.projection.input.storage) ||
      !Matrix(d.projection.input, 1, 4096, 4) || !Matrix(d.projection.output, 1, p.vocab, 4) ||
      d.projection.weights.rows != p.vocab || d.projection.weights.columns != 4096 ||
      d.projection.path != kg::Ds4Q8VectorPath::kAligned || d.projection.prepared ||
      d.projection.generation != c.storage_generation ||
      d.projection.quantized.generation != c.storage_generation)
    return Fail("frontier substituted final-row HC collapse/full original output head");
  return CheckedStages({kg::CheckDs4Rms(d.hc_rms), kg::CheckDs4F16Vector(d.hc_projection),
                        kg::CheckDs4HcHeadWeights(d.hc_weights), kg::CheckDs4HcWeighted(d.collapse),
                        kg::CheckDs4Rms(d.norm), kg::CheckDs4Q8Vector(d.projection)});
}
}  // namespace

Result CheckChunk(const model::Dsv4Profile& p, const Chunk& c) {
  if (auto check = Profile(p); !check) return check;
  if (c.artifact_id != kCommunityArtifact || (c.first != 0 && c.first != kRows) ||
      c.context != 2 * kRows || c.layers.size() != kLayers || c.device_sms == 0 ||
      c.storage_generation == 0 || c.model_generation == 0 ||
      c.frontier.has_value() != (c.first == kRows))
    return Fail("incomplete initial all43/two4096/full-head8K recipe");
  if (!Matrix(c.embedding.weights, p.vocab, 4096, 2) ||
      !Matrix(c.embedding.output, kRows, 16384, 4) || c.embedding.hyper_connections != 4)
    return Fail("embedding is not the canonical F16 four-HC producer");
  if (auto check = kg::CheckDs4Embedding(c.embedding); !check) return Fail(check.error().detail);
  for (std::uint32_t i = 0; i < kLayers; ++i) {
    const auto& l = c.layers[i];
    if (l.index != i) return Fail("missing, repeated or reordered original layer");
    if (auto check = LayerOf(p, c, l); !check)
      return std::unexpected("layer " + std::to_string(i) + ": " + check.error());
    if (i == 0) {
      if (!Same(l.hc_attention_pre.residual, c.embedding.output.storage))
        return Fail("first layer does not consume current embedding");
    } else {
      const auto& before = c.layers[i - 1].ffn_expand;
      if (!Same(l.hc_attention_pre.residual, before.values) ||
          !Same(View(l.hc_attention_projection.input.storage), before.values_f16))
        return Fail("HC layer boundary does not consume preceding fused expansion");
    }
  }
  if (c.frontier) {
    if (auto check = FrontierOf(p, c); !check) return check;
  }
  return Paid(c);
}
Result CheckMappedOperands(const Chunk& c) { return Paid(c); }

Result CheckHashTables(const model::Dsv4Profile& p, const Chunk& c,
                       std::span<const HashTable> tables) {
  if (auto check = Profile(p); !check) return check;
  if (c.layers.size() != kLayers || tables.size() != p.hash_layers)
    return Fail("missing authenticated original hash table");
  for (std::uint32_t i = 0; i < p.hash_layers; ++i) {
    const auto& t = tables[i];
    if (t.layer != i || !Same(t.device, c.layers[i].router.hash) ||
        t.entries.size() != static_cast<std::uint64_t>(p.vocab) * 6 ||
        t.device.bytes < t.entries.size_bytes())
      return Fail("hash table identity/capacity differs from bound model");
    for (std::uint32_t row = 0; row < p.vocab;) {
      const auto rows = std::min(kRows, p.vocab - row);
      auto check = kg::CheckDs4MoeIds(
          t.entries.subspan(static_cast<std::size_t>(row) * 6, static_cast<std::size_t>(rows) * 6),
          rows, 6);
      if (!check) return Fail(check.error().detail);
      row += rows;
    }
  }
  return {};
}

std::expected<Progress, std::string> Begin(const model::Dsv4Profile& p, const Chunk& c,
                                           const Progress* previous) {
  if (auto check = CheckChunk(p, c); !check) return std::unexpected(check.error());
  Progress result;
  result.first = c.first;
  result.storage_generation = c.storage_generation;
  result.model_generation = c.model_generation;
  if (c.first == 0) {
    if (previous != nullptr)
      return std::unexpected("fresh state cannot reuse a previous progress ledger");
  } else {
    if (previous == nullptr || previous->phase != Phase::kComplete || previous->first != 0 ||
        previous->next_layer != kLayers || previous->storage_generation != c.storage_generation ||
        previous->model_generation != c.model_generation)
      return std::unexpected("second chunk requires completed original first chunk/current state");
    if (previous->completion_owner_ != nullptr &&
        (previous->completion_identity_ != ProgressIdentity(*previous) ||
         !previous->completion_token_ || previous->completion_token_->poisoned ||
         previous->completion_token_->owner != previous->completion_owner_ ||
         previous->completion_serial_ != previous->completion_token_->serial))
      return std::unexpected("completed native ledger was modified before continuation");
    result.compressed = previous->compressed;
    result.indexed = previous->indexed;
    for (std::uint32_t i = 0; i < kLayers; ++i) {
      const auto ratio = p.compress_ratios[i];
      if (result.compressed[i] != (ratio == 0 ? 0U : c.first / ratio) ||
          result.indexed[i] != (ratio == 4 ? c.first / 4 : 0U))
        return std::unexpected("previous completed cache counts do not match new chunk frontier");
    }
    if (previous->completion_owner_ != nullptr) {
      result.completion_owner_ = previous->completion_owner_;
      result.completion_identity_ = ProgressIdentity(result);
      result.completion_token_ = previous->completion_token_;
      result.completion_serial_ = previous->completion_serial_;
    }
  }
  return result;
}
Result CheckProgress(const Chunk& c, const Progress& x, Phase phase, std::uint32_t layer) {
  if ((c.first != 0 && c.first != kRows) || c.context != 2 * kRows || c.storage_generation == 0 ||
      c.model_generation == 0 || x.phase != phase || x.first != c.first ||
      x.storage_generation != c.storage_generation || x.model_generation != c.model_generation ||
      x.next_layer > kLayers ||
      (phase == Phase::kLayers && (layer != x.next_layer || layer >= kLayers)) ||
      (phase == Phase::kEmbedding && x.next_layer != 0) ||
      (phase == Phase::kFrontier && (x.next_layer != kLayers || !c.frontier)) ||
      c.layers.size() != kLayers)
    return Fail("wrong, stale, failed or out-of-order completed progress");
  for (std::uint32_t i = 0; i < kLayers; ++i) {
    const auto ratio = c.layers[i].ratio;
    const auto expected = OriginalRatio(i);
    if (ratio != expected || c.layers[i].index != i)
      return Fail("progress recipe layer schedule changed");
    const auto end = c.first + (i < x.next_layer ? kRows : 0U);
    if (x.compressed[i] != (ratio == 0 ? 0U : end / ratio) ||
        x.indexed[i] != (ratio == 4 ? end / 4 : 0U))
      return Fail("host cache count advanced before its layer fence");
  }
  return {};
}
Result CompleteEmbedding(const Chunk& c, Progress& x) {
  if (auto check = CheckProgress(c, x, Phase::kEmbedding); !check) return check;
  x.phase = Phase::kLayers;
  return {};
}
Result CompleteLayer(const Chunk& c, std::uint32_t layer, Progress& x) {
  if (auto check = CheckProgress(c, x, Phase::kLayers, layer); !check) return check;
  const auto ratio = c.layers[layer].ratio;
  x.compressed[layer] = ratio == 0 ? 0U : (c.first + kRows) / ratio;
  x.indexed[layer] = ratio == 4 ? (c.first + kRows) / 4 : 0U;
  ++x.next_layer;
  if (x.next_layer == kLayers) x.phase = c.frontier ? Phase::kFrontier : Phase::kComplete;
  return {};
}
Result CompleteFrontier(const Chunk& c, Progress& x) {
  if (auto check = CheckProgress(c, x, Phase::kFrontier); !check) return check;
  x.phase = Phase::kComplete;
  return {};
}
void Poison(Progress& x) {
  x.phase = Phase::kPoisoned;
  if (x.completion_token_) x.completion_token_->poisoned = true;
}

struct RuntimeAccess {
  static Result CheckCompletion(const Progress& x, const kg::LaunchContext& launch, bool fresh) {
    if (fresh && x.completion_owner_ == nullptr && x.completion_identity_.empty() &&
        !x.completion_token_)
      return {};
    if (x.completion_owner_ != &launch || x.completion_identity_ != ProgressIdentity(x) ||
        !x.completion_token_ || x.completion_token_->owner != &launch ||
        x.completion_token_->poisoned || x.completion_serial_ != x.completion_token_->serial ||
        x.completion_token_->serial == UINT64_MAX)
      return Fail("no unchanged native completion receipt for this progress ledger");
    return {};
  }
  static void Completed(Progress& x, const kg::LaunchContext& launch) {
    x.completion_owner_ = &launch;
    x.completion_identity_ = ProgressIdentity(x);
    if (!x.completion_token_) {
      x.completion_token_ = std::make_shared<CompletionToken>();
      x.completion_token_->owner = &launch;
    }
    ++x.completion_token_->serial;
    x.completion_serial_ = x.completion_token_->serial;
  }
  static void Clear(Chunk& c) {
    c.resolved_owner_ = nullptr;
    c.resolved_profile_.reset();
    c.resolved_identity_.clear();
    c.resolved_dispatch_.reset();
  }
  static void Seal(Chunk& c, const kg::LaunchContext& launch, const model::Dsv4Profile& profile,
                   ResolvedDispatch dispatch) {
    c.resolved_profile_ = profile;
    c.resolved_identity_ = Identity(c);
    c.resolved_owner_ = &launch;
    c.resolved_dispatch_ = std::move(dispatch);
  }
  static Result Check(const Chunk& c, const kg::LaunchContext& launch) {
    if (c.resolved_owner_ != &launch || !c.resolved_profile_)
      return Fail("recipe is not resolved for this native launch owner");
    if (c.artifact_id != kCommunityArtifact || c.resolved_identity_ != Identity(c))
      return Fail("operand binding or numerical recipe changed after Resolve");
    return {};
  }
  static std::expected<ResolvedDispatch, std::string> Dispatch(const Chunk& c) {
    if (!c.resolved_owner_ || !c.resolved_profile_ || !c.resolved_dispatch_ ||
        c.resolved_identity_ != Identity(c))
      return std::unexpected("no current actual resolved dispatch receipt");
    if (auto check = CheckChunk(*c.resolved_profile_, c); !check)
      return std::unexpected(check.error());
    return *c.resolved_dispatch_;
  }
};
std::expected<ResolvedDispatch, std::string> ReadResolvedDispatch(const Chunk& c) {
  return RuntimeAccess::Dispatch(c);
}

#ifdef JITLLM_DS4_COMPLETE_CUDA
namespace {
Result Owner(engine::PagedNode& node, kg::LaunchContext& launch, std::uint32_t stream) {
  if (launch.faulted() || launch.capturing() ||
      !launch.UsesStream(node.execution(), node.stream(stream)))
    return Fail("wrong native provider/stream, capture, or faulted execution owner");
  return {};
}
class Queue {
 public:
  bool Add(Checked r) {
    if (!r) {
      error = r.error().detail;
      unknown = r.error().error == kg::KernelError::kUnknown;
      return false;
    }
    queued = true;
    return true;
  }
  bool Add(providers::DeviceStatus r) {
    if (!r.ok()) {
      error = r.text();
      unknown = true;
      return false;
    }
    queued = true;
    return true;
  }
  scheduler::JobResult Outcome() const {
    if (unknown) return scheduler::JobResult::kUnknown;
    if (!error.empty())
      return queued ? scheduler::JobResult::kFailed : scheduler::JobResult::kNotStarted;
    return scheduler::JobResult::kQueued;
  }
  std::string error;
  bool queued = false;
  bool unknown = false;
};
Result CheckProfile(ProfileMarks* profile) {
  if (profile == nullptr) return {};
  if (std::ranges::any_of(profile->marks, [](auto mark) { return mark.handle == nullptr; }))
    return Fail("diagnostic profile requires all sixteen runner-owned timing marks");
  profile->milliseconds.fill(0);
  profile->active.fill(false);
  return {};
}
bool Mark(Queue& q, providers::NativeStream stream, ProfileMarks* profile, std::size_t chain,
          bool end = false) {
  if (profile == nullptr || !profile->active[chain]) return true;
  return q.Add(providers::RecordTimingMark(profile->marks[(2 * chain) + (end ? 1 : 0)], stream));
}
Result ReadProfile(ProfileMarks* profile) {
  if (profile == nullptr) return {};
  // Called only after node.Job proved completion on this native stream.
  for (std::size_t chain = 0; chain < kProfileChains; ++chain) {
    if (!profile->active[chain]) continue;
    auto elapsed =
        providers::ElapsedMilliseconds(profile->marks[2 * chain], profile->marks[(2 * chain) + 1]);
    if (!elapsed) return Fail(elapsed.error().text());
    if (!std::isfinite(*elapsed) || *elapsed <= 0)
      return Fail("active diagnostic chain has a nonpositive/nonfinite GPU duration");
    profile->milliseconds[chain] = *elapsed;
  }
  return {};
}
bool Project(Queue& q, kg::LaunchContext& launch, const Compression& d) {
  return q.Add(kg::RunDs4F16Product(launch, d.kv_projection)) &&
         q.Add(kg::RunDs4F16Product(launch, d.score_projection));
}
bool Emit(Queue& q, kg::LaunchContext& launch, const Compression& d) {
  if (!q.Add(kg::RunDs4CompChunk(launch, d.chunk))) return false;
  if (d.refresh) {
    if (!d.refresh_kv || !d.refresh_score) {
      q.error = "missing original compression refresh projections";
      return false;
    }
    return q.Add(kg::RunDs4F16Vector(launch, *d.refresh_kv)) &&
           q.Add(kg::RunDs4F16Vector(launch, *d.refresh_score)) &&
           q.Add(kg::RunDs4CompRefresh(launch, *d.refresh));
  }
  return true;
}
bool EncodeLayer(Queue& q, providers::NativeStream native, kg::LaunchContext& launch,
                 const Chunk& c, const Layer& l, ProfileMarks* profile) {
  // Fresh chunk invalidates the prior HC mirror; subsequent layers reuse
  // the current same-stream fused expand producer, never a global registry.
  if (!Mark(q, native, profile, 0)) return false;
  if (l.index == 0 && !q.Add(kg::RunDs4Rms(launch, l.hc_attention_rms))) return false;
  if (!q.Add(kg::RunDs4F16Product(launch, l.hc_attention_projection)) ||
      !q.Add(kg::RunDs4HcPre(launch, l.hc_attention_pre)) ||
      !q.Add(kg::RunDs4Rms(launch, l.attention_norm)))
    return false;
  if (!Mark(q, native, profile, 0, true) || !Mark(q, native, profile, 1)) return false;
  if (!q.Add(kg::RunDs4Q8Product(launch, l.query_a)) ||
      !q.Add(kg::RunDs4Q8Product(launch, l.kv_projection)) ||
      !q.Add(kg::RunDs4QkvNorm(launch, l.qkv_norm)) ||
      !q.Add(kg::RunDs4Q8Product(launch, l.query_b)) ||
      !q.Add(kg::RunDs4HeadRope(launch, l.query_rope)) ||
      !q.Add(kg::RunDs4HeadRope(launch, l.kv_rope)) ||
      !q.Add(kg::RunDs4CacheQat(launch, l.raw_qat)))
    return false;
  // Preserve the original zero-prefix store before compressor work; a
  // nonzero compressed chunk stores after emit and before score/select.
  if ((c.first == 0 || l.ratio == 0) && !q.Add(kg::RunDs4CacheRawStore(launch, l.raw_store)))
    return false;
  if (!Mark(q, native, profile, 1, true) || !Mark(q, native, profile, 2)) return false;
  if (l.compression && (!Project(q, launch, *l.compression) || !Emit(q, launch, *l.compression)))
    return false;
  if (l.indexer) {
    const auto& d = *l.indexer;
    // Original producers sit between indexer projections and emit.
    if (!Project(q, launch, d.compression) ||
        !q.Add(kg::RunDs4F16Conversion(launch, d.query_conversion)) ||
        !q.Add(kg::RunDs4F16Product(launch, d.query_projection)) ||
        !q.Add(kg::RunDs4HeadRope(launch, d.query_rope)) ||
        !q.Add(kg::RunDs4CacheQat(launch, d.query_qat)) ||
        !q.Add(kg::RunDs4F16Product(launch, d.weight_projection)) ||
        !Emit(q, launch, d.compression))
      return false;
  }
  if (c.first != 0 && l.ratio != 0 && !q.Add(kg::RunDs4CacheRawStore(launch, l.raw_store)))
    return false;
  if (l.indexer &&
      !q.Add(kg::RunDs4IndexerScoreSelect(launch, l.indexer->scores, l.indexer->select)))
    return false;
  if (!Mark(q, native, profile, 2, true) || !Mark(q, native, profile, 3)) return false;
  if (!q.Add(kg::RunDs4Attention(launch, l.attention))) return false;
  if (!Mark(q, native, profile, 3, true) || !Mark(q, native, profile, 4)) return false;
  if (!q.Add(kg::RunDs4OutA(launch, l.output_a)) ||
      !q.Add(kg::RunDs4Q8Product(launch, l.output_b)) ||
      !q.Add(kg::RunDs4HcExpand(launch, l.attention_expand)))
    return false;
  if (!Mark(q, native, profile, 4, true) || !Mark(q, native, profile, 5)) return false;
  if (!q.Add(kg::RunDs4F16Product(launch, l.hc_ffn_projection)) ||
      !q.Add(kg::RunDs4HcPre(launch, l.hc_ffn_pre)) || !q.Add(kg::RunDs4Rms(launch, l.ffn_norm)) ||
      !q.Add(kg::RunDs4F16Product(launch, l.router_projection)) ||
      !q.Add(kg::RunDs4Router(launch, l.router)))
    return false;
  if (!Mark(q, native, profile, 5, true) || !Mark(q, native, profile, 6)) return false;
  // Explicit charged safety work: the fused RMS writes only D4 payload.
  // Initialize the original guarded consumer slack before its first G1
  // read. The original default blanket arena/output memsets stay absent.
  const auto payload = static_cast<std::uint64_t>(kRows) * (4096 / 128) * 144;
  auto* tail =
      std::bit_cast<void*>(static_cast<std::uintptr_t>(l.ffn_norm.q8_d4.address + payload));
  if (!q.Add(providers::FillAsync(native, tail, 0, std::size_t{256} * 144)) ||
      !q.Add(kg::RunDs4Moe(launch, l.routed)))
    return false;
  if (!Mark(q, native, profile, 6, true) || !Mark(q, native, profile, 7)) return false;
  if (!q.Add(kg::RunDs4Q8Product(launch, l.shared_gate)) ||
      !q.Add(kg::RunDs4Q8Product(launch, l.shared_up)) ||
      !q.Add(kg::RunDs4SharedSwiglu(launch, l.shared_swiglu)) ||
      !q.Add(kg::RunDs4Q8Product(launch, l.shared_down)))
    return false;
  if (l.final_sum && !q.Add(kg::RunDs4MoeSum(launch, *l.final_sum))) return false;
  return q.Add(kg::RunDs4HcExpand(launch, l.ffn_expand)) && Mark(q, native, profile, 7, true);
}
Result JobResult(const Result& job, const Queue& q) {
  if (!job) {
    if (!q.error.empty()) return std::unexpected(q.error + "; native Job: " + job.error());
    return job;
  }
  if (!q.error.empty()) return Fail(q.error);
  return {};
}
Result Workspace(const kg::LaunchContext& launch, const Chunk& c) {
  const auto w = launch.workspace();
  const Buffer pool{w.base, w.size.value()};
  const auto* handle = launch.cublas();
  if (!handle || handle->workspace().size.value() != (32U << 20U) || !Range(pool) ||
      std::ranges::none_of(c.paid_ranges, [pool](Buffer b) { return Covers(b, pool); }))
    return Fail("original F16 products need the paid 32MiB cuBLAS and native pool");
  const Buffer blas{handle->workspace().base, handle->workspace().size.value()};
  if (Overlap(pool, blas)) return Fail("original native and cuBLAS workspaces overlap");
  if (std::ranges::none_of(c.paid_ranges, [blas](Buffer b) { return Covers(b, blas); }))
    return Fail("cuBLAS workspace is outside charged catalog ranges");
  for (const auto& b : Operands(c).ranges) {
    if (Overlap(b, pool) || Overlap(b, blas))
      return Fail("native workspace aliases a live recipe operand");
  }
  for (const auto& l : c.layers) {
    for (const auto* d : {&l.query_a, &l.kv_projection, &l.query_b, &l.output_b, &l.shared_gate,
                          &l.shared_up, &l.shared_down}) {
      auto bytes = kg::PlanDs4Q8Product(launch, *d);
      if (!bytes || *bytes > pool.bytes)
        return Fail("unpaid original Q8 native stream-K workspace");
    }
    auto moe = kg::PlanDs4MoeScratch(launch, l.routed);
    if (!moe || *moe > pool.bytes) return Fail("unpaid original routed native workspace");
    const Compression* compression = nullptr;
    const Compression* indexer = nullptr;
    const auto& compression_recipe = l.compression;
    const auto& indexer_recipe = l.indexer;
    if (compression_recipe) compression = &*compression_recipe;
    if (indexer_recipe) indexer = &indexer_recipe->compression;
    for (const auto* d : {compression, indexer}) {
      if (d == nullptr || !d->refresh) continue;
      if (!d->refresh_kv || !d->refresh_score)
        return Fail("missing original refresh workspace recipe");
      for (const auto* v : {&*d->refresh_kv, &*d->refresh_score}) {
        auto bytes = kg::PlanDs4F16Vector(*v);
        if (!bytes || *bytes > pool.bytes)
          return Fail("unpaid original last-four split-K workspace");
      }
    }
  }
  if (c.frontier) {
    auto bytes = kg::PlanDs4F16Vector(c.frontier->hc_projection);
    if (!bytes || *bytes > pool.bytes) return Fail("unpaid original frontier split-K workspace");
  }
  return {};
}
}  // namespace

Result Resolve(engine::PagedNode& node, kg::LaunchContext& launch, const catalog::Closure& closure,
               std::uint32_t stream, const model::Dsv4Profile& p, Chunk& c,
               std::span<const HashTable> tables) {
  RuntimeAccess::Clear(c);
  if (auto check = Owner(node, launch, stream); !check) return check;
  if (auto check = CheckChunk(p, c); !check) return check;
  if (auto check = CheckHashTables(p, c, tables); !check) return check;
  for (const auto& l : c.layers) {
    if (l.attention.kind != kg::Ds4AttentionKind::kOriginalDefault ||
        (l.indexer && (l.indexer->scores.kind != kg::Ds4IndexerScoreKind::kOriginalDefault ||
                       l.indexer->select.kind != kg::Ds4IndexerSelectKind::kOriginalDefault)))
      return Fail("Resolve must authenticate original default selection before freezing tiers");
  }
  std::array<kg::Ds4IndexerDispatch, kLayers> index_dispatch{};
  ResolvedDispatch dispatch{
      .device = launch.device(),
      .stream = stream,
      .planned_sms = c.device_sms,
      .native_pool_bytes = launch.workspace().size.value(),
      .cublas_bytes = launch.cublas() ? launch.cublas()->workspace().size.value() : 0,
      .stages = {}};
  std::string error;
  auto resolved = node.Job(
      closure,
      [&](providers::NativeStream) {
        const auto facts = providers::QueryDeviceFacts(launch.device());
        if (!facts || facts->architecture != 1210) {
          error = "initial original packed baseline requires actual GB10 architecture1210";
          return scheduler::JobResult::kNotStarted;
        }
        dispatch.architecture = facts->architecture;
        auto prepared = kg::PrepareDs4Attention(launch);
        if (prepared) prepared = kg::PrepareDs4Indexer(launch);
        if (!prepared) {
          error = prepared.error().detail;
          return prepared.error().error == kg::KernelError::kUnknown
                     ? scheduler::JobResult::kUnknown
                     : scheduler::JobResult::kNotStarted;
        }
        if (auto check = Workspace(launch, c); !check) {
          error = check.error();
          return scheduler::JobResult::kNotStarted;
        }
        for (const auto& l : c.layers) {
          auto attention = kg::DescribeDs4AttentionDispatch(launch, l.attention);
          if (!attention || attention->kind != kg::Ds4AttentionKind::kTokenTile) {
            error = "original packed token-tile tier unavailable; F32 fallback is forbidden";
            return scheduler::JobResult::kNotStarted;
          }
          dispatch.stages.push_back(
              {l.index, "attention", "original-token-tile", attention->scratch.bytes});
          if (l.indexer) {
            auto indexer =
                kg::DescribeDs4IndexerDispatch(launch, l.indexer->scores, l.indexer->select);
            if (!indexer || indexer->score_kind != kg::Ds4IndexerScoreKind::kMxf4) {
              error = "original packed MXF4 tier unavailable; F32 fallback is forbidden";
              return scheduler::JobResult::kNotStarted;
            }
            const auto expected_select = l.indexer->scores.cells == 1024
                                             ? kg::Ds4IndexerSelectKind::kBitonic1024
                                             : kg::Ds4IndexerSelectKind::kBitonic2048;
            if (indexer->select_kind != expected_select) {
              error = "unexpected original first8K selector dispatch";
              return scheduler::JobResult::kNotStarted;
            }
            index_dispatch[l.index] = *indexer;
            auto score_desc = l.indexer->scores;
            auto select_desc = l.indexer->select;
            score_desc.kind = indexer->score_kind;
            select_desc.kind = indexer->select_kind;
            const auto score_bytes = kg::Ds4IndexerScoreScratchBytes(score_desc);
            const auto select_bytes = kg::Ds4IndexerSelectScratchBytes(select_desc);
            if (!score_bytes || !select_bytes) {
              error = "resolved indexer workspace unavailable";
              return scheduler::JobResult::kNotStarted;
            }
            dispatch.stages.push_back({l.index, "indexer-score", "original-mxf4", *score_bytes});
            dispatch.stages.push_back(
                {l.index, "indexer-select",
                 indexer->select_kind == kg::Ds4IndexerSelectKind::kBitonic1024
                     ? "original-bitonic1024"
                     : "original-bitonic2048",
                 *select_bytes});
          }
          const std::array<std::pair<std::string_view, const kg::Ds4Q8Product*>, 7> products{
              {{"query-a", &l.query_a},
               {"kv-projection", &l.kv_projection},
               {"query-b", &l.query_b},
               {"output-b", &l.output_b},
               {"shared-gate", &l.shared_gate},
               {"shared-up", &l.shared_up},
               {"shared-down", &l.shared_down}}};
          for (const auto& [stage, product] : products) {
            const auto bytes = kg::PlanDs4Q8Product(launch, *product);
            if (!bytes) {
              error = bytes.error().detail;
              return scheduler::JobResult::kNotStarted;
            }
            dispatch.stages.push_back({l.index, stage,
                                       product->path == kg::Ds4Q8Path::kMmq
                                           ? "original-aligned-q8-mmq"
                                           : "original-aligned-q8-dense-d2r",
                                       *bytes});
          }
          auto moe = kg::PlanDs4MoeScratch(launch, l.routed);
          if (!moe) {
            error = moe.error().detail;
            return scheduler::JobResult::kNotStarted;
          }
          dispatch.stages.push_back({l.index, "routed-ffn", "original-direct-g1-d2s6-q2", *moe});
          dispatch.stages.push_back({l.index, "output-a", "original-own-hmma-inverse-rope-d4", 0});
          dispatch.stages.push_back({l.index, "router", "original-batch-top6-warp", 0});
          dispatch.stages.push_back(
              {l.index, "hc-projections", "original-wide-f16-gemm", dispatch.cublas_bytes});
        }
        if (c.frontier) {
          const auto bytes = kg::PlanDs4F16Vector(c.frontier->hc_projection);
          if (!bytes) {
            error = bytes.error().detail;
            return scheduler::JobResult::kNotStarted;
          }
          dispatch.stages.push_back({kLayers, "frontier-hc", "original-small-f16-split-k", *bytes});
          dispatch.stages.push_back(
              {kLayers, "frontier-full-head", "original-aligned-q8-vector", 0});
        }
        // The Job records native completion of setup. No model operation ran.
        return scheduler::JobResult::kQueued;
      },
      "resolve original ds4 complete dispatch", stream);
  if (!resolved)
    return std::unexpected(error.empty() ? resolved.error() : error + "; " + resolved.error());
  if (!error.empty()) return Fail(error);
  for (auto& l : c.layers) {
    l.attention.kind = kg::Ds4AttentionKind::kTokenTile;
    if (l.indexer) {
      l.indexer->scores.kind = index_dispatch[l.index].score_kind;
      l.indexer->select.kind = index_dispatch[l.index].select_kind;
    }
  }
  if (auto check = CheckChunk(p, c); !check) return check;
  RuntimeAccess::Seal(c, launch, p, std::move(dispatch));
  return {};
}
Result RunEmbedding(engine::PagedNode& node, kg::LaunchContext& launch,
                    const catalog::Closure& closure, std::uint32_t stream, const Chunk& c,
                    Progress& x, ProfileMarks* profile) {
  if (auto check = Owner(node, launch, stream); !check) return check;
  if (auto check = RuntimeAccess::Check(c, launch); !check) return check;
  if (auto check = CheckProgress(c, x, Phase::kEmbedding); !check) return check;
  if (auto check = RuntimeAccess::CheckCompletion(x, launch, c.first == 0); !check) return check;
  if (auto check = CheckProfile(profile); !check) return check;
  if (profile != nullptr) profile->active[0] = true;
  Queue q;
  auto job = node.Job(
      closure,
      [&](providers::NativeStream native) {
        if (Mark(q, native, profile, 0) && q.Add(kg::RunDs4Embedding(launch, c.embedding)))
          Mark(q, native, profile, 0, true);
        return q.Outcome();
      },
      "original ds4 complete embedding", stream);
  if (auto check = JobResult(job, q); !check) {
    Poison(x);
    return check;
  }
  if (auto check = ReadProfile(profile); !check) {
    Poison(x);
    return check;
  }
  if (auto check = CompleteEmbedding(c, x); !check) {
    Poison(x);
    return check;
  }
  RuntimeAccess::Completed(x, launch);
  return {};
}
Result RunLayer(engine::PagedNode& node, kg::LaunchContext& launch, const catalog::Closure& closure,
                std::uint32_t stream, const Chunk& c, std::uint32_t layer, Progress& x,
                ProfileMarks* profile) {
  if (auto check = Owner(node, launch, stream); !check) return check;
  if (auto check = RuntimeAccess::Check(c, launch); !check) return check;
  if (auto check = CheckProgress(c, x, Phase::kLayers, layer); !check) return check;
  if (auto check = RuntimeAccess::CheckCompletion(x, launch, false); !check) return check;
  if (auto check = CheckProfile(profile); !check) return check;
  if (profile != nullptr) {
    profile->active.fill(true);
    profile->active[2] = c.layers[layer].ratio != 0;
  }
  Queue q;
  auto job = node.Job(
      closure,
      [&](providers::NativeStream native) {
        EncodeLayer(q, native, launch, c, c.layers[layer], profile);
        return q.Outcome();
      },
      "original ds4 complete attention+FFN layer", stream);
  if (auto check = JobResult(job, q); !check) {
    Poison(x);
    return check;
  }
  if (auto check = ReadProfile(profile); !check) {
    Poison(x);
    return check;
  }
  if (auto check = CompleteLayer(c, layer, x); !check) {
    Poison(x);
    return check;
  }
  RuntimeAccess::Completed(x, launch);
  return {};
}
Result RunFrontier(engine::PagedNode& node, kg::LaunchContext& launch,
                   const catalog::Closure& closure, std::uint32_t stream, const Chunk& c,
                   Progress& x, ProfileMarks* profile) {
  if (auto check = Owner(node, launch, stream); !check) return check;
  if (auto check = RuntimeAccess::Check(c, launch); !check) return check;
  if (auto check = CheckProgress(c, x, Phase::kFrontier); !check) return check;
  if (auto check = RuntimeAccess::CheckCompletion(x, launch, false); !check) return check;
  if (!c.frontier) return Fail("missing original final head recipe");
  if (auto check = CheckProfile(profile); !check) return check;
  if (profile != nullptr) profile->active[0] = true;
  Queue q;
  const auto& d = *c.frontier;
  auto job = node.Job(
      closure,
      [&](providers::NativeStream native) {
        if (Mark(q, native, profile, 0) && q.Add(kg::RunDs4Rms(launch, d.hc_rms)) &&
            q.Add(kg::RunDs4F16Vector(launch, d.hc_projection)) &&
            q.Add(kg::RunDs4HcHeadWeights(launch, d.hc_weights)) &&
            q.Add(kg::RunDs4HcWeighted(launch, d.collapse)) &&
            q.Add(kg::RunDs4Rms(launch, d.norm)) && q.Add(kg::RunDs4Q8Vector(launch, d.projection)))
          Mark(q, native, profile, 0, true);
        return q.Outcome();
      },
      "original ds4 complete final HC/full head", stream);
  if (auto check = JobResult(job, q); !check) {
    Poison(x);
    return check;
  }
  if (auto check = ReadProfile(profile); !check) {
    Poison(x);
    return check;
  }
  if (auto check = CompleteFrontier(c, x); !check) {
    Poison(x);
    return check;
  }
  RuntimeAccess::Completed(x, launch);
  return {};
}
#else
Result Resolve(engine::PagedNode& /*node*/, kg::LaunchContext& /*launch*/,
               const catalog::Closure& /*closure*/, std::uint32_t /*stream*/,
               const model::Dsv4Profile& /*profile*/, Chunk& /*chunk*/,
               std::span<const HashTable> /*hash_tables*/) {
  return Fail("complete ds4 execution requires the native CUDA benchmark");
}
Result RunEmbedding(engine::PagedNode& /*node*/, kg::LaunchContext& /*launch*/,
                    const catalog::Closure& /*closure*/, std::uint32_t /*stream*/,
                    const Chunk& /*c*/, Progress& /*x*/, ProfileMarks* /*profile*/) {
  return Fail("complete ds4 execution requires the native CUDA benchmark");
}
Result RunLayer(engine::PagedNode& /*node*/, kg::LaunchContext& /*launch*/,
                const catalog::Closure& /*closure*/, std::uint32_t /*stream*/, const Chunk& /*c*/,
                std::uint32_t /*layer*/, Progress& /*x*/, ProfileMarks* /*profile*/) {
  return Fail("complete ds4 execution requires the native CUDA benchmark");
}
Result RunFrontier(engine::PagedNode& /*node*/, kg::LaunchContext& /*launch*/,
                   const catalog::Closure& /*closure*/, std::uint32_t /*stream*/,
                   const Chunk& /*c*/, Progress& /*x*/, ProfileMarks* /*profile*/) {
  return Fail("complete ds4 execution requires the native CUDA benchmark");
}
#endif

}  // namespace jitllm::benchmarks::ds4_complete
