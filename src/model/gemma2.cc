// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "model/gemma2.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <format>
#include <initializer_list>
#include <limits>
#include <unordered_map>
#include <utility>

#include "artifact/artifact.h"
#include "artifact/representation.h"
#include "base/check.h"
#include "base/sha256.h"

namespace jitllm::model {
namespace {
// Every multiplication follows equality with the closed, bounded profile.
// Keep the existing artifact's 512-element quantized over-read contract:
// width2304 requires one extra 256-element Q8_0 row tail (272 bytes).
bool StorageValid(std::string_view type, std::span<const std::uint64_t> ne, std::uint64_t readable,
                  std::string_view expected_type, std::span<const std::uint64_t> expected_ne) {
  const auto* traits = artifact::FindGgmlType(type);
  if (type != expected_type || traits == nullptr || ne.empty() ||
      !std::ranges::equal(ne, expected_ne) || ne[0] % traits->block_elements != 0)
    return false;
  auto bytes = ne[0] / traits->block_elements * traits->block_bytes;
  for (std::size_t i = 1; i < ne.size(); ++i) bytes *= ne[i];
  if (traits->block_elements > 1 && ne[0] % artifact::kGgmlRowPadding != 0) {
    bytes += (artifact::kGgmlRowPadding - ne[0] % artifact::kGgmlRowPadding) /
             traits->block_elements * traits->block_bytes;
  }
  return readable >= bytes;
}
constexpr std::uint64_t kExtent = 2U << 20U;
constexpr std::uint16_t kZero = 0, kNegInf = 0xFC00;
std::unexpected<std::string> Refused(std::string detail) {
  return std::unexpected(std::move(detail));
}
std::uint64_t Pad(std::uint64_t bytes, std::uint64_t align) {
  return ((bytes + align - 1) / align) * align;
}
bool ProfileValid(const Gemma2Profile& p) { return p == Gemma2_2B(); }
bool AlignValid(std::uint32_t align) { return align >= 256 && align <= 8192 && align % 256 == 0; }
std::uint32_t ReadCells(std::uint32_t positions, std::uint32_t cells, std::uint32_t align) {
  return static_cast<std::uint32_t>(std::min<std::uint64_t>(cells, Pad(positions, align)));
}
// Public layouts are checked too: an edited offset/width must never produce
// out-of-region input indices or understate materialization ranges.
bool LayoutValid(const Gemma2Profile& p, const Gemma2StateLayout& s) {
  if (!ProfileValid(p) || s.context == 0 || s.context > p.context || s.max_rows == 0 ||
      s.max_rows > std::min(s.context, kGemma2MaxRows) || s.global_cells != Pad(s.context, 256) ||
      s.local_cells != Pad(std::min<std::uint64_t>(s.context, p.window + s.max_rows), 256) ||
      s.tensors.size() != std::size_t{p.layers} * 2) {
    return false;
  }
  std::uint64_t at = 0;
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    for (const bool value : {false, true}) {
      const auto& t = s.tensors[std::size_t{il} * 2 + (value ? 1U : 0U)];
      const auto cells = p.local(il) ? s.local_cells : s.global_cells;
      const auto width = (value ? p.value_dim : p.key_dim) * p.kv_heads;
      const auto bytes = std::uint64_t{width} * cells * 2;
      if (t.layer != il || t.value != value || t.local != p.local(il) || t.width != width ||
          t.cells != cells || t.offset != at || t.bytes != bytes) {
        return false;
      }
      at += Pad(bytes, kExtent);
    }
  }
  return s.bytes == at;
}
std::expected<std::uint64_t, std::string> InputBytes(const Gemma2Profile& p,
                                                     const Gemma2StateLayout& s,
                                                     std::span<const Gemma2Segment> segments,
                                                     bool masks, std::uint32_t align,
                                                     std::uint32_t max_total_rows) {
  const auto limit = max_total_rows == 0 ? s.max_rows : max_total_rows;
  if (!LayoutValid(p, s) || !AlignValid(align) || segments.empty() ||
      segments.size() > kGemma2MaxSlots || limit < s.max_rows || limit > kGemma2MaxRows ||
      limit > std::uint64_t{s.max_rows} * kGemma2MaxSlots) {
    return Refused("invalid Gemma2 layout, alignment or segment count");
  }
  std::array<bool, kGemma2MaxSlots> seen{};
  std::uint64_t bytes = segments.size() * sizeof(Gemma2SegmentInputs), rows = 0;
  for (const auto& seg : segments) {
    if (seg.slot >= kGemma2MaxSlots || seen[seg.slot] || seg.tokens.empty() ||
        seg.n_past > s.context || seg.tokens.size() > s.context - seg.n_past ||
        seg.tokens.size() > s.max_rows || seg.tokens.size() > limit - rows) {
      return Refused("empty, repeated or out-of-bounds Gemma2 segment");
    }
    seen[seg.slot] = true;
    for (const auto token : seg.tokens) {
      if (token < 0 || std::cmp_greater_equal(token, p.vocab)) {
        return Refused("Gemma2 token is outside the vocabulary");
      }
    }
    rows += seg.tokens.size();
    // Token, position, output ID and the global/local I64 cell indices.
    bytes += seg.tokens.size() * (3 * sizeof(std::int32_t) + 2 * sizeof(std::int64_t));
    if (masks) {
      const auto end = seg.n_past + static_cast<std::uint32_t>(seg.tokens.size());
      const auto global =
          std::uint64_t{ReadCells(end, s.global_cells, align)} * seg.tokens.size() * 2;
      const auto local =
          std::uint64_t{ReadCells(end, s.local_cells, align)} * seg.tokens.size() * 2;
      if (global > std::numeric_limits<std::int32_t>::max() ||
          local > std::numeric_limits<std::int32_t>::max()) {
        return Refused("Gemma2 reference mask exceeds the GGML I32 byte stride");
      }
      bytes += global + local;
    }
  }
  return bytes;
}
}  // namespace

const Gemma2Profile& Gemma2_2B() {
  // Actual metadata plus d812 defaults: absent RoPE base/scale => 10000/1.
  // The 2B attention scale is 1/sqrt(256), not 1/sqrt(width/heads).
  static constexpr Gemma2Profile profile{26,      2304,    9216,  8,    4,        256,
                                         256,     256000,  8192,  4096, 10000.0F, 1.0F,
                                         1.0e-6F, 0.0625F, 50.0F, 30.0F};
  return profile;
}

std::expected<Gemma2Binding, std::string> BindApprovedGemma2(const artifact::Artifact& artifact) {
  const auto sources = artifact.sources();
  if (sources.size() != 1 || sources[0].name != "gemma-2-2b-it-Q8_0.gguf" ||
      sources[0].bytes.value() != 2784495456ULL ||
      base::ToHex(sources[0].sha256) !=
          "2d448a9aab894b8e8e18168cf3f490cb9f65632222f29f93514ac9ecc754debe")
    return std::unexpected("Gemma2 execution needs the approved prepared Q8_0 source identity");
  return BindGemma2(Gemma2_2B(), artifact);
}
std::expected<Gemma2Binding, std::string> BindGemma2(const Gemma2Profile& p,
                                                     std::string_view architecture,
                                                     std::span<const Gemma2Resource> resources) {
  if (p != Gemma2_2B() || architecture != "gemma2" || resources.size() != 288) {
    return std::unexpected("unsupported Gemma2 profile, architecture or resource count");
  }
  std::unordered_map<std::string, std::uint32_t> roles;
  for (std::size_t i = 0; i < resources.size(); ++i) {
    const auto& resource = resources[i];
    if (resource.roles.empty() || resource.roles.size() > 2) {
      return std::unexpected("invalid Gemma2 resource roles");
    }
    for (const auto& role : resource.roles) {
      if (role.empty() || !roles.emplace(role, static_cast<std::uint32_t>(i)).second) {
        return std::unexpected("empty or duplicate Gemma2 role");
      }
    }
  }
  const auto take =
      [&](const std::string& role, std::string_view type,
          std::initializer_list<std::uint64_t> shape) -> std::expected<Gemma2Tensor, std::string> {
    const auto found = roles.find(role);
    if (found == roles.end()) {
      return std::unexpected("missing Gemma2 role: " + role);
    }
    const auto index = found->second;
    const auto& resource = resources[index];
    const std::vector<std::uint64_t> ne(shape);
    if (!StorageValid(resource.type, resource.ne, resource.readable, type, ne)) {
      return std::unexpected("wrong Gemma2 type, shape or readable storage: " + role);
    }
    roles.erase(found);
    return Gemma2Tensor{index, resource.type, resource.ne, resource.readable};
  };
  Gemma2Binding binding;
  const auto assign =
      [&](Gemma2Tensor& target, const std::string& role, std::string_view type,
          std::initializer_list<std::uint64_t> shape) -> std::expected<void, std::string> {
    auto tensor = take(role, type, shape);
    if (!tensor) {
      return std::unexpected(tensor.error());
    }
    target = std::move(*tensor);
    return {};
  };
  if (auto result = assign(binding.token_embd, "token_embd.weight", "Q8_0", {p.width, p.vocab});
      !result) {
    return std::unexpected(result.error());
  }
  binding.output = binding.token_embd;
  if (const auto head = roles.find("output.weight"); head != roles.end()) {
    if (head->second != binding.token_embd.index) {
      return std::unexpected("Gemma2 output must alias token embeddings");
    }
    roles.erase(head);
  }
  if (auto result = assign(binding.output_norm, "output_norm.weight", "F32", {p.width}); !result) {
    return std::unexpected(result.error());
  }
  binding.layers.resize(p.layers);
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    auto& layer = binding.layers[il];
    const auto prefix = "blk." + std::to_string(il) + '.';
    const auto norm = [&](Gemma2Tensor& tensor, std::string_view suffix, std::uint64_t width) {
      return assign(tensor, prefix + std::string(suffix) + ".weight", "F32", {width});
    };
    const auto matrix = [&](Gemma2Tensor& tensor, std::string_view suffix, std::uint64_t input,
                            std::uint64_t output) {
      return assign(tensor, prefix + std::string(suffix) + ".weight", "Q8_0", {input, output});
    };
    // Q/K/V are distinct resources; this profile has no Q/K norms.
    for (auto result :
         {norm(layer.attn_norm, "attn_norm", p.width),
          matrix(layer.q, "attn_q", p.width, p.heads * p.key_dim),
          matrix(layer.k, "attn_k", p.width, p.kv_heads * p.key_dim),
          matrix(layer.v, "attn_v", p.width, p.kv_heads * p.value_dim),
          matrix(layer.out, "attn_output", p.heads * p.value_dim, p.width),
          norm(layer.attn_post_norm, "post_attention_norm", p.width),
          norm(layer.ffn_norm, "ffn_norm", p.width), matrix(layer.gate, "ffn_gate", p.width, p.ffn),
          matrix(layer.up, "ffn_up", p.width, p.ffn),
          matrix(layer.down, "ffn_down", p.ffn, p.width),
          norm(layer.ffn_post_norm, "post_ffw_norm", p.width)}) {
      if (!result) {
        return std::unexpected(result.error());
      }
    }
  }
  if (!roles.empty()) {
    return std::unexpected("unconsumed Gemma2 roles");
  }
  return binding;
}

std::expected<Gemma2Binding, std::string> BindGemma2(const Gemma2Profile& p,
                                                     const artifact::Artifact& artifact) {
  if (artifact.model().expert_count != 0 || !artifact.expert_arrays().empty() ||
      artifact.resources().size() != 288) {
    return std::unexpected("Gemma2 requires 288 ordinary resources and no expert storage");
  }
  std::vector<Gemma2Resource> resources;
  resources.reserve(artifact.resources().size());
  for (const auto& resource : artifact.resources()) {
    if (resource.repr.family != artifact::Family::kGgml) {
      return std::unexpected("Gemma2 requires GGML resources");
    }
    resources.push_back({resource.roles, std::string(resource.repr.type), resource.repr.dims,
                         resource.readable.value()});
  }
  return BindGemma2(p, artifact.model().architecture, resources);
}
std::expected<void, std::string> CheckGemma2Binding(const Gemma2Profile& p,
                                                    const Gemma2Binding& binding) {
  if (p != Gemma2_2B() || binding.layers.size() != p.layers ||
      binding.output != binding.token_embd) {
    return std::unexpected("invalid Gemma2 profile, layer count or tied head");
  }
  // Fixed stack membership replaces reconstruction/rebinding on future waves.
  std::array<bool, 288> seen{};
  const auto check = [&](const Gemma2Tensor& t, std::string_view type,
                         std::initializer_list<std::uint64_t> shape) {
    if (t.index >= seen.size() || seen[t.index] ||
        !StorageValid(t.type, t.ne, t.readable, type, shape))
      return false;
    seen[t.index] = true;
    return true;
  };
  if (!check(binding.token_embd, "Q8_0", {p.width, p.vocab}) ||
      !check(binding.output_norm, "F32", {p.width})) {
    return std::unexpected("invalid Gemma2 embedding/head descriptors");
  }
  for (const auto& l : binding.layers) {
    if (!check(l.attn_norm, "F32", {p.width}) ||
        !check(l.q, "Q8_0", {p.width, p.heads * p.key_dim}) ||
        !check(l.k, "Q8_0", {p.width, p.kv_heads * p.key_dim}) ||
        !check(l.v, "Q8_0", {p.width, p.kv_heads * p.value_dim}) ||
        !check(l.out, "Q8_0", {p.heads * p.value_dim, p.width}) ||
        !check(l.attn_post_norm, "F32", {p.width}) || !check(l.ffn_norm, "F32", {p.width}) ||
        !check(l.gate, "Q8_0", {p.width, p.ffn}) || !check(l.up, "Q8_0", {p.width, p.ffn}) ||
        !check(l.down, "Q8_0", {p.ffn, p.width}) || !check(l.ffn_post_norm, "F32", {p.width})) {
      return std::unexpected("invalid Gemma2 layer descriptor or resource identity");
    }
  }
  return {};
}
std::expected<Gemma2StateLayout, std::string> Gemma2State(const Gemma2Profile& p,
                                                          std::uint32_t context,
                                                          std::uint32_t max_rows) {
  if (!ProfileValid(p) || context == 0 || context > p.context || max_rows == 0 ||
      max_rows > std::min(context, kGemma2MaxRows)) {
    return Refused("Gemma2 context or chunk bound is invalid");
  }
  Gemma2StateLayout s;
  s.context = context;
  s.max_rows = max_rows;
  s.global_cells = static_cast<std::uint32_t>(Pad(context, 256));
  s.local_cells =
      static_cast<std::uint32_t>(Pad(std::min<std::uint64_t>(context, p.window + max_rows), 256));
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    for (const bool value : {false, true}) {
      const auto width = (value ? p.value_dim : p.key_dim) * p.kv_heads;
      const auto cells = p.local(il) ? s.local_cells : s.global_cells;
      const auto bytes = std::uint64_t{width} * cells * 2;
      s.tensors.push_back({.layer = il,
                           .value = value,
                           .local = p.local(il),
                           .width = width,
                           .cells = cells,
                           .offset = s.bytes,
                           .bytes = bytes});
      s.bytes += Pad(bytes, kExtent);
    }
  }
  return s;
}
std::expected<std::vector<StateRepresentation>, std::string> Gemma2StateLayout::Representations(
    const Gemma2Profile& p) const {
  if (!LayoutValid(p, *this)) {
    return Refused("Gemma2 representation layout is invalid");
  }
  std::vector<StateRepresentation> out;
  for (const auto& t : tensors) {
    out.push_back({.name = std::format("gemma2.{}.{}", t.layer, t.value ? "v" : "k"),
                   .block_positions = 0,
                   .block_bytes = base::Bytes{Pad(t.bytes, kExtent)},
                   .capabilities = t.local ? static_cast<std::uint8_t>(StateCapability::kAppend)
                                           : StateCapability::kAppend | StateCapability::kTruncate,
                   .max_snapshots = 0,
                   .snapshot_bytes = base::Bytes{0}});
  }
  return out;
}
std::expected<std::vector<StateRange>, std::string> Gemma2UsedState(const Gemma2Profile& p,
                                                                    const Gemma2StateLayout& s,
                                                                    std::uint32_t positions,
                                                                    std::uint32_t align) {
  if (!LayoutValid(p, s) || !AlignValid(align) || positions > s.context) {
    return Refused("Gemma2 initialized footprint is out of bounds");
  }
  std::vector<StateRange> out;
  if (positions != 0) {
    for (const auto& t : s.tensors) {
      out.push_back({.offset = t.offset,
                     .bytes = std::uint64_t{ReadCells(positions, t.cells, align)} * t.width * 2});
    }
  }
  return out;
}
std::expected<std::uint64_t, std::string> Gemma2HostInputBytes(
    const Gemma2Profile& p, const Gemma2StateLayout& s, std::span<const Gemma2Segment> segments,
    bool masks, std::uint32_t align, std::uint32_t max_total_rows) {
  return InputBytes(p, s, segments, masks, align, max_total_rows);
}
std::expected<Gemma2ChunkInputs, std::string> Gemma2Chunk(const Gemma2Profile& p,
                                                          const Gemma2StateLayout& s,
                                                          std::span<const Gemma2Segment> segments,
                                                          bool masks, std::uint32_t align,
                                                          std::uint32_t max_total_rows) {
  const auto bytes = InputBytes(p, s, segments, masks, align, max_total_rows);
  if (!bytes) {
    return std::unexpected(bytes.error());
  }
  Gemma2ChunkInputs in;
  std::size_t total_rows = 0;
  for (const auto& seg : segments) {
    total_rows += seg.tokens.size();
  }
  in.tokens.reserve(total_rows);
  in.positions.reserve(total_rows);
  in.out_ids.reserve(total_rows);
  in.segments.reserve(segments.size());
  for (const auto& seg : segments) {
    Gemma2SegmentInputs x;
    x.slot = seg.slot;
    x.n_past = seg.n_past;
    x.first_row = static_cast<std::uint32_t>(in.tokens.size());
    x.rows = static_cast<std::uint32_t>(seg.tokens.size());
    const auto end = x.n_past + x.rows;
    x.global_n_kv = ReadCells(end, s.global_cells, align);
    x.local_n_kv = ReadCells(end, s.local_cells, align);
    x.global_cells.resize(x.rows);
    x.local_cells.resize(x.rows);
    if (masks) {
      x.global_mask.assign(std::size_t{x.rows} * x.global_n_kv, kNegInf);
      x.local_mask.assign(std::size_t{x.rows} * x.local_n_kv, kNegInf);
    }
    for (std::uint32_t i = 0; i < x.rows; ++i) {
      const auto pos = x.n_past + i;
      in.tokens.push_back(seg.tokens[i]);
      in.positions.push_back(static_cast<std::int32_t>(pos));
      in.out_ids.push_back(static_cast<std::int32_t>(in.out_ids.size()));
      x.global_cells[i] = pos;
      x.local_cells[i] = pos % s.local_cells;
      if (masks) {
        for (std::uint32_t cell = 0; cell < x.global_n_kv; ++cell) {
          if (cell <= pos) {
            x.global_mask[std::size_t{i} * x.global_n_kv + cell] = kZero;
          }
        }
        for (std::uint32_t cell = 0; cell < x.local_n_kv; ++cell) {
          if (cell >= end) {
            continue;
          }
          const auto held = cell + ((end - 1 - cell) / s.local_cells) * s.local_cells;
          if (held <= pos && pos - held < p.window) {
            x.local_mask[std::size_t{i} * x.local_n_kv + cell] = kZero;
          }
        }
      }
    }
    in.segments.push_back(std::move(x));
  }
  std::uint64_t allocated =
      (in.tokens.capacity() + in.positions.capacity() + in.out_ids.capacity()) *
          sizeof(std::int32_t) +
      in.segments.capacity() * sizeof(Gemma2SegmentInputs);
  for (const auto& seg : in.segments) {
    allocated += (seg.global_cells.capacity() + seg.local_cells.capacity()) * sizeof(std::int64_t);
    allocated += (seg.global_mask.capacity() + seg.local_mask.capacity()) * sizeof(std::uint16_t);
  }
  base::Check(allocated == *bytes, "Gemma2 inputs exceeded their preflight storage envelope");
  return in;
}
std::expected<std::vector<StateRange>, std::string> Gemma2ChunkWrites(const Gemma2Profile& p,
                                                                      const Gemma2StateLayout& s,
                                                                      std::uint32_t past,
                                                                      std::uint32_t rows) {
  if (!LayoutValid(p, s) || rows == 0 || rows > s.max_rows || past > s.context ||
      rows > s.context - past) {
    return Refused("Gemma2 write footprint is out of bounds");
  }
  std::vector<StateRange> out;
  for (const auto& t : s.tensors) {
    const auto first = t.local ? past % t.cells : past;
    const auto count = std::min(rows, t.cells - first);
    const auto row_bytes = std::uint64_t{t.width} * 2;
    out.push_back({.offset = t.offset + first * row_bytes, .bytes = count * row_bytes});
    if (count < rows) {
      out.push_back({.offset = t.offset, .bytes = (rows - count) * row_bytes});
    }
  }
  return out;
}
}  // namespace jitllm::model
