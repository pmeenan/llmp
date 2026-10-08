// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "model/qwen2.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/representation.h"

namespace llmp::model {
namespace {

std::unexpected<std::string> Refused(std::string detail) {
  return std::unexpected(std::move(detail));
}

// A tensor the architecture reads: its role, type and ne.
struct Expected {
  std::string role;
  std::string_view type;
  std::vector<std::uint64_t> ne;
};

std::vector<Expected> ExpectedTensors(const Qwen2Profile& p) {
  const std::string_view w = p.weight_type;
  std::vector<Expected> out;
  out.reserve(3 + (std::size_t{p.layers} * 12));
  out.push_back({"token_embd.weight", w, {p.width, p.vocab}});
  out.push_back({"output_norm.weight", "F32", {p.width}});
  out.push_back({"output.weight", w, {p.width, p.vocab}});
  const std::uint64_t q_width = std::uint64_t{p.heads} * p.head_dim;
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    const std::string b = std::format("blk.{}.", il);
    out.push_back({b + "attn_norm.weight", "F32", {p.width}});
    out.push_back({b + "attn_q.weight", w, {p.width, q_width}});
    out.push_back({b + "attn_q.bias", "F32", {q_width}});
    out.push_back({b + "attn_k.weight", w, {p.width, p.kv_width()}});
    out.push_back({b + "attn_k.bias", "F32", {p.kv_width()}});
    out.push_back({b + "attn_v.weight", w, {p.width, p.kv_width()}});
    out.push_back({b + "attn_v.bias", "F32", {p.kv_width()}});
    out.push_back({b + "attn_output.weight", w, {q_width, p.width}});
    out.push_back({b + "ffn_norm.weight", "F32", {p.width}});
    out.push_back({b + "ffn_gate.weight", w, {p.width, p.ffn}});
    out.push_back({b + "ffn_up.weight", w, {p.width, p.ffn}});
    out.push_back({b + "ffn_down.weight", w, {p.ffn, p.width}});
  }
  return out;
}

std::string Shape(std::span<const std::uint64_t> ne) {
  std::string out = "[";
  for (std::size_t i = 0; i < ne.size(); ++i) {
    out += std::format("{}{}", i == 0 ? "" : ", ", ne[i]);
  }
  return out + "]";
}

}  // namespace

const Qwen2Profile& Qwen25Instruct05B() {
  // qwen2.* key/values of qwen2.5-0.5b-instruct-fp16.gguf (SHA-256
  // 8e0ae260...), checked by gguf_profile_check.py.
  static constexpr Qwen2Profile kProfile = {.name = "qwen2.5-0.5b-instruct",
                                            .layers = 24,
                                            .width = 896,
                                            .heads = 14,
                                            .kv_heads = 2,
                                            .head_dim = 64,
                                            .ffn = 4864,
                                            .vocab = 151936,
                                            .train_context = 8192,
                                            .rms_eps = 1e-6f,
                                            .rope_base = 1000000.0f,
                                            .weight_type = "F16"};
  return kProfile;
}

std::expected<Qwen2Binding, std::string> BindQwen2(const Qwen2Profile& profile,
                                                   std::string_view architecture,
                                                   std::span<const ResourceShape> resources) {
  if (architecture != "qwen2") {
    return Refused(std::format("the artifact's architecture is {}, not qwen2", architecture));
  }
  if (profile.layers == 0 || profile.heads == 0 || profile.kv_heads == 0 ||
      profile.heads % profile.kv_heads != 0) {
    return Refused("the profile's head counts are not a Qwen2 model's");
  }
  const std::vector<Expected> expected = ExpectedTensors(profile);
  // Every bound role must be one the architecture reads.
  for (const ResourceShape& resource : resources) {
    for (const std::string& role : resource.roles) {
      if (std::ranges::none_of(expected, [&](const Expected& e) { return e.role == role; })) {
        return Refused(std::format("the artifact binds {}, which Qwen2 does not read", role));
      }
    }
  }
  std::vector<std::uint32_t> index(expected.size());
  for (std::size_t t = 0; t < expected.size(); ++t) {
    const Expected& e = expected[t];
    const auto found = std::ranges::find_if(
        resources, [&](const ResourceShape& r) { return std::ranges::contains(r.roles, e.role); });
    if (found == resources.end()) {
      return Refused(std::format("the artifact has no {}", e.role));
    }
    if (found->type != e.type || !std::ranges::equal(found->ne, e.ne)) {
      return Refused(std::format("{} is {} {}, not {} {}", e.role, found->type, Shape(found->ne),
                                 e.type, Shape(e.ne)));
    }
    index[t] = static_cast<std::uint32_t>(found - resources.begin());
  }
  Qwen2Binding binding;
  binding.token_embd = index[0];
  binding.output_norm = index[1];
  binding.output = index[2];
  binding.layers.resize(profile.layers);
  for (std::uint32_t il = 0; il < profile.layers; ++il) {
    const std::size_t b = 3 + (std::size_t{il} * 12);
    binding.layers[il] = {.attn_norm = index[b],
                          .q = index[b + 1],
                          .q_bias = index[b + 2],
                          .k = index[b + 3],
                          .k_bias = index[b + 4],
                          .v = index[b + 5],
                          .v_bias = index[b + 6],
                          .out = index[b + 7],
                          .ffn_norm = index[b + 8],
                          .gate = index[b + 9],
                          .up = index[b + 10],
                          .down = index[b + 11]};
  }
  return binding;
}

std::expected<Qwen2Binding, std::string> BindQwen2(const Qwen2Profile& profile,
                                                   const artifact::Artifact& artifact) {
  std::vector<ResourceShape> shapes;
  shapes.reserve(artifact.resources().size());
  for (const artifact::Resource& resource : artifact.resources()) {
    if (resource.repr.family != artifact::Family::kGgml) {
      return Refused(std::format("{} is not a GGML representation", resource.name));
    }
    shapes.push_back({.roles = resource.roles,
                      .type = std::string(resource.repr.type),
                      .ne = resource.repr.dims});
  }
  if (!artifact.expert_arrays().empty()) {
    return Refused("a dense Qwen2 artifact has no expert arrays");
  }
  return BindQwen2(profile, artifact.model().architecture, shapes);
}

std::uint32_t PaddedKv(std::uint32_t used_cells, std::uint32_t cells) {
  constexpr std::uint32_t kPad = 256;
  const std::uint64_t padded = (std::uint64_t{used_cells} + kPad - 1) / kPad * kPad;
  return static_cast<std::uint32_t>(
      std::min<std::uint64_t>(cells, std::max<std::uint64_t>(kPad, padded)));
}

std::expected<ChunkInputs, std::string> Qwen2ChunkInputs(const Qwen2Profile& profile,
                                                         std::uint32_t cells, std::uint32_t n_past,
                                                         std::uint32_t rows) {
  if (rows == 0 || cells == 0 || n_past > cells || rows > cells - n_past) {
    return Refused(std::format("{} rows after {} do not fit {} cells", rows, n_past, cells));
  }
  const std::uint32_t used = n_past + rows;
  if (used > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
    return Refused("positions beyond int32");
  }
  ChunkInputs in;
  in.n_kv = PaddedKv(used, cells);
  const std::uint32_t kv_width = profile.kv_width();
  in.positions.resize(rows);
  in.k_idxs.resize(rows);
  in.out_ids.resize(rows);
  in.v_idxs.resize(std::size_t{rows} * kv_width);
  in.mask.assign(std::size_t{rows} * in.n_kv, -std::numeric_limits<float>::infinity());
  for (std::uint32_t i = 0; i < rows; ++i) {
    const std::uint32_t cell = n_past + i;
    in.positions[i] = static_cast<std::int32_t>(cell);
    in.k_idxs[i] = cell;
    in.out_ids[i] = static_cast<std::int32_t>(i);
    for (std::uint32_t j = 0; j < kv_width; ++j) {
      in.v_idxs[(std::size_t{i} * kv_width) + j] = (std::int64_t{j} * cells) + cell;
    }
    // Cells hold positions equal to their index; a token attends to every
    // occupied cell at or before its own position.
    for (std::uint32_t j = 0; j <= cell && j < in.n_kv; ++j) {
      in.mask[(std::size_t{i} * in.n_kv) + j] = 0.0f;
    }
  }
  return in;
}

float HalfToFloat(std::uint16_t half) {
  const std::uint32_t sign = std::uint32_t{half & 0x8000U} << 16U;
  std::uint32_t exponent = (half >> 10U) & 0x1FU;
  std::uint32_t mantissa = half & 0x3FFU;
  std::uint32_t bits = 0;
  if (exponent == 0x1FU) {
    bits = sign | 0x7F800000U | (mantissa << 13U);
  } else if (exponent != 0) {
    bits = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
  } else if (mantissa == 0) {
    bits = sign;
  } else {
    // Subnormal: normalize.
    exponent = 113;
    while ((mantissa & 0x400U) == 0) {
      mantissa <<= 1U;
      --exponent;
    }
    bits = sign | (exponent << 23U) | ((mantissa & 0x3FFU) << 13U);
  }
  return std::bit_cast<float>(bits);
}

std::expected<void, std::string> EmbedRows(std::span<const std::uint16_t> table,
                                           std::uint32_t width, std::uint32_t vocab,
                                           std::span<const std::int32_t> tokens,
                                           std::span<float> out) {
  if (table.size() != std::size_t{width} * vocab || out.size() != tokens.size() * width) {
    return Refused("the table or the output has the wrong size");
  }
  for (std::size_t t = 0; t < tokens.size(); ++t) {
    if (tokens[t] < 0 || std::cmp_greater_equal(tokens[t], vocab)) {
      return Refused(std::format("token {} is outside the vocabulary", tokens[t]));
    }
    const auto row = table.subspan(static_cast<std::size_t>(tokens[t]) * width, width);
    std::ranges::transform(row, out.subspan(t * width, width).begin(), HalfToFloat);
  }
  return {};
}

}  // namespace llmp::model
