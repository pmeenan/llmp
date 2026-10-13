// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef LLMP_TESTS_UNIT_QWEN2_GGUF_NORMALIZATION_FIXTURE_H_
#define LLMP_TESTS_UNIT_QWEN2_GGUF_NORMALIZATION_FIXTURE_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <format>
#include <span>
#include <string>
#include <string_view>

#include "base/check.h"
#include "format/layout.h"
#include "import/checkpoint.h"

namespace llmp::test_support {
// Independent bounded header inventory of qwen2.5-0.5b-instruct-fp16.gguf
// (SHA-256 8e0ae260..., 291 tensors: 170 F16 + 121 F32, 1,260,477,952 payload
// bytes); synthetic extents, no payload.
inline import::Checkpoint Qwen2GgufCheckpoint(bool untied = true) {
  import::Checkpoint out{
      .container = "gguf", .architecture = "qwen2", .files = {{.name = "memory", .bytes = 0}}};
  out.metadata.push_back({"general.architecture", std::string("qwen2")});
  out.metadata.push_back({"qwen2.attention.head_count", std::int64_t{14}});
  out.metadata.push_back({"qwen2.attention.head_count_kv", std::int64_t{2}});
  out.metadata.push_back({"qwen2.attention.layer_norm_rms_epsilon", static_cast<double>(1e-6F)});
  out.metadata.push_back({"qwen2.block_count", std::int64_t{24}});
  out.metadata.push_back({"qwen2.context_length", std::int64_t{8192}});
  out.metadata.push_back({"qwen2.embedding_length", std::int64_t{896}});
  out.metadata.push_back({"qwen2.feed_forward_length", std::int64_t{4864}});
  out.metadata.push_back({"qwen2.rope.freq_base", 1000000.0});
  std::ranges::sort(out.metadata, {}, &import::MetaEntry::key);
  struct Descriptor {
    std::string_view name;
    std::array<std::uint64_t, 2> dims;
    std::uint8_t rank;
    format::Encoding encoding;
    bool per_layer;
  };
  constexpr std::array descriptors{
      Descriptor{"blk.{}.attn_k.bias", {128, 1}, 1, format::Encoding::kF32, true},
      Descriptor{"blk.{}.attn_k.weight", {896, 128}, 2, format::Encoding::kF16, true},
      Descriptor{"blk.{}.attn_norm.weight", {896, 1}, 1, format::Encoding::kF32, true},
      Descriptor{"blk.{}.attn_output.weight", {896, 896}, 2, format::Encoding::kF16, true},
      Descriptor{"blk.{}.attn_q.bias", {896, 1}, 1, format::Encoding::kF32, true},
      Descriptor{"blk.{}.attn_q.weight", {896, 896}, 2, format::Encoding::kF16, true},
      Descriptor{"blk.{}.attn_v.bias", {128, 1}, 1, format::Encoding::kF32, true},
      Descriptor{"blk.{}.attn_v.weight", {896, 128}, 2, format::Encoding::kF16, true},
      Descriptor{"blk.{}.ffn_down.weight", {4864, 896}, 2, format::Encoding::kF16, true},
      Descriptor{"blk.{}.ffn_gate.weight", {896, 4864}, 2, format::Encoding::kF16, true},
      Descriptor{"blk.{}.ffn_norm.weight", {896, 1}, 1, format::Encoding::kF32, true},
      Descriptor{"blk.{}.ffn_up.weight", {896, 4864}, 2, format::Encoding::kF16, true},
      Descriptor{"output.weight", {896, 151936}, 2, format::Encoding::kF16, false},
      Descriptor{"output_norm.weight", {896, 1}, 1, format::Encoding::kF32, false},
      Descriptor{"token_embd.weight", {896, 151936}, 2, format::Encoding::kF16, false},
  };
  const auto add = [&](std::string name, const Descriptor& d) {
    auto shape = format::ShapeOf(std::span(d.dims).first(d.rank));
    base::Check(shape.has_value(), "captured Qwen2 shape");
    format::TensorFormat f{
        .encoding = d.encoding, .layout = format::Layout::kDense, .shape = *shape};
    auto parts = format::PartsOf(f);
    base::Check(parts && parts->size() == 1, "captured Qwen2 part");
    auto at = (out.files[0].bytes + 31) / 32 * 32;
    out.tensors.push_back({.name = std::move(name),
                           .format = f,
                           .extent = {.offset = at, .bytes = (*parts)[0].bytes},
                           .dtype = format::Info(d.encoding).name,
                           .dims = {d.dims.begin(), d.dims.begin() + d.rank}});
    out.files[0].bytes = at + (*parts)[0].bytes;
  };
  for (const auto& d : descriptors) {
    if (!untied && d.name == "output.weight") continue;
    if (!d.per_layer) {
      add(std::string(d.name), d);
      continue;
    }
    for (unsigned layer = 0; layer < 24; ++layer)
      add(std::vformat(d.name, std::make_format_args(layer)), d);
  }
  return out;
}
}  // namespace llmp::test_support
#endif  // LLMP_TESTS_UNIT_QWEN2_GGUF_NORMALIZATION_FIXTURE_H_
