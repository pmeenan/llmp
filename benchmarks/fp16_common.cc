// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "fp16_common.h"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include "base/sha256.h"
#include "ggml.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/qwen2_graph.h"
#include "kernels/ggml/tensors.h"
#include "model/qwen2.h"

namespace llmp::benchmarks {
namespace {

namespace kg = kernels::ggml;

std::unexpected<std::string> Error(std::string what) { return std::unexpected(std::move(what)); }

std::uint64_t Address(const void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer); }

}  // namespace

std::expected<Trajectory, std::string> LoadTrajectory(std::string_view name,
                                                      const std::filesystem::path& tokens) {
  Trajectory t;
  t.name = std::string(name);
  std::ifstream file(tokens, std::ios::binary);
  if (!file) {
    return Error(std::format("cannot read {}", tokens.string()));
  }
  const std::string bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  if (name == "control") {
    std::istringstream in(bytes);
    std::int64_t id = 0;
    while (in >> id) {
      t.tokens.push_back(static_cast<std::int32_t>(id));
    }
    if (t.tokens.size() != 76) {
      return Error(std::format("control has 76 tokens, not {}", t.tokens.size()));
    }
    t.chunks.push_back(32);
    t.chunks.insert(t.chunks.end(), 44, 1);
    t.cells = 512;
    t.restore_after = 32;
  } else if (name == "heldout") {
    base::Sha256 hash;
    hash.Update(std::as_bytes(std::span(bytes)));
    const std::string digest = base::ToHex(hash.Finish());
    if (bytes.size() != std::size_t{1040} * 8 ||
        digest != "6dd8da897821f5615e7796f4d882795faf306a941f050e2eb68b619096e3fb0c") {
      return Error("not the declared held-out IDs");
    }
    t.chunks = {16, 17};
    t.chunks.insert(t.chunks.end(), 16, 1);
    t.chunks.push_back(512);
    t.chunks.insert(t.chunks.end(), 16, 1);
    std::size_t total = 0;
    for (const std::uint32_t c : t.chunks) {
      total += c;
    }
    for (std::size_t i = 0; i < total; ++i) {
      std::uint64_t id = 0;
      for (int b = 7; b >= 0; --b) {
        id = (id << 8U) | static_cast<unsigned char>(bytes[(i * 8) + static_cast<std::size_t>(b)]);
      }
      t.tokens.push_back(static_cast<std::int32_t>(id));
    }
    t.cells = 1024;
    t.restore_after = 33;
  } else {
    return Error("the trajectory is control or heldout");
  }
  return t;
}

std::expected<ChunkGraph, std::string> PlanChunk(const model::Qwen2Profile& profile,
                                                 const model::Qwen2Binding& binding,
                                                 const ChunkMemory& memory, std::uint32_t cells,
                                                 std::uint32_t rows, std::uint32_t n_kv,
                                                 bool fusion, const kg::DeviceChoices& choices) {
  ChunkGraph out;
  auto arena = kg::TensorArena::Create(kg::Qwen2GraphTensors(profile));
  if (!arena) {
    return Error(arena.error().detail);
  }
  out.arena.emplace(std::move(*arena));
  auto graph =
      kg::BuildQwen2Graph(*out.arena, profile, {.rows = rows, .n_kv = n_kv, .cells = cells});
  if (!graph) {
    return Error(graph.error().detail);
  }
  out.graph = std::move(*graph);
  kg::Qwen2Graph& g = out.graph;
  const auto bind = [&](ggml_tensor* t, std::uint32_t resource) {
    kg::TensorArena::Bind(t, memory.weight(resource));
  };
  const std::uint64_t layer_cache = std::uint64_t{profile.kv_width()} * cells * 2;
  for (std::uint32_t il = 0; il < profile.layers; ++il) {
    const auto& b = binding.layers[il];
    auto& l = g.layers[il];
    bind(l.attn_norm, b.attn_norm);
    bind(l.q, b.q);
    bind(l.q_bias, b.q_bias);
    bind(l.k, b.k);
    bind(l.k_bias, b.k_bias);
    bind(l.v, b.v);
    bind(l.v_bias, b.v_bias);
    bind(l.out, b.out);
    bind(l.ffn_norm, b.ffn_norm);
    bind(l.gate, b.gate);
    bind(l.up, b.up);
    bind(l.down, b.down);
    kg::TensorArena::Bind(l.k_cache, memory.kv + (std::uint64_t{2} * il * layer_cache));
    kg::TensorArena::Bind(l.v_cache, memory.kv + (((std::uint64_t{2} * il) + 1) * layer_cache));
  }
  bind(g.output_norm, binding.output_norm);
  bind(g.output, binding.output);

  // First pass: every computed tensor at its own address.
  constexpr std::uint64_t kDistinct = std::uint64_t{1} << 46U;
  for (ggml_tensor* input : g.inputs()) {
    kg::TensorArena::Bind(input, kDistinct - (std::uint64_t{1} << 40U));
  }
  kg::BindDistinct(g.nodes, kDistinct);
  auto first = kg::PlanGraph(g.nodes, fusion, choices);
  if (!first) {
    return Error(first.error().detail);
  }
  auto inputs = g.inputs();
  auto placement = kg::PlaceActivations(g.nodes, *first, inputs, 128);
  if (!placement) {
    return Error(placement.error().detail);
  }
  out.placement = std::move(*placement);
  if (memory.activations == 0) {  // measuring only
    out.plan = std::move(*first);
    return out;
  }
  if (out.placement.extent > memory.activation_bytes) {
    return Error("the activations exceed their region");
  }
  for (const auto& [tensor, offset] : out.placement.offsets) {
    kg::TensorArena::Bind(tensor, memory.activations + offset);
  }
  kg::BindViews(g.nodes);
  auto second = kg::PlanGraph(g.nodes, fusion, choices);
  if (!second) {
    return Error(second.error().detail);
  }
  if (!kg::SamePlan(*first, *second)) {
    return Error("the plan changed once the activations were placed");
  }
  out.plan = std::move(*second);
  // 128-byte alignment of every tensor bound (the recording sees no
  // addresses).
  for (const ggml_tensor* t : g.nodes) {
    for (const ggml_tensor* src : t->src) {
      if (src != nullptr && src->view_src == nullptr && Address(src->data) % 128 != 0) {
        return Error(std::format("{} is not 128-byte aligned", src->name));
      }
    }
    if (t->view_src == nullptr && Address(t->data) % 128 != 0) {
      return Error(std::format("node {} is not 128-byte aligned", ggml_op_desc(t)));
    }
  }
  return out;
}

}  // namespace llmp::benchmarks
