// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The GGML graph of one Qwen2 chunk (model/qwen2.h), built with GGML's
// graph functions over tensor descriptors (tensors.h) exactly as llama.cpp
// b29c606e2 builds it for the FP16 bridge's settings: flash attention off,
// an F16 K/V cache holding one sequence, every row an output, no LoRA and
// no control vector (src/models/qwen2.cpp and llm_graph_context's
// build_norm, build_qkv, build_attn, build_attn_mha and build_ffn;
// llama_kv_cache's cpy_k, cpy_v, get_k and get_v). The same operations,
// parameters, views and ggml_build_forward_expand calls give the same
// nodes in the same order (fusion.h GraphOrder), which is what the FP16
// gate's recorded plan follows (docs/backend-proof.md, Tier E).
//
// What llama.cpp runs on the CPU, the token embedding lookup, is not part
// of the graph: the looked-up rows are an input, as the scheduler's copy of
// them is to the bridge's GPU split. The host-built inputs are leaves in
// the order the bridge's scheduler copies them.
//
// Every tensor is created unbound. Bind the leaves (inputs, weights and the
// cache), give every computed node that is no view memory, then bind the
// views (graph_plan.h BindViews). Every profile builds this; nothing here launches.

#ifndef LLMP_KERNELS_GGML_QWEN2_GRAPH_H_
#define LLMP_KERNELS_GGML_QWEN2_GRAPH_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <vector>

#include "ggml.h"
#include "kernels/ggml/tensors.h"
#include "model/qwen2.h"

namespace llmp::kernels::ggml {

// A chunk's shape: its rows, the cells attention reads (n_kv, as
// model::PaddedKv gives it) and the cache's size in cells.
struct Qwen2ChunkShape {
  std::int64_t rows = 0;
  std::int64_t n_kv = 0;
  std::int64_t cells = 0;
};

struct Qwen2LayerTensors {
  ggml_tensor* attn_norm = nullptr;
  ggml_tensor* q = nullptr;
  ggml_tensor* q_bias = nullptr;
  ggml_tensor* k = nullptr;
  ggml_tensor* k_bias = nullptr;
  ggml_tensor* v = nullptr;
  ggml_tensor* v_bias = nullptr;
  ggml_tensor* out = nullptr;
  ggml_tensor* ffn_norm = nullptr;
  ggml_tensor* gate = nullptr;
  ggml_tensor* up = nullptr;
  ggml_tensor* down = nullptr;
  // The layer's F16 cache: K as [kv_width, cells], V transposed.
  ggml_tensor* k_cache = nullptr;
  ggml_tensor* v_cache = nullptr;
};

struct Qwen2Graph {
  // Host-built inputs, in the bridge's copy order: the embedding rows (F32
  // [width, rows]), positions (I32), K cells (I64), V elements (I64
  // [rows × kv_width]), the mask (F32 [n_kv, rows]) and output rows (I32).
  ggml_tensor* embd = nullptr;
  ggml_tensor* positions = nullptr;
  ggml_tensor* k_idxs = nullptr;
  ggml_tensor* v_idxs = nullptr;
  ggml_tensor* mask = nullptr;
  ggml_tensor* out_ids = nullptr;
  std::vector<Qwen2LayerTensors> layers;
  ggml_tensor* output_norm = nullptr;
  ggml_tensor* output = nullptr;
  // The F32 logits, [vocab, rows]: the last node.
  ggml_tensor* logits = nullptr;
  // Every node in GGML's order, views included.
  std::vector<ggml_tensor*> nodes;

  std::array<ggml_tensor*, 6> inputs() const {
    return {embd, positions, k_idxs, v_idxs, mask, out_ids};
  }
};

// How many tensors a chunk's graph creates at most, for TensorArena.
std::size_t Qwen2GraphTensors(const model::Qwen2Profile& profile);

// Builds the chunk's graph on `arena`. Refused if the shape is not one the
// cache holds (rows and n_kv at most the cells, n_kv at least rows) or the
// arena lacks room.
std::expected<Qwen2Graph, KernelFailure> BuildQwen2Graph(TensorArena& arena,
                                                         const model::Qwen2Profile& profile,
                                                         const Qwen2ChunkShape& shape);

}  // namespace llmp::kernels::ggml

#endif  // LLMP_KERNELS_GGML_QWEN2_GRAPH_H_
