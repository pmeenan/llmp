// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The Qwen2 dense architecture adapter, as small as backend-proof P2 needs
// it (docs/architecture.md#adapters-and-request-programs; D-068): the
// model's hyperparameters, the binding of a prepared artifact's resources to
// the tensors the architecture reads, and each chunk's host-built inputs
// for a single-sequence F16 KV cache laid out as llama.cpp lays it out
// (docs/backend-proof.md, Tier E: n_kv padding, the KV cell layout, the
// mask, positions and output rows).
//
// M2 has no GGUF reader or importer, so the hyperparameters are compiled
// in: a profile per supported checkpoint, validated against the artifact's
// resource shapes when it is bound. The profile's scalars (RoPE base and
// original context, the norm epsilon, head counts) were cross-checked
// against the checkpoint's GGUF key/values by a reference-side script
// (docs/experiments/backend-proof-p2/gguf_profile_check.py). This is a
// harness-level model description, not a format: nothing here reads or
// writes model metadata.
//
// The model layer holds no vendor or kernel-module types; the GGML graph
// that computes a chunk is built from this adapter by the GGML kernel
// module (kernels/ggml/qwen2_graph.h).

#ifndef LLMP_MODEL_QWEN2_H_
#define LLMP_MODEL_QWEN2_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace llmp::artifact {
class Artifact;
}

namespace llmp::model {

// A Qwen2 dense checkpoint's hyperparameters.
struct Qwen2Profile {
  std::string_view name;
  std::uint32_t layers = 0;
  std::uint32_t width = 0;     // n_embd
  std::uint32_t heads = 0;     // query heads
  std::uint32_t kv_heads = 0;  // key and value heads
  std::uint32_t head_dim = 0;  // also the RoPE dimensions (NEOX over the whole head)
  std::uint32_t ffn = 0;       // feed-forward width
  std::uint32_t vocab = 0;
  std::uint32_t train_context = 0;  // RoPE's original context (n_ctx_orig)
  float rms_eps = 0.0f;
  float rope_base = 0.0f;
  // Weights of matrices; norms and biases are F32.
  std::string_view weight_type;

  std::uint32_t kv_width() const { return kv_heads * head_dim; }
};

// Qwen2.5-0.5B-Instruct (the FP16 fixture of docs/first-slice.md).
const Qwen2Profile& Qwen25Instruct05B();

// The tensors of one layer, as artifact resource indices.
struct Qwen2Layer {
  std::uint32_t attn_norm = 0;
  std::uint32_t q = 0, q_bias = 0;
  std::uint32_t k = 0, k_bias = 0;
  std::uint32_t v = 0, v_bias = 0;
  std::uint32_t out = 0;
  std::uint32_t ffn_norm = 0;
  std::uint32_t gate = 0, up = 0, down = 0;
};

struct Qwen2Binding {
  std::vector<Qwen2Layer> layers;
  std::uint32_t token_embd = 0;
  std::uint32_t output_norm = 0;
  std::uint32_t output = 0;  // token_embd's resource when the output head is tied
};

// A resource as the adapter sees it: every name it is bound to (its own and
// its tied aliases), and its GGML type and `ne`.
struct ResourceShape {
  std::vector<std::string> roles;
  std::string type;
  std::vector<std::uint64_t> ne;
};

// Binds every tensor `profile` reads to the resource of that role, which
// must have the profile's type and shape exactly; refused, naming the
// tensor, if one is missing or differs, if `architecture` is not "qwen2",
// or if a resource is bound to a role the architecture does not read.
std::expected<Qwen2Binding, std::string> BindQwen2(const Qwen2Profile& profile,
                                                   std::string_view architecture,
                                                   std::span<const ResourceShape> resources);
// The same over a validated v0 artifact (artifact/artifact.h).
std::expected<Qwen2Binding, std::string> BindQwen2(const Qwen2Profile& profile,
                                                   const artifact::Artifact& artifact);

// llama.cpp's n_kv (llama_kv_cache::get_n_kv at b29c606e2): the attended
// cells, the used cells padded to 256 and at most the cache's size.
std::uint32_t PaddedKv(std::uint32_t used_cells, std::uint32_t cells);

// One chunk's inputs built on the host, for a single-sequence cache of
// `cells` cells that holds positions [0, n_past) in cells [0, n_past), and
// this chunk's `rows` tokens at positions (and cells) n_past onwards: what
// llama.cpp's set_input functions write for that cache.
struct ChunkInputs {
  std::uint32_t n_kv = 0;
  std::vector<std::int32_t> positions;  // rows
  std::vector<std::int64_t> k_idxs;     // rows: each token's cell
  // rows × kv_width: the transposed V cache's element per head element
  // (flash attention off), j × cells + cell for element j.
  std::vector<std::int64_t> v_idxs;
  // rows × n_kv: 0 where the token attends to the cell (an occupied cell at
  // or before its position), -inf elsewhere.
  std::vector<float> mask;
  std::vector<std::int32_t> out_ids;  // rows: every row is an output
};

std::expected<ChunkInputs, std::string> Qwen2ChunkInputs(const Qwen2Profile& profile,
                                                         std::uint32_t cells, std::uint32_t n_past,
                                                         std::uint32_t rows);

// The embedding rows of `tokens` from an F16 token table of `vocab` rows of
// `width` elements, widened exactly to F32 (llama.cpp looks them up on the
// host). Refused if a token is outside the table or `out` is not
// tokens × width.
std::expected<void, std::string> EmbedRows(std::span<const std::uint16_t> table,
                                           std::uint32_t width, std::uint32_t vocab,
                                           std::span<const std::int32_t> tokens,
                                           std::span<float> out);

// IEEE binary16 to binary32, exactly (every half is a float).
float HalfToFloat(std::uint16_t half);

}  // namespace llmp::model

#endif  // LLMP_MODEL_QWEN2_H_
