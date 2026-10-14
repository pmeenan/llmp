// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The Qwen3.8 Flash Next architecture adapter (M3; docs/plan.md, "Model
// graphs and state"; docs/experiments/qwen38-native/README.md): the model's
// hyperparameters, the binding of its prepared artifact (the ModelOpt NVFP4
// and MXFP8 checkpoint as import_m3.py writes it) to the tensors the
// architecture reads, its state (the QSA layers' KV and indexer caches, the
// linear-attention layers' recurrent and convolution state, the n-gram
// layer's convolution history) as explicit, bounded state representations
// (D-068), and each chunk's host-built inputs for one sequence, as llama.cpp
// b29c606e2 builds them for qwen4exp (src/models/qwen4exp.cpp,
// src/llama-memory-hybrid-idx.cpp) with no speculative rollback planes and
// one stream.
//
// The hyperparameters are compiled in, a profile per supported checkpoint,
// and validated against the artifact's resource shapes when it is bound;
// the n-gram hash's constants are the artifact's data, checked before use
// (CheckQwen38PleHash).
//
// The model layer holds no vendor or kernel-module types; the GGML graph is
// lowered through the GGML kernel module.

#ifndef LLMP_MODEL_QWEN38_H_
#define LLMP_MODEL_QWEN38_H_

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

struct Qwen38Profile {
  std::string_view name;
  std::uint32_t layers = 0;
  std::uint32_t width = 0;     // n_embd
  std::uint32_t heads = 0;     // QSA query heads
  std::uint32_t kv_heads = 0;  // QSA KV heads
  std::uint32_t head_dim = 0;
  std::uint32_t rope_dims = 0;
  std::array<std::int32_t, 4> rope_sections{};  // interleaved mrope
  float rope_base = 0.0f;
  std::uint32_t lin_k_heads = 0;  // Gated DeltaNet
  std::uint32_t lin_v_heads = 0;
  std::uint32_t lin_head_dim = 0;  // key and value
  std::uint32_t conv = 0;          // its convolution kernel
  std::uint32_t experts = 0;
  std::uint32_t experts_used = 0;
  std::uint32_t expert_ffn = 0;
  std::uint32_t shared_ffn = 0;
  std::uint32_t hc = 0;  // hyper-connection streams
  std::uint32_t hc_rank = 0;
  std::uint32_t indexer_heads = 0;
  std::uint32_t indexer_head_dim = 0;
  std::uint32_t indexer_ratio = 0;   // tokens a block pools
  std::uint32_t indexer_budget = 0;  // tokens attention keeps (whole blocks, plus the tail)
  std::uint32_t ple_layer = 0;       // the n-gram embedding's layer
  std::uint32_t ngram = 0;
  std::uint32_t heads_per_ngram = 0;
  std::uint32_t ple_row = 0;  // values a table row holds
  std::uint32_t ple_conv = 0;
  std::int32_t ple_eos = 0;  // resets the n-gram window
  std::uint32_t vocab = 0;
  float rms_eps = 0.0f;

  // Every full_attention_interval-th layer is QSA, the rest linear.
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static): the profile's rule
  bool linear(std::uint32_t layer) const { return (layer + 1) % 4 != 0; }
  std::uint32_t hc_width() const { return hc * width; }
  std::uint32_t lin_k_width() const { return lin_k_heads * lin_head_dim; }
  std::uint32_t lin_v_width() const { return lin_v_heads * lin_head_dim; }
  std::uint32_t conv_channels() const { return (2 * lin_k_width()) + lin_v_width(); }
  std::uint32_t ple_heads() const { return (ngram - 1) * heads_per_ngram; }
  std::uint32_t ple_width() const { return ple_heads() * ple_row; }
  // The n-gram convolution's history: (kernel - 1) taps dilated by the
  // n-gram size.
  std::uint32_t ple_history() const { return (ple_conv - 1) * ngram; }
};

// Qwen3.8 Flash Next (Qwen/Qwen3.8-Flash-Next@de4b8e4d's text model, as
// Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6 carries it).
inline constexpr std::uint32_t kQwen38FlashContext = 262144;
inline constexpr std::uint32_t kQwen38FlashVocab = 248320;
const Qwen38Profile& Qwen38Flash();

// A bound tensor: its artifact resource (or expert array), whether it is a
// plain (checkpoint-layout) or GGML representation, its type name (GGML's,
// or the container dtype) and its shape in GGML order (`ne`; a plain
// [rows, cols] is ne [cols, rows]).
struct Qwen38Tensor {
  std::uint32_t index = 0;
  bool plain = false;
  std::string type;
  std::vector<std::uint64_t> ne;
  // An expert array's slice: its offset in each expert group and its
  // readable bytes (the slice plus the kernels' over-read, which the
  // artifact reserves and zeroes; docs/artifact-format.md).
  std::uint64_t group_offset = 0;
  std::uint64_t readable = 0;
};

// A linear's weights: MXFP8, E4M3 codes ne [k, n] and E8M0 scales ne
// [k / 32, n]; or, in the MTP drafter, whose linears the checkpoint keeps
// in BF16, `bf16` alone (GGML BF16 ne [k, n]; codes and scales unbound);
// or, in a GGUF checkpoint's artifact (Qwen38Format::kGguf), `matrix`
// alone: a GGML matrix ne [k, n] of the type the checkpoint stores (a
// k- or i-quant, Q8_0, BF16 or F32; the graph refuses one its products do
// not take).
struct Qwen38Mxfp8 {
  Qwen38Tensor codes, scales;
  Qwen38Tensor bf16;
  Qwen38Tensor matrix;
  bool is_bf16() const { return !bf16.type.empty(); }
  bool is_matrix() const { return !matrix.type.empty(); }
};

struct Qwen38Layer {
  bool linear = false;
  Qwen38Tensor hc_attn_norm, hc_attn_down, hc_attn_up, hc_attn_inject;
  Qwen38Tensor hc_ffn_norm, hc_ffn_down, hc_ffn_up, hc_ffn_inject;
  // Gated DeltaNet.
  Qwen38Mxfp8 qkv, z, beta, alpha, ssm_out;
  Qwen38Tensor dt_bias, ssm_a, conv1d, ssm_norm;
  // QSA. The indexer's query and key projection is one fused matrix in the
  // ModelOpt artifact (idx_qk: its query heads' rows, then the key's), and
  // two in a GGUF checkpoint's (idx_q and idx_k: llama.cpp's converter
  // splits it).
  Qwen38Mxfp8 q, k, v, o, idx_qk, idx_q, idx_k;
  Qwen38Tensor q_norm, k_norm, idx_q_norm, idx_k_norm;
  // The n-gram embedding (its layer only). The hash's constants are I64
  // tensors in the ModelOpt artifact; a GGUF checkpoint keeps them in its
  // metadata (ReadQwen38GgufHash), so they are unbound there.
  Qwen38Tensor ple_key, ple_value, ple_norm_key, ple_norm_query, ple_norm_conv, ple_conv1d;
  Qwen38Tensor ple_multipliers, ple_head_offsets, ple_head_vocab;
  // MoE: the router, the shared expert and its gate, the routed experts
  // (expert arrays, `ne` of one slice) and their per-expert global scales
  // (NVFP4's; a GGUF checkpoint's experts have none, so they are unbound).
  Qwen38Tensor router, shared_gate;
  Qwen38Mxfp8 gate_shexp, up_shexp, down_shexp;
  // The routed experts in GGML's layout (Qwen38Experts::kGgml): NVFP4, or
  // a GGUF checkpoint's types (each projection its own, as the
  // checkpoint's mix gives them).
  Qwen38Tensor gate_exps, up_exps, down_exps;
  // Or in the CUTLASS layout (kCutlass): four I8 arrays packed from each
  // expert group's start (kernels/ggml/moe_layout.h, checked contiguous):
  // gate's and up's codes as one block of 2f rows ne [w / 2, 2f], their
  // E4M3 scales swizzled in 512-byte atoms ne [512, atoms], then down's
  // codes ne [f / 2, w] and scales.
  Qwen38Tensor gate_up_codes, gate_up_scales, down_codes, down_scales;
  Qwen38Tensor gate_exps_scale, up_exps_scale, down_exps_scale;

  // The layer's routed-expert arrays of `experts`' layout, in group order.
  std::vector<const Qwen38Tensor*> expert_arrays(bool cutlass) const {
    if (cutlass) {
      return {&gate_up_codes, &gate_up_scales, &down_codes, &down_scales};
    }
    return {&gate_exps, &up_exps, &down_exps};
  }
};

// The routed experts' layout in the artifact: GGML's block_nvfp4 per
// projection (the first Qwen3.8 import), or the CUTLASS layout the importer
// writes since (docs/artifact-format.md, "Executable views").
enum class Qwen38Experts : std::uint8_t { kGgml, kCutlass };

// The checkpoint the artifact was imported from (docs/artifact-format.md):
// Mia's ModelOpt NVFP4 experts and MXFP8 linears, repacked by
// modelopt_qwen38.py; or a GGUF checkpoint (llama.cpp's qwen4exp tensors,
// unsloth's and others' quantizations), imported verbatim by layout.py:
// every tensor a GGML representation of the checkpoint's own type, the
// indexer's projection split in two, no expert or table scales, and the
// n-gram hash's constants in the kept GGUF metadata.
enum class Qwen38Format : std::uint8_t { kModelOpt, kGguf };

struct Qwen38Binding {
  std::vector<Qwen38Layer> layers;
  // ple_table: the ModelOpt artifact's packed NVFP4 rows (plain U8 [90,
  // rows]) with ple_table_scale; a GGUF artifact's GGML table [ple_row,
  // rows] of its type (ple_table_scale unbound).
  Qwen38Tensor token_embd, ple_table, ple_table_scale, output, output_hc_norm, output_hc_down,
      output_hc_up;
  Qwen38Experts experts = Qwen38Experts::kGgml;
  Qwen38Format format = Qwen38Format::kModelOpt;
  bool cutlass() const { return experts == Qwen38Experts::kCutlass; }
  bool gguf() const { return format == Qwen38Format::kGguf; }
};

// A resource or expert array as the adapter sees it.
struct Qwen38Resource {
  std::vector<std::string> roles;  // for an expert array, its name
  bool plain = false;
  std::string type;
  std::vector<std::uint64_t> ne;  // GGML order; an expert array's: one slice's
  bool expert_array = false;
  std::uint32_t count = 0;
  std::uint64_t group_offset = 0;  // an expert array's, as Qwen38Tensor's
  // Readable bytes: a slice's (as Qwen38Tensor's), or a resource's, its
  // bytes plus the kernels' over-read the artifact reserves (0: none
  // recorded).
  std::uint64_t readable = 0;
};

// Binds every tensor the profile reads to the resource of that role, which
// must have its representation, type and shape exactly; refused, naming the
// tensor, if one is missing or differs, if `architecture` is not
// "qwen4exp", or if the artifact binds a role Qwen3.8 does not read. A GGUF
// artifact (every resource and array a GGML representation, the n-gram
// table among them) is bound as Qwen38Format::kGguf: its matrices,
// routed experts and table take any GGML type whose rows are whole blocks
// (the artifact's reader has checked that), its norms and recurrent
// parameters F32, exactly as llama.cpp's converter writes them.
std::expected<Qwen38Binding, std::string> BindQwen38(const Qwen38Profile& profile,
                                                     std::string_view architecture,
                                                     std::span<const Qwen38Resource> resources);
std::expected<Qwen38Binding, std::string> BindQwen38(const Qwen38Profile& profile,
                                                     const artifact::Artifact& artifact);

// The n-gram hash's constants, the artifact's data (I64): each head's rows
// are offset + (hash mod vocab), so every head's range must lie within the
// table and the I32 row index the lookup takes. Refused otherwise, naming
// the head. Qwen38Chunk checks the ranges again against `table_rows`, so a
// hash built any other way is refused there, not used.
struct Qwen38PleHash {
  std::vector<std::uint64_t> multipliers;  // ngram
  std::vector<std::uint64_t> offsets;      // ple_heads
  std::vector<std::uint64_t> vocab;        // ple_heads
  std::uint64_t table_rows = 0;            // the table's rows it was checked against
};
std::expected<Qwen38PleHash, std::string> CheckQwen38PleHash(
    const Qwen38Profile& profile, std::span<const std::int64_t> multipliers,
    std::span<const std::int64_t> offsets, std::span<const std::int64_t> vocab,
    std::uint64_t table_rows);

// A GGUF artifact's n-gram hash, from the GGUF metadata it keeps
// (docs/artifact-format.md, import rule 7: the first source shard's,
// artifact/gguf_metadata.h's rules): qwen4exp.ple.layer_multipliers,
// .head_offsets and .head_vocab_sizes, checked as CheckQwen38PleHash checks
// them against `table_rows`. The hyperparameters the tensors' shapes do not
// fix must be the profile's too: the architecture, the layer and expert
// counts and widths, the experts used, the attention, indexer, Gated
// DeltaNet and hyper-connection extents, the indexer's budget and every
// layer's compression ratio (which places the QSA layers), the RMS epsilon,
// RoPE's dimensions, sections and base, the context, and the n-gram size,
// heads, layer, row, convolution and EOS. Refused, naming the key, if one
// is missing, mistyped or differs.
std::expected<Qwen38PleHash, std::string> ReadQwen38GgufHash(const Qwen38Profile& profile,
                                                             const artifact::Artifact& artifact,
                                                             std::uint64_t table_rows);
// The same from the kept metadata file's bytes.
std::expected<Qwen38PleHash, std::string> ReadQwen38GgufHash(const Qwen38Profile& profile,
                                                             std::span<const std::byte> metadata,
                                                             std::uint64_t table_rows);

// The widest chunk a state admits (the harness measures up to 2,048 rows).
inline constexpr std::uint32_t kQwen38MaxRows = 8192;

// The widest chunk Qwen38 admits at `context`.
std::uint32_t Qwen38MostRows(std::uint32_t context, bool host_masks = true);

// State region byte size at context and chunk bounds.
std::uint64_t Qwen38StateBytes(const Qwen38Profile& profile, std::uint32_t context,
                               std::uint32_t max_rows);
std::uint64_t Qwen38MtpStateBytes(const Qwen38Profile& profile, std::uint32_t context,
                                  std::uint32_t max_rows);

// The n-gram rows of the token at `position` (exposed for tests):
// ple_heads rows, llm_graph_input_ple::set_input's hash.
std::vector<std::int32_t> Qwen38PleRows(const Qwen38Profile& profile, const Qwen38PleHash& hash,
                                        std::span<const std::int32_t> history,
                                        std::uint32_t position);

// ---------------------------------------------------------------- the MTP drafter

// Qwen3.8's MTP block (D-068's stored MTP layer), its own drafter artifact
// (architecture qwen4exp-mtp; docs/experiments/artifact-layout/
// modelopt_qwen38.py plan_mtp), as vLLM's Qwen3_8FlashNextMTP reads it
// (vllm/models/qwen3_8_flash_next/nvidia/mtp.py at 8e685d198, the oracle's
// engine): one full-attention layer (hyper-connections, QSA with its
// indexer, the routed experts in the CUTLASS layout and the gated shared
// expert) whose linears, indexer projection, router and shared expert are
// BF16; fc_embedding and fc_hidden with their norms, which fuse the next
// token's embedding and the target's streams before the head's mix; and its
// own final mixer. It holds no token table or head: the target's are bound
// (Qwen38Binding::token_embd and ::output), as vLLM loads the same two.
struct Qwen38MtpBinding {
  Qwen38Layer layer;
  Qwen38Tensor fc_embd, fc_hidden, norm_embd, norm_hidden;
  Qwen38Tensor output_hc_norm, output_hc_down, output_hc_up;
  // Optional paired selected BF16 head and I32 [1, rows] original token IDs.
  // IDs are strictly ascending, checked after loading before any draft.
  Qwen38Tensor draft_output, draft_ids;
  bool selected_head() const { return !draft_output.type.empty(); }
};

// Refused, naming the tensor, as BindQwen38 is, and if `architecture` is
// not "qwen4exp-mtp".
std::expected<Qwen38MtpBinding, std::string> BindQwen38Mtp(
    const Qwen38Profile& profile, std::string_view architecture,
    std::span<const Qwen38Resource> resources);
std::expected<Qwen38MtpBinding, std::string> BindQwen38Mtp(const Qwen38Profile& profile,
                                                           const artifact::Artifact& artifact);

std::expected<void, std::string> CheckQwen38DraftIds(std::span<const std::int32_t> ids,
                                                     std::uint32_t vocab);

}  // namespace llmp::model

#endif  // LLMP_MODEL_QWEN38_H_
