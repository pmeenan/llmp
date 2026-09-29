// SPDX-FileCopyrightText: 2026 jitLLM contributors
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
// built from this adapter by the GGML kernel module
// (kernels/ggml/qwen38_graph.h).

#ifndef JITLLM_MODEL_QWEN38_H_
#define JITLLM_MODEL_QWEN38_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "model/state.h"

namespace jitllm::artifact {
class Artifact;
}

namespace jitllm::model {

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

// An MXFP8 matrix: E4M3 codes ne [k, n] and E8M0 scales ne [k / 32, n]; or,
// in the MTP drafter, whose linears the checkpoint keeps in BF16, `bf16`
// alone (GGML BF16 ne [k, n]; codes and scales unbound).
struct Qwen38Mxfp8 {
  Qwen38Tensor codes, scales;
  Qwen38Tensor bf16;
  bool is_bf16() const { return !bf16.type.empty(); }
};

struct Qwen38Layer {
  bool linear = false;
  Qwen38Tensor hc_attn_norm, hc_attn_down, hc_attn_up, hc_attn_inject;
  Qwen38Tensor hc_ffn_norm, hc_ffn_down, hc_ffn_up, hc_ffn_inject;
  // Gated DeltaNet.
  Qwen38Mxfp8 qkv, z, beta, alpha, ssm_out;
  Qwen38Tensor dt_bias, ssm_a, conv1d, ssm_norm;
  // QSA.
  Qwen38Mxfp8 q, k, v, o, idx_qk;
  Qwen38Tensor q_norm, k_norm, idx_q_norm, idx_k_norm;
  // The n-gram embedding (its layer only).
  Qwen38Tensor ple_key, ple_value, ple_norm_key, ple_norm_query, ple_norm_conv, ple_conv1d;
  Qwen38Tensor ple_multipliers, ple_head_offsets, ple_head_vocab;
  // MoE: the router, the shared expert and its gate, the routed experts
  // (expert arrays, `ne` of one slice) and their per-expert global scales.
  Qwen38Tensor router, shared_gate;
  Qwen38Mxfp8 gate_shexp, up_shexp, down_shexp;
  // The routed experts in GGML's layout (Qwen38Experts::kGgml): NVFP4.
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

struct Qwen38Binding {
  std::vector<Qwen38Layer> layers;
  Qwen38Tensor token_embd, ple_table, ple_table_scale, output, output_hc_norm, output_hc_down,
      output_hc_up;
  Qwen38Experts experts = Qwen38Experts::kGgml;
  bool cutlass() const { return experts == Qwen38Experts::kCutlass; }
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
  std::uint64_t readable = 0;
};

// Binds every tensor the profile reads to the resource of that role, which
// must have its representation, type and shape exactly; refused, naming the
// tensor, if one is missing or differs, if `architecture` is not
// "qwen4exp", or if the artifact binds a role Qwen3.8 does not read.
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

// ---------------------------------------------------------------- state

// One sequence's state for `context` positions and chunks of at most
// `max_rows` rows, in one region at 256-byte aligned offsets. The QSA
// caches hold a cell per position, cells = pad(context, 256), position p in
// cell p (as llama.cpp's unified cache places one sequence); attention reads
// the first pad(positions, 256) cells, masking the rest. Everything starts
// zeroed: the recurrent and convolution state of an empty sequence is zero,
// as llama.cpp's cleared state rows are.
struct Qwen38StateTensor {
  enum class Kind : std::uint8_t {
    kK,         // F16 [head_dim · kv_heads, cells]
    kV,         // F16 [head_dim · kv_heads, cells]
    kIndexerK,  // F32 [indexer_head_dim, cells]: the indexer's raw keys
    // BF16 [indexer_head_dim, cells / indexer_ratio]: each complete block's
    // pooled key, normalized and rotated (the fast graph writes each once,
    // when its block completes, and reads them; kernels/ggml/jitllm_ops.h
    // jitllm.qsa.pool)
    kIndexerBlocks,
    kConv,       // F32 [(conv - 1) · channels]: the last conv - 1 inputs, time fastest
    kRecurrent,  // F32 [head_dim · head_dim · v_heads]
    kPleConv,    // F32 [ple_history · hc_width]: the n-gram layer's
  };
  Kind kind = Kind::kK;
  std::uint32_t layer = 0;
  bool f16 = false;  // two-byte elements (F16, or kIndexerBlocks' BF16)
  std::uint64_t ne0 = 0;
  std::uint64_t ne1 = 0;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
};

struct Qwen38StateLayout {
  std::uint32_t context = 0;
  std::uint32_t max_rows = 0;
  std::uint32_t cells = 0;
  std::vector<Qwen38StateTensor> tensors;
  std::uint64_t bytes = 0;

  std::int64_t Find(std::uint32_t layer, Qwen38StateTensor::Kind kind) const;
  // As D-068 representations: the attention and indexer caches (one
  // fixed-size block sized for the context, append only) and the recurrent
  // and convolution state (one fixed-size block, overwritten each chunk:
  // rollback needs the snapshot planes a drafter brings).
  std::vector<StateRepresentation> Representations() const;
};

// The widest chunk a state admits (the harness measures up to 2,048 rows).
inline constexpr std::uint32_t kQwen38MaxRows = 8192;

// Refused if the context or chunk bound is zero, the context is past the
// graph's I32 positions, a chunk is longer than the context or than
// kQwen38MaxRows, or the profile is not Qwen3.8's; and, with `host_masks`
// (the graphs that select over masks of every cell by every row: the
// reference and unfused forms, kernels/ggml/qwen38_graph.h), if a chunk's
// F32 [padded context, rows] tensors (x 4 bytes, a 32-bit stride in GGML's
// flash attention and ggml_permute: RE-037) would pass I32 bytes. The fast
// graph builds no such tensor.
std::expected<Qwen38StateLayout, std::string> Qwen38State(const Qwen38Profile& profile,
                                                          std::uint32_t context,
                                                          std::uint32_t max_rows,
                                                          bool host_masks = true);

// The widest chunk Qwen38State admits at `context`: the context,
// kQwen38MaxRows and, with `host_masks`, the F32 [n_kv, rows] tensors' I32
// bytes (0 when none, or when the context is refused whatever the chunk).
std::uint32_t Qwen38MostRows(std::uint32_t context, bool host_masks = true);

// ---------------------------------------------------------------- chunk inputs

// QSA's block tables for a chunk (llama_memory_hybrid_idx::set_input_qsa,
// one sequence, cell = position): which block each cell pools into, each
// block's cells and RoPE position, and each token's per-block bias.
struct Qwen38QsaInputs {
  std::uint32_t blocks = 0;               // ceil(n_kv / ratio)
  std::vector<std::int32_t> cell_block;   // n_kv
  std::vector<std::int32_t> block_cells;  // ratio x blocks
  std::vector<std::int32_t> block_pos;    // 4 x blocks, section-major
  std::vector<float> bias;                // blocks x rows
};

struct Qwen38ChunkInputs {
  std::uint32_t n_past = 0;
  std::uint32_t rows = 0;
  std::uint32_t n_kv = 0;               // cells attention reads, pad(n_past + rows, 256)
  std::vector<std::int32_t> tokens;     // rows
  std::vector<std::int32_t> positions;  // 4 x rows (mrope sections, all the position)
  std::vector<std::int64_t> cells;      // rows: each token's cache cell
  std::vector<std::uint16_t> mask;      // F16 bits [n_kv, rows]: 0 visible, -inf not
  std::vector<float> mask_f32;          // the same in F32, for the indexer's scores
                                        // (both empty where not asked for)
  std::vector<std::int32_t> ple_rows;   // ple_heads x rows: the n-gram table's rows
  bool qsa_select = false;              // the indexer's budget is below n_kv
  Qwen38QsaInputs qsa;                  // when qsa_select
};

// A chunk's inputs: `rows` tokens at positions n_past onwards; `history`
// holds the sequence's tokens from position 0 through the chunk's last (the
// n-gram hash reads each token's predecessors). Refused if the chunk is
// empty, runs past the layout's context, is longer than its chunk bound,
// the history is not n_past + rows tokens, or a token is outside the
// vocabulary. With `selection_masks` false, a chunk whose QSA selects gets
// neither host-built masks nor block tables (only `qsa_select` and
// `qsa.blocks`): the fast graph selects on the device from the cached block
// keys and attends the kept cells alone (kernels/ggml/qwen38_graph.h). The
// masks are [n_kv, rows] and the bias [n_kv / ratio, rows]: at a context of
// 8,192 192 MiB a chunk, growing with the context.
std::expected<Qwen38ChunkInputs, std::string> Qwen38Chunk(const Qwen38Profile& profile,
                                                          const Qwen38StateLayout& state,
                                                          const Qwen38PleHash& hash,
                                                          std::span<const std::int32_t> history,
                                                          std::uint32_t n_past, std::uint32_t rows,
                                                          bool selection_masks = true);

// The n-gram rows of the token at `position` (exposed for tests):
// ple_heads rows, llm_graph_input_ple::set_input's hash.
std::vector<std::int32_t> Qwen38PleRows(const Qwen38Profile& profile, const Qwen38PleHash& hash,
                                        std::span<const std::int32_t> history,
                                        std::uint32_t position);

// The positions, cells, masks and QSA tables of `rows` rows at n_past over
// a cache of `cells` cells (position p in cell p), attention reading `read`
// cells (the chunk's n_kv: at least pad(n_past + rows, 256), at most
// `cells`, a multiple of 256): Qwen38Chunk's, without tokens or n-gram rows
// (the MTP drafter's passes, which share one n_kv). Refused if the rows do
// not fit.
std::expected<Qwen38ChunkInputs, std::string> Qwen38Rows(const Qwen38Profile& profile,
                                                         std::uint32_t cells, std::uint32_t n_past,
                                                         std::uint32_t rows, std::uint32_t read,
                                                         bool selection_masks);

inline constexpr std::uint16_t kQwen38HalfZero = 0x0000;
inline constexpr std::uint16_t kQwen38HalfNegInf = 0xFC00;

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

// The drafter's state, one region: its layer's F16 K and V caches and F32
// indexer keys (a cell per position, as the target's QSA layers'), its BF16
// block keys (kIndexerBlocks' layout), and the streams it reads:
// `hidden_rows` rows of the target's streams before the head's mix (F32
// [hc_width]), row 0 the pending one a prefill chunk leaves
// (docs/experiments/qwen38-mtp/). A D-068 representation: the caches append,
// and cells past the committed positions (a draft's) are rewritten before
// any row reads them, a block's key when the pass that writes its last cell
// runs.
struct Qwen38MtpState {
  std::uint32_t context = 0;
  std::uint32_t cells = 0;
  std::uint32_t hidden_rows = 0;
  std::uint64_t k = 0;  // offsets
  std::uint64_t v = 0;
  std::uint64_t indexer = 0;
  std::uint64_t blocks = 0;
  std::uint64_t hidden = 0;
  std::uint64_t bytes = 0;
  std::vector<StateRepresentation> Representations() const;
};
// For the target's `state` (its context and chunk bound): hidden_rows =
// max_rows + 1.
std::expected<Qwen38MtpState, std::string> Qwen38MtpStateOf(const Qwen38Profile& profile,
                                                            const Qwen38StateLayout& state);

// What a speculative verify saves so that its accepted rows' writes, and
// nothing else, reach the target's recurrent, convolution and n-gram
// state (Qwen38Commit replays them; the verify writes none of that state):
// per linear-attention layer, each row's convolution output (the
// recurrence's q, k and v, F32 [channels]), its input (the QKV rows, F32
// [channels], for the convolution history), its gate and beta (F32
// [v_heads] each); and the n-gram layer's convolution input rows (F32
// [hc_width]). Each part a [width, rows] block at a 256-byte aligned offset.
struct Qwen38CommitLayout {
  std::uint32_t rows = 0;
  std::uint32_t channels = 0;
  std::uint32_t v_heads = 0;
  std::uint32_t hc_width = 0;
  std::vector<std::uint32_t> layers;  // the linear-attention layers, in order
  std::uint64_t layer_bytes = 0;      // one layer's block
  std::uint64_t bytes = 0;
  // Layer `layers[i]`'s parts.
  std::uint64_t conv_out(std::size_t i) const { return i * layer_bytes; }
  std::uint64_t qkv(std::size_t i) const { return conv_out(i) + Part(channels); }
  std::uint64_t gate(std::size_t i) const { return qkv(i) + Part(channels); }
  std::uint64_t beta(std::size_t i) const { return gate(i) + Part(v_heads); }
  std::uint64_t ple() const { return layers.size() * layer_bytes; }
  std::uint64_t Part(std::uint32_t width) const {
    return ((std::uint64_t{width} * rows * 4) + 255) / 256 * 256;
  }
};
// For verifies of at most `rows` rows (1 to 8).
std::expected<Qwen38CommitLayout, std::string> Qwen38Commit(const Qwen38Profile& profile,
                                                            std::uint32_t rows);

}  // namespace jitllm::model

#endif  // JITLLM_MODEL_QWEN38_H_
