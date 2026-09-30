// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The GGML graph of one Qwen3.8 Flash Next chunk (model/qwen38.h), built
// with GGML's graph functions over tensor descriptors (tensors.h) after
// llama.cpp b29c606e2's qwen4exp graph (src/models/qwen4exp.cpp and the
// llm_graph_context parts it calls, MIT): hyper-connections, the n-gram
// embedding layer, Gated DeltaNet with the fused gated delta rule, QSA with
// its indexer, the MoE with the shared expert, and the head. Settings: one
// sequence and stream, F16 caches, flash attention, no speculative rollback
// planes, no LoRA. The oracle is Mia's vLLM on the same checkpoint, not
// llama.cpp, so the port keeps upstream's operation plan but not its node
// order; where they differ:
//   - the checkpoint's formats: MXFP8 products run jitLLM's own operations
//     (jitllm_ops.h), the vector product up to 8 rows and otherwise the
//     weights dequantized to BF16 for GGML's float product; the n-gram
//     table's NVFP4 rows are gathered by jitllm.nvfp4.get_rows; routed
//     experts are GGML NVFP4 with each expert's global scale applied after
//     its product, as llama.cpp's build_lora_mm_id applies a `_s` tensor;
//   - the indexer's fused q/k projection is one product whose rows are
//     viewed (llama.cpp's converter splits it);
//   - state is written back with set_rows at row 0 of each state tensor
//     (llama.cpp copies into a view of its recurrent cache);
//   - the F32 copy of the causal mask the indexer adds is an input, not a
//     cast of the F16 one, and the indexer's key cache is F32 (GGML's row
//     gather takes F32 or BF16 rows);
//   - the shared expert's gate (one value a token) is the gate row, read as
//     F32, times each token, summed: GGML's products refuse a one-row
//     output;
//   - the chunk's rows are transposed into packed rows before the
//     convolution histories are concatenated to them;
//   - when attention reads no more cells than the indexer's budget keeps,
//     the selection keeps every cell, so its scoring is not built (the
//     indexer's keys are still cached);
//   - the rows the head computes are gathered before the final mixer
//     (llama.cpp gathers them in the last layer);
//   - with Qwen38GraphOptions::fused and ::exact (the reference form) the
//     hyper-connections' and the MoE output's elementwise nodes run as
//     jitLLM's fusions of them, and wide float products read activations
//     converted to BF16 once (jitllm_ops.h), with the unfused graph's
//     result;
//   - by default (fused, not exact: the fast form, D-085's speed before
//     bit exactness) the MXFP8 products past 8 rows run on tensor cores
//     over activations quantized to MXFP8, as the oracle runs them; each
//     block's output is combined into the streams by the next mix's
//     jitllm.hc.prep, which also normalizes them into BF16 and gives the
//     next combine's logits, and the mix reads those and the BF16 up
//     product; the router's softmax, top experts and the shared expert's
//     gate are one kernel; Gated DeltaNet's QKV and z rows are BF16 and its
//     gated norm is quantized for the output product in one pass; QSA's
//     heads are normalized and rotated in one pass, and the output gate is
//     fused with the output product's quantization. Each is checked against
//     an FP64 reference (tests/unit/qwen38_fast_test.cc), and the model
//     against the oracle coarsely (docs/experiments/qwen38-native/README.md);
//   - and in the fast form QSA's cost per token does not grow with the
//     context but for the indexer's scoring (docs/experiments/long-context/):
//     each block's pooled, normalized and rotated key is cached in the state
//     once, when the chunk that completes the block runs (jitllm.qsa.pool;
//     the reference form pools every block again at every step); the
//     selection scores the cached keys and keeps its cells on the device at
//     any depth, deterministically, ties to the lower cell
//     (jitllm.qsa.topk); and attention reads the kept cells alone
//     (jitllm.qsa.attn) rather than every cell under a mask. No tensor of
//     every cell by every row is built, and no mask or block table on the
//     host.
//
// Routed experts are 3D weights [k, n, experts] at the caller's expert
// stride (the resident expert layout, docs/artifact-format.md#executable-views);
// the down projection's rows (640 elements) are not whole 512-element steps,
// so its weights are marked as padded (validate_ext.h
// MarkRowPaddingReadable): the caller's slab must hold the artifact's
// readable bytes past each slice, as the artifact reserves them.
//
// Every tensor is created unbound; bind the leaves, place the computed
// nodes, then bind the views (graph_plan.h BindViews). Nothing here
// launches.

#ifndef JITLLM_KERNELS_GGML_QWEN38_GRAPH_H_
#define JITLLM_KERNELS_GGML_QWEN38_GRAPH_H_

#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <utility>
#include <vector>

#include "ggml.h"
#include "kernels/ggml/tensors.h"
#include "model/qwen38.h"

namespace jitllm::kernels::ggml {

struct Qwen38ChunkShape {
  std::int64_t rows = 0;
  std::int64_t n_kv = 0;
  std::int64_t cells = 0;
  std::int64_t outputs = 0;  // rows whose logits are computed: the last `outputs`
  bool qsa_select = false;
  std::int64_t qsa_blocks = 0;

  bool operator==(const Qwen38ChunkShape&) const = default;
};

Qwen38ChunkShape Qwen38ShapeOf(const model::Qwen38StateLayout& state,
                               const model::Qwen38ChunkInputs& chunk, std::int64_t outputs);

struct Qwen38Mxfp8Tensors {
  ggml_tensor* codes = nullptr;   // I8 [k, n]
  ggml_tensor* scales = nullptr;  // I8 [k / 32, n]
  // Or, the MTP drafter's (model/qwen38.h Qwen38Mxfp8::bf16), BF16 [k, n]:
  // GGML's float product up to kQwen38Bf16Rows rows, else jitllm.gemm.bf16.
  ggml_tensor* bf16 = nullptr;
};

struct Qwen38LayerTensors {
  ggml_tensor* hc_attn_norm = nullptr;
  ggml_tensor* hc_attn_down = nullptr;
  ggml_tensor* hc_attn_up = nullptr;
  ggml_tensor* hc_attn_inject = nullptr;
  ggml_tensor* hc_ffn_norm = nullptr;
  ggml_tensor* hc_ffn_down = nullptr;
  ggml_tensor* hc_ffn_up = nullptr;
  ggml_tensor* hc_ffn_inject = nullptr;
  Qwen38Mxfp8Tensors qkv, z, beta, alpha, ssm_out;
  ggml_tensor* dt_bias = nullptr;
  ggml_tensor* ssm_a = nullptr;
  ggml_tensor* conv1d = nullptr;
  ggml_tensor* ssm_norm = nullptr;
  Qwen38Mxfp8Tensors q, k, v, o, idx_qk;
  ggml_tensor* q_norm = nullptr;
  ggml_tensor* k_norm = nullptr;
  ggml_tensor* idx_q_norm = nullptr;
  ggml_tensor* idx_k_norm = nullptr;
  ggml_tensor* ple_key = nullptr;
  ggml_tensor* ple_value = nullptr;
  ggml_tensor* ple_norm_key = nullptr;
  ggml_tensor* ple_norm_query = nullptr;
  ggml_tensor* ple_norm_conv = nullptr;
  ggml_tensor* ple_conv1d = nullptr;
  ggml_tensor* router = nullptr;
  ggml_tensor* shared_gate = nullptr;
  Qwen38Mxfp8Tensors gate_shexp, up_shexp, down_shexp;
  ggml_tensor* gate_exps = nullptr;  // [k, n, experts] at the caller's stride
  ggml_tensor* up_exps = nullptr;
  ggml_tensor* down_exps = nullptr;
  // The layer's expert slab as bytes [stride, experts] (the CUTLASS layout,
  // moe_layout.h), in place of the three GGML weights.
  ggml_tensor* experts = nullptr;
  ggml_tensor* gate_exps_scale = nullptr;  // F32 [experts]
  ggml_tensor* up_exps_scale = nullptr;
  ggml_tensor* down_exps_scale = nullptr;
  // State (bind at the state layout's offsets).
  ggml_tensor* cache_k = nullptr;     // F16 [head_dim · kv_heads, cells]
  ggml_tensor* cache_v = nullptr;     // F16 [head_dim · kv_heads, cells]
  ggml_tensor* cache_idx = nullptr;   // F32 [indexer_head_dim, cells]
  ggml_tensor* cache_pool = nullptr;  // BF16 [indexer_head_dim, cells / ratio]: block keys
  ggml_tensor* conv_state = nullptr;  // F32 [(conv - 1) · channels, 1]
  ggml_tensor* recurrent = nullptr;   // F32 [head² · v_heads, 1]
  ggml_tensor* ple_state = nullptr;   // F32 [ple_history · hc_width, 1]
  // A verify's saves (Qwen38GraphOptions::verify; bind at the commit
  // layout's offsets, model/qwen38.h Qwen38CommitLayout): F32 [width, rows].
  ggml_tensor* commit_conv = nullptr;  // the convolution's output [channels]
  ggml_tensor* commit_qkv = nullptr;   // its input [channels]
  ggml_tensor* commit_gate = nullptr;  // [v_heads]
  ggml_tensor* commit_beta = nullptr;  // [v_heads]
  ggml_tensor* commit_ple = nullptr;   // the n-gram layer's convolution input [hc_width]
};

struct Qwen38GraphOptions {
  // Each layer's routed-expert stride in bytes (nb[2] of the three expert
  // weights); 0 for the packed stride of one slice.
  std::vector<std::uint64_t> expert_stride;
  // jitLLM's fusions (jitllm_ops.h) in place of the GGML nodes they repeat:
  // the hyper-connections' combine, norm and mix, the experts' scales with
  // SwiGLU, and their weighted sum with the gated shared expert; and, above
  // kQwen38Bf16Rows rows, each float product's activations converted to
  // BF16 once for all the products that read them (GGML's cuBLAS path
  // converts them per product). false builds the unfused graph.
  bool fused = true;
  // With `fused`: the reference form, whose fusions repeat GGML's nodes bit
  // for bit and whose MXFP8 products past the vector product's columns run
  // on the weights dequantized to BF16 through cuBLAS (the unfused graph's
  // result, but for the chunked delta rule's order of sums). Otherwise the
  // fast form, the default (D-085: speed before bit exactness): those
  // products on tensor cores over activations quantized to MXFP8
  // (jitllm.mxfp8.*, as the oracle's vLLM runs the checkpoint's MXFP8
  // linears), judged against the oracle coarsely
  // (docs/experiments/qwen38-native/README.md).
  bool exact = false;
  // The routed experts' resident layout: GGML's block_nvfp4 slices (GGML's
  // mul_mat_id: MMVQ and MMQ), or the CUTLASS layout (moe_layout.h;
  // jitllm.moe.gemv up to 8 rows, else CUTLASS's grouped GEMM over rows
  // sorted by expert, jitllm_ops.h), which needs the fused graph and an
  // expert stride per layer.
  enum class Experts : std::uint8_t { kGgml, kCutlass };
  Experts experts = Experts::kGgml;
  // A speculative verify (the fast form; docs/experiments/qwen38-mtp/): the
  // recurrent, convolution and n-gram state are read but not written; each
  // linear-attention layer's recurrence inputs and convolution input, and
  // the n-gram layer's convolution input, are saved row by row (the layers'
  // commit_* tensors, rows `row_ids`) for the runner's commit of the
  // accepted rows (qwen38_commit.h). Its convolutions run jitllm.gdn.conv
  // at every width. Its KV and indexer cells are written as any chunk's.
  bool verify = false;
  // Each row's streams after the last layer (before the head's mix) set
  // into `streams` at rows `stream_rows`: what the MTP drafter reads.
  bool export_streams = false;
  // The rows `streams` holds (model/qwen38.h Qwen38MtpState::hidden_rows).
  std::int64_t stream_rows = 0;
  // Diagnostic only: retain routed-layer operands at at most three
  // layers of a small verify. No arithmetic or new nodes are introduced.
  std::uint64_t capture_routed = 0;
};

inline bool Qwen38RoutedCaptureFits(std::uint64_t mask, std::uint32_t layers) {
  return layers > 0 && layers < 64 && (mask >> layers) == 0 && std::popcount(mask) <= 3;
}

struct Qwen38RoutedTensors {
  std::uint32_t layer = 0;
  ggml_tensor* input = nullptr;       // F32 [width, 1, rows], before gate/up
  ggml_tensor* activation = nullptr;  // F32 [ffn, used, rows], after SwiGLU
  ggml_tensor* down = nullptr;        // F32 [width, used, rows], before scale2
  ggml_tensor* shared = nullptr;      // F32 [width, rows], before sigmoid gate
  ggml_tensor* gate = nullptr;        // F32 [1, rows], shared gate logits
  ggml_tensor* weights = nullptr;     // F32 [1, used, rows]
  ggml_tensor* ids = nullptr;         // I32 [used, rows]
  ggml_tensor* combined = nullptr;    // F32 [width, rows]
};

// The rows above which GGML's float products run on cuBLAS (MMVF and MMF
// stop at 16), so that the fused graph's BF16 activations feed the same
// cuBLAS call.
inline constexpr std::int64_t kQwen38Bf16Rows = 16;

struct Qwen38Graph {
  // Inputs, host-built (model/qwen38.h Qwen38ChunkInputs).
  ggml_tensor* tokens = nullptr;     // I32 [rows]
  ggml_tensor* positions = nullptr;  // I32 [4 · rows]
  ggml_tensor* cells = nullptr;      // I64 [rows]
  ggml_tensor* mask = nullptr;       // F16 [n_kv, rows, 1, 1] (none where the device selects)
  ggml_tensor* mask_f32 = nullptr;   // F32 [n_kv, rows] (QSA selection only)
  ggml_tensor* ple_rows = nullptr;   // I32 [ple_heads · rows]
  ggml_tensor* state_row = nullptr;  // I64 [1]: 0, the state tensors' only row
  ggml_tensor* row_zero = nullptr;   // I32 [1]: 0, for reading a weight row as F32
  ggml_tensor* out_ids = nullptr;    // I32 [outputs]: the rows the head computes
  // The host's selection tables (none where the device selects).
  ggml_tensor* cell_block = nullptr;   // I32 [n_kv] (QSA selection only)
  ggml_tensor* block_cells = nullptr;  // I32 [ratio · blocks]
  ggml_tensor* block_pos = nullptr;    // I32 [4 · blocks]
  ggml_tensor* block_bias = nullptr;   // F32 [blocks, rows]
  ggml_tensor* row_ids = nullptr;      // I64 [rows]: 0 .. rows - 1 (a verify's saves)
  ggml_tensor* stream_rows = nullptr;  // I64 [rows]: where each row's streams go
  // The MTP drafter's streams (export_streams), F32 [hc_width, rows]: bind
  // at its state's (model/qwen38.h Qwen38MtpState::hidden).
  ggml_tensor* streams = nullptr;
  // Weights.
  ggml_tensor* token_embd = nullptr;
  ggml_tensor* ple_table = nullptr;        // I8 [row bytes, rows]
  ggml_tensor* ple_table_scale = nullptr;  // F32 [1]
  ggml_tensor* output = nullptr;
  ggml_tensor* output_hc_norm = nullptr;
  ggml_tensor* output_hc_down = nullptr;
  ggml_tensor* output_hc_up = nullptr;
  std::vector<Qwen38LayerTensors> layers;
  ggml_tensor* logits = nullptr;  // F32 [vocab, outputs]
  // A verify's: each row's argmax (I32 [outputs], the lowest index among
  // equals, jitllm.argmax), the last node; the logits stay live beside it.
  ggml_tensor* argmax = nullptr;
  std::vector<ggml_tensor*> nodes;
  std::vector<Qwen38RoutedTensors> routed;
  // Intermediates under llama.cpp's callback names ("l_last-7", ...).
  std::vector<std::pair<std::string, ggml_tensor*>> named;

  // The host-built inputs, in the order they are copied (those the shape
  // does not use left out).
  std::vector<ggml_tensor*> inputs() const;
  ggml_tensor* Named(std::string_view name) const;
};

// How many tensors a chunk's graph creates at most, for TensorArena.
std::size_t Qwen38GraphTensors(const model::Qwen38Profile& profile);

// Builds the chunk's graph on `arena`. Refused if the shape is not one the
// state holds, a type is not one the operations take, an expert stride is
// not a whole number of the expert slices' blocks, or the arena lacks room.
std::expected<Qwen38Graph, KernelFailure> BuildQwen38Graph(TensorArena& arena,
                                                           const model::Qwen38Profile& profile,
                                                           const model::Qwen38Binding& binding,
                                                           const Qwen38ChunkShape& shape,
                                                           const Qwen38GraphOptions& options);

// ---------------------------------------------------------------- the MTP drafter

// The MTP drafter's graph (model/qwen38.h Qwen38MtpBinding), as vLLM's
// Qwen3_8FlashNextMultiTokenPredictor.forward computes it (the oracle's
// engine, vllm/models/qwen3_8_flash_next/nvidia/mtp.py at 8e685d198), on
// the target's fast-form kernels: for each row, the next token's embedding
// row (the target's table) under norm_embd and fc_embd, added to each of
// the target's streams under norm_hidden (over all four together) and
// fc_hidden (each stream); the full-attention layer (its mixes, QSA with
// its indexer over the drafter's own caches, the MoE), its output combined
// into the streams; then its final mixer and the target's head.
//
// Pass 0 takes `rows` rows at positions from the inputs, reading the
// streams from `streams` rows hidden_row..; each later pass takes one row at
// the next position, its token the previous pass's draft and its streams
// the previous pass's last row's combined streams (vLLM's scheme A). A
// pass's draft is the argmax of its last row's logits over the head's
// first `head_rows` rows (a lowest-token-ID prefix, or a prepared selected
// head with a map back to original token IDs; 0: every available row), the
// lowest token ID among equals. Without `head` (a
// prefill pass) nothing past the caches' writes is computed.
struct Qwen38MtpShape {
  std::int64_t rows = 0;  // pass 0's
  std::int64_t passes = 1;
  std::int64_t n_kv = 0;  // every pass's
  std::int64_t cells = 0;
  bool qsa_select = false;
  std::int64_t qsa_blocks = 0;
  bool head = false;
  std::int64_t head_rows = 0;
  bool confidence = false;  // also return each draft's softmax probability
  // Diagnostic only: keep each pass's unrounded mixed input and F32 head
  // logits for a post-run copy. Requires a head of at most 65,536 rows.
  bool capture_head = false;
  std::int64_t hidden_row = 0;   // pass 0's first streams row
  std::int64_t hidden_rows = 0;  // the streams' rows

  bool operator==(const Qwen38MtpShape&) const = default;
};

// One pass's host-built inputs (as Qwen38Graph's; model/qwen38.h
// Qwen38Rows builds them).
struct Qwen38MtpPass {
  ggml_tensor* tokens = nullptr;  // I32 [rows]: pass 0 only
  ggml_tensor* positions = nullptr;
  ggml_tensor* cells = nullptr;
  ggml_tensor* mask = nullptr;     // F16 [n_kv, rows]: a pass that selects has none
  ggml_tensor* out_ids = nullptr;  // unused by the drafter, kept for the shared builder
};

struct Qwen38MtpGraph {
  std::vector<Qwen38MtpPass> passes;
  ggml_tensor* state_row = nullptr;  // I64 [1]: 0
  ggml_tensor* row_zero = nullptr;   // I32 [1]: 0
  // The drafter's weights (its layer's caches: bind at its state's offsets).
  Qwen38LayerTensors layer;
  ggml_tensor* fc_embd = nullptr;
  ggml_tensor* fc_hidden = nullptr;
  ggml_tensor* norm_embd = nullptr;
  ggml_tensor* norm_hidden = nullptr;
  ggml_tensor* output_hc_norm = nullptr;
  ggml_tensor* output_hc_down = nullptr;
  ggml_tensor* output_hc_up = nullptr;
  // The target's token table and head.
  ggml_tensor* token_embd = nullptr;
  ggml_tensor* output = nullptr;
  ggml_tensor* draft_ids = nullptr;  // optional I32 [1, selected rows]
  ggml_tensor* streams = nullptr;    // F32 [hc_width, hidden_rows]: the drafter's state
  std::vector<ggml_tensor*> drafts;  // I32 [1] a pass (with its head)
  // I32 [1] a pass: its draft's softmax probability over the draft head's
  // rows, as F32 bits (an adaptive window's confidence).
  std::vector<ggml_tensor*> probabilities;
  std::vector<ggml_tensor*> head_inputs;  // F32 [width, 1], unrounded; capture_head only
  std::vector<ggml_tensor*> head_logits;  // F32 [head rows, 1]; capture_head only
  std::vector<ggml_tensor*> nodes;

  // The host-built inputs, in the order they are copied.
  std::vector<ggml_tensor*> inputs() const;
};

std::size_t Qwen38MtpGraphTensors(const model::Qwen38Profile& profile, std::int64_t passes);

// Builds the drafter's graph on `arena`, its experts at `expert_stride`.
// Refused if the shape is not one its state holds, a pass's rows past the
// first are not one, or the arena lacks room.
std::expected<Qwen38MtpGraph, KernelFailure> BuildQwen38MtpGraph(
    TensorArena& arena, const model::Qwen38Profile& profile, const model::Qwen38Binding& target,
    const model::Qwen38MtpBinding& drafter, const Qwen38MtpShape& shape,
    std::uint64_t expert_stride);

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_QWEN38_GRAPH_H_
