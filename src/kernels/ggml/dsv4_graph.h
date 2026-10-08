// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The GGML graph of one DeepSeek V4 chunk (model/dsv4.h), built with GGML's
// graph functions over tensor descriptors (tensors.h) as llama.cpp b29c606e2
// builds it (src/models/deepseek4.cpp, MIT) for these settings: flash
// attention on, F16 caches, one sequence and stream, no speculative rollback
// planes, every row an output, no LoRA or control vector, and upstream's
// defaults for the model-level fused operations (dsv4_hc_pre, dsv4_hc_comb,
// dsv4_hc_post and the lightning indexer). The same operations, parameters,
// views and ggml_build_forward_expand calls give the same nodes in the same
// order (fusion.h GraphOrder), with four differences, none of which changes
// a computed value:
//   - the compressor state is read straight from the ring (llama.cpp's
//     restore copies run only with rollback planes, n_rs_seq > 0);
//   - a cache is viewed once at the width attention reads, where llama.cpp
//     views it at the cache's width and then narrows the view;
//   - the token embedding rows are an input, looked up on the host as
//     llama.cpp looks them up on the CPU;
//   - the indexer's Hadamard matrix is an input the transform never reads.
//
// That is the reference form. The fast plan (Dsv4GraphOptions::fused)
// departs from it also in the attention at depth (docs/experiments/
// long-context, phase 2): no concatenated K, the masks built on the device
// from each row's visible counts, llmpalooza's deterministic indexer, and the
// MMA kernel's sparse gather, so its host inputs carry no CSA, HCA or
// indexer mask (Dsv4Graph::csa_visible, hca_visible instead).
// A positive shape.outputs selects trailing rows before the final mix,
// norm and vocabulary head, preserving every trunk/state/feature row.
// Shape outputs = 0 leaves the reference/default all-row head unchanged.
//
// Speculation (M3; docs/experiments/dspark/) adds, by option: a verify's
// row-invariant plan (D-092: attention per query row; the planner picks
// the row-invariant products); the target's features and their injection
// into the DSpark drafter's ring after the logits (dflash.cpp's
// feature capture and KV injection); and BuildDsparkGraph, the drafter's
// own block (dflash.cpp graph_dsv4). Rollback is not llama.cpp's rollback
// planes: the runner saves the bytes a verify writes and restores the
// rejected rows' outside the graph (model/dsv4.h Dsv4ChunkWrites).
//
// Routed experts are 3D weights [k, n, experts] whose expert stride (nb[2])
// is the caller's: the GGUF's packed stride, or the stride of the repacked
// expert groups laid out at a uniform stride (the resident expert layout,
// docs/artifact-format.md#executable-views). GGML's mul_mat_id kernels
// address expert e at base + e·nb[2] and count the stride in whole blocks,
// so it must be a multiple of every expert projection's block size.
//
// Every tensor is created unbound. Bind the leaves (inputs, weights and the
// state), give every computed node that is no view memory, then bind the
// views (graph_plan.h BindViews). Every profile builds this; nothing here
// launches.

#ifndef LLMP_KERNELS_GGML_DSV4_GRAPH_H_
#define LLMP_KERNELS_GGML_DSV4_GRAPH_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ggml.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/tensors.h"
#include "model/dspark.h"
#include "model/dsv4.h"

namespace llmp::kernels::ggml {

// The measured compact expert scheduling floor for production DeepSeek
// prefill. Native model controls share it; direct operation controls may
// deliberately exercise smaller experimental shapes.
inline constexpr std::int64_t kDsv4CompactMinRows = 2048;

// A chunk's shape, from its host-built inputs.
struct Dsv4ChunkShape {
  std::int64_t rows = 0;
  // Plain one-row generation publishes its host-greedy-compatible I32 token.
  bool token = false;
  // The head's last rows; 0 keeps the reference/default's every-row head.
  // This never narrows the streams a DSpark feature capture reads.
  std::int64_t outputs = 0;
  std::int64_t raw_n_kv = 0;
  std::int64_t raw_cells = 0;
  std::int64_t csa_n_kv = 0;  // also the indexer's
  std::int64_t hca_n_kv = 0;
  std::int64_t csa_cells = 0;
  std::int64_t hca_cells = 0;
  std::int64_t csa_blocks = 0;
  std::int64_t hca_blocks = 0;
  std::int64_t csa_persist = 0;
  std::int64_t hca_persist = 0;
  std::int64_t csa_state_rows = 0;
  std::int64_t hca_state_rows = 0;
  bool operator==(const Dsv4ChunkShape&) const = default;
};

Dsv4ChunkShape Dsv4ShapeOf(const model::Dsv4StateLayout& state, const model::Dsv4ChunkInputs& chunk,
                           std::int64_t outputs = 0);

// One compressor's host-built inputs.
struct Dsv4CompInputs {
  ggml_tensor* state_pos = nullptr;    // I32 [rows]
  ggml_tensor* persist_src = nullptr;  // I32
  ggml_tensor* persist_dst = nullptr;  // I32
  ggml_tensor* read_idxs = nullptr;    // I32
  ggml_tensor* write_idxs = nullptr;   // I64 [blocks]
  ggml_tensor* write_pos = nullptr;    // I32 [blocks]
  ggml_tensor* mask = nullptr;         // F16 [n_kv, rows, 1, 1]; none in the fast plan
};

// A layer's weights (unbound leaves: bind each at its resource's address)
// and state (bind at the state layout's offsets).
struct Dsv4LayerTensors {
  ggml_tensor* attn_norm = nullptr;
  ggml_tensor* attn_sinks = nullptr;
  ggml_tensor* q_a = nullptr;
  ggml_tensor* q_a_norm = nullptr;
  ggml_tensor* q_b = nullptr;
  ggml_tensor* kv = nullptr;
  ggml_tensor* kv_norm = nullptr;
  ggml_tensor* out_a = nullptr;  // [heads·head / groups, o_lora, groups]
  ggml_tensor* out_b = nullptr;
  ggml_tensor* hc_attn_fn = nullptr;
  ggml_tensor* hc_attn_base = nullptr;
  ggml_tensor* hc_attn_scale = nullptr;
  ggml_tensor* hc_ffn_fn = nullptr;
  ggml_tensor* hc_ffn_base = nullptr;
  ggml_tensor* hc_ffn_scale = nullptr;
  ggml_tensor* comp_kv = nullptr;
  ggml_tensor* comp_gate = nullptr;
  ggml_tensor* comp_ape = nullptr;
  ggml_tensor* comp_norm = nullptr;
  ggml_tensor* idx_q_b = nullptr;
  ggml_tensor* idx_proj = nullptr;
  ggml_tensor* idx_comp_kv = nullptr;
  ggml_tensor* idx_comp_gate = nullptr;
  ggml_tensor* idx_comp_ape = nullptr;
  ggml_tensor* idx_comp_norm = nullptr;
  ggml_tensor* ffn_norm = nullptr;
  ggml_tensor* router = nullptr;
  ggml_tensor* router_bias = nullptr;  // routed layers
  ggml_tensor* tid2eid = nullptr;      // hash layers
  ggml_tensor* up_exps = nullptr;      // [width, ffn, experts] at the caller's stride
  ggml_tensor* gate_exps = nullptr;
  ggml_tensor* down_exps = nullptr;
  ggml_tensor* up_shexp = nullptr;
  ggml_tensor* gate_shexp = nullptr;
  ggml_tensor* down_shexp = nullptr;
  // State.
  // F16 [head, raw_cells, 1]; in the fast plan a compressed layer's [head,
  // raw_cells + compressed cells, 1], its compressed cache (csa_k, hca_k) a
  // view of it (the state lays the two out together, model/dsv4.h).
  ggml_tensor* raw_k = nullptr;
  ggml_tensor* csa_k = nullptr;         // F16 [head, csa_cells, 1]
  ggml_tensor* csa_state_kv = nullptr;  // F32 [2·head, 8]
  ggml_tensor* csa_state_score = nullptr;
  ggml_tensor* lid_k = nullptr;  // F16 [indexer head, csa_cells, 1]
  ggml_tensor* lid_state_kv = nullptr;
  ggml_tensor* lid_state_score = nullptr;
  ggml_tensor* hca_k = nullptr;         // F16 [head, hca_cells, 1]
  ggml_tensor* hca_state_kv = nullptr;  // F32 [head, 128]
  ggml_tensor* hca_state_score = nullptr;
};

// The DSpark drafter's part of a target chunk (model/dspark.h): the chunk's
// features fused and projected into the drafter's KV ring, as llama.cpp's
// draft-dspark decodes every target batch's features into its context
// (common/speculative.cpp process, dflash.cpp graph_dsv4's embd batch).
struct Dsv4Injection {
  const model::DsparkProfile* profile = nullptr;
  const model::DsparkBinding* binding = nullptr;
  std::int64_t rows = 0;  // the chunk's last rows injected (model/dspark.h DsparkInject)
  std::int64_t ring = 0;  // the drafter ring's cells
};

// The drafter's weights and ring a chunk's injection binds.
struct DsparkInjectTensors {
  ggml_tensor* cells = nullptr;  // I64 [injected rows]: each row's ring cell (an input)
  ggml_tensor* fc = nullptr;
  ggml_tensor* enc_norm = nullptr;
  std::vector<ggml_tensor*> kv;       // per drafter block: wkv
  std::vector<ggml_tensor*> kv_norm;  // per drafter block
  std::vector<ggml_tensor*> ring;     // per drafter block: F16 [head, ring, 1]
};

struct Dsv4GraphOptions {
  // Each layer's routed-expert stride in bytes (nb[2] of the three expert
  // weights); 0 for the packed stride of one [k, n] slice.
  // NOLINTNEXTLINE(readability-redundant-member-init): designated initializers may omit it
  std::vector<std::uint64_t> expert_stride = {};
  // A speculative verify (D-092): flash attention runs each query row
  // alone, as its one-row chunk runs it, and the rows are concatenated;
  // the plan's products are the row-invariant ones (graph_plan.h
  // DeviceChoices::row_invariant). Every other operation's rows are
  // independent of the chunk's already.
  bool row_invariant = false;
  // The layers whose inputs a drafter reads (DSpark's target layers; the
  // layer count names the stream leaving the last layer): each's residual
  // streams' mean, concatenated per row into Dsv4Graph::features.
  // NOLINTNEXTLINE(readability-redundant-member-init): designated initializers may omit it
  std::vector<std::uint32_t> features = {};
  // With `features`, the drafter's KV injection in the same graph.
  std::optional<Dsv4Injection> inject = std::nullopt;
  // The fast plan (llmp_ops.h, "DeepSeek V4's fast plan"; the owner's
  // policy, 2026-09-28): for chunks of at most kVecQTokens rows, each
  // hyper-connection pre-mix and the norm after it as llmp.dsv4.hc_mix
  // and hc_pre, and each MoE block as the routing, one activation
  // quantization, the routed and shared experts' products with their SwiGLU
  // in the kernel (each distinct expert read once for the chunk) and the
  // combination. And for every chunk, sparse attention at depth
  // (llmp_ops.h, "DeepSeek V4's sparse attention"): each layer's window
  // cells and, in a compressed layer, its compressed rows read in place as
  // one K (no concatenation), masked on the device (the indexer's selection
  // for CSA, the visible rows for HCA), attended through the MMA kernel's
  // gather of the unmasked cells; the indexer's scores and selection as
  // llmp.dsv4.lid_topk (deterministic, ties to the lower row). A token's
  // attention and indexer then cost the same at any depth but for the
  // indexer's scoring and HCA's one row per 128 positions, over a ring
  // window (model/dsv4.h Dsv4Window::kRing) or the full one. Not llama.cpp's
  // arithmetic: its logits differ in the last bits and beyond. Off: the
  // graph node for node as llama.cpp builds it. Wide contiguous512-value
  // Q heads with64 normal rotary tail values use one RMSNorm/RoPE kernel,
  // retaining native F32 rounding without a materialized norm tensor.
  bool fused = false;
  // Private opt-in for the qualified GB10 4096-row staged output-A shape.
  // Unsupported shapes and the exact graph retain native inverse RoPE/MMQ.
  bool outa_prefill = false;
  // The ds4 prefill stage mechanisms (docs/experiments/ds4-prefill-stages),
  // each byte-identical to the graph without it; with `fused`, the fast
  // plan's defaults (SetDsv4PrefillStages). Each HC mix input of at least
  // 64 rows with F16 mix weights is normalized straight to F16 rows that
  // the mix product reads (with F32 mix weights, to F32 rows: rms_norm's
  // own), fused with the preceding HC post where one exists.
  bool hc_f16_rows = false;
  // The attention input's F16-weight products (compressor, indexer
  // compressor and projection) of at least 64 rows read one shared F16 copy
  // of it instead of each converting it.
  bool shared_f16_inputs = false;
  // The Q-head writes F16 Q rows (the RN values attention rounds F32 Q to)
  // for chunks of at least 64 rows.
  bool f16_q = false;
  // Nonzero: graph-owned exact-row raw causal/ring masks, bounded by this
  // actual state context. Compressed masks and DSpark's non-causal masks
  // keep their separate contracts.
  std::uint32_t raw_mask_context = 0;
  // DSpark-only exact-row noncausal block mask, independent of target context.
  bool device_draft_masks = false;
  // A nonfinal prompt needs state and requested features, without a head.
  // If no feature reads final streams, stop after the last layer's stores.
  bool state_only = false;
};

// The fast plan's graph-side prefill stage mechanisms (above), all on or off.
inline void SetDsv4PrefillStages(Dsv4GraphOptions& options, bool on) {
  options.hc_f16_rows = on;
  options.shared_f16_inputs = on;
  options.f16_q = on;
}

struct Dsv4Graph {
  ggml_tensor* embd = nullptr;       // F32 [width, rows]: the embedding rows
  ggml_tensor* tokens = nullptr;     // I32 [rows]: the hash layers' routing
  ggml_tensor* positions = nullptr;  // I32 [rows]
  // An engine-planned HCA scalar, authenticated against the actual host
  // positions before dispatch, including when an operation falls back.
  std::optional<std::uint32_t> prefill_first_position = std::nullopt;
  ggml_tensor* raw_k_idxs = nullptr;  // I64 [rows]: each token's ring cell
  ggml_tensor* raw_mask = nullptr;    // F16 [raw_n_kv, rows, 1, 1]
  bool device_raw_mask = false;       // graph activation rather than host input
  ggml_tensor* out_ids = nullptr;     // I32 [outputs]: the head's requested trailing rows
  Dsv4CompInputs csa, hca, lid;
  ggml_tensor* lid_rot = nullptr;  // F32 [indexer head, indexer head]: never read
  ggml_tensor* top_k_zeros =
      nullptr;  // F16: the zero fill's source, never read (not the fast plan's)
  // The fast plan's: I32 [rows], the compressed rows each row sees (CSA's and
  // the indexer's; HCA's), from which it masks them on the device.
  ggml_tensor* csa_visible = nullptr;
  ggml_tensor* hca_visible = nullptr;
  // Actual output-A eligibility, including the potential packed attention
  // result of a pruned final layer. Pruning cannot broaden combined HCA.
  std::uint32_t headed_outa_layers = 0;
  std::vector<Dsv4LayerTensors> layers;
  ggml_tensor* output_norm = nullptr;
  ggml_tensor* output = nullptr;
  ggml_tensor* hc_head_fn = nullptr;
  ggml_tensor* hc_head_base = nullptr;
  ggml_tensor* hc_head_scale = nullptr;
  ggml_tensor* logits = nullptr;  // F32 [vocab, outputs]; default outputs = chunk rows
  ggml_tensor* token = nullptr;   // I32 [outputs], only explicit plain token shapes
  // With options.features: F32 [width · features, rows], each row the
  // listed layers' stream means in order (DSpark's fc input).
  ggml_tensor* features = nullptr;
  // With options.inject.
  std::optional<DsparkInjectTensors> inject;
  bool state_only_tail_cut = false;  // final trunk omitted; required exports retain it
  std::vector<ggml_tensor*> nodes;   // GGML's order, views included
  // Intermediate tensors under llama.cpp's callback names ("l_last-7",
  // "attn_out-7", "ffn_moe_out-7", "hc_head-1", ...) and a few of llmpalooza's
  // ("kq_mask-7", "lid_topk-7", the fast plan's "ffn_moe_route-7"), for
  // comparisons.
  std::vector<std::pair<std::string, ggml_tensor*>> named;

  // The host-built inputs, in the order they are copied.
  std::vector<ggml_tensor*> inputs() const;
  ggml_tensor* Named(std::string_view name) const;
};

// How many tensors a chunk's graph creates at most, for TensorArena.
std::size_t Dsv4GraphTensors(const model::Dsv4Profile& profile);

// ---------------------------------------------------------------- waves

// A wave (engine/dsv4_runner.h): up to kDsv4WaveSlots independent
// sequences' decode or verify chunks in one graph of the fast plan. Every
// row-local operation (the products, norms and rotations, the
// hyper-connections, the routing, the routed and shared experts, the head,
// the drafter's features) runs once over every slot's rows together, so a
// weight is read once for the wave and a routed expert once for every row
// of any slot that selected it. Each slot's own state stays its own: its
// window cells, compressors, compressed and indexer rows, its indexer's
// selection and its attention run over its own inputs and state at its own
// rows, and the attention outputs are concatenated again; so does its
// DSpark injection (GGML's quantized products of the drafter's projections
// follow the column count). At most kDsv4WaveRows rows in all: the fast
// plan's vector products and GGML's float vector kernel give each column
// the same arithmetic at any count from two to eight, so a slot's verify
// rows in a wave equal its verify alone bit for bit; a wave of one-row
// steps gives its vector products the one-token launch (llmp_ops.h
// SetVecQOneToken), so each slot's row equals its step alone bit for bit.
// A wave mixing one-row and wider slots keeps the wider launch. Every
// slot's rows count toward kDsv4WaveRows, so sixteen one-row steps (the
// engine's kMaxRequestSlots) fill a wave.
inline constexpr std::size_t kDsv4WaveSlots = 16;
inline constexpr std::int64_t kDsv4WaveRows = 16;

// Each slot's chunk shape, in wave order (their layout fields equal), and
// the rows of each slot's DSpark injection (its last rows; 0 without one).
struct Dsv4WaveShape {
  std::vector<Dsv4ChunkShape> slots;
  std::vector<std::int64_t> inject_rows;
  bool token = false;  // homogeneous non-speculative one-row owners
  bool operator==(const Dsv4WaveShape&) const = default;
};

struct Dsv4WaveGraph {
  // The weights (and the drafter's injection weights), the joined inputs
  // (embd, tokens, positions, each compressor's state_pos: every slot's
  // rows in wave order), lid_rot, the logits of every row, features, the
  // nodes and the joined tensors' names.
  Dsv4Graph joined;
  // Each slot's own inputs (raw_k_idxs, raw_mask, the compressors' other
  // lists, the visible counts, the injection's cells) and its state tensors
  // (layers, the injection's ring); the joined fields are null.
  std::vector<Dsv4Graph> slots;
  std::vector<std::int64_t> first;  // each slot's first row in the wave
  // Each layer's slots' attention and state operations, one concurrent lane
  // a slot and a region a layer (graph_plan.h AssignLanes); their outputs'
  // joining is the stream's, in the region.
  LaneTags lanes;

  // The host-built inputs, in the order they are copied: the joined ones,
  // then each slot's.
  std::vector<ggml_tensor*> inputs() const;
};

std::size_t Dsv4WaveGraphTensors(const model::Dsv4Profile& profile, std::size_t slots);

// Whether the binding's weights let every layer take the fast plan's fused
// form, which a wave needs: the profile's widths, and each layer's routed
// and shared expert products llmp.vecq types with gate and up alike. The
// mixing weights' type is not a condition (F32, F16 and BF16 take
// llmp.dsv4.hc_mix; another type GGML's product). Refused with the first
// layer that does not, and its types: a runner then serves one request at a
// time rather than refusing the model.
std::expected<void, KernelFailure> Dsv4WaveSupport(const model::Dsv4Profile& profile,
                                                   const model::Dsv4Binding& binding);

// Builds a wave's graph (options.fused, the fast plan only: no reference,
// row-invariant or prefill stage form). Refused if a slot's shape is not
// one the state holds, the slots' layouts differ, the rows exceed
// kDsv4WaveRows, an injection's rows do not fit its slot, or a layer would
// not take the fast plan's fused form (Dsv4WaveSupport).
std::expected<Dsv4WaveGraph, KernelFailure> BuildDsv4WaveGraph(TensorArena& arena,
                                                               const model::Dsv4Profile& profile,
                                                               const model::Dsv4Binding& binding,
                                                               const Dsv4WaveShape& shape,
                                                               const Dsv4GraphOptions& options);

// Builds the chunk's graph on `arena` with the binding's weight types and
// shapes. Refused if the shape is not one the state holds, a weight type is
// not a GGML type, an expert stride is not a whole number of the expert
// projections' blocks, or the arena lacks room.
std::expected<Dsv4Graph, KernelFailure> BuildDsv4Graph(TensorArena& arena,
                                                       const model::Dsv4Profile& profile,
                                                       const model::Dsv4Binding& binding,
                                                       const Dsv4ChunkShape& shape,
                                                       const Dsv4GraphOptions& options);

// ---------------------------------------------------------------- DSpark

// The DSpark drafter's draft block (model/dspark.h), as dflash.cpp's
// graph_dsv4 builds its token batch: the block's rows embedded from the
// target's table (an input, as for the target), through the drafter's
// DeepSeek V4 blocks over its KV ring (each block's K written at its rows'
// cells, then attention over the whole ring under the block's non-causal
// window mask), its hyper-connection head and final norm, the target's
// head, and the Markov head: slot i's logits plus markov_w2 · markov_w1[prev],
// prev the anchor for slot 0 and slot i - 1's argmax after it. Each slot's
// draft is its biased logits' argmax (llama.cpp's draft sampler, top-k 10
// then the most probable, is that argmax).
struct DsparkGraph {
  Dsv4Graph core;  // embd, positions, ring cells; raw_mask is a leaf or block producer
  ggml_tensor* tokens = nullptr;  // I32 [rows]: the block's tokens (the anchor first)
  ggml_tensor* markov_w1 = nullptr;
  ggml_tensor* markov_w2 = nullptr;
  ggml_tensor* logits = nullptr;  // F32 [vocab, rows]: the Markov-biased slots
  ggml_tensor* drafts = nullptr;  // I32 [rows]: the last node

  // The host-built inputs, in the order they are copied.
  std::vector<ggml_tensor*> inputs() const;
};

// How many tensors a draft block's graph creates at most.
std::size_t DsparkGraphTensors(const model::DsparkProfile& profile, std::int64_t rows);

// Builds a draft block of `rows` rows over a ring of `ring` cells. Refused
// if the rows or ring are not the profile's, a weight type is not a GGML
// type, or the arena lacks room.
std::expected<DsparkGraph, KernelFailure> BuildDsparkGraph(TensorArena& arena,
                                                           const model::DsparkProfile& profile,
                                                           const model::DsparkBinding& binding,
                                                           std::int64_t rows, std::int64_t ring,
                                                           const Dsv4GraphOptions& options);

// Several requests' draft blocks as one graph (a DSpark wave's drafts): the
// blocks' rows joined, in wave order, through every row-local operation
// (the products read each weight once for every slot's rows), and each
// slot's attention over its own ring and its Markov head over its own rows,
// as BuildDsv4WaveGraph joins a wave's verifies. Each slot's drafts equal
// its own draft block's (BuildDsparkGraph) bit for bit: llmp.vecq gives
// each row the same sums at any count of two or more, a float product past
// GGML's column-invariant count runs over groups of whole slots that fit
// its eight columns, and the rest is per row or per slot.
struct DsparkWaveGraph {
  // The joined rows: embd and positions every slot's; the weights, the
  // head and its logits; no state.
  Dsv4Graph joined;
  // Each slot's cells (raw_k_idxs), window mask (raw_mask) and ring (its
  // layers' raw_k).
  std::vector<Dsv4Graph> slots;
  ggml_tensor* tokens = nullptr;  // I32 [rows]: every slot's block (each anchor first)
  ggml_tensor* markov_w1 = nullptr;
  ggml_tensor* markov_w2 = nullptr;
  std::vector<ggml_tensor*> drafts;  // each slot's I32 [its rows]
  std::vector<std::int64_t> first;   // each slot's first row in the joined rows
  // Each slot's attention and Markov head on a concurrent lane (graph_plan.h
  // AssignLanes), as Dsv4WaveGraph::lanes.
  LaneTags lanes;

  // The host-built inputs, in the order they are copied: the joined ones,
  // then each slot's.
  std::vector<ggml_tensor*> inputs() const;
};

// How many tensors a joined draft of `slots` blocks creates at most.
std::size_t DsparkWaveGraphTensors(const model::DsparkProfile& profile, std::int64_t rows,
                                   std::size_t slots);

// Builds `slots` draft blocks of `rows` rows each over rings of `ring`
// cells. Refused as BuildDsparkGraph refuses, for fewer than two slots or
// more than kDsv4WaveSlots, past kDsv4WaveRows rows in all, or if a block
// layer is not in the fast plan's fused form (whose products are the
// column-invariant ones).
std::expected<DsparkWaveGraph, KernelFailure> BuildDsparkWaveGraph(
    TensorArena& arena, const model::DsparkProfile& profile, const model::DsparkBinding& binding,
    std::int64_t rows, std::size_t slots, std::int64_t ring, const Dsv4GraphOptions& options);

// The GGML type of a type name, if GGML has one.
std::expected<ggml_type, KernelFailure> GgmlTypeOf(std::string_view name);

// The Hadamard matrix llama.cpp gives the indexer (ggml_gen_hadamard), n x n
// F32, row-major.
std::vector<float> HadamardMatrix(std::int64_t n);

}  // namespace llmp::kernels::ggml

#endif  // LLMP_KERNELS_GGML_DSV4_GRAPH_H_
