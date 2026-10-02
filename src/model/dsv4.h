// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The DeepSeek V4 Flash architecture adapter (M3; docs/plan.md, "Model
// graphs and state"): the model's hyperparameters, the binding of a
// prepared artifact's resources and expert arrays to the tensors the
// architecture reads, its state (the sliding-window cache, the compressed
// caches, the indexer's cache and the compressors' ring state) as explicit,
// bounded state representations (D-068), and each chunk's host-built
// inputs for one sequence, as llama.cpp b29c606e2 builds them
// (src/llama-kv-cache-dsv4.cpp's dsv4_build_comp_plan and
// llm_graph_input_dsv4) with no speculative rollback planes (n_rs_seq 0)
// and one stream.
//
// As with Qwen2 (model/qwen2.h), the hyperparameters are compiled in: a
// profile per supported checkpoint, cross-checked against the GGUF's
// key/values by a reference-side script
// (docs/experiments/dsv4-native/gguf_profile_check.py) and validated
// against the artifact's resource shapes when it is bound. Weight types are
// the artifact's: the binding records each tensor's GGML type, and the
// kernel module refuses at planning a type it does not compile.
//
// The model layer holds no vendor or kernel-module types; the GGML graph
// that computes a chunk is built from this adapter by the GGML kernel
// module (kernels/ggml/dsv4_graph.h).

#ifndef JITLLM_MODEL_DSV4_H_
#define JITLLM_MODEL_DSV4_H_

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

// A DeepSeek V4 checkpoint's hyperparameters (the deepseek4.* key/values).
struct Dsv4Profile {
  std::string_view name;
  std::uint32_t layers = 0;
  std::uint32_t width = 0;      // n_embd
  std::uint32_t heads = 0;      // query heads; one KV head (K = V)
  std::uint32_t head_dim = 0;   // key and value length
  std::uint32_t rope_dims = 0;  // the rotated tail of each head
  std::uint32_t q_lora = 0;     // q_lora_rank
  std::uint32_t o_lora = 0;     // output_lora_rank
  std::uint32_t o_groups = 0;   // output_group_count
  std::uint32_t window = 0;     // sliding_window
  std::uint32_t experts = 0;
  std::uint32_t experts_used = 0;
  std::uint32_t expert_ffn = 0;  // per routed and shared expert
  std::uint32_t shared_experts = 0;
  std::uint32_t hash_layers = 0;  // the first layers route by token id
  std::uint32_t hc = 0;           // hyper-connection streams
  std::uint32_t sinkhorn_iterations = 0;
  std::uint32_t indexer_heads = 0;
  std::uint32_t indexer_head_dim = 0;
  std::uint32_t indexer_top_k = 0;
  std::uint32_t vocab = 0;
  std::uint32_t yarn_original_context = 0;
  float rms_eps = 0.0f;
  float hc_eps = 0.0f;
  float rope_base = 0.0f;           // uncompressed layers
  float compress_rope_base = 0.0f;  // compressed layers, the indexer and compressed rows
  float rope_scale = 0.0f;          // YaRN factor (freq_scale = 1 / factor)
  float yarn_beta_fast = 0.0f;
  float yarn_beta_slow = 0.0f;
  float expert_weights_scale = 0.0f;
  bool expert_weights_norm = false;
  float swiglu_limit = 0.0f;         // routed experts (every layer)
  float swiglu_limit_shared = 0.0f;  // shared expert (every layer)
  // Per layer: 0 (sliding window only), 4 (CSA, with the indexer) or 128
  // (HCA).
  std::vector<std::uint32_t> compress_ratios;

  std::uint32_t hc_width() const { return hc * width; }
  std::uint32_t hc_mix() const { return (2 + hc) * hc; }
};

inline constexpr std::uint32_t kDsv4CsaRatio = 4;
inline constexpr std::uint32_t kDsv4HcaRatio = 128;
// The supported Flash checkpoint's trained position ceiling.
inline constexpr std::uint32_t kDsv4FlashContext = 1048576;

// DeepSeek V4 Flash (both the e3aa0d6a and 0731 GGUF revisions share it).
const Dsv4Profile& Dsv4Flash();

// A tensor as the adapter binds it: its artifact resource (or, for routed
// experts, expert array), GGML type and `ne`.
struct Dsv4Tensor {
  std::uint32_t index = 0;
  std::string type;
  std::vector<std::uint64_t> ne;
};

struct Dsv4Layer {
  std::uint32_t ratio = 0;
  Dsv4Tensor attn_norm, attn_sinks, q_a, q_a_norm, q_b, kv, kv_norm, out_a, out_b;
  Dsv4Tensor hc_attn_fn, hc_attn_base, hc_attn_scale, hc_ffn_fn, hc_ffn_base, hc_ffn_scale;
  // Compressor (ratio 4 or 128).
  Dsv4Tensor comp_kv, comp_gate, comp_ape, comp_norm;
  // Indexer (ratio 4).
  Dsv4Tensor idx_q_b, idx_proj, idx_comp_kv, idx_comp_gate, idx_comp_ape, idx_comp_norm;
  // MoE: the router, its bias (routed layers) or token table (hash layers),
  // the routed experts (expert arrays, `ne` of one slice) and the shared one.
  Dsv4Tensor ffn_norm, router, router_bias, tid2eid;
  Dsv4Tensor up_exps, gate_exps, down_exps;
  Dsv4Tensor up_shexp, gate_shexp, down_shexp;
  bool hash = false;
};

struct Dsv4Binding {
  std::vector<Dsv4Layer> layers;
  Dsv4Tensor token_embd, output_norm, output, hc_head_fn, hc_head_base, hc_head_scale;
};

// A resource or expert array as the adapter sees it.
struct Dsv4Resource {
  std::vector<std::string> roles;  // for an expert array, its name
  std::string type;
  std::vector<std::uint64_t> ne;  // an expert array's: one slice's
  bool expert_array = false;
  std::uint32_t count = 0;  // an expert array's experts
};

// Binds every tensor the profile reads to the resource of that role (an
// expert array for the routed experts), which must have its shape exactly
// and a type of its kind (F32 for norms, scales and biases, F32 or F16 for
// APE tables, I32 for the hash tables, a GGML matrix type otherwise); refused, naming the tensor,
// if one is missing or differs, if `architecture` is not "deepseek4", or if a resource is bound to
// a role the architecture does not read.
std::expected<Dsv4Binding, std::string> BindDsv4(const Dsv4Profile& profile,
                                                 std::string_view architecture,
                                                 std::span<const Dsv4Resource> resources);
// The same over a validated v0 artifact (artifact/artifact.h). Resource and
// expert-array indices are the artifact's.
std::expected<Dsv4Binding, std::string> BindDsv4(const Dsv4Profile& profile,
                                                 const artifact::Artifact& artifact);

// A role a binder asks of an artifact beside its DeepSeek V4 blocks' (the
// DSpark drafter's, model/dspark.h).
struct Dsv4ExtraRole {
  std::string role;
  bool f32 = false;  // F32, else a matrix type
  std::vector<std::uint64_t> ne;
  Dsv4Tensor* into = nullptr;
};

// BindDsv4's rules for an artifact of `architecture` whose stages are
// DeepSeek V4 blocks: the profile's layers, its hyper-connection head and
// final norm; with `tables` the token table and the head too; and every
// `extra` role. Every role the artifact binds must be one of these.
std::expected<Dsv4Binding, std::string> BindDsv4Roles(const Dsv4Profile& profile,
                                                      std::string_view want_architecture,
                                                      std::string_view architecture, bool tables,
                                                      std::span<const Dsv4Resource> resources,
                                                      std::span<Dsv4ExtraRole> extra);
// The same over a validated v0 artifact, its expert arrays indexed among
// the artifact's.
std::expected<Dsv4Binding, std::string> BindDsv4Roles(const Dsv4Profile& profile,
                                                      std::string_view want_architecture,
                                                      const artifact::Artifact& artifact,
                                                      bool tables, std::span<Dsv4ExtraRole> extra);

// A hash-routed layer's token-to-expert table (ffn_gate_tid2eid, I32
// [experts_used, vocab]) as the artifact holds it. Its entries are untrusted
// data that the graph's get_rows and mul_mat_id kernels index device memory
// with, unchecked, so every one must name an expert in [0, experts) before
// any chunk runs. Refused, naming the first entry outside, or if the table
// is not [experts_used, vocab].
std::expected<void, std::string> CheckDsv4HashRouting(const Dsv4Profile& profile,
                                                      std::span<const std::int32_t> table);

// ---------------------------------------------------------------- state

// The state of one sequence for a context of `context` positions and chunks
// of at most `max_rows` rows: every layer's tensors in one region, each at a
// 256-byte aligned offset. With Dsv4Window::kFull (the reference mode's)
// the sliding-window cache holds a cell per position of the context,
// raw_cells = pad(context, 256), position p in cell p, as llama.cpp's
// default full-size SWA cache does (swa_full): attention reads the first
// pad(positions, 256) cells, the window masking the rest, so the attention
// length, and with it the kernels' summation order, is llama.cpp's. With
// Dsv4Window::kRing (the fast plan's, as llama-server runs with swa_full
// off) it is a ring of the window plus a chunk, raw_cells = pad(window +
// max_rows, 256) (at most the full size), position p in cell p mod
// raw_cells: a chunk's rows and the window before each of them are always
// in it, so its size, and a token's attention, no longer grow with the
// context. The compressed caches hold one row per
// completed block, pad(ceil(context / ratio), 256) rows; the compressors'
// ring state holds each layer's last 2·ratio (CSA, indexer) or ratio (HCA)
// positions' projections. Each compressed layer's compressed cache follows
// its window cache directly in the region (the graph's fast plan reads the
// two as one tensor: the window's cells, then the compressed rows).
// Everything starts zeroed: attention masks the cells not yet written, and
// llama.cpp zeroes the compressed caches.
enum class Dsv4Window : std::uint8_t { kFull, kRing };
struct Dsv4StateTensor {
  enum class Kind : std::uint8_t {
    kRawK,        // F16 [head_dim, raw_cells]
    kCsaK,        // F16 [head_dim, csa_cells]
    kCsaStateKv,  // F32 [2·head_dim, 2·4]
    kCsaStateScore,
    kLidK,        // F16 [indexer_head_dim, csa_cells]
    kLidStateKv,  // F32 [2·indexer_head_dim, 2·4]
    kLidStateScore,
    kHcaK,        // F16 [head_dim, hca_cells]
    kHcaStateKv,  // F32 [head_dim, 128]
    kHcaStateScore,
  };
  Kind kind = Kind::kRawK;
  std::uint32_t layer = 0;
  bool f16 = true;
  std::uint64_t ne0 = 0;
  std::uint64_t ne1 = 0;
  std::uint64_t offset = 0;  // in the state region
  std::uint64_t bytes = 0;
};

struct Dsv4StateLayout {
  std::uint32_t context = 0;
  std::uint32_t max_rows = 0;
  Dsv4Window window = Dsv4Window::kFull;
  std::uint32_t raw_cells = 0;
  std::uint32_t csa_cells = 0;  // also the indexer's
  std::uint32_t hca_cells = 0;
  std::uint32_t csa_state_rows = 0;  // 2 x 4
  std::uint32_t hca_state_rows = 0;  // 128
  std::vector<Dsv4StateTensor> tensors;
  std::uint64_t bytes = 0;  // the region

  // The index of layer `layer`'s tensor of `kind`, or -1.
  std::int64_t Find(std::uint32_t layer, Dsv4StateTensor::Kind kind) const;
  // The state as D-068 representations: the window cache, the compressor
  // rings and the compressed and indexer caches, each one fixed-size block
  // sized for the context. Append only without speculation; with a
  // speculative verify of up to `max_verify` rows (a DSpark drafter), each
  // also truncates to any position at or above the committed prefix: a
  // verify first saves every row it will write (Dsv4ChunkWrites), and a
  // rejection restores the rejected positions' rows and the chunk's scratch
  // rows from that snapshot, which is the request's working state
  // (Dsv4VerifySnapshotBytes), not a representation's.
  std::vector<StateRepresentation> Representations(std::uint32_t max_verify = 0) const;
};

// Refused if the context or chunk bound is zero, the context is past the
// graph's I32 positions (2^31 - 256), a chunk leaves the window
// no room in the ring (max_rows > raw_cells - window), or the profile is not
// DeepSeek V4's.
std::expected<Dsv4StateLayout, std::string> Dsv4State(const Dsv4Profile& profile,
                                                      std::uint32_t context, std::uint32_t max_rows,
                                                      Dsv4Window window = Dsv4Window::kFull);

// The cache prefixes and fixed ring/scratch ranges a chunk ending at
// `positions` may touch. The dummy compressed row is always included.
std::expected<std::vector<StateRange>, std::string> Dsv4UsedState(const Dsv4StateLayout& state,
                                                                  std::uint32_t positions);
// Ranges later tokens may overwrite. Checkpoints save their complete used
// physical pages, including padded bytes beyond the current read prefix.
std::expected<std::vector<StateRange>, std::string> Dsv4CheckpointWrites(
    const Dsv4StateLayout& state, std::uint32_t positions);

// The widest chunk Dsv4State admits at `context` (either window: a ring is
// never larger than the full cache, and holds any chunk the full cache
// does): the context, and the full cache's cells less the window (0 when
// none, or when the context is refused whatever the chunk); and, for the
// fast plan's ring, the widest whose attention mask (F16 [ring cells +
// compressed cells, rows]) stays under 2^31 bytes (RE-037): 13,530 rows
// at 262,144, 4,100 at 1,030,144, 4,029 at 1,048,576.
std::uint32_t Dsv4MostRows(const Dsv4Profile& profile, std::uint32_t context);

// ---------------------------------------------------------------- chunk inputs

// One compressor's recipe for a chunk (llama_kv_cache_dsv4_context::comp_plan
// for one sequence, one stream, no rollback): what the graph reads and writes
// of the compressor's ring state and its compressed cache, and the mask of
// the compressed rows each token sees.
struct Dsv4CompPlan {
  std::uint32_t ratio = 0;
  bool overlap = false;                 // CSA and the indexer: blocks read the previous window too
  std::vector<std::int32_t> state_pos;  // rows: pos mod ratio (the APE row)
  std::vector<std::int32_t> persist_src;  // chunk rows kept in the ring state
  std::vector<std::int32_t> persist_dst;  // their ring rows, ascending
  // Rows of [ring state | chunk rows | a zero (-inf) row] each block reads:
  // overlap, every block's previous window, then every block's own.
  std::vector<std::int32_t> read_idxs;
  std::vector<std::int64_t> write_idxs;  // blocks: the compressed cache row
  std::vector<std::int32_t> write_pos;   // blocks: the RoPE position (block start)
  std::vector<std::int32_t> n_visible;   // rows: completed rows each token sees
  std::uint32_t n_kv = 0;                // mask width, pad(max visible, 256)

  std::uint32_t blocks() const { return static_cast<std::uint32_t>(write_idxs.size()); }
};

// A chunk's host-built inputs: `rows` tokens at positions n_past onwards of
// a sequence whose earlier positions are all in the state.
struct Dsv4ChunkInputs {
  std::uint32_t n_past = 0;
  std::uint32_t rows = 0;
  std::vector<std::int32_t> positions;  // rows
  std::uint32_t raw_n_kv = 0;           // cells attention reads
  std::vector<std::int64_t> raw_cells;  // rows: each token's ring cell
  // F16 bit patterns, 0 where a token attends, -inf elsewhere.
  std::vector<std::uint16_t> raw_mask;  // rows x raw_n_kv
  Dsv4CompPlan csa, hca, lid;
  // Only with `masks` (the reference mode's graph reads them): the fast
  // plan's graph masks the compressed rows on the device from each plan's
  // n_visible, so none of its inputs grows with the context but a full
  // window's mask.
  std::vector<std::uint16_t> csa_mask;  // rows x csa.n_kv
  std::vector<std::uint16_t> hca_mask;  // rows x hca.n_kv
  std::vector<std::uint16_t> lid_mask;  // rows x lid.n_kv
};

// Refused if the chunk is empty, runs past the layout's context, or is
// longer than its chunk bound. A ring's chunk reads all of the ring's cells
// (raw_n_kv = raw_cells), so every chunk of a width has one shape until
// its compressed rows cross a multiple of 256. Without `masks`, the
// compressed caches' masks are left empty.
std::expected<Dsv4ChunkInputs, std::string> Dsv4Chunk(const Dsv4Profile& profile,
                                                      const Dsv4StateLayout& state,
                                                      std::uint32_t n_past, std::uint32_t rows,
                                                      bool masks = true);

// One compressor's plan (exposed for tests).
std::expected<Dsv4CompPlan, std::string> Dsv4CompressorPlan(std::uint32_t ratio, bool overlap,
                                                            std::uint32_t state_rows,
                                                            std::uint32_t cache_rows,
                                                            std::uint32_t n_past,
                                                            std::uint32_t rows);

// ---------------------------------------------------------------- speculation

// Every state byte a chunk writes, by what writes it (D-068 truncation for
// a speculative verify): row i's own writes (its position's window cell in
// every layer, the compressor ring rows its position persists, and the
// compressed and indexer rows of the blocks its position completes), and
// the chunk's scratch writes, whatever is accepted: the dummy blocks' rows
// (each compressed cache's last row, always masked). A verify saves all of
// them before it runs; after accepting rows [0, m], it restores rows m + 1
// onwards and the scratch rows, which leaves exactly what a chunk of the
// accepted rows alone would have left. Ranges are 256-byte aligned
// multiples of 256 bytes, disjoint within a chunk of at most 8 rows (the
// rings hold 8 and 128 positions).
struct Dsv4Writes {
  std::vector<std::vector<StateRange>> rows;
  std::vector<StateRange> scratch;
};
Dsv4Writes Dsv4ChunkWrites(const Dsv4Profile& profile, const Dsv4StateLayout& state,
                           const Dsv4ChunkInputs& chunk);
// The bytes a verify of up to `max_rows` rows saves at most: the request's
// rollback snapshot (D-068 working state).
std::uint64_t Dsv4VerifySnapshotBytes(const Dsv4Profile& profile, const Dsv4StateLayout& state,
                                      std::uint32_t max_rows);

// Whether a verify of `rows` rows after `n_past` runs at every mask width a
// one-row step of each of its positions would (the window cells attention
// reads and the compressed rows each layer's mask spans are padded to 256):
// only then can its rows equal those steps bit for bit (D-092). A draft
// that would cross a width is shortened to end before it.
bool Dsv4SameWidths(const Dsv4StateLayout& state, std::uint32_t n_past, std::uint32_t rows);

inline constexpr std::uint16_t kHalfZero = 0x0000;
inline constexpr std::uint16_t kHalfNegInf = 0xFC00;

}  // namespace jitllm::model

#endif  // JITLLM_MODEL_DSV4_H_
