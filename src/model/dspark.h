// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// DeepSeek V4 Flash's DSpark drafter (M3, "Speculative decoding in the
// core"; D-068), as llama.cpp b29c606e2 runs it (src/models/dflash.cpp's
// graph_dsv4 and common/speculative.cpp's draft-dspark, MIT): three
// DeepSeek V4 blocks over a window-only KV ring of their own, a
// hyper-connection head, the target's head, and a Markov head that biases
// each block position's logits by the position before it.
//
// - Its artifact (architecture "dflash") is its own v0 artifact; it ships
//   no token table or head and binds its target's (the target's resources:
//   one set of extents, leased and charged once).
// - Its state is one KV ring per block: `ring` cells of head_dim F16, cell
//   p mod ring for position p, holding for every committed position the K
//   of the target's features there (the hc-mean of the residual streams
//   entering target layers 41 and 42 and leaving the last, fused by `fc`
//   and projected by each block's wkv: the injection, which runs in the
//   target's own chunk), and while a draft runs, the draft block's own K.
// - A draft is one block of `rows` positions from pos0, the position of
//   the last sampled token (the anchor, not yet in the target's state):
//   [anchor, mask, mask, ...], attending non-causally (dflash.attention.
//   causal is absent: llama.cpp sets causal attention off) to each other
//   and to the ring's positions within the window. With sample_from_anchor
//   (the default, the key absent), slot i's logits propose the token at
//   pos0 + i + 1: `rows` drafts from a block of `rows`.
// - Rollback: the ring truncates to any position through a snapshot of the
//   cells a verify writes (DsparkWrites), as the target's state does
//   (model/dsv4.h Dsv4ChunkWrites).
//
// llama.cpp's confidence head (conf_proj) only truncates a draft below
// --spec-draft-p-min, 0 by default: bound, never computed.

#ifndef JITLLM_MODEL_DSPARK_H_
#define JITLLM_MODEL_DSPARK_H_

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "model/dsv4.h"
#include "model/state.h"

namespace jitllm::artifact {
class Artifact;
}

namespace jitllm::model {

struct DsparkProfile {
  std::string_view name;
  // The drafter's DeepSeek V4 stages: layers window-only (ratio 0), every
  // layer routed (no hash layers), the rest as the GGUF's dflash.* keys.
  Dsv4Profile blocks;
  std::uint32_t block_size = 0;              // dflash.block_size: the most drafts
  std::vector<std::uint32_t> target_layers;  // dflash.target_layers
  std::int32_t mask_token = 0;               // tokenizer.ggml.mask_token_id
  std::uint32_t markov_rank = 0;             // markov_w1's rows
  std::uint32_t ring = 0;                    // the KV ring's cells
};

// DeepSeek-V4-Flash-0731's DSpark drafter (dspark-DeepSeek-V4-Flash-0731-
// Q8_0.gguf), its keys as the GGUF has them
// (docs/experiments/dspark/README.md).
const DsparkProfile& DsparkDeepSeekV4Flash();

struct DsparkBinding {
  // Its blocks, hyper-connection head and final norm, the drafter
  // artifact's; token_embd and output name the TARGET artifact's
  // resources.
  Dsv4Binding blocks;
  Dsv4Tensor fc;         // [width · target layers, width]
  Dsv4Tensor enc_norm;   // enc.output_norm, [width]
  Dsv4Tensor markov_w1;  // [rank, vocab]
  Dsv4Tensor markov_w2;  // [rank, vocab]
  Dsv4Tensor conf_proj;  // [width + rank, 1]: bound, not read
};

// Binds the drafter artifact's resources (architecture "dflash") and its
// target's tables. Refused, naming the reason, if a role is missing, of
// another shape or kind, or not one the drafter reads; if the drafter's
// profile is not a DeepSeek V4 stage's (layers not window-only, hash
// layers); or if the target does not match it: width, vocabulary, a target
// layer past the target's last (its count names the stream leaving the
// last layer), no target layers, or a drafter wider than its block.
std::expected<DsparkBinding, std::string> BindDspark(const DsparkProfile& profile,
                                                     const artifact::Artifact& drafter,
                                                     const Dsv4Profile& target_profile,
                                                     const Dsv4Binding& target);
// The same over the drafter's resources as the adapter sees them.
std::expected<DsparkBinding, std::string> BindDspark(const DsparkProfile& profile,
                                                     std::string_view architecture,
                                                     std::span<const Dsv4Resource> drafter,
                                                     const Dsv4Profile& target_profile,
                                                     const Dsv4Binding& target);

// ---------------------------------------------------------------- state

struct DsparkStateLayout {
  std::uint32_t ring = 0;
  std::uint32_t max_rows = 0;          // a draft block, or a chunk's injected rows
  std::vector<std::uint64_t> offsets;  // each block's ring, F16 [head_dim, ring]
  std::uint64_t bytes = 0;

  // The ring as a D-068 representation: one fixed-size block; with a
  // verify of up to `max_verify` rows, truncated to any position through
  // the verify's snapshot (DsparkWrites).
  StateRepresentation Representation(std::uint32_t max_verify) const;
};

// Refused if the ring cannot hold the window and a block beside it (every
// position a block row sees is in a cell of its own), or `max_rows` is 0.
std::expected<DsparkStateLayout, std::string> DsparkState(const DsparkProfile& profile,
                                                          std::uint32_t max_rows);

// ---------------------------------------------------------------- inputs

// A draft block's host-built inputs.
struct DsparkBlockInputs {
  std::uint32_t pos0 = 0;
  std::uint32_t rows = 0;
  std::vector<std::int32_t> tokens;     // [anchor, mask, ...]
  std::vector<std::int32_t> positions;  // pos0 ...
  std::vector<std::int64_t> cells;      // each row's ring cell
  // F16 bit patterns [ring, rows]: 0 where a row attends, -inf elsewhere:
  // cells holding a position in [pos0 + i - window + 1, pos0 + rows) for row
  // i (the block's own, and the committed positions in its window).
  std::vector<std::uint16_t> mask;
};

// Refused if `rows` is 0 or past the block size or the layout's bound, or
// the anchor is outside the vocabulary, the block ends past INT32_MAX, or
// the ring cannot retain the window and block. With materialize_mask=false,
// the same tokens, positions and cells are checked and built; only the
// optional host mask matrix is omitted.
std::expected<DsparkBlockInputs, std::string> DsparkBlock(const DsparkProfile& profile,
                                                          const DsparkStateLayout& state,
                                                          std::uint32_t pos0, std::int32_t anchor,
                                                          std::uint32_t rows,
                                                          bool materialize_mask = true);

// The ring cells a target chunk of `rows` rows from `n_past` injects:
// its last min(rows, ring) rows, whose positions have cells of their own.
struct DsparkInjection {
  std::uint32_t first = 0;  // the chunk row of the first injected position
  std::vector<std::int64_t> cells;
};
DsparkInjection DsparkInject(const DsparkStateLayout& state, std::uint32_t n_past,
                             std::uint32_t rows);

// The ring bytes a verify chunk's injection writes, by chunk row (rows
// before the injected ones write none).
std::vector<std::vector<StateRange>> DsparkWrites(const DsparkProfile& profile,
                                                  const DsparkStateLayout& state,
                                                  std::uint32_t n_past, std::uint32_t rows);

}  // namespace jitllm::model

#endif  // JITLLM_MODEL_DSPARK_H_
