// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "model/dspark.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "model/dsv4.h"
#include "model/state.h"

namespace llmp::model {
namespace {

std::unexpected<std::string> Refused(std::string detail) {
  return std::unexpected(std::move(detail));
}

std::uint64_t Pad(std::uint64_t n, std::uint64_t to) { return (n + to - 1) / to * to; }

}  // namespace

const DsparkProfile& DsparkDeepSeekV4Flash() {
  // dflash.* key/values of dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf (pinned
  // in docs/experiments/fast-swap/pins.json): the stages' hyperparameters
  // equal the target's (deepseek4.*) but for three blocks, all
  // window-only (compress_ratios [0, 0, 0]) and routed (hash_layer_count 0).
  static const DsparkProfile kProfile = [] {
    DsparkProfile d;
    d.name = "deepseek-v4-flash-dspark";
    d.blocks = Dsv4Flash();
    d.blocks.name = "deepseek-v4-flash-dspark-blocks";
    d.blocks.layers = 3;
    d.blocks.hash_layers = 0;
    d.blocks.compress_ratios.assign(3, 0);
    d.block_size = 5;
    d.target_layers = {41, 42, 43};
    d.mask_token = 128799;
    d.markov_rank = 256;
    // The window and a block beside it, padded to attention's 256-cell
    // steps.
    d.ring = 256;
    return d;
  }();
  return kProfile;
}

namespace {

// What both binders check, and the roles beside the blocks' they bind.
std::expected<std::vector<Dsv4ExtraRole>, std::string> DsparkRoles(
    const DsparkProfile& profile, const Dsv4Profile& target_profile, const Dsv4Binding& target,
    DsparkBinding& b) {
  const Dsv4Profile& p = profile.blocks;
  if (std::ranges::any_of(p.compress_ratios, [](std::uint32_t r) { return r != 0; }) ||
      p.hash_layers != 0) {
    return Refused("the drafter's stages are window-only and routed");
  }
  if (profile.target_layers.empty() || profile.block_size == 0 || profile.markov_rank == 0 ||
      profile.mask_token < 0 || std::cmp_greater_equal(profile.mask_token, p.vocab)) {
    return Refused("the drafter's profile has no target layers, block, Markov rank or mask token");
  }
  if (target_profile.width != p.width || target_profile.vocab != p.vocab ||
      std::ranges::any_of(profile.target_layers,
                          [&](std::uint32_t l) { return l > target_profile.layers; })) {
    return Refused("the drafter does not match its target's width, vocabulary or layers");
  }
  // The tables the drafter reads as [width, vocab]: the target's, bound to
  // its profile, which the drafter's matches.
  const std::vector<std::uint64_t> table{p.width, p.vocab};
  if (target.layers.size() != target_profile.layers || target.token_embd.ne != table ||
      target.output.ne != table) {
    return Refused("the target's binding is not its profile's");
  }
  const std::uint64_t width = p.width;
  const std::uint64_t features = width * profile.target_layers.size();
  return std::vector<Dsv4ExtraRole>{
      {.role = "fc.weight", .f32 = false, .ne = {features, width}, .into = &b.fc},
      {.role = "enc.output_norm.weight", .f32 = true, .ne = {width}, .into = &b.enc_norm},
      {.role = "markov_w1.weight",
       .f32 = false,
       .ne = {profile.markov_rank, p.vocab},
       .into = &b.markov_w1},
      {.role = "markov_w2.weight",
       .f32 = false,
       .ne = {profile.markov_rank, p.vocab},
       .into = &b.markov_w2},
      {.role = "conf_proj.weight",
       .f32 = false,
       .ne = {width + profile.markov_rank, 1},
       .into = &b.conf_proj},
  };
}

std::expected<DsparkBinding, std::string> Finish(std::expected<Dsv4Binding, std::string> blocks,
                                                 const Dsv4Binding& target, DsparkBinding& b) {
  if (!blocks) {
    return std::unexpected(blocks.error());
  }
  b.blocks = std::move(*blocks);
  // The target's tables, whose shapes the drafter reads.
  b.blocks.token_embd = target.token_embd;
  b.blocks.output = target.output;
  return std::move(b);
}

}  // namespace

std::expected<DsparkBinding, std::string> BindDspark(const DsparkProfile& profile,
                                                     const artifact::Artifact& drafter,
                                                     const Dsv4Profile& target_profile,
                                                     const Dsv4Binding& target) {
  DsparkBinding b;
  auto extra = DsparkRoles(profile, target_profile, target, b);
  if (!extra) {
    return std::unexpected(extra.error());
  }
  return Finish(BindDsv4Roles(profile.blocks, "dflash", drafter, false, *extra), target, b);
}

std::expected<DsparkBinding, std::string> BindDspark(const DsparkProfile& profile,
                                                     std::string_view architecture,
                                                     std::span<const Dsv4Resource> drafter,
                                                     const Dsv4Profile& target_profile,
                                                     const Dsv4Binding& target) {
  DsparkBinding b;
  auto extra = DsparkRoles(profile, target_profile, target, b);
  if (!extra) {
    return std::unexpected(extra.error());
  }
  return Finish(BindDsv4Roles(profile.blocks, "dflash", architecture, false, drafter, *extra),
                target, b);
}

// ---------------------------------------------------------------- state

StateRepresentation DsparkStateLayout::Representation(std::uint32_t max_verify) const {
  return StateRepresentation{
      .name = "dspark.ring",
      .block_positions = 0,
      .block_bytes = Bytes(bytes),
      .capabilities = max_verify == 0 ? static_cast<std::uint8_t>(StateCapability::kAppend)
                                      : (StateCapability::kAppend | StateCapability::kTruncate),
      .max_snapshots = 0,
      .snapshot_bytes = Bytes(0)};
}

std::expected<DsparkStateLayout, std::string> DsparkState(const DsparkProfile& profile,
                                                          std::uint32_t max_rows) {
  const Dsv4Profile& p = profile.blocks;
  if (max_rows == 0 || profile.ring == 0 || profile.ring % 256 != 0 ||
      std::uint64_t{profile.ring} < std::uint64_t{p.window} + profile.block_size) {
    return Refused("the drafter's ring must hold its window and a block, in 256-cell steps");
  }
  DsparkStateLayout s;
  s.ring = profile.ring;
  s.max_rows = max_rows;
  const std::uint64_t ring_bytes = std::uint64_t{p.head_dim} * profile.ring * 2;
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    s.offsets.push_back(s.bytes);
    s.bytes = Pad(s.bytes + ring_bytes, 256);
  }
  return s;
}

// ---------------------------------------------------------------- inputs

std::expected<DsparkBlockInputs, std::string> DsparkBlock(const DsparkProfile& profile,
                                                          const DsparkStateLayout& state,
                                                          std::uint32_t pos0, std::int32_t anchor,
                                                          std::uint32_t rows,
                                                          bool materialize_mask) {
  const Dsv4Profile& p = profile.blocks;
  if (rows == 0 || rows > profile.block_size || rows > state.max_rows) {
    return Refused(std::format("a draft block of {} rows (at most {})", rows,
                               std::min(profile.block_size, state.max_rows)));
  }
  if (anchor < 0 || std::cmp_greater_equal(anchor, p.vocab)) {
    return Refused(std::format("the anchor {} is outside the vocabulary", anchor));
  }
  if (std::uint64_t{pos0} + rows >
      static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
    return Refused("positions beyond int32");
  }
  if (state.ring != profile.ring || p.window == 0 ||
      std::uint64_t{state.ring} < std::uint64_t{p.window} + rows ||
      std::uint64_t{rows} * state.ring > INT32_MAX / sizeof(std::uint16_t))
    return Refused("a draft ring must retain its window and whole block within input bounds");
  DsparkBlockInputs in;
  in.pos0 = pos0;
  in.rows = rows;
  const std::uint64_t ring = state.ring;
  const std::uint64_t end = std::uint64_t{pos0} + rows;  // one past the block
  if (materialize_mask) in.mask.assign(std::size_t{rows} * ring, kHalfNegInf);
  for (std::uint32_t i = 0; i < rows; ++i) {
    in.tokens.push_back(i == 0 ? anchor : profile.mask_token);
    const std::uint64_t pos = std::uint64_t{pos0} + i;
    in.positions.push_back(static_cast<std::int32_t>(pos));
    in.cells.push_back(static_cast<std::int64_t>(pos % ring));
    for (std::uint64_t c = 0; materialize_mask && c < ring; ++c) {
      if (c >= end) {
        continue;  // no position has reached this cell yet
      }
      // The position cell c holds once the block is written: the latest
      // one below `end` congruent to it.
      const std::uint64_t q = c + (((end - 1 - c) / ring) * ring);
      // Non-causal within the window (llama.cpp's is_masked_swa: masked
      // when p1 - p0 >= n_swa, never for a later position).
      if (q > pos || pos - q < p.window) {
        in.mask[(std::size_t{i} * ring) + c] = kHalfZero;
      }
    }
  }
  return in;
}

DsparkInjection DsparkInject(const DsparkStateLayout& state, std::uint32_t n_past,
                             std::uint32_t rows) {
  DsparkInjection out;
  const std::uint32_t injected = std::min(rows, state.ring);
  out.first = rows - injected;
  for (std::uint32_t i = out.first; i < rows; ++i) {
    out.cells.push_back(static_cast<std::int64_t>((std::uint64_t{n_past} + i) % state.ring));
  }
  return out;
}

std::vector<std::vector<StateRange>> DsparkWrites(const DsparkProfile& profile,
                                                  const DsparkStateLayout& state,
                                                  std::uint32_t n_past, std::uint32_t rows) {
  std::vector<std::vector<StateRange>> out(rows);
  const DsparkInjection inject = DsparkInject(state, n_past, rows);
  const std::uint64_t cell = std::uint64_t{profile.blocks.head_dim} * 2;
  for (std::size_t k = 0; k < inject.cells.size(); ++k) {
    for (const std::uint64_t offset : state.offsets) {
      out[inject.first + k].push_back(
          {.offset = offset + (static_cast<std::uint64_t>(inject.cells[k]) * cell), .bytes = cell});
    }
  }
  return out;
}

}  // namespace llmp::model
