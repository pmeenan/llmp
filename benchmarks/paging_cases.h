// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// BP-P2's partial evictions (docs/backend-proof.md, case matrix), shared
// by the paged harnesses (fp16_paged.cc, exl3_paged.cc): each case names a
// set of weight chunks, by group and group-relative chunk, taken from the
// artifact's own placements, so the harness evicts exactly those extents
// and the next phase's materialization must bring back only them.

#ifndef LLMP_BENCHMARKS_PAGING_CASES_H_
#define LLMP_BENCHMARKS_PAGING_CASES_H_

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "artifact/artifact.h"

namespace llmp::benchmarks {

struct PartialCase {
  std::string name;
  std::set<std::pair<std::uint32_t, std::uint32_t>> chunks;  // (group, chunk)
};

// The cases, in order, each non-empty:
// - one layer: every chunk the resources whose names start with
//   `layer_prefix` need;
// - side vectors and biases: every chunk holding a resource of at most
//   64 KiB (norms, biases and, in EXL3, the Hadamard sign vectors); a
//   chunk evicts whatever else shares it too;
// - the trellis only (EXL3, `trellis`): the chunks of that layer's
//   `.trellis` resources;
// - a shared small-tensor chunk: the first chunk that three or more
//   resources share;
// - padded tails: each group's last chunk where the group's stored bytes
//   do not fill it;
// - a tensor crossing a chunk boundary: the chunks of the first resource
//   of that layer whose range spans two or more.
// A resource's chunks are those its readable range needs (its placement's
// closure). Empty cases are left out.
inline std::vector<PartialCase> PartialCases(const artifact::Artifact& a,
                                             std::string_view layer_prefix, bool trellis) {
  constexpr std::uint64_t kSmall = 64ULL << 10U;
  constexpr std::uint64_t kChunk = 2ULL << 20U;
  const auto closure_of = [&](std::uint32_t r) {
    std::set<std::pair<std::uint32_t, std::uint32_t>> chunks;
    const auto placed = a.ResourcePlacement(r);
    if (placed) {
      for (std::uint32_t c = placed->closure.first; c <= placed->closure.last; ++c) {
        chunks.emplace(placed->closure.group, c);
      }
    }
    return chunks;
  };
  const auto resources = a.resources();
  PartialCase layer{.name = "one layer", .chunks = {}};
  PartialCase side{.name = "side vectors and biases", .chunks = {}};
  PartialCase trellis_only{.name = "trellis only", .chunks = {}};
  PartialCase crossing{.name = "a tensor crossing a chunk boundary", .chunks = {}};
  std::map<std::pair<std::uint32_t, std::uint32_t>, int> sharing;
  for (std::uint32_t r = 0; r < resources.size(); ++r) {
    const auto& resource = resources[r];
    const auto chunks = closure_of(r);
    for (const auto& chunk : chunks) {
      ++sharing[chunk];
    }
    if (resource.bytes.value() <= kSmall) {
      side.chunks.insert(chunks.begin(), chunks.end());
    }
    if (!resource.name.starts_with(layer_prefix)) {
      continue;
    }
    layer.chunks.insert(chunks.begin(), chunks.end());
    if (trellis && resource.name.ends_with(".trellis")) {
      trellis_only.chunks.insert(chunks.begin(), chunks.end());
    }
    if (crossing.chunks.empty() && chunks.size() >= 2) {
      crossing.chunks = chunks;
    }
  }
  PartialCase shared{.name = "a shared small-tensor chunk", .chunks = {}};
  for (const auto& [chunk, count] : sharing) {
    if (count >= 3) {
      shared.chunks.insert(chunk);
      break;
    }
  }
  PartialCase tails{.name = "padded tails", .chunks = {}};
  const auto groups = a.groups();
  for (std::uint32_t g = 0; g < groups.size(); ++g) {
    if (groups[g].chunks > 0 && groups[g].stored.value() % kChunk != 0) {
      tails.chunks.emplace(g, groups[g].chunks - 1);
    }
  }
  std::vector<PartialCase> cases;
  for (PartialCase* c : {&layer, &side, &trellis_only, &shared, &tails, &crossing}) {
    if (!c->chunks.empty()) {
      cases.push_back(std::move(*c));
    }
  }
  return cases;
}

}  // namespace llmp::benchmarks

#endif  // LLMP_BENCHMARKS_PAGING_CASES_H_
