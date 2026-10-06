// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/paged_weights.h"

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <format>
#include <numeric>
#include <optional>
#include <utility>

#include "artifact/layout.h"
#include "engine/support.h"
#include "ggml.h"
#include "kernels/ggml/dsv4_graph.h"  // GgmlTypeOf

namespace jitllm::engine {

namespace {

namespace sc = jitllm::scheduler;
using base::Bytes;
using catalog::ExtentId;
using catalog::MemoryClass;
using catalog::Recovery;
using support::Error;
using support::Round;

constexpr std::uint64_t kExtent = kPagedExtent;
constexpr std::uint64_t kFileAlignment = 4096;

}  // namespace

// ------------------------------------------------------------------ slabs

std::expected<SlabLayout, std::string> LayOutSlab(std::span<const std::uint32_t> shard,
                                                  std::span<const std::uint64_t> file,
                                                  std::uint64_t stored, std::uint64_t stride,
                                                  std::uint64_t alignment) {
  const std::size_t n = file.size();
  if (n == 0 || shard.size() != n || stored == 0 || stride < stored ||
      stored % kFileAlignment != 0) {
    return Error("not a slab: no groups, or a stride below the groups' stored bytes");
  }
  if (alignment < 16 || alignment > kFileAlignment || (alignment & (alignment - 1)) != 0) {
    return Error("a slab's alignment is a power of two from 16 to 4,096");
  }
  // Consecutive in the file within a shard, changing shard at most once.
  std::optional<std::size_t> change;
  for (std::size_t e = 1; e < n; ++e) {
    if (shard[e] == shard[e - 1]) {
      if (file[e] != file[e - 1] + stored) {
        return Error(
            std::format("expert {}'s group does not follow expert {}'s in the file", e, e - 1));
      }
    } else if (change) {
      return Error("the slab's groups change shard more than once");
    } else {
      change = e;
    }
  }
  SlabLayout out;
  out.stride = stride;
  const std::uint64_t gap = stride - stored;
  if (change) {
    // A page boundary in the gap before the group that starts the second
    // shard: δ ≡ -e·S (mod 2 MiB), rounded up to the alignment within the
    // gap.
    const std::uint64_t at = (*change * stride) % kExtent;
    const std::uint64_t r = (kExtent - at) % kExtent;
    const std::uint64_t delta = Round(r, alignment);
    if (delta - r > gap) {
      return Error("the gap between groups is too small to align a shard change with a page");
    }
    out.delta = delta % kExtent;
  }
  const std::uint64_t end = out.delta + ((n - 1) * stride) + stored;
  const std::uint64_t pages = (end + kExtent - 1) / kExtent;
  out.pages.reserve(pages);
  for (std::uint64_t p = 0; p < pages; ++p) {
    const std::uint64_t lo = p * kExtent;
    const std::uint64_t hi = lo + kExtent;
    SlabPage page;
    std::uint64_t first = 0;  // the first piece's file position
    std::uint64_t last = 0;   // the last piece's file end
    // Experts whose groups overlap [lo, hi): from the one containing lo.
    std::size_t e = lo > out.delta ? static_cast<std::size_t>((lo - out.delta) / stride) : 0;
    for (; e < n && out.delta + (e * stride) < hi; ++e) {
      const std::uint64_t start = out.delta + (e * stride);
      const std::uint64_t piece_lo = std::max(lo, start);
      const std::uint64_t piece_hi = std::min(hi, start + stored);
      if (piece_lo >= piece_hi) {
        continue;  // the page starts in the gap after this group
      }
      const std::uint64_t position = file[e] + (piece_lo - start);
      if (page.pieces == 0) {
        page.shard = shard[e];
        first = position;
      } else if (shard[e] != page.shard || position != last) {
        return Error(std::format("page {} needs bytes that are not one range of one file", p));
      }
      if (page.pieces == kSlabPieces) {
        return Error(std::format("page {} needs more than {} pieces", p, kSlabPieces));
      }
      page.page_offset.at(page.pieces) = piece_lo - lo;
      page.bytes.at(page.pieces) = piece_hi - piece_lo;
      page.slot_offset.at(page.pieces) = position;  // made relative below
      ++page.pieces;
      last = position + (piece_hi - piece_lo);
    }
    if (page.pieces == 0) {
      return Error(std::format("page {} holds no group's bytes", p));
    }
    page.file_offset = first / kFileAlignment * kFileAlignment;
    page.length = Round(last, kFileAlignment) - page.file_offset;
    if (page.length > kSlabSlotBytes) {
      return Error(std::format("page {}'s read is longer than a slot", p));
    }
    for (std::size_t i = 0; i < page.pieces; ++i) {
      page.slot_offset.at(i) -= page.file_offset;
    }
    out.pages.push_back(page);
  }
  return out;
}

std::expected<SlabSpec, std::string> ExpertSlab(
    const artifact::Artifact& artifact,
    std::span<const std::pair<std::uint32_t, std::string_view>> arrays, std::uint32_t count,
    std::uint64_t alignment, std::uint32_t layer) {
  std::uint64_t unit = 16;
  std::optional<std::uint32_t> first_group;
  for (const auto& [index, type_name] : arrays) {
    const auto& a = artifact.expert_arrays()[index];
    auto type = kernels::ggml::GgmlTypeOf(type_name);
    if (!type) {
      return Error(type.error().detail);
    }
    unit = std::lcm(unit, static_cast<std::uint64_t>(ggml_type_size(*type)));
    if (first_group && *first_group != a.first_group) {
      return Error(std::format("layer {}'s expert arrays do not share their groups", layer));
    }
    first_group = a.first_group;
  }
  const std::uint32_t g = first_group.value_or(0);
  if (g >= artifact.groups().size()) {
    return Error(std::format("layer {}'s expert arrays name no group", layer));
  }
  return SlabSpec{.first_group = g,
                  .count = count,
                  .stride = Round(artifact.groups()[g].stored.value(), unit),
                  .alignment = alignment};
}

// ------------------------------------------------------------------ weights

std::array<std::uint8_t, 32> ArtifactKey(const artifact::Artifact& artifact) {
  std::array<std::uint8_t, 32> id{};
  const std::string& hex = artifact.id();
  for (std::size_t i = 0; i < id.size() && (2 * i) + 1 < hex.size(); ++i) {
    (void)std::from_chars(hex.data() + (2 * i), hex.data() + (2 * i) + 2, id.at(i), 16);
  }
  return id;
}

PagedWeights::Status PagedWeights::Open(const std::filesystem::path& path,
                                        const artifact::OpenOptions& options) {
  checked_places_.reset();
  auto opened = artifact::Artifact::Open(path, options);
  if (!opened) {
    return Error(
        std::format("the artifact {} was refused: {}", path.string(), opened.error().ToString()));
  }
  artifact_ = std::make_unique<artifact::Artifact>(std::move(*opened));
  for (std::uint32_t s = 0; s < artifact_->shards().size(); ++s) {
    auto fd = artifact_->OpenShardForDirectRead(s);
    if (!fd) {
      return Error(
          std::format("shard {} of {} cannot be opened for direct reads", s, path.string()));
    }
    shards_.push_back(std::move(*fd));
  }
  // The artifact's identity orders victim ties (catalog.h ContentKey).
  key_ = ArtifactKey(*artifact_);
  return {};
}

PagedWeights::Status PagedWeights::Reserve(PagedNode& node, std::span<const GroupPlace> place,
                                           std::span<const SlabSpec> slabs) {
  checked_places_.reset();
  if (!opened()) {
    return Error("the weights' artifact is not open");
  }
  const artifact::Artifact& artifact = *artifact_;
  const auto groups = artifact.groups();
  if (place.size() != groups.size()) {
    return Error("a place per group");
  }
  // Each slab's layout, and which slab each of its groups is in.
  std::vector<SlabLayout> layouts(slabs.size());
  std::vector<std::int64_t> slab_of(groups.size(), -1);
  for (std::size_t s = 0; s < slabs.size(); ++s) {
    const SlabSpec& spec = slabs[s];
    if (spec.count == 0 || std::uint64_t{spec.first_group} + spec.count > groups.size()) {
      return Error(std::format("slab {} names groups the artifact does not have", s));
    }
    const std::uint64_t stored = groups[spec.first_group].stored.value();
    std::vector<std::uint32_t> shard(spec.count);
    std::vector<std::uint64_t> file(spec.count);
    for (std::uint32_t e = 0; e < spec.count; ++e) {
      const std::uint32_t g = spec.first_group + e;
      if (groups[g].kind != artifact::GroupKind::kExpert || groups[g].stored.value() != stored ||
          place[g] != GroupPlace::kNone || slab_of[g] >= 0) {
        return Error(std::format("slab {}'s expert groups are not uniform", s));
      }
      const auto range = artifact::ChunkRangeOf(artifact.layout(), {.group = g, .chunk = 0});
      if (!range) {
        return Error("an expert group's file range");
      }
      shard[e] = range->shard;
      file[e] = range->file_offset.value();
      slab_of[g] = static_cast<std::int64_t>(s);
    }
    auto layout = LayOutSlab(shard, file, stored, spec.stride, spec.alignment);
    if (!layout) {
      return Error(std::format("slab {}: {}", s, layout.error()));
    }
    layouts[s] = std::move(*layout);
    slab_padding_ += (spec.stride - stored) * spec.count;
  }
  for (std::size_t g = 0; g < groups.size(); ++g) {
    if (place[g] != GroupPlace::kNone && groups[g].kind == artifact::GroupKind::kExpert) {
      return Error(std::format("expert group {} is placed as a dense group", g));
    }
  }
  // Places, in file order: dense regions and slabs in the device
  // reservation, host groups in the host one.
  std::vector<std::uint32_t> order(groups.size());
  std::ranges::iota(order, 0U);
  std::ranges::sort(order, [&](std::uint32_t a, std::uint32_t b) {
    return std::pair(groups[a].shard, groups[a].offset.value()) <
           std::pair(groups[b].shard, groups[b].offset.value());
  });
  std::vector<std::uint64_t> region(groups.size(), 0);
  std::vector<std::uint64_t> slab_region(slabs.size(), 0);
  std::vector<bool> slab_placed(slabs.size(), false);
  for (const std::uint32_t g : order) {
    if (slab_of[g] >= 0) {
      const auto s = static_cast<std::size_t>(slab_of[g]);
      if (!slab_placed[s]) {
        slab_placed[s] = true;
        slab_region[s] = bytes_;
        bytes_ += layouts[s].bytes();
      }
    } else if (place[g] == GroupPlace::kDevice) {
      region[g] = bytes_;
      bytes_ += std::uint64_t{groups[g].chunks} * kExtent;
    } else if (place[g] == GroupPlace::kHost) {
      region[g] = host_bytes_;
      host_bytes_ += std::uint64_t{groups[g].chunks} * kExtent;
    }
  }
  auto& memory = node.memory();
  auto reservation = memory.Reserve(Bytes(std::max<std::uint64_t>(bytes_, kExtent)));
  if (!reservation) {
    return Error(std::format("reserving the weights: {}", reservation.error().detail));
  }
  reservation_ = *reservation;
  const std::uint64_t base = memory.RangeOf(reservation_).value().base;
  std::uint64_t host_base = 0;
  if (host_bytes_ != 0) {
    auto host = memory.Reserve(Bytes(host_bytes_));
    if (!host) {
      return Error(std::format("reserving the weights' host groups: {}", host.error().detail));
    }
    host_reservation_ = *host;
    host_base = memory.RangeOf(host_reservation_).value().base;
  }
  group_address_.assign(groups.size(), 0);
  for (std::uint32_t g = 0; g < groups.size(); ++g) {
    if (slab_of[g] >= 0) {
      const auto s = static_cast<std::size_t>(slab_of[g]);
      group_address_[g] = base + slab_region[s] + layouts[s].delta +
                          (std::uint64_t{g - slabs[s].first_group} * slabs[s].stride);
    } else if (place[g] == GroupPlace::kDevice) {
      group_address_[g] = base + region[g];
    } else if (place[g] == GroupPlace::kHost) {
      group_address_[g] = host_base + region[g];
    }
  }

  const auto add = [&](catalog::ContentKey content) -> std::expected<ExtentId, std::string> {
    auto extent = node.catalog().AddExtent({.domain = node.domain(),
                                            .memory_class = MemoryClass::kWeights,
                                            .recovery = Recovery::kFromArtifact,
                                            .size = Bytes(kExtent),
                                            .content = content});
    if (!extent) {
      return Error("cataloging a weight extent");
    }
    return *extent;
  };
  std::ranges::fill(slab_placed, false);
  for (const std::uint32_t g : order) {
    if (slab_of[g] >= 0) {
      const auto s = static_cast<std::size_t>(slab_of[g]);
      if (slab_placed[s]) {
        continue;
      }
      slab_placed[s] = true;
      const SlabLayout& slab = layouts[s];
      for (std::uint32_t p = 0; p < slab.pages.size(); ++p) {
        const SlabPage& page = slab.pages[p];
        auto extent = add({.artifact = key_, .group = slabs[s].first_group, .chunk = p});
        if (!extent) {
          return std::unexpected(extent.error());
        }
        const std::uint64_t offset = slab_region[s] + (std::uint64_t{p} * kExtent);
        sc::PageSource source{.read = {.fd = shards_.at(page.shard).get(),
                                       .offset = page.file_offset,
                                       .memory = nullptr,
                                       .length = page.length},
                              .landed = true,
                              .destination = 0,
                              .pieces = {},
                              .piece_count = page.pieces,
                              .backing = sc::BackingPlace{.reservation = reservation_,
                                                          .offset = Bytes(offset),
                                                          .size = Bytes(kExtent),
                                                          .allocation_class = node.device_class()}};
        for (std::size_t i = 0; i < page.pieces; ++i) {
          source.pieces.at(i) =
              sc::LandedPiece{.slot_offset = page.slot_offset.at(i),
                              .destination = base + offset + page.page_offset.at(i),
                              .length = Bytes(page.bytes.at(i))};
        }
        read_bytes_ += page.length;
        extents_.push_back(*extent);
        sources_.push_back({.address = base + offset,
                            .group = slabs[s].first_group,
                            .host = false,
                            .source = source});
      }
      continue;
    }
    if (place[g] == GroupPlace::kNone) {
      continue;
    }
    const bool host = place[g] == GroupPlace::kHost;
    for (std::uint32_t c = 0; c < groups[g].chunks; ++c) {
      const auto range = artifact::ChunkRangeOf(artifact.layout(), {.group = g, .chunk = c});
      if (!range) {
        return Error("a chunk's range");
      }
      auto extent = add({.artifact = key_, .group = g, .chunk = c});
      if (!extent) {
        return std::unexpected(extent.error());
      }
      const std::uint64_t offset = region[g] + (std::uint64_t{c} * kExtent);
      const std::uint64_t address = (host ? host_base : base) + offset;
      sc::PageSource source{
          .read = {.fd = shards_.at(range->shard).get(),
                   .offset = range->file_offset.value(),
                   // A host group's chunk in place, into host VMM the CPU
                   // maps (D-034).
                   // NOLINTNEXTLINE(performance-no-int-to-ptr)
                   .memory = host ? reinterpret_cast<std::byte*>(address) : nullptr,
                   .length = range->length.value()},
          .landed = !host,
          .destination = host ? 0 : address,
          .backing =
              sc::BackingPlace{.reservation = host ? host_reservation_ : reservation_,
                               .offset = Bytes(offset),
                               .size = Bytes(kExtent),
                               .allocation_class = host ? node.host_class() : node.device_class()}};
      read_bytes_ += range->length.value();
      extents_.push_back(*extent);
      sources_.push_back({.address = address, .group = g, .host = host, .source = source});
    }
  }
  return {};
}

PagedWeights::Status PagedWeights::Register(PagedNode& node, int owner) {
  checked_places_.reset();
  for (std::size_t i = 0; i < extents_.size(); ++i) {
    auto set = node.scheduler().SetSource(extents_[i], sources_[i].source);
    if (!set) {
      return Error(std::format("a weight's source: {}", sc::ToString(set.error())));
    }
    node.AddSpan({.base = sources_[i].address,
                  .size = kExtent,
                  .extent = extents_[i],
                  .memory_class = MemoryClass::kWeights,
                  .device = !sources_[i].host,
                  .owner = owner});
  }
  return {};
}

void PagedWeights::CheckPlaces(const sc::Scheduler& scheduler, PlaceCheck& check) const {
  (void)CheckPlacesImpl(scheduler, check);
}

bool PagedWeights::CheckPlacesImpl(const sc::Scheduler& scheduler, PlaceCheck& check) const {
  const auto stamp = scheduler.placement_stamp();
  if (stamp.cacheable() && checked_places_ == stamp) return true;
  checked_places_.reset();
  const auto before = check.moved;
  for (std::size_t i = 0; i < extents_.size() && i < sources_.size(); ++i) {
    check.Check(scheduler, extents_[i], sources_[i].source);
  }
  if (stamp.cacheable() && extents_.size() == sources_.size() && check.moved == before) {
    checked_places_ = stamp;
  }
  return false;
}

std::uint64_t PagedWeights::resource_address(std::uint32_t resource) const {
  const auto& r = artifact_->resources()[resource];
  return group_address_.at(r.group) + r.offset.value();
}

std::uint64_t PagedWeights::array_address(std::uint32_t array) const {
  const auto& a = artifact_->expert_arrays()[array];
  return group_address_.at(a.first_group) + a.group_offset.value();
}

std::vector<ExtentId> PagedWeights::ExtentsWhere(
    const std::function<bool(std::uint32_t group, bool host)>& which) const {
  std::vector<ExtentId> out;
  for (std::size_t i = 0; i < extents_.size() && i < sources_.size(); ++i) {
    if (which(sources_[i].group, sources_[i].host)) {
      out.push_back(extents_[i]);
    }
  }
  return out;
}

std::vector<PagedWeights::Range> PagedWeights::Unwritten() const {
  std::vector<Range> out;
  for (const Source& s : sources_) {
    if (s.host) {
      continue;
    }
    const std::uint64_t start = s.address;
    const std::uint64_t end = start + kExtent;
    if (s.source.piece_count == 0) {
      const std::uint64_t written = s.source.destination + s.source.read.length;
      if (written < end) {
        out.push_back({.address = written, .bytes = end - written, .slab = false});
      }
      continue;
    }
    // Pieces ascend within the page (LayOutSlab).
    std::uint64_t at = start;
    for (std::size_t i = 0; i < s.source.piece_count; ++i) {
      const sc::LandedPiece& piece = s.source.pieces.at(i);
      if (piece.destination > at) {
        out.push_back({.address = at, .bytes = piece.destination - at, .slab = true});
      }
      at = std::max(at, piece.destination + piece.length.value());
    }
    if (at < end) {
      out.push_back({.address = at, .bytes = end - at, .slab = true});
    }
  }
  return out;
}

PagedWeights::Status PagedWeights::Release(providers::VmmProvider& memory) {
  checked_places_.reset();
  std::vector<std::string> problems;
  for (providers::ReservationId* reservation : {&reservation_, &host_reservation_}) {
    if (reservation->valid() && !memory.Free(*reservation)) {
      problems.emplace_back("a weights reservation still has mappings");
      continue;
    }
    *reservation = {};
  }
  return support::Joined(problems);
}

}  // namespace jitllm::engine
