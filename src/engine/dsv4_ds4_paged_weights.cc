// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/dsv4_ds4_paged_weights.h"

#include <algorithm>
#include <format>
#include <set>
#include <string_view>
#include <utility>

namespace jitllm::engine {
namespace {
namespace kg = kernels::ggml;
constexpr std::uint64_t kExtent = kPagedExtent;
constexpr std::string_view kCommunityArtifact =
    "cd39d504dc2dbfe911a4a521fa8efc8053dc3e80e99738a9b25fa6b70c97a1ac";

bool SameLayout(const kg::Ds4AlignedLayout& a, const kg::Ds4AlignedLayout& b) {
  return a.raw_bytes == b.raw_bytes && a.packed_bytes == b.packed_bytes && a.blocks == b.blocks &&
         a.scales_offset == b.scales_offset && a.codes_offset == b.codes_offset &&
         a.blocks_per_row == b.blocks_per_row && a.raw_block_bytes == b.raw_block_bytes;
}

base::Sha256Digest ContentKey(const Ds4PreparedWeight& tensor, const Ds4PreparedWeightFile& file) {
  base::Sha256 hash;
  // Delimited fixed fields distinguish canonical raw content, expert/dense
  // namespaces and physical layouts. Names never carry addresses or FDs.
  hash.Update("jitllm-temporary-ds4-aligned-v1\n");
  hash.Update(kCommunityArtifact).Update("\n");
  hash.Update("76d51ef82a81b70b78e51a3a6ea11946286de976\n");
  hash.Update(std::format("{}:{}:{}:{}:{}:{}:{}\n", tensor.expert_array, tensor.index,
                          static_cast<unsigned>(tensor.shape.kind), tensor.shape.input,
                          tensor.shape.output, tensor.shape.groups, file.stored_bytes.value()));
  hash.Update(base::ToHex(file.sha256));
  return hash.Finish();
}
}  // namespace

Status Ds4PagedAlignedWeights::Reserve(PagedNode& node, const Ds4PreparedWeightSet& plan,
                                       std::vector<Ds4PreparedWeightFile>& files) {
  if (reservation_.valid() || node.threaded() || plan.artifact_id != kCommunityArtifact ||
      plan.tensors.empty() || plan.tensors.size() > 1024 || plan.tensors.size() != files.size()) {
    return std::unexpected("invalid complete aligned weight file inventory");
  }
  std::set<std::pair<bool, std::uint32_t>> identities;
  std::uint64_t total = 0;
  for (std::size_t i = 0; i < files.size(); ++i) {
    const auto& tensor = plan.tensors[i];
    const auto& file = files[i];
    const auto layout = kg::Ds4AlignedLayoutOf(tensor.shape);
    const auto payload = file.payload_bytes.value();
    const auto stored = file.stored_bytes.value();
    std::uint64_t rounded = 0;
    if (!layout || !SameLayout(*layout, tensor.layout) || file.file.get() < 0 ||
        payload != layout->packed_bytes.value() || payload == 0 ||
        __builtin_add_overflow(payload, artifact::kFileAlignment - 1, &rounded) ||
        stored != rounded / artifact::kFileAlignment * artifact::kFileAlignment ||
        !identities.emplace(tensor.expert_array, tensor.index).second ||
        __builtin_add_overflow(stored, kExtent - 1, &rounded)) {
      return std::unexpected("aligned weight file does not match its validated tensor");
    }
    const auto region = rounded / kExtent * kExtent;
    if (__builtin_add_overflow(total, region, &total) || total > (std::uint64_t{1} << 40U)) {
      return std::unexpected("aligned weight virtual reservation overflows");
    }
  }
  auto reservation = node.memory().Reserve(base::Bytes(total));
  if (!reservation) return std::unexpected(reservation.error().detail);
  reservation_ = *reservation;
  auto range = node.memory().RangeOf(reservation_);
  if (!range) return std::unexpected(range.error().detail);
  bytes_ = total;
  files_ = std::move(files);
  std::uint64_t offset = 0;
  for (std::size_t i = 0; i < files_.size(); ++i) {
    const auto& tensor = plan.tensors[i];
    const auto& file = files_[i];
    const auto stored = file.stored_bytes.value();
    const auto pages = (stored + kExtent - 1) / kExtent;
    const auto key = ContentKey(tensor, file);
    const auto memory_class =
        tensor.expert_array ? catalog::MemoryClass::kRoutedExperts : catalog::MemoryClass::kWeights;
    views_.push_back({.expert_array = tensor.expert_array,
                      .index = tensor.index,
                      .shape = tensor.shape,
                      .layout = tensor.layout,
                      .address = range->base + offset,
                      .stored_sha256 = file.sha256,
                      .content_key = key});
    for (std::uint32_t page = 0; page < pages; ++page) {
      const auto in_file = static_cast<std::uint64_t>(page) * kExtent;
      const auto address = range->base + offset;
      auto extent =
          node.catalog().AddExtent({.domain = node.domain(),
                                    .memory_class = memory_class,
                                    .recovery = catalog::Recovery::kFromArtifact,
                                    .size = base::Bytes(kExtent),
                                    .content = {.artifact = key, .group = 0, .chunk = page}});
      if (!extent) return std::unexpected("catalog refused an aligned weight extent");
      const auto length = std::min(kExtent, stored - in_file);
      scheduler::PageSource source{
          .read = {.fd = file.file.get(), .offset = in_file, .memory = nullptr, .length = length},
          .landed = true,
          .destination = address,
          .backing = scheduler::BackingPlace{.reservation = reservation_,
                                             .offset = base::Bytes(offset),
                                             .size = base::Bytes(kExtent),
                                             .allocation_class = node.device_class()}};
      extents_.push_back(*extent);
      sources_.push_back({.address = address, .memory_class = memory_class, .source = source});
      read_bytes_ += length;
      offset += kExtent;
    }
  }
  return {};
}

Status Ds4PagedAlignedWeights::Register(PagedNode& node, int owner) {
  if (!reservation_.valid() || registered_ || node.threaded() ||
      sources_.size() != extents_.size()) {
    return std::unexpected("aligned weight sources registered out of order");
  }
  for (std::size_t i = 0; i < sources_.size(); ++i) {
    auto registered = node.scheduler().SetSource(extents_[i], sources_[i].source);
    if (!registered) return std::unexpected(scheduler::ToString(registered.error()));
    node.AddSpan({.base = sources_[i].address,
                  .size = kExtent,
                  .extent = extents_[i],
                  .memory_class = sources_[i].memory_class,
                  .device = true,
                  .owner = owner});
  }
  registered_ = true;
  return {};
}

void Ds4PagedAlignedWeights::CheckPlaces(const scheduler::Scheduler& scheduler,
                                         PlaceCheck& check) const {
  for (std::size_t i = 0; i < extents_.size(); ++i) {
    check.Check(scheduler, extents_[i], sources_[i].source);
  }
}

const Ds4AlignedWeightView* Ds4PagedAlignedWeights::Find(std::uint32_t index,
                                                         bool expert_array) const {
  const auto found = std::ranges::find_if(views_, [=](const auto& view) {
    return view.index == index && view.expert_array == expert_array;
  });
  return found == views_.end() ? nullptr : &*found;
}

Status Ds4PagedAlignedWeights::Release(providers::VmmProvider& memory) {
  if (reservation_.valid()) {
    auto freed = memory.Free(reservation_);
    if (!freed) return std::unexpected(freed.error().detail);
    reservation_ = {};
  }
  // Freeing the reservation proves no backing remains; native node teardown
  // is additionally required to prove all file reads and consumers retired.
  files_.clear();
  views_.clear();
  sources_.clear();
  extents_.clear();
  registered_ = false;
  bytes_ = 0;
  read_bytes_ = 0;
  return {};
}

}  // namespace jitllm::engine
