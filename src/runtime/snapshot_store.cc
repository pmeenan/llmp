// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/snapshot_store.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "base/check.h"
#include "engine/paged_node.h"

namespace llmp::runtime {
namespace {
using Range = engine::LiveState::Range;
// Private in-process metadata, never a disk schema. All variable-length
// fields follow this header inside the same pinned allocation as the payload.
struct CaptureHeader {
  std::uint64_t generation = 0;
  std::uint64_t payload_bytes = 0;
  std::uint64_t total_bytes = 0;
  std::uint64_t ranges_at = 0, range_count = 0;
  std::uint64_t history_at = 0, history_count = 0;
  std::uint64_t layout_at = 0, layout_bytes = 0;
  std::uint32_t cursor = 0;
  std::optional<base::Sha256Digest> execution_identity;
  std::array<std::uint64_t, execution::AdaptiveDepth::kWords> decoding{};
};
static_assert(std::is_trivially_copyable_v<CaptureHeader>);
static_assert(std::is_trivially_destructible_v<CaptureHeader>);
static_assert(std::is_trivially_copyable_v<Range>);
bool Add(std::uint64_t& total, std::uint64_t count, std::uint64_t size) {
  if (count > (std::numeric_limits<std::uint64_t>::max() - total) / size) return false;
  total += count * size;
  return true;
}
bool Align(std::uint64_t& total, std::uint64_t alignment) {
  if (total > std::numeric_limits<std::uint64_t>::max() - alignment + 1) return false;
  total = (total + alignment - 1) / alignment * alignment;
  return true;
}
}  // namespace

struct SnapshotControl {
  void* allocation = nullptr;
  std::uint64_t capacity = 0;
  std::uint64_t header_at = 0;
  std::uint64_t generation = 0;
  std::weak_ptr<const std::byte> issuer;
  bool completed = false;
  bool unproven = false;
  // The stream whose fence frees the quarantined allocation (unproven).
  std::optional<std::uint32_t> unproven_stream;

  const CaptureHeader& Header() const {
    return *reinterpret_cast<const CaptureHeader*>(static_cast<const std::byte*>(allocation) +
                                                   header_at);
  }
};

SnapshotStore::SnapshotStore(engine::PagedNode& node, std::uint64_t ceiling, Reclaim reclaim)
    : node_(&node),
      node_lifetime_(node.staging_lifetime()),
      ceiling_(ceiling),
      reclaim_(std::move(reclaim)),
      control_(std::make_shared<SnapshotControl>()) {}

SnapshotStore::~SnapshotStore() {
  control_->completed = false;
  control_->issuer.reset();
  if (NodeAlive() && control_->allocation != nullptr && !control_->unproven) {
    if (!node_->FreePinned(control_->allocation)) node_->KeepPinned(control_->allocation);
  }
}

bool SnapshotStore::NodeAlive() const {
  const auto observer = node_lifetime_.lock();
  return observer && observer->live();
}
bool SnapshotStore::usable() const {
  return NodeAlive() && (!control_->unproven || (control_->unproven_stream.has_value() &&
                                                 !node_->Quarantining(*control_->unproven_stream)));
}
std::uint64_t SnapshotStore::capacity() const { return control_->capacity; }

std::expected<void, std::string> SnapshotStore::CheckNode(const engine::PagedNode* node) const {
  if (!usable() || node != node_ || ceiling_ == 0) {
    return std::unexpected("the snapshot store has no live, matching bounded node");
  }
  // A fence of the stream freed what an unproven copy left: a store whose
  // allocation went to the quarantine starts afresh (Prepare allocates).
  if (control_->unproven) {
    control_->unproven = false;
    control_->unproven_stream.reset();
  }
  return {};
}

std::expected<void, std::string> SnapshotStore::Prepare(
    std::weak_ptr<const std::byte> issuer, std::optional<base::Sha256Digest> execution_identity,
    std::string_view layout, std::uint32_t cursor, std::span<const Range> ranges,
    std::span<const std::int32_t> history, const execution::AdaptiveDepth& decoding) {
  if (auto checked = CheckNode(node_); !checked) return checked;
  if (issuer.expired() || control_->generation == std::numeric_limits<std::uint64_t>::max()) {
    return std::unexpected("the snapshot issuer or capture generation is expired");
  }
  CaptureHeader header;
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    const auto& range = ranges[i];
    if ((i != 0 && (ranges[i - 1].region > range.region ||
                    (ranges[i - 1].region == range.region &&
                     ranges[i - 1].offset + ranges[i - 1].bytes > range.offset))) ||
        range.bytes == 0 ||
        range.offset > std::numeric_limits<std::uint64_t>::max() - range.bytes ||
        !Add(header.payload_bytes, range.bytes, 1)) {
      return std::unexpected("the snapshot ranges overlap, are unordered, empty or overflow");
    }
  }
  std::uint64_t total = header.payload_bytes;
  if (!Align(total, alignof(CaptureHeader))) return std::unexpected("snapshot size overflow");
  const auto header_at = total;
  if (!Add(total, 1, sizeof(CaptureHeader)) || !Align(total, alignof(Range))) {
    return std::unexpected("snapshot size overflow");
  }
  header.ranges_at = total;
  header.range_count = ranges.size();
  if (!Add(total, ranges.size(), sizeof(Range)) || !Align(total, alignof(std::int32_t))) {
    return std::unexpected("snapshot size overflow");
  }
  header.history_at = total;
  header.history_count = history.size();
  if (!Add(total, history.size(), sizeof(std::int32_t)))
    return std::unexpected("snapshot size overflow");
  header.layout_at = total;
  header.layout_bytes = layout.size();
  if (!Add(total, layout.size(), 1) || total > ceiling_ ||
      total > std::numeric_limits<std::size_t>::max()) {
    return std::unexpected("the complete snapshot exceeds its storage ceiling");
  }
  // Match PagedNode::Pinned's minimum charge, including an empty payload.
  const auto allocation_bytes = std::max<std::uint64_t>(total, 256);
  if (allocation_bytes > ceiling_) {
    return std::unexpected("the minimum snapshot allocation exceeds its storage ceiling");
  }
  // No old/new payload overlap. Capacity refusal happens before any capture
  // publication; a growing allocation invalidates the old capability first.
  control_->completed = false;
  control_->issuer.reset();
  ++control_->generation;
  if (allocation_bytes > control_->capacity) {
    if (auto freed = node_->FreePinned(control_->allocation); !freed) {
      // FreePinned refuses before any physical release (a failure after it
      // fail-stops), but never infer ownership or reuse from an error:
      // KeepPinned retains it only if the node still owns the allocation.
      Unproven();
      return freed;
    }
    control_->allocation = nullptr;
    control_->capacity = 0;
    std::vector<catalog::ExtentId> staging;
    auto allocated = node_->Pinned(allocation_bytes, engine::kShared, staging);
    if (!allocated && reclaim_ && reclaim_(allocation_bytes)) {
      staging.clear();
      allocated = node_->Pinned(allocation_bytes, engine::kShared, staging);
    }
    if (!allocated) return std::unexpected(allocated.error());
    control_->allocation = *allocated;
    control_->capacity = allocation_bytes;
  }
  if (reinterpret_cast<std::uintptr_t>(control_->allocation) % alignof(CaptureHeader) != 0 ||
      reinterpret_cast<std::uintptr_t>(control_->allocation) % alignof(Range) != 0) {
    return std::unexpected("the pinned snapshot allocator broke its metadata alignment");
  }
  header.total_bytes = total;
  header.generation = control_->generation;
  header.cursor = cursor;
  header.execution_identity = execution_identity;
  header.decoding = decoding.Save();
  auto* bytes = static_cast<std::byte*>(control_->allocation);
  std::memcpy(bytes + header_at, &header, sizeof(header));
  if (!ranges.empty()) std::memcpy(bytes + header.ranges_at, ranges.data(), ranges.size_bytes());
  if (!history.empty())
    std::memcpy(bytes + header.history_at, history.data(), history.size_bytes());
  if (!layout.empty()) std::memcpy(bytes + header.layout_at, layout.data(), layout.size());
  control_->header_at = header_at;
  control_->issuer = std::move(issuer);
  return {};
}

void* SnapshotStore::WriteAddress() const { return control_->allocation; }

StateSnapshot SnapshotStore::Publish() {
  base::Check(usable() && !control_->issuer.expired() && control_->allocation != nullptr,
              "publishing an invalid snapshot capture");
  control_->completed = true;
  StateSnapshot out;
  out.control_ = control_;
  out.generation_ = control_->generation;
  return out;
}

void SnapshotStore::Unproven(std::optional<std::uint32_t> stream) {
  control_->completed = false;
  control_->unproven = true;
  control_->unproven_stream = stream;
  // Retain a possibly borrowed allocation only while the node still owns it
  // (the copy may have quarantined it already, tagged with its stream). A
  // failed release may already have freed it: KeepPinned then does nothing.
  // Either way, this store never reads, reuses or releases the address again.
  // Retained header/history/ranges need no pool or branch accounting owner.
  if (NodeAlive()) node_->KeepPinned(control_->allocation, stream);
  control_->allocation = nullptr;
  control_->capacity = 0;
}

std::expected<void, std::string> SnapshotStore::Authenticate(const StateSnapshot& snapshot) const {
  if (!usable() || snapshot.control_.lock() != control_ || !control_->completed ||
      snapshot.generation_ != control_->generation || control_->issuer.expired()) {
    return std::unexpected("the snapshot is foreign, stale, incomplete or its issuer has expired");
  }
  // The metadata is private and was copied only by Prepare. Still validate
  // its exact partition before following any span, so a bad transfer cannot
  // turn a completed flag into permission to read outside the allocation.
  if (control_->allocation == nullptr || control_->header_at > control_->capacity ||
      sizeof(CaptureHeader) > control_->capacity - control_->header_at) {
    return std::unexpected("the snapshot metadata header is outside its allocation");
  }
  const auto& header = control_->Header();
  std::uint64_t end = header.payload_bytes;
  if (header.generation != control_->generation || !Align(end, alignof(CaptureHeader)) ||
      end != control_->header_at || !Add(end, 1, sizeof(CaptureHeader)) ||
      !Align(end, alignof(Range)) || end != header.ranges_at ||
      !Add(end, header.range_count, sizeof(Range)) || !Align(end, alignof(std::int32_t)) ||
      end != header.history_at || !Add(end, header.history_count, sizeof(std::int32_t)) ||
      end != header.layout_at || !Add(end, header.layout_bytes, 1) || end != header.total_bytes ||
      end > control_->capacity || end > ceiling_) {
    return std::unexpected("the snapshot metadata does not describe its captured allocation");
  }
  return {};
}

std::expected<SnapshotStore::CaptureView, std::string> SnapshotStore::Read(
    const StateSnapshot& snapshot, const std::shared_ptr<const std::byte>& issuer,
    std::optional<base::Sha256Digest> execution_identity, std::string_view layout) const {
  if (auto authenticated = Authenticate(snapshot); !authenticated) {
    return std::unexpected(authenticated.error());
  }
  if (!issuer || control_->issuer.lock() != issuer) {
    return std::unexpected("the snapshot belongs to another branch issuer");
  }
  const auto& header = control_->Header();
  const auto* bytes = static_cast<const std::byte*>(control_->allocation);
  const std::string_view captured_layout(reinterpret_cast<const char*>(bytes + header.layout_at),
                                         header.layout_bytes);
  if (header.execution_identity != execution_identity || captured_layout != layout) {
    return std::unexpected("the snapshot execution or kept layout is incompatible");
  }
  const auto decoding = execution::AdaptiveDepth::Load(header.decoding);
  if (!decoding) return std::unexpected("the captured decoding metadata is invalid");
  return CaptureView{
      .payload = control_->allocation,
      .ranges = {reinterpret_cast<const Range*>(bytes + header.ranges_at), header.range_count},
      .history = {reinterpret_cast<const std::int32_t*>(bytes + header.history_at),
                  header.history_count},
      .layout = captured_layout,
      .cursor = header.cursor,
      .decoding = *decoding};
}

std::expected<base::Sha256Digest, std::string> SnapshotStore::Digest(
    const StateSnapshot& snapshot) const {
  if (auto authenticated = Authenticate(snapshot); !authenticated) {
    return std::unexpected(authenticated.error());
  }
  base::Sha256 hash;
  hash.Update(
      {static_cast<const std::byte*>(control_->allocation), control_->Header().payload_bytes});
  return hash.Finish();
}

}  // namespace llmp::runtime
