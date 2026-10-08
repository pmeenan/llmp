// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "model/state.h"

#include <algorithm>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <utility>

#include "base/bytes.h"

namespace llmp::model {
namespace {

std::optional<Bytes> Times(Bytes bytes, std::uint64_t count) {
  std::uint64_t product = 0;
  if (__builtin_mul_overflow(bytes.value(), count, &product)) {
    return std::nullopt;
  }
  return Bytes(product);
}

}  // namespace

bool IsValid(const StateRepresentation& representation) {
  const bool snapshots = representation.max_snapshots != 0;
  return representation.block_bytes != Bytes() &&
         snapshots == (representation.snapshot_bytes != Bytes()) &&
         (!snapshots || representation.Can(StateCapability::kTruncateAtSnapshot));
}

std::string ToString(StateError error) {
  switch (error) {
    case StateError::kInvalid:
      return "invalid state representation";
    case StateError::kUnsupported:
      return "the state representation lacks that capability";
    case StateError::kExceedsBound:
      return "the state would grow past its admitted bound";
    case StateError::kBelowCommitted:
      return "the truncation would discard committed positions";
    case StateError::kBeyondWritten:
      return "the positions were never written";
    case StateError::kNoSnapshot:
      return "no snapshot at the truncation target";
    case StateError::kSnapshotsFull:
      return "the state holds all the snapshots it may";
  }
  return "unknown state error";
}

std::optional<std::uint64_t> BlocksFor(const StateRepresentation& representation,
                                       std::uint64_t positions) {
  if (representation.block_positions == 0) {
    return 1;  // one fixed-size block
  }
  // Rounded up without overflowing at the top of the range.
  return (positions / representation.block_positions) +
         (positions % representation.block_positions != 0 ? 1 : 0);
}

std::optional<Bytes> StateAllowance(const StateRepresentation& representation,
                                    std::uint64_t positions) {
  return BlocksFor(representation, positions)
      .and_then([&](std::uint64_t blocks) { return Times(representation.block_bytes, blocks); })
      .and_then([&](Bytes blocks) {
        return Times(representation.snapshot_bytes, representation.max_snapshots)
            .and_then([&](Bytes snapshots) { return blocks.Plus(snapshots); });
      });
}

std::expected<StateCursor, StateError> StateCursor::Create(StateRepresentation representation,
                                                           std::uint64_t bound) {
  if (!IsValid(representation)) {
    return std::unexpected(StateError::kInvalid);
  }
  const std::optional<Bytes> allowance = StateAllowance(representation, bound);
  if (!allowance) {
    return std::unexpected(StateError::kExceedsBound);
  }
  return StateCursor(std::move(representation), bound, *allowance);
}

std::uint64_t StateCursor::blocks() const {
  // written_ <= bound_, whose blocks were counted without overflow.
  return BlocksFor(representation_, written_).value_or(0);
}

std::expected<void, StateError> StateCursor::Append(std::uint64_t positions) {
  if (!representation_.Can(StateCapability::kAppend)) {
    return std::unexpected(StateError::kUnsupported);
  }
  if (positions > room()) {
    return std::unexpected(StateError::kExceedsBound);  // before anything is allocated
  }
  written_ += positions;
  return {};
}

std::expected<void, StateError> StateCursor::Commit(std::uint64_t positions) {
  if (positions > written_ - committed_) {
    return std::unexpected(StateError::kBeyondWritten);
  }
  committed_ += positions;
  std::erase_if(snapshots_, [&](std::uint64_t at) { return at < committed_; });
  return {};
}

std::expected<void, StateError> StateCursor::Truncate(std::uint64_t position) {
  if (position < committed_) {
    return std::unexpected(StateError::kBelowCommitted);
  }
  if (position > written_) {
    return std::unexpected(StateError::kBeyondWritten);
  }
  if (position == written_) {
    return {};
  }
  if (!representation_.Can(StateCapability::kTruncate)) {
    if (!representation_.Can(StateCapability::kTruncateAtSnapshot)) {
      return std::unexpected(StateError::kUnsupported);
    }
    if (!std::ranges::binary_search(snapshots_, position)) {
      return std::unexpected(StateError::kNoSnapshot);
    }
  }
  written_ = position;
  std::erase_if(snapshots_, [&](std::uint64_t at) { return at > written_; });
  return {};
}

std::expected<void, StateError> StateCursor::Snapshot() {
  if (!representation_.Can(StateCapability::kTruncateAtSnapshot)) {
    return std::unexpected(StateError::kUnsupported);
  }
  if (!snapshots_.empty() && snapshots_.back() == written_) {
    return {};
  }
  if (snapshots_.size() >= representation_.max_snapshots) {
    return std::unexpected(StateError::kSnapshotsFull);
  }
  snapshots_.push_back(written_);
  return {};
}

}  // namespace llmp::model
