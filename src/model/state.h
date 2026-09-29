// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// State representations and their capabilities, as a state adapter
// declares them (D-055, D-068; docs/architecture.md#adapters-and-request-programs),
// and one request's live state in one representation. Internal: the
// operation-contract decision (backend-proof P6) settles these types.
//
// A representation declares its backing block (positions per block and the
// block's physical bytes, provider granularity included), what it can do
// (append; truncate to any position, or only to a snapshot) and how many
// snapshots it may hold at once. Its allowance at a request's bound is what
// D-050's R(G) charges for it, fixed at admission.
//
// A StateCursor tracks one request's positions in that representation: the
// committed prefix, positions written tentatively beyond it (a verify of
// drafted tokens, a canvas commit in progress), and the admitted bound. It
// only keeps the books: the caller materializes and releases the blocks
// blocks() names, through the catalog. It refuses, before anything is
// allocated, an append past the bound (an exceeded envelope) and a
// truncation below the committed prefix or to a position the
// representation cannot restore. Truncation returns backing to the
// request's unused allowance; the allowance itself stays committed until
// the request retires (D-050).
//
// Nothing here names a model architecture.

#ifndef JITLLM_MODEL_STATE_H_
#define JITLLM_MODEL_STATE_H_

#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "base/bytes.h"

namespace jitllm::model {

using base::Bytes;

// A byte range inside one model-owned state region.
struct StateRange {
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
};

// What a representation can do, as a bit set.
enum class StateCapability : std::uint8_t {
  kAppend = 1U << 0U,
  // Truncate to any position at or above the committed prefix, such as a
  // paged KV cache after a rejected draft.
  kTruncate = 1U << 1U,
  // Truncate only to a position a snapshot was taken at, such as recurrent
  // state restored from a per-step snapshot.
  kTruncateAtSnapshot = 1U << 2U,
};

constexpr std::uint8_t operator|(StateCapability a, StateCapability b) {
  return static_cast<std::uint8_t>(static_cast<std::uint8_t>(a) | static_cast<std::uint8_t>(b));
}

struct StateRepresentation {
  std::string name;
  // Positions per backing block; 0 means one fixed-size block whatever the
  // position count (recurrent state).
  std::uint64_t block_positions = 0;
  Bytes block_bytes;  // a block's physical backing
  std::uint8_t capabilities = 0;
  std::uint32_t max_snapshots = 0;  // held at once, for kTruncateAtSnapshot
  Bytes snapshot_bytes;             // each snapshot's physical backing

  bool Can(StateCapability capability) const {
    return (capabilities & static_cast<std::uint8_t>(capability)) != 0;
  }
  // Truncation of either kind: what a rejected draft needs.
  bool CanTruncate() const {
    return Can(StateCapability::kTruncate) || Can(StateCapability::kTruncateAtSnapshot);
  }
};

// Whether a representation is well formed: it has a block, and snapshot
// bytes exactly when it may hold snapshots, which only a representation
// truncated at snapshots does.
bool IsValid(const StateRepresentation& representation);

// Blocks that `positions` positions occupy.
std::optional<std::uint64_t> BlocksFor(const StateRepresentation& representation,
                                       std::uint64_t positions);

// The representation's allowance through `positions`: its blocks and every
// snapshot it may hold, with checked arithmetic. Empty on overflow.
std::optional<Bytes> StateAllowance(const StateRepresentation& representation,
                                    std::uint64_t positions);

enum class StateError : std::uint8_t {
  kInvalid,         // a representation with no block, or snapshots without their bytes
  kUnsupported,     // the representation lacks the capability
  kExceedsBound,    // past the admitted bound: an exceeded envelope
  kBelowCommitted,  // would discard committed positions
  kBeyondWritten,   // commits or truncates to positions never written
  kNoSnapshot,      // a truncation target without a snapshot
  kSnapshotsFull,
};

std::string ToString(StateError error);

class StateCursor {
 public:
  // Live state in `representation` for a request admitted through `bound`
  // positions, with nothing written yet.
  static std::expected<StateCursor, StateError> Create(StateRepresentation representation,
                                                       std::uint64_t bound);

  // Writes `positions` more tentatively.
  std::expected<void, StateError> Append(std::uint64_t positions);
  // The first `positions` tentative positions become committed: accepted
  // drafts, a committed canvas. Snapshots below the new prefix are dropped.
  std::expected<void, StateError> Commit(std::uint64_t positions);
  // Discards every position from `position` on. Never below the committed
  // prefix; with kTruncateAtSnapshot, only to a snapshot (or a no-op).
  std::expected<void, StateError> Truncate(std::uint64_t position);
  // Records a snapshot at the written position (kTruncateAtSnapshot).
  std::expected<void, StateError> Snapshot();

  const StateRepresentation& representation() const { return representation_; }
  std::uint64_t committed() const { return committed_; }
  std::uint64_t written() const { return written_; }
  std::uint64_t bound() const { return bound_; }
  // Positions that may still be written before the bound.
  std::uint64_t room() const { return bound_ - written_; }
  // The blocks the written positions occupy.
  std::uint64_t blocks() const;
  std::span<const std::uint64_t> snapshots() const { return snapshots_; }
  // What R(G) charges for this state: StateAllowance at the bound.
  Bytes allowance() const { return allowance_; }

 private:
  StateCursor(StateRepresentation representation, std::uint64_t bound, Bytes allowance)
      : representation_(std::move(representation)), bound_(bound), allowance_(allowance) {}

  StateRepresentation representation_;
  std::uint64_t bound_ = 0;
  Bytes allowance_;
  std::uint64_t committed_ = 0;
  std::uint64_t written_ = 0;
  std::vector<std::uint64_t> snapshots_;  // ascending, each in [committed, written]
};

}  // namespace jitllm::model

#endif  // JITLLM_MODEL_STATE_H_
