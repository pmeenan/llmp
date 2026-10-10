// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A drafter's declared token/cache/feature coordinates and its durable
// repeatable first-pass schedule. A physical catch-up does not advance
// this schedule: only a completed target transition publishes a new one.

#ifndef LLMP_MODEL_DRAFT_PROGRESS_H_
#define LLMP_MODEL_DRAFT_PROGRESS_H_

#include <array>
#include <cstdint>
#include <expected>
#include <string>

namespace llmp::model {

enum class DraftRelation : std::uint8_t {
  kTokenAtCache,     // cache q consumes token q and target feature q-1
  kTokenAfterCache,  // cache q consumes token q+1 and target feature q
};

enum class DraftProgressPolicy : std::uint8_t {
  kCursor,         // the existing transient, token-aligned cursor
  kRepeatPending,  // replay the same first-pass rows until target acceptance
};

struct DraftPolicy {
  DraftRelation relation = DraftRelation::kTokenAtCache;
  DraftProgressPolicy progress = DraftProgressPolicy::kCursor;
  bool operator==(const DraftPolicy&) const = default;
};

bool ValidDraftPolicy(DraftPolicy policy);

// Exactly eight dense I64 values in the ordinary state tensor, not a
// numeric KV encoding. Word 0 is schema 1, 1..5 are Progress's fields,
// 6 is the trusted relation's ID and 7 is reserved zero. Empty state has
// one representation: all zero. No process identity is serialized.
using DraftProgressWords = std::array<std::int64_t, 8>;
inline constexpr std::uint64_t kDraftProgressBytes = 64;
static_assert(sizeof(DraftProgressWords) == kDraftProgressBytes);

struct DraftProgress {
  std::uint32_t positions = 0;
  std::uint32_t pending_first = 0;
  std::uint32_t tap_first = 0;
  std::uint32_t tap_rows = 0;
  std::uint32_t preceding = 0;
  bool operator==(const DraftProgress&) const = default;
};

// Derived from the actual stage layout, never from restored words. The
// layout also qualifies its private cache for replay of completed rows.
struct DraftProgressBounds {
  std::uint32_t context = 0;
  std::uint32_t max_rows = 0;
  std::uint32_t carry = 0;
  DraftRelation relation = DraftRelation::kTokenAtCache;
};

std::expected<void, std::string> CheckDraftProgress(const DraftProgress& progress,
                                                    DraftProgressBounds bounds);
std::expected<DraftProgressWords, std::string> EncodeDraftProgress(const DraftProgress& progress,
                                                                   DraftProgressBounds bounds);
std::expected<DraftProgress, std::string> DecodeDraftProgress(const DraftProgressWords& words,
                                                              std::uint32_t positions,
                                                              DraftProgressBounds bounds);

struct DraftRows {
  std::uint32_t cache_first = 0;
  std::uint32_t token_first = 0;
  std::uint32_t feature_first = 0;
  std::uint32_t count = 0;
  bool operator==(const DraftRows&) const = default;
};

// The complete first pass, including the known anchor token t[N]. This
// does not publish progress or qualify a completion witness.
std::expected<DraftRows, std::string> DraftFirstPass(const DraftProgress& progress,
                                                     DraftProgressBounds bounds);

struct DraftCompanions {
  DraftRows before;
  DraftRows after;
  // Usable only after the whole pre/target/post sequence and its control
  // copy retire. A completed `before` alone leaves old progress unchanged.
  std::uint32_t final_pending_first = 0;
};

// Shapes of the two headless companions around a target prompt chunk.
// The caller acquires compound ownership before `before`, keeps the old
// record through the whole sequence, and supplies authoritative history.
std::expected<DraftCompanions, std::string> DraftPrefillCompanions(const DraftProgress& progress,
                                                                   DraftProgressBounds bounds,
                                                                   std::uint32_t rows);

// A trusted layout bound for checkpoint capture and restore, before an
// unloaded control can be read. A completed record's pending rows fit it.
std::uint32_t DraftCheckpointFirst(std::uint32_t positions, std::uint32_t max_rows);

}  // namespace llmp::model

#endif  // LLMP_MODEL_DRAFT_PROGRESS_H_
