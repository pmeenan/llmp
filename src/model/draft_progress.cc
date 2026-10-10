// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "model/draft_progress.h"

#include <cstddef>
#include <limits>

namespace llmp::model {
namespace {

bool ValidRelation(DraftRelation relation) {
  return relation == DraftRelation::kTokenAtCache || relation == DraftRelation::kTokenAfterCache;
}

std::uint64_t Ahead(DraftRelation relation) {
  return relation == DraftRelation::kTokenAfterCache ? 1 : 0;
}

std::unexpected<std::string> Refused(const char* what) {
  return std::unexpected(std::string(what));
}

}  // namespace

bool ValidDraftPolicy(DraftPolicy policy) {
  return ValidRelation(policy.relation) &&
         (policy.progress == DraftProgressPolicy::kRepeatPending ||
          (policy.progress == DraftProgressPolicy::kCursor &&
           policy.relation == DraftRelation::kTokenAtCache));
}

std::expected<void, std::string> CheckDraftProgress(const DraftProgress& p, DraftProgressBounds b) {
  if (!ValidRelation(b.relation) || b.context == 0 || b.max_rows == 0 || b.max_rows > b.context ||
      p.positions > b.context) {
    return Refused("invalid drafter progress bounds");
  }
  if (p.positions == 0) {
    if (p.pending_first || p.tap_first || p.tap_rows || p.preceding)
      return Refused("nonempty drafter progress at zero positions");
    return {};
  }
  const std::uint64_t a = Ahead(b.relation);
  const std::uint64_t end = std::uint64_t{p.positions} + 1 - a;
  const std::uint64_t first = p.pending_first;
  if (first + a == 0 || first >= end || end - first > b.max_rows)
    return Refused("pending cache rows outside replay bound");
  if (p.tap_rows == 0 || p.tap_rows > b.max_rows ||
      std::uint64_t{p.tap_first} + p.tap_rows != p.positions || p.preceding > b.carry ||
      p.preceding > p.tap_first)
    return Refused("invalid retained feature span");
  const std::uint64_t feature_first = first + a - 1;
  if (feature_first < std::uint64_t{p.tap_first} - p.preceding)
    return Refused("pending rows require absent features");
  return {};
}

std::expected<DraftProgressWords, std::string> EncodeDraftProgress(const DraftProgress& p,
                                                                   DraftProgressBounds b) {
  if (auto checked = CheckDraftProgress(p, b); !checked) return std::unexpected(checked.error());
  if (p.positions == 0) return DraftProgressWords{};
  return DraftProgressWords{1,
                            p.positions,
                            p.pending_first,
                            p.tap_first,
                            p.tap_rows,
                            p.preceding,
                            static_cast<std::int64_t>(b.relation),
                            0};
}

std::expected<DraftProgress, std::string> DecodeDraftProgress(const DraftProgressWords& words,
                                                              std::uint32_t positions,
                                                              DraftProgressBounds b) {
  if (positions == 0) {
    for (const auto word : words)
      if (word != 0) return Refused("nonzero empty drafter control");
    DraftProgress p;
    if (auto checked = CheckDraftProgress(p, b); !checked) return std::unexpected(checked.error());
    return p;
  }
  if (words[0] != 1 || words[6] != static_cast<std::int64_t>(b.relation) || words[7] != 0)
    return Refused("drafter control version, relation or reserved field differs");
  for (std::size_t i = 1; i <= 5; ++i) {
    if (words[i] < 0 || std::uint64_t(words[i]) > std::numeric_limits<std::uint32_t>::max())
      return Refused("drafter control coordinate cannot be represented");
  }
  DraftProgress p{.positions = static_cast<std::uint32_t>(words[1]),
                  .pending_first = static_cast<std::uint32_t>(words[2]),
                  .tap_first = static_cast<std::uint32_t>(words[3]),
                  .tap_rows = static_cast<std::uint32_t>(words[4]),
                  .preceding = static_cast<std::uint32_t>(words[5])};
  if (p.positions != positions) return Refused("drafter control position differs from restore");
  if (auto checked = CheckDraftProgress(p, b); !checked) return std::unexpected(checked.error());
  return p;
}

std::expected<DraftRows, std::string> DraftFirstPass(const DraftProgress& p,
                                                     DraftProgressBounds b) {
  if (auto checked = CheckDraftProgress(p, b); !checked) return std::unexpected(checked.error());
  if (p.positions == 0) return Refused("empty target has no feature");
  if (p.positions == b.context) return Refused("target has no room for an anchor");
  const std::uint64_t a = Ahead(b.relation);
  const std::uint64_t end = std::uint64_t{p.positions} + 1 - a;
  if (end > b.context) return Refused("draft cache rows pass context");
  return DraftRows{.cache_first = p.pending_first,
                   .token_first = static_cast<std::uint32_t>(p.pending_first + a),
                   .feature_first = static_cast<std::uint32_t>(p.pending_first + a - 1),
                   .count = static_cast<std::uint32_t>(end - p.pending_first)};
}

std::expected<DraftCompanions, std::string> DraftPrefillCompanions(const DraftProgress& p,
                                                                   DraftProgressBounds b,
                                                                   std::uint32_t rows) {
  if (auto checked = CheckDraftProgress(p, b); !checked) return std::unexpected(checked.error());
  if (rows == 0 || rows > b.max_rows || rows > b.context - p.positions)
    return Refused("target chunk outside drafter bounds");
  if (p.positions != 0 && b.carry == 0)
    return Refused("nonempty prompt companion requires tap carry");
  const std::uint64_t a = Ahead(b.relation);
  const std::uint64_t n = p.positions;
  DraftCompanions c;
  if (n != 0) {
    const std::uint64_t count = n - a - p.pending_first;
    if (count != 0) {
      c.before = {.cache_first = p.pending_first,
                  .token_first = static_cast<std::uint32_t>(p.pending_first + a),
                  .feature_first = static_cast<std::uint32_t>(p.pending_first + a - 1),
                  .count = static_cast<std::uint32_t>(count)};
    }
    c.after = {.cache_first = static_cast<std::uint32_t>(n - a),
               .token_first = p.positions,
               .feature_first = p.positions - 1,
               .count = rows};
  } else if (rows > 1) {
    c.after = {.cache_first = static_cast<std::uint32_t>(1 - a),
               .token_first = 1,
               .feature_first = 0,
               .count = rows - 1};
  }
  c.final_pending_first = static_cast<std::uint32_t>(n + rows - a);
  return c;
}

std::uint32_t DraftCheckpointFirst(std::uint32_t positions, std::uint32_t max_rows) {
  return positions > max_rows ? positions - max_rows : 0;
}

}  // namespace llmp::model
