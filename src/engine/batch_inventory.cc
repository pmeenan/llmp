// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/batch_inventory.h"

#include <algorithm>
#include <limits>

#include "model/state_contract.h"

namespace llmp::engine {
namespace {
namespace pl = planner;
namespace kn = kernels;
constexpr std::uint64_t kAllocationMetadata = 64;
std::unexpected<kn::ResourceFailure> Refused(kn::ResourceError kind, std::string what) {
  return std::unexpected(kn::ResourceFailure{kind, std::move(what)});
}
bool AddStorage(std::uint64_t& bytes, std::uint64_t count, std::uint64_t size) {
  std::uint64_t added = 0;
  return !__builtin_mul_overflow(count, size, &added) &&
         !__builtin_add_overflow(bytes, added, &bytes) && bytes <= PTRDIFF_MAX;
}
}  // namespace

std::expected<PromptBatchInventory, kn::ResourceFailure> PromptBatchInventory::Preview(
    ModelRunner& runner, const WaveComposer& composer, std::span<const PendingWork> target) {
  auto root = runner.RetainedOwner();
  if (!root)
    return Refused(kn::ResourceError::kOwnership, "batch preview needs its retained runner");
  if (target.empty() || composer.stages() != runner.slots().stages() ||
      composer.stages() > std::uint64_t{UINT16_MAX} + 1)
    return Refused(kn::ResourceError::kBounds, "batch preview has no exact admitted owners/stages");
  auto identity = runner.ExecutionIdentity();
  if (!identity) return Refused(kn::ResourceError::kFault, identity.error());
  for (std::size_t s = 0; s < composer.stages(); ++s)
    if (composer.context(s) != runner.slots().context(s) ||
        composer.read_align(s) != runner.slots().read_align(s) ||
        composer.read_buckets(s) != runner.slots().read_buckets() ||
        composer.frozen(s) != (s != 0 && model::ReadsFrozen(runner.slots().layout(s))))
      return Refused(kn::ResourceError::kBounds, "batch composer differs from its source layouts");
  for (std::size_t i = 0; i < target.size(); ++i) {
    const auto& p = target[i];
    if (p.slot >= runner.slots().slots() || p.stage != 0 || p.rows == 0 ||
        (p.phase != pl::Phase::kPrefill && p.phase != pl::Phase::kDecode &&
         p.phase != pl::Phase::kScore) ||
        p.past != runner.slots().positions(p.slot) || !runner.slots().CheckActive(p.slot) ||
        std::ranges::any_of(target.first(i),
                            [&](const auto& prior) { return prior.slot == p.slot; }))
      return Refused(kn::ResourceError::kBounds, "batch target does not continue one active owner");
  }
  // Scratch capacity follows the actual simultaneous owner count. Composition
  // and all copies below reuse it; no stage/context-sized speculative loops.
  std::uint64_t scratch_bytes = 0, one_segments = 0;
  if (__builtin_mul_overflow(std::uint64_t{target.size()}, std::uint64_t{sizeof(pl::Segment)},
                             &one_segments) ||
      !AddStorage(scratch_bytes, target.size(),
                  sizeof(PendingWork) + sizeof(std::size_t) + sizeof(pl::StepShape) +
                      kAllocationMetadata) ||
      !AddStorage(scratch_bytes, target.size(), one_segments) ||
      !AddStorage(scratch_bytes, 4, kAllocationMetadata))
    return Refused(kn::ResourceError::kBounds, "batch composition storage overflows");
  auto scratch_payment = runner.PayHostMetadata(scratch_bytes);
  if (!scratch_payment) return Refused(kn::ResourceError::kCapacity, scratch_payment.error());
  std::vector<PendingWork> pending;
  pending.reserve(target.size());
  std::vector<std::size_t> order(target.size());
  std::vector<pl::StepShape> scratch(target.size());
  for (auto& shape : scratch) shape.segments.reserve(target.size());

  // Count/fill share this canonical composition traversal. Singleton shapes
  // are explicit failure alternatives, avoiding exponentially many subsets.
  const auto emit = [&](const auto& output) -> Status {
    const auto wave = [&]() -> Status {
      if (pending.empty()) return {};
      const auto count = composer.ComposePrediction(pending, scratch, order);
      if (!count) return std::unexpected("batch composition exceeds actual geometry/capacity");
      for (const auto& shape : std::span(scratch).first(*count))
        if (auto accepted = output(shape, false); !accepted) return accepted;
      if (*count == pending.size()) return {};  // already exact singleton alternatives
      // A joined failure may isolate owners without admitting an unplanned
      // subset. The caller switches its remaining work to these singleton
      // shapes; this preview itself grants no fallback execution permission.
      for (auto& p : pending) p.alone = true;
      const auto singles = composer.ComposePrediction(pending, scratch, order);
      if (!singles) return std::unexpected("batch singleton composition failed");
      for (const auto& shape : std::span(scratch).first(*singles))
        if (auto accepted = output(shape, true); !accepted) return accepted;
      return {};
    };
    pending.assign(target.begin(), target.end());
    if (auto accepted = wave(); !accepted) return accepted;
    return {};
  };
  std::uint64_t shapes = 0, segments = 0;
  const auto counted = emit([&](const pl::StepShape& shape, bool) -> Status {
    if (__builtin_add_overflow(shapes, std::uint64_t{1}, &shapes) ||
        __builtin_add_overflow(segments, std::uint64_t{shape.segments.size()}, &segments))
      return std::unexpected("batch shape count overflows");
    return {};
  });
  if (!counted) return Refused(kn::ResourceError::kBounds, counted.error());
  std::uint64_t bytes = 0;
  if (!AddStorage(bytes, shapes,
                  sizeof(pl::StepShape) + sizeof(std::uint8_t) + kAllocationMetadata) ||
      !AddStorage(bytes, segments, sizeof(pl::Segment)) ||
      !AddStorage(bytes, target.size(), sizeof(PendingWork)) ||
      !AddStorage(bytes, 3, kAllocationMetadata))
    return Refused(kn::ResourceError::kBounds, "batch immutable inventory overflows");
  auto paid = runner.PayHostMetadata(bytes);
  if (!paid) return Refused(kn::ResourceError::kCapacity, paid.error());
  PromptBatchInventory result;
  result.payment_ = std::move(*paid);
  result.owner_ = std::move(root);
  result.identity_ = *identity;
  result.shapes_.reserve(static_cast<std::size_t>(shapes));
  result.fallback_.reserve(static_cast<std::size_t>(shapes));
  result.target_.assign(target.begin(), target.end());
  const auto filled = emit([&](const pl::StepShape& shape, bool is_fallback) -> Status {
    if (result.shapes_.size() == shapes) return std::unexpected("batch count/fill differs");
    result.fallback_.push_back(is_fallback ? 1 : 0);
    result.shapes_.push_back(shape);
    return {};
  });
  if (!filled || result.shapes_.size() != shapes)
    return Refused(kn::ResourceError::kFault, "batch count/fill differs");
  if (auto current = result.Revalidate(); !current)
    return Refused(kn::ResourceError::kFault, current.error());
  return result;
}

Status PromptBatchInventory::CheckSource() const {
  if (!owner_ || !owner_->bound_ || owner_->released_ || owner_->closing_)
    return std::unexpected("batch preview owner is absent or draining");
  auto identity = owner_->ExecutionIdentity();
  if (!identity || *identity != identity_)
    return std::unexpected("batch preview source identity changed");
  return {};
}
Status PromptBatchInventory::Revalidate() const {
  if (auto source = CheckSource(); !source) return source;
  if (owner_->slots().faulted()) return std::unexpected("batch preview cohort faulted");
  for (const auto& target : target_)
    if (owner_->slots().positions(target.slot) != target.past ||
        !owner_->slots().CheckActive(target.slot))
      return std::unexpected("batch preview target changed before mint");
  return {};
}
}  // namespace llmp::engine
