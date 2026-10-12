// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "import/route.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <string_view>
#include <utility>

namespace llmp::import {
namespace {
std::unexpected<Error> Bad(std::string_view reason) {
  return std::unexpected(Error{.rule = Rule::kBounds, .reason = reason});
}
bool Overlap(std::span<const std::byte> a, std::span<const std::byte> b) {
  if (a.empty() || b.empty()) return false;
  const std::less<const std::byte*> less;
  return less(a.data(), b.data() + b.size()) && less(b.data(), a.data() + a.size());
}
}  // namespace

std::expected<PartRoutePlan, Error> PartRoutePlan::Copy(std::uint64_t bytes) {
  if (bytes == 0) return Bad("a part route must cover positive bytes");
  PartRoutePlan out;
  out.input_bytes_ = out.output_bytes_ = bytes;
  return out;
}
std::expected<PartRoutePlan, Error> PartRoutePlan::Permute(const HeadPermutationPlan& plan,
                                                           format::PartRole part) {
  auto cursor = plan.CursorFor(part);
  if (!cursor) return std::unexpected(cursor.error());
  const auto found = std::ranges::find(plan.parts(), part, &format::PartExtent::role);
  if (found == plan.parts().end() || found->bytes == 0)
    return Bad("a permutation route lacks its complete part");
  PartRoutePlan out;
  out.input_bytes_ = out.output_bytes_ = found->bytes;
  out.permutation_ = *cursor;
  return out;
}
std::expected<PartRoutePlan, Error> PartRoutePlan::Fold(ScalarFold fold, std::uint64_t elements) {
  if ((fold != ScalarFold::kNormPlusOne && fold != ScalarFold::kANegExp) || elements == 0 ||
      elements > std::numeric_limits<std::uint64_t>::max() / 4)
    return Bad("invalid scalar route or byte product");
  PartRoutePlan out;
  out.input_bytes_ = elements * 2;
  out.output_bytes_ = elements * 4;
  out.fold_ = fold;
  return out;
}
std::expected<PartRouteExecution, Error> PartRoutePlan::Start(
    std::span<std::byte> workspace) const {
  if (fold_ && workspace.size() < 4) return Bad("scalar route workspace was not provided");
  PartRouteExecution out;
  out.input_bytes_ = input_bytes_;
  out.output_bytes_ = output_bytes_;
  out.permutation_ = permutation_;
  out.fold_ = fold_;
  if (fold_) out.workspace_ = workspace.first(std::min<std::size_t>(workspace.size(), 65536));
  return out;
}
PartRouteExecution::PartRouteExecution(PartRouteExecution&& other) noexcept {
  *this = std::move(other);
}
PartRouteExecution& PartRouteExecution::operator=(PartRouteExecution&& other) noexcept {
  // A sink can hold this cursor's address. Never replace the bounds or
  // progress underneath its active Consume, even with a fresh source.
  if (busy_) {
    failed_ = true;
    return *this;
  }
  if (this == &other) return *this;
  input_bytes_ = other.input_bytes_;
  output_bytes_ = other.output_bytes_;
  received_ = other.received_;
  acknowledged_ = other.acknowledged_;
  permutation_ = other.permutation_;
  fold_ = other.fold_;
  workspace_ = other.workspace_;
  carry_ = other.carry_;
  carried_ = other.carried_;
  failed_ = other.failed_ || other.busy_;
  busy_ = false;
  other.failed_ = true;
  other.workspace_ = {};
  return *this;
}
std::expected<void, Error> PartRouteExecution::Consume(std::uint64_t offset,
                                                       std::span<const std::byte> input,
                                                       const PartSink& sink) {
  if (failed_ || busy_) {
    failed_ = true;
    return Bad("part route is cancelled, moved, busy or failed");
  }
  if (!sink || offset != received_ || received_ > input_bytes_ ||
      input.size() > input_bytes_ - received_ || Overlap(input, workspace_)) {
    failed_ = true;
    return Bad("part route input is not the next disjoint bounded span");
  }
  // Poison while inside a callback as well: a callback cannot reenter this
  // cursor, acknowledge it early or recover an ambiguous partial write.
  busy_ = true;
  bool complete = false;
  struct Guard {
    bool& busy;
    bool& failed;
    bool& complete;
    ~Guard() {
      busy = false;
      if (!complete) failed = true;
    }
  } guard{busy_, failed_, complete};
  const auto emit = [&](std::uint64_t at, std::span<const std::byte> bytes) {
    auto result = sink(at, bytes);
    if (!result) return result;
    if (failed_) return std::expected<void, Error>(Bad("part route cancelled during its sink"));
    return result;
  };
  while (!input.empty()) {
    if (!fold_) {
      CopyRun run{.source = received_, .destination = received_, .bytes = input.size()};
      if (permutation_) {
        auto next = permutation_->Next(input.size());
        if (!next) return std::unexpected(next.error());
        if (!*next || (**next).source != received_ || (**next).bytes == 0 ||
            (**next).bytes > input.size())
          return Bad("permutation cursor lost source coverage");
        run = **next;
      }
      const auto piece = input.first(static_cast<std::size_t>(run.bytes));
      if (auto ok = emit(run.destination, piece); !ok) return ok;
      received_ += run.bytes;
      acknowledged_ += run.bytes;
      input = input.subspan(piece.size());
      continue;
    }
    if (carried_) {
      carry_[1] = input.front();
      auto output = workspace_.first(4);
      if (auto ok = FoldBf16(*fold_, carry_, output); !ok) return ok;
      if (auto ok = emit(acknowledged_, output); !ok) return ok;
      ++received_;
      acknowledged_ += 4;
      carried_ = false;
      input = input.subspan(1);
      continue;
    }
    const auto count = std::min(input.size(), workspace_.size() / 2) & ~std::size_t{1};
    if (count == 0) {
      carry_[0] = input.front();
      carried_ = true;
      ++received_;
      input = input.subspan(1);
      continue;
    }
    auto output = workspace_.first(count * 2);
    if (auto ok = FoldBf16(*fold_, input.first(count), output); !ok) return ok;
    if (auto ok = emit(acknowledged_, output); !ok) return ok;
    received_ += count;
    acknowledged_ += output.size();
    input = input.subspan(count);
  }
  complete = true;
  return {};
}
std::expected<void, Error> PartRouteExecution::Finish() const {
  if (failed_ || busy_ || carried_ || received_ != input_bytes_ || acknowledged_ != output_bytes_)
    return Bad("part route has not acknowledged complete input and output");
  return {};
}
}  // namespace llmp::import
