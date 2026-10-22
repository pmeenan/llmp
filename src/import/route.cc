// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0
#include "import/route.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <string_view>
#include <utility>

#include "format/encoding.h"
#include "format/layout.h"

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
std::expected<PartRoutePlan, Error> PartRoutePlan::Fold(ScalarFold fold, std::uint64_t elements,
                                                        format::Encoding source_encoding) {
  if ((fold != ScalarFold::kNormPlusOne && fold != ScalarFold::kANegExp &&
       fold != ScalarFold::kIdentity) ||
      elements == 0 ||
      (source_encoding != format::Encoding::kBF16 && source_encoding != format::Encoding::kF16 &&
       source_encoding != format::Encoding::kF32) ||
      elements > std::numeric_limits<std::uint64_t>::max() / 4)
    return Bad("invalid scalar route or byte product");
  const std::uint64_t elem_size = (source_encoding == format::Encoding::kF32) ? 4 : 2;
  PartRoutePlan out;
  out.input_bytes_ = elements * elem_size;
  out.output_bytes_ = elements * 4;
  out.fold_ = fold;
  out.source_encoding_ = source_encoding;
  return out;
}
std::expected<PartRoutePlan, Error> PartRoutePlan::PermuteFold(const HeadPermutationPlan& plan,
                                                               format::PartRole part,
                                                               ScalarFold fold) {
  auto out = Permute(plan, part);
  if (!out) return out;
  const auto enc = plan.format().encoding;
  if ((fold != ScalarFold::kNormPlusOne && fold != ScalarFold::kANegExp &&
       fold != ScalarFold::kIdentity) ||
      (enc != format::Encoding::kBF16 && enc != format::Encoding::kF16 &&
       enc != format::Encoding::kF32))
    return Bad("a permuted fold needs a small whole BF16/F16/F32 part");
  const std::uint64_t elem_size = (enc == format::Encoding::kF32) ? 4 : 2;
  if (out->input_bytes_ % elem_size != 0 || out->input_bytes_ > elem_size * 8192)
    return Bad("a permuted fold needs a small whole BF16/F16/F32 part");
  out->output_bytes_ = (out->input_bytes_ / elem_size) * 4;
  out->fold_ = fold;
  out->source_encoding_ = enc;
  return out;
}
std::expected<PartRoutePlan, Error> PartRoutePlan::ScaleAtoms(std::uint64_t rows,
                                                              std::uint64_t blocks) {
  if (rows == 0 || rows % format::kScaleAtomRows != 0 || blocks == 0 ||
      blocks % format::kScaleAtomBlocks != 0 || blocks > 256 ||
      rows > std::numeric_limits<std::uint64_t>::max() / blocks)
    return Bad("scale atoms need whole 128-row atoms of 4 to 256 scales a row");
  PartRoutePlan out;
  out.input_bytes_ = out.output_bytes_ = rows * blocks;
  out.atom_blocks_ = blocks;
  return out;
}
std::expected<PartRoutePlan, Error> PartRoutePlan::Strided(std::uint64_t rows,
                                                           std::uint64_t row_bytes,
                                                           std::uint64_t stride,
                                                           std::uint64_t lane) {
  if (rows == 0 || row_bytes == 0 || stride == 0 || lane > stride || row_bytes > stride - lane ||
      rows > std::numeric_limits<std::uint64_t>::max() / stride)
    return Bad("a strided route needs whole rows within their stride");
  PartRoutePlan out;
  out.input_bytes_ = out.output_bytes_ = rows * row_bytes;
  out.row_bytes_ = row_bytes;
  out.stride_ = stride;
  out.lane_ = lane;
  return out;
}
std::expected<PartRouteExecution, Error> PartRoutePlan::Start(
    std::span<std::byte> workspace) const {
  if (fold_ && workspace.size() < 4) return Bad("scalar route workspace was not provided");
  if (fold_ && permutation_ && workspace.size() < input_bytes_ + output_bytes_)
    return Bad("permuted fold workspace was not provided");
  const auto band = format::kScaleAtomRows * atom_blocks_;
  if (atom_blocks_ != 0 && workspace.size() < 2 * band)
    return Bad("scale-atom route workspace was not provided");
  PartRouteExecution out;
  out.input_bytes_ = input_bytes_;
  out.output_bytes_ = output_bytes_;
  out.permutation_ = permutation_;
  out.fold_ = fold_;
  out.atom_blocks_ = atom_blocks_;
  out.row_bytes_ = row_bytes_;
  out.stride_ = stride_;
  out.lane_ = lane_;
  out.source_encoding_ = source_encoding_;
  if (fold_) out.workspace_ = workspace.first(std::min<std::size_t>(workspace.size(), 65536));
  if (atom_blocks_ != 0) out.workspace_ = workspace.first(static_cast<std::size_t>(2 * band));
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
  source_encoding_ = other.source_encoding_;
  atom_blocks_ = other.atom_blocks_;
  row_bytes_ = other.row_bytes_;
  stride_ = other.stride_;
  lane_ = other.lane_;
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
  if (stride_ != 0) {
    // Each row's received bytes at its lane of its output row.
    while (!input.empty()) {
      const auto row = received_ / row_bytes_, at = received_ % row_bytes_;
      const auto count =
          static_cast<std::size_t>(std::min<std::uint64_t>(input.size(), row_bytes_ - at));
      if (auto ok = emit((row * stride_) + lane_ + at, input.first(count)); !ok) return ok;
      received_ += count;
      acknowledged_ += count;
      input = input.subspan(count);
    }
    complete = true;
    return {};
  }
  if (permutation_ && fold_) {
    // The whole source, then each permuted run folded into its place.
    std::memcpy(workspace_.data() + received_, input.data(), input.size());
    received_ += input.size();
    if (received_ == input_bytes_) {
      const auto whole =
          std::span<const std::byte>(workspace_).first(static_cast<std::size_t>(input_bytes_));
      auto output = workspace_.subspan(static_cast<std::size_t>(input_bytes_),
                                       static_cast<std::size_t>(output_bytes_));
      const auto elem_size = (source_encoding_ == format::Encoding::kF32) ? 4U : 2U;
      for (std::uint64_t at = 0; at < input_bytes_;) {
        auto next = permutation_->Next(input_bytes_ - at);
        if (!next) return std::unexpected(next.error());
        if (!*next || (**next).source != at || (**next).bytes == 0 ||
            (**next).bytes % elem_size != 0 || (**next).destination % elem_size != 0 ||
            (**next).bytes > input_bytes_ - at ||
            (**next).destination > input_bytes_ - (**next).bytes)
          return Bad("permuted fold lost source coverage");
        const auto& run = **next;
        const std::uint64_t out_dest = (run.destination / elem_size) * 4;
        const std::uint64_t out_bytes = (run.bytes / elem_size) * 4;
        if (auto ok = FoldScalar(*fold_, source_encoding_,
                                 whole.subspan(static_cast<std::size_t>(run.source),
                                               static_cast<std::size_t>(run.bytes)),
                                 output.subspan(static_cast<std::size_t>(out_dest),
                                                static_cast<std::size_t>(out_bytes)));
            !ok)
          return ok;
        at += run.bytes;
      }
      if (auto ok = emit(0, output); !ok) return ok;
      acknowledged_ = output_bytes_;
    }
    complete = true;
    return {};
  }
  if (atom_blocks_ != 0) {
    // Rows into the first band buffer; a whole band swizzled into the
    // second and emitted at the band's own offset.
    const std::uint64_t band = format::kScaleAtomRows * atom_blocks_;
    auto rows = workspace_.first(static_cast<std::size_t>(band));
    auto atoms = workspace_.subspan(static_cast<std::size_t>(band));
    while (!input.empty()) {
      const auto at = received_ % band;
      const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(input.size(), band - at));
      std::memcpy(rows.data() + at, input.data(), count);
      received_ += count;
      input = input.subspan(count);
      if (received_ % band != 0) continue;
      for (std::uint64_t r = 0; r < format::kScaleAtomRows; ++r)
        for (std::uint64_t b = 0; b < atom_blocks_; ++b)
          atoms[static_cast<std::size_t>(format::ScaleAtomOffset(r, b, atom_blocks_))] =
              rows[static_cast<std::size_t>((r * atom_blocks_) + b)];
      if (auto ok = emit(acknowledged_, atoms); !ok) return ok;
      acknowledged_ += band;
    }
    complete = true;
    return {};
  }
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
    const auto elem_size = (source_encoding_ == format::Encoding::kF32) ? 4U : 2U;
    if (carried_ > 0) {
      const auto need = elem_size - carried_;
      if (input.size() < need) {
        std::memcpy(carry_.data() + carried_, input.data(), input.size());
        carried_ += static_cast<std::uint8_t>(input.size());
        received_ += input.size();
        input = {};
        continue;
      }
      std::memcpy(carry_.data() + carried_, input.data(), need);
      auto output = workspace_.first(4);
      if (auto ok = FoldScalar(*fold_, source_encoding_,
                               std::span<const std::byte>(carry_).first(elem_size), output);
          !ok)
        return ok;
      if (auto ok = emit(acknowledged_, output); !ok) return ok;
      received_ += need;
      acknowledged_ += 4;
      carried_ = 0;
      input = input.subspan(need);
      continue;
    }
    const auto max_elements = std::min(input.size() / elem_size, workspace_.size() / 4);
    const auto count = max_elements * elem_size;
    if (count == 0) {
      std::memcpy(carry_.data(), input.data(), input.size());
      carried_ = static_cast<std::uint8_t>(input.size());
      received_ += input.size();
      input = {};
      continue;
    }
    auto output = workspace_.first(max_elements * 4);
    if (auto ok = FoldScalar(*fold_, source_encoding_, input.first(count), output); !ok) return ok;
    if (auto ok = emit(acknowledged_, output); !ok) return ok;
    received_ += count;
    acknowledged_ += output.size();
    input = input.subspan(count);
  }
  complete = true;
  return {};
}
std::expected<void, Error> PartRouteExecution::Finish() const {
  if (failed_ || busy_ || carried_ != 0 || received_ != input_bytes_ ||
      acknowledged_ != output_bytes_)
    return Bad("part route has not acknowledged complete input and output");
  return {};
}
}  // namespace llmp::import
