// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/step_inputs.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <limits>

#include "components/step.h"
#include "engine/support.h"

namespace llmp::engine::step_inputs {
namespace kn = llmp::kernels;
namespace pl = llmp::planner;
using support::Error;

// The I32 values of one segment's rows (or every segment's, joined) for a
// tokens or positions input, written into `out` and the rest zeroed.
std::expected<void, std::string> WriteRows(std::span<std::byte> out,
                                           std::span<const std::int32_t> values) {
  const std::uint64_t bytes = values.size_bytes();
  if (bytes > out.size()) return Error("an input's rows exceed its slot");
  if (bytes != 0) std::memcpy(out.data(), values.data(), bytes);
  std::fill(out.begin() + static_cast<std::ptrdiff_t>(bytes), out.end(), std::byte{0});
  return {};
}

// The values of an input the step's shape decides (step_program.h
// kOutputRows, kStoreCells, kMask), by components/step.h InputValues' rules.
std::expected<void, std::string> WriteComputed(const kn::InputSlot& in, const pl::StepShape& shape,
                                               std::span<std::byte> out) {
  namespace co = llmp::components;
  const co::CellKey cells{.capacity = in.cells.capacity,
                          .ring = in.cells.ring,
                          .frozen = in.cells.frozen,
                          .fixed = in.cells.fixed,
                          .block = in.cells.block,
                          .per_row = in.cells.per_row,
                          .first_row = in.cells.first_row,
                          .positions = in.cells.positions,
                          .recent = in.cells.recent};
  co::StepInput input{.segment = in.segment,
                      .cells = cells,
                      .read = in.read,
                      .ratio = in.ratio,
                      .overlap = in.overlap,
                      .ring = in.ring,
                      .visible_capacity = in.visible_capacity};
  std::size_t width = sizeof(std::int32_t);
  switch (in.kind) {
    case kn::InputKind::kHadamard:
      return co::WriteHadamard(in.read, out);
    case kn::InputKind::kVisibleCells:
      input.kind = co::InputKind::kVisibleCells;
      break;
    case kn::InputKind::kPositionPhases:
      input.kind = co::InputKind::kPositionPhases;
      break;
    case kn::InputKind::kKeptRows:
      input.kind = co::InputKind::kKeptRows;
      break;
    case kn::InputKind::kKeptCells:
      input.kind = co::InputKind::kKeptCells;
      width = sizeof(std::int64_t);
      break;
    case kn::InputKind::kPoolReads:
      input.kind = co::InputKind::kPoolReads;
      break;
    case kn::InputKind::kBlockCells:
      input.kind = co::InputKind::kBlockCells;
      width = sizeof(std::int64_t);
      break;
    case kn::InputKind::kBlockStarts:
      input.kind = co::InputKind::kBlockStarts;
      break;
    case kn::InputKind::kOutputRows:
      input.kind = co::InputKind::kOutputRows;
      break;
    case kn::InputKind::kBlockPositions:
      input.kind = co::InputKind::kBlockPositions;
      input.sections = static_cast<std::uint8_t>(in.sections);
      break;
    case kn::InputKind::kStoreCells:
      if (in.outputs != 0 &&
          (in.segment >= shape.segments.size() || in.outputs != shape.segments[in.segment].outputs))
        return Error("a tap's cells name no segment's outputs");
      input.kind = in.outputs != 0 ? co::InputKind::kOutputCells : co::InputKind::kStoreCells;
      width = sizeof(std::int64_t);
      break;
    case kn::InputKind::kMask: {
      if (in.segment >= shape.segments.size()) return Error("a mask names no segment");
      const auto& segment = shape.segments[in.segment];
      std::uint64_t output_cells = 0, bytes = 0;
      if (in.mask_rows < segment.rows ||
          __builtin_mul_overflow(in.read, std::uint64_t{in.mask_rows}, &output_cells) ||
          __builtin_mul_overflow(output_cells, std::uint64_t{sizeof(std::uint16_t)}, &bytes) ||
          bytes != out.size())
        return Error("a mask's slot does not hold its cells and rows");
      constexpr std::uint16_t kHidden = 0xFC00;
      for (std::size_t i = 0; i < out.size(); i += sizeof(kHidden))
        std::memcpy(out.data() + i, &kHidden, sizeof(kHidden));
      struct Mask {
        const kn::InputSlot& slot;
        const pl::Segment& segment;
        std::span<std::byte> into;
      } mask{in, segment, out};
      co::InputValueSink sink(&mask, [](void* context, std::size_t cell, std::int64_t position) {
        auto& m = *static_cast<Mask*>(context);
        if (cell >= m.slot.read) return false;
        for (std::uint32_t row = 0; row < m.segment.rows; ++row) {
          const auto at = std::int64_t{m.segment.past} + row;
          const bool seen = position >= 0 && (m.slot.ahead || position <= at) &&
                            (m.slot.window == 0 || at - position < std::int64_t{m.slot.window});
          if (seen) {
            const std::uint16_t value = 0;
            std::memcpy(m.into.data() + (std::uint64_t{row} * m.slot.read + cell) * sizeof(value),
                        &value, sizeof(value));
          }
        }
        return true;
      });
      input.kind = co::InputKind::kCellPositions;
      if (auto emitted = co::EmitInputValues(input, shape, {}, sink); !emitted) return emitted;
      if (sink.size() != in.read) return Error("a mask has a different cell inventory");
      return {};
    }
    default:
      return Error("not a computed input");
  }
  co::InputValueSink sink(out, width,
                          width == sizeof(std::int32_t) || in.kind == kn::InputKind::kStoreCells);
  return co::EmitInputValues(input, shape, {}, sink);
}

}  // namespace llmp::engine::step_inputs
