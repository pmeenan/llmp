// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_ENGINE_SPECIALIZATION_H_
#define LLMP_ENGINE_SPECIALIZATION_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "execution/step_shape.h"
#include "model/shape_parameters.h"

namespace llmp::engine {

// The intermediates a stage's verify stores for its commit, per owner.
struct CommitBuffer {
  std::uint32_t layer = 0;
  std::uint16_t component = 0;
  std::string what;
  std::uint64_t row_bytes = 0;
};

// Stage-level execution facts exchanged between a step source and the model
// runner at setup, without family-named source inspection methods.
struct StageExecutionFacts {
  bool uses_side_rows = false;
  bool joined_repeat_pending = false;
  std::function<std::expected<std::vector<CommitBuffer>, std::string>(std::uint32_t max_rows)>
      commit_buffers;
};

// Pure shape facts, not contents, residency, a mutable ledger or execution
// authority. The immutable source owns the provider/state geometry revision.
struct SpecializationRequest {
  const execution::StepShape& shape;
  std::size_t owner = 0;
  // Bounds are candidate past coordinates. Through leaves rows within context,
  // so the inclusive latest past is context - rows, not context.
  std::uint32_t after = 0, through = 0, context = 0;
};
struct SpecializationBoundary {
  std::uint32_t position = 0;
  std::uint64_t fact = 0;  // opaque source/provider reason, never an address
};
struct SpecializationChange {
  enum class Coverage : std::uint8_t { kUnknown, kComplete };
  Coverage coverage = Coverage::kUnknown;
  std::optional<SpecializationBoundary> next{};
};
// Calls allocate nothing and inspect no mutable state. A complete answer uses
// the same facts as the source's canonical shape-parameter emitter, including
// its read-geometry dependencies. Unknown never means position invariant.
enum class SpecializationError : std::uint8_t { kRequest, kOverflow, kProvider };
class SpecializationQuery {
 public:
  virtual ~SpecializationQuery() = default;
  virtual std::expected<SpecializationChange, SpecializationError> NextSpecializationChange(
      const SpecializationRequest&) const {
    return SpecializationChange{};
  }
};

// Shape-level execution facts exchanged per step shape: canonical baked
// dimensions, position-dependent specialization boundaries, and side-row
// derivation from token history.
class ShapeExecutionFacts : public SpecializationQuery {
 public:
  // Canonical baked graph dimensions. The same allocation-free emitter counts
  // before storage admission, then fills an exact paid span. It must be pure.
  virtual void EmitShapeParameters(const execution::StepShape&, model::ShapeParameterSink&) const {}
  std::expected<std::size_t, model::ShapeParameterSink::Error> CountShapeParameters(
      const execution::StepShape& shape) const {
    model::ShapeParameterSink sink;
    EmitShapeParameters(shape, sink);
    return sink.Finish();
  }
  std::expected<std::size_t, model::ShapeParameterSink::Error> FillShapeParameters(
      const execution::StepShape& shape, std::span<std::uint64_t> into) const {
    model::ShapeParameterSink sink(into);
    EmitShapeParameters(shape, sink);
    return sink.Finish();
  }
  // A side-table component's row IDs from the owner's complete token
  // history. Pure host work before any state admission or device writes.
  virtual std::expected<void, std::string> SideRows(std::uint16_t /*stage*/,
                                                    std::uint16_t /*component*/,
                                                    std::span<const std::int32_t> /*history*/,
                                                    std::uint32_t /*position*/,
                                                    std::span<std::int32_t> /*out*/) const {
    return std::unexpected("the step source cannot derive side rows");
  }
};

inline bool ValidSpecializationRequest(const SpecializationRequest& request) {
  return request.owner < request.shape.segments.size() &&
         request.shape.segments[request.owner].past == request.after &&
         request.after < request.through && request.through <= request.context &&
         request.shape.segments[request.owner].rows != 0 &&
         request.shape.segments[request.owner].rows <= request.context - request.through;
}
inline bool ValidSpecializationChange(const SpecializationRequest& request,
                                      const SpecializationChange& change) {
  return ValidSpecializationRequest(request) &&
         (change.coverage == SpecializationChange::Coverage::kUnknown ||
          change.coverage == SpecializationChange::Coverage::kComplete) &&
         (change.coverage == SpecializationChange::Coverage::kComplete || !change.next) &&
         (!change.next ||
          (change.next->position > request.after && change.next->position <= request.through));
}
}  // namespace llmp::engine
#endif  // LLMP_ENGINE_SPECIALIZATION_H_
