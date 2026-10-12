// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A selected routed dependency inside one placed step. This describes actual
// source/read edges, not residency permission. Only ModelRunner mints access
// after its checked prefix completion and selected closure admission.
#ifndef LLMP_KERNELS_COMMON_ROUTED_PROGRAM_H_
#define LLMP_KERNELS_COMMON_ROUTED_PROGRAM_H_
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "kernels/common/grouped_worklist.h"
#include "kernels/common/step_program.h"

namespace llmp::engine {
class ModelRunner;
}
namespace llmp::kernels {
struct RoutedCopy {
  std::uint64_t device = 0, host_offset = 0, bytes = 0;
  // Readback only: each device row goes to the next packed host row. A
  // contiguous range retains the single-copy default; uploads stay contiguous.
  std::uint64_t rows = 1, device_stride = 0;
};
struct RoutedLayout {
  GroupedRouteShape shape;
  // The five complete input spans; outputs remain in this same pinned owner.
  std::uint64_t ids = 0, offsets = 0, position = 0, token = 0, expert = 0;
  std::uint64_t work_offsets = 0, work_tokens = 0, work_slots = 0, seen = 0;
  std::uint64_t host_bytes = 0, host_offset = 0;
  std::vector<RoutedCopy> readback;
};
// Sequence bookkeeping only. The caller supplies completion after its actual
// completion certificate; unknown work deliberately leaves awaiting() true.
// Recording checks Ready without Submit and cannot advance this protocol.
class RoutedProgress {
 public:
  bool Start(std::uint64_t epoch, std::size_t groups) {
    if (epoch == 0 || groups == 0 || groups == SIZE_MAX || epoch <= epoch_ || awaiting_)
      return false;
    epoch_ = epoch;
    groups_ = groups;
    next_ = 0;
    prepared_ = false;
    return true;
  }
  bool Ready(std::uint64_t epoch, std::size_t segment) const {
    return epoch == epoch_ && segment == next_ && segment <= groups_ && !awaiting_ &&
           (segment == 0 || prepared_);
  }
  bool Submit(std::uint64_t epoch, std::size_t segment) {
    if (!Ready(epoch, segment)) return false;
    awaiting_ = true;
    return true;
  }
  bool Complete(std::uint64_t epoch, std::size_t segment) {
    if (epoch != epoch_ || segment != next_ || !awaiting_) return false;
    awaiting_ = false;
    prepared_ = false;
    ++next_;
    return true;
  }
  bool CanPrepare(std::uint64_t epoch, std::size_t group) const {
    return epoch == epoch_ && group < groups_ && next_ == group + 1 && !awaiting_ && !prepared_;
  }
  bool Prepared(std::uint64_t epoch, std::size_t group) {
    if (!CanPrepare(epoch, group)) return false;
    prepared_ = true;
    return true;
  }
  std::uint64_t epoch() const { return epoch_; }
  bool awaiting() const { return awaiting_; }

 private:
  std::uint64_t epoch_ = 0;
  std::size_t groups_ = 0, next_ = 0;
  bool awaiting_ = false, prepared_ = false;
};

class RoutedProgram;
class RoutedAccess final {
 public:
  bool Names(const RoutedProgram* owner, std::uint64_t epoch) const {
    return owner_ == owner && epoch_ == epoch && epoch != 0;
  }
  std::uint64_t epoch() const { return epoch_; }
  std::size_t segment() const { return segment_; }

 private:
  friend class llmp::engine::ModelRunner;
  RoutedAccess(const RoutedProgram* owner, std::uint64_t epoch, std::size_t segment = 0)
      : owner_(owner), epoch_(epoch), segment_(segment) {}
  const RoutedProgram* owner_;
  std::uint64_t epoch_;
  std::size_t segment_;
};
// Deliberately not a Recorded: ordinary StepLaunch::Launch cannot replay it.
class RoutedRecording {
 public:
  virtual ~RoutedRecording() = default;
  virtual const RoutedProgram* owner() const = 0;
  virtual std::size_t segment() const = 0;
  virtual std::size_t nodes() const = 0;
  virtual double capture_seconds() const = 0;
  virtual double instantiate_seconds() const = 0;
  virtual std::expected<void, StepFailure> Replay(const RoutedAccess&, StepLaunch&,
                                                  providers::NativeStream) const = 0;
};
class RoutedProgram {
 public:
  virtual ~RoutedProgram() = default;
  virtual std::span<const RoutedWeight> weights() const = 0;
  virtual std::size_t groups() const = 0;
  virtual const RoutedLayout& layout(std::size_t group = 0) const = 0;
  virtual std::span<const std::size_t> members(std::size_t group) const = 0;
  virtual std::uint64_t pinned_bytes() const = 0;
  virtual const base::Sha256Digest& routing_identity() const = 0;
  // Source graph + immutable backend execution identity are additionally
  // checked by the runner's source/cache identity and exact weight inventory.
  virtual const base::Sha256Digest& graph_identity() const = 0;
  // Prefix runs/captures only the original graph prefix and route preparation.
  // Input/readback copies are supplied by the runner in body. Every entry checks
  // current backend Bind generation; continuation additionally needs Prepare.
  virtual std::expected<void, StepFailure> RunSegment(const RoutedAccess&, StepLaunch&,
                                                      providers::NativeStream) = 0;
  // Called only by ModelRunner after positive job and page-in retirement.
  virtual std::expected<void, StepFailure> CompleteSegment(const RoutedAccess&) = 0;
  virtual std::expected<void, StepFailure> Prepare(const RoutedAccess&,
                                                   std::span<std::byte> pinned) = 0;
  virtual std::expected<std::unique_ptr<RoutedRecording>, StepFailure> RecordSegment(
      const RoutedAccess&, StepLaunch&, providers::NativeStream, const StepLaunch::Body& body) = 0;
  virtual std::expected<void, StepFailure> ReplaySegment(const RoutedAccess&, StepLaunch&,
                                                         providers::NativeStream,
                                                         const RoutedRecording&) = 0;
};
}  // namespace llmp::kernels
#endif  // LLMP_KERNELS_COMMON_ROUTED_PROGRAM_H_
