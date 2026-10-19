// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_ENGINE_STEP_AHEAD_H_
#define LLMP_ENGINE_STEP_AHEAD_H_

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <utility>

#include "engine/paged_node.h"
#include "execution/step_shape.h"

namespace llmp::engine {

// Optional immutable owned predictions. Counted storage replaces a fixed
// candidate cap. Copies share one payment; no caller can reset its receipt
// independently of live shapes or retain a borrowed request span.
class StepAhead {
 public:
  StepAhead() = default;
  StepAhead(const StepAhead& other)
      : build(other.build),
        capture_ahead(other.capture_ahead),
        prepare(other.prepare),
        inventory_(other.inventory_) {
    if (inventory_) inventory_->references.fetch_add(1, std::memory_order_relaxed);
  }
  StepAhead(StepAhead&& other) noexcept
      : build(std::exchange(other.build, 0)),
        capture_ahead(std::exchange(other.capture_ahead, false)),
        prepare(std::exchange(other.prepare, false)),
        inventory_(std::exchange(other.inventory_, nullptr)) {}
  StepAhead& operator=(const StepAhead& other) {
    StepAhead replacement(other);
    Swap(replacement);
    return *this;
  }
  StepAhead& operator=(StepAhead&& other) noexcept {
    StepAhead replacement(std::move(other));
    Swap(replacement);
    return *this;
  }
  ~StepAhead() {
    if (inventory_ && inventory_->references.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      auto payment = std::move(inventory_->payment);
      auto* owned = inventory_->data();
      for (std::size_t i = 0; i < inventory_->count; ++i) owned[i].~optional();
      inventory_->~Inventory();
      ::operator delete(inventory_);
      payment.Reset();
    }
  }
  std::span<const std::optional<execution::StepShape>> shapes() const {
    return inventory_ ? std::span(inventory_->data(), inventory_->count)
                      : std::span<const std::optional<execution::StepShape>>{};
  }
  std::uint64_t paid_bytes() const { return inventory_ ? inventory_->payment.bytes() : 0; }
  static std::expected<StepAhead, std::string> Create(
      PagedNode& node, std::span<const std::optional<execution::StepShape>> shapes,
      std::uint32_t builds = 0, bool capture = false, bool preparation = false) {
    StepAhead result;
    if (shapes.empty()) return result;
    constexpr std::uint64_t kAllocatorMetadata = 64;
    std::uint64_t array_bytes = 0, allocated = 0, charge = 0;
    if (__builtin_mul_overflow(std::uint64_t{shapes.size()},
                               std::uint64_t{sizeof(std::optional<execution::StepShape>)},
                               &array_bytes) ||
        __builtin_add_overflow(std::uint64_t{sizeof(Inventory)}, array_bytes, &allocated) ||
        allocated > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
        __builtin_add_overflow(allocated, kAllocatorMetadata, &charge))
      return std::unexpected("prediction inventory overflows");
    for (const auto& shape : shapes) {
      if (!shape || shape->segments.empty()) continue;
      std::uint64_t bytes = 0;
      if (__builtin_mul_overflow(std::uint64_t{shape->segments.size()},
                                 std::uint64_t{sizeof(execution::Segment)}, &bytes) ||
          __builtin_add_overflow(bytes, kAllocatorMetadata, &bytes) ||
          __builtin_add_overflow(charge, bytes, &charge))
        return std::unexpected("prediction segment inventory overflows");
    }
    auto payment = node.PrepayHost(charge);
    if (!payment) return std::unexpected(payment.error());
    void* allocation = ::operator new(static_cast<std::size_t>(allocated), std::nothrow);
    if (!allocation) return std::unexpected("allocating paid prediction inventory failed");
    result.inventory_ = new (allocation) Inventory(std::move(*payment), shapes.size());
    for (std::size_t i = 0; i < shapes.size(); ++i)
      new (result.inventory_->data() + i) std::optional<execution::StepShape>(shapes[i]);
    result.build = static_cast<std::uint32_t>(std::min<std::size_t>(builds, shapes.size()));
    result.capture_ahead = capture;
    result.prepare = preparation;
    return result;
  }
  // Actual optional compiler grants requested, independent of candidate count.
  // The transaction may admit fewer; no work is promised by this value.
  std::uint32_t build = 0;
  bool capture_ahead = false, prepare = false;

 private:
  struct alignas(std::max_align_t) Inventory {
    Inventory(HostPayment paid, std::size_t size) : payment(std::move(paid)), count(size) {}
    std::optional<execution::StepShape>* data() {
      return reinterpret_cast<std::optional<execution::StepShape>*>(this + 1);
    }
    std::atomic<std::size_t> references{1};
    HostPayment payment;
    std::size_t count = 0;
  };
  static_assert(alignof(Inventory) >= alignof(std::optional<execution::StepShape>));
  static_assert(sizeof(Inventory) % alignof(std::optional<execution::StepShape>) == 0);
  void Swap(StepAhead& other) noexcept {
    std::swap(inventory_, other.inventory_);
    std::swap(build, other.build);
    std::swap(capture_ahead, other.capture_ahead);
    std::swap(prepare, other.prepare);
  }
  Inventory* inventory_ = nullptr;
};
}  // namespace llmp::engine
#endif  // LLMP_ENGINE_STEP_AHEAD_H_
