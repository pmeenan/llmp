// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_ENGINE_HOST_BUFFER_H_
#define LLMP_ENGINE_HOST_BUFFER_H_

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <iterator>
#include <limits>
#include <new>
#include <span>
#include <string>
#include <utility>

#include "base/check.h"
#include "engine/paged_node.h"

namespace llmp::engine {

// Immutable owned bytes. Copies retain one paid allocation; no staging or node
// pointer escapes. Its intrusive control header is destroyed with the payload,
// before returning the payment, even on a consumer thread.
class HostBuffer {
 public:
  HostBuffer() = default;
  HostBuffer(const HostBuffer& other)
      : owner_(other.owner_), offset_(other.offset_), size_(other.size_) {
    if (owner_) owner_->references.fetch_add(1, std::memory_order_relaxed);
  }
  HostBuffer& operator=(const HostBuffer& other) {
    HostBuffer replacement(other);
    Swap(replacement);
    return *this;
  }
  HostBuffer(HostBuffer&& other) noexcept
      : owner_(std::exchange(other.owner_, nullptr)),
        offset_(std::exchange(other.offset_, 0)),
        size_(std::exchange(other.size_, 0)) {}
  HostBuffer& operator=(HostBuffer&& other) noexcept {
    HostBuffer replacement(std::move(other));
    Swap(replacement);
    return *this;
  }
  ~HostBuffer() {
    if (owner_ && owner_->references.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      auto payment = std::move(owner_->payment);
      owner_->~Backing();
      ::operator delete(owner_);
      payment.Reset();
    }
  }
  // Caller input becomes immutable owned storage before the caller's source
  // lifetime ends. Payload, control header and allocator metadata are prepaid.
  static std::expected<HostBuffer, std::string> Copy(PagedNode& node,
                                                     std::span<const std::byte> source) {
    auto buffer = Allocate(node, source.size());
    if (!buffer) return std::unexpected(buffer.error());
    if (!source.empty()) std::memcpy(buffer->Writable().data(), source.data(), source.size());
    return buffer;
  }
  HostBuffer Slice(std::size_t offset, std::size_t bytes) const {
    base::Check(offset <= size_ && bytes <= size_ - offset, "host result slice is in bounds");
    HostBuffer view = *this;
    view.offset_ += offset;
    view.size_ = bytes;
    return view;
  }
  const std::byte* data() const {
    return owner_ ? reinterpret_cast<const std::byte*>(owner_ + 1) + offset_ : nullptr;
  }
  std::size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  const std::byte* begin() const { return data(); }
  const std::byte* end() const { return size_ == 0 ? data() : data() + size_; }
  // The backing's one prepaid receipt, shared by every slice (not additive).
  std::uint64_t paid_bytes() const { return owner_ ? owner_->payment.bytes() : 0; }
  friend bool operator==(const HostBuffer& a, const HostBuffer& b) {
    return std::ranges::equal(a, b);
  }

 private:
  friend class StepTransaction;
  friend class StagedRunner;
  friend class HostOutputs;
  struct alignas(std::max_align_t) Backing {
    explicit Backing(HostPayment payment_in) : payment(std::move(payment_in)) {}
    std::atomic<std::size_t> references{1};
    HostPayment payment;
  };
  static std::expected<HostBuffer, std::string> Allocate(PagedNode& node, std::uint64_t bytes,
                                                         bool* over_budget = nullptr) {
    if (over_budget) *over_budget = false;
    if (bytes == 0) return HostBuffer{};
    // One allocation: actual control header, exact payload, named malloc header.
    constexpr std::uint64_t kAllocationMetadataBytes = 64;
    std::uint64_t allocated = 0, charge = 0;
    if (bytes > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
        __builtin_add_overflow(bytes, std::uint64_t{sizeof(Backing)}, &allocated) ||
        allocated > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
        __builtin_add_overflow(allocated, kAllocationMetadataBytes, &charge))
      return std::unexpected("owned host result size overflows");
    auto paid = node.PrepayHost(charge);
    if (!paid) {
      if (over_budget) *over_budget = true;
      return std::unexpected(paid.error());
    }
    void* allocation = ::operator new(static_cast<std::size_t>(allocated), std::nothrow);
    if (!allocation) {
      if (over_budget) *over_budget = true;
      return std::unexpected("allocating paid host result bytes failed");
    }
    HostBuffer buffer;
    buffer.owner_ = new (allocation) Backing(std::move(*paid));
    buffer.size_ = static_cast<std::size_t>(bytes);
    return buffer;
  }
  void Swap(HostBuffer& other) noexcept {
    std::swap(owner_, other.owner_);
    std::swap(offset_, other.offset_);
    std::swap(size_, other.size_);
  }
  std::span<std::byte> Writable() { return {const_cast<std::byte*>(data()), size_}; }
  Backing* owner_ = nullptr;
  std::size_t offset_ = 0, size_ = 0;
};

// A result's segment descriptors and bytes have one private lifetime/payment.
// Indexing yields an owning immutable byte view, so a caller may retain any
// segment independently. A completely empty result needs no allocation.
class HostOutputs {
 public:
  HostOutputs() = default;
  HostOutputs(const HostOutputs&) = default;
  HostOutputs& operator=(const HostOutputs&) = default;
  HostOutputs(HostOutputs&& other) noexcept
      : storage_(std::move(other.storage_)), count_(std::exchange(other.count_, 0)) {}
  HostOutputs& operator=(HostOutputs&& other) noexcept {
    if (this != &other) {
      storage_ = std::move(other.storage_);
      count_ = std::exchange(other.count_, 0);
    }
    return *this;
  }
  // Public construction returns initialized immutable bytes. Only a producing
  // transaction can allocate uninitialized backing, retained privately until
  // all returned output bytes have been copied after proven retirement.
  static std::expected<HostOutputs, std::string> Zeroed(PagedNode& node,
                                                        std::span<const std::uint64_t> segments,
                                                        std::uint64_t joined) {
    auto result = Allocate(node, segments, joined);
    if (!result) return result;
    for (std::size_t i = 0; i < result->size(); ++i)
      std::ranges::fill(result->at(i).Writable(), std::byte{});
    std::ranges::fill(result->joined().Writable(), std::byte{});
    return result;
  }
  std::size_t size() const { return count_; }
  bool empty() const { return count_ == 0; }
  HostBuffer at(std::size_t i) const {
    base::Check(i < count_, "host output segment exists");
    return View(i);
  }
  HostBuffer operator[](std::size_t i) const { return at(i); }
  HostBuffer front() const { return at(0); }
  HostBuffer joined() const { return View(count_); }
  class Iterator {
   public:
    using value_type = HostBuffer;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::input_iterator_tag;
    Iterator() = default;
    HostBuffer operator*() const { return owner_->at(index_); }
    Iterator& operator++() {
      ++index_;
      return *this;
    }
    Iterator operator++(int) {
      auto old = *this;
      ++*this;
      return old;
    }
    friend bool operator==(const Iterator&, const Iterator&) = default;

   private:
    friend class HostOutputs;
    Iterator(const HostOutputs* owner, std::size_t index) : owner_(owner), index_(index) {}
    const HostOutputs* owner_ = nullptr;
    std::size_t index_ = 0;
  };
  Iterator begin() const { return Iterator(this, 0); }
  Iterator end() const { return Iterator(this, count_); }

 private:
  friend class StepTransaction;
  friend class StagedRunner;
  static std::expected<HostOutputs, std::string> Allocate(PagedNode& node,
                                                          std::span<const std::uint64_t> segments,
                                                          std::uint64_t joined,
                                                          bool* over_budget = nullptr) {
    if (over_budget) *over_budget = false;
    HostOutputs result;
    result.count_ = segments.size();
    if (joined == 0 && std::ranges::all_of(segments, [](auto n) { return n == 0; })) return result;
    std::uint64_t descriptors = 0, total = 0;
    if (segments.size() == SIZE_MAX ||
        __builtin_mul_overflow(std::uint64_t{segments.size() + 1},
                               std::uint64_t{sizeof(Descriptor)}, &descriptors))
      return std::unexpected("host output descriptor inventory overflows");
    total = descriptors;
    auto append = [&](std::uint64_t bytes) {
      return Align(total) && !__builtin_add_overflow(total, bytes, &total);
    };
    for (auto bytes : segments)
      if (!append(bytes)) return std::unexpected("host output payload inventory overflows");
    if (!append(joined)) return std::unexpected("host output joined inventory overflows");
    auto storage = HostBuffer::Allocate(node, total, over_budget);
    if (!storage) return std::unexpected(storage.error());
    result.storage_ = std::move(*storage);
    std::uint64_t at = descriptors;
    auto describe = [&](std::size_t i, std::uint64_t bytes) {
      (void)Align(at);  // The identical checked inventory above proved every addition.
      const Descriptor descriptor{at, bytes};
      std::memcpy(result.storage_.Writable().data() + i * sizeof(Descriptor), &descriptor,
                  sizeof(descriptor));
      at += bytes;
    };
    for (std::size_t i = 0; i < segments.size(); ++i) describe(i, segments[i]);
    describe(segments.size(), joined);
    return result;
  }
  struct Descriptor {
    std::uint64_t offset, bytes;
  };
  static bool Align(std::uint64_t& value) {
    constexpr std::uint64_t kAlignment = alignof(std::max_align_t);
    std::uint64_t adjusted = 0;
    if (__builtin_add_overflow(value, kAlignment - 1, &adjusted)) return false;
    value = adjusted / kAlignment * kAlignment;
    return true;
  }
  HostBuffer View(std::size_t index) const {
    if (storage_.empty()) return {};
    Descriptor descriptor{};
    std::memcpy(&descriptor, storage_.data() + index * sizeof(Descriptor), sizeof(descriptor));
    return storage_.Slice(descriptor.offset, descriptor.bytes);
  }
  HostBuffer storage_;
  std::size_t count_ = 0;
};

}  // namespace llmp::engine

#endif  // LLMP_ENGINE_HOST_BUFFER_H_
