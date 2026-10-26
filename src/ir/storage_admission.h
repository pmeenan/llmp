// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Paid ownership for optional comparison-only graph construction. This
// interface deliberately depends on no execution or device types.

#ifndef LLMP_IR_STORAGE_ADMISSION_H_
#define LLMP_IR_STORAGE_ADMISSION_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

namespace llmp::ir {

inline constexpr std::uint64_t kAdmissionHeapBlockOverhead =
    2 * sizeof(void*) + alignof(std::max_align_t);

constexpr bool AddAdmissionBytes(std::uint64_t& total, std::uint64_t amount) {
  return !__builtin_add_overflow(total, amount, &total);
}

constexpr bool MulAdmissionBytes(std::uint64_t a, std::uint64_t b, std::uint64_t& out) {
  return !__builtin_mul_overflow(a, b, &out);
}

constexpr std::optional<std::uint64_t> AlignAdmissionBytes(std::uint64_t bytes,
                                                           std::uint64_t alignment) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) return std::nullopt;
  std::uint64_t padded = 0;
  if (__builtin_add_overflow(bytes, alignment - 1, &padded)) return std::nullopt;
  return padded & ~(alignment - 1);
}

constexpr std::optional<std::uint64_t> VectorAdmissionBytes(
    std::uint64_t count, std::uint64_t stride,
    std::uint64_t overhead = kAdmissionHeapBlockOverhead) {
  if (count == 0) return std::uint64_t{0};
  std::uint64_t bytes = 0;
  if (stride == 0 || !MulAdmissionBytes(count, stride, bytes) ||
      !AddAdmissionBytes(bytes, overhead))
    return std::nullopt;
  return bytes;
}

constexpr bool AddVectorAdmissionBytes(std::uint64_t count, std::uint64_t stride,
                                       std::uint64_t& total,
                                       std::uint64_t overhead = kAdmissionHeapBlockOverhead) {
  const auto bytes = VectorAdmissionBytes(count, stride, overhead);
  return bytes.has_value() && AddAdmissionBytes(total, *bytes);
}

constexpr std::optional<std::uint64_t> StringAdmissionBytes(
    std::uint64_t length, std::uint64_t overhead = kAdmissionHeapBlockOverhead) {
  std::uint64_t bytes = length;
  if (!AddAdmissionBytes(bytes, 1) || !AddAdmissionBytes(bytes, overhead)) return std::nullopt;
  return bytes;
}

constexpr bool AddStringAdmissionBytes(std::uint64_t length, std::uint64_t& total,
                                       std::uint64_t overhead = kAdmissionHeapBlockOverhead) {
  const auto bytes = StringAdmissionBytes(length, overhead);
  return bytes.has_value() && AddAdmissionBytes(total, *bytes);
}

class StorageFunder {
 public:
  virtual ~StorageFunder() = default;
  virtual bool Charge(std::uint64_t bytes) const = 0;
  virtual void Release(std::uint64_t bytes) const = 0;
};

// Owns one prepaid allocation envelope. Moving a loan is allocation-free;
// its funder owner is retained until after the associated storage dies.
class StorageLoan {
 public:
  StorageLoan() = default;
  StorageLoan(StorageLoan&& other) noexcept
      : funder_(std::move(other.funder_)), bytes_(std::exchange(other.bytes_, 0)) {}
  StorageLoan& operator=(StorageLoan&& other) noexcept {
    if (this != &other) {
      Reset();
      funder_ = std::move(other.funder_);
      bytes_ = std::exchange(other.bytes_, 0);
    }
    return *this;
  }
  StorageLoan(const StorageLoan&) = delete;
  StorageLoan& operator=(const StorageLoan&) = delete;
  ~StorageLoan() { Reset(); }

  explicit operator bool() const { return bytes_ != 0; }
  std::shared_ptr<const StorageFunder> funder() const { return funder_; }
  std::uint64_t bytes() const { return bytes_; }
  // Extend a live high-water loan without releasing its existing charge.
  // The fixed owner and inline counter make this allocation-free.
  bool Grow(std::uint64_t additional) {
    if (!funder_ || additional > std::numeric_limits<std::uint64_t>::max() - bytes_) return false;
    if (additional != 0 && !funder_->Charge(additional)) return false;
    bytes_ += additional;
    return true;
  }
  // Bind `funder` if not yet bound and extend this loan to `target_bytes`
  // without releasing its existing charge.
  bool GrowTo(const std::shared_ptr<const StorageFunder>& funder, std::uint64_t target_bytes) {
    if (!funder || target_bytes < bytes_) return false;
    if (!funder_) {
      funder_ = funder;
    } else if (funder_.get() != funder.get()) {
      return false;
    }
    return Grow(target_bytes - bytes_);
  }
  void Reset() {
    if (bytes_ != 0) funder_->Release(std::exchange(bytes_, 0));
    funder_.reset();
  }

 private:
  friend class Builder;
  friend std::optional<StorageLoan> AcquireStorageLoan(
      const std::shared_ptr<const StorageFunder>& funder, std::uint64_t bytes);
  StorageLoan(std::shared_ptr<const StorageFunder> funder, std::uint64_t bytes)
      : funder_(std::move(funder)), bytes_(bytes) {}
  std::shared_ptr<const StorageFunder> funder_{};
  std::uint64_t bytes_ = 0;
};

inline std::optional<StorageLoan> AcquireStorageLoan(
    const std::shared_ptr<const StorageFunder>& funder, std::uint64_t bytes) {
  if (!funder) return std::nullopt;
  if (bytes != 0 && !funder->Charge(bytes)) return std::nullopt;
  return StorageLoan(funder, bytes);
}

}  // namespace llmp::ir

#endif  // LLMP_IR_STORAGE_ADMISSION_H_
