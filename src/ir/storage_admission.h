// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Paid ownership for optional comparison-only graph construction. This
// interface deliberately depends on no execution or device types.

#ifndef LLMP_IR_STORAGE_ADMISSION_H_
#define LLMP_IR_STORAGE_ADMISSION_H_

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

namespace llmp::ir {

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
