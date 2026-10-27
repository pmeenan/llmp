// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The vendor-free view a kernel source has of a tensor another backend owns
// (D-053, D-110). The step backend that lowers a graph (today GGML's) makes
// the descriptors and places them; a selected source (the decoders, EXL3)
// reads them only through a TensorRef: an opaque handle plus the owning
// backend's reader, which reports the descriptor's facts at the moment they
// are read. Addresses are final only when a call is bound, so a source reads
// the facts again at Bind rather than keeping what Lower saw.
#ifndef LLMP_KERNELS_COMMON_TENSOR_REF_H_
#define LLMP_KERNELS_COMMON_TENSOR_REF_H_

#include <array>
#include <cstdint>
#include <optional>

namespace llmp::kernels {

// One element's (or block's) storage as the selected sources check it.
// kOther: anything else, which every source refuses.
enum class StorageType : std::uint8_t { kOther, kF32, kF16, kBF16, kI8, kI32, kQ8_0, kNvfp4 };

// What a backend-owned tensor is when read: extents (ne) and byte strides
// (nb) in the backend's dimension order, its address once placed, and the
// facts the sources check.
struct TensorFacts {
  StorageType type = StorageType::kOther;
  std::array<std::int64_t, 4> ne{};
  std::array<std::uint64_t, 4> nb{};
  void* data = nullptr;
  // The backend's size of the tensor: its last element's (or block row's)
  // end measured from data, with the backend's own (wrapping) arithmetic.
  std::uint64_t bytes = 0;
  std::uint64_t element_bytes = 0;  // one element, or one block of a blocked type
  std::uint64_t row_bytes = 0;      // one packed row of ne[0] elements
  std::int64_t elements = 0;
  // Its rows are packed (the backend's contiguity test).
  bool contiguous = false;
  // Every view in its chain still points where its source does.
  bool current = false;
  // Its bytes from the first element to one past the last, by checked
  // arithmetic: nothing for an empty tensor, a stride the backend treats as
  // negative, or an extent or end address that overflows.
  std::optional<std::uint64_t> extent{};
};

class TensorRef {
 public:
  using Reader = TensorFacts (*)(const void* handle);

  constexpr TensorRef() = default;
  constexpr TensorRef(const void* handle, Reader reader)
      : handle_(handle), reader_(handle != nullptr ? reader : nullptr) {}

  explicit operator bool() const { return handle_ != nullptr && reader_ != nullptr; }
  // Only on a non-empty reference.
  TensorFacts facts() const { return reader_(handle_); }
  // The owning backend's identity tag and handle: only that backend may
  // interpret the handle (it checks the reader is its own).
  Reader reader() const { return reader_; }
  const void* handle() const { return handle_; }

  // The same descriptor (not merely equal facts).
  friend bool operator==(const TensorRef& a, const TensorRef& b) {
    return a.handle_ == b.handle_ && a.reader_ == b.reader_;
  }

 private:
  const void* handle_ = nullptr;
  Reader reader_ = nullptr;
};

inline std::uint64_t AddressOf(const TensorFacts& facts) {
  return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(facts.data));
}

}  // namespace llmp::kernels

#endif  // LLMP_KERNELS_COMMON_TENSOR_REF_H_
