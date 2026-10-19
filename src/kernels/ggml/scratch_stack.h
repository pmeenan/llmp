// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_KERNELS_GGML_SCRATCH_STACK_H_
#define LLMP_KERNELS_GGML_SCRATCH_STACK_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace llmp::kernels::ggml {

// Fixed, caller-paid launch metadata. Every nonempty entry has a distinct
// 256-byte start; interior dead entries also count until the top is released.
// This bounds capacity by the operation's actual maximum scratch, independent
// of allocation order or sizes. Empty allocations consume no descriptor.
class ScratchStack {
 public:
  struct Block {
    std::uint64_t offset, end;
    bool live;
  };
  static std::uint64_t Capacity(std::uint64_t bytes) { return bytes / 256 + (bytes % 256 != 0); }
  explicit ScratchStack(std::span<Block> storage = {}) : storage_(storage) {}
  std::optional<std::uint64_t> Allocate(std::uint64_t size, std::uint64_t limit) {
    if (size == 0) return 0;
    if (top_ > UINT64_MAX - 255) return std::nullopt;
    const auto offset = ((top_ + 255) / 256) * 256;
    if (offset > limit || size > limit - offset || count_ == storage_.size()) return std::nullopt;
    storage_[count_++] = {.offset = offset, .end = offset + size, .live = true};
    top_ = offset + size;
    peak_ = std::max(peak_, top_);
    return offset;
  }
  bool Free(std::uint64_t offset, std::uint64_t size) {
    if (size == 0) return true;
    bool found = false;
    for (auto& block : storage_.first(count_)) {
      if (block.live && block.offset == offset && block.end - offset == size) {
        block.live = false;
        found = true;
        break;
      }
    }
    if (!found) return false;
    while (count_ && !storage_[count_ - 1].live) --count_;
    top_ = count_ ? storage_[count_ - 1].end : 0;
    return true;
  }
  bool empty() const { return count_ == 0; }
  std::uint64_t peak() const { return peak_; }
  void ResetPeak() { peak_ = top_; }

 private:
  std::span<Block> storage_;
  std::size_t count_ = 0;
  std::uint64_t top_ = 0, peak_ = 0;
};
}  // namespace llmp::kernels::ggml
#endif  // LLMP_KERNELS_GGML_SCRATCH_STACK_H_
