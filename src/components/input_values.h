// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_COMPONENTS_INPUT_VALUES_H_
#define LLMP_COMPONENTS_INPUT_VALUES_H_
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace llmp::components {
// Allocation-free count, fixed typed byte fill or the ordinary owning vector
// adapter for one canonical emitter. A fixed span is never silently widened.
class InputValueSink {
 public:
  InputValueSink() = default;
  explicit InputValueSink(std::vector<std::int64_t>& values) : vector_(&values) {}
  InputValueSink(std::span<std::byte> into, std::size_t width, bool padded = false)
      : into_(into), width_(width), fixed_(true), padded_(padded) {
    good_ = (width == sizeof(std::int32_t) || width == sizeof(std::int64_t)) &&
            into.size() % width == 0;
  }
  using Emit = bool (*)(void*, std::size_t, std::int64_t);
  InputValueSink(void* context, Emit emit) : context_(context), emit_(emit) {}
  void push_back(std::int64_t value) {
    if (!good_ || count_ == SIZE_MAX) {
      good_ = false;
      return;
    }
    if (fixed_) {
      if (count_ >= into_.size() / width_) {
        good_ = false;
        return;
      }
      if (width_ == sizeof(std::int32_t)) {
        if (value < INT32_MIN || value > INT32_MAX) {
          good_ = false;
          return;
        }
        const auto narrow = static_cast<std::int32_t>(value);
        std::memcpy(into_.data() + count_ * width_, &narrow, width_);
      } else {
        std::memcpy(into_.data() + count_ * width_, &value, width_);
      }
    } else if (vector_)
      vector_->push_back(value);
    else if (emit_ && !emit_(context_, count_, value)) {
      good_ = false;
      return;
    }
    ++count_;
  }
  std::expected<void, std::string> Finish() const {
    if (!good_ || (fixed_ && !padded_ && count_ != into_.size() / width_))
      return std::unexpected("input values exceed their exact typed/count storage");
    if (fixed_ && padded_ && count_ * width_ < into_.size())
      std::memset(into_.data() + count_ * width_, 0, into_.size() - count_ * width_);
    return {};
  }
  std::size_t size() const { return count_; }

 private:
  std::vector<std::int64_t>* vector_ = nullptr;
  std::span<std::byte> into_;
  std::size_t width_ = 0, count_ = 0;
  void* context_ = nullptr;
  Emit emit_ = nullptr;
  bool fixed_ = false, good_ = true, padded_ = false;
};
}  // namespace llmp::components
#endif  // LLMP_COMPONENTS_INPUT_VALUES_H_
