// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_ENGINE_COPY_SINK_H_
#define LLMP_ENGINE_COPY_SINK_H_

#include <cstddef>
#include <cstdint>
#include <span>

#include "engine/run_stats.h"

namespace llmp::engine {

// The same copy emitter serves ordinary vectors and a batch's fixed, already
// paid span. The default sink counts without allocation or touching addresses.
// Coalescing preserves a row sequence's original boundaries when requested.
class CopySink {
 public:
  CopySink() = default;
  explicit CopySink(Copies& storage) : vector_(&storage), count_(storage.size()) {
    if (count_) last_ = storage.back();
  }
  explicit CopySink(std::span<RunCopy> storage) : storage_(storage), fixed_(true) {}
  bool Push(RunCopy copy, bool coalesce = false) {
    if (coalesce && count_ != 0 && last_[2] <= UINT64_MAX - last_[0] &&
        last_[2] <= UINT64_MAX - last_[1] && last_[0] + last_[2] == copy[0] &&
        last_[1] + last_[2] == copy[1] && copy[2] <= UINT64_MAX - last_[2]) {
      last_[2] += copy[2];
      if (vector_)
        vector_->back() = last_;
      else if (fixed_)
        storage_[count_ - 1] = last_;
      return true;
    }
    if (count_ == SIZE_MAX || (fixed_ && count_ == storage_.size())) return false;
    if (vector_)
      vector_->push_back(copy);
    else if (fixed_)
      storage_[count_] = copy;
    last_ = copy;
    ++count_;
    return true;
  }
  std::size_t size() const { return count_; }

 private:
  Copies* vector_ = nullptr;
  std::span<RunCopy> storage_;
  RunCopy last_{};
  std::size_t count_ = 0;
  bool fixed_ = false;
};
}  // namespace llmp::engine
#endif  // LLMP_ENGINE_COPY_SINK_H_
