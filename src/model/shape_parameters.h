// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef LLMP_MODEL_SHAPE_PARAMETERS_H_
#define LLMP_MODEL_SHAPE_PARAMETERS_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <span>

namespace llmp::model {

// A canonical geometry value and the first coordinate where that value changes.
// Absence proves invariance over the addressable coordinate range. Providers
// derive both from the same calculation used by graph construction.
struct ShapeParameterFact {
  std::uint64_t value = 0;
  std::optional<std::uint64_t> next{};
};

// One canonical emitter supports a no-allocation count and an exact fixed-span
// fill. It never grows storage. Neither operation grants state access.
class ShapeParameterSink {
 public:
  enum class Error : std::uint8_t { kOverflow, kExtent };
  struct Observation {
    bool complete = true;
    std::optional<std::uint32_t> next{};
    std::uint64_t word = 0;
  };
  static ShapeParameterSink Observe(std::uint32_t after, std::uint32_t through) {
    ShapeParameterSink sink;
    sink.observing_ = true;
    sink.after_ = after;
    sink.through_ = through;
    if (after >= through) sink.error_ = Error::kExtent;
    return sink;
  }
  std::expected<Observation, Error> Next() const {
    if (error_) return std::unexpected(*error_);
    if (!observing_) return std::unexpected(Error::kExtent);
    return observed_;
  }
  ShapeParameterSink() = default;  // count only
  explicit ShapeParameterSink(std::span<std::uint64_t> into) : into_(into), filling_(true) {}
  void push_back(std::uint64_t word) {
    if (observing_) observed_.complete = false;
    Push(word);
  }
  void push_back(std::uint64_t word, std::optional<std::uint64_t> next) {
    if (observing_ && !error_) {
      if (next && *next <= after_) {
        error_ = Error::kExtent;
        return;
      }
      if (next && *next <= through_ && (!observed_.next || *next < *observed_.next)) {
        observed_.next = static_cast<std::uint32_t>(*next);
        observed_.word = count_;
      }
    }
    Push(word);
  }
  void push_back(const ShapeParameterFact& fact) { push_back(fact.value, fact.next); }
  std::expected<std::size_t, Error> Finish() const {
    if (error_) return std::unexpected(*error_);
    if (filling_ && count_ != into_.size()) return std::unexpected(Error::kExtent);
    return count_;
  }

 private:
  void Push(std::uint64_t word) {
    if (error_) return;
    if (count_ == std::numeric_limits<std::size_t>::max()) {
      error_ = Error::kOverflow;
      return;
    }
    if (filling_) {
      if (count_ == into_.size()) {
        error_ = Error::kExtent;
        return;
      }
      into_[count_] = word;
    }
    ++count_;
  }
  std::span<std::uint64_t> into_;
  std::size_t count_ = 0;
  bool filling_ = false, observing_ = false;
  std::uint32_t after_ = 0, through_ = 0;
  Observation observed_;
  std::optional<Error> error_;
};
}  // namespace llmp::model
#endif  // LLMP_MODEL_SHAPE_PARAMETERS_H_
