// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Typed, generation-checked identities (D-006, D-048): an index into a
// table and the generation the slot had when the identity was issued. A
// slot's generation advances whenever it is reused, so a stale identity
// never names the slot's new occupant. Generation 0 is never issued: a
// default identity names nothing.

#ifndef LLMP_BASE_IDS_H_
#define LLMP_BASE_IDS_H_

#include <compare>
#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "base/check.h"

namespace llmp::base {

template <typename Tag>
class Id {
 public:
  constexpr Id() = default;
  constexpr Id(std::uint32_t index, std::uint32_t generation)
      : index_(index), generation_(generation) {}

  constexpr std::uint32_t index() const { return index_; }
  constexpr std::uint32_t generation() const { return generation_; }
  constexpr bool valid() const { return generation_ != 0; }

  constexpr auto operator<=>(const Id&) const = default;

  std::string ToString() const { return std::format("{}#{}.{}", Tag::kName, index_, generation_); }

 private:
  std::uint32_t index_ = 0;
  std::uint32_t generation_ = 0;
};

// A table of slots addressed by Id<Tag>. Erasing a slot advances its
// generation; a slot whose generation would wrap is retired, never reused
// (D-048: exhaust the ID space by refusing, not by reuse).
//
// A default table grows as needed and reuses the most recently freed slot.
// A bounded table has all its slots from the start, allocates nothing
// more, and serves its free slots in rotation (the longest free first), so
// every slot's generation advances at the same pace: exhaustion comes
// after about capacity x 2^32 insertions, not one slot at a time from the
// first 2^32 (docs/async-model.md).
template <typename Tag, typename T>
class SlotTable {
 public:
  SlotTable() = default;
  // A bounded table of `capacity` slots. `first_generation` (at least 1)
  // is the generation every slot starts at: a test hook that reaches
  // exhaustion quickly. Production code leaves it at 1; a larger value
  // only brings retirement sooner, since a fresh table never issues an
  // identity twice.
  explicit SlotTable(std::size_t capacity, std::uint32_t first_generation = 1)
      : bounded_(true), slots_(capacity), free_(capacity) {
    Check(capacity > 0 && capacity <= kMaxSlots && first_generation > 0,
          "a bounded slot table needs slots and a valid first generation");
    for (std::size_t i = 0; i < capacity; ++i) {
      slots_[i].generation = first_generation;
      free_[i] = static_cast<std::uint32_t>(i);
    }
    free_count_ = capacity;
  }

  // The identity of the new element; invalid if the table is exhausted
  // (a bounded table: every slot live or retired).
  template <typename... Args>
  Id<Tag> Insert(Args&&... args) {
    std::uint32_t index = 0;
    if (bounded_) {
      if (free_count_ == 0) {
        return {};
      }
      index = free_[free_head_];
      free_head_ = (free_head_ + 1) % free_.size();
      --free_count_;
    } else if (!free_.empty()) {
      index = free_.back();
      free_.pop_back();
    } else {
      if (slots_.size() >= kMaxSlots) {
        return {};
      }
      index = static_cast<std::uint32_t>(slots_.size());
      slots_.push_back(Slot{});
    }
    Slot& slot = slots_[index];
    slot.value.emplace(std::forward<Args>(args)...);
    ++live_;
    return {index, slot.generation};
  }

  // The element an identity names, or nullptr if it is stale or invalid.
  T* Find(Id<Tag> id) {
    if (!id.valid() || id.index() >= slots_.size()) {
      return nullptr;
    }
    Slot& slot = slots_[id.index()];
    return slot.generation == id.generation() && slot.value ? &*slot.value : nullptr;
  }
  const T* Find(Id<Tag> id) const { return const_cast<SlotTable*>(this)->Find(id); }  // NOLINT

  // Removes the element; false if the identity is stale or invalid.
  bool Erase(Id<Tag> id) {
    if (Find(id) == nullptr) {
      return false;
    }
    Slot& slot = slots_[id.index()];
    slot.value.reset();
    --live_;
    if (slot.generation == UINT32_MAX) {
      ++retired_;
      return true;  // retired: its generation cannot advance
    }
    ++slot.generation;
    if (bounded_) {
      // At most every slot is free, so the ring never overflows.
      free_[(free_head_ + free_count_) % free_.size()] = id.index();
      ++free_count_;
    } else {
      free_.push_back(id.index());
    }
    return true;
  }

  std::size_t size() const { return live_; }
  // Slots whose generation is exhausted: never issued again.
  std::size_t retired() const { return retired_; }

  // Calls fn(id, value) for every live element, in index order.
  template <typename Fn>
  void ForEach(Fn&& fn) const {
    for (std::uint32_t i = 0; i < slots_.size(); ++i) {
      const Slot& slot = slots_[i];
      if (slot.value.has_value()) {
        fn(Id<Tag>(i, slot.generation),
           slot.value.value());  // NOLINT(bugprone-unchecked-optional-access)
      }
    }
  }

 private:
  static constexpr std::size_t kMaxSlots = UINT32_MAX;
  struct Slot {
    std::uint32_t generation = 1;
    std::optional<T> value;
  };
  bool bounded_ = false;
  std::vector<Slot> slots_;
  // Free slots: a stack, or in a bounded table a ring of free_count_
  // entries from free_head_.
  std::vector<std::uint32_t> free_;
  std::size_t free_head_ = 0;
  std::size_t free_count_ = 0;
  std::size_t live_ = 0;
  std::size_t retired_ = 0;
};

}  // namespace llmp::base

#endif  // LLMP_BASE_IDS_H_
