// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/intake_limits.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>

#include "base/check.h"
#include "config/node_config.h"

namespace llmp::runtime {
namespace {

constexpr std::uint64_t kMiB = std::uint64_t{1} << 20U;
// The grant moves in whole extents, as the catalog charges them.
constexpr std::uint64_t kGrantStep = 2 * kMiB;

// a * b, saturating.
std::uint64_t Times(std::uint64_t a, std::uint64_t b) {
  if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
    return std::numeric_limits<std::uint64_t>::max();
  }
  return a * b;
}

std::uint64_t RoundUp(std::uint64_t bytes) {
  return bytes > std::numeric_limits<std::uint64_t>::max() - kGrantStep
             ? bytes
             : (bytes + kGrantStep - 1) / kGrantStep * kGrantStep;
}

}  // namespace

std::uint64_t RequestFloor(const config::ClientConfig& client) {
  return std::min(kRequestMemoryFloor, client.request_memory_bytes.value_or(kRequestMemoryFloor));
}

IntakeLimits DeriveIntakeLimits(std::uint64_t floor, std::uint64_t capacity,
                                std::uint64_t context_bytes, const config::ClientConfig& client) {
  IntakeLimits l;
  l.request_floor = floor;
  l.request_capacity = std::max(floor, capacity == 0 ? kNominalRequestCapacity : capacity);
  if (client.request_memory_bytes) {
    l.request_capacity = *client.request_memory_bytes;
    l.request_floor = std::min(l.request_floor, l.request_capacity);
  }
  std::uint64_t body = std::min(l.request_capacity / 16, config::kMaxBodyCeiling);
  if (context_bytes != 0) {
    body = std::min(body, context_bytes);
  }
  l.max_body = client.max_body_bytes.value_or(std::max<std::uint64_t>(body, 1024));
  l.stream_buffer = client.stream_buffer_bytes.value_or(
      std::clamp<std::uint64_t>(l.request_floor / 64, kMiB, 64 * kMiB));
  return l;
}

IntakeLimits NominalIntakeLimits() {
  return DeriveIntakeLimits(kRequestMemoryFloor, kNominalRequestCapacity, 0, {});
}

std::uint64_t RenderBytes(std::uint32_t context, std::size_t longest_token, bool normalizes) {
  const std::uint64_t fits =
      Times(Times(context, std::max<std::size_t>(longest_token, 1)), normalizes ? 4 : 1);
  return std::max(fits, kMinRenderBytes);
}

std::uint64_t ContextBodyBytes(std::uint32_t context, std::size_t longest_token) {
  // A token as text: its bytes, each JSON-escaped at worst to six; as an
  // ID: at most eleven ("2147483647,"). And a megabyte of framing.
  const std::uint64_t per_token = std::max<std::uint64_t>(
      Times(std::max<std::size_t>(longest_token, 1), kJsonBytesPerTokenByte), 11);
  const std::uint64_t bytes = Times(context, per_token);
  return bytes > std::numeric_limits<std::uint64_t>::max() - kMiB ? bytes : bytes + kMiB;
}

std::uint64_t ConnectionsFor(std::uint64_t open_files) {
  return open_files > kReservedFiles + 16 ? open_files - kReservedFiles : 16;
}

// ---------------------------------------------------------------- RequestMemory

std::uint64_t RequestMemory::grant() const {
  const std::scoped_lock lock(mutex_);
  return grant_;
}

std::uint64_t RequestMemory::used() const {
  const std::scoped_lock lock(mutex_);
  return used_;
}

std::uint64_t RequestMemory::denials() const {
  const std::scoped_lock lock(mutex_);
  return denials_;
}

bool RequestMemory::grows() const {
  const std::scoped_lock lock(mutex_);
  return static_cast<bool>(grower_) && grant_ < capacity_;
}

bool RequestMemory::ChargeLocked(std::uint64_t bytes, bool record) {
  if (bytes > grant_ || used_ > grant_ - bytes) {
    if (record && bytes <= capacity_ && used_ <= capacity_ - bytes) {
      wanting_ = true;
      wanted_ = std::max(wanted_, bytes);
      demand_at_ = std::chrono::steady_clock::now();
    }
    return false;
  }
  used_ += bytes;
  return true;
}

bool RequestMemory::TryCharge(std::uint64_t bytes) {
  std::unique_lock lock(mutex_);
  const bool driver = std::this_thread::get_id() == driver_ && grower_ && !growing_;
  // Recorded for Settle unless the driver grows for it now.
  if (ChargeLocked(bytes, !driver)) {
    return true;
  }
  if (!driver || bytes > capacity_ || used_ > capacity_ - bytes) {
    return false;
  }
  // The driver's own charge: the grant grows now, through the reclaim
  // order, or the charge is refused.
  if (!Grow(lock, std::min(capacity_, exact_ ? used_ + bytes : RoundUp(used_ + bytes)))) {
    ++denials_;
    ++epoch_;
    return false;
  }
  return ChargeLocked(bytes, false);
}

std::uint64_t RequestMemory::epoch() const {
  const std::scoped_lock lock(mutex_);
  return epoch_;
}

bool RequestMemory::TryChargeGranted(std::uint64_t bytes) {
  const std::scoped_lock lock(mutex_);
  return ChargeLocked(bytes, true);
}

void RequestMemory::Force(std::uint64_t bytes) {
  const std::scoped_lock lock(mutex_);
  used_ = bytes > std::numeric_limits<std::uint64_t>::max() - used_
              ? std::numeric_limits<std::uint64_t>::max()
              : used_ + bytes;
  if (used_ > grant_) {
    wanting_ = true;  // what is used already passes the grant
    demand_at_ = std::chrono::steady_clock::now();
  }
}

void RequestMemory::Release(std::uint64_t bytes) {
  const std::scoped_lock lock(mutex_);
  used_ -= std::min(bytes, used_);
}

bool RequestMemory::Grow(std::unique_lock<std::mutex>& lock, std::uint64_t to) {
  if (to <= grant_) {
    return true;
  }
  const std::uint64_t incoming = to - used_;
  growing_ = true;
  const Grower grower = grower_;
  lock.unlock();
  const bool grown = grower(exact_ ? incoming : to);
  lock.lock();
  growing_ = false;
  if (grown) {
    grant_ = exact_ ? used_ + incoming : to;
    demand_at_ = std::chrono::steady_clock::now();
    ++epoch_;
  }
  return grown;
}

void RequestMemory::SetDriver(std::thread::id driver, Grower grower, std::uint64_t slack,
                              std::chrono::milliseconds quiet) {
  const std::scoped_lock lock(mutex_);
  driver_ = driver;
  grower_ = std::move(grower);
  slack_ = slack;
  quiet_ = quiet;
}

void RequestMemory::Settle() {
  std::uint64_t slack = 0;
  std::chrono::milliseconds quiet{};
  {
    const std::scoped_lock lock(mutex_);
    slack = slack_;
    quiet = quiet_;
  }
  Settle(slack, quiet);
}

void RequestMemory::Settle(std::uint64_t slack, std::chrono::milliseconds quiet) {
  std::unique_lock lock(mutex_);
  if (!grower_) {
    return;
  }
  if (exact_) {
    // A reclaim inside Grow releases histories on this same driver. Reduce
    // the grant/catalog charge now, while preserving the outer growth guard.
    if (std::this_thread::get_id() != driver_) {
      return;
    }
    const std::uint64_t before = grant_;
    grant_ = std::max(floor_, used_);
    const bool was_growing = std::exchange(growing_, true);
    const Grower grower = grower_;
    lock.unlock();
    const bool shrunk = grower(0);
    lock.lock();
    growing_ = was_growing;
    if (!shrunk) {
      grant_ = before;
    }
    ++epoch_;
    return;
  }
  if (growing_) {
    return;
  }
  if (wanting_) {
    // What is used now and the largest refused charge beside it.
    const std::uint64_t need =
        wanted_ > capacity_ - std::min(used_, capacity_) ? capacity_ : used_ + wanted_;
    wanting_ = false;
    wanted_ = 0;
    if (need > grant_) {
      if (!Grow(lock, std::min(capacity_, RoundUp(need + slack)))) {
        ++denials_;
        ++epoch_;
      }
    } else {
      ++epoch_;  // it fits now (others released): its waiters try again
    }
    return;
  }
  // Given back toward what is used, a slack's worth above it, once nothing
  // has wanted more for `quiet` (so a parse or a body growing in steps is
  // not shrunk under between its steps): lowered first, so no charge
  // passes the new grant while the catalog is uncharged.
  const std::uint64_t keep = std::max(floor_, RoundUp(used_ + slack));
  if (grant_ <= keep || grant_ - keep < slack ||
      std::chrono::steady_clock::now() - demand_at_ < quiet) {
    return;
  }
  const std::uint64_t before = grant_;
  grant_ = keep;
  growing_ = true;
  const Grower grower = grower_;
  lock.unlock();
  const bool shrunk = grower(keep);
  lock.lock();
  growing_ = false;
  if (!shrunk) {
    grant_ = before;
  }
  ++epoch_;
}

// ---------------------------------------------------------------- MemoryCharge

MemoryCharge::MemoryCharge(MemoryCharge&& other) noexcept
    : pool_(std::exchange(other.pool_, nullptr)), bytes_(std::exchange(other.bytes_, 0)) {}

MemoryCharge& MemoryCharge::operator=(MemoryCharge&& other) noexcept {
  if (this != &other) {
    Reset();
    pool_ = std::exchange(other.pool_, nullptr);
    bytes_ = std::exchange(other.bytes_, 0);
  }
  return *this;
}

bool MemoryCharge::Add(RequestMemory& pool, std::uint64_t bytes, bool granted) {
  if (pool_ != nullptr && pool_ != &pool) {
    return false;  // one pool a charge
  }
  if (!(granted ? pool.TryChargeGranted(bytes) : pool.TryCharge(bytes))) {
    return false;
  }
  pool_ = &pool;
  bytes_ += bytes;
  return true;
}

void MemoryCharge::Force(RequestMemory& pool, std::uint64_t bytes) {
  if (pool_ != nullptr && pool_ != &pool) {
    return;
  }
  pool.Force(bytes);
  pool_ = &pool;
  bytes_ += bytes;
}

void MemoryCharge::Reset() {
  if (pool_ != nullptr && bytes_ != 0) {
    pool_->Release(bytes_);
  }
  pool_ = nullptr;
  bytes_ = 0;
}

bool ReserveTokenStorage(std::vector<std::int32_t>& tokens, MemoryCharge& charge,
                         RequestMemory& memory, std::size_t capacity) {
  if (capacity <= tokens.capacity()) {
    return true;
  }
  if (capacity > std::numeric_limits<std::uint64_t>::max() / sizeof(std::int32_t)) {
    return false;
  }
  MemoryCharge replacement_charge;
  if (!replacement_charge.Add(memory, capacity * sizeof(std::int32_t))) {
    return false;
  }
  std::vector<std::int32_t> replacement;
  replacement.reserve(capacity);
  // The pinned libstdc++ allocates exactly reserve(n) on an empty vector.
  base::Check(replacement.capacity() == capacity, "token vector exceeded its funded capacity");
  replacement.insert(replacement.end(), tokens.begin(), tokens.end());
  tokens.swap(replacement);
  // Free old storage before releasing its charge.
  std::vector<std::int32_t>().swap(replacement);
  charge = std::move(replacement_charge);
  return true;
}

std::vector<TokenReclaimGroup> GroupTokenReclaim(std::span<const std::uint64_t> capacities,
                                                 std::uint64_t used, std::uint64_t quantum,
                                                 std::uint64_t needed) {
  std::vector<TokenReclaimGroup> groups;
  if (quantum == 0) {
    return groups;
  }
  const auto rounded = [quantum](std::uint64_t bytes) {
    return bytes / quantum + (bytes % quantum != 0 ? 1U : 0U);
  };
  const std::uint64_t charged = rounded(used);
  for (std::size_t i = 0; i < capacities.size(); ++i) {
    used -= std::min(used, capacities[i]);
    const auto after = rounded(used);
    const auto credit = (charged - after) * quantum;
    if (credit != 0) {
      if (groups.empty()) {
        groups.push_back({.end = i + 1, .catalog_bytes = credit});
      } else if (credit > groups.front().catalog_bytes) {
        groups.front() = {.end = i + 1, .catalog_bytes = credit};
      }
      if (credit >= needed) {
        break;
      }
    }
  }
  return groups;
}

std::uint64_t TokenReclaimCredit(std::uint64_t catalog_drop, std::uint64_t token_drop,
                                 std::uint64_t target_drop) {
  return catalog_drop - std::min(catalog_drop, token_drop) + target_drop;
}

void ReleaseTokenStorage(std::vector<std::int32_t>& tokens, MemoryCharge& charge) {
  std::vector<std::int32_t>().swap(tokens);
  charge.Reset();
}

}  // namespace llmp::runtime
