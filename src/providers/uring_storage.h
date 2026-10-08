// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The Linux storage provider: Storage over one io_uring ring
// (platform/io_uring.h, D-034). Requests are handed to the kernel as they
// are submitted; a failure to hand them over leaves their outcome unknown,
// since the kernel may have consumed some. Cancellation completions are
// consumed here and never reported: only the original's completion retires
// its memory. A vectored request is one READV or WRITEV entry, whose
// iovecs this provider keeps until its completion.
//
// Wake writes to an eventfd that a read in the same ring waits on, armed
// only while Harvest waits: that read completing ends the wait. A flag
// coalesces wakes; the lane clears it when it reaps that read, before it
// looks for commands again, so a wake is either seen by that look or
// writes the eventfd again.

#ifndef LLMP_PROVIDERS_URING_STORAGE_H_
#define LLMP_PROVIDERS_URING_STORAGE_H_

#include <sys/uio.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <system_error>
#include <vector>

#include "platform/io_uring.h"
#include "providers/storage.h"

namespace llmp::providers {

class UringStorage final : public Storage {
 public:
  // A ring for `depth` requests in flight, with room for a cancellation
  // of each and the wake read. std::errc::function_not_supported where
  // there is no io_uring.
  static std::expected<std::unique_ptr<UringStorage>, std::error_code> Create(std::size_t depth);

  // Takes ownership of `wake_fd`, a blocking eventfd.
  UringStorage(platform::IoUring ring, std::size_t depth, int wake_fd);
  // Its owner must have harvested every request first (D-048): closing the
  // ring does not stop reads landing in memory, so anything left is fatal.
  ~UringStorage() override;

  UringStorage(const UringStorage&) = delete;
  UringStorage& operator=(const UringStorage&) = delete;
  UringStorage(UringStorage&&) = delete;
  UringStorage& operator=(UringStorage&&) = delete;

  std::size_t depth() const override { return depth_; }
  // Requests and cancellations whose completions are still to come: the
  // owner drains until this is zero before destroying the provider.
  std::size_t in_flight() const override { return in_flight_.size() + cancels_in_flight_; }

  Submission Submit(const IoRequest& request) override;
  Submission Cancel(std::uint64_t token) override;
  std::size_t Harvest(std::span<IoCompletion> out, bool wait) override;
  void Wake() override;

 private:
  // Tokens with this bit are cancellations, never reported.
  static constexpr std::uint64_t kCancelBit = std::uint64_t{1} << 63;
  // The wake read's token; the one request token whose cancellation would
  // collide with it is refused.
  static constexpr std::uint64_t kWakeToken = ~std::uint64_t{0};

  // Arms the wake read unless it is armed; false if the ring has no room.
  bool ArmWake();
  // Adds one to the eventfd, completing the wake read.
  void Signal() const;

  Submission Hand();
  // io_uring_enter, remembering a failure for diagnostics: what it consumed
  // is still judged by the ring itself.
  void Enter(unsigned wait_for);

  platform::IoUring ring_;
  std::size_t depth_;
  std::set<std::uint64_t> in_flight_;
  // A vectored request's iovecs, by token, until its completion.
  std::map<std::uint64_t, std::vector<iovec>> vectors_;
  std::size_t cancels_in_flight_ = 0;
  // Cancellations not yet completed, by the token they target: a token is
  // not reused until its cancellation is done, or it could hit the new
  // request.
  std::map<std::uint64_t, std::uint32_t> cancelling_;
  std::vector<platform::Completion> scratch_;
  std::error_code last_error_;
  int wake_fd_ = -1;
  std::atomic<bool> woken_{false};  // any thread sets it; the owner clears it
  bool wake_armed_ = false;
  std::uint64_t wake_count_ = 0;  // the wake read's destination
};

}  // namespace llmp::providers

#endif  // LLMP_PROVIDERS_URING_STORAGE_H_
