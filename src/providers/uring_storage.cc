// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "providers/uring_storage.h"

#include <sys/eventfd.h>
#include <sys/uio.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <system_error>
#include <utility>
#include <vector>

#include "base/check.h"
#include "platform/io_uring.h"
#include "providers/storage.h"

namespace llmp::providers {

std::expected<std::unique_ptr<UringStorage>, std::error_code> UringStorage::Create(
    std::size_t depth) {
  if (depth == 0 || depth > 4096) {
    return std::unexpected(std::make_error_code(std::errc::invalid_argument));
  }
  // Room for every request, a cancellation of each, and the wake read.
  auto ring = platform::IoUring::Create(static_cast<unsigned>((depth * 2) + 1));
  if (!ring) {
    return std::unexpected(ring.error());
  }
  // Blocking: a nonblocking eventfd would complete the wake read at once.
  const int wake_fd = ::eventfd(0, EFD_CLOEXEC);
  if (wake_fd < 0) {
    return std::unexpected(std::error_code(errno, std::generic_category()));
  }
  return std::make_unique<UringStorage>(*std::move(ring), depth, wake_fd);
}

std::expected<std::unique_ptr<Storage>, std::error_code> OpenStorage(std::size_t depth) {
  auto created = UringStorage::Create(depth);
  if (!created) {
    return std::unexpected(created.error());
  }
  return std::unique_ptr<Storage>(std::move(*created));
}

UringStorage::UringStorage(platform::IoUring ring, std::size_t depth, int wake_fd)
    : ring_(std::move(ring)),
      depth_(depth),
      scratch_(ring_.completion_entries()),
      wake_fd_(wake_fd) {}

UringStorage::~UringStorage() {
  base::Check(in_flight_.empty() && cancels_in_flight_ == 0,
              "a storage ring destroyed with requests in flight");
  // The wake read is ours: end it and reap it before the ring and the
  // eventfd go, since closing the ring does not stop it writing wake_count_.
  if (wake_armed_) {
    Signal();
  }
  while (wake_armed_) {
    last_error_.clear();
    Enter(1);
    const std::size_t reaped = ring_.Reap(scratch_);
    for (std::size_t i = 0; i < reaped; ++i) {
      base::Check(scratch_[i].user_data == kWakeToken,
                  "a storage ring destroyed with requests in flight");
      wake_armed_ = false;
    }
    base::Check(reaped > 0 || !last_error_, "the storage ring cannot reap its wake read");
  }
  base::Check(ring_.unconsumed() == 0 && ring_.prepared() == 0,
              "a storage ring destroyed with requests in flight");
  (void)::close(wake_fd_);
}

void UringStorage::Signal() const {
  const std::uint64_t one = 1;
  ssize_t written = 0;
  do {
    written = ::write(wake_fd_, &one, sizeof(one));
  } while (written < 0 && errno == EINTR);
}

void UringStorage::Wake() {
  // Coalesced: one write until the lane reaps the wake read.
  if (!woken_.exchange(true, std::memory_order_seq_cst)) {
    Signal();
  }
}

bool UringStorage::ArmWake() {
  if (!wake_armed_) {
    if (!ring_.PrepareRead(wake_fd_, &wake_count_, sizeof(wake_count_), 0, kWakeToken)) {
      return false;
    }
    wake_armed_ = true;
  }
  return true;
}

Submission UringStorage::Hand() {
  // Once published, a request reaches the kernel on this or a later
  // Submit(): if the kernel has not consumed it yet, its outcome is not
  // known, but it is in flight either way.
  Enter(0);
  return ring_.unconsumed() == 0 ? Submission::kAccepted : Submission::kUnknown;
}

void UringStorage::Enter(unsigned wait_for) {
  if (auto entered = ring_.Submit(wait_for); !entered) {
    last_error_ = entered.error();
  }
}

Submission UringStorage::Submit(const IoRequest& request) {
  if (in_flight_.size() >= depth_ || (request.token & kCancelBit) != 0 ||
      (request.token | kCancelBit) == kWakeToken || in_flight_.contains(request.token) ||
      cancelling_.contains(request.token)) {
    return Submission::kNotStarted;
  }
  if (!request.segments.empty()) {
    // Vectored: the iovecs stay here until the completion, since the
    // kernel may read them only when it consumes the entry.
    if (request.segments.size() > kMaxSegments) {
      return Submission::kNotStarted;
    }
    std::vector<iovec> vectors;
    vectors.reserve(request.segments.size());
    std::uint64_t total = 0;
    for (const IoSegment& segment : request.segments) {
      if (segment.memory == nullptr || segment.length == 0) {
        return Submission::kNotStarted;
      }
      vectors.push_back(iovec{.iov_base = segment.memory, .iov_len = segment.length});
      total += segment.length;
    }
    if (total != request.length) {
      return Submission::kNotStarted;  // `length` is the segments' sum (storage.h)
    }
    // Kept before the entry is prepared: once prepared, it will reach the
    // kernel. (A token's iovecs go with its completion, and the token is
    // not in flight, so none are kept for it.)
    const auto [kept, inserted] = vectors_.emplace(request.token, std::move(vectors));
    if (!inserted) {
      return Submission::kNotStarted;
    }
    const auto count = static_cast<unsigned>(kept->second.size());
    const bool prepared = request.kind == IoKind::kRead
                              ? ring_.PrepareReadVectored(request.fd, kept->second.data(), count,
                                                          request.offset, request.token)
                              : ring_.PrepareWriteVectored(request.fd, kept->second.data(), count,
                                                           request.offset, request.token);
    if (!prepared) {
      vectors_.erase(kept);  // the ring had no room: nothing reaches the kernel
      return Submission::kNotStarted;
    }
    in_flight_.insert(request.token);
    return Hand();
  }
  if (request.memory == nullptr || request.length == 0) {
    return Submission::kNotStarted;
  }
  const bool prepared = request.kind == IoKind::kRead
                            ? ring_.PrepareRead(request.fd, request.memory, request.length,
                                                request.offset, request.token)
                            : ring_.PrepareWrite(request.fd, request.memory, request.length,
                                                 request.offset, request.token);
  if (!prepared) {
    return Submission::kNotStarted;
  }
  in_flight_.insert(request.token);
  return Hand();
}

Submission UringStorage::Cancel(std::uint64_t token) {
  if (!in_flight_.contains(token) || cancels_in_flight_ >= depth_ ||
      !ring_.PrepareCancel(token, token | kCancelBit)) {
    return Submission::kNotStarted;
  }
  ++cancels_in_flight_;
  ++cancelling_[token];
  return Hand();
}

std::size_t UringStorage::Harvest(std::span<IoCompletion> out, bool wait) {
  // Waits only with the wake read armed, so Wake can end the wait; with no
  // room to arm it, it does not wait.
  if (wait && (!in_flight_.empty() || cancels_in_flight_ > 0) && ArmWake()) {
    Enter(1);
  } else if (ring_.unconsumed() > 0 || ring_.prepared() > 0) {
    Enter(0);
  }
  std::size_t produced = 0;
  while (produced < out.size()) {
    const std::size_t room = std::min(out.size() - produced, scratch_.size());
    const std::size_t reaped = ring_.Reap(std::span(scratch_).first(room));
    if (reaped == 0) {
      break;
    }
    for (std::size_t i = 0; i < reaped; ++i) {
      const platform::Completion& completion = scratch_[i];
      if (completion.user_data == kWakeToken) {
        // Cleared before the lane looks for commands again: a wake after
        // this writes the eventfd again.
        wake_armed_ = false;
        woken_.store(false, std::memory_order_seq_cst);
        continue;
      }
      if ((completion.user_data & kCancelBit) != 0) {
        --cancels_in_flight_;  // the cancellation's own result: never reported
        const std::uint64_t target = completion.user_data & ~kCancelBit;
        if (--cancelling_[target] == 0) {
          cancelling_.erase(target);
        }
        continue;
      }
      in_flight_.erase(completion.user_data);
      vectors_.erase(completion.user_data);  // the kernel is done with them
      out[produced++] = IoCompletion{.token = completion.user_data, .result = completion.result};
    }
  }
  return produced;
}

}  // namespace llmp::providers
