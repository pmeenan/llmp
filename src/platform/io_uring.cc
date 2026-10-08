// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "platform/io_uring.h"

#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <span>
#include <system_error>
#include <utility>

namespace llmp::platform {
namespace {

std::error_code LastError() { return {errno, std::generic_category()}; }

template <typename T>
T* Field(void* ring, std::uint32_t offset) {
  return reinterpret_cast<T*>(static_cast<std::byte*>(ring) +
                              offset);  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
}

unsigned Acquire(unsigned* shared) {
  return std::atomic_ref<unsigned>(*shared).load(std::memory_order_acquire);
}
void Publish(unsigned* shared, unsigned value) {
  std::atomic_ref<unsigned>(*shared).store(value, std::memory_order_release);
}

}  // namespace

std::expected<IoUring, std::error_code> IoUring::Create(unsigned entries) {
  io_uring_params params{};
  const long fd = ::syscall(__NR_io_uring_setup, entries, &params);
  if (fd < 0) {
    const int error = errno;
    if (error == ENOSYS) {
      return std::unexpected(std::make_error_code(std::errc::function_not_supported));
    }
    return std::unexpected(std::error_code(error, std::generic_category()));
  }
  IoUring ring;
  ring.fd_ = static_cast<int>(fd);
  ring.sq_entries_ = params.sq_entries;
  ring.cq_entries_ = params.cq_entries;
  ring.sq_ring_size_ = params.sq_off.array + (params.sq_entries * sizeof(unsigned));
  ring.cq_ring_size_ = params.cq_off.cqes + (params.cq_entries * sizeof(io_uring_cqe));
  const bool single = (params.features & IORING_FEAT_SINGLE_MMAP) != 0;
  if (single) {
    ring.sq_ring_size_ = ring.cq_ring_size_ = std::max(ring.sq_ring_size_, ring.cq_ring_size_);
  }
  ring.sq_ring_ = ::mmap(nullptr, ring.sq_ring_size_, PROT_READ | PROT_WRITE,
                         MAP_SHARED | MAP_POPULATE, ring.fd_, IORING_OFF_SQ_RING);
  if (ring.sq_ring_ == MAP_FAILED) {
    ring.sq_ring_ = nullptr;
    return std::unexpected(LastError());
  }
  if (single) {
    ring.cq_ring_ = ring.sq_ring_;
  } else {
    ring.cq_ring_ = ::mmap(nullptr, ring.cq_ring_size_, PROT_READ | PROT_WRITE,
                           MAP_SHARED | MAP_POPULATE, ring.fd_, IORING_OFF_CQ_RING);
    if (ring.cq_ring_ == MAP_FAILED) {
      ring.cq_ring_ = nullptr;
      return std::unexpected(LastError());
    }
  }
  ring.sqes_size_ = params.sq_entries * sizeof(io_uring_sqe);
  ring.sqes_ = ::mmap(nullptr, ring.sqes_size_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                      ring.fd_, IORING_OFF_SQES);
  if (ring.sqes_ == MAP_FAILED) {
    ring.sqes_ = nullptr;
    return std::unexpected(LastError());
  }
  ring.sq_head_ = Field<unsigned>(ring.sq_ring_, params.sq_off.head);
  ring.sq_tail_ = Field<unsigned>(ring.sq_ring_, params.sq_off.tail);
  ring.sq_mask_ = Field<unsigned>(ring.sq_ring_, params.sq_off.ring_mask);
  ring.sq_array_ = Field<unsigned>(ring.sq_ring_, params.sq_off.array);
  ring.cq_head_ = Field<unsigned>(ring.cq_ring_, params.cq_off.head);
  ring.cq_tail_ = Field<unsigned>(ring.cq_ring_, params.cq_off.tail);
  ring.cq_mask_ = Field<unsigned>(ring.cq_ring_, params.cq_off.ring_mask);
  ring.cqes_ = Field<void>(ring.cq_ring_, params.cq_off.cqes);
  return ring;
}

IoUring::IoUring(IoUring&& other) noexcept { *this = std::move(other); }

IoUring& IoUring::operator=(IoUring&& other) noexcept {
  if (this != &other) {
    Release();
    fd_ = std::exchange(other.fd_, -1);
    sq_ring_ = std::exchange(other.sq_ring_, nullptr);
    sq_ring_size_ = std::exchange(other.sq_ring_size_, 0);
    cq_ring_ = std::exchange(other.cq_ring_, nullptr);
    cq_ring_size_ = std::exchange(other.cq_ring_size_, 0);
    sqes_ = std::exchange(other.sqes_, nullptr);
    sqes_size_ = std::exchange(other.sqes_size_, 0);
    sq_entries_ = std::exchange(other.sq_entries_, 0);
    cq_entries_ = std::exchange(other.cq_entries_, 0);
    sq_head_ = std::exchange(other.sq_head_, nullptr);
    sq_tail_ = std::exchange(other.sq_tail_, nullptr);
    sq_mask_ = std::exchange(other.sq_mask_, nullptr);
    sq_array_ = std::exchange(other.sq_array_, nullptr);
    cq_head_ = std::exchange(other.cq_head_, nullptr);
    cq_tail_ = std::exchange(other.cq_tail_, nullptr);
    cq_mask_ = std::exchange(other.cq_mask_, nullptr);
    cqes_ = std::exchange(other.cqes_, nullptr);
    sq_prepared_ = std::exchange(other.sq_prepared_, 0);
  }
  return *this;
}

IoUring::~IoUring() { Release(); }

void IoUring::Release() {
  // Closing the ring does not wait for requests in flight; its owner must
  // have drained them (D-048: destruction is not proof of completion).
  if (sqes_ != nullptr) {
    (void)::munmap(sqes_, sqes_size_);
  }
  if (cq_ring_ != nullptr && cq_ring_ != sq_ring_) {
    (void)::munmap(cq_ring_, cq_ring_size_);
  }
  if (sq_ring_ != nullptr) {
    (void)::munmap(sq_ring_, sq_ring_size_);
  }
  if (fd_ >= 0) {
    (void)::close(fd_);
  }
  sqes_ = cq_ring_ = sq_ring_ = nullptr;
  fd_ = -1;
}

void* IoUring::NextEntry() {
  const unsigned head = Acquire(sq_head_);
  const unsigned tail = *sq_tail_ + sq_prepared_;
  if (tail - head >= sq_entries_) {
    return nullptr;
  }
  const unsigned index = tail & *sq_mask_;
  auto* entry = static_cast<io_uring_sqe*>(sqes_) +
                index;  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  std::memset(entry, 0, sizeof(*entry));
  sq_array_[index] = index;  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  ++sq_prepared_;
  return entry;
}

bool IoUring::PrepareRead(int fd, void* memory, std::uint32_t length, std::uint64_t offset,
                          std::uint64_t user_data) {
  auto* entry = static_cast<io_uring_sqe*>(NextEntry());
  if (entry == nullptr) {
    return false;
  }
  entry->opcode = IORING_OP_READ;
  entry->fd = fd;
  entry->off = offset;
  entry->addr = reinterpret_cast<std::uint64_t>(memory);
  entry->len = length;
  entry->user_data = user_data;
  return true;
}

bool IoUring::PrepareWrite(int fd, const void* memory, std::uint32_t length, std::uint64_t offset,
                           std::uint64_t user_data) {
  auto* entry = static_cast<io_uring_sqe*>(NextEntry());
  if (entry == nullptr) {
    return false;
  }
  entry->opcode = IORING_OP_WRITE;
  entry->fd = fd;
  entry->off = offset;
  entry->addr = reinterpret_cast<std::uint64_t>(memory);
  entry->len = length;
  entry->user_data = user_data;
  return true;
}

bool IoUring::PrepareVectored(std::uint8_t opcode, int fd, const struct iovec* vectors,
                              unsigned count, std::uint64_t offset, std::uint64_t user_data) {
  auto* entry = static_cast<io_uring_sqe*>(NextEntry());
  if (entry == nullptr) {
    return false;
  }
  entry->opcode = opcode;
  entry->fd = fd;
  entry->off = offset;
  entry->addr = reinterpret_cast<std::uint64_t>(vectors);
  entry->len = count;  // iovecs, not bytes
  entry->user_data = user_data;
  return true;
}

bool IoUring::PrepareReadVectored(int fd, const struct iovec* vectors, unsigned count,
                                  std::uint64_t offset, std::uint64_t user_data) {
  return PrepareVectored(IORING_OP_READV, fd, vectors, count, offset, user_data);
}

bool IoUring::PrepareWriteVectored(int fd, const struct iovec* vectors, unsigned count,
                                   std::uint64_t offset, std::uint64_t user_data) {
  return PrepareVectored(IORING_OP_WRITEV, fd, vectors, count, offset, user_data);
}

bool IoUring::PrepareCancel(std::uint64_t target, std::uint64_t user_data) {
  auto* entry = static_cast<io_uring_sqe*>(NextEntry());
  if (entry == nullptr) {
    return false;
  }
  entry->opcode = IORING_OP_ASYNC_CANCEL;
  entry->fd = -1;
  entry->addr = target;
  entry->user_data = user_data;
  return true;
}

std::expected<unsigned, std::error_code> IoUring::Submit(unsigned wait_for) {
  // Publish what was prepared, then tell the kernel.
  Publish(sq_tail_, *sq_tail_ + sq_prepared_);
  sq_prepared_ = 0;
  // Everything published and not yet consumed, including what an earlier
  // call left behind.
  const unsigned count = unconsumed();
  const unsigned flags = wait_for > 0 ? IORING_ENTER_GETEVENTS : 0;
  long consumed = 0;
  do {
    consumed = ::syscall(__NR_io_uring_enter, fd_, count, wait_for, flags, nullptr, 0);
  } while (consumed < 0 && errno == EINTR);
  if (consumed < 0) {
    return std::unexpected(LastError());
  }
  return static_cast<unsigned>(consumed);
}

unsigned IoUring::unconsumed() const { return *sq_tail_ - Acquire(sq_head_); }

std::size_t IoUring::Reap(std::span<Completion> out) {
  unsigned head = *cq_head_;
  const unsigned tail = Acquire(cq_tail_);
  std::size_t reaped = 0;
  while (head != tail && reaped < out.size()) {
    const auto* entry =
        static_cast<const io_uring_cqe*>(cqes_) +
        (head & *cq_mask_);  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    out[reaped++] = Completion{.user_data = entry->user_data, .result = entry->res};
    ++head;
  }
  Publish(cq_head_, head);
  return reaped;
}

}  // namespace llmp::platform
