// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// A minimal io_uring ring over the raw system calls (D-034): reads, writes
// and cancellation, with no library dependency, as M0's I/O experiment
// measured it (docs/experiments/io-path/). One thread uses a ring at a
// time; the storage lane owns it (D-048).
//
// The submission and completion queues are shared with the kernel: this
// side publishes a submission tail with release ordering and reads the
// completion tail with acquire ordering, as the kernel's ABI requires.

#ifndef LLMP_PLATFORM_IO_URING_H_
#define LLMP_PLATFORM_IO_URING_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <system_error>

struct iovec;

namespace llmp::platform {

struct Completion {
  std::uint64_t user_data = 0;
  std::int32_t result = 0;  // bytes transferred, or -errno
};

class IoUring {
 public:
  // A ring with room for `entries` submissions (rounded up by the kernel).
  // std::errc::function_not_supported where the kernel or an emulator
  // offers no io_uring.
  static std::expected<IoUring, std::error_code> Create(unsigned entries);

  IoUring(const IoUring&) = delete;
  IoUring& operator=(const IoUring&) = delete;
  IoUring(IoUring&& other) noexcept;
  IoUring& operator=(IoUring&& other) noexcept;
  ~IoUring();

  unsigned submission_entries() const { return sq_entries_; }
  unsigned completion_entries() const { return cq_entries_; }

  // Queue one request; false if the submission queue is full. Nothing
  // reaches the kernel until Submit().
  bool PrepareRead(int fd, void* memory, std::uint32_t length, std::uint64_t offset,
                   std::uint64_t user_data);
  bool PrepareWrite(int fd, const void* memory, std::uint32_t length, std::uint64_t offset,
                    std::uint64_t user_data);
  // Vectored (IORING_OP_READV, IORING_OP_WRITEV): `count` iovecs at
  // `vectors`, which must stay valid until the request completes (the
  // kernel may read them when it consumes the entry, on a later Submit()).
  bool PrepareReadVectored(int fd, const struct iovec* vectors, unsigned count,
                           std::uint64_t offset, std::uint64_t user_data);
  bool PrepareWriteVectored(int fd, const struct iovec* vectors, unsigned count,
                            std::uint64_t offset, std::uint64_t user_data);
  // Asks the kernel to cancel the request with `target`; the cancellation
  // completes with its own `user_data`, and the target still completes.
  bool PrepareCancel(std::uint64_t target, std::uint64_t user_data);

  // Hands prepared requests to the kernel, waiting for at least
  // `wait_for` completions. Returns how many the kernel consumed.
  std::expected<unsigned, std::error_code> Submit(unsigned wait_for = 0);
  // Requests prepared but not yet published to the kernel.
  unsigned prepared() const { return sq_prepared_; }
  // Requests published but not yet consumed by the kernel: they will be on
  // a later Submit(), so their outcome is not yet known.
  unsigned unconsumed() const;

  // Copies out and consumes up to out.size() completions.
  std::size_t Reap(std::span<Completion> out);

 private:
  IoUring() = default;
  void Release();
  // The next free submission entry, or nullptr.
  void* NextEntry();
  bool PrepareVectored(std::uint8_t opcode, int fd, const struct iovec* vectors, unsigned count,
                       std::uint64_t offset, std::uint64_t user_data);

  int fd_ = -1;
  void* sq_ring_ = nullptr;
  std::size_t sq_ring_size_ = 0;
  void* cq_ring_ = nullptr;  // the same mapping as sq_ring_ with a single mmap
  std::size_t cq_ring_size_ = 0;
  void* sqes_ = nullptr;
  std::size_t sqes_size_ = 0;
  unsigned sq_entries_ = 0;
  unsigned cq_entries_ = 0;
  // Pointers into the shared rings.
  unsigned* sq_head_ = nullptr;
  unsigned* sq_tail_ = nullptr;
  unsigned* sq_mask_ = nullptr;
  unsigned* sq_array_ = nullptr;
  unsigned* cq_head_ = nullptr;
  unsigned* cq_tail_ = nullptr;
  unsigned* cq_mask_ = nullptr;
  void* cqes_ = nullptr;
  unsigned sq_prepared_ = 0;  // entries written past the published tail
};

}  // namespace llmp::platform

#endif  // LLMP_PLATFORM_IO_URING_H_
