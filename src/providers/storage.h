// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The storage provider (D-034, D-048; docs/architecture.md#providers):
// direct reads into, and writes from, protected backing ranges, with a
// bounded number in flight. Linux implements it with io_uring
// (providers/uring_storage.h); the fake scripts completions
// (providers/fake/fake_storage.h). The storage lane owns one provider and
// is its only caller.
//
// Every submission resolves as not started, accepted, or unknown. An
// accepted or unknown request produces exactly one completion, whether it
// succeeded, failed, was cancelled or transferred fewer bytes than asked;
// a completion also proves the provider will touch that memory no more.
// A vectored request's count covers its segments in order: a short one
// filled a prefix of them.
// Cancellation is a request with its own result: the original still
// completes, and only its completion retires the memory (D-048).
//
// Wake is the one call any thread may make: it lets the lane leave a
// waiting Harvest to take a new command, such as a cancellation, instead of
// waiting for a completion that may never come (D-048).
//
// Requests carry file descriptors the caller opened for direct I/O beneath
// a storage role (config/storage_roles.h), and memory that the caller
// keeps leased until the completion arrives.

#ifndef LLMP_PROVIDERS_STORAGE_H_
#define LLMP_PROVIDERS_STORAGE_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <system_error>

namespace llmp::providers {

enum class IoKind : std::uint8_t { kRead, kWrite };

// One piece of memory of a vectored request.
struct IoSegment {
  std::byte* memory = nullptr;
  std::uint32_t length = 0;
};

// The most segments one request may carry: the kernel's iovec limit
// (IOV_MAX, 1,024 on both Sparks; docs/artifact-format.md#page-in-contract).
inline constexpr std::size_t kMaxSegments = 1024;

struct IoRequest {
  std::uint64_t token = 0;  // the caller's identity for it, echoed back
  IoKind kind = IoKind::kRead;
  int fd = -1;
  std::uint64_t offset = 0;  // in the file
  std::byte* memory = nullptr;
  std::uint32_t length = 0;
  // A vectored request when not empty (D-056's coalesced reads): the file
  // range from `offset`, `length` bytes long, moves to or from these
  // segments in order, and `memory` is unused. `length` must be the
  // segments' sum, or the request is refused. Each segment is aligned as
  // a plain request's memory and length are; at most kMaxSegments. The
  // provider keeps what it needs of them, so they need outlive only Submit.
  std::span<const IoSegment> segments;
};

enum class Submission : std::uint8_t {
  kNotStarted,  // proven never to have started: the queue was full, or it was refused
  kAccepted,
  kUnknown,  // it may have started: wait for its completion, and fault if none comes
};

struct IoCompletion {
  std::uint64_t token = 0;
  // Bytes transferred (possibly fewer than asked), or -errno.
  std::int64_t result = 0;
};

class Storage {
 public:
  Storage() = default;
  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;
  Storage(Storage&&) = delete;
  Storage& operator=(Storage&&) = delete;
  virtual ~Storage() = default;

  // The most requests in flight at once.
  virtual std::size_t depth() const = 0;
  // Requests, and any cancellations of them, whose completions are still to
  // come; the owner drains it to zero before destroying the provider.
  virtual std::size_t in_flight() const = 0;

  virtual Submission Submit(const IoRequest& request) = 0;
  // Asks for the request with `token` to be cancelled. Not started if
  // there is no such request in flight or no room to ask; the request
  // still completes either way.
  virtual Submission Cancel(std::uint64_t token) = 0;

  // Completions, up to out.size(); with `wait`, blocks until at least one
  // arrives if any request is in flight, or until woken.
  virtual std::size_t Harvest(std::span<IoCompletion> out, bool wait) = 0;
  // Any thread: a Harvest waiting now returns, possibly with nothing, and so
  // does the next one to wait if none is. Wakes before the lane harvests
  // coalesce into one.
  virtual void Wake() = 0;
};

// The system's storage provider, a ring for `depth` requests in flight: on
// Linux io_uring's (uring_storage.h), which the storage library builds
// there. Another system's library defines it over its own asynchronous
// reads (docs/portability.md: a thread pool of pread/pwrite or dispatch_io
// on macOS, overlapped I/O on an I/O completion port on Windows).
// std::errc::function_not_supported where the system has none.
std::expected<std::unique_ptr<Storage>, std::error_code> OpenStorage(std::size_t depth);

}  // namespace llmp::providers

#endif  // LLMP_PROVIDERS_STORAGE_H_
