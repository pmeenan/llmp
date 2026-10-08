// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// Whole reads over the storage provider (D-034, D-048;
// docs/async-model.md): a range of a file into memory, all of it or an
// explicit failure, and write-back's whole writes the same way (ReadSpec).
// The storage lane owns one reader over its provider.
//
//   - Alignment is checked before anything starts: memory, file offset and
//     length must be multiples of the direct-I/O alignment (4 KiB for
//     D-056's artifacts).
//   - A read is split into pieces of at most `request_bytes`, each a
//     request of its own unless it coalesces (below). A short transfer
//     continues from where it stopped; one that ends at the file's end, or
//     at an unaligned point, ends the read as end of file.
//   - Interrupted or retry-later errors are retried, a bounded number of
//     times per read; any other error fails the read.
//   - Duplicate reads coalesce: a read for a key already in flight, with
//     the same range and memory, adds a waiter (a bounded list) instead of
//     reading again.
//   - Withdrawing a waiter removes only its interest. When the last one
//     leaves, the read stops starting requests and asks the provider to
//     cancel those in flight, but it still drains: it ends only when every
//     request it started has completed, so its memory is then untouched.
//   - Reads start in the order they arrived: a read's requests reach the
//     provider before a later read's, and a continuation or retry goes
//     ahead of reads that have not started. The device sees the ranges in
//     the order the caller asked for them (sequential for a load), which
//     matters: out-of-order 2 MiB reads at 4 KiB-aligned offsets ran about
//     18% slower on the Spark's SSD (docs/experiments/pagein-perf/).
//   - A request the provider could not start waits for the next Poll; one
//     whose start is unknown is waited for like any other.
//   - Adjacent reads may coalesce (D-056, BP-P1; an option, off by
//     default, ReaderSettings::span_bytes): when the provider has room
//     again, the pieces waiting to start that continue one another in the
//     same file, in start order, go as one vectored request, one segment
//     per piece into that piece's own memory, up to `span_bytes` and
//     `span_segments`. A run breaks wherever the file range does not
//     continue (a resident chunk between two misses, another shard), at a
//     write (only reads coalesce), and at a read whose span failed (below).
//     Nothing waits in order to coalesce: a read starts as soon as the
//     provider takes it, and only reads already waiting for room join.
//   - A span's count fills its pieces in order. Pieces it filled are done;
//     the first it did not fill continues or ends as a lone request would
//     (a count of zero, or one ending at an unaligned point, is the end of
//     the file); the pieces after it start again, still ahead of later
//     reads. A retryable error retries each of its reads; any other error
//     starts each of its reads again on its own, so the error lands on the
//     read it belongs to. A span is cancelled only once every read in it
//     is stopping: withdrawing one read never cancels its neighbours'
//     bytes, and a withdrawn read still ends only when the span completes.
// A read that fails or is cancelled has no usable contents: the caller
// publishes nothing from it.

#ifndef LLMP_PROVIDERS_DIRECT_READER_H_
#define LLMP_PROVIDERS_DIRECT_READER_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <map>
#include <span>
#include <string>
#include <vector>

#include "providers/storage.h"

namespace llmp::providers {

// The most one coalesced request moves when coalescing is on: the artifact
// prototype's run size, a tuning value (docs/artifact-format.md#page-in-contract).
inline constexpr std::uint32_t kSpanBytes = 64U << 20U;
// A span_bytes that never coalesces: the default.
inline constexpr std::uint32_t kNoCoalescing = 0;

struct ReaderSettings {
  std::uint32_t alignment = 4096;
  std::uint32_t request_bytes = 2U << 20U;  // the most one piece (a segment) moves
  std::uint32_t retries = 3;                // transient errors per read
  std::size_t reads = 64;                   // reads in flight at once
  std::size_t waiters = 8;                  // per read
  // Coalescing: the most one request moves, and the most pieces it
  // carries (at most kMaxSegments). A span no larger than `request_bytes`,
  // or one segment, never coalesces. Off by default (one request per
  // piece): BP-P1's A/B on the GB10 measured coalescing slower (D-085;
  // docs/experiments/backend-proof/README.md#bp-p1-coalesced-reads).
  // kSpanBytes turns it on.
  std::uint32_t span_bytes = kNoCoalescing;
  std::size_t span_segments = kMaxSegments;
};

// A whole transfer: a read of the file range into memory, or with kWrite
// (write-back, docs/architecture.md#page-in-and-eviction-lifecycles-8) a
// write of the memory into the file range. A write is split, continued and
// retried as a read is; a write that stops short of its range (no progress,
// or at an unaligned point) fails rather than ending at end of file.
struct ReadSpec {
  int fd = -1;
  std::uint64_t offset = 0;
  std::byte* memory = nullptr;
  std::uint64_t length = 0;
  IoKind kind = IoKind::kRead;
  bool operator==(const ReadSpec&) const = default;
};

enum class ReadError : std::uint8_t {
  kFull,            // as many reads in flight as allowed
  kUnaligned,       // memory, offset or length off the alignment
  kInvalidRange,    // the file or memory range cannot be represented
  kMismatch,        // the key is in flight for a different range or memory
  kTooManyWaiters,  // the read's waiter list is full
  kDraining,        // cancelled and still draining: retry once it ends
  kUnknownRead,     // no such read, or not a waiter of it
};

std::string ToString(ReadError error);

enum class ReadOutcome : std::uint8_t {
  kComplete,
  kEndOfFile,  // the file ended before the range did
  kFailed,     // an I/O error; `error` holds its errno
  kCancelled,  // every waiter withdrew
};

struct FinishedRead {
  std::uint64_t key = 0;
  ReadOutcome outcome = ReadOutcome::kComplete;
  std::uint64_t bytes = 0;  // transferred into memory
  int error = 0;
  std::vector<std::uint64_t> waiters;
};

class DirectReader {
 public:
  // `settings` must be sane: a non-zero alignment, and requests a non-zero
  // multiple of it, with non-zero read and waiter limits (a configuration
  // error is fatal). Accepted requests must be drained before destruction.
  DirectReader(Storage& storage, ReaderSettings settings);
  ~DirectReader();

  DirectReader(const DirectReader&) = delete;
  DirectReader& operator=(const DirectReader&) = delete;
  DirectReader(DirectReader&&) = delete;
  DirectReader& operator=(DirectReader&&) = delete;

  // Starts reading `spec` for `key`, or joins the read in flight for it.
  // True if it joined. Repeating the same waiter is idempotent.
  std::expected<bool, ReadError> Read(std::uint64_t key, const ReadSpec& spec,
                                      std::uint64_t waiter);
  std::expected<void, ReadError> Withdraw(std::uint64_t key, std::uint64_t waiter);

  // Starts what can start, harvests completions (waiting for one, with
  // `wait`, if anything is in flight), and returns the reads that ended.
  std::vector<FinishedRead> Poll(bool wait);

  std::size_t reads() const { return reads_.size(); }

 private:
  struct Piece {
    std::uint64_t start = 0;  // within the read
    std::uint64_t length = 0;
    std::uint64_t done = 0;
    bool in_flight = false;
    std::uint64_t token = 0;
  };
  struct Reading {
    ReadSpec spec;
    std::vector<std::uint64_t> waiters;
    std::vector<Piece> pieces;
    std::uint64_t arrival = 0;  // its place in the start order
    std::uint32_t retries = 0;
    bool queued = false;    // in queue_: it has a piece to start
    bool stopping = false;  // failed, cancelled or at end of file: start nothing more
    bool alone = false;     // a span it was in failed: its pieces start one at a time
    ReadOutcome outcome = ReadOutcome::kComplete;
    int error = 0;
  };
  // One piece of a request in flight.
  struct Member {
    std::uint64_t key = 0;
    std::size_t piece = 0;
  };
  struct Request {
    std::vector<Member> members;  // in file order
    bool cancelled = false;       // asked of the provider once
  };

  // Queues a read with a piece to start, in its arrival order.
  void Queue(std::uint64_t key, Reading& reading);
  // Starts queued pieces in order, coalescing runs, until the provider is
  // full; then drops the reads with nothing left to start from the front.
  void StartQueued();
  // Submits one run; false if the provider did not take it.
  bool Submit(std::span<const Member> run);
  // Settles one completed request's members.
  void Settle(const Request& request, std::int64_t result);
  // Asks the provider to cancel the request once every read in it stops.
  void CancelIfAbandoned(std::uint64_t token);
  static void Stop(Reading& reading, ReadOutcome outcome, int error);
  static bool Finished(const Reading& reading);
  static bool HasStart(const Reading& reading);

  Storage& storage_;
  ReaderSettings settings_;
  std::map<std::uint64_t, Reading> reads_;  // by key
  // Reads with a piece to start, by arrival: each at most once.
  std::map<std::uint64_t, std::uint64_t> queue_;  // arrival -> key
  std::uint64_t next_arrival_ = 0;
  // Reads a completion or a withdrawal may have finished, for Poll to look at.
  std::vector<std::uint64_t> settled_;
  std::map<std::uint64_t, Request> requests_;  // in flight, by token
  std::uint64_t next_token_ = 1;
  // StartQueued's scratch, kept to avoid allocating on each call.
  std::vector<Member> run_;
  std::vector<IoSegment> segments_;
};

}  // namespace llmp::providers

#endif  // LLMP_PROVIDERS_DIRECT_READER_H_
