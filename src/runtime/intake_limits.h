// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// What a request may hold in host memory (D-102;
// docs/runtime-serving.md#the-chat-route). Requests live in host memory the
// catalog does not count otherwise, so they are charged to one pool of their
// own, the request memory, by what each actually holds or is about to build:
//
// - a body's bytes as they arrive (not as declared), until it is parsed;
// - its parse's working set (json::ParseWorkingBytes: the document's nodes,
//   its strings and the request built from them), while it parses;
// - the parsed request (its strings, prompt IDs and stop matcher), from its
//   parse until it is done;
// - a stream's unread output, as it grows (past a stream buffer, or when
//   the pool cannot take more, the request pauses until its client reads);
// - its rendering and tokenization's working set, while they run;
// - a non-streaming response's text and its body, until the socket has
//   taken it.
//
// The pool grows and shrinks (D-102 as the owner decided 2026-10-03): a
// small floor (kRequestMemoryFloor) is set apart at the start beside the
// memory guard's margin; past it, the driver charges what requests hold to
// the catalog's budget like any other need, through the node's reclaim
// order (displacing plans, graphs and idle conversations, D-055), and gives
// it back as requests end. Request memory itself is never reclaimed. The
// I/O and parse threads charge only what has been granted: a body that
// cannot grow waits, unread, until the driver grants more (or refuses), and
// a parse waits likewise. The pool's capacity is the most it could ever
// reach: [client] request_memory_bytes when set, otherwise the floor and
// the state room the budget leaves beside the largest model. A charge past
// the capacity is refused before the work (413); one the driver could not
// grant now gets a 503 with Retry-After. Refusals name
// [client] request_memory_bytes.
//
// Sizes. A body is at most a sixteenth of the capacity and at most the
// largest configured context's worth of bytes; a JSON body's working set
// is up to about thirteen times its bytes (a dense token-ID array: a value
// every two bytes at twenty bytes a value, and the request built from it).
// A stream's unread output before it pauses is a sixty-fourth of the floor,
// 1 to 64 MiB. A chat rendering is at most what a prompt that fits the
// model's context could occupy: the usable context times the vocabulary's
// longest token, four times over when the tokenizer normalizes (NFC
// composition shortens text at most threefold). Vendor-free and pure but
// for RequestMemory's lock, so the CPU tests hold the arithmetic.

#ifndef JITLLM_RUNTIME_INTAKE_LIMITS_H_
#define JITLLM_RUNTIME_INTAKE_LIMITS_H_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <vector>

#include "config/node_config.h"

namespace jitllm::runtime {

// The request memory set apart at the start, beside the guard's margin.
inline constexpr std::uint64_t kRequestMemoryFloor = std::uint64_t{256} << 20U;
// The capacity when nothing says how much the budget could give (tests,
// CPU-only).
inline constexpr std::uint64_t kNominalRequestCapacity = std::uint64_t{1} << 30U;
// A rendering's least bound, for template text around a short context.
inline constexpr std::uint64_t kMinRenderBytes = std::uint64_t{1} << 20U;
// Bytes a JSON request takes a token, at most, for the context's bound on a
// body: the token's own bytes JSON-escaped (six a byte) and separators.
inline constexpr std::uint64_t kJsonBytesPerTokenByte = 6;
// Queued requests may hold at most this share of the capacity (as 1/n):
// the rest is the driver's headroom for the request it takes up, so the
// head never fails for room it waited for.
inline constexpr std::uint64_t kQueueHeadroomShare = 4;

struct IntakeLimits {
  std::uint64_t request_floor = 0;     // the request memory set apart at the start
  std::uint64_t request_capacity = 0;  // the most it may grow to
  std::uint64_t max_body = 0;          // one request body
  std::uint64_t stream_buffer = 0;     // a stream's unread output before it pauses
};

// The floor: kRequestMemoryFloor, or [client] request_memory_bytes when
// that is smaller.
std::uint64_t RequestFloor(const config::ClientConfig& client);

// The limits for a request memory of `floor` bytes that may grow to
// `capacity` (the floor and the state room; 0: kNominalRequestCapacity),
// with [client] request_memory_bytes capping it, and models whose largest
// context in body bytes is `context_bytes` (0: none), with what `client`
// sets in their place.
IntakeLimits DeriveIntakeLimits(std::uint64_t floor, std::uint64_t capacity,
                                std::uint64_t context_bytes, const config::ClientConfig& client);

// The server's defaults when nothing else is given (tests).
IntakeLimits NominalIntakeLimits();

// A chat rendering's most bytes for a model of `context` usable tokens
// whose vocabulary's longest token text is `longest_token` bytes (above).
std::uint64_t RenderBytes(std::uint32_t context, std::size_t longest_token, bool normalizes);

// The bytes of a body that holds a prompt of `context` tokens of at most
// `longest_token` bytes each, JSON-escaped, and its framing.
std::uint64_t ContextBodyBytes(std::uint32_t context, std::size_t longest_token);

// The connections the route may keep when [client] max_connections is
// absent: the open-file limit less what the runtime keeps for itself
// (model files, the node's own), at least 16.
inline constexpr std::uint64_t kReservedFiles = 256;
std::uint64_t ConnectionsFor(std::uint64_t open_files);

// The pool's accounting, shared by the I/O thread, the parse thread and the
// driver. Thread-safe.
//
// What is charged may not pass the grant (the floor, and what the driver
// has charged to the catalog beside it). A charge that does not fit records
// what was wanted; the driver, between units (Settle), grows the grant to
// it through its grower, or counts a denial. On the driver's own thread
// (SetDriver) a charge that does not fit asks the grower at once. Growth
// needs the reclaim order to free room in the budget (plans, graphs, idle
// conversations): what active conversations hold is never taken for it, so
// the capacity is a bound, not a promise.
class RequestMemory {
 public:
  // Grows the grant to `to` bytes (on the driver, through the reclaim
  // order) or shrinks it: whether it now holds `to`.
  using Grower = std::function<bool(std::uint64_t to)>;

  // `exact`: the driver grower receives incoming bytes for growth and zero
  // for trim, and charges used() + incoming. Releases during reclaim can
  // therefore reduce the target; no floor/slack belongs to this pool.
  explicit RequestMemory(std::uint64_t floor, std::uint64_t capacity = 0, bool exact = false)
      : floor_(floor), capacity_(capacity == 0 ? floor : capacity), exact_(exact), grant_(floor) {}
  RequestMemory(const RequestMemory&) = delete;
  RequestMemory& operator=(const RequestMemory&) = delete;
  RequestMemory(RequestMemory&&) = delete;
  RequestMemory& operator=(RequestMemory&&) = delete;
  ~RequestMemory() = default;

  std::uint64_t floor() const { return floor_; }
  std::uint64_t capacity() const { return capacity_; }
  std::uint64_t grant() const;
  std::uint64_t used() const;
  // How often the driver could not grow the grant to what was wanted.
  std::uint64_t denials() const;
  // Whether a driver grows the grant (SetDriver): without one, a charge
  // that does not fit never will.
  bool grows() const;

  // Charges `bytes` if they fit within the grant (asking the grower first
  // on the driver's thread); false otherwise, recording what was wanted.
  bool TryCharge(std::uint64_t bytes);
  // Charges `bytes` within the grant, never growing it (a charge made in
  // the middle of a unit, where the driver must not reclaim).
  bool TryChargeGranted(std::uint64_t bytes);
  // Charges `bytes` whatever the grant (memory already built that refusing
  // would not free); the excess is wanted, and later charges wait for it.
  void Force(std::uint64_t bytes);
  void Release(std::uint64_t bytes);

  // The driver: its thread, its grower, and how it settles. Settle runs on
  // the driver between units, between a serial request's steps and while it
  // waits for readers: the grant grown to what was wanted (or a denial
  // counted), or, when what was wanted fits now, the waiters told so; and
  // shrunk toward what is used, leaving `slack`, once nothing has wanted
  // more for `quiet` (a parse or a body growing in steps is not shrunk
  // under, and a burst of requests does not grow and shrink it, displacing
  // idle conversations, again and again).
  void SetDriver(std::thread::id driver, Grower grower, std::uint64_t slack = kSettleSlack,
                 std::chrono::milliseconds quiet = kSettleQuiet);
  void Settle();
  void Settle(std::uint64_t slack, std::chrono::milliseconds quiet);
  static constexpr std::uint64_t kSettleSlack = std::uint64_t{64} << 20U;
  static constexpr std::chrono::milliseconds kSettleQuiet{10'000};
  // Counts what may let a refused charge fit: the grant grown or shrunk, a
  // denial, or room now for what was wanted (not every release). A charge
  // waiting for room tries again when it moves (the parse thread).
  std::uint64_t epoch() const;

 private:
  // `record`: a refusal is wanted (for Settle).
  bool ChargeLocked(std::uint64_t bytes, bool record);
  bool Grow(std::unique_lock<std::mutex>& lock, std::uint64_t to);

  const std::uint64_t floor_;
  const std::uint64_t capacity_;
  // Native histories: no grant slack, and reclaim may release owners during growth.
  const bool exact_;
  mutable std::mutex mutex_;
  std::uint64_t grant_;
  std::uint64_t used_ = 0;
  // Since the last Settle: whether a charge was refused (or forced past the
  // grant), and the largest refused; Settle wants what is used and it.
  bool wanting_ = false;
  std::uint64_t wanted_ = 0;
  std::chrono::steady_clock::time_point demand_at_;  // the last refused charge or growth
  std::uint64_t denials_ = 0;
  std::uint64_t epoch_ = 0;
  bool growing_ = false;
  std::thread::id driver_;
  Grower grower_;
  std::uint64_t slack_ = kSettleSlack;
  std::chrono::milliseconds quiet_ = kSettleQuiet;
};

// A charge that releases itself. Move-only.
class MemoryCharge {
 public:
  MemoryCharge() = default;
  MemoryCharge(const MemoryCharge&) = delete;
  MemoryCharge& operator=(const MemoryCharge&) = delete;
  MemoryCharge(MemoryCharge&& other) noexcept;
  MemoryCharge& operator=(MemoryCharge&& other) noexcept;
  ~MemoryCharge() { Reset(); }

  // Tries to charge `bytes` more to `pool` (the pool this charge holds, if
  // any); false, and nothing charged, when they do not fit. `granted`:
  // within the grant only (RequestMemory::TryChargeGranted).
  bool Add(RequestMemory& pool, std::uint64_t bytes, bool granted = false);
  // Charges `bytes` more whatever the grant (RequestMemory::Force).
  void Force(RequestMemory& pool, std::uint64_t bytes);
  std::uint64_t bytes() const { return bytes_; }
  void Reset();

 private:
  RequestMemory* pool_ = nullptr;
  std::uint64_t bytes_ = 0;
};

// Fund replacement capacity, including simultaneous old/new storage, before
// allocating. On refusal neither storage nor its ownership charge changes.
bool ReserveTokenStorage(std::vector<std::int32_t>& tokens, MemoryCharge& charge,
                         RequestMemory& memory, std::size_t capacity);
struct TokenReclaimGroup {
  std::size_t end = 0;  // exclusive end in the recency-ordered history capacities
  std::uint64_t catalog_bytes = 0;
};
// One recency prefix just far enough to cover `needed` catalog bytes (or
// all it can free). A single candidate keeps rounding credits independent
// of the selector's victim order. `used` includes pending admission bytes.
// Partial trailing capacities that free no quantum stay retained.
std::vector<TokenReclaimGroup> GroupTokenReclaim(
    std::span<const std::uint64_t> capacities, std::uint64_t used, std::uint64_t quantum,
    std::uint64_t needed = std::numeric_limits<std::uint64_t>::max());
// During native admission, replace actual token-catalog release with the
// decrease in the rounded prospective target; retain other catalog release.
std::uint64_t TokenReclaimCredit(std::uint64_t catalog_drop, std::uint64_t token_drop,
                                 std::uint64_t target_drop);

void ReleaseTokenStorage(std::vector<std::int32_t>& tokens, MemoryCharge& charge);

}  // namespace jitllm::runtime

#endif  // JITLLM_RUNTIME_INTAKE_LIMITS_H_
