// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The completion board (D-048,
// docs/async-model.md#submission-and-completion-protocol): one mailbox per
// in-flight operation, where provider lanes publish what they observed and
// the scheduler thread, the board's owner, harvests it. An admitted
// operation owns its mailbox until the owner closes it, so a terminal
// result can never be lost to a full queue. The number of mailboxes is fixed
// when the board is built.
//
// Each mailbox holds two independent observations, keyed by the operation's
// generation-tagged identity:
//
//   - acceptance: how the provider resolved the submission (not started,
//     accepted, or unknown);
//   - the terminal result: success, failure or cancellation and the bytes
//     transferred, and separately the provider's proof that there will be
//     no further access to the operation's memory, which may come with the
//     result or later (a cancelled transfer proven quiescent afterwards).
//
// Either can arrive first, and the owner retires nothing before
// reconciling both. A repeated observation equal to the first is ignored;
// one that differs marks the mailbox contradictory, which faults the
// provider, unless the owner closed the operation first (then it is
// stale). An observation for a stale identity (one the owner already
// closed) changes nothing: generations are part of every atomic state
// word, so a late write cannot land in the mailbox's next operation.
//
// The board keeps a proof that follows its result, but the scheduler
// (scheduler.h) reads a result harvested without its proof as unproven
// and quarantines the operation for good: a lane publishes the proof with
// the result, as every lane does today, or never.
//
// Publishing is release/acquire synchronized, and the board signals the
// owner's WakeFlag after every publication that needs harvesting.
//
// Free mailboxes are issued in rotation (the longest free first), so every
// mailbox's generation advances at the same pace. A mailbox whose
// generation is exhausted is retired, never reused: with rotation that
// comes after about capacity x 2^32 operations, not one mailbox at a time
// from the first 2^32 (docs/async-model.md).

#ifndef JITLLM_SCHEDULER_COMPLETIONS_H_
#define JITLLM_SCHEDULER_COMPLETIONS_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>

#include "base/ids.h"
#include "base/wake.h"

namespace jitllm::scheduler {

struct OperationTag {
  static constexpr const char* kName = "operation";
};
using OperationId = base::Id<OperationTag>;

enum class Acceptance : std::uint8_t {
  kNone,        // not yet resolved
  kNotStarted,  // proven never to have started
  kAccepted,
  kUnknown,  // the provider cannot tell: treat as accepted, never as not started
};

enum class Outcome : std::uint8_t { kSucceeded, kFailed, kCancelled };

struct Terminal {
  Outcome outcome = Outcome::kSucceeded;
  std::uint64_t bytes = 0;
  // The provider proved the operation will touch its memory no more.
  bool no_further_access = false;
  bool operator==(const Terminal&) const = default;
};

// What the owner sees when it harvests a mailbox.
struct Observation {
  OperationId operation;
  Acceptance acceptance = Acceptance::kNone;
  std::optional<Terminal> terminal;
  bool contradictory = false;
};

enum class Published : std::uint8_t {
  kRecorded,
  kDuplicate,      // equal to what was already recorded
  kContradiction,  // differs from what was already recorded
  kStale,          // the identity names no open operation
};

// Owner-thread counters. Read only by that owner, or after it has joined.
// Locks count successful nonempty removals, excluding empty/final-more checks.
struct HarvestStats {
  std::uint64_t indices = 0;
  std::uint64_t pop_locks = 0;
};

class CompletionBoard {
 public:
  // `mailboxes` operations may be in flight at once; `wake` is the owner's.
  // Every mailbox's generation starts at `first_generation` (at least 1):
  // a test hook that reaches exhaustion quickly. Production code leaves it
  // at 1; a larger value only brings retirement sooner, since a fresh
  // board never issues an identity twice.
  CompletionBoard(std::size_t mailboxes, base::WakeFlag& wake, std::uint32_t first_generation = 1);
  CompletionBoard(const CompletionBoard&) = delete;
  CompletionBoard& operator=(const CompletionBoard&) = delete;
  CompletionBoard(CompletionBoard&&) = delete;
  CompletionBoard& operator=(CompletionBoard&&) = delete;
  ~CompletionBoard() = default;

  // The owner: a mailbox for a new operation, or an invalid identity when
  // every mailbox is in use (the caller does not submit).
  OperationId Open();
  // The owner: frees a mailbox once its operation is reconciled: proven not
  // started, or accepted (or unknown) with a terminal result that proves no
  // further access. A mailbox whose operation may still touch memory, or
  // whose observations contradict each other, stays open. False otherwise,
  // or if the identity is stale.
  bool Close(OperationId operation);

  // Any thread.
  Published Accept(OperationId operation, Acceptance acceptance);
  Published Complete(OperationId operation, const Terminal& terminal);

  // The owner: up to `limit` mailboxes with news, in the order they got it.
  // Each is reported once per burst of publications; a publication after
  // the harvest reports it again. News left over re-signals the owner, so
  // an owner that sleeps after a partial harvest still wakes for it.
  // "Not started" together with a terminal result is a contradiction.
  std::vector<Observation> Harvest(std::size_t limit);

  // Any thread: wakes the owner with no news, for a lane that made room in
  // its queue without publishing anything (a cancellation it took), so a
  // command the owner kept because that queue was full is offered again.
  void Nudge() { wake_.Signal(); }
  // Any thread: a publication is likely by `until` (a fence nearing its
  // expected end), so the owner should poll until then rather than sleep
  // (base::WakeFlag::Anticipate). A hint; it publishes nothing.
  void Anticipate(base::WakeFlag::Clock::time_point until) { wake_.Anticipate(until); }

  // Any thread: every new observation recorded so far (an acceptance, a
  // terminal result or a later proof), of any operation: a count that
  // moves whenever any lane completes anything (a fence seen complete, a
  // read or write landed, a VMM operation). The runtime's hang watch reads
  // it as progress (D-102; engine/paged_node.h PagedNode::progress).
  std::uint64_t publications() const { return publications_.load(std::memory_order_relaxed); }

  const HarvestStats& harvest_stats() const { return harvest_stats_; }

  std::size_t open() const { return open_count_; }
  std::size_t capacity() const { return count_; }
  // The owner: mailboxes Open can issue now. A mailbox whose generation is
  // exhausted is retired, so this can be zero with fewer than capacity()
  // open.
  std::size_t available() const { return free_count_; }
  // The owner: every mailbox is retired, so Open never issues one again.
  bool exhausted() const { return retired_ == count_; }

 private:
  // State words: the mailbox generation in the high bits, the rest in the
  // low byte. The terminal word also carries the outcome and the
  // no-further-access proof, so a reader needs one acquire load for all of
  // it; the byte count sits in its own atomic, read between two loads of
  // the terminal word.
  static constexpr std::uint64_t Word(std::uint32_t generation, std::uint8_t value) {
    return (std::uint64_t{generation} << 8) | value;
  }
  static constexpr std::uint32_t GenerationOf(std::uint64_t word) {
    return static_cast<std::uint32_t>(word >> 8);
  }
  static constexpr std::uint8_t ValueOf(std::uint64_t word) {
    return static_cast<std::uint8_t>(word & 0xff);
  }
  // Terminal word values: 0 empty, 1 being written, or kWritten with the
  // outcome and the proof.
  static constexpr std::uint8_t kEmpty = 0;
  static constexpr std::uint8_t kWriting = 1;
  static constexpr std::uint8_t kWritten = 2;
  // The contradiction word of a mailbox retired at its last generation: a
  // value no publisher sets.
  static constexpr std::uint8_t kRetiredMark = 2;
  static std::uint8_t Encode(const Terminal& terminal);
  static Terminal Decode(std::uint8_t value, std::uint64_t bytes, bool proven);

  struct Mailbox {
    std::atomic<std::uint64_t> acceptance{Word(1, 0)};
    std::atomic<std::uint64_t> terminal{Word(1, kEmpty)};
    std::atomic<std::uint64_t> bytes{0};
    std::atomic<std::uint64_t> proof{Word(1, 0)};          // Word(generation, 1) once proven
    std::atomic<std::uint64_t> contradictory{Word(1, 0)};  // Word(generation, 1) once contradicted
    std::atomic<bool> queued{false};                       // in the news queue
    std::uint32_t generation = 1;                          // owner only
    bool open = false;                                     // owner only
  };

  // The terminal result recorded for `generation`, if it is complete and
  // still current.
  static std::optional<Terminal> ReadTerminal(const Mailbox& mailbox, std::uint32_t generation);
  // The mailbox was closed at its last generation, `generation`: every
  // publication for it is stale.
  static bool IsRetired(const Mailbox& mailbox, std::uint32_t generation);
  void Announce(std::uint32_t index);
  // Marks the mailbox contradictory, if its generation is still current.
  Published Contradict(Mailbox& mailbox, std::uint32_t generation, std::uint32_t index);
  // Records the no-further-access proof; true if it is new.
  bool Prove(Mailbox& mailbox, std::uint32_t generation, std::uint32_t index);

  // Built in place, once: atomics do not move.
  std::vector<Mailbox> mailboxes_;
  std::size_t count_;
  base::WakeFlag& wake_;
  std::atomic<std::uint64_t> publications_{0};  // publications()
  // Owner only: free mailboxes, a ring of free_count_ entries from
  // free_head_, in the order they were freed.
  std::vector<std::uint32_t> free_;
  std::size_t free_head_ = 0;
  std::size_t free_count_ = 0;
  std::size_t open_count_ = 0;
  std::size_t retired_ = 0;
  HarvestStats harvest_stats_;  // owner only
  // The news queue: each mailbox at most once, so it never overflows.
  std::mutex news_mutex_;
  std::vector<std::uint32_t> news_;
  std::size_t news_head_ = 0;
  std::size_t news_count_ = 0;
};

}  // namespace jitllm::scheduler

#endif  // JITLLM_SCHEDULER_COMPLETIONS_H_
