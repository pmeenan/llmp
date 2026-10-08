// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "scheduler/completions.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "base/check.h"
#include "base/wake.h"

namespace jitllm::scheduler {
namespace {

constexpr std::uint8_t kStateMask = 0x3;

}  // namespace

std::uint8_t CompletionBoard::Encode(const Terminal& terminal) {
  return static_cast<std::uint8_t>(kWritten | (static_cast<std::uint8_t>(terminal.outcome) << 2));
}

Terminal CompletionBoard::Decode(std::uint8_t value, std::uint64_t bytes, bool proven) {
  return Terminal{.outcome = static_cast<Outcome>((value >> 2) & 0x3),
                  .bytes = bytes,
                  .no_further_access = proven};
}

CompletionBoard::CompletionBoard(std::size_t mailboxes, base::WakeFlag& wake,
                                 std::uint32_t first_generation)
    : mailboxes_(mailboxes),
      count_(mailboxes),
      wake_(wake),
      free_(mailboxes),
      free_count_(mailboxes),
      news_(mailboxes) {
  base::Check(mailboxes > 0 && mailboxes <= UINT32_MAX && first_generation > 0,
              "a completion board needs mailboxes and a valid first generation");
  for (std::size_t i = 0; i < mailboxes; ++i) {
    Mailbox& mailbox = mailboxes_[i];
    mailbox.generation = first_generation;
    mailbox.acceptance.store(Word(first_generation, 0), std::memory_order_relaxed);
    mailbox.terminal.store(Word(first_generation, kEmpty), std::memory_order_relaxed);
    mailbox.proof.store(Word(first_generation, 0), std::memory_order_relaxed);
    mailbox.contradictory.store(Word(first_generation, 0), std::memory_order_relaxed);
    free_[i] = static_cast<std::uint32_t>(i);
  }
}

OperationId CompletionBoard::Open() {
  if (free_count_ == 0) {
    return {};  // every mailbox open or retired
  }
  const std::uint32_t index = free_[free_head_];
  free_head_ = (free_head_ + 1) % count_;
  --free_count_;
  Mailbox& mailbox = mailboxes_[index];
  mailbox.open = true;
  ++open_count_;
  return {index, mailbox.generation};
}

bool CompletionBoard::Close(OperationId operation) {
  if (!operation.valid() || operation.index() >= count_) {
    return false;
  }
  Mailbox& mailbox = mailboxes_[operation.index()];
  if (!mailbox.open || mailbox.generation != operation.generation()) {
    return false;
  }
  const std::uint32_t generation = mailbox.generation;
  const std::uint64_t acceptance = mailbox.acceptance.load(std::memory_order_acquire);
  const std::optional<Terminal> terminal = ReadTerminal(mailbox, generation);
  const bool not_started =
      acceptance == Word(generation, static_cast<std::uint8_t>(Acceptance::kNotStarted));
  const bool started = GenerationOf(acceptance) == generation &&
                       (ValueOf(acceptance) == static_cast<std::uint8_t>(Acceptance::kAccepted) ||
                        ValueOf(acceptance) == static_cast<std::uint8_t>(Acceptance::kUnknown));
  const bool reconciled =
      not_started ? !terminal : started && terminal && terminal->no_further_access;
  if (!reconciled) {
    return false;
  }
  // A not-started operation has no terminal result; freeze its terminal
  // word, so a late Complete loses to Close (and is stale) instead of being
  // recorded and then overwritten.
  if (not_started) {
    std::uint64_t empty = Word(generation, kEmpty);
    if (!mailbox.terminal.compare_exchange_strong(empty, Word(generation, kWriting),
                                                  std::memory_order_acq_rel)) {
      return false;  // a result arrived: the owner harvests the contradiction
    }
  }
  // The gate: a contradiction lands before this or not at all. A slot whose
  // generation cannot advance is retired, with a marker no one can set.
  const bool retired = generation == UINT32_MAX;
  const std::uint32_t next = retired ? generation : generation + 1;
  std::uint64_t clean = Word(generation, 0);
  if (!mailbox.contradictory.compare_exchange_strong(
          clean, retired ? Word(generation, kRetiredMark) : Word(next, 0),
          std::memory_order_acq_rel)) {
    if (not_started) {
      mailbox.terminal.store(Word(generation, kEmpty), std::memory_order_release);  // thaw
    }
    return false;  // contradicted: stays open for the fault
  }
  mailbox.open = false;
  --open_count_;
  if (retired) {
    ++retired_;
    if (not_started) {
      // Release anyone waiting on the frozen word: they find the marker set
      // and report stale.
      mailbox.terminal.store(Word(generation, kWritten), std::memory_order_release);
    }
    return true;
  }
  mailbox.generation = next;
  // From here on every publication for the old generation is stale. The
  // byte count is left for the next result to overwrite.
  mailbox.terminal.store(Word(next, kEmpty), std::memory_order_release);
  mailbox.proof.store(Word(next, 0), std::memory_order_release);
  mailbox.acceptance.store(Word(next, static_cast<std::uint8_t>(Acceptance::kNone)),
                           std::memory_order_release);
  // Behind every other free mailbox: rotation spreads generations evenly.
  // At most every mailbox is free, so the ring never overflows.
  free_[(free_head_ + free_count_) % count_] = operation.index();
  ++free_count_;
  return true;
}

bool CompletionBoard::IsRetired(const Mailbox& mailbox, std::uint32_t generation) {
  return mailbox.contradictory.load(std::memory_order_acquire) == Word(generation, kRetiredMark);
}

std::optional<Terminal> CompletionBoard::ReadTerminal(const Mailbox& mailbox,
                                                      std::uint32_t generation) {
  const std::uint64_t word = mailbox.terminal.load(std::memory_order_acquire);
  if (GenerationOf(word) != generation || (ValueOf(word) & kStateMask) != kWritten) {
    return std::nullopt;
  }
  const std::uint64_t bytes = mailbox.bytes.load(std::memory_order_relaxed);
  // The byte count belongs to this result only if the word did not change.
  std::atomic_thread_fence(std::memory_order_acquire);
  if (mailbox.terminal.load(std::memory_order_relaxed) != word) {
    return std::nullopt;
  }
  return Decode(ValueOf(word), bytes,
                mailbox.proof.load(std::memory_order_acquire) == Word(generation, 1));
}

void CompletionBoard::Announce(std::uint32_t index) {
  publications_.fetch_add(1, std::memory_order_relaxed);
  if (mailboxes_[index].queued.exchange(true, std::memory_order_acq_rel)) {
    return;  // already queued; the owner reads it after this publication
  }
  {
    const std::scoped_lock lock(news_mutex_);
    base::Check(news_count_ < count_, "completion news overflow");
    news_[(news_head_ + news_count_) % count_] = index;
    ++news_count_;
  }
  wake_.Signal();
}

Published CompletionBoard::Contradict(Mailbox& mailbox, std::uint32_t generation,
                                      std::uint32_t index) {
  // Only for this generation: a delayed publisher of an older one cannot
  // touch the marker of the mailbox's next operation, and one that loses to
  // Close is stale.
  std::uint64_t expected = Word(generation, 0);
  if (mailbox.contradictory.compare_exchange_strong(expected, Word(generation, 1),
                                                    std::memory_order_acq_rel)) {
    Announce(index);
    return Published::kContradiction;
  }
  return expected == Word(generation, 1) ? Published::kContradiction : Published::kStale;
}

bool CompletionBoard::Prove(Mailbox& mailbox, std::uint32_t generation, std::uint32_t index) {
  std::uint64_t expected = Word(generation, 0);
  if (!mailbox.proof.compare_exchange_strong(expected, Word(generation, 1),
                                             std::memory_order_acq_rel)) {
    return false;  // already proven, or the operation is gone
  }
  Announce(index);
  return true;
}

Published CompletionBoard::Accept(OperationId operation, Acceptance acceptance) {
  base::Check(acceptance != Acceptance::kNone, "acceptance must resolve the submission");
  if (!operation.valid() || operation.index() >= count_) {
    return Published::kStale;
  }
  Mailbox& mailbox = mailboxes_[operation.index()];
  const std::uint32_t generation = operation.generation();
  const auto value = static_cast<std::uint8_t>(acceptance);
  std::uint64_t expected = Word(generation, static_cast<std::uint8_t>(Acceptance::kNone));
  if (mailbox.acceptance.compare_exchange_strong(expected, Word(generation, value),
                                                 std::memory_order_acq_rel,
                                                 std::memory_order_acquire)) {
    Announce(operation.index());
    return Published::kRecorded;
  }
  if (GenerationOf(expected) != generation || IsRetired(mailbox, generation)) {
    return Published::kStale;
  }
  if (ValueOf(expected) == value) {
    return Published::kDuplicate;
  }
  return Contradict(mailbox, generation, operation.index());
}

Published CompletionBoard::Complete(OperationId operation, const Terminal& terminal) {
  if (!operation.valid() || operation.index() >= count_) {
    return Published::kStale;
  }
  Mailbox& mailbox = mailboxes_[operation.index()];
  const std::uint32_t generation = operation.generation();
  std::uint64_t expected = Word(generation, kEmpty);
  while (!mailbox.terminal.compare_exchange_strong(expected, Word(generation, kWriting),
                                                   std::memory_order_acquire)) {
    // Another publication for this operation is being written, or Close
    // froze the word; wait for either to settle.
    while (GenerationOf(expected) == generation && (ValueOf(expected) & kStateMask) == kWriting) {
      std::this_thread::yield();
      expected = mailbox.terminal.load(std::memory_order_acquire);
    }
    if (GenerationOf(expected) != generation) {
      return Published::kStale;
    }
    if ((ValueOf(expected) & kStateMask) != kEmpty) {
      break;  // a result is recorded: compare below
    }
    // Close thawed the word (the mailbox stays open): try again.
  }
  if ((ValueOf(expected) & kStateMask) == kEmpty) {
    // A sequence lock: a reader that sees these bytes also sees the word
    // change (ReadTerminal).
    std::atomic_thread_fence(std::memory_order_release);
    mailbox.bytes.store(terminal.bytes, std::memory_order_relaxed);
    if (terminal.no_further_access) {
      std::uint64_t unproven = Word(generation, 0);
      (void)mailbox.proof.compare_exchange_strong(unproven, Word(generation, 1),
                                                  std::memory_order_acq_rel);
    }
    mailbox.terminal.store(Word(generation, Encode(terminal)), std::memory_order_release);
    Announce(operation.index());
    return Published::kRecorded;
  }
  const std::optional<Terminal> recorded = ReadTerminal(mailbox, generation);
  if (!recorded || IsRetired(mailbox, generation)) {
    return Published::kStale;  // closed (and reused, or retired) while we looked
  }
  if (recorded->outcome == terminal.outcome && recorded->bytes == terminal.bytes) {
    // The proof may follow the result; it is never withdrawn.
    return terminal.no_further_access && Prove(mailbox, generation, operation.index())
               ? Published::kRecorded
               : Published::kDuplicate;
  }
  return Contradict(mailbox, generation, operation.index());
}

std::vector<Observation> CompletionBoard::Harvest(std::size_t limit) {
  std::vector<Observation> observations;
  std::array<std::uint32_t, 64> batch{};
  while (observations.size() < limit) {
    std::size_t taken = 0;
    {
      const std::scoped_lock lock(news_mutex_);
      taken = std::min({batch.size(), limit - observations.size(), news_count_});
      if (taken == 0) break;
      for (std::size_t i = 0; i < taken; ++i) {
        batch[i] = news_[news_head_];
        news_head_ = (news_head_ + 1) % count_;
      }
      news_count_ -= taken;
      ++harvest_stats_.pop_locks;
      harvest_stats_.indices += taken;
    }
    // Take only existing news, without waiting to fill the local batch.
    // Publishers can keep recording while the owner reads these mailboxes.
    // Their queued flags remain set until each index is processed below.
    for (std::size_t i = 0; i < taken; ++i) {
      const std::uint32_t index = batch[i];
      Mailbox& mailbox = mailboxes_[index];
      // Cleared before reading, so a later publication queues it again.
      mailbox.queued.exchange(false, std::memory_order_acq_rel);
      if (!mailbox.open) {
        continue;  // news of an operation the owner already closed
      }
      const std::uint32_t generation = mailbox.generation;
      const std::uint64_t acceptance = mailbox.acceptance.load(std::memory_order_acquire);
      Observation seen{
          .operation = OperationId(index, generation),
          .acceptance = GenerationOf(acceptance) == generation
                            ? static_cast<Acceptance>(ValueOf(acceptance))
                            : Acceptance::kNone,
          .terminal = ReadTerminal(mailbox, generation),
          .contradictory =
              mailbox.contradictory.load(std::memory_order_acquire) == Word(generation, 1),
      };
      seen.contradictory =
          seen.contradictory || (seen.acceptance == Acceptance::kNotStarted && seen.terminal);
      // A mailbox queued again during this harvest is reported once, as last read.
      const auto earlier = std::ranges::find(observations, seen.operation, &Observation::operation);
      if (earlier != observations.end()) {
        *earlier = seen;
      } else {
        observations.push_back(seen);
      }
    }
  }
  // A partial harvest leaves news behind: keep the owner awake for it.
  bool more = false;
  {
    const std::scoped_lock lock(news_mutex_);
    more = news_count_ > 0;
  }
  if (more) {
    wake_.Signal();
  }
  return observations;
}

}  // namespace jitllm::scheduler
