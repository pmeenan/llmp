// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The keeper of conversation records (D-105; kept_record.h): what makes a
// spilled conversation survive a restart, off the driver's path.
//
// A record exists for a request slot only while the slot's state is wholly
// on disk and nothing has written its spill file since: the driver tells
// the keeper before anything may change the file or make the state live
// again (Invalidate: a restore, a swap bringing the model in, a clear, a
// discard), and after a write of the whole state completed (Keep: a
// spill, a swap's write-back). Keep queues the record; a worker thread
// then syncs the files, computes the digests of what they hold (each used
// extent of the spill file, each page of the turn checkpoints, which are
// immutable and hashed once), and writes the record atomically, unless an
// Invalidate came meanwhile (a sequence number a slot: the record would
// describe a file that has changed since). The record is written and
// synced beside its name off the lock, renamed over it under the lock if
// its slot's sequence still holds, and its directory synced after, so the
// driver never waits for hashing or a sync. A crash leaves a record that
// describes its files exactly, or none, or one whose removal had not yet
// reached the device (Invalidate syncs nothing): its file may have changed
// since, which the next start's check of every listed extent refuses, and
// its adoption empties every extent the record does not list (D-105).
//
// Vendor-free: the files are named beneath each model's private directory
// (platform/kept_files.h), read with direct I/O. The CPU tests drive it
// with files of their own.

#ifndef JITLLM_RUNTIME_STATE_KEEPER_H_
#define JITLLM_RUNTIME_STATE_KEEPER_H_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "runtime/kept_record.h"

namespace jitllm::runtime {

// The SHA-256 of each place in the open file `fd`, read with direct I/O
// (each place 4 KiB-aligned and at most kept::kExtentBytes), over at most
// `threads` threads; none for a place that could not be read whole.
// `hashed`, if given, counts the bytes as they are hashed (progress).
// `go_on`, if given, is asked before each place (it may wait); false stops
// the hashing (the places left get none). `background`: each thread runs
// at the lowest CPU and I/O priority (platform::LowerThreadPriority).
std::vector<std::optional<kept::Digest>> HashPlaces(int fd, std::span<const kept::Place> places,
                                                    std::size_t threads,
                                                    std::atomic<std::uint64_t>* hashed = nullptr,
                                                    const std::function<bool()>& go_on = {},
                                                    bool background = false);

// Empties what of a kept spill file its record does not list (kept::
// UnlistedPlaces), beneath the model's private directory `dir`: the very
// file the record names (its identity and size), opened again. Those
// extents then read as zeros, as a fresh slot's do. Why not, or empty.
std::string EmptyUnlisted(int dir, const kept::Record& record);

class StateKeeper {
 public:
  using Clock = std::chrono::steady_clock;
  using Log = std::function<void(std::string_view)>;

  // `threads`: how many hash one record's files at once.
  StateKeeper(std::size_t threads, Log log);
  StateKeeper(const StateKeeper&) = delete;
  StateKeeper& operator=(const StateKeeper&) = delete;
  StateKeeper(StateKeeper&&) = delete;
  StateKeeper& operator=(StateKeeper&&) = delete;
  // Stops the worker: what is still queued is abandoned (no record).
  ~StateKeeper();

  // A model's private directory, taken (closed at the keeper's end): its
  // index for the calls below. Before the first Keep.
  std::size_t AddModel(int directory);
  int directory(std::size_t model) const { return models_.at(model); }

  // The driver: slot `slot`'s spill file may change, or its state become
  // live again: its record is removed now, and a record still being made
  // for it is not written.
  void Invalidate(std::size_t model, std::uint32_t slot);
  // The driver: the slot's spill file holds `record`'s state, written
  // whole and completed; its digests (and its checkpoints', unless known)
  // are computed and the record written in the background.
  void Keep(std::size_t model, kept::Record record);
  // A record adopted at start: it exists (Invalidate removes it); its
  // checkpoints' digests are known.
  void Adopted(std::size_t model, const kept::Record& record);
  // Whether a slot has a record now.
  bool Kept(std::size_t model, std::uint32_t slot) const;

  // Waits until every queued record is written or dropped, or `deadline`;
  // true if none is left. Hashing no longer yields (Quiet) once asked.
  bool Drain(Clock::time_point deadline);

  // The driver's: while a Quiet lives (a swap, a prefill), and kLinger
  // after the last ends, the hashing waits between places, so its reads
  // and cores stay out of the way of the model's own (it runs in the
  // background otherwise, at the lowest priority). Nests. Null: nothing.
  class Quiet {
   public:
    explicit Quiet(StateKeeper* keeper) : keeper_(keeper) {
      if (keeper_ != nullptr) {
        keeper_->quiet_.fetch_add(1, std::memory_order_acq_rel);
      }
    }
    Quiet(const Quiet&) = delete;
    Quiet& operator=(const Quiet&) = delete;
    Quiet(Quiet&&) = delete;
    Quiet& operator=(Quiet&&) = delete;
    ~Quiet() {
      if (keeper_ != nullptr) {
        keeper_->Unquiet();
      }
    }

   private:
    StateKeeper* keeper_;
  };
  static constexpr std::chrono::milliseconds kLinger{250};

  struct Stats {
    std::uint64_t written = 0;  // records written
    std::uint64_t stale = 0;    // dropped: invalidated while being made
    std::uint64_t failed = 0;   // dropped: a file could not be read or written
    std::uint64_t hashed_bytes = 0;
    double hash_seconds = 0;
  };
  Stats stats() const;
  std::size_t pending() const;

 private:
  struct Job {
    std::size_t model = 0;
    std::uint64_t sequence = 0;
    kept::Record record;
  };
  struct SlotState {
    std::uint64_t sequence = 0;  // advanced by every Invalidate
    bool kept = false;           // its record is on disk
  };
  using SlotKey = std::pair<std::size_t, std::uint32_t>;

  void Work(const std::stop_token& stop);
  // Hashes the job's files (no lock held); false with why if they do not
  // hold what it says, or if it went stale meanwhile (`stale` set).
  // `seen`: the Invalidates counted when its sequence was last checked.
  bool Hash(Job& job, std::uint64_t seen, const std::stop_token& stop, std::string& why,
            bool& stale);
  // Between places: waits while quiet (unless draining); false once the
  // job went stale (an Invalidate of its slot) or the keeper stops.
  bool GoOn(const Job& job, std::uint64_t seen, const std::stop_token& stop);
  void Unquiet();
  void Say(std::string_view line) const;

  const std::size_t threads_;
  Log log_;
  // Read unlocked in Hash and Work: safe only as AddModel runs at startup, before any Keep.
  std::vector<int> models_;
  mutable std::mutex mutex_;
  std::condition_variable_any ready_;
  std::condition_variable idle_;
  std::deque<Job> queue_;
  bool busy_ = false;
  std::map<SlotKey, SlotState> slots_;
  // Checkpoint pages' digests by file and identity: they never change.
  std::map<std::pair<std::size_t, std::string>, std::pair<kept::FileId, std::vector<kept::Digest>>>
      checkpoint_digests_;
  Stats stats_;
  std::atomic<std::uint64_t> hashed_{0};  // Stats::hashed_bytes, as it goes
  // Invalidates so far: a job compares it with the count read with its
  // sequence check before it looks up its slot's sequence (under the lock).
  std::atomic<std::uint64_t> invalidations_{0};
  std::atomic<int> quiet_{0};                    // Quiets alive
  std::atomic<std::int64_t> quiet_until_ns_{0};  // the linger after the last
  std::atomic<bool> draining_{false};            // Drain asked: yield no more
  std::jthread worker_;
};

}  // namespace jitllm::runtime

#endif  // JITLLM_RUNTIME_STATE_KEEPER_H_
