// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The deterministic storage fake (docs/architecture.md#providers): files
// held in memory, and completions whose order, results and acceptance a
// test scripts. It checks direct-I/O alignment the way the kernel does
// (memory, offset and length multiples of the alignment, else -EINVAL),
// and moves data only when a request completes, so a reader that uses
// memory before its completion sees what was there before. A vectored
// request's count fills its segments in order, as the kernel's does. It
// never blocks.

#ifndef LLMP_PROVIDERS_FAKE_FAKE_STORAGE_H_
#define LLMP_PROVIDERS_FAKE_FAKE_STORAGE_H_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <span>
#include <vector>

#include "providers/storage.h"

namespace llmp::providers::fake {

class FakeStorage final : public Storage {
 public:
  FakeStorage(std::size_t depth, std::uint32_t alignment);

  // A file with these contents; returns its descriptor.
  int AddFile(std::vector<std::byte> contents);
  std::span<const std::byte> Contents(int fd) const;

  // How the next submission it has room for goes.
  struct Script {
    Submission submission = Submission::kAccepted;
    // Overrides the result: a short count, or -errno.
    std::optional<std::int64_t> result;
    // Held requests complete only when released (or cancelled).
    bool hold = false;
  };
  void ScriptNext(const Script& script) { scripts_.push_back(script); }
  // Lets a held request complete at the next harvest.
  bool Release(std::uint64_t token);

  // Changes how many requests it takes at once: fewer than are in flight
  // refuses new ones until enough complete.
  void SetDepth(std::size_t depth) { depth_ = depth; }

  std::size_t depth() const override { return depth_; }
  std::size_t in_flight() const override { return requests_.size(); }
  Submission Submit(const IoRequest& request) override;
  // A held request is cancelled (it completes with -ECANCELED and moves no
  // data); one already due completes as it would have.
  Submission Cancel(std::uint64_t token) override;
  std::size_t Harvest(std::span<IoCompletion> out, bool wait) override;
  void Wake() override {}  // it never waits

  // What was submitted, in order, for tests to inspect.
  const std::vector<IoRequest>& submitted() const { return submitted_; }

 private:
  struct Pending {
    IoRequest request;
    std::optional<std::int64_t> result;
    bool held = false;
    bool cancelled = false;
    std::uint64_t sequence = 0;
  };
  std::int64_t Perform(const Pending& pending);

  std::size_t depth_;
  std::uint32_t alignment_;
  std::map<int, std::vector<std::byte>> files_;
  int next_fd_ = 1000;
  std::deque<Script> scripts_;
  std::map<std::uint64_t, Pending> requests_;  // by token
  std::uint64_t next_sequence_ = 0;
  std::vector<IoRequest> submitted_;
  std::deque<std::vector<IoSegment>> segments_;  // vectored requests', for their lifetime
};

}  // namespace llmp::providers::fake

#endif  // LLMP_PROVIDERS_FAKE_FAKE_STORAGE_H_
