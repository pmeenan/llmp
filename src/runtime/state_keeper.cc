// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/state_keeper.h"

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "base/sha256.h"
#include "platform/direct_io.h"
#include "platform/kept_files.h"

namespace jitllm::runtime {
namespace {

struct FreeAligned {
  void operator()(std::byte* p) const { std::free(p); }  // NOLINT(cppcoreguidelines-no-malloc)
};

std::string Errno(int error) { return std::system_category().message(error); }

kept::FileId IdOf(const platform::FileIdentity& id) {
  return {.device = id.device, .inode = id.inode, .generation = id.generation};
}

}  // namespace

std::vector<std::optional<kept::Digest>> HashPlaces(int fd, std::span<const kept::Place> places,
                                                    std::size_t threads,
                                                    std::atomic<std::uint64_t>* hashed,
                                                    const std::function<bool()>& go_on,
                                                    bool background) {
  std::vector<std::optional<kept::Digest>> digests(places.size());
  std::atomic<std::size_t> next{0};
  std::atomic<bool> stopped{false};
  const auto work = [&](bool helper) {
    if (background && helper) {
      (void)platform::LowerThreadPriority();
    }
    // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
    std::unique_ptr<std::byte, FreeAligned> buffer(static_cast<std::byte*>(
        std::aligned_alloc(platform::kDirectIoAlignment, kept::kExtentBytes)));
    if (buffer == nullptr) {
      return;
    }
    for (std::size_t i = next.fetch_add(1); i < places.size(); i = next.fetch_add(1)) {
      if (stopped.load(std::memory_order_relaxed) || (go_on && !go_on())) {
        stopped.store(true, std::memory_order_relaxed);
        return;
      }
      const kept::Place& place = places[i];
      if (place.bytes == 0 || place.bytes > kept::kExtentBytes ||
          place.bytes % platform::kDirectIoAlignment != 0 ||
          place.offset % platform::kDirectIoAlignment != 0) {
        continue;
      }
      const std::span<std::byte> bytes(buffer.get(), static_cast<std::size_t>(place.bytes));
      if (!platform::TransferDirectFile(fd, place.offset, bytes, false)) {
        continue;
      }
      digests[i] = base::Sha256().Update(bytes).Finish();
      if (hashed != nullptr) {
        hashed->fetch_add(place.bytes, std::memory_order_relaxed);
      }
    }
  };
  const std::size_t helpers = std::min(std::max<std::size_t>(threads, 1), places.size());
  std::vector<std::jthread> pool;
  for (std::size_t i = 1; i < helpers; ++i) {
    pool.emplace_back(work, true);
  }
  work(false);
  pool.clear();  // joined
  return digests;
}

StateKeeper::StateKeeper(std::size_t threads, Log log)
    : threads_(std::max<std::size_t>(threads, 1)),
      log_(std::move(log)),
      worker_([this](const std::stop_token& stop) { Work(stop); }) {}

StateKeeper::~StateKeeper() {
  worker_.request_stop();
  ready_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
  for (const int fd : models_) {
    (void)::close(fd);
  }
}

void StateKeeper::Say(std::string_view line) const {
  if (log_) {
    log_(line);
  }
}

std::size_t StateKeeper::AddModel(int directory) {
  const std::scoped_lock lock(mutex_);
  models_.push_back(directory);
  return models_.size() - 1;
}

void StateKeeper::Invalidate(std::size_t model, std::uint32_t slot) {
  const std::scoped_lock lock(mutex_);
  SlotState& state = slots_[{model, slot}];
  ++state.sequence;
  invalidations_.fetch_add(1, std::memory_order_release);
  if (state.kept) {
    const std::string name = kept::RecordFileName(slot);
    if (auto removed = platform::RemovePrivate(models_.at(model), name.c_str()); !removed) {
      // A record that cannot be removed would describe a file about to
      // change: say so (the next start refuses it if its digests no longer
      // match, as they then will not).
      Say(std::format("a kept conversation's record {} could not be removed: {}", name,
                      Errno(removed.error())));
    }
    state.kept = false;
  }
}

void StateKeeper::Keep(std::size_t model, kept::Record record) {
  {
    const std::scoped_lock lock(mutex_);
    const SlotState& state = slots_[{model, record.slot}];
    queue_.push_back({.model = model, .sequence = state.sequence, .record = std::move(record)});
  }
  ready_.notify_one();
}

void StateKeeper::Adopted(std::size_t model, const kept::Record& record) {
  const std::scoped_lock lock(mutex_);
  slots_[{model, record.slot}].kept = true;
  for (const kept::Checkpoint& c : record.checkpoints) {
    checkpoint_digests_[{model, c.file}] = {c.id, c.digests};
  }
}

bool StateKeeper::Kept(std::size_t model, std::uint32_t slot) const {
  const std::scoped_lock lock(mutex_);
  const auto found = slots_.find({model, slot});
  return found != slots_.end() && found->second.kept;
}

bool StateKeeper::Drain(Clock::time_point deadline) {
  draining_.store(true, std::memory_order_release);
  std::unique_lock lock(mutex_);
  return idle_.wait_until(lock, deadline, [this] { return queue_.empty() && !busy_; });
}

void StateKeeper::Unquiet() {
  const auto until = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         (Clock::now() + kLinger).time_since_epoch())
                         .count();
  quiet_until_ns_.store(until, std::memory_order_release);
  quiet_.fetch_sub(1, std::memory_order_acq_rel);
}

bool StateKeeper::GoOn(const Job& job, std::uint64_t seen, const std::stop_token& stop) {
  for (;;) {
    if (stop.stop_requested()) {
      return false;
    }
    if (invalidations_.load(std::memory_order_acquire) != seen) {
      const std::scoped_lock lock(mutex_);
      const auto found = slots_.find({job.model, job.record.slot});
      if (found != slots_.end() && found->second.sequence != job.sequence) {
        return false;  // stale: its file may change; stop reading it now
      }
    }
    const auto now =
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
            .count();
    const bool quiet = quiet_.load(std::memory_order_acquire) > 0 ||
                       now < quiet_until_ns_.load(std::memory_order_acquire);
    if (!quiet || draining_.load(std::memory_order_acquire)) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

StateKeeper::Stats StateKeeper::stats() const {
  const std::scoped_lock lock(mutex_);
  Stats stats = stats_;
  stats.hashed_bytes = hashed_.load(std::memory_order_relaxed);
  return stats;
}

std::size_t StateKeeper::pending() const {
  const std::scoped_lock lock(mutex_);
  return queue_.size() + (busy_ ? 1 : 0);
}

bool StateKeeper::Hash(Job& job, const std::stop_token& stop, std::string& why, bool& stale) {
  kept::Record& r = job.record;
  const int dir = models_.at(job.model);
  const auto started = Clock::now();
  const std::uint64_t seen = invalidations_.load(std::memory_order_acquire);
  const std::function<bool()> go_on = [&]() { return GoOn(job, seen, stop); };
  stale = false;
  {
    auto file = platform::OpenPrivateFile(
        dir, r.file.c_str(), {.write = false, .create = false, .truncate = false, .direct = true});
    if (!file) {
      why = std::format("{}: {}", r.file, Errno(file.error()));
      return false;
    }
    const int fd = file->fd;
    const auto close = [fd] { (void)::close(fd); };
    if (IdOf(file->identity) != r.id || file->bytes != r.file_bytes) {
      close();
      why = std::format("{} is not the file the record was made for", r.file);
      return false;
    }
    // What the writes put there reaches the device before a record says so.
    if (auto synced = platform::SyncFileData(fd); !synced) {
      close();
      why = std::format("{}: {}", r.file, Errno(synced.error()));
      return false;
    }
    std::vector<kept::Place> places;
    places.reserve(r.extents.size());
    for (const kept::Extent& extent : r.extents) {
      places.push_back(
          {.offset = kept::ExtentOffset(r.regions, extent), .bytes = kept::kExtentBytes});
    }
    const auto digests = HashPlaces(fd, places, threads_, &hashed_, go_on, true);
    close();
    for (std::size_t i = 0; i < digests.size(); ++i) {
      const std::optional<kept::Digest>& digest = digests[i];
      if (!digest.has_value()) {
        if (!go_on()) {
          stale = true;  // stopped: invalidated, or the keeper stops
          return false;
        }
        why = std::format("{} could not be read whole", r.file);
        return false;
      }
      r.extents[i].digest = *digest;
    }
  }
  // Each checkpoint once: its pages never change after its capture.
  std::vector<kept::Checkpoint> checkpoints;
  for (kept::Checkpoint& c : r.checkpoints) {
    {
      const std::scoped_lock lock(mutex_);
      const auto known = checkpoint_digests_.find({job.model, c.file});
      if (known != checkpoint_digests_.end() && known->second.first == c.id &&
          known->second.second.size() == c.ranges.size()) {
        c.digests = known->second.second;
        checkpoints.push_back(std::move(c));
        continue;
      }
    }
    auto file = platform::OpenPrivateFile(
        dir, c.file.c_str(), {.write = false, .create = false, .truncate = false, .direct = true});
    if (!file) {
      continue;  // dropped since (its turn rolled back past it): kept without it
    }
    const int fd = file->fd;
    if (IdOf(file->identity) != c.id || file->bytes != c.file_bytes ||
        !platform::SyncFileData(fd)) {
      (void)::close(fd);
      continue;
    }
    const std::vector<kept::Place> places = kept::CheckpointPlaces(c);
    const auto digests = HashPlaces(fd, places, threads_, &hashed_, go_on, true);
    (void)::close(fd);
    c.digests.clear();
    for (const std::optional<kept::Digest>& digest : digests) {
      if (!digest.has_value()) {
        break;
      }
      c.digests.push_back(*digest);
    }
    if (c.digests.size() != digests.size()) {
      continue;  // a page could not be read whole: kept without it
    }
    {
      const std::scoped_lock lock(mutex_);
      checkpoint_digests_[{job.model, c.file}] = {c.id, c.digests};
    }
    checkpoints.push_back(std::move(c));
  }
  r.checkpoints = std::move(checkpoints);
  {
    const std::scoped_lock lock(mutex_);
    stats_.hash_seconds += std::chrono::duration<double>(Clock::now() - started).count();
  }
  return true;
}

void StateKeeper::Work(const std::stop_token& stop) {
  // Background work: the model's reads and the driver's cores come first.
  (void)platform::LowerThreadPriority();
  for (;;) {
    Job job;
    {
      std::unique_lock lock(mutex_);
      if (!ready_.wait(lock, stop, [this] { return !queue_.empty(); })) {
        return;  // stopping
      }
      job = std::move(queue_.front());
      queue_.pop_front();
      if (slots_[{job.model, job.record.slot}].sequence != job.sequence) {
        ++stats_.stale;
        if (queue_.empty()) {
          idle_.notify_all();
        }
        continue;
      }
      busy_ = true;
    }
    std::string why;
    bool stale = false;
    const bool hashed = Hash(job, stop, why, stale);
    {
      const std::scoped_lock lock(mutex_);
      SlotState& state = slots_[{job.model, job.record.slot}];
      if (stale || state.sequence != job.sequence) {
        ++stats_.stale;
      } else if (!hashed) {
        ++stats_.failed;
        Say(std::format("a kept conversation's record was not written: {}", why));
      } else {
        const std::string name = kept::RecordFileName(job.record.slot);
        if (auto written = platform::ReplacePrivateFile(models_.at(job.model), name.c_str(),
                                                        kept::Encode(job.record));
            !written) {
          ++stats_.failed;
          Say(std::format("a kept conversation's record {} was not written: {}", name,
                          Errno(written.error())));
        } else {
          state.kept = true;
          ++stats_.written;
          // The slot's checkpoints the record does not name are gone.
          const std::string prefix = std::format("slot-{}.turn-", job.record.slot);
          std::erase_if(checkpoint_digests_, [&](const auto& entry) {
            return entry.first.first == job.model && entry.first.second.starts_with(prefix) &&
                   std::ranges::none_of(job.record.checkpoints, [&](const kept::Checkpoint& c) {
                     return c.file == entry.first.second;
                   });
          });
        }
      }
      busy_ = false;
    }
    idle_.notify_all();
  }
}

}  // namespace jitllm::runtime
