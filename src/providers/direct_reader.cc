// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "providers/direct_reader.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "base/check.h"
#include "providers/storage.h"

namespace llmp::providers {

std::string ToString(ReadError error) {
  switch (error) {
    case ReadError::kFull:
      return "as many reads are in flight as allowed";
    case ReadError::kUnaligned:
      return "the memory, offset or length is not aligned for direct I/O";
    case ReadError::kInvalidRange:
      return "the file or memory range cannot be represented";
    case ReadError::kMismatch:
      return "the key is being read into a different range or memory";
    case ReadError::kTooManyWaiters:
      return "the read has as many waiters as allowed";
    case ReadError::kDraining:
      return "the read was cancelled and is still draining";
    case ReadError::kUnknownRead:
      return "no such read or waiter";
  }
  return "unknown read error";
}

DirectReader::DirectReader(Storage& storage, ReaderSettings settings)
    : storage_(storage), settings_(settings) {
  base::Check(settings.alignment > 0 && settings.request_bytes > 0 &&
                  settings.request_bytes % settings.alignment == 0 && settings.reads > 0 &&
                  settings.waiters > 0,
              "direct reads need aligned non-zero requests and non-zero read and waiter limits");
  // A span's count comes back as a 32-bit result (io_uring): 1 GiB bounds
  // it well inside that.
  base::Check(settings.span_segments > 0 && settings.span_segments <= kMaxSegments &&
                  settings.span_bytes <= (1U << 30U),
              "coalesced reads need 1 to kMaxSegments segments and at most 1 GiB");
  run_.reserve(settings.span_segments);
  segments_.reserve(settings.span_segments);
}

DirectReader::~DirectReader() {
  base::Check(requests_.empty(), "a direct reader destroyed with requests in flight");
}

std::expected<bool, ReadError> DirectReader::Read(std::uint64_t key, const ReadSpec& spec,
                                                  std::uint64_t waiter) {
  if (const auto found = reads_.find(key); found != reads_.end()) {
    Reading& reading = found->second;
    if (reading.spec != spec) {
      return std::unexpected(ReadError::kMismatch);
    }
    if (reading.stopping && reading.waiters.empty()) {
      return std::unexpected(ReadError::kDraining);
    }
    if (std::ranges::find(reading.waiters, waiter) != reading.waiters.end()) {
      return true;
    }
    if (reading.waiters.size() >= settings_.waiters) {
      return std::unexpected(ReadError::kTooManyWaiters);
    }
    reading.waiters.push_back(waiter);
    return true;
  }
  const auto address = reinterpret_cast<std::uintptr_t>(spec.memory);
  if (spec.memory == nullptr || spec.length == 0 || address % settings_.alignment != 0 ||
      spec.offset % settings_.alignment != 0 || spec.length % settings_.alignment != 0) {
    return std::unexpected(ReadError::kUnaligned);
  }
  if (spec.length > std::numeric_limits<std::uint64_t>::max() - spec.offset ||
      spec.length > std::numeric_limits<std::uintptr_t>::max() - address ||
      spec.length > static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
    return std::unexpected(ReadError::kInvalidRange);
  }
  if (reads_.size() >= settings_.reads) {
    return std::unexpected(ReadError::kFull);
  }
  Reading reading;
  reading.spec = spec;
  reading.arrival = next_arrival_++;
  reading.waiters.push_back(waiter);
  for (std::uint64_t start = 0; start < spec.length; start += settings_.request_bytes) {
    reading.pieces.push_back(
        Piece{.start = start,
              .length = std::min<std::uint64_t>(settings_.request_bytes, spec.length - start),
              .done = 0,
              .in_flight = false,
              .token = 0});
  }
  Queue(key, reads_.emplace(key, std::move(reading)).first->second);
  StartQueued();  // behind any older read still waiting for room
  return false;
}

std::expected<void, ReadError> DirectReader::Withdraw(std::uint64_t key, std::uint64_t waiter) {
  const auto found = reads_.find(key);
  if (found == reads_.end()) {
    return std::unexpected(ReadError::kUnknownRead);
  }
  Reading& reading = found->second;
  const auto position = std::ranges::find(reading.waiters, waiter);
  if (position == reading.waiters.end()) {
    return std::unexpected(ReadError::kUnknownRead);
  }
  reading.waiters.erase(position);
  if (reading.waiters.empty() && !reading.stopping) {
    Stop(reading, ReadOutcome::kCancelled, 0);
    for (const Piece& piece : reading.pieces) {
      if (piece.in_flight) {
        CancelIfAbandoned(piece.token);
      }
    }
    settled_.push_back(key);  // with nothing in flight, it has ended
  }
  return {};
}

void DirectReader::CancelIfAbandoned(std::uint64_t token) {
  const auto found = requests_.find(token);
  if (found == requests_.end() || found->second.cancelled) {
    return;
  }
  // A span also carries other reads' bytes: cancelled only once none of
  // them wants it.
  const bool abandoned = std::ranges::all_of(found->second.members, [this](const Member& member) {
    return reads_.at(member.key).stopping;
  });
  if (abandoned) {
    found->second.cancelled = true;
    (void)storage_.Cancel(token);  // best effort: it still completes
  }
}

void DirectReader::Stop(Reading& reading, ReadOutcome outcome, int error) {
  if (!reading.stopping) {
    reading.stopping = true;
    reading.outcome = outcome;
    reading.error = error;
  }
}

void DirectReader::Queue(std::uint64_t key, Reading& reading) {
  if (!reading.queued && !reading.stopping) {
    reading.queued = true;
    queue_.emplace(reading.arrival, key);
  }
}

bool DirectReader::HasStart(const Reading& reading) {
  return !reading.stopping && std::ranges::any_of(reading.pieces, [](const Piece& piece) {
    return !piece.in_flight && piece.done < piece.length;
  });
}

void DirectReader::StartQueued() {
  // Pieces in start order (reads by arrival, a read's pieces in turn),
  // gathered into runs that continue one another in the file.
  run_.clear();
  std::uint64_t run_end = 0;  // the file offset the run reaches
  std::uint64_t run_bytes = 0;
  bool full = false;
  const bool coalescing = settings_.span_bytes > settings_.request_bytes;
  for (const auto& [arrival, key] : queue_) {
    Reading& reading = reads_.at(key);
    if (reading.stopping) {
      continue;
    }
    for (std::size_t i = 0; i < reading.pieces.size() && !full; ++i) {
      const Piece& piece = reading.pieces[i];
      if (piece.in_flight || piece.done == piece.length) {
        continue;
      }
      const std::uint64_t offset = reading.spec.offset + piece.start + piece.done;
      const std::uint64_t length = piece.length - piece.done;
      if (!run_.empty()) {
        const Reading& last = reads_.at(run_.back().key);
        const bool joins = coalescing && !last.alone && !reading.alone &&
                           last.spec.fd == reading.spec.fd && last.spec.kind == IoKind::kRead &&
                           reading.spec.kind == IoKind::kRead && run_end == offset &&
                           run_bytes + length <= settings_.span_bytes &&
                           run_.size() < settings_.span_segments;
        if (!joins) {
          if (!Submit(run_)) {
            full = true;  // the provider is full: the rest wait, in order
            break;
          }
          run_.clear();
          run_bytes = 0;
        }
      }
      run_.push_back(Member{.key = key, .piece = i});
      run_end = offset + length;
      run_bytes += length;
    }
    if (full) {
      break;
    }
  }
  if (!full && !run_.empty()) {
    (void)Submit(run_);
  }
  run_.clear();
  // Reads with nothing left to start leave the queue. They are a prefix:
  // runs start in order, and the first the provider refused ends the pass.
  while (!queue_.empty()) {
    const auto oldest = queue_.begin();
    Reading& reading = reads_.at(oldest->second);
    if (HasStart(reading)) {
      return;
    }
    reading.queued = false;
    queue_.erase(oldest);
  }
}

bool DirectReader::Submit(std::span<const Member> run) {
  const std::uint64_t token = next_token_++;
  const Reading& first = reads_.at(run.front().key);
  const Piece& head = first.pieces[run.front().piece];
  IoRequest request{
      .token = token,
      .kind = first.spec.kind,
      .fd = first.spec.fd,
      .offset = first.spec.offset + head.start + head.done,
      .memory = nullptr,
      .length = 0,
      .segments = {},
  };
  segments_.clear();
  std::uint64_t total = 0;
  for (const Member& member : run) {
    const Reading& reading = reads_.at(member.key);
    const Piece& piece = reading.pieces[member.piece];
    const IoSegment segment{
        .memory = reading.spec.memory + piece.start +
                  piece.done,  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        .length = static_cast<std::uint32_t>(piece.length - piece.done)};
    segments_.push_back(segment);
    total += segment.length;
  }
  request.length = static_cast<std::uint32_t>(total);
  if (run.size() == 1) {
    request.memory = segments_.front().memory;  // a lone piece: a plain request
  } else {
    request.segments = segments_;
  }
  if (storage_.Submit(request) == Submission::kNotStarted) {
    return false;  // the next Poll tries again
  }
  // Accepted or unknown: either way a completion is owed.
  Request& started = requests_[token];
  started.members.assign(run.begin(), run.end());
  for (const Member& member : run) {
    Piece& piece = reads_.at(member.key).pieces[member.piece];
    piece.in_flight = true;
    piece.token = token;
  }
  return true;
}

void DirectReader::Settle(const Request& request, std::int64_t result) {
  // Each read once, where a request carries several of its pieces.
  const auto each_read = [&request](auto&& action) {
    for (std::size_t m = 0; m < request.members.size(); ++m) {
      if (m == 0 || request.members[m].key != request.members[m - 1].key) {
        action(request.members[m].key);
      }
    }
  };
  if (result < 0) {
    const int error = static_cast<int>(-result);
    if (error == ECANCELED) {
      each_read([this](std::uint64_t key) { Stop(reads_.at(key), ReadOutcome::kCancelled, 0); });
    } else if (error == EINTR || error == EAGAIN) {
      each_read([this, error](std::uint64_t key) {
        Reading& reading = reads_.at(key);
        if (reading.retries < settings_.retries) {
          ++reading.retries;  // transient: started again, ahead of later reads
          Queue(key, reading);
        } else {
          Stop(reading, ReadOutcome::kFailed, error);
        }
      });
    } else if (request.members.size() > 1) {
      // Which piece failed is unknown: each read starts again on its own,
      // so the error lands on the one it belongs to.
      each_read([this](std::uint64_t key) {
        Reading& reading = reads_.at(key);
        reading.alone = true;
        Queue(key, reading);
      });
    } else {
      Stop(reads_.at(request.members.front().key), ReadOutcome::kFailed, error);
    }
    return;
  }
  // The count fills the pieces in order.
  auto remaining = static_cast<std::uint64_t>(result);
  bool reached = true;  // the transfer reached this piece's end
  for (const Member& member : request.members) {
    Reading& reading = reads_.at(member.key);
    if (!reached) {
      Queue(member.key, reading);  // not reached: starts again, ahead of later reads
      continue;
    }
    Piece& piece = reading.pieces[member.piece];
    const std::uint64_t moved = std::min(remaining, piece.length - piece.done);
    piece.done += moved;
    remaining -= moved;
    if (piece.done == piece.length) {
      continue;
    }
    reached = false;
    // The first piece left short: a lone request that stopped here.
    if (result == 0 || piece.done % settings_.alignment != 0) {
      if (reading.spec.kind == IoKind::kWrite) {
        Stop(reading, ReadOutcome::kFailed, EIO);  // a write that stops short failed
      } else {
        Stop(reading, ReadOutcome::kEndOfFile, 0);  // the file ended here
      }
    } else {
      Queue(member.key, reading);  // continues, ahead of later reads
    }
  }
}

bool DirectReader::Finished(const Reading& reading) {
  return std::ranges::all_of(reading.pieces, [&reading](const Piece& piece) {
    return !piece.in_flight && (reading.stopping || piece.done == piece.length);
  });
}

std::vector<FinishedRead> DirectReader::Poll(bool wait) {
  StartQueued();
  std::array<IoCompletion, 64> completions{};
  std::size_t harvested = storage_.Harvest(completions, wait);
  while (harvested > 0) {
    for (std::size_t c = 0; c < harvested; ++c) {
      const IoCompletion& completion = completions[c];
      const auto found = requests_.find(completion.token);
      if (found == requests_.end()) {
        continue;  // not ours, or seen already; the provider reports only what it accepted
      }
      const Request request = std::move(found->second);
      requests_.erase(found);
      for (const Member& member : request.members) {
        reads_.at(member.key).pieces[member.piece].in_flight = false;
        settled_.push_back(member.key);
      }
      Settle(request, completion.result);  // restarted pieces start below, ahead of later reads
    }
    harvested = storage_.Harvest(completions, false);
  }
  StartQueued();
  std::vector<FinishedRead> finished;
  for (const std::uint64_t key : settled_) {
    const auto it = reads_.find(key);
    if (it == reads_.end() || !Finished(it->second)) {
      continue;  // ended already (listed twice), or still reading
    }
    Reading& reading = it->second;
    if (reading.queued) {
      queue_.erase(reading.arrival);  // stopping: nothing more will start
    }
    std::uint64_t bytes = 0;
    for (const Piece& piece : reading.pieces) {
      bytes += piece.done;
    }
    finished.push_back(
        FinishedRead{.key = it->first,
                     .outcome = reading.stopping ? reading.outcome : ReadOutcome::kComplete,
                     .bytes = bytes,
                     .error = reading.error,
                     .waiters = std::move(reading.waiters)});
    reads_.erase(it);
  }
  settled_.clear();
  return finished;
}

}  // namespace llmp::providers
