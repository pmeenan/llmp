// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

#include "providers/fake/fake_device_execution.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "base/bytes.h"
#include "base/check.h"
#include "providers/device_execution.h"

namespace llmp::providers::fake {
namespace {

std::unexpected<Failure> Invalid(std::string detail) {
  return std::unexpected(Failure{.error = ProviderError::kInvalid, .detail = std::move(detail)});
}

void* At(std::uint64_t address) {
  return reinterpret_cast<void*>(address);  // NOLINT(performance-no-int-to-ptr)
}

std::uint64_t TokenOf(NativeStream stream) {
  return reinterpret_cast<std::uint64_t>(stream.handle);
}

// Every live fake stream's native token, across the process's fakes. Its
// lock is never held with a fake's own.
struct Registry {
  std::mutex mutex;
  std::unordered_map<std::uint64_t, std::pair<FakeDeviceExecution*, StreamId>> streams;
};
Registry& Tokens() {
  static Registry registry;
  return registry;
}
// Distinct and non-null, at no real address, never reused in the process.
std::uint64_t NextToken() {
  static std::atomic<std::uint64_t> next{0};
  return next.fetch_add(16, std::memory_order_relaxed) + 16;
}

}  // namespace

FakeDeviceExecution::~FakeDeviceExecution() {
  std::vector<std::uint64_t> tokens;
  {
    const std::scoped_lock lock(mutex_);
    streams_.ForEach(
        [&](StreamId /*id*/, const Stream& stream) { tokens.push_back(stream.native); });
  }
  Registry& registry = Tokens();
  const std::scoped_lock lock(registry.mutex);
  for (const std::uint64_t token : tokens) {
    registry.streams.erase(token);
  }
}

std::expected<StreamId, Failure> FakeDeviceExecution::CreateStream() {
  const std::uint64_t token = NextToken();
  StreamId id;
  {
    const std::scoped_lock lock(mutex_);
    id = streams_.Insert(
        Stream{.steps = {}, .fences = 0, .unfenced = false, .native = token, .recording = nullptr});
  }
  if (!id.valid()) {
    return std::unexpected(
        Failure{.error = ProviderError::kFailed, .detail = "the stream table is full"});
  }
  Registry& registry = Tokens();
  const std::scoped_lock lock(registry.mutex);
  registry.streams.emplace(token, std::pair{this, id});
  return id;
}

std::expected<void, Failure> FakeDeviceExecution::DestroyStream(StreamId stream) {
  std::uint64_t token = 0;
  {
    const std::scoped_lock lock(mutex_);
    const Stream* found = streams_.Find(stream);
    if (found == nullptr) {
      return Invalid("stale or unknown stream");
    }
    if (found->fences > 0 || found->unfenced || !found->steps.empty() ||
        found->recording != nullptr) {
      return Invalid("the stream has work not yet behind a completed fence");
    }
    token = found->native;
    (void)streams_.Erase(stream);
  }
  Registry& registry = Tokens();
  const std::scoped_lock lock(registry.mutex);
  registry.streams.erase(token);
  return {};
}

std::optional<FakeDeviceExecution::Owner> FakeDeviceExecution::OwnerOf(NativeStream stream) {
  Registry& registry = Tokens();
  const std::scoped_lock lock(registry.mutex);
  const auto found = registry.streams.find(TokenOf(stream));
  if (found == registry.streams.end()) {
    return std::nullopt;
  }
  return Owner{.execution = found->second.first, .stream = found->second.second};
}

std::expected<void, Failure> FakeDeviceExecution::Launch(NativeStream stream, HostWork work) {
  const auto owner = OwnerOf(stream);
  if (!owner) {
    return Invalid("no fake stream has this native handle");
  }
  return owner->execution->Enqueue(owner->stream, std::move(work));
}

std::expected<void, Failure> FakeDeviceExecution::BeginRecording(NativeStream stream) {
  const auto owner = OwnerOf(stream);
  if (!owner) {
    return Invalid("no fake stream has this native handle");
  }
  return owner->execution->Begin(owner->stream);
}

std::expected<std::unique_ptr<FakeDeviceExecution::Recording>, Failure>
FakeDeviceExecution::EndRecording(NativeStream stream) {
  const auto owner = OwnerOf(stream);
  if (!owner) {
    return Invalid("no fake stream has this native handle");
  }
  return owner->execution->End(owner->stream);
}

std::expected<void, Failure> FakeDeviceExecution::Enqueue(StreamId stream, HostWork work) {
  const std::scoped_lock lock(mutex_);
  Stream* found = streams_.Find(stream);
  if (found == nullptr) {
    return Invalid("stale or unknown stream");
  }
  if (found->recording != nullptr) {
    found->recording->push_back(std::move(work));
    return {};
  }
  found->unfenced = true;
  found->steps.push_back(Queued{.kind = Queued::Kind::kHost,
                                .destination = 0,
                                .source = 0,
                                .size = {},
                                .fence = {},
                                .work = std::move(work)});
  return {};
}

std::expected<void, Failure> FakeDeviceExecution::Begin(StreamId stream) {
  const std::scoped_lock lock(mutex_);
  Stream* found = streams_.Find(stream);
  if (found == nullptr) {
    return Invalid("stale or unknown stream");
  }
  if (found->recording != nullptr) {
    return Invalid("the stream is already recording");
  }
  found->recording = std::make_unique<Recording>();
  return {};
}

std::expected<std::unique_ptr<FakeDeviceExecution::Recording>, Failure> FakeDeviceExecution::End(
    StreamId stream) {
  const std::scoped_lock lock(mutex_);
  Stream* found = streams_.Find(stream);
  if (found == nullptr) {
    return Invalid("stale or unknown stream");
  }
  if (found->recording == nullptr) {
    return Invalid("the stream is not recording");
  }
  return std::move(found->recording);
}

std::expected<NativeStream, Failure> FakeDeviceExecution::Submission(StreamId stream) {
  const std::scoped_lock lock(mutex_);
  Stream* found = streams_.Find(stream);
  if (found == nullptr) {
    return Invalid("stale or unknown stream");
  }
  found->unfenced = true;
  return NativeStream{.handle = At(found->native)};
}

std::expected<void, Failure> FakeDeviceExecution::Copy(StreamId stream, std::uint64_t destination,
                                                       std::uint64_t source, Bytes size) {
  const std::scoped_lock lock(mutex_);
  Stream* found = streams_.Find(stream);
  if (found == nullptr) {
    return Invalid("stale or unknown stream");
  }
  const auto fault = Take(copy_fault_, FenceId{});
  if (fault && *fault != ProviderError::kUnknown) {
    return std::unexpected(Failure{.error = *fault, .detail = "scripted copy refusal"});
  }
  if (found->recording != nullptr) {
    found->recording->push_back([destination, source, bytes = size.value()] {
      std::memmove(At(destination), At(source), bytes);
    });
  } else {
    found->unfenced = true;
    found->steps.push_back(Queued{.kind = Queued::Kind::kCopy,
                                  .destination = destination,
                                  .source = source,
                                  .size = size,
                                  .fence = {},
                                  .work = {}});
  }
  if (fault) {
    return std::unexpected(Failure{.error = *fault, .detail = "scripted copy fault"});
  }
  return {};
}

std::expected<void, Failure> FakeDeviceExecution::Zero(StreamId stream, std::uint64_t destination,
                                                       Bytes size) {
  const std::scoped_lock lock(mutex_);
  Stream* found = streams_.Find(stream);
  if (found == nullptr) {
    return Invalid("stale or unknown stream");
  }
  const auto fault = Take(copy_fault_, FenceId{});
  if (fault && *fault != ProviderError::kUnknown) {
    return std::unexpected(Failure{.error = *fault, .detail = "scripted zeroing refusal"});
  }
  if (found->recording != nullptr) {
    found->recording->push_back(
        [destination, bytes = size.value()] { std::memset(At(destination), 0, bytes); });
  } else {
    found->unfenced = true;
    found->steps.push_back(Queued{.kind = Queued::Kind::kZero,
                                  .destination = destination,
                                  .source = 0,
                                  .size = size,
                                  .fence = {},
                                  .work = {}});
  }
  if (fault) {
    return std::unexpected(Failure{.error = *fault, .detail = "scripted zeroing fault"});
  }
  return {};
}

std::expected<void, Failure> FakeDeviceExecution::Wait(StreamId stream, FenceId fence) {
  const std::scoped_lock lock(mutex_);
  Stream* found = streams_.Find(stream);
  if (found == nullptr || fences_.Find(fence) == nullptr) {
    return Invalid("stale or unknown stream or fence");
  }
  if (found->recording != nullptr) {
    return Invalid("the stream is recording");
  }
  found->unfenced = true;
  found->steps.push_back(Queued{.kind = Queued::Kind::kWait,
                                .destination = 0,
                                .source = 0,
                                .size = {},
                                .fence = fence,
                                .work = {}});
  return {};
}

std::expected<FenceId, Failure> FakeDeviceExecution::Record(StreamId stream) {
  const std::scoped_lock lock(mutex_);
  Stream* found = streams_.Find(stream);
  if (found == nullptr) {
    return Invalid("stale or unknown stream");
  }
  if (found->recording != nullptr) {
    return Invalid("the stream is recording");
  }
  const FenceId fence = fences_.Insert(Fence{.stream = stream, .complete = false, .seen = false});
  if (!fence.valid()) {
    return std::unexpected(
        Failure{.error = ProviderError::kFailed, .detail = "the fence table is full"});
  }
  found = streams_.Find(stream);
  found->steps.push_back(Queued{.kind = Queued::Kind::kFence,
                                .destination = 0,
                                .source = 0,
                                .size = {},
                                .fence = fence,
                                .work = {}});
  ++found->fences;
  found->unfenced = false;
  return fence;
}

std::optional<ProviderError> FakeDeviceExecution::Take(Fault& fault, FenceId fence) {
  if (fault.times == 0 || (fault.fence.valid() && fault.fence != fence)) {
    return std::nullopt;
  }
  --fault.times;
  return fault.error;
}

std::expected<FenceState, Failure> FakeDeviceExecution::Query(FenceId fence) {
  const std::scoped_lock lock(mutex_);
  Fence* found = fences_.Find(fence);
  if (found == nullptr) {
    return Invalid("stale or unknown fence");
  }
  if (const auto error = Take(fault_, fence)) {
    return std::unexpected(Failure{.error = *error, .detail = "scripted device fault"});
  }
  if (progress_ == Progress::kOnQuery && !found->complete) {
    // What the device ran meanwhile. The fence table does not change, so
    // `found` stays valid.
    DrainLocked();
  }
  if (found->complete) {
    found->seen = true;
    return FenceState::kComplete;
  }
  return FenceState::kPending;
}

std::expected<void, Failure> FakeDeviceExecution::Release(FenceId fence) {
  const std::scoped_lock lock(mutex_);
  const Fence* found = fences_.Find(fence);
  if (found == nullptr) {
    return Invalid("stale or unknown fence");
  }
  if (!found->seen) {
    return Invalid("the fence has not been seen complete");
  }
  if (const auto error = Take(release_fault_, fence)) {
    return std::unexpected(Failure{.error = *error, .detail = "scripted release failure"});
  }
  Stream* stream = streams_.Find(found->stream);
  base::Check(stream != nullptr && stream->fences > 0, "a fence outlived its stream");
  --stream->fences;
  (void)fences_.Erase(fence);
  return {};
}

bool FakeDeviceExecution::Step(StreamId stream) {
  const std::scoped_lock lock(mutex_);
  return StepLocked(stream);
}

bool FakeDeviceExecution::StepLocked(StreamId stream) {
  Stream* found = streams_.Find(stream);
  if (found == nullptr || found->steps.empty()) {
    return false;
  }
  Queued& step = found->steps.front();
  switch (step.kind) {
    case Queued::Kind::kCopy:
      std::memmove(At(step.destination), At(step.source), step.size.value());
      break;
    case Queued::Kind::kZero:
      std::memset(At(step.destination), 0, step.size.value());
      break;
    case Queued::Kind::kWait: {
      const Fence* awaited = fences_.Find(step.fence);
      if (awaited != nullptr && !awaited->complete) {
        return false;  // blocked on the other stream
      }
      break;
    }
    case Queued::Kind::kFence:
      fences_.Find(step.fence)->complete = true;
      break;
    case Queued::Kind::kHost:
      step.work();
      break;
  }
  found->steps.pop_front();
  return true;
}

void FakeDeviceExecution::Drain() {
  const std::scoped_lock lock(mutex_);
  DrainLocked();
}

void FakeDeviceExecution::DrainLocked() {
  bool moved = true;
  while (moved) {
    moved = false;
    streams_.ForEach([&](StreamId id, const Stream& /*stream*/) {
      while (StepLocked(id)) {
        moved = true;
      }
    });
  }
}

}  // namespace llmp::providers::fake
