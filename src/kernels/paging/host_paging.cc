// SPDX-FileCopyrightText: 2026 llmpalooza contributors
// SPDX-License-Identifier: Apache-2.0

// The paging kernels (paging.h) as host loops for the fake device runtime
// (providers/fake/fake_device_runtime.h): each is queued on the fake stream
// behind `stream` and runs when the fake runs that stream, reading its
// operands in place then, as the CUDA kernels read theirs when they run.
// Launch limits are CUDA's (a grid's x dimension), so a launch the GPU
// refuses is refused here too.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

#include "kernels/paging/paging.h"
#include "providers/device_execution.h"
#include "providers/fake/fake_device_execution.h"

namespace llmp::kernels::paging {

namespace {

using providers::fake::FakeDeviceExecution;

constexpr std::uint64_t kMaxGrid = 0x7fffffff;  // CUDA's grid x limit

void* At(std::uint64_t address) {
  return reinterpret_cast<void*>(address);  // NOLINT(performance-no-int-to-ptr)
}

bool Queue(void* stream, FakeDeviceExecution::HostWork work) {
  return FakeDeviceExecution::Launch(providers::NativeStream{.handle = stream}, std::move(work))
      .has_value();
}

}  // namespace

bool FillRanges(const std::uint64_t* ranges, std::uint32_t count, std::uint8_t value,
                void* stream) {
  if (count == 0) {
    return true;
  }
  if (count > kMaxGrid) {
    return false;
  }
  return Queue(stream, [ranges, count, value] {
    for (std::uint32_t i = 0; i < count; ++i) {
      std::memset(At(ranges[2 * std::size_t{i}]), value, ranges[(2 * std::size_t{i}) + 1]);
    }
  });
}

bool GatherPleRows(const std::byte* landing, const std::uint64_t* sources,
                   const std::uint32_t* count, std::uint32_t max_count, std::uint32_t row_bytes,
                   std::byte* slots, void* stream) {
  if (max_count == 0) {
    return true;
  }
  if (max_count > kMaxGrid) {
    return false;
  }
  return Queue(stream, [landing, sources, count, max_count, row_bytes, slots] {
    const std::uint32_t rows = std::min(*count, max_count);
    for (std::uint32_t slot = 0; slot < rows; ++slot) {
      std::memcpy(slots + (std::uint64_t{slot} * row_bytes), landing + sources[slot], row_bytes);
    }
  });
}

bool ClampTokens(std::int32_t* tokens, std::uint32_t count, std::int32_t limit, void* stream) {
  if (count == 0) {
    return true;
  }
  return Queue(stream, [tokens, count, limit] {
    for (std::uint32_t i = 0; i < count; ++i) {
      if (tokens[i] < 0 || tokens[i] >= limit) {
        tokens[i] = 0;
      }
    }
  });
}

bool CopyRanges(const RangeCopy* ranges, std::uint32_t count, void* stream) {
  if (count == 0) {
    return true;
  }
  if (!RangeCopiesValid(ranges, count)) {
    return false;
  }
  return Queue(stream, [ranges, count] {
    for (std::uint32_t i = 0; i < count; ++i) {
      std::memmove(At(ranges[i].to), At(ranges[i].from), ranges[i].bytes);
    }
  });
}

}  // namespace llmp::kernels::paging
